"""Loopback web demo for AMD Ryzen AI direct scoring and JSON generation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from .core import validate_row
from .ryzenai_backend import NPU_LABELS


DEFAULT_MODEL = "models/Qwen3-4B-npu-4k"
DEFAULT_REVISION = "d6fb03663d78ae5034d4594bfe9d92b35a5e213a"
MAX_BODY_BYTES = 64 * 1024
REPO_ROOT = Path(__file__).resolve().parents[2]


class BusyError(RuntimeError):
    """Another request already owns the single model worker."""


def decision_row(payload: object) -> dict:
    """Accept the browser's editable fields, never executable model settings."""
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    for field in ("state", "question"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    options = payload.get("options")
    if (not isinstance(options, list) or not 2 <= len(options) <= len(NPU_LABELS)
            or any(not isinstance(option, str) or not option.strip() for option in options)):
        raise ValueError(f"Provide 2–{len(NPU_LABELS)} nonempty option descriptions")
    row = {
        "id": "browser-decision", "state": payload["state"], "question": payload["question"],
        "options": [{"id": NPU_LABELS[index], "description": option} for index, option in enumerate(options)],
    }
    validate_row(row, max_options=len(NPU_LABELS))
    return row


class DemoService:
    """Keep model initialization and every inference on one dedicated thread."""

    def __init__(self, source: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION):
        self.source = source
        self.revision = revision
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ryzenai")
        self._gate = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = "unloaded"
        self._error = None
        self._load_seconds = None
        self._loaded = None

    def _state_is(self, state: str, error: str | None = None):
        with self._state_lock:
            self._state, self._error = state, error

    def status(self) -> dict:
        with self._state_lock:
            return {
                "state": self._state, "error": self._error, "load_seconds": self._load_seconds,
                "model": {
                    "name": "AMD Qwen3-4B · Ryzen AI 1.8.0 NPU",
                    "revision": self.revision, "backend": "ryzenai-npu",
                    "context_limit": self._loaded[2]["context_ceiling"] if self._loaded else 4096,
                    "max_options": len(NPU_LABELS),
                },
            }

    def load(self) -> dict:
        if not self._gate.acquire(blocking=False):
            raise BusyError("The NPU is busy. Wait for the current operation to finish.")
        try:
            if self._loaded is None:
                self._state_is("loading")
                self._worker.submit(self._load).result()
            return self.status()
        finally:
            self._gate.release()

    def _load(self):
        from .ryzenai_backend import load_model

        started = time.perf_counter()
        try:
            self._loaded = load_model(self.source, self.revision)
            self._load_seconds = time.perf_counter() - started
            self._state_is("ready")
        except Exception as error:
            self._state_is("error", str(error))
            raise

    def reserve_run(self):
        if not self._gate.acquire(blocking=False):
            raise BusyError("The NPU is busy. Wait for the current operation to finish.")
        if self._loaded is None:
            self._gate.release()
            raise BusyError("Load the model before running a comparison.")
        self._state_is("running")

    def run_reserved(self, row: dict, emit):
        """Called only after reserve_run; emit failures cancel decoding promptly."""
        try:
            self._worker.submit(self._compare, row, emit).result()
        finally:
            self._state_is("ready")
            self._gate.release()

    def _compare(self, row: dict, emit):
        from .ryzenai_backend import score
        from .ryzenai_generation import generate

        model, tokenizer, metadata = self._loaded
        direct = score(model, tokenizer, row, metadata)
        emit({"type": "direct", "result": direct})
        generated = generate(model, tokenizer, row, metadata,
                             max_new_tokens=1024 if len(row["options"]) > 16 else 512,
                             on_token=lambda text: emit({"type": "token", "text": text}))
        emit({"type": "generation", "result": generated})
        emit({"type": "done"})

    def close(self):
        # Destruction also happens on the owning thread.
        self._worker.submit(lambda: setattr(self, "_loaded", None)).result()
        self._worker.shutdown(wait=True)


def make_server(service: DemoService, port: int = 8008,
                root: Path = REPO_ROOT) -> ThreadingHTTPServer:
    assets = {
        "/": (root / "npu-demo/index.html", "text/html; charset=utf-8"),
        "/index.html": (root / "npu-demo/index.html", "text/html; charset=utf-8"),
        "/en/": (root / "npu-demo/en.html", "text/html; charset=utf-8"),
        "/en": (root / "npu-demo/en.html", "text/html; charset=utf-8"),
        "/en.html": (root / "npu-demo/en.html", "text/html; charset=utf-8"),
        "/app.js": (root / "npu-demo/app.js", "text/javascript; charset=utf-8"),
        "/style.css": (root / "webgpu-demo/style.css", "text/css; charset=utf-8"),
        "/npu.css": (root / "npu-demo/npu.css", "text/css; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 closes each stream at completion, with no fake Content-Length.
        server_version = "SemIfRyzenAI/1.0"

        def _headers(self, code: int, content_type: str, length: int | None = None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; "
                             "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                             "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            if length is not None:
                self.send_header("Content-Length", str(length))
            self.end_headers()

        def _json(self, code: int, payload: dict):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._headers(code, "application/json; charset=utf-8", len(data))
            self.wfile.write(data)

        def _local_request(self) -> bool:
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            public_asset = self.command == "GET" and urlsplit(self.path).path in assets
            if host not in allowed or (not public_asset and origin is not None and origin != f"http://{host}"):
                self._json(403, {"error": "Only requests from this local demo are accepted."})
                return False
            if not public_asset and self.headers.get("Sec-Fetch-Site") == "cross-site":
                self._json(403, {"error": "Cross-site requests are not accepted."})
                return False
            return True

        def do_GET(self):
            if not self._local_request():
                return
            path = urlsplit(self.path).path
            if path == "/api/status":
                self._json(200, service.status())
                return
            asset = assets.get(path)
            if asset is None or not asset[0].is_file():
                self._json(404, {"error": "Not found"})
                return
            data = asset[0].read_bytes()
            self._headers(200, asset[1], len(data))
            self.wfile.write(data)

        def _payload(self):
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is not supported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError(f"JSON body must contain 1–{MAX_BODY_BYTES} bytes")
            self.connection.settimeout(30)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete JSON body")
            def reject_constant(value):
                raise ValueError(f"Non-finite JSON value: {value}")
            return json.loads(raw.decode("utf-8"), parse_constant=reject_constant)

        def do_POST(self):
            if not self._local_request():
                return
            path = urlsplit(self.path).path
            if path not in {"/api/load", "/api/run"}:
                self._json(404, {"error": "Not found"})
                return
            try:
                payload = self._payload()
                if path == "/api/load":
                    if not isinstance(payload, dict):
                        raise ValueError("Expected a JSON object")
                    self._json(200, service.load())
                    return
                row = decision_row(payload)
                service.reserve_run()
            except (ValueError, UnicodeError, RecursionError, TimeoutError) as error:
                self._json(400, {"error": str(error)})
                return
            except BusyError as error:
                self._json(409, {"error": str(error)})
                return
            except Exception as error:
                self._json(500, {"error": str(error)})
                return

            def emit(event):
                self.wfile.write(json.dumps(event, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
                self.wfile.flush()

            # Headers are written from within the reserved job so disconnects
            # always reach its finally block and release the model gate.
            sent_headers = False
            def stream(event):
                nonlocal sent_headers
                if not sent_headers:
                    self._headers(200, "application/x-ndjson; charset=utf-8")
                    sent_headers = True
                emit(event)
            try:
                service.run_reserved(row, stream)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                pass
            except Exception as error:
                try:
                    stream({"type": "error", "error": str(error)})
                except OSError:
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = False
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    service = DemoService(args.model, args.revision)
    try:
        with make_server(service, args.port) as server:
            print(f"SemIf Ryzen AI demo: http://127.0.0.1:{server.server_port}", flush=True)
            print("Open the page and load the model. Ctrl+C stops the server.", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        service.close()


if __name__ == "__main__":
    main()
