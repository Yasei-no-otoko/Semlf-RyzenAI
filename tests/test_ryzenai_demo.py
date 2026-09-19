"""Loopback-only regression tests for the Ryzen AI browser demo."""

from http.client import HTTPConnection
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from semif_phase1 import ryzenai_demo as demo


PAYLOAD = {"state": "The deployment passed.", "question": "What happened?", "options": ["Passed", "Failed"]}


class _Service:
    def __init__(self, *, loaded=True, busy=False):
        self.loaded = loaded
        self.busy = busy
        self.rows = []

    def status(self):
        return {"state": "ready" if self.loaded else "unloaded"}

    def load(self):
        self.loaded = True
        return self.status()

    def reserve_run(self):
        if self.busy:
            raise demo.BusyError("busy")
        if not self.loaded:
            raise demo.BusyError("Load the model before running a comparison.")

    def run_reserved(self, row, emit):
        self.rows.append(row)
        emit({"type": "direct", "result": {"option_logits": [1.0, 0.0]}})
        emit({"type": "done"})


@pytest.fixture
def loopback(tmp_path):
    (tmp_path / "npu-demo").mkdir()
    (tmp_path / "npu-demo" / "index.html").write_text("<html lang='ja'>日本語</html>", encoding="utf-8")
    (tmp_path / "npu-demo" / "en.html").write_text("<html lang='en'>English</html>", encoding="utf-8")
    (tmp_path / "npu-demo" / "app.js").write_text("window.demo = true;", encoding="utf-8")
    (tmp_path / "npu-demo" / "npu.css").write_text("body { color: black; }", encoding="utf-8")
    (tmp_path / "webgpu-demo").mkdir()
    (tmp_path / "webgpu-demo" / "style.css").write_text("body { margin: 0; }", encoding="utf-8")
    service = _Service()
    server = demo.make_server(service, port=0, root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield service, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _request(port, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    request_headers = dict(headers or {})
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    data = response.read()
    result = response.status, dict(response.getheaders()), data
    connection.close()
    return result


def _json_request(port, path, payload, headers=None):
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    return _request(port, "POST", path, json.dumps(payload), request_headers)


def test_status_and_valid_run_stream_are_loopback_only(loopback):
    service, port = loopback

    status, headers, data = _request(port, "GET", "/api/status")
    assert status == 200
    assert json.loads(data) == {"state": "ready"}
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"

    status, headers, data = _json_request(port, "/api/run", PAYLOAD)
    assert status == 200
    assert headers["Content-Type"].startswith("application/x-ndjson")
    assert [json.loads(line)["type"] for line in data.splitlines()] == ["direct", "done"]
    assert service.rows == [{
        "id": "browser-decision", "state": PAYLOAD["state"], "question": PAYLOAD["question"],
        "options": [{"id": "A", "description": "Passed"}, {"id": "B", "description": "Failed"}],
    }]


def test_language_pages_and_shared_assets_are_served_with_external_origins(loopback):
    _, port = loopback
    foreign_origin = {"Origin": "https://example.test", "Sec-Fetch-Site": "cross-site"}

    status, headers, japanese = _request(port, "GET", "/", headers=foreign_origin)
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "日本語" in japanese.decode("utf-8")

    for route in ("/en", "/en/", "/en.html"):
        status, headers, english = _request(port, "GET", route, headers=foreign_origin)
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert english == b"<html lang='en'>English</html>"

    for route, expected_type in (("/app.js", "text/javascript"), ("/npu.css", "text/css"),
                                 ("/style.css", "text/css")):
        status, headers, _ = _request(port, "GET", route, headers=foreign_origin)
        assert status == 200
        assert headers["Content-Type"].startswith(expected_type)

    status, _, data = _request(port, "GET", "/api/status", headers=foreign_origin)
    assert status == 403
    assert "local demo" in json.loads(data)["error"]


@pytest.mark.parametrize("body", [b"{", b"[]", b'{"state": 3, "question": "q", "options": ["a", "b"]}'])
def test_run_rejects_invalid_json_and_field_types(loopback, body):
    _, port = loopback
    status, _, data = _request(port, "POST", "/api/run", body, {"Content-Type": "application/json"})
    assert status == 400
    assert "error" in json.loads(data)


@pytest.mark.parametrize("options", [["one"], [str(index) for index in range(33)]])
def test_run_enforces_option_count_bounds(loopback, options):
    _, port = loopback
    payload = {**PAYLOAD, "options": options}
    status, _, data = _json_request(port, "/api/run", payload)
    assert status == 400
    assert "2–32" in json.loads(data)["error"]


def test_run_accepts_exactly_32_distinct_npu_option_labels(loopback):
    service, port = loopback
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    assert demo.NPU_LABELS == labels
    payload = {**PAYLOAD, "options": list(labels)}

    status, headers, data = _json_request(port, "/api/run", payload)

    assert status == 200
    assert headers["Content-Type"].startswith("application/x-ndjson")
    assert [json.loads(line)["type"] for line in data.splitlines()] == ["direct", "done"]
    assert [option["id"] for option in service.rows[0]["options"]] == list(labels)
    assert [option["description"] for option in service.rows[0]["options"]] == list(labels)


def test_run_before_load_and_busy_requests_return_conflict(tmp_path):
    for service in (_Service(loaded=False), _Service(busy=True)):
        server = demo.make_server(service, port=0, root=tmp_path)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, _, data = _json_request(server.server_port, "/api/run", PAYLOAD)
            assert status == 409
            assert "error" in json.loads(data)
            assert service.rows == []
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


@pytest.mark.parametrize("headers", [
    {"Host": "example.test"},
    {"Origin": "http://example.test"},
    {"Sec-Fetch-Site": "cross-site"},
])
def test_external_host_origin_and_cross_site_requests_are_rejected(loopback, headers):
    _, port = loopback
    status, _, data = _request(port, "GET", "/api/status", headers=headers)
    assert status == 403
    assert "local demo" in json.loads(data)["error"] or "Cross-site" in json.loads(data)["error"]


@pytest.mark.parametrize("path", ["/..%2fAGENTS.md", "/npu-demo/../index.html", "/%2e%2e/pyproject.toml"])
def test_static_routes_do_not_allow_filesystem_traversal(loopback, path):
    _, port = loopback
    status, _, data = _request(port, "GET", path)
    assert status == 404
    assert json.loads(data)["error"] == "Not found"


@pytest.mark.parametrize("callback_error", [RuntimeError("callback failed"), BrokenPipeError("disconnected")])
def test_service_loads_once_on_one_worker_and_releases_gate_after_callback_error(monkeypatch, callback_error):
    calls = []

    def load_model(source, revision):
        calls.append(("load", threading.get_ident(), source, revision))
        return object(), object(), {"context_ceiling": 31}

    def score(model, tokenizer, row, metadata):
        calls.append(("score", threading.get_ident()))
        return {"id": row["id"]}

    def generate(model, tokenizer, row, metadata, max_new_tokens, on_token):
        calls.append(("generate", threading.get_ident(), max_new_tokens))
        on_token("A")
        return {"text": "A"}

    monkeypatch.setitem(sys.modules, "semif_phase1.ryzenai_backend", SimpleNamespace(
        load_model=load_model, score=score,
    ))
    monkeypatch.setitem(sys.modules, "semif_phase1.ryzenai_generation", SimpleNamespace(generate=generate))
    service = demo.DemoService("fixture", "revision")
    try:
        assert service.load()["state"] == "ready"
        assert service.load()["state"] == "ready"
        assert [call[0] for call in calls] == ["load"]

        service.reserve_run()
        with pytest.raises(type(callback_error), match=str(callback_error)):
            service.run_reserved(demo.decision_row(PAYLOAD), lambda event: (_ for _ in ()).throw(callback_error))

        # The callback exception must not strand the exclusive NPU gate.
        service.reserve_run()
        events = []
        service.run_reserved(demo.decision_row(PAYLOAD), events.append)
        worker_ids = {call[1] for call in calls}
        assert len(worker_ids) == 1
        assert threading.get_ident() not in worker_ids
        assert [event["type"] for event in events] == ["direct", "token", "generation", "done"]

        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        service.reserve_run()
        service.run_reserved(demo.decision_row({**PAYLOAD, "options": list(labels)}), lambda event: None)
        assert [call[2] for call in calls if call[0] == "generate"] == [512, 1024]
    finally:
        service.close()
