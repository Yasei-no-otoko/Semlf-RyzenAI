"""Run the frozen Ryzen AI benchmark with full-vocabulary finite checks.

The workload, warmups, metrics, and token limits remain in ryzenai_benchmark.
Use the same command-line arguments. Evidence is create-only; observations
are written after measured work, including when a Python exception stops it.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def code_hashes():
    paths = [*Path("src/semif_phase1").glob("*.py"),
             *Path("benchmarks").glob("*.py")]
    return {path.as_posix(): file_sha256(path) for path in sorted(paths)}


def runtime_hashes(oga):
    directory = Path(oga.__file__).parent
    return {path.relative_to(directory).as_posix(): file_sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".dll", ".pyd"}}


@contextmanager
def describe_observed_run(benchmark, provenance):
    """Adapt descriptions without changing scoring or measurement functions."""
    original_load, original_write = benchmark.backend.load_model, benchmark.write
    loaded_metadata = []

    def load_model(*args, **kwargs):
        model, tokenizer, metadata = original_load(*args, **kwargs)
        metadata = dict(metadata)
        metadata["execution"] = (
            "RyzenAI NPU configuration with CPU graph/host components; "
            "no GPU provider configured. Artifact identity is recorded below.")
        loaded_metadata.append(metadata)
        return model, tokenizer, metadata

    def write(path, value):
        if Path(path).name == "manifest.json":
            value = dict(value)
            value["observation"] = provenance
            value["timing_scope"] += (
                " Full-vocabulary readback and finite checks are included; "
                "observation serialization and provenance hashing are excluded.")
            value["limitations"] = [
                "Model families, tokenizers, quantization recipes and NPU graphs differ; "
                "this compares the recorded deployed artifacts, not architecture alone.",
                "NPU execution includes CPU graph/host work; no GPU offload is configured.",
                "Speed retains the original 4096-token prompt budget, three compact "
                "repeats and 128 generated-token limit; Quality uses the model context ceiling.",
                "Invalid JSON and generation truncation are results, not grounds for retry.",
                "No source text or reference document is copied into predictions.",
                "This is a sequential local systems measurement, not an isolated appliance.",
            ]
        original_write(path, value)

    benchmark.backend.load_model, benchmark.write = load_model, write
    try:
        yield loaded_metadata
    finally:
        benchmark.backend.load_model, benchmark.write = original_load, original_write


def run_observed(benchmark, oga, guard, *, model_path, output):
    """The caller supplies the original benchmark argv and an unused output."""
    output, model_path = Path(output), Path(model_path)
    if output.exists():
        raise ValueError("Output directory must be new")
    config = json.loads((model_path / "genai_config.json").read_text(encoding="utf-8"))
    package_manifest = model_path / "package-manifest.json"
    provenance = {
        "version": "ryzenai-observed-v1",
        "code_sha256": code_hashes(),
        "runtime_binary_sha256": runtime_hashes(oga),
        "package_manifest_sha256": file_sha256(package_manifest) if package_manifest.is_file() else None,
        "guard": "Every direct readout and each pre-sampling readout checks the full vocabulary; "
                 "the following sample reuses cached logits. No post-EOS readout.",
        "speed_prompt_budget": 4096,
        "max_generated_tokens": 128,
        "thread_environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "sdk_environment": os.environ.get("RYZEN_AI_INSTALLATION_PATH"),
        "partial_evidence": "Quality flushes each row; Speed flushes each 21-row group. "
                            "A hard process exit can lose the current group and final observations.",
    }
    records = []
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    started = time.perf_counter()
    try:
        with describe_observed_run(benchmark, provenance) as metadata:
            with guard(oga, vocab_size=config["model"]["vocab_size"], observations=records,
                       max_generated_tokens=128, hash_logits=False):
                benchmark.main()
        report["status"] = "complete"
        report["code_unchanged"] = code_hashes() == provenance["code_sha256"]
        report["runtime_unchanged"] = runtime_hashes(oga) == provenance["runtime_binary_sha256"]
        report["package_manifest_unchanged"] = (
            (file_sha256(package_manifest) if package_manifest.is_file() else None)
            == provenance["package_manifest_sha256"])
        report["model_artifacts_unchanged"] = bool(metadata) and (
            benchmark.backend._artifact_hashes(model_path) == metadata[0]["source_artifact_sha256"])
        if not all(report[key] for key in ("code_unchanged", "runtime_unchanged",
                                           "package_manifest_unchanged", "model_artifacts_unchanged")):
            raise RuntimeError("Code, runtime, or model artifacts changed during the benchmark")
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        report["provenance"] = provenance
        report["generator_observations"] = records
        output.mkdir(parents=True, exist_ok=True)
        benchmark.write(output / "observations.json", report)


def main():
    import argparse
    import onnxruntime_genai as oga
    import ryzenai_benchmark as benchmark
    from ryzenai_guard import guard_generators

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args, _ = parser.parse_known_args()
    run_observed(benchmark, oga, guard_generators, model_path=args.model, output=args.output)


if __name__ == "__main__":
    # Script execution already puts benchmarks/ on sys.path. Module execution
    # needs it for the existing benchmark's sibling evaluator imports.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
