from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from benchmarks import ryzenai_observed as observed


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "genai_config.json").write_text(json.dumps({"model": {"vocab_size": 7}}))
    output = tmp_path / "evidence"
    actions = []
    native_refs = []
    original_metadata = {"source_artifact_sha256": {"model.onnx": "model-hash"},
                         "execution": "original description"}

    class Native:
        pass

    def load_model(*args, **kwargs):
        actions.append("load")
        native, tokenizer = Native(), Native()
        native_refs.extend([weakref.ref(native), weakref.ref(tokenizer)])
        return native, tokenizer, original_metadata

    def write(path, value):
        actions.append("write:" + Path(path).name)
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False)

    backend = SimpleNamespace(load_model=load_model,
                              _artifact_hashes=lambda path: {"model.onnx": "model-hash"})
    benchmark = SimpleNamespace(backend=backend, write=write)
    oga = SimpleNamespace(Generator=Native)
    kwargs_seen = []

    @contextmanager
    def guard(oga, **kwargs):
        kwargs_seen.append(kwargs)
        original = oga.Generator
        oga.Generator = object
        actions.append("guard-enter")
        try:
            kwargs["observations"].append({"generator_index": 1, "finite": True})
            yield
        finally:
            actions.append("guard-exit")
            oga.Generator = original

    def main():
        actions.append("main-start")
        output.mkdir()
        native, tokenizer, metadata = benchmark.backend.load_model(str(model), "revision")
        benchmark.write(output / "manifest.json", {
            "model": metadata, "timing_scope": "Frozen timing scope.",
            "limitations": ["old model-specific description"],
            "unchanged_metric": 4.5,
        })
        actions.append("measured-work")
        benchmark.write(output / "summary.json", {"median_seconds": 1.25})
        actions.append("main-end")

    benchmark.main = main

    def code_hashes():
        actions.append("code-hashes")
        return {"runner.py": "code-hash"}

    def runtime_hashes(oga):
        actions.append("runtime-hashes")
        return {"runtime.dll": "runtime-hash"}

    monkeypatch.setattr(observed, "code_hashes", code_hashes)
    monkeypatch.setattr(observed, "runtime_hashes", runtime_hashes)
    return SimpleNamespace(model=model, output=output, benchmark=benchmark, oga=oga,
                           guard=guard, native=Native, native_refs=native_refs,
                           original_load=load_model, original_write=write,
                           original_metadata=original_metadata, actions=actions,
                           kwargs_seen=kwargs_seen)


def run(setup):
    return observed.run_observed(setup.benchmark, setup.oga, setup.guard,
                                 model_path=setup.model, output=setup.output)


def report(setup):
    return json.loads((setup.output / "observations.json").read_text())


def assert_restored(setup):
    assert setup.benchmark.backend.load_model is setup.original_load
    assert setup.benchmark.write is setup.original_write
    assert setup.oga.Generator is setup.native


def test_complete_run_keeps_protocol_values_and_restores_every_patch(setup):
    run(setup)
    result = report(setup)
    manifest = json.loads((setup.output / "manifest.json").read_text())
    assert result["status"] == "complete"
    assert all(result[key] for key in ("code_unchanged", "runtime_unchanged",
                                      "package_manifest_unchanged", "model_artifacts_unchanged"))
    assert result["generator_observations"] == [{"generator_index": 1, "finite": True}]
    assert manifest["unchanged_metric"] == 4.5
    assert json.loads((setup.output / "summary.json").read_text()) == {"median_seconds": 1.25}
    assert "Full-vocabulary" in manifest["timing_scope"]
    assert "CPU graph/host" in manifest["model"]["execution"]
    assert setup.original_metadata["execution"] == "original description"
    assert setup.kwargs_seen[0]["vocab_size"] == 7
    assert setup.kwargs_seen[0]["max_generated_tokens"] == 128
    assert setup.kwargs_seen[0]["hash_logits"] is False
    assert setup.actions.count("load") == 1
    measured = setup.actions[setup.actions.index("main-start"):setup.actions.index("main-end") + 1]
    assert "code-hashes" not in measured and "runtime-hashes" not in measured
    assert "write:observations.json" not in measured
    assert setup.actions.index("write:observations.json") > setup.actions.index("guard-exit")
    assert all(ref() is None for ref in setup.native_refs)
    assert_restored(setup)


@pytest.mark.parametrize("error", [RuntimeError("benchmark failed"), KeyboardInterrupt(), SystemExit(3)])
def test_failure_preserves_partial_observations_and_reraises_original(setup, error):
    def fail():
        setup.output.mkdir()
        setup.benchmark.backend.load_model()
        raise error

    setup.benchmark.main = fail
    with pytest.raises(type(error)) as caught:
        run(setup)
    assert caught.value is error
    result = report(setup)
    assert result["status"] == "failed" and result["error"]["type"] == type(error).__name__
    assert result["generator_observations"]
    assert "code_unchanged" not in result
    assert all(ref() is None for ref in setup.native_refs)
    assert_restored(setup)


def test_failure_before_benchmark_mkdir_still_records_failure(setup):
    def fail():
        raise RuntimeError("before output creation")

    setup.benchmark.main = fail
    with pytest.raises(RuntimeError, match="before output creation"):
        run(setup)
    assert report(setup)["status"] == "failed"
    assert_restored(setup)


def test_existing_output_is_rejected_without_mutation_or_model_load(setup):
    setup.output.mkdir()
    sentinel = setup.output / "keep.txt"
    sentinel.write_bytes(b"original")
    with pytest.raises(ValueError, match="must be new"):
        run(setup)
    assert list(setup.output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"original" and setup.actions == []
    assert_restored(setup)


@pytest.mark.parametrize("kind", ["code", "runtime", "model"])
def test_postrun_identity_changes_fail_closed(setup, monkeypatch, kind):
    if kind in ("code", "runtime"):
        calls = iter(({"file": "before"}, {"file": "after"}))
        monkeypatch.setattr(observed, kind + "_hashes", lambda *args: next(calls))
    else:
        setup.benchmark.backend._artifact_hashes = lambda path: {"model.onnx": "changed"}
    with pytest.raises(RuntimeError, match="artifacts changed"):
        run(setup)
    result = report(setup)
    assert result["status"] == "failed"
    key = "model_artifacts_unchanged" if kind == "model" else kind + "_unchanged"
    assert result[key] is False
    assert_restored(setup)


def test_missing_native_load_cannot_be_reported_complete(setup):
    setup.benchmark.main = lambda: setup.output.mkdir()
    with pytest.raises(RuntimeError, match="artifacts changed"):
        run(setup)
    assert report(setup)["model_artifacts_unchanged"] is False
    assert_restored(setup)


def test_optional_package_manifest_is_pinned_when_present(setup):
    path = setup.model / "package-manifest.json"
    path.write_bytes(b'{"owned": true}\n')
    run(setup)
    result = report(setup)
    assert result["provenance"]["package_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["package_manifest_unchanged"] is True


def test_package_manifest_change_fails_even_if_model_artifact_hashes_are_equal(setup):
    path = setup.model / "package-manifest.json"
    path.write_bytes(b'{"owned": true}\n')
    original = setup.benchmark.main

    def mutate():
        original()
        path.write_bytes(b'{"owned": false}\n')

    setup.benchmark.main = mutate
    with pytest.raises(RuntimeError):
        run(setup)
    assert report(setup)["status"] == "failed"
    assert_restored(setup)


def test_official_artifact_without_package_manifest_is_supported(setup):
    run(setup)
    assert report(setup)["provenance"]["package_manifest_sha256"] is None


def test_runtime_hash_inventory_reads_only_native_binary_files(tmp_path):
    (tmp_path / "entry.py").write_bytes(b"python")
    (tmp_path / "runtime.DLL").write_bytes(b"dll")
    (tmp_path / "extension.pyd").write_bytes(b"pyd")
    (tmp_path / "weights.bin").write_bytes(b"not runtime")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "dep.dll").write_bytes(b"dependency")
    result = observed.runtime_hashes(SimpleNamespace(__file__=str(tmp_path / "entry.py")))
    assert set(result) == {"runtime.DLL", "extension.pyd", "sub/dep.dll"}
    assert result["runtime.DLL"] == hashlib.sha256(b"dll").hexdigest()
