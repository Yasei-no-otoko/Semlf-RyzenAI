"""Offline integrity checks for Ryzen AI evidence."""

import importlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def verifier():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        yield importlib.import_module("verify_ryzenai")
    finally:
        sys.path.remove(str(ROOT / "benchmarks"))


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _owned_run(tmp_path, verifier):
    gold = verifier.read_jsonl(ROOT / "benchmarks/data/authored144.jsonl")
    model = {
        "backend": "ryzenai-npu", "oga_distribution": "onnxruntime-genai-directml-ryzenai",
        "oga_version": "0.14.0", "revision": "0" * 40, "context_ceiling": 4096,
        "source_artifact_sha256": {"genai_config.json": "a" * 64},
    }
    predictions = []
    for row in gold:
        ids = [option["id"] for option in row["options"]]
        probabilities = [0.0] * len(ids)
        probabilities[row["label"]] = 1.0
        predictions.append({
            "id": row["id"], "option_ids": ids, "probabilities": probabilities,
            "option_logits": [0.0] * len(ids), "input_tokens": 12,
            "prompt_sha256": "b" * 64, "model_reference": "manifest.json#/model",
        })
    (tmp_path / "authored144.predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    quality = {"authored144": importlib.import_module("evaluate").evaluate(gold, predictions)}
    _write_json(tmp_path / "quality.json", quality)
    manifest = {
        "version": "ryzenai-benchmark-v1", "model": model,
        "hardware": {"runtime_versions": {
            "onnxruntime-genai-directml-ryzenai": "0.14.0", "onnxruntime-vitisai": "1.27.0",
            "onnxruntime-providers-ryzenai": "1.8.0", "ryzenai-dynamic-dispatch": "1.8.0", "voe": "1.8.0",
        }},
        "inputs": {"authored144": {
            "file": "authored144.jsonl", "sha256": verifier.digest(ROOT / "benchmarks/data/authored144.jsonl"),
            "rows": 144,
        }},
    }
    _write_json(tmp_path / "manifest.json", manifest)
    return predictions


def test_owned_quality_verifies_without_external_source_snapshots(tmp_path, verifier):
    _owned_run(tmp_path, verifier)

    result = verifier.verify(tmp_path)

    assert result["verified_quality_sets"] == ["authored144"]
    assert result["external_unverified"] == ["wanli256", "typesafe102", "every204"]
    assert result["status"] == "ok"


def test_verifier_catches_tampered_prediction_probability(tmp_path, verifier):
    _owned_run(tmp_path, verifier)
    path = tmp_path / "authored144.predictions.jsonl"
    rows = verifier.read_jsonl(path)
    rows[0]["probabilities"] = [0.7, 0.7] if len(rows[0]["option_ids"]) == 2 else [0.7, 0.2, 0.2]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="Probabilities do not sum"):
        verifier.verify(tmp_path)


def test_verifier_rejects_an_unpinned_model_revision(tmp_path, verifier):
    _owned_run(tmp_path, verifier)
    manifest = verifier.read_json(tmp_path / "manifest.json")
    manifest["model"]["revision"] = "main"
    _write_json(tmp_path / "manifest.json", manifest)

    with pytest.raises(AssertionError, match="40-character"):
        verifier.verify(tmp_path)


def test_context_limit_rows_must_match_the_preflight_population(tmp_path, verifier):
    gold = [{"id": "over", "options": [{"id": "yes"}, {"id": "no"}]}]
    model = {"context_ceiling": 4096}
    prediction = {
        "id": "over", "option_ids": ["yes", "no"], "input_tokens": 5000,
        "error_type": "context_limit", "context_limit": 4096,
        "error": "no truncation", "model_reference": "manifest.json#/model",
    }
    path = tmp_path / "typesafe102.predictions.jsonl"
    path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")

    verifier.verify_predictions(path, gold, model, {"over": 5000})
    with pytest.raises(AssertionError, match="Unexpected context rejection"):
        verifier.verify_predictions(path, gold, model, {})


def test_xrt_decimal_string_counters_are_compared_numerically(tmp_path, verifier):
    _owned_run(tmp_path, verifier)
    counters = {
        "before": {"available": True, "benchmark_contexts": [
            {"context_id": "1", "command_submissions": "9", "errors": "0"}]},
        "after": {"available": True, "benchmark_contexts": [
            {"context_id": "1", "command_submissions": "10", "errors": "0"}]},
    }
    _write_json(tmp_path / "hardware.json", counters)
    assert verifier.verify(tmp_path)["status"] == "ok"
    counters["after"]["benchmark_contexts"][0]["errors"] = "1"
    _write_json(tmp_path / "hardware.json", counters)
    with pytest.raises(AssertionError, match="hardware errors"):
        verifier.verify(tmp_path)


@pytest.mark.parametrize("value", [None, True, -1, "-1", "1.0", "unknown"])
def test_invalid_hardware_counters_fail_closed(value, verifier):
    with pytest.raises(AssertionError, match="hardware counter"):
        verifier.hardware_counter({"errors": value}, "errors")
