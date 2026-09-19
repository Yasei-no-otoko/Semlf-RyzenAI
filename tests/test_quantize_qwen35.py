from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "quantize_qwen35.py"
SPEC = importlib.util.spec_from_file_location("quantize_qwen35", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
quantize_qwen35 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantize_qwen35)


def test_conversion_rejects_an_unreviewed_quark_release(monkeypatch):
    monkeypatch.setattr(quantize_qwen35, "_quark_version", lambda: "0.12")
    with pytest.raises(RuntimeError, match="requires reviewed AMD Quark 0.11, found 0.12"):
        quantize_qwen35._require_quark_version()


def test_conversion_accepts_the_reviewed_quark_release(monkeypatch):
    monkeypatch.setattr(quantize_qwen35, "_quark_version", lambda: "0.11")
    assert quantize_qwen35._require_quark_version() == "0.11"


@pytest.fixture
def pinned_source(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}),
        encoding="utf-8",
    )
    shards = {}
    for name in quantize_qwen35.SOURCE_WEIGHT_SHA256:
        (tmp_path / name).write_bytes(name.encode())
        shards[name] = quantize_qwen35._sha256(tmp_path / name)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {str(index): name for index, name in enumerate(shards)}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(quantize_qwen35, "SOURCE_CONFIG_SHA256", quantize_qwen35._sha256(tmp_path / "config.json"))
    monkeypatch.setattr(quantize_qwen35, "SOURCE_WEIGHT_SHA256", shards)
    return tmp_path


def test_source_validation_hashes_both_shards_and_index(pinned_source):
    hashes = quantize_qwen35._require_source(pinned_source, quantize_qwen35.SOURCE_REVISION)
    assert set(hashes) == {"config.json", "model.safetensors.index.json", *quantize_qwen35.SOURCE_WEIGHT_SHA256}
    for name, expected in quantize_qwen35.SOURCE_WEIGHT_SHA256.items():
        assert hashes[name] == expected
    first_shard = next(iter(quantize_qwen35.SOURCE_WEIGHT_SHA256))
    (pinned_source / first_shard).write_bytes(b"changed source weight")
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        quantize_qwen35._require_source(pinned_source, quantize_qwen35.SOURCE_REVISION)


def test_source_validation_rejects_different_revision_and_config(pinned_source):
    with pytest.raises(ValueError, match="reviewed Qwen3.5-4B commit"):
        quantize_qwen35._require_source(pinned_source, "0" * 40)
    config_path = pinned_source / "config.json"
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source SHA256 mismatch for config.json"):
        quantize_qwen35._require_source(pinned_source, quantize_qwen35.SOURCE_REVISION)


def test_source_validation_rejects_unbound_loader_inputs(pinned_source):
    alternate = pinned_source / "model.safetensors"
    alternate.write_bytes(b"would override the sharded checkpoint")
    with pytest.raises(ValueError, match="exactly the two pinned"):
        quantize_qwen35._require_source(pinned_source, quantize_qwen35.SOURCE_REVISION)
    alternate.unlink()
    (pinned_source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "unbound.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="weight index must reference"):
        quantize_qwen35._require_source(pinned_source, quantize_qwen35.SOURCE_REVISION)
