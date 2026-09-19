from __future__ import annotations

import json
import sys
from types import ModuleType

import onnx

import pytest

from benchmarks.prepare_qwen35 import (
    _required_absolute_option,
    _replace_from_tmp,
    _set_eager_search_options,
    _validate_chunk_size,
    assert_ryzenai_genai_config,
    finalize_existing,
    install_precision_free_cast_recovery,
    main,
    prepare,
)


def test_recovers_real_onnx_extractor_value_info_and_rejects_wrong_cast():
    matcher = pytest.importorskip("ryzenai_onnx_utils.matcher")
    prefix = "InsertedPrecisionFreeCast_"
    raw = prefix + "linear/output"
    inputs = [onnx.helper.make_tensor_value_info(f"in{i}", onnx.TensorProto.FLOAT, [1, 2]) for i in range(6)]
    linear = onnx.helper.make_node(
        "LinearAttention", inputs=[x.name for x in inputs], outputs=[raw], domain="com.microsoft", update_rule="gated_delta"
    )
    cast = onnx.helper.make_node("Cast", [raw], ["linear/output"], to=onnx.TensorProto.FLOAT16)
    canonical = onnx.helper.make_tensor_value_info("linear/output", onnx.TensorProto.FLOAT16, [1, 2])
    model = onnx.helper.make_model(onnx.helper.make_graph([linear, cast], "test", inputs, [], value_info=[canonical]))
    extractor = matcher.get_extractor(model)

    install_precision_free_cast_recovery()
    assert matcher.get_dtype(raw, extractor) == onnx.TensorProto.FLOAT
    assert matcher.get_shape(raw, extractor) == (1, 2)

    bad_raw = prefix + "linear/bad"
    bad_linear = onnx.helper.make_node(
        "LinearAttention", inputs=[x.name for x in inputs], outputs=[bad_raw], domain="com.microsoft", update_rule="gated_delta"
    )
    bad_cast = onnx.helper.make_node("Cast", [bad_raw], ["linear/bad"], to=onnx.TensorProto.FLOAT)
    bad_canonical = onnx.helper.make_tensor_value_info("linear/bad", onnx.TensorProto.FLOAT16, [1, 2])
    bad_model = onnx.helper.make_model(
        onnx.helper.make_graph([bad_linear, bad_cast], "bad", inputs, [], value_info=[bad_canonical])
    )
    with pytest.raises(ValueError, match="FP32-to-FP16"):
        matcher.get_dtype(bad_raw, matcher.get_extractor(bad_model))


def test_requires_absolute_optimizer_model_paths():
    assert _required_absolute_option(["--input-model", "C:\\model.onnx"], "--input-model").is_absolute()
    with pytest.raises(ValueError, match="must be absolute"):
        _required_absolute_option(["--input-model", "model.onnx"], "--input-model")


def test_wrapper_keeps_global_onnx_utils_flags_before_optimize(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "benchmarks.prepare_qwen35.run_onnx_utils",
        lambda args, work_dir: captured.update(args=args, work_dir=work_dir),
    )
    main(["--work-dir", str(tmp_path), "-v", "optimize", "llm"])
    assert captured["args"] == ["-v", "optimize", "llm"]


def test_requires_exact_16k_qwen35_ryzenai_config(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"model")
    (tmp_path / "genai_config.json").write_text(
        """{
          "model": {"type": "qwen3_5_text", "context_length": 40960,
            "decoder": {"session_options": {"provider_options": [
              {"RyzenAI": {"hybrid_opt_token_backend": "npu", "hybrid_opt_chunk_context": "1", "hybrid_opt_max_seq_length": "4096"}}
            ]}}},
          "search": {"max_length": 16384, "chunk_size": 4096, "past_present_share_buffer": true}
        }""",
        encoding="utf-8",
    )
    assert_ryzenai_genai_config(tmp_path, chunk_size=4096)

    config_path = tmp_path / "genai_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"]["decoder"]["session_options"]["provider_options"].append({"DmlExecutionProvider": {}})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly one RyzenAI"):
        assert_ryzenai_genai_config(tmp_path, chunk_size=4096)


def test_prepare_is_create_only(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    with pytest.raises(ValueError, match="new path"):
        prepare(source, output, tmp_path)


def test_prepare_rejects_output_nested_in_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="separate directories"):
        prepare(source, source / "output", tmp_path)


@pytest.mark.parametrize("work_parent", ["source", "output"])
@pytest.mark.parametrize("finalize_existing_only", [False, True])
def test_prepare_rejects_work_dir_inside_model_directories(tmp_path, work_parent, finalize_existing_only):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    work_dir = tmp_path / work_parent / "diagnostics"
    work_dir.mkdir()
    with pytest.raises(ValueError, match="separate directories"):
        prepare(source, output, work_dir, finalize_existing_only=finalize_existing_only)
    with pytest.raises(ValueError, match="separate directories"):
        finalize_existing(source, output, work_dir, {"chunk_size": 64})
    assert list(work_dir.iterdir()) == []


def test_prepare_uses_fixed_shape_npu_eager_pipeline(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"source")
    captured = {}

    def fake_eager(input_model, output_model, work_dir, *, chunk_size):
        captured.update(input_model=input_model, output_model=output_model, work_dir=work_dir, chunk_size=chunk_size)

    monkeypatch.setattr("benchmarks.prepare_qwen35._run_fixed_shape_npu_eager", fake_eager)
    monkeypatch.setattr(
        "benchmarks.prepare_qwen35.finalize_existing",
        lambda source, output, work_dir, settings: captured.update(
            source=source, finalized_output=output, settings=settings
        ) or output / "conversion-manifest.json",
    )
    output = tmp_path / "output"
    prepare(source, output, tmp_path)

    assert captured["input_model"] == (output / "model.onnx").resolve()
    assert captured["output_model"] == (output / "tmp" / "model.onnx").resolve()
    assert captured["settings"]["prefill"] == "npu_eager"
    assert captured["settings"]["token"] == "npu_eager"
    assert captured["chunk_size"] == 64
    assert captured["settings"]["provider_max_seq_length"] == 4096


def test_chunk_size_is_bounded_and_configurable(tmp_path):
    assert _validate_chunk_size(64) == 64
    with pytest.raises(ValueError, match="1 through 4096"):
        _validate_chunk_size(0)
    with pytest.raises(ValueError, match="1 through 4096"):
        _validate_chunk_size(4097)

    (tmp_path / "model.onnx").write_bytes(b"model")
    (tmp_path / "genai_config.json").write_text(
        """{"model":{"type":"qwen3_5_text","context_length":16384,"decoder":{"session_options":{"provider_options":[{"RyzenAI":{"hybrid_opt_token_backend":"npu","hybrid_opt_chunk_context":"1","hybrid_opt_max_seq_length":"4096"}}]}}},"search":{"max_length":16384,"chunk_size":64,"past_present_share_buffer":true}}""",
        encoding="utf-8",
    )
    assert_ryzenai_genai_config(tmp_path, chunk_size=64)


def test_eager_search_options_set_16k_ceiling_and_selected_chunk():
    class Config:
        def __init__(self):
            self.options = {}

        def set_search_option(self, name, value):
            self.options[name] = value

    config = Config()
    _set_eager_search_options(config, 64)
    assert config.options == {"max_length": 16384, "chunk_size": 64}


def test_promotion_preserves_sdk_diagnostics_and_keeps_runtime_bin(tmp_path):
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    (tmp / "model.onnx").write_bytes(b"model")
    (tmp / "model.bin").write_bytes(b"runtime")
    (tmp / "genai_config.json").write_text("{}", encoding="utf-8")
    (tmp / "optimized_model.onnx").write_bytes(b"intermediate")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _replace_from_tmp(tmp_path, work_dir)
    assert (tmp_path / "model.bin").read_bytes() == b"runtime"
    assert (work_dir / f"{tmp_path.name}-sdk-diagnostics" / "optimized_model.onnx").read_bytes() == b"intermediate"


def _write_eager_payload(model_dir, *, chunk_size):
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("model.onnx", "model.onnx.data", "model.bin", "model.pb.bin"):
        (model_dir / filename).write_bytes(f"optimizer output: {filename}".encode())
    config = {
        "model": {
            "type": "qwen3_5_text",
            "context_length": 16384,
            "decoder": {
                "session_options": {
                    "provider_options": [{"RyzenAI": {
                        "hybrid_opt_token_backend": "npu",
                        "hybrid_opt_chunk_context": "1",
                        "hybrid_opt_max_seq_length": "4096",
                    }}],
                },
            },
        },
        "search": {"max_length": 16384, "chunk_size": chunk_size, "past_present_share_buffer": True},
    }
    (model_dir / "genai_config.json").write_text(json.dumps(config), encoding="utf-8")


def test_finalize_wrong_chunk_preserves_complete_staging_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    tmp = output / "tmp"
    _write_eager_payload(tmp, chunk_size=4096)
    (output / "model.onnx").write_bytes(b"original copied OGA model")
    (tmp / "sdk-diagnostic.txt").write_text("keep this", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}

    with pytest.raises(RuntimeError, match="search.chunk_size must equal 64"):
        prepare(source, output, work_dir, finalize_existing_only=True)

    assert tmp.is_dir()
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == before
    assert list(work_dir.iterdir()) == []


def test_finalize_retries_after_sdk_error_without_repeating_promotion(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"source OGA")
    output = tmp_path / "output"
    tmp = output / "tmp"
    _write_eager_payload(tmp, chunk_size=64)
    expected_payload = {p.name: p.read_bytes() for p in tmp.iterdir()}
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    calls = []

    def fake_finalize(output_dir, model_name, *, model_type):
        calls.append("finalize")
        assert output_dir == output
        assert model_name == "model.onnx"
        assert model_type == "qwen3.5"
        (output_dir / "rai_config.json").write_text("{}", encoding="utf-8")
        if calls.count("finalize") == 1:
            raise RuntimeError("simulated SDK finalization failure")

    package = ModuleType("model_generate")
    package.__path__ = []
    runner = ModuleType("model_generate.runner")
    runner.finalize_output = fake_finalize
    filtering = ModuleType("model_generate.filtering")
    filtering.resolve_dyn_bins = lambda: tmp_path / "dyn_bins.zip"
    filtering.filter_bins = lambda *args, **kwargs: calls.append("filter")
    for module in (package, runner, filtering):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(RuntimeError, match="simulated SDK"):
        prepare(source, output, work_dir, finalize_existing_only=True)
    assert not tmp.exists()
    assert not (output / "conversion-manifest.json").exists()
    assert {name: (output / name).read_bytes() for name in expected_payload} == expected_payload

    manifest_path = prepare(source, output, work_dir, finalize_existing_only=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert calls == ["finalize", "finalize", "filter"]
    assert manifest["settings"]["chunk_size"] == 64
    assert set(expected_payload) <= manifest["output_artifact_sha256"].keys()
    assert {name: (output / name).read_bytes() for name in expected_payload} == expected_payload
    with pytest.raises(RuntimeError, match="conversion-manifest.json already exists"):
        prepare(source, output, work_dir, finalize_existing_only=True)


def test_finalize_rejects_incomplete_promoted_payload(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    _write_eager_payload(output, chunk_size=64)
    (output / "model.pb.bin").unlink()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    with pytest.raises(RuntimeError, match="incomplete optimizer output: missing model.pb.bin"):
        prepare(source, output, work_dir, finalize_existing_only=True)
    assert not (output / "conversion-manifest.json").exists()
