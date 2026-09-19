from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "export_qwen35.py"
SPEC = importlib.util.spec_from_file_location("export_qwen35", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
export_qwen35 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_qwen35)


def _quark_attrs() -> dict:
    return {
        "config": {
            "global_quant_config": {"weight": {"dtype": "uint4", "group_size": 128}},
            "layer_quant_config": {},
        }
    }


def _linear_tensors(torch):
    tensors = {}
    prefix = "model.language_model.layers.0.linear_attn."
    for projection in export_qwen35.LINEAR_PROJECTIONS:
        tensors[prefix + projection + ".weight"] = torch.zeros((128, 1), dtype=torch.int32)
        tensors[prefix + projection + ".weight_scale"] = torch.ones((1, 8), dtype=torch.float32)
        tensors[prefix + projection + ".weight_zero_point"] = torch.zeros((1, 1), dtype=torch.int32)
    tensors[prefix + "conv1d.weight"] = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tensors[prefix + "A_log"] = torch.tensor([1.25, 2.5], dtype=torch.float32)
    tensors[prefix + "dt_bias"] = torch.tensor([0.25, 0.5], dtype=torch.float32)
    tensors[prefix + "norm.weight"] = torch.tensor([0.75, 1.25], dtype=torch.float32)
    tensors["mtp.layers.0.mlp.gate_proj.weight"] = torch.ones((2, 2), dtype=torch.float32)
    tensors["model.visual.encoder.weight"] = torch.ones((2, 2), dtype=torch.float32)
    return tensors


def test_qwen35_adapter_preserves_linear_attention_and_uses_quark_layout(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    quantized_model = pytest.importorskip("onnxruntime_genai.models.quantized_model")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    source_tensors = _linear_tensors(torch)
    source_tensors = {
        name.replace(".weight_scale", ".weight_quantizer.scale").replace(
            ".weight_zero_point", ".weight_quantizer.zero_point"
        ): tensor
        for name, tensor in source_tensors.items()
    }

    original_load_file = quantized_model.load_file
    monkeypatch.setattr(quantized_model, "load_file", lambda _path: source_tensors)
    try:
        with export_qwen35.qwen35_quark_adapter():
            model = quantized_model.QuantModel.from_pretrained(
                "quark",
                input_path=str(checkpoint),
                quant_attrs=_quark_attrs(),
                q_size=1,
                kv_size=1,
                intermediate_size=1,
                num_layers=1,
            )
    finally:
        monkeypatch.setattr(quantized_model, "load_file", original_load_file)

    linear = model.layers[0].linear_attn
    assert linear is not None
    assert torch.equal(linear.conv1d.weight, source_tensors["model.language_model.layers.0.linear_attn.conv1d.weight"])
    assert torch.equal(linear.A_log, source_tensors["model.language_model.layers.0.linear_attn.A_log"])
    assert torch.equal(linear.dt_bias, source_tensors["model.language_model.layers.0.linear_attn.dt_bias"])
    assert torch.equal(linear.norm.weight, source_tensors["model.language_model.layers.0.linear_attn.norm.weight"])
    for name in export_qwen35.LINEAR_PROJECTIONS:
        projection = getattr(linear, name)
        assert projection.bits == 4
        assert projection.group_size == 128
        assert projection.in_features == 128
        assert projection.out_features == 8
        assert projection.g_idx is None

        # Reproduce the reviewed OGA unpack/repack recipe from the original
        # Quark tensors.  Exact equality checks the emitted uint4 payload and
        # its scale/zero-point transformation, not merely field names.
        raw_prefix = f"model.language_model.layers.0.linear_attn.{name}"
        reference = quantized_model.QuantizedTensorModule()
        reference.qweight = source_tensors[raw_prefix + ".weight"].clone()
        reference.scales = source_tensors[raw_prefix + ".weight_quantizer.scale"].clone()
        reference.qzeros = source_tensors[raw_prefix + ".weight_quantizer.zero_point"].clone()
        reference.bits = 4
        reference.group_size = 128
        reference.in_features = 128
        reference.out_features = 8
        model.set_g_idx(reference)
        model.unpack(reference)
        model.repack(reference)
        assert torch.equal(projection.qweight, reference.qweight)
        assert torch.equal(projection.scales, reference.scales)
        assert torch.equal(projection.qzeros, reference.qzeros)

    # The bounded adapter packer must be byte-for-byte equivalent to the
    # reviewed upstream routine for both output types and transpose modes.
    for packed_dtype in (torch.int32, torch.uint8):
        for transpose in (False, True):
            values = torch.randint(0, 16, (257, 19), dtype=torch.int32)
            expected = quantized_model.QuarkModel.pack_on_row_for_2_4_8_bits(
                model, values, 4, transpose, packed_dtype
            )
            actual = model.pack_on_row_for_2_4_8_bits(values, 4, transpose, packed_dtype)
            assert torch.equal(actual, expected)


def test_adapter_restores_upstream_factory_after_failure():
    quantized_model = pytest.importorskip("onnxruntime_genai.models.quantized_model")
    original_factory = quantized_model.QuantModel.from_pretrained
    with pytest.raises(RuntimeError, match="stop"):
        with export_qwen35.qwen35_quark_adapter():
            raise RuntimeError("stop")
    assert quantized_model.QuantModel.from_pretrained is original_factory


def test_checkpoint_validation_requires_conditional_outer_architecture(tmp_path):
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForCausalLM"],
        "text_config": {"model_type": "qwen3_5_text"},
        "quantization_config": {
            "quant_method": "quark",
            "global_quant_config": {"weight": {"dtype": "uint4", "group_size": 128}},
        },
    }
    (tmp_path / "config.json").write_text(__import__("json").dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Qwen3_5ForConditionalGeneration"):
        export_qwen35._require_checkpoint(tmp_path, export_qwen35.SOURCE_REVISION)


def test_projection_width_must_fit_quark_uint4_words():
    torch = pytest.importorskip("torch")

    class Projection:
        qweight = torch.zeros((128, 1), dtype=torch.int32)
        scales = torch.ones((1, 1), dtype=torch.float32)
        bits = 4
        group_size = 128
        in_features = 0
        out_features = 0

    class Model:
        def set_g_idx(self, _projection):
            raise AssertionError("invalid width must fail before g_idx setup")

    with pytest.raises(ValueError, match="divisible by 8"):
        export_qwen35._set_projection_properties(Model(), Projection())


def test_graph_repair_connects_quantized_projection_without_touching_external_weights(tmp_path):
    onnx = pytest.importorskip("onnx")
    produced = "/layer/proj/MatMulNBits/output_0"
    consumed = "/layer/proj/MatMul/output_0"
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMulNBits", ["input"], [produced], domain="com.microsoft"),
            onnx.helper.make_node("Identity", [consumed], ["output"]),
        ],
        "repair-fixture",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    onnx.save_model(onnx.helper.make_model(graph), tmp_path / "model.onnx")
    payload = tmp_path / "model.onnx.data"
    payload.write_bytes(b"external-weights-must-stay-identical")

    assert export_qwen35.repair_int4_matmul_output_names(tmp_path) == 1
    repaired = onnx.load(tmp_path / "model.onnx", load_external_data=False)
    assert repaired.graph.node[0].output[0] == repaired.graph.node[1].input[0] == consumed
    assert payload.read_bytes() == b"external-weights-must-stay-identical"
    assert export_qwen35.repair_int4_matmul_output_names(tmp_path) == 0


def test_graph_repair_rejects_unrelated_missing_input_without_saving(tmp_path):
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["missing"], ["output"])],
        "invalid-fixture", [],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "model.onnx"
    onnx.save_model(onnx.helper.make_model(graph), path)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="unresolved inputs"):
        export_qwen35.repair_int4_matmul_output_names(tmp_path)
    assert path.read_bytes() == original


@pytest.fixture
def manifested_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {
        "model_type": "qwen3_5",
        "architectures": [export_qwen35.OUTER_ARCHITECTURE],
        "text_config": {
            "model_type": "qwen3_5_text", "head_dim": 256, "partial_rotary_factor": 0.25,
            "rope_parameters": {"mrope_section": [11, 11, 10]},
        },
        "quantization_config": {
            "quant_method": "quark",
            "global_quant_config": {"weight": {"dtype": "uint4", "group_size": 128}},
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"packed weights")
    manifest = {
        "source_revision": export_qwen35.SOURCE_REVISION,
        "source_config_sha256": export_qwen35.SOURCE_CONFIG_SHA256,
        "quantizer": "amd-quark", "quantizer_version": "0.11",
        "method": "uint4_rtn_minmax_weight_only",
        "output_file_sha256": {
            path.name: export_qwen35._sha256(path) for path in checkpoint.iterdir()
        },
    }
    (checkpoint / "quantize-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint


def test_export_requires_manifest_binding_actual_weights(manifested_checkpoint):
    hashes = export_qwen35._require_quantize_manifest(manifested_checkpoint, export_qwen35.SOURCE_REVISION)
    manifest_path = manifested_checkpoint / "quantize-manifest.json"
    assert hashes[manifest_path.name] == export_qwen35._sha256(manifest_path)
    (manifested_checkpoint / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="SHA256 mismatch for model.safetensors"):
        export_qwen35._require_quantize_manifest(manifested_checkpoint, export_qwen35.SOURCE_REVISION)


@pytest.mark.parametrize("field,value", [("source_revision", "0" * 40), ("source_config_sha256", "0" * 64)])
def test_export_rejects_false_source_provenance(manifested_checkpoint, field, value):
    path = manifested_checkpoint / "quantize-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned source"):
        export_qwen35._require_quantize_manifest(manifested_checkpoint, export_qwen35.SOURCE_REVISION)


def test_export_rejects_unbound_additional_weights(manifested_checkpoint):
    (manifested_checkpoint / "extra.safetensors").write_bytes(b"unmanifested weights")
    with pytest.raises(ValueError, match="checkpoint files differ"):
        export_qwen35._require_quantize_manifest(manifested_checkpoint, export_qwen35.SOURCE_REVISION)


def test_export_manifest_binds_inputs_loader_and_repaired_outputs(manifested_checkpoint, tmp_path, monkeypatch):
    import contextlib
    import sys
    from types import ModuleType

    output = tmp_path / "export"
    builder = ModuleType("onnxruntime_genai.models.builder")

    def create_model(**kwargs):
        assert kwargs["input_path"] == str(manifested_checkpoint)
        output.mkdir()
        (output / "model.onnx").write_bytes(b"unrepaired graph")
        (output / "model.onnx.data").write_bytes(b"external weights")
        (output / "genai_config.json").write_text("{}", encoding="utf-8")

    def repair(directory):
        (directory / "model.onnx").write_bytes(b"repaired graph")
        return 248

    builder.create_model = create_model
    monkeypatch.setitem(sys.modules, "onnxruntime_genai.models.builder", builder)
    monkeypatch.setattr(export_qwen35, "qwen35_quark_adapter", lambda _sha: contextlib.nullcontext())
    monkeypatch.setattr(export_qwen35, "repair_int4_matmul_output_names", repair)
    export_qwen35.export_oga(
        manifested_checkpoint, output, export_qwen35.SOURCE_REVISION,
        export_qwen35.PUBLIC_OGA_014_QUANTIZED_MODEL_SHA256,
    )
    manifest_path = output / "export-manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    assert manifest["source_revision"] == export_qwen35.SOURCE_REVISION
    assert manifest["public_oga_quantized_loader_sha256"] == export_qwen35.PUBLIC_OGA_014_QUANTIZED_MODEL_SHA256
    assert manifest["input_file_sha256"]["model.safetensors"] == export_qwen35._sha256(manifested_checkpoint / "model.safetensors")
    assert manifest["output_file_sha256"]["model.onnx"] == export_qwen35._sha256(output / "model.onnx")
    assert manifest["output_file_sha256"]["model.onnx.data"] == export_qwen35._sha256(output / "model.onnx.data")
    assert manifest["repaired_matmul_output_names"] == 248
    with pytest.raises(FileExistsError):
        export_qwen35.export_oga(
            manifested_checkpoint, output, export_qwen35.SOURCE_REVISION,
            export_qwen35.PUBLIC_OGA_014_QUANTIZED_MODEL_SHA256,
        )
    assert manifest_path.read_bytes() == original
    assert not list(tmp_path.glob(".export-builder-*"))
