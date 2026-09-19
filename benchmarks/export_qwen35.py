"""Export a Quark-quantized Qwen3.5 text decoder through public OGA 0.14.

The public OGA 0.14 loader understands Quark checkpoints for ordinary
attention and MLP projections, but its ``QuantizedModel`` does not yet model
Qwen3.5's GatedDeltaNet ``linear_attn`` component.  This entry point installs
a process-local adapter around that loader.  It delegates all existing keys to
the upstream implementation and handles only the missing Qwen3.5 tensors.

Run in the isolated conversion environment.  It never modifies the Ryzen AI
SDK or the project runtime environment.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


SOURCE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_CONFIG_SHA256 = "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
OUTER_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
TEXT_MODEL_TYPE = "qwen3_5_text"
PUBLIC_OGA_014_QUANTIZED_MODEL_SHA256 = "c9eb4254be202300fc071bec8e5cc291133223b98a9cb38b89504cc1d836f086"
LINEAR_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
PACK_ROWS = 128
_PROJECTION_KEY = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.linear_attn\."
    r"(?P<projection>in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\."
    r"(?P<parameter>qweight|weight|scales|weight_scale|qzeros|weight_zero_point|g_idx|bias)$"
)
_RAW_LINEAR_KEY = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.linear_attn\."
    r"(?P<parameter>conv1d\.weight|conv1d\.bias|A_log|dt_bias|norm\.weight|norm\.bias)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_checkpoint(source: Path, revision: str) -> dict[str, Any]:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise ValueError(f"checkpoint must contain config.json: {source}")
    if revision != SOURCE_REVISION:
        raise ValueError(f"source revision must be the reviewed Qwen3.5-4B commit {SOURCE_REVISION}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError("checkpoint config model_type must be qwen3_5")
    if config.get("architectures", [None])[0] != OUTER_ARCHITECTURE:
        raise ValueError(f"checkpoint architecture must remain {OUTER_ARCHITECTURE}")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or text_config.get("model_type") != TEXT_MODEL_TYPE:
        raise ValueError(f"checkpoint text_config.model_type must be {TEXT_MODEL_TYPE}")
    rope = text_config.get("rope_parameters")
    sections = rope.get("mrope_section") if isinstance(rope, dict) else None
    head_dim = text_config.get("head_dim")
    partial = text_config.get("partial_rotary_factor")
    if (
        not isinstance(sections, list)
        or len(sections) != 3
        or any(not isinstance(section, int) or isinstance(section, bool) or section <= 0 for section in sections)
        or not isinstance(head_dim, int)
        or isinstance(head_dim, bool)
        or not isinstance(partial, (int, float))
        or isinstance(partial, bool)
        or sum(sections) * 2 != head_dim * partial
    ):
        raise ValueError(
            "checkpoint text_config must provide three mrope_section values "
            "summing to half of head_dim * partial_rotary_factor"
        )
    quant = config.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("quant_method") != "quark":
        raise ValueError("checkpoint quantization_config.quant_method must be quark")
    weight = quant.get("global_quant_config", {}).get("weight")
    if not isinstance(weight, dict) or weight.get("dtype") != "uint4" or weight.get("group_size") != 128:
        raise ValueError("adapter supports the pinned Quark uint4 group_size=128 checkpoint only")
    return config


def _require_quantize_manifest(source: Path, revision: str) -> dict[str, str]:
    """Bind the pinned source claim to every actual Quark checkpoint file."""

    path = source / "quantize-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source_revision") != revision or revision != SOURCE_REVISION:
        raise ValueError("quantize manifest source revision does not match the pinned source")
    if manifest.get("source_config_sha256") != SOURCE_CONFIG_SHA256:
        raise ValueError("quantize manifest source config SHA256 does not match the pinned source")
    if (
        manifest.get("quantizer") != "amd-quark"
        or manifest.get("quantizer_version") != "0.11"
        or manifest.get("method") != "uint4_rtn_minmax_weight_only"
    ):
        raise ValueError("quantize manifest must describe the reviewed Quark 0.11 RTN conversion")
    hashes = manifest.get("output_file_sha256")
    if not isinstance(hashes, dict) or not {"config.json", "model.safetensors"}.issubset(hashes):
        raise ValueError("quantize manifest must bind config.json and model.safetensors")
    actual_names = {
        item.relative_to(source).as_posix()
        for item in source.rglob("*")
        if item.is_file() and item != path
    }
    if actual_names != set(hashes):
        raise ValueError("checkpoint files differ from quantize manifest output files")
    verified = {}
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError(f"invalid checkpoint path in quantize manifest: {name}")
        item = source / relative
        if not item.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"checkpoint path leaves source directory: {name}")
        actual = _sha256(item)
        if actual != expected:
            raise ValueError(f"quantize manifest SHA256 mismatch for {name}")
        verified[name] = actual
    verified[path.name] = _sha256(path)
    return verified


def _write_export_manifest(
    source: Path,
    output: Path,
    revision: str,
    input_hashes: dict[str, str],
    loader_sha256: str,
    repaired_outputs: int,
) -> None:
    output_hashes = {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "export-manifest.json"
    }
    if not {"model.onnx", "genai_config.json"}.issubset(output_hashes):
        raise RuntimeError("OGA export did not produce model.onnx and genai_config.json")
    manifest = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "source_revision": revision,
        "input_file_sha256": input_hashes,
        "public_oga_quantized_loader_sha256": loader_sha256,
        "builder_precision": "int4",
        "builder_execution_provider": "dml",
        "runtime_validation": "not_performed_by_export",
        "repaired_matmul_output_names": repaired_outputs,
        "output_file_sha256": output_hashes,
    }
    with (output / "export-manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


class _LinearAttention:
    """The subset of a Qwen3.5 GatedDeltaNet layer consumed by Qwen35TextModel."""

    def __init__(self, quantized_tensor: type[Any], tensor: type[Any]) -> None:
        for name in LINEAR_PROJECTIONS:
            setattr(self, name, quantized_tensor())
        self.conv1d = tensor()
        self.norm = tensor()
        self.A_log: Any = None
        self.dt_bias: Any = None


def _linear_parameter(name: str) -> tuple[int, str, str] | None:
    projection = _PROJECTION_KEY.fullmatch(name)
    if projection:
        return int(projection["layer"]), projection["projection"], projection["parameter"]
    raw = _RAW_LINEAR_KEY.fullmatch(name)
    if raw:
        return int(raw["layer"]), "", raw["parameter"]
    return None


def _canonical_tensor_name(name: str) -> str:
    """Apply the Quark aliases before selecting Qwen3.5 linear-attention keys."""

    return name.replace(".weight_quantizer.scale", ".weight_scale").replace(
        ".weight_quantizer.zero_point", ".weight_zero_point"
    )


def _is_ignored_non_text_key(name: str) -> bool:
    # Vision is already filtered by normalize_vlm_weight_name.  MTP is not a
    # part of Qwen35TextModel and the generic loader would otherwise reject it.
    return name.startswith(("mtp.", "model.mtp."))


def _set_projection_properties(model: Any, projection: Any) -> None:
    if projection.qweight is None:
        raise ValueError("Qwen3.5 linear-attention projection is missing qweight")
    if projection.scales is None:
        raise ValueError("Qwen3.5 linear-attention projection is missing weight_scale/scales")
    projection.out_features = projection.scales.shape[1]
    projection.in_features = projection.qweight.shape[0]
    # Quark's reviewed OGA uint4 reorder works on complete int32 words.  The
    # real Qwen3.5 projections are 32/4096/8192-wide; reject malformed toy
    # configurations here instead of surfacing its internal assertion.
    values_per_word = 32 // projection.bits
    if projection.out_features % values_per_word:
        raise ValueError(
            "Quark uint4 projection output width must be divisible by "
            f"{values_per_word}, got {projection.out_features}"
        )
    model.set_g_idx(projection)


def _set_linear_mlp_properties(model: Any, layer: Any) -> None:
    """Finish dense MLP setup when the upstream pass skips a linear layer."""

    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(layer.mlp, name)
        if projection.qweight is not None:
            _set_projection_properties(model, projection)


def _attach_linear_attention(model: Any, captured: dict[int, dict[str, Any]], quantized_model: ModuleType) -> None:
    """Attach captured Qwen3.5 tensors and reuse AMD's Quark unpack/repack path."""

    for layer_id, tensors in sorted(captured.items()):
        if layer_id >= model.num_layers:
            raise ValueError(f"linear-attention tensor refers to layer {layer_id}, beyond configured layers")
        if isinstance(model.layers, dict):
            layer = model.layers.get(layer_id)
            if layer is None:
                layer = quantized_model.QuantizedDecoderLayer(layer_id)
                model.layers[layer_id] = layer
        else:
            layer = next((item for item in model.layers if item.layer_id == layer_id), None)
            if layer is None:
                layer = quantized_model.QuantizedDecoderLayer(layer_id)
                model.layers.append(layer)
        linear = _LinearAttention(quantized_model.QuantizedTensorModule, quantized_model.TensorModule)
        layer.linear_attn = linear

        for key, tensor in tensors.items():
            parsed = _linear_parameter(key)
            assert parsed is not None
            _unused_layer, projection_name, parameter = parsed
            if projection_name:
                target = getattr(linear, projection_name)
                aliases = {
                    "weight": "qweight",
                    "qweight": "qweight",
                    "scales": "scales",
                    "weight_scale": "scales",
                    "qzeros": "qzeros",
                    "weight_zero_point": "qzeros",
                    "g_idx": "g_idx",
                    "bias": "bias",
                }
                setattr(target, aliases[parameter], tensor)
                target.bits = model.get_layer_bits(key)
                target.group_size = model.get_layer_group_size(key)
            elif parameter.startswith("conv1d."):
                setattr(linear.conv1d, parameter.removeprefix("conv1d."), tensor)
            elif parameter.startswith("norm."):
                setattr(linear.norm, parameter.removeprefix("norm."), tensor)
            else:
                setattr(linear, parameter, tensor)

        if linear.conv1d.weight is None or linear.norm.weight is None or linear.A_log is None or linear.dt_bias is None:
            raise ValueError(f"Qwen3.5 linear-attention layer {layer_id} is missing a required raw tensor")
        for projection_name in LINEAR_PROJECTIONS:
            projection = getattr(linear, projection_name)
            _set_projection_properties(model, projection)
            model.unpack(projection)
            model.repack(projection)
            projection.g_idx = None

    if not isinstance(model.layers, dict):
        model.layers.sort(key=lambda item: item.layer_id)


def _require_loader_hash(quantized_model: ModuleType, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError("expected quantized-model SHA256 must be 64 lowercase hexadecimal characters")
    actual = _sha256(Path(inspect.getfile(quantized_model)))
    if actual != expected_sha256:
        raise RuntimeError(
            "refusing to patch an unexpected onnxruntime_genai quantized_model.py "
            f"(expected {expected_sha256}, found {actual})"
        )


def repair_int4_matmul_output_names(output: Path) -> int:
    """Repair public OGA's Qwen3.5 int4 MatMulNBits consumer-name mismatch."""

    import onnx

    model_path = output / "model.onnx"
    model = onnx.load(model_path, load_external_data=False)
    produced = {name for node in model.graph.node for name in node.output if name}
    replacements: dict[str, str] = {}
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "MatMulNBits":
            continue
        for name in node.output:
            expected = name.replace("/MatMulNBits/output_", "/MatMul/output_")
            if expected == name:
                continue
            if expected in produced:
                raise RuntimeError(f"cannot repair colliding MatMul output name: {expected}")
            replacements[name] = expected
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
        for index, name in enumerate(node.output):
            if name in replacements:
                node.output[index] = replacements[name]
    for value in (*model.graph.value_info, *model.graph.output):
        if value.name in replacements:
            value.name = replacements[value.name]
    available = {
        *(name for node in model.graph.node for name in node.output if name),
        *(initializer.name for initializer in model.graph.initializer),
        *(graph_input.name for graph_input in model.graph.input),
    }
    unresolved = sorted(
        {
            name
            for node in model.graph.node
            for name in node.input
            if name and name not in available
        }
    )
    if unresolved:
        raise RuntimeError(f"refusing to save graph with unresolved inputs: {', '.join(unresolved[:5])}")
    if replacements:
        onnx.save_model(model, model_path)
    return len(replacements)


@contextlib.contextmanager
def qwen35_quark_adapter(expected_loader_sha256: str | None = None) -> Iterator[None]:
    """Temporarily route Quark loading through the Qwen3.5 adapter.

    The patch is restored even if OGA export raises.  It changes neither the
    installed wheel nor non-Quark model loading in this interpreter.
    """

    from onnxruntime_genai.models import quantized_model

    _require_loader_hash(quantized_model, expected_loader_sha256)
    missing_module = object()
    original_bare_module = sys.modules.get("quantized_model", missing_module)
    original_factory = quantized_model.QuantModel.from_pretrained
    original_decoder_init = quantized_model.QuantizedDecoderLayer.__init__

    def decoder_init(self: Any, layer_id: int) -> None:
        original_decoder_init(self, layer_id)
        self.linear_attn = None

    class Qwen35QuarkModel(quantized_model.QuarkModel):
        def pack_on_row_for_2_4_8_bits(
            self, tensor: Any, bits: int, transpose: bool, packed_dtype: Any = None
        ) -> Any:
            """Apply OGA's exact bit-packer in bounded row batches."""

            if packed_dtype is None:
                import torch

                packed_dtype = torch.int32
            rows = tensor.T if transpose else tensor
            packed_rows = [
                quantized_model.QuarkModel.pack_on_row_for_2_4_8_bits(
                    self, rows[start : start + PACK_ROWS], bits, False, packed_dtype
                )
                for start in range(0, rows.shape[0], PACK_ROWS)
            ]
            if len(packed_rows) == 1:
                packed = packed_rows[0]
            else:
                import torch

                packed = torch.cat(packed_rows, dim=0)
            return packed.T if transpose else packed

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured: dict[int, dict[str, Any]] = {}
            original_load_file = quantized_model.load_file
            original_set_properties = quantized_model.QuantizedModel.set_properties

            def load_and_capture(path: str) -> dict[str, Any]:
                retained: dict[str, Any] = {}
                for raw_name, tensor in original_load_file(path).items():
                    name = quantized_model.normalize_vlm_weight_name(raw_name)
                    if name is None:
                        continue
                    name = _canonical_tensor_name(name)
                    if _is_ignored_non_text_key(name):
                        continue
                    parsed = _linear_parameter(name)
                    if parsed is None:
                        retained[raw_name] = tensor
                    else:
                        layer_id, _projection, _parameter = parsed
                        captured.setdefault(layer_id, {})[name] = tensor
                return retained

            quantized_model.load_file = load_and_capture

            def set_properties_with_linear_attention(instance: Any) -> None:
                """Use upstream setup for attention layers and finish dense MLPs here.

                The generic loader assumes every decoder layer has q/k/v/o
                attention.  Qwen3.5 GatedDeltaNet layers do not, but still
                contain an ordinary dense MLP that QuarkModel later repacks.
                """

                all_layers = instance.layers
                layer_values = list(all_layers.values()) if isinstance(all_layers, dict) else list(all_layers)
                if isinstance(all_layers, dict):
                    instance.layers = {
                        layer.layer_id: layer for layer in layer_values if layer.layer_id not in captured
                    }
                else:
                    instance.layers = [layer for layer in layer_values if layer.layer_id not in captured]
                try:
                    original_set_properties(instance)
                finally:
                    instance.layers = all_layers
                for layer in layer_values:
                    if layer.layer_id in captured:
                        _set_linear_mlp_properties(instance, layer)

            quantized_model.QuantizedModel.set_properties = set_properties_with_linear_attention
            try:
                super().__init__(*args, **kwargs)
            finally:
                quantized_model.load_file = original_load_file
                quantized_model.QuantizedModel.set_properties = original_set_properties
            _attach_linear_attention(self, captured, quantized_model)

    def from_pretrained(quant_type: str, **kwargs: Any) -> Any:
        if quant_type != "quark":
            return original_factory(quant_type, **kwargs)
        return Qwen35QuarkModel(quant_type, **kwargs)

    quantized_model.QuantizedDecoderLayer.__init__ = decoder_init
    quantized_model.QuantModel.from_pretrained = staticmethod(from_pretrained)
    # Model.load_weights imports this module by its unqualified name.  Make
    # that import resolve to this patched public OGA module for this export.
    sys.modules["quantized_model"] = quantized_model
    try:
        yield
    finally:
        quantized_model.QuantModel.from_pretrained = original_factory
        quantized_model.QuantizedDecoderLayer.__init__ = original_decoder_init
        if original_bare_module is missing_module:
            del sys.modules["quantized_model"]
        else:
            sys.modules["quantized_model"] = original_bare_module


def export_oga(source: Path, output: Path, revision: str, quantized_loader_sha256: str) -> None:
    output_path = output.resolve(strict=False)
    output_parent = output_path.parent
    if not output_parent.is_dir():
        raise FileNotFoundError(f"output parent must exist: {output_parent}")
    if output_path.exists():
        raise FileExistsError(f"output is create-only and already exists: {output_path}")
    _require_checkpoint(source, revision)
    input_hashes = _require_quantize_manifest(source, revision)
    if quantized_loader_sha256 != PUBLIC_OGA_014_QUANTIZED_MODEL_SHA256:
        raise ValueError("export requires the pinned public OGA 0.14 quantized loader SHA256")
    from onnxruntime_genai.models.builder import create_model

    cache_dir = Path(tempfile.mkdtemp(prefix=f".{output_path.name}-builder-", dir=output_parent))
    cache_created = True
    try:
        with qwen35_quark_adapter(quantized_loader_sha256):
            create_model(
                model_name="Qwen3.5-4B-Quark",
                input_path=str(source),
                output_dir=str(output_path),
                precision="int4",
                execution_provider="dml",
                cache_dir=str(cache_dir),
                exclude_embeds=False,
                hf_remote=False,
            )
        repaired_outputs = repair_int4_matmul_output_names(output_path)
        _write_export_manifest(
            source, output_path, revision, input_hashes, quantized_loader_sha256, repaired_outputs
        )
    finally:
        if cache_created and cache_dir.exists():
            cache_resolved = cache_dir.resolve(strict=True)
            is_reparse_point = cache_dir.is_symlink() or (
                hasattr(cache_dir, "is_junction") and cache_dir.is_junction()
            )
            if is_reparse_point or cache_resolved.parent != output_parent:
                raise RuntimeError(f"refusing to remove unverified builder cache: {cache_dir}")
            shutil.rmtree(cache_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Quark uint4 group-128 checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new OGA output directory")
    parser.add_argument("--revision", default=SOURCE_REVISION, help="pinned Qwen3.5 source commit")
    parser.add_argument(
        "--quantized-loader-sha256",
        default=PUBLIC_OGA_014_QUANTIZED_MODEL_SHA256,
        help="SHA256 of the reviewed public onnxruntime_genai/models/quantized_model.py",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    export_oga(args.source, args.output, args.revision, args.quantized_loader_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
