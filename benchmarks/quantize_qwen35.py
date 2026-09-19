"""Create a Quark uint4 RTN/minmax checkpoint for Qwen3.5 text inference.

This is deliberately a weight-only baseline.  It does not enable AWQ, GPTQ,
SmoothQuant, activation quantization, or KV-cache quantization.  In
particular, do not describe its output as AWQ: the per-group MinMax observer
performs unsigned RTN-style weight quantization only.

Run this in a separate conversion environment containing AMD Quark 0.11 and
Transformers with Qwen3.5 conditional-generation classes. The verified route
uses Windows and CPU PyTorch; keep it separate from the Ryzen AI runtime.

The Qwen3.5 outer config and `model.language_model.*` names are intentionally
preserved because Ryzen AI OGA 0.14 dispatches Qwen3.5 only when its outer
architecture is `Qwen3_5ForConditionalGeneration`.  An OGA export adapter
must validate the resulting checkpoint before it is treated as deployable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


SOURCE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_CONFIG_SHA256 = "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
SOURCE_WEIGHT_SHA256 = {
    "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}
GROUP_SIZE = 128
QUARK_VERSION = "0.11"
QUARK_EXCLUDED_MODULES = (
    "model.visual.*",
    "mtp.*",
    "model.language_model.embed_tokens",
    "model.language_model.layers.*.linear_attn.conv1d",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_hashes(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "quantize-manifest.json"
    }


def _quark_version() -> str:
    for distribution in ("amd-quark", "amd_quark"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    raise RuntimeError("AMD Quark distribution metadata is unavailable")


def _require_quark_version() -> str:
    """Keep the converter on the reviewed Quark release, not just its API."""

    installed = _quark_version()
    if installed != QUARK_VERSION:
        raise RuntimeError(
            f"Qwen3.5 conversion requires reviewed AMD Quark {QUARK_VERSION}, found {installed}"
        )
    return installed


def _require_source(source: Path, revision: str) -> dict[str, str]:
    if revision != SOURCE_REVISION:
        raise ValueError(f"source revision must be the reviewed Qwen3.5-4B commit {SOURCE_REVISION}")
    config_path = source / "config.json"
    if not config_path.is_file():
        raise ValueError(f"source must contain config.json: {source}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError("source config model_type must be qwen3_5")
    text = config.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "qwen3_5_text":
        raise ValueError("source config must contain text_config.model_type=qwen3_5_text")
    expected = {"config.json": SOURCE_CONFIG_SHA256, **SOURCE_WEIGHT_SHA256}
    if {path.name for path in source.glob("*.safetensors")} != set(SOURCE_WEIGHT_SHA256):
        raise ValueError("source safetensors must be exactly the two pinned Qwen3.5-4B shards")
    hashes = {}
    for name, expected_sha256 in expected.items():
        actual = _sha256(source / name)
        if actual != expected_sha256:
            raise ValueError(f"source SHA256 mismatch for {name}")
        hashes[name] = actual
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or set(weight_map.values()) != set(SOURCE_WEIGHT_SHA256):
        raise ValueError("source weight index must reference exactly the two pinned shards")
    hashes[index_path.name] = _sha256(index_path)
    # Record tokenizer inputs as well: they are copied into the checkpoint.
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json"):
        path = source / name
        if path.is_file():
            hashes[name] = _sha256(path)
    return hashes


def _weight_only_config() -> Any:
    """Build Quark's public uint4 per-group MinMax configuration.

    No algorithm configuration is supplied, which is the material difference
    between this RTN/minmax baseline and an AWQ conversion.
    """

    from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
    from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
    from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

    weight = QTensorConfig(
        dtype=Dtype.uint4,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=False,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        qscheme=QSchemeType.per_group,
        ch_axis=1,
        is_dynamic=False,
        group_size=GROUP_SIZE,
    )
    return QConfig(
        global_quant_config=QLayerConfig(weight=weight),
        # The original source is multimodal and includes an MTP head.  This
        # baseline quantizes only the text backbone while retaining the outer
        # HF architecture and untouched non-text weights for key compatibility.
        # The GatedDeltaNet causal convolution is depthwise [8192, 1, 4] and
        # must stay raw for the Ryzen AI CausalConv kernel.  Quark 0.11 eager
        # mode has no QuantConv1d replacement, but retain the explicit policy
        # in case module coverage changes.
        exclude=list(QUARK_EXCLUDED_MODULES),
        # The explicit policy is recorded in the conversion manifest too.
        # Keep the rationale close to the Quark configuration because these
        # tensors are required unquantized by the text-only OGA loader.
        #
        # OGA consumes the embedding as a dense initializer.  Quark has an
        # eager QuantEmbedding replacement, so keep this explicit even though
        # the tied lm_head remains eligible for uint4 conversion.
        # The GatedDeltaNet causal convolution is depthwise [8192, 1, 4] and
        # must stay raw for the Ryzen AI CausalConv kernel.
        #
        # The patterns themselves live in QUARK_EXCLUDED_MODULES so the
        # manifest cannot understate the conversion policy.
        # (Quark 0.11 eager mode has no QuantConv1d replacement, but retain
        # the explicit policy in case module coverage changes.)
    )


def _self_test() -> None:
    """Exercise the Quark API on CPU without downloading or exporting a model."""

    import torch
    from quark.torch import ModelQuantizer

    linear = torch.nn.Sequential(torch.nn.Linear(GROUP_SIZE, GROUP_SIZE, bias=False)).eval()
    quantized = ModelQuantizer(_weight_only_config()).quantize_model(linear)
    frozen = ModelQuantizer.freeze(quantized)
    with torch.inference_mode():
        output = frozen(torch.ones((1, GROUP_SIZE)))
    if output.shape != (1, GROUP_SIZE):
        raise RuntimeError(f"unexpected self-test output shape: {tuple(output.shape)}")


def _self_test_export() -> None:
    """Verify Quark's packed safetensors export with a tiny Transformers model."""

    from safetensors import safe_open
    from transformers import BertConfig, BertForMaskedLM
    from quark.torch import ModelQuantizer, export_safetensors

    config = BertConfig(
        vocab_size=256,
        hidden_size=GROUP_SIZE,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=GROUP_SIZE * 2,
    )
    model = BertForMaskedLM(config).eval()
    quantized = ModelQuantizer(_weight_only_config()).quantize_model(model)
    frozen = ModelQuantizer.freeze(quantized)
    with tempfile.TemporaryDirectory(prefix="quark-rtn-export-") as directory:
        output = Path(directory) / "output"
        export_safetensors(frozen, output, custom_mode="quark", weight_format="real_quantized", pack_method="reorder")
        exported_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        quantization_config = exported_config.get("quantization_config", {})
        weight_config = quantization_config.get("global_quant_config", {}).get("weight", {})
        if quantization_config.get("quant_method") != "quark":
            raise RuntimeError("Quark export did not retain quant_method=quark")
        if weight_config.get("dtype") != "uint4" or weight_config.get("group_size") != GROUP_SIZE:
            raise RuntimeError("Quark export did not retain uint4 group-128 quantization metadata")
        with safe_open(output / "model.safetensors", framework="pt") as archive:
            keys = list(archive.keys())
        if not any(key.endswith("qweight") or key.endswith("weight") for key in keys):
            raise RuntimeError("Quark export did not emit packed weight tensors")
        if not any(key.endswith("weight_scale") or key.endswith("weight_quantizer.scale") for key in keys):
            raise RuntimeError("Quark export did not emit weight-scale tensors")
        if not any(key.endswith("weight_zero_point") or key.endswith("weight_quantizer.zero_point") for key in keys):
            raise RuntimeError("Quark export did not emit weight zero-point tensors")


def _load_wrapper_with_text_backbone(source: Path, device: str, torch_threads: int) -> tuple[Any, Any]:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    # Qwen3.5-4B is distributed as a multimodal wrapper.  Keep its outer
    # architecture and the text keys exactly as published: the Ryzen AI 1.8
    # OGA builder recognizes this model family by that outer architecture.
    torch.set_num_threads(torch_threads)
    load_options: dict[str, Any] = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
    }
    # Quark rejects an Accelerate device map containing CPU placements.  A
    # CPU conversion therefore uses ordinary Transformers loading; ROCm/CUDA
    # conversion keeps the model on the requested accelerator.
    if device.lower() != "cpu":
        load_options["device_map"] = {"": device}
    try:
        wrapped = Qwen3_5ForConditionalGeneration.from_pretrained(
            source,
            **load_options,
        )
    except Exception as error:
        raise RuntimeError(
            "this Transformers build cannot load the Qwen3.5 conditional-generation wrapper; "
            "use a Qwen3.5-capable Transformers build"
        ) from error
    if getattr(wrapped.config, "model_type", None) != "qwen3_5":
        raise RuntimeError("loaded model is not Qwen3.5")
    language_model = getattr(getattr(wrapped, "model", None), "language_model", None)
    text_config = getattr(wrapped.config, "text_config", None)
    if language_model is None or getattr(text_config, "model_type", None) != "qwen3_5_text":
        raise RuntimeError("Qwen3.5 wrapper does not expose its expected text backbone")
    wrapped.eval()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    return wrapped, tokenizer


def convert(source: Path, output: Path, revision: str, device: str, torch_threads: int) -> None:
    if output.exists():
        raise FileExistsError(f"output is create-only and already exists: {output}")
    source_hashes = _require_source(source, revision)
    _require_quark_version()

    from quark.torch import ModelQuantizer, export_safetensors

    model, tokenizer = _load_wrapper_with_text_backbone(source, device, torch_threads)
    quantized = ModelQuantizer(_weight_only_config()).quantize_model(model)
    frozen = ModelQuantizer.freeze(quantized)
    export_safetensors(frozen, output, custom_mode="quark", weight_format="real_quantized", pack_method="reorder")
    tokenizer.save_pretrained(output)

    manifest = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "source_revision": revision,
        "source_config_sha256": source_hashes["config.json"],
        "source_file_sha256": source_hashes,
        "model_type": "qwen3_5",
        "text_model_type": "qwen3_5_text",
        "source_wrapper": "Qwen3_5ForConditionalGeneration",
        "export_model": "Qwen3_5ForConditionalGeneration",
        "quantized_component": "model.language_model",
        "excluded_components": list(QUARK_EXCLUDED_MODULES),
        "quantizer": "amd-quark",
        "quantizer_version": _quark_version(),
        "method": "uint4_rtn_minmax_weight_only",
        "awq": False,
        "weight_dtype": "uint4",
        "group_size": GROUP_SIZE,
        "symmetric": False,
        "round_method": "half_even",
        "scale_type": "float",
        "activation_quantization": False,
        "kv_cache_quantization": False,
        "quark_disable_compile": os.environ.get("QUARK_DISABLE_COMPILE") == "1",
        "export": {"format": "quark_safetensors", "weight_format": "real_quantized", "pack_method": "reorder"},
        "oga_status": "requires_export_adapter_validation",
        "output_file_sha256": _output_hashes(output),
    }
    with (output / "quantize-manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local Qwen3.5-4B source checkout")
    parser.add_argument("--output", type=Path, help="new output directory for Quark safetensors")
    parser.add_argument("--revision", default=SOURCE_REVISION, help="pinned source commit")
    parser.add_argument("--device", default="cuda:0", help="PyTorch ROCm/CUDA device, or cpu")
    parser.add_argument("--torch-threads", type=int, default=32, help="CPU thread count used during conversion")
    parser.add_argument(
        "--allow-quark-compile",
        action="store_true",
        help="allow Quark's torch.compile fake-quantizer path; disabled by default for reproducible CPU conversion",
    )
    parser.add_argument("--self-test", action="store_true", help="run the CPU-only Quark API smoke test")
    parser.add_argument("--self-test-export", action="store_true", help="run the temporary packed-safetensors export check")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.allow_quark_compile:
        # Quark 0.11 enables torch.compile for fake quantization by default.
        # Native extension startup may still require the MSVC developer shell.
        os.environ["QUARK_DISABLE_COMPILE"] = "1"
    if args.self_test:
        _self_test()
        return 0
    if args.self_test_export:
        _self_test_export()
        return 0
    if args.source is None or args.output is None:
        _parser().error("--source and --output are required unless --self-test is used")
    if args.torch_threads < 1:
        _parser().error("--torch-threads must be positive")
    convert(args.source, args.output, args.revision, args.device, args.torch_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
