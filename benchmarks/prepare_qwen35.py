"""Run Ryzen AI's Qwen3.5 optimizer with one SDK 1.8.0 metadata recovery.

ORT's extended optimizer can rename an FP32 ``LinearAttention`` output to an
``InsertedPrecisionFreeCast_*`` value while dropping its value-info entry.
The following SDK pass then needs that dtype to restore the explicit FP16 cast.
This module supplies it only for the documented, structurally verified case;
it does not alter the installed SDK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence


_PREFIX = "InsertedPrecisionFreeCast_"
_SDK_SHA256 = {
    "matcher.py": "d2a6671fab85889267fccbab4c77d8890d13e4ce017b48e7fd3ef681c5043c4f",
    "add_fp16_casts.py": "1de099e32c07277fa81957d00d6be3a1567b025e47935dc4ca94d74e31e80a82",
    "llm.py": "a5f4cb5d153335c9469aee762dca6fc9039f684e25e1cb45d4f89ddf93937982",
}
_MODEL_NAME = "model.onnx"
_MAX_CONTEXT = 16384
_DEFAULT_CHUNK_SIZE = 64
_PROVIDER_MAX_SEQ_LENGTH = 4096
_RUNTIME_TMP_ARTIFACTS = {"cache", "genai_config.json", "model.bin", "model.onnx", "model.onnx.data", "model.pb.bin"}
_RUNTIME_MODEL_FILES = ("model.onnx", "model.onnx.data", "model.bin", "model.pb.bin", "genai_config.json")


def _recover_precision_free_cast_value_info(name: str, extractor, matcher, onnx) -> None:
    """Restore one strictly identified FP32 LinearAttention output ValueInfo.

    The AMD preprocessor leaves the original post-cast ValueInfo named without
    ``_PREFIX``. Its shape is still authoritative; only the raw producer output
    lost its type. Every condition below was observed in the failed Qwen3.5
    optimized graph and is required before adding the recovered entry.
    """
    if not name.startswith(_PREFIX):
        raise ValueError(f"{name} not found in graph")
    producers = matcher.find_nodes_by_output(name, extractor.graph)
    if len(producers) != 1:
        raise ValueError(f"{name} has no unique producer")
    node = producers[0]
    update_rule = matcher.get_attribute(node, "update_rule", None)
    if isinstance(update_rule, bytes):
        update_rule = update_rule.decode("utf-8")
    if (
        node.domain != "com.microsoft"
        or node.op_type != "LinearAttention"
        or update_rule != "gated_delta"
        or len(node.input) != 6
        or any(matcher.get_dtype(input_name, extractor) != onnx.TensorProto.FLOAT for input_name in node.input)
    ):
        raise ValueError(f"{name} is not a recoverable GatedDeltaNet LinearAttention output")
    consumers = matcher.find_nodes_by_input(name, extractor.graph)
    if len(consumers) != 1 or consumers[0].op_type not in {"Cast", "CastAvx"}:
        raise ValueError(f"{name} does not have one FP32-to-FP16 Cast consumer")
    cast_to = matcher.get_attribute(consumers[0], "to", None)
    if cast_to != onnx.TensorProto.FLOAT16:
        raise ValueError(f"{name} does not have one FP32-to-FP16 Cast consumer")
    original_name = name.removeprefix(_PREFIX)
    original_tvi = matcher.get_tvi(original_name, extractor)
    if matcher.get_dtype(original_tvi) != onnx.TensorProto.FLOAT16:
        raise ValueError(f"{original_name} is not the expected FP16 Cast output")
    matcher.replace_tvis(
        [matcher.build_tvi(original_name, extractor, name=name, dtype=onnx.TensorProto.FLOAT)], extractor
    )


def install_precision_free_cast_recovery() -> None:
    """Install a process-local dtype fallback for the affected SDK pass once."""
    import onnx
    import ryzenai_onnx_utils.matcher as matcher
    from ryzenai_onnx_utils.model_preprocessing import llm as llm_preprocessing
    from ryzenai_onnx_utils.passes.llm import add_fp16_casts

    if getattr(matcher, "_qwen35_recovery_installed", False):
        return
    for module in (matcher, add_fp16_casts, llm_preprocessing):
        path = Path(module.__file__).resolve()
        expected = _SDK_SHA256[path.name]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"unsupported Ryzen AI 1.8.0 SDK file: {path.name}")
    native_get_dtype = matcher.get_dtype

    def get_dtype_with_qwen35_recovery(obj, extractor=None):
        try:
            return native_get_dtype(obj, extractor)
        except ValueError:
            if not isinstance(obj, str) or extractor is None:
                raise
            _recover_precision_free_cast_value_info(obj, extractor, matcher, onnx)
            return native_get_dtype(obj, extractor)

    matcher.get_dtype = get_dtype_with_qwen35_recovery
    matcher._qwen35_recovery_installed = True


def _as_int(value, field: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{field} must be an integer") from error
    if str(result) != str(value):
        raise RuntimeError(f"{field} must be an integer")
    return result


def _validate_chunk_size(chunk_size: int) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= _PROVIDER_MAX_SEQ_LENGTH:
        raise ValueError("chunk_size must be an integer from 1 through 4096")
    return chunk_size


def _set_eager_search_options(config, chunk_size: int) -> None:
    """Set OGA's 16K ceiling and its selected prefill chunk independently."""
    config.set_search_option("max_length", _MAX_CONTEXT)
    config.set_search_option("chunk_size", _validate_chunk_size(chunk_size))


def assert_ryzenai_genai_config(model_dir: Path, *, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> None:
    """Require the Qwen3.5 NPU-eager hybrid 16K RyzenAI configuration."""
    chunk_size = _validate_chunk_size(chunk_size)
    config_path = model_dir / "genai_config.json"
    if not (model_dir / "model.onnx").is_file() or not config_path.is_file():
        raise RuntimeError("optimizer did not create model.onnx and genai_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config.get("model", {})
    if model.get("type") != "qwen3_5_text":
        raise RuntimeError("genai_config.json is not a qwen3_5_text model")
    if _as_int(config.get("model", {}).get("context_length"), "model.context_length") < _MAX_CONTEXT:
        raise RuntimeError("model.context_length is below 16384")
    search = config.get("search", {})
    if _as_int(search.get("max_length"), "search.max_length") != _MAX_CONTEXT:
        raise RuntimeError("search.max_length must equal 16384")
    if _as_int(search.get("chunk_size"), "search.chunk_size") != chunk_size:
        raise RuntimeError(f"search.chunk_size must equal {chunk_size}")
    if search.get("past_present_share_buffer") is not True:
        raise RuntimeError("search.past_present_share_buffer must be true for Qwen3.5 recurrent state")
    decoder = model.get("decoder", {})
    options = decoder.get("session_options", {}).get("provider_options", [])
    if not isinstance(options, list) or len(options) != 1:
        raise RuntimeError("genai_config.json must have exactly one RyzenAI decoder provider")
    option = options[0]
    if not isinstance(option, dict) or set(option) != {"RyzenAI"} or not isinstance(option["RyzenAI"], dict):
        raise RuntimeError("genai_config.json must have exactly one RyzenAI decoder provider")
    provider = option["RyzenAI"]
    if provider.get("hybrid_opt_token_backend") != "npu":
        raise RuntimeError("RyzenAI token backend must be npu")
    chunk_context = provider.get("hybrid_opt_chunk_context")
    if isinstance(chunk_context, bool) or chunk_context not in {"1", 1}:
        raise RuntimeError("RyzenAI chunk context must be enabled")
    if _as_int(provider.get("hybrid_opt_max_seq_length"), "RyzenAI.hybrid_opt_max_seq_length") != _PROVIDER_MAX_SEQ_LENGTH:
        raise RuntimeError("RyzenAI.hybrid_opt_max_seq_length must equal 4096")


def _required_absolute_option(argv: Sequence[str], option: str) -> Path:
    try:
        value = Path(argv[argv.index(option) + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(f"missing {option}") from error
    if not value.is_absolute():
        raise ValueError(f"{option} must be absolute so SDK temporary files stay in --work-dir")
    return value


def run_onnx_utils(argv: Sequence[str], work_dir: Path) -> None:
    """Invoke ``onnx_utils`` in an isolated working directory."""
    _required_absolute_option(argv, "--input-model")
    _required_absolute_option(argv, "--output-model")
    work_dir = work_dir.resolve()
    if not work_dir.is_dir():
        raise ValueError(f"--work-dir must exist: {work_dir}")
    original_argv = sys.argv
    original_cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        import ryzenai_onnx_utils.partitioner as partitioner

        install_precision_free_cast_recovery()
        sys.argv = ["onnx_utils", *argv]
        partitioner.main()
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)


def _run_fixed_shape_npu_eager(input_model: Path, output_model: Path, work_dir: Path, *, chunk_size: int) -> None:
    """Run the SDK eager strategy after its required 16K prefill shape fix.

    The public ``optimize --prefill npu_eager --token npu_eager`` entry point
    does not call ``_llm_fix_shapes``.  Qwen3.5 needs that prefill-phase fix
    so OGA receives fixed recurrent/KV state shapes and a dynamic prompt
    length.  This is the same SDK sequence used by the successful eager
    prefill graph, kept process-local rather than changing the SDK.
    """
    chunk_size = _validate_chunk_size(chunk_size)
    input_model = input_model.resolve()
    output_model = output_model.resolve()
    if not input_model.is_file() or not output_model.is_relative_to(input_model.parent):
        raise ValueError("input model and temporary eager output must share one model directory")
    original_cwd = Path.cwd()
    try:
        os.chdir(work_dir.resolve())
        import ryzenai_onnx_utils.optimize as optimize
        import ryzenai_onnx_utils.partitioner as partitioner

        install_precision_free_cast_recovery()
        parser = partitioner.get_parser()
        namespace = parser.parse_args(
            [
                "optimize", "--input-model", str(input_model), "--output-model", str(output_model), "--force", "llm",
                "--prefill", "npu_eager", "--token", "npu_eager", "--model-type", "qwen3.5",
                "--max-seq-len", str(_MAX_CONTEXT), "--no-prune-logits",
            ]
        )
        args = optimize.LlmArgs(namespace)
        optimized = output_model.parent / f"optimized_{input_model.name}"
        optimize.llm_preprocess_optimize(args, optimized)
        args.input_model = optimized
        fixed = optimize._llm_fix_shapes(args, optimize.Phase.PREFILL)
        args.input_model = fixed
        args.output_model = output_model
        builder = optimize.llm_npu_eager(args)
        config = builder.generate_genai_config(output_model.stem)
        # Keep the SDK eager provider's tested 4K allocation bound.  Search
        # options control OGA's context/chunking and do not change graph shapes.
        _set_eager_search_options(config, chunk_size)
        config_path = config.save(input_model.parent, output_model, args.dry_run)
        if config_path is None:
            raise RuntimeError("SDK did not create a GenAI configuration")
        config_path.rename(config_path.with_stem("genai_config"))
        # These are created beneath the output's temporary directory, never
        # beside the caller's source model.  Remove only the SDK's known files
        # before _replace_from_tmp promotes the final artifact.
        fixed_shape_models = set(output_model.parent.glob(f"tmp_model_{_MAX_CONTEXT}_prompt_*.onnx"))
        for intermediate in {optimized, fixed, *fixed_shape_models}:
            _require_child(intermediate, output_model.parent)
            for candidate in (
                intermediate,
                intermediate.with_name(f"{intermediate.name}.data"),
                intermediate.with_suffix(".pb.bin"),
                intermediate.with_suffix(".bin"),
            ):
                _require_child(candidate, output_model.parent)
                candidate.unlink(missing_ok=True)
        _require_child(output_model.with_name(f"{output_model.stem}_strategy.yaml"), output_model.parent).unlink(
            missing_ok=True
        )
    finally:
        os.chdir(original_cwd)


def _require_child(path: Path, parent: Path) -> Path:
    if path.is_symlink() or getattr(os.path, "isjunction", lambda _: False)(path):
        raise RuntimeError(f"unsafe link or junction: {path}")
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or resolved.is_symlink():
        raise RuntimeError(f"unsafe output artifact path: {path}")
    return resolved


def _preserve_tmp_diagnostics(tmp_dir: Path, work_dir: Path) -> None:
    diagnostics = [path for path in tmp_dir.iterdir() if path.name not in _RUNTIME_TMP_ARTIFACTS]
    if not diagnostics:
        return
    target = work_dir / f"{tmp_dir.parent.name}-sdk-diagnostics"
    if target.exists():
        raise RuntimeError(f"SDK diagnostics destination already exists: {target}")
    target.mkdir()
    for source in diagnostics:
        source = _require_child(source, tmp_dir)
        shutil.move(str(source), str(_require_child(target / source.name, target)))


def _replace_from_tmp(output_dir: Path, work_dir: Path) -> None:
    tmp_dir = _require_child(output_dir / "tmp", output_dir)
    if not tmp_dir.is_dir():
        raise RuntimeError("optimizer did not create tmp output")
    artifacts = list(tmp_dir.iterdir())
    if not artifacts or not (tmp_dir / _MODEL_NAME).is_file() or not (tmp_dir / "genai_config.json").is_file():
        raise RuntimeError("optimizer tmp output lacks model.onnx or genai_config.json")
    _preserve_tmp_diagnostics(tmp_dir, work_dir)
    artifacts = list(tmp_dir.iterdir())
    for source in artifacts:
        source = _require_child(source, tmp_dir)
        destination = _require_child(output_dir / source.name, output_dir)
        if destination.exists():
            if destination.is_symlink() or getattr(os.path, "isjunction", lambda _: False)(destination):
                raise RuntimeError(f"refusing to replace symlink: {destination}")
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(source), str(destination))
    shutil.rmtree(tmp_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(model_dir: Path) -> dict[str, str]:
    suffixes = {".onnx", ".data", ".bin", ".json", ".model", ".zip", ".fconst", ".state", ".meta", ".super", ".ctrlpkt"}
    return {
        path.relative_to(model_dir).as_posix(): _sha256(path)
        for path in sorted(model_dir.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in suffixes
    }


def _write_manifest(source: Path, output: Path, settings: dict[str, object]) -> Path:
    import onnxruntime
    import onnxruntime_genai

    manifest = {
        "source_dir": source.name,
        "output_dir": output.name,
        "source_artifact_sha256": _artifact_hashes(source),
        "output_artifact_sha256": _artifact_hashes(output),
        "settings": settings,
        "sdk": {
            "onnxruntime": onnxruntime.__version__,
            "onnxruntime_genai": onnxruntime_genai.__version__,
            "ryzen_ai": getattr(onnxruntime_genai, "__rai_version__", None),
            "patched_sdk_sha256": _SDK_SHA256,
        },
    }
    path = output / "conversion-manifest.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _require_separate_directories(source: Path, output: Path, work_dir: Path) -> None:
    if (
        output.is_relative_to(source)
        or source.is_relative_to(output)
        or work_dir.is_relative_to(source)
        or work_dir.is_relative_to(output)
    ):
        raise ValueError("source, output, and work-dir must be separate directories")


def _assert_finalize_input(model_dir: Path, *, chunk_size: int) -> None:
    """Check the complete eager payload before promotion or SDK writes."""
    for filename in _RUNTIME_MODEL_FILES:
        path = _require_child(model_dir / filename, model_dir)
        if not path.is_file():
            raise RuntimeError(f"incomplete optimizer output: missing {filename}")
    assert_ryzenai_genai_config(model_dir, chunk_size=chunk_size)


def finalize_existing(source: Path, output: Path, work_dir: Path, settings: dict[str, object]) -> Path:
    """Finalize staged output, or retry SDK finalization after its promotion."""
    source = source.resolve()
    output = output.resolve()
    work_dir = work_dir.resolve()
    _require_separate_directories(source, output, work_dir)
    if not source.is_dir() or not output.is_dir() or not work_dir.is_dir() or source == output:
        raise ValueError("source and existing output must be distinct directories")
    if _require_child(output / "conversion-manifest.json", output).exists():
        raise RuntimeError("conversion-manifest.json already exists; output is not create-only")
    chunk_size = _validate_chunk_size(settings["chunk_size"])
    tmp_dir = _require_child(output / "tmp", output)
    if tmp_dir.exists():
        # In particular, a wrong --chunk-size must leave the completed SDK
        # output intact so the caller can retry with the matching setting.
        _assert_finalize_input(tmp_dir, chunk_size=chunk_size)
    else:
        # Promotion may have completed before an SDK finalization/import error.
        # Resume only a complete, matching NPU artifact, never a source OGA copy.
        _assert_finalize_input(output, chunk_size=chunk_size)
    for filename in ("optimized_model.onnx", "optimized_model.onnx.data"):
        _require_child(output / filename, output)
    if tmp_dir.exists():
        _replace_from_tmp(output, work_dir)
    original_cwd = Path.cwd()
    try:
        os.chdir(work_dir.resolve())
        from model_generate.filtering import filter_bins, resolve_dyn_bins
        from model_generate.runner import finalize_output

        finalize_output(output, _MODEL_NAME, model_type="qwen3.5")
        dyn_bins = resolve_dyn_bins()
        if dyn_bins is not None:
            filter_bins(output, dyn_bins, model_filename=_MODEL_NAME)
    finally:
        os.chdir(original_cwd)
    _assert_finalize_input(output, chunk_size=chunk_size)
    return _write_manifest(source, output, settings)


def prepare(
    source: Path,
    output: Path,
    work_dir: Path,
    *,
    finalize_existing_only: bool = False,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Path:
    """Create a fresh Qwen3.5 NPU-eager hybrid 16K output and finalize it deterministically."""
    source = source.resolve()
    output = output.resolve()
    work_dir = work_dir.resolve()
    chunk_size = _validate_chunk_size(chunk_size)
    settings = {
        "model_type": "qwen3.5",
        "prefill": "npu_eager",
        "token": "npu_eager",
        "max_seq_len": _MAX_CONTEXT,
        "chunk_size": chunk_size,
        "provider_max_seq_length": _PROVIDER_MAX_SEQ_LENGTH,
        "no_prune_logits": True,
    }
    _require_separate_directories(source, output, work_dir)
    if finalize_existing_only:
        return finalize_existing(source, output, work_dir, settings)
    if not source.is_dir() or output.exists() or not work_dir.is_dir():
        raise ValueError("source and work-dir must exist and output must be a new path")
    shutil.copytree(source, output)
    copied_manifest = _require_child(output / "conversion-manifest.json", output)
    if copied_manifest.exists():
        if not copied_manifest.is_file():
            raise RuntimeError("source conversion-manifest.json is not a file")
        copied_manifest.unlink()
    (output / "tmp").mkdir()
    input_model = (output / _MODEL_NAME).resolve()
    tmp_model = (output / "tmp" / _MODEL_NAME).resolve()
    _run_fixed_shape_npu_eager(input_model, tmp_model, work_dir, chunk_size=chunk_size)
    return finalize_existing(source, output, work_dir, settings)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="existing ignored diagnostics directory")
    parser.add_argument("--source", type=Path, help="source OGA model directory")
    parser.add_argument("--output", type=Path, help="new model output directory")
    parser.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE, help="OGA prefill chunk size (1-4096; default: 64)")
    parser.add_argument("--finalize-existing", action="store_true", help="only finish an existing output/tmp optimizer result")
    args, onnx_utils_args = parser.parse_known_args(argv)
    if args.source or args.output or args.finalize_existing:
        if args.source is None or args.output is None or onnx_utils_args:
            parser.error("--source/--output mode does not accept onnx_utils arguments")
        prepare(args.source, args.output, args.work_dir, finalize_existing_only=args.finalize_existing, chunk_size=args.chunk_size)
        return
    if args.chunk_size != _DEFAULT_CHUNK_SIZE:
        parser.error("--chunk-size requires --source and --output")
    if "optimize" not in onnx_utils_args:
        parser.error("pass normal onnx_utils arguments containing the 'optimize' command")
    run_onnx_utils(onnx_utils_args, args.work_dir)


if __name__ == "__main__":
    main()
