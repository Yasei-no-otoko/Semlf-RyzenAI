"""Build the experimental Qwen3.5 DD token path with a pinned Ryzen AI SDK.

This is a local conversion tool, not an inference or SDK installer. The profile
identifies the exact measured OGA/eager inputs. A successful build means CPU
structural checks passed; run the separate validator to establish NPU behavior.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_PROFILE = HERE / "qwen35_rai18_profile.json"
SOURCE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def local_file(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Unsafe relative artifact path: {relative}")
    candidate = root
    for part in rel.parts:
        candidate = candidate / part
        if candidate.is_symlink() or candidate.is_junction():
            raise ValueError(f"Linked artifact is unsupported: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Artifact is not a confined file: {relative}")
    return resolved


def file_identity(path: Path, expected: str) -> dict:
    before = path.stat()
    actual = sha256(path)
    after = path.stat()
    def fields(st):
        return st.st_size, st.st_mtime_ns, st.st_ino
    if fields(before) != fields(after) or actual != expected:
        raise ValueError(f"Artifact changed or is outside the compatibility profile: {path.name}")
    return {"sha256": actual, "size": after.st_size,
            "mtime_ns": after.st_mtime_ns, "inode": after.st_ino}


def input_identities(root: Path, pins: dict[str, str]) -> dict:
    return {name: file_identity(local_file(root, name), expected)
            for name, expected in sorted(pins.items())}


def check_sdk(sdk_root: Path, profile: dict) -> dict:
    """Check files without importing the AMD runtime or registering a device."""
    versions = {}
    for name, expected in profile["package_versions"].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Use the pinned SDK environment: {name} must be {expected}, got {actual}")
        versions[name] = actual
    packages = {}
    for name, pins in profile["package_files"].items():
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"Installed SDK package is missing: {name}")
        root = Path(spec.origin).resolve().parent
        packages[name] = {"root": str(root), "files": input_identities(root, pins)}
    return {"versions": versions, "packages": packages,
            "sdk_root": str(sdk_root), "sdk_files": input_identities(sdk_root, profile["sdk_files"])}


def separate_paths(source: Path, prefill: Path, work: Path, output: Path) -> None:
    paths = (source, prefill, work, output)
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Source, prefill, work, and output must be separate directories")
    if work.exists() or output.exists():
        raise FileExistsError("Work and output must be new paths; failed builds cannot be overwritten")
    if not source.is_dir() or not prefill.is_dir() or not work.parent.is_dir() or not output.parent.is_dir():
        raise ValueError("Input directories and output parent directories must exist")


def code_hashes() -> dict[str, str]:
    names = ("prepare_qwen35_token_fusion.py", "prepare_qwen35.py", "qwen35_dd.py", "qwen35_package.py", "qwen35_prefill.py",
             "qwen35_adaptive_prefill.py")
    return {name: sha256(HERE / name) for name in names}


def create_plan(source: Path, prefill: Path, sdk_root: Path, work: Path,
                output: Path, profile_path: Path = DEFAULT_PROFILE, *,
                prefill_linear_attention: str = "native", prefill_chunk_size: int | None = None) -> dict:
    if prefill_linear_attention not in {"native", "token_loop", "adaptive"}:
        raise ValueError("Unsupported prefill LinearAttention mode")
    if prefill_linear_attention == "adaptive":
        prefill_chunk_size = 1024 if prefill_chunk_size is None else prefill_chunk_size
        if type(prefill_chunk_size) is not int or prefill_chunk_size not in (1024, 4096):
            raise ValueError("Adaptive prefill chunk size must be 1024 or 4096")
    elif prefill_chunk_size is not None:
        raise ValueError("A prefill chunk override requires adaptive mode")
    source, prefill, sdk_root, work, output, profile_path = (
        p.resolve() for p in (source, prefill, sdk_root, work, output, profile_path))
    separate_paths(source, prefill, work, output)
    profile = read_json(profile_path)
    if profile.get("schema_version") != 1 or profile.get("source_revision") != SOURCE_REVISION:
        raise ValueError("Unsupported Qwen3.5 compatibility profile")
    try:
        from benchmarks.prepare_qwen35 import assert_ryzenai_genai_config
    except ModuleNotFoundError:
        from prepare_qwen35 import assert_ryzenai_genai_config
    assert_ryzenai_genai_config(prefill, chunk_size=64)
    return {"schema_version": 1, "status": "plan_only", "source_revision": SOURCE_REVISION,
            "prefill_linear_attention": prefill_linear_attention,
            "prefill_chunk_size": prefill_chunk_size,
            "source": str(source), "prefill": str(prefill), "sdk_root": str(sdk_root),
            "work": str(work), "output": str(output), "profile": str(profile_path),
            "profile_sha256": sha256(profile_path), "code_sha256": code_hashes(),
            "source_artifacts": input_identities(source, profile["source_artifacts"]),
            "prefill_artifacts": input_identities(prefill, profile["prefill_artifacts"]),
            "sdk": check_sdk(sdk_root, profile),
            "qualification": "Pinned local CPU conversion plan; no model or NPU session has been created"}


def copy_inputs(source: Path, destination: Path, identities: dict) -> None:
    destination.mkdir(exist_ok=False)
    for name, identity in identities.items():
        original = local_file(source, name)
        file_identity(original, identity["sha256"])
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        file_identity(target, identity["sha256"])
        if target.stat().st_ino == original.stat().st_ino or target.stat().st_nlink != 1:
            raise ValueError("Conversion inputs must be independent copies")


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


def verify_fixed_token(path: Path, profile: dict) -> dict:
    import onnx
    model = onnx.load(path, load_external_data=False)
    for obj in (model, model.graph):
        for entry in list(obj.metadata_props):
            if entry.key == "onnx_utils_load":
                obj.metadata_props.remove(entry)
    locations = set()
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                locations.add(entry.value)
                entry.value = "__fixed_token_external_data__"
    if len(locations) != 1:
        raise ValueError("Fixed token graph must reference one external data file")
    graph_hash = hashlib.sha256(model.SerializeToString()).hexdigest()
    expected = profile["fixed_token"]
    if graph_hash != expected["normalized_graph_sha256"]:
        raise ValueError("Regenerated token graph differs from the measured structural contract")
    external = local_file(path.parent, locations.pop())
    data = file_identity(external, expected["external_data_sha256"])
    return {"path": str(path), "header_sha256": sha256(path),
            "normalized_graph_sha256": graph_hash, "external_data": data}


def regenerate_token(source: Path, work: Path, identities: dict, profile: dict) -> Path:
    """Reproduce the fixed token input without the unsupported fusion pass."""
    try:
        from benchmarks.prepare_qwen35 import install_precision_free_cast_recovery
    except ModuleNotFoundError:
        from prepare_qwen35 import install_precision_free_cast_recovery
    copied = work / "sdk-inputs"
    copy_inputs(source, copied, identities)
    fixed_dir = work / "fixed-token"
    fixed_dir.mkdir(exist_ok=False)
    with working_directory(work):
        import ryzenai_onnx_utils.optimize as optimize
        import ryzenai_onnx_utils.partitioner as partitioner
        install_precision_free_cast_recovery()
        namespace = partitioner.get_parser().parse_args([
            "optimize", "--input-model", str(copied / "model.onnx"),
            "--output-model", str(fixed_dir / "model.onnx"), "llm",
            "--prefill", "npu_eager", "--token", "npu_fusion", "--model-type", "qwen3.5",
            "--max-seq-len", "16384", "--no-prune-logits"])
        args = optimize.LlmArgs(namespace)
        optimized = fixed_dir / "optimized_model.onnx"
        optimize.llm_preprocess_optimize(args, optimized)
        args.input_model = optimized
        # Match the SDK token-fusion shape sequence without executing either
        # partitioning pass: establish 16K state shapes before fixing q_seq=1.
        args.input_model = optimize._llm_fix_shapes(args, optimize.Phase.PREFILL)
        fixed = Path(optimize._llm_fix_shapes(args, optimize.Phase.TOKEN)).resolve()
    evidence = verify_fixed_token(fixed, profile)
    write_new(work / "fixed-token-manifest.json", evidence)
    return fixed


def build(plan_path: Path, plan_sha256: str) -> dict:
    if sha256(plan_path) != plan_sha256:
        raise ValueError("Plan hash mismatch")
    plan = read_json(plan_path)
    fresh = create_plan(*(Path(plan[name]) for name in ("source", "prefill", "sdk_root", "work", "output", "profile")),
                        prefill_linear_attention=plan.get("prefill_linear_attention", "native"),
                        prefill_chunk_size=plan.get("prefill_chunk_size"))
    if fresh != plan:
        raise ValueError("Plan inputs, code, profile, or SDK changed")
    profile = read_json(Path(plan["profile"]))
    work, output = Path(plan["work"]), Path(plan["output"])
    work.mkdir(exist_ok=False)
    report = {"status": "failed", "plan_sha256": plan_sha256, "source_revision": SOURCE_REVISION,
              "prefill_linear_attention": plan["prefill_linear_attention"],
              "prefill_chunk_size": plan["prefill_chunk_size"],
              "code_sha256": plan["code_sha256"], "profile_sha256": plan["profile_sha256"],
              "runtime_validated": False, "stages": []}
    try:
        os.environ["RYZEN_AI_INSTALLATION_PATH"] = plan["sdk_root"]
        fixed = regenerate_token(Path(plan["source"]), work, plan["source_artifacts"], profile)
        report["stages"].append("fixed_token_regenerated_and_verified")
        try:
            from benchmarks.qwen35_dd import lower_token_graph
            from benchmarks.qwen35_package import package_model
        except ModuleNotFoundError:
            from qwen35_dd import lower_token_graph
            from qwen35_package import package_model
        token = work / "dd-token"
        report["lowering"] = lower_token_graph(fixed, token, profile=profile)
        report["stages"].append("token_lowered")
        report["package"] = package_model(Path(plan["prefill"]), token, output,
                                            sdk_root=Path(plan["sdk_root"]), profile=profile,
                                            prefill_linear_attention=plan["prefill_linear_attention"],
                                            prefill_chunk_size=plan["prefill_chunk_size"])
        for name in ("source", "prefill"):
            if input_identities(Path(plan[name]), profile[name + "_artifacts"]) != plan[name + "_artifacts"]:
                raise ValueError(f"Original {name} files changed during conversion")
        if check_sdk(Path(plan["sdk_root"]), profile) != plan["sdk"] or code_hashes() != plan["code_sha256"]:
            raise ValueError("SDK or conversion code changed during build")
        report["stages"].append("package_and_source_immutability_checked")
        report["status"] = "materialized_cpu_checked"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        write_new(work / "build-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("plan")
    for name in ("source", "prefill", "sdk-root", "work-dir", "output", "plan"):
        plan.add_argument("--" + name, required=True, type=Path)
    plan.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    plan.add_argument("--prefill-linear-attention", choices=("native", "token_loop", "adaptive"), default="native")
    plan.add_argument("--prefill-chunk-size", type=int, choices=(1024, 4096),
                      help="Global OGA chunk size for adaptive mode only (default: 1024)")
    run = sub.add_parser("build")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        target = args.plan.resolve()
        if target.exists() or any(target.is_relative_to(p.resolve()) for p in (args.source, args.prefill, args.output, args.work_dir)):
            parser.error("Plan must be a new file outside all input/output directories")
        result = create_plan(args.source, args.prefill, args.sdk_root, args.work_dir, args.output, args.profile,
                             prefill_linear_attention=args.prefill_linear_attention,
                             prefill_chunk_size=args.prefill_chunk_size)
        write_new(target, result)
        print(json.dumps({"status": result["status"], "plan": str(target), "sha256": sha256(target)}))
    else:
        result = build(args.plan.resolve(), args.plan_sha256)
        print(json.dumps({"status": result["status"], "runtime_validated": False}))


if __name__ == "__main__":
    main()
