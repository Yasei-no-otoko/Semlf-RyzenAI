"""Verify create-only Ryzen AI evidence without invoking the NPU runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

import evaluate
import evaluate_external
import evaluate_perturbations


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-10
OWNED = {
    "authored144": ROOT / "benchmarks/data/authored144.jsonl",
    "perturbations108": ROOT / "benchmarks/data/perturbations108.jsonl",
    "shape777": ROOT / "benchmarks/data/shape777.jsonl",
}
EXTERNAL = {
    "wanli256": Path("wanli256.jsonl"),
    "typesafe102": Path("typesafe102.jsonl"),
    "every204": Path("every/inference204.jsonl"),
    "every_gold154": Path("every/gold154.jsonl"),
    "firewall_actions": Path("every/firewall-actions.json"),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_digest(content: bytes) -> str:
    """Hash source using the same newline normalization as ``Path.read_text``."""
    text = content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def _historical_code_revision(relative: str, expected: str) -> str | None:
    """Find an available Git revision containing the exact recorded source.

    Evidence records source-text hashes, rather than Git blob hashes.  Searching
    the local history lets a later working tree verify an older run without
    accepting an unchecked version or revision allowlist.  Missing Git metadata
    (for example, a source archive) simply leaves the hash unverifiable.
    """
    if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        return None
    try:
        path = (ROOT / relative).resolve()
        git_path = path.relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return None
    try:
        revisions = subprocess.run(
            ["git", "log", "--all", "--format=%H", "--", git_path],
            cwd=ROOT, capture_output=True, text=True, check=False, timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    for revision in revisions[:512]:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            continue
        try:
            source = subprocess.run(
                ["git", "show", f"{revision}:{git_path}"],
                cwd=ROOT, capture_output=True, check=False, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if source.returncode == 0 and _text_digest(source.stdout) == expected:
            return revision
    return None


def verify_code_hash(relative: str, expected: str) -> str | None:
    """Return how a recorded source hash was verified, or ``None`` on failure."""
    if not isinstance(relative, str) or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        return None
    try:
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
    except (OSError, ValueError):
        return None
    if path.is_file() and _text_digest(path.read_bytes()) == expected:
        return "working_tree"
    return _historical_code_revision(relative, expected)


def close(actual, expected, path="$"):
    if isinstance(actual, float) and isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in actual:
            close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            close(left, right, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def require(value, message):
    if not value:
        raise AssertionError(message)


def input_paths(data_dir: Path | None):
    paths = dict(OWNED)
    if data_dir is not None:
        paths.update({name: data_dir / relative for name, relative in EXTERNAL.items()})
    return paths


def verify_manifest(manifest, data_dir: Path | None, code_sources: dict[str, str] | None = None):
    require(manifest.get("version") == "ryzenai-benchmark-v1", "Unexpected benchmark manifest version")
    model = manifest.get("model")
    require(isinstance(model, dict), "Missing model metadata")
    require(model.get("backend") == "ryzenai-npu", "Evidence was not scored by ryzenai-npu")
    require(model.get("oga_distribution") == "onnxruntime-genai-directml-ryzenai", "Unexpected OGA distribution")
    require(model.get("oga_version") == "0.14.0", "Unexpected OGA version")
    require(isinstance(model.get("revision"), str) and re.fullmatch(r"[0-9a-f]{40}", model["revision"]),
            "Model revision must be a pinned 40-character commit")
    require(isinstance(model.get("context_ceiling"), int) and model["context_ceiling"] > 0,
            "Missing valid NPU context ceiling")
    artifacts = model.get("source_artifact_sha256")
    require(isinstance(artifacts, dict) and "genai_config.json" in artifacts, "Missing NPU config hash")
    require(all(isinstance(value, str) and len(value) == 64 for value in artifacts.values()), "Invalid artifact hash")
    runtime = manifest.get("hardware", {}).get("runtime_versions", {})
    expected_runtime = {
        "onnxruntime-genai-directml-ryzenai": "0.14.0", "onnxruntime-vitisai": "1.27.0",
        "onnxruntime-providers-ryzenai": "1.8.0", "ryzenai-dynamic-dispatch": "1.8.0", "voe": "1.8.0",
    }
    for name, expected in expected_runtime.items():
        require(runtime.get(name) == expected, f"Unexpected runtime version for {name}")
    paths = input_paths(data_dir)
    for name, details in manifest.get("inputs", {}).items():
        path = paths.get(name)
        if path is None or not path.is_file():
            continue
        require(details.get("file") == path.name, f"Input filename differs for {name}")
        require(details.get("sha256") == digest(path), f"Input checksum differs for {name}")
        if name != "firewall_actions":
            require(details.get("rows") == len(read_jsonl(path)), f"Input row count differs for {name}")
    for relative, expected in manifest.get("benchmark_code_sha256", {}).items():
        source = verify_code_hash(relative, expected)
        require(source is not None, f"Code checksum differs for {relative}")
        if code_sources is not None:
            code_sources[relative] = "working_tree" if source == "working_tree" else f"git:{source}"
    return model


def verify_predictions(path: Path, gold, model, context_errors=None):
    rows = read_jsonl(path)
    require(len(rows) == len(gold), f"Wrong prediction count: {path.name}")
    by_id = {row.get("id"): row for row in rows}
    require(len(by_id) == len(rows) and set(by_id) == {row["id"] for row in gold}, f"Prediction IDs differ: {path.name}")
    context_errors = context_errors or {}
    for source in gold:
        row = by_id[source["id"]]
        require("model" not in row and row.get("model_reference") == "manifest.json#/model",
                f"Invalid provenance reference: {source['id']}")
        require(row.get("option_ids") == [option["id"] for option in source["options"]], f"Option IDs differ: {source['id']}")
        if row.get("error"):
            expected_tokens = context_errors.get(source["id"])
            require(expected_tokens is not None and row.get("error_type") == "context_limit"
                    and row.get("context_limit") == model["context_ceiling"]
                    and row.get("input_tokens") == expected_tokens,
                    f"Unexpected context rejection: {source['id']}")
            continue
        require(source["id"] not in context_errors, f"Missing context rejection: {source['id']}")
        evaluate.vector(row.get("probabilities"), row["option_ids"])
        logits = row.get("option_logits")
        require(isinstance(logits, list) and len(logits) == len(row["option_ids"]) and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in logits),
            f"Invalid logits: {source['id']}")
        require(isinstance(row.get("input_tokens"), int) and 0 < row["input_tokens"] <= model["context_ceiling"],
                f"Invalid input token count: {source['id']}")
        require(isinstance(row.get("prompt_sha256"), str) and len(row["prompt_sha256"]) == 64,
                f"Invalid prompt hash: {source['id']}")
    return rows


def verify_quality(run_dir: Path, manifest, data_dir: Path | None):
    quality_path = run_dir / "quality.json"
    if not quality_path.is_file():
        return [], []
    paths = input_paths(data_dir)
    datasets = {name: read_jsonl(path) for name, path in paths.items()
                if name != "firewall_actions" and path.is_file()}
    preflight = read_json(run_dir / "preflight.json") if (run_dir / "preflight.json").is_file() else {}
    predictions = {}
    for name in ("authored144", "perturbations108", "wanli256", "typesafe102", "every204"):
        path = run_dir / f"{name}.predictions.jsonl"
        if path.is_file():
            require(name in datasets, f"Cannot verify {name} without its frozen source rows")
            over = {item["id"]: item["input_tokens"] for item in preflight.get(name, {}).get("over_context_limit", [])}
            require(len(over) == len(preflight.get(name, {}).get("over_context_limit", [])), f"Duplicate preflight IDs: {name}")
            require(all(identifier in {row["id"] for row in datasets[name]} and tokens > manifest["model"]["context_ceiling"]
                        for identifier, tokens in over.items()), f"Invalid preflight context limit: {name}")
            predictions[name] = verify_predictions(path, datasets[name], manifest["model"], over)
    expected = {}
    for name in ("authored144", "perturbations108", "wanli256"):
        if name in predictions:
            expected[name] = evaluate.evaluate(datasets[name], predictions[name])
    if "authored144" in predictions and "perturbations108" in predictions:
        expected["perturbation_stability"] = evaluate_perturbations.evaluate_system(
            datasets["authored144"], datasets["perturbations108"], predictions["authored144"], predictions["perturbations108"])
    if "typesafe102" in predictions:
        all_predictions = predictions["typesafe102"]
        valid = [row for row in all_predictions if not row.get("error")]
        valid_ids = {row["id"] for row in valid}
        covered_gold = [row for row in datasets["typesafe102"] if row["id"] in valid_ids]
        aligned = evaluate.align(datasets["typesafe102"], all_predictions)
        groups = evaluate.clusters(aligned)
        expected["typesafe102"] = {
            "status": "complete" if len(valid) == len(all_predictions) else "partial_context_limit",
            "frozen_rows": len(all_predictions), "scored_rows": len(valid),
            "rejected_rows": len(all_predictions) - len(valid),
            "full_denominator_equal_case_modal_agreement": statistics.mean(
                statistics.mean(float(row["correct"]) for row in group) for group in groups.values()),
            "full_denominator_alignment": evaluate.evaluate(datasets["typesafe102"], all_predictions),
            "covered_only": evaluate_external.type_safe_systems(
                covered_gold, {"ryzenai_npu": valid}) if valid else None,
            "limitation": "Covered-only metrics use fewer rows/cases and are not the published 102-row comparison; context rejections remain failures in full-denominator metrics.",
        }
        if expected["typesafe102"]["status"] == "partial_context_limit":
            require((expected["typesafe102"]["frozen_rows"], expected["typesafe102"]["scored_rows"],
                     expected["typesafe102"]["rejected_rows"]) == (102, 73, 29), "Unexpected TypeSafe coverage")
    if "every204" in predictions:
        actions_path = paths.get("firewall_actions")
        require("every_gold154" in datasets and actions_path is not None and actions_path.is_file(),
                "Every predictions require local gold rows and firewall actions")
        expected["every204"] = evaluate_external.every_systems(
            datasets["every_gold154"], datasets["every204"], {"ryzenai_npu": predictions["every204"]}, read_json(actions_path))
    close(read_json(quality_path), expected, "quality")
    unavailable = [name for name in ("wanli256", "typesafe102", "every204") if name not in datasets]
    return list(predictions), unavailable


def verify_shape(run_dir: Path, manifest):
    path = run_dir / "shape777.json"
    if not path.is_file():
        return 0
    report = read_json(path)
    close(report.get("model"), manifest["model"], "shape777.model")
    gold = read_jsonl(OWNED["shape777"])
    outputs = report.get("outputs")
    verify_predictions_rows(outputs, gold, manifest["model"], "shape777 outputs")
    groups = report.get("groups")
    require(isinstance(groups, list) and len(groups) == 37, "Shape777 requires 37 state timings")
    expected_groups = {row["group_id"] for row in gold}
    require({group.get("group_id") for group in groups} == expected_groups and all(group.get("decisions") == 21 for group in groups),
            "Shape777 group coverage differs")
    require(report.get("wall_seconds", 0) > 0, "Invalid Shape777 wall time")
    close(report.get("decisions_per_second"), 777 / report["wall_seconds"], "shape throughput")
    close(report.get("state_p50_seconds"), statistics.median(group["total_seconds"] for group in groups), "shape p50")
    emitted = read_jsonl(run_dir / "shape777.predictions.jsonl")
    require(len(emitted) == 777 and all(row.get("section") == "shape777_fresh" for row in emitted), "Shape777 emitted rows differ")
    require([row.get("id") for row in emitted] == [row["id"] for row in outputs], "Shape777 emitted IDs differ")
    return 777


def verify_predictions_rows(rows, gold, model, label):
    require(isinstance(rows, list) and len(rows) == len(gold), f"Wrong {label} count")
    by_id = {row.get("id"): row for row in rows}
    require(len(by_id) == len(rows) and set(by_id) == {row["id"] for row in gold}, f"{label} IDs differ")
    for source in gold:
        row = by_id[source["id"]]
        require(row.get("option_ids") == [option["id"] for option in source["options"]], f"{label} options differ")
        evaluate.vector(row.get("probabilities"), row["option_ids"])
        require(isinstance(row.get("input_tokens"), int) and 0 < row["input_tokens"] <= model["context_ceiling"], f"{label} token count differs")


def verify_compact(run_dir: Path, manifest):
    path = run_dir / "compact.json"
    if not path.is_file():
        return 0
    report = read_json(path)
    close(report.get("model"), manifest["model"], "compact.model")
    direct = report.get("direct_fresh", {}).get("runs")
    generated = report.get("compact_generation", {}).get("runs")
    gold = read_jsonl(OWNED["shape777"])[:21]
    require(isinstance(direct, list) and isinstance(generated, list) and len(direct) == len(generated) == 3,
            "Compact benchmark requires three paired runs")
    for index, (fresh, generation) in enumerate(zip(direct, generated)):
        require(fresh.get("repeat") == generation.get("repeat") == index and fresh.get("total_seconds", 0) > 0,
                "Compact repeat metadata differs")
        verify_predictions_rows(fresh.get("outputs"), gold, manifest["model"], f"compact direct {index}")
        timeline = generation.get("timeline")
        choices = generation.get("choices")
        require(generation.get("valid_complete_array") is True and isinstance(choices, list) and len(choices) == 21
                and all(choice in {"yes", "no"} for choice in choices), "Compact choices are not strict")
        require(isinstance(timeline, list) and generation.get("output_tokens") == len(timeline), "Compact token timeline differs")
        require(timeline and generation.get("time_to_first_token_seconds") == timeline[0]["seconds"]
                and generation.get("total_seconds", 0) >= timeline[-1]["seconds"], "Compact timings differ")
        require(all(left["seconds"] <= right["seconds"] for left, right in zip(timeline, timeline[1:])), "Compact timeline is unordered")
    direct_median = statistics.median(item["total_seconds"] for item in direct)
    generation_median = statistics.median(item["total_seconds"] for item in generated)
    close(report["direct_fresh"]["median_total_seconds"], direct_median, "compact direct median")
    close(report["compact_generation"]["median_total_seconds"], generation_median, "compact generation median")
    close(report["median_wall_ratio_generation_over_direct"], generation_median / direct_median, "compact ratio")
    emitted = read_jsonl(run_dir / "compact.predictions.jsonl")
    require(len(emitted) == 63 and all(row.get("section") == "decision21_direct" for row in emitted), "Compact emitted rows differ")
    return 63


def hardware_counter(context: dict, name: str) -> int:
    value = context.get(name)
    require(type(value) is int or (isinstance(value, str) and value.isascii() and value.isdigit()),
            f"Invalid NPU hardware counter: {name}")
    value = int(value)
    require(value >= 0, f"Negative NPU hardware counter: {name}")
    return value


def verify(run_dir: Path, data_dir: Path | None = None):
    run_dir = run_dir.resolve()
    manifest = read_json(run_dir / "manifest.json")
    code_sources = {}
    verify_manifest(manifest, data_dir, code_sources)
    quality, external_unverified = verify_quality(run_dir, manifest, data_dir)
    shape = verify_shape(run_dir, manifest)
    compact = verify_compact(run_dir, manifest)
    hardware_path = run_dir / "hardware.json"
    if hardware_path.is_file():
        hardware = read_json(hardware_path)
        before, after = hardware.get("before"), hardware.get("after")
        require(isinstance(before, dict) and isinstance(after, dict), "Invalid filtered hardware counters")
        if before.get("available") or after.get("available"):
            require(before.get("available") is True and after.get("available") is True,
                    "Hardware counter availability changed during the benchmark")
            old = {item.get("context_id"): item for item in before.get("benchmark_contexts", [])}
            new = {item.get("context_id"): item for item in after.get("benchmark_contexts", [])}
            shared = old.keys() & new.keys()
            require(shared, "No persistent benchmark NPU context was observed")
            require(all(hardware_counter(new[key], "errors") == 0 for key in shared), "NPU hardware errors were reported")
            require(any(hardware_counter(new[key], "command_submissions") > hardware_counter(old[key], "command_submissions")
                        for key in shared), "NPU command submissions did not increase")
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        require(summary.get("unavailable") == manifest.get("unavailable", {}), "Summary unavailable sources differ")
        if compact:
            compact_report = read_json(run_dir / "compact.json")
            expected_compact = {
                "direct_median_seconds": compact_report["direct_fresh"]["median_total_seconds"],
                "generation_median_seconds": compact_report["compact_generation"]["median_total_seconds"],
                "median_output_tokens": compact_report["compact_generation"]["median_output_tokens"],
                "all_runs_valid_complete_arrays": compact_report["compact_generation"]["all_runs_valid_complete_arrays"],
                "generation_over_direct_ratio": compact_report["median_wall_ratio_generation_over_direct"],
            }
            close(summary.get("compact"), expected_compact, "summary.compact")
        if shape:
            shape_report = read_json(run_dir / "shape777.json")
            close(summary.get("shape777"), {key: shape_report[key] for key in
                  ("wall_seconds", "decisions_per_second", "state_p50_seconds")}, "summary.shape777")
    result = {"run": str(run_dir), "verified_quality_sets": quality, "verified_speed_rows": shape + compact,
              "external_unverified": external_unverified, "code_source_verification": code_sources,
              "status": "ok"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.data_dir), ensure_ascii=False))
