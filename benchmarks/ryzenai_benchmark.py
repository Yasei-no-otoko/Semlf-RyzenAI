"""Create-only Ryzen AI evidence using the frozen SemIf speed/quality workloads.

Run from the repository root with the AMD 1.8 environment. Source records stay
in the local data directory; published predictions contain no input text.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

import evaluate
import evaluate_external
import evaluate_perturbations
from semif_phase1 import ryzenai_backend as backend
from semif_phase1.core import direct_messages
from semif_phase1.ryzenai_demo import DEFAULT_MODEL, DEFAULT_REVISION


def read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_prediction(row):
    result = {key: value for key, value in row.items() if key != "model"}
    result["model_reference"] = "manifest.json#/model"
    return result


def record(stream, row):
    stream.write(json.dumps(public_prediction(row), ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def quality_reports(datasets, predictions, actions=None):
    """Reuse the original metrics, without a synthetic second model."""
    report = {}
    for name in ("authored144", "perturbations108", "wanli256"):
        if name in predictions:
            report[name] = evaluate.evaluate(datasets[name], predictions[name])
    if "authored144" in predictions and "perturbations108" in predictions:
        report["perturbation_stability"] = evaluate_perturbations.evaluate_system(
            datasets["authored144"], datasets["perturbations108"],
            predictions["authored144"], predictions["perturbations108"])
    if "typesafe102" in predictions:
        all_predictions = predictions["typesafe102"]
        valid = [row for row in all_predictions if not row.get("error")]
        valid_ids = {row["id"] for row in valid}
        covered_gold = [row for row in datasets["typesafe102"] if row["id"] in valid_ids]
        aligned = evaluate.align(datasets["typesafe102"], all_predictions)
        groups = evaluate.clusters(aligned)
        report["typesafe102"] = {
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
    if "every204" in predictions:
        report["every204"] = evaluate_external.every_systems(
            datasets["every_gold154"], datasets["every204"],
            {"ryzenai_npu": predictions["every204"]}, actions)
    return report


def hardware_info():
    result = {"os": platform.platform(), "python": platform.python_version(),
              "logical_cpus": os.cpu_count(), "processor": platform.processor()}
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            result["processor"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    result["runtime_versions"] = {
        name: version(name) for name in (
            "onnxruntime-genai-directml-ryzenai", "onnxruntime-vitisai",
            "onnxruntime-providers-ryzenai", "ryzenai-dynamic-dispatch", "voe", "numpy")}
    return result


def npu_activity(output, phase):
    """Keep raw device snapshots local; publish only benchmark counters."""
    executable = shutil.which("xrt-smi")
    if executable is None:
        return {"available": False}
    raw_path = output / f"hardware-private-{phase}.json"
    if raw_path.exists():
        raise ValueError("Hardware snapshot path already exists")
    subprocess.run([executable, "--batch", "examine", "-r", "aie-partitions",
                    "-f", "JSON", "-o", str(raw_path)], check=True, capture_output=True, text=True)
    snapshot = json.loads(raw_path.read_text(encoding="utf-8"))
    contexts, other_submissions = [], 0
    for device in snapshot.get("devices", []):
        for partition in device.get("aie_partitions", {}).get("partitions", []):
            for context in partition.get("hw_contexts", []):
                if int(context["pid"]) == os.getpid():
                    contexts.append({key: context[key] for key in
                                     ("context_id", "command_submissions", "command_completions", "errors")})
                else:
                    other_submissions += int(context["command_submissions"])
    return {"available": True, "benchmark_contexts": contexts,
            "other_process_command_submissions": other_submissions}


def main():
    from ryzenai_speed import public_input

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Local outputs of the frozen WANLI/Every/TypeSafe builders")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "quality", "speed"), default="all")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new; benchmark evidence is never overwritten")
    paths = {name: Path("benchmarks/data") / f"{name}.jsonl"
             for name in ("authored144", "perturbations108", "shape777")}
    paths.update({"wanli256": args.data_dir / "wanli256.jsonl",
                  "typesafe102": args.data_dir / "typesafe102.jsonl",
                  "every204": args.data_dir / "every/inference204.jsonl",
                  "every_gold154": args.data_dir / "every/gold154.jsonl"})
    counts = {"authored144": 144, "perturbations108": 108, "shape777": 777,
              "wanli256": 256, "typesafe102": 102, "every204": 204, "every_gold154": 154}
    datasets, unavailable = {}, {}
    for name, path in paths.items():
        if not path.is_file():
            if name in {"authored144", "perturbations108", "shape777"}:
                raise ValueError(f"Missing owned fixture: {path}")
            unavailable[name] = "Frozen source snapshot unavailable; not measured"
            continue
        rows = read(path)
        if len(rows) != counts[name] or len({row["id"] for row in rows}) != counts[name]:
            raise ValueError(f"Wrong frozen row population for {name}")
        datasets[name] = rows
    actions_path = args.data_dir / "every/firewall-actions.json"
    actions = json.loads(actions_path.read_text(encoding="utf-8")) if actions_path.is_file() else None
    if "every204" in datasets and ("every_gold154" not in datasets or actions is None):
        raise ValueError("Every requires both gold154 and the frozen firewall actions")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    model, tokenizer, metadata = backend.load_model(args.model, args.revision)
    load_seconds = time.perf_counter() - started
    manifest = {
        "version": "ryzenai-benchmark-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite, "model": metadata, "hardware": hardware_info(),
        "process_id": os.getpid(), "model_load_seconds": load_seconds,
        "git_base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "benchmark_code_sha256": {name: hashlib.sha256(Path(name).read_text(encoding="utf-8").encode()).hexdigest()
                                   for name in ("benchmarks/ryzenai_benchmark.py", "benchmarks/ryzenai_speed.py",
                                                "benchmarks/evaluate.py", "benchmarks/evaluate_external.py",
                                                "benchmarks/evaluate_perturbations.py",
                                                "src/semif_phase1/ryzenai_backend.py", "src/semif_phase1/core.py")},
        "inputs": {name: {**(public_input(path, Path.cwd()) if name in
                             {"authored144", "perturbations108", "shape777"} else {}),
                          "file": path.name, "sha256": sha256(path), "rows": len(datasets[name])}
                   for name, path in paths.items() if name in datasets},
        "source_selection_sha256": sha256("benchmarks/manifests/source-selection.jsonl"),
        "evaluation_matrix_sha256": sha256("benchmarks/manifests/evaluation-matrix.jsonl"),
        "unavailable": unavailable,
        "timing_scope": "Warm model; prompt construction, tokenization, forward and readout included; model load and evidence writes excluded.",
        "unsupported_paths": ["serial_prefix", "parallel_shared", "native_reranker"],
        "limitations": ["AMD Qwen3-4B quantized NPU artifact differs from the upstream Qwen3.5-4B BF16 model.",
                        "Official NPU configuration includes CPU graph/host work; no GPU offload is configured.",
                        "No source text or reference document is copied into predictions.",
                        "Other idle demo processes may remain resident; this is a local systems measurement."],
    }
    if actions is not None:
        manifest["inputs"]["firewall_actions"] = {"file": actions_path.name, "sha256": sha256(actions_path), "actions": 10}
    write(args.output / "manifest.json", manifest)
    preflight = {}
    for name, rows in datasets.items():
        if name == "every_gold154":
            continue
        lengths = []
        for row in rows:
            prompt = tokenizer.reference.apply_chat_template(
                direct_messages(row, labels=backend.NPU_LABELS), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            lengths.append(len(tokenizer.reference.encode(prompt, add_special_tokens=False)))
        over = [{"id": row["id"], "input_tokens": length} for row, length in zip(rows, lengths)
                if length > metadata["context_ceiling"]]
        preflight[name] = {"rows": len(rows), "min_input_tokens": min(lengths),
                           "max_input_tokens": max(lengths), "over_context_limit": over}
    write(args.output / "preflight.json", preflight)
    if any(item["over_context_limit"] for name, item in preflight.items() if name != "typesafe102"):
        raise ValueError("Frozen inputs exceed the model context; see preflight.json. No truncation performed.")
    print(json.dumps({"loaded": load_seconds, "pid": os.getpid(), "preflight": preflight}), flush=True)
    hardware_before = npu_activity(args.output, "before")
    summary = {"unavailable": unavailable}
    if args.suite in {"all", "quality"}:
        predictions, timings = {}, {}
        for name in ("authored144", "perturbations108", "wanli256", "typesafe102", "every204"):
            if name not in datasets:
                continue
            rows = datasets[name]
            rejected = {item["id"]: item["input_tokens"] for item in preflight[name]["over_context_limit"]}
            first_valid = next((row for row in rows if row["id"] not in rejected), None)
            if first_valid is not None:
                backend.score(model, tokenizer, first_valid, metadata, metadata["context_ceiling"])
            values, durations = [], []
            with (args.output / f"{name}.predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for index, row in enumerate(rows):
                    mark = time.perf_counter()
                    if row["id"] in rejected:
                        result = {"id": row["id"], "option_ids": [option["id"] for option in row["options"]],
                                  "input_tokens": rejected[row["id"]], "error_type": "context_limit",
                                  "context_limit": metadata["context_ceiling"],
                                  "error": "Frozen prompt exceeds the compiled NPU context; no truncation performed"}
                    else:
                        result = backend.score(model, tokenizer, row, metadata, metadata["context_ceiling"])
                    durations.append(time.perf_counter() - mark)
                    values.append(public_prediction(result))
                    record(stream, result)
                    if (index + 1) % 25 == 0 or index + 1 == len(rows):
                        print(f"quality/{name}: {index + 1}/{len(rows)}", flush=True)
            predictions[name] = values
            timings[name] = {"rows": len(values), "scoring_seconds": sum(durations),
                             "median_row_seconds": statistics.median(durations)}
        quality = quality_reports(datasets, predictions, actions)
        write(args.output / "quality.json", quality)
        write(args.output / "quality-timing.json", timings)
        summary["quality"] = {name: quality[name]["mean_family_balanced_accuracy"]
                              for name in ("authored144", "perturbations108", "wanli256") if name in quality}
        if "typesafe102" in quality:
            summary["quality"]["typesafe102"] = {
                key: value for key, value in quality["typesafe102"].items()
                if key != "full_denominator_alignment"}
        if "every204" in quality:
            summary["quality"]["every204"] = quality["every204"]
    if args.suite in {"all", "speed"}:
        from ryzenai_speed import benchmark_decision21, benchmark_shape777_fresh

        for name, function, rows in (
                ("compact", benchmark_decision21, datasets["shape777"][:21]),
                ("shape777", benchmark_shape777_fresh, datasets["shape777"])):
            with (args.output / f"{name}.predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                def on_row(section, repeat, result):
                    record(stream, {"section": section, "repeat": repeat, **result})
                extra = {"on_progress": lambda done, total: print(f"shape777: {done}/{total} states", flush=True)} if name == "shape777" else {}
                report = function(model, tokenizer, metadata, rows, on_row=on_row, **extra)
            write(args.output / f"{name}.json", report)
            if name == "compact":
                summary[name] = {
                    "direct_median_seconds": report["direct_fresh"]["median_total_seconds"],
                    "generation_median_seconds": report["compact_generation"]["median_total_seconds"],
                    "median_output_tokens": report["compact_generation"]["median_output_tokens"],
                    "all_runs_valid_complete_arrays": report["compact_generation"]["all_runs_valid_complete_arrays"],
                    "generation_over_direct_ratio": report["median_wall_ratio_generation_over_direct"],
                }
            else:
                summary[name] = {key: report[key] for key in
                                 ("wall_seconds", "decisions_per_second", "state_p50_seconds")}
    hardware_after = npu_activity(args.output, "after")
    write(args.output / "hardware.json", {"before": hardware_before, "after": hardware_after})
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
