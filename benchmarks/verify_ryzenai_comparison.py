"""CPU verification of a sanitized comparison bundle, without model inference.

Default: checksums, all prediction rows, speed arithmetic, owned Quality, saved
guard/counter/control evidence. --data-dir additionally recomputes full Quality.
Generated JSON text is intentionally unavailable: strict parsing is not repeated.
No inference or private cache helpers are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import verify_ryzenai as v  # noqa: E402

QUALITY = ("authored144", "perturbations108", "wanli256", "typesafe102", "every204")
COUNTS = dict(
    authored144=144, perturbations108=108, wanli256=256, typesafe102=102, every204=204
)
VOCAB = {"qwen3": 151936, "qwen35": 248320}
REVISIONS = {
    "qwen3": "715d60818350b685ca2af3566e5ae38f4780daf0",
    "qwen35": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}
LABELS = ("qwen3-speed", "qwen35-all", "qwen3-quality-historical")
require, close, read, rows = v.require, v.close, v.read_json, v.read_jsonl


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value, *, positive=False):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        "Invalid finite timing/count",
    )
    return value


def sha_value(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
        "Invalid SHA-256",
    )


def expected_files():
    names = {"comparison.json", "execution.json"}
    for index, label in enumerate(LABELS):
        children = {"manifest.json", "preflight.json", "hardware.json"}
        if index < 2:
            children |= {
                "observations.json",
                "speed.json",
                "load-gate.json",
                "shape777.predictions.jsonl",
                "compact.predictions.jsonl",
            }
        if index > 0:
            children |= {
                "quality.json",
                "quality-timing.json",
                *(name + ".predictions.jsonl" for name in QUALITY),
            }
        names.update(label + "/" + child for child in children)
    return names


def check_manifest(bundle):
    manifest = read(bundle / "manifest.json")
    require(
        manifest["schema_version"] == 1
        and manifest["status"] == "completed_comparison_published",
        "Unsupported/incomplete publication",
    )
    require(
        set(manifest["artifacts"]) == expected_files(),
        "Publication file population differs",
    )
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    require(
        actual == expected_files() | {"manifest.json"}, "Extra/missing public files"
    )
    for name, identity in manifest["artifacts"].items():
        path = (bundle / name).resolve()
        require(
            path.is_relative_to(bundle)
            and path.stat().st_size == identity["bytes"]
            and digest(path) == identity["sha256"],
            "Curated checksum/size differs: " + name,
        )
    predicted = {}
    for index, label in enumerate(LABELS):
        if index < 2:
            predicted.update(
                {
                    label + "/shape777.predictions.jsonl": 777,
                    label + "/compact.predictions.jsonl": 63,
                }
            )
        if index > 0:
            predicted.update(
                {
                    label + "/" + name + ".predictions.jsonl": count
                    for name, count in COUNTS.items()
                }
            )
    require(
        manifest["prediction_rows"] == predicted
        and manifest["total_prediction_rows"] == 3308,
        "Declared prediction populations differ",
    )
    for name, count in predicted.items():
        require(len(rows(bundle / name)) == count, "Actual prediction count differs")
    return manifest


def check_predictions(predicted, expected=None):
    require(
        len({row["id"] for row in predicted}) == len(predicted),
        "Duplicate prediction ID",
    )
    if expected is not None:
        require(
            [row["id"] for row in predicted] == [row["id"] for row in expected],
            "Prediction order/population differs",
        )
    for index, row in enumerate(predicted):
        require(
            "error" not in row and row["model_reference"] == "manifest.json#/model",
            "Invalid prediction provenance",
        )
        require(
            type(row["input_tokens"]) is int and 0 < row["input_tokens"] <= 16384,
            "Invalid input length",
        )
        if expected is not None:
            require(
                row["option_ids"]
                == [option["id"] for option in expected[index]["options"]],
                "Option IDs differ",
            )
        v.evaluate.vector(row["probabilities"], row["option_ids"])
        require(
            len(row["option_logits"])
            == len(row["answer_token_ids"])
            == len(row["option_ids"])
            and all(
                type(x) in (int, float) and math.isfinite(x)
                for x in row["option_logits"]
            ),
            "Invalid option logits",
        )
        require(
            all(type(x) is int and x >= 0 for x in row["answer_token_ids"]),
            "Invalid answer-slot IDs",
        )
        for key in ("prompt_sha256", "input_ids_sha256"):
            sha_value(row[key])
        number(row["forward_seconds"], positive=True)
        number(row["total_seconds"], positive=True)


def speed(run):
    raw = read(run / "speed.json")
    shape, compact = raw["shape777"], raw["compact"]
    predictions = rows(run / "shape777.predictions.jsonl")
    gold = rows(v.OWNED["shape777"])
    check_predictions(predictions, gold)
    require(
        all(row["input_tokens"] <= 4096 for row in predictions),
        "Speed prompt budget differs",
    )
    groups = shape["groups"]
    require(
        len(groups) == 37
        and [g["group_id"] for g in groups]
        == list(dict.fromkeys(r["group_id"] for r in gold)),
        "Shape group ordering differs",
    )
    require(all(g["decisions"] == 21 for g in groups), "Shape group width differs")
    for index, row in enumerate(predictions):
        require(
            row["section"] == "shape777_fresh"
            and row["repeat"] == groups[index // 21]["group_id"],
            "Shape emitted group differs",
        )
    durations = [number(group["total_seconds"], positive=True) for group in groups]
    wall = sum(durations)
    close(shape["wall_seconds"], wall)
    close(shape["decisions_per_second"], 777 / wall)
    close(shape["state_p50_seconds"], statistics.median(durations))
    direct, generated = (
        compact["direct_fresh"]["runs"],
        compact["compact_generation"]["runs"],
    )
    require(
        len(direct) == len(generated) == 3
        and compact["compact_generation"]["max_new_tokens"] == 128,
        "Speed repeat/generation protocol differs",
    )
    emitted = rows(run / "compact.predictions.jsonl")
    require(len(emitted) == 63, "Compact direct population differs")
    result_rows = []
    for i, (fresh, generation) in enumerate(zip(direct, generated, strict=True)):
        selected = emitted[21 * i : 21 * (i + 1)]
        check_predictions(selected, gold[:21])
        require(
            all(
                r["section"] == "decision21_direct" and r["repeat"] == i
                for r in selected
            ),
            "Direct repeat differs",
        )
        for a, b in zip(selected, predictions[:21], strict=True):
            require(
                all(
                    a[key] == b[key]
                    for key in ("input_tokens", "prompt_sha256", "input_ids_sha256")
                ),
                "Direct input fingerprint differs",
            )
        require(fresh["repeat"] == generation["repeat"] == i, "Repeat order differs")
        number(fresh["total_seconds"], positive=True)
        number(generation["total_seconds"], positive=True)
        require(
            0 < generation["input_tokens"] <= 4096 - 128,
            "Generation context reserve differs",
        )
        sha_value(generation["prompt_sha256"])
        sha_value(generation["output_text_sha256"])
        timeline = generation["timeline"]
        require(
            1 <= len(timeline) <= 128 and generation["output_tokens"] == len(timeline),
            "Generated population differs",
        )
        for index, sample in enumerate(timeline, 1):
            require(
                sample["sample"] == index and type(sample["is_configured_eos"]) is bool,
                "Sample sequence malformed",
            )
            number(sample["seconds"])
        require(
            all(a["seconds"] <= b["seconds"] for a, b in zip(timeline, timeline[1:]))
            and generation["total_seconds"] >= timeline[-1]["seconds"]
            and generation["time_to_first_token_seconds"] == timeline[0]["seconds"],
            "Token timing differs",
        )
        require(
            not any(item["is_configured_eos"] for item in timeline[:-1]),
            "Samples continued after EOS",
        )
        eos = timeline[-1]["is_configured_eos"]
        require(
            generation["ended_by_eos"] is eos
            and generation["truncated"] is (not eos)
            and (eos or len(timeline) == 128),
            "EOS/truncation accounting differs",
        )
        valid = generation["valid_complete_array"]
        require(type(valid) is bool, "Invalid strict-JSON result flag")
        choices = generation["choices"]
        require(
            (
                isinstance(choices, list)
                and len(choices) == 21
                and all(x in ("yes", "no") for x in choices)
            )
            if valid
            else choices is None,
            "Recorded strict choices malformed",
        )
        chosen = [
            r["option_ids"][
                max(range(len(r["probabilities"])), key=r["probabilities"].__getitem__)
            ]
            for r in selected
        ]
        agreement = sum(a == b for a, b in zip(chosen, choices)) / 21 if valid else None
        close(generation["agreement_with_direct_argmax"], agreement)
        result_rows.append(
            dict(
                repeat=i,
                direct21_seconds=fresh["total_seconds"],
                compact_seconds=generation["total_seconds"],
                input_tokens=generation["input_tokens"],
                output_tokens=len(timeline),
                valid_complete_array=valid,
                ended_by_eos=eos,
                truncated=not eos,
                agreement_with_direct_argmax=agreement,
            )
        )
    dm = statistics.median(r["direct21_seconds"] for r in result_rows)
    gm = statistics.median(r["compact_seconds"] for r in result_rows)
    close(compact["direct_fresh"]["median_total_seconds"], dm)
    close(compact["compact_generation"]["median_total_seconds"], gm)
    close(compact["median_wall_ratio_generation_over_direct"], gm / dm)
    truncated = [r["repeat"] for r in result_rows if r["truncated"]]
    require(
        compact["compact_generation"]["truncated_repeats"] == truncated
        and compact["compact_generation"]["all_runs_valid_complete_arrays"]
        is all(r["valid_complete_array"] for r in result_rows)
        and compact["compact_generation"]["all_runs_ended_by_eos"]
        is all(r["ended_by_eos"] for r in result_rows),
        "Aggregate generation status differs",
    )
    median_tokens = statistics.median(r["output_tokens"] for r in result_rows)
    close(compact["compact_generation"]["median_output_tokens"], median_tokens)
    return dict(
        direct21_median_seconds=dm,
        compact_median_seconds=gm,
        compact_over_direct_ratio=gm / dm,
        valid_compact_repeats=sum(r["valid_complete_array"] for r in result_rows),
        repeats=3,
        eos_repeats=sum(r["ended_by_eos"] for r in result_rows),
        truncated_repeats=truncated,
        median_output_tokens=median_tokens,
        runs=result_rows,
        shape777_total_seconds=wall,
        shape777_decisions_per_second=777 / wall,
        shape777_state_p50_seconds=statistics.median(durations),
    )


def quality(run, manifest, data_dir=None):
    report = read(run / "quality.json")
    predicted = {}
    for name in QUALITY:
        predicted[name] = rows(run / (name + ".predictions.jsonl"))
        require(len(predicted[name]) == COUNTS[name], "Quality population differs")
        check_predictions(
            predicted[name], rows(v.OWNED[name]) if name in v.OWNED else None
        )
    for name in ("authored144", "perturbations108"):
        gold = rows(v.OWNED[name])
        require(
            manifest["inputs"][name]["sha256"] == digest(v.OWNED[name]),
            "Owned input differs",
        )
        close(report[name], v.evaluate.evaluate(gold, predicted[name]), name)
    close(
        report["perturbation_stability"],
        v.evaluate_perturbations.evaluate_system(
            rows(v.OWNED["authored144"]),
            rows(v.OWNED["perturbations108"]),
            predicted["authored144"],
            predicted["perturbations108"],
        ),
    )
    if data_dir is not None:
        for name, path in v.input_paths(data_dir).items():
            require(
                path.is_file() and manifest["inputs"][name]["sha256"] == digest(path),
                "Missing/changed frozen external input",
            )
        names, unavailable = v.verify_quality(run, manifest, data_dir)
        require(
            set(names) == set(QUALITY) and not unavailable,
            "Full external Quality incomplete",
        )
    ts = report["typesafe102"]
    require(
        (ts["status"], ts["frozen_rows"], ts["scored_rows"], ts["rejected_rows"])
        == ("complete", 102, 102, 0),
        "TypeSafe coverage differs",
    )
    tc = ts["covered_only"]["ryzenai_npu"]
    require(tc["rows"] == 102 and tc["cases"] == 20, "TypeSafe case population differs")
    every = report["every204"]["ryzenai_npu"]
    require(
        {k: x["rows"] for k, x in every.items()}
        == {
            "judge-grid": 36,
            "action-firewall": 50,
            "code-rag": 48,
            "company-brain": 70,
        },
        "Every population differs",
    )
    result = {
        name: dict(
            rows=COUNTS[name],
            mean_family_balanced_accuracy=report[name]["mean_family_balanced_accuracy"],
        )
        for name in QUALITY[:3]
    }
    result["typesafe102"] = dict(
        rows=102,
        cases=20,
        equal_case_modal_agreement=tc["equal_case_modal_agreement"],
        equal_case_total_variation=tc["equal_case_total_variation"],
    )
    result.update(
        every_judge_grid=dict(rows=36, accuracy=every["judge-grid"]["accuracy"]),
        every_action_firewall=dict(
            rows=50, actions=10, accuracy=every["action-firewall"]["accuracy"]
        ),
        every_code_retrieval=dict(
            rows=48, queries=6, recall_at_1=every["code-rag"]["recall_at_1"]
        ),
        every_company_knowledge=dict(
            rows=70, queries=7, recall_at_1=every["company-brain"]["recall_at_1"]
        ),
    )
    return result


def counters(value):
    def table(snapshot):
        require(snapshot["available"] is True, "Counter availability missing")
        result = {}
        for row in snapshot["contexts"]:
            label = row["label"]
            require(
                isinstance(label, str)
                and re.fullmatch(r"context-[1-9][0-9]*", label)
                and label not in result,
                "Duplicate/malformed context label",
            )
            item = {
                key: v.hardware_counter(row, key)
                for key in ("command_submissions", "command_completions", "errors")
            }
            require(
                item["errors"] == 0
                and item["command_submissions"] == item["command_completions"],
                "NPU error/unfinished work",
            )
            result[label] = item
        return result

    before, after = table(value["before"]), table(value["after"])
    require(before.keys() <= after.keys(), "Context disappeared")
    total = dict(command_submissions=0, command_completions=0, errors=0)
    for label, item in after.items():
        for key in total:
            delta = item[key] - before.get(label, {}).get(key, 0)
            require(delta >= 0, "Counter regressed")
            total[key] += delta
    require(
        total["command_submissions"] == total["command_completions"] > 0,
        "No completed NPU work",
    )
    close(value["total_delta"], total)
    return total


def guard(run, manifest, key):
    observed = read(run / "observations.json")
    require(
        observed["status"] == "complete"
        and all(
            observed[name] is True
            for name in (
                "code_unchanged",
                "runtime_unchanged",
                "package_manifest_unchanged",
                "model_artifacts_unchanged",
            )
        ),
        "Observed identity gate failed",
    )
    close(observed["provenance"], manifest["observation"])
    schedule = []

    def add(values):
        schedule.extend((value["input_tokens"], 0) for value in values)

    if key == "qwen35":
        for name in QUALITY:
            values = rows(run / (name + ".predictions.jsonl"))
            add(values[:1])
            add(values)
    shape = rows(run / "shape777.predictions.jsonl")
    compact = read(run / "speed.json")["compact"]["compact_generation"]["runs"]
    add(shape[:21])
    schedule.append((None, 16))
    for generation in compact:
        add(shape[:21])
        schedule.append((generation["input_tokens"], generation["output_tokens"]))
    add(shape[:1])
    add(shape)
    records = observed["generator_observations"]
    require(
        len(records) == len(schedule) == (1685 if key == "qwen35" else 866),
        "Generator schedule differs",
    )
    checks, sample_counts = 0, []
    for index, (record, (length, samples)) in enumerate(
        zip(records, schedule, strict=True), 1
    ):
        require(
            record["generator_index"] == index
            and record["constructed"] is True
            and record["errors"] == [],
            "Failed/reordered Generator",
        )
        require(
            record["append_calls_started"] == record["append_calls_completed"] == 1,
            "Unexpected append count",
        )
        lengths = record["appended_token_counts"]
        require(
            len(lengths) == 1 and lengths == [length]
            if length is not None
            else len(lengths) == 1 and 0 < lengths[0] <= 4096 - 16,
            "Scheduled input length differs",
        )
        count = record["sampling_calls_completed"]
        require(
            type(count) is int
            and record["sampling_calls_started"] == count
            and (1 <= count <= 16 if length is None else count == samples),
            "Sampling schedule differs",
        )
        n = count if count else 1
        require(
            record["native_logits_calls_started"]
            == record["native_logits_calls_completed"]
            == len(record["logits_checks"])
            == n
            and record["explicit_logits_calls"] == (0 if count else 1),
            "Logits call population differs",
        )
        for check in record["logits_checks"]:
            require(
                check["status"] == "passed"
                and check["finite"] is True
                and check["nonfinite_count"] == 0
                and check["elements"] == check["finite_count"] == VOCAB[key]
                and check["dtype"] == "float32"
                and check["shape"] in ([1, VOCAB[key]], [1, 1, VOCAB[key]])
                and check["phase"]
                == ("before_sample" if count else "explicit_readout"),
                "Full-vocabulary guard differs",
            )
        checks += n
        if count:
            sample_counts.append(count)
    return dict(generators=len(records), checks=checks, sample_counts=sample_counts)


def idle(value):
    require(
        value["policy"]
        == dict(
            maximum_cpu_percent=30,
            minimum_free_gib=32,
            build_process_count=0,
            consecutive_samples_required=2,
        ),
        "Idle policy differs",
    )
    samples = value["samples"]
    require(len(samples) >= 2, "Insufficient idle samples")
    consecutive = 0
    for sample in samples:
        native = sample["native_cpu"]
        require(
            native["api"] == "GetSystemTimes"
            and 1 <= native["logical_processors"] <= 64,
            "CPU sampler differs",
        )
        number(native["seconds"], positive=True)
        delta = {
            name: native["end"][name] - native["begin"][name]
            for name in ("idle", "kernel", "user")
        }
        require(
            all(type(x) is int and x >= 0 for x in delta.values())
            and native["delta"] == delta,
            "CPU counter delta differs",
        )
        total = delta["kernel"] + delta["user"]
        require(total > 0 and delta["idle"] <= total, "Invalid CPU counter interval")
        close(sample["cpu_percent"], 100 * (total - delta["idle"]) / total)
        okay = (
            sample["cpu_percent"] <= 30
            and sample["free_gib"] >= 32
            and sample["build_process_count"] == 0
        )
        consecutive = consecutive + 1 if okay else 0
        require(sample["consecutive_idle"] == consecutive, "Idle sequence differs")
    require(consecutive == 2, "Launch did not follow two idle intervals")
    return len(samples)


def identity_links(execution, compared):
    for key, label in (("qwen3", "qwen3-speed"), ("qwen35", "qwen35-all")):
        proof = compared[key]["guard"]
        require(
            proof["code_sha256"] and proof["runtime_binary_sha256"],
            "Missing code/runtime identity",
        )
        for name, expected in proof["code_sha256"].items():
            sha_value(expected)
            require(
                execution["sources"][name]["sha256"] == expected,
                "Queue/observer source hash differs",
            )
        for name, expected in proof["runtime_binary_sha256"].items():
            sha_value(expected)
            require(
                execution["runtime"]["Lib/site-packages/onnxruntime_genai/" + name][
                    "sha256"
                ]
                == expected,
                "Queue/observer runtime hash differs",
            )
        model = compared[key]["identity"]["source_artifact_sha256"]
        pinned = execution["model_hashes"][label]
        require(
            pinned["model.onnx"]["sha256"] == model["model.onnx"],
            "Observed model header differs",
        )
        require(
            all(
                row["sha256"] == model[name]
                for name, row in pinned.items()
                if name in model
            ),
            "Queue/observer artifact hash differs",
        )
    for key in ("contract_sha256", "comparator_sha256"):
        expected_key = "comparison_helper_sha256" if key == "comparator_sha256" else key
        require(
            execution["execution_policy"][key] == compared[expected_key],
            "Verifier code association differs",
        )


def model_scope_links(execution, compared, manifests):
    require(
        len(manifests) == 3 and len(execution["jobs"]) == 2,
        "Model scope population differs",
    )
    for index, (key, suite) in enumerate((("qwen3", "speed"), ("qwen35", "all"))):
        deployed, identity, job = (
            manifests[index],
            compared[key]["identity"],
            execution["jobs"][index]["job"],
        )
        require(
            deployed["model"]["revision"]
            == identity["revision"]
            == job["revision"]
            == REVISIONS[key],
            "Deployed/comparison/job model revision differs",
        )
        require(
            deployed["suite"] == identity["suite"] == job["suite"] == suite,
            "Deployed/comparison/job suite differs",
        )
    require(
        execution["selection"]["revision"] == REVISIONS["qwen35"],
        "Selected source revision differs",
    )
    historical, identity = manifests[2], compared["qwen3"]["quality_identity"]
    require(
        historical["model"]["revision"] == identity["revision"] == REVISIONS["qwen3"]
        and historical["suite"] == identity["suite"] == "quality",
        "Historical revision/suite differs",
    )


def input_population_links(bundle, manifests):
    """Cross-run evidence check needs no external record bodies or tokenizer."""
    populations = dict(COUNTS, shape777=777, every_gold154=154, firewall_actions=10)
    for manifest in manifests:
        require(
            set(manifest["inputs"]) == set(populations),
            "Frozen input population differs",
        )
        for name, count in populations.items():
            item = manifest["inputs"][name]
            sha_value(item["sha256"])
            field = "actions" if name == "firewall_actions" else "rows"
            require(
                type(item[field]) is int and item[field] == count,
                "Frozen input count differs",
            )
        close(
            manifest["inputs"],
            manifests[0]["inputs"],
            "cross-run frozen input identities",
        )
    for name, count in COUNTS.items():
        new = rows(bundle / LABELS[1] / (name + ".predictions.jsonl"))
        historical = rows(bundle / LABELS[2] / (name + ".predictions.jsonl"))
        require(
            len(new) == len(historical) == count,
            "Cross-run Quality row population differs",
        )
        require(
            [(row["id"], row["option_ids"]) for row in new]
            == [(row["id"], row["option_ids"]) for row in historical],
            "Cross-run Quality IDs/options/order differ",
        )


def verify(bundle, data_dir=None):
    bundle = Path(bundle).resolve()
    manifest = check_manifest(bundle)
    compared, execution = (
        read(bundle / "comparison.json"),
        read(bundle / "execution.json"),
    )
    require(
        compared["status"] == "completed_comparison_verified"
        and execution["status"] == "complete"
        and execution["exit_code"] == 0
        and execution["retries"] == 0
        and execution["heartbeat_errors"] == []
        and execution["not_started_jobs"] == []
        and len(execution["jobs"]) == 2,
        "Execution incomplete",
    )
    original = manifest["original_control_evidence"]
    require(
        compared["selection_sha256"]
        == execution["selection_sha256"]
        == original["selection_sha256"]
        and execution["pins_sha256"] == original["pins_sha256"]
        and compared["queue_report_sha256"] == original["queue_report_sha256"],
        "Control evidence association differs",
    )
    require(
        execution["child_memory_bytes"]
        == execution["execution_policy"]["memory_bytes"]
        == 20 * 1024**3,
        "Child memory cap differs",
    )
    identity_links(execution, compared)
    manifests = [read(bundle / label / "manifest.json") for label in LABELS]
    model_scope_links(execution, compared, manifests)
    input_population_links(bundle, manifests)
    for index, key in enumerate(("qwen3", "qwen35")):
        run, deployed = bundle / LABELS[index], manifests[index]
        require(
            deployed["historical_measurement"] is False
            and deployed["dll_identity_evidence"] is True
            and deployed["full_vocabulary_guard_evidence"] is True
            and deployed["model"]["context_ceiling"] == 16384,
            "New model scope differs",
        )
        v.verify_manifest(deployed, None)
        close(compared[key]["speed"], speed(run))
        close(compared[key]["npu_counter_delta"], counters(read(run / "hardware.json")))
        observed = guard(run, deployed, key)
        reference = compared[key]["guard"]
        require(
            observed["generators"] == reference["generators"]
            and observed["checks"] == reference["full_vocabulary_checks"]
            and observed["sample_counts"]
            == reference["compact_samples_including_warmup"],
            "Comparison guard summary differs",
        )
        for name in (
            "code_sha256",
            "runtime_binary_sha256",
            "package_manifest_sha256",
            "thread_environment",
        ):
            close(deployed["observation"][name], reference[name])
        close(
            deployed["model"]["source_artifact_sha256"],
            compared[key]["identity"]["source_artifact_sha256"],
        )
        job = execution["jobs"][index]
        load_samples = idle(read(run / "load-gate.json"))
        require(
            job["load_gate"]["status"] == "idle_gate_passed"
            and job["load_gate"]["samples"] == load_samples,
            "Queue idle evidence association differs",
        )
        require(
            job["job"]["name"] == LABELS[index]
            and job["status"] == "complete"
            and job["child_started"] is True
            and job["child"]["status"] == "returned"
            and job["child"]["exit_code"] == 0
            and job["retries"] == 0
            and job["job"]["timeout_seconds"]
            == execution["selection"][key + "_timeout_seconds"],
            "Child/deadline evidence differs",
        )
        close(job["completion"]["npu_delta"], compared[key]["npu_counter_delta"])
        require(
            job["child"]["memory_limit_bytes"] == 20 * 1024**3
            and job["child"]["kill_on_job_close"] is True
            and job["child"]["process_and_job_commit_limited"] is True
            and job["preflight"]["model"]["files"]
            == job["postflight"]["model"]["files"]
            == len(execution["model_hashes"][LABELS[index]]),
            "Model post-hash/resource record differs",
        )
        require(
            job["completion"]["observation_sha256"]
            == manifest["original_evidence"][LABELS[index] + "/observations.json"][
                "sha256"
            ],
            "Child observation identity differs",
        )
    for name in ("code_sha256", "runtime_binary_sha256", "thread_environment"):
        close(compared["qwen3"]["guard"][name], compared["qwen35"]["guard"][name])
    require(
        execution["selection"]["package_manifest_sha256"]
        == compared["qwen35"]["guard"]["package_manifest_sha256"],
        "Selected model manifest differs",
    )
    old = manifests[2]
    require(
        old["historical_measurement"] is True
        and old["created_utc"].startswith("2026-09-19")
        and old["dll_identity_evidence"] is False
        and old["full_vocabulary_guard_evidence"] is False,
        "Historical guard/runtime scope misrepresented",
    )
    close(
        old["model"]["source_artifact_sha256"],
        manifests[0]["model"]["source_artifact_sha256"],
    )
    v.verify_manifest(old, None)
    for key, index in (("qwen3", 2), ("qwen35", 1)):
        close(
            compared[key]["quality"],
            quality(bundle / LABELS[index], manifests[index], data_dir),
        )
    return dict(
        status="public_comparison_checks_passed",
        prediction_rows=3308,
        inference_calls=0,
        speed="Recomputed from all saved group/repeat/sample timings and 1680 direct prediction rows.",
        quality="Full Quality recomputed with matching external inputs."
        if data_dir is not None
        else "Owned144/Perturbations108 and stability recomputed; external metrics retained and associated but not recomputed without their inputs.",
        strict_json="Not reparsed: public bundle omits generated text/token IDs. Recorded result, choices, EOS, truncation and timing consistency checked.",
        guard="Checks saved instrumentation metadata; full logits arrays and warmup token-ID fingerprints were not retained.",
        provenance="Checks saved hashes/associations; it does not load or rehash model/runtime payloads or reproduce the hardware measurement.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle, args.data_dir), ensure_ascii=False, indent=2))
