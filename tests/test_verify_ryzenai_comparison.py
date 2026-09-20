"""CPU verification tests using owned fixtures and committed comparison evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import verify_ryzenai_comparison as check

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/raw/ryzenai-16k-20260919"
SPEED = ROOT / "results/raw/ryzenai-4k-20260919"



def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def speed_report(shape, compact):
    """Convert owned raw speed reports to the public verifier's schema locally."""
    result = {
        "shape777": {
            key: copy.deepcopy(shape[key])
            for key in (
                "version",
                "wall_seconds",
                "decisions_per_second",
                "state_p50_seconds",
                "groups",
                "scope",
                "unsupported_modes",
            )
        },
        "compact": {
            "version": compact["version"],
            "scope": compact["scope"],
            "unsupported_modes": compact["unsupported_modes"],
            "median_wall_ratio_generation_over_direct": compact[
                "median_wall_ratio_generation_over_direct"
            ],
            "direct_fresh": {
                key: copy.deepcopy(value)
                for key, value in compact["direct_fresh"].items()
                if key != "runs"
            },
            "compact_generation": {
                key: copy.deepcopy(value)
                for key, value in compact["compact_generation"].items()
                if key != "runs"
            },
        },
    }
    result["compact"]["direct_fresh"]["runs"] = [
        {"repeat": row["repeat"], "total_seconds": row["total_seconds"]}
        for row in compact["direct_fresh"]["runs"]
    ]
    generated = []
    for row in compact["compact_generation"]["runs"]:
        item = {
            key: copy.deepcopy(row[key])
            for key in (
                "repeat",
                "prompt_sha256",
                "input_tokens",
                "output_tokens",
                "time_to_first_token_seconds",
                "total_seconds",
                "valid_complete_array",
                "choices",
                "ended_by_eos",
                "truncated",
                "agreement_with_direct_argmax",
            )
        }
        item["output_text_sha256"] = hashlib.sha256(
            row["output_text"].encode()
        ).hexdigest()
        item["timeline"] = [
            {
                "sample": index,
                "seconds": step["seconds"],
                "is_configured_eos": step["token_id"] in row["eos_token_ids"],
            }
            for index, step in enumerate(row["timeline"], 1)
        ]
        generated.append(item)
    result["compact"]["compact_generation"]["runs"] = generated
    return result


@pytest.fixture
def speed_fixture(tmp_path):
    save(
        tmp_path / "speed.json",
        speed_report(
            check.read(SPEED / "shape777.json"), check.read(SPEED / "compact.json")
        ),
    )
    for name in ("shape777", "compact"):
        (tmp_path / (name + ".predictions.jsonl")).write_bytes(
            (SPEED / (name + ".predictions.jsonl")).read_bytes()
        )
    return tmp_path


def test_speed_recomputed_from_owned_rows(speed_fixture):
    result = check.speed(speed_fixture)
    raw = check.read(SPEED / "compact.json")
    assert (
        result["direct21_median_seconds"] == raw["direct_fresh"]["median_total_seconds"]
    )
    assert (
        result["shape777_decisions_per_second"]
        == check.read(SPEED / "shape777.json")["decisions_per_second"]
    )


@pytest.mark.parametrize(
    "mutation", ["timing", "tokens", "eos", "group", "median", "ratio", "choices"]
)
def test_speed_mutations_rejected(speed_fixture, mutation):
    value = check.read(speed_fixture / "speed.json")
    generation = value["compact"]["compact_generation"]["runs"][0]
    if mutation == "timing":
        generation["timeline"][0]["seconds"] = -1
    elif mutation == "tokens":
        generation["output_tokens"] += 1
    elif mutation == "eos":
        generation["timeline"][0]["is_configured_eos"] = True
    elif mutation == "group":
        value["shape777"]["groups"][0]["decisions"] = 20
    elif mutation == "median":
        value["compact"]["direct_fresh"]["median_total_seconds"] += 1
    elif mutation == "ratio":
        value["compact"]["median_wall_ratio_generation_over_direct"] += 1
    else:
        generation["choices"][0] = "invalid_label"
    save(speed_fixture / "speed.json", value)
    with pytest.raises(AssertionError):
        check.speed(speed_fixture)


def test_recorded_truncation_is_an_outcome(speed_fixture):
    value = check.read(speed_fixture / "speed.json")
    compact = value["compact"]["compact_generation"]
    for row in compact["runs"]:
        row["timeline"] = [
            {
                "sample": i + 1,
                "seconds": row["total_seconds"] * (i + 1) / 128,
                "is_configured_eos": False,
            }
            for i in range(128)
        ]
        row.update(
            output_tokens=128,
            choices=None,
            valid_complete_array=False,
            ended_by_eos=False,
            truncated=True,
            agreement_with_direct_argmax=None,
            time_to_first_token_seconds=row["timeline"][0]["seconds"],
        )
    compact.update(
        all_runs_valid_complete_arrays=False,
        all_runs_ended_by_eos=False,
        truncated_repeats=[0, 1, 2],
        median_output_tokens=128,
    )
    save(speed_fixture / "speed.json", value)
    result = check.speed(speed_fixture)
    assert result["valid_compact_repeats"] == 0 and result["truncated_repeats"] == [
        0,
        1,
        2,
    ]


def test_guard_wrong_vocabulary_and_schedule_rejected(tmp_path, speed_fixture):
    manifest = {"observation": {"fixture": True}}
    shape = check.rows(speed_fixture / "shape777.predictions.jsonl")
    generations = check.read(speed_fixture / "speed.json")["compact"][
        "compact_generation"
    ]["runs"]
    schedule = [(row["input_tokens"], 0) for row in shape[:21]] + [(1866, 16)]
    for row in generations:
        schedule += [(item["input_tokens"], 0) for item in shape[:21]] + [
            (row["input_tokens"], row["output_tokens"])
        ]
    schedule += [(shape[0]["input_tokens"], 0)] + [
        (row["input_tokens"], 0) for row in shape
    ]
    observed = dict(
        status="complete",
        code_unchanged=True,
        runtime_unchanged=True,
        package_manifest_unchanged=True,
        model_artifacts_unchanged=True,
        provenance={"fixture": True},
        generator_observations=[],
    )
    for index, (length, samples) in enumerate(schedule, 1):
        calls = samples or 1
        observed["generator_observations"].append(
            dict(
                generator_index=index,
                constructed=True,
                errors=[],
                append_calls_started=1,
                append_calls_completed=1,
                appended_token_counts=[length],
                sampling_calls_started=samples,
                sampling_calls_completed=samples,
                native_logits_calls_started=calls,
                native_logits_calls_completed=calls,
                explicit_logits_calls=0 if samples else 1,
                logits_checks=[
                    dict(
                        status="passed",
                        finite=True,
                        nonfinite_count=0,
                        elements=151936,
                        finite_count=151936,
                        dtype="float32",
                        shape=[1, 151936],
                        phase="before_sample" if samples else "explicit_readout",
                    )
                    for _ in range(calls)
                ],
            )
        )
    save(speed_fixture / "observations.json", observed)
    assert check.guard(speed_fixture, manifest, "qwen3")["generators"] == 866
    altered = copy.deepcopy(observed)
    altered["generator_observations"][0]["logits_checks"][0]["nonfinite_count"] = 1
    save(speed_fixture / "observations.json", altered)
    with pytest.raises(AssertionError, match="vocabulary"):
        check.guard(speed_fixture, manifest, "qwen3")
    observed["generator_observations"][0]["appended_token_counts"][0] += 1
    save(speed_fixture / "observations.json", observed)
    with pytest.raises(AssertionError, match="input length"):
        check.guard(speed_fixture, manifest, "qwen3")


def test_owned_quality_recompute_needs_no_external_cache():
    manifest = check.read(OLD / "manifest.json")
    result = check.quality(OLD, manifest)
    assert result["typesafe102"]["rows"] == 102
    assert result["every_company_knowledge"]["rows"] == 70


def test_missing_external_data_is_rejected_explicitly(tmp_path):
    with pytest.raises(AssertionError, match="external input"):
        check.quality(OLD, check.read(OLD / "manifest.json"), tmp_path)


def test_counter_labels_and_idle_formula():
    counters = dict(
        before=dict(available=True, contexts=[]),
        after=dict(
            available=True,
            contexts=[
                dict(
                    label="context-1",
                    command_submissions=12,
                    command_completions=12,
                    errors=0,
                )
            ],
        ),
        total_delta=dict(command_submissions=12, command_completions=12, errors=0),
    )
    assert check.counters(counters)["command_submissions"] == 12
    counters["after"]["contexts"][0]["errors"] = 1
    with pytest.raises(AssertionError):
        check.counters(counters)
    native = dict(
        api="GetSystemTimes",
        logical_processors=32,
        seconds=3.0,
        begin=dict(idle=100, kernel=150, user=10),
        end=dict(idle=190, kernel=240, user=20),
        delta=dict(idle=90, kernel=90, user=10),
    )
    idle = dict(
        policy=dict(
            maximum_cpu_percent=30,
            minimum_free_gib=32,
            build_process_count=0,
            consecutive_samples_required=2,
        ),
        samples=[
            dict(
                native_cpu=copy.deepcopy(native),
                cpu_percent=10.0,
                free_gib=100,
                build_process_count=0,
                consecutive_idle=i,
            )
            for i in (1, 2)
        ],
    )
    assert check.idle(idle) == 2
    idle["samples"][1]["cpu_percent"] = 9.0
    with pytest.raises(AssertionError):
        check.idle(idle)


def test_duplicate_prediction_id_rejected():
    value = check.rows(OLD / "every204.predictions.jsonl")
    value[1]["id"] = value[0]["id"]
    with pytest.raises(AssertionError, match="Duplicate"):
        check.check_predictions(value)


def test_artifact_checksum_and_population_rejected(tmp_path):
    names = check.expected_files()
    for name in names:
        path = tmp_path / name
        if name.endswith(".predictions.jsonl"):
            count = (
                777
                if "/shape777." in name
                else 63
                if "/compact." in name
                else check.COUNTS[path.name.removesuffix(".predictions.jsonl")]
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" * count, encoding="utf-8")
        else:
            save(path, {})
    counts = {
        name: len(check.rows(tmp_path / name))
        for name in names
        if name.endswith(".jsonl")
    }
    manifest = dict(
        schema_version=1,
        status="completed_comparison_published",
        artifacts={
            name: dict(
                bytes=(tmp_path / name).stat().st_size,
                sha256=check.digest(tmp_path / name),
            )
            for name in names
        },
        prediction_rows=counts,
        total_prediction_rows=3308,
    )
    save(tmp_path / "manifest.json", manifest)
    assert check.check_manifest(tmp_path)["total_prediction_rows"] == 3308
    (tmp_path / "comparison.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(AssertionError, match="checksum"):
        check.check_manifest(tmp_path)


@pytest.mark.parametrize("mutation", [None, "source", "runtime", "model", "helper"])
def test_identity_hash_associations(mutation):
    execution = dict(
        sources={"benchmarks/test.py": {"sha256": "a" * 64}},
        runtime={"Lib/site-packages/onnxruntime_genai/test.dll": {"sha256": "b" * 64}},
        model_hashes={
            name: {"model.onnx": {"sha256": "c" * 64}}
            for name in ("qwen3-speed", "qwen35-all")
        },
        execution_policy=dict(contract_sha256="d" * 64, comparator_sha256="e" * 64),
    )
    compared = dict(contract_sha256="d" * 64, comparison_helper_sha256="e" * 64)
    for model in ("qwen3", "qwen35"):
        compared[model] = dict(
            guard=dict(
                code_sha256={"benchmarks/test.py": "a" * 64},
                runtime_binary_sha256={"test.dll": "b" * 64},
            ),
            identity=dict(source_artifact_sha256={"model.onnx": "c" * 64}),
        )
    if mutation == "source":
        execution["sources"]["benchmarks/test.py"]["sha256"] = "0" * 64
    elif mutation == "runtime":
        execution["runtime"]["Lib/site-packages/onnxruntime_genai/test.dll"][
            "sha256"
        ] = "0" * 64
    elif mutation == "model":
        execution["model_hashes"]["qwen35-all"]["model.onnx"]["sha256"] = "0" * 64
    elif mutation == "helper":
        execution["execution_policy"]["contract_sha256"] = "0" * 64
    if mutation is None:
        check.identity_links(execution, compared)
    else:
        with pytest.raises(AssertionError):
            check.identity_links(execution, compared)


@pytest.mark.parametrize(
    "target",
    [
        ("manifests", 0, "model", "revision"),
        ("manifests", 1, "model", "revision"),
        ("compared", "qwen3", "identity", "revision"),
        ("compared", "qwen35", "identity", "revision"),
        ("execution", "jobs", 0, "job", "revision"),
        ("execution", "jobs", 1, "job", "revision"),
        ("execution", "selection", "revision"),
        ("manifests", 2, "model", "revision"),
        ("compared", "qwen3", "quality_identity", "revision"),
        ("manifests", 0, "suite"),
        ("manifests", 1, "suite"),
        ("manifests", 2, "suite"),
        ("compared", "qwen3", "identity", "suite"),
        ("compared", "qwen35", "identity", "suite"),
        ("compared", "qwen3", "quality_identity", "suite"),
        ("execution", "jobs", 0, "job", "suite"),
        ("execution", "jobs", 1, "job", "suite"),
    ],
)
def test_model_scope_association_rejected(target):
    manifests = [
        dict(model=dict(revision=check.REVISIONS[key]), suite=suite)
        for key, suite in (("qwen3", "speed"), ("qwen35", "all"), ("qwen3", "quality"))
    ]
    compared = {
        key: dict(identity=dict(revision=check.REVISIONS[key], suite=suite))
        for key, suite in (("qwen3", "speed"), ("qwen35", "all"))
    }
    compared["qwen3"]["quality_identity"] = dict(
        revision=check.REVISIONS["qwen3"], suite="quality"
    )
    execution = dict(
        selection=dict(revision=check.REVISIONS["qwen35"]),
        jobs=[
            dict(job=dict(revision=check.REVISIONS[key], suite=suite))
            for key, suite in (("qwen3", "speed"), ("qwen35", "all"))
        ],
    )
    fixture = dict(manifests=manifests, compared=compared, execution=execution)
    entry = fixture
    for key in target[:-1]:
        entry = entry[key]
    entry[target[-1]] = "0" * 40 if target[-1] == "revision" else "wrong-suite"
    with pytest.raises(AssertionError, match="revision|suite"):
        check.model_scope_links(**fixture)


@pytest.mark.parametrize(
    "mutation",
    [None, "id", "option_ids", "order", "input_sha", "input_rows", "missing_input"],
)
def test_cross_run_quality_population_without_external_cache(tmp_path, mutation):
    manifests = [copy.deepcopy(check.read(OLD / "manifest.json")) for _ in range(3)]
    for label in (check.LABELS[1], check.LABELS[2]):
        folder = tmp_path / label
        folder.mkdir()
        for name in check.QUALITY:
            (folder / (name + ".predictions.jsonl")).write_bytes(
                (OLD / (name + ".predictions.jsonl")).read_bytes()
            )
    bundle = tmp_path
    if mutation in ("id", "option_ids", "order"):
        path = bundle / check.LABELS[1] / "wanli256.predictions.jsonl"
        values = check.rows(path)
        if mutation == "id":
            values[0]["id"] = "different-case"
        elif mutation == "option_ids":
            values[0]["option_ids"].reverse()
        else:
            values[0], values[1] = values[1], values[0]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in values), encoding="utf-8"
        )
    elif mutation == "input_sha":
        manifests[1]["inputs"]["wanli256"]["sha256"] = "0" * 64
    elif mutation == "input_rows":
        manifests[2]["inputs"]["wanli256"]["rows"] = 255
    elif mutation == "missing_input":
        del manifests[0]["inputs"]["every_gold154"]
    if mutation is None:
        check.input_population_links(bundle, manifests)
    else:
        with pytest.raises(AssertionError):
            check.input_population_links(bundle, manifests)


def test_committed_comparison_recomputes_without_external_sources():
    result = check.verify(ROOT / "results/raw/qwen35-vs-qwen3-20260920")
    assert result["status"] == "public_comparison_checks_passed"
    assert result["prediction_rows"] == 3308
    assert result["inference_calls"] == 0
