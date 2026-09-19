"""Backward-compatible reusable evaluation entry points."""

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evaluators():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    try:
        yield importlib.import_module("evaluate_external"), importlib.import_module("evaluate_perturbations")
    finally:
        sys.path.remove(str(ROOT / "benchmarks"))


def _prediction(identifier, probabilities):
    return {"id": identifier, "option_ids": ["yes", "no"], "probabilities": probabilities}


def _assert_json_close(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_json_close(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_json_close(left, right)
    elif isinstance(actual, float):
        assert actual == pytest.approx(expected, abs=5e-10)
    else:
        assert actual == expected


def test_type_safe_systems_preserves_two_system_wrapper(evaluators):
    external, _ = evaluators
    gold = [
        {"id": "a", "group_id": "case-a", "label": 0, "options": [{"id": "yes"}, {"id": "no"}],
         "target_distribution": [0.75, 0.25], "published_models": {"typesafe": {"distribution": [0.8, 0.2]}}},
        {"id": "b", "group_id": "case-b", "label": 1, "options": [{"id": "yes"}, {"id": "no"}],
         "target_distribution": [0.25, 0.75], "published_models": {"typesafe": {"distribution": [0.2, 0.8]}}},
    ]
    direct, reranker = [_prediction("a", [0.9, 0.1]), _prediction("b", [0.1, 0.9])], [
        _prediction("a", [0.4, 0.6]), _prediction("b", [0.6, 0.4])]

    assert external.type_safe(gold, direct, reranker) == external.type_safe_systems(
        gold, {"direct": direct, "reranker": reranker})
    assert set(external.type_safe_systems(gold, {"ryzenai_npu": direct})) == {"ryzenai_npu", "published_jev"}


def test_every_systems_preserves_two_system_wrapper(evaluators):
    external, _ = evaluators
    def row(identifier, experiment, *, source_item="action", question_id="signal", label=0):
        return {"id": identifier, "label": label, "options": [{"id": "yes"}, {"id": "no"}],
                "provenance": {"experiment": experiment, "source_item": source_item, "question_id": question_id}}

    gold = [row("judge", "judge-grid"), row("code", "code-rag"), row("company", "company-brain")]
    inference = gold + [
        row("every/action-firewall/action/destructive", "action-firewall", question_id="destructive"),
        row("every/action-firewall/action/reversible", "action-firewall", question_id="reversible"),
        row("every/action-firewall/action/scope", "action-firewall", question_id="exceeds_scope"),
        row("every/action-firewall/action/sensitive", "action-firewall", question_id="shares_sensitive"),
        row("every/action-firewall/action/confirm", "action-firewall", question_id="needs_confirmation"),
    ]
    predictions = [_prediction(item["id"], [0.9, 0.1]) for item in inference]
    actions = {"expected_actions": {"action": "block"}}

    assert external.every(gold, inference, predictions, predictions, actions) == external.every_systems(
        gold, inference, {"direct": predictions, "reranker": predictions}, actions)
    assert set(external.every_systems(gold, inference, {"ryzenai_npu": predictions}, actions)) == {"ryzenai_npu"}


def test_perturbation_reuse_matches_committed_per_system_reports(evaluators):
    _, perturb = evaluators
    gold = perturb.read(ROOT / "benchmarks/data/authored144.jsonl")
    variants = perturb.read(ROOT / "benchmarks/data/perturbations108.jsonl")
    published = json.loads((ROOT / "results/raw/perturbation-comparison.json").read_text())
    for name in ("direct_logits", "reranker"):
        actual = perturb.evaluate_system(
            gold, variants,
            perturb.read(ROOT / f"results/raw/predictions/{'direct' if name == 'direct_logits' else name}-authored144.jsonl"),
            perturb.read(ROOT / f"results/raw/predictions/{'direct' if name == 'direct_logits' else name}-perturbations108.jsonl"),
        )
        _assert_json_close(actual, published["systems"][name])


def test_published_perturbation_report_and_summary_verifier_remain_unchanged(tmp_path):
    output = tmp_path / "perturbation-comparison.json"
    command = [
        sys.executable, "benchmarks/evaluate_perturbations.py",
        "--gold", "benchmarks/data/authored144.jsonl",
        "--perturbations", "benchmarks/data/perturbations108.jsonl",
        "--direct-base", "results/raw/predictions/direct-authored144.jsonl",
        "--direct-perturbations", "results/raw/predictions/direct-perturbations108.jsonl",
        "--reranker-base", "results/raw/predictions/reranker-authored144.jsonl",
        "--reranker-perturbations", "results/raw/predictions/reranker-perturbations108.jsonl",
        "--output", str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    # The historical artifact uses LF and was emitted on an older Python;
    # evaluator float accumulation can differ in the final binary digit.
    _assert_json_close(json.loads(output.read_text()), json.loads(
        (ROOT / "results/raw/perturbation-comparison.json").read_text()))

    verified = subprocess.run([sys.executable, "benchmarks/verify_published.py"], cwd=ROOT,
                              check=True, capture_output=True, text=True)
    assert json.loads(verified.stdout)["verified_summary_claims"] == 69
