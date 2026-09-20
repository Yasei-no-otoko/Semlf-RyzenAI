import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import ryzenai_speed as speed
from semif_phase1.ryzenai_backend import RyzenAiTokenizer


def _rows(count, *, groups=False):
    return [
        {
            "id": f"row-{index}", "group_id": f"group-{index // 21}" if groups else "group-0",
            "state": "shared evidence", "question": f"criterion {index}",
            "options": [{"id": "yes", "description": "yes"}, {"id": "no", "description": "no"}],
        }
        for index in range(count)
    ]


class ReferenceTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "compact prompt"

    def encode(self, prompt, add_special_tokens=False):
        assert not add_special_tokens
        return [10, 11]


class OgaTokenizer:
    def encode(self, prompt):
        return np.asarray([10, 11], dtype=np.int32)

    def create_stream(self):
        return SimpleNamespace(decode=lambda token: "" if token == 99 else self.text)


def _install_oga(monkeypatch, text, generated_tokens=(1, 99)):
    events = []

    class Params:
        def __init__(self, model):
            events.append("params")

        def set_search_options(self, **kwargs):
            events.append(("search", kwargs))

    class Generator:
        def __init__(self, model, params):
            self.index = 0
            events.append("generator")

        def append_tokens(self, tokens):
            assert tokens.dtype == np.int32 and tokens.tolist() == [10, 11]
            events.append("append")

        def is_done(self):
            return False

        def generate_next_token(self):
            self.index += 1

        def get_next_tokens(self):
            return np.asarray([generated_tokens[min(self.index - 1, len(generated_tokens) - 1)]], dtype=np.int32)

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(GeneratorParams=Params, Generator=Generator))
    oga = OgaTokenizer()
    oga.text = text
    return events, RyzenAiTokenizer(ReferenceTokenizer(), oga)


def test_compact_generation_requires_parity_records_timeline_and_eos(monkeypatch):
    rows = _rows(21)
    events, tokenizer = _install_oga(monkeypatch, json.dumps(["yes"] * 21))
    result = speed.run_compact_generation(object(), tokenizer, {"context_ceiling": 64}, "shared evidence", rows,
                                          max_tokens=64, max_new_tokens=8)
    assert events[:3] == ["params", ("search", {"max_length": 10, "batch_size": 1, "do_sample": False}), "generator"]
    assert result["choices"] == ["yes"] * 21
    assert result["valid_complete_array"] and result["ended_by_eos"] and not result["truncated"]
    assert result["output_tokens"] == 2 and result["timeline"][-1]["token_id"] == 99


def test_decision21_runs_fresh_direct_rows_and_separates_model_metadata(monkeypatch):
    rows = _rows(21)
    _, tokenizer = _install_oga(monkeypatch, json.dumps(["no"] * 21))
    calls, emitted = [], []

    def fake_score(model, tokenizer, row, metadata, max_tokens):
        calls.append(row["id"])
        return {"id": row["id"], "option_ids": ["yes", "no"], "probabilities": [0.8, 0.2],
                "model": metadata, "total_seconds": 0.01}

    monkeypatch.setattr(speed, "score", fake_score)
    report = speed.benchmark_decision21(object(), tokenizer, {"context_ceiling": 64, "source": "local"}, rows,
                                        repeats=2, max_tokens=64, max_new_tokens=8,
                                        on_row=lambda section, repeat, row: emitted.append((section, repeat, row)))
    assert len(calls) == 21 * 3  # complete direct warmup plus two fresh 21-row runs
    assert len(emitted) == 42
    assert report["direct_fresh"]["runs"][0]["outputs"][0].get("model") is None
    assert report["compact_generation"]["all_runs_valid_complete_arrays"]
    assert report["compact_prompt_messages"][0]["role"] == "system"
    assert report["unsupported_modes"] == speed.UNSUPPORTED_MODES


def test_shape777_groups_all_rows_and_callback_runs_after_each_group(monkeypatch):
    rows = _rows(777, groups=True)
    callbacks, progress = [], []

    def fake_score(model, tokenizer, row, metadata, max_tokens):
        return {"id": row["id"], "option_ids": ["yes", "no"], "probabilities": [0.5, 0.5],
                "model": metadata}

    monkeypatch.setattr(speed, "score", fake_score)
    report = speed.benchmark_shape777_fresh(object(), object(), {"context_ceiling": 4096}, rows,
                                             on_row=lambda section, group, row: callbacks.append((section, group, row["id"])),
                                             on_progress=lambda done, total: progress.append((done, total)))
    assert len(report["groups"]) == len(set(row["group_id"] for row in rows)) == 37
    assert len(report["outputs"]) == len(callbacks) == 777
    assert report["groups"][0]["group_id"] == "group-0"
    assert progress == [(index, 37) for index in range(1, 38)]


def test_public_input_exposes_only_repository_relative_path(tmp_path):
    root, fixture = tmp_path / "repo", tmp_path / "repo" / "benchmarks" / "data.jsonl"
    fixture.parent.mkdir(parents=True)
    fixture.write_text("{}\n", encoding="utf-8")
    identity = speed.public_input(fixture, root)
    assert identity["path"] == "benchmarks/data.jsonl"
    assert len(identity["sha256"]) == 64
    with pytest.raises(ValueError, match="inside the repository"):
        speed.public_input(tmp_path / "outside.jsonl", root)


def test_generation_uses_loaded_configured_eos_authoritatively(tmp_path):
    tokenizer = RyzenAiTokenizer(ReferenceTokenizer(), OgaTokenizer())
    assert speed._eos_ids(tokenizer, {}) == {99}
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"eos_token_id": [100]}}), encoding="utf-8"
    )
    assert speed._eos_ids(tokenizer, {"source": str(tmp_path)}) == {100}

    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"eos_token_id": [99]}}), encoding="utf-8"
    )
    assert speed._eos_ids(tokenizer, {"source": str(tmp_path)}) == {99}


def test_generation_continues_past_reference_eos_to_loaded_model_stop(monkeypatch, tmp_path):
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"eos_token_id": 100, "vocab_size": 101}}), encoding="utf-8"
    )
    _, tokenizer = _install_oga(monkeypatch, json.dumps(["yes"] * 21), generated_tokens=(1, 99, 100))
    result = speed.run_compact_generation(object(), tokenizer, {"context_ceiling": 64, "source": str(tmp_path)},
                                         "shared evidence", _rows(21), max_tokens=64, max_new_tokens=8)
    assert result["eos_token_ids"] == [100]
    assert [event["token_id"] for event in result["timeline"]] == [1, 99, 100]
    assert result["choices"] == ["yes"] * 21
    assert result["ended_by_eos"] and not result["truncated"]


@pytest.mark.parametrize("configured", [[-1], [100], [True], ["99"]])
def test_malformed_configured_eos_fails_before_generator_or_append(monkeypatch, tmp_path, configured):
    (tmp_path / "genai_config.json").write_text(
        json.dumps({"model": {"eos_token_id": configured, "vocab_size": 100}}), encoding="utf-8"
    )
    events, tokenizer = _install_oga(monkeypatch, json.dumps(["yes"] * 21))
    with pytest.raises(ValueError):
        speed.run_compact_generation(object(), tokenizer, {"context_ceiling": 64, "source": str(tmp_path)},
                                     "shared evidence", _rows(21), max_tokens=64, max_new_tokens=8)
    assert events == []
