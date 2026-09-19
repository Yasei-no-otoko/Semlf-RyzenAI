import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from semif_phase1 import ryzenai_generation as generation
from semif_phase1.ryzenai_backend import NPU_LABELS, RyzenAiTokenizer


ROW = {
    "id": "decision-1", "state": "The deployment passed.", "question": "Did it pass?",
    "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
}


class ReferenceTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        assert messages[1]["content"].startswith("State:\nThe deployment passed.")
        return "generation prompt"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [10, 11]


class OgaTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def encode(self, text):
        return np.asarray([10, 11], dtype=np.int32)

    def create_stream(self):
        return SimpleNamespace(decode=lambda token: self.pieces[token])


def test_legacy_generation_prompt_is_unchanged_through_16_options():
    user = generation._messages(ROW)[1]["content"]
    assert user == (
        "State:\nThe deployment passed.\n\nQuestion:\nDid it pass?\n\nAllowed options:\nA. Yes\nB. No\n\n"
        "Estimate the probability that each allowed option is the correct decision. "
        "Return only one JSON object mapping each option to its probability. "
        'Form every key as "<label>: <full option text>" using the allowed options above. '
        'For example, if the unrelated options were "A. Route north" and "B. Route south", valid output would be: '
        '{"A: Route north": 0.65, "B: Route south": 0.35}\n'
        "For the actual decision, include every supplied option exactly once and in order. "
        "Each value must be a JSON number from 0 to 1, and the probabilities must sum to 1. "
        "Output JSON only, with no markdown or explanation."
    )


def test_parse_requires_exact_finite_distribution_without_duplicates():
    parsed, error = generation._parse('{"A: Yes": 0.7, "B: No": 0.3}', ROW)
    assert parsed == {"A: Yes": 0.7, "B: No": 0.3}
    assert error is None

    for text in ('{"A: Yes": 1}', '{"A: Yes": 0.5, "B: No": 0.5, "C: Other": 0}',
                 '{"A: Yes": 0.5, "A: Yes": 0.5, "B: No": 0.5}', '{"A: Yes": true, "B: No": 0}'):
        parsed, error = generation._parse(text, ROW)
        assert parsed is None
        assert error


def test_parse_accepts_all_32_npu_option_labels_including_digits():
    row = {
        "id": "decision-32", "state": "evidence", "question": "Choose one.",
        "options": [{"id": str(index), "description": f"Option {index}"} for index in range(32)],
    }
    expected = {f"{label}: Option {index}": 1 / 32 for index, label in enumerate(NPU_LABELS)}
    parsed, error = generation._parse(json.dumps(expected), row)
    assert parsed == expected and error is None
    parsed, error = generation._parse(json.dumps(dict(list(expected.items())[:5])), row)
    assert parsed is None and error == "expected 32 exact option keys; received 5"
    assert generation.GENERATION_PROMPT_VERSION_32 != generation.GENERATION_PROMPT_VERSION
    assert "Z. Option 25" in generation._messages(row)[1]["content"]
    assert "0. Option 26" in generation._messages(row)[1]["content"]


def test_32_option_prompt_has_a_complete_safely_quoted_key_checklist():
    row = {
        "id": "decision-32", "state": "evidence", "question": "Choose one.",
        "options": [{"id": str(index), "description": f'Option "{index}"'} for index in range(32)],
    }
    user = generation._messages(row)[1]["content"]
    assert "exactly 32 members" in user
    assert "Include zero-probability options too; do not use Others or grouping." in user
    assert "For example" not in user
    checklist = user.split("Required JSON keys:\n", 1)[1].splitlines()
    assert checklist == [
        "- " + json.dumps(f'{label}: Option "{index}"', ensure_ascii=False)
        for index, label in enumerate(NPU_LABELS)
    ]


def test_generate_marks_output_cap_truncated_even_when_oga_reports_done(monkeypatch):
    events, received = [], []
    pieces = {1: '{"A: Yes": 0.7, ', 2: '"B: No": 0.3}'}

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
            return self.index == len(pieces)

        def generate_next_token(self):
            self.index += 1
            events.append("generate")

        def get_next_tokens(self):
            return np.asarray([self.index], dtype=np.int32)

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(GeneratorParams=Params, Generator=Generator))
    result = generation.generate(
        object(), RyzenAiTokenizer(ReferenceTokenizer(), OgaTokenizer(pieces)), ROW,
        {"context_ceiling": 8}, max_tokens=8, max_new_tokens=2, on_token=received.append,
    )
    assert events == ["params", ("search", {"max_length": 4, "batch_size": 1, "do_sample": False}),
                      "generator", "append", "generate", "generate"]
    assert received == list(pieces.values())
    assert result["text"] == ''.join(pieces.values())
    assert result["valid_json"] and result["parsed"] == {"A: Yes": 0.7, "B: No": 0.3}
    # OGA's `is_done()` is also true at max_length, even when no EOS arrived.
    assert result["output_tokens"] == 2 and result["truncated"]
    assert result["ttft_seconds"] is not None


def test_generate_reports_truncation_and_rejects_prompt_over_reserved_budget(monkeypatch):
    class Params:
        def __init__(self, model):
            pass

        def set_search_options(self, **kwargs):
            pass

    class Generator:
        def __init__(self, model, params):
            self.index = 0

        def append_tokens(self, tokens):
            pass

        def is_done(self):
            return False

        def generate_next_token(self):
            self.index += 1

        def get_next_tokens(self):
            return np.asarray([1], dtype=np.int32)

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(GeneratorParams=Params, Generator=Generator))
    adapter = RyzenAiTokenizer(ReferenceTokenizer(), OgaTokenizer({1: "{"}))
    result = generation.generate(object(), adapter, ROW, {"context_ceiling": 5}, max_tokens=5, max_new_tokens=3)
    assert result["truncated"] and not result["valid_json"] and "validation_error" in result

    with pytest.raises(ValueError, match="reserved limit"):
        generation.generate(object(), adapter, ROW, {"context_ceiling": 4}, max_tokens=4, max_new_tokens=3)

    with pytest.raises(ValueError, match="context ceiling"):
        generation.generate(object(), adapter, ROW, {"context_ceiling": True})


def test_generate_releases_generator_when_stream_creation_fails(monkeypatch):
    events = []

    class Params:
        def __init__(self, model):
            pass

        def set_search_options(self, **kwargs):
            pass

    class Generator:
        def __init__(self, model, params):
            pass

        def __del__(self):
            events.append("released")

    class BrokenOgaTokenizer(OgaTokenizer):
        def create_stream(self):
            raise RuntimeError("stream unavailable")

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(GeneratorParams=Params, Generator=Generator))
    adapter = RyzenAiTokenizer(ReferenceTokenizer(), BrokenOgaTokenizer({}))
    with pytest.raises(RuntimeError, match="stream unavailable"):
        generation.generate(object(), adapter, ROW, {"context_ceiling": 8}, max_new_tokens=2)
    assert events == ["released"]
