"""Pure regression coverage for the Qwen3.5 NPU validator's proof gate."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import validate_qwen35_npu as validator


def _activity(submissions, completions, errors=0):
    return {
        "available": True,
        "benchmark_contexts": [{
            "command_submissions": submissions,
            "command_completions": completions,
            "errors": errors,
        }],
    }


def test_npu_proof_requires_a_completed_submission_and_no_current_errors():
    assert validator._verify_npu_activity(_activity("1", "1"), _activity("3", "3")) == {
        "command_submissions": 2,
        "command_completions": 2,
        "errors": 0,
    }

    with pytest.raises(validator.ValidationError, match="completions"):
        validator._verify_npu_activity(_activity(1, 1), _activity(2, 1))
    with pytest.raises(validator.ValidationError, match="errors"):
        validator._verify_npu_activity(_activity(1, 1), _activity(2, 2, 1))
    with pytest.raises(validator.ValidationError, match="must be an integer"):
        validator._verify_npu_activity(_activity(1, 1), _activity(True, 2))


def _greedy_fixture(monkeypatch, tokens, *, eos=248044, nonfinite_at=None):
    """Model OGA's lazy GetLogits and IsDone contract without loading a model."""
    monkeypatch.setattr(validator, "GENERATION_TOKENS", 3)
    monkeypatch.setattr(validator.backend, "_encode_prompt", lambda *_: ([1, 2], [], "prompt-hash"))

    class Params:
        def __init__(self, model):
            pass

        def set_search_options(self, **options):
            self.max_length = options["max_length"]

    class Generator:
        def __init__(self, model, params):
            self.params = params
            self.generated = []
            self.logit_positions = []
            self.computed_logits = False
            self.done = False

        def append_tokens(self, ids):
            self.prompt_length = len(ids)
            self.computed_logits = True

        def is_done(self):
            return False if self.computed_logits else self.done

        def get_logits(self):
            # Calling this after EOS/length completion would hide IsDone in OGA.
            assert not self.done, "GetLogits called after generation completed"
            self.logit_positions.append(len(self.generated))
            self.computed_logits = True
            return np.array([np.nan if len(self.generated) == nonfinite_at else 1.0, 0.0])

        def generate_next_token(self):
            assert self.computed_logits
            token = tokens[len(self.generated)]
            self.generated.append(token)
            self.computed_logits = False
            self.done = token == eos or self.prompt_length + len(self.generated) == self.params.max_length

        def get_next_tokens(self):
            return np.array(self.generated[-1:], dtype=np.int32)

    instance = None

    def create_generator(model, params):
        nonlocal instance
        instance = Generator(model, params)
        return instance

    monkeypatch.setitem(sys.modules, "onnxruntime_genai", SimpleNamespace(
        GeneratorParams=Params, Generator=create_generator))
    tokenizer = SimpleNamespace(oga=SimpleNamespace(
        create_stream=lambda: SimpleNamespace(decode=lambda token: str(token))))
    return tokenizer, lambda: instance


@pytest.mark.parametrize("tokens", [[248044], [32, 248044], [32, 198, 32]])
def test_greedy_checks_done_before_requesting_more_logits(monkeypatch, tokens):
    tokenizer, generated = _greedy_fixture(monkeypatch, tokens)
    result = validator._greedy_continuation(None, tokenizer, {"context_ceiling": 16384}, {})
    assert result["token_ids"] == tokens
    assert result["ended_by_oga"] is True  # Includes completion exactly at the output limit.
    assert result["finite_logits"] is True
    assert generated().logit_positions == list(range(len(tokens)))


@pytest.mark.parametrize("nonfinite_at", [0, 1])
def test_greedy_rejects_nonfinite_logits_before_sampling(monkeypatch, nonfinite_at):
    tokenizer, generated = _greedy_fixture(monkeypatch, [32, 198, 32], nonfinite_at=nonfinite_at)
    with pytest.raises(validator.ValidationError, match="non-finite logits"):
        validator._greedy_continuation(None, tokenizer, {"context_ceiling": 16384}, {})
    assert len(generated().generated) == nonfinite_at
