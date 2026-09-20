from __future__ import annotations

import gc
import hashlib
import json
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from benchmarks.ryzenai_guard import guard_generators


class NativeGenerator:
    """CPU model of OGA's cached-logits state machine, without SDK imports."""

    instances = []

    def __init__(self, *, shape=(1, 1, 7), eos_after=3):
        self.shape = shape
        self.eos_after = eos_after
        self.samples = self.forwards = self.reads = 0
        self.cached = False
        self.value = np.arange(7, dtype=np.float32).reshape(shape)
        self.instances.append(weakref.ref(self))

    def append_tokens(self, tokens):
        assert not self.cached
        self.forwards += 1
        self.cached = True

    def get_logits(self):
        assert not self.is_done(), "Illegal native read after EOS"
        self.reads += 1
        if not self.cached:
            self.forwards += 1
            self.cached = True
        return self.value

    def generate_next_token(self):
        assert not self.is_done(), "Illegal native sample after EOS"
        if not self.cached:
            self.get_logits()
        self.samples += 1
        self.cached = False

    def is_done(self):
        return self.samples >= self.eos_after

    def get_next_tokens(self):
        return np.array([6], dtype=np.int32)


@pytest.fixture
def oga():
    NativeGenerator.instances = []
    return SimpleNamespace(Generator=NativeGenerator)


@pytest.mark.parametrize("shape", [(1, 7), (1, 1, 7)])
def test_direct_reads_cached_logits_without_an_extra_forward(oga, shape):
    observations = []
    with guard_generators(oga, vocab_size=7, observations=observations, hash_logits=True):
        generator = oga.Generator(shape=shape)
        generator.append_tokens(np.array([1, 2], dtype=np.int32))
        values = generator.get_logits()
        native = NativeGenerator.instances[-1]()
        assert native.forwards == 1 and native.samples == 0
        assert values is native.value
    assert oga.Generator is NativeGenerator
    row = observations[0]
    assert row["appended_token_counts"] == [2]
    assert row["explicit_logits_calls"] == row["native_logits_calls_completed"] == 1
    check = row["logits_checks"][0]
    assert check["finite"] and check["finite_count"] == 7 and check["nonfinite_count"] == 0
    assert check["shape"] == list(shape)
    assert check["raw_sha256"] == hashlib.sha256(values.tobytes()).hexdigest()
    json.dumps(observations, allow_nan=False)


def test_each_sample_checks_logits_without_duplicate_forward_and_stops_at_eos(oga):
    with guard_generators(oga, vocab_size=7) as rows:
        generator = oga.Generator(eos_after=3)
        generator.append_tokens(np.array([1], dtype=np.int32))
        native = NativeGenerator.instances[-1]()
        generator.get_logits()  # Explicit inspection before the first sample.
        for count in range(1, 4):
            generator.generate_next_token()
            assert native.forwards == count
            assert native.samples == count
        assert generator.is_done()
        assert generator.get_next_tokens().tolist() == [6]
        with pytest.raises(RuntimeError, match="after is_done"):
            generator.get_logits()
        assert native.forwards == 3 and native.reads == 4
    assert rows[0]["sampling_calls_completed"] == 3
    assert len([r for r in rows[0]["logits_checks"] if r["status"] == "passed"]) == 4


def test_sampling_after_eos_does_not_read_or_run_native(oga):
    with guard_generators(oga, vocab_size=7):
        generator = oga.Generator(eos_after=1)
        generator.append_tokens([1])
        generator.generate_next_token()
        native = NativeGenerator.instances[-1]()
        with pytest.raises(RuntimeError, match="after is_done"):
            generator.generate_next_token()
        assert (native.forwards, native.samples, native.reads) == (1, 1, 1)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method", ["get_logits", "generate_next_token"])
def test_nonfinite_outside_selected_options_fails_before_sampling(oga, bad, method):
    rows = []
    with pytest.raises(RuntimeError, match="non-finite"):
        with guard_generators(oga, vocab_size=7, observations=rows):
            generator = oga.Generator()
            generator.append_tokens([1])
            native = NativeGenerator.instances[-1]()
            native.value[..., -1] = bad  # Selected answer slots 0/1 remain finite.
            getattr(generator, method)()
    assert oga.Generator is NativeGenerator
    assert native.samples == 0 and native.forwards == 1
    assert rows[0]["logits_checks"][0]["nonfinite_count"] == 1
    assert rows[0]["logits_checks"][0]["status"] == "failed"
    assert rows[0]["errors"]
    with pytest.raises(RuntimeError, match="already rejected"):
        generator.generate_next_token()
    assert native.reads == 1
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("value", [
    np.zeros((7,), dtype=np.float32),
    np.zeros((2, 7), dtype=np.float32),
    np.zeros((1, 2, 7), dtype=np.float32),
    np.zeros((1, 1, 6), dtype=np.float32),
    np.zeros((1, 1, 0), dtype=np.float32),
    np.zeros((1, 1, 7), dtype=np.int32),
])
def test_malformed_logits_are_rejected_and_recorded(oga, value):
    with guard_generators(oga, vocab_size=7) as rows:
        generator = oga.Generator()
        generator.append_tokens([1])
        NativeGenerator.instances[-1]().value = value
        with pytest.raises(RuntimeError, match="Generator guard"):
            generator.get_logits()
        assert rows[0]["logits_checks"][0]["shape"] == list(value.shape)
        assert rows[0]["logits_checks"][0]["status"] == "failed"


def test_token_budget_rejects_before_a_new_read_or_forward(oga):
    with guard_generators(oga, vocab_size=7, max_generated_tokens=2) as rows:
        generator = oga.Generator(eos_after=99)
        generator.append_tokens([1])
        generator.generate_next_token()
        generator.generate_next_token()
        native = NativeGenerator.instances[-1]()
        with pytest.raises(RuntimeError, match="limit exceeded"):
            generator.generate_next_token()
        assert (native.forwards, native.reads, native.samples) == (2, 2, 2)
        assert rows[0]["sampling_calls_started"] == 2


@pytest.mark.parametrize("error", [RuntimeError("native failure"), KeyboardInterrupt(), SystemExit(3)])
def test_context_restores_original_on_any_exit(oga, error):
    with pytest.raises(type(error)):
        with guard_generators(oga, vocab_size=7):
            assert oga.Generator is not NativeGenerator
            raise error
    assert oga.Generator is NativeGenerator


def test_native_failure_keeps_partial_counters_and_restores_class(oga):
    rows = []
    with pytest.raises(RuntimeError, match="native failure"):
        with guard_generators(oga, vocab_size=7, observations=rows):
            generator = oga.Generator()
            generator.append_tokens([1])
            native = NativeGenerator.instances[-1]()
            native.get_logits = lambda: (_ for _ in ()).throw(RuntimeError("native failure"))
            generator.get_logits()
    assert oga.Generator is NativeGenerator
    assert rows[0]["native_logits_calls_started"] == 1
    assert rows[0]["native_logits_calls_completed"] == 0
    assert rows[0]["logits_checks"][0]["status"] == "failed"
    assert rows[0]["errors"][0]["message"] == "native failure"


def test_records_do_not_keep_native_generators_or_arrays_alive(oga):
    with guard_generators(oga, vocab_size=7) as rows:
        for _ in range(20):
            generator = oga.Generator()
            native = NativeGenerator.instances[-1]()
            array_ref = weakref.ref(native.value)
            native_ref = weakref.ref(native)
            generator.append_tokens([1])
            generator.get_logits()
            generator.generate_next_token()
            del native, generator
            assert native_ref() is None and array_ref() is None
        assert len(rows) == 20
        assert all(ref() is None for ref in NativeGenerator.instances)
        json.dumps(rows, allow_nan=False)
    gc.collect()
    assert all(ref() is None for ref in NativeGenerator.instances)


@pytest.mark.parametrize("kwargs", [
    {"vocab_size": 0}, {"vocab_size": True}, {"vocab_size": 7.0},
    {"vocab_size": 7, "max_generated_tokens": 129},
    {"vocab_size": 7, "max_generated_tokens": 0},
    {"vocab_size": 7, "max_generated_tokens": True},
    {"vocab_size": 7, "hash_logits": 1},
    {"vocab_size": 7, "observations": {}},
])
def test_invalid_guard_configuration_does_not_patch_sdk(oga, kwargs):
    with pytest.raises((TypeError, ValueError)):
        with guard_generators(oga, **kwargs):
            pytest.fail("Invalid configuration entered the context")
    assert oga.Generator is NativeGenerator
