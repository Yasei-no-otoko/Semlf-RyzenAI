"""Temporary full-vocabulary checks for an already configured OGA benchmark.

This module does not import OGA, load a model, or write evidence. The caller
owns the process limits, model identity, workload, and timing boundaries.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import time

import numpy as np


@contextmanager
def guard_generators(
    oga,
    *,
    vocab_size: int,
    observations: list[dict] | None = None,
    max_generated_tokens: int = 128,
    hash_logits: bool = False,
):
    """Check every explicit readout and every sample without another forward.

    OGA reuses logits obtained by get_logits in the following
    generate_next_token. Both the readout and this cache behavior must be
    supported by the caller's pinned runtime. Checks add host work to timings.

    Observations contain scalars and small lists only, including failed calls.
    They retain neither generators nor vocabulary arrays. Dump them outside the
    desired timing boundary. This process-global patch is for one sequential
    benchmark; it is not a thread-safe guard for concurrent OGA users.
    """
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("vocab_size must be a positive integer")
    if type(max_generated_tokens) is not int or not 1 <= max_generated_tokens <= 128:
        raise ValueError("max_generated_tokens must be an integer from 1 to 128")
    if type(hash_logits) is not bool:
        raise ValueError("hash_logits must be a boolean")
    records = [] if observations is None else observations
    if not isinstance(records, list):
        raise TypeError("observations must be a list")
    original = oga.Generator
    started = time.perf_counter()

    class CheckedGenerator:
        def __init__(self, *args, **kwargs):
            self._failed = False
            self._record = {
                "generator_index": len(records) + 1,
                "constructed": False,
                "append_calls_started": 0,
                "append_calls_completed": 0,
                "appended_token_counts": [],
                "explicit_logits_calls": 0,
                "native_logits_calls_started": 0,
                "native_logits_calls_completed": 0,
                "sampling_calls_started": 0,
                "sampling_calls_completed": 0,
                "logits_checks": [],
                "errors": [],
            }
            records.append(self._record)
            try:
                self._generator = original(*args, **kwargs)
                self._record["constructed"] = True
            except Exception as error:
                self._error("construction", error)
                raise

        def _error(self, phase, error):
            self._failed = True
            self._record["errors"].append({
                "phase": phase,
                "type": type(error).__name__,
                "message": str(error),
                "seconds": time.perf_counter() - started,
            })

        def _require_live(self):
            if self._failed:
                raise RuntimeError("Generator guard already rejected this generator")
            if self._generator.is_done():
                raise RuntimeError("Generator guard prohibits logits or sampling after is_done")

        def __getattr__(self, name):
            # No wrapper registry retains this native generator. When the caller
            # releases the proxy, normal reference counting releases it too.
            if name.startswith("_"):
                raise AttributeError(name)
            return getattr(self._generator, name)

        def append_tokens(self, tokens, *args, **kwargs):
            self._record["append_calls_started"] += 1
            try:
                if self._failed:
                    raise RuntimeError("Generator guard already rejected this generator")
                self._record["appended_token_counts"].append(int(np.asarray(tokens).size))
                result = self._generator.append_tokens(tokens, *args, **kwargs)
                self._record["append_calls_completed"] += 1
                return result
            except Exception as error:
                self._error("append", error)
                raise

        def _checked_logits(self, phase, *args, **kwargs):
            check = {"phase": phase, "status": "started", "seconds": time.perf_counter() - started}
            self._record["logits_checks"].append(check)
            try:
                self._require_live()
                self._record["native_logits_calls_started"] += 1
                values = self._generator.get_logits(*args, **kwargs)
                self._record["native_logits_calls_completed"] += 1
                array = np.asarray(values)
                check.update(shape=list(array.shape), dtype=str(array.dtype), elements=int(array.size))
                if array.shape not in ((1, vocab_size), (1, 1, vocab_size)):
                    raise RuntimeError("Generator guard received an unexpected full-vocabulary logits shape")
                if array.dtype.kind != "f":
                    raise RuntimeError("Generator guard requires floating-point logits")
                finite_count = int(np.count_nonzero(np.isfinite(array)))
                check.update(finite_count=finite_count, nonfinite_count=int(array.size) - finite_count,
                             finite=finite_count == array.size)
                if hash_logits:
                    check["raw_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
                if not check["finite"]:
                    raise RuntimeError("Generator guard rejected non-finite full-vocabulary logits")
                check["status"] = "passed"
                return values
            except Exception as error:
                check["status"] = "failed"
                self._error(phase, error)
                raise
            finally:
                check["completed_seconds"] = time.perf_counter() - started

        def get_logits(self, *args, **kwargs):
            self._record["explicit_logits_calls"] += 1
            return self._checked_logits("explicit_readout", *args, **kwargs)

        def generate_next_token(self, *args, **kwargs):
            try:
                self._require_live()
                if self._record["sampling_calls_started"] >= max_generated_tokens:
                    raise RuntimeError("Generator guard generated-token limit exceeded")
                # Do not retain the array after checking. The pinned OGA runtime
                # consumes its cached logits below instead of running again.
                self._checked_logits("before_sample")
                self._record["sampling_calls_started"] += 1
                result = self._generator.generate_next_token(*args, **kwargs)
                self._record["sampling_calls_completed"] += 1
                return result
            except Exception as error:
                if not self._failed:
                    self._error("sampling", error)
                raise

    oga.Generator = CheckedGenerator
    try:
        yield records
    finally:
        oga.Generator = original
