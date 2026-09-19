"""Create-only validation for a custom Quark Qwen3.5 Ryzen AI NPU artifact.

This is deliberately separate from the published benchmark suites.  It checks
the completed 16K NPU artifact with owned prompts only, including direct-logit
prefills at the 16K boundary.  It must be run from the AMD 1.8
environment after the artifact is ready; it never falls back to CPU or GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

try:  # Works as either ``python benchmarks/...`` or an imported module.
    from benchmarks.ryzenai_benchmark import npu_activity
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI process.
    benchmark_directory = str(Path(__file__).resolve().parent)
    if benchmark_directory not in sys.path:
        sys.path.insert(0, benchmark_directory)
    from ryzenai_benchmark import npu_activity
from semif_phase1 import ryzenai_backend as backend


SOURCE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_MODEL_TYPE = "qwen3_5_text"
EXPECTED_CONTEXT_CEILING = 16_384
GENERATION_TOKENS = 16
MAX_BOUNDARY_GAP_TOKENS = 8


class ValidationError(RuntimeError):
    """A reportable validation failure."""


def _write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_genai_config(source: Path) -> dict[str, Any]:
    path = source / "genai_config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"Cannot read {path}") from error
    if not isinstance(config, dict):
        raise ValidationError("genai_config.json must contain an object")
    model = config.get("model")
    if not isinstance(model, dict) or model.get("type") != EXPECTED_MODEL_TYPE:
        raise ValidationError(f"Artifact model.type must be {EXPECTED_MODEL_TYPE!r}")
    return config


def _owned_rows() -> list[tuple[dict[str, Any], str]]:
    """Short, owned sanity checks; correctness is reported, never asserted."""
    return [
        ({
            "id": "owned-en-blue-first", "state": "The marker is BLUE.",
            "question": "What color is the marker?",
            "options": [{"id": "blue", "description": "BLUE"},
                        {"id": "red", "description": "RED"}],
        }, "blue"),
        ({
            "id": "owned-ja-blue-second", "state": "空の色は青です。",
            "question": "空の色は何ですか？",
            "options": [{"id": "red", "description": "赤"},
                        {"id": "blue", "description": "青"}],
        }, "blue"),
        ({
            "id": "owned-en-blue-reversed", "state": "The marker is BLUE.",
            "question": "What color is the marker?",
            "options": [{"id": "red", "description": "RED"},
                        {"id": "blue", "description": "BLUE"}],
        }, "blue"),
    ]


def _direct_result(model: Any, tokenizer: backend.RyzenAiTokenizer, metadata: dict[str, Any],
                   row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    result = backend.score(model, tokenizer, row, metadata, max_tokens=max_tokens)
    logits = np.asarray(result["option_logits"], dtype=np.float64)
    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    if not (np.isfinite(logits).all() and np.isfinite(probabilities).all()):
        raise ValidationError(f"{row['id']} returned non-finite categorical values")
    if probabilities.ndim != 1 or len(probabilities) != len(row["options"]):
        raise ValidationError(f"{row['id']} returned a malformed categorical distribution")
    if not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValidationError(f"{row['id']} probabilities do not sum to one")
    return result


def _greedy_continuation(model: Any, tokenizer: backend.RyzenAiTokenizer,
                         metadata: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Exercise native OGA cache updates without manually supplying state tensors."""
    import onnxruntime_genai as oga

    ids, _slots, prompt_hash = backend._encode_prompt(tokenizer, row, metadata["context_ceiling"])
    params = oga.GeneratorParams(model)
    params.set_search_options(max_length=len(ids) + GENERATION_TOKENS, batch_size=1, do_sample=False)
    generator = oga.Generator(model, params)
    stream = tokenizer.oga.create_stream()
    token_ids: list[int] = []
    pieces: list[str] = []
    finite_logits = True
    ended_by_oga = False
    started = time.perf_counter()
    try:
        generator.append_tokens(np.asarray(ids, dtype=np.int32))
        ended_by_oga = bool(generator.is_done())
        while len(token_ids) < GENERATION_TOKENS and not ended_by_oga:
            # GetLogits may run the next forward pass and make IsDone false.
            # Inspect only logits that will actually be sampled, before generation.
            finite_logits = bool(np.isfinite(np.asarray(generator.get_logits())).all())
            if not finite_logits:
                raise ValidationError("Native OGA greedy continuation returned non-finite logits")
            generator.generate_next_token()
            next_ids = backend._ids(generator.get_next_tokens(), label="OGA generated")
            if len(next_ids) != 1:
                raise ValidationError("Native OGA greedy generation returned other than one token")
            token_ids.append(next_ids[0])
            pieces.append(stream.decode(next_ids[0]))
            ended_by_oga = bool(generator.is_done())
    finally:
        del generator
    return {
        "prompt_sha256": prompt_hash,
        "input_tokens": len(ids),
        "requested_output_tokens": GENERATION_TOKENS,
        "output_tokens": len(token_ids),
        "token_ids": token_ids,
        "decoded_text": "".join(pieces),
        "ended_by_oga": ended_by_oga,
        "finite_logits": finite_logits,
        "seconds": time.perf_counter() - started,
        "cache_mode": "native OGA Generator append_tokens/generate_next_token; no external recurrent-state inputs",
    }


def _long_row(filler_count: int, *, sentinel_at_beginning: bool = False) -> dict[str, Any]:
    state = (
        "BLUE sentinel. Neutral filler follows." + " x" * filler_count
        if sentinel_at_beginning
        else "Neutral filler follows. " + " x" * filler_count + " BLUE sentinel."
    )
    return {
        "id": "owned-long-blue-sentinel-head" if sentinel_at_beginning else "owned-long-blue-sentinel-tail",
        "state": state,
        "question": "What is the sentinel color?",
        "options": [{"id": "blue", "description": "BLUE"},
                    {"id": "red", "description": "RED"}],
    }


def _overflow_token_count(tokenizer: backend.RyzenAiTokenizer, row: dict[str, Any]) -> int:
    """Count a rejected canonical prompt while still requiring tokenizer agreement."""
    prompt = tokenizer.reference.apply_chat_template(
        backend.direct_messages(row, labels=backend.NPU_LABELS), tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    reference_ids = backend._ids(
        tokenizer.reference.encode(prompt, add_special_tokens=False), label="reference"
    )
    oga_ids = backend._ids(tokenizer.oga.encode(prompt), label="OGA")
    if oga_ids != reference_ids:
        raise ValidationError("Overflow prompt tokenizer IDs differ between OGA and the reference tokenizer")
    return len(oga_ids)


def _longest_owned_row(tokenizer: backend.RyzenAiTokenizer, ceiling: int, *, sentinel_at_beginning: bool = False) -> tuple[dict[str, Any], int, int, str]:
    """Find a boundary prompt and a separately rejected, one-step-longer prompt."""
    def count(filler_count: int) -> int:
        row = _long_row(filler_count, sentinel_at_beginning=sentinel_at_beginning)
        ids, _slots, _digest = backend._encode_prompt(tokenizer, row, ceiling)
        return len(ids)

    low, low_tokens = 0, count(0)
    high = 1
    overflow_rejection: str | None = None
    while True:
        try:
            tokens = count(high)
        except ValueError as error:
            if "exceed limit" not in str(error):
                raise
            overflow_rejection = str(error)
            break
        if tokens > ceiling:
            break
        low, low_tokens, high = high, tokens, high * 2
        if high > ceiling * 4:
            raise ValidationError("Neutral filler could not reach the configured context ceiling")

    while low + 1 < high:
        middle = (low + high) // 2
        try:
            tokens = count(middle)
        except ValueError as error:
            if "exceed limit" not in str(error):
                raise
            high = middle
            continue
        if tokens <= ceiling:
            low, low_tokens = middle, tokens
        else:
            high = middle
    overflow_tokens = _overflow_token_count(tokenizer, _long_row(high, sentinel_at_beginning=sentinel_at_beginning))
    if overflow_tokens <= ceiling:
        raise ValidationError("Expected the first rejected neutral-filler prompt to exceed the context ceiling")
    try:
        count(high)
    except ValueError as error:
        if "exceed limit" not in str(error):
            raise
        overflow_rejection = str(error)
    else:
        raise ValidationError("Overflow prompt was not rejected by the canonical prompt boundary")
    return _long_row(low, sentinel_at_beginning=sentinel_at_beginning), low_tokens, overflow_tokens, overflow_rejection


def _counter_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"NPU counter {name} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isdecimal():
        parsed = int(value)
    else:
        raise ValidationError(f"NPU counter {name} must be an integer")
    if parsed < 0:
        raise ValidationError(f"NPU counter {name} must not be negative")
    return parsed


def _counter_totals(activity: dict[str, Any], *, require_context: bool) -> dict[str, int]:
    if not activity.get("available"):
        raise ValidationError("xrt-smi activity counters are unavailable; cannot prove NPU execution")
    contexts = activity.get("benchmark_contexts")
    if not isinstance(contexts, list) or (require_context and not contexts):
        raise ValidationError("NPU activity counters did not contain this process's contexts")
    totals = {"command_submissions": 0, "command_completions": 0, "errors": 0}
    for index, context in enumerate(contexts):
        if not isinstance(context, dict):
            raise ValidationError(f"NPU context {index} is not an object")
        for key in totals:
            if key not in context:
                raise ValidationError(f"NPU context {index} is missing {key}")
            totals[key] += _counter_value(context[key], key)
    return totals


def _verify_npu_activity(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    previous = _counter_totals(before, require_context=False)
    current = _counter_totals(after, require_context=True)
    delta = {key: current[key] - previous[key] for key in current}
    if delta["command_submissions"] <= 0:
        raise ValidationError("NPU command submissions did not increase")
    if delta["command_completions"] <= 0:
        raise ValidationError("NPU command completions did not increase")
    if current["errors"] != 0:
        raise ValidationError(f"NPU context reports {current['errors']} errors")
    return delta


def _validate_loaded_model(source: Path, revision: str, metadata: dict[str, Any]) -> None:
    if revision != SOURCE_REVISION:
        raise ValidationError(f"Expected pinned source revision {SOURCE_REVISION}")
    _read_genai_config(source)
    if metadata.get("context_ceiling") != EXPECTED_CONTEXT_CEILING:
        raise ValidationError(
            f"Expected a {EXPECTED_CONTEXT_CEILING}-token NPU ceiling, got {metadata.get('context_ceiling')!r}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="completed local 16K NPU artifact")
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--revision", default=SOURCE_REVISION, help="pinned Qwen3.5 source revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.model.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("Output directory must be new; validation evidence is create-only")
    if not source.is_dir():
        raise SystemExit(f"Model directory does not exist: {source}")
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "version": "qwen35-custom-rtn-npu-validation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "source_revision": args.revision,
        "model_directory": str(source),
        "execution": "Custom Quark 0.11 uint4 group-128 RTN artifact through AMD RyzenAI 1.8 NPU; no GPU provider configured.",
        "private_local_artifacts": ["hardware-private-before.json", "hardware-private-after.json"],
        "status": "failed",
    }
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    error: Exception | None = None
    try:
        before = npu_activity(output, "before")
        print("Loading custom Qwen3.5 RyzenAI model...", flush=True)
        model, tokenizer, metadata = backend.load_model(str(source), args.revision)
        metadata = dict(metadata)
        metadata["execution"] = report["execution"]
        _validate_loaded_model(source, args.revision, metadata)
        report["model"] = metadata
        report["artifact_hashes"] = metadata["source_artifact_sha256"]
        report["genai_config_sha256"] = _sha256(source / "genai_config.json")
        report["validator_sha256"] = _sha256(Path(__file__))
        _write_new(output / "stage-model-loaded.json", report)

        short_results = []
        report["owned_direct_examples"] = short_results
        for row, expected_id in _owned_rows():
            print(f"Scoring {row['id']}...", flush=True)
            result = _direct_result(model, tokenizer, metadata, row, EXPECTED_CONTEXT_CEILING)
            choice = result["option_ids"][int(np.argmax(result["probabilities"]))]
            short_results.append({
                "id": row["id"], "expected_option_id": expected_id, "argmax_option_id": choice,
                "matches_expected": choice == expected_id, "result": result,
            })
            _write_new(output / f"stage-{row['id']}.json", short_results[-1])
        report["owned_direct_correct_count"] = sum(item["matches_expected"] for item in short_results)

        print("Checking native greedy continuation...", flush=True)
        report["native_greedy_continuation"] = _greedy_continuation(
            model, tokenizer, metadata, _owned_rows()[0][0])
        _write_new(output / "stage-native-greedy.json", report["native_greedy_continuation"])

        long_row, planned_tokens, overflow_tokens, overflow_rejection = _longest_owned_row(
            tokenizer, EXPECTED_CONTEXT_CEILING
        )
        if planned_tokens <= 4096:
            raise ValidationError(f"Long owned prompt reached only {planned_tokens} tokens; it did not cross 4K")
        boundary_gap = EXPECTED_CONTEXT_CEILING - planned_tokens
        if boundary_gap > MAX_BOUNDARY_GAP_TOKENS:
            raise ValidationError(
                f"Long owned prompt is {boundary_gap} tokens below the context ceiling; "
                f"maximum permitted gap is {MAX_BOUNDARY_GAP_TOKENS}"
            )
        print(f"Scoring tail sentinel at {planned_tokens} input tokens...", flush=True)
        long_result = _direct_result(model, tokenizer, metadata, long_row, EXPECTED_CONTEXT_CEILING)
        long_choice = long_result["option_ids"][int(np.argmax(long_result["probabilities"]))]
        report["long_direct_score"] = {
            "id": long_row["id"], "constructed_input_tokens": planned_tokens,
            "expected_option_id": "blue", "argmax_option_id": long_choice,
            "matches_expected": long_choice == "blue",
            "actual_input_tokens": long_result["input_tokens"], "crosses_4k": long_result["input_tokens"] > 4096,
            "context_ceiling": EXPECTED_CONTEXT_CEILING, "boundary_gap_tokens": boundary_gap,
            "overflow_prompt_input_tokens": overflow_tokens,
            "overflow_rejected_before_inference": overflow_tokens > EXPECTED_CONTEXT_CEILING,
            "overflow_rejection": overflow_rejection,
            "result": long_result,
        }
        if long_result["input_tokens"] != planned_tokens:
            raise ValidationError("Long prompt token count changed between construction and score")
        _write_new(output / "stage-long-tail.json", report["long_direct_score"])

        head_row, head_tokens, head_overflow_tokens, head_overflow_rejection = _longest_owned_row(
            tokenizer, EXPECTED_CONTEXT_CEILING, sentinel_at_beginning=True
        )
        head_gap = EXPECTED_CONTEXT_CEILING - head_tokens
        if head_tokens <= 4096 or head_gap > MAX_BOUNDARY_GAP_TOKENS:
            raise ValidationError(
                f"Head-sentinel prompt has {head_tokens} tokens and a {head_gap}-token boundary gap"
            )
        print(f"Scoring head sentinel at {head_tokens} input tokens...", flush=True)
        head_result = _direct_result(model, tokenizer, metadata, head_row, EXPECTED_CONTEXT_CEILING)
        head_choice = head_result["option_ids"][int(np.argmax(head_result["probabilities"]))]
        report["long_head_sentinel_direct_score"] = {
            "id": head_row["id"], "expected_option_id": "blue", "argmax_option_id": head_choice,
            "matches_expected": head_choice == "blue", "constructed_input_tokens": head_tokens,
            "actual_input_tokens": head_result["input_tokens"], "crosses_4k": head_result["input_tokens"] > 4096,
            "context_ceiling": EXPECTED_CONTEXT_CEILING, "boundary_gap_tokens": head_gap,
            "overflow_prompt_input_tokens": head_overflow_tokens,
            "overflow_rejected_before_inference": head_overflow_tokens > EXPECTED_CONTEXT_CEILING,
            "overflow_rejection": head_overflow_rejection,
            "purpose": "Check that the BLUE sentinel in the first prefill chunk remains available after later chunks.",
            "result": head_result,
        }
        if head_result["input_tokens"] != head_tokens:
            raise ValidationError("Head-sentinel prompt token count changed between construction and score")
        _write_new(output / "stage-long-head.json", report["long_head_sentinel_direct_score"])
    except Exception as caught:  # preserve evidence, then return a failing status below
        error = caught
        report["error"] = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        try:
            after = npu_activity(output, "after")
        except Exception as caught:
            if error is None:
                error = caught
                report["error"] = {"type": type(caught).__name__, "message": str(caught)}
            else:
                report["after_counter_error"] = {"type": type(caught).__name__, "message": str(caught)}
        report["npu_activity"] = {"before": before, "after": after}
        if error is None and before is not None and after is not None:
            try:
                report["npu_counter_delta"] = _verify_npu_activity(before, after)
            except Exception as caught:
                error = caught
                report["error"] = {"type": type(caught).__name__, "message": str(caught)}
        report["status"] = "passed" if error is None else "failed"
        _write_new(output / "manifest.json", {
            "version": report["version"],
            "created_utc": report["created_utc"],
            "process_id": report["process_id"],
            "source_revision": report["source_revision"],
            "model_directory": report["model_directory"],
            "execution": report["execution"],
            "model": report.get("model"),
            "artifact_hashes": report.get("artifact_hashes"),
            "status": report["status"],
        })
        _write_new(output / "report.json", report)

    print(json.dumps({"status": report["status"], "output": str(output)}, ensure_ascii=False), flush=True)
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
