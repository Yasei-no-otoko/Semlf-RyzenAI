"""OGA JSON-probability generation for comparison with SemIf direct logits."""

from __future__ import annotations

import json
import math
import re
import time

from .core import validate_row
from .ryzenai_backend import NPU_LABELS, RyzenAiTokenizer, _ids


GENERATION_PROMPT_VERSION = "webgpu-probability-json-v1"
GENERATION_PROMPT_VERSION_32 = "webgpu-probability-json-32-checklist-v2"


def _messages(row: dict) -> list[dict]:
    labels = NPU_LABELS[:len(row["options"])]
    options = "\n".join(
        f"{label}. {option['description']}" for label, option in zip(labels, row["options"])
    )
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    if len(labels) <= 16:
        instruction = (
            "Estimate the probability that each allowed option is the correct decision. "
            "Return only one JSON object mapping each option to its probability. "
            'Form every key as "<label>: <full option text>" using the allowed options above. '
            'For example, if the unrelated options were "A. Route north" and "B. Route south", valid output would be: '
            '{"A: Route north": 0.65, "B: Route south": 0.35}\n'
            "For the actual decision, include every supplied option exactly once and in order. "
            "Each value must be a JSON number from 0 to 1, and the probabilities must sum to 1. "
            "Output JSON only, with no markdown or explanation."
        )
    else:
        checklist = "\n".join(
            "- " + json.dumps(f"{label}: {option['description']}", ensure_ascii=False)
            for label, option in zip(labels, row["options"])
        )
        instruction = (
            "Estimate the probability that each allowed option is the correct decision. "
            f"Return only one JSON object with exactly {len(labels)} members. "
            "Use every quoted key in the following checklist exactly once and in this exact order. "
            "Include zero-probability options too; do not use Others or grouping. "
            "Do not omit, rename, merge, or add a key; do not substitute a category name. "
            "Each value must be a JSON number from 0 to 1, and all values must sum to 1. "
            "Output JSON only, with no markdown or explanation.\n"
            f"Required JSON keys:\n{checklist}"
        )
    return [
        {"role": "system", "content": "Make the requested decision from the supplied state. Follow the output format exactly."},
        {"role": "user", "content": f"State:\n{state}\n\nQuestion:\n{row['question']}\n\nAllowed options:\n{options}\n\n{instruction}"},
    ]


def _prompt_ids(tokenizer: RyzenAiTokenizer, row: dict, input_limit: int) -> list[int]:
    prompt = tokenizer.reference.apply_chat_template(
        _messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    reference_ids = _ids(tokenizer.reference.encode(prompt, add_special_tokens=False), label="reference")
    ids = _ids(tokenizer.oga.encode(prompt), label="OGA")
    if ids != reference_ids:
        raise ValueError("OGA tokenizer IDs differ from the reference generation tokenizer")
    if not ids or len(ids) > input_limit:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed reserved limit {input_limit}; no truncation allowed")
    return ids


def _parse(text: str, row: dict) -> tuple[dict | None, str | None]:
    payload = re.sub(r"^<think>[\s\S]*?</think>\s*", "", text.strip(), flags=re.IGNORECASE)
    try:
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate probability option key")
                result[key] = value
            return result

        parsed = json.loads(payload, object_pairs_hook=unique_pairs)
        labels = NPU_LABELS[:len(row["options"])]
        expected = [f"{label}: {option['description']}" for label, option in zip(labels, row["options"])]
        if not isinstance(parsed, dict):
            raise ValueError("expected one JSON object")
        if set(parsed) != set(expected) or len(parsed) != len(expected):
            raise ValueError(f"expected {len(expected)} exact option keys; received {len(parsed)}")
        probabilities = []
        for key in expected:
            value = parsed[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("probabilities must be numbers from 0 to 1")
            probabilities.append(float(value))
        if abs(sum(probabilities) - 1) > 0.02:
            raise ValueError("probabilities must sum to 1")
        return parsed, None
    except (json.JSONDecodeError, ValueError, TypeError, OverflowError) as error:
        return None, str(error) or "invalid JSON"


def generate(model, tokenizer: RyzenAiTokenizer, row: dict, metadata: dict, max_tokens: int = 4096,
             max_new_tokens: int = 256, on_token=None) -> dict:
    """Generate one strict browser-comparison probability object with OGA."""
    import numpy as np
    import onnxruntime_genai as oga

    validate_row(row, max_options=len(NPU_LABELS))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if on_token is not None and not callable(on_token):
        raise ValueError("on_token must be callable")
    ceiling = metadata.get("context_ceiling")
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 1:
        raise ValueError("Missing valid RyzenAI context ceiling in model metadata")
    budget = min(max_tokens, ceiling)
    if max_new_tokens >= budget:
        raise ValueError("max_new_tokens must leave room for the input prompt")

    started = time.perf_counter()
    ids = _prompt_ids(tokenizer, row, budget - max_new_tokens)
    params = oga.GeneratorParams(model)
    params.set_search_options(max_length=len(ids) + max_new_tokens, batch_size=1, do_sample=False)
    pieces, output_tokens, first_token_at = [], 0, None
    generator = None
    try:
        generator = oga.Generator(model, params)
        stream = tokenizer.oga.create_stream()
        generator.append_tokens(np.asarray(ids, dtype=np.int32))
        while output_tokens < max_new_tokens and not generator.is_done():
            generator.generate_next_token()
            generated_ids = _ids(generator.get_next_tokens(), label="OGA generated")
            if len(generated_ids) != 1:
                raise RuntimeError("OGA generation step returned anything other than one token")
            token = generated_ids[0]
            piece = stream.decode(token)
            output_tokens += 1
            if piece:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(piece)
                if on_token is not None:
                    on_token(piece)
        # OGA reports done when max_length is reached even without EOS. Treat an
        # exhausted caller budget as truncated conservatively.
        truncated = output_tokens >= max_new_tokens or not generator.is_done()
    finally:
        if generator is not None:
            del generator

    text = "".join(pieces).strip()
    parsed, validation_error = _parse(text, row)
    result = {
        "text": text,
        "total_seconds": time.perf_counter() - started,
        "ttft_seconds": None if first_token_at is None else first_token_at - started,
        "input_tokens": len(ids),
        "output_tokens": output_tokens,
        "parsed": parsed,
        "valid_json": parsed is not None,
        "truncated": truncated,
        "prompt_version": (GENERATION_PROMPT_VERSION if len(row["options"]) <= 16
                           else GENERATION_PROMPT_VERSION_32),
    }
    if validation_error is not None:
        result["validation_error"] = validation_error
    return result
