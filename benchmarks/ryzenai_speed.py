"""AMD Ryzen AI timing helpers for an already-loaded SemIf NPU model.

The caller owns model loading and create-only evidence files.  These helpers
never import Torch or GPU libraries, and make no claim that unsupported SemIf
cache modes run on OGA.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

try:  # Works both as `benchmarks.ryzenai_speed` and script-neighbor import.
    from benchmarks.decision_vs_generation import compact_messages
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI process.
    from decision_vs_generation import compact_messages
from semif_phase1.ryzenai_backend import RyzenAiTokenizer, _ids, score


UNSUPPORTED_MODES = {
    "serial_prefix": "Ryzen AI OGA backend has no SemIf serial-prefix cache support.",
    "parallel_shared": "Ryzen AI OGA backend has no SemIf shared-state parallel support.",
    "reranker": "Ryzen AI OGA backend supports direct categorical logits only.",
}


def public_input(path: Path, repository_root: Path) -> dict:
    """Return a reproducible relative fixture identity without a local path."""
    resolved_root = repository_root.resolve()
    try:
        relative = path.resolve().relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError("Benchmark input must be inside the repository") from error
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _positive_int(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strip_model(row: dict) -> dict:
    """Avoid repeating immutable model provenance in every persisted row."""
    return {key: value for key, value in row.items() if key != "model"}


def _validate_decision21(rows: list[dict]) -> None:
    if len(rows) != 21 or len({row.get("state") for row in rows}) != 1:
        raise ValueError("Expected one shared-state group of exactly 21 decisions")


def _validate_generation_rows(rows: list[dict]) -> None:
    if not 1 <= len(rows) <= 21 or len({row.get("state") for row in rows}) != 1:
        raise ValueError("Expected 1-21 decisions sharing one state for compact generation")


def _eos_ids(tokenizer: RyzenAiTokenizer, metadata: dict) -> set[int]:
    """Use the loaded OGA model's stop IDs, which may differ from HF metadata."""
    value = getattr(tokenizer.reference, "eos_token_id", None)
    values = value if isinstance(value, (list, tuple, set)) else [value]
    if not values or any(not isinstance(token, int) or isinstance(token, bool) for token in values):
        raise ValueError("Reference tokenizer must expose integer eos_token_id values")
    reference_ids = set(values)
    source = metadata.get("source")
    config_path = Path(source) / "genai_config.json" if isinstance(source, str) else None
    if config_path is not None and config_path.is_file():
        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))["model"]
            configured = model_config["eos_token_id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("Cannot read model eos_token_id from genai_config.json") from error
        configured_values = configured if isinstance(configured, list) else [configured]
        if (not configured_values or any(not isinstance(token, int) or isinstance(token, bool)
                                        or token < 0 for token in configured_values)):
            raise ValueError("genai_config.json eos_token_id must contain nonnegative integers")
        vocab_size = model_config.get("vocab_size", metadata.get("vocab_size"))
        if vocab_size is not None:
            if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
                raise ValueError("genai_config.json vocab_size must be a positive integer")
            if any(token >= vocab_size for token in configured_values):
                raise ValueError("genai_config.json eos_token_id exceeds vocab_size")
        return set(configured_values)
    if any(token < 0 for token in reference_ids):
        raise ValueError("Reference tokenizer eos_token_id values must be nonnegative")
    return reference_ids


def _parse_choices(text: str, rows: list[dict]) -> list[str] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if (isinstance(parsed, list) and len(parsed) == len(rows)
            and all(isinstance(choice, str) and choice in {"yes", "no"} for choice in parsed)):
        return parsed
    return None


def run_compact_generation(model, tokenizer: RyzenAiTokenizer, metadata: dict, state: str,
                           rows: list[dict], *, max_tokens: int = 4096,
                           max_new_tokens: int = 128) -> dict:
    """Run OGA-native greedy generation of the compact 21-choice JSON array."""
    import numpy as np
    import onnxruntime_genai as oga

    _validate_generation_rows(rows)
    if state != rows[0]["state"]:
        raise ValueError("Compact generation state must equal the shared decision state")
    _positive_int(max_tokens, "max_tokens")
    _positive_int(max_new_tokens, "max_new_tokens")
    ceiling = metadata.get("context_ceiling")
    _positive_int(ceiling, "metadata.context_ceiling")
    budget = min(max_tokens, ceiling)
    if max_new_tokens >= budget:
        raise ValueError("max_new_tokens must leave room for the compact prompt")

    started = time.perf_counter()
    prompt = tokenizer.reference.apply_chat_template(
        compact_messages(state, rows), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    reference_ids = _ids(tokenizer.reference.encode(prompt, add_special_tokens=False), label="reference")
    ids = _ids(tokenizer.oga.encode(prompt), label="OGA")
    if ids != reference_ids:
        raise ValueError("OGA tokenizer IDs differ from the reference compact-generation tokenizer")
    if not ids or len(ids) > budget - max_new_tokens:
        raise ValueError(f"Compact generation has {len(ids)} input tokens beyond reserved limit {budget - max_new_tokens}")
    eos_ids = _eos_ids(tokenizer, metadata)

    params = oga.GeneratorParams(model)
    params.set_search_options(max_length=len(ids) + max_new_tokens, batch_size=1, do_sample=False)
    generator = None
    pieces, timeline, output_tokens, first_token_at, saw_eos = [], [], 0, None, False
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
            output_tokens += 1
            elapsed = time.perf_counter() - started
            if first_token_at is None:
                first_token_at = elapsed
            piece = stream.decode(token)
            timeline.append({"seconds": elapsed, "token_id": token, "text": piece})
            if piece and token not in eos_ids:
                pieces.append(piece)
            if token in eos_ids:
                saw_eos = True
                break
    finally:
        if generator is not None:
            del generator

    text = "".join(pieces).strip()
    choices = _parse_choices(text, rows)
    return {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_tokens": len(ids),
        "output_tokens": output_tokens,
        "time_to_first_token_seconds": first_token_at,
        "total_seconds": time.perf_counter() - started,
        "output_text": text,
        "valid_complete_array": choices is not None,
        "choices": choices,
        "timeline": timeline,
        "eos_token_ids": sorted(eos_ids),
        "ended_by_eos": saw_eos,
        "truncated": not saw_eos,
    }


def _choice(row: dict) -> str:
    return row["option_ids"][max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)]


def benchmark_decision21(model, tokenizer: RyzenAiTokenizer, metadata: dict, rows: list[dict], *,
                         repeats: int = 3, max_tokens: int = 4096, max_new_tokens: int = 128,
                         on_row=None) -> dict:
    """Warm and time fresh 21-row direct scoring against compact OGA generation."""
    _validate_decision21(rows)
    _positive_int(repeats, "repeats")
    _positive_int(max_tokens, "max_tokens")
    _positive_int(max_new_tokens, "max_new_tokens")
    for row in rows:
        score(model, tokenizer, row, metadata, max_tokens)
    warmup = run_compact_generation(model, tokenizer, metadata, rows[0]["state"], rows[:1],
                                    max_tokens=max_tokens, max_new_tokens=16)
    if not warmup["timeline"]:
        raise RuntimeError("Compact generation warmup emitted no token events")

    direct_runs, generation_runs = [], []
    for repeat in range(repeats):
        started = time.perf_counter()
        outputs = []
        for row in rows:
            output = _strip_model(score(model, tokenizer, row, metadata, max_tokens))
            outputs.append(output)
        direct_runs.append({"repeat": repeat, "total_seconds": time.perf_counter() - started,
                            "outputs": outputs})
        if on_row is not None:
            for output in outputs:
                on_row("decision21_direct", repeat, output)

        generated = run_compact_generation(model, tokenizer, metadata, rows[0]["state"], rows,
                                           max_tokens=max_tokens, max_new_tokens=max_new_tokens)
        generated["repeat"] = repeat
        generated["agreement_with_direct_argmax"] = (
            sum(_choice(row) == choice for row, choice in zip(outputs, generated["choices"])) / len(rows)
            if generated["choices"] is not None else None
        )
        generation_runs.append(generated)

    direct_median = statistics.median(run["total_seconds"] for run in direct_runs)
    generation_median = statistics.median(run["total_seconds"] for run in generation_runs)
    return {
        "version": "ryzenai-decision21-fresh-v1",
        "model": metadata,
        "compact_prompt_messages": compact_messages(rows[0]["state"], rows),
        "scope": "Warm AMD OGA model; 21 independent fresh direct calls versus one compact greedy JSON array generation.",
        "unsupported_modes": UNSUPPORTED_MODES,
        "direct_fresh": {"runs": direct_runs, "median_total_seconds": direct_median},
        "compact_generation": {"max_new_tokens": max_new_tokens, "runs": generation_runs, "median_total_seconds": generation_median,
                                 "median_output_tokens": statistics.median(run["output_tokens"] for run in generation_runs),
                                 "all_runs_valid_complete_arrays": all(run["valid_complete_array"] for run in generation_runs),
                                 "all_runs_ended_by_eos": all(run["ended_by_eos"] for run in generation_runs),
                                 "truncated_repeats": [run["repeat"] for run in generation_runs if run["truncated"]]},
        "median_wall_ratio_generation_over_direct": generation_median / direct_median,
    }


def benchmark_shape777_fresh(model, tokenizer: RyzenAiTokenizer, metadata: dict, rows: list[dict], *,
                             max_tokens: int = 4096, on_row=None, on_progress=None) -> dict:
    """Time only the supported fresh-direct OGA path over frozen 37x21 Shape777."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("group_id")].append(row)
    if len(rows) != 777 or len(groups) != 37 or any(len(group) != 21 for group in groups.values()):
        raise ValueError("Expected the frozen 37-state x 21-question Shape777 fixture")
    _positive_int(max_tokens, "max_tokens")
    first = next(iter(groups.values()))
    score(model, tokenizer, first[0], metadata, max_tokens)

    group_runs, output_records = [], []
    elapsed = 0.0
    for group_index, (group_id, group) in enumerate(groups.items(), start=1):
        group_started = time.perf_counter()
        group_outputs = []
        for row in group:
            output = _strip_model(score(model, tokenizer, row, metadata, max_tokens))
            group_outputs.append(output)
            output_records.append((group_id, output))
        group_seconds = time.perf_counter() - group_started
        elapsed += group_seconds
        group_runs.append({"group_id": group_id, "total_seconds": group_seconds,
                           "decisions": len(group_outputs)})
        if on_row is not None:
            for output in group_outputs:
                on_row("shape777_fresh", group_id, output)
        if on_progress is not None:
            on_progress(group_index, len(groups))
    return {
        "version": "ryzenai-shape777-fresh-v1",
        "model": metadata,
        "scope": "Warm AMD OGA model; every Shape777 decision is scored through an independent fresh direct generator.",
        "unsupported_modes": UNSUPPORTED_MODES,
        "wall_seconds": elapsed,
        "decisions_per_second": len(output_records) / elapsed,
        "state_p50_seconds": statistics.median(group["total_seconds"] for group in group_runs),
        "groups": group_runs,
        "outputs": [output for _, output in output_records],
    }
