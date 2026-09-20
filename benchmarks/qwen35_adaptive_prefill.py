"""Pure ONNX header rewrite for adaptive Ryzen AI Qwen3.5 prefill attention.

Segment lengths bound per-head absolute log-gate sums without changing gates.
Native computation and recurrent state remain BF16. Attention outputs accumulate
in a FLOAT tensor sequence, then cast back exactly for finite BF16 values.

Runtime input sequences must contain at least one token. This header transform
does not run a model or inspect runtime inputs, load external tensor data, import
device libraries, or change source files. The caller checks the saved model in
its destination directory with the installed ORT schemas after packaging.
"""

from __future__ import annotations

import copy
import hashlib
import math

import onnx
from onnx import TensorProto as T, helper as h

try:
    from benchmarks.qwen35_prefill import _declarations, _graphs, _validate
except ModuleNotFoundError as error:
    if error.name != "benchmarks":
        raise
    from qwen35_prefill import _declarations, _graphs, _validate


MAX_SEGMENT = 64
LOG_BUDGET = 64.0
WIDTHS = {0: 2048, 1: 2048, 2: 4096, 4: 32, 5: 32}
STATE_SHAPE = [1, 32, 128, 128]


def _constant(name, dtype, shape, values):
    return h.make_node(
        "Constant",
        [],
        [name],
        name=name + "/Constant",
        value=h.make_tensor(name + "/value", dtype, shape, values),
    )


def _replacement(
    original, declarations, *, log_budget=LOG_BUDGET, max_segment=MAX_SEGMENT
):
    """Return replacement nodes for one exact original native BF16 contract."""
    _validate(original, declarations)
    if (
        not isinstance(max_segment, int)
        or isinstance(max_segment, bool)
        or not 1 <= max_segment <= 64
    ):
        raise ValueError("max_segment must be an integer from 1 through 64")
    if (
        not isinstance(log_budget, (int, float))
        or isinstance(log_budget, bool)
        or not math.isfinite(log_budget)
        or not 0 < log_budget <= 64
    ):
        raise ValueError("log_budget must be finite and in (0, 64]")
    prefix = original.name + "/AdaptiveLoop"

    def n(suffix):
        return prefix + "/" + suffix

    vi = h.make_tensor_value_info
    state_in, position_in, attention_in = (
        n(x) for x in ("state", "position", "attention")
    )
    nodes = [
        _constant(n("zero"), T.INT64, [], [0]),
        _constant(n("one"), T.INT64, [], [1]),
        _constant(n("one_vector"), T.INT64, [1], [1]),
        _constant(n("zero_axis"), T.INT64, [1], [0]),
        _constant(n("max_segment"), T.INT64, [], [max_segment]),
        _constant(n("budget"), T.FLOAT, [], [log_budget]),
        _constant(n("head_axes"), T.INT64, [2], [0, 2]),
        h.make_node(
            "SequenceEmpty",
            [],
            [n("empty_attention")],
            dtype=T.FLOAT,
            name=n("attention_sequence_empty"),
        ),
        h.make_node(
            "Shape", [original.input[0]], [n("query_shape")], name=n("query_shape_node")
        ),
        h.make_node(
            "Gather",
            [n("query_shape"), n("one")],
            [n("sequence_length")],
            axis=0,
            name=n("sequence_length_node"),
        ),
        h.make_node(
            "Less",
            [n("zero"), n("sequence_length")],
            [n("initial_condition")],
            name=n("initial_condition_node"),
        ),
        h.make_node(
            "Cast",
            [original.input[4]],
            [n("gate_float")],
            to=T.FLOAT,
            name=n("gate_float_node"),
        ),
    ]
    body = [
        h.make_node(
            "Add",
            [position_in, n("max_segment")],
            [n("window_unbounded")],
            name=n("window_add"),
        ),
        h.make_node(
            "Min",
            [n("window_unbounded"), n("sequence_length")],
            [n("window_end")],
            name=n("window_min"),
        ),
        h.make_node(
            "Unsqueeze",
            [position_in, n("zero_axis")],
            [n("start_vector")],
            name=n("start_vector_node"),
        ),
        h.make_node(
            "Unsqueeze",
            [n("window_end"), n("zero_axis")],
            [n("window_end_vector")],
            name=n("window_end_vector_node"),
        ),
        h.make_node(
            "Slice",
            [
                n("gate_float"),
                n("start_vector"),
                n("window_end_vector"),
                n("one_vector"),
            ],
            [n("gate_window")],
            name=n("gate_window_node"),
        ),
        h.make_node(
            "Abs",
            [n("gate_window")],
            [n("absolute_gate")],
            name=n("absolute_gate_node"),
        ),
        h.make_node(
            "CumSum",
            [n("absolute_gate"), n("one")],
            [n("cumulative_gate")],
            exclusive=0,
            reverse=0,
            name=n("cumulative_gate_node"),
        ),
        h.make_node(
            "ReduceMax",
            [n("cumulative_gate"), n("head_axes")],
            [n("maximum_head_sum")],
            keepdims=0,
            name=n("maximum_head_sum_node"),
        ),
        h.make_node(
            "LessOrEqual",
            [n("maximum_head_sum"), n("budget")],
            [n("safe_prefix_mask")],
            name=n("safe_prefix_mask_node"),
        ),
        h.make_node(
            "Cast",
            [n("safe_prefix_mask")],
            [n("safe_prefix_int")],
            to=T.INT64,
            name=n("safe_prefix_int_node"),
        ),
        h.make_node(
            "ReduceSum",
            [n("safe_prefix_int"), n("zero_axis")],
            [n("safe_prefix_length")],
            keepdims=0,
            name=n("safe_prefix_length_node"),
        ),
        h.make_node(
            "Max",
            [n("safe_prefix_length"), n("one")],
            [n("segment_length")],
            name=n("segment_length_node"),
        ),
        h.make_node(
            "Add",
            [position_in, n("segment_length")],
            [n("next_position")],
            name=n("next_position_node"),
        ),
        h.make_node(
            "Unsqueeze",
            [n("next_position"), n("zero_axis")],
            [n("end_vector")],
            name=n("end_vector_node"),
        ),
    ]
    steps = {}
    infos = []
    for index, width in WIDTHS.items():
        steps[index] = n(f"input_{index}")
        body.append(
            h.make_node(
                "Slice",
                [
                    original.input[index],
                    n("start_vector"),
                    n("end_vector"),
                    n("one_vector"),
                ],
                [steps[index]],
                name=n(f"slice_{index}"),
            )
        )
        infos.append(vi(steps[index], T.BFLOAT16, [1, "segment_length", width]))
    native = copy.deepcopy(original)
    del native.input[:]
    native.input.extend([steps[0], steps[1], steps[2], state_in, steps[4], steps[5]])
    del native.output[:]
    native.output.extend([n("segment_attention"), n("next_state")])
    # Keep exactly one native node with its original name, attributes and order.
    restored = copy.deepcopy(native)
    del restored.input[:]
    restored.input.extend(original.input)
    del restored.output[:]
    restored.output.extend(original.output)
    if restored != original:
        raise ValueError("Native contract changed")
    body.extend(
        [
            native,
            h.make_node(
                "Cast",
                [n("segment_attention")],
                [n("segment_attention_float")],
                to=T.FLOAT,
                name=n("segment_attention_to_float"),
            ),
            h.make_node(
                "SequenceInsert",
                [attention_in, n("segment_attention_float")],
                [n("next_attention")],
                name=n("attention_sequence_insert"),
            ),
            h.make_node(
                "Less",
                [n("next_position"), n("sequence_length")],
                [n("continue")],
                name=n("continue_node"),
            ),
        ]
    )
    loop_body = h.make_graph(
        body,
        n("body"),
        [
            vi(n("iteration"), T.INT64, []),
            vi(n("condition"), T.BOOL, []),
            vi(state_in, T.BFLOAT16, STATE_SHAPE),
            vi(position_in, T.INT64, []),
            h.make_tensor_sequence_value_info(
                attention_in, T.FLOAT, [1, "segment_length", 4096]
            ),
        ],
        [
            vi(n("continue"), T.BOOL, []),
            vi(n("next_state"), T.BFLOAT16, STATE_SHAPE),
            vi(n("next_position"), T.INT64, []),
            h.make_tensor_sequence_value_info(
                n("next_attention"), T.FLOAT, [1, "segment_length", 4096]
            ),
        ],
        value_info=infos,
    )
    nodes.append(
        h.make_node(
            "Loop",
            [
                n("sequence_length"),
                n("initial_condition"),
                original.input[3],
                n("zero"),
                n("empty_attention"),
            ],
            [original.output[1], n("final_position"), n("attention_sequence")],
            body=loop_body,
            name=n("loop"),
        )
    )
    nodes.extend(
        [
            h.make_node(
                "ConcatFromSequence",
                [n("attention_sequence")],
                [n("attention_float")],
                axis=1,
                new_axis=0,
                name=n("attention_sequence_concat"),
            ),
            h.make_node(
                "Cast",
                [n("attention_float")],
                [original.output[0]],
                to=T.BFLOAT16,
                name=n("attention_to_bf16"),
            ),
        ]
    )
    return nodes


def _check_generated_names(model, original, nodes, used_nodes, used_values):
    """Reject collisions before inserting generated nodes into an existing graph."""
    existing_nodes = {
        n.name for graph in _graphs(model.graph) for n in graph.node if n.name
    }
    existing_values = {
        name
        for graph in _graphs(model.graph)
        for name in [
            *(
                v.name
                for v in [
                    *graph.input,
                    *graph.output,
                    *graph.value_info,
                    *graph.initializer,
                ]
            ),
            *(name for n in graph.node for name in [*n.input, *n.output]),
        ]
        if name
    }
    graph = h.make_graph(nodes, "generated_name_check", [], [])
    generated_nodes = [n.name for g in _graphs(graph) for n in g.node]
    generated_value_definitions = [
        name
        for g in _graphs(graph)
        for name in [
            *(v.name for v in g.input),
            *(name for n in g.node for name in n.output),
        ]
        if name
    ]
    generated_values = set(generated_value_definitions)
    if len(generated_nodes) != len(set(generated_nodes)):
        raise ValueError("Duplicate generated node names")
    if len(generated_value_definitions) != len(generated_values):
        raise ValueError("Duplicate generated tensor names")
    if (set(generated_nodes) - {original.name}) & existing_nodes or set(
        generated_nodes
    ) & used_nodes:
        raise ValueError("Generated node name collides with an existing node")
    if (
        generated_values - set(original.output)
    ) & existing_values or generated_values & used_values:
        raise ValueError("Generated tensor name collides with an existing tensor")
    used_nodes.update(generated_nodes)
    used_values.update(generated_values)


def transform_prefill_adaptive_attention(
    model: onnx.ModelProto,
    *,
    expected_count: int = 24,
    log_budget: float = LOG_BUDGET,
    max_segment: int = MAX_SEGMENT,
) -> tuple[onnx.ModelProto, dict]:
    """Return a copied header and metadata for the exact supported native contract.

    The graph is intended for nonempty runtime sequences. Static zero lengths
    and unsupported declarations are rejected by the shared native validator.
    log_budget is finite in (0, 64]; max_segment is an integer in [1, 64].
    Native kernel numerical accuracy remains a separate runtime validation.
    """
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("model must be an ONNX ModelProto")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
    ):
        raise ValueError("expected_count must be a positive integer")
    if not any(
        x.domain in ("", "ai.onnx") and x.version >= 18 for x in model.opset_import
    ):
        raise ValueError("Adaptive rewrite requires ONNX opset 18 or newer")
    result = copy.deepcopy(model)
    originals = [
        n
        for n in result.graph.node
        if n.domain == "com.ryzenai" and n.op_type == "LinearAttention"
    ]
    if len(originals) != expected_count:
        raise ValueError("Unexpected native LinearAttention count")
    declarations = _declarations(
        result.graph, {s for n in originals for s in [*n.input, *n.output]}
    )
    if len({n.name for n in originals}) != expected_count:
        raise ValueError("Duplicate native node names")
    replacements = {}
    used_nodes, used_values = set(), set()
    for native in originals:
        generated = _replacement(
            native, declarations, log_budget=log_budget, max_segment=max_segment
        )
        _check_generated_names(result, native, generated, used_nodes, used_values)
        replacements[native.name] = generated
    nodes = []
    for node in result.graph.node:
        nodes.extend(replacements[node.name] if node in originals else [node])
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    source_sha = hashlib.sha256(model.SerializeToString()).hexdigest()
    return result, {
        "source_header_sha256": source_sha,
        "native_nodes": expected_count,
        "max_segment": max_segment,
        "absolute_log_gate_budget": log_budget,
        "gate_values_changed": False,
        "attention_accumulator": "FLOAT_sequence_with_BF16_boundary_casts",
        "native_state_dtype": "BF16",
        "nonempty_sequence_required": True,
        "npu_runtime_verified": False,
    }
