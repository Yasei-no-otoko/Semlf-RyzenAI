"""Pure header transformation for Ryzen AI 1.8 Qwen3.5 prefill LinearAttention.

Only the measured BF16, batch-one, 24-layer native contract is supported. No
external tensor data, SDK modules, runtime sessions or device APIs are accessed.
"""

from __future__ import annotations

import copy
import hashlib

import onnx
from onnx import TensorProto as T, helper as h

_WIDTHS = {0: 2048, 1: 2048, 2: 4096, 4: 32, 5: 32}
_STATE = [1, 32, 128, 128]
_ATTRIBUTES = {
    "q_num_heads": 16,
    "kv_num_heads": 32,
    "update_rule": b"gated_delta",
    "chunk_size": 64,
    "ryzenai_absorb_input_cast_indices": [4],
    "ryzenai_absorb_input_cast_dtypes": [10],
    "ryzenai_absorb_input_order": [0],
    "seq_length": 4096,
    "q_seq": 4096,
    "k_heads": 16,
    "v_heads": 32,
    "k_head_dim": 128,
    "v_head_dim": 128,
    "scale": 1.0,
    "hybrid_llm_cast_input": [],
    "hybrid_llm_cast_output": [],
}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(proto):
    return hashlib.sha256(proto.SerializeToString()).hexdigest()


def _graphs(graph):
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _graphs(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for child in attr.graphs:
                    yield from _graphs(child)


def _native_keys(model):
    return sorted(
        (n.name, n.domain, n.op_type)
        for g in _graphs(model.graph)
        for n in g.node
        if n.domain == "com.ryzenai"
    )


def _shape(value):
    _require(
        value.type.HasField("tensor_type") and value.type.tensor_type.HasField("shape"),
        "Missing tensor shape declaration",
    )
    return [
        d.dim_value if d.HasField("dim_value") else d.dim_param
        for d in value.type.tensor_type.shape.dim
    ]


def _declarations(graph, relevant_names):
    found = {}
    for value in [*graph.input, *graph.output, *graph.value_info]:
        if value.name not in relevant_names:
            continue
        if value.name in found:
            _require(
                found[value.name].type == value.type, "Conflicting tensor declaration"
            )
        found[value.name] = value
    return found


def _validate(node, declarations):
    _require(
        node.name and len(node.input) == 6 and len(node.output) == 2,
        "LinearAttention requires a name, six inputs and two outputs",
    )
    _require(
        len(set(node.output)) == 2 and all(node.input) and all(node.output),
        "Invalid LinearAttention tensor names",
    )
    _require(
        len(node.attribute) == len(_ATTRIBUTES),
        "Unsupported LinearAttention attribute count",
    )
    actual = {a.name: h.get_attribute_value(a) for a in node.attribute}
    _require(actual == _ATTRIBUTES, "Unsupported native LinearAttention attributes")
    for attr in node.attribute:
        expected = _ATTRIBUTES[attr.name]
        expected_type = (
            onnx.AttributeProto.INTS
            if isinstance(expected, list)
            else onnx.AttributeProto.STRING
            if isinstance(expected, bytes)
            else onnx.AttributeProto.FLOAT
            if isinstance(expected, float)
            else onnx.AttributeProto.INT
        )
        _require(attr.type == expected_type, "Unsupported native attribute type")
    for name in [*node.input, *node.output]:
        _require(
            name in declarations, "Missing LinearAttention tensor declaration: " + name
        )
        _require(
            declarations[name].type.tensor_type.elem_type == T.BFLOAT16,
            "LinearAttention boundary must be BF16",
        )
    sequence = _shape(declarations[node.input[0]])
    _require(
        len(sequence) == 3
        and sequence[0] == 1
        and isinstance(sequence[1], str)
        and sequence[1],
        "Expected batch-one dynamic sequence",
    )
    for index, width in _WIDTHS.items():
        _require(
            _shape(declarations[node.input[index]]) == [1, sequence[1], width],
            "Incompatible LinearAttention sequence shape",
        )
    _require(
        _shape(declarations[node.input[3]]) == _STATE,
        "Incompatible recurrent state shape",
    )
    _require(
        _shape(declarations[node.output[0]]) == [1, sequence[1], 4096],
        "Incompatible attention output shape",
    )
    _require(
        _shape(declarations[node.output[1]]) == _STATE,
        "Incompatible state output shape",
    )


def _replacement(original, declarations):
    _validate(original, declarations)
    prefix = original.name + "/TokenLoop"

    def local(suffix):
        return prefix + "/" + suffix

    def vi(name, dtype, shape):
        return h.make_tensor_value_info(name, dtype, shape)

    def const(name, dtype, shape, values):
        return h.make_node(
            "Constant",
            [],
            [name],
            name=name + "/Constant",
            value=h.make_tensor(name + "/value", dtype, shape, values),
        )

    body_nodes = [const(local("sequence_axis"), T.INT64, [1], [1])]
    infos, steps = [], {}
    for index, width in _WIDTHS.items():
        sliced, step = local(f"slice_{index}"), local(f"step_{index}")
        body_nodes.extend(
            [
                h.make_node(
                    "Gather",
                    [original.input[index], local("iteration")],
                    [sliced],
                    axis=1,
                    name=local(f"gather_{index}"),
                ),
                h.make_node(
                    "Unsqueeze",
                    [sliced, local("sequence_axis")],
                    [step],
                    name=local(f"unsqueeze_{index}"),
                ),
            ]
        )
        infos.extend(
            [vi(sliced, T.BFLOAT16, [1, width]), vi(step, T.BFLOAT16, [1, 1, width])]
        )
        steps[index] = step
    token = copy.deepcopy(original)
    del token.input[:]
    token.input.extend(
        [steps[0], steps[1], steps[2], local("carried_state"), steps[4], steps[5]]
    )
    del token.output[:]
    token.output.extend([local("step_attention"), local("next_state")])
    restored = copy.deepcopy(token)
    del restored.input[:]
    restored.input.extend(original.input)
    del restored.output[:]
    restored.output.extend(original.output)
    _require(restored == original, "Native name/attributes unexpectedly changed")
    body_nodes.extend(
        [
            token,
            h.make_node(
                "Identity",
                [local("condition")],
                [local("condition_out")],
                name=local("condition_identity"),
            ),
        ]
    )
    body = h.make_graph(
        body_nodes,
        local("body"),
        [
            vi(local("iteration"), T.INT64, []),
            vi(local("condition"), T.BOOL, []),
            vi(local("carried_state"), T.BFLOAT16, _STATE),
        ],
        [
            vi(local("condition_out"), T.BOOL, []),
            vi(local("next_state"), T.BFLOAT16, _STATE),
            vi(local("step_attention"), T.BFLOAT16, [1, 1, 4096]),
        ],
        value_info=infos,
    )
    nodes = [
        h.make_node(
            "Shape",
            [original.input[0]],
            [local("query_shape")],
            name=local("query_shape_node"),
        ),
        const(local("sequence_index"), T.INT64, [], [1]),
        h.make_node(
            "Gather",
            [local("query_shape"), local("sequence_index")],
            [local("trip_count")],
            axis=0,
            name=local("trip_count_node"),
        ),
        const(local("initial_condition"), T.BOOL, [], [True]),
        h.make_node(
            "Loop",
            [local("trip_count"), local("initial_condition"), original.input[3]],
            [original.output[1], local("scan_attention")],
            body=body,
            name=local("loop"),
        ),
        const(local("attention_shape"), T.INT64, [3], [1, -1, 4096]),
        h.make_node(
            "Reshape",
            [local("scan_attention"), local("attention_shape")],
            [original.output[0]],
            name=local("restore_attention_shape"),
        ),
    ]
    return nodes, dict(
        native_name=original.name,
        original_node_sha256=_sha(original),
        body_native_node_sha256=_sha(token),
        prefix=prefix,
        original_inputs=list(original.input),
        original_outputs=list(original.output),
        trip_count_source=original.input[0],
        sequence_axis=1,
    )


def transform_prefill_linear_attention(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, dict]:
    """Return a new header and evidence; do not mutate or materialize the input.

    Sequence length must be positive at execution. The existing model interface
    and caller enforce that runtime constraint; this transform adds no padding.
    """
    _require(isinstance(model, onnx.ModelProto), "Expected an ONNX ModelProto")
    standard_versions = {
        o.version for o in model.opset_import if o.domain in ("", "ai.onnx")
    }
    _require(standard_versions == {21}, "Expected the pinned standard opset21 contract")
    _require(
        any(o.domain == "com.ryzenai" and o.version == 1 for o in model.opset_import),
        "Missing native RyzenAI opset1",
    )
    original_native = _native_keys(model)
    _require(
        all(n for n, _, _ in original_native)
        and len({n for n, _, _ in original_native}) == len(original_native),
        "Native operator names must be unique",
    )
    targets = [
        n
        for n in model.graph.node
        if n.domain == "com.ryzenai" and n.op_type == "LinearAttention"
    ]
    _require(
        len(targets) == 24
        and sum(op == "LinearAttention" for _, _, op in original_native) == 24,
        "Expected exactly24 top-level native LinearAttention nodes",
    )
    declarations = _declarations(
        model.graph, {name for node in targets for name in [*node.input, *node.output]}
    )
    occupied = set()
    for graph in _graphs(model.graph):
        occupied.update(
            v.name
            for v in [
                *graph.input,
                *graph.output,
                *graph.value_info,
                *graph.initializer,
            ]
        )
        for node in graph.node:
            occupied.add(node.name)
            occupied.update([*node.input, *node.output])
    transformed = copy.deepcopy(model)
    nodes, reconstructed, contracts = [], [], []
    for original in model.graph.node:
        if original.domain == "com.ryzenai" and original.op_type == "LinearAttention":
            replacement, contract = _replacement(original, declarations)
            prefix = contract["prefix"]
            _require(
                not any(
                    name == prefix or name.startswith(prefix + "/") for name in occupied
                ),
                "Generated Loop namespace collision",
            )
            for item in replacement:
                occupied.add(item.name)
                occupied.update(item.output)
            nodes.extend(replacement)
            contracts.append(contract)
        else:
            nodes.append(original)
        reconstructed.append(original)
    del transformed.graph.node[:]
    transformed.graph.node.extend(nodes)
    restored = copy.deepcopy(transformed)
    del restored.graph.node[:]
    restored.graph.node.extend(reconstructed)
    _require(
        restored.SerializeToString() == model.SerializeToString(),
        "Non-LA model content changed",
    )
    _require(
        _native_keys(transformed) == original_native,
        "Native eager protobuf name/type coverage changed",
    )
    report = dict(
        name="native_linear_attention_token_loop",
        version=1,
        source_header_proto_sha256=_sha(model),
        transformed_prefill_header_sha256=_sha(transformed),
        changed_nodes=len(contracts),
        preserved_native_operator_names_and_types=True,
        preserved_original_external_tensor_extents=True,
        preserved_non_la_model_content=True,
        required_global_chunk_size=64,
        trip_count="Shape(query)[1]",
        state_boundary_dtype="BF16",
        runtime_validation="not performed by this conversion command",
        limitations=[
            "Positive sequence length and batch1 only",
            "Host Loop scheduling; native NPU token LinearAttention",
            "State rounds at BF16 token boundaries; batched-prefill bit equality is not asserted",
        ],
        nodes=contracts,
    )
    return transformed, report
