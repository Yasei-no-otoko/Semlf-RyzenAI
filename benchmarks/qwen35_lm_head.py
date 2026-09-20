"""Copy-only last-token LM-head rewrite for combined Ryzen AI Qwen3.5 graphs.

Both output branches and the top-level logits declaration must be [1,1,V]:
OGA v0.14 Model::IsPruned checks the session's top-level output dimension 1.
The original full sequence remains available to every state-producing node.
"""

from __future__ import annotations

import copy
import hashlib

import onnx
from onnx import TensorProto as T, helper as h

HEAD = "/lm_head/MatMulNBits"
PRODUCER = "/model/layers.31/mlp/SSMLP"
PREFIX = "/lm_head/LastToken"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _graphs(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graphs(attribute.g)


def _shape(value):
    return [
        d.dim_value if d.HasField("dim_value") else d.dim_param
        for d in value.type.tensor_type.shape.dim
    ]


def _digest(message):
    return hashlib.sha256(message.SerializeToString()).hexdigest()


def _branches(model):
    switches = [
        n for n in model.graph.node if n.op_type == "If" and n.domain in ("", "ai.onnx")
    ]
    _require(len(switches) == 1, "Expected exactly one top-level If")
    switch = switches[0]
    mapping = {
        a.name: a.g for a in switch.attribute if a.type == onnx.AttributeProto.GRAPH
    }
    _require(
        set(mapping) == {"then_branch", "else_branch"}, "Unexpected branch attributes"
    )
    _require(
        not mapping["then_branch"].input and not mapping["else_branch"].input,
        "Branches must capture inputs",
    )
    _require(switch.output[0] == "logits", "Expected first output logits")
    return switch, mapping["then_branch"], mapping["else_branch"]


def transform_prefill_lm_head(
    model: onnx.ModelProto,
    *,
    hidden_size: int = 2560,
    vocab_size: int = 248320,
    expected_states: int = 64,
) -> tuple[onnx.ModelProto, dict]:
    """Return a copied header and provenance for a last-token-only prefill head.

    The default dimensions match the qualified Qwen3.5-4B contract. This function
    performs no numerical execution or external-data access. State-producing
    nodes still process the complete sequence. A separate runtime test is
    required because changing native MatMul sequence length can change rounding."""
    _require(isinstance(model, onnx.ModelProto), "Expected ModelProto")
    for value in (hidden_size, vocab_size, expected_states):
        _require(
            type(value) is int and value > 0, "Positive integer dimensions required"
        )
    result = copy.deepcopy(model)
    switch, token, prefill = _branches(result)
    _require(
        len(prefill.output) == len(token.output) == expected_states + 1,
        "State output count differs",
    )
    _require(
        [v.name for v in prefill.output]
        == list(switch.output)
        == [v.name for v in token.output],
        "Branch output ordering differs",
    )
    _require(
        all(v.name.startswith("present.") for v in prefill.output[1:]),
        "Unknown state outputs",
    )
    _require(
        _shape(token.output[0]) == [1, 1, vocab_size]
        and token.output[0].type.tensor_type.elem_type == T.FLOAT,
        "Token logits must already be trimmed FLOAT",
    )
    heads = [n for n in prefill.node if n.name == HEAD]
    _require(len(heads) == 1, "Expected one prefill LM head")
    head = heads[0]
    _require(
        (head.domain, head.op_type, len(head.input), len(head.output))
        == ("com.ryzenai", "MatMulNBitsBf", 6, 1),
        "Unsupported native LM-head ABI",
    )
    attrs = {a.name: h.get_attribute_value(a) for a in head.attribute}
    _require(
        attrs
        == dict(
            accuracy_level=0,
            bits=4,
            N=vocab_size,
            block_size=128,
            K=hidden_size,
            mladf_version=b"v2",
            real_K=hidden_size,
            real_N=vocab_size,
            hybrid_llm_cast_input=[],
            hybrid_llm_cast_output=[],
        ),
        "Native head attributes differ",
    )
    producers = [n for n in prefill.node if head.input[0] in n.output]
    _require(
        len(producers) == 1
        and producers[0].name == PRODUCER
        and producers[0].op_type == "SSMLP"
        and producers[0].domain == "com.ryzenai",
        "Expected exact final SSMLP producer",
    )
    _require(
        [n.name for n in prefill.node if head.input[0] in n.input] == [HEAD],
        "Final activation has other consumers",
    )
    casts = [n for n in prefill.node if head.output[0] in n.input]
    _require(len(casts) == 1, "Head output has unexpected consumers")
    cast = casts[0]
    _require(
        (cast.domain, cast.op_type, list(cast.input), list(cast.output))
        == ("com.ryzenai", "CastAvx", list(head.output), ["logits"])
        and {a.name: h.get_attribute_value(a) for a in cast.attribute}
        == {"to": T.FLOAT},
        "Expected direct FLOAT cast from head to logits",
    )
    declarations = [v for v in prefill.value_info if v.name == head.input[0]]
    _require(
        len(declarations) == 1
        and declarations[0].type.tensor_type.elem_type == T.BFLOAT16,
        "Missing unique BF16 final activation declaration",
    )
    source_shape = _shape(declarations[0])
    _require(
        len(source_shape) == 3
        and source_shape[0] == 1
        and source_shape[2] == hidden_size
        and isinstance(source_shape[1], str)
        and source_shape[1],
        "Expected dynamic [1,S,K] activation",
    )
    _require(
        _shape(prefill.output[0]) == [1, source_shape[1], vocab_size]
        and prefill.output[0].type.tensor_type.elem_type == T.FLOAT,
        "Prefill logits declaration differs",
    )
    _require(
        result.graph.output[0].name == "logits"
        and _shape(result.graph.output[0]) == [1, source_shape[1], vocab_size]
        and result.graph.output[0].type.tensor_type.elem_type == T.FLOAT,
        "Top-level unpruned logits declaration differs",
    )
    for graph in _graphs(result.graph):
        names = [
            v.name
            for v in [
                *graph.input,
                *graph.output,
                *graph.value_info,
                *graph.initializer,
            ]
        ]
        names += [name for n in graph.node for name in [n.name, *n.input, *n.output]]
        _require(
            not any(name.startswith(PREFIX) for name in names),
            "Generated name collision",
        )

    old_input = head.input[0]
    added = [
        h.make_node(
            "Constant",
            [],
            [PREFIX + "/index"],
            name=PREFIX + "/index_constant",
            value=h.make_tensor(PREFIX + "/index_value", T.INT64, [], [-1]),
        ),
        h.make_node(
            "Constant",
            [],
            [PREFIX + "/axes"],
            name=PREFIX + "/axes_constant",
            value=h.make_tensor(PREFIX + "/axes_value", T.INT64, [1], [1]),
        ),
        h.make_node(
            "Gather",
            [old_input, PREFIX + "/index"],
            [PREFIX + "/selected"],
            name=PREFIX + "/Gather",
            axis=1,
        ),
        h.make_node(
            "Unsqueeze",
            [PREFIX + "/selected", PREFIX + "/axes"],
            [PREFIX + "/activation"],
            name=PREFIX + "/Unsqueeze",
        ),
    ]
    head.input[0] = PREFIX + "/activation"
    new_nodes = []
    for node in prefill.node:
        if node.name == HEAD:
            new_nodes.extend(added)
        new_nodes.append(copy.deepcopy(node))
    del prefill.node[:]
    prefill.node.extend(new_nodes)
    changed_declarations = []
    for scope, values in (
        ("prefill.value_info", prefill.value_info),
        ("prefill.output", prefill.output),
        ("model.value_info", result.graph.value_info),
        ("model.output", result.graph.output),
    ):
        for value in values:
            if value.name not in (head.output[0], "logits"):
                continue
            before = dict(shape=_shape(value), dtype=value.type.tensor_type.elem_type)
            _require(
                len(before["shape"]) == 3
                and before["shape"][0] == 1
                and before["shape"][2] == vocab_size,
                "Unexpected logits annotation",
            )
            value.type.tensor_type.shape.dim[1].ClearField("dim_param")
            value.type.tensor_type.shape.dim[1].dim_value = 1
            value.type.tensor_type.elem_type = (
                T.FLOAT if value.name == "logits" else T.BFLOAT16
            )
            changed_declarations.append(
                dict(
                    scope=scope,
                    name=value.name,
                    before=before,
                    after=dict(
                        shape=_shape(value), dtype=value.type.tensor_type.elem_type
                    ),
                )
            )
    prefill.value_info.extend(
        [
            h.make_tensor_value_info(
                PREFIX + "/selected", T.BFLOAT16, [1, hidden_size]
            ),
            h.make_tensor_value_info(
                PREFIX + "/activation", T.BFLOAT16, [1, 1, hidden_size]
            ),
        ]
    )
    _, old_token, old_prefill = _branches(model)
    _require(
        token.SerializeToString() == old_token.SerializeToString(),
        "Token branch changed",
    )
    _require(
        prefill.output[1:] == old_prefill.output[1:]
        and result.graph.output[1:] == model.graph.output[1:],
        "State output contract changed",
    )
    _require(
        result.graph.input == model.graph.input
        and result.graph.initializer == model.graph.initializer,
        "Input/initializer contract changed",
    )
    return result, dict(
        source_header_sha256=_digest(model),
        transformed_header_sha256=_digest(result),
        native_head=HEAD,
        input_producer=PRODUCER,
        original_input=old_input,
        new_input=head.input[0],
        added_nodes=[n.name for n in added],
        declarations=changed_declarations,
        token_branch_sha256=_digest(token),
        state_outputs_unchanged=expected_states,
        native_head_attributes_unchanged=True,
        native_head_weights_unchanged=True,
        top_level_logits_trimmed_for_oga=True,
        original_input_nonempty_required=True,
        numerical_equivalence_not_assumed=True,
        npu_runtime_verified=False,
    )
