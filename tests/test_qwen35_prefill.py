"""Owned header fixtures and a standard-op topology interpreter; no runtime."""

from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
from onnx import TensorProto as T, helper as h  # noqa: E402
from benchmarks import qwen35_prefill as prefill  # noqa: E402


def native_node(name, inputs, outputs):
    node = h.make_node(
        "LinearAttention",
        inputs,
        outputs,
        name=name,
        domain="com.ryzenai",
        q_num_heads=16,
        kv_num_heads=32,
        update_rule="gated_delta",
        chunk_size=64,
        ryzenai_absorb_input_cast_indices=[4],
        ryzenai_absorb_input_cast_dtypes=[10],
        ryzenai_absorb_input_order=[0],
        seq_length=4096,
        q_seq=4096,
        k_heads=16,
        v_heads=32,
        k_head_dim=128,
        v_head_dim=128,
        scale=1.0,
    )
    for name in ("hybrid_llm_cast_input", "hybrid_llm_cast_output"):
        node.attribute.append(
            h.make_attribute(name, [], attr_type=onnx.AttributeProto.INTS)
        )
    return node


def fixture_model(count=24):
    inputs = [
        h.make_tensor_value_info(name, T.BFLOAT16, shape)
        for name, shape in (
            ("query", [1, "sequence_length", 2048]),
            ("key", [1, "sequence_length", 2048]),
            ("value", [1, "sequence_length", 4096]),
            ("gate", [1, "sequence_length", 32]),
            ("beta", [1, "sequence_length", 32]),
        )
    ]
    outputs, nodes = [], []
    for i in range(count):
        inputs.append(
            h.make_tensor_value_info(
                f"past.{i}.recurrent_state", T.BFLOAT16, [1, 32, 128, 128]
            )
        )
        outputs.extend(
            [
                h.make_tensor_value_info(
                    f"attention_{i}", T.BFLOAT16, [1, "sequence_length", 4096]
                ),
                h.make_tensor_value_info(
                    f"present.{i}.recurrent_state", T.BFLOAT16, [1, 32, 128, 128]
                ),
            ]
        )
        nodes.append(
            native_node(
                f"layer{i}/LinearAttention",
                ["query", "key", "value", f"past.{i}.recurrent_state", "gate", "beta"],
                [f"attention_{i}", f"present.{i}.recurrent_state"],
            )
        )
    # Other forty state tensors are header-only; no large arrays are allocated.
    for i in range(40):
        inputs.append(
            h.make_tensor_value_info(f"past.other_state_{i}", T.FLOAT16, [1, 2, 3])
        )
        outputs.append(
            h.make_tensor_value_info(f"present.other_state_{i}", T.FLOAT16, [1, 2, 3])
        )
        nodes.append(
            h.make_node(
                "Identity",
                [inputs[-1].name],
                [outputs[-1].name],
                name=f"retain_other_state_{i}",
            )
        )
    weight = T(
        name="unloaded_owned_weight",
        data_type=T.FLOAT,
        dims=[2],
        data_location=T.EXTERNAL,
    )
    for key, value in (
        ("location", "does-not-exist.bin"),
        ("offset", "0"),
        ("length", "8"),
    ):
        weight.external_data.add(key=key, value=value)
    model = h.make_model(
        h.make_graph(nodes, "header", inputs, outputs, [weight]),
        opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.ryzenai", 1)],
    )
    model.metadata_props.add(key="retain", value="original metadata")
    return model


def test_preserves_all_original_content_and_64_states_without_reading_external_data():
    source = fixture_model()
    before = source.SerializeToString()
    result, report = prefill.transform_prefill_linear_attention(source)
    assert source.SerializeToString() == before
    assert report["changed_nodes"] == 24
    assert len([v for v in result.graph.output if v.name.startswith("present.")]) == 64
    assert list(result.graph.input) == list(source.graph.input)
    assert list(result.graph.output) == list(source.graph.output)
    assert list(result.graph.initializer) == list(source.graph.initializer)
    assert list(result.graph.node[-40:]) == list(source.graph.node[-40:])
    assert report["preserved_native_operator_names_and_types"]
    assert result.metadata_props == source.metadata_props
    for i, contract in enumerate(report["nodes"]):
        body = next(
            a.g for a in result.graph.node[i * 7 + 4].attribute if a.name == "body"
        )
        native = next(n for n in body.node if n.domain == "com.ryzenai")
        assert native.name == source.graph.node[i].name
        assert list(native.attribute) == list(source.graph.node[i].attribute)
        assert contract["native_name"] == native.name
    with pytest.raises(ValueError, match="top-level"):
        prefill.transform_prefill_linear_attention(result)


def test_unrelated_legacy_value_info_is_preserved_but_la_conflicts_are_rejected():
    model = fixture_model()
    # The measured eager source has an unrelated logits FLOAT/FLOAT16 metadata
    # mismatch. A local LA replacement must neither interpret nor rewrite it.
    model.graph.value_info.append(
        h.make_tensor_value_info("present.other_state_0", T.FLOAT, [1, 2, 3])
    )
    transformed, _ = prefill.transform_prefill_linear_attention(model)
    assert transformed.graph.value_info == model.graph.value_info
    model.graph.value_info.append(
        h.make_tensor_value_info("query", T.FLOAT16, [1, "sequence_length", 2048])
    )
    with pytest.raises(ValueError, match="Conflicting tensor declaration"):
        prefill.transform_prefill_linear_attention(model)


@pytest.mark.parametrize(
    "mutation",
    [
        "input_dtype",
        "output_dtype",
        "state_shape",
        "width",
        "batch",
        "sequence",
        "different_sequence",
        "arity",
        "output_arity",
        "missing_declaration",
        "attribute",
        "attribute_type",
        "collision",
        "node_collision",
        "duplicate_native_name",
        "count",
        "opset",
    ],
)
def test_rejects_unmeasured_or_ambiguous_contract(mutation):
    model = fixture_model(23 if mutation == "count" else 24)
    if mutation == "input_dtype":
        model.graph.input[0].type.tensor_type.elem_type = T.FLOAT16
    elif mutation == "output_dtype":
        model.graph.output[1].type.tensor_type.elem_type = T.FLOAT16
    elif mutation == "state_shape":
        model.graph.input[5].type.tensor_type.shape.dim[2].dim_value = 127
    elif mutation == "width":
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 4096
    elif mutation == "batch":
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
    elif mutation == "sequence":
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 64
    elif mutation == "different_sequence":
        model.graph.input[1].type.tensor_type.shape.dim[1].dim_param = "other_sequence"
    elif mutation == "arity":
        model.graph.node[0].input.append("extra")
    elif mutation == "output_arity":
        model.graph.node[0].output.pop()
    elif mutation == "missing_declaration":
        model.graph.input.remove(model.graph.input[0])
    elif mutation == "attribute":
        next(a for a in model.graph.node[0].attribute if a.name == "scale").f = 0.5
    elif mutation == "attribute_type":
        attr = next(a for a in model.graph.node[0].attribute if a.name == "scale")
        attr.ClearField("f")
        attr.type = onnx.AttributeProto.INT
        attr.i = 1
    elif mutation in {"collision", "node_collision"}:
        name = "layer0/LinearAttention/TokenLoop/step_0"
        if mutation == "collision":
            model.graph.value_info.append(
                h.make_tensor_value_info(name, T.BFLOAT16, [1, 1, 2048])
            )
        else:
            model.graph.node[-1].name = name
    elif mutation == "duplicate_native_name":
        model.graph.node[1].name = model.graph.node[0].name
    elif mutation == "opset":
        model.opset_import[0].version = 13
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        prefill.transform_prefill_linear_attention(model)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("sequence", [1, 17, 64])
def test_dynamic_loop_slices_and_carries_state_for_actual_sequence(sequence):
    source = fixture_model()
    result, _ = prefill.transform_prefill_linear_attention(source)
    raw = {
        name: np.broadcast_to(
            np.arange(sequence, dtype=np.uint16)[None, :, None] + 100 * j,
            (1, sequence, width),
        ).copy()
        for j, (name, width) in enumerate(
            [
                ("query", 2048),
                ("key", 2048),
                ("value", 4096),
                ("gate", 32),
                ("beta", 32),
            ]
        )
    }
    raw["past.0.recurrent_state"] = np.zeros((1, 32, 128, 128), np.uint16)
    saved = {k: v.copy() for k, v in raw.items()}
    calls = []

    def execute(nodes, values):
        for node in nodes:
            attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
            args = [values[n] for n in node.input]
            if node.op_type == "Constant":
                outputs = [onnx.numpy_helper.to_array(attrs["value"])]
            elif node.op_type == "Shape":
                outputs = [np.array(args[0].shape, np.int64)]
            elif node.op_type == "Gather":
                outputs = [np.take(args[0], args[1], axis=attrs["axis"])]
            elif node.op_type == "Unsqueeze":
                outputs = [np.expand_dims(args[0], tuple(args[1].tolist()))]
            elif node.op_type == "Identity":
                outputs = args
            elif node.op_type == "Reshape":
                outputs = [args[0].reshape(args[1])]
            elif node.op_type == "LinearAttention":
                t = len(calls)
                for j, index in enumerate((0, 1, 2, 4, 5)):
                    assert args[index].shape[1] == 1 and np.all(
                        args[index] == t + 100 * j
                    )
                assert np.all(args[3] == t)
                calls.append(t)
                outputs = [
                    args[2].copy(),
                    args[3] + 1,
                ]  # counting fake, not a numerical LA reference
            elif node.op_type == "Loop":
                body = attrs["body"]
                state = args[2]
                scans = []
                assert int(args[0]) == sequence and bool(args[1])
                for t in range(int(args[0])):
                    local = {
                        **values,
                        body.input[0].name: np.array(t, np.int64),
                        body.input[1].name: np.array(True),
                        body.input[2].name: state,
                    }
                    execute(body.node, local)
                    assert local[body.output[0].name]
                    state = local[body.output[1].name]
                    scans.append(local[body.output[2].name])
                outputs = [state, np.stack(scans)]
            else:
                raise AssertionError(node.op_type)
            values.update(zip(node.output, outputs, strict=True))

    env = dict(raw)
    execute(result.graph.node[:7], env)
    assert calls == list(range(sequence))
    assert np.all(env["present.0.recurrent_state"] == sequence)
    assert np.array_equal(env["attention_0"], raw["value"])
    assert all(np.array_equal(raw[k], v) for k, v in saved.items())
