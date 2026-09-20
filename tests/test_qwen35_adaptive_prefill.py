"""Owned portable ONNX fixtures; only standard operators run on the CPU."""

from __future__ import annotations
import copy
import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
from onnx import TensorProto as T, helper as h  # noqa: E402
from benchmarks import qwen35_adaptive_prefill as adaptive  # noqa: E402
from benchmarks.qwen35_prefill import _ATTRIBUTES, _graphs  # noqa: E402


def original_model():
    names = ["q", "k", "v", "state", "g", "beta"]
    shapes = [
        [1, "S", 2048],
        [1, "S", 2048],
        [1, "S", 4096],
        [1, 32, 128, 128],
        [1, "S", 32],
        [1, "S", 32],
    ]
    node = h.make_node(
        "LinearAttention",
        names,
        ["attention_out", "state_out"],
        name="layer14/LinearAttention",
        domain="com.ryzenai",
    )
    for key, value in _ATTRIBUTES.items():
        node.attribute.append(
            h.make_attribute(key, value, attr_type=onnx.AttributeProto.INTS)
            if isinstance(value, list)
            else h.make_attribute(key, value)
        )
    model = h.make_model(
        h.make_graph(
            [node],
            "owned_native_contract",
            [h.make_tensor_value_info(n, T.BFLOAT16, s) for n, s in zip(names, shapes)],
            [
                h.make_tensor_value_info("attention_out", T.BFLOAT16, [1, "S", 4096]),
                h.make_tensor_value_info("state_out", T.BFLOAT16, [1, 32, 128, 128]),
            ],
        ),
        opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.ryzenai", 1)],
        ir_version=10,
    )
    return model


def constant(name, dtype, shape, values):
    return h.make_node(
        "Constant",
        [],
        [name],
        name=name + "/constant",
        value=h.make_tensor(name, dtype, shape, values),
    )


def mock_nodes(native):
    """All five sequence roles affect a recurrent prefix; state is not reset."""
    p = native.name + "/OwnedCPUMock/"

    def n(text):
        return p + text

    nodes = [
        constant(n("zero"), T.INT64, [], [0]),
        constant(n("one"), T.INT64, [], [1]),
        constant(n("last_axis"), T.INT64, [1], [2]),
        constant(n("flat_shape"), T.INT64, [1], [-1]),
        constant(n("sum_axes"), T.INT64, [2], [0, 1]),
    ]
    terms = []
    for index, weight in zip([0, 1, 2, 4, 5], [1, 2, 3, 5, 7]):
        nodes.extend(
            [
                h.make_node(
                    "Gather",
                    [native.input[index], n("zero")],
                    [n(f"value{index}")],
                    axis=2,
                    name=n(f"gather{index}"),
                ),
                constant(n(f"weight{index}"), T.FLOAT, [], [weight]),
                h.make_node(
                    "Mul",
                    [n(f"value{index}"), n(f"weight{index}")],
                    [n(f"term{index}")],
                    name=n(f"mul{index}"),
                ),
            ]
        )
        terms.append(n(f"term{index}"))
    for i in range(1, len(terms)):
        nodes.append(
            h.make_node(
                "Add",
                [terms[0] if i == 1 else n(f"sum{i - 1}"), terms[i]],
                [n(f"sum{i}")],
                name=n(f"sum_node{i}"),
            )
        )
    increments = n("sum4")
    nodes.extend(
        [
            h.make_node(
                "Reshape",
                [native.input[3], n("flat_shape")],
                [n("flat_state")],
                name=n("flatten_state"),
            ),
            h.make_node(
                "Gather",
                [n("flat_state"), n("zero")],
                [n("initial")],
                axis=0,
                name=n("initial_state"),
            ),
            h.make_node(
                "CumSum", [increments, n("one")], [n("prefix")], name=n("prefix_node")
            ),
            h.make_node(
                "Add",
                [n("prefix"), n("initial")],
                [n("running")],
                name=n("running_node"),
            ),
            h.make_node(
                "Unsqueeze",
                [n("running"), n("last_axis")],
                [n("running_3d")],
                name=n("running_3d_node"),
            ),
            h.make_node(
                "Add",
                [native.input[2], n("running_3d")],
                [native.output[0]],
                name=n("attention"),
            ),
            h.make_node(
                "ReduceSum",
                [increments, n("sum_axes")],
                [n("total")],
                keepdims=0,
                name=n("total_node"),
            ),
            h.make_node(
                "Add",
                [native.input[3], n("total")],
                [native.output[1]],
                name=n("state"),
            ),
        ]
    )
    return nodes


def fixture(sequence, mode):
    arrays = {
        name: np.zeros([1, sequence, width], np.float32)
        for name, width in [
            ("q", 2048),
            ("k", 2048),
            ("v", 4096),
            ("g", 32),
            ("beta", 32),
        ]
    }
    positions = np.arange(sequence, dtype=np.float32)
    arrays["q"][0, :, 0] = positions + 1
    arrays["k"][0, :, 0] = 2 * positions + 1
    arrays["v"][0, :, :] = (positions + 3)[:, None]
    arrays["beta"][0, :, :] = ((positions % 3) + 1)[:, None]
    if mode == "small":
        arrays["g"][:] = -0.5
    elif mode == "strong":
        arrays["g"][:] = -128
    elif mode == "mixed":
        arrays["g"][:] = -0.5
        arrays["g"][0, :, 5] = -4
        arrays["g"][0, 8::13, 5] = -128
        arrays["g"][0, 12::17, 31] = 16  # absolute value and non-leading head
    elif mode != "zero":
        raise ValueError(mode)
    arrays["state"] = np.full([1, 32, 128, 128], 1.5, np.float32)
    return arrays


def reference(inputs, budget, max_segment):
    increments = sum(
        inputs[name][..., 0] * weight
        for name, weight in [("q", 1), ("k", 2), ("v", 3), ("g", 5), ("beta", 7)]
    )
    attention = inputs["v"] + (np.cumsum(increments, axis=1) + 1.5)[..., None]
    state = inputs["state"] + increments.sum()
    gates = inputs["g"]
    start, ends = 0, []
    while start < gates.shape[1]:
        sums = np.zeros(32, np.float32)
        end = start
        for index in range(start, min(start + max_segment, gates.shape[1])):
            candidate = sums + np.abs(gates[0, index])
            if np.max(candidate) > budget:
                break
            sums = candidate
            end = index + 1
        end = max(end, start + 1)
        ends.append(end)
        start = end
    return attention, state, np.array(ends, dtype=np.int64)


def cpu_session(model):
    ort = pytest.importorskip("onnxruntime")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    assert session.get_providers() == ["CPUExecutionProvider"]
    return session


def float_mock(model):
    model = copy.deepcopy(model)
    for graph in _graphs(model.graph):
        for value in [*graph.input, *graph.output, *graph.value_info]:
            if value.type.tensor_type.elem_type == T.BFLOAT16:
                value.type.tensor_type.elem_type = T.FLOAT
        nodes = []
        for node in graph.node:
            if node.op_type == "Cast":
                target = next(a for a in node.attribute if a.name == "to")
                if target.i == T.BFLOAT16:
                    target.i = T.FLOAT
            nodes.extend(mock_nodes(node) if node.domain == "com.ryzenai" else [node])
        del graph.node[:]
        graph.node.extend(nodes)
    add_trace(model)
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize("length", [1, 2, 7, 8, 63, 64, 96])
@pytest.mark.parametrize("mode", ["zero", "small", "strong", "mixed"])
@pytest.mark.parametrize("budget", [32.0, 64.0])
def test_segment_order_all_five_input_roles_and_nonzero_state(length, mode, budget):
    model, _ = adaptive.transform_prefill_adaptive_attention(
        original_model(), expected_count=1, log_budget=budget
    )
    inputs = fixture(length, mode)
    for actual, expected in zip(
        cpu_session(float_mock(model)).run(None, inputs),
        reference(inputs, budget, 64),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def bf16_bits(value):
    bits = np.asarray(value, np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def add_trace(model):
    loop = next(n for n in model.graph.node if n.op_type == "Loop")
    body = next(a.g for a in loop.attribute if a.name == "body")
    body.node.append(
        h.make_node(
            "Identity", [body.output[2].name], ["trace_end"], name="trace_end_node"
        )
    )
    body.output.append(h.make_tensor_value_info("trace_end", T.INT64, []))
    loop.output.append("trace_ends")
    model.graph.output.append(
        h.make_tensor_value_info("trace_ends", T.INT64, ["segments"])
    )


@pytest.mark.parametrize(
    "length,mode", [(64, "mixed"), (64, "strong"), (1024, "mixed"), (1024, "strong")]
)
def test_real_bf16_boundaries_and_state_iobinding(length, mode):
    ort = pytest.importorskip("onnxruntime")
    if not hasattr(ort.OrtValue, "ortvalue_from_numpy_with_onnx_type"):
        pytest.skip("BF16 OrtValue binding requires a recent ORT build")
    model, _ = adaptive.transform_prefill_adaptive_attention(
        original_model(), expected_count=1, log_budget=64
    )
    for graph in _graphs(model.graph):
        nodes = []
        for native in graph.node:
            if native.domain != "com.ryzenai":
                nodes.append(native)
                continue
            p = native.name + "/Mock/"
            nodes.extend(
                [
                    h.make_node(
                        "Identity",
                        [native.input[2]],
                        [native.output[0]],
                        name=p + "attention",
                    ),
                    h.make_node(
                        "Cast",
                        [native.input[3]],
                        [p + "state_float"],
                        to=T.FLOAT,
                        name=p + "cast_float",
                    ),
                    constant(p + "one", T.FLOAT, [], [1]),
                    h.make_node(
                        "Add",
                        [p + "state_float", p + "one"],
                        [p + "next_float"],
                        name=p + "add",
                    ),
                    h.make_node(
                        "Cast",
                        [p + "next_float"],
                        [native.output[1]],
                        to=T.BFLOAT16,
                        name=p + "cast_bf16",
                    ),
                ]
            )
        del graph.node[:]
        graph.node.extend(nodes)
    add_trace(model)
    onnx.checker.check_model(model)
    session = cpu_session(model)
    raw = {name: bf16_bits(value) for name, value in fixture(length, mode).items()}
    decoded = {
        name: (value.astype(np.uint32) << 16).view(np.float32)
        for name, value in raw.items()
    }
    ends = reference(decoded, 64, 64)[2]
    inputs = {
        name: ort.OrtValue.ortvalue_from_numpy_with_onnx_type(value, T.BFLOAT16)
        for name, value in raw.items()
    }
    raw_results = [np.zeros_like(raw["v"]), np.zeros_like(raw["state"])]
    result_ends = np.zeros(len(ends), np.int64)
    outputs = [
        ort.OrtValue.ortvalue_from_numpy_with_onnx_type(value, T.BFLOAT16)
        for value in raw_results
    ]
    outputs.append(ort.OrtValue.ortvalue_from_numpy(result_ends))
    binding = session.io_binding()
    for name, value in inputs.items():
        binding.bind_ortvalue_input(name, value)
    for name, value in zip(["attention_out", "state_out", "trace_ends"], outputs):
        binding.bind_ortvalue_output(name, value)
    session.run_with_iobinding(binding)
    np.testing.assert_array_equal(raw_results[0], raw["v"])
    np.testing.assert_array_equal(result_ends, ends)
    expected_state = np.float32(1.5)
    for _ in ends:
        expected_bits = bf16_bits(expected_state + np.float32(1))
        expected_state = (np.asarray([expected_bits], np.uint32) << 16).view(
            np.float32
        )[0]
    np.testing.assert_array_equal(
        raw_results[1], np.full_like(raw_results[1], expected_bits)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_count": 0},
        {"expected_count": True},
        {"expected_count": 1.5},
        {"log_budget": 0},
        {"log_budget": -1},
        {"log_budget": 65},
        {"log_budget": None},
        {"log_budget": "32"},
        {"log_budget": True},
        {"log_budget": float("nan")},
        {"log_budget": float("inf")},
        {"max_segment": 0},
        {"max_segment": 65},
        {"max_segment": True},
        {"max_segment": 1.5},
    ],
)
def test_parameter_guards(kwargs):
    arguments = {"expected_count": 1, **kwargs}
    with pytest.raises(ValueError):
        adaptive.transform_prefill_adaptive_attention(original_model(), **arguments)


def test_model_type_and_static_empty_sequence_rejected():
    with pytest.raises(TypeError):
        adaptive.transform_prefill_adaptive_attention(None)
    model = original_model()
    dim = model.graph.input[0].type.tensor_type.shape.dim[1]
    dim.ClearField("dim_param")
    dim.dim_value = 0
    with pytest.raises(ValueError, match="dynamic sequence"):
        adaptive.transform_prefill_adaptive_attention(model, expected_count=1)


@pytest.mark.parametrize("kind", ["node", "tensor", "initializer"])
def test_generated_name_collision_is_rejected(kind):
    model = original_model()
    prefix = "layer14/LinearAttention/AdaptiveLoop/"
    if kind == "node":
        model.graph.node.append(
            h.make_node(
                "Identity", ["beta"], ["owned_output"], name=prefix + "window_add"
            )
        )
    elif kind == "tensor":
        model.graph.node.append(
            h.make_node(
                "Identity",
                ["beta"],
                [prefix + "segment_attention_float"],
                name="owned_identity",
            )
        )
    else:
        model.graph.initializer.append(
            h.make_tensor(prefix + "budget", T.FLOAT, [], [1])
        )
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="collides"):
        adaptive.transform_prefill_adaptive_attention(model, expected_count=1)
    assert model.SerializeToString() == before


def test_duplicate_native_node_names_rejected():
    model = original_model()
    model.graph.node.append(model.graph.node[0])
    with pytest.raises(ValueError, match="Duplicate native node"):
        adaptive.transform_prefill_adaptive_attention(model, expected_count=2)


def test_native_output_aliasing_generated_value_rejected():
    model = original_model()
    collision = "layer14/LinearAttention/AdaptiveLoop/segment_attention_float"
    model.graph.node[0].output[0] = collision
    model.graph.output[0].name = collision
    with pytest.raises(ValueError, match="Duplicate generated tensor"):
        adaptive.transform_prefill_adaptive_attention(model, expected_count=1)


def test_transform_does_not_read_external_data_or_run_checker(monkeypatch):
    model = original_model()
    tensor = onnx.TensorProto()
    tensor.name = "owned_external"
    tensor.data_type = T.FLOAT
    tensor.dims.extend([1])
    tensor.data_location = T.EXTERNAL
    entry = tensor.external_data.add()
    entry.key = "location"
    entry.value = "missing_external_tensor_for_owned_no_io_test.data"
    model.graph.initializer.append(tensor)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Header transformation must not read external data or run the whole-model checker"
        )

    monkeypatch.setattr(onnx.checker, "check_model", forbidden)
    monkeypatch.setattr(
        onnx.external_data_helper, "load_external_data_for_model", forbidden
    )
    result, _ = adaptive.transform_prefill_adaptive_attention(model, expected_count=1)
    assert result.graph.initializer == model.graph.initializer


def test_all24_native_contracts_and_64_state_outputs_are_preserved():
    model = original_model()
    original = copy.deepcopy(model.graph.node[0])
    state_in = copy.deepcopy(model.graph.input[3])
    state_out = copy.deepcopy(model.graph.output[1])
    attention = copy.deepcopy(model.graph.output[0])
    for index in range(1, 24):
        node = copy.deepcopy(original)
        node.name = f"layer{index}/LinearAttention"
        if index == 14:
            node.name = "layer0/LinearAttention"
        node.input[3] = f"past_{index}"
        node.output[:] = [f"attention_{index}", f"state_{index}"]
        inp, out, attn = (
            copy.deepcopy(state_in),
            copy.deepcopy(state_out),
            copy.deepcopy(attention),
        )
        inp.name, out.name, attn.name = node.input[3], node.output[1], node.output[0]
        model.graph.input.append(inp)
        model.graph.output.extend([attn, out])
        model.graph.node.append(node)
    for index in range(40):
        inp, out = f"other_past_{index}", f"other_present_{index}"
        model.graph.input.append(h.make_tensor_value_info(inp, T.FLOAT16, [1, 2, 3]))
        model.graph.output.append(h.make_tensor_value_info(out, T.FLOAT16, [1, 2, 3]))
        model.graph.node.append(
            h.make_node("Identity", [inp], [out], name=f"other_state_{index}")
        )
    model.metadata_props.add(key="keep", value="owned metadata")
    before = model.SerializeToString()
    result, info = adaptive.transform_prefill_adaptive_attention(model)
    assert info["native_nodes"] == 24 and info["npu_runtime_verified"] is False
    assert info["absolute_log_gate_budget"] == 64 and info["max_segment"] == 64
    assert model.SerializeToString() == before
    assert (
        result.graph.input == model.graph.input
        and result.graph.output == model.graph.output
    )
    assert len(result.graph.output) == 24 * 2 + 40  # 24 attention and 64 state tensors
    assert result.graph.node[-40:] == model.graph.node[-40:]
    assert result.metadata_props == model.metadata_props
    assert (
        result.opset_import == model.opset_import
        and result.ir_version == model.ir_version
    )
    native = {
        node.name: node
        for graph in _graphs(result.graph)
        for node in graph.node
        if node.domain == "com.ryzenai"
    }
    assert len(native) == 24
    for source in model.graph.node[:24]:
        assert native[source.name].attribute == source.attribute
        assert native[source.name].op_type == source.op_type
    onnx.checker.check_model(result)


@pytest.mark.parametrize("max_segment", [1, 8, 64])
def test_segment_cap_is_independent_of_gate_budget(max_segment):
    model, _ = adaptive.transform_prefill_adaptive_attention(
        original_model(), expected_count=1, max_segment=max_segment
    )
    inputs = fixture(96, "mixed")
    for actual, expected in zip(
        cpu_session(float_mock(model)).run(None, inputs),
        reference(inputs, 64, max_segment),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("missing", ["benchmarks", "an_unavailable_nested_dependency"])
def test_script_directory_import_fallback_only_handles_missing_package(
    monkeypatch, missing
):
    import builtins
    import runpy
    from pathlib import Path

    original_import = builtins.__import__

    def limited_import(name, *args, **kwargs):
        if name == "benchmarks.qwen35_prefill":
            raise ModuleNotFoundError("owned import failure", name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.syspath_prepend(str(Path(adaptive.__file__).parent))
    monkeypatch.setattr(builtins, "__import__", limited_import)
    if missing != "benchmarks":
        with pytest.raises(ModuleNotFoundError) as error:
            runpy.run_path(adaptive.__file__)
        assert error.value.name == missing
    else:
        namespace = runpy.run_path(adaptive.__file__)
        transformed, _ = namespace["transform_prefill_adaptive_attention"](
            original_model(), expected_count=1
        )
        expected, _ = adaptive.transform_prefill_adaptive_attention(
            original_model(), expected_count=1
        )
        assert transformed.SerializeToString() == expected.SerializeToString()
