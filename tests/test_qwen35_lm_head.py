"""Owned portable BF16 CPU fixtures for prefill-only LM-head pruning."""

from __future__ import annotations
import copy
import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
from onnx import TensorProto as T, helper as h  # noqa: E402
from benchmarks import qwen35_lm_head as prune  # noqa: E402
from benchmarks import qwen35_package as package  # noqa: E402


def owned_model():
    def vi(name, dtype, dims):
        return h.make_tensor_value_info(name, dtype, dims)

    attrs = dict(
        accuracy_level=0,
        bits=4,
        N=8,
        block_size=128,
        K=4,
        mladf_version="v2",
        real_K=4,
        real_N=8,
    )
    head = h.make_node(
        "MatMulNBitsBf",
        ["activation", "W", "empty", "empty", "empty", "empty"],
        ["logits_bf16"],
        name=prune.HEAD,
        domain="com.ryzenai",
        **attrs,
    )
    for name in ("hybrid_llm_cast_input", "hybrid_llm_cast_output"):
        head.attribute.append(
            h.make_attribute(name, [], attr_type=onnx.AttributeProto.INTS)
        )
    state_nodes = [
        h.make_node("Cast", ["x"], ["x_float"], name="state_cast", to=T.FLOAT),
        h.make_node(
            "ReduceSum", ["x_float", "axis1"], ["sum"], name="all_tokens", keepdims=0
        ),
        h.make_node("Add", ["sum", "past"], ["present.total"], name="next_total"),
        h.make_node("Identity", ["past"], ["present.other"], name="other_state"),
    ]
    output = [
        vi("logits", T.FLOAT, [1, "S", 8]),
        vi("present.total", T.FLOAT, [1, 4]),
        vi("present.other", T.FLOAT, [1, 4]),
    ]
    prefill = h.make_graph(
        [
            h.make_node(
                "SSMLP",
                ["x"],
                ["activation"],
                name=prune.PRODUCER,
                domain="com.ryzenai",
            ),
            *copy.deepcopy(state_nodes),
            head,
            h.make_node(
                "CastAvx",
                ["logits_bf16"],
                ["logits"],
                name="final_cast",
                domain="com.ryzenai",
                to=T.FLOAT,
            ),
        ],
        "owned_prefill",
        [],
        output,
        value_info=[
            vi("activation", T.BFLOAT16, [1, "S", 4]),
            vi("logits_bf16", T.BFLOAT16, [1, "S", 8]),
        ],
    )
    token_output = copy.deepcopy(output)
    token_output[0].type.tensor_type.shape.dim[1].ClearField("dim_param")
    token_output[0].type.tensor_type.shape.dim[1].dim_value = 1
    token = h.make_graph(
        [
            *copy.deepcopy(state_nodes),
            h.make_node(
                "DynamicDispatch",
                ["x"],
                ["token_bf16"],
                name="owned_DD",
                domain="com.ryzenai",
                model_type=9,
            ),
            h.make_node(
                "CastAvx",
                ["token_bf16"],
                ["logits"],
                name="token_cast",
                domain="com.ryzenai",
                to=T.FLOAT,
            ),
        ],
        "owned_token",
        [],
        token_output,
    )
    model = h.make_model(
        h.make_graph(
            [
                h.make_node(
                    "If",
                    ["is_token"],
                    [v.name for v in output],
                    then_branch=token,
                    else_branch=prefill,
                )
            ],
            "owned_combined",
            [
                vi("x", T.BFLOAT16, [1, "S", 4]),
                vi("past", T.FLOAT, [1, 4]),
                vi("is_token", T.BOOL, []),
            ],
            output,
            initializer=[
                h.make_tensor("W", T.FLOAT, [4, 8], list(range(32))),
                h.make_tensor("axis1", T.INT64, [1], [1]),
                h.make_tensor("empty", T.FLOAT, [0], []),
            ],
        ),
        opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.ryzenai", 1)],
        ir_version=10,
    )
    return model


def convert(model):
    return prune.transform_prefill_lm_head(
        model, hidden_size=4, vocab_size=8, expected_states=2
    )


def standard_mock(model):
    model = copy.deepcopy(model)
    for graph in prune._graphs(model.graph):
        nodes = []
        for node in graph.node:
            if node.domain != "com.ryzenai":
                nodes.append(node)
            elif node.op_type == "SSMLP":
                nodes.append(
                    h.make_node(
                        "Identity", [node.input[0]], list(node.output), name=node.name
                    )
                )
            elif node.op_type == "CastAvx":
                nodes.append(
                    h.make_node(
                        "Cast",
                        list(node.input),
                        list(node.output),
                        name=node.name,
                        to=T.FLOAT,
                    )
                )
            elif node.op_type in ("MatMulNBitsBf", "DynamicDispatch"):
                prefix = node.name + "/OwnedMock/"
                nodes.extend(
                    [
                        h.make_node(
                            "Cast",
                            [node.input[0]],
                            [prefix + "input"],
                            name=prefix + "cast",
                            to=T.FLOAT,
                        ),
                        h.make_node(
                            "MatMul",
                            [prefix + "input", "W"],
                            [prefix + "float"],
                            name=prefix + "matmul",
                        ),
                        h.make_node(
                            "Cast",
                            [prefix + "float"],
                            list(node.output),
                            name=prefix + "result",
                            to=T.BFLOAT16,
                        ),
                    ]
                )
            else:
                raise AssertionError(node.op_type)
        del graph.node[:]
        graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def run_cpu(model, seq):
    ort = pytest.importorskip("onnxruntime")
    if not hasattr(ort.OrtValue, "ortvalue_from_numpy_with_onnx_type"):
        pytest.skip("BF16 OrtValue binding requires a recent ORT build")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        standard_mock(model).SerializeToString(),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    assert session.get_providers() == ["CPUExecutionProvider"]
    values = (np.arange(seq * 4).reshape(1, seq, 4) % 7).astype(np.float32)
    raw = (values.view(np.uint32) >> 16).astype(np.uint16)
    binding = session.io_binding()
    binding.bind_ortvalue_input(
        "x", ort.OrtValue.ortvalue_from_numpy_with_onnx_type(raw, T.BFLOAT16)
    )
    binding.bind_cpu_input("past", np.full((1, 4), 1.5, np.float32))
    binding.bind_cpu_input("is_token", np.array(seq == 1))
    for name in ("logits", "present.total", "present.other"):
        binding.bind_output(name, "cpu")
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()


@pytest.mark.parametrize("seq", [1, 2, 7, 64, 1024])
def test_last_logits_match_and_all_sequence_state_updates_survive(seq):
    model = owned_model()
    original_bytes = model.SerializeToString()
    pruned, info = convert(model)
    before = run_cpu(model, seq)
    after = run_cpu(pruned, seq)
    assert model.SerializeToString() == original_bytes
    assert after[0].shape == (1, 1, 8)
    np.testing.assert_array_equal(after[0], before[0][:, -1:, :])
    np.testing.assert_array_equal(after[1], before[1])
    np.testing.assert_array_equal(after[2], before[2])
    assert info["top_level_logits_trimmed_for_oga"] and not info["npu_runtime_verified"]


@pytest.mark.parametrize(
    "mutation",
    [
        "head_attr",
        "producer",
        "other_consumer",
        "head_cast",
        "token_shape",
        "outer_shape",
        "collision",
        "input_dtype",
        "state_count",
        "duplicate_head",
    ],
)
def test_rejects_changed_contract_without_modifying_input(mutation):
    model = owned_model()
    _, token, prefill = prune._branches(model)
    head = next(n for n in prefill.node if n.name == prune.HEAD)
    if mutation == "head_attr":
        next(a for a in head.attribute if a.name == "K").i = 2
    elif mutation == "producer":
        prefill.node[0].name = "different"
    elif mutation == "other_consumer":
        prefill.node.append(h.make_node("Identity", [head.input[0]], ["extra"]))
    elif mutation == "head_cast":
        prefill.node[-1].attribute[0].i = T.FLOAT16
    elif mutation == "token_shape":
        token.output[0].type.tensor_type.shape.dim[1].dim_value = 2
    elif mutation == "outer_shape":
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1
    elif mutation == "collision":
        prefill.node.append(
            h.make_node("Identity", [head.input[0]], [prune.PREFIX + "/activation"])
        )
    elif mutation == "input_dtype":
        prefill.value_info[0].type.tensor_type.elem_type = T.FLOAT16
    elif mutation == "state_count":
        prefill.output.pop()
    elif mutation == "duplicate_head":
        prefill.node.append(copy.deepcopy(head))
    before = model.SerializeToString()
    with pytest.raises(ValueError):
        convert(model)
    assert model.SerializeToString() == before


def test_no_external_tensor_load_or_checker_during_transform(monkeypatch):
    model = owned_model()
    tensor = T(
        name="owned_unloaded", data_type=T.FLOAT, dims=[1], data_location=T.EXTERNAL
    )
    tensor.external_data.add(key="location", value="missing-owned-external.bin")
    model.graph.initializer.append(tensor)

    def forbidden(*args, **kwargs):
        pytest.fail("Transform must only copy the supplied header")

    monkeypatch.setattr(onnx, "load", forbidden)
    monkeypatch.setattr(onnx.checker, "check_model", forbidden)
    result, _ = convert(model)
    assert result.graph.initializer == model.graph.initializer


def split_fixture(tmp_path):
    combined = owned_model()
    _, token, prefill = prune._branches(combined)
    paths = []
    for name, graph, token_mode in (
        ("prefill", prefill, False),
        ("token", token, True),
    ):
        graph = copy.deepcopy(graph)
        graph.input.extend(combined.graph.input)
        del graph.input[-1]  # The actual combined helper derives its own condition.
        graph.input.append(h.make_tensor_value_info("input_ids", T.INT64, [1, "S"]))
        if token_mode:
            for tensor in graph.input:
                if tensor.name in ("x", "input_ids"):
                    tensor.type.tensor_type.shape.dim[1].ClearField("dim_param")
                    tensor.type.tensor_type.shape.dim[1].dim_value = 1
        graph.initializer.extend(combined.graph.initializer)
        model = h.make_model(graph, opset_imports=combined.opset_import, ir_version=10)
        path = tmp_path / (name + ".onnx")
        path.write_bytes(model.SerializeToString())
        paths.append(path)
    return paths


def test_combine_hook_returns_identical_header_to_posthoc_transform(tmp_path):
    prefill, token = split_fixture(tmp_path)
    original = tmp_path / "original"
    proposed = tmp_path / "proposed"
    package._combine_branches(prefill, prefill, token, original)
    hook_calls = []

    def transform(model):
        assert not proposed.exists()
        hook_calls.append(1)
        return prune.transform_prefill_lm_head(
            model, hidden_size=4, vocab_size=8, expected_states=2
        )[0]

    package._combine_branches(
        prefill, prefill, token, proposed, combined_transform=transform
    )
    baseline = onnx.load(original / "model.onnx", load_external_data=False)
    expected, _ = prune.transform_prefill_lm_head(
        baseline, hidden_size=4, vocab_size=8, expected_states=2
    )
    assert (proposed / "model.onnx").read_bytes() == expected.SerializeToString()
    assert hook_calls == [1]


def test_disabled_hook_keeps_default_output_bytes(tmp_path):
    prefill, token = split_fixture(tmp_path)
    original, proposed = tmp_path / "old", tmp_path / "new"
    package._combine_branches(prefill, prefill, token, original)
    package._combine_branches(
        prefill, prefill, token, proposed, combined_transform=None
    )
    assert (original / "model.onnx").read_bytes() == (
        proposed / "model.onnx"
    ).read_bytes()


@pytest.mark.parametrize("failure", ["exception", "non_model", "unresolved_input"])
def test_hook_failure_leaves_no_output_directory(tmp_path, failure):
    prefill, token = split_fixture(tmp_path)
    output = tmp_path / "rejected"

    def transform(model):
        if failure == "exception":
            raise ValueError("owned rejected transform")
        if failure == "non_model":
            return None
        model.graph.node[0].input[0] = "unknown_capture"
        return model

    with pytest.raises(ValueError):
        package._combine_branches(
            prefill, prefill, token, output, combined_transform=transform
        )
    assert not output.exists()
