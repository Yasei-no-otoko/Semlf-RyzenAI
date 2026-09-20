"""Small synthetic graph and SDK-contract tests; no real weights or NPU."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
T, h, nh = onnx.TensorProto, onnx.helper, onnx.numpy_helper
from benchmarks import qwen35_dd as dd  # noqa: E402


def _profile():
    return {"token_lowering": dict(matmul_count=1, linear_attention_count=1,
        input_count=8, output_count=3, state_tensor_count=1, vocab_size=32,
        bits=4, block_size=128, max_tensor_bytes=512 * 1024**2, max_projection_bytes=768 * 1024**2,
        matmul_xclbin="reviewed_matmul_resource", linear_attention_xclbin="reviewed_la_resource",
        host_count_tensor="mask_count", linear_attention=dict(k_heads=16, v_heads=32,
            k_head_dim=128, v_head_dim=128, query_scale=1.0, chunk_size=64))}


def _fixture(tmp_path, *, bias=True):
    source = tmp_path / "source"
    source.mkdir()
    arrays = {"w": np.arange(32 * 64, dtype=np.uint8).reshape(32, 1, 64),
              "s": np.ones((32, 1), dtype=np.float16), "z": np.full((32, 1), 0x88, dtype=np.uint8),
              "b": np.full(32, 2, dtype=np.float16), "preserved": np.array([3], dtype=np.float32)}
    tensors, offset = [], 0
    with (source / "tensors.bin").open("xb") as stream:
        for name, array in arrays.items():
            tensor = nh.from_array(array, name)
            payload = array.tobytes()
            stream.write(payload)
            onnx.external_data_helper.set_external_data(tensor, "tensors.bin", offset, len(payload))
            tensor.ClearField("raw_data")
            tensor.data_location = T.EXTERNAL
            tensors.append(tensor)
            offset += len(payload)
    tensors.append(nh.from_array(np.array([1], dtype=np.int64), "axis"))
    preserved_alias = copy.deepcopy(tensors[-2])
    preserved_alias.name = "preserved_alias"
    tensors.append(preserved_alias)
    make = h.make_tensor_value_info
    inputs = [make("attention_mask", T.INT64, [1, "total_length"]), make("x", T.FLOAT16, [1, 1, 128]),
              make("q", T.FLOAT16, [1, 1, 2048]), make("k", T.FLOAT16, [1, 1, 2048]),
              make("v", T.FLOAT16, [1, 1, 4096]), make("past_key_values.0.value", T.FLOAT16, [1, 32, 128, 128]),
              make("gate", T.FLOAT16, [1, 1, 32]), make("beta_before_sigmoid", T.FLOAT16, [1, 1, 32])]
    outputs = [make("logits", T.FLOAT16, [1, 1, 32]), make("attention", T.FLOAT16, [1, 1, 4096]),
               make("present.0.value", T.FLOAT16, [1, 32, 128, 128])]
    nodes = [h.make_node("ReduceSum", ["attention_mask", "axis"], ["mask_count"], name="mask_reduce", keepdims=0, noop_with_empty_axes=0),
             h.make_node("MatMulNBits", ["x", "w", "s", "z", "", "b"] if bias else ["x", "w", "s", "z"],
                         ["mm_out"], name="projection", domain="com.microsoft", K=128, N=32, bits=4, block_size=128),
             h.make_node("Identity", ["mm_out"], ["logits"], name="surrounding_identity"),
             h.make_node("Sigmoid", ["beta_before_sigmoid"], ["beta"], name="original_sigmoid"),
             h.make_node("LinearAttention", ["q", "k", "v", "past_key_values.0.value", "gate", "beta"],
                         ["attention", "present.0.value"], name="linear", domain="com.microsoft", q_num_heads=16,
                         kv_num_heads=32, update_rule="gated_delta", scale=1.0, chunk_size=64)]
    values = [make("mask_count", T.INT64, [1]), make("mm_out", T.FLOAT16, [1, 1, 32]), make("beta", T.FLOAT16, [1, 1, 32])]
    model = h.make_model(h.make_graph(nodes, "tiny", inputs, outputs, tensors, value_info=values),
                         opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)], ir_version=10)
    path = source / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return path, model


def _mock_sdk(monkeypatch, *, fallback=False, wrong_aux=False, wrong_version=False):
    observed = {"projections": [], "wrappers": []}

    class Params:
        def __init__(self, options, output, cache, abs_cache):
            self.options, self.cache = options, cache

    def replacement(unit, pass_id, subgraph, params):
        node = subgraph[0]
        constants = list(unit.graph.initializer)
        observed["projections"].append({t.name: nh.to_array(t) for t in constants})
        if fallback:
            return [copy.deepcopy(node)], [], []
        input_value, output_value = unit.graph.input[0], unit.graph.output[0]
        input_name, output_name = node.name + "_bf16_input", node.name + "_bf16_output"
        nodes = [h.make_node("Cast", [node.input[0]], [input_name], name=node.name + "_cast_in", to=T.BFLOAT16),
                 h.make_node("MladfMatMul", [input_name, *[t.name for t in constants]], [output_name], name=node.name,
                             domain="com.ryzenai", op_version="flat" if wrong_version else "v2"),
                 h.make_node("Cast", [output_name], [node.output[0]], name=node.name + "_cast_out", to=T.FLOAT16)]
        values = [h.make_tensor_value_info(name, T.BFLOAT16, dd._shape(v)) for name, v in
                  ((input_name, input_value), (output_name, output_value))]
        return nodes, constants, values

    def build(unit, nodes, params, *, extra_attributes, meta_name=""):
        assert len(nodes) == 1
        native = nodes[0]
        consts = {t.name: t for t in unit.graph.initializer}
        ins = [name for name in native.input if name not in consts]
        outs = list(native.output)
        values = {v.name: v for v in [*unit.graph.input, *unit.graph.output, *unit.graph.value_info]}
        outer_name = meta_name + native.name
        attrs = {f"input_shape_{i}": dd._shape(values[name]) for i, name in enumerate(ins)}
        attrs.update({f"output_shape_{i}": dd._shape(values[name]) for i, name in enumerate(outs)})
        attrs.update(extra_attributes, xclbin=params.options["xclbins"])
        outer = h.make_node("DynamicDispatch", ins, outs, name=outer_name, domain="com.ryzenai", **attrs)
        metadata = dict(op_list=[dict(name=native.name, type=native.op_type, in_args=ins, out_args=outs, const_args=list(consts))],
                        fused_tensors={"in": {"packed_tensors": ins}, "out": {"packed_tensors": outs}},
                        tensor_map={name: dict(dtype="bfloat16", shape=dd._shape(values[name])) for name in [*ins, *outs]},
                        aux_info={}, state_table_updates=[])
        if params.options["is_llm"]:
            metadata["aux_info"] = dict(is_llm=True, states=["bad"] if wrong_aux else [])
            metadata["state_table_updates"] = [dict(state_table_idx=0, update_func=1, update_arg=1)]
        for index, (name, tensor) in enumerate(consts.items()):
            path = params.cache / f"{outer_name}_{index}.const"
            payload = nh.to_array(tensor).tobytes()
            path.write_bytes(payload)
            metadata["tensor_map"][name] = dict(dtype="mock_const", shape=list(tensor.dims), file_name=str(path), file_size=len(payload))
        (params.cache / (outer_name + "_meta.json")).write_text(json.dumps(metadata), encoding="utf-8")
        observed["wrappers"].append(copy.deepcopy(native))
        return outer

    monkeypatch.setattr(dd, "_sdk_api", lambda: SimpleNamespace(ReplaceParams=Params, get_extractor=lambda model: model,
                                                              replacement=replacement, build_dd_node=build))
    def check(path):
        onnx.checker.check_model(str(path), full_check=False)
        return dict(status="passed", synthetic_graph_standard_checker=True)
    monkeypatch.setattr(dd, "_check_final_model", check)
    return observed


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("explicit_reduce_default", [False, True])
def test_lowers_once_preserving_state_surrounding_ops_and_bounded_payload(tmp_path, monkeypatch, bias, explicit_reduce_default):
    source, original = _fixture(tmp_path, bias=bias)
    if not explicit_reduce_default:
        reduction = original.graph.node[0]
        reduction.attribute.remove(next(a for a in reduction.attribute if a.name == "noop_with_empty_axes"))
        source.write_bytes(original.SerializeToString())
    source_hash, data_hash = dd._sha(source), dd._sha(source.parent / "tensors.bin")
    observed = _mock_sdk(monkeypatch)
    output = tmp_path / "output"
    cwd = Path.cwd()
    report = dd.lower_token_graph(source, output, profile=_profile())
    assert Path.cwd() == cwd
    assert report["status"] == "lowered_cpu_checked" and len(report["partitions"]) == 2
    assert report["source_immutability_checked"] and not report["npu_execution"] and not report["host_compile"]
    assert dd._sha(source) == source_hash and dd._sha(source.parent / "tensors.bin") == data_hash
    packed_bias = observed["projections"][0]["b" if bias else "w.dd_zero_bias"]
    np.testing.assert_array_equal(packed_bias, np.full(32, 2 if bias else 0, dtype=np.float16))
    model = onnx.load(output / "model.onnx", load_external_data=False)
    assert [v.SerializeToString() for v in model.graph.input] == [v.SerializeToString() for v in original.graph.input]
    assert model.graph.output[0].type.tensor_type.elem_type == T.FLOAT
    assert model.graph.output[2].SerializeToString() == original.graph.output[2].SerializeToString()
    old_sigmoid = next(n for n in original.graph.node if n.name == "original_sigmoid")
    assert next(n for n in model.graph.node if n.name == old_sigmoid.name).SerializeToString() == old_sigmoid.SerializeToString()
    wrappers = [n for n in model.graph.node if n.op_type == "DynamicDispatch"]
    assert len(wrappers) == 2
    assert sum(n.output == [dd.HOST_COUNT] for n in model.graph.node) == 1
    for wrapper in wrappers:
        attrs = dd._attrs(wrapper)
        meta = json.loads((output / "cache" / (wrapper.name + "_meta.json")).read_text())
        assert wrapper.input[-1] == dd.HOST_COUNT and attrs["input_num"] in (1, 6)
        assert attrs[f"input_shape_{attrs['input_num']}"] == [1] and attrs["model_type"] == 9
        assert list(wrapper.input[:-1]) == meta["op_list"][0]["in_args"] and dd.HOST_COUNT not in meta["tensor_map"]
        assert meta["state_table_updates"] == []
        assert meta["aux_info"] == ({"is_llm": False, "states": []} if attrs["input_num"] == 1 else {})
    native_la = observed["wrappers"][1]
    assert dd._attrs(native_la) == dict(k_heads=16, v_heads=32, k_head_dim=128, v_head_dim=128, q_seq=1)
    loaded = onnx.load(output / "model.onnx")
    preserved = next(t for t in loaded.graph.initializer if t.name == "preserved")
    np.testing.assert_array_equal(nh.to_array(preserved), [3])
    # The two remaining external constants share one original range and are copied once.
    matching = [r for r in report["external_data"]["ranges"] if r["length"] == 4]
    assert len(matching) == 1
    for artifact in report["artifacts"]:
        path = output / artifact["path"]
        assert path.stat().st_size == artifact["size"] and dd._sha(path) == artifact["sha256"]
    with pytest.raises(dd.LoweringError, match="must be new"):
        dd.lower_token_graph(source, output, profile=_profile())


@pytest.mark.parametrize("change,match", [("g_idx", "g_idx"), ("bits", "quantization"), ("group", "quantization"),
                                          ("beta", "Sigmoid"), ("axis", "axis1"), ("empty_axes", "Host control"),
                                          ("reserved", "reserved"), ("shape", "projection interface")])
def test_rejects_unsupported_source_contract(tmp_path, monkeypatch, change, match):
    source, model = _fixture(tmp_path)
    projection = model.graph.node[1]
    if change == "g_idx":
        projection.input[4] = "axis"
    elif change in ("bits", "group"):
        key = "bits" if change == "bits" else "block_size"
        next(a for a in projection.attribute if a.name == key).i = 8 if change == "bits" else 64
    elif change == "beta":
        model.graph.node[3].op_type = "Identity"
    elif change == "axis":
        axis = next(t for t in model.graph.initializer if t.name == "axis")
        axis.CopyFrom(nh.from_array(np.array([-1], dtype=np.int64), "axis"))
    elif change == "empty_axes":
        next(a for a in model.graph.node[0].attribute if a.name == "noop_with_empty_axes").i = 1
    elif change == "reserved":
        model.graph.node[3].name = "__qwen35_existing"
    elif change == "shape":
        model.graph.input[1].type.tensor_type.shape.dim[2].dim_value = 64
    source.write_bytes(model.SerializeToString())
    before = source.read_bytes()
    _mock_sdk(monkeypatch)
    output = tmp_path / "failed"
    cwd = Path.cwd()
    with pytest.raises(dd.LoweringError, match=match):
        dd.lower_token_graph(source, output, profile=_profile())
    assert source.read_bytes() == before and Path.cwd() == cwd
    if output.exists():
        assert json.loads((output / "lowering-manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("mode,match", [("fallback", "fallback"), ("wrong_aux", "aux_info"), ("wrong_version", "no-control")])
def test_rejects_sdk_fallback_and_contract_changes(tmp_path, monkeypatch, mode, match):
    source, _ = _fixture(tmp_path)
    _mock_sdk(monkeypatch, **{mode: True})
    output = tmp_path / "failed"
    with pytest.raises(dd.LoweringError, match=match):
        dd.lower_token_graph(source, output, profile=_profile())
    assert not (output / "model.onnx").exists()


@pytest.mark.parametrize("mode", ["escape", "range", "cap"])
def test_external_paths_ranges_and_caps_fail_closed(tmp_path, monkeypatch, mode):
    source, model = _fixture(tmp_path)
    if mode != "cap":
        weights = next(t for t in model.graph.initializer if t.name == "w")
        entry = next(e for e in weights.external_data if e.key == ("location" if mode == "escape" else "length"))
        entry.value = "../outside.bin" if mode == "escape" else "999999"
        source.write_bytes(model.SerializeToString())
    profile = _profile()
    if mode == "cap":
        profile["token_lowering"]["max_tensor_bytes"] = 16
    observed = _mock_sdk(monkeypatch)
    with pytest.raises(dd.LoweringError, match="source model directory|byte range|read bound"):
        dd.lower_token_graph(source, tmp_path / "failed", profile=profile)
    assert observed["projections"] == []


def test_disallows_source_child_and_preserves_existing_output(tmp_path):
    source, _ = _fixture(tmp_path)
    with pytest.raises(dd.LoweringError, match="disjoint"):
        dd.lower_token_graph(source, source.parent / "nested", profile=_profile())
    output = tmp_path / "already"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(dd.LoweringError, match="must be new"):
        dd.lower_token_graph(source, output, profile=_profile())
    assert sentinel.read_text() == "keep"


def test_checker_failure_never_publishes_success(tmp_path, monkeypatch):
    source, _ = _fixture(tmp_path)
    _mock_sdk(monkeypatch)
    def reject(path):
        raise RuntimeError("synthetic checker failure")
    monkeypatch.setattr(dd, "_check_final_model", reject)
    with pytest.raises(RuntimeError, match="checker failure"):
        dd.lower_token_graph(source, tmp_path / "failed", profile=_profile())
    report = json.loads((tmp_path / "failed/lowering-manifest.json").read_text())
    assert report["status"] == "failed" and "artifacts" not in report


def test_checker_import_from_direct_script_path_and_changed_cwd(tmp_path):
    directory = str(Path(dd.__file__).resolve().parent)
    script = f"""
import importlib.util
import sys
sys.path.insert(0, {directory!r})
assert importlib.util.find_spec('benchmarks') is None
import qwen35_dd
import qwen35_package
qwen35_package.check_model_with_installed_ort_schema = lambda path: {{'checked': path}}
assert qwen35_dd._check_final_model('owned-fixture.onnx') == {{'checked': 'owned-fixture.onnx'}}
print('direct-script checker import passed')
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "direct-script checker import passed"
