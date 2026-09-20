"""Tiny CPU fixtures; no SDK imports, ORT sessions or model weights."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
import zipfile

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto as T, helper as h  # noqa: E402
from benchmarks import qwen35_package as package  # noqa: E402


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _model(folder, *, token):
    seq = 1 if token else "sequence_length"
    weights = h.make_tensor("shared_weight", T.FLOAT, [2], [1., 2.])
    # The common initializer is byte-identical in independent external files.
    weights.ClearField("float_data")
    weights.raw_data = b"\x00\x00\x80\x3f\x00\x00\x00\x40"
    host_count = h.make_tensor("attention_mask_const_uint", T.UINT32, [1], [1])
    nodes = [h.make_node("Cast", ["input_ids"], ["x"], to=T.BFLOAT16)]
    if token:
        nodes.append(h.make_node("DynamicDispatch", ["x", "attention_mask_const_uint"], ["logits"],
                                 name="projection", domain="com.ryzenai", model_type=9, input_num=1,
                                 input_shape_0=[1, 1], input_shape_1=[1], output_shape_0=[1, 1, 2],
                                 xclbin="synthetic", mladf_version="v2"))
    else:
        nodes.append(h.make_node("MatMulNBitsBf", ["x"], ["logits"], name="eager_projection", domain="com.ryzenai"))
    graph = h.make_graph(nodes, "fixture", [h.make_tensor_value_info("input_ids", T.INT64, [1, seq])],
                         [h.make_tensor_value_info("logits", T.FLOAT, [1, seq, 2])],
                         initializer=[weights] + ([host_count] if token else []))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.ryzenai", 1)])
    onnx.save_model(model, folder / "model.onnx", save_as_external_data=True,
                    all_tensors_to_one_file=True, location="initializers.data", size_threshold=0)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    prefill, token, sdk = [tmp_path / n for n in ("prefill", "token", "sdk")]
    for path in (prefill, token, sdk):
        path.mkdir()
    for path in (prefill / "cache", token / "cache", sdk / "deployment"):
        path.mkdir()
    _model(prefill, token=False)
    _model(token, token=True)
    (prefill / "model.pb.bin").write_bytes(b"synthetic protobuf header")
    (prefill / "model.bin").write_bytes(b"eager weights")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        _json(prefill / name, {})
    config = dict(model=dict(type="qwen3_5_text", decoder=dict(filename="model.onnx", session_options=dict(
        provider_options=[dict(RyzenAI=dict(external_data_file="model.pb.bin", hybrid_opt_token_backend="npu",
                                          hybrid_opt_max_seq_length="4096"))]))),
                  search=dict(max_length=16384, chunk_size=64))
    _json(prefill / "genai_config.json", config)
    constant = token / "cache/projection_0.const"
    constant.write_bytes(b"constant")
    metadata = dict(op_list=[dict(type="MladfMatMul", in_args=["x"], out_args=["logits"], const_args=["w"], attrs={})],
                    state_table_updates=[], aux_info=dict(is_llm=False),
                    fused_tensors=dict(**{"in": dict(packed_tensors=["x"]), "out": dict(packed_tensors=["logits"])}),
                    tensor_map=dict(x=dict(shape=[1, 1]), logits=dict(shape=[1, 1, 2]),
                                    w=dict(file_name=str(constant), file_size=8)))
    _json(token / "cache/projection_meta.json", metadata)
    with zipfile.ZipFile(sdk / "deployment/dyn_bins.zip", "w") as full:
        full.writestr("family/first.bin", b"first")
        full.writestr("family/second.bin", b"second")
    with zipfile.ZipFile(prefill / "cache/txn_bins.zip", "w") as old:
        old.writestr("family/first.bin", b"first")
    header = SimpleNamespace(external_data=SimpleNamespace(filename="model.bin", npu=True, gpu=False),
                             operators={"eager_projection": SimpleNamespace(op_type="MatMulNBitsBf",
                                         data=[SimpleNamespace(offset=0, size=13)])})
    monkeypatch.setattr(package, "_eager_header", lambda p: (header, dict(descriptor_sha256="synthetic")))
    def checker(path):
        onnx.checker.check_model(str(path), full_check=False)
        return dict(status="passed", full_check=False, mock_installed_schema=True)
    monkeypatch.setattr(package, "check_model_with_installed_ort_schema", checker)
    profile = dict(source_revision="synthetic-revision", expected_dd_nodes=1,
                   expected_dd_counts={"MladfMatMul": 1})
    return SimpleNamespace(prefill=prefill, token=token, sdk=sdk, profile=profile,
                           output=tmp_path / "package", header=header, metadata=metadata, config=config)


def _package(f):
    return package.package_model(f.prefill, f.token, f.output, sdk_root=f.sdk, profile=f.profile)


def test_package_copies_relocates_and_preserves_eager_configuration(fixture):
    f = fixture
    originals = {p: p.read_bytes() for root in (f.prefill, f.token, f.sdk) for p in root.rglob("*") if p.is_file()}
    result = _package(f)
    assert result["status"] == "materialized_cpu_checked" and result["runtime_unverified"]
    assert result["source_models"]["source_revision"] == "synthetic-revision"
    assert result["metadata_count"] == 1
    assert result["transaction_archive"]["entry_count"] == 2
    assert len(result["transaction_archive"]["eager_subset_entries"]) == 1
    assert (f.output / "cache/txn_bins.zip").read_bytes() == (f.sdk / "deployment/dyn_bins.zip").read_bytes()
    meta = json.loads((f.output / "cache/projection_meta.json").read_text())
    assert meta["tensor_map"]["w"]["file_name"] == str(f.output / "cache/projection_0.const")
    before = copy.deepcopy(f.metadata)
    before["tensor_map"]["w"]["file_name"] = meta["tensor_map"]["w"]["file_name"]
    assert before == meta
    config = json.loads((f.output / "genai_config.json").read_text())
    opts = config["model"]["decoder"]["session_options"]["provider_options"][0]["RyzenAI"]
    assert opts["external_data_file"] == "model.pb.bin" and opts["hybrid_opt_max_seq_length"] == "4096"
    assert opts["dd_cache"] == (f.output / "cache").as_posix() and opts["compile_fusion_rt"] == "1"
    assert opts["onnx_custom_ops_const_key"] == ""
    assert config["search"] == f.config["search"]
    combined = onnx.load(f.output / "model.onnx", load_external_data=False)
    switch = next(n for n in combined.graph.node if n.op_type == "If")
    branches = {a.name: a.g for a in switch.attribute}
    assert any(n.op_type == "DynamicDispatch" for n in branches["then_branch"].node)
    assert any(n.op_type == "MatMulNBitsBf" for n in branches["else_branch"].node)
    assert not branches["then_branch"].input and not branches["else_branch"].input
    assert all(p.read_bytes() == content for p, content in originals.items())
    for row in result["artifacts"]:
        path = f.output / row["path"]
        assert package._sha256(path) == row["sha256"] and path.stat().st_nlink == 1


@pytest.mark.parametrize("case", ["existing", "inside_source", "absolute_external", "traversal_external", "constant_escape", "shape", "eager_extent", "count", "archive_mismatch", "opset"])
def test_rejects_unsafe_or_incompatible_input_before_manifest(fixture, case):
    f = fixture
    if case == "existing":
        f.output.mkdir()
    elif case == "inside_source":
        f.output = f.prefill / "nested"
    elif case in {"absolute_external", "traversal_external", "opset"}:
        model = onnx.load(f.token / "model.onnx", load_external_data=False)
        if case == "opset":
            model.opset_import[0].version = 20
        else:
            location = str(f.token / "initializers.data") if case == "absolute_external" else "../prefill/initializers.data"
            next(e for e in model.graph.initializer[0].external_data if e.key == "location").value = location
        (f.token / "model.onnx").write_bytes(model.SerializeToString())
    elif case in {"constant_escape", "shape"}:
        if case == "constant_escape":
            f.metadata["tensor_map"]["w"]["file_name"] = str(f.prefill / "model.bin")
        else:
            f.metadata["tensor_map"]["x"]["shape"] = [1, 2]
        _json(f.token / "cache/projection_meta.json", f.metadata)
    elif case == "eager_extent":
        f.header.operators["eager_projection"].data[0].size = 100
    elif case == "count":
        f.profile["expected_dd_nodes"] = 2
    elif case == "archive_mismatch":
        with zipfile.ZipFile(f.prefill / "cache/txn_bins.zip", "w") as z:
            z.writestr("family/first.bin", b"different")
    with pytest.raises((package.PackagingError, FileExistsError)):
        _package(f)
    assert not (f.output / "package-manifest.json").exists()


def test_schema_inspection_is_separate_from_fresh_onnx_check(tmp_path, monkeypatch):
    path = tmp_path / "tiny.onnx"
    path.write_bytes(b"header-only-mock")
    calls = []
    snapshot = dict(onnxruntime_version="test", schema={"name": "SimplifiedLayerNormalization"})
    def run(command, *, input_text=None):
        calls.append((command, input_text))
        return snapshot if len(calls) == 1 else dict(status="passed", full_check=False)
    monkeypatch.setattr(package, "_schema_subprocess", run)
    assert package.check_model_with_installed_ort_schema(path)["status"] == "passed"
    assert len(calls) == 2
    assert calls[0][0][2] == "-c" and "get_all_operator_schema" in calls[0][0][3]
    assert calls[1][0][-2:] == ["--check-schema", str(path)]
    assert json.loads(calls[1][1]) == snapshot
    assert "InferenceSession" not in package._ORT_SCHEMA_EXPORT_CODE


def test_source_revision_is_required(fixture):
    fixture.profile.pop("source_revision")
    with pytest.raises(package.PackagingError, match="revision"):
        _package(fixture)
    assert not fixture.output.exists()


def test_source_mutation_during_packaging_never_writes_success_manifest(fixture, monkeypatch):
    original_copy = package._copy_independent
    changed = False
    def copy_then_mutate(source, destination):
        nonlocal changed
        digest = original_copy(source, destination)
        if not changed:
            changed = True
            path = fixture.token / "cache/projection_meta.json"
            path.write_text(path.read_text() + "\n", encoding="utf-8")
        return digest
    monkeypatch.setattr(package, "_copy_independent", copy_then_mutate)
    with pytest.raises(package.PackagingError, match="Source changed"):
        _package(fixture)
    assert fixture.output.is_dir()  # Partial evidence is retained, never reused.
    assert not (fixture.output / "package-manifest.json").exists()
