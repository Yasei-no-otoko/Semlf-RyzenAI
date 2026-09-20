from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks import prepare_qwen35_token_fusion as conversion

onnx = pytest.importorskip("onnx")
TensorProto, helper = onnx.TensorProto, onnx.helper


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def small_inputs(tmp_path, monkeypatch):
    source, prefill, sdk = [tmp_path / name for name in ("source", "prefill", "sdk")]
    for directory in (source, prefill, sdk):
        directory.mkdir()
    (source / "source.bin").write_bytes(b"owned source fixture")
    (prefill / "model.onnx").write_bytes(b"header fixture")
    config = {
        "model": {
            "type": "qwen3_5_text", "context_length": 16384,
            "decoder": {"session_options": {"provider_options": [{"RyzenAI": {
                "hybrid_opt_token_backend": "npu", "hybrid_opt_chunk_context": "1",
                "hybrid_opt_max_seq_length": "4096",
            }}]}},
        },
        "search": {"max_length": 16384, "chunk_size": 64, "past_present_share_buffer": True},
    }
    _write_json(prefill / "genai_config.json", config)
    profile = {
        "schema_version": 1, "source_revision": conversion.SOURCE_REVISION,
        "source_artifacts": {"source.bin": conversion.sha256(source / "source.bin")},
        "prefill_artifacts": {name: conversion.sha256(prefill / name)
                              for name in ("model.onnx", "genai_config.json")},
    }
    profile_path = tmp_path / "profile.json"
    _write_json(profile_path, profile)
    monkeypatch.setattr(conversion, "check_sdk", lambda root, profile: {"fixture": "CPU-only SDK identity"})
    # build() sets this process-local variable. Restore the original test runner
    # environment even when exercising a deliberately failed build.
    monkeypatch.setenv("RYZEN_AI_INSTALLATION_PATH", str(sdk))
    return SimpleNamespace(source=source, prefill=prefill, sdk=sdk, work=tmp_path / "work",
                           output=tmp_path / "output", profile=profile_path, plan=tmp_path / "plan.json")


def _plan(inputs):
    return conversion.create_plan(inputs.source, inputs.prefill, inputs.sdk,
                                  inputs.work, inputs.output, inputs.profile)


def _save_plan(inputs):
    conversion.write_new(inputs.plan, _plan(inputs))
    return conversion.sha256(inputs.plan)


def test_plan_hashes_owned_files_without_creating_work_or_output(small_inputs):
    plan = _plan(small_inputs)
    assert plan["status"] == "plan_only"
    assert plan["source_artifacts"]["source.bin"]["sha256"] == _digest(b"owned source fixture")
    assert plan["profile_sha256"] == conversion.sha256(small_inputs.profile)
    assert not small_inputs.work.exists()
    assert not small_inputs.output.exists()


@pytest.mark.parametrize("parent", ["source", "prefill", "output"])
def test_overlapping_work_path_is_rejected_before_any_write(small_inputs, parent):
    small_inputs.work = getattr(small_inputs, parent) / "work"
    with pytest.raises(ValueError, match="separate directories"):
        _plan(small_inputs)
    assert not small_inputs.work.exists()


def test_existing_failed_work_is_preserved(small_inputs):
    small_inputs.work.mkdir()
    marker = small_inputs.work / "failure-evidence.json"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="new paths"):
        _plan(small_inputs)
    assert marker.read_bytes() == b"preserve"


@pytest.mark.parametrize("changed", ["source", "profile", "code"])
def test_changed_plan_dependency_stops_before_build(small_inputs, monkeypatch, changed):
    digest = _save_plan(small_inputs)
    if changed == "source":
        (small_inputs.source / "source.bin").write_bytes(b"changed source")
    elif changed == "profile":
        profile = json.loads(small_inputs.profile.read_text(encoding="utf-8"))
        profile["annotation"] = "changed after planning"
        _write_json(small_inputs.profile, profile)
    else:
        original = conversion.code_hashes()
        monkeypatch.setattr(conversion, "code_hashes", lambda: {**original, "changed.py": "0" * 64})
    with pytest.raises(ValueError, match="Artifact changed|Plan inputs"):
        conversion.build(small_inputs.plan, digest)
    assert not small_inputs.work.exists()
    assert not small_inputs.output.exists()


def test_wrong_plan_digest_prevents_all_conversion(small_inputs):
    _save_plan(small_inputs)
    with pytest.raises(ValueError, match="Plan hash mismatch"):
        conversion.build(small_inputs.plan, "0" * 64)
    assert not small_inputs.work.exists()


def test_fixed_graph_failure_records_failure_without_lowering_or_packaging(small_inputs, monkeypatch):
    digest = _save_plan(small_inputs)

    def fail(*_args):
        (small_inputs.work / "partial-owned-header.onnx").write_bytes(b"preserve for diagnosis")
        raise ValueError("Regenerated token graph differs from the measured structural contract")

    monkeypatch.setattr(conversion, "regenerate_token", fail)
    with pytest.raises(ValueError, match="Regenerated token graph"):
        conversion.build(small_inputs.plan, digest)
    report = json.loads((small_inputs.work / "build-report.json").read_text())
    assert report["status"] == "failed"
    assert report["runtime_validated"] is False
    assert report["stages"] == []
    assert report["error"]["type"] == "ValueError"
    assert (small_inputs.work / "partial-owned-header.onnx").read_bytes() == b"preserve for diagnosis"
    assert not small_inputs.output.exists()


def test_copy_inputs_are_independent_and_do_not_modify_original(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = source / "input.bin"
    original.write_bytes(b"original fixture")
    identities = conversion.input_identities(source, {original.name: conversion.sha256(original)})
    destination = tmp_path / "copy"
    conversion.copy_inputs(source, destination, identities)
    (destination / original.name).write_bytes(b"changed copy")
    assert original.read_bytes() == b"original fixture"
    with pytest.raises(FileExistsError):
        conversion.copy_inputs(source, destination, identities)


@pytest.mark.parametrize("relative", ["../outside.bin", "nested/../../outside.bin"])
def test_input_paths_reject_parent_traversal(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe relative"):
        conversion.local_file(tmp_path, relative)


def _token_fixture(directory, location="weights.bin", transient="local source"):
    directory.mkdir()
    data = struct.pack("<ff", 1.25, -2.5)
    (directory / location).write_bytes(data)
    tensor = TensorProto(name="weight", data_type=TensorProto.FLOAT, dims=[2], data_location=TensorProto.EXTERNAL)
    for key, value in (("location", location), ("offset", "0"), ("length", "8")):
        tensor.external_data.add(key=key, value=value)
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"])], "tiny-token",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])], [tensor],
    ), opset_imports=[helper.make_opsetid("", 17)])
    model.metadata_props.add(key="onnx_utils_load", value=transient)
    model.graph.metadata_props.add(key="onnx_utils_load", value=transient)
    # Construct the expected fixture independently: only these known transient
    # fields are excluded, while the graph and tensor byte ranges are retained.
    canonical = copy.deepcopy(model)
    del canonical.metadata_props[:]
    del canonical.graph.metadata_props[:]
    canonical.graph.initializer[0].external_data[0].value = "__fixed_token_external_data__"
    profile = {"fixed_token": {
        "normalized_graph_sha256": _digest(canonical.SerializeToString()),
        "external_data_sha256": _digest(data),
    }}
    path = directory / "model.onnx"
    path.write_bytes(model.SerializeToString())
    return path, model, profile


def test_fixed_token_allows_only_path_and_transient_metadata_changes(tmp_path):
    first, _, expected = _token_fixture(tmp_path / "first")
    relocated, _, relocated_expected = _token_fixture(tmp_path / "relocated", "renamed.bin", "another source")
    before = first.read_bytes(), relocated.read_bytes()
    assert expected == relocated_expected
    evidence = conversion.verify_fixed_token(relocated, expected)
    assert evidence["normalized_graph_sha256"] == expected["fixed_token"]["normalized_graph_sha256"]
    assert evidence["external_data"]["size"] == 8
    assert (first.read_bytes(), relocated.read_bytes()) == before


@pytest.mark.parametrize("mutation", ["operator", "offset", "dtype", "persistent_metadata"])
def test_fixed_token_rejects_semantic_or_byte_range_changes(tmp_path, mutation):
    path, model, profile = _token_fixture(tmp_path / "token")
    if mutation == "operator":
        model.graph.node[0].op_type = "Sub"
    elif mutation == "offset":
        model.graph.initializer[0].external_data[1].value = "4"
    elif mutation == "dtype":
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    else:
        model.metadata_props.add(key="persistent_attribute", value="must not be ignored")
    path.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match="structural contract"):
        conversion.verify_fixed_token(path, profile)


def test_fixed_token_rejects_external_data_corruption(tmp_path):
    path, _, profile = _token_fixture(tmp_path / "token")
    (path.parent / "weights.bin").write_bytes(struct.pack("<ff", 1.25, 2.5))
    with pytest.raises(ValueError, match="Artifact changed"):
        conversion.verify_fixed_token(path, profile)


def test_fixed_token_rejects_a_second_external_file(tmp_path):
    path, model, profile = _token_fixture(tmp_path / "token")
    second = copy.deepcopy(model.graph.initializer[0])
    second.name = "other"
    second.external_data[0].value = "other.bin"
    model.graph.initializer.append(second)
    path.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match="one external data file"):
        conversion.verify_fixed_token(path, profile)


def test_sdk_file_hash_rejects_same_version_changed_content(tmp_path, monkeypatch):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    module = package_dir / "critical.py"
    module.write_bytes(b"changed implementation")
    monkeypatch.setattr(conversion.importlib.metadata, "version", lambda name: "1.8.0")
    monkeypatch.setattr(conversion.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin=str(package_dir / "__init__.py")))
    profile = {"package_versions": {"vendor": "1.8.0"},
               "package_files": {"vendor": {"critical.py": _digest(b"reviewed implementation")}}, "sdk_files": {}}
    with pytest.raises(ValueError, match="compatibility profile"):
        conversion.check_sdk(tmp_path, profile)


def test_sdk_version_mismatch_stops_before_package_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(conversion.importlib.metadata, "version", lambda name: "1.9.0")
    monkeypatch.setattr(conversion.importlib.util, "find_spec", lambda name: pytest.fail("must not inspect package"))
    with pytest.raises(ValueError, match="must be 1.8.0"):
        conversion.check_sdk(tmp_path, {"package_versions": {"vendor": "1.8.0"}})


def test_regeneration_retains_dynamic_prefill_fix_before_static_token(tmp_path, monkeypatch):
    """Record the real SDK's two shape-fix stages, without importing its runtime."""
    from benchmarks import prepare_qwen35

    package = ModuleType("ryzenai_onnx_utils")
    package.__path__ = []
    optimize = ModuleType("ryzenai_onnx_utils.optimize")
    partitioner = ModuleType("ryzenai_onnx_utils.partitioner")
    work = tmp_path / "work"
    work.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"tiny source")
    calls = []
    phase = SimpleNamespace(PREFILL="prefill", TOKEN="token")
    optimize.Phase = phase

    def parse_args(argv):
        return SimpleNamespace(input_model=Path(argv[argv.index("--input-model") + 1]),
                               output_model=Path(argv[argv.index("--output-model") + 1]))

    partitioner.get_parser = lambda: SimpleNamespace(parse_args=parse_args)
    optimize.LlmArgs = lambda args: args

    def preprocess(args, output):
        calls.append(("optimize", args.input_model))
        output.write_bytes(b"optimized header")

    def fix(args, stage):
        calls.append((stage, args.input_model))
        output = work / "fixed-token" / (stage + ".onnx")
        output.write_bytes(stage.encode())
        args.output_model = output
        return output

    optimize.llm_preprocess_optimize = preprocess
    optimize._llm_fix_shapes = fix
    package.optimize = optimize
    package.partitioner = partitioner
    for module in (package, optimize, partitioner):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(prepare_qwen35, "install_precision_free_cast_recovery", lambda: None)
    monkeypatch.setattr(conversion, "verify_fixed_token", lambda path, profile: {"verified_fixture": str(path)})
    before_cwd = Path.cwd()
    fixed = conversion.regenerate_token(source, work,
        conversion.input_identities(source, {"model.onnx": _digest(b"tiny source")}), {})
    assert [name for name, _ in calls] == ["optimize", "prefill", "token"]
    assert calls[1][1] == work / "fixed-token" / "optimized_model.onnx"
    assert calls[2][1] == work / "fixed-token" / "prefill.onnx"
    assert fixed == work / "fixed-token" / "token.onnx"
    assert Path.cwd() == before_cwd
    assert (source / "model.onnx").read_bytes() == b"tiny source"
