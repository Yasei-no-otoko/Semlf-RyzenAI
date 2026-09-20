from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto as T, helper as h  # noqa: E402
from benchmarks import (  # noqa: E402
    prepare_qwen35_token_fusion as cli,
    qwen35_package as package,
    qwen35_prefill,
)
import test_prepare_qwen35_token_fusion as original_cli_tests  # noqa: E402
import test_qwen35_package as original_package_tests  # noqa: E402
from test_qwen35_prefill import native_node  # noqa: E402

small_inputs = original_cli_tests.small_inputs
fixture = original_package_tests.fixture


def test_package_native_header_matches_legacy_owned_fixture(fixture):
    f = fixture
    new_path = package._combine_branches(
        f.prefill / "model.onnx",
        f.prefill / "model.onnx",
        f.token / "model.onnx",
        f.output.with_name("new"),
    )
    # Golden comes from the previous packager's owned eight-byte fixture. Fix
    # only the test's IR version so ONNX package version defaults cannot affect
    # the comparison. No real model, cache path or historical report is needed.
    normalized = onnx.load(new_path, load_external_data=False)
    normalized.ir_version = 10
    assert hashlib.sha256(normalized.SerializeToString()).hexdigest() == (
        "5bd0a446f7e849ee321ed655eff47fd0a32aaaffe7f39507f7ab87e74d8b3794"
    )
    assert all(
        p.read_bytes() == b"\x00\x00\x80\x3f\x00\x00\x00\x40"
        for p in new_path.parent.glob("weights-*.bin")
    )


def add_native_layers(f):
    model = onnx.load(f.prefill / "model.onnx", load_external_data=False)
    shapes = [
        (1, "sequence_length", 2048),
        (1, "sequence_length", 2048),
        (1, "sequence_length", 4096),
        (1, 32, 128, 128),
        (1, "sequence_length", 32),
        (1, "sequence_length", 32),
    ]
    names = ["query", "key", "value", "state", "gate", "beta"]
    for name, shape in zip(names, shapes, strict=True):
        model.graph.input.append(h.make_tensor_value_info(name, T.BFLOAT16, shape))
    for i in range(24):
        node = native_node(
            f"layer{i}/LinearAttention", names, [f"attention_{i}", f"state_{i}"]
        )
        model.graph.node.append(node)
        model.graph.value_info.extend(
            [
                h.make_tensor_value_info(
                    node.output[0], T.BFLOAT16, [1, "sequence_length", 4096]
                ),
                h.make_tensor_value_info(node.output[1], T.BFLOAT16, [1, 32, 128, 128]),
            ]
        )
        f.header.operators[node.name] = SimpleNamespace(
            op_type="LinearAttention", data=[]
        )
    (f.prefill / "model.onnx").write_bytes(model.SerializeToString())
    f.config["search"]["chunk_size"] = 64
    f.config["model"]["decoder"]["session_options"]["provider_options"][0]["RyzenAI"][
        "hybrid_opt_token_backend"
    ] = "npu"
    (f.prefill / "genai_config.json").write_text(json.dumps(f.config))
    return model


def test_optional_package_transforms_header_and_copies_source_data_once(
    fixture, monkeypatch
):
    f = fixture
    source = add_native_layers(f)
    originals = {
        p: p.read_bytes()
        for folder in (f.prefill, f.token, f.sdk)
        for p in folder.rglob("*")
        if p.is_file()
    }
    copies = []
    original_copy = package._copy_independent

    def copy(src, dst):
        copies.append((src, dst))
        return original_copy(src, dst)

    monkeypatch.setattr(package, "_copy_independent", copy)
    result = package.package_model(
        f.prefill,
        f.token,
        f.output,
        sdk_root=f.sdk,
        profile=f.profile,
        prefill_linear_attention="token_loop",
    )
    transform = result["prefill_transformation"]
    assert (
        transform["required_global_chunk_size"]
        == transform["observed_global_chunk_size"]
        == 64
    )
    assert (
        transform["changed_nodes"] == 24
        and transform["preserved_native_operator_names_and_types"]
    )
    assert (
        transform["source_model_sha256"]
        == result["source_models"]["prefill"]["sha256"]
        == package._sha256(f.prefill / "model.onnx")
    )
    assert transform["transformer_sha256"] == package._sha256(
        Path(qwen35_prefill.__file__)
    )
    assert (
        transform["runtime_validation"] == "not performed by this conversion command"
        and result["runtime_unverified"]
    )
    combined = onnx.load(f.output / "model.onnx", load_external_data=False)
    branch = next(
        a.g
        for n in combined.graph.node
        if n.op_type == "If"
        for a in n.attribute
        if a.name == "else_branch"
    )
    assert sum(n.op_type == "Loop" for n in branch.node) == 24
    assert [
        n.name
        for g in package._graphs(branch)
        for n in g.node
        if n.op_type == "LinearAttention"
    ] == [n.name for n in source.graph.node if n.op_type == "LinearAttention"]
    for path in (
        f.prefill / "initializers.data",
        f.token / "initializers.data",
        f.prefill / "model.bin",
    ):
        assert sum(src == path for src, dst in copies) == 1
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert (f.output / "model.pb.bin").read_bytes() == (
        f.prefill / "model.pb.bin"
    ).read_bytes()
    config = json.loads((f.output / "genai_config.json").read_text())
    assert config["search"]["chunk_size"] == 64


@pytest.mark.parametrize("mode", ["native", "token_loop"])
def test_option_is_pinned_in_plan_and_reaches_packaging(
    small_inputs, monkeypatch, mode
):
    f = small_inputs
    plan = cli.create_plan(
        f.source,
        f.prefill,
        f.sdk,
        f.work,
        f.output,
        f.profile,
        prefill_linear_attention=mode,
    )
    assert (
        plan["prefill_linear_attention"] == mode
        and "qwen35_prefill.py" in plan["code_sha256"]
    )
    cli.write_new(f.plan, plan)
    monkeypatch.setattr(cli, "regenerate_token", lambda *args: f.source / "fixed.onnx")
    from benchmarks import qwen35_dd

    monkeypatch.setattr(
        qwen35_dd, "lower_token_graph", lambda *args, **kw: {"status": "mock"}
    )
    received = {}

    def mock_package(*args, **kw):
        received.update(kw)
        return {"status": "mock"}

    monkeypatch.setattr(package, "package_model", mock_package)
    result = cli.build(f.plan, cli.sha256(f.plan))
    assert (
        received["prefill_linear_attention"] == mode
        and result["prefill_linear_attention"] == mode
    )


def test_cli_plan_flag_roundtrip(small_inputs, monkeypatch):
    f = small_inputs
    args = [
        "tool",
        "plan",
        "--source",
        str(f.source),
        "--prefill",
        str(f.prefill),
        "--sdk-root",
        str(f.sdk),
        "--work-dir",
        str(f.work),
        "--output",
        str(f.output),
        "--plan",
        str(f.plan),
        "--profile",
        str(f.profile),
        "--prefill-linear-attention",
        "token_loop",
    ]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert json.loads(f.plan.read_text())["prefill_linear_attention"] == "token_loop"
    assert not f.work.exists() and not f.output.exists()


def test_invalid_mode_and_tampered_plan_fail_before_build(small_inputs):
    f = small_inputs
    with pytest.raises(ValueError, match="mode"):
        cli.create_plan(
            f.source,
            f.prefill,
            f.sdk,
            f.work,
            f.output,
            f.profile,
            prefill_linear_attention="auto",
        )
    plan = cli.create_plan(f.source, f.prefill, f.sdk, f.work, f.output, f.profile)
    cli.write_new(f.plan, plan)
    digest = cli.sha256(f.plan)
    plan["prefill_linear_attention"] = "token_loop"
    f.plan.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="hash"):
        cli.build(f.plan, digest)
    assert not f.work.exists() and not f.output.exists()


def test_token_loop_requires_existing_npu_option_and_chunk64(fixture):
    f = fixture
    add_native_layers(f)
    f.config["search"]["chunk_size"] = 32
    (f.prefill / "genai_config.json").write_text(json.dumps(f.config))
    with pytest.raises(package.PackagingError, match="chunk64"):
        package.package_model(
            f.prefill,
            f.token,
            f.output,
            sdk_root=f.sdk,
            profile=f.profile,
            prefill_linear_attention="token_loop",
        )
    assert not f.output.exists()
