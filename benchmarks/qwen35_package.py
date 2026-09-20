"""Create-only CPU packaging for eager prefill and native RyzenAI DD token graphs.

Requires ONNX, the installed Ryzen AI ONNX utilities protobuf descriptor, and ORT
schema metadata. Never creates an ORT session, OGA model, or device context.
Large external data is streamed into independent files only by package_model().
"""

from __future__ import annotations

import copy
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import onnx


def _require(condition, message):
    if not condition:
        raise PackagingError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_json_bytes(value))


def _stat(path):
    value = Path(path).stat()
    return (value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino, value.st_nlink)


def _relative_file(root, name):
    """Resolve a contained local path, rejecting absolute, traversal and symlink escape."""
    relative = Path(name)
    _require(name and not relative.is_absolute() and ".." not in relative.parts,
             f"Expected a relative contained file: {name!r}")
    path = (root / relative).resolve(strict=True)
    _require(path.is_relative_to(root.resolve()) and path.is_file(), f"File escapes source directory: {name}")
    return path


class PackagingError(ValueError):
    """The models do not satisfy the integration contract."""


def _named(values, label):
    result = {}
    for value in values:
        if not value.name or value.name in result:
            raise PackagingError(f"{label}: empty or duplicate name {value.name!r}")
        result[value.name] = value
    return result


def _compatible(reference, branch, label):
    if not reference.type.HasField("tensor_type") or not branch.type.HasField(
        "tensor_type"
    ):
        raise PackagingError(f"{label}: only tensor interfaces are supported")
    left, right = reference.type.tensor_type, branch.type.tensor_type
    if left.elem_type != right.elem_type:
        raise PackagingError(f"{label}: dtype differs")
    if not left.HasField("shape") or not right.HasField("shape"):
        raise PackagingError(f"{label}: tensor rank is unspecified")
    if len(left.shape.dim) != len(right.shape.dim):
        raise PackagingError(f"{label}: rank differs")
    for index, (expected, actual) in enumerate(
        zip(left.shape.dim, right.shape.dim, strict=True)
    ):
        if expected.HasField("dim_value"):
            if (
                not actual.HasField("dim_value")
                or actual.dim_value != expected.dim_value
            ):
                raise PackagingError(f"{label}: fixed dimension {index} differs")
        elif expected.dim_param and not actual.HasField("dim_value"):
            if actual.dim_param != expected.dim_param:
                raise PackagingError(f"{label}: symbolic dimension {index} differs")


def _prepare_branch(reference, branch, label):
    reference_inputs = _named(reference.input, "reference inputs")
    branch_inputs = _named(branch.input, f"{label} inputs")
    for name, value in branch_inputs.items():
        if name not in reference_inputs:
            raise PackagingError(f"{label}: extra input {name!r} cannot be captured")
        _compatible(reference_inputs[name], value, f"{label} input {name}")
    reference_outputs = _named(reference.output, "reference outputs")
    branch_outputs = _named(branch.output, f"{label} outputs")
    if branch_outputs.keys() != reference_outputs.keys():
        raise PackagingError(f"{label}: output names differ from reference")
    ordered = []
    for name, value in reference_outputs.items():
        _compatible(value, branch_outputs[name], f"{label} output {name}")
        ordered.append(copy.deepcopy(branch_outputs[name]))
    del branch.output[:]
    branch.output.extend(ordered)
    del branch.input[:]


def _graphs(graph) -> Iterator[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from _graphs(child)


def _tensors(graph) -> Iterator[onnx.TensorProto]:
    for child in _graphs(graph):
        if getattr(child, "sparse_initializer", []):
            raise PackagingError(
                "Sparse initializers require a separate external-data adapter"
            )
        yield from getattr(child, "initializer", [])
        for node in child.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    yield attribute.t
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    yield from attribute.tensors
                elif attribute.type in {
                    onnx.AttributeProto.SPARSE_TENSOR,
                    onnx.AttributeProto.SPARSE_TENSORS,
                }:
                    raise PackagingError(
                        "Sparse tensor attributes require a separate external-data adapter"
                    )


def _check_captures(graph, outer_names):
    available = (
        set(outer_names)
        | {value.name for value in graph.input}
        | {t.name for t in graph.initializer}
    )
    for node in graph.node:
        missing = {name for name in node.input if name and name not in available}
        if missing:
            raise PackagingError(
                f"Unresolved inputs for {node.name or node.op_type}: {sorted(missing)}"
            )
        for attribute in node.attribute:
            children = (
                [attribute.g]
                if attribute.type == onnx.AttributeProto.GRAPH
                else attribute.graphs
            )
            for child in children:
                _check_captures(child, available)
        available.update(name for name in node.output if name)
    if any(value.name not in available for value in graph.output):
        raise PackagingError(f"Unresolved graph output in {graph.name}")


def _external(tensor, source_dir):
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        return None
    info = {item.key: item.value for item in tensor.external_data}
    location = info.get("location", "")
    if not location or Path(location).is_absolute():
        raise PackagingError("External data must use a nonempty relative path")
    path = (source_dir / location).resolve(strict=True)
    if not path.is_relative_to(source_dir.resolve()) or not path.is_file():
        raise PackagingError(f"External data escapes its model directory: {location}")
    offset = int(info.get("offset", 0))
    length = int(info.get("length", path.stat().st_size - offset))
    if offset < 0 or length < 0 or offset + length > path.stat().st_size:
        raise PackagingError(f"Invalid external-data extent: {tensor.name}")
    return path, offset, length


def _fingerprint(tensor, source_dir):
    digest = hashlib.sha256()
    digest.update(f"{tensor.data_type}:{list(tensor.dims)}:".encode())
    external = _external(tensor, source_dir)
    if any(dimension < 0 for dimension in tensor.dims):
        raise PackagingError(f"Negative initializer dimension: {tensor.name}")
    if tensor.data_type == onnx.TensorProto.STRING:
        if external or tensor.HasField("raw_data"):
            raise PackagingError("String tensors must use string_data")
    else:
        dtype_name = onnx.TensorProto.DataType.Name(tensor.data_type)
        bits = (
            4
            if dtype_name in {"INT4", "UINT4", "FLOAT4E2M1"}
            else onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type).itemsize * 8
        )
        expected_bytes = (math.prod(tensor.dims) * bits + 7) // 8
        stored_bytes = (
            external[2]
            if external
            else len(tensor.raw_data)
            if tensor.HasField("raw_data")
            else expected_bytes
        )
        if stored_bytes != expected_bytes:
            raise PackagingError(
                f"Initializer byte length differs from dtype/shape: {tensor.name}"
            )
    if external:
        path, offset, remaining = external
        with path.open("rb") as handle:
            handle.seek(offset)
            while remaining:
                block = handle.read(min(remaining, 1024 * 1024))
                if not block:
                    raise PackagingError(f"Truncated external data: {path}")
                digest.update(block)
                remaining -= len(block)
    elif tensor.HasField("raw_data"):
        digest.update(tensor.raw_data)
    elif tensor.data_type == onnx.TensorProto.STRING:
        for value in tensor.string_data:
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
    else:
        dtype_name = onnx.TensorProto.DataType.Name(tensor.data_type)
        if dtype_name in {"INT4", "UINT4", "FLOAT4E2M1"}:
            raise PackagingError("Sub-byte typed-field tensors require raw_data")
        array = onnx.numpy_helper.to_array(tensor)
        digest.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest(), external


def _opsets_and_functions(models, paths):
    versions, functions, function_contents = {}, {}, {}
    for model, path in zip(models, paths, strict=True):
        for item in model.opset_import:
            domain = "" if item.domain in {"", "ai.onnx"} else item.domain
            if domain in versions and versions[domain] != item.version:
                raise PackagingError(f"Conflicting opset versions for {domain!r}")
            versions[domain] = item.version
        for function in model.functions:
            identity = function.domain, function.name, function.overload
            if identity in functions and functions[identity] != function:
                raise PackagingError(f"Conflicting function definition: {identity}")
            # Equal protobufs can still point to different bytes in their source directories.
            contents = [
                _fingerprint(tensor, path.parent)[0] for tensor in _tensors(function)
            ]
            if (
                identity in function_contents
                and function_contents[identity] != contents
            ):
                raise PackagingError(f"Conflicting function content: {identity}")
            function_contents[identity] = contents
            functions[identity] = function
    if versions.get("", 0) < 15:
        raise PackagingError("Shape start/end requires default opset >= 15")
    for model in models:
        for graph in _graphs(model.graph):
            for node in graph.node:
                domain = "" if node.domain in {"", "ai.onnx"} else node.domain
                if domain not in versions:
                    raise PackagingError(f"Missing opset import for {domain!r}")
                if node.domain == "ai.onnx":
                    node.domain = ""
    return [
        onnx.helper.make_opsetid(domain, version)
        for domain, version in versions.items()
    ], list(functions.values())


def _combine_branches(reference_path, prefill_path, token_path, output_dir, *, prefill_model=None,
                      combined_transform=None) -> Path:
    """Write a new model directory; never replace an existing path or source file.

    Interface, hashes, and collisions are checked before creating output_dir.
    An I/O/checker failure after directory creation leaves a partial, non-reusable result;
    the caller must choose a fresh path. External locations must stay within each
    source model's directory. Files are copied byte-for-byte, preserving extents.
    """
    paths = [
        Path(path).resolve(strict=True)
        for path in (reference_path, prefill_path, token_path)
    ]
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    _require(prefill_model is None or isinstance(prefill_model, onnx.ModelProto), "Expected a prefill ModelProto")
    models = [copy.deepcopy(prefill_model) if index == 1 and prefill_model is not None
              else onnx.load(path, load_external_data=False) for index, path in enumerate(paths)]
    for model in models:
        for graph in _graphs(model.graph):
            for index in reversed(range(len(graph.metadata_props))):
                if graph.metadata_props[index].key == "onnx_utils_load":
                    del graph.metadata_props[index]
    reference, prefill, token = models
    opsets, functions = _opsets_and_functions(models, paths)
    for model, label in [(prefill, "prefill"), (token, "token")]:
        _prepare_branch(reference.graph, model.graph, label)

    inputs = _named(reference.graph.input, "reference inputs")
    if "input_ids" not in inputs:
        raise PackagingError("A named input_ids tensor is required")
    input_type = inputs["input_ids"].type.tensor_type
    if len(input_type.shape.dim) != 2:
        raise PackagingError(
            "input_ids must be rank 2 with sequence length at axis 1"
        )
    reserved = {"branch_sequence_shape", "branch_one", "branch_is_token"}
    names = set(inputs) | {value.name for value in reference.graph.output}
    for model in (prefill, token):
        for graph in _graphs(model.graph):
            names.update(t.name for t in graph.initializer)
            names.update(
                name for node in graph.node for name in (*node.input, *node.output)
            )
    if reserved & names:
        raise PackagingError("The model uses a reserved integration tensor name")

    defaults = [
        copy.deepcopy(tensor)
        for tensor in reference.graph.initializer
        if tensor.name in inputs
    ]
    tensors = [(tensor, paths[0].parent) for tensor in defaults]
    tensors += [
        (tensor, path.parent)
        for model, path in zip((prefill, token), paths[1:], strict=True)
        for tensor in _tensors(model.graph)
    ]
    tensors += [
        (tensor, path.parent)
        for model, path in zip(models, paths, strict=True)
        for function in model.functions
        for tensor in _tensors(function)
    ]
    fingerprints, external_files, relocations = {}, {}, []
    for tensor, source_dir in tensors:
        fingerprint, external = _fingerprint(tensor, source_dir)
        if tensor.name:
            if tensor.name in fingerprints and fingerprints[tensor.name] != fingerprint:
                raise PackagingError(
                    f"Conflicting initializer content: {tensor.name}"
                )
            fingerprints[tensor.name] = fingerprint
        if external:
            path, _, _ = external
            if path not in external_files:
                external_files[path] = f"weights-{len(external_files):04d}.bin"
            relocations.append((tensor, external_files[path]))
    # Relocate before make_graph/make_node copy the TensorProto objects.
    for tensor, location in relocations:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = location
    nodes = [
        onnx.helper.make_node(
            "Shape", ["input_ids"], ["branch_sequence_shape"], start=1, end=2
        ),
        onnx.helper.make_node(
            "Constant",
            [],
            ["branch_one"],
            value=onnx.helper.make_tensor("", onnx.TensorProto.INT64, [1], [1]),
        ),
        onnx.helper.make_node(
            "Equal", ["branch_sequence_shape", "branch_one"], ["branch_is_token"]
        ),
        onnx.helper.make_node(
            "If",
            ["branch_is_token"],
            [value.name for value in reference.graph.output],
            then_branch=token.graph,
            else_branch=prefill.graph,
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "qwen35_branch_integration",
        reference.graph.input,
        reference.graph.output,
        initializer=defaults,
    )
    _check_captures(graph, set())
    graph.doc_string = reference.graph.doc_string
    graph.metadata_props.extend(reference.graph.metadata_props)
    combined = onnx.helper.make_model(graph, opset_imports=opsets, functions=functions)
    combined.ir_version = max(model.ir_version for model in models)
    for field in (
        "domain",
        "model_version",
        "doc_string",
        "producer_name",
        "producer_version",
    ):
        setattr(combined, field, getattr(reference, field))
    combined.metadata_props.extend(reference.metadata_props)
    if combined_transform is not None:
        combined = combined_transform(combined)
        _require(isinstance(combined, onnx.ModelProto), "Combined transform must return ModelProto")
        _check_captures(combined.graph, set())

    output_dir.mkdir(parents=False, exist_ok=False)
    for source, filename in external_files.items():
        _copy_independent(source, output_dir / filename)
    output_path = output_dir / "model.onnx"
    with output_path.open("xb") as handle:
        handle.write(combined.SerializeToString())
    return output_path


_ORT_SCHEMA_EXPORT_CODE = r'''
import json
import onnxruntime
from onnxruntime.capi import _pybind_state
schemas = [s for s in _pybind_state.get_all_operator_schema()
           if s.domain == "" and s.name == "SimplifiedLayerNormalization"]
if len(schemas) != 1:
    raise RuntimeError("Expected one installed ORT SimplifiedLayerNormalization schema")
s = schemas[0]
def formal(p):
    return dict(name=p.name, type=p.typeStr, option=str(p.option).split(".")[-1])
print(json.dumps(dict(onnxruntime_version=onnxruntime.__version__, schema=dict(
    name=s.name, domain=s.domain, since_version=s.since_version,
    min_input=s.min_input, max_input=s.max_input, min_output=s.min_output, max_output=s.max_output,
    inputs=[formal(p) for p in s.inputs], outputs=[formal(p) for p in s.outputs],
    attributes=[dict(name=n, type=str(a.type).split(".")[-1], required=a.required)
                for n,a in sorted(s.attributes.items())],
    type_constraints=[(c.type_param_str, c.allowed_type_strs, "") for c in s.type_constraints]
))))
'''


def _schema_subprocess(command, *, input_text=None):
    """Bounded CPU schema inspection; no session or inference is created."""
    completed = subprocess.run(
        command, input=input_text, text=True, capture_output=True, timeout=120,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
    )
    _require(completed.returncode == 0, "Schema checker failed: " + completed.stderr[-4000:])
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise PackagingError("Schema helper returned no valid JSON") from error


def _check_with_schema(model_path, snapshot):
    s = snapshot["schema"]
    _require((s["name"], s["domain"], s["since_version"]) ==
             ("SimplifiedLayerNormalization", "", 1), "Unexpected installed ORT schema")
    S = onnx.defs.OpSchema
    def formal(p):
        return S.FormalParameter(p["name"], p["type"], param_option=getattr(S.FormalParameterOption, p["option"]))
    schema = S(s["name"], s["domain"], s["since_version"],
               inputs=[formal(p) for p in s["inputs"]], outputs=[formal(p) for p in s["outputs"]],
               attributes=[S.Attribute(a["name"], getattr(S.AttrType, a["type"]), required=a["required"]) for a in s["attributes"]],
               type_constraints=s["type_constraints"])
    _require((schema.min_input, schema.max_input, schema.min_output, schema.max_output) ==
             (s["min_input"], s["max_input"], s["min_output"], s["max_output"]), "Native schema IO cardinality differs")
    _require(not onnx.defs.has(s["name"], s["since_version"], s["domain"]),
             "Fresh ONNX registry unexpectedly already contains this ORT-only schema")
    onnx.defs.register_schema(schema)
    onnx.checker.check_model(str(model_path), full_check=False)
    return dict(status="passed", full_check=False, model_sha256=_sha256(model_path),
                onnxruntime_version=snapshot["onnxruntime_version"],
                schema_sha256=hashlib.sha256(_json_bytes(snapshot)).hexdigest(),
                limitation="Native shape inference/default-value payloads are not transferred; custom domains are not shape-inferred.")


def check_model_with_installed_ort_schema(model_path: Path) -> dict:
    """Check a saved ONNX model in isolation using the installed ORT's metadata.

ORT schema export and ONNX checking use separate fresh CPU subprocesses because
importing both libraries can change the standard ONNX schema registry. No ORT
session, model, execution provider registration, or device initialization occurs.
"""
    model_path = Path(model_path).resolve(strict=True)
    snapshot = _schema_subprocess([sys.executable, "-B", "-c", _ORT_SCHEMA_EXPORT_CODE])
    return _schema_subprocess(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--check-schema", str(model_path)],
        input_text=json.dumps(snapshot),
    )


def _eager_header(header_path):
    """Parse the installed protobuf descriptor without importing SDK runtime code."""
    from google.protobuf import descriptor_pool, message_factory

    distribution = importlib.metadata.distribution("ryzenai-onnx-utils")
    candidates = [distribution.locate_file(path) for path in distribution.files or []
                  if str(path).replace("\\", "/").endswith("ryzenai_onnx_utils/proto/external_data_pb2.py")]
    _require(len(candidates) == 1, "Installed RyzenAI external-data protobuf descriptor not found")
    schema_path = Path(candidates[0])
    tree = ast.parse(schema_path.read_text(encoding="utf-8"))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "DESCRIPTOR" for t in node.targets)]
    _require(len(assignments) == 1, "Unexpected protobuf descriptor source")
    call = assignments[0].value
    _require(isinstance(call, ast.Call) and len(call.args) == 1
             and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, bytes),
             "Protobuf descriptor must be a bytes literal")
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(call.args[0].value)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("ryzenai.onnx_utils.proto.Header"))
    header = cls()
    header.ParseFromString(Path(header_path).read_bytes())
    return header, dict(distribution_version=distribution.version, descriptor_sha256=_sha256(schema_path))


def _copy_independent(source, destination):
    before = _stat(source)
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
            dst.write(block)
    _require(_stat(source) == before, f"Source changed during copy: {source.name}")
    _require(destination.stat().st_nlink == 1, "Destination must be an independent copy")
    _require(_sha256(destination) == digest.hexdigest(), "Copied data hash differs")
    return digest.hexdigest()


def _transaction_coverage(archive, eager_archive):
    """Keep every SDK entry and verify any existing eager subset byte-for-byte."""
    with zipfile.ZipFile(archive) as full:
        names = [entry.filename for entry in full.infolist() if not entry.is_dir()]
        _require(names and len(names) == len(set(names)), "Empty or duplicate SDK transaction archive")
        matched = []
        if eager_archive.is_file():
            with zipfile.ZipFile(eager_archive) as old:
                old_names = [entry.filename for entry in old.infolist() if not entry.is_dir()]
                _require(len(old_names) == len(set(old_names)), "Duplicate eager transaction entry")
                for name in sorted(old_names):
                    _require(name in names, "Eager transaction is absent from SDK archive: " + name)
                    digest = hashlib.sha256(old.read(name)).hexdigest()
                    _require(digest == hashlib.sha256(full.read(name)).hexdigest(), "SDK/eager transaction mismatch: " + name)
                    matched.append(dict(path=name, sha256=digest))
    return dict(entry_count=len(names), sha256=_sha256(archive),
                coverage_kind="Complete installed SDK transaction archive, no filtering",
                eager_subset_entries=matched,
                limitation="Archive completeness does not establish operator shape support or runtime correctness")


def package_model(prefill_dir: Path, token_dir: Path, output_dir: Path, *, sdk_root: Path, profile: dict,
                  prefill_linear_attention: str = "native", prefill_chunk_size: int | None = None,
                  prune_prefill_lm_head: bool = False) -> dict:
    """Create a fresh package from an eager prefill and a lowered DD token graph.

    ``profile['source_revision']`` is required. Optional ``expected_dd_nodes``
    and ``expected_dd_counts`` pin the number/types of metadata operations. SDK
    version/hash/source authorization belongs to the calling profile validator.
    ``prefill_linear_attention='token_loop'`` applies the guarded pure header
    transform before final copying. The default preserves native batched prefill.
    ``'adaptive'`` uses unchanged gates with adaptive native segments and a FLOAT
    sequence accumulator. Only this mode accepts a global chunk size, 1024
    (the default) or 4096; the source configuration must still use chunk64.
    ``prune_prefill_lm_head=True`` is an additional adaptive-only opt-in that
    projects the last prefill token while retaining every state update.
    Token inputs are ``model.onnx`` and ``cache/<node.name>_meta.json`` plus their
    referenced constants. This function never initializes/executes the runtime.
    """
    _require(prefill_linear_attention in {"native", "token_loop", "adaptive"}, "Unsupported prefill LinearAttention mode")
    _require(type(prune_prefill_lm_head) is bool, "prune_prefill_lm_head must be a boolean")
    _require(not prune_prefill_lm_head or prefill_linear_attention == "adaptive",
             "Prefill LM-head pruning requires adaptive mode")
    if prefill_linear_attention == "adaptive":
        prefill_chunk_size = 1024 if prefill_chunk_size is None else prefill_chunk_size
        _require(type(prefill_chunk_size) is int and prefill_chunk_size in (1024, 4096),
                 "Adaptive prefill chunk size must be 1024 or 4096")
    else:
        _require(prefill_chunk_size is None, "A prefill chunk override requires adaptive mode")
    prefill_dir, token_dir = (Path(p).resolve(strict=True) for p in (prefill_dir, token_dir))
    sdk_root, output_dir = Path(sdk_root).resolve(strict=True), Path(output_dir).absolute()
    _require(isinstance(profile.get("source_revision"), str) and profile["source_revision"], "Source revision is required")
    _require(not output_dir.exists() and not output_dir.is_symlink(), "Output directory must be new")
    _require(output_dir.parent.is_dir(), "Output parent must already exist")
    _require(not output_dir.resolve().is_relative_to(prefill_dir) and not output_dir.resolve().is_relative_to(token_dir),
             "Output must not be inside a source model")
    prefill_path, token_path = prefill_dir / "model.onnx", token_dir / "model.onnx"
    prefill, token = [onnx.load(p, load_external_data=False) for p in (prefill_path, token_path)]
    config_path = prefill_dir / "genai_config.json"
    config = _read_json(config_path)
    decoder = config["model"]["decoder"]
    providers = decoder["session_options"]["provider_options"]
    _require(len(providers) == 1 and set(providers[0]) == {"RyzenAI"}, "Expected a single RyzenAI provider")
    options = providers[0]["RyzenAI"]
    _require("custom_ops_library" not in decoder["session_options"], "Unexpected extra custom-op DLL registration")
    header_path = _relative_file(prefill_dir, options["external_data_file"])
    _require(header_path.parent == prefill_dir, "Eager header must reside at model root")
    header, descriptor_evidence = _eager_header(header_path)
    eager_weights = _relative_file(prefill_dir, header.external_data.filename)
    _require(eager_weights.parent == prefill_dir, "Eager weights must reside at model root")
    _require(header.external_data.npu and not header.external_data.gpu, "Expected NPU eager weights")
    eager_nodes = {n.name: n for g in _graphs(prefill.graph) for n in g.node if n.domain == "com.ryzenai" and n.op_type != "CastAvx"}
    _require(set(header.operators) == set(eager_nodes), "Eager header/operator-name coverage differs")
    for name, operator in header.operators.items():
        _require(operator.op_type == eager_nodes[name].op_type, "Eager header operator type differs")
        for tensor in operator.data:
            _require(0 <= tensor.offset <= tensor.offset + tensor.size <= eager_weights.stat().st_size,
                     "Eager external-data extent is out of bounds")
    prefill_transformation = None
    transformer_path = None
    transformation_dependencies = {}
    if prefill_linear_attention == "token_loop":
        _require(config["search"]["chunk_size"] == 64 and options.get("hybrid_opt_token_backend") == "npu",
                 "Token Loop requires unchanged chunk64 and NPU token backend")
        try:
            import benchmarks.qwen35_prefill as qwen35_prefill
        except ModuleNotFoundError:
            import qwen35_prefill
        transformer_path = Path(qwen35_prefill.__file__).resolve(strict=True)
        prefill, prefill_transformation = qwen35_prefill.transform_prefill_linear_attention(prefill)
        prefill_transformation["transformer_sha256"] = _sha256(transformer_path)
        prefill_transformation["observed_global_chunk_size"] = config["search"]["chunk_size"]
    elif prefill_linear_attention == "adaptive":
        _require(config["search"]["chunk_size"] == 64 and options.get("hybrid_opt_token_backend") == "npu",
                 "Adaptive prefill requires source chunk64 and NPU token backend")
        try:
            from benchmarks import qwen35_adaptive_prefill, qwen35_prefill
        except ModuleNotFoundError as error:
            if error.name != "benchmarks":
                raise
            import qwen35_adaptive_prefill
            import qwen35_prefill
        transformer_path = Path(qwen35_adaptive_prefill.__file__).resolve(strict=True)
        helper_path = Path(qwen35_prefill.__file__).resolve(strict=True)
        transformation_dependencies = {path: _sha256(path) for path in (transformer_path, helper_path)}
        prefill, prefill_transformation = qwen35_adaptive_prefill.transform_prefill_adaptive_attention(prefill)
        prefill_transformation.update(
            transformer_sha256=transformation_dependencies[transformer_path],
            transformation_dependencies=[dict(path=str(path), sha256=digest)
                                         for path, digest in transformation_dependencies.items()],
            source_global_chunk_size=config["search"]["chunk_size"],
            observed_global_chunk_size=prefill_chunk_size,
        )
        config["search"]["chunk_size"] = prefill_chunk_size
    lm_head_transformation = None
    combine_options = {}
    if prune_prefill_lm_head:
        try:
            from benchmarks import qwen35_lm_head
        except ModuleNotFoundError as error:
            if error.name != "benchmarks":
                raise
            import qwen35_lm_head
        lm_head_path = Path(qwen35_lm_head.__file__).resolve(strict=True)
        transformation_dependencies[lm_head_path] = _sha256(lm_head_path)
        def transform_combined(combined):
            nonlocal lm_head_transformation
            result, lm_head_transformation = qwen35_lm_head.transform_prefill_lm_head(combined)
            lm_head_transformation.update(transformer_path=str(lm_head_path),
                                          transformer_sha256=transformation_dependencies[lm_head_path])
            return result
        combine_options["combined_transform"] = transform_combined
    source_paths = {prefill_path, token_path, config_path, header_path, eager_weights}
    source_paths.update(transformation_dependencies)
    if transformer_path is not None:
        source_paths.add(transformer_path)
    metadata, constants, dd_types = [], {}, {}
    dd_names = set()
    cache = token_dir / "cache"
    for graph in _graphs(token.graph):
        for node in graph.node:
            if node.domain != "com.ryzenai" or node.op_type != "DynamicDispatch":
                continue
            _require(node.name and node.name not in dd_names and Path(node.name).name == node.name
                     and not any(c in node.name for c in "\\/"), "DD node needs a unique safe filename")
            dd_names.add(node.name)
            attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            _require(attributes.get("model_type") == 9 and attributes.get("input_num") in (1, 6), "Unexpected DD wrapper model/input count")
            _require(len(node.input) == attributes["input_num"] + 1
                     and node.input[-1] == "attention_mask_const_uint", "DD host count contract differs")
            path = _relative_file(cache, node.name + "_meta.json")
            source_paths.add(path)
            meta = _read_json(path)
            _require(not meta.get("state_table_updates") and not meta.get("aux_info", {}).get("is_llm", False),
                     "Unexpected generic DD state/scratch contract")
            _require(len(meta["op_list"]) == 1, "Expected one operation per DD partition")
            op_type = meta["op_list"][0]["type"]
            _require(op_type in {"MladfMatMul", "linear_attention_token"}, "Unsupported DD metadata operator")
            _require(len(meta["fused_tensors"]["in"]["packed_tensors"]) == attributes["input_num"], "DD metadata input count differs")
            for direction, names, prefix in (("in", node.input[:-1], "input"), ("out", node.output, "output")):
                packed = meta["fused_tensors"][direction]["packed_tensors"]
                _require(list(packed) == list(names), "DD wrapper/metadata tensor order differs")
                for index, tensor_name in enumerate(packed):
                    _require(attributes.get(f"{prefix}_shape_{index}") == meta["tensor_map"][tensor_name]["shape"],
                             "DD wrapper/metadata tensor shape differs")
            _require(attributes.get(f"input_shape_{attributes['input_num']}") == [1], "DD host count shape differs")
            dd_types[op_type] = dd_types.get(op_type, 0) + 1
            for tensor in meta["tensor_map"].values():
                if "file_name" not in tensor:
                    continue
                path_value = Path(tensor["file_name"])
                original = path_value.resolve(strict=True) if path_value.is_absolute() else _relative_file(cache, tensor["file_name"])
                _require(original.is_file() and original.parent == cache.resolve(), "DD constant escapes token cache")
                _require(original.stat().st_size == tensor["file_size"], "DD constant size differs")
                _require(original.name not in constants or constants[original.name] == original, "DD constant filename collision")
                constants[original.name] = original
                source_paths.add(original)
                tensor["file_name"] = str(output_dir / "cache" / original.name)
            metadata.append((node.name, path, meta))
    _require(metadata, "Token graph has no DD nodes")
    if "expected_dd_nodes" in profile:
        _require(len(metadata) == profile["expected_dd_nodes"], "DD node count differs from profile")
    if "expected_dd_counts" in profile:
        _require(dd_types == profile["expected_dd_counts"], "DD operator counts differ from profile")
    archive = sdk_root / "deployment/dyn_bins.zip"
    source_paths.add(archive)
    eager_archive = prefill_dir / "cache/txn_bins.zip"
    if eager_archive.exists():
        source_paths.add(eager_archive)
    transactions = _transaction_coverage(archive, eager_archive)
    # Include every external TensorProto source in immutability checks.
    for model, folder in ((prefill, prefill_dir), (token, token_dir)):
        for scope in (model.graph, *model.functions):
            for tensor in _tensors(scope):
                external = _external(tensor, folder)
                if external:
                    source_paths.add(external[0])
    support_files = [prefill_dir / name for name in
                     ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")
                     if (prefill_dir / name).is_file()]
    _require({p.name for p in support_files} >= {"config.json", "tokenizer.json", "tokenizer_config.json"}, "Missing tokenizer/config files")
    source_paths.update(support_files)
    before = {p: _stat(p) for p in source_paths}
    hashes = {p: _sha256(p) for p in (prefill_path, token_path, config_path, header_path)}
    hashes.update(transformation_dependencies)
    if prefill_transformation is None:
        _combine_branches(prefill_path, prefill_path, token_path, output_dir, **combine_options)
    else:
        prefill_transformation["source_model_sha256"] = hashes[prefill_path]
        _combine_branches(prefill_path, prefill_path, token_path, output_dir, prefill_model=prefill, **combine_options)
    (output_dir / "cache").mkdir(exist_ok=False)
    roles = {"model.onnx": "combined_header"}
    for path in output_dir.glob("weights-*.bin"):
        roles[path.name] = "onnx_external_tensor"
        _require(path.stat().st_nlink == 1, "ONNX external data must be independently copied")
    for source in [header_path, eager_weights, *support_files]:
        _copy_independent(source, output_dir / source.name)
        roles[source.name] = "eager_data" if source == eager_weights else "eager_metadata_or_tokenizer"
    for name, source in sorted(constants.items()):
        _copy_independent(source, output_dir / "cache" / name)
        roles["cache/" + name] = "dd_constant"
    for name, _, meta in metadata:
        relative = "cache/" + name + "_meta.json"
        _write_new(output_dir / relative, meta)
        roles[relative] = "dd_metadata"
    _require(_copy_independent(archive, output_dir / "cache/txn_bins.zip") == transactions["sha256"], "SDK archive changed")
    roles["cache/txn_bins.zip"] = "complete_sdk_transactions"
    decoder["filename"] = "model.onnx"
    options.update(dd_cache=(output_dir / "cache").as_posix(), compile_fusion_rt="1", onnx_custom_ops_const_key="")
    decoder["session_options"]["graph_optimization_level"] = "ORT_DISABLE_ALL"
    _write_new(output_dir / "genai_config.json", config)
    roles["genai_config.json"] = "genai_config"
    checker = check_model_with_installed_ort_schema(output_dir / "model.onnx")
    for path, original in before.items():
        _require(_stat(path) == original, "Source changed while packaging: " + path.name)
    for path, digest in hashes.items():
        _require(_sha256(path) == digest, "Source header/config changed while packaging")
    manifest = dict(
        schema_version=1, status="materialized_cpu_checked", package_root=str(output_dir),
        source_models=dict(source_revision=profile["source_revision"],
                           prefill=dict(path=str(prefill_path), sha256=hashes[prefill_path]),
                           token=dict(path=str(token_path), sha256=hashes[token_path])),
        artifacts=[dict(path=name, size=(output_dir / name).stat().st_size, sha256=_sha256(output_dir / name), role=role)
                   for name, role in sorted(roles.items())],
        metadata_count=len(metadata), dd_operator_counts=dd_types,
        transaction_archive=dict(path="cache/txn_bins.zip", **transactions),
        eager_protobuf=descriptor_evidence, checker=checker,
        source_immutability_rechecked=True, runtime_unverified=True,
        limitations=["No inference, numerical correctness, 16K execution or shared-context resource claim",
                     "Initial native runtime compilation is required; no serialized state is guessed or renamed",
                     "DD cache/constant paths are absolute; moving the package requires a new relocation and manifest"],
    )
    if prefill_transformation is not None:
        manifest["prefill_transformation"] = prefill_transformation
    if lm_head_transformation is not None:
        manifest["prefill_lm_head_pruning"] = lm_head_transformation
    _write_new(output_dir / "package-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--check-schema":
        raise SystemExit("This module is a library; use the Qwen3.5 preparation command.")
    print(json.dumps(_check_with_schema(Path(sys.argv[2]).resolve(strict=True), json.load(sys.stdin))))
