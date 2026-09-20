"""Verify compact owned synthetic evidence; optionally materialize runner fixtures.

No ONNX runtime, SDK import, model activation, or NPU call is used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402 -- limit BLAS threads before importing NumPy


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def raw_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def decode(array):
    require(array.dtype == np.uint16, "Expected raw BF16")
    return (array.astype(np.uint32) << 16).view(np.float32)


def encode(array):
    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def generate(gate, length):
    require(gate in (-4, -80, -128) and length in (1, 64), "Unknown fixture")
    rng = np.random.default_rng(20260920)
    q = rng.normal(size=(1, 64, 16, 128)).astype(np.float32)
    k = rng.normal(size=(1, 64, 16, 128)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True) * np.float32(np.sqrt(128))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.uniform(-.125, .125, size=(1, 64, 4096)).astype(np.float32)
    arrays = dict(query=encode(q.reshape(1, 64, 2048)), key=encode(k.reshape(1, 64, 2048)),
                  value=encode(v), state=np.zeros((1, 32, 128, 128), np.uint16),
                  log_gate=encode(np.full((1, 64, 32), gate, np.float32)),
                  beta=encode(np.full((1, 64, 32), .5, np.float32)))
    return {name: value if name == "state" else np.ascontiguousarray(value[:, :length])
            for name, value in arrays.items()}


def reference(raw, round_state=False):
    """Stable FP64 K-major recurrence; no cumulative-exp division."""
    length = raw["query"].shape[1]
    q = decode(raw["query"]).astype(np.float64).reshape(length, 16, 128)
    k = decode(raw["key"]).astype(np.float64).reshape(length, 16, 128)
    v = decode(raw["value"]).astype(np.float64).reshape(length, 32, 128)
    g = decode(raw["log_gate"]).astype(np.float64).reshape(length, 32)
    beta = decode(raw["beta"]).astype(np.float64).reshape(length, 32)
    state = decode(raw["state"]).astype(np.float64).reshape(32, 128, 128)
    output = np.empty((length, 32, 128), np.float64)
    for t in range(length):
        qt, kt = np.repeat(q[t], 2, axis=0), np.repeat(k[t], 2, axis=0)
        state *= np.exp(g[t])[:, None, None]
        correction = beta[t, :, None] * (v[t] - np.einsum("hkv,hk->hv", state, kt, optimize=False))
        state += np.einsum("hk,hv->hkv", kt, correction, optimize=False)
        if round_state:
            state = decode(encode(state)).astype(np.float64)
        output[t] = np.einsum("hkv,hk->hv", state, qt, optimize=False)
        require(np.isfinite(state).all() and np.isfinite(output[t]).all(), "Nonfinite CPU reference")
    return dict(attention=output.reshape(1, length, 4096), state_out=state.reshape(1, 32, 128, 128))


def metric(raw, ref):
    with np.errstate(invalid="ignore"):
        actual = decode(raw).astype(np.float64)
    row = dict(elements=actual.size, finite_count=int(np.isfinite(actual).sum()),
               nan_count=int(np.isnan(actual).sum()), positive_inf_count=int(np.isposinf(actual).sum()),
               negative_inf_count=int(np.isneginf(actual).sum()), all_finite=bool(np.isfinite(actual).all()))
    if row["all_finite"]:
        diff = actual - ref
        rmse = float(np.sqrt(np.mean(diff**2)))
        row.update(max_abs_error=float(np.abs(diff).max()), rmse=rmse,
                   relative_rmse=rmse / float(np.sqrt(np.mean(ref**2))))
    return row


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        path = root / name
        require(Path(name).name == name, "Unexpected path")
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], f"Changed {name}")
    evidence = json.loads((root / "evidence.json").read_text())
    references = {}
    report = {"status": "cpu_verified", "npu_calls": 0, "numpy": np.__version__, "cases": {}}
    with zipfile.ZipFile(root / "synthetic-observed.zip") as archive:
        require(sum(item.file_size for item in archive.infolist()) <= 16 * 1024**2, "Unexpected archive expansion")
        require(set(archive.namelist()) == set(evidence["observation_members"]), "Unexpected archive members")
        for case_id, case in evidence["synthetic"].items():
            raw = generate(case["log_gate"], case["sequence_length"])
            for role, digest in case["inputs"].items():
                require(raw_sha(raw[role]) == digest, f"Seed/input mismatch: {case_id}/{role}")
            key = (case["log_gate"], case["sequence_length"])
            if key not in references:
                primary = reference(raw)
                secondary = reference(raw, True)
                references[key] = {**primary, **{f"bf16_each_step_{k}": v for k, v in secondary.items()}}
            refs = references[key]
            for role, digest in case["references"].items():
                require(raw_sha(refs[role]) == digest, f"FP64 reference mismatch: {case_id}/{role}")
            for role, digest in case["observations"].items():
                info = evidence["arrays"][digest]
                payload = archive.read(f"{digest}.bf16")
                require(len(payload) == info["bytes"] and hashlib.sha256(payload).hexdigest() == digest, "Corrupt raw observation")
                actual = np.frombuffer(payload, dtype="<u2").reshape(info["shape"])
                for prefix in ("", "bf16_each_step_"):
                    measured = metric(actual, refs[prefix + role])
                    require(measured == case["metrics"][prefix + role], f"Metric mismatch: {case_id}/{prefix}{role}")
            report["cases"][case_id] = {"inputs_exact": True, "references_exact": True,
                                         "observed_bits_exact": True, "metrics_recomputed": True}
    report["captured_la14"] = "Metadata-only provenance; private arrays intentionally unavailable for public numerical recomputation."
    return evidence, report


def prepare(root, output, evidence):
    """Expand five supported runner cases locally; generated tensors are not publication files."""
    output.mkdir(parents=True, exist_ok=False)
    mappings = dict(batched_m4="native_s64_m4", batched_m128="native_s64_m128",
                    loop_m4="loop_s64_m4", loop_m128="loop_s64_m128",
                    no_cast_hints_m128="native_s64_m128_no_cast_hints")
    manifests, shared = {}, {}
    for runner_id, case_id in mappings.items():
        case = evidence["synthetic"][case_id]
        gate = case["log_gate"]
        if gate not in shared:
            raw = generate(gate, 64)
            refs = reference(raw)
            paths = {}
            for role, arrays in (("inputs", raw), ("reference", refs)):
                path = output / f"m{abs(gate)}-{role}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                paths[role] = path
            shared[gate] = paths
        model = output / case["model"]
        if not model.exists():
            with (root / case["model"]).open("rb") as src, model.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        paths = {**shared[gate], "model": model}
        manifests[runner_id] = {"token_backend": case["token_backend"],
                                "npu_commands": case["npu_counters"]["command_submissions"],
                                "source_case": case_id,
                                "artifacts": {role: {"path": path.name, "bytes": path.stat().st_size,
                                                       "sha256": sha(path)} for role, path in paths.items()}}
    with (root / "repro_native.py").open("rb") as src, (output / "repro_native.py").open("xb") as dst:
        shutil.copyfileobj(src, dst)
    write(output / "repro-fixtures.json", {"schema_version": 1, "cases": manifests})
    write(output / "source-association.json", {"source_evidence_sha256": sha(root / "evidence.json"),
          "public_runner_hardware_validated": False, "runner_sha256": sha(output / "repro_native.py"),
          "materialization_npu_calls": 0,
          "unsupported_in_this_runner": ["native_s64_m80", "native_s1_m128"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--prepare", type=Path, help="Create local inputs/references for the unchanged five-case native runner")
    args = parser.parse_args()
    root = args.root.resolve()
    evidence, report = verify(root)
    if args.prepare:
        prepare(root, args.prepare.resolve(), evidence)
    if args.report:
        write(args.report.resolve(), report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
