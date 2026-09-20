"""Owned seven-token CPU model: inspect terminal EOS in get_sequence.

CPUExecutionProvider only, no AMD provider or original model/runtime changes.
The graph predicts input_id+1, so prompt ending in2 produces3,4,5,6(EOS).
Final sequences from the earlier NPU run remain unknown; these are CPU toys.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REVISION = "b7a6ec307bea84e3b64aa33d59bcad817122d9af"
SOURCES = ("src/search.cpp", "src/search.h", "src/sequences.h", "src/sequences.cpp",
           "src/generators.cpp", "src/ort_genai_c.cpp", "src/python/python.cpp")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    output = HERE / "cpu-run1"
    output.mkdir(exist_ok=False)
    source_dir = output / "public-source"
    source_dir.mkdir()
    source_hashes = {}
    for name in SOURCES:
        raw = subprocess.check_output(["git", "show", REVISION + ":" + name], cwd=ROOT / "cache/upstream/onnxruntime-genai")
        with (source_dir / name.replace("/", "_")).open("xb") as stream:
            stream.write(raw)
        source_hashes[name] = digest(raw)
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    import onnxruntime_genai as oga
    nodes = [h.make_node("Add", ["input_ids", "one"], ["next_ids"]),
             h.make_node("OneHot", ["next_ids", "depth", "scores"], ["logits"], axis=-1)]
    model_proto = h.make_model(h.make_graph(nodes, "owned_cpu_eos_sequence",
        [h.make_tensor_value_info("input_ids", T.INT64, [1, "sequence_length"]),
         h.make_tensor_value_info("position_ids", T.INT64, [1, "sequence_length"]),
         h.make_tensor_value_info("attention_mask", T.INT64, [1, "total_sequence_length"])],
        [h.make_tensor_value_info("logits", T.FLOAT, [1, "sequence_length", 7])],
        [nh.from_array(np.array(1, dtype=np.int64), "one"),
         nh.from_array(np.array(7, dtype=np.int64), "depth"),
         nh.from_array(np.array([-10, 10], dtype=np.float32), "scores")]),
        opset_imports=[h.make_opsetid("", 17)])
    model_proto.ir_version = 10
    onnx.checker.check_model(model_proto)
    base = json.loads((ROOT / "cache/qwen35-conversion/chunk-api-audit-run1/tiny-cpu-model/genai_config.json").read_text())
    assert base["model"]["decoder"]["session_options"]["provider_options"] == []
    base["model"]["context_length"] = 32768
    base["search"]["max_length"] = 32768
    report = dict(status="running_cpu_toys", python=sys.version,
                  packages={k: version(k) for k in ("numpy", "onnx", "onnxruntime-genai-directml-ryzenai", "onnxruntime-vitisai")},
                  probe_sha256=sha(__file__), public_source_revision=REVISION, public_source_sha256=source_hashes,
                  original_fixture_config_sha256=sha(ROOT / "cache/qwen35-conversion/chunk-api-audit-run1/tiny-cpu-model/genai_config.json"),
                  npu_sessions=0, original_model_reads=0, cases=[],
                  qualification="Owned CPU toys only. The failed NPU run's fourth returned sequence was not saved and is not reconstructed here.")
    cases = (("eos_at_small_boundary", 12, 16, 6),
             ("eos_before_small_boundary", 12, 32, 6),
             ("non_eos_at_small_boundary", 12, 16, 0),
             ("non_eos_earlier_max", 12, 15, 6),
             ("eos_at_16k_candidate_boundary", 16380, 16384, 6),
             ("eos_before_16k_candidate_boundary", 16380, 16385, 6),
             ("non_eos_at_16k_candidate_boundary", 16380, 16384, 0))
    for label, count, limit, eos in cases:
        folder = output / label
        folder.mkdir()
        config = json.loads(json.dumps(base))
        config["model"]["eos_token_id"] = eos
        with (folder / "model.onnx").open("xb") as stream:
            stream.write(model_proto.SerializeToString())
        write(folder / "genai_config.json", config)
        model = oga.Model(str(folder))
        assert model.device_type == "CPU", model.device_type
        params = oga.GeneratorParams(model)
        params.set_search_options(max_length=limit, batch_size=1, do_sample=False)
        generator = oga.Generator(model, params)
        ids = [2] * count
        generator.append_tokens(np.asarray(ids, dtype=np.int32))
        record = dict(label=label, input_tokens=count, configured_max_length=limit, configured_eos_id=eos,
                      device_type=model.device_type, provider_options=[], model_sha256=sha(folder / "model.onnx"),
                      config_sha256=sha(folder / "genai_config.json"),
                      input_ids_sha256=digest(json.dumps(ids).encode()), steps=[], logits_calls=0, samples=0)
        report["cases"].append(record)
        generated = []
        for index in range(4):
            if generator.is_done():
                break
            logits = np.asarray(generator.get_logits()).copy()
            record["logits_calls"] += 1
            assert logits.shape == (1, 1, 7) and np.isfinite(logits).all()
            generator.generate_next_token()
            record["samples"] += 1
            next_ids = np.asarray(generator.get_next_tokens()).copy().tolist()
            assert len(next_ids) == 1
            generated.extend(next_ids)
            before_done = np.asarray(generator.get_sequence(0)).copy()
            before_count = generator.token_count()
            done = bool(generator.is_done())
            after_done = np.asarray(generator.get_sequence(0)).copy()
            after_count = generator.token_count()
            is_eos = next_ids[0] == eos
            expected = ids + (generated[:-1] if is_eos else generated)
            sequence_paths = []
            for tag, array in (("before-is-done", before_done), ("after-is-done", after_done)):
                path = folder / f"step{index+1}-{tag}.npy"
                with path.open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                sequence_paths.append(dict(file=str(path.relative_to(output)).replace("\\", "/"), sha256=sha(path)))
            step = dict(index=index+1, get_next_tokens=next_ids, all_sampled_ids=list(generated),
                        configured_eos_sampled=is_eos, is_done=done,
                        before_is_done_sequence_length=int(before_done.size), after_is_done_sequence_length=int(after_done.size),
                        token_count_before_is_done=before_count, token_count_after_is_done=after_count,
                        sequence_unchanged_by_is_done=bool(np.array_equal(before_done, after_done)),
                        sequence_equals_prompt_plus_all_sampled=bool(np.array_equal(before_done, ids+generated)),
                        sequence_equals_prompt_plus_sampled_without_final_eos=bool(np.array_equal(before_done, expected)),
                        last_8_returned_ids=before_done[-8:].tolist(), saved_sequences=sequence_paths)
            record["steps"].append(step)
            assert np.array_equal(before_done, after_done)
            assert before_done.tolist() == expected
            assert before_count == after_count == len(expected)
            if is_eos:
                assert done
                break
            if done:
                assert len(expected) == limit
                break
        record.update(generated_ids=generated, final_returned_sequence_length=record["steps"][-1]["after_is_done_sequence_length"],
                      final_is_done=record["steps"][-1]["is_done"], no_logits_after_done=True)
        assert record["final_is_done"]
        generator = params = model = None
    report["status"] = "cpu_toys_confirm_eos_excluded_from_sequence"
    write(output / "report.json", report)
    print(json.dumps(dict(status=report["status"], report_sha256=sha(output / "report.json"),
                         cases=[{k:c[k] for k in ("label", "generated_ids", "final_returned_sequence_length")} for c in report["cases"]])))


if __name__ == "__main__":
    main()
