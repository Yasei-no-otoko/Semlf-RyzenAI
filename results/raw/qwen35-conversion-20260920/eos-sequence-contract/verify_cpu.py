"""Verify saved owned CPU sequences; no model/runtime/SDK execution."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, row in manifest["files"].items():
        data = (root / name).read_bytes()
        require(sha(data) == row["sha256"] and len(data) == row["bytes"], "Publication file changed")
    report = json.loads((root / "cpu-report.json").read_text())
    require(report["status"] == "cpu_toys_confirm_eos_excluded_from_sequence", "Wrong report")
    require(report["npu_sessions"] == report["original_model_reads"] == 0 and len(report["cases"]) == 7, "Wrong scope")
    arrays = 0
    rows = []
    with zipfile.ZipFile(root / "owned-fixtures-and-sequences.zip") as archive:
        require(set(archive.namelist()) == set(manifest["archive_members"]), "Unexpected archive member")
        require(sum(item.file_size for item in archive.infolist()) < 4 * 1024**2, "Archive expansion exceeds bound")
        for item in archive.infolist():
            require(item.date_time == (1980, 1, 1, 0, 0, 0), "Nondeterministic ZIP time")
            data = archive.read(item.filename)
            expected = manifest["archive_members"][item.filename]
            require(sha(data) == expected["sha256"] and len(data) == expected["bytes"], "Archived source changed")
        for case in report["cases"]:
            require(case["device_type"] == "CPU" and case["provider_options"] == [], "Non-CPU model")
            prompt = [2] * case["input_tokens"]
            require(sha(json.dumps(prompt).encode()) == case["input_ids_sha256"], "Owned input mismatch")
            require(len(case["steps"]) == case["samples"] == case["logits_calls"], "Call counts differ")
            sampled = []
            for index, step in enumerate(case["steps"], 1):
                require(step["index"] == index and step["get_next_tokens"] == [index + 2], "Unexpected toy prediction")
                sampled += step["get_next_tokens"]
                require(step["all_sampled_ids"] == sampled, "Sample list mismatch")
                eos = sampled[-1] == case["configured_eos_id"]
                expected = prompt + (sampled[:-1] if eos else sampled)
                require(step["configured_eos_sampled"] is eos and step["sequence_equals_prompt_plus_all_sampled"] is (not eos), "EOS assertion mismatch")
                require(step["sequence_equals_prompt_plus_sampled_without_final_eos"] and step["sequence_unchanged_by_is_done"], "Sequence invariant differs")
                require(len(step["saved_sequences"]) == 2, "Need before/after arrays")
                for stored in step["saved_sequences"]:
                    data = archive.read(stored["file"])
                    require(sha(data) == stored["sha256"], "Recorded array SHA mismatch")
                    array = np.load(io.BytesIO(data), allow_pickle=False)
                    require(array.ndim == 1 and array.dtype.kind in "iu" and array.tolist() == expected, "Saved sequence differs")
                    arrays += 1
                for name in ("before_is_done_sequence_length", "after_is_done_sequence_length", "token_count_before_is_done", "token_count_after_is_done"):
                    require(step[name] == len(expected), "API length/count mismatch")
                require(step["last_8_returned_ids"] == expected[-8:], "Saved tail mismatch")
                require(step["is_done"] is (eos or len(expected) == case["configured_max_length"]), "Done condition mismatch")
            require(case["generated_ids"] == sampled and case["final_is_done"] and case["no_logits_after_done"], "Final metadata differs")
            require(case["final_returned_sequence_length"] == len(expected), "Final length differs")
            rows.append({"label": case["label"], "saved_arrays_verified": 2 * len(case["steps"]), "final_returned_length": len(expected)})
    require(arrays == 54, "Expected all 54 sequence arrays")
    return {"status": "saved_cpu_sequences_verified", "cases": rows, "saved_arrays": arrays,
            "new_cpu_model_sessions": 0, "npu_sessions": 0, "failed_npu_fourth_sequence_reconstructed": False}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent)))
