# Qwen3.5-4B conversion for Ryzen AI 1.8

**Experimental and incomplete.** Quark quantization and OGA export completed.
A custom NPU-eager artifact scores three short English/Japanese examples with
finite, correct outputs. **Token Fusion is unsupported by the installed SDK
recipe, and 16K inference has not completed successfully.** This is not a
replacement for the working AMD Qwen3-4B demo model.

The pinned source is [Qwen/Qwen3.5-4B at
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a).
The requested reference, [AMD's Qwen3-4B NPU 16K
model](https://huggingface.co/amd/Qwen3-4B_rai_1.8.0_npu_16K), uses AWQ and
Token Fusion. This custom conversion uses **Quark 0.11 UINT4 RTN/MinMax,
asymmetric group size 128**, not AWQ. It has no activation or KV-cache
quantization calibration. It does not inherit the reference model's quality
or speed results.

## What has been verified

| Stage | Result |
|---|---|
| Quark weight-only quantization | Complete; 249 packed projections |
| Public OGA 0.14 export | Complete; repaired graph loads on CPU |
| CPU direct scoring | Three owned short examples are finite and correct |
| SDK Token Fusion | Failed: missing Qwen3.5 DD pattern and unsupported projection shapes |
| SDK NPU eager, chunk size 4096 | English short example passes; Japanese 96-token example returns NaNs |
| SDK NPU eager, chunk size 64 | Three owned short examples pass, including Japanese |
| Native generation | Finite logits observed, but the original validator checked logits after EOS; normal termination is not validated |
| 16,384-token prefill | Two attempts interrupted by host reboots; no completed result |
| Full scripted eager post-processing rerun | Interrupted by the second reboot |

The candidate graph contains 1,398 nodes. Its NPU operators include 153
`MatMulNBitsBf`, 32 `SSMLP` groups (covering the remaining 96 heavy matmuls),
and 24 `LinearAttention` operations. The 24 `CausalConvWithState` and 8 GQA
operations remain on CPU, with other host operations and casts. No GPU
provider is configured. The 64 recurrent/conv/KV state inputs are float16 at
the graph boundary. The final artifact is text-only: vision and MTP are not
exported, and embeddings and depthwise convolution weights remain raw.

The chunk-4096 Japanese failure first appears in layer 8 `LinearAttention`,
in one of 32 state heads. CPU evaluation of the same layer remains finite.
Chunk size 64 avoids the observed short-example failure; this does not prove
stability for arbitrary inputs or long contexts.

Both long runs reached the 16,384-token tail-sentinel stage, but neither
produced a final result. Windows recorded unexpected reboots at approximately
00:47 and 00:59 JST on 2026-09-20, with preceding WHEA corrected
Bus/Interconnect errors. This correlation does not establish a root cause.
Further long hardware runs require resolving the host instability first.

The owned evidence and exact artifact hashes are in
[the conversion evidence](../results/raw/qwen35-conversion-20260920/evidence.json).
Weights, caches, machine-identifying hardware reports, and third-party data
are not committed.

## Why Token Fusion did not complete

The installed SDK Qwen3.5 strategy specializes the eager path. Its generic
`llm.dd_graph.llm_token_to_dd` pass recognizes supported MLADF/MHA and LFM2
patterns, but has no partition/state contract for Qwen3.5's recurrent
`LinearAttention`, convolution states, and CPU GQA boundaries. It finds zero
valid token partitions. Weight prepacking also rejects decode projections
with `(K, N)` equal to `(2560, 32)`, `(2560, 8192)`, and `(9216, 2560)`.

Removing the partition assertion or changing a context-length setting would
not supply these missing compiled operator/transaction contracts.
`split_dd_fusion` operates after a valid partition is found. The available
SDK cannot produce the requested Qwen3.5 Token Fusion artifact with a
Python-only post-processing change. NPU eager is a separate experimental
route, not a Token Fusion result.

## Reproducible conversion environments

Run from the repository root. Keep the conversion environment separate from
the Ryzen AI runtime: SDK 1.8 VOE pins Transformers 4.57.6, whereas the public
OGA Qwen3.5 conversion below uses Transformers 5.17.0.

Create a new ignored environment with the SDK's Python 3.12, install
`pip install -e '.[test]'`, and use these exact conversion versions:

```text
amd-quark 0.11
Python 3.12.11
torch 2.8.0+cpu
transformers 5.17.0
onnxruntime-genai 0.14.0 (public Microsoft package)
onnxruntime 1.27.0
onnx 1.19.0
onnx-ir 1.0.0
numpy 1.26.4
ml-dtypes 0.5.4
scipy 1.15.3
pandas 2.2.3
accelerate 1.12.0
safetensors 0.8.0
tokenizers 0.23.2
```

Quark is from [AMD's 0.11 archive](https://download.amd.com/opendownload/Quark/amd_quark-0.11.zip).
The reviewed wheel SHA256 is
`2af8a359ed574cecb9cbfdb33134199fcd57ba2403f1030d718285a2e19df26e`.
Use `pip check` after installation. The current local conversion Python is
`cache/envs/qwen35-convert/Scripts/python.exe`.

Download the pinned source revision separately to
`models/Qwen3.5-4B-source`; preserve its configuration and tokenizer files.
Do not commit source weights. All conversion output directories must be new.

```powershell
$convertPython = 'cache\envs\qwen35-convert\Scripts\python.exe'
$revision = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
$quantized = 'models\Qwen3.5-4B-quark-new'
$oga = 'models\Qwen3.5-4B-oga-new'
$env:MAX_JOBS = '32'
$env:CMAKE_BUILD_PARALLEL_LEVEL = '32'
$env:OMP_NUM_THREADS = '32'
$env:MKL_NUM_THREADS = '32'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'

& $convertPython benchmarks\quantize_qwen35.py `
  --source models\Qwen3.5-4B-source --output $quantized `
  --revision $revision --device cpu --torch-threads 32
& $convertPython benchmarks\export_qwen35.py `
  --source $quantized --output $oga --revision $revision
```

Quark disables its `torch.compile` path by default, but native startup
compilation may still require the MSVC x64 developer shell. Set both build
parallelism variables to 32. The exporter verifies the reviewed OGA loader
hash before applying a process-local Qwen3.5 adapter and bounded UINT4
packing. It also repairs only the observed `MatMulNBits` producer-name
mismatch; it refuses collisions or unresolved graph inputs. Neither stage
modifies installed SDK packages.

The OGA builder uses its INT4/DML export schema for the intermediate. The
intermediate was tested with CPUExecutionProvider; GPU inference was not
used. SDK post-processing supplies the final RyzenAI execution configuration.

The quantizer records its exclusion policy and source/output hashes. The
first produced manifest listed only vision/MTP exclusions; an artifact audit
also verified the raw embedding and convolution weights. Future manifests
record all four exclusions explicitly. New conversions verify the pinned
source configuration and both weight shards. The exporter verifies the
Quark manifest against the actual inputs and writes `export-manifest.json`
binding those inputs, the reviewed loader, and the exported ONNX files.

## Experimental eager post-processing

This command expresses the intended reproducible eager route. Its full
rerun was interrupted; it is not a completed Token Fusion procedure.

```powershell
$sdkPython = Join-Path $env:USERPROFILE 'miniforge3\envs\ryzen-ai-1.8.0\python.exe'
$sdkRoot = 'C:\Program Files\RyzenAI\1.8.0'
$sdkEnvironment = Split-Path -Parent $sdkPython
$env:RYZEN_AI_INSTALLATION_PATH = $sdkRoot
$env:PATH = "$sdkEnvironment\Scripts;$sdkEnvironment;$sdkRoot;$env:PATH"
$workDir = [IO.Path]::GetFullPath('cache\qwen35-conversion\prepare-new')
New-Item -ItemType Directory -Path $workDir | Out-Null
$npu = 'models\Qwen3.5-4B-npu-eager-new'

& $sdkPython benchmarks\prepare_qwen35.py `
  --source $oga --output $npu --work-dir $workDir --chunk-size 64
```

The SDK environment must already contain the `model_generate` wheel supplied
with Ryzen AI 1.8. The wrapper checks the reviewed SDK files and restores only
one structurally identified missing FP32 `LinearAttention` value-info entry.
It uses native SDK preprocessing, the prefill shape fix, and eager
post-processing, followed by native finalization and transaction filtering.
It explicitly sets `search.max_length=16384` and `search.chunk_size=64`,
while keeping the provider allocation bound at 4096. It writes a conversion
manifest and moves SDK diagnostics to the ignored working directory.

`--finalize-existing` can finish a known successful optimizer output whose
`tmp` directory is complete. It cannot resume an interrupted optimization
pass. Preserve failed/interrupted output directories and choose a new one
when rerunning conversion.

The materialized short-tested candidate at
`models/Qwen3.5-4B-npu-eager-16k-chunk64-run1` was recovered from the completed
prefill graph of the failed Token Fusion run. Its semantic eager strategy
matches the scripted route. The incomplete `...chunk64-run2` output must not
be used for inference.

## Validation before deployment

After host stability is restored, use a new output directory:

```powershell
& .\.venv\Scripts\python.exe benchmarks\validate_qwen35_npu.py `
  --model $npu --revision $revision `
  --output cache\qwen35-conversion\validation-new
```

The validator records each completed stage separately. It marks a report
`passed` only after short direct scoring, native generation, tail/head sentinel
checks at the 16K boundary, overflow rejection, and completed NPU commands.
Failed reports can be written earlier; require `status: passed`, not merely
the existence of `report.json`. Partial stage files are not a passed run.
Correctness is reported separately
from finite-output and hardware checks. Model quality and generation speed
still require their own benchmarks.

Repository checks remain:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
Push-Location results\raw
& 'C:\Program Files\Git\usr\bin\sha256sum.exe' -c SHA256SUMS
Pop-Location
& .\.venv\Scripts\python.exe benchmarks\verify_published.py
```
