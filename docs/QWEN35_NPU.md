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
| SDK default Token Fusion recipe | Failed: missing Qwen3.5 DD partition and incompatible default projection format |
| Custom DD LinearAttention | Two direct-DD single-token NPU calls complete; nonzero-state reference identifies the state/gate/head contract |
| v2/no-control-packet MatMul DD | Real layer-0 `in_proj_b` direct-NPU check passes with DD LLM mode disabled; original mode failed |
| Projection transaction inventory | Exact M=1 v2/no-control-packet entries exist for all 249 projections across 8 shapes |
| All-projection DD graph prototype | 249 projections converted; CPU structural checks pass with the native ORT normalization schema; full model execution untested |
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
Bus/Interconnect errors. The operator subsequently reported simultaneous
DiffusionGemma localjev testing and a 32-way LichtFeld Studio build during
these runs. This context and the event correlation do not establish a root
cause. Resume with isolated, bounded operator checks after checking current
load, before repeating long model runs.

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

Follow-up inspection corrects the earlier conclusion that a Python
post-processing extension could not work. The installed DD 1.8 binary
contains `linear_attention_token` and the required projection transactions.
For all three projection sizes above, transactions exist in **v2 format
without control packets**, whereas the default token strategy selects flat
format with control packets. A synthetic `(2560, 32)` UINT4 group-128 weight
prepack succeeds with v2/no-control-packet settings. A synthetic single-token
LinearAttention DD graph also compiles and saves successfully on the host.
Neither result alone proves numerical correctness or 16K inference.
The [transaction and packing results](../results/raw/qwen35-conversion-20260920/dd-contracts.json)
and [host compilation report](../results/raw/qwen35-conversion-20260920/dd-linear-host-compile.json)
record the exact SDK source and generated artifact hashes.

The [MatMul host-compilation report](../results/raw/qwen35-conversion-20260920/dd-matmul-host-compile.json)
also covers the real layer-0 `in_proj_b` projection. Only 42,240 external
weight/scale bytes were read; zero points were inline. Restoring the
preformatted weight, zero-point, and scale layout reproduces the original
dequantized matrix exactly. Four CPU linear/Sigmoid reference inputs agree
with that restored representation. Sigmoid remains outside DD. This checks
weight layout and host compilation, not NPU rounding, model quality, or
the native provider's single-node `model_type` routing.

The [complete projection inventory](../results/raw/qwen35-conversion-20260920/dd-projection-inventory.json)
maps all 249 projections to eight `(K, N)` shapes. Each has an exact M=1,
4-bit/group-128 v2/no-control-packet transaction returned by the SDK query
and present in the transaction archive's index. This includes the
`(2560, 248320)` output projection. No shape is missing from this inventory;
only the small `(2560, 32)` projection has been host-compiled in these tests.
Inventory presence alone does not prove all projections can compile or run.

The [all-projection graph report](../results/raw/qwen35-conversion-20260920/dd-projection-graph.json)
records a CPU-only conversion of all 249 projections into separate DD nodes.
The 1,297 surrounding nodes and the exact 67-input/65-output protobufs are
preserved. SSA, topological references, and external-data ranges were checked.
The standard ONNX checker rejects the source model's existing default-domain
`SimplifiedLayerNormalization`; importing that operator's actual ORT 1.27
schema permits the structural check. This does not shape-infer custom
operators or validate their execution. Constants use independent files.

The [direct-DD LinearAttention NPU report](../results/raw/qwen35-conversion-20260920/dd-linear-token-npu.json)
contains two owned synthetic single-token cases, including an asymmetric
nonzero recurrent state. Both calls complete: the process's NPU context
records two submissions, two completions, and zero errors. Among the tested
contracts, the nonzero case agrees best with K-major input/output state,
log-decay gate values, and the kernel's internal `repeat_interleave(2)` head
mapping. No external gate exponentiation or Q/K head replication is needed.
The query has already been scaled upstream. Attention relative RMSE versus
the FP32 reference is 0.305% and 0.328%; state relative RMSE is 0.236% and
0.284%. All 32 case/hypothesis rows are retained. These are diagnostic
observations, not model-quality, speed, or 16K results.

The [first real MatMul NPU report](../results/raw/qwen35-conversion-20260920/dd-matmul-npu-failure.json)
records successful initialization followed by `ERT_CMD_STATE_ERROR` on the
single execution attempt: one submission, zero completions, and one error.
Its output cannot be used for numerical comparison.

A [subsequent controlled MatMul check](../results/raw/qwen35-conversion-20260920/dd-matmul-npu-no-state.json)
uses the same weights, input, xclbin, PDI, and predeclared numerical thresholds,
with metadata `aux_info.is_llm=false` and no state-table updates. This removes
three state-management instructions and disables LLM input/output scratch
copies together; it does not isolate either change as the sole cause. The
single NPU call completes with zero errors. All 32 outputs are finite and
within their component budgets; relative RMSE is 0.875% versus the FP64
dequantized reference, below the predeclared 3.125% diagnostic limit. The
report preserves every output row and both failed/successful attempt records.
Sigmoid is evaluated separately on CPU. Other projection shapes, native
provider integration, full-model behavior, and 16K context remain unverified
by this single-projection check.

The two operator families require different xclbins and execution interfaces.
They need separate DD subgraphs, validated BF16/state boundaries, and an
explicit Qwen3.5 partitioning extension. Removing a partition assertion or
changing a context-length setting is insufficient. The generic combiner also
needs output-order validation, custom opset preservation, and transaction
collection for branches that contain both DD and eager operators. This
custom route is under development; NPU eager remains a separate experimental
result.

An independent OGA exporter memory improvement was submitted as
[onnxruntime-genai PR #2596](https://github.com/microsoft/onnxruntime-genai/pull/2596).
Its bit-packing equivalence and allocation tests pass (157 tests). The change
reduces intermediate packing allocations; it does not implement Token Fusion
or establish whole-model memory usage. The PR is not yet merged.

### Official availability check (2026-09-20 JST)

AMD's public Qwen model query returned 203 models, including nine Qwen3.5
models in 9B, 35B, and 397B sizes, but no Qwen3.5-4B model. The public API
returned 25 items from the Ryzen AI 1.8 NPU 16K collection, with no Qwen3.5
entry. These are bounded public-query results, not a claim about unpublished
AMD work. The [official model list](https://ryzenai.docs.amd.com/en/latest/llm_list.html)
also has no Qwen3.5 entry.

The [generic 16K preparation command](https://ryzenai.docs.amd.com/en/latest/oga_model_prepare.html)
applies to supported architectures. The [additional operator compilation
flow](https://ryzenai.docs.amd.com/en/latest/oga_op_prepare.html) is experimental
and primarily supports hybrid execution; it does not establish a Qwen3.5
Token Fusion route.

The installed SDK's `ryzenai_onnx_utils` and `ryzenai_dynamic_dispatch` 1.8.0
Windows wheels exactly match the SHA256 values for the same filenames in
AMD's public package indexes. This rules out a replacement of these two
specific files in the checked indexes. Query times, URLs, counts, and both
wheel hashes are recorded in the
[availability report](../results/raw/qwen35-conversion-20260920/availability.json).
This check did not run inference and does not change the incomplete status.

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
