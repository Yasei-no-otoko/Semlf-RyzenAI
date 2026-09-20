# Qwen3.5-4B conversion for Ryzen AI 1.8

**Experimental and incomplete.** Quark quantization and OGA export completed.
A combined eager-prefill and custom DD-token model now scores three owned
English/Japanese short examples correctly and completes a native greedy
continuation at EOS. That short run's full-vocabulary logits are finite, with
5,307 NPU submissions/completions and zero errors. **The installed SDK recipe
does not support this custom integration, and the combined 16K validation
failed.** The tail-information case passed at exactly 16,384 input tokens;
the head-information case produced non-finite logits. A fresh-process head
check also returned all-NaN logits. This is not a replacement for the working
AMD Qwen3-4B demo model.

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
| Custom DD LinearAttention | Two direct-DD calls and three native ORT calls complete; BF16 outputs, and FP16 outputs after the boundary cast, match direct DD exactly |
| v2/no-control-packet MatMul DD | Real layer-0 `in_proj_b` passes direct DD and native ORT checks with DD LLM mode disabled; original mode failed |
| Projection transaction inventory | Exact M=1 v2/no-control-packet entries exist for all 249 projections across 8 shapes |
| Representative projection sizes | All eight shapes host-compile and pass one direct-DD NPU check each, including the 248,320-output LM head |
| All-projection DD graph prototype | 249 projections converted; CPU structural checks pass with the native ORT normalization schema |
| SDK NPU eager, chunk size 4096 | English short example passes; Japanese 96-token example returns NaNs |
| SDK NPU eager, chunk size 64 | Three owned short examples pass, including Japanese |
| Combined eager-prefill / DD-token model | One full-model load, three correct short cases; option logits match the eager baseline exactly |
| Native generation, combined model | Four generated tokens ending at configured EOS; all seven observed full-vocabulary logit vectors are finite |
| 16,384-token prefill | Tail-information case passed; head-information case failed the full-vocabulary finite check. Earlier two runs were interrupted by reboots |
| Fresh-process 16,384-token head prefill | All 248,320 logits are NaN; 64 final states read, with the lowest-numbered non-finite state at layer 14, head 5 |
| Exact first 64 tokens of the head input | All logits are NaN after one eager chunk; same lowest non-finite state at layer 14, head 5. Incomplete prompt, no answer-quality check |
| Public custom-DD CLI rebuild | CPU conversion completes; independent artifact/graph comparison matches the diagnostic candidate under documented normalization. New package runtime is untested |
| Full scripted eager post-processing rerun | Interrupted by the second reboot |

The eager prefill graph contains 1,398 nodes. Its NPU operators include 153
`MatMulNBitsBf`, 32 `SSMLP` groups (covering the remaining 96 heavy matmuls),
and 24 `LinearAttention` operations. The 24 `CausalConvWithState` and 8 GQA
operations remain on CPU, with other host operations and casts. No GPU
provider is configured. The 64 recurrent/conv/KV state inputs are float16 at
the graph boundary. The final artifact is text-only: vision and MTP are not
exported, and embeddings and depthwise convolution weights remain raw.

The chunk-4096 Japanese failure's lowest-numbered non-finite final state is
layer 8's recurrent state, in one of 32 heads. CPU evaluation of the same
layer remains finite.
Chunk size 64 avoids the observed short-example failure; this does not prove
stability for arbitrary inputs or long contexts.

The two earlier long runs reached the 16,384-token tail-sentinel stage, but neither
produced a final result. Windows recorded unexpected reboots at approximately
00:47 and 00:59 JST on 2026-09-20, with preceding WHEA corrected
Bus/Interconnect errors. The operator subsequently reported simultaneous
DiffusionGemma localjev testing and a 32-way LichtFeld Studio build during
these runs. This context and the event correlation do not establish a root
cause. Subsequent isolated operator checks and the combined short-model run
completed successfully on an idle host. The bounded long-context run then
returned a numerical failure without a reboot or NPU command error. These
observations do not identify the cause of either the earlier reboots or the
new numerical failure.

The owned evidence and exact artifact hashes are in
[the conversion evidence](../results/raw/qwen35-conversion-20260920/evidence.json).
Weights, caches, machine-identifying hardware reports, and third-party data
are not committed.

## SDK recipe failure and the custom DD integration

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
`(2560, 248320)` output projection. No shape is missing from this inventory.
Inventory presence alone does not prove execution; the separate representative
shape checks below provide compilation and numerical evidence.

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

The [native ORT LinearAttention report](../results/raw/qwen35-conversion-20260920/dd-native-linear-attention.json)
adds three independent one-session/one-execution checks: zero-state BF16,
asymmetric-state BF16, and asymmetric-state FP16 boundaries. Each records one
completed NPU command and zero errors. The wrapper uses `model_type=9`, six
DD inputs (`input_num=6`), and a seventh UINT32 host input containing the valid
token count. The DD metadata still has six inputs. This is an experimentally
verified connection, not a claim of AMD-supported Qwen3.5 integration.
All 1,585,152 output elements across the three cases match the corresponding
direct-DD storage bits exactly after the defined output conversion. Twelve
state elements round differently when cast from BF16 to FP16; they match the
cast reference, not the original BF16 bits. The report preserves all 48 native
and 32 direct reference-hypothesis rows. The FP16 runner's numerical metrics
re-round its output to BF16; the independent bit comparison uses the actual
FP16 output. The seeded asymmetric state is not derived from a real 128-token
prefix. These checks do not establish multi-token evolution or full-model
correctness.

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
Sigmoid is evaluated separately on CPU. Other projection shapes, full-model
behavior, and 16K context remain unverified by this single-projection check.

The [native ORT MatMul report](../results/raw/qwen35-conversion-20260920/dd-native-matmul.json)
then verifies the same projection through the RyzenAI NPU provider with
`model_type=9`, `mladf_version=v2`, `input_num=1`, and a second host count input.
One session construction and one execution produce one completed NPU command
and zero errors. All 32 BF16 output values match the successful direct-DD
result bit for bit, with the same predeclared error limits. The graph contains
one DD node; its separately reported Sigmoid values are CPU post-processing.
The native provider's internal compile configuration has not been proven
identical to the direct-DD configuration.

The [representative-shape report](../results/raw/qwen35-conversion-20260920/dd-all-shapes-npu.json)
covers one real projection for each of the eight shapes, with one seeded
synthetic BF16 input per shape. Each host compilation succeeds, and each
direct-DD NPU call completes with zero errors. All outputs are finite and
within the predeclared component error budgets and 3.125% relative-RMSE limit.
The [compressed output rows](../results/raw/qwen35-conversion-20260920/dd-all-shapes-rows.csv.gz)
retain individual references, results, and error budgets, including every
output of the LM head.

| K | N | Relative RMSE versus FP64 dequantized reference |
|---:|---:|---:|
| 2,560 | 32 | 0.875% |
| 2,560 | 1,024 | 0.952% |
| 2,560 | 4,096 | 0.973% |
| 4,096 | 2,560 | 0.985% |
| 2,560 | 8,192 | 0.971% |
| 2,560 | 9,216 | 0.938% |
| 9,216 | 2,560 | 0.959% |
| 2,560 | 248,320 | 0.785% |

These are eight representative weight tensors, not numerical validation of
all 249 projections. The inputs are not captured model activations, and the
results are not model-quality or throughput benchmarks. The seven new CPU
references use chunks of at most 18 MiB of FP64 weights. Host compilation of
the largest projection peaks at approximately 2.03 GiB of process private
commit; this is not a bound on whole-model runtime memory.

The two operator families require different xclbins and execution interfaces.
The combined candidate uses separate DD subgraphs with the checked BF16/state
boundaries and an explicit Qwen3.5 partitioning extension. Its combiner checks
output order, captured inputs, custom opset versions, and external data. The
package retains the complete installed SDK transaction archive, including the
eager branch's entries, instead of relying on the SDK filter's incomplete
coverage of mixed branches. Removing a partition assertion or changing a
context-length setting alone would not provide these contracts.

### Combined-model short validation

The [integrated short-run report](../results/raw/qwen35-conversion-20260920/dd-integrated-short.json)
records one model load and four generators, with no retry. The combined model
selects the chunk-64 eager graph for prefill and a token graph containing
249 MatMul DD nodes and 24 LinearAttention DD nodes for single-token input.
CPU convolution, GQA, host operations, and casts remain; no GPU provider is
configured. This is a custom conversion, not an AMD-published Qwen3.5 model.

| Owned short case | Input tokens | Option logits | Correct |
|---|---:|---|---|
| English, BLUE first | 94 | `[27.625, 20.875]` | Yes |
| Japanese, BLUE second | 96 | `[19.25, 28.0]` | Yes |
| English, BLUE second | 94 | `[21.0, 28.375]` | Yes |

All three recorded option-logit pairs, probabilities, option IDs, and
prompt/tokenization identities match the earlier eager result exactly.
These direct cases exercise the eager prefill branch. Full-vocabulary arrays
were checked for finiteness but were not retained for an equality comparison.
Three owned color questions are functional checks, not a quality benchmark.

The greedy continuation generated four tokens, ending with the configured
EOS token `248044`, and decoded to `A\n`. It used OGA's native state/cache
management. The three direct calls and four greedy steps produced seven
observed last-position vectors of 248,320 logits, all finite. This validates
normal termination for this short continuation, not arbitrary generation.

Four process-owned NPU contexts together recorded **5,307 submissions,
5,307 completions, and zero errors**. After accounting for the repeated
94-token prefill, the continuation's residual counters are consistent with
three decode steps: `3 × 249 = 747` MatMul commands and `3 × 24 = 72`
LinearAttention commands. This is an inference from the graph inventory and
context counters, not a per-operator execution trace or proof that CPU
components were offloaded.

The first load, including compilation of all 273 DD metadata files, reached
the model-loaded stage after 238.50 seconds. Peak process/job commit was
about 9.60 GiB under a 20 GiB limit. These are single-run observations, not
speed benchmarks or bounds on all driver/device memory. The evidence retains
the source/model/runtime hashes, stages, counters, and equality comparison.
The model's configured 16K ceiling remains separate from this short-run
result.

### Combined-model 16K failure

The [subsequent supervised run](../results/raw/qwen35-conversion-20260920/dd-integrated-16k-failure.json)
completed the three short cases and greedy
continuation, then scored an owned tail-information prompt at exactly
16,384 input tokens. BLUE was correct, with option logits `[26.25, 22.875]`,
all 248,320 logits finite, and a single observed forward time of 618.786 seconds.
The 16,385-token overflow prompt was rejected before inference.

The next generator processed the owned head-information prompt at exactly
16,384 input tokens but failed the full-vocabulary finite check. Its scores
were not accepted. The saved event identifies a non-finite vector; it does
not retain the NaN/Inf counts or establish which layer first failed.
The overall validator exited with failure. All four NPU contexts together
recorded 292,539 submissions and completions, with zero reported command
errors. Process commit peaked at 11,260,719,104 bytes, below the 20 GiB job
limit; this was neither a timeout nor a host reboot. Numerical correctness
is therefore unresolved even though the NPU completed its commands.

The counter totals are consistent with all 520 expected prefill chunks
(four short prompts with two chunks each, plus two 256-chunk long prompts)
and the three short decode steps: `481 × 520 + 249 × 3 = 250867`,
`24 × 520 = 12480`, `56 × 520 = 29120`, and `24 × 3 = 72` across
the four contexts. This is an inference from the previously measured chunk
and decode counts, not a per-node trace. The failed long case reads prefill
logits; it does not exercise long-context DD decoding.

The [fresh-process head check](../results/raw/qwen35-conversion-20260920/dd-fresh-head-failure.json)
then used the same model/configuration and head-prompt construction, with one
model, one generator, exactly 16,384 input tokens, and no generated tokens.
All 248,320 logits were NaN. The [compressed original NumPy tensor](../results/raw/qwen35-conversion-20260920/dd-fresh-head-logits.npy.gz)
preserves every output bit; decompression reproduces the original file and
its SHA256 recorded in the report. The report also retains the complete
owned prompt, token IDs, source/runtime hashes, timings, and final-state
statistics. A preceding tail generator is therefore not required to trigger
this failure.

All 64 final states were read. Using zero-based layer/head indices, the
lowest-numbered non-finite state was `present.14.recurrent_state`: all
16,384 elements of head 5 were NaN, while its other 31 heads, layer 14's
convolution state, and all shallower saved states were finite. Both layer 15
KV outputs had their first recorded non-finite coordinate at `[0, 0, 40, 0]`.
Only the first 16 bad coordinates and aggregate counts were retained for KV,
so the complete distribution across token positions is unknown. These final
observations do not identify the first failing operation or chunk.

The fresh run completed 143,616 NPU submissions/completions with zero errors;
reading the states added no NPU commands. It exited with numerical failure,
not a timeout.

A [separate prefix diagnostic](../results/raw/qwen35-conversion-20260920/dd-head-prefix64-failure.json)
used only the exact first 64 token IDs of that saved head input, with the
same model/configuration and a 16K generator limit. All logits were NaN
after one eager chunk: 561 NPU submissions/completions, zero errors, and
zero additional commands for the 64 final-state reads. Layer 14, head 5
again contained 16,384 NaNs; both layer 15 KV tensors had their first
recorded bad coordinate at `[0, 0, 40, 0]`, with 24,576 NaNs each. The
[compressed original logits](../results/raw/qwen35-conversion-20260920/dd-head-prefix64-logits.npy.gz)
and all state/head statistics are retained. This incomplete prompt was a
finiteness diagnostic, not an answer-quality test. Its first 64 tokens are
sufficient to reproduce the failure; processing the full long context is
not required. KV statistics include unused allocated positions and do not
recover the complete distribution across valid tokens. The record preserves
all 273 changed generated DD `.state` hashes relative to the fresh-head run;
original model/configuration identities match, and no causal significance is
assigned to those serialization differences.

Chunk-dependent arithmetic or valid-token masking is a
candidate for further investigation, informed by the earlier short Japanese
case's improvement at chunk size 64. The native mechanism is unconfirmed:
OGA's `search.chunk_size=64` does not establish a 64-token native NPU kernel.
A smaller OGA chunk is a controlled follow-up, not an established fix. Neither
failed long-prefill check validates DD decoding near the 16K boundary.

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

## Public custom-DD conversion command

The [conversion CLI](../benchmarks/prepare_qwen35_token_fusion.py),
[compatibility profile](../benchmarks/qwen35_rai18_profile.json),
[DD lowering](../benchmarks/qwen35_dd.py), and
[packager](../benchmarks/qwen35_package.py) expose the custom route without
depending on private diagnostic scripts or editing installed SDK packages.
The [public CLI rebuild report](../results/raw/qwen35-conversion-20260920/dd-public-build-reproducibility.json)
records successful CPU conversion from the pinned existing OGA export and
eager prefill. The supervised build returned exit code 0 in 1,051.109 seconds,
with peak process commit of 21,489,287,168 bytes. These are single-build
observations, not inference performance measurements.

The independent comparison verified all 1,280 manifested artifacts in each
package against their recorded hashes. All three graph comparisons and all
273 DD contracts/constants matched after the documented name/path
normalization; graph I/O, 64 state boundaries, opsets, and configuration
apart from package paths also matched. The report retains every artifact
hash and comparison row, plus code, profile, and SDK identities.
**The rebuilt package has not run inference.** The short success and 16K
failures above belong to the separately materialized diagnostic candidate;
CPU reproducibility does not establish long-context numerical correctness.

Use the SDK 1.8 environment, rather than the export environment. The profile
pins SDK package/file versions and the exact measured OGA and eager artifacts.
The example paths below identify those artifacts; a different export must
pass the same profile checks before it can be used. Source, prefill, work,
output, and plan paths must be separate, and all output paths must be new.

```powershell
$tfWork = 'cache\qwen35-conversion\token-fusion-build-new'
$tfOutput = 'models\Qwen3.5-4B-token-fusion-new'
$tfPlan = 'cache\qwen35-conversion\token-fusion-plan-new.json'

& $sdkPython benchmarks\prepare_qwen35_token_fusion.py plan `
  --source models\Qwen3.5-4B-oga-run2 `
  --prefill models\Qwen3.5-4B-npu-eager-16k-chunk64-run1 `
  --sdk-root $sdkRoot --work-dir $tfWork --output $tfOutput `
  --profile benchmarks\qwen35_rai18_profile.json --plan $tfPlan
if ($LASTEXITCODE -ne 0) { throw 'Conversion plan validation failed.' }

$tfPlanSha = (Get-FileHash -LiteralPath $tfPlan -Algorithm SHA256).Hash.ToLowerInvariant()
& $sdkPython benchmarks\prepare_qwen35_token_fusion.py build `
  --plan $tfPlan --plan-sha256 $tfPlanSha
```

Planning verifies the pinned inputs, SDK, and code without creating an
inference session. Building regenerates the token graph, lowers 249 MatMul
and 24 LinearAttention partitions, and packages them with the supplied eager
prefill. It copies external data independently, relocates all DD constants,
preserves the eager protobuf header and weights, and includes the complete
SDK transaction archive. It uses metadata from the installed ORT in an
isolated CPU schema check; no model inference or NPU session is performed by
the conversion command.

The output includes `package-manifest.json`; the work directory contains
`build-report.json`. `materialized_cpu_checked` means file/graph validation
completed. Require that status in **`build-report.json`**, which is written
after the final source/SDK checks; a package manifest alone does not establish
that the overall build succeeded. It does not mean the new artifact passed
inference. Runtime initialization still compiles the native DD metadata. Absolute cache and
constant paths are recorded, so moving a package requires relocation and a
new manifest. Run short functional validation before attempting the separate
16K validation.

## Validation before deployment

The combined short run passed after two low-load samples (CPU at most 30%,
at least 32 GiB available RAM, and no build processes). A supervised full
validation subsequently failed on non-finite head-information logits, so
long-context deployment remains unvalidated. The following is the underlying
validator, which needs a new evidence directory:

```powershell
& $sdkPython benchmarks\validate_qwen35_npu.py `
  --model $tfOutput --revision $revision `
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
