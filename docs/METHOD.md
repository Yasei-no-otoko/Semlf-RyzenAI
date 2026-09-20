# Method

## Question and systems

Phase 1 tests whether open, generation-free readouts reproduce the useful part of Jev's public claim: accept unstructured state plus runtime-defined natural-language decisions and return typed scores cheaply enough to embed in ordinary software.

The direct system prompts frozen Qwen3.5-4B with a state, criterion, and 2-16 described options. It performs one native forward pass and applies a softmax only to the logits of fixed uppercase answer tokens. It does not decode a token.

The reranker system follows Qwen3-Reranker-4B's native yes/no contract. Each candidate answer becomes a separate query/document relevance proposition. The system computes `logit(yes) - logit(no)` for every option and softmaxes those log-odds across options. That last normalization is our comparison rule; it is not part of the upstream reranker's calibration contract.

## Frozen evaluation matrix

Prompts, IDs, labels, task semantics, revisions, and metrics were frozen before the full reranker outputs were evaluated. The complete local matrix had 706 rows:

| Source | Rows | Purpose |
|---|---:|---|
| Project-authored | 144 | Evidence interpretation, rule application, candidate selection; original and missing-evidence cases |
| WANLI | 256 | External natural-language inference |
| TypeSafe public evaluation subset | 102 | Distribution/reference agreement on 20 available cases |
| Every public lab artifacts | 204 | Judgment grid, retrieval, company knowledge, composed action policy |

Unlike tasks were not collapsed into one accuracy number. Hard-label tasks use full-denominator accuracy, balanced accuracy, macro F1, NLL/Brier where applicable, and source-group bootstrap intervals. Retrieval reports query ranking metrics. TypeSafe rows compare distributions and use an equal-case macro so cases with more questions do not dominate.

The TypeSafe comparison is the 102 public rows that could be aligned locally, not its advertised 711-row aggregate and not a live Jev run. Published Jev, Opus, and Sol values were read from those public records. Raw third-party fixtures are excluded from this candidate.

The exact evaluated IDs are committed in `benchmarks/manifests/source-selection.jsonl`. WANLI uses revision `61c95318fd71c55b6ba355d76253254615f387ec`: malformed or over-4,000-character rows and components touching pilot-training sources were excluded, remaining IDs were sorted then shuffled with seed 291607, and 86 entailment/85 contradiction/85 neutral rows were selected with at most one row per connected premise/pair-ID component. Premise becomes state; hypothesis becomes the criterion; entailment/neutral/contradiction map to supported/insufficient/contradicted.

TypeSafe extraction reads four locally supplied, hash-verified `*-cases.js` snapshots. It keeps published, successfully run Choice/Noul nodes having one unambiguous document/question binding, a released reference answer, and a unique reference-distribution argmax. Selection is outcome-blind round-robin over workflow, case, and primitive, with hashed-ID order inside buckets. Score primitives and tied targets are excluded from the 102-row comparison. Every mappings and original experiment data are available from the directly linked experiment JSON and source archive.

## Perturbations

Thirty-six owned original cases received three output-blind variants: reverse the displayed option order while preserving semantic IDs, wrap the criterion in meaning-preserving wording, and append irrelevant owned context. A separate 36-row missing-evidence population tests whether a system selects `insufficient`. Stability is measured after aligning probabilities by semantic option ID.

## Browser model ladder

Qwen3-0.6B, MiniCPM5-2B, and Qwen3.5-4B use the same frozen prompt and native BF16 final-position option-logit scorer on the 144 authored, 108 perturbation, and 102 selected TypeSafe rows. TypeSafe modal agreement is averaged within each of the 20 source cases and then equally across cases. The browser artifacts are independently pinned GGUF quantizations. Browser smoke timings begin after the page initiates each operation; model files were served from a local SSD to exclude internet transfer time. A successful smoke requires model load, warmup, finite logits for every displayed option, and completion of the generated path. It does not establish full quantized quality or portable latency.

## Shape-matched systems benchmark

An owned fixture contains 37 states and 21 fixed binary criteria per state, giving 777 decisions. States are roughly 8,000 characters and exercise repeated-context computation. It matches the count geometry of the public Every/Jev demonstration, but does not reproduce its unpublished documents, token lengths, hardware, API path, or model. Therefore it is a systems measurement, not a Jev head-to-head benchmark.

Direct modes are fresh batch-one scoring, serial suffixes after one state prefill, and parallel suffix branches after one state prefill. The reranker repeats the state for two independent yes/no option pairs per binary decision and tests ordinary pair batching. Timings use one RTX 3090 with a warm-loaded BF16 model and include prompt construction, tokenization, transfers, forward passes, and CPU readout; model loading and result-file writes are outside the timed region.

## Interpretation rules

- A forced typed output can still be semantically wrong.
- Softmax over allowed tokens is conditional on the supplied alternatives; it is not calibrated operational confidence.
- Prefix-cache speedups are implementation results, not evidence about Jev's disclosed architecture.
- A reranker is expected to be strongest on ranking. Its categorical threshold metrics should not be confused with ranking quality.

## Ryzen AI 1.8 evidence

The Ryzen AI runs use AMD's pinned ONNX Runtime GenAI NPU artifacts through the
repository's `ryzenai-npu` direct backend. The 4K quality run uses the local
`amd/Qwen3-4B_rai_1.8.0_npu_4K` artifact at revision
`d6fb03663d78ae5034d4594bfe9d92b35a5e213a`. The separate long-context quality
run uses AMD's `amd/Qwen3-4B_rai_1.8.0_npu_16K` Token Fusion artifact at
revision `715d60818350b685ca2af3566e5ae38f4780daf0`. These are distinct
compiled model artifacts; results are reported by artifact and are not a
Qwen3.5-to-Qwen3 quality transfer.

Both runs use the same frozen SemIf inputs and direct probability scorer. The
official NPU configuration permits CPU host/tokenization and other CPU graph
work; no GPU offload is configured. The benchmark records runtime version
metadata, model revisions, and model/input/prompt/code hashes in its manifest.
Predictions contain option IDs, probabilities/logits, token counts, timings,
and errors, but no source text or reference documents.

The speed scope is deliberately narrow: three fresh direct repeats for the
21-decision compact comparison against one compact JSON generation with a
128-token output cap, plus one fresh-direct pass over all 777 Shape777
decisions. Serial prefix caching, shared-state execution, and the native
reranker are unsupported paths for this backend. The run performs a context
preflight and never silently truncates a prompt. The 4K TypeSafe run is
recorded its compiled-context rejections explicitly (29 rejected and 73 scored
rows). The 16K Token Fusion run scored all 102 TypeSafe rows; its largest
observed input was 12,621 tokens. The two results remain measurements of
distinct compiled artifacts, not a context-only comparison.

Compact generation uses the loaded model's `genai_config.json` EOS IDs.
These may differ from the reference tokenizer's chat-end ID, as in the custom
Qwen3.5 conversion. The report retains the actual stop IDs and generated-token
timeline. Strict JSON validity, EOS termination, and the 128-token limit are
reported separately; an invalid or truncated answer remains a benchmark outcome.

`verify_ryzenai.py` compares each recorded source-text hash with the working
tree, then with exact source content in the available local Git history.
Its output identifies a historical match by commit rather than implying that
the current implementation is unchanged. A shallow clone or source ZIP that
lacks the matching historical source cannot complete that check.

### Qwen3.5-versus-Qwen3 comparison

The completed Qwen3.5 comparison uses the same fresh 21-decision Speed
protocol for both models and compares Qwen3.5 quality with the dated
2026-09-19 Qwen3 16K Quality artifact. It adds three fresh direct repeats,
three compact JSON repeats, and one fresh-direct Shape777 pass per model;
the compact path has a 128-token cap and reports syntax validity, parsed
array length, EOS, and truncation independently. Full-vocabulary guard time
is included in the fresh Speed measurements. The deployed artifacts differ
in architecture, tokenizer, and quantization, so the comparison describes
these pinned systems; CPU host, prefill, and LM-head graph components are
permitted by the NPU backend. The historical Qwen3 Quality record has no DLL
identity and no full-vocabulary guard record.

The final comparison is summarized in
[`RESULTS.md`](RESULTS.md#qwen35-versus-qwen3-speed-and-quality) and its
create-only evidence bundle is
[`results/raw/qwen35-vs-qwen3-20260920`](../results/raw/qwen35-vs-qwen3-20260920).
After the bundle is present, the public integrity check is:

```powershell
.venv\Scripts\python.exe -B benchmarks\verify_ryzenai_comparison.py results\raw\qwen35-vs-qwen3-20260920
.venv\Scripts\python.exe -B benchmarks\verify_ryzenai_comparison.py results\raw\qwen35-vs-qwen3-20260920 --data-dir cache\benchmark-built-20260919-215735
```

The default check verifies all 3,308 prediction rows, timing arithmetic,
recorded guards and provenance associations, and recomputes the owned
Authored/Perturbations metrics. WANLI, TypeSafe, and Every metrics are only
recomputed when `--data-dir` supplies their matching frozen inputs. The
second command was also run for this publication; when reproducing it,
substitute the directory built by the repository's pinned source workflow.
Missing or mismatched inputs fail the check. `verify_published.py` now also
runs the default comparison check, separately from its original 69 claims.

Generated text and token sequences are omitted from the public bundle, so
it cannot independently repeat strict JSON parsing. It retains recorded
validity, choices when valid, output hashes, EOS/truncation, and timing.
The [output and recovery audit](../results/raw/qwen35-speed-20260920/benchmark-recovery-audit.json)
records the independently checked 20-versus-21 array lengths and earlier
failed attempts. Full logits arrays and warmup token-ID fingerprints were
not retained; finite-logit verification checks the saved guard records.
Hash checks establish consistency of recorded evidence and do not rerun
the model or independently establish hardware execution. Neither verifier
command performs inference or downloads model weights or source records.

The model distinction and context modes follow AMD's [Ryzen AI 1.8 OGA
documentation](https://ryzenai.docs.amd.com/en/latest/oga_model_prepare.html). The
official model cards are [Qwen3-4B NPU 4K](https://huggingface.co/amd/Qwen3-4B_rai_1.8.0_npu_4K)
and [Qwen3-4B NPU 16K](https://huggingface.co/amd/Qwen3-4B_rai_1.8.0_npu_16K).
