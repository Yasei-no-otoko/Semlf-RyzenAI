# Reproduction guide

## Environment

Create an isolated virtual environment and place caches on a drive with room for model weights:

```bash
python -m venv .venv
. .venv/bin/activate
export HF_HOME=/path/to/large-drive/huggingface
pip install -r requirements.txt
pip install -e '.[test,torch]'
pytest -q
```

Use one GPU per scorer process. The measured environment was Ubuntu 22.04 on Linux x86_64, Python 3.10.12, NVIDIA driver 595.71.05, CUDA 12.8, PyTorch 2.10.0+cu128, Transformers 5.17.0, BF16, and an RTX 3090. `requirements.txt` pins the observed Python runtime packages; the CUDA-enabled PyTorch wheel still requires a compatible NVIDIA driver. Exact model commit IDs are in [../manifests/models.json](../manifests/models.json).

`pytest -q` runs all core and browser-source tests. Timing is hardware-sensitive, and BF16/kernel differences can change borderline probabilities or choices. Treat committed row counts, schemas, source hashes, and checksums as exact acceptance criteria; treat timings and model outputs as measurements to compare with the committed row-level evidence, not byte-identical golden outputs.

## Score owned examples

```bash
CUDA_VISIBLE_DEVICES=0 semif-score --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results-direct.jsonl

CUDA_VISIBLE_DEVICES=0 semif-score --mode serial \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results-serial.jsonl

CUDA_VISIBLE_DEVICES=0 semif-score --mode reranker \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input examples/decisions.jsonl --output results-reranker.jsonl
```

The command refuses an existing output path and refuses silent input truncation. Each output embeds the exact revision, library versions, prompt hash, token count, timings, and an explicit probability-status warning. State may be a nonempty string, JSON object, or JSON array. `serial` caches consecutive equal states. `shared` requires every input row to carry the same exact state and is exercised by the 37×21 runner below.

## Third-party evaluations

TypeSafe source records are not included. To reproduce that comparison, supply local snapshots in the source directory. The helper fetches the remaining public evaluation inputs with hash verification:

```bash
python benchmarks/fetch_sources.py --output /path/on/large-drive/semif-sources
```

The frozen 706-row matrix and source IDs are in `benchmarks/manifests/`. Row-level direct and reranker outputs are in `results/raw/predictions/`. The complete owned 144-row labeled workload is distributed in `benchmarks/data/authored144.jsonl`.

Build the exact external evaluation rows and recompute their metrics with the commands in [the benchmark guide](../benchmarks/README.md#quality-evidence). The builders verify source hashes and frozen selection IDs; the TypeSafe and Every evaluators accept the rebuilt gold rows plus the committed row-level predictions.

## Reproduce perturbation evidence

Rebuild the frozen 108-row fixture from the 36 owned originals, then verify it matches the committed fixture:

```bash
python benchmarks/build_perturbations.py \
  --source benchmarks/data/authored144.jsonl \
  --output /tmp/perturbations108.jsonl \
  --manifest /tmp/perturbations108-manifest.json
cmp /tmp/perturbations108.jsonl benchmarks/data/perturbations108.jsonl
```

Regenerate direct and reranker predictions with `semif-score --mode serial` and `--mode reranker`, respectively, or recompute the exact committed report from the included row-level predictions:

```bash
python benchmarks/evaluate_perturbations.py \
  --gold benchmarks/data/authored144.jsonl \
  --perturbations benchmarks/data/perturbations108.jsonl \
  --direct-base results/raw/predictions/direct-authored144.jsonl \
  --direct-perturbations results/raw/predictions/direct-perturbations108.jsonl \
  --reranker-base results/raw/predictions/reranker-authored144.jsonl \
  --reranker-perturbations results/raw/predictions/reranker-perturbations108.jsonl \
  --output perturbation-report.json
cmp perturbation-report.json results/raw/perturbation-comparison.json
```

## Reproduce the headline speed results

Run the focused three-repeat direct-versus-compact-array comparison:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/decision_vs_generation.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output compact-array-run.json
```

Run the complete 777-decision fresh, serial-cache, and parallel shared-state comparison:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/shape777.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output shape777-run.json
```

Run the complete native-reranker comparison at the published pair batch sizes:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/shape777_reranker.py \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input benchmarks/data/shape777.jsonl \
  --pair-batch-sizes 1,4,8 \
  --output shape777-reranker-run.json
```

All scripts require a new output path. Timing includes prompt construction, tokenization, transfers, model execution, and CPU readout after a warmup; model loading and final result-file writes are excluded.

Verify the committed evidence bundle and confirm that every selected scalar in the machine-readable summary matches its raw report:

```bash
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```

The source-specific quality commands above regenerate the metrics stored in `results/raw/quality-comparison.json`. `verify_published.py` checks 69 published summary values against that report plus the perturbation, systems, and generation reports. It deliberately does not require byte-identical GPU reruns.

## Reproduce Ryzen AI 1.8 evidence

These commands run from the repository root in the AMD 1.8 environment. Install
the optional Ryzen AI dependencies and the five locally supplied AMD wheels as
described in [RYZENAI.md](RYZENAI.md); do not substitute the older machine-wide
SDK. The model directories are local downloads and are never committed:

```powershell
$built = "cache/benchmark-built-20260919-215735"
$sdkRoot = "C:\Program Files\RyzenAI\1.8.0"
$python = ".\.venv\Scripts\python.exe"
$env:RYZEN_AI_INSTALLATION_PATH = $sdkRoot
$env:PATH = "$sdkRoot;$env:PATH"
$model4k = "models/Qwen3-4B-npu-4k"
$revision4k = "d6fb03663d78ae5034d4594bfe9d92b35a5e213a"
$out4k = "cache/results/ryzenai-4k-$(Get-Date -Format yyyyMMdd-HHmmss)"
if (Test-Path $out4k) { throw "Output must be new: $out4k" }
& $python benchmarks/ryzenai_benchmark.py --model $model4k --revision $revision4k `
  --data-dir $built --output $out4k --suite all
& $python benchmarks/verify_ryzenai.py --data-dir $built $out4k
```

`$built` must contain the locally rebuilt, hash-verified `wanli256.jsonl`,
`typesafe102.jsonl`, `every/inference204.jsonl`,
`every/gold154.jsonl`, and `every/firewall-actions.json` outputs. Build those
inputs with the existing commands in [benchmarks/README.md](../benchmarks/README.md)
and keep the source records local. The benchmark also consumes the committed
`benchmarks/data/authored144.jsonl`, `perturbations108.jsonl`, and
`shape777.jsonl` fixtures.

Run the separate AMD 16K Token Fusion quality suite with a new output path:

```powershell
$model16k = "models/Qwen3-4B-npu-16k"
$revision16k = "715d60818350b685ca2af3566e5ae38f4780daf0"
$out16k = "cache/results/ryzenai-16k-quality-$(Get-Date -Format yyyyMMdd-HHmmss)"
if (Test-Path $out16k) { throw "Output must be new: $out16k" }
& $python benchmarks/ryzenai_benchmark.py --model $model16k --revision $revision16k `
  --data-dir $built --output $out16k --suite quality
& $python benchmarks/verify_ryzenai.py --data-dir $built $out16k
```

The 4K and 16K runs are separate artifact measurements, not a context-only
comparison. Each run writes a context preflight before scoring and fails rather
than truncating non-TypeSafe inputs that exceed the compiled context ceiling.
The completed 4K TypeSafe run has 29 explicit context rejections and 73 scored
rows. The completed 16K run scored all 102 TypeSafe rows; its largest observed
input was 12,621 tokens. The 4K Full Fusion and 16K Token Fusion artifacts
retain their own pinned revisions and manifests.

For speed evidence, use the same 4K model and a new output directory with
`--suite speed`; this executes three fresh 21-decision direct repeats against
compact JSON generation capped at 128 output tokens, followed by one fresh
direct pass over all 777 Shape777 decisions. The benchmark does not invoke
serial/shared prefix modes or the native reranker. It records only IDs,
vectors, hashes, token counts, timings, and metadata in the output bundle.

The benchmark's AMD model/runtime and artifact revisions are documented in
[the AMD OGA preparation guide](https://ryzenai.docs.amd.com/en/latest/oga_model_prepare.html),
[the 4K model card](https://huggingface.co/amd/Qwen3-4B_rai_1.8.0_npu_4K), and
[the 16K model card](https://huggingface.co/amd/Qwen3-4B_rai_1.8.0_npu_16K).

For a completed run, check that the create-only bundle contains the manifest,
preflight, quality/speed reports, predictions, and private hardware snapshots
before archiving it:

```powershell
Get-ChildItem -LiteralPath $out4k -Force
Get-FileHash "$out4k\manifest.json", "$out4k\quality.json", "$out4k\compact.json" -Algorithm SHA256
& $python benchmarks\verify_ryzenai.py --data-dir $built $out4k
```
