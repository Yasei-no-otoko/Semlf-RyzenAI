# Ryzen AI NPU backend

This is the SemIf direct scorer for AMD Ryzen AI Software 1.8. It targets the
official AMD Qwen3 4B Full Fusion NPU 4K model:

* model: `amd/Qwen3-4B_rai_1.8.0_npu_4K`
* pinned revision: `d6fb03663d78ae5034d4594bfe9d92b35a5e213a`
* source: [AMD model card](https://huggingface.co/amd/Qwen3-4B_rai_1.8.0_npu_4K)

The model is an OGA deployment artifact, not a Transformers checkpoint. Models
made for earlier Ryzen AI releases are incompatible with Ryzen AI 1.8; use the
1.8 model and runtime together. AMD documents Full Fusion NPU models as having
a 4096-token total input-plus-output limit.

## Install the AMD runtime first

Install Ryzen AI Software 1.8 and the matching NPU driver from AMD. For a
standalone Python environment, follow AMD's [OGA flow](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html)
and use the AMD Ryzen AI 1.8 package source. Do not substitute similarly named
packages from public PyPI. The OGA stack used by this integration is the set of
AMD 1.8 artifacts below (the exact filenames are useful when checking a local
wheel cache):

```text
onnxruntime_genai_directml_ryzenai-0.14.0-py3-none-win_amd64.whl
onnxruntime_vitisai-1.27.0-py3-none-win_amd64.whl
onnxruntime_providers_ryzenai-1.8.0-py3-none-win_amd64.whl
ryzenai_dynamic_dispatch-1.8.0-py3-none-win_amd64.whl
voe-1.8.0-py3-none-win_amd64.whl
```

Use the wheel files provided by AMD rather than guessing a PyPI replacement.
This repository does not install the SDK, alter DLL search paths, or configure
global environment variables for you. The Ryzen AI 1.27 VitisAI bindings are
built against the NumPy 1.x ABI, so the `[ryzenai]` extra pins NumPy 1.26.4.
The CUDA/Torch and MLX extras retain the original NumPy 2.2.6 pin; do not mix
those extras into the NPU environment.

## Download and run

Use Python 3.12 from the installed SDK to create an isolated environment. From
the repository root, install SemIf and the five AMD-provided wheels in one
dependency-resolution step. The Torch/Torchvision versions below match AMD's
`env.yaml`; they are SDK dependencies, not the NPU scorer's execution backend.

```powershell
conda activate ryzen-ai-1.8.0
python -m venv .venv
$sdkDir = 'C:\Program Files\RyzenAI\1.8.0'
$wheelNames = @(
  'onnxruntime_genai_directml_ryzenai-0.14.0-py3-none-win_amd64.whl'
  'onnxruntime_vitisai-1.27.0-py3-none-win_amd64.whl'
  'onnxruntime_providers_ryzenai-1.8.0-py3-none-win_amd64.whl'
  'ryzenai_dynamic_dispatch-1.8.0-py3-none-win_amd64.whl'
  'voe-1.8.0-py3-none-win_amd64.whl'
)
$wheels = $wheelNames | ForEach-Object { Join-Path $sdkDir $_ }
& .\.venv\Scripts\python.exe -m pip install -e '.[test,ryzenai]' `
  'torch==2.4.1' 'torchvision==0.19.1' @wheels
& .\.venv\Scripts\python.exe -m pip check
```

Do not add the AMD wheels as ordinary project dependencies or use `--no-deps`;
the local wheel install must resolve their declared dependencies. Then download
the exact model snapshot and run:

```powershell
$modelDir = 'models\Qwen3-4B-npu-4k'
& .\.venv\Scripts\hf.exe download amd/Qwen3-4B_rai_1.8.0_npu_4K `
  --revision d6fb03663d78ae5034d4594bfe9d92b35a5e213a `
  --local-dir $modelDir
```

The pinned snapshot is the official [AMD model card](https://huggingface.co/amd/Qwen3-4B_rai_1.8.0_npu_4K). Then run:

```powershell
.\run_semif_npu.ps1
```

The launcher always passes `--backend ryzenai-npu --mode direct`, uses
`.venv\Scripts\python.exe`, and creates a new ignored output under
`cache\results\`. To select a local model, revision label, input, or output:

```powershell
.\run_semif_npu.ps1 `
  -Model 'D:\models\Qwen3-4B-npu-4k' `
  -Revision 'd6fb03663d78ae5034d4594bfe9d92b35a5e213a' `
  -Input 'examples\decisions.jsonl' `
  -Output 'cache\results\semif-npu-my-run.jsonl'
```

Outputs retain SemIf's existing JSONL input and output shape. The NPU performs
the model compute selected by AMD's official NPU configuration. Host CPU work
includes tokenization, prompt construction, option readout, and result
serialization; the model's official prefill/LM-head CPU components remain in
that configuration. The launcher configures no GPU offload and makes no claim
that every operator runs on the NPU. This backend currently supports direct
scoring only: serial and shared modes are not supported. It preserves the
SemIf interface and data format, but it is not an official Jev binary or
network API and does not transfer the Qwen3.5 quality claims to Qwen3.

## Local hardware verification

### Browser demo

To run the web demo on the same Ryzen AI model, start the local server:

```powershell
.\run_semif_npu_demo.ps1
```

Open the [Japanese page](http://127.0.0.1:8008/) or the
[English page](http://127.0.0.1:8008/en/), load the model, and run the comparison.
The language links switch pages; both pages share the same server and loaded
model. Both interfaces preserve the original demo's two methods: direct option logits and
autoregressive JSON probabilities, streamed as the model produces tokens.
The server loads one model and executes the two methods sequentially on one
worker. The original `webgpu-demo/` remains a standalone browser demo;
`npu-demo/` uses the local Python/OGA backend and shares its stylesheet.

The demo reads the model from the local model directory configured in the
setup steps above and does not download it. Inputs stay on this machine and are not
saved by the server. No external assets or inference services are requested by
the NPU page. The server listens only on loopback and accepts same-origin web
requests. Stop it with Ctrl+C in its terminal; use `-Port 8009` for another port.

The demo opens with a 16-option receipt-routing example and supports 2–32
options within the model's 4096-token context. A 32-option preset is also available. The
single-token labels are A–Z followed by 0–5. JSON generation reserves up to
512 output tokens for 2–16 options, or 1024 for 17–32, and rejects prompts that do not
fit; there is no silent truncation. Generated JSON is checked against the
exact option names and probability range/sum. Invalid or truncated output is
shown as such. For 17–32 options, the generation prompt lists every required
JSON key explicitly and requests zero-probability entries too, to reduce
prematurely shortened answers. The model can still violate the requested
format; missing entries are reported with expected and received key counts.
Generated probability estimates and direct normalized logits
are different quantities; neither is calibrated decision confidence. The first
run includes initial kernel preparation, so cold and later timings can differ.

On 2026-09-19, the browser demo was exercised in Edge on this Ryzen AI Max+ 395:
model loading took 25.925 seconds. The Japanese account preset measured 1.394
seconds for direct scoring and 4.910 seconds for generation (48 output tokens).
The Japanese email preset measured 1.253 seconds and 4.401 seconds respectively
(43 output tokens). Both generated objects passed the exact-option JSON checks.
The account preset's direct and generated choices disagreed, which illustrates
why these outputs should not be treated as interchangeable confidence scores.
Empty-question validation and a successful retry were also checked in the UI.

The server process (PID 8816) completed 24 prefill-context and 89 token-context
NPU commands with zero hardware errors. Its local device snapshot is
`cache/results/npu-demo-hardware-20260919-210648.json`. These are smoke-run
observations, not published quality or speed benchmark claims.

### Direct scorer smoke run

A local smoke run on 2026-09-19 used Ryzen AI Max+ 395 (Strix Halo), Ryzen AI
1.8.0, OGA 0.14.0, and the pinned model above. AMD's SDK quicktest passed.
For the SemIf process (PID 29808), `xrt-smi` recorded NPU submissions and
completions increasing from 0 to 20 across five direct decisions, with zero
hardware errors. The model reported device type `RyzenAI`.

The three bundled examples, a reversed option order, and a Japanese input all
produced finite categorical scores. The reversed-order test selected the same
semantic answer, and the Japanese color test selected blue. The policy example
selected `required` although the supplied policy implies `not_required`; this
is a model decision error, and the scores are not calibrated confidence.
This smoke run is execution evidence, not a quality benchmark.

Model loading and artifact hashing took 23.05 seconds. The first English
decision took 1.73 seconds; subsequent English decisions took 0.23–0.24 seconds,
and the first Japanese input took 0.66 seconds. These are single-run timings
and exclude model startup.

The local evidence is retained under ignored `cache/results/`:
`npu-verified-20260919-204811-29808.jsonl` and
`npu-hardware-20260919-204811-29808-{before,after}.json`.
