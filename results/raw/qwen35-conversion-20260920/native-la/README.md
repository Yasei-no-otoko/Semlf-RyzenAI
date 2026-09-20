# Native LinearAttention numerical evidence (publication evidence)

This bundle preserves seven owned synthetic observations without committing repeated inputs or FP64 reference tensors. `synthetic-observed.zip` contains only unique BF16 raw **synthetic outputs**, indexed by SHA256. The inputs and two CPU references are regenerated from seed 20260920. Four small ONNX graphs contain no trained weights. `evidence.json` associates every row with original preparation, execution, supervisor, graph, input/reference/output and runner hashes.

| Case | Finite attention | Finite state | FP64 relative RMSE attention/state | NPU submitted/completed/errors | Exit |
|---|---:|---:|---:|---:|---:|
| native_s64_m4 | 262144/262144 | 524288/524288 | 0.011612952 / 0.0072734776 | 1/1/0 | 0 |
| native_s64_m80 | 0/262144 | 0/524288 | not defined (NaN) | 1/1/0 | 1 |
| native_s64_m128 | 0/262144 | 0/524288 | not defined (NaN) | 1/1/0 | 1 |
| native_s1_m128 | 4096/4096 | 524288/524288 | 0.0021267222 / 0.0016528113 | 1/1/0 | 0 |
| loop_s64_m4 | 262144/262144 | 524288/524288 | 0.002431077 / 0.0017680722 | 64/64/0 | 0 |
| loop_s64_m128 | 262144/262144 | 524288/524288 | 0.002347279 / 0.0016548687 | 64/64/0 | 0 |
| native_s64_m128_no_cast_hints | 0/262144 | 0/524288 | not defined (NaN) | 1/1/0 | 1 |

Exit 1 here is an intentional result for nonfinite output, with normal process return; it is not an access violation. Every listed test constructed one session and ran one host execute, without retries. Loop cases submitted 64 native NPU operations; their Loop/Gather/assembly is host work. These are numerical observations, not preselected quality thresholds. Batched native and token Loop need not be bit equivalent because state is rounded to BF16 between token steps.

The S1 -128 case uses the first token and zero initial state, and explicitly enables `hybrid_opt_token_backend=npu`. An earlier S1 test without that option stopped with `NPU backend must be used for LinearAttention`, submitted no NPU commands, and is recorded separately as a configuration failure. S1 does not prove decay accuracy for nonzero prior state. The constant -80 gate fixture is also nonfinite; no clamping transformation was applied to the captured model inputs. Removing only the three cast-absorption attributes also did not resolve it. Native internal overflow remains a hypothesis; no closed-source kernel cause is asserted.

The private captured layer14 comparison is metadata-only. Its original capture saved results then exited with 0xC0000005; that capture is **not** a clean run. The subsequent isolated native replay returned normally with exit 1 and reproduced every captured output bit, including NaN signs/payloads. Its attention had 3,072 NaNs (head 5, tokens 40..63) and state had 16,384 NaNs (head 5). The isolated Loop replay returned exit 0 with all 786,432 outputs finite, and used identical six BF16 inputs. Attention/state FP64 relative RMSE was 0.00643015/0.01448674; the previously nonfinite subset was 0.00251863/0.00251665. Both replays had zero NPU errors (native 1/1, Loop 64/64). The original capture access violation remains unresolved. No private activation, output tensor or model weight is distributed, so that private numerical comparison cannot be independently recomputed from this public bundle.

## CPU verification

Use NumPy 1.26.4 and Python with standard-library LZMA support:

```powershell
python -B verify_cpu.py
python -B verify_cpu.py --prepare C:/path/to/new-fixtures
python -B C:/path/to/new-fixtures/repro_native.py --evidence-root C:/path/to/new-fixtures --case loop_m128
```

Verification checks all publication hashes, regenerates each synthetic input and both stable references, checks exact array SHA256, and recomputes finite counts and errors from the archived observation bytes. The optional create-only `--prepare` command materializes inputs/reference files locally and copies the unchanged five-case runner. These generated files are not intended for publication. SHA differences fail explicitly instead of silently relaxing accuracy. No SDK or runtime is imported by `verify_cpu.py`.

## Optional native reproduction, not executed by this publication build

`repro_native.py` is the previously CPU-mocked minimal runner (SHA in the evidence). **This new public runner itself has not been executed on the NPU.** The original observed cases came from the per-case pinned `probe.py` or `probe_npu_token.py`, not this file. The unchanged runner supports `batched_m4`, `batched_m128`, `loop_m4`, `loop_m128`, and `no_cast_hints_m128`; it does not implement S1/-80 or private-capture replay. Those results retain explicit original-source associations, not a false claim of new-runner coverage.

Native execution must be explicitly selected and needs the user's matching installed Ryzen AI 1.8 provider, transaction archive, and xrt-smi. They are deliberately not redistributed. On an otherwise idle NPU:

```powershell
python -B C:/path/to/new-fixtures/repro_native.py --evidence-root C:/path/to/new-fixtures --case loop_m128 --execute-npu --provider-dll C:/path/to/onnxruntime_providers_ryzenai.dll --txn-archive C:/path/to/txn_bins.zip --xrt-smi C:/path/to/xrt-smi.exe --output C:/path/to/new-result
```

The runner pins the provider/transaction hashes, selects the typed NPU, disables fallback, binds BF16 buffers, permits one session/one execute/no retry, records own-process counters, and caps the child at 180 seconds and 2 GiB. Source fixtures are checked again after execution. A saved report and a clean process exit are recorded separately. It does not modify the SDK. The publication contains our scripts/generated synthetic data only, and no vendor runtime, transaction, xclbin, or source implementation.
