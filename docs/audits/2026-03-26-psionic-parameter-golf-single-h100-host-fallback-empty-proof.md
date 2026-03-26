# 2026-03-26 Psionic Parameter Golf Single-H100 Host-Fallback-Empty Proof

This audit records a fresh exact public-shape same-node H100 rerun of the
current direct-banked PGOLF train step with the CUDA host-fallback profiler
enabled.

## Conclusion

Current `main` still takes hundreds of seconds on the exact public single-H100
train step, but that remaining wallclock is not explained by the CUDA backend's
explicit host-fallback path.

Fresh same-node H100 results on the exact public train shape:

- `train_time=472430ms`
- `forward_ms=357241`
- `backward_ms=109342`
- `host_materialization_ms=580`
- `retained_binding_f32=36642059168`

The profiler sink for
`PSIONIC_CUDA_HOST_FALLBACK_PROFILE_PATH=<run-root>/host_fallback.jsonl`
remained empty at the end of the retained train-step proof.

That means the exact-shape `direct_banked` step still misses the score target,
but the miss is now a narrower one: it is inside the admitted CUDA path rather
than hidden `Permute`, `ReduceSum`, `Expand`, `Add`, or other explicit
host-fallback replay inside the train step.

## Run Identity

- pod device: `NVIDIA H100 80GB HBM3`
- proof binary:
  `/root/psionic-target-proof-df70fcea/release/parameter_golf_single_h100_train`
- live run root:
  `/workspace/parameter-golf-runpod/single-h100-proof-20260326T091705Z-host-fallback-profile`
- runtime posture:
  `matrix_execution_mode=direct_banked`
- final validation mode:
  `roundtrip_only`
- host fallback sink:
  `/workspace/parameter-golf-runpod/single-h100-proof-20260326T091705Z-host-fallback-profile/host_fallback.jsonl`

Exact trainer command shape:

```bash
PSIONIC_PARAMETER_GOLF_MATRIX_EXECUTION_MODE=direct_banked \
PSIONIC_CUDA_HOST_FALLBACK_PROFILE_PATH=/workspace/parameter-golf-runpod/single-h100-proof-20260326T091705Z-host-fallback-profile/host_fallback.jsonl \
  /root/psionic-target-proof-df70fcea/release/parameter_golf_single_h100_train \
  /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /workspace/parameter-golf-runpod/single-h100-proof-20260326T091705Z-host-fallback-profile/parameter_golf_single_h100_training.json \
  1 \
  roundtrip_only \
  non_overlapping
```

The run was launched through `script -qefc` so the trainer emitted line-buffered
output to a PTY-backed transcript.

## Observed Same-Node H100 Output

Fresh PTY-backed output captured during the retained train-step proof:

```text
micro_step_complete step=1/1 micro_step=8/8 window_id=42845a4917f7d6f04439320a0c3f0c34ca83982ff80685ae3eea363b67381c13 train_loss=8.29108715 forward_ms=357241 backward_ms=109342 host_materialization_ms=580 retained_binding_f32=36642059168 gradient_f32=136479296
train_step_complete step=1 mean_microbatch_loss=8.28840446 lr_mult=1.00000000 muon_momentum=0.85000002 host_materialization_ms=580 optimizer_step_ms=3535
train_runtime_receipt step=1 path=device_resident_cuda_training_graph_v1 graph_surface=parameter_golf_baseline_training_graph_v2 matrix_execution_mode=direct_banked sessions=1 stable_parameter_buffers=42 stable_parameter_values=17059912 resident_parameter_upload_us=30836 parameter_refresh_us=0 input_token_write_us=1466 target_token_write_us=952 resident_buffers_reused=true
step:1/1 train_loss:8.2884 train_time:472430ms step_avg:472430.00ms
final_validation_skipped mode=roundtrip_only reason=explicit_final_validation_mode
```

The proof wrapper then interrupted the follow-on roundtrip validation sweep
after the train-step receipt was captured.

## Host-Fallback Profile Outcome

The live run root retained an empty fallback profile sink:

```text
/workspace/parameter-golf-runpod/single-h100-proof-20260326T091705Z-host-fallback-profile/host_fallback.jsonl
size: 0 bytes
```

That is the important result from this audit.

If the exact-shape train step were still spending meaningful time in the CUDA
backend's explicit host-fallback surface, this sink would contain one or more
JSON lines with per-op fallback timings. Instead, it remained empty on the run
that emitted the `472430ms` train-step receipt.

The final JSON report was not retained from this rerun because the proof wrapper
stopped the follow-on roundtrip validation sweep after the train-step receipt
landed. The retained truth for this audit is the PTY-backed console output plus
the empty profiler sink in the run root above.

## Comparison Against The Earlier Same-Node Baseline

Retained earlier direct-banked same-node H100 baseline:

- `478.511s`
- source:
  `docs/audits/2026-03-26-psionic-parameter-golf-single-h100-banked-vs-split-audit.md`

Fresh profiler-enabled direct-banked same-node H100 proof:

- `472.430s`

Relative delta:

- step wallclock: `478.511s -> 472.430s` down about `1.27%`

That is a real but still minor movement. It is nowhere near enough to close the
exact public-shape hot-path issue.

## Issue Impact

This proof does not close any open issue.

It sharpens `#546`:

- the remaining exact-shape same-node wallclock is still hundreds of seconds
- the admitted CUDA forward and backward kernels remain the dominant blocker
- the remaining miss is not explained by the backend's explicit host-fallback
  path on the retained train step
