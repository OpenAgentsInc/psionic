# Psionic Parameter Golf Single-H100 Banked-vs-Split Audit

> Date: 2026-03-26
> Scope: same-node H100 attribution proof for direct banked PGOLF execution
> vs explicit split-sliced comparator on the exact public single-H100 train
> step shape.

## Summary

Current `main` now has a retained same-node H100 attribution proof showing that
the direct banked PGOLF matrix path is materially faster than the explicit
split-sliced comparator on the exact public train-step shape.

Direct banked outcome:

- run root:
  `/workspace/parameter-golf-runpod/single-h100-bank-vs-split-20260326T041827Z-direct_banked`
- exact public-shape bounded command:
  `/root/psionic-target-proof-bank-vs-split/release/parameter_golf_single_h100_train /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model <report> 1 live_only non_overlapping`
- runtime posture:
  `matrix_execution_mode=direct_banked`
- retained receipt:
  `train_runtime_receipt` reached in `478.511s`

Explicit split comparator outcome:

- run root:
  `/workspace/parameter-golf-runpod/single-h100-bank-vs-split-20260326T042639Z-split_sliced`
- exact public-shape bounded command:
  `PSIONIC_PARAMETER_GOLF_MATRIX_EXECUTION_MODE=split_sliced /root/psionic-target-proof-bank-vs-split/release/parameter_golf_single_h100_train /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model <report> 1 live_only non_overlapping`
- runtime posture:
  `matrix_execution_mode=split_sliced`
- retained outcome:
  no `train_runtime_receipt` after `970s`, so the comparator was terminated
  once it exceeded `2x` the direct-banked receipt time (`957.022s`)

That is enough to attribute a material same-node H100 train-step improvement to
the direct banked execution path itself, not only to the other score-path work
that landed nearby.

## Exact H100 Results

Direct banked retained:

- `elapsed_to_train_runtime_receipt_seconds=478.511`
- `micro_step_count=8`
- `micro_step_forward_ms_sum=1490303`
- `micro_step_backward_ms_sum=506084`
- `micro_step_host_materialization_ms_sum=2667`
- `micro_step_retained_binding_f32_max=36642059168`
- `resident_parameter_upload_us=29280`
- `matrix_execution_mode=direct_banked`

Split-sliced retained before termination:

- `elapsed_seconds_without_train_runtime_receipt=970`
- `micro_step_count=6`
- `micro_step_forward_ms_sum=919528`
- `micro_step_backward_ms_sum=2023040`
- `micro_step_host_materialization_ms_sum=11458`
- `micro_step_retained_binding_f32_max=27580634808`
- `matrix_execution_mode=split_sliced`

Normalized over the first six micro-steps completed by both postures:

- direct banked forward: `870739 ms`
- split-sliced forward: `919528 ms`
- direct banked backward: `295550 ms`
- split-sliced backward: `2023040 ms`
- direct banked host materialization: `1555 ms`
- split-sliced host materialization: `11458 ms`

Relative deltas over those same six micro-steps:

- forward: split-sliced is `1.056x` slower
- backward: split-sliced is `6.845x` slower
- host materialization: split-sliced is `7.368x` slower

## Pod Observations

The same RunPod `NVIDIA H100 80GB HBM3` node was used for both passes.

Direct banked:

- early sample showed the trainer using about `39.9 GiB`
- later sample dropped to about `13.2 GiB`
- the run reached `train_runtime_receipt`

Split-sliced:

- repeated samples stayed around `29.6 GiB`
- `nvidia-smi` sampled `0%` GPU utilization while the trainer process stayed at
  `100%` host CPU near the cutoff
- the run never reached `train_runtime_receipt`

That is consistent with the explicit split comparator falling back into a
host-dominated execution posture while the direct banked path keeps the hot
matrix lane on the admitted CUDA banked execution surface.

## Conclusion

This audit closes the acceptance bar for:

- `#551` banked PGOLF model surface on the real train path
- `#558` real banked PGOLF execution replacing split-sliced bank execution

It does not close the broader train-step wallclock issue in `#546`. The direct
banked same-node H100 step is materially better, but the resulting
`478.511s` step receipt is still catastrophically slow relative to the public
scoreboard posture.
