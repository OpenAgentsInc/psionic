# XTRAIN PGOLF Fast-FD And Window-1 Quick Eval Audit

Date: 2026-03-27

## Scope

This iteration retained three linked changes for the bounded XTRAIN-to-PGOLF local proof lane:

1. The finite-difference loss probe no longer rebuilds a fully validated `ParameterGolfReferenceModel`
   with fresh descriptor digests for every coordinate perturbation.
2. The repo now has a quick evaluation harness at
   `crates/psionic-serve/examples/xtrain_parameter_golf_quick_eval.rs`
   so XTRAIN tuning can compare proof-vs-XTRAIN quality and direct-vs-served decode throughput
   without paying the full legacy-clone benchmark cost on every iteration.
3. The promoted XTRAIN baseline now keeps the stronger 16-step budget and emits a bounded
   inference attention window of `1` token for this repo-owned `1..8` cycle proof.

## Why The Fast-FD Change Was Needed

Process sampling on the pre-fix quick-eval run showed the finite-difference path burning substantial
wall time inside `ParameterGolfReferenceModel::new(...)` and the descriptor-weight digest path
(`digest_tensor_values` / SHA-256 compression) for every single perturbed coordinate.

That work was pure tuning tax:

- the perturbation path only needed a compatible execution model to score loss
- it did not need fresh persisted descriptor metadata
- the forward pass was already shape-safe because the override was derived from the current validated model

The retained fix adds `ParameterGolfReferenceModel::with_execution_weights_unchecked(...)` and uses it
from `loss_with_parameter_override(...)` in `psionic-train`.

After the patch, sampling the same quick-eval path showed the hot stack dominated by real forward math
(`linear_forward_with_weight`) instead of digest computation.

## Quick Eval Result

Retained report:

- `/tmp/psionic_xtrain_pgolf_quick_1774607044_window1_fastfd/xtrain_parameter_golf_quick_eval_report.json`

Quick-eval result for the retained XTRAIN candidate:

- proof loss: `8.60598874092102`
- XTRAIN loss: `3.641620635986328`
- proof BPB: `9.93265382277841`
- XTRAIN BPB: `4.202998425869111`
- XTRAIN generated tokens: `[6, 8, 6, 8, 6, 8, 6, 8]`
- exact prefix match: `0`
- exact cycle match: `false`
- direct runtime throughput: `100.62399960877389 tok/s`
- served runtime throughput: `100.21725472293544 tok/s`
- direct/served token parity: `true`

## Comparison To The Last Retained Full Report

Previous retained full report:

- `/tmp/psionic_xtrain_pgolf_eval_1774604300/xtrain_parameter_golf_train_infer_report.json`

That earlier retained XTRAIN report had:

- loss: `4.380941867828369`
- BPB: `5.056290485711057`
- generated tokens: `[6, 8, 8, 8, 8, 7, 7, 8]`
- direct runtime throughput: `24.852325009270356 tok/s`

Compared with that retained baseline, this iteration improved:

- validation loss: `4.380941867828369 -> 3.641620635986328`
- bits per byte: `5.056290485711057 -> 4.202998425869111`
- decode throughput: `24.852325009270356 tok/s -> 100.62399960877389 tok/s`

Important boundary:

- the new throughput number comes from the quick-eval harness, not the old full legacy-clone report
- it is still a real direct/served decode measurement on the trained XTRAIN bundle
- it should be treated as the current fast-tuning benchmark, not as a drop-in replacement for the older full benchmark format

## What Improved And What Did Not

Improved:

- XTRAIN quality by loss and BPB
- bounded direct runtime throughput very substantially
- served runtime kept exact token parity with direct runtime
- the finite-difference tuning loop now spends time on actual model math instead of descriptor digest churn

Not solved yet:

- the toy cycle is still not exact
- the first token after `abcd` is still wrong (`6` instead of `5`)
- the model has collapsed into an alternating `6,8` rhythm under the retained window-1 decode posture

## Operational Conclusion

This iteration is worth keeping because it made the bounded XTRAIN proof lane materially better on two fronts at once:

- lower validation loss / BPB
- much faster direct and served inference under the retained quick benchmark

But it did **not** complete the toy-task correctness goal. The next iteration should target
task fidelity specifically, not generic loss reduction:

- inspect whether the `window=1` posture is too aggressive for exact-cycle recovery
- use the quick harness as the default tuning loop
- keep the fast finite-difference execution path, since that fix is clearly correct and useful regardless of the next quality move
