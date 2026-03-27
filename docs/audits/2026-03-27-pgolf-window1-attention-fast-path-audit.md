# 2026-03-27 PGOLF Window-1 Attention Fast Path Audit

## Summary

This iteration does not change the trained PGOLF-shaped model family, the XTRAIN
lane, or the emitted promoted bundle format. It improves inference throughput
for the already-retained bounded decode path by adding a mathematically
equivalent fast path for the `bounded_attention_window_tokens = 1` case inside
the CPU Parameter Golf reference model.

The retained bundle benchmark target was the existing XTRAIN output at:

- `/tmp/psionic_xtrain_pgolf_quick_1774607044_window1_fastfd/xtrain_bundle`

That bundle already powered the previous retained report at:

- `/tmp/psionic_xtrain_pgolf_quick_1774607044_window1_fastfd/xtrain_parameter_golf_quick_eval_report.json`

## What Changed

The new fast path lives in:

- `crates/psionic-models/src/parameter_golf.rs`

When the model is asked to evaluate exactly one token under
`forward_logits_with_attention_window(..., 1)`, the causal attention law
collapses:

- there is only one legal source position
- the softmax weight is therefore exactly `1`
- the Q/K score path, RoPE application, and `q_gain` scaling cannot change the
  selected source position
- the attention output is just the grouped-query expansion of the current
  value-path vector, optionally followed by XSA and then the output projection

So the retained fast path skips the generic Q/K scoring loop and directly
materializes the one-token attended value surface before the existing output
projection. The dense path is unchanged. This means:

- training semantics are unchanged
- full-context inference semantics are unchanged
- bounded window `1` inference should stay output-identical while running
  faster

I also added a reusable benchmark example at:

- `crates/psionic-serve/examples/parameter_golf_promoted_runtime_benchmark.rs`

That example benchmarks direct runtime and `psionic-serve` runtime against an
already-emitted promoted PGOLF bundle without retraining it.

## Validation

Correctness tests run:

```sh
cargo test -q -p psionic-models single_token_window_one_matches_dense_forward_path -- --nocapture
cargo test -q -p psionic-models single_token_window_one_matches_dense_forward_path_with_xsa -- --nocapture
```

Those tests prove the fast path matches the unchanged dense forward path for:

- the baseline PGOLF config
- an XSA-enabled PGOLF variant

Benchmark run:

```sh
cargo run -q -p psionic-serve --example parameter_golf_promoted_runtime_benchmark -- \
  /tmp/psionic_xtrain_pgolf_quick_1774607044_window1_fastfd/xtrain_bundle abcd 16 2
```

## Measured Result

Previous retained quick-eval report:

- direct runtime: `100.62399960877389 tok/s`
- served runtime: `100.21725472293544 tok/s`
- generated tokens: `[6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8]`

New benchmark after the fast path:

- direct runtime: `123.35635320934547 tok/s`
- served runtime: `124.04032279555877 tok/s`
- generated tokens: `[6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8]`
- direct/served parity: `true`

Derived throughput improvement:

- direct runtime: about `22.6%` faster
- served runtime: about `23.8%` faster

## Claim Boundary

This iteration improves inference speed only. It does **not** claim:

- better trained-model quality
- a new retained XTRAIN training result
- exact toy-cycle correctness

The current retained XTRAIN model still emits the same alternating `6, 8`
pattern as before. The improvement here is that the bounded PGOLF inference
path now serves that retained model materially faster while preserving output
parity.
