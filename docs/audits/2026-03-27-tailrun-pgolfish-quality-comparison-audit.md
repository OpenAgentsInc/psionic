# Tailrun PGOLF-ish Quality Comparison Audit

> Status: retained 2026-03-27 audit for `TAILRUN-5`, covering the first honest
> PGOLF-shaped quality comparison across the admitted home-device short-run
> training artifacts.

## Purpose

The earlier Tailnet short-run work answered a speed question:

- how fast can the local M5 run the current bounded open-adapter lane
- how fast can the remote RTX 4080 run the same lane
- can one admitted mixed-device Tailnet run complete honestly

That was necessary, but not sufficient.

The daily goal is model improvement in bounded roughly-10-minute runs, not just
higher step counts. This audit adds one shared quality lens over the retained
artifacts:

- the same-node M5 retained bundle
- the same-node RTX 4080 retained bundle
- the admitted mixed-device Tailnet run, included honestly as training-summary
  context until it emits a promotable eval artifact

## Evaluation Flow

The retained evaluator is now:

- `cargo run -q -p psionic-train --bin open_adapter_pgolfish_quality_compare -- --output-root fixtures/apple_adapter/runs/tailrun_pgolfish_quality_compare_20260327`

That binary:

- reuses the widened open-adapter PGOLF-ish profile already used by the
  10-minute benchmark lane
- loads retained `portable_bundle.safetensors` artifacts from the committed
  M5 and RTX 4080 runs
- scores them on one distinct held-out synthetic split
- reports held-out cross-entropy loss and bits per token
- preserves the admitted mixed-device Tailnet run beside the same-node rows,
  but only through its retained training summary because that run still has no
  promoted PGOLF runtime bundle

The retained machine-readable output lives at:

- `fixtures/apple_adapter/runs/tailrun_pgolfish_quality_compare_20260327/quality_report.json`

## What “Mostly Under PGOLF Constraints” Means Here

This comparison is PGOLF-shaped, not exact PGOLF.

What stays aligned:

- bounded short-run training windows
- one explicit `SentencePiece`-style `SP1024`-ish tokenizer digest
- one fixed hidden width, vocab width, LoRA rank, and batch size
- one held-out loss and bits-per-token comparison instead of pure throughput
- portable retained artifacts rather than hand-waved claims

What stays outside strict PGOLF:

- this is an open-adapter lane, not the full promoted Parameter Golf decoder
  family
- the held-out set is synthetic and repo-owned, not a public contest dataset
- the admitted mixed-device swarm row is still training-summary-only, not a
  promoted inferable PGOLF bundle

## Retained Same-Node Quality Result

### M5 Same-Node Retained Bundle

- training steps per second: `162.53061053630358`
- training source tokens per second: `4649675.706222572`
- training final loss: `0.0`
- held-out mean loss: `15.942383766174316`
- held-out bits per token: `22.999997999408404`

### RTX 4080 Same-Node Retained Bundle

- training steps per second: `122.89196860911672`
- training source tokens per second: `3515693.437969611`
- training final loss: `0.0`
- held-out mean loss: `15.942383766174316`
- held-out bits per token: `22.999997999408404`

### Honest Reading

The current retained same-node result is:

- the M5 is still materially faster on the bounded Rust-only lane
- the RTX 4080 is now close enough to matter operationally
- neither retained same-node bundle shows any held-out quality advantage over
  the other

In plain language: today the extra throughput gap is real, but it is not yet
turning into a held-out quality gap on this bounded PGOLF-ish comparison.

That matters. It means the current short-run lane is still mostly measuring:

- how quickly each device can drive the current adapter-only training loop
- not yet how much useful generalization each device can buy in ten minutes

## Admitted Mixed-Device Tailnet Context

The admitted mixed-device run:

- run id: `tailrun-home-admitted-20260327e`
- admitted devices: `local_m5_mlx`, `archlinux_rtx4080_cuda`
- result classification: `bounded_success`
- publish disposition: `refused`
- promotion disposition: `held`
- aggregate retained training final mean loss: `5.364421e-7`
- weighted retained estimated steps per second: `3050.8474576271187`

Per-device retained summary:

- coordinator M5 MLX: `101.69491525423727 steps/s`, final mean loss `5.364421e-7`
- contributor RTX 4080 CUDA: `6000.0 steps/s`, final mean loss `5.364421e-7`

That is a real and useful retained result, but it is **not** yet part of the
same held-out winner table.

Why not:

- the run proved admitted mixed-device contribution, validation, replay, and
  merge
- it did not yet emit the promoted PGOLF runtime artifact needed for the same
  direct held-out bundle evaluation path

So the honest current boundary is:

- same-node rows: directly bundle-evaluated on the shared held-out profile
- admitted mixed-device row: retained training-summary context only

That exact gap is what `TAILRUN-6` is meant to close.

## Operational Takeaway

If the goal today is “best useful work in a 10-minute home-network run,” the
current best reading is:

1. use the M5 as the default same-node short-run device because it is still the
   fastest retained single-box lane
2. keep the RTX 4080 in the loop because the throughput gap is now small enough
   to justify continued tuning
3. do not pretend that either same-node retained artifact currently beats the
   other on held-out PGOLF-ish quality
4. prioritize the next promotion/export work so the admitted mixed-device run
   can enter this quality table honestly

## Next Epic Boundary

The next immediate issue after this audit is not more same-node profiling.

The next immediate issue is:

- `TAILRUN-6`: take one bounded home-device training artifact and line it up
  with the promoted PGOLF-shaped inference and serving path

That is the missing bridge between:

- “we can train bounded artifacts on reachable home devices”
- and “we can judge useful model improvement through the same inferable runtime
  that matters downstream”
