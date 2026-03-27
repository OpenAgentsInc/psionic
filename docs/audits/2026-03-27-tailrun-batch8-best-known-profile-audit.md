# Tailrun Batch-8 Best-Known Profile Audit

> Status: retained 2026-03-27 audit for `TAILRUN-8`, freezing the first
> best-known 10-minute PGOLF-ish home-device profile for the admitted daily
> Tailnet lane.

## Purpose

The daily loop was already real after `TAILRUN-7`, but it still needed one
bounded tuning program.

The important constraint in this pass was to improve the short-run lane without
breaking compatibility with the retained artifact family. That ruled out shape
changes like:

- hidden size
- vocab size
- LoRA rank

Those would have invalidated old retained bundles and made the comparison muddy.

So this pass tuned the first meaningful no-shape-break knob:

- PGOLF-ish batch size

## Search Space

The retained short search used the local M5 lane only, with the same open-adapter
family and a shorter wallclock budget to choose a candidate before spending the
full 10-minute admitted run.

Candidates tested:

- batch `8`
- batch `16`
- batch `24`
- batch `32`

Each candidate used:

- `45` second same-node M5 benchmark
- the same held-out PGOLF-ish quality compare path

## Short Search Result

### Batch 8

- M5 steps per second: `305.3211009174312`
- held-out mean loss: `15.942383766174316`

### Batch 16

- M5 steps per second: `160.42613191289362`
- held-out mean loss: `15.942383766174316`

### Batch 24

- M5 steps per second: `102.29565842840303`
- held-out mean loss: `15.942384481430054`

### Batch 32

- M5 steps per second: `74.47137324775913`
- held-out mean loss: `15.942383766174316`

## Why Batch 8 Won

Batch `8` was the first honest winner because:

- it improved M5 throughput by about `90.32%` over the old batch-`16` baseline
  in the short search
- it did **not** worsen held-out loss in the short search
- it preserved the same model shape and artifact compatibility

That made it strong enough to spend the full retained 10-minute admitted run on.

## Retained 10-Minute Result

The retained full run now lives at:

- `fixtures/apple_adapter/daily/tailrun-daily-batch8-retained-20260327/`

The retained scoreboard verdict is:

- `throughput_improved`

### M5

- old retained baseline: `162.53061053630358 steps/s`
- new batch-8 retained run: `304.3399208380537 steps/s`
- retained gain: `87.25%`

### RTX 4080 CUDA

- old retained baseline: `82.40252049829174 steps/s`
- new batch-8 retained run: `122.27966003430116 steps/s`
- retained gain: `48.39%`

### Held-Out Quality

- old best held-out mean loss: `15.942383766174316`
- new best held-out mean loss: `15.942383766174316`
- retained quality verdict: `noise_band`

### Infer/Serve Bridge

- direct match: passed
- served overlay match: passed

So the honest retained result is:

- throughput improved materially on both admitted devices
- held-out quality did not improve, but also did not regress
- the infer/serve bridge stayed green

## Best-Known Profile

The current best-known 10-minute admitted-device PGOLF-ish profile is now:

- hidden size: `512`
- vocab size: `1024`
- LoRA rank: `32`
- batch size: `8`
- training sample count: `128`
- holdout sample count: `64`
- wallclock budget: `600` seconds
- admitted device order:
  - local M5 MLX first
  - remote RTX 4080 CUDA second
  - M2 opportunistic only

That profile is now reflected in the default batch-size settings for:

- `open_adapter_pgolfish_profile.rs`
- `scripts/run-open-adapter-tailnet-matrix.sh`
- `scripts/run-tailrun-daily-loop.sh`

## What Was Rejected

The following candidates were rejected as best-known defaults:

- batch `16`: too much slower than batch `8` with no quality win
- batch `24`: much slower and slightly worse held-out loss
- batch `32`: much slower with no quality win

## Remaining Gap To Useful Small-Model Training

This still does **not** mean the home-device lane is training a useful product
model yet.

What improved:

- same exact 10-minute budget now buys much more throughput
- the daily operator default is now better than the old profile
- the loop still ends in a working infer/serve proof

What is still missing:

- held-out quality is still basically flat on this synthetic PGOLF-ish lane
- the model family is still bounded open-adapter training, not a useful dense
  small LM
- the admitted mixed-device swarm artifact still lacks the same inferable
  promoted or near-equivalent bundle path
- the M2 is still not part of the stable admitted daily set

So the right honest reading is:

- batch `8` is the best current way to spend the 10-minute budget on this lane
- it is a real operational improvement
- it is not the end of the small-model training problem
