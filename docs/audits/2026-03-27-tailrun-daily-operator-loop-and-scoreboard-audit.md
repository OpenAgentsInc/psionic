# Tailrun Daily Operator Loop And Scoreboard Audit

> Status: retained 2026-03-27 audit for `TAILRUN-7`, freezing the daily
> admitted-device short-run operator loop and the first explicit scorekeeping
> contract for that loop.

## Purpose

The earlier Tailrun work proved separate pieces:

- an admitted-device same-node matrix for the local M5 and remote RTX 4080
- one PGOLF-ish held-out quality comparison
- one bounded near-equivalent infer/serve bridge for the retained M5 artifact

That was enough to make the work real, but not enough to make it repeatable.

The missing piece was one stable daily loop with:

- the exact command sequence
- the exact artifact roots
- the exact scorecard
- one explicit rule for “improvement” versus “noise”

This audit freezes that loop.

## Canonical Daily Entry Point

The daily operator entry point now lives at:

- `scripts/run-tailrun-daily-loop.sh`

The current best-known default profile for that loop is now:

- batch size `8`
- documented in `docs/audits/2026-03-27-tailrun-batch8-best-known-profile-audit.md`

Its job is to run or reuse the three retained stages:

1. admitted-device same-node matrix
2. PGOLF-ish held-out quality compare
3. M5 near-equivalent infer/serve bridge

Then it emits:

- `daily_scoreboard.json`
- `daily_scoreboard.md`

under one daily artifact root.

## Canonical Daily Order

The admitted daily operator order is now:

1. run the local M5 MLX same-node lane first
2. run the remote `archlinux` RTX 4080 CUDA lane in the same matrix second
3. run the PGOLF-ish held-out quality compare on the just-produced bundles
4. run the M5 near-equivalent infer/serve bridge on the just-produced M5 bundle
5. treat the M2 as opportunistic only and do not block the daily loop on it

That is deliberate. The loop is now designed to keep moving on the admitted
reachable set instead of waiting for the unstable M2.

## Exact Commands

The daily script freezes these command contracts:

### Matrix

```bash
scripts/run-open-adapter-tailnet-matrix.sh \
  --run-id <run_id> \
  --bundle-dir <matrix_root> \
  --target-seconds 600 \
  --batch-size 8 \
  --remote-host archlinux
```

### Quality

```bash
cargo run -q -p psionic-train --bin open_adapter_pgolfish_quality_compare -- \
  --output-root <quality_root> \
  --m5-report <matrix_root>/m5_mlx/report.json \
  --m5-bundle <matrix_root>/m5_mlx/portable_bundle.safetensors \
  --cuda-report <matrix_root>/archlinux_cuda/report.json \
  --cuda-bundle <matrix_root>/archlinux_cuda/portable_bundle.safetensors \
  --batch-size 8 \
  --admitted-home-summary \
    fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json
```

### Near-Equivalent Infer/Serve Bridge

```bash
cargo run -q -p psionic-serve \
  --example tailrun_open_adapter_near_equivalent_operator -- \
  --source-report <matrix_root>/m5_mlx/report.json \
  --source-bundle <matrix_root>/m5_mlx/portable_bundle.safetensors \
  --output-root <near_equivalent_root>
```

## Improvement Versus Noise

The daily scoreboard now uses explicit thresholds:

- throughput improvement threshold: `5%`
- held-out loss improvement threshold: `1%`
- near-equivalent bridge must pass both:
  - direct token match
  - served overlay token match

Daily score meanings:

- `meaningful_improvement`: metric moved past the threshold in the right direction
- `noise_band`: metric moved, but not enough to count honestly
- `meaningful_regression`: metric moved past the threshold in the wrong direction
- `passed` / `failed`: used for the near-equivalent bridge

The overall daily verdict is then summarized as one of:

- `quality_and_throughput_improved`
- `quality_improved`
- `throughput_improved`
- `stable_no_clear_gain`
- `regression`
- `bridge_failed`

## Artifact Layout

The default daily artifact root is:

- `fixtures/apple_adapter/daily/<run_id>/`

Inside that root the loop freezes:

- `matrix/`
- `quality/`
- `near_equivalent/`
- `daily_scoreboard.json`
- `daily_scoreboard.md`

The scoreboard also stores the baseline references it used for comparison:

- `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/matrix_report.json`
- `fixtures/apple_adapter/runs/tailrun_pgolfish_quality_compare_20260327/quality_report.json`
- `fixtures/apple_adapter/promoted/tailrun_m5_near_equivalent_20260327/near_equivalent_report.json`

## Retained Validation

The full live matrix stage costs real wallclock, so the retained validation in
this issue used the existing admitted matrix artifact and reran the later daily
stages against it:

```bash
scripts/run-tailrun-daily-loop.sh \
  --run-id tailrun-daily-scoreboard-validation-20260327 \
  --root-dir fixtures/apple_adapter/daily/tailrun-daily-scoreboard-validation-20260327 \
  --skip-matrix \
  --matrix-root fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b
```

That validation produced:

- `fixtures/apple_adapter/daily/tailrun-daily-scoreboard-validation-20260327/daily_scoreboard.json`
- `fixtures/apple_adapter/daily/tailrun-daily-scoreboard-validation-20260327/daily_scoreboard.md`

The retained verdict is currently:

- `stable_no_clear_gain`

That is the correct honest result, because the validation reused the same
retained baseline matrix artifact. The important result here is not a new speed
win. The important result is that the loop now produces one repeatable
scoreboard instead of scattered prose.

## Operational Meaning

This issue does **not** claim that the daily lane is now a useful product-model
trainer.

It does claim:

- the home Tailnet short-run loop is now explicit and repeatable
- the exact commands are frozen
- the scorekeeping threshold is explicit
- the M2 no longer blocks the daily operator flow
- the current best-known default profile is now batch size `8`
- later tuning can now be judged against a stable scorecard instead of ad hoc
  memory

## Next Boundary

With the daily loop frozen, the next work is narrower and cleaner:

- `TAILRUN-8`: improve the best-known 10-minute profile inside this exact loop

That is where actual retained quality and throughput gains should now be judged.
