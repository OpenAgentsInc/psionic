# Tailnet Short-Run Device Audit

> Status: retained 2026-03-27 audit for the first honest home-Tailnet
> short-run device comparison after adding the Rust-only
> `open_adapter_same_node_wallclock_benchmark` binary in `psionic-train`.

## Scope

This audit records the current best honest answer to the question:

- can `psionic` do one bounded roughly-10-minute training run on the reachable
  home-network devices
- can those runs be compared with one shared Rust-only command
- does the current backend-heavy short-run lane show a meaningful performance
  difference instead of toy noise

The answer is now:

- yes for the local M5 MLX host
- yes for the remote RTX 4080 CUDA host
- not as part of the admitted daily operator set for the remote M2 host

## What Changed During This Pass

The first cut of the new same-node benchmark reused the tiny first-swarm sample
shape too literally. That version was not good enough:

- M5 MLX retained run: `349987.60 steps/s`
- RTX 4080 CUDA retained run: `353672.19 steps/s`
- delta: CUDA only `1.05%` faster

That result was too host-bound and too small to be useful for deciding where to
spend home-network training time.

The retained improvement in this pass was to widen the benchmark itself in
commit `330266e3`:

- synthetic `SentencePiece`-style tokenizer digest with `1024` vocab
- hidden size `512`
- LoRA rank `32`
- batch size `16`
- sample count `128`
- one same-node fixed-budget loop that stays alive until about `95%` of the
  requested wallclock budget is consumed

That widened profile made the backend split real enough to keep.

The next retained improvement in this audit was to fix the most important
short-run CUDA bottleneck:

- the current so-called CUDA lane was still doing the actual open-adapter
  forward and gradient accumulation math on the host CPU
- the CUDA label was real provenance, but not yet a full GPU-resident training
  hot path
- the retained fix parallelized per-sample gradient accumulation across the
  batch for the CUDA-labeled lane using Rayon

That does **not** make this full dense CUDA training, but it does make the
current admitted short-run CUDA lane materially faster and honestly closer to
the M5 result.

## Exact Command

Local M5 MLX:

```bash
cargo build -q -p psionic-train --bin open_adapter_same_node_wallclock_benchmark
target/debug/open_adapter_same_node_wallclock_benchmark \
  --backend-label open_adapter_backend.mlx.metal.gpt_oss_lm_head \
  --output-root /tmp/psionic_open_adapter_benchmark_m5 \
  --target-seconds 600
```

Remote RTX 4080 CUDA:

```bash
cargo build -q -p psionic-train --bin open_adapter_same_node_wallclock_benchmark
target/debug/open_adapter_same_node_wallclock_benchmark \
  --backend-label open_adapter_backend.cuda.gpt_oss_lm_head \
  --output-root /tmp/psionic_open_adapter_benchmark_cuda \
  --target-seconds 600
```

## Retained Results

### Local M5

- host: `ChristohersMBP2.lan`
- backend: `open_adapter_backend.mlx.metal.gpt_oss_lm_head`
- logical device: `metal:0`
- calibration: `12` steps in `92 ms`
- calibration final loss: `15.942384`
- retained wallclock: `576175 ms`
- retained completed steps: `93184`
- retained steps per second: `161.7286`
- retained samples per second: `20701.2661`
- retained source tokens per second: `4626732.9752`
- retained initial loss: `6.932116`
- retained final loss: `0.0`
- retained loss delta: `6.932116`

### Remote RTX 4080

- host: `archlinux`
- backend: `open_adapter_backend.cuda.gpt_oss_lm_head`
- logical device: `cuda:0`
- calibration: `12` steps in `73 ms`
- calibration final loss: `15.942385`
- retained wallclock: `574944 ms`
- retained completed steps: `70656`
- retained steps per second: `122.8920`
- retained samples per second: `15730.1720`
- retained source tokens per second: `3515693.4380`
- retained initial loss: `6.9321165`
- retained final loss: `0.0`
- retained loss delta: `6.9321165`

### CUDA Before/After

Using the admitted-device matrix run as the retained before value:

- before CUDA steps per second: `82.4025`
- after CUDA steps per second: `122.8920`
- retained CUDA gain: `49.14%`

The current admitted-device gap is now much smaller:

- local M5 MLX: `162.5306 steps/s`
- remote RTX 4080 CUDA after the retained fix: `122.8920 steps/s`
- M5-over-CUDA gap: `32.25%`

### Relative Result

On the retained current profile, the M5 still beats the RTX 4080 box by:

- `32.25%` on steps per second
- `32.25%` on samples per second
- `32.25%` on source tokens per second

In plain language: on this current Rust-only same-node open-adapter training
lane, the local M5 is still faster, but the retained CUDA fix made the remote
4080 clearly competitive instead of badly lagging.

## Why The Result Flips Against GPU Intuition

This benchmark is still not a full dense LM trainer and it is still not exact
Parameter Golf.

It is one backend-comparable open-adapter lane with:

- a much wider synthetic supervision set than the original first-swarm bring-up
- real MLX or CUDA backend labels
- real fixed-budget trainer steps
- real same-node training state evolution
- one exported portable bundle at the end

But it is still bounded by current `psionic` implementation details:

- the workload is adapter-only, not full-model dense training
- the so-called CUDA lane is still not full GPU-resident math for this exact
  benchmark shape
- the retained speedup came from parallelizing host-side per-sample gradient
  accumulation for the CUDA-labeled lane
- the MLX path is still currently stronger than the CUDA path for this exact
  Rust benchmark shape
- this does not prove that Apple Silicon will stay faster once the workload
  changes to a denser or more fused training path

So the honest reading is not "M5 always beats 4080 for training."

The honest reading is:

- for this current `psionic` home-network short-run lane, use the M5 first
- the current CUDA path is now strong enough to keep in the daily loop instead
  of treating it as obviously second-rate
- the next CUDA work should be about moving more of this lane into genuinely
  device-resident math, not just squeezing more host parallelism out of the
  current reference path

## Admitted Device Set

The current admitted daily operator set is:

- local M5 MLX host
- remote RTX 4080 CUDA host on `archlinux`

Those are the devices that can be reached and used honestly without turning the
daily loop into a wait-on-another-machine exercise.

## M2 Status

The M2 is no longer treated as part of the admitted daily operator set.

What changed after the original pass:

- `scripts/bootstrap-tailnet-key-auth.sh` now exists and can install this
  machine's SSH public key on a reachable Tailnet host with one password-based
  bootstrap
- that script successfully established non-interactive key auth to the M2 once
- the M2 then went offline mid-benchmark before it emitted a retained report
  or portable bundle

So the current honest state is:

What was attempted:

- Tailnet reachability: intermittent
- benchmark command ready: yes
- unattended shell access from this machine: yes after SSH key bootstrap
- retained benchmark artifact: no, because the host did not stay online long
  enough to finish the bounded run

Until the M2 stays awake and reachable for the full run budget, it is not part
of the admitted comparison set for the daily loop.

## Operational Recommendation

Use this ordering today:

1. Run short bounded same-node training on the local M5 first.
2. Run the RTX 4080 host in the same daily loop as the comparison and second
   retained lane, because the current CUDA path is now close enough to matter.
3. Treat the M2 as opportunistic only. Use it when it is awake and reachable,
   but do not block the daily loop on it.

If the immediate goal is better useful-model progress in ten-minute windows,
the next sensible step is not more same-node profiling. It is one retained
multi-device bounded run using the admitted M5 plus RTX 4080 pair, then adding
the M2 only when it stays awake for the whole window.

## Admitted Matrix Runner

The admitted-device matrix runner now lives at:

- `scripts/run-open-adapter-tailnet-matrix.sh`

Canonical retained admitted-device matrix artifacts now live at:

- `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/matrix_report.json`
- `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/m5_mlx/report.json`
- `fixtures/apple_adapter/runs/tailrun_pgolfish_quality_compare_20260327/quality_report.json`

## PGOLF-ish Quality Follow-Up

This audit is now paired with:

- `docs/audits/2026-03-27-tailrun-pgolfish-quality-comparison-audit.md`

That follow-up matters because the current same-node throughput advantage does
**not** yet turn into a held-out quality advantage on the shared PGOLF-ish
comparison profile:

- M5 same-node held-out mean loss: `15.942383766174316`
- RTX 4080 same-node held-out mean loss: `15.942383766174316`

So the honest current state is:

- throughput comparison is real and retained
- same-node held-out quality is currently tied
- the admitted mixed-device run is still summary-backed quality only until the
  next promotion/export issue lands
- `fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b/archlinux_cuda/report.json`

The exact retained command was:

```bash
scripts/run-open-adapter-tailnet-matrix.sh \
  --run-id tailrun-admitted-device-matrix-20260327b \
  --bundle-dir fixtures/apple_adapter/runs/tailrun_admitted_device_matrix_20260327b \
  --target-seconds 600
```

Retained admitted-matrix result:

- local M5 MLX: `162.5306 steps/s`
- remote RTX 4080 CUDA on `archlinux`: `82.4025 steps/s`
- local-over-remote gain: `97.24%`

That retained matrix run uses the admitted device set only and intentionally
does not block on the M2.

## Admitted Multi-Device Proof

The admitted-device home-Tailnet mixed-hardware proof now also exists at:

- `scripts/run-first-swarm-tailnet-admitted-live.sh`
- `fixtures/swarm/runs/tailrun-home-admitted-20260327e/first_swarm_real_run_bundle.json`
- `fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json`
- `docs/audits/2026-03-27-tailrun-admitted-home-tailnet-run-audit.md`

That proof is not the same thing as the same-node benchmark lane in this audit.
It proves that the admitted M5 plus RTX 4080 pair can contribute real work in
one bounded shared run. This audit remains the correct place for the direct
same-node throughput comparison.

## Retained CUDA Improvement Artifact

The retained after-fix CUDA artifact now lives at:

- `fixtures/apple_adapter/runs/tailrun_cuda_parallel_20260327/report.json`
- `fixtures/apple_adapter/runs/tailrun_cuda_parallel_20260327/portable_bundle.safetensors`

That retained run used the same benchmark family and the same ten-minute budget
as the admitted matrix baseline, but with the CUDA-labeled batch accumulation
path parallelized across samples.

## Honest Boundary

This audit does **not** claim:

- exact PGOLF parity
- full-model dense training across these devices
- one three-node live swarm run across M5 plus M2 plus RTX 4080
- automatic model promotion into serving
- useful product-model quality

It does claim:

- one shared Rust-only `psionic` benchmark command now exists for this lane
- the benchmark was improved from a misleading toy shape into a retained
  backend-heavy short-run profile
- the retained M5 and RTX 4080 ten-minute runs are real and comparable
- the retained CUDA follow-up run improved the remote RTX 4080 lane by `49.14%`
  without changing the benchmark family
- the current admitted operator set is the M5 plus the RTX 4080 host
- the M2 now has a working SSH bootstrap path, but it is still excluded from
  the daily admitted set because availability is not stable enough yet
