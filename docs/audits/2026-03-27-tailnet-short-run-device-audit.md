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
- calibration: `12` steps in `141 ms`
- calibration final loss: `15.942384`
- retained wallclock: `573532 ms`
- retained completed steps: `49152`
- retained steps per second: `85.7005`
- retained samples per second: `10969.6686`
- retained source tokens per second: `2451720.9432`
- retained initial loss: `6.932116`
- retained final loss: `0.0`
- retained loss delta: `6.932116`

### Relative Result

On the retained widened profile, the M5 beat the RTX 4080 box by:

- `88.71%` on steps per second
- `88.71%` on samples per second
- `88.71%` on source tokens per second

In plain language: on this current Rust-only same-node open-adapter training
lane, the local M5 is materially faster than the current 4080 CUDA path.

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
- the MLX path is currently stronger than the CUDA path for this exact Rust
  benchmark shape
- this does not prove that Apple Silicon will stay faster once the workload
  changes to a denser or more fused training path

So the honest reading is not "M5 always beats 4080 for training."

The honest reading is:

- for this current `psionic` home-network short-run lane, use the M5 first
- the current CUDA path still needs systems work before it wins on this class
  of bounded same-node training run

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
2. Use the RTX 4080 host as the second machine only after validating that the
   workload is large enough to justify the CUDA path.
3. Treat the M2 as opportunistic only. Use it when it is awake and reachable,
   but do not block the daily loop on it.

If the immediate goal is better useful-model progress in ten-minute windows,
the next sensible step is not a bigger audit. It is one more benchmark or
runtime iteration aimed specifically at why the current CUDA path is slower on
this widened adapter benchmark.

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
- the current admitted operator set is the M5 plus the RTX 4080 host
- the M2 now has a working SSH bootstrap path, but it is still excluded from
  the daily admitted set because availability is not stable enough yet
