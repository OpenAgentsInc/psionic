# Tailrun Admitted Home Tailnet Run Audit

> Status: retained 2026-03-27 audit for the first admitted-device home-Tailnet
> mixed-hardware run using the real `psionic-train`
> `first_swarm_trusted_lan_live_runtime` path.

## Scope

This audit records the first honest admitted-device home-Tailnet run after the
same-node M5 and RTX 4080 benchmark passes.

The admitted set for this proof is:

- local M5 MLX host
- remote `archlinux` RTX 4080 CUDA host

The M2 is explicitly **not** part of this proof. It stayed opportunistic and
did not block this run.

## Exact Command

The retained run used:

```bash
scripts/run-first-swarm-tailnet-admitted-live.sh \
  --run-id tailrun-home-admitted-20260327e \
  --bundle-dir fixtures/swarm/runs/tailrun-home-admitted-20260327e \
  --coordinator-port 35200 \
  --contributor-port 35201
```

That operator script:

- stages a minimal `psionic` source snapshot to remote `/tmp`
- avoids the dirty/full remote home checkout
- uses Tailnet IPv4 endpoints instead of bare LAN discovery
- runs the real `first_swarm_trusted_lan_live_runtime` on both machines
- writes the standard first-swarm bundle plus a Tailnet-specific per-device
  contribution summary

## Retained Artifacts

- bundle:
  `fixtures/swarm/runs/tailrun-home-admitted-20260327e/first_swarm_real_run_bundle.json`
- per-device summary:
  `fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json`
- coordinator report:
  `fixtures/swarm/runs/tailrun-home-admitted-20260327e/coordinator_runtime_report.json`
- contributor report:
  `fixtures/swarm/runs/tailrun-home-admitted-20260327e/contributor_runtime_report.json`
- operator manifest:
  `fixtures/swarm/runs/tailrun-home-admitted-20260327e/operator_manifest.json`
- logs:
  `fixtures/swarm/runs/tailrun-home-admitted-20260327e/logs/`

## Verified Outcome

The retained bundle verifies with:

```bash
scripts/check-first-swarm-trusted-lan-real-run.sh \
  --bundle fixtures/swarm/runs/tailrun-home-admitted-20260327e/first_swarm_real_run_bundle.json
```

Retained result:

- result classification: `bounded_success`
- total contributions: `2`
- accepted contributions: `2`
- replay-checked contributions: `2`
- submission receipt count: `2`
- merge disposition: `merged`
- publish disposition: `refused`
- promotion disposition: `held`

## Per-Device Contribution Split

The retained summary records one accepted contribution from each admitted
device.

### Local M5 MLX Coordinator

- endpoint: `100.127.107.31:35200`
- backend: `open_adapter_backend.mlx.metal.gpt_oss_lm_head`
- runtime role:
  `swarm.mac.mlx.coordinator_validator_contributor`
- executed steps: `12`
- batch count: `2`
- sample count: `12`
- payload bytes: `1472`
- final mean loss: `5.364421e-7`
- contribution share: `0.5`

### Remote RTX 4080 Contributor

- endpoint: `100.108.56.85:35201`
- backend: `open_adapter_backend.cuda.gpt_oss_lm_head`
- runtime role:
  `swarm.linux.cuda.rtx4080.contributor`
- executed steps: `12`
- batch count: `2`
- sample count: `12`
- payload bytes: `1480`
- final mean loss: `5.364421e-7`
- contribution share: `0.5`

## Timing Interpretation

The summary also records local execution windows and observed wallclock values,
but they should be read carefully:

- the coordinator wallclock includes validator and aggregation duties, not just
  local training work
- the contributor local execution window is very short and timestamp granularity
  is coarse enough that naive per-second rates are noisy

So this proof is strong on contribution truth, not on exact cross-device
throughput comparison. The throughput comparison still belongs to the bounded
same-node benchmark audit.

## Why This Operator Path Exists

The first attempts at this run surfaced two practical blockers that are now
encoded into the retained operator script:

- staging the whole repo was wasteful because tracked large fixture trees were
  not needed for this runtime
- the remote build still needed a specific compile-time fixture subset because
  `psionic-train` and its dependencies use `include_str!` on several fixture
  files

The retained script now stages only the actual build surface plus the exact
fixture trees required by the runtime compile.

## Honest Boundary

This audit does **not** claim:

- M2 participation
- open internet swarm membership
- elastic world-size changes
- full-model dense mixed-backend training
- automatic publish or automatic served-model promotion

It does claim:

- one admitted-device home-Tailnet run completed with the real `psionic-train`
  mixed-hardware runtime
- both admitted devices contributed real work in the same bounded run
- the run produced replay-checked contribution truth and a retained bundle
- the operator path is now repeatable without depending on the dirty/full
  remote home checkout
