# Coordinator Evolution Training (Khala M6, P3–P5)

This documents the learning-side primitives added for the Sakana-style learned
coordinator (TRINITY substrate), and how to actually train it. It is the
companion runbook for `crates/psionic-train/src/coordinator_evolution.rs` and
the GCloud job at `scripts/psion-coordinator-evolution-gcloud-job.sh`.

Roadmap source (in the `openagents` repo):
`docs/sakana/psionic-coordinator-roadmap.md` (P3/P4/P5 specs) and
`docs/sakana/coordinator-as-verified-work.md`.

## What landed

Built on the merged P1/P2 substrate (`Cs336A1TransformerLm::forward_with_hidden`
and `psionic_models::CoordinatorHead`):

- **P3 — separable CMA-ES** (`SepCmaEs`). Gradient-free. Samples a population of
  perturbed `CoordinatorHead` parameter vectors via
  `CoordinatorHead::flatten_parameters` / `with_flat_parameters`, evaluates each
  through the P4 hook, recombines by fitness-weighted rank mean, and updates a
  diagonal (separable) covariance plus a global step size. A `RandomSearch`
  baseline shares the same surface (the paper's control / sanity gate). The PRNG
  is an internal deterministic splitmix64 + Box–Muller, so a seeded run is
  bit-for-bit reproducible — required for stable ES fitness.

- **P4 — scalar terminal-reward adapter** (`TerminalRewardAdapter`) +
  atomic-evaluation hook (`CoordinatorFitness::evaluate_coordinator(params)->f32`).
  Reward is a verification **verdict** (`VerificationVerdict::{Verified,Rejected}`)
  mapped to `verified ? 1.0 : 0.0`, with optional cost shaping
  `reward − λ·cost` ("verified-work-per-sat"). Two concrete hooks ship:
  `ClosureFitness` (tests/smoke) and `FixtureCoordinatorEval` (materializes a
  real head per candidate, runs a fixture verdict batch — the offline lane).

- **P5 — typed, capability-filtered worker-pool binding** (`WorkerPoolBinding`).
  A candidate worker set is filtered to those whose **receipted capability
  envelope** covers the required capability, then sorted into a stable order;
  the head's `L` worker logits index into that filtered list. The coordinator
  selects *within* the eligible set and can never name a worker outside the
  receipt gate. Frontier endpoints are first-class members (`WorkerKind`).
  ACCEPT/halt is deliberately NOT a head output — it is the replay-validator
  verdict (per the verified-work doc).

## CPU smoke (this is a SMOKE, not a frontier result)

```
cargo run -q -p psionic-train --bin coordinator_evolution_smoke
```

Observed locally (deterministic, seed `0x5A6A4A4A`):

```
P5 worker pool: 2 eligible for `rust_build` -> ["frontier-a", "open-z"]
P2 head: hidden_dim=8 num_workers=2 num_roles=3 -> 40 params
P3 sep-CMA-ES result:
  initial fitness : 0.0000   (zero head routes the probe to the WRONG worker)
  best fitness    : 1.0000   (verified routing)
  improved        : true
  evaluations     : 1441
```

The fitness is a deterministic **fixture verdict** (Verified iff the head routes
a fixed probe hidden state to the capability-eligible "correct" worker). This
proves the optimizer drives a real `CoordinatorHead` from a failing start to a
verified routing on CPU. It says nothing about real coordination quality.

Unit tests (`cargo test -p psionic-train --lib coordinator_evolution`, 11 tests)
cover the reward adapter, the capability filter, ES improvement, ES-vs-random
at equal budget, determinism, and the full P2↔P3↔P4 seam on a real head.

## GCloud training-job spec

`scripts/psion-coordinator-evolution-gcloud-job.sh` (project `openagentsgemini`):

- **CPU-only by default** (`e2-standard-4`). Sep-CMA-ES is gradient-free and the
  TRINITY head is ~10K params, so the *optimizer* needs no GPU. The expensive
  part is the per-eval (real workers / sats on the live lane), which is metered
  and budgeted — not a VM cost.
- **Dry-run by default** — prints the exact `gcloud` command and provisions
  nothing. `--submit` creates a **bounded** VM (`--max-run-duration 900s`,
  `--instance-termination-action DELETE`, self-deleting startup script).
  `--teardown` deletes it.
- The startup script clones this branch, builds the crate, and runs the CPU
  smoke. **No GPU VM is provisioned without an explicit code change** — flagged
  by design so spend stays conservative.

## Pylon-served eval (where it slots in)

The P4 atomic eval is the natural seam for the **Pylon network** to do
distributed, parallel evaluation. Each `evaluate_coordinator(params)` call is an
independent end-to-end trajectory (select → role → dispatch → verify) — i.e. an
embarrassingly parallel population sweep. On the live lane:

- The sep-CMA-ES population (`λ` candidates/generation) fans out as `λ`
  independent eval jobs dispatched to Pylon contributor nodes, each running one
  coordinated trajectory over the P5 worker pool. This reuses the existing
  Pylon dispatch path (cf. the `qwen_legal_pylon_*` lanes in this crate) rather
  than a bespoke executor.
- The reward is the **Tassadar verdict** (`training.verification_classes.v1`:
  `exact_trace_replay` at sample rate 1.0 for deterministic work,
  `seeded_replication`/`statistical_cross_check` for stochastic LLM work) — the
  same verdict that releases settlement. The Verifier role binds to the replay
  validator, never a prompted LLM.
- A per-generation eval-budget cap (sats) must gate the fan-out and be emitted
  as a receipt; the fleet metering already attributes per-call cost, so the
  cost-aware denominator in `TerminalRewardAdapter::cost_aware` is free.

A live `CoordinatorFitness` impl that drives `probe_gepa_rollout_coordinator.rs`
and aggregates Pylon-returned verdicts is the next build step — see "What real
training still needs" below.

## M6 live-training lane (issue #6014, this branch)

The live-training wiring lands in
`crates/psionic-train/src/coordinator_live_training.rs` with a CPU driver bin
`crates/psionic-train/src/bin/coordinator_live_train.rs`:

- **`LiveCoordinatorFitness`** — the live `CoordinatorFitness`. Each candidate
  materializes a real `CoordinatorHead`, and for every `EvalSample` it computes
  the backbone hidden state via an injected `forward_with_hidden` closure (a
  frozen cs336 stack here; swap in Qwen3-0.6B on the production lane), runs
  `head.decide(h)`, resolves the worker through the P5 `WorkerPoolBinding`, asks
  an `EvalVerdictSource` for the verdict + spend, **debits the spend against the
  cap first (fail-closed)**, and aggregates with `TerminalRewardAdapter`.
- **`DailySpendCap`** — a hard, fail-closed daily cap clamped to the owner's
  10,000 sats/day (`OWNER_DAILY_CAP_MSATS = 10_000_000` msats). Same
  `spent_today / day_key / daily_cap` semantics as the shared buy-mode ceiling.
- **`EvalVerdictSource`** — the seam for the verdict. `SimulatedVerdictSource`
  is the deterministic, **zero-spend** validation source; the live lane binds this
  to `DispatchBackedVerdictSource` over either the signed Pylon TCP dispatcher or
  the default-off OpenAgents Worker HTTP dispatcher, both reading the Tassadar
  replay verdict instead of a prompted judge.

### Cap integration point (exact, cross-repo)

The owner budget is ONE shared autonomous daily ceiling whose authority is the
`openagents` Worker, not psionic:

- File: `apps/openagents.com/workers/api/src/buy-mode-dispatcher.ts`.
- Columns (minified at runtime, semantics stable): `daily_cap_msats`,
  `spent_today_msats`, `day_key`, `per_job_cap_msats`, `spend_enabled` on the
  buy-mode campaign row.
- Enforcement: `dispatchJob` / `recordSettlement` reject when
  `campaign.spentTodayMsats + amountMsats > campaign.dailyCapMsats`, emit
  `alert.buy_mode.daily_cap_breach`, and halt the campaign. Roll-over is by
  `day_key` (UTC date).

`DailySpendCap::can_spend` uses the **identical predicate**
(`spent_today + amount <= daily_cap`). On a real run, each
`LiveCoordinatorFitness` eval dispatched to Pylon MUST be one buy-mode
`dispatchJob` call against the SAME campaign row, carrying its
`per_job_cap_msats` / `daily_cap_msats`, so the local cap and the shared cap
debit the same budget. The local cap is the fail-closed backstop, never a
parallel budget.

### Validation result (no-spend smoke)

```
cargo run -q -p psionic-train --bin coordinator_live_train          # validate (no spend)
cargo run -q -p psionic-train --bin coordinator_live_train -- --real # bounded real run (HELD)
```

Observed (deterministic, seed `0x6014_6017`):

```
P5 worker pool: 3 eligible for `rust_build` -> ["frontier-claude", "open-pylon-a", "open-pylon-b"]
P2 head: hidden_dim=8 num_workers=3 num_roles=3 -> 48 params
LiveRunReport (simulated, no-spend):
  initial fitness : 0.3333
  best fitness    : 1.0000   (real frozen-backbone hidden states routed correctly)
  improved        : true
  evaluations     : 8001
  spent (msats)   : 0
  cap   (msats)   : 10000000
  halted on cap   : false
```

sep-CMA-ES drove a real `CoordinatorHead`, reading REAL frozen-backbone
`forward_with_hidden` hidden states, to route all 3 distinct samples to their
correct capability-eligible workers, **spending zero sats** while exercising the
cap path. This is a SMOKE (simulated verdict source), not a frontier ML result.

The `--real` run is **HELD**: a sat-moving run needs a live Pylon verdict source
(dispatch each trajectory as a buy-mode eval job), the Tassadar
`training.verification_classes.v1` verdict, an owner-armed dispatcher, and a
spend-enabled buy-mode campaign row to debit the shared cap. The signed TCP
dispatcher (`coordinator_live_buymode_dispatch.rs`) and the default-off HTTP
Worker bridge (`coordinator_http_buymode_dispatch.rs`) are now both present, but
neither arms itself and neither proves a live sat-moving deployment run. The
driver refuses to spend without shared-cap authority or to fabricate verdicts;
cap fail-closed is proven by
`coordinator_live_training::tests::live_fitness_fails_closed_at_the_cap`.

## What real training still needs

1. A **live `CoordinatorFitness`** wired to `probe_gepa_rollout_coordinator.rs`
   (replacing `FixtureCoordinatorEval`), aggregating Tassadar verdicts.
2. **`forward_with_hidden` on a frozen backbone** (Qwen3-0.6B) feeding real
   hidden states into the head, instead of the fixture probe tensor.
3. **Pylon fan-out** of the population eval + a **per-generation sat budget cap**
   emitted as a receipt.
4. **Compute/budget**: the optimizer is CPU-cheap; the cost is per-eval worker
   spend. The paper operates at 1.5k–40k evals for a ~10K-dim problem; our
   per-eval cost is higher (real workers), which is exactly the budget-tight
   regime where ES is supposed to win — but only if metered. Budget the live
   run before launching; the CPU smoke above is free.
5. **Shadow ship**: emit the trained head as a `Candidate` in
   `CompiledAgentPromotedArtifactContract`, shadow vs the NB route model on
   verified-work-per-sat, promote on a clean win.
