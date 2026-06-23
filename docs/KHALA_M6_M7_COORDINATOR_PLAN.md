# Khala M6/M7 Learned-Coordinator Build Plan (Psionic)

Sequenced, honest build plan for the Khala learned-coordinator capability
(OpenAgents EPIC #6017; issues #6014 M6 TRINITY, #6015 M7 Conductor). The
coordinator **primitives** live in this repo (Psionic owns execution/coordinator
truth); the **spec** lives in the `openagents` repo under `docs/sakana/`
(`psionic-coordinator-roadmap.md` is the P1–P5 source of record) and
`docs/research/tmax/synthesis.md` §5 (DPPO + FP32 LM head recipe).

This doc states what P1–P5 concretely are in our stack, what already exists vs is
new, the dependency edges (especially the M4 real-pool blocker for full M6 and
M5+M6 for M7), the training loop, the shadow-candidate + promotion-gate
mechanics, and how the M7 GRPO Conductor layers on.

## TL;DR state (2026-06-22)

The P1–P5 learning-side substrate is **already implemented and merged** to
`psionic` `origin/main`:

- `#1133` P1 `forward_with_hidden` + P2 `CoordinatorHead` (+ adapter metadata).
- `#1134` P3 sep-CMA-ES + P4 reward adapter/atomic-eval + P5 worker-pool binding.
- `#1135` live coordinator-evolution training scaffold with a hard 10k sat/day
  fail-closed cap and a no-spend simulated validation pass.

What was still missing and is added in this change: the **Phase-4 shadow-decision
primitive** (`coordinator_shadow_comparison.rs`) — verified-work-per-sat learned
vs heuristic, confidence band, promote/hold/rollback recommendation — which is
the M6 "Done when" gate's decision logic and the seam the eventual
candidate-artifact emission consumes.

What remains genuinely new and unbuilt: the **typed candidate-artifact emission**
into `CompiledAgentPromotedArtifactContract` (contract-touching, deferred to a
reviewed change), the **live paid lane** (blocked on M4 real pool + a reachable
Pylon verdict source), and the entire **M7 Conductor GRPO lane**.

## P1–P5 in our stack: what each is, exists vs new

| Primitive | What it is here | Symbol(s) | State |
|---|---|---|---|
| P1 hidden-state extraction | `forward_with_hidden` on the frozen cs336 backbone returns last-token `h` off the same forward pass; deterministic. Qwen3-0.6B on the production lane. | `Cs336A1TransformerLm::forward_with_hidden` (`psionic-models/src/cs336_a1_reference_stack.rs:453`) | **EXISTS** (`#1133`) |
| P2 coordinator head + frozen backbone | Bias-free linear head `h -> (worker logits ‖ role logits)`, independent block softmaxes, argmax `decide`; `flatten_parameters`/`with_flat_parameters` is the optimizer seam. Adapter metadata `CoordinatorHead`/`CoordinatorRouter`. SVF is **not** built. | `psionic_models::CoordinatorHead` (`coordinator_head.rs`); `AdapterArtifactKind::CoordinatorHead`, `AdapterTargetFamily::CoordinatorRouter` | **EXISTS** (`#1133`); SVF NEW (unbuilt) |
| P3 separable CMA-ES | Gradient-free diagonal-covariance ES (Ros & Hansen scaling) maximizing fitness, deterministic splitmix64+Box–Muller PRNG, with a `RandomSearch` control. | `SepCmaEs`, `RandomSearch`, `SepCmaEsConfig` (`coordinator_evolution.rs`) | **EXISTS** (`#1134`) |
| P4 scalar terminal-reward adapter + atomic-eval | `VerificationVerdict -> verified?1:0` with optional `reward − λ·cost`; `CoordinatorFitness::evaluate_coordinator(params)->f32` hook with `ClosureFitness`, `FixtureCoordinatorEval`, and the live `LiveCoordinatorFitness`. | `TerminalRewardAdapter`, `TrajectoryOutcome`, `CoordinatorFitness` (`coordinator_evolution.rs`); `LiveCoordinatorFitness` (`coordinator_live_training.rs`) | **EXISTS** (`#1134`/`#1135`) |
| P5 typed worker-pool binding | Capability-filtered, stably-ordered `L` eligible workers; head worker logits index the eligible set, never overriding the receipt gate. Frontier endpoints are first-class members. | `WorkerPoolBinding`, `WorkerPoolMember`, `WorkerKind` (`coordinator_evolution.rs`) | **EXISTS** (`#1134`) |

Reward = the M2 verification verdict. Train on the verdict, monetize on
settlement (`docs/sakana/coordinator-as-verified-work.md`). ACCEPT/halt is the
replay-validator verdict, never a head output — enforced structurally:
`CoordinatorDecision` has no halt field, and the verdict enters only through
`EvalVerdictSource`.

## What this change adds (Phase-4 decision primitive)

`crates/psionic-train/src/coordinator_shadow_comparison.rs`:

- `ArmOutcome::from_outcomes` aggregates a `TrajectoryOutcome` stream into
  `(trajectories, verified, total_cost)`, with `verified_rate()` and
  `verified_work_per_sat()` (the roadmap's business metric; `None` when no sats
  moved).
- `ShadowComparison::compare(learned, heuristic)` produces the paired comparison:
  it picks the **verified-work-per-sat** lane when both arms have a positive sat
  denominator, else falls back to **verified rate** (offline/simulated) so the
  result is always defined; assigns a `ShadowConfidenceBand`
  (High ≥ 0.80 / Watch ≥ 0.60 / Review, matching
  `docs/COMPILED_AGENT_SHADOW_GOVERNANCE.md`); and emits a
  `ShadowRecommendation`:
  - `PromoteCandidate` — strict win **and** High band (eligible for an
    approval-gated `runtime_promotion`, never automatic);
  - `Rollback` — learned regressed vs the heuristic baseline;
  - `HoldShadow` — otherwise (watch band, tie, weak band).

It moves no sats, dispatches no work, starts no training. 10 unit tests cover the
paid lane, the offline fallback, the mixed lane, regression-rollback,
watch-band-holds-not-promotes, ties, and empty arms.

This is deliberately **not** the contract-emission wiring (see Phase 4 below).

## Dependency edges (honest)

```
P1 forward_with_hidden ─┐
P2 head (+ SVF NEW)     ─┼─> P4 reward/atomic-eval ─> P3 sep-CMA-ES ─> shadow-comparison ─> candidate-artifact ─> approval-gated promotion
P5 pool binding        ─┘        (offline lane first, paid lane second)

full M6 (paid, in-distribution win) ── BLOCKED ON ──> M4 real Pylon pool + reachable verdict source
M7 Conductor ── DEPENDS ON ──> M5 (Verse viz) + M6 (learned-coordinator substrate)
```

- **Offline lane is unblocked.** The whole P1–P5 + shadow-comparison loop runs on
  CPU today against fixture/simulated verdicts (zero spend). The
  `coordinator_live_train --validate` smoke already drives a real frozen cs336
  backbone + real head + sep-CMA-ES to a verified routing, 0 sats.
- **Paid lane (full M6) is blocked on M4.** A real shadow win on the
  in-distribution task set needs the real Pylon worker pool (M4, #6012) and a
  reachable verdict source (the Tassadar `training.verification_classes.v1`
  verdict, dispatched as buy-mode eval jobs). Until M4, the demo runs on the
  heuristic router and the learned coordinator stays an offline-validated
  candidate.
- **M7 is blocked on M5 + M6.** The Conductor's Verse fan-out view needs M5; its
  substrate (worker pool, verdict reward, governance) is M6.

## The sep-CMA-ES training loop (reward = verdict, cost-per-accepted-outcome)

1. Build the P5 `WorkerPoolBinding` for the task's required capability →
   `L = pool.len()` eligible workers (capability-filtered, stably ordered).
2. Configure a `CoordinatorHead` with `num_workers == L`, `num_roles == 3`.
3. For each sample: frozen `forward_with_hidden` → `h`; `head.decide(h)` →
   `(worker_index, role_index)`; resolve worker via the binding.
4. `EvalVerdictSource` returns `(verdict, spend_msats)`. On the paid lane debit
   spend against the `DailySpendCap` **first** (fail-closed at the owner's 10k
   sat/day ceiling); on the offline lane spend is 0.
5. `TerminalRewardAdapter` aggregates: offline `verified?1:0`; cost-aware
   `reward − λ·cost` so fitness is **verified-work-per-sat**, the actual business
   objective, not raw pass rate.
6. `SepCmaEs::optimize` samples a population of perturbed head params, evaluates
   each via the fitness hook, recombines by fitness-weighted rank mean, updates
   the diagonal covariance + step size. `RandomSearch` at equal budget is the
   sanity control.

Budget discipline: every paid eval may move sats, so the per-generation eval
budget × per-eval spend must fit under the daily cap; the cap halts the run
before breach.

## Shadow-candidate + promotion-gate mechanics

1. Train the head offline (fixture/simulated verdict) until the optimizer
   reliably improves and the `coordinator_live_train --validate` smoke is green.
2. Collect two `TrajectoryOutcome` streams over the **same** in-distribution
   samples — one from the learned coordinator, one from the heuristic router —
   and run `ShadowComparison::compare`. `PromoteCandidate` requires a strict
   verified-work-per-sat win in the High band.
3. **Emit a Candidate entry** (deferred, reviewed) in
   `CompiledAgentPromotedArtifactContract` with
   `candidate_label = "coordinator_sep_cmaes_v1"`, alongside the promoted
   heuristic/route model, keeping the heuristic router as `rollback_artifact_id`.
   This step touches an existing runtime contract (the contract is currently
   keyed to `CompiledAgentModuleKind` route/grounded modules, not a generic
   coordinator), so it lands as a **reviewed PR**, not a direct push.
4. Shadow-run on the paid lane (needs M4); compare on verified-work-per-sat under
   the confidence bands. Promotion is an **approval-gated `runtime_promotion`**
   (same governance as Artanis authority) — never automatic. Any held-out
   regression trips `Rollback`.

## M7 GRPO Conductor (DPPO + FP32 LM head)

The Conductor lane (#6015) reuses P4/P5/governance and the shadow-comparison
primitive unchanged, and swaps P1–P3 (ES over a tiny head) for a **GRPO loop over
a 7B base** that emits an NL workflow (`model_id` / `subtasks` / `access_list`
parallel lists). From `docs/research/tmax/synthesis.md` §5, adopt the stability
recipe:

- **FP32 LM head** — the cheap, high-leverage fix for the training–inference
  logprob mismatch (high-frequency tokens like `\n` drive the worst mismatch when
  rollouts come from a fast serving path and gradients from the trainer).
- **DPPO over GRPO** — mask tokens where inference/training logprobs diverge
  (binary TV threshold 0.1); limits training collapse.
- **Filter zero-std samples**, group size 32, KL β = 0, constant LR 1e-6,
  centered advantage (TMAX Table-13 starting config).

The ES (TRINITY) lane mostly avoids logprob mismatch entirely, which is why it is
the cheaper, higher-leverage first build; DPPO/FP32 matter only for the
Conductor RL lane. M7 also needs the Verse multi-worker fan-out view (M5).

## Effort / risk

| Slice | Effort | Risk | Blocked on |
|---|---|---|---|
| P1–P5 substrate | done | — | — |
| Phase-4 shadow-comparison primitive | done (this change) | low | — |
| SVF adapter (P2 optional) | small | low | — |
| Candidate-artifact emission into the contract | medium | medium (contract-touching) | reviewed PR |
| Live paid lane (full M6 win) | medium | medium (cost, LLM nondeterminism) | **M4 real pool** + Pylon verdict source |
| M7 Conductor GRPO (DPPO+FP32) | large | high (RL stability) | **M5 viz + M6** |

## Concrete next slice (after this change)

1. **SVF adapter** (P2 optional) behind `CoordinatorHead` — small, additive, may
   lift representation per TRINITY ablations; gate it on whether the offline lane
   plateaus.
2. **Candidate-artifact emission** — DONE (`#1136`, merged):
   `coordinator_candidate_emission.rs` ships a trained head as a digest-pinned
   Candidate under `CompiledAgentPromotedArtifactContract` with the heuristic
   router as `rollback_artifact_id`, consuming `ShadowComparison` for the
   promote/hold/rollback decision.
3. **Pylon verdict source** for the paid lane — PLUMBING DONE (this change):
   `coordinator_eval_verdict_source.rs` adds `DispatchBackedVerdictSource`, a real
   `EvalVerdictSource` over the buy-mode dispatch path (`BuyModeDispatch` seam) that
   reads the `training.verification_classes.v1` verdict, yields the scalar terminal
   reward, and feeds `LiveCoordinatorFitness` / `ShadowComparison`. It is
   **disarmed by default** (`CoordinatorArmState::Disarmed`) and **fail-closed
   behind the daily cap** (an over-cap or disarmed request dispatches nothing and
   moves no sats). The remaining **owner-gated** work is a *live* `BuyModeDispatch`
   that publishes to the real Pylon pool (M4, #6012, merged) and reads the settled
   gateway verdict, an armed source, and a spend-enabled buy-mode campaign row.
   This change provides the seam and the fixture lane; it never fabricates a
   verdict and never dispatches in tests.

## Build / run anchors

- Offline smoke: `cargo run -q -p psionic-train --bin coordinator_evolution_smoke`
- Live validation (no spend): `cargo run -q -p psionic-train --bin coordinator_live_train`
- Tests: `cargo test -p psionic-train --lib coordinator`,
  `cargo test -p psionic-train --lib coordinator_eval_verdict`,
  `cargo test -p psionic-models --lib coordinator_head`
- Companion runbook: `docs/COORDINATOR_EVOLUTION_TRAINING.md`
- Spec (openagents): `docs/sakana/psionic-coordinator-roadmap.md`,
  `docs/sakana/coordinator-as-verified-work.md`,
  `docs/sakana/tassadar-run-integration.md`, `docs/sakana/trinity.md`,
  `docs/sakana/conductor.md`, `docs/research/tmax/synthesis.md` §5.
