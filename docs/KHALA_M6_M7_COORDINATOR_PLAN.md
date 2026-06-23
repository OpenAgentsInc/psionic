# Khala M6/M7/M8 Learned-Coordinator Build Plan (Psionic)

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
Pylon verdict source), and the **M7 Conductor real GRPO training run** + the
paid composition demo (the **M7 scaffold** — plan contract, planner, GRPO/DPPO
trainer — is now built and tested as `coordinator_conductor.rs`, inert until
armed; see the M7 scaffold-status section below).

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

### M7 scaffold status (2026-06-23) — `coordinator_conductor.rs`

The M7 Conductor scaffold is landed as a reviewed PR
(`crates/psionic-train/src/coordinator_conductor.rs`), built + tested on CPU with
no spend and no model weights. It reuses the M6 substrate unchanged. What it
contains, and what stays owner/compute-gated:

**Real + offline-proven (tests green):**

- **The plan contract** — `ConductorPlan` materializes the Conductor paper's
  three parallel lists (`model_id` / `subtasks` / `access_list`) as a validated
  `Vec<ConductorStep>`. `AccessList` (`All` / `None` / `Indices`) is the
  access-list topology; validation enforces a DAG over the linear ordering (no
  self/forward edges), bounds the step count to `max_steps` (the paper's 5-step
  cap), and rejects any `worker_index` outside the **M6 `WorkerPoolBinding`** —
  so the language plan can never name a worker the capability gate already
  filtered out. `parse` is a deterministic, bounded-field structuring step that
  runs *after* the semantic decode (no intent keyword-matching). `worker_fanout`
  is the compose-across-the-map set the Verse view (M5) will render;
  `resolve_worker_ids` maps indices to stable ids through the binding.
- **The planner stepping interface** — `ConductorPlanner<P: ConductorPolicy>`:
  decode (via an injected `ConductorPolicy` text-generation seam, so tests run
  with a fixture, not a 7B) → parse → validate → `step()` yields a
  `PlanStepOutcome` (resolved worker id, subtask, access, `is_final`) for the
  plan→implement→verify→refine loop. A malformed model output is surfaced as a
  structured `ConductorError` (the GRPO format-condition reward-0 signal), never
  a panic.
- **The GRPO/DPPO trainer scaffold with the TMAX recipe** —
  `ConductorTrainerConfig::tmax_table13()` captures FP32 head ON, DPPO TV mask
  0.1, zero-std filtering ON, group size 32, KL β 0, LR 1e-6, centered advantage.
  `ConductorTrainer::update_step` groups rollouts by `prompt_id`, **filters
  zero-std GRPO groups** (`GrpoGroup::is_zero_std`), computes **centered
  advantages**, applies the **DPPO TV-mask** (`DpppoUpdate::token_is_masked`,
  with the **FP32-head** path treating the trainer logprob as exact so the
  measured divergence is the real policy gap, not a precision artifact), and
  emits a deterministic `GrpoUpdateStep` summary. **Reward = the M6
  `EvalVerdictSource` verdict** via `TerminalRewardAdapter` (verified-work, not
  raw pass rate). It applies **no gradient** — there is no autograd/serving
  backend in the scaffold; it proves the *loop steps*, not that the Conductor is
  good.

**Default-off / fail-closed (tests assert):**

- `ConductorTrainer` is **`Disarmed` by default** (same `CoordinatorArmState` as
  the M6 lanes). `guard_paid_rollout` refuses cleanly while disarmed (no spend,
  no dispatch); even when **armed** it pre-checks the **`DailySpendCap`** and
  fails closed over-cap; the cap clamps to the owner's 10,000 sat/day ceiling
  (`OWNER_DAILY_CAP_MSATS`) as a hard upper bound. Nothing in this module
  dispatches work, moves sats, or starts a training run. The openagents gateway
  verdict shapes are consumed read-only.

**Owner / compute-gated — the remaining gates (`ConductorReadiness`):** every
field is `false` in the shipped scaffold; flipping them is owner/compute work,
not code that lands in this scaffold:

1. `policy_backend_wired` — a 7B base policy + FP32 head + autograd/serving split
   (compute).
2. `training_run_executed` — a real GRPO run that converges (H100-hours).
3. `paid_verdict_source_armed` — `EvalVerdictSource` **armed** over the live
   Pylon pool (M4, #6012) + a spend-enabled buy-mode campaign (owner).
4. `paid_shadow_win_recorded` — an M6 paid `ShadowComparison`
   verified-work-per-sat win over single-model (owner + M6).
5. `crossy_road_composition_verified` — the crossy-road composition beats
   single-model cost at comparable quality under the M2 rubric (the #6015
   Done-when proof).

## M8 head-to-head evaluation harness (2026-06-23) — `coordinator_m8_head_to_head.rs`

M8 (OpenAgents issue #6016) is the north-star head-to-head demo: publish our own
Fugu-Ultra-vs-frontier comparison where **`openagents/khala` solves the
benchmark BY COMPOSITION, verified, cheaper than a single model**. The
publication-side machinery (evidence manifest, metric reducer, closure audit,
fixture pack) already lives in the `openagents` repo
(`scripts/khala-demo/reduce-head-to-head.mjs`,
`docs/inference/khala-head-to-head-demo.md`, the `khala_head_to_head_evidence.v1`
fixture). What was missing — and what this change adds in Psionic, which owns
coordinator/execution truth — is the **evaluation harness that produces a lane's
metrics in the first place**: composed (M7 Conductor over the M6 pool) vs a
single-model baseline on a fixture task set, scored on quality + cost, with the
deterministic win verdict the "compose to win, cheaper" claim rests on.

### M8 scaffold status — what is real vs owner/compute-gated

`crates/psionic-train/src/coordinator_m8_head_to_head.rs`:

**Real + offline-proven (tests green, fixture-backed, no spend):**

- **`M8HeadToHeadHarness`** runs the composed arm vs the single-model baseline
  over a `HeadToHeadTaskSet`. For each task it **builds + validates the Conductor
  plan over the M6 `WorkerPoolBinding`** (so the composed arm can only fan out
  across receipt-eligible workers — the capability gate is honored, a plan naming
  an out-of-pool worker is a structured `ComposedPlanFailed`, never a silently
  scored outcome), and it **validates the baseline's worker index against the
  same pool**. It reuses the M7 `ConductorPlanner`/`ConductorPolicy` seam, so
  tests run with a fixture plan, not a 7B.
- **`ArmCostMetric`** computes, per arm, the exact gateway / demo-reducer
  vocabulary: **accepted rate** (the reducer's `verifiedRate`),
  **cost-per-accepted-outcome** (`total_cost / accepted`, the gateway's headline
  metric; `None`/"not_applicable" when nothing was accepted), and
  **verified-work-per-sat** (`accepted / total_cost`, the roadmap business
  metric; `None` only on the offline lane where no sats moved). Both arms are
  aggregated via the M6 `ArmOutcome`/`TerminalRewardAdapter` and scored through
  the M6 `ShadowComparison` decision logic unchanged.
- **`HeadToHeadReport`** is the deterministic, typed, serde-round-tripping report:
  composed vs single-model on quality + cost, the underlying `ShadowComparison`,
  the composed worker fan-out, and the M8 **`HeadToHeadVerdict`**:
  - `ComposeToWinCheaper` — fires ONLY when the comparison ran on the paid
    (verified-work-per-sat) lane, composition **strictly** beat single-model on
    that cost lane, AND quality stayed comparable (composed accepted rate within
    `QUALITY_PARITY_EPSILON` = 0.05 of single-model's). This is the only verdict
    that supports publishing "compose to win, cheaper".
  - `CheaperButLowerQuality` — composition was cheaper per-sat but its quality
    dropped below parity (the per-sat win came from accepting fewer outcomes, not
    from being better). Not an honest win.
  - `SingleModelNotBeaten` — single-model was at least as good on cost (ties
    included; a tie is not a win).
  - `NoCostLaneOffline` — at least one arm moved no sats, so there is no cost win
    to claim; the cheaper-than-single-model claim requires the paid lane.

The harness has **no dispatch seam at all**: a run only consumes fixture
outcomes and produces a report. It dispatches no work, moves no sats, and starts
no training. 17 unit tests cover the metric math, the win verdict (each
verdict variant), the quality-parity tolerance, the capability-gate honoring
(invalid composed plan + out-of-pool baseline both error, not panic), empty-set
rejection, determinism, and serde round-trip.

**Owner / compute-gated — the remaining gates (`M8HeadToHeadReadiness`):** every
field is `false` in the shipped harness (mirrors M7's `ConductorReadiness`);
flipping them is owner/compute work, not code that lands here:

1. `composed_arm_live` — the composed arm runs a real trained Conductor policy
   (7B over the M6 pool), not a fixture plan (compute; depends on M7
   `policy_backend_wired` + `training_run_executed`).
2. `single_model_baseline_live` — the baseline runs a real frontier endpoint, not
   a fixture outcome.
3. `paid_verdict_source_armed` — the `EvalVerdictSource` is **armed** over the
   live Pylon pool (M4, #6012) with a spend-enabled buy-mode campaign, so the
   per-arm outcomes are real verified evals (owner).
4. `paid_compose_to_win_recorded` — a paid head-to-head where composition's
   verified-work-per-sat beat single-model at comparable quality under the M2
   rubric (the #6016 Done-when proof; owner + M4 + M6).
5. `demo_closure_audit_passes` — the openagents-repo reducer's closure audit
   returns `canClose: true` for a *live* manifest (publication-side, owner).

### M8 Done-when next slice (after this change)

1. **Wire the composed arm to the M7 live policy** once `ConductorReadiness`'s
   `policy_backend_wired` + `training_run_executed` flip — replace the fixture
   `ConductorPolicy` with the trained 7B served over the M6 pool. Compute-gated.
2. **Arm the verdict source** over the live M4 Pylon pool with a spend-enabled
   buy-mode campaign and a real frontier baseline endpoint, then run the harness
   on the live crossy-road task set to record a paid `ComposeToWinCheaper`
   verdict. Owner-gated (arming + spend-enabled campaign + the daily cap).
3. **Hand the per-lane metrics to the publication reducer** — the harness's
   `ArmCostMetric` is the per-lane shape the openagents
   `reduce-head-to-head.mjs` manifest carries; feeding a live report into a live
   manifest is what flips the reducer's `canClose` to `true` and closes #6016.

## Effort / risk

| Slice | Effort | Risk | Blocked on |
|---|---|---|---|
| P1–P5 substrate | done | — | — |
| Phase-4 shadow-comparison primitive | done (this change) | low | — |
| SVF adapter (P2 optional) | small | low | — |
| Candidate-artifact emission into the contract | medium | medium (contract-touching) | reviewed PR |
| Live paid lane (full M6 win) | medium | medium (cost, LLM nondeterminism) | **M4 real pool** + Pylon verdict source |
| M7 Conductor scaffold (plan contract + planner + GRPO/DPPO trainer, inert) | done (this change) | low | — |
| M7 Conductor real GRPO run (DPPO+FP32 over a 7B) | large | high (RL stability) | **compute** (policy backend + H100-hours) |
| M7 paid composition demo (crossy-road, cheaper than single-model) | large | high | **arm + M4 paid lane + M6 shadow-win + M2 rubric** |
| M8 head-to-head eval harness (composed-vs-single, fixture, inert) | done (this change) | low | — |
| M8 armed head-to-head run (live composed + frontier baseline, paid) | large | high | **M7 live policy + arm + M4 paid lane + M2 rubric + demo closure audit** |

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
4. **M7 Conductor scaffold** — DONE (this change):
   `coordinator_conductor.rs` ships the typed plan contract (`ConductorPlan` over
   the M6 pool), the planner stepping interface (`ConductorPlanner` /
   `ConductorPolicy` / `PlanStepOutcome`), and the GRPO/DPPO trainer scaffold
   (`ConductorTrainer` / `ConductorTrainerConfig::tmax_table13` / `DpppoUpdate` /
   `GrpoGroup` / `GrpoUpdateStep`) with reward = the M6 `EvalVerdictSource`
   verdict, **inert until armed**. The remaining work toward the #6015 Done-when
   is the `ConductorReadiness` gate list: a real GRPO run over a 7B policy
   (compute) and the paid crossy-road composition demo (arm + M4 paid lane + M6
   shadow-win + M2 rubric).

### M7 Done-when next slice (after this change)

1. **Policy backend** — wire a 7B base policy + the FP32 LM head to a real
   autograd/serving split behind `ConductorPolicy` and the trainer's logprob
   path. Compute-gated; the scaffold ships only the typed loop, no weights.
2. **First GRPO run** — drive `ConductorTrainer::update_step` over real rollouts
   (format + correctness reward via `EvalVerdictSource`) until the loop converges
   on the crossy-road task set. H100-hours; owner/compute decision.
3. **Paid composition demo** — arm the verdict source over the live M4 Pylon
   pool, collect a learned-vs-single-model `ShadowComparison` on
   verified-work-per-sat, and prove the crossy-road composition beats
   single-model cost at comparable quality under the M2 rubric. Owner-gated
   (arming + spend-enabled campaign + daily cap).

## Build / run anchors

- Offline smoke: `cargo run -q -p psionic-train --bin coordinator_evolution_smoke`
- Live validation (no spend): `cargo run -q -p psionic-train --bin coordinator_live_train`
- Tests: `cargo test -p psionic-train --lib coordinator`,
  `cargo test -p psionic-train --lib coordinator_eval_verdict`,
  `cargo test -p psionic-train --lib coordinator_conductor`,
  `cargo test -p psionic-train --lib coordinator_m8_head_to_head`,
  `cargo test -p psionic-models --lib coordinator_head`
- Companion runbook: `docs/COORDINATOR_EVOLUTION_TRAINING.md`
- Spec (openagents): `docs/sakana/psionic-coordinator-roadmap.md`,
  `docs/sakana/coordinator-as-verified-work.md`,
  `docs/sakana/tassadar-run-integration.md`, `docs/sakana/trinity.md`,
  `docs/sakana/conductor.md`, `docs/research/tmax/synthesis.md` §5.
