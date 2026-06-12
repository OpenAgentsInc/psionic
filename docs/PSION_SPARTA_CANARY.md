# Psion SPARTA Sparse-Averaging Canary Harness

> Status: canonical `psionic#1127` record, 2026-06-12, after landing the
> SPARTA canary harness with pre-registered bounds (Pluralis roadmap item
> P2.4).

Authored by Fable (claude-fable-5) for psionic#1127.

## W3 Standing Order

Per the Tassadar research plan W3: **no public gradients into the main
optimizer, ever**. Sparse-averaging decentralized training runs only as a
side experiment with its own canary evals and pre-written kill criteria. It
is never the main run, never the main optimizer, and never a default. This
harness exists so that side experiment runs by the book: harness before
claim, pre-registration before run.

The standing order is enforced in code: every
`SpartaCanaryPreRegistration` must carry `SPARTA_CANARY_STANDING_ORDER`
verbatim, and a weakened standing order is a typed validation error.

## What This Closes

Psionic now owns the pre-registered canary harness for SPARTA-class sparse
parameter averaging — the harness only, not a training run:

- `crates/psionic-train/src/sparta_canary.rs`
- `crates/psionic-train/src/bin/sparta_canary_contract.rs`
- `fixtures/training/sparta_canary_v1.json`
- `scripts/check-sparta-canary.sh`

The derisking ledger entry for this mechanism
(`docs/PSION_DERISKING_LEDGER.md`, entry 1) stays
`deferred (side experiment only)`; the cross-reference now records that the
harness exists (`harness_ready`).

## Harness Surfaces

### Pre-registration contract

`SpartaCanaryPreRegistration` is a typed, digest-pinned record written
before any run:

- The grid (`SpartaCanaryGridPoint`): sparsity fraction, averaging cadence,
  partition strategy. The committed grid starts from the Pluralis published
  tuning (sparsity fraction 0.05, average state every 5 steps, random
  rotating partitions). That is their tuning, not ours; the grid is config,
  not constants.
- The baseline arm: synchronous full-parameter averaging
  (`SpartaCanaryBaselineArm::SynchronousAveraging`). The canary is judged
  against it, never against nothing.
- The eval schema (`SpartaCanaryEvalSchema`): a named first-divergence-step
  metric plus held-out eval families. Per the research plan, perplexity
  alone never decides the canary; a schema whose held-out families are all
  perplexity surfaces is a typed validation error.
- The kill bound (`SpartaCanaryKillBound`): a maximum held-out eval
  degradation fraction vs the synchronous baseline, and a communications-
  savings materiality floor. Both clauses are pre-registered.

`preregistration_digest` pins the whole record: changing any grid point,
bound, or schema field changes the digest, and a stale digest over a changed
grid is refused.

### SPARTA partition mechanics

`SpartaPartitionedIndexSelector` is the owned Rust equivalent of the
Pluralis AsyncMesh `PartitionedIndexSelector` (read-only reference lane
`projects/pluralis/repos/AsyncMesh`, `sparta/sparta.py`):

- Each tensor's indices are split into `min(ceil(1/p), n)` random
  partitions (their `rand(n).argsort() % num_partitions` becomes a seeded
  Fisher-Yates shuffle followed by `shuffled position % partition_count`).
- Each averaging round consumes exactly one rotating partition.
- A full rotation covers every index exactly once, then partitions
  re-randomize for the next rotation epoch.

Determinism is counter-based and caller-seeded: `sparta_counter_rng(seed,
stream, counter)` is a stateless splitmix64 mix (the house seeded-
determinism pattern), so the same `(seed, tensor, rotation epoch)` always
yields the same partition schedule regardless of call order. The harness
never reads ambient entropy or wall clocks; timestamps are caller-passed.

### Bounded two-arm comparison runner

`run_sparta_canary_comparison` simulates N in-process replicas with seeded
synthetic parameter drift and runs both arms side by side at toy scale:

- sparse arm: one rotating partition averaged per averaging round;
- synchronous arm: full-parameter averaging on the same cadence, fed the
  identical drift stream.

Per round it records, for each arm, the mean pairwise L2 divergence between
replicas (quantized to nano-units for exact JSON round-trips), whether the
round averaged, and the simulated bytes communicated (elements averaged x 8
bytes x replica count). The artifact also carries averaging-rounds-to-full-
coverage and per-arm byte totals. Artifact validation replays the simulation
from its own config, so recorded values cannot drift from what the runner
deterministically produces.

This demonstrates the harness works end to end at toy scale. It is not the
R1/R2-scale canary; that run is gated and not claimed.

### Outcome recording

`SpartaCanaryOutcomeRecord` is the typed write-back record for the
derisking ledger: status (`held` / `killed` / `pending`), the deciding
pre-registered bound for a kill, the observed values behind a decision, and
the binding `preregistration_digest`. `evaluate_sparta_kill_bound` applies
the pre-registered clauses to an observed outcome:

- killed by `EvalDegradationBound` if held-out eval degradation vs the
  synchronous baseline exceeds the pre-registered maximum;
- killed by `CommsSavingsMaterialityFloor` if communications savings fall
  below the pre-registered floor (immaterial at our fleet scale);
- held only when every bound passes on a real gated run.

A killed canary still produces the outcome record and the ledger
write-back: negative results are recorded, not discarded. The committed
record is `pending` — the contract refuses a held or killed status while the
comparison artifact is synthetic, because a bounded toy simulation never
decides the canary.

## Pluralis Source

Adapted from the Pluralis AsyncMesh SPARTA implementation
(`sparta/sparta.py` in the read-only reference lane
`projects/pluralis/repos/AsyncMesh`): the `PartitionedIndexSelector`
rotation (`ceil(1/p)` partitions, one per round, re-randomized per
rotation), and the published tuning defaults that seed the first grid point.
We port ideas, not code; the mechanics are owned Rust, and the buffering,
EMA, and async-delay variants in the reference are deliberately out of scope
for this harness.

## Pre-Registration-Before-Run Discipline

The order is fixed: the pre-registration record is written and its digest
committed before any canary run; the run binds to that digest; the outcome
record names the digest and the deciding bound. A run whose grid, baseline,
eval schema, or kill bound does not match a committed pre-registration
digest is not the pre-registered canary, and its outcome cannot hold the
mechanism.

## Kill Criteria

From the roadmap, pre-registered and pinned:

- kill if first-divergence/held-out evals degrade beyond the pre-registered
  bound vs the synchronous-averaging baseline;
- kill if communications savings are immaterial at our fleet scale (below
  the pre-registered materiality floor);
- a killed canary still produces the ledger entry and receipts.

## Honest Current Meaning

- This is a harness plus a toy-scale demonstration only. No training run of
  any kind happened, and no claim about real convergence behavior is made.
- Every comparison value in the committed fixture is synthetic output of the
  bounded in-process simulation (`measurement_status: synthetic_bounded`),
  and the contract refuses a synthetic artifact that does not say so.
- The toy simulation's sparsity (0.25) is a walkthrough input chosen so the
  fixture shows a complete rotation; it is not the pre-registered grid and
  decides nothing.
- The R1/R2-scale canary is gated and not claimed. Until a real gated run
  exists, the outcome record stays `pending` by typed rule.
- The W3 standing order stands regardless of any canary outcome: a held
  canary earns a side-experiment lane, never entry into the main optimizer.

## Validation

- `cargo test -p psionic-train --lib sparta_canary::` covers: full-rotation
  coverage (every index exactly once, including after re-randomization),
  partition determinism under the same seed and divergence under different
  seeds, sparse rounds moving only the selected partition, both arms'
  metrics from the comparison runner with sparse bytes strictly below
  synchronous bytes, the pre-registration digest pinning the grid, the
  standing-order and perplexity-alone refusals, kill-bound evaluation on
  both sides of each bound, outcome-record state rules, tamper refusal via
  deterministic replay, and fixture round-trip against the committed
  artifact. (Run with the module filter: `cargo test -p psionic-train
  --lib` carries pre-existing unrelated failures tracked as psionic#1129.)
- `./scripts/check-sparta-canary.sh` regenerates the contract and verifies
  the committed fixture has not drifted, the standing order and Pluralis
  starting grid point are intact, perplexity never decides alone, the
  sparse arm communicates fewer simulated bytes, and the outcome stays
  pending with the synthetic boundary explicit.
