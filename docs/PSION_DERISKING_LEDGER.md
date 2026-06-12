# Psion Derisking Ledger

> Status: canonical `#1124` record, created 2026-06-12 for the
> Pluralis-derived mechanism entries required by the adaptation roadmap
> item P0.3 (openagents
> `docs/training/2026-06-12-pluralis-to-pylon-adaptation-roadmap.md`,
> commit `463b0d76c`).

## Purpose

The buildout plan's rule (openagents
`docs/training/2026-06-10-psion-full-pipeline-buildout-plan.md`, §6):
every considered-and-deferred training mechanism gets a one-line ledger
entry with the reason, so "we considered it" is on the record and the
ledger — current baseline plus every tested delta plus verdict — is a
committed doc, not tribal memory. This file is that ledger for psionic.

Nothing in this ledger is a capability claim. An entry records a
decision about whether and how a mechanism may enter the training
substrate, not that the mechanism works, helps, or exists in our code.
Entry statuses are decision states (`deferred`, `blocked`,
`answered`), not the repo capability vocabulary.

The existing per-decision ablation records
(`docs/PSION_EXECUTOR_*_ABLATION.md` and their fixtures under
`fixtures/psion/executor/`) remain the fixture pattern for decisions
that were actually run. This ledger is the index of mechanisms that
were considered and deferred, blocked, or gated before any run.

## Pluralis-Derived Entries (roadmap P0.3)

Source material is the read-only workspace reference lane
`projects/pluralis/repos/` (agora, node0, AsyncPP, AsyncMesh). We port
ideas, not code.

| Entry | Status | Reason | Cross-reference |
| --- | --- | --- | --- |
| SPARTA-class sparse parameter averaging | deferred (side experiment only) | W3 standing order: no public gradients into the main optimizer, ever; sparse averaging runs only as a canary side experiment with its own evals and pre-written kill criteria | psionic#1127 (canary; harness_ready 2026-06-12: pre-registered harness landed in `docs/PSION_SPARTA_CANARY.md` + `crates/psionic-train/src/sparta_canary.rs`; harness only, no run, status stays deferred/side-experiment); Pluralis AsyncMesh `sparta/sparta.py` (read-only lane `projects/pluralis/repos/AsyncMesh/`) |
| PowerSGD rank-compressed gradient averaging (Pluralis node0 ships `averager_rank: 64`) | answered (2026-06-12) | Compression composes with the algebra but not the provenance: low-rank factors admit cheap Freivalds-style consistency probes, but lossy compression severs the algebraic identity to the true gradient, so compressed contributions ride seeded_replication or stay inside the trust boundary | psionic#1128; `docs/2026-06-12-powersgd-freivalds-compatibility.md` |
| Subspace-compressed pipeline-stage boundaries (node0 `compression_rate: 100`) | deferred (R3+/R4-conditional) | Tied to pipeline-stage-sharded windows, which are deliberately unfiled until the R2 economics gate clears twice against the rented-cluster comparator | openagents roadmap P3.1 |
| AsyncPP delay-corrected optimizers (weight stashing, Nesterov correction; ICML 2025) | deferred | Enters only via the ablation manifest if a side experiment earns it; staleness is currently a dispatch-layer concern (openagents#4849 / openagents#4853), not an optimizer concern | Pluralis AsyncPP reference (read-only lane `projects/pluralis/repos/AsyncPP/`) |

No implementation work attaches to this file. The entries exist to gate
and record. A mechanism leaves this ledger only by earning an ablation
fixture through the per-decision pattern above, or by having its
blocking question answered in a committed doc, as entry 2 now has.

Future considered-and-deferred mechanisms from other lanes append here
under their own issue references.

Authored by Fable (claude-fable-5) for psionic#1124 / #1128.
