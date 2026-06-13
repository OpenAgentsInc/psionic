# Psion Derisking Ledger

> Status: canonical `#1124` and `#1118` record. Created 2026-06-12 for
> the Pluralis-derived mechanism entries required by the adaptation
> roadmap item P0.3 (openagents
> `docs/training/2026-06-12-pluralis-to-pylon-adaptation-roadmap.md`,
> commit `463b0d76c`); extended 2026-06-12 with the QVAC-derived
> deferred-technique entries promised by the buildout plan (openagents
> `docs/training/2026-06-10-qvac-edge-stack-analysis.md`, absorbed via
> psionic #1115-#1118).

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

## QVAC-Derived Entries (psionic#1118)

Source material is the QVAC edge-stack analysis (openagents
`docs/training/2026-06-10-qvac-edge-stack-analysis.md`, Tier 2/4) and
the read-only `tetherto` reference repos it covers. We port ideas, not
code.

| Entry | Status | Reason | Cross-reference |
| --- | --- | --- | --- |
| Vulkan backend (cross-vendor edge GPU) | deferred | QVAC's production Android evidence makes Vulkan the stated direction for psionic's fourth backend, but no ladder rung needs contributor Android/cross-vendor devices yet, and psionic's posture stays standard OS driver stacks (Metal/CUDA/ROCm) with experimental sovereign-driver paths explicitly gated | `docs/deep-research-tinygrad.md` (driver-posture note); openagents QVAC analysis Tier 2 |
| Dynamic tiling under mobile memory limits | deferred | Exists to work around mobile-GPU buffer ceilings (Adreno 128 MiB SSBO constraint); psionic has no mobile-GPU training target, so there is nothing for the technique to bound yet | openagents QVAC analysis Tier 2; revisit at the first mobile-GPU training target |
| KV-cache quantization (TurboQuant/PolarQuant-class) | deferred | Served decode still stores dense host `f32` KV with an `f16` device mirror; device-resident decode KV ownership and explicit KV-cache encoding contracts must exist before a quantized cache has anywhere honest to live | `docs/audits/2026-03-24-psionic-turboquant-integration-audit.md` |
| BitNet-b1.58 QAT pretraining | deferred | A published recipe, not a Psion rung: it enters only through the ablation manifest at R2+ when the ablation system can price the architecture delta | psionic#1115 (ternary TQ-class serving formats with cross-backend determinism receipts); openagents QVAC analysis §4 |

No implementation work attaches to this file. The entries exist to gate
and record. A mechanism leaves this ledger only by earning an ablation
fixture through the per-decision pattern above, or by having its
blocking question answered in a committed doc, as entry 2 now has.

Future considered-and-deferred mechanisms from other lanes append here
under their own issue references.

Authored by Fable (claude-fable-5) for psionic#1124 / #1128 / #1118.
