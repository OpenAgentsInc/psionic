# Psion Family Serve Vocabulary

> Status: canonical `PSION-0902` / `#786` family serve-and-claim vocabulary
> packet, written 2026-03-30 after landing the first generic
> `Psion` load-and-generate packet.

This document freezes one family-level serve vocabulary for `Psion` without
flattening the lane-specific evidence classes that already exist in the repo.

The problem it solves is simple:

- the umbrella family name is `Psion`
- the current codebase already has multiple real served lanes
- those lanes do not all mean the same thing

The family needed one explicit vocabulary packet so readers, operators, and
future issues can say "Psion" without silently laundering generic learned
truth into plugin-conditioned or executor-capable claims.

## Canonical Artifacts

- `docs/PSION_FAMILY_SERVE_VOCABULARY.md` is the canonical human-readable
  contract.
- `crates/psionic-serve/src/psion_family_serve_vocabulary.rs` owns the
  retained machine-readable packet.
- `crates/psionic-serve/examples/psion_family_serve_vocabulary_fixtures.rs`
  writes the canonical fixture.
- `fixtures/psion/serve/psion_family_serve_vocabulary_v1.json` is the
  canonical retained packet.

Stable schema version:

- `psion.family_serve_vocabulary.v1`

## Retained Packet Truth

The retained family vocabulary packet digest is:

- `ee70a0b1544c629d3266bdbaea8c5a9c02b939fa186dadd023f6ad804f2b9185`

The packet freezes three lane rows:

1. `generic_compact_decoder`
2. `plugin_conditioned`
3. `executor_capable_tassadar_profile`

And three distinct evidence classes:

1. `learned_judgment_only`
2. `learned_plugin_reasoning_with_runtime_receipt_gate`
3. `bounded_executor_replacement_and_admitted_fast_route_truth`

Those class labels are the heart of this issue. The family now has one shared
vocabulary, but the evidence classes stay distinct.

## Lane Meanings

### Generic compact-decoder `Psion`

This is the generic learned lane.

Its serve vocabulary is:

- direct artifact-backed generation
- shared served-evidence contract
- shared served-output-claim posture contract
- no hidden execution
- learned judgment only by default

Canonical authorities:

- `docs/PSION_PROGRAM_MAP.md`
- `docs/PSION_GENERIC_LOAD_AND_GENERATE.md`
- `docs/PSION_SERVED_EVIDENCE.md`
- `docs/PSION_SERVED_OUTPUT_CLAIMS.md`

### Plugin-conditioned `Psion`

This is still a learned lane, but not the same learned lane as the generic
compact-decoder path.

Its serve vocabulary is:

- lane-specific served posture over the shared claim contracts
- learned plugin reasoning
- runtime-owned plugin execution
- explicit receipt gate for actual execution claims
- no flattening into generic plugin-platform or arbitrary software claims

Canonical authorities:

- `docs/PSION_PLUGIN_CLAIM_BOUNDARY_AND_CAPABILITY_POSTURE.md`
- `docs/PSION_PLUGIN_HOST_NATIVE_CAPABILITY_MATRIX_V1.md`
- `docs/PSION_PLUGIN_MIXED_CAPABILITY_MATRIX_V2.md`
- `docs/PSION_SERVED_EVIDENCE.md`
- `docs/PSION_SERVED_OUTPUT_CLAIMS.md`

### Executor-capable `Psion` / `Tassadar`

This is the bounded executor-capable profile.

Its serve vocabulary is:

- bounded executor profile
- explicit promotion and replacement packets
- admitted fast-route posture
- explicit executor-capable claim boundary
- consumer-seam validation before promotion

Canonical authorities:

- `docs/PSION_EXECUTOR_PROGRAM.md`
- `docs/PSION_EXECUTOR_TRAINED_V1_PROMOTION.md`
- `docs/PSION_EXECUTOR_TRAINED_V1_REPLACEMENT_REPORT.md`
- `docs/ROADMAP_TASSADAR.md`

## Shared Rules

The retained packet freezes four family rules:

1. family name does not flatten lane truth
2. shared claim contract reuse
3. no hidden execution
4. executor lane is not the generic default

That means:

- saying "`Psion`" is allowed
- using "`Psion`" to erase lane-specific evidence classes is not
- shared served-evidence and served-output-claim contracts should be reused
  where applicable
- lane-specific posture is still allowed above those shared lower contracts

## Why This Matters

The generic family now has:

- one real operator-facing generic serve packet
- existing plugin-conditioned served posture
- existing executor-capable promotion and replacement truth

Without this vocabulary packet, the umbrella family can become misleading just
through shorthand.

This document prevents that. It makes the umbrella usable while keeping the
served claim boundary honest for each lane.
