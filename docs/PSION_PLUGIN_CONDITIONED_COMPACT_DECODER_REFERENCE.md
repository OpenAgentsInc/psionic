# PSION Plugin-Conditioned Compact Decoder Reference

> Status: canonical `PSION_PLUGIN-14` reference model-config contract for the
> first plugin-conditioned learned lane, written 2026-03-22 after landing the
> lane-specific compact-decoder wrapper in `psionic-train`.

This document freezes the first lane-bound model-config contract for the
plugin-conditioned Psion training tranche.

It reuses the shared Psion compact-decoder family instead of inventing a new
plugin-specific model family.

The same contract now has two committed bounded lane instances:

- host-native reference
- mixed host-native plus guest-artifact reference

## Canonical Artifacts

- `crates/psionic-train/src/psion_plugin_conditioned_compact_decoder.rs` owns
  the lane-specific reference config around the shared compact-decoder
  descriptor.
- `crates/psionic-train/examples/psion_plugin_conditioned_compact_decoder_fixtures.rs`
  writes the committed host-native reference config.
- `crates/psionic-train/examples/psion_plugin_conditioned_mixed_compact_decoder_fixtures.rs`
  writes the committed mixed reference config.
- `fixtures/psion/plugins/models/psion_plugin_conditioned_compact_decoder_reference_config_v1.json`
  is the committed host-native reference output.
- `fixtures/psion/plugins/models/psion_plugin_conditioned_mixed_compact_decoder_reference_config_v1.json`
  is the committed mixed reference output.

Stable schema version:

- `psionic.psion.plugin_conditioned_compact_decoder_reference.v1`

## What This Config Freezes

The first reference config now freezes:

- one pilot-sized compact-decoder descriptor for the plugin-conditioned lane
- one explicit `8192` token context window
- one unchanged `32768` vocabulary contract on top of the existing tokenizer
- one no-custom-plugin-token serialization posture
- one structured JSON serialization strategy carrying schema ids, tool names,
  and receipt refs in ordinary token space
- one lane-bound checkpoint family, stage-run-bundle ref, and export-directory
  naming posture
- one direct binding to the committed stage dataset identity instead of only an
  unversioned dataset ref

## Context And Vocabulary Assumptions

The first plugin-conditioned lane keeps the model-family choice narrow:

- size anchor: `pilot_32m`
- tokenizer family: `sentence_piece`
- tokenizer id: `psion_sentencepiece_seed`
- vocab size: `32768`
- context window: `8192`

The context budget is explicitly reserved for:

- directive text
- serialized schema and packet structure
- receipt anchors and refusal evidence
- assistant completion

That keeps later plugin-conditioned audits from pretending the lane had
unbounded prompt room.

## Serialization Strategy

The first plugin-conditioned lane does **not** add bespoke plugin-only tokens.

Instead it freezes:

- structured JSON supervision
- schema ids serialized verbatim
- tool names serialized verbatim
- receipt refs serialized verbatim

That keeps the learned lane aligned with the benchmark packages, training
records, and runtime receipts that already exist in the repo.

## Checkpoint And Export Naming

The lane-specific config keeps the shared compact-decoder file contract:

- `descriptor.json`
- `model.safetensors`
- the shared decoder export-format id

But it binds those files to:

- the plugin-conditioned checkpoint family from the stage manifest
- the committed plugin-conditioned stage-bundle ref
- the committed stage dataset identity
- a lane-specific checkpoint-ref prefix
- a lane-specific export directory name

That keeps the model config tied to the plugin-conditioned stage instead of
floating as a generic decoder preset.

## Boundary

These configs do not yet claim:

- a trained plugin-conditioned checkpoint
- broader model-family proliferation
- served capability closure
- a Google training audit

Guest-artifact training is now exercised by the mixed reference lane, but that
does not widen the served claim boundary. These are only the first truthful
reference configs for the host-native and mixed bounded lanes.
