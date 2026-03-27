# Parameter Golf Promoted Family Contract

## Purpose

This document freezes the family contract for the first serious Psion-owned
small decoder that should be:

- trainable inside `psionic-train`
- exportable as one honest portable bundle
- loadable through a public runtime path
- inferable locally
- serveable through `psionic-serve`

The family is the existing `parameter_golf_decoder` core, not the older toy
`PsionCompactDecoderReferencePilotModel`.

## Frozen Shared Family

Stable family id:

- `parameter_golf_decoder`

Stable first-shipping baseline:

- model id: `parameter-golf-sp1024-9x512`
- revision: `public-2026-03-18`
- config: `9` layers, `512` hidden size, `8` query heads, `4` KV heads,
  `1024` context, tied embeddings, tanh logit softcap, RoPE

Shared capabilities admitted across both promoted profiles:

- tied embeddings
- grouped-query attention with explicit KV-head count
- RoPE
- optional partial RoPE
- learned skip weights / U-Net-like skip structure
- optional `BigramHash`
- optional late-layer value embeddings
- optional XSA on deep layers
- `relu^2`
- `LeakyReLU(0.5)^2`
- parameter-banked weight surfaces
- quantized small-model export

These capabilities are frozen in code at:

- `crates/psionic-models/src/parameter_golf.rs`

## Promoted Profiles

### 1. General Psion Small Decoder

Stable profile id:

- `psion_small_decoder_pgolf_core_v0`

Intent:

- use the PGOLF core architecture as Psionic's first real small decoder
- avoid hard-binding every run to challenge-only contest rules

Challenge-only requirements intentionally disabled:

- exact `SP1024` tokenizer
- exact FineWeb challenge dataset lane
- exact `16 MiB` compressed artifact cap
- score-first TTT
- contest BPB accounting

This is the default profile the train-to-infer PINF stack should target first.

### 2. Strict PGOLF Challenge Overlay

Stable profile id:

- `parameter_golf_challenge_sp1024_v0`

Intent:

- preserve the public Parameter Golf contest posture when Psionic wants to run
  the same core family under the strict challenge rules

Challenge-only requirements enabled:

- exact `SP1024` tokenizer
- exact FineWeb challenge dataset lane
- exact `16 MiB` compressed artifact cap
- legal score-first TTT
- tokenizer-agnostic BPB accounting

This is an overlay profile on the same core family, not a separate core model
family.

## Machine-Readable Surfaces

The promoted-family split is now carried in:

- `ParameterGolfPromotedProfileContract` in
  `crates/psionic-models/src/parameter_golf.rs`
- `PortableModelProfileContract` in
  `crates/psionic-train/src/model_io.rs`
- `parameter_golf_general_psion_small_decoder_profile_contract()` in
  `crates/psionic-train/src/parameter_golf_reference.rs`
- `parameter_golf_strict_challenge_profile_contract()` in
  `crates/psionic-train/src/parameter_golf_reference.rs`

The first canonical promoted training proof path now also exists at:

- `run_parameter_golf_promoted_reference_run(...)` in
  `crates/psionic-train/src/parameter_golf_reference.rs`
- `crates/psionic-train/examples/parameter_golf_promoted_reference_run.rs`

That proof path binds the general promoted profile to the existing full
`parameter_golf_decoder` local-reference trainer, verifies that the emitted
checkpoint tensor surface exactly matches the promoted descriptor, and proves
restore-plus-resume parity from the emitted checkpoint lineage.

The profile split is also carried directly through the training config and
checkpoint manifest:

- `ParameterGolfReferenceTrainingConfig.promoted_profile`
- `ParameterGolfCheckpointManifest.promoted_profile`

The bounded local-reference trainer only admits the general Psion profile. If
the strict challenge overlay is requested before the exact public challenge
tokenizer, FineWeb lane, legal score-first TTT path, contest BPB accounting,
and exported artifact-cap accounting exist, the lane now refuses explicitly
instead of silently pretending those prerequisites are satisfied.

For train-to-infer work, this promoted PGOLF path is now the primary first-model
target. The older `PsionCompactDecoderReferencePilotModel` remains the bounded
smoke-test lane for receipt and pipeline closure only.

## Scope Boundary

This contract freeze does not itself:

- emit the later self-contained promoted bundle
- load the bundle for inference
- run local generation
- wire `psionic-serve`

Those remain the later PINF issues. This document only freezes the family and
profile identity so the rest of the work can converge on one honest target.
