# Psion Compact Decoder

> Status: canonical `PSION-11` / `#367` compact decoder-family contract,
> written 2026-03-22 after landing the first Psion sampling policy.

This document freezes the first general-purpose compact causal decoder family
for the `Psion` lane.

It keeps the learned lane on top of Psionic's existing generic decoder
substrate instead of introducing a separate trainer-owned model stack.

## Canonical Artifacts

- `crates/psionic-models/src/psion_compact_decoder.rs` owns the typed Psion
  compact-decoder descriptor contract.
- `fixtures/psion/models/psion_compact_decoder_pilot_descriptor_v1.json` is
  the canonical pilot-anchor descriptor fixture.
- `fixtures/psion/models/psion_compact_decoder_internal_descriptor_v1.json`
  is the canonical first internal-anchor descriptor fixture.
- `docs/PSION_PLUGIN_CONDITIONED_COMPACT_DECODER_REFERENCE.md` is the first
  lane-specific child config built on top of this shared compact-decoder
  family for the host-native plugin-conditioned reference lane.
- `docs/PSION_GENERIC_LOAD_AND_GENERATE.md` is the first retained generic
  learned-lane serve packet built on top of the current artifact-backed decoder
  runtime.

The stable schema version is `psion.compact_decoder_descriptor.v1`.

## What The Descriptor Freezes

The descriptor now carries:

- a stable Psion decoder family label
- an admitted size anchor
- explicit decoder config, including context length
- explicit tokenizer binding, including tokenizer id, version, family, digest,
  and vocab size
- stable checkpoint tensor naming and tensor shapes
- stable export file naming
- a bridge into the generic `DecoderModelDescriptor` once real weight metadata
  exists

That keeps pilot and first internal model descriptors challengeable before the
pretrain stage starts emitting checkpoints.

The new generic load-and-generate packet matters here because it proves the
current compact-decoder substrate can close one honest artifact-backed
load-to-serve loop without pretending that descriptor closure already implies
full pilot-anchor serving parity.

## Size Anchors

The first admitted anchors are intentionally narrow:

- `pilot_32m`
- `internal_128m`

They are descriptor anchors, not a promise that larger closure or broader model
families are already done.

## Checkpoint And Export Naming

The family now freezes:

- `model.safetensors` as the checkpoint file name
- `descriptor.json` as the descriptor file name
- `decoder` as the checkpoint tensor namespace prefix
- one ordered tensor layout derived mechanically from the descriptor config

That keeps training, evaluation, and serving on one stable naming surface.

## Mechanical Enforcement

`psionic-models` now validates that:

- tokenizer vocab size matches the model config vocab size
- context length is explicit and non-zero
- each size anchor maps to one exact decoder config
- checkpoint tensor names and shapes rebuild deterministically from the config
- export naming stays aligned with the checkpoint contract
- later generic decoder descriptors can only be produced from matching tensor
  layouts and checkpoint formats
