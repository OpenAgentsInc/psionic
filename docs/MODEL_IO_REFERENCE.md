# Model IO Reference

> Status: canonical `PSI-282` / `#3587` plus `PLIB-204` / `#3719` reference
> record, updated 2026-03-25 after landing the typed model-IO portability,
> compatibility-boundary, and bounded selective-import layer in
> `crates/psionic-train/src/model_io.rs`.

This document records the first explicit Rust-native model-IO contract for the
Psionic train stack.

## Canonical Runner

Run the contract harness from the repo root:

```bash
scripts/release/check-psionic-model-io-contracts.sh
```

## What Landed

`psionic-train` now owns a typed model-IO portability layer that sits between
training state and serving-compatible artifact surfaces.

The new typed surfaces include:

- `PortableModelStateDict`
- `ModelStateGroupAssignment`
- `ModelStateTensorManifest`
- `PortableTokenizerBinding`
- `PortableModelBundle`
- `ModelIoCompatibilityContract`
- `ModelIoArtifactReceipt`
- `ModelAdapterDelta`
- `PortableModelImportRequest`
- `PortableModelBundleImportPlan`

## Portability Surfaces

The model-IO layer now supports these explicit portability surfaces:

- Psionic-native in-memory state dict ownership
- dense safetensors export and import with embedded Psionic manifest metadata
- JSON torch-style state-dict compatibility artifacts
- GGUF import with tensor inventory, tokenizer binding, and chat-template digest
- additive adapter merge and unmerge on parameter tensors
- selective state-dict import with include and exclude filters
- explicit source-to-target state-key remap during import
- deferred safetensors tensor materialization through an import plan

## What The Contract Makes Explicit

The model-IO issue was not just "save some tensors to disk." The new contract
makes all of the following typed and inspectable:

- named state-dict traversal records
- Rust model-tree assignment paths for each tensor
- training-group-to-state-dict assignment contracts
- optimizer-state portability alongside train-visible parameters
- checkpoint family and checkpoint reference binding
- tokenizer family, digest, asset version, and special-token posture
- chat-template digest binding
- portability receipts for artifact import and export
- typed adapter delta derivation and reversal

## Current Interop Boundaries

The current portable layer is intentionally specific about what it supports,
and it now exposes that scope as a machine-readable compatibility contract
instead of only prose:

- Psionic-native typed state dict ownership is the canonical import/export path
- safetensors export and import are for dense `f32` training-state artifacts
  that carry the embedded Psionic manifest metadata
- the torch-compatible surface is a typed JSON artifact, not a Python pickle
  or opaque `.pt` loader
- GGUF support is currently import-focused, because that is the relevant
  train-to-serve portability seam in the retained stack
- opaque Python pickle or legacy opaque checkpoint archives are intentionally
  unsupported and called out as such by the typed compatibility contract
- GGUF-imported quantized tensors are preserved in portable state, but they are
  not re-emitted as safetensors without an explicit dequantization or conversion
  step
- selective and remapped import is bounded by training-group structural
  integrity; partial group selection refuses instead of silently producing
  invalid optimizer assignment state
- deferred materialization is currently a safetensors import-plan feature for
  dense `f32` manifest-carrying artifacts, not a blanket lazy loader for every
  interop surface

Those limits are deliberate. The goal of this issue was to stop trained or
served artifacts from being stranded behind bespoke conversion scripts, not to
pretend every foreign binary format is already supported.

## Pass Criteria

The contract is green only if all of the following remain true:

- portable state dicts can roundtrip training groups without losing optimizer
  state, residency posture, or applied-step truth
- safetensors artifacts carry enough embedded metadata to recover Psionic
  assignment contracts
- JSON state-dict artifacts remain machine-legible and replay-safe
- GGUF import binds tokenizer and chat-template identity instead of treating
  them as detached side files
- adapter deltas can be derived, merged, and unmerged against the same typed
  state-dict surface
- selective import can admit a complete tensor subset without silently dropping
  part of a training-group assignment
- remap collisions and missing include keys refuse deterministically
- safetensors import plans can keep admitted tensor payloads deferred until the
  caller materializes the bundle or one named tensor

## Selective Import And Deferred Materialization

The bounded import layer now owns three explicit controls:

- `PortableTensorImportSelection`
- `PortableTensorKeyRemap`
- `TensorMaterializationPolicy`

Callers can use those through:

- `PortableModelBundle::import_torch_state_dict_json_with_request(...)`
- `PortableModelBundle::import_safetensors_with_request(...)`
- `PortableModelBundle::plan_safetensors_import(...)`
- `PortableModelStateDict::select_and_remap(...)`

The intended use is direct:

- select only the tensor subset you need
- remap source keys into the target portable state surface
- refuse if the selection leaves a training group structurally incomplete
- defer safetensors tensor decoding when the consumer only needs an admitted
  import plan first

The import plan is not a second checkpoint truth model. It is a bounded staging
surface that lets Psionic inspect, filter, remap, and later materialize a
portable bundle without eagerly decoding every admitted tensor.

## Current Limits

This issue does not claim that the whole train-to-serve conversion story is
finished. It does not yet implement:

- direct Python pickle or opaque `.pt` checkpoint decoding
- full GGUF export
- tokenizer asset re-emission beyond the typed binding contract
- model-family-specific structural assignment for full decoder trees

What it does do is give Psionic one canonical, Rust-owned portability contract
that training and serving work can both target.
