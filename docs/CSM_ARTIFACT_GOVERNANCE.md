# CSM Artifact Governance

Status: canonical Psionic governance contract for CSM speech artifacts

Related specs:

- [CSM Audio Runtime](CSM_AUDIO_RUNTIME.md)
- [CSM Rust Runtime Decision Record](CSM_RUST_RUNTIME_DECISION_RECORD.md)

## Purpose

This document defines the governance metadata that Psionic must publish before
Autopilot can promote CSM speech generation beyond shadow or limited canary.

CSM provider output is evidence, not instruction. CSM has no CRM, CEO, Legal,
Memory, Blueprint, HUD, approval, action, or receipt authority.

## Stable Artifact Identity

Every CSM worker publishes:

```text
artifact_id
artifact_hash
model_id
model_version
source_repositories
runtime_image_ref
quantization
tokenizer_dependency
audio_codec_dependency
```

The current artifact id is derived from:

```text
{model_id}@{csm_model_digest}
```

The current artifact hash is the CSM model digest from the committed fixture
descriptor. Tokenizer and audio-codec dependencies remain separate digests
because the worker needs all three classes of artifact to produce governed
audio.

Quantized or accelerated variants must publish a new artifact id. They must not
silently replace the current full-precision artifact.

## License Posture

The current license posture is:

```text
license_review_required_before_public_or_customer_use_operator_dogfood_only
```

That posture means:

- CSM may be used only for OpenAgents-operated dogfood and Psionic/Autopilot
  shadow or limited canary tests.
- CSM must not be promoted to public, customer, or broad user-facing use until
  license review is complete and recorded in governance metadata.
- License review status is a promotion gate, not a note in a launch checklist.

## Voice Profile Policy

Served requests must use governed assistant voice profile ids. Raw prompt
fixture ids are not served voice ids.

The current profile id is:

```text
openagents/default_female_v1
```

The id is an OpenAgents-owned CSM voice profile id. It is not a raw Sesame
prompt id and it is not a public voice-cloning permission.

The current policy disallows:

```text
public_user_voice_clone
customer_voice_clone
contact_voice_clone
arbitrary_reference_audio_upload
```

Arbitrary user, customer, investor, lawyer, witness, contact, or employee voice
cloning remains blocked until Psionic has a governed profile creation process,
consent evidence, retention policy, watermark or equivalent safety control, and
Autopilot/Blueprint receipt linkage.

## Watermark Posture

The current watermark posture is:

```text
unsupported_operator_accepted_limited_dogfood
```

The refusal code is:

```text
csm_watermarking_unavailable
```

Public demo watermark keys are not production safety controls. Missing private
watermarking or an equivalent voice-safety control blocks public use,
customer-facing use, broad canary, and primary provider promotion.

## Worker Metadata

The worker metadata endpoint is:

```text
GET /psionic/csm/worker/metadata
```

The metadata includes:

```text
governance.schema_version
governance.voice_profile_governance_schema_version
governance.artifact_id
governance.artifact_hash
governance.license_posture
governance.runtime_image_ref
governance.quantization
governance.allowed_voice_profile_ids
governance.disallowed_voice_use_cases
governance.watermark_status
governance.watermark_refusal_code
governance.canary_promotion
governance.primary_promotion
governance.rollback_target
governance.missing_governance_blocks
```

Autopilot can consume this surface without depending on Psionic internals.

## Response Headers

Every request that reaches validation publishes governance headers on success
and fail-closed runtime responses:

```text
x-psionic-csm-artifact-id
x-psionic-csm-governance-schema
x-psionic-csm-license-posture
x-psionic-csm-runtime-image-ref
x-psionic-csm-voice-profile-id
x-psionic-csm-voice-approval-status
x-psionic-csm-runtime-admission
x-psionic-csm-consent-posture
x-psionic-csm-watermarking
x-psionic-csm-watermark-refusal-code
x-psionic-csm-promotion-gate
x-psionic-csm-rollback-target
```

Stream terminal metadata repeats the artifact id, governance schema, license
posture, runtime admission, watermark status, and watermark refusal code.

## Promotion Gate

The current canary and primary promotion gate is:

```text
blocked_without_artifact_license_voice_profile_watermark_and_runtime_image_governance
```

Promotion remains blocked when any of these are missing:

- license review;
- private watermark or equivalent voice-safety control;
- runtime image ref from the deploy receipt;
- Autopilot shadow business-outcome evidence;
- governed voice profile id and consent posture;
- rollback target;
- artifact hash and dependency digests.

## Rollback Target

The current rollback target is:

```text
fallback_to_current_autopilot_tts_provider
```

Autopilot must keep a non-CSM speech fallback until CSM meets the provider
promotion bar in the Autopilot voice roadmap.

## Release Requirements

Any CSM release that changes artifact, runtime, voice profile, watermark,
license, or response metadata must update this document and run the focused CSM
tests.

Minimum verification:

```bash
git diff --check
rustfmt --edition 2024 --check crates/psionic-serve/src/csm_speech.rs
cargo test -p psionic-serve csm_
```

Workspace-wide formatting is intentionally not the gate for targeted CSM
changes because this repository currently has unrelated Rust formatting drift.
