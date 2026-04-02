# Train Sampler Service Reference

> Status: canonical bounded sampler-service record, updated 2026-04-01 after
> landing the first trainer-integrated sampler service in `psionic-train`.

This document records the first training-aware sampler service in Psionic.

## What Landed

`psionic-train` already owned:

- policy revision identity
- checkpoint lineage
- rollout artifacts
- fixed-budget trainer steps
- repo-owned open-adapter training and export

This issue adds the missing live inference bridge above that substrate:

- a reusable `TrainingSamplerService`
- `OpenAdapterTrainingSamplerConfig` for the first bounded backend
- health and readiness inspection through `status`
- completions-style inference
- chat-completions-style inference
- per-token logprob queries
- explicit active-revision refresh without process restart
- fail-closed refusal when the active revision is stale or the requested
  revision is unavailable
- revision identity, adapter identity, and optional weight-broadcast identity in
  status and response surfaces

## Canonical Shape

The first service is intentionally bounded and training-owned.

It currently serves one backend:

- the repo-owned open-adapter reference lane for
  `gpt_oss.decoder_lm_head_lora`

It owns:

1. active revision loading from `PolicyRevision` plus one adapter artifact
2. optional checkpoint lineage carried through the policy revision
3. optional datastream weight-broadcast identity carried through refresh input
4. synchronous bounded generation over a deterministic prompt encoder
5. per-token logprob materialization suitable for RL diagnostics and
   distillation-compatible consumers
6. explicit refresh to a newly promoted revision
7. explicit freshness gating on the request path

The current service does not hide which revision served a response. Every
generation and logprob response carries the active revision snapshot used for
the request.

## Canonical Runner

Run the focused harness from the repo root:

```bash
scripts/release/check-psionic-train-sampler-service.sh
```

## Pass Criteria

The bounded sampler service is green only if all of the following remain true:

- active revision status is inspectable before and after refresh
- generation responses include policy revision identity
- chat and completions requests both execute against the same active revision
- per-token logprob queries return inspectable token-level records
- refresh can adopt a newer promoted revision without a process restart
- stale or unavailable revisions refuse explicitly instead of silently falling
  back
- the integration harness proves one training-backed revision swap in one run

## Current Boundary

This first service is still bounded.

It does not yet claim:

- a network server or OpenAI-compatible HTTP surface
- dense checkpoint loading for general full-model revisions
- continuous batching or multi-tenant admission
- distributed sampler replicas or live weight-stream application

The first version closes the training-to-sampler bridge for the repo-owned
open-adapter reference lane. Dense checkpoint serving remains later work.
