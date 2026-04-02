# Psion Actual Pretraining Baseline Tools Bundle

> Status: canonical actual-lane baseline-tools bundle, written 2026-04-02
> after binding selective CS336 A1 work into `psion_actual_pretraining_v1`
> without creating a second pedagogical trainer stack.

This document freezes the machine-readable baseline-tools surface for the
canonical actual `Psion` pretraining lane.

It does not create a separate curriculum trainer or a detached tokenizer lab.
It binds one honest bring-up trainer, one tokenizer reproducibility binding,
one operator-readable resource-accounting table, and one bounded ablation
family directly into the actual lane.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_baseline_tools_bundle.rs`
  owns the typed baseline-tools contract.
- `crates/psionic-train/examples/psion_actual_pretraining_baseline_tools_bundle_fixtures.rs`
  regenerates the committed baseline-tools fixture plus the committed bring-up
  and ablation stage configs.
- `fixtures/psion/pretrain/psion_actual_pretraining_baseline_tools_bundle_v1.json`
  carries the canonical actual-lane baseline-tools bundle.
- `fixtures/psion/pretrain/psion_actual_pretraining_bringup_stage_config_v1.json`
  is the honest internal-128M bring-up stage config consumed by the actual
  lane.
- `fixtures/psion/pretrain/psion_actual_pretraining_pilot32m_ablation_stage_config_v1.json`
  is the bounded cheaper pilot32m ablation config kept in the same family.

Stable schema version:

- `psion.actual_pretraining_baseline_tools_bundle.v1`

## What The Bundle Freezes

The baseline-tools bundle binds:

- one `psion_pretrain_stage` bring-up surface pointed at the frozen internal
  128M recipe shape
- one tokenizer reproducibility binding across the tokenizer training manifest,
  tokenizer artifact bundle, tokenized corpus manifest, and held-out exposure
  partitions
- one smaller resource-accounting table for the actual 128M lane plus one
  bounded pilot32m ablation row
- one short actual-lane smoke ablation and one bounded cheaper pilot32m replay
  ablation

That is the selective A1 port. It is useful because the actual lane still
needs smaller truthful surfaces for correctness bring-up, tokenizer drift
checks, and operator-readable resource accounting. It would be wasteful to
build a second full trainer stack for that.

## Current Claim Boundary

This bundle honestly claims:

- the actual lane has one direct bring-up config above the frozen recipe
- tokenizer reproducibility is now machine-legible inside the actual-lane
  contract set
- bounded ablations now reuse the same tokenizer, dataset identity, and stage
  program surface as the actual lane

It does not yet claim:

- a second independently supported pedagogical LM implementation
- detached curriculum parity with every CS336 starter exercise
- that the bounded ablations themselves prove the admitted four-node H100 lane

## Why This Exists

Before this bundle landed, the actual lane already had recipe, data, scaling,
systems, evidence, launcher, and continuation surfaces. What it did not have
was one smaller operator-owned place to put selective A1 work without letting
that work turn into a parallel teaching stack.

This bundle closes that gap. The actual launcher now consumes it directly as
one frozen contract ref, so the retained actual-lane manifests show that the
bring-up trainer, tokenizer reproducibility binding, and bounded ablation
family belong to the real lane instead of floating nearby as optional notes.
