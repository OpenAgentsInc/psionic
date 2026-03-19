# Tassadar TCM.v1 Substrate Audit

Date: 2026-03-19

## Purpose

`TCM.v1` is the declared terminal substrate model for Psionic/Tassadar. This
audit exists so later universality work refers to one explicit machine model
instead of a growing bag of “broad compute” surfaces.

Canonical machine-readable artifacts:

- `fixtures/tassadar/reports/tassadar_tcm_v1_model.json`
- `fixtures/tassadar/reports/tassadar_tcm_v1_runtime_contract_report.json`

## What TCM.v1 means

`TCM.v1` is a bounded small-step resumable machine with:

- explicit conditional control and indirect dispatch
- mutable heap segments with checkpoint-backed extension
- persisted continuation through checkpoints, process objects, and spill-tape
  state
- declared effect profiles only
- explicit refusal on ambient host effects or undeclared publication widening

## What TCM.v1 does not mean

This declaration alone does not prove:

- universal-machine encodings
- a universality witness suite
- a minimal universal-substrate acceptance gate
- a theory/operator/served verdict split
- broad served universality posture

## Why this matters

The terminal claim can now refer to one declared substrate instead of to
rhetorical “near-universality.” Every later universality witness and gate must
be a construction over `TCM.v1`.
