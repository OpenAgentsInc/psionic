# Public Work Assignment Reference

> Status: canonical `XTRAIN-33` / `#577` record, updated 2026-03-26 after
> landing the first deterministic public-work assignment contract.

## What This Closes

Psionic now owns one deterministic assignment surface for public miners and
validators.

The new contract lives in:

- `crates/psionic-train/src/public_work_assignment_contract.rs`
- `crates/psionic-train/src/bin/public_work_assignment_contract.rs`
- `fixtures/training/public_work_assignment_contract_v1.json`
- `scripts/check-public-work-assignment-contract.sh`

This issue closes the first truthful answer to:

- which network window was active
- which public miner worked on which dataset page slice
- which validator challenged which miner slice
- why post-close work is refused

## Contract Shape

The canonical contract freezes:

- two public windows
- eight public work assignments
- eight assignment receipts
- one late-window refusal

## Current Canonical Windows

The fixture keeps two deterministic windows explicit:

- `window.public.1230`
- `window.public.1231`

Each window binds:

- one stable assignment seed
- the current checkpoint-authority pair
- two miner assignments
- two validator assignments

## Current Canonical Work Selection

The current miner assignments are:

- Google on `dataset.page.train.0001_0004`
- Apple MLX on `dataset.page.train.0005_0008`
- Google on `dataset.page.train.0009_0012`
- Apple MLX on `dataset.page.train.0013_0016`

The current validator challenges are the cross-checks over those miner slices,
so the validator lane is explicit rather than implied by later score output.

## Honest Refusal Boundary

The first timing refusal is explicit:

- `late_refusal.window1230.google.replay`

That refusal proves window closure matters: a replay attempt against the sealed
Google miner assignment in window `1230` is rejected after close instead of
being treated as still valid work.

## Existing Psionic Binding

The work-assignment contract binds directly to:

- `DecentralizedNetworkContract`
- `PublicNetworkRegistryContract`
- `ElasticDeviceMeshContract`

That means assignment truth stays grounded in:

- admitted node identity
- admitted public roles
- current active miner, validator, and checkpoint-authority lanes

## Pass Criteria

The contract is green only if all of the following stay true:

- both windows remain ordered and non-overlapping
- each window keeps two miner and two validator assignments
- every assignment keeps one deterministic receipt
- validator assignments still target miner assignments in the same window
- the late-window refusal remains after close

## Current Limits

This issue intentionally does not claim:

- page-proofed dataset authority
- anti-replay data receipts
- content-addressed artifact exchange
- full public miner execution protocol

This issue freezes public time and work truth first: window clocks, assignment
selection, validator challenge targeting, and late-window refusal.
