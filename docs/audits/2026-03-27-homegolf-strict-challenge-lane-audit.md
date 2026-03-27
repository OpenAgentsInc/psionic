# HOMEGOLF Strict Challenge Lane Audit

> Status: retained 2026-03-27 proof for `HOMEGOLF-8`

This audit records the canonical runnable strict HOMEGOLF lane surface.

## What Landed

Generator:

- `crates/psionic-train/src/parameter_golf_homegolf_strict_challenge.rs`

Entrypoint:

- `crates/psionic-train/src/bin/parameter_golf_homegolf_strict_challenge_lane.rs`

Retained report:

- `fixtures/parameter_golf/reports/parameter_golf_homegolf_strict_challenge_lane.json`

Checker:

- `scripts/check-parameter-golf-homegolf-strict-challenge-lane.sh`

## What It Changes

Before this issue, the only easy HOMEGOLF rerun command was the bounded
exact-family bundle proof. That path was real, but it still carried:

- `evaluation_identity = local_reference_validation`
- `contest_bits_per_byte_accounting_required = false`
- `exact_compressed_artifact_cap_required = false`

So it was a real proof lane, not a contest-style lane.

The new strict HOMEGOLF lane fixes that by freezing the real contest overlay in
the canonical runnable command surface:

- strict challenge profile id
- exact FineWeb `SP1024` data-lane requirement
- exact challenge tokenizer requirement
- `sliding_window:64` evaluation
- legal score-first TTT
- contest bits-per-byte accounting
- exact `16,000,000`-byte artifact-cap law

## Current Retained Result

The retained report is currently a refusal, on purpose.

Current disposition:

- `refused_missing_challenge_inputs`

Current refusal reason:

- exact dataset root not supplied
- exact tokenizer path not supplied
- local-reference fallback explicitly denied

That is the correct current truth for the canonical runnable lane. The new lane
does not pretend the old local-reference proof is contest-valid. It either gets
the exact challenge inputs or it refuses.

## What It Proves

- HOMEGOLF now has a canonical runnable strict challenge lane surface
- the emitted profile is the strict challenge profile, not
  `general_psion_small_decoder`
- the emitted policy keeps contest BPB accounting and exact artifact-cap
  accounting enabled
- missing challenge inputs now fail closed with a typed refusal

## What It Does Not Prove

- that the exact challenge dataset and tokenizer are already available locally
- that a live strict dense HOMEGOLF run has already completed
- that the mixed-device home cluster already produces the scored bundle

That next step belongs to the later HOMEGOLF execution issues, not this one.
