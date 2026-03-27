# Psion Pilot Pretraining Run

> Status: canonical `PSION-14` / `#370` pilot-run contract, written
> 2026-03-22 after landing the first pretrain-stage and observability receipts.

This document freezes the first bounded `Psion` pilot pretraining run.

It does not claim broader pretraining closure. It records one explicit pilot
bundle that proves the corpus, tokenizer, decoder, trainer, replay, checkpoint,
and promotion surfaces can survive one bounded end-to-end run.

It also does not claim accelerator-backed training. The current reference pilot
is the canonical CPU-reference lane. It remains valuable for bounded receipts,
checkpoint, replay, and promotion truth, but it is not a valid Google GPU
training proof target by itself.

It is also no longer the primary first-model train-to-infer target. The
promoted PGOLF-shaped first-model proof path now lives separately at:

- `crates/psionic-train/examples/parameter_golf_promoted_reference_run.rs`

That promoted path is the honest closure target for the first serious
train/checkpoint/restore/resume artifact family and now emits the canonical
`descriptor.json` + `model.safetensors` + `tokenizer.json` + `generation_config.json`
bundle surface for later inference and serve work; this Psion pilot remains the
bounded smoke-test and receipt lane.

The canonical bounded accelerator-backed single-node trainer now exists
separately at:

- `crates/psionic-train/examples/psion_accelerated_reference_pilot.rs`

That accelerated lane reuses the same bounded corpus, decoder, checkpoint, and
receipt contract, but binds the training step path to CUDA instead of the CPU
reference optimizer lane.

## Canonical Artifacts

- `crates/psionic-train/src/psion_pilot_pretraining_run.rs` owns the pilot
  held-out-loss receipt, route/refusal probe receipt, and full pilot run
  bundle.
- `crates/psionic-train/examples/psion_reference_pilot.rs` is the canonical
  CPU-reference trainer entrypoint for bounded receipt and checkpoint truth.
- `crates/psionic-train/examples/psion_accelerated_reference_pilot.rs` is the
  canonical bounded CUDA-backed single-node trainer entrypoint.
- `crates/psionic-train/examples/psion_pilot_pretraining_run_fixtures.rs`
  regenerates the canonical pilot fixtures.
- `fixtures/psion/pilot/psion_pilot_held_out_loss_receipt_v1.json` is the
  canonical held-out-loss receipt across textbooks, specs, and technical docs.
- `fixtures/psion/pilot/psion_pilot_route_probe_receipt_v1.json` is the
  canonical pilot route/refusal probe receipt.
- `fixtures/psion/pilot/psion_pilot_pretraining_run_bundle_v1.json` is the
  canonical full pilot run bundle.

The stable schema versions are:

- `psion.pilot_held_out_loss_receipt.v1`
- `psion.pilot_route_probe_receipt.v1`
- `psion.pilot_pretraining_run_bundle.v1`

## What The Pilot Bundle Freezes

The first pilot bundle now binds:

- one explicit pretrain-stage receipt from `PSION-12`
- one explicit run-observability receipt from `PSION-13`
- held-out loss deltas across textbooks, normative specs, and technical docs
- explicit route probes for direct answer, exact-executor handoff, and refusal
- one canonical refusal-calibration receipt tied to the unsupported-request
  refusal benchmark package and capability matrix
- one promotion decision that is actually recordable through the canonical
  acceptance-matrix ledger

That makes the first pilot a repo-owned evidence bundle instead of a narrative
claim.

## Mechanical Enforcement

`psionic-train` now validates that:

- the pilot bundle still points at one exact pilot pretrain-stage receipt and
  one exact pilot observability receipt
- held-out loss improvement is positive across textbooks, specs, and technical
  docs
- the route probe surface covers direct answer, exact-executor handoff, and
  refusal explicitly
- the refusal evidence in the promotion decision stays bound to the canonical
  unsupported-request refusal package digest and refusal-calibration receipt
- the acceptance-matrix decision still targets the promoted checkpoint emitted
  by the pilot stage
- replay evidence in the promotion decision matches the replay facts carried by
  the pilot stage
- the held-out improvement metric in the promotion decision still matches the
  minimum improvement surfaced by the held-out-loss receipt
- the pilot bundle also requires a bounded specification-vs-implementation
  benchmark receipt with pass rate at or above `8600` bps before the pilot
  publication can stay green
