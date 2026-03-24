# Parameter Golf Same-Node Parity Harness

> Status: canonical pre-H100 parity-harness contract for `#472`, written on
> 2026-03-24 after landing the first typed upstream-receipt and comparison
> surfaces.

This document records the current same-node parity harness surface for the
single-H100 Parameter Golf lane.

It is narrower than a completed H100 parity receipt and narrower than closing
the issue itself.

## Canonical Surfaces

- upstream normalized receipt example:
  `crates/psionic-train/examples/parameter_golf_train_gpt_reference_run_receipt.rs`
- same-node parity report example:
  `crates/psionic-train/examples/parameter_golf_same_node_parity_report.rs`
- typed report logic:
  `crates/psionic-train/src/parameter_golf_same_node_parity.rs`

## What Landed

The harness now has two explicit machine-readable surfaces:

- one normalized upstream `train_gpt.py` receipt that binds:
  - the exact source log path and digest
  - operator-supplied hardware and input identity that the upstream log does
    not carry safely by itself
  - parsed train-step timing
  - parsed peak allocated/reserved memory
  - parsed final validation metrics
  - parsed final int8+zlib roundtrip metrics
  - parsed compressed-model bytes
- one same-node parity report that compares that upstream receipt against the
  existing Psionic single-H100 trainer report and emits explicit judgments for:
  - train-step wallclock
  - validation wallclock
  - peak memory allocated
  - final roundtrip `val_loss`
  - final roundtrip `val_bpb`
  - compressed-model bytes

The parity report is fail-closed:

- it blocks if the Psionic receipt is not a real single-H100 executed run
- it blocks if hardware, dataset, tokenizer, or geometry do not match
- it blocks if one side is missing a required comparison metric

That keeps the later H100 parity review honest instead of relying on ad hoc log
reading.

## Commands

Normalize one upstream `train_gpt.py` run:

```bash
cargo run -q -p psionic-train --example parameter_golf_train_gpt_reference_run_receipt -- \
  --run-id <run-id> \
  --log <train-gpt-log-path> \
  --output <upstream-receipt-path> \
  --device-name "NVIDIA H100 80GB HBM3" \
  --dataset-manifest-digest <digest> \
  --tokenizer-digest <digest> \
  --world-size 1 \
  --train-batch-tokens 524288 \
  --validation-batch-tokens 524288 \
  --train-sequence-length 1024 \
  --grad-accum-steps 8
```

Build one parity report:

```bash
cargo run -q -p psionic-train --example parameter_golf_same_node_parity_report -- \
  <psionic-single-h100-report.json> \
  <upstream-train-gpt-receipt.json> \
  <same-node-parity-report.json>
```

## Honest Boundary

This harness surface does not claim:

- that a same-node H100 parity receipt already exists
- that the current repo is already faster than the upstream baseline
- that missing metrics like peak memory or validation timing are already
  populated on both sides

It closes one narrower but important thing:

- the repo now has one typed fail-closed comparison surface ready to consume a
  real Psionic single-H100 receipt plus a real normalized `train_gpt.py`
  receipt, so the later H100 run only needs evidence rather than fresh schema
  or parser work
