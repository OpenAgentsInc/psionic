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
- RunPod upstream operator wrapper:
  `scripts/parameter-golf-runpod-run-train-gpt-reference.sh`
- same-node parity build wrapper:
  `scripts/parameter-golf-build-same-node-parity.sh`
- same-node parity chain wrapper:
  `scripts/parameter-golf-runpod-run-same-node-parity-chain.sh`

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

Run the upstream single-H100 PyTorch baseline on a RunPod node and preserve the
raw log:

```bash
bash scripts/parameter-golf-runpod-run-train-gpt-reference.sh \
  --repo-root /workspace/parameter-golf \
  --log /workspace/parameter_golf_train_gpt_reference.log
```

The upstream runner now prefers `/workspace/parameter-golf/.venv/bin/python`
when it exists. If the current Python environment cannot import the required
packages or cannot execute `scaled_dot_product_attention(..., enable_gqa=...)`,
the runner bootstraps or upgrades `repo-root/.venv` from
`requirements.txt` before it launches `train_gpt.py`. That keeps the same-node
parity path reproducible on pods whose system Python ships an older `torch`.

Build one parity report:

```bash
cargo run -q -p psionic-train --example parameter_golf_same_node_parity_report -- \
  <psionic-single-h100-report.json> \
  <upstream-train-gpt-receipt.json> \
  <same-node-parity-report.json>
```

Or use the repo-owned wrapper to derive the upstream identity fields from the
completed Psionic report and emit both artifacts in one step:

```bash
bash scripts/parameter-golf-build-same-node-parity.sh \
  --psionic-report <psionic-single-h100-report.json> \
  --upstream-log /workspace/parameter_golf_train_gpt_reference.log \
  --output-dir <same-node-parity-artifacts-dir>
```

Or use the repo-owned chain wrapper on the RunPod node to wait for the live
Psionic run, fast-forward the repo checkout, run the upstream baseline, and
then emit the same-node parity artifacts:

```bash
bash scripts/parameter-golf-runpod-run-same-node-parity-chain.sh \
  --psionic-repo-root /workspace/psionic \
  --upstream-repo-root /workspace/parameter-golf \
  --run-root /workspace/issue454_single_h100_run_20260324T215106Z \
  --trainer-pid 10414 \
  --cargo-target-dir /tmp/psionic-target
```

On shared or long-lived checkouts, prefer a scratch `CARGO_TARGET_DIR` for the
follow-up parity build. The chain wrapper now exports that target dir before it
invokes the Rust receipt builders, which avoids exhausting space under the repo
checkout when the node has limited free capacity.

## Honest Boundary

This harness surface does not claim:

- that a same-node H100 parity receipt already exists
- that the current repo is already faster than the upstream baseline
- that missing metrics like peak memory or validation timing are already
  populated on both sides

It closes one narrower but important thing:

- the repo now has one typed fail-closed comparison surface ready to consume a
  real Psionic single-H100 receipt plus a real normalized `train_gpt.py`
  receipt, and the repo now also ships explicit RunPod wrappers for producing
  those inputs, including one repo-owned chain wrapper for the live-node
  follow-up, so the later H100 run only needs evidence rather than fresh
  schema, parser, or operator glue work
