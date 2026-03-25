# Psionic Parameter Golf Single-H100 Trainer

> Status: canonical bounded Rust-native Parameter Golf single-H100 trainer
> command, written 2026-03-23 after landing
> `crates/psionic-train/src/parameter_golf_single_h100_training.rs` and
> `crates/psionic-train/src/bin/parameter_golf_single_h100_train.rs`.

This document records the current honest trainer posture for the Psionic
Parameter Golf single-H100 lane.

It is narrower than the real `8xH100` record lane, but it now follows the
same public single-device control-loop shape as `train_gpt.py` by default.

## What Landed

`psionic-train` now exposes:

- `ParameterGolfSingleH100TrainingConfig`
- `ParameterGolfSingleH100TrainingDisposition`
- `ParameterGolfSingleH100TrainingReport`
- `ParameterGolfSingleH100LiveVisualizationWriter`
- `build_parameter_golf_single_h100_training_report(...)`
- `write_parameter_golf_single_h100_training_report(...)`
- `parameter_golf_single_h100_visualization` as a Rust materializer for the
  provider-neutral app mirror
- `parameter_golf_single_h100_train` as a Rust binary entrypoint

That means the repo now owns one real Rust entrypoint for the public
single-device baseline path rather than only the narrower bring-up seam and the
older bounded local-reference trainer.

## Command

The binary defaults to the local `~/code/parameter-golf` clone paths from the
public README and now enters the widened challenge-style control loop when no
explicit step cap is passed:

```bash
cargo run -q -p psionic-train --bin parameter_golf_single_h100_train
```

You can also pass the explicit cached-dataset and tokenizer paths, an output
report path, an explicit bounded proof step count, an optional explicit
final-validation mode, an optional explicit validation eval mode, and an
optional trailing legal score-first TTT selector:

```bash
cargo run -q -p psionic-train --bin parameter_golf_single_h100_train -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/parameter_golf_single_h100_training.json \
  1 \
  roundtrip_only \
  sliding_window:64 \
  score_first_ttt
```

Passing the final positional step count selects the old bounded proof posture:

- `warmup_steps=0`
- `validation_loss_every=0`
- `train_log_every=1`
- default `final_validation_mode=roundtrip_only`
- no wallclock stop cap

Omitting that positional step count selects the widened baseline posture:

- `max_steps=20_000`
- `warmup_steps=20`
- `validation_loss_every=1000`
- `train_log_every=200`
- default `final_validation_mode=both`
- `max_wallclock_seconds=600`

Supported explicit final-validation modes are:

- `live_only`
- `roundtrip_only`
- `both`

Supported explicit validation eval modes are:

- `non_overlapping`
- `sliding_window:<stride>`

Today the widened trainer CLI accepts sliding-window eval on the single-H100
lane so the final live or roundtrip validation pass can score only the suffix
of each full-context window instead of only non-overlapping chunks.

Supported legal score-first TTT selectors are:

- `score_first_ttt`
- `legal_score_first_ttt`
- `score_first_ttt:stride=<n>,chunk_tokens=<n>,epochs=<n>,freeze_blocks=<n>,learning_rate=<f>,momentum=<f>,batch_sequences=<n>,grad_clip_norm=<f>`

The default `score_first_ttt` label expands to the current public leaderboard
README posture:

- `stride=64`
- `chunk_tokens=32768`
- `epochs=3`
- `freeze_blocks=0`
- `learning_rate=0.002`
- `momentum=0.9`
- `batch_sequences=32`
- `grad_clip_norm=1.0`

This selector is only valid when `validation_eval_mode=sliding_window:<stride>`
uses the same stride value.

The single-H100 trainer now also carries an explicit
`validation_batch_sequences` contract that is independent from the train batch
geometry. The default remains the train-geometry batch size for
`non_overlapping`, but sliding-window evaluation now widens onto the explicit
scoreboard batch surface instead of inheriting the train token cap implicitly.

The legality boundary now matches the README contract explicitly:

- each validation window is scored before any adaptation can see that chunk
- window ownership is assigned by the first scored token, not by the full
  context prefix
- chunk adaptation happens only after the score phase for that chunk completes
- the final chunk is scored but never trained
- adaptation mutates a cloned validation model and does not change the trained
  live-model weights or the exported `int8+zlib` artifact

The final validation summary now carries a machine-readable
`score_first_ttt_receipt` with:

- the resolved TTT config
- planned chunk boundaries
- scored window counts per chunk
- per-step adaptation receipts with learning-rate, clip, and token-count facts
- final aggregated validation metrics

For bounded same-node validation-runtime comparisons, the repo also exposes:

```bash
cargo run -q -p psionic-train --bin parameter_golf_validation_runtime_receipt -- \
  ~/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
  ~/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /tmp/parameter_golf_validation_runtime_comparison.json \
  8 \
  2
```

That sidecar receipt is intentionally a local runtime comparison tool, not a
contest metric surface.

## Data Setup

This doc assumes the public challenge cache has already been downloaded into
the local `~/code/parameter-golf` clone using the public README workflow:

```bash
cd ~/code/parameter-golf
.venv/bin/python data/cached_challenge_fineweb.py --variant sp1024
```

On the current local machine, that populates:

- `data/datasets/fineweb10B_sp1024/`
- `data/tokenizers/fineweb_1024_bpe.model`
- `data/tokenizers/fineweb_1024_bpe.vocab`

## What The Command Binds

The command is explicit about what it treats as trainer truth. It binds:

- the local cached FineWeb `sp1024` shard directory into a versioned
  `DatasetManifest`
- the local tokenizer file into a machine-readable `TokenizerDigest`
- the sibling `fineweb_1024_bpe.vocab` surface beside that tokenizer model
  into the validation byte-accounting LUTs, so reported `val_bpb` is derived
  from the real challenge tokenizer vocabulary instead of the tiny oracle
  parity fixture
- the local CUDA inventory into explicit single-H100 machine-admission truth
  using the repo-owned CUDA discovery substrate
- the public single-device batch geometry from
  `ParameterGolfBatchGeometry::challenge_single_device_defaults()`
- the public baseline `9x512` model contract and optimizer-plan digest
- the upstream-style four-bank matrix runtime descriptor
  (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) for the train-visible
  matrix surface, so the lowered graph can now bind the same bank vocabulary
  cited by the public top records instead of only the fully split per-layer
  matrix tensor list
- integer token ids directly into the lowered graph, where token embedding
  lookup now happens on-device rather than through a host-owned embedded-input
  `Vec<f32>` gather before each train or validation batch
- integer target ids directly into the on-device projection-loss path rather
  than routing them through dense `f32` target tensors
- the upstream mixed-precision optimizer split at the trainer-state boundary:
  BF16 train-visible embeddings and matrix weights, FP32 control tensors, and
  FP32 master weights for the Adam-managed embedding/head groups
- BF16 graph inputs for the train-visible token-embedding and linear weight
  surface, with the CUDA path now widening BF16 embedding tables and casting
  F32 activations to BF16 on-device before the existing BF16 matmul lane so
  the hot-path weight residency is no longer dense `f32` end to end
- BF16 autodiff admission for that same train-visible weight surface, so the
  single-H100 backward launcher now binds retained BF16 primal values and BF16
  seed or gradient tensors through the graph-declared dtype instead of
  silently forcing the hot path back to dense `f32`
- a narrowed backward-output and retained-primal surface on the shared CUDA
  train path, so the trainer now only materializes parameter-input gradients
  back to host and only keeps backward-live primal bindings instead of reading
  every gradient-bearing tensor and retaining every primal value that the full
  autodiff plan can expose
- slice-wise Muon updates over that banked matrix surface, so rank-3 bank
  tensors are now treated as stacks of equal-shaped matrices rather than
  forcing the optimizer path back to the split surface before every update
- the same single-device warmup-and-restore, repeated-step, periodic
  validation, train-log, and wallclock-stop control-loop shape the public
  `train_gpt.py` path uses
- one measured CUDA training run when the machine and CUDA-capability
  contracts are satisfied
- preserved initial, periodic, and final validation receipts directly from the
  Psionic path, with an explicit `final_validation_mode` telling the report and
  logs whether the last-step live-model validation, the exported int8+zlib
  roundtrip validation, or both were requested
- a device-resident validation runner that keeps the stable parameter surface
  resident on device across validation batches, reuses mutable token buffers,
  runs through the explicit `parameter_golf_baseline_eval_graph_v2` surface
  instead of the training-graph surface, now consumes one per-token projection
  loss surface on device, and records a machine-readable
  validation runtime receipt with the resident parameter buffer count,
  stable-buffer allocation posture, named eval graph surface, token-write
  cost, byte-accounting cost, and explicit eval mode for each validation pass
- a device-resident training runner that keeps the stable parameter surface
  resident on device across repeated train batches for each admitted batch
  shape, reuses mutable token and target buffers, refreshes the resident
  parameter buffers once per optimizer step instead of rebuilding full graph
  inputs every batch, reuses prepacked host `bf16` staging for BF16-visible
  parameter-state refresh instead of repacking those tensors on every step,
  and records one machine-readable training runtime receipt per completed step
  with resident-buffer counts, named graph surface, resident upload cost,
  parameter-refresh cost, and mutable token-write cost
- preserved initial, periodic, and final validation receipts directly from the
  Psionic path, with the pre-export live-model validation retained separately
  whenever that posture is requested
- post-step int8-plus-zlib artifact bytes, artifact ref, and artifact digest
- canonical final contest metrics from the exported int8-plus-zlib roundtrip
  artifact, including the preserved roundtrip eval time, when the requested
  final-validation mode includes the roundtrip pass
- stop reason plus measured warmup, training, validation, and per-step timing
  receipts so later same-node comparison work can reuse the same trainer report

The report also preserves explicit refusal when:

- the dataset root is missing or malformed
- the tokenizer path is missing
- the machine contract is not one qualifying non-MIG H100
- the committed Parameter Golf CUDA capability report still carries trainer
  blockers

## Live Visualization Mirror

When the trainer receives remote-training metadata through:

- `PSIONIC_REMOTE_TRAINING_PROVIDER`
- `PSIONIC_REMOTE_TRAINING_PROFILE_ID`
- `PSIONIC_REMOTE_TRAINING_LANE_ID`
- `PSIONIC_REMOTE_TRAINING_REPO_REVISION`

it now writes a provider-neutral live mirror beside the trainer report:

- `training_visualization/parameter_golf_single_h100_remote_training_visualization_bundle_v1.json`
- `training_visualization/remote_training_run_index_v1.json`

That mirror updates on the same one-second cadence frozen in
`docs/REMOTE_TRAINING_VISUALIZATION.md`.

It preserves:

- heartbeat truth about the active phase and micro-step position
- retained loss, optimizer-math, and runtime series for every completed step
- local GPU sampling when the host can run `nvidia-smi`
- explicit terminal posture once the trainer report lands

If the trainer report stays absent at finalization time, the materializer keeps
an explicit partial-series fallback from the retained training log instead of
inventing full-series truth.

## Current Honest Boundary

This command is still a single-H100 baseline parity step rather than a stronger
contest claim.

Today the single-H100 trainer doc does **not** claim:

- a Google operator lane
- `8xH100` distributed closure
- leaderboard-speed runtime closure
- distributed sliding-window eval closure yet; the single-H100 lane now
  accepts explicit sliding-window eval, but the distributed `8xH100` runtime
  still needs the same scoreboard-grade semantics wired end to end
- exported-folder score-first TTT closure; the single-H100 CUDA trainer can now
  execute the legal score-first TTT path, but the shipped bounded
  local-reference exported-folder runtime still refuses that request with a
  typed unsupported-validation error instead of pretending it ran it
- record-track accounting closure
- full BF16 activation-kernel closure yet; the current report now records BF16
  graph uploads for the train-visible token-embedding and linear weight path,
  but scalar/control tensors and retained activations remain explicit `f32`
  until the wider BF16 graph-runtime slice lands
- challenge-speed closure; the trainer now reports final contest metrics from
  the exported int8+zlib roundtrip artifact like `train_gpt.py`, but that does
  not by itself make the lane competitive yet

Instead, it gives the repo one narrower but important thing:

- a real Rust-owned single-H100 baseline training command that binds the
  challenge dataset, tokenizer, machine contract, challenge-style control
  loop, validation cadence, stop reason, pre-export live-model validation,
  canonical final int8+zlib roundtrip metrics, and compressed-model accounting
  surfaces into one machine-readable report
- one real bounded RunPod H100 proof that the same Rust trainer can complete
  remotely outside the local review host; see
  `fixtures/parameter_golf/reports/parameter_golf_runpod_single_h100_first_real_audit.json`
  and
  `docs/audits/2026-03-25-psionic-parameter-golf-runpod-single-h100-first-run-audit.md`

The narrower machine-admission seam from
`docs/PARAMETER_GOLF_SINGLE_H100_BRINGUP.md` remains useful, but it is no
longer the only repo-owned single-H100 entrypoint, and the old one-step proof
mode now exists only as an explicit CLI selection for bounded bring-up runs.
