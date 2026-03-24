# Psionic Parameter Golf Validation Runtime Audit

> Status: written 2026-03-24 after reading
> `~/code/parameter-golf/README.md`,
> `~/code/parameter-golf/train_gpt.py`,
> `crates/psionic-train/src/parameter_golf_single_h100_training.rs`,
> `crates/psionic-train/src/parameter_golf_baseline_graph.rs`,
> `crates/psionic-backend-cuda/src/lib.rs`, and the live RunPod single-H100
> run rooted at `/workspace/issue454_single_h100_run_20260324T182307Z`.

## Summary

The current Psionic single-H100 trainer is now far enough along to expose the
real validation problem clearly.

The validation path is no longer primarily blocked by the old host-fallback
surface. The live H100 receipt shows that validation fallback is already down
to a small bounded `permute` family, while the first final-validation batch
still took `97,003 ms`.

That means the dominant problem is now the core forward execution model, not
the remaining fallback profile.

As of March 24, 2026:

- the live bounded run completed all `8/8` train microsteps and one optimizer
  step
- the first final-validation batch logged at `97,003 ms`
- the trainer still performs two full post-train validations:
  - one live-model final validation
  - one final int8+zlib roundtrip validation

At the current observed rate:

- one full validation pass projects to about `25.5` hours on `1xH100`
- the two-pass post-train evaluation surface projects to about `51` hours on
  `1xH100`

That is materially better than the earlier `30+` hour projection for a single
validation pass, but it is still nowhere near honest challenge-speed.

## Current Evidence

The live run at `/workspace/issue454_single_h100_run_20260324T182307Z` has
already logged:

- `micro_step=1/8` through `micro_step=8/8`
- `train_step_complete step=1 mean_microbatch_loss=8.28842449`
- `step:1/1 train_loss:8.2884 train_time:1900968ms`
- `final_validation_start sequences=60568 batch_sequences=64`
- `validation_progress stage=final_validation batch=1/947 ... elapsed_ms=97003`

The same run's validation fallback receipts are already narrow:

- repeated forward validation receipts show only `permute`
- recent forward validation receipts are roughly `76-103 ms` total host
  fallback

So the forward validation batch cost is about `97 s`, while only about `0.08 s`
to `0.10 s` of that shows up in host fallback. Validation is therefore not
currently dominated by the fallback surface.

The train path still has real backward fallback costs, but that is a separate
issue. The live microstep receipts and the fallback profile show:

- validation slowness is a forward-path problem
- training slowness still has a backward-path problem

Those should not be conflated.

## What The Current Code Does

### 1. Validation runs one full host-built batch loop in Rust

`evaluate_validation_on_cuda(...)` in
`crates/psionic-train/src/parameter_golf_single_h100_training.rs:1509-1623`
iterates over all validation sequences in batches of `64`, and for each batch
it:

- slices the validation token stream on the host
- rebuilds `input_ids` as `Vec<Vec<u32>>`
- rebuilds `target_ids` as `Vec<Vec<u32>>`
- binds a fresh graph input map
- executes the graph
- re-flattens tokens again to count bytes

This is not the same execution posture as `train_gpt.py`.

### 2. Every validation batch rebinds the entire model surface from host memory

`bind_parameter_golf_baseline_training_graph_inputs(...)` in
`crates/psionic-train/src/parameter_golf_baseline_graph.rs:234-310` rebuilds a
fresh `BTreeMap<TensorId, TensorData>` for every batch.

That path:

- re-derives `parameter_vectors(...)`
- clones each parameter payload into fresh `TensorData`
- re-materializes token-id and target-id buffers on the host

So even though the model itself is logically unchanged during validation, the
validation loop does not keep the model resident as a stable device-side eval
state.

### 3. Every graph input is turned into a fresh CUDA buffer for every batch

`execute_cuda_graph_outputs(...)` in
`crates/psionic-train/src/parameter_golf_single_h100_training.rs:1798-1858`
walks every input tensor and allocates a fresh CUDA buffer from the host
payload:

- `input_buffer(...)` for `f32`
- `input_bf16_buffer(...)` for `bf16`
- `input_i32_buffer(...)` for token ids

Then it calls `compile_and_execute(...)`.

Graph structure is cached by batch size, and CUDA execution plans are also
cached, so this is not "recompile the whole graph every batch" in the simple
sense. But it is still "re-stage the whole input surface every batch", which is
very different from the upstream PyTorch model residency posture.

### 4. The current attention forward path is fundamentally not a fast training/eval kernel

The critical validation bottleneck is in
`crates/psionic-backend-cuda/src/lib.rs:3484-3643`.

`execute_scaled_dot_product_attention_step(...)` does not run one full-sequence
attention kernel across `[batch, heads, seq, head_dim]`.

Instead it:

- allocates scratch buffers
- loops over every `batch_index`
- loops over every `position`
- packs one token into scratch
- calls `attention_decode(...)`
- scatters the token output back out
- copies one cache row into `key_cache` and `value_cache`

This is effectively a host-orchestrated decode-style loop used to emulate
training-time self-attention. It is a bounded correctness path, but it is not a
challenge-speed validation path.

The live receipt lines up with that design:

- the first validation batch takes about as long as one train forward microstep
- forward fallback is already tiny
- therefore the main forward wallclock is now inside the "supported" path, not
  the fallback path

### 5. The trainer pays for two full final validations

The trainer performs:

- one pre-export final validation inside the main loop when `last_step` is true
  in `crates/psionic-train/src/parameter_golf_single_h100_training.rs:776-818`
- one full roundtrip validation after int8+zlib export in
  `crates/psionic-train/src/parameter_golf_single_h100_training.rs:887-918`

That mirrors the upstream control-loop shape, but at the current runtime it
means we pay the full validation cost twice.

This is acceptable only if validation itself is already cheap. Right now it is
not.

## Comparison To `train_gpt.py`

The upstream validation path in `~/code/parameter-golf/train_gpt.py:217-278`
and `~/code/parameter-golf/train_gpt.py:976-1116` differs in three important
ways:

- the model stays resident on device during validation
- validation runs under `torch.inference_mode()` plus BF16 autocast
- each batch is a direct `model(x, y)` call rather than a host-built graph
  input map plus device re-staging plus a token-by-token attention forward loop

The batch geometry is not the main mismatch. Psionic already uses the same
public single-device batch geometry:

- `validation_batch_tokens = 524,288`
- `train_seq_len = 1024`
- `grad_accum_steps = 8`
- local validation batch size `= 64` sequences

That means the shape contract is mostly aligned. The execution model is not.

## Gap To A 10-Minute Evaluation Budget

The challenge requires evaluation to fit within a separate `10` minute budget
on `8xH100`.

With the public geometry:

- total validation sequences: `60,568`
- local validation batch size: `64` sequences
- on `8xH100`, that still implies about `119` local batches per rank

To finish one validation pass in `600` seconds, the local per-rank validation
batch needs to average about:

- `600 / 119 ~= 5.0` seconds per local batch

The current live Psionic rate is:

- `97.0` seconds for the first local validation batch on `1xH100`

Even granting ideal `8x` sharding, that is still nowhere near enough. The local
batch needs to become about `19x` faster before the eval path is even in the
same order of magnitude as the challenge budget.

This is why continuing to shave tens of milliseconds off validation fallback is
not the main answer anymore.

## Path Forward

### Priority 0: Stop Paying For Two Full Validations Unless Explicitly Needed

This is the fastest near-term wallclock win.

Add an explicit validation mode to the single-H100 trainer, for example:

- `live_only`
- `roundtrip_only`
- `both`

Recommended immediate default for bounded proof runs:

- `roundtrip_only`

Recommended explicit debug posture:

- `both`

This does not solve challenge-speed validation, but it immediately avoids
paying for two `25.5` hour-class passes when only the final roundtrip metric is
needed.

### Priority 1: Build A Device-Resident Validation Runner

The current validation loop should stop re-staging the whole model every batch.

Required changes:

- keep validation weights resident on device across all batches
- keep one cached eval graph and one cached compiled plan per batch size
- preallocate reusable device buffers for:
  - input token ids
  - target ids
  - any small control inputs
- update only the mutable token buffers per batch instead of rebuilding a full
  `BTreeMap<TensorId, TensorData>`
- stop cloning parameter vectors into fresh `TensorData` for every batch

This is the minimum structural fix needed before the validation path can be
considered a real GPU eval runner rather than a repeated host-to-device replay
loop.

### Priority 2: Replace The Attention Forward Path

This is the main validation-speed problem.

Validation is not currently slow because of fallback. It is slow because the
"supported" attention forward path is a host-orchestrated token loop.

The next major kernel milestone for validation should be:

- one full-sequence causal attention forward implementation for the bounded
  Parameter Golf shapes
- operating directly over `[batch, query_heads, seq, head_dim]` and grouped
  query heads
- BF16-first where the upstream path is BF16-first

In practical terms:

- do not keep using the per-position `attention_decode(...)` loop for training
  or validation forward
- validation speed will not become acceptable until that loop is replaced by a
  true sequence-parallel attention path

If only one runtime project is chosen specifically to speed up validation, it
should be this one.

### Priority 3: Stop Recomputing Validation Byte Accounting Batch By Batch

`evaluate_validation_on_cuda(...)` currently rebuilds flat token vectors per
batch solely to call `byte_luts.count_target_bytes(...)`.

This is not the dominant `97 s` cost, but it is still unnecessary overhead.

A cleaner approach is:

- precompute target-byte counts for the validation stream once
- or precompute per-window byte counts aligned to the sequence layout
- or compute byte counts from stable flat token slices without rebuilding
  `Vec<Vec<u32>>` and then flattening again

This should be treated as cleanup after device residency and attention forward,
not as the first optimization.

### Priority 4: Add A Real Distributed Validation Path

Even after the single-rank validation runner is corrected, the record-track
target is still `8xH100`, not `1xH100`.

Validation needs a real distributed posture that mirrors the upstream logic:

- shard validation sequences across `world_size`
- keep per-rank local batch sequences at the expected geometry
- aggregate `loss_sum`, `token_count`, and `byte_count` across ranks
- report one honest end-to-end eval time on the real `8xH100` lane

Without this, even a much faster single-H100 validation path is still not a
record-track evaluation path.

### Priority 5: Split Eval Graph Truth From Train Graph Truth

Today validation reuses the lowered training graph and its surrounding trainer
plumbing.

That is acceptable as a bounded correctness path, but it is not the cleanest
route to fast eval.

After the device-resident validation runner exists, it is worth adding a
separate eval-oriented graph or program surface that:

- accepts persistent resident weights
- accepts mutable token-id / target-id buffers
- emits only what validation needs
- avoids training-specific assumptions or scaffolding

This should come after the earlier priorities, not before them.

## What Not To Prioritize For Validation

The current live receipts make several tempting but wrong optimization paths
clear:

- do not treat `permute` fallback cleanup as the main validation problem
- do not treat `add` fallback cleanup as the main validation problem
- do not expect `scaled_dot_product_attention_query_backward` work to speed up
  final validation; that is a train-path issue
- do not assume compile caching is the main missing piece; graph and plan reuse
  already exist, and the batch is still ~`97 s`

The main validation problem is the forward runtime structure.

## Recommended Execution Order

1. Add explicit validation mode and stop forcing two full validations for every
   bounded proof run.
2. Build a device-resident validation runner with persistent parameter buffers
   and reusable token buffers.
3. Replace the current token-by-token attention forward path with a real
   full-sequence attention implementation.
4. Clean up byte-accounting overhead and other per-batch host churn.
5. Move the corrected validation runner onto the real `8xH100` distributed
   lane.

## Honest Current Bottom Line

The current Psionic validation path is no longer blocked by the early obvious
CUDA failures.

It is now blocked by a harder but clearer truth:

- validation is mostly spending time inside the "supported" forward execution
  model
- that execution model is still structurally much closer to a bounded
  correctness proof than to the upstream fast eval path

Until the validation runner becomes device-resident and the attention forward
path stops using the current per-position decode loop, the repo should not
claim anything close to challenge-speed validation.
