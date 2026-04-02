# Psion CS336 A1 Full Port Matrix

> Status: full Stanford CS336 Assignment 1 adapter coverage is complete in the
> bounded `psionic` reference lane as of 2026-04-02.

This matrix is the hard completion bar for claiming a full Stanford CS336 A1
port in `psionic`.

It maps the adapter families declared in the Stanford reference
`assignment1-basics/tests/adapters.py` onto owned Rust implementation surfaces,
unit tests, and retained proof artifacts.

Retained report:

- `fixtures/training/cs336_a1_full_port_conformance_report_v1.json`

Retained training proof bundle:

- `fixtures/training/cs336_a1_reference_tiny_training_bundle_v1.json`

Claim boundary:

- This closes full A1 only as a bounded reference lane.
- It does not promote the A1 lane into the actual `Psion` pretraining operator lane.
- It does not claim scalable broader-pretraining backward support beyond the
  tiny finite-difference reference trainer.

| Stanford adapter | Category | Owned `psionic` surface | Proof surface | Status |
| --- | --- | --- | --- | --- |
| `run_linear` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `linear_matches_manual_projection` | `green` |
| `run_embedding` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `embedding_matches_table_lookup` | `green` |
| `run_swiglu` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `swiglu_composes_gate_and_value_paths` | `green` |
| `run_scaled_dot_product_attention` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `scaled_dot_product_attention_matches_manual_example` | `green` |
| `run_multihead_self_attention` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `multihead_self_attention_supports_identity_projection_path` | `green` |
| `run_multihead_self_attention_with_rope` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `multihead_self_attention_with_rope_executes_end_to_end` | `green` |
| `run_rope` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `rope_rotates_even_odd_pairs` | `green` |
| `run_transformer_block` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `transformer_block_with_zero_submodules_is_identity` | `green` |
| `run_transformer_lm` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `transformer_lm_executes_end_to_end_and_exposes_expected_state_dict_keys` | `green` |
| `run_rmsnorm` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `rms_norm_scales_by_root_mean_square` | `green` |
| `run_silu` | `model` | `crates/psionic-models/src/cs336_a1_reference_stack.rs` | `silu_matches_reference_formula` | `green` |
| `run_get_batch` | `training` | `crates/psionic-train/src/cs336_a1_reference_training.rs` | `get_batch_cycles_deterministically` | `green` |
| `run_softmax` | `training` | `crates/psionic-train/src/cs336_a1_reference_training.rs` | `softmax_normalizes_requested_dimension` | `green` |
| `run_cross_entropy` | `training` | `crates/psionic-train/src/cs336_a1_reference_training.rs` | `cross_entropy_matches_expected_average` | `green` |
| `run_gradient_clipping` | `training` | `crates/psionic-train/src/cs336_a1_reference_training.rs` | `gradient_clipping_enforces_global_norm_bound` | `green` |
| `get_adamw_cls` | `training` | `crates/psionic-train/src/cs336_a1_reference_training.rs` | `adamw_config_updates_parameters_on_first_step` | `green` |
| `run_get_lr_cosine_schedule` | `training` | `crates/psionic-train/src/cs336_a1_reference_training.rs` | `cosine_schedule_warms_up_and_decays` | `green` |
| `run_save_checkpoint` | `training` | `crates/psionic-train/src/cs336_a1_reference_training.rs` | `checkpoint_round_trip_preserves_iteration_and_state_digests` plus retained checkpoint/bundle fixtures | `green` |
| `run_load_checkpoint` | `training` | `crates/psionic-train/src/cs336_a1_reference_training.rs` | `checkpoint_round_trip_preserves_iteration_and_state_digests` plus retained checkpoint/bundle fixtures | `green` |
| `get_tokenizer` | `tokenizer` | `crates/psionic-models/src/cs336_a1_tokenizer.rs` | `tokenizer_round_trips_text_and_special_tokens`, `tokenizer_prefers_the_longest_special_token`, `tokenizer_streaming_surface_matches_direct_encoding` | `green` |
| `run_train_bpe` | `tokenizer` | `crates/psionic-data/src/cs336_a1_bpe.rs` | `trainer_uses_lexicographically_greatest_pair_for_ties`, `trainer_emits_reconstructible_artifacts` | `green` |
