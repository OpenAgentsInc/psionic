# Psion CS336 A2 Full Port Matrix

> Status: current Stanford CS336 Assignment 2 adapter coverage is no longer
> fully green. As of 2026-04-27 the bounded `psionic` reference lane covers
> FlashAttention and sharded-optimizer surfaces, maps the current DDP adapter
> names to older bounded DDP receipts, implements bounded `get_fsdp` wrapper
> lifecycle evidence, implements bounded `fsdp_on_after_backward`
> reduce-scatter evidence, and implements bounded `fsdp_gather_full_params`
> full-state reconstruction evidence. The lane has no current adapter row left
> in `missing_tracked`, but it is still bounded reference evidence rather than
> full transport-backed A2 parity.

This matrix is the hard truth bar for discussing Stanford CS336 A2 coverage in
`psionic`. It is not currently a full-parity claim.

It maps the adapter families declared in the Stanford reference
`assignment2-systems/tests/adapters.py` onto owned Rust implementation surfaces,
checked-in proof bundles, and the bounded systems-lane docs.

The current Spring 2026 reference adapter set is:

- `get_flashattention_autograd_function_pytorch`
- `get_flashattention_autograd_function_triton`
- `get_ddp`
- `ddp_on_after_backward`
- `get_fsdp`
- `fsdp_on_after_backward`
- `fsdp_gather_full_params`
- `get_sharded_optimizer`

Retained report:

- `fixtures/training/cs336_a2_full_port_conformance_report_v1.json`

Retained proof bundles:

- `fixtures/training/cs336_a2_baseline_profile_bundle_v1.json`
- `fixtures/training/cs336_a2_flashattention_reference_receipt_v1.json`
- `fixtures/training/cs336_a2_flashattention_fused_cuda_receipt_v1.json`
- `fixtures/training/cs336_a2_ddp_individual_parameters_receipt_v1.json`
- `fixtures/training/cs336_a2_ddp_bucketed_receipt_v1.json`
- `fixtures/training/cs336_a2_sharded_optimizer_receipt_v1.json`
- `fixtures/training/cs336_a2_fsdp_wrapper_receipt_v1.json`
- `fixtures/training/cs336_a2_fsdp_after_backward_receipt_v1.json`
- `fixtures/training/cs336_a2_fsdp_full_params_receipt_v1.json`

Reference-lane doc:

- `docs/PSION_CS336_A2_REFERENCE_LANE.md`

Claim boundary:

- This no longer closes full current A2 parity.
- It does not promote the A2 lane into the actual `Psion` pretraining operator lane.
- It does not claim admitted distributed throughput, transport-backed cluster
  execution, or actual-lane checkpoint sharding.
- It is not a prerequisite for `a1_minimal_distributed_lm_001`.

| Stanford adapter | Category | Owned `psionic` surface | Proof surface | Status |
| --- | --- | --- | --- | --- |
| `get_flashattention_autograd_function_pytorch` | `attention` | `crates/psionic-models/src/cs336_a2_flashattention_reference.rs`, `crates/psionic-train/src/cs336_a2_flashattention_reference_receipt.rs` | `build_cs336_a2_flashattention_reference_receipt` plus retained reference receipt fixture | `green_bounded_reference` |
| `get_flashattention_autograd_function_triton` | `attention` | `crates/psionic-backend-cuda/src/lib.rs`, `crates/psionic-train/src/cs336_a2_flashattention_fused_cuda_receipt.rs` | `build_cs336_a2_flashattention_fused_cuda_receipt` plus retained fused receipt or refusal fixture | `partial_bounded_reference`: fused CUDA analogue, not the Stanford Triton kernel surface |
| `get_ddp` | `distributed` | `crates/psionic-train/src/cs336_a2_ddp_individual_parameters_receipt.rs` | `build_cs336_a2_ddp_individual_parameters_receipt` plus retained two-rank DDP receipt | `partial_bounded_reference`: maps current name to the older individual-parameter receipt, without async overlap or transport-backed collectives |
| `ddp_on_after_backward` | `distributed` | `crates/psionic-train/src/cs336_a2_ddp_individual_parameters_receipt.rs` | retained `parameter_syncs` surface plus two-rank DDP receipt | `partial_bounded_reference`: proves bounded post-backward sync, not true async overlap |
| `get_fsdp` | `fsdp` | `crates/psionic-train/src/cs336_a2_fsdp_wrapper_receipt.rs` | `build_cs336_a2_fsdp_wrapper_receipt` plus retained wrapper lifecycle fixture | `partial_bounded_reference`: proves bounded Linear/Embedding sharding, all-gather planning, fp32 master restoration, fp16 compute-dtype admission, and model-state reconstruction, not transport-backed FSDP |
| `fsdp_on_after_backward` | `fsdp` | `crates/psionic-train/src/cs336_a2_fsdp_after_backward_receipt.rs` | `build_cs336_a2_fsdp_after_backward_receipt` plus retained after-backward fixture | `partial_bounded_reference`: proves bounded sharded-gradient reduce-scatter, replicated-gradient all-reduce equivalence, fp32 master-gradient restoration, and fp32/fp16 parity before optimizer step, not transport-backed FSDP |
| `fsdp_gather_full_params` | `fsdp` | `crates/psionic-train/src/cs336_a2_fsdp_full_params_receipt.rs` | `build_cs336_a2_fsdp_full_params_receipt` plus retained full-parameter gather fixture | `partial_bounded_reference`: proves bounded full state-dict reconstruction after each step, sharded Linear/Embedding all-gather, replicated parameter return-as-is handling, and fp32/fp16 baseline comparison, not transport-backed FSDP |
| `get_sharded_optimizer` | `optimizer` | `crates/psionic-train/src/cs336_a2_sharded_optimizer_receipt.rs` | `build_cs336_a2_sharded_optimizer_receipt` plus retained sharded-optimizer receipt | `green_bounded_reference` |

The older bucketed DDP receipt remains useful bounded systems evidence, but it
is no longer a current Stanford adapter-family row because the current
`adapters.py` surface does not expose `get_ddp_bucketed`,
`ddp_bucketed_on_after_backward`, or `ddp_bucketed_on_train_batch_start`.
