# Psion CS336 A2 Full Port Matrix

> Status: full Stanford CS336 Assignment 2 adapter coverage is complete in the
> bounded `psionic` reference lane as of 2026-04-03.

This matrix is the hard completion bar for claiming a full Stanford CS336 A2
port in `psionic`.

It maps the adapter families declared in the Stanford reference
`assignment2-systems/tests/adapters.py` onto owned Rust implementation surfaces,
checked-in proof bundles, and the bounded systems-lane docs.

Retained report:

- `fixtures/training/cs336_a2_full_port_conformance_report_v1.json`

Retained proof bundles:

- `fixtures/training/cs336_a2_baseline_profile_bundle_v1.json`
- `fixtures/training/cs336_a2_flashattention_reference_receipt_v1.json`
- `fixtures/training/cs336_a2_flashattention_fused_cuda_receipt_v1.json`
- `fixtures/training/cs336_a2_ddp_individual_parameters_receipt_v1.json`
- `fixtures/training/cs336_a2_ddp_bucketed_receipt_v1.json`
- `fixtures/training/cs336_a2_sharded_optimizer_receipt_v1.json`

Reference-lane doc:

- `docs/PSION_CS336_A2_REFERENCE_LANE.md`

Claim boundary:

- This closes full A2 only as a bounded systems reference lane.
- It does not promote the A2 lane into the actual `Psion` pretraining operator lane.
- It does not claim admitted distributed throughput, transport-backed cluster
  execution, or actual-lane checkpoint sharding.

| Stanford adapter | Category | Owned `psionic` surface | Proof surface | Status |
| --- | --- | --- | --- | --- |
| `get_flashattention_autograd_function_pytorch` | `attention` | `crates/psionic-models/src/cs336_a2_flashattention_reference.rs`, `crates/psionic-train/src/cs336_a2_flashattention_reference_receipt.rs` | `build_cs336_a2_flashattention_reference_receipt` plus retained reference receipt fixture | `green` |
| `get_flashattention_autograd_function_triton` | `attention` | `crates/psionic-backend-cuda/src/lib.rs`, `crates/psionic-train/src/cs336_a2_flashattention_fused_cuda_receipt.rs` | `build_cs336_a2_flashattention_fused_cuda_receipt` plus retained fused receipt or refusal fixture | `green` |
| `get_ddp_individual_parameters` | `distributed` | `crates/psionic-train/src/cs336_a2_ddp_individual_parameters_receipt.rs` | `build_cs336_a2_ddp_individual_parameters_receipt` plus retained two-rank DDP receipt | `green` |
| `ddp_individual_parameters_on_after_backward` | `distributed` | `crates/psionic-train/src/cs336_a2_ddp_individual_parameters_receipt.rs` | retained `parameter_syncs` surface plus two-rank DDP receipt | `green` |
| `get_ddp_bucketed` | `distributed` | `crates/psionic-train/src/cs336_a2_ddp_bucketed_receipt.rs` | `build_cs336_a2_ddp_bucketed_receipt` plus retained bucketed DDP receipt | `green` |
| `ddp_bucketed_on_after_backward` | `distributed` | `crates/psionic-train/src/cs336_a2_ddp_bucketed_receipt.rs` | `build_after_backward_receipt` plus retained bucket completion receipt | `green` |
| `ddp_bucketed_on_train_batch_start` | `distributed` | `crates/psionic-train/src/cs336_a2_ddp_bucketed_receipt.rs` | `build_train_batch_start_receipt` plus retained reset receipt | `green` |
| `get_sharded_optimizer` | `optimizer` | `crates/psionic-train/src/cs336_a2_sharded_optimizer_receipt.rs` | `build_cs336_a2_sharded_optimizer_receipt` plus retained sharded-optimizer receipt | `green` |
