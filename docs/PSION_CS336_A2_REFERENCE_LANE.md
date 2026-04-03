# Psion CS336 A2 Reference Lane

> Status: bounded full-port tranche in progress, updated 2026-04-02 after
> landing the profiling baseline receipts and the owned FlashAttention2
> reference path, plus the bounded fused CUDA receipt family.

This document records the owned `psionic` surfaces for the bounded Stanford
CS336 Assignment 2 port program.

It exists so the repo can distinguish between:

- the older actual-lane systems bundle that ports A2 ideas into the real Psion
  operator lane, and
- the newer bounded full-port reference lane work that aims to cover all of A2
  directly inside `psionic`.

## What Has Landed So Far

The first bounded A2 tranche now owns:

- one deterministic profiling and benchmark bundle for the bounded A2 lane
- one naive-attention baseline receipt tied to owned A1-style attention code
- one tiny A1-backed training-step baseline receipt with real loss and optimizer
  state digests
- one analytical distributed-step baseline receipt that fixes the pre-DDP
  communication and optimizer-state comparison surface for later A2 work

Primary landing surfaces:

- `crates/psionic-train/src/cs336_a2_profiling.rs`
- `crates/psionic-train/examples/psion_cs336_a2_baseline_profile_bundle.rs`
- `fixtures/training/cs336_a2_baseline_profile_bundle_v1.json`

That baseline tranche now provides:

- deterministic naive-attention output digests for a bounded causal attention
  shape
- deterministic training-step receipts over the existing A1 tiny reference lane
- machine-legible parameter, optimizer-state, communication, and bucket-count
  baselines for the later FlashAttention, DDP, and sharded-optimizer tranches
- a clean claim boundary that says these receipts are bounded analytical
  baselines, not admitted actual-lane throughput claims

The second bounded A2 tranche now owns:

- one tiled owned FlashAttention2-style reference forward path
- one tiled owned FlashAttention2-style reference backward path
- one retained comparison receipt against the bounded naive baseline for output,
  logsumexp, gradient, and memory-surface agreement

Primary landing surfaces:

- `crates/psionic-models/src/cs336_a2_flashattention_reference.rs`
- `crates/psionic-train/src/cs336_a2_flashattention_reference_receipt.rs`
- `crates/psionic-train/examples/psion_cs336_a2_flashattention_reference_receipt.rs`
- `fixtures/training/cs336_a2_flashattention_reference_receipt_v1.json`

That FlashAttention tranche now provides:

- one bounded owned tiled attention path that does not just alias the naive
  probability-trace implementation
- one bounded recompute-style backward pass that matches the naive baseline
  within the declared tolerance
- one machine-readable memory comparison showing the smaller tiled score and
  probability surfaces relative to the naive full matrix surface
- one retained proof bundle that stays attached to the shared bounded A2
  receipt family

The third bounded A2 tranche now owns:

- one explicit CUDA capability and refusal surface for the bounded fused path
- one retained fused CUDA receipt family that compares the backend path against
  the owned tiled reference path
- one bounded benchmark family that records naive CPU, tiled CPU, and fused
  CUDA forward timings in the same A2 lane when the admitted CUDA path exists

Primary landing surfaces:

- `crates/psionic-backend-cuda/src/lib.rs`
- `crates/psionic-train/src/cs336_a2_flashattention_fused_cuda_receipt.rs`
- `crates/psionic-train/examples/psion_cs336_a2_flashattention_fused_cuda_receipt.rs`
- `fixtures/training/cs336_a2_flashattention_fused_cuda_receipt_v1.json`

That fused tranche now provides:

- one honest bounded backend-accelerated attention path tied to owned CUDA code
  instead of prose about future acceleration
- one explicit refusal receipt on hosts where the admitted CUDA path does not
  exist
- one retained correctness bundle that checks fused output and backward
  gradients against the owned tiled reference path within the declared
  tolerance on admitted hardware
- one shared benchmark surface that records naive, tiled, and fused forward
  timings without pretending that this tiny lane is a production throughput
  claim

The current checked-in fused receipt was generated on a non-CUDA host, so the
retained artifact records the explicit refusal path rather than a successful
backend execution. The same receipt family records correctness and fused timing
data when regenerated on admitted CUDA hardware.

The fourth bounded A2 tranche now owns:

- one bounded two-rank individual-parameter DDP synchronization receipt above
  the owned A1 tiny trainer
- one retained proof that rank-0 broadcast, per-parameter gradient averaging,
  and bounded update application stay aligned with a non-parallel baseline
- one explicit statement that the collective path is host-owned reference
  emulation, not transport-backed distributed execution

Primary landing surfaces:

- `crates/psionic-train/src/cs336_a2_ddp_individual_parameters_receipt.rs`
- `crates/psionic-train/examples/psion_cs336_a2_ddp_individual_parameters_receipt.rs`
- `fixtures/training/cs336_a2_ddp_individual_parameters_receipt_v1.json`

That individual-parameter DDP tranche now provides:

- one owned bounded DDP path that synchronizes each trainable parameter tensor
  through a host-owned reference averaging path inside `psionic-train`
- one bounded update path pinned to the same global finite-difference gradient
  surface as the non-parallel reference trainer so retained parity stays
  deterministic
- one retained two-rank proof bundle showing broadcast closure and synchronized
  updates matching the non-parallel baseline
- one honest boundary note that the current path uses host-owned reference
  collective emulation and does not claim backend transport execution

The fifth bounded A2 tranche now owns:

- one bounded bucketed DDP synchronization receipt above the owned A1 tiny
  trainer
- one explicit start-of-step reset surface and one after-backward bucket
  completion surface
- one retained proof that the bucketed coordination lane stays aligned with the
  non-parallel baseline while still recording bucket-plan variation

Primary landing surfaces:

- `crates/psionic-train/src/cs336_a2_ddp_bucketed_receipt.rs`
- `crates/psionic-train/examples/psion_cs336_a2_ddp_bucketed_receipt.rs`
- `fixtures/training/cs336_a2_ddp_bucketed_receipt_v1.json`

That bucketed DDP tranche now provides:

- one owned bounded bucket-planning surface with single-bucket, profile-bucket,
  and small-bucket retained cases
- one explicit train-batch-start receipt that resets the pending bucket set for
  each step
- one explicit after-backward receipt that records deterministic reverse-order
  bucket completion and bucket gradient digests
- one bounded update path pinned to the same global finite-difference gradient
  surface as the non-parallel reference trainer so retained parity stays
  deterministic
- one honest boundary note that the current path records bucket coordination
  truth but does not claim asynchronous transport overlap or backend collectives

## Current Claim Boundary

This lane now honestly claims:

- `psionic` owns a bounded A2 profiling bundle instead of only prose about
  future systems work
- `psionic` owns a bounded FlashAttention2-style reference implementation with
  retained parity evidence against the naive baseline
- `psionic` owns a bounded fused CUDA attention receipt family with explicit
  refusal posture on non-CUDA hosts
- `psionic` owns a bounded individual-parameter DDP proof lane with retained
  two-rank synchronization evidence against the non-parallel baseline
- `psionic` owns a bounded bucketed DDP proof lane with retained bucket
  planning, start-of-step, and after-backward coordination evidence
- the bounded A2 lane is anchored to the existing A1 tiny reference lane rather
  than to a detached synthetic benchmark toy
- later A2 tranches now have one retained receipt family to plug into

It does not yet claim:

- sharded-optimizer execution
- full Stanford CS336 A2 parity
- admitted actual-lane throughput or distributed-cluster qualification
