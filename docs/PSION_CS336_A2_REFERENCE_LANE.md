# Psion CS336 A2 Reference Lane

> Status: bounded full-port tranche in progress, updated 2026-04-02 after
> landing the profiling and benchmark baseline receipts.

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

## Current Claim Boundary

This lane now honestly claims:

- `psionic` owns a bounded A2 profiling bundle instead of only prose about
  future systems work
- the bounded A2 lane is anchored to the existing A1 tiny reference lane rather
  than to a detached synthetic benchmark toy
- later A2 tranches now have one retained receipt family to plug into

It does not yet claim:

- a bounded FlashAttention2 reference path
- a fused backend FlashAttention2 path
- bounded DDP or sharded-optimizer execution
- full Stanford CS336 A2 parity
- admitted actual-lane throughput or distributed-cluster qualification
