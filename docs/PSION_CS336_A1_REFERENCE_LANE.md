# Psion CS336 A1 Reference Lane

> Status: bounded full port complete, updated 2026-04-02 after landing the
> tokenizer train/runtime pair, the forward/model stack, the end-to-end
> training/checkpoint tranche, and the full A1 conformance harness.

This document records the owned `psionic` surfaces for the full Stanford CS336
Assignment 1 port program.

It exists so the repo can distinguish between:

- the older selective actual-lane baseline-tools bundle, and
- the newer bounded full-port reference lane work that now covers all of A1.

## What Landed So Far

The bounded full-port lane now owns:

- one real CS336-style byte-level BPE trainer
- one matching runtime tokenizer
- one bounded forward-only Transformer reference stack with state-dict paths
  aligned to the Stanford A1 module naming
- one bounded end-to-end CS336 A1 training loop with deterministic batching,
  softmax, cross-entropy, gradient clipping, AdamW, cosine schedule,
  checkpoint save/load, and exact resume proof on a tiny admitted corpus
- one typed conformance harness, one retained conformance report, and one
  checked-in completion matrix that map every Stanford A1 adapter family to an
  owned `psionic` proof surface

Primary landing surfaces:

- `crates/psionic-data/src/cs336_a1_bpe.rs`
- `crates/psionic-models/src/cs336_a1_tokenizer.rs`
- `crates/psionic-models/src/cs336_a1_reference_stack.rs`
- `crates/psionic-train/src/cs336_a1_reference_training.rs`
- `crates/psionic-train/src/cs336_a1_full_port_conformance.rs`
- `crates/psionic-train/examples/psion_cs336_a1_reference_training_bundle.rs`
- `crates/psionic-train/examples/psion_cs336_a1_full_port_conformance_report.rs`
- `docs/PSION_CS336_A1_FULL_PORT_MATRIX.md`
- `fixtures/training/cs336_a1_full_port_conformance_report_v1.json`
- `fixtures/training/cs336_a1_reference_tiny_training_bundle_v1.json`

That implementation now provides:

- deterministic byte-level BPE training from text or corpus paths
- exact lexicographic-greatest tiebreaking on raw bytes
- retained artifact bundles with stable digests, GPT-2 printable-byte vocab
  entries, and ordered merge lists
- runtime construction from raw vocab bytes, ordered merge pairs, and explicit
  special-token inventory
- longest-match special-token preservation during encoding
- UTF-8 round-trip decoding back to original text
- a bounded `encode_iterable` surface for streamed callers
- a bounded forward-only CS336 A1 reference stack covering linear, embedding,
  SiLU, SwiGLU, RoPE, scaled dot-product attention, multi-head self-attention,
  transformer blocks, and a full Transformer LM forward path
- a real `Module` tree and loadable state-dict layout matching Stanford A1
  paths such as `token_embeddings.weight`, `layers.0.attn.q_proj.weight`,
  `layers.0.ffn.w1.weight`, `ln_final.weight`, and `lm_head.weight`
- a bounded trainer that turns the tokenizer + tiny corpus into next-token
  batches, runs the A1 forward stack, computes cross-entropy loss, clips
  gradients globally, applies AdamW updates, writes checkpoints, reloads them,
  and proves resumed execution matches an uninterrupted four-step run
- committed retained fixtures for the tiny corpus, the step-2 checkpoint, the
  step-4 checkpoint, and the end-to-end training bundle
- a typed conformance report that refuses to go green unless every Stanford A1
  adapter family maps to an owned implementation surface plus an in-repo proof
  surface
- a checked-in markdown matrix that makes the bounded completion bar legible in
  docs instead of leaving “full port” as a vague claim

## Current Claim Boundary

This lane now honestly claims:

- `psionic` owns a real CS336 A1 tokenizer trainer instead of only tokenizer
  manifest bookkeeping
- `psionic` owns a real CS336 A1 tokenizer runtime that can consume the raw
  vocab and merge outputs produced by that trainer
- `psionic` owns a complete bounded forward-only CS336 A1 Transformer stack
  above existing `psionic` primitives
- `psionic` owns a bounded end-to-end CS336 A1 trainer that covers the
  Stanford A1 training-side helper surfaces in owned Rust and emits retained
  bundle/checkpoint artifacts
- `psionic` owns a full Stanford CS336 A1 adapter-coverage matrix with an
  in-repo retained conformance report and direct proof rows for every adapter
  family in the bounded reference lane
- tokenizer reproducibility for the A1 reference lane is now machine-legible
  and test-covered inside owned Rust code

It does not yet claim:

- scalable broader-pretraining backward support beyond the tiny reference lane,
  because the bounded trainer currently uses finite-difference reference math
  for gradients on the tiny model rather than a production autograd path
- that the bounded A1 lane is the same thing as the actual `Psion`
  pretraining operator lane

The packaged machine-runtime version of this work now lives separately in
`docs/PSION_CS336_A1_DEMO_LANE.md`. That demo lane reuses the same tiny corpus,
training config, and checkpoint family, but it exists to make one honest
`Pylon` and `Nexus` demo path possible. It should not be confused with either
the broader reference-lane completion claim here or the actual pretraining
lane.

The canonical completion matrix lives in
`docs/PSION_CS336_A1_FULL_PORT_MATRIX.md`, and the retained machine-readable
report lives in
`fixtures/training/cs336_a1_full_port_conformance_report_v1.json`.
