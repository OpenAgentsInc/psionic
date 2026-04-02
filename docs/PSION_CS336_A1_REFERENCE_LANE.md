# Psion CS336 A1 Reference Lane

> Status: partial full-port program, updated 2026-04-02 after landing the
> bounded CS336 A1 tokenizer train/runtime pair in `psionic-data` and
> `psionic-models`.

This document records the owned `psionic` surfaces for the full Stanford CS336
Assignment 1 port program.

It exists so the repo can distinguish between:

- the older selective actual-lane baseline-tools bundle, and
- the newer bounded full-port reference lane work needed to cover all of A1.

## What Landed So Far

The first full-port tranches now own:

- one real CS336-style byte-level BPE trainer
- one matching runtime tokenizer
- one bounded forward-only Transformer reference stack with state-dict paths
  aligned to the Stanford A1 module naming

Primary landing surfaces:

- `crates/psionic-data/src/cs336_a1_bpe.rs`
- `crates/psionic-models/src/cs336_a1_tokenizer.rs`
- `crates/psionic-models/src/cs336_a1_reference_stack.rs`

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

## Current Claim Boundary

This lane now honestly claims:

- `psionic` owns a real CS336 A1 tokenizer trainer instead of only tokenizer
  manifest bookkeeping
- `psionic` owns a real CS336 A1 tokenizer runtime that can consume the raw
  vocab and merge outputs produced by that trainer
- `psionic` owns a complete bounded forward-only CS336 A1 Transformer stack
  above existing `psionic` primitives
- tokenizer reproducibility for the A1 reference lane is now machine-legible
  and test-covered inside owned Rust code

It does not yet claim:

- full Assignment 1 end-to-end trainer, optimizer, scheduler, batching, and
  checkpoint parity
- full Assignment 1 conformance proof coverage across every Stanford adapter
  surface

Those remaining surfaces stay in the follow-on issues for the bounded full-port
program.
