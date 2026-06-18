# Tassadar ALM Linked Dense Module

> Status: `implemented_early` for C5 module composition/linking, landed
> 2026-06-18 under OpenAgents issue #5325.

This document is the contract for
`crates/psionic-compiler/src/tassadar_alm_linked_dense_module.rs`.

## Identity

- module schema: `TassadarAlmLinkedDenseModule` v1
- module kind: `tassadar_alm_linked_dense_module.v1`
- fixture: `fixtures/tassadar/tassadar-linked-dense-module-v1.json`
- source banks:
  - `tassadar_corpus.mul_add_v1`
  - `tassadar_corpus.memory_roundtrip_v1`
- linked module digest:
  `cc1403674fc0d38892610d9e9c6c9230075494061f720c45bfa4f7b5a961756a`
- composed dense module digest:
  `2f3fa15120f0a078d4ede4e074e288fed24533ffa46f2d4b8aa4ca418c876602`
- composed replay trace digest:
  `0caa43ace27a5b86da14cfe037e65c30f250f0c0a0ac1c01f1fe3a3a45a230b2`

## What This Phase Adds

The linked dense fixture composes two digest-pinned dense ALM weight modules
into one block-separated dense module. The builder:

- builds source dense fixtures for `mul_add_v1` and `memory_roundtrip_v1`,
- presents each source bank as a computational module manifest,
- resolves the dependency graph through `tassadar_module_linker`,
- offsets residual slots and keyed-memory channels to avoid collisions,
- materializes one composed `TassadarAlmDenseWeightModule`,
- executes the composed module through the dense replay path,
- projects each bank's output columns from the composed trace, and
- requires every projected trace digest to equal the source bank trace digest.

The fixture carries the linker resolution, the composed dense module, source
bank digests, replay conformance cases, compile receipt refs, and public-safe
marketplace artifact refs. OpenAgents consumes the fixture as the first
digest-pinned compiled-weight-module listing.

## Reproducibility Bar

`build_tassadar_alm_linked_dense_program_fixture_v1` rebuilds the linked
fixture from the existing dense module builder. The ignored
`dump_linked_dense_module_fixture` test regenerates the committed JSON at
`/tmp/tassadar-linked-dense-module-v1.json`.

The conformance proof is exact replay over projections of the composed trace.
If a source bank uses a different input arity, step schedule, or projected rows
that do not digest to the source trace, the fixture builder refuses.

## What This Phase Does Not Do

No arbitrary module installation, no learned weights, no softmax runtime, no
serving route, no real settlement, and no purchase authority. The artifact is a
bounded, digest-pinned composition of two verified dense ALM banks inside the
existing numeric exactness window.
