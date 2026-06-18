# Tassadar ALM Dense Weight Module

> Status: `implemented_early` for W1.2 dense loadable materialization
> (bounded), landed 2026-06-18 under OpenAgents issue #5323.

This document is the contract for
`crates/psionic-compiler/src/tassadar_alm_dense_module.rs`.

## Identity

- module schema: `TassadarAlmDenseWeightModule` v1
- module kind: `tassadar_alm_dense_weight_module.v1`
- fixture: `fixtures/tassadar/tassadar-dense-weight-module-v1.json`
- source program: `tassadar_corpus.loop_sum_v1`
- dense module digest:
  `cfda0fe5dcf42e16db9e18696731427f0f30915fd3100d38da2dcc8411433e2c`
- replay trace digest:
  `2465d2c2af5077b4cf44c6eddbdc5aba2859029e30062f49a30e669acfc8e9d2`

## What This Phase Adds

The numeric model remains the sparse scalar-lane execution contract. The
dense module is a loadable, content-addressed representation of that
same ALM program as full-width matrices:

- residual wiring blocks as row-major dense `wResidual` matrices,
- attention blocks with explicit `wQ`, `wK`, `wV`, and `wO` matrices,
- FFN blocks with dense `wValue`, `wGate`, and `wOut` matrices,
- seed writes, output slots, and end-of-step write rows,
- source numeric model digest, graph digest, bundle digest, and public
  claim boundary.

ALM hard-max channel memory is not hidden in the matrices. Attention head
descriptors name keyed-read and accumulator channel semantics, while the
matrices carry the loadable residual projections. That keeps the artifact
honest: the module can replay the compiled ALM workload deterministically
without pretending to be a conventional softmax transformer.

## Reproducibility Bar

`build_tassadar_alm_dense_program_fixture_v1` rebuilds the fixture from
the existing psionic numeric program corpus. The builder materializes the
dense module, decodes it back into an executable numeric representation,
executes the same input steps, and requires the trace digest to match the
source fixture. The ignored `dump_dense_weight_module_fixture` test
regenerates the committed JSON at `/tmp/tassadar-dense-weight-module-v1.json`.

The decoded model may not have the exact same sparse JSON digest as the
source numeric model because duplicate sparse wiring terms collapse into
one dense slot coefficient. The module therefore stores the source model
digest as provenance and uses exact trace replay as the semantic proof.

## What This Phase Does Not Do

No training, no learned weights, no softmax, no temperature, no serving
route, no performance claim, and no live settlement authority. The
artifact is a bounded, digest-pinned, loadable replay module for compiled
ALM workloads inside the existing numeric exactness window.
