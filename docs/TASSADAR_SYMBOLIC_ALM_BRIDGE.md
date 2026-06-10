# Tassadar Symbolic-To-ALM Bridge

> Status: `implemented_early`, landed 2026-06-10 under issue #1105. The ALM
> stack is E1–E3/E5 (`docs/TASSADAR_ALM_GRAPH.md`,
> `docs/TASSADAR_ALM_BACKEND.md`, `docs/TASSADAR_ALM_SPECIALIZER.md`,
> `docs/TASSADAR_ALM_STACK_ISA.md`); the bounded symbolic lane is
> `psionic-ir::tassadar_symbolic`.

This document is the contract for the bridge in
`crates/psionic-compiler/src/tassadar_symbolic_alm_bridge.rs`, the first
integration between the new executor-compiler stack and the existing
Tassadar bounded symbolic lane.

## Identity

- bridge family: `tassadar_symbolic_alm_bridge`
- bridge version: `v1`
- claim boundary: the symbolic-to-ALM bridge lowers the bounded
  straight-line symbolic IR into channel-free single-step ALM graphs; the
  symbolic evaluator saturates in i32 while the ALM is checked-exact i64,
  so parity is claimed only for executions that neither saturate in i32
  nor overflow i64; no control flow, Wasm, or served-route claim is
  created.

## Lowering Rules

`compile_tassadar_symbolic_to_alm` performs SSA-style lowering of one
`TassadarSymbolicProgram`:

- Declared inputs become ALM `Input` fields in declaration order and seed
  both the name map and their bound memory slots.
- `initial_memory` cells become constants on their slots.
- Mutable memory is compile-time SSA: a slot→value map that `Store`
  rebinds and `MemorySlot` operands read; unseeded slots read as a zero
  constant (matching the symbolic zero-initialized memory image).
- `Let` bindings extend the name map; reusing a name rebinds it.
- `Add`/`Sub` lower to `Linear`; `Mul` lowers to the exact two-ReGLU
  product; `Lt` lowers to the ReLU step indicator
  `1[right − left >= 1]`.
- `Output` statements collect as graph outputs (a program with no outputs
  refuses).

The result is a channel-free, seed-free, single-step ALM graph: the whole
straight-line program is one token step.

## Parity Bar (Landed)

For **every** committed example in
`psionic_ir::tassadar_symbolic_program_examples()`, three legs agree with
the example's expected outputs: the symbolic evaluator, the bridged graph
under the E1 evaluator, and the E2-compiled bundle execution. Typed
refusals cover unknown names, out-of-range slots, and output-free
programs; bridge reports are digest-stable.

## Semantics Boundary

The symbolic evaluator uses saturating i32 arithmetic
(`saturating_add/sub/mul`); the ALM uses checked-exact i64 with overflow
refusals. The behaviors agree exactly on executions whose intermediate
values stay within i32 without saturating — which all committed examples
satisfy. Programs relying on saturation are outside the bridge's claim;
the bridge does not attempt to reproduce saturation.

## Why This Bridge Matters

The bounded symbolic lane already has committed example programs, an
artifact path, and runtime lowering. The bridge makes those same programs
first-class executor-compiler workloads: every symbolic example is now
also an ALM conformance case, and anything expressible in the symbolic
subset can be scheduled (E2) or — when its data is static — specialized
(E5) without new authoring.
