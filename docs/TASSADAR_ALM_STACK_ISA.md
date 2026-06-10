# Tassadar ALM Stack-ISA Universal Interpreter

> Status: `implemented_early` for executor-compiler phase E3 (bounded),
> landed 2026-06-10 under issue #1104. E1/E2/E5 are
> `docs/TASSADAR_ALM_GRAPH.md`, `docs/TASSADAR_ALM_BACKEND.md`, and
> `docs/TASSADAR_ALM_SPECIALIZER.md`.

This document is the contract for the bounded universal-interpreter lane
in `crates/psionic-compiler/src/tassadar_alm_stack_isa.rs`.

## Identity

- lane id: `tassadar.alm_stack_isa_interpreter.v1`
- claim boundary: the ALM stack-ISA interpreter executes a bounded
  straight-line stack instruction set (push, add, sub, mul, out, halt)
  with the program in a static seeded channel; it claims integer-exact
  reference semantics for encoded programs only and makes no branch,
  loop, call, Wasm, or served-route claim.

## What This Phase Proves

The complete program-to-weights pipeline shape — universal interpreter
versus specialized executor — now exists at integer-exact IR level:

1. **The universal interpreter** is one ALM gate graph. The program lives
   in a static keyed channel (opcode at key `2i`, operand at key
   `2i + 1`); the cursor is an accumulator; instruction fetch is two
   keyed reads; opcode decode is ReLU step indicators; the stack is a
   keyed channel addressed by a depth accumulator, with one masked
   unconditional write per step (operands read strictly-prior state;
   binary-op results overwrite the new top through next-step
   visibility); dynamic multiply uses
   `second·relu(top) − second·relu(−top)`.
2. **Specialization (E5)** on the program channel bakes the instruction
   table into step-function gate structure and deletes the channel — the
   interpreter becomes a dedicated executor for that program.
3. **Compilation (E2)** schedules either graph into a digest-pinned
   bundle.

The landed six-way agreement test runs one arithmetic program —
`(3 + 5) · 2 − 4` — through a plain Rust reference machine, the universal
graph under the E1 evaluator, the specialized graph under the evaluator,
and both graphs through E2 compiled execution: all five machine legs
match the reference row-for-row.

## Encoder Discipline

`tassadar_stack_isa_validate` statically checks stack discipline before
any graph exists: empty programs, stack underflow (binary ops below
depth 2, OUT below depth 1), and overflow past the declared maximum
depth are typed refusals with the offending instruction index. The
seeded stack floor covers keys `-1..=max_depth`, so in-discipline
programs never hit a missing-key refusal at runtime.

## Landed Tests

5 tests: reference parity for the arithmetic program, the six-way
agreement, masked-multiply correctness for negative and zero operands,
encoder refusals (empty/underflow/overflow), and a depth-3 expression
exercising the full declared stack.

## What This Phase Does Not Do

No branches, loops, calls, locals, or memory beyond the operand stack;
no Wasm decoding (the full `core_i32_v2` frontend remains the unbounded
E3); no tensor weights; no serving. The lane exists to prove the
pipeline shape and to be the first conformance workload for everything
downstream.
