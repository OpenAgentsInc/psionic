# Tassadar ALM Window Interpreter (Branch-Capable)

> Status: `implemented_early` for executor-compiler phase E3 (full,
> bounded), landed 2026-06-10 under issue #1112. The straight-line
> stack-ISA predecessor is `docs/TASSADAR_ALM_STACK_ISA.md`; the
> window-convergence map is `docs/TASSADAR_WASM_WINDOW_ALIGNMENT.md`.

This document is the contract for the branch-capable interpreter in
`crates/psionic-compiler/src/tassadar_alm_wasm_interpreter.rs`.

## Identity

- lane id: `tassadar.alm_wasm_interpreter.v1`
- claim boundary: the ALM window interpreter executes the bounded
  twelve-opcode Tassadar i32 window (const, local get/set, add, sub,
  mul, lt, load, store, br_if, output, return) under a fixed step budget
  with the program in a static specializable channel; parity is claimed
  only for programs the CPU reference runner accepts, because the gate
  graph yields seeded zeros where the runner refuses malformed stack
  discipline; integer-exact, no f32, no serving.

## What This Phase Proves

The executor-compiler now interprets the runtime's **actual program
format**: a real `TassadarProgram` converts directly into the
interpreter's static program channel and runs as one ALM gate graph —
branches included — with outputs cross-validated against the production
`TassadarCpuReferenceRunner`.

The mechanism, entirely inside the five ALM primitives:

- **All machine state in keyed channels** read at strictly-prior-step
  visibility: pc/depth/halted in a state channel under fixed keys, the
  operand stack keyed by depth, locals and memory keyed by index, the
  program static at keys `2·pc` / `2·pc + 1`.
- **Branching without control flow**: the new pc is
  `pc + not_halted · (1 + taken · (target − pc − 1))`, where `taken` is
  `is_br_if · 1[condition ≠ 0]` and every product runs through the
  two-ReGLU identity. Backward branches are just negative displacements.
- **Halting as a sticky bit**: `halted OR is_return`, with every effect
  masked by `not_halted` — frozen pc, zero stack delta, sink-keyed
  writes, suppressed outputs. Virtual `Return` padding at `pc = len` and
  `len + 1` makes fall-off-the-end halt identically to the runner's
  `FellOffEnd`.
- **Masked effects**: locals/memory reads use gated keys (key 0, always
  seeded, when inactive); stack/local/memory writes route to a sink key
  when their opcode is not active, so the single unconditional
  `ChannelWrite` per channel per step never disturbs live state.

## Parity Bars (Landed, All Against The Production Runner)

7 tests: straight-line arithmetic; a **backward-branch loop**
(`acc += i` for `i in 1..6` via locals, `i32.lt`, and `br_if 4`); a
forward conditional skip exercised on both arms; a memory roundtrip with
`initial_memory`; fall-off-the-end halting; converter refusals
(branch/local/slot out of range); and the full-pipeline test —
`TassadarProgram` → ALM graph → E2 compiled execution **and**
E5-specialized (program channel baked into step-function gate structure)
→ compiled execution, all equal to `TassadarCpuReferenceRunner` outputs.

That last bar is the one the lane has been building toward: a real
Tassadar runtime program, baked into pure static gate structure with no
program channel at all, reproducing the production runner's outputs.

## What This Phase Does Not Do

Fixed step budgets only (the caller chooses; the halt flag reports
completion). No calls, globals, sub-word memory, or unsigned comparisons
(the `core_i32_v3` convergence plan in the alignment audit governs
widenings). Programs the runner would refuse (stack underflow) execute
on seeded zeros here — the runner stays the arbiter of well-formedness.
No f32 weights, no serving, no public capability copy.
