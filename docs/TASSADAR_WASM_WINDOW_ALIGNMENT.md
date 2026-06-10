# Tassadar Wasm-Window Alignment: core_i32_v2 vs transformer-vm

> Status: alignment audit, landed 2026-06-10 under issue #1108. Doc only;
> no capability claim changes. Sources: `psionic-runtime/src/tassadar.rs`
> (the `TassadarOpcode` set and `TassadarWasmProfile` registry) and the
> read-only reference clone `projects/repos/transformer-vm`
> (`transformer_vm/wasm/interpreter.py` OPCODES, `compilation/lower.py`).

## Why This Doc Exists

The eventual full E3 frontend (Wasm → ALM) and the planned
cross-validation harness (running transformer-vm's example programs as
external conformance cases) both need a precise opcode-level convergence
map between psionic's Wasm semantic windows and Percepta's compiled
core. This is that map.

## The Two Windows

**psionic `core_i32_v2`** (also `sudoku_*`/`hungarian_*` profiles): the
11-opcode `TassadarOpcode::ALL` set. The article profile
(`article_i32_compute_v1`) adds `i32.lt` for 12. Explicit bounds per
profile (core_i32_v2: 8 locals, 16 memory slots, 128 program length,
512 steps), `br_if`-nonzero branch mode, host-level `output`, typed
refusals outside the window.

**transformer-vm core**: 36 dispatch opcodes (their post says 35;
`input_base` is a pseudo-op), plus a lowering pass that expands harder
ops into the core. No declared execution bounds; memory grows with the
token stream.

## Opcode Matrix

| Opcode | psionic core_i32_v2 | transformer-vm | Notes |
| --- | --- | --- | --- |
| `i32.const` | native | native | — |
| `local.get` / `local.set` | native | native | tvm adds `local.tee` |
| `local.tee` | absent | native | trivial lowering: `tee = set + get` |
| `global.get` / `global.set` | absent | native | psionic has no globals; lowerable onto reserved memory slots |
| `i32.load` / `i32.store` | native (slot-addressed) | native (byte-addressed) | **addressing-model divergence**, see below |
| `i32.load8_s/u`, `i32.load16_s/u` | absent | native | sub-word memory; needed for tvm's byte-wise bitop lowering |
| `i32.store8` / `i32.store16` | absent | native | same |
| `i32.add` / `i32.sub` | native | native | — |
| `i32.mul` | **native** | **lowered** (shift-add / byte expansion) | psionic is *wider* here |
| `i32.div_u/s`, `i32.rem_u` | absent | lowered (const divisor) | tvm lowers only when the divisor is a preceding const |
| `i32.and/or/xor` | absent | lowered (byte-wise via sub-word memory) | depends on load8/store8 |
| `i32.shl`, `i32.shr_u` | absent | lowered (const shift) | — |
| `i32.rotl/rotr`, `clz`, `ctz`, `popcnt`, `extend8_s/16_s` | absent | lowered | — |
| `i32.eqz`, `i32.eq`, `i32.ne` | absent | native | psionic expresses eq via lt compositions only in authored programs |
| `i32.lt_s` | `i32.lt` in the article profile | native | **psionic's `i32.lt` is one comparison; tvm carries all ten signed/unsigned comparisons natively** |
| `i32.lt_u`, `gt_s/u`, `le_s/u`, `ge_s/u` | absent | native | — |
| `br` | absent | native | psionic is `br_if`-only |
| `br_if` | native (nonzero mode) | native | semantics agree (branch on nonzero) |
| `call` / `return` | `return` only | both | psionic has no calls; tvm tracks call depth via cumsum |
| `select`, `drop` | absent | native | trivial ALM lowerings |
| `output` | native (host opcode) | native (host opcode) | both are host-level emission pseudo-ops |
| `input_base` | absent | pseudo-op | tvm's input-binding convention |
| `halt` | via `return` | native | naming difference only |

## Structural Divergences That Matter More Than Opcodes

1. **Memory addressing.** psionic's window is slot-addressed (bounded
   `max_memory_slots`); transformer-vm is byte-addressed Wasm MVP memory
   with sub-word access. A converged frontend must either widen psionic
   to byte addressing or define the slot window as a checked sub-profile
   with byte ops refused. Recommendation: keep the slot profile as the
   bounded tier and add a byte-addressed profile as a separate window —
   never silently reinterpret.
2. **Signedness coverage.** psionic exposes one comparison (`i32.lt`,
   signed); tvm exposes all ten. Cross-validation on tvm examples will
   hit unsigned comparisons immediately. The ALM lowering for unsigned
   comparisons over i64-held i32 values is cheap (bias by 2^31 then
   signed compare); the window decision is whether to admit them as
   opcodes or as authored lowerings.
3. **Bounds and refusal posture.** psionic profiles carry explicit
   program/step/local/slot bounds and typed refusals; tvm declares none
   and grows with the trace. For homework dispatch this is a psionic
   advantage to keep, not a gap to close: any converged profile should
   stay bounds-declared.
4. **Lowering philosophy.** tvm lowers `mul/div/shift/bitops` into a
   smaller trusted core, partly only when operands are constants;
   psionic carries `i32.mul` natively (the ALM two-ReGLU product is
   exact for dynamic operands — see `tassadar_alm_stack_isa`). Neither
   subset contains the other: psionic is wider on dynamic multiply,
   tvm is wider on everything bitwise.
5. **Calls and control.** tvm has `call`/`return`/`br`; psionic has
   `br_if`/`return` only. Calls are the largest functional gap for
   running tvm's C-derived examples (`fibonacci.c` uses calls via
   sscanf/printf shims; `collatz.c` is loop-only).

## Convergence Recommendations For The Full E3 Frontend

1. Define a new profile id (e.g. `core_i32_v3`) as the explicit
   intersection-plus: current `core_i32_v2` ∪ {`local.tee`, `drop`,
   `select`, `i32.eqz/eq/ne`, the remaining comparisons with an explicit
   signedness table} — all cheap ALM lowerings, no new memory model.
2. Treat byte-addressed memory and sub-word access as a separate profile
   decision with its own bounds, not an increment.
3. Adopt tvm's example set as external conformance cases in two tiers:
   tier 1 (`addition.c`, `collatz.c` without calls) against the v3
   profile; tier 2 (the full set) gated on calls + byte memory.
4. Keep dynamic `i32.mul` native in any converged profile; do not adopt
   tvm's const-only lowering restriction.
5. Every widening lands as a profile version bump with refusal-surface
   updates — the existing frozen-core-Wasm discipline
   (`tassadar_frozen_core_wasm`) already models exactly this.

## Boundary

This audit changes no claims. The windows are different by design today;
the doc exists so that when the full E3 frontend lands, its profile
decisions are deliberate, versioned, and cross-validatable instead of
incidental.
