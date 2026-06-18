# Tassadar W1.1 Wasm Window Ladder

> Status: implemented for C4/#5324 on 2026-06-18.

This document records the current W1.1 opcode-window ladder in
`crates/psionic-runtime/src/tassadar.rs` and the ALM interpreter in
`crates/psionic-compiler/src/tassadar_alm_wasm_interpreter.rs`.

## Profile

- profile id: `tassadar.wasm.core_i32_w1_1.v1`
- supported opcode count: 21
- opcode vocabulary: `i32.const`, `local.get`, `local.set`, `i32.add`,
  `i32.sub`, `i32.mul`, `i32.lt`, `i32.load`, `i32.store`, `br_if`,
  `output`, `return`, `nop`, `local.tee`, `drop`, `i32.eqz`, `i32.eq`,
  `i32.ne`, `i32.gt`, `i32.le`, `i32.ge`
- claim boundary: bounded i32-only programs accepted by the CPU reference
  runner; no arbitrary Wasm closure, no host imports, no f32/f64, no serving.

## Validation

The W1.1 program `tassadar_corpus.w1_1_window_v1` exercises the newly
added tier (`nop`, `local.tee`, `drop`, `i32.eqz`, `i32.eq`, `i32.ne`,
`i32.gt`, `i32.le`, `i32.ge`) and is compiled into the run-facing
numeric corpus. The corpus digest after adding that slot is:

`0d347bc3081acd2740761673f0b70d3e17a5ae467e9f865b5e6ef12009bfeb49`

Focused verification:

```bash
cargo test -p psionic-compiler tassadar_alm_wasm_interpreter -- --nocapture
cargo test -p psionic-compiler tassadar_alm_numeric::tests::program_corpus_fixture_is_pipeline_derived_and_deterministic -- --nocapture
```

The interpreter test compares the direct ALM evaluator, compiled row
executor, and specialized compiled executor against the CPU reference
runner for the W1.1 program.
