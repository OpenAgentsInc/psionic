//! W3 student program (openagents#4749): the four-baseline sweep over
//! verified Tassadar trace corpora, evaluated by first divergence behind
//! replay — never perplexity.
//!
//! Psion lane naming rule (PSION_EXECUTOR_PROGRAM.md): every model here
//! is a *student* producing bounded statistics that the shipped replay
//! verifier checks; only the frozen analytic executor (baseline d's
//! core) carries Tassadar-grade exactness, and it is not trained.
//!
//! Modules:
//! * [`prep`] — `student_prep.v0.1` reader and the student token
//!   protocol over verified trace records;
//! * [`tensor`] — deterministic f32 math (gemm, AdamW, splitmix);
//! * [`model`] — the shared transformer backbone with hand-derived
//!   backprop and incremental decode;
//! * [`train`] — baselines (a)/(b)/(c) training with receipts;
//! * [`interface`] — baseline (d): learned marshaling around the frozen
//!   psionic numeric executor;
//! * [`evalrun`] — first-divergence evaluation with replay acceptance.

pub mod evalrun;
pub mod interface;
pub mod model;
pub mod prep;
pub mod tensor;
pub mod train;
