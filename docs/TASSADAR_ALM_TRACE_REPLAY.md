# Tassadar ALM Exact Trace-Replay Verification

> Status: `implemented_early`, landed 2026-06-10 under issue #1106.

This document is the contract for the exact-replay verification class in
`crates/psionic-compiler/src/tassadar_alm_trace_replay.rs`.

## Identity

- verification class id: `exact_trace_replay.alm_compiled.v1`
- claim boundary: exact trace replay verifies compiled-ALM-bundle
  executions only, by deterministic re-execution and bitwise comparison;
  it grants no serving, payment, or settlement authority and makes no
  claim about non-ALM trace formats.

## The Two Verdict Paths

- **`tassadar_alm_verify_full_replay`**: given the bundle, the inputs,
  and a claim (`bundle_digest`, `trace_digest`), check the bundle digest
  first (a forged bundle rejects before any replay), re-execute, and
  compare trace digests. Outcome is `verified` or `rejected` with a typed
  reason: `bundle_digest_mismatch`, `trace_digest_mismatch` (carrying the
  actual digest), or `execution_refused` (typed execution errors carried
  through).
- **`tassadar_alm_verify_window`**: given claimed output rows for one
  sampled step window, replay and diff the window row-for-row, rejecting
  with the **exact first mismatching step index**. Empty or out-of-range
  windows are typed request errors, not verdicts.

Both produce a deterministic, digest-stable `TassadarAlmReplayVerdict`
(class id, outcome, replayed/compared step counts, bundle digest)
suitable for embedding in receipts.

## Cost Note

ALM execution is sequential, so window checks replay from step zero.
Replay therefore costs the same as the original work — which, for this
work class, is still the cheapest verification grade the network has:
no probabilistic machinery, no graders, no quorum needed for honest
workloads. Checkpointed mid-trace state for cheaper window entry is a
later optimization, not a v1 requirement.

## Relation To The Homework Economy

This is the reference implementation of the `exact_trace_replay`
verification class that the OpenAgents homework epic registers in its
pluggable verification queue (openagents #4674) and that executor-trace
homework binds to (openagents #4684). The worker-side TypeScript class
mirrors these verdict semantics; rejection reasons map onto the queue's
typed failure codes. Verdicts inform acceptance; they never replace
acceptance authority or move money.

## Landed Tests

7 tests: honest full-replay verification, tampered trace-digest
rejection, forged bundle-digest rejection before replay, honest window
verification plus tampered-row rejection naming the exact step,
invalid-window refusals, execution-refusal propagation, and verdict
digest stability.
