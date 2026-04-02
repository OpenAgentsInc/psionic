# Psionic Conformance Harness

This document describes the reusable Ollama-to-Psionic conformance harness added by
`PSI-117`.

The implementation lives in `crates/psionic-serve/src/conformance.rs` and stays
inside the reusable serve layer. It does not require app/UI participation.

## What It Covers

The harness normalizes and compares the subset of behavior the desktop depends
on:

- `tags` / installed-model discovery
- `show` / model detail facts
- `ps` / loaded-model state
- non-streaming `generate`
- streaming `generate`
- `embed`

Each check records one of four statuses:

- `passed`
- `failed`
- `unsupported`
- `intentional_difference`

`intentional_difference` is the escape hatch for cases where Psionic is explicitly
not at Ollama parity yet but the gap is known and documented. That keeps the
cutover artifact honest instead of forcing silent drift or fake parity.

## Subject Types

Current reusable subject implementations:

- `OllamaHttpSubject`
  - live HTTP adapter over `/api/tags`, `/api/show`, `/api/ps`,
    `/api/generate`, and `/api/embed`
  - uses Ollama `_debug_render_only` on non-streaming generate cases so prompt
    rendering can be compared without sampling noise
- `RecordedConformanceSubject`
  - in-memory subject for tests or for callers that already have comparable Psionic
    observations
  - suitable for current Psionic integration while some parity surfaces are still
    landing

## Fixture-Driven Cases

`GenerateConformanceCase::from_generate_compatible_prompt_fixture(...)` builds a
non-streaming prompt-render case from the real golden prompt corpus in
`psionic-models`.

Today that builder intentionally accepts only the subset of fixture cases that
can map honestly onto `/api/generate`:

- optional leading `system` or `developer`
- one `user` turn
- `add_generation_prompt = true`

That is enough to anchor single-turn families such as `phi3`, `qwen2`, the
first `qwen35` prompt-projection pilot, and the bounded `gemma4:e4b`
instruction-plus-user lane without pretending that multi-turn or raw
content-part multimodal chat-template parity is already solved.

Embeddings cases also carry an explicit `EmbeddingParityBudget` from
`psionic-runtime` so vector comparisons use the shared drift-budget policy instead
of one-off tolerance numbers.

## Report Shape

The harness emits a `ConformanceReport` JSON artifact with this top-level shape:

```json
{
  "suite_id": "qwen2-prompt-render",
  "baseline_subject": "ollama@http://127.0.0.1:11434",
  "candidate_subject": "psionic-candidate",
  "checks": [
    {
      "surface": "generate",
      "case_id": "qwen2.default_system",
      "status": "intentional_difference",
      "detail": "current Psionic prompt rendering is tracked separately in PSI-114; candidate marked unsupported: prompt rendering not yet implemented in Psionic",
      "baseline": { "...": "..." },
      "candidate": { "...": "..." }
    }
  ],
  "summary": {
    "passed": 0,
    "failed": 0,
    "unsupported": 0,
    "intentional_differences": 1
  }
}
```

`ConformanceReport::cutover_ready()` returns `true` only when there are no
`failed` or `unsupported` checks.

`PSI-138` adds a separate performance gate on top of that semantic gate:

```rust
let thresholds = CutoverPerformanceThresholds::default();
let performance = report.performance_gate(&thresholds);
let ready = report.cutover_ready_with_performance(&thresholds);
```

The default thresholds are ratio-based against the Ollama baseline for the
same case:

- generation total duration: candidate must stay within `1.25x`
- generation load duration: candidate must stay within `1.25x`
- generation prompt throughput: candidate must stay above `0.80x`
- generation decode throughput: candidate must stay above `0.80x`
- embeddings total duration: candidate must stay within `1.25x`
- embeddings load duration: candidate must stay within `1.25x`

If either side omits the required timing evidence for a compared `generate` or
`embed` case, the performance gate reports `insufficient_evidence` instead of
guessing. That is intentional: cutover should fail closed when performance
evidence is missing.

## Documented Run

Repeatable repo-owned harness coverage:

```bash
cargo test -p psionic-serve conformance --manifest-path Cargo.toml --no-default-features
cargo test -p psionic-serve gemma4 --manifest-path Cargo.toml --no-default-features
```

Those runs cover:

- real `qwen2`, `qwen35`, and `gemma4` prompt-fixture case construction from
  the golden corpus
- matching conformance-suite coverage for bounded prompt-render cases,
  including the real `gemma4:e4b` instruction-first fixture shape
- structured `intentional_difference` reporting for surfaces that still fail
  closed honestly
- live HTTP normalization of Ollama `tags` / `show` / `ps` / `generate(stream)`
  / `embed` semantics via a local test server
- bounded Gemma 4 server smoke and refusal coverage on the native CUDA lane

The real-artifact repeat lane for `gemma4:e4b` is:

```bash
PSIONIC_GEMMA4_PILOT_GGUF_PATH=/abs/path/to/gemma4-e4b-ollama.gguf \
  cargo test -p psionic-serve \
  gemma4_e4b_cuda_conformance_repeat_is_machine_checkable_when_available \
  --manifest-path Cargo.toml --no-default-features
```

That test skips cleanly when the pilot GGUF or a CUDA backend is unavailable,
and the same lane now guards the whole-model KV-cache geometry that a real
multi-layer `gemma4:e4b` artifact requires.

## Controlled Local Validation

For a local cutover check against a real Ollama daemon:

1. Start Ollama with the target model family installed.
2. Build a `ConformanceSuite` using real golden fixture cases plus any
   candidate-specific expected differences.
3. Run the suite with:

```rust
let mut baseline = OllamaHttpSubject::new("http://127.0.0.1:11434")?;
let mut candidate = RecordedConformanceSubject::new("psionic-candidate");
let report = run_conformance_suite(&suite, &mut baseline, &mut candidate)?;
write_conformance_report("target/psionic-conformance.json", &report)?;
```

When Psionic gains direct adapters for more surfaces, those adapters should
implement `ConformanceSubject` rather than inventing a second cutover harness.
