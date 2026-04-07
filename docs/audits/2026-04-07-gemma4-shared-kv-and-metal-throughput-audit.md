# Gemma 4 Shared-KV Fix And Metal Throughput Audit

> Status: audit, 2026-04-07
>
> Scope: document the Gemma 4 correctness regression in current `psionic-serve`,
> the fix landed in this change, the local Metal hot-path waste removed in the
> same pass, and the remaining throughput gap against local Ollama on the same
> host and prompt.

## Why This Audit Exists

`Gemma 4` on current `main` had two separate problems:

1. A real correctness regression.
2. A real performance gap against local Ollama even after Gemma execution was
   working again.

This pass fixed the correctness regression, reduced one concrete source of
decode overhead in the native Metal local path, and added better benchmarking
surface area for future work. It did not close the remaining Metal throughput
gap.

## What Was Broken

The regression was in shared-KV handling for Gemma 4 tail layers.

The current loader logic used tensor presence to decide whether a layer owned KV
or reused KV. The public Ollama GGUF advertises
`gemma4.attention.shared_kv_layers`, and the tail-layer tensors do not map
cleanly onto the old "tensor exists means this layer owns KV" assumption.

That meant Psionic built the wrong reuse mapping for the tail of the layer
stack. The runtime still executed, but output quality regressed badly.

## Fixes Landed In This Change

### 1. Shared-KV mapping now prefers Gemma metadata over raw tensor presence

Files:

- `crates/psionic-serve/src/gguf.rs`

What changed:

- added `gemma4_declared_shared_kv_layers(...)`
- changed `gemma4_layer_has_kv(...)` to accept the decoder descriptor and layer
  index
- added `gemma4_reused_kv_layer_index_from_metadata(...)`
- updated CPU, Metal, and CUDA Gemma 4 model loaders to use the metadata-driven
  mapping when `gemma4.attention.shared_kv_layers` is present

Why it matters:

- tail layers now reuse the intended earlier KV source instead of fabricating an
  incorrect ownership mapping from tensor presence
- the local Metal and CUDA Gemma 4 lanes return coherent output again

### 2. The local Metal decode path no longer allocates unused host KV buffers

Files:

- `crates/psionic-serve/src/gguf.rs`
- `crates/psionic-serve/src/lib.rs`

What changed:

- `InMemoryKvCache::append_token_without_kv(...)` now appends empty host KV
  vectors instead of allocating zero-filled `width`-sized host buffers
- `MetalGemma4ModelInner::forward_local_step_with_layer_caches(...)` now
  materializes host KV buffers only when they are actually needed:
  staged execution, forwarded KV, or explicit host materialization
- the Metal fast path now returns empty key/value vectors when the local decode
  step kept KV fully device-side

Why it matters:

- the local single-node Metal decode path no longer pays a full-width host KV
  allocation tax per generated token when those host vectors are never used

### 3. The benchmark harness is more useful now

File:

- `crates/psionic-serve/examples/gemma4_bench.rs`

What changed:

- added `--backend cpu`
- added `--prompt-mode rendered-tokens|text`
- recorded `output_token_ids` in the JSON report

Why it matters:

- future Gemma comparisons no longer need ad hoc shell glue to compare token
  trajectories or to force text-path versus token-path runs

## Validation

Test run:

```bash
cargo test -p psionic-serve gemma4_shared_kv_metadata_overrides_tensor_presence_for_tail_layers -- --nocapture
```

Result:

- passed

New test:

- `gguf::tests::gemma4_shared_kv_metadata_overrides_tensor_presence_for_tail_layers`

That test writes a small synthetic Gemma 4 GGUF with
`gemma4.attention.shared_kv_layers = 2` and proves that the tail layers are
treated as reused-KV layers even when key tensors are present in the layout.

## Benchmark Receipts

### Historical clean reference before the regression

Command:

```bash
cargo run --release -p psionic-serve --example gemma4_bench -- \
  --model-path /Users/christopherdavid/models/gemma4/gemma4-e4b-ollama.gguf \
  --backend metal \
  --max-output-tokens 8 \
  --repeats 1 \
  --json-out /tmp/psionic-gemma-0404.json
```

Receipt:

- checkout: historical clean commit `04cbd840`
- `8` output tokens
- total wall clock `0.352841458 s`
- output was coherent English in the interactive receipt from that run

### Broken current-main behavior before this fix

Exploratory receipt:

- file: `/tmp/psionic-gemma-main.json`
- the same Gemma 4 benchmark path produced incoherent output in the interactive
  run from that checkpoint
- the report-side decode split was clearly invalid for that broken run
  (`decode_s` near zero while total wall clock stayed large), so that receipt is
  only useful as evidence that the regression existed, not as a trustworthy
  throughput number

### Exploratory post-fix receipt

Command:

```bash
cargo run --release -p psionic-serve --example gemma4_bench -- \
  --model-path /Users/christopherdavid/models/gemma4/gemma4-e4b-ollama.gguf \
  --backend metal \
  --max-output-tokens 32 \
  --repeats 3 \
  --json-out /tmp/psionic-gemma-fixed-32x3.json
```

Receipt:

- load time `1.746888208 s`
- output tokens per run `26`
- warm decode times:
  - `0.871468625 s`
  - `0.932283416 s`
- warm decode throughput:
  - `29.83 tok/s`
  - `27.89 tok/s`

Meaning:

- the shared-KV fix restored coherent output
- Psionic was still materially slower than Ollama

### Clean release receipt from the landed code

Command:

```bash
cargo run --release -q -p psionic-serve --example gemma4_bench -- \
  --model-path /Users/christopherdavid/models/gemma4/gemma4-e4b-ollama.gguf \
  --backend metal \
  --max-output-tokens 32 \
  --repeats 3 \
  --json-out /tmp/psionic-gemma-clean-fixed-32x3.json
```

Receipt:

- file: `/tmp/psionic-gemma-clean-fixed-32x3.json`
- load time `1.750375417 s`
- output tokens per run `26`
- warm decode times:
  - `0.842737875 s`
  - `0.882629833 s`
- warm decode throughput:
  - `30.85 tok/s`
  - `29.46 tok/s`
- mean reported decode throughput across all three runs: `30.35 tok/s`

Meaning:

- the clean shipping code reproduces the exploratory post-fix throughput band
- the correctness fix is real
- the remaining throughput gap is still very large

### Exploratory post-fix receipt after removing host KV placeholder allocation

Command:

```bash
cargo run --release -p psionic-serve --example gemma4_bench -- \
  --model-path /Users/christopherdavid/models/gemma4/gemma4-e4b-ollama.gguf \
  --backend metal \
  --max-output-tokens 32 \
  --repeats 3 \
  --json-out /tmp/psionic-gemma-fastpath-opt-32x3.json
```

Receipt:

- load time `1.754935375 s`
- output tokens per run `26`
- warm decode times:
  - `0.905126916 s`
  - `0.857086375 s`
- warm decode throughput:
  - `28.73 tok/s`
  - `30.34 tok/s`

Meaning:

- the local hot-path allocation cut helped prompt-side cost and one warm run
- it did not close the decode gap in any serious way

### Fresh Ollama same-host comparator

Command:

```bash
curl -s http://127.0.0.1:11434/api/generate -d '{
  "model":"gemma4-e4b-local:latest",
  "prompt":"Write one short sentence about decentralized Gemma inference.",
  "stream":false,
  "options":{"temperature":0,"num_predict":32}
}'
```

Aggregated receipt:

- file: `/tmp/ollama-gemma4-32x3-current.json`
- `3` runs
- each run generated `32` tokens
- decode durations:
  - `322690747 ns`
  - `320755292 ns`
  - `321554546 ns`
- mean decode throughput: `99.48 tok/s`

### Native-load proof from the local Metal lane

Command:

```bash
PSIONIC_GEMMA4_METAL_DEBUG_LOAD=1 cargo run --release -q -p psionic-serve --example gemma4_bench -- \
  --model-path /Users/christopherdavid/models/gemma4/gemma4-e4b-ollama.gguf \
  --backend metal \
  --max-output-tokens 1 \
  --repeats 1 2>&1 | rg 'psionic\.gemma4\.metal\.load|^model:|^load:|^run '
```

Receipt:

- `fused_qkv_layers=23`
- `fused_gate_up_layers=42`
- `native_attention_layers=42`
- `native_ffn_layers=42`
- `per_layer_model_proj=native_GgmlQ4K`

Meaning:

- the remaining performance gap is not explained by a silent host projection
  fallback on the local Metal path
- the hot path is already fully native for attention, FFN, and the per-layer
  model projection

### Early negative tuning results

Decode-attention simdgroup sweep:

- partial warm-run receipts from `/tmp/psionic-gemma-simd1.json`,
  `/tmp/psionic-gemma-simd2.json`, `/tmp/psionic-gemma-simd4.json`, and
  `/tmp/psionic-gemma-simd8.json`
- warm decode throughput after the first run:
  - `1 simdgroup`: about `26.13 tok/s`
  - `2 simdgroups`: about `28.55 tok/s`
  - `4 simdgroups`: about `26.95 tok/s`
  - `8 simdgroups`: about `30.27 tok/s`

Meaning:

- the current token-count-based default is not the whole problem
- `8` simdgroups was roughly tied with the clean default receipt, not a real
  breakthrough

Output-head argmax fan-out experiment:

- raised `METAL_QUANTIZED_ARGMAX_ROWS_PER_THREADGROUP` from `8` to `32`
- clean receipt from `/tmp/psionic-gemma-argmax32-32x3.json`
- resulting mean decode throughput: `29.48 tok/s`

Meaning:

- a coarser candidate fan-out in the current quantized argmax kernel made the
  Gemma 4 lane slightly worse
- the remaining head-selection work needs a more targeted kernel change than
  simply increasing rows per threadgroup

Per-layer input exploratory upper bound:

- collected on the same direct benchmark binary that still contained the
  temporary `32`-row argmax fan-out experiment
- baseline direct receipt from `/tmp/psionic-direct-current.json`:
  - `28.88 tok/s`
  - `26` output tokens
- direct receipt with `PSIONIC_GEMMA4_DISABLE_PER_LAYER_INPUTS=1` from
  `/tmp/psionic-direct-no-per-layer.json`:
  - `40.34 tok/s`
  - `32` output tokens

Meaning:

- the Gemma per-layer input path is expensive enough to move throughput by a
  large margin
- the flag is not a shippable optimization because it changes generation
  behavior
- the next profiling pass should treat per-layer input projection and
  application as a first-class suspected bottleneck rather than a side detail

## What This Means

The landed fixes closed the correctness problem and removed one measurable piece
of local decode waste.

They did not make Psionic faster than Ollama on this Gemma 4 Metal lane.

The remaining gap is still about:

- Psionic exploratory warm decode: roughly `28-30 tok/s`
- Ollama same-host warm decode: roughly `99.5 tok/s`

That is still about a `3.3x-3.6x` gap.

## Where The Remaining Gap Lives

The remaining gap is not in GGUF admission anymore.

It is in the Metal decode system itself:

1. Native decode attention scheduling still needs more work.
2. Output-head selection for the large quantized vocab matrix is still a likely
   hot spot.
3. The local Gemma decode path still issues a long sequence of small kernels per
   layer instead of a more graph-owned or more aggressively fused schedule.

The evidence that this is now a runtime problem rather than a loader problem is
straightforward:

- the model is coherent again after the shared-KV fix
- the fast path already uses native Metal projections and native dense decode
  attention admission
- removing host KV placeholder allocation only moved the numbers slightly
- Ollama remains far ahead on the same artifact and prompt

## Recommended Follow-On Work

1. Benchmark the output-head argmax path in isolation and tune
   `quantized_matvec_argmax_*` kernels for the Gemma 4 output matrix shape.
2. Profile the per-layer input path directly now that loader fallback has been
   ruled out as the explanation.
3. Revisit the Metal decode-attention simdgroup heuristic and benchmark beyond
   the current token-count-based default.
4. Add one explicit Gemma 4 vs Ollama benchmark script or report lane so future
   regressions are visible in one command instead of hidden in ad hoc notes.

## Files Touched In This Change

- `crates/psionic-serve/src/gguf.rs`
- `crates/psionic-serve/src/lib.rs`
- `crates/psionic-serve/examples/gemma4_bench.rs`
