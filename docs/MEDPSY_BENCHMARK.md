# MedPsy Benchmark Harness

> Status: `implemented_early` for direct Psionic BF16 safetensors / GGUF CPU
> runtime benchmarking. Comparator benchmarking against Tether-recommended
> `llama.cpp`, QVAC SDK, vLLM, or Transformers paths remains planned in issue
> `#982`.

The direct benchmark runner is:

```bash
cargo run --release -p psionic-models --example medpsy_bench -- \
  --model-path <path> \
  --artifact-kind safetensors \
  --model-size 1.7b \
  --backend cpu \
  --prompt-token-ids 151644 \
  --max-new-tokens 1 \
  --repeats 1 \
  --json-out fixtures/medpsy/benchmarks/manual/medpsy_17b_safetensors_cpu.json
```

For the first GGUF row:

```bash
cargo run --release -p psionic-models --example medpsy_bench -- \
  --model-path <path> \
  --artifact-kind gguf \
  --model-size 1.7b \
  --backend cpu \
  --prompt-token-ids 151644 \
  --max-new-tokens 1 \
  --repeats 1 \
  --json-out fixtures/medpsy/benchmarks/manual/medpsy_17b_q4_k_m_gguf_cpu.json
```

The convenience wrapper is:

```bash
PSIONIC_MEDPSY_17B_SAFETENSORS_PATH=/abs/path/model.safetensors \
PSIONIC_MEDPSY_17B_Q4_K_M_GGUF_PATH=/abs/path/model.gguf \
  scripts/release/run-medpsy-local-bench.sh
```

On a CUDA host, build and run with:

```bash
PSIONIC_MEDPSY_BENCH_BACKEND=cuda \
PSIONIC_MEDPSY_17B_Q4_K_M_GGUF_PATH=/abs/path/model.gguf \
  cargo run --release -p psionic-models --features medpsy-cuda --example medpsy_bench -- \
    --model-path /abs/path/model.gguf \
    --artifact-kind gguf \
    --model-size 1.7b \
    --backend cuda \
    --prompt-token-ids 151644 \
    --max-new-tokens 1 \
    --repeats 1 \
    --json-out fixtures/medpsy/benchmarks/manual/medpsy_17b_q4_k_m_gguf_cuda.json
```

The report schema is `psionic.medpsy.bench.v1`. It records:

- artifact path
- artifact kind
- model size
- backend
- execution engine
- prompt token IDs
- generated token IDs
- total wall time
- decode tokens per second
- artifact digest
- medical policy posture

The benchmark is runtime evidence only. It is not a clinical quality claim and
does not replace medical safety evaluation, HealthBench, closed-ended medical
benchmarks, or the comparator matrix planned for `#982`.

## Current Comparator Matrix

The first retained comparator matrix is:

```text
fixtures/medpsy/benchmarks/medpsy_comparator_matrix_20260511_local.json
```

It records one completed Psionic CPU row on
`medpsy-1.7b-q4_k_m-imat.gguf` and one attempted `llama-cli` comparator row.
The `llama-cli` row timed out after `300s` on the special-token smoke prompt, so
the matrix is `partial_with_comparator_timeout`. It is valid harness evidence,
not a competitive throughput claim.

Follow-up comparator work must:

- switch the llama.cpp row to a normal rendered MedPsy prompt;
- parse llama.cpp timing output into the same JSON fields as Psionic;
- add QVAC SDK when the local Node/Bare runtime harness is available;
- add a BF16 vLLM or Transformers row when the source model is available.
