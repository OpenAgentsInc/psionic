# Psionic Parameter Golf Roadmap

> Status: written 2026-03-18 after reviewing `docs/ROADMAP.md`,
> `docs/ARCHITECTURE.md`, `docs/TRAIN_SYSTEM.md`, `README.md`,
> `~/code/parameter-golf/PSIONIC_PARAMETER_GOLF_SPEC.md`,
> `~/code/parameter-golf/README.md`,
> `~/code/parameter-golf/data/README.md`,
> `~/code/parameter-golf/train_gpt.py`,
> `~/code/parameter-golf/train_gpt_mlx.py`,
> `~/code/parameter-golf/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`,
> and
> `~/code/parameter-golf/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`.
>
> This is the lane-specific roadmap for building a Psionic-owned Parameter Golf
> stack inside `crates/psionic-*`. It is intentionally narrower than
> `docs/ROADMAP.md`: it is about FineWeb challenge data, compact causal-decoder
> training, exact `val_bpb` reproduction, artifact compression and accounting,
> and truthful submission packaging, not the whole Psionic library roadmap.

Agent execution instruction: implement this roadmap in dependency order, not by
whichever local benchmark or architecture idea looks most fun first. Freeze the
challenge oracle before widening the model family, keep non-record versus
record-track claims separate, and do not treat a thin Python launcher as proof
that Psionic owns the real training path.

Reference-first instruction: Parameter Golf work must not be implemented from
memory. Choose the reference that owns the layer being changed:

- start with `~/code/parameter-golf/PSIONIC_PARAMETER_GOLF_SPEC.md` for the
  intended Psionic-owned scope and the required claim discipline
- start with `~/code/parameter-golf/README.md` for the current public challenge
  rules, the active leaderboard, and the operator workflow as of 2026-03-18
- start with `~/code/parameter-golf/data/README.md` and
  `~/code/parameter-golf/data/cached_challenge_fineweb.py` for the published
  FineWeb shard layout, tokenizer artifacts, and manifest-driven download path
- start with `~/code/parameter-golf/train_gpt.py` for the current CUDA oracle:
  shard format, exact `val_bpb` logic, 9x512 baseline architecture, optimizer
  split, wallclock budgeting, and int8+zlib export path
- start with `~/code/parameter-golf/train_gpt_mlx.py` for the current local
  Apple iteration path and the challenge-compatible local parity expectations
- start with `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, and
  `docs/TRAIN_SYSTEM.md` for the canonical Psionic owner split, status
  vocabulary, train substrate, and refusal posture the lane must reuse instead
  of bypass

Psionic-only execution rule: the shipped Parameter Golf lane must remain
Psionic-owned end to end. Do not close roadmap items by training in PyTorch or
MLX and then calling the result "Psionic-backed" because the logs or metadata
landed in this repo. If Psionic does not own the model execution, optimizer
execution, evaluation, artifact export, and acceptance reporting, the issue is
not closed.

## Decision

Parameter Golf belongs in a separate `ROADMAP_PARAMETERGOLF.md`, not only as a
small bullet list inside `docs/ROADMAP.md`.

That decision is deliberate for four reasons:

1. `docs/ROADMAP.md` is the canonical full-library roadmap and should remain
   the answer to "what is Psionic overall?"
2. Parameter Golf cuts across `psionic-data`, `psionic-models`,
   `psionic-nn`, `psionic-train`, `psionic-eval`, `psionic-distributed`,
   `psionic-array`, and packaging or compatibility work, so the challenge lane
   is too specific to keep only as one paragraph in the main roadmap.
3. The repo already uses lane-specific deep dives such as
   `docs/ROADMAP_CLUSTER.md`, `docs/ROADMAP_FM.md`, `docs/ROADMAP_MLX.md`, and
   `docs/ROADMAP_TASSADAR.md`.
4. Parameter Golf needs its own issue queue, rules-accounting closure, and
   record versus non-record claim split without implying that the challenge is
   the whole Psionic training program.

So the structural rule is:

- `docs/ROADMAP.md` remains canonical
- `docs/ROADMAP_PARAMETERGOLF.md` is the Parameter Golf-specific deep dive

## Objective

Build a Psionic-owned Parameter Golf lane with:

- exact reproduction of the current public challenge oracle for data,
  tokenizer-byte accounting, and `val_bpb`
- a compact causal-decoder family in `psionic-models` that matches the current
  public 9x512 SP-1024 baseline before architecture search begins
- a Psionic-owned train loop, optimizer split, artifact export path, and eval
  report path that does not delegate the real work to PyTorch or MLX
- explicit non-record versus record-track packaging posture
- honest throughput, artifact-size, and accounting receipts for `8xH100`
  attempts instead of inferred leaderboard claims

This is not a plan to:

- rebrand the current Python scripts as a Psionic lane without replacing their
  execution ownership
- treat local Apple iteration as equivalent to `8xH100` challenge closure
- ignore code-size accounting because the Rust runtime lives outside the
  records folder
- claim record-track readiness before the public challenge accounting posture is
  explicit

## Relationship To The Main Roadmap

This roadmap is subordinate to `docs/ROADMAP.md`.

It depends on and refines work already named there:

- Epic 2 semantics and compatibility for tokenizer, layer, optimizer, and
  state behavior
- Epic 4 backend truth and performance for CUDA kernel coverage and honest
  throughput claims
- Epic 5 cluster and execution truth for multi-GPU topology, launch, and
  communication posture
- Epic 6 training, eval, and research for train-loop, checkpoint, artifact,
  and benchmark ownership

This roadmap does not widen product scope in `docs/MVP.md`, and it does not
move ownership boundaries out of `docs/OWNERSHIP.md`.

## Ownership Rules

This roadmap must continue to respect `docs/OWNERSHIP.md`:

- `crates/psionic-*` owns the reusable data, model, train, eval, optimizer,
  distributed, runtime, and artifact-accounting substrate for the challenge
- a thin compatibility `train_gpt.py` wrapper is acceptable only if challenge
  rules force it and only if the real execution is still Psionic-owned
- app UX, provider-market behavior, wallet or payout flows, and authority truth
  stay out of this lane

More specifically:

- `psionic-data` owns dataset and tokenizer contracts, split manifests,
  iteration, byte accounting, and batch planning
- `psionic-models` owns the compact decoder family and parameter-accounting
  metadata
- `psionic-nn` owns reusable layers, losses, optimizer shells, and quantized
  wrappers used by the lane
- `psionic-train` owns the trainer, optimizer execution, checkpoint, artifact
  export, and benchmark receipts
- `psionic-eval` owns challenge eval and acceptance reporting
- `psionic-distributed` plus lower runtime or backend crates own multi-GPU
  execution posture and capability truth

## Challenge Snapshot

As of 2026-03-18, the public `parameter-golf` repo says:

- the challenge runs from March 18, 2026 through April 30, 2026
- the artifact cap is decimal `16,000,000` bytes
- record-track runs must train in under `10` minutes on `8xH100`
- the metric is tokenizer-agnostic FineWeb validation compression in
  `bits per byte`
- evaluation must be self-contained with no external downloads or network
- the repository currently assumes a `train_gpt.py`-style submission artifact

The current public baseline target is also explicit:

- model shape: SP-1024, `9` layers, width `512`, `8` attention heads,
  `4` KV heads, tied embeddings, sequence length `1024`
- batch target: `524,288` train tokens per step
- eval target: full fixed `fineweb_val_*` split with exact `val_loss` and
  `val_bpb`
- artifact target: post-train int8 plus zlib roundtrip
- current record-track baseline on 2026-03-18:
  `val_bpb = 1.22436570`, code bytes `47,642`, total bytes `15,863,489`
- current notable non-record 4-hour run on 2026-03-18:
  `val_bpb = 1.20737944`, total bytes `15,810,161`

The current CUDA oracle in `train_gpt.py` matters in detail:

- the shard format is a binary header with magic `20240520`, version `1`, and
  little-endian `u16` token payloads
- `val_bpb` uses SentencePiece piece-byte lookup plus leading-space adjustment,
  not ordinary token-count normalization
- the baseline architecture uses GQA, RoPE, RMSNorm, learned residual mixing,
  tied embeddings, and a tanh logit softcap
- optimizer behavior is split across Adam groups plus Muon for matrix-shaped
  transformer parameters
- the reported leaderboard artifact is the int8 plus zlib roundtripped model,
  not only the raw checkpoint

## Current Position

Psionic already has real substrate this lane can reuse:

- `psionic-data`
  - `DatasetManifest`, `TokenizerDigest`, deterministic iteration windows,
    packing plans, data-ingress contracts, and distributed data-feed planning
- `psionic-models`
  - reusable model metadata plus runtime tokenizer support for GGUF-backed
    SentencePiece, GPT-style BPE, and WordPiece families
- `psionic-nn`
  - public `Module` trees, `Linear`, `Embedding`, `RmsNorm`,
    `cross_entropy_loss`, optimizer shells, and eval-oriented quantized module
    wrappers
- `psionic-train`
  - fixed-budget training core, checkpoint and replay truth, reusable
    optimizer math (`SGD`, `Adam`, `AdamW`, `LARS`, `LAMB`), mixed-precision
    policy, distributed optimizer contracts, and model IO
- `psionic-eval`
  - benchmark-package contracts, eval-run state, artifact capture, and
    repeated-run aggregation
- `psionic-distributed`
  - public groups, collective helpers, gradient averaging, tensor-parallel
    wrappers, FSDP-style update helpers, and topology-aware backend mapping
- `psionic-array` and backends
  - a bounded public CPU, Metal, and CUDA array surface, plus backend kernels
    that already include matmul/add and some RMSNorm or quantized helper paths

Psionic does not yet ship the challenge lane itself:

- no exact challenge-eval oracle above the new FineWeb shard contract and token-stream substrate
- no exact SentencePiece byte-accounting oracle for challenge `val_bpb`
- no Parameter Golf benchmark package or eval receipts
- no compact challenge decoder family matching the public baseline
- no Muon optimizer or challenge-matching optimizer grouping in Psionic
- no single-device trainer that exports the challenge int8 plus zlib artifact
- no transport-backed `8xH100` train lane proved at challenge throughput
- no challenge wrapper or record-folder output contract for record-track review

## Gap Map

| Area | Current posture | Required Parameter Golf work |
| --- | --- | --- |
| `psionic-data` | `implemented_early` generic manifests, tokenizer digests, iteration, packing, and distributed feed planning, plus a landed Parameter Golf shard loader, local-dir manifest builder, and deterministic token-stream contract | add challenge tokenizer byte lookup generation and wire the data contract into exact eval receipts and trainer ingestion |
| `psionic-models` | `implemented` model metadata plus bounded model families and runtime tokenizers | add the compact challenge decoder family with tied embeddings, GQA, RoPE, RMSNorm, residual mixing, and explicit parameter-count or byte-accounting facts |
| `psionic-nn` | `implemented` reusable layers, losses, optimizer shells, and eval-only quantization wrappers | widen or compose the current layer set into a reusable causal-decoder path and add any missing train-time primitive support the lane needs |
| `psionic-train` | `implemented_early` fixed-budget training core, optimizer math, mixed precision, model IO, checkpoint, and replay truth | add challenge-specific train loop, optimizer grouping, Muon support, grad accumulation, wallclock budgeting, and int8+zlib roundtrip export |
| `psionic-eval` | `implemented` benchmark and eval contracts, with a tracking-only lane acceptance checker but not the oracle itself | add exact Parameter Golf `val_loss` plus `val_bpb` eval reports, oracle parity fixtures, and challenge acceptance reporting tied to real lane receipts |
| `psionic-distributed` | `implemented_early` public distributed helpers with reference-emulated public collectives | add the real multi-GPU training path we intend to use for `8xH100`, plus honest topology, communication, and refusal receipts |
| `psionic-array` and backends | `implemented_early` bounded CPU/Metal/CUDA public surface, with wider CUDA backend kernels below it | widen the CUDA train-time kernel and runtime path needed for competitive small-decoder throughput without overclaiming the public array surface |
| packaging and compatibility | `planned` for this lane | define the non-record versus record-track wrapper, code-byte accounting posture, record-folder output, and submission metadata path |

## Strategy

The lane should be delivered in two separate answers:

- Deliverable A: a Psionic-native research and training lane that owns model
  execution, training, evaluation, artifact export, and reporting
- Deliverable B: a challenge-compliant submission lane whose code accounting,
  wrapper shape, and record-folder output are acceptable under the public
  challenge rules

We should not assume A automatically solves B.

## GitHub Issue Queue

No Parameter Golf issue block existed in `OpenAgentsInc/psionic` when this
roadmap was written. This roadmap therefore opens the first repo-local issue
queue for the lane.

The GitHub issue queue for this roadmap now exists under
`PGOLF-000` / [#159](https://github.com/OpenAgentsInc/psionic/issues/159),
with `PGOLF-001` / [#160](https://github.com/OpenAgentsInc/psionic/issues/160)
reserved for the roadmap itself and `PGOLF-002` /
[#161](https://github.com/OpenAgentsInc/psionic/issues/161) plus `PGOLF-003` /
[#162](https://github.com/OpenAgentsInc/psionic/issues/162) now closing the
governance baseline for claim language and acceptance tracking.

## Epic 0: Governance And Acceptance

### Goal

Freeze the claim boundary before we chase benchmarks.

### Exit Criteria

- the repo has one explicit Parameter Golf claim vocabulary
- record versus non-record posture is explicit
- the lane has one machine-readable acceptance report or checker contract

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-001` / [#160](https://github.com/OpenAgentsInc/psionic/issues/160) | done (2026-03-18) | `Psionic Parameter Golf: create the lane-specific roadmap and issue program` | This document closes the issue. It records the owner split, the public challenge snapshot as of 2026-03-18, and the dependency-ordered issue queue. |
| `PGOLF-002` / [#161](https://github.com/OpenAgentsInc/psionic/issues/161) | done (2026-03-18) | `Psionic Parameter Golf: freeze challenge-accounting posture and record-vs-non-record claim language` | The repo now has `docs/PARAMETER_GOLF_ACCOUNTING.md`, which freezes the allowed claim vocabulary, makes counted-runtime and wrapper posture explicit, and records that the current lane posture is `research` until stronger categories turn green. |
| `PGOLF-003` / [#162](https://github.com/OpenAgentsInc/psionic/issues/162) | done (2026-03-18) | `Psionic Parameter Golf: add an acceptance matrix and checker for oracle parity, trainer parity, distributed closure, and submission posture` | The repo now has `docs/PARAMETER_GOLF_ACCEPTANCE_MATRIX.md`, `docs/parameter_golf_acceptance_report.schema.json`, `fixtures/parameter_golf/reports/parameter_golf_acceptance_report.json`, and `scripts/check-parameter-golf-acceptance.sh`, which turn the lane into a schema-backed tracking contract instead of free-form benchmark prose. |

## Epic 1: Data And Metric Oracle

### Goal

Rebuild the challenge oracle in Psionic before model work.

### Exit Criteria

- FineWeb shard loading is deterministic and challenge-compatible
- SP-1024 tokenizer identity and validation split identity are explicit
- Psionic reproduces the current `val_loss` and `val_bpb` on frozen fixtures

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-101` / [#163](https://github.com/OpenAgentsInc/psionic/issues/163) | done (2026-03-18) | `Psionic Parameter Golf: add FineWeb challenge manifests, shard loading, and deterministic token streaming` | `psionic-data` now exposes a dedicated Parameter Golf shard ABI, exact `u16` shard loading, a local-dir bundle builder that produces train-prefix plus fixed-validation manifests with challenge metadata and datastream bindings, and a replay-safe token-stream contract that matches the public contiguous shard order. |
| `PGOLF-102` / [#164](https://github.com/OpenAgentsInc/psionic/issues/164) | open | `Psionic Parameter Golf: add SentencePiece byte-accounting and exact val_bpb reproduction` | Rebuild the current `build_sentencepiece_luts(...)` and leading-space byte-accounting logic in Psionic so the lane can compute the same tokenizer-agnostic `val_bpb` as `train_gpt.py` and `train_gpt_mlx.py` on the same tokens and logits. |
| `PGOLF-103` / [#165](https://github.com/OpenAgentsInc/psionic/issues/165) | open | `Psionic Parameter Golf: add frozen parity fixtures against train_gpt.py and train_gpt_mlx.py` | Add fixture-level parity tests that compare shard loading, validation token slicing, `val_loss`, and `val_bpb` against the current Python and MLX oracles. The lane must prove metric parity before any architecture or throughput claims widen. |

## Epic 2: Baseline Model And Single-Device Trainer

### Goal

Port the current public baseline exactly before searching for better models.

### Exit Criteria

- a Psionic decoder family matches the current public baseline shape
- optimizer grouping and Muon behavior are reproduced honestly
- one single-device run can train, validate, export, reload, and re-evaluate

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-201` / [#166](https://github.com/OpenAgentsInc/psionic/issues/166) | open | `Psionic Parameter Golf: add the compact challenge decoder family in psionic-models` | Add the first Parameter Golf model family with tied embeddings, GQA, RoPE, RMSNorm, learned residual mixing, skip weights, and the current tanh logit softcap. The first closure bar is parity with the public 9x512 baseline, not novelty. |
| `PGOLF-202` / [#167](https://github.com/OpenAgentsInc/psionic/issues/167) | open | `Psionic Parameter Golf: add challenge optimizer grouping and Muon parity in psionic-train` | Rebuild the current optimizer split, including token-embedding Adam, scalar or control Adam groups, and Muon for matrix-shaped transformer parameters. Do not silently substitute an existing optimizer just because it is already in-tree. |
| `PGOLF-203` / [#168](https://github.com/OpenAgentsInc/psionic/issues/168) | open | `Psionic Parameter Golf: add a single-device reference trainer with int8+zlib roundtrip validation` | Add one end-to-end trainer path in `psionic-train` plus `psionic-eval` that runs the baseline family, enforces the challenge batch geometry, exports the raw artifact plus the roundtripped int8+zlib artifact, reloads it, and re-runs eval under Psionic ownership. |

## Epic 3: Distributed Throughput And Backend Closure

### Goal

Turn single-device parity into honest `8xH100` throughput closure.

### Exit Criteria

- the repo has one explicit `8xH100` train topology plan
- multi-GPU logs and receipts are preserved as benchmark truth
- the required CUDA runtime or kernel widenings are explicit

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-301` / [#169](https://github.com/OpenAgentsInc/psionic/issues/169) | open | `Psionic Parameter Golf: add challenge benchmark packages, eval receipts, and leaderboard-facing reports` | Add a dedicated benchmark or report lane for Parameter Golf so the repo can preserve wallclock, memory, artifact-size, and score receipts under the same eval substrate instead of using ad hoc logs only. |
| `PGOLF-302` / [#170](https://github.com/OpenAgentsInc/psionic/issues/170) | open | `Psionic Parameter Golf: add the real distributed 8xH100 training lane and throughput receipts` | Decide and implement the actual distributed execution posture for the challenge run, then emit explicit topology, communication, timing, and memory receipts for that path. The target is measured closure, not assumed DDP equivalence. |
| `PGOLF-303` / [#171](https://github.com/OpenAgentsInc/psionic/issues/171) | open | `Psionic Parameter Golf: widen CUDA runtime and kernel coverage required by the challenge baseline` | Close the gap between today's bounded public array surface and the kernels or runtime path needed for competitive small-decoder training. This includes any train-time RoPE, RMSNorm, attention, residual, optimizer, or quantization kernels that must be widened for the chosen `8xH100` path. |

## Epic 4: Packaging, Submission, And Research Widening

### Goal

Turn the parity lane into a defensible challenge entry, then widen into
research without losing the oracle.

### Exit Criteria

- one non-record packaging path exists under explicit accounting rules
- the repo has a reproducible research harness for post-parity variants
- record-track packaging remains blocked on explicit public accounting closure
  until the rule story is honest

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-401` / [#172](https://github.com/OpenAgentsInc/psionic/issues/172) | open | `Psionic Parameter Golf: add a non-record submission wrapper and record-folder output contract` | Build the first packaging answer as a clearly labeled non-record lane if needed, including wrapper shape, README generation, `submission.json`, train log preservation, and artifact-byte accounting that does not pretend the Rust runtime is free. |
| `PGOLF-402` / [#173](https://github.com/OpenAgentsInc/psionic/issues/173) | open | `Psionic Parameter Golf: add a research harness for post-parity architecture and compression variants` | After parity is real, add a controlled harness for recurrent/shared-depth variants, stronger parameter tying, better quantization, and compression experiments under the same oracle and artifact-accounting rules. |
| `PGOLF-403` / [#174](https://github.com/OpenAgentsInc/psionic/issues/174) | open | `Psionic Parameter Golf: add a record-track submission contract once public accounting is explicit` | Only after the public wrapper and code-size story is clear should the repo claim record-track readiness. This issue owns the final record-submission path, not the earlier research or non-record lane. |

## Current Execution Order

### Phase 1: freeze challenge posture and acceptance (done 2026-03-18)

- `PGOLF-001` -> roadmap and issue queue
- `PGOLF-002` -> accounting and claim-language contract
- `PGOLF-003` -> acceptance matrix, schema, report, and checker

### Phase 2: rebuild the challenge oracle

- `PGOLF-101` -> landed FineWeb shard ABI, manifest builder, and deterministic token-stream contract
- `PGOLF-102`
- `PGOLF-103`

### Phase 3: port the public baseline exactly

- `PGOLF-201`
- `PGOLF-202`
- `PGOLF-203`

### Phase 4: close measured 8xH100 execution truth

- `PGOLF-301`
- `PGOLF-302`
- `PGOLF-303`

### Phase 5: package the lane honestly, then widen research

- `PGOLF-401`
- `PGOLF-402`
- `PGOLF-403`

## Bottom Line

Parameter Golf is a good forcing function for compact-model training, exact
tokenizer-byte accounting, artifact compression, and honest benchmark
packaging.

Psionic is now strong enough to justify a real lane, but the repo should stay
precise about what is true on 2026-03-18:

- Psionic already has strong reusable train, eval, model, tokenizer, and
  distributed substrate
- Psionic does not yet own the exact challenge oracle or the public baseline
  path
- the first truthful result should be parity against `train_gpt.py`, not a new
  architecture
- record-track claims stay blocked until the actual counted-runtime submission
  path is implemented, not merely documented
