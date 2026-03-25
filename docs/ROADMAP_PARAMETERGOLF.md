# Psionic Parameter Golf Roadmap

> Status: refreshed 2026-03-18 after re-reading `docs/ROADMAP.md`,
> `docs/ARCHITECTURE.md`, `docs/TRAIN_SYSTEM.md`,
> `docs/PARAMETER_GOLF_ACCOUNTING.md`,
> `docs/PARAMETER_GOLF_ACCEPTANCE_MATRIX.md`,
> `docs/PARAMETER_GOLF_RECORD_TRACK_CONTRACT.md`, `README.md`,
> `~/code/parameter-golf/PSIONIC_PARAMETER_GOLF_SPEC.md`,
> `~/code/parameter-golf/README.md`,
> `~/code/parameter-golf/data/README.md`,
> `~/code/parameter-golf/train_gpt.py`,
> `~/code/parameter-golf/train_gpt_mlx.py`,
> `~/code/parameter-golf/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`,
> and
> `~/code/parameter-golf/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`.
>
> Active development on this lane stopped on 2026-03-19 by explicit user
> direction. See `docs/PARAMETER_GOLF_AFTER_ACTION.md` and
> `fixtures/parameter_golf/reports/parameter_golf_after_action_report.json`
> for the final stop record and the closed-not-planned resolution of the
> remaining open PGOLF issues.
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
- a PR-submittable `records/...` folder export that matches the public
  `parameter-golf` repo contract instead of a repo-local review bundle only
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
- all submissions are pull requests that only add one new folder under
  `records/track_10min_16mb` or `records/track_non_record_16mb`
- each submission folder must include at least `README.md`,
  `submission.json`, `train.log`, `train_gpt.py`, and any extra dependencies
  required to compile and run inside that folder
- new SOTA record PRs must beat the current record by at least `0.005` nats
  and carry enough run logs to justify `p < 0.01`, unless the README's
  systems-only waiver applies

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

- a bounded local-reference benchmark package, challenge score report, and wallclock, memory, artifact-size, plus train or eval bundle-root receipts now exist, but only for the local-reference CPU lane
- a bounded single-device local-reference trainer now exists in `psionic-train`, with explicit batch geometry, grad accumulation, checkpoint and restart posture, raw safetensors export, and int8 plus zlib roundtrip validation, but it is still a CPU reference path rather than measured challenge throughput closure
- an exact `8xH100` DDP-style receipt lane now exists with topology,
  communication, wallclock, memory, and explicit refusal truth, but the public
  CUDA train path still carries typed blocker notes for decoder-kernel and
  runtime widening
- a first honest non-record runtime package and record-folder output contract
  now exist, with a top-level `train_gpt.py` launcher that executes a shipped
  Psionic payload inside the folder, but the path is still a bounded local-
  reference replay rather than a promoted record-track runtime, so neither
  record-track nor true challenge-PR submission posture is closed yet

## Gap Map

| Area | Current posture | Required Parameter Golf work |
| --- | --- | --- |
| `psionic-data` | `implemented_early` generic manifests, tokenizer digests, iteration, packing, and distributed feed planning, plus a landed Parameter Golf shard loader, local-dir manifest builder, deterministic token-stream contract, SentencePiece byte-accounting LUT oracle, and frozen parity fixtures for shard loading, validation slicing, `val_loss`, and `val_bpb` parity against the public Python and MLX scripts | wire the data and tokenizer contracts into exact eval receipts and trainer ingestion |
| `psionic-models` | `implemented_early` model metadata plus bounded model families and runtime tokenizers, now including a landed Parameter Golf reference decoder with tied embeddings, GQA, RoPE, RMSNorm-without-weight, learned residual mixing, skip weights, tanh logit softcap, stable tensor naming, baseline parameter accounting, and a frozen `train_gpt.py` parity fixture at the public `9x512` shape, plus new named-parameter override and export helpers used by the single-device reference trainer, a typed PGOLF MLP activation surface that covers both baseline `relu_squared` and competitive `leaky_relu_squared(0.5)` across the reference-model, graph, IR, CPU, and CUDA lanes, and an explicit `BigramHash` score-path surface with deterministic hash ids, optional projection, scalar scale, and stable tensor naming | retain real `H100` or `8xH100` receipts for the non-baseline activation and bigram paths, then widen the family toward faster train-time or research variants without losing the frozen oracle |
| `psionic-nn` | `implemented` reusable layers, losses, optimizer shells, and eval-only quantization wrappers | widen or compose the current layer set into a reusable causal-decoder path and add any missing train-time primitive support the lane needs |
| `psionic-train` | `implemented_early` fixed-budget training core, optimizer math, mixed precision, model IO, checkpoint, and replay truth, plus a landed Parameter Golf lane module for the public optimizer split, baseline schedule helpers, exact Muon reference math, a frozen `train_gpt.py` optimizer parity fixture, a bounded local-reference trainer with challenge batch geometry, grad accumulation, checkpoint or restart state, raw safetensors export, int8+zlib roundtrip restore, and a local-reference benchmark bundle builder that emits score, wallclock, memory, artifact-size, and train or eval bundle-root receipts plus a shipped non-record runtime replay path | widen from the CPU reference path to the measured distributed challenge lane and carry the new receipt lane forward to real multi-GPU evidence |
| `psionic-eval` | `implemented_early` benchmark and eval contracts, with a tracking-only lane acceptance checker, a landed Parameter Golf validation eval report used by the local-reference trainer and roundtrip checks, and now a dedicated Parameter Golf benchmark package plus challenge score and receipt contracts | connect the landed receipt lane to real distributed runs, benchmark-package bundles, and leaderboard-facing review without overclaiming local-reference measurements |
| `psionic-distributed` | `implemented_early` public distributed helpers with reference-emulated public collectives | add the real multi-GPU training path we intend to use for `8xH100`, plus honest topology, communication, and refusal receipts |
| `psionic-array` and backends | `implemented_early` bounded CPU/Metal/CUDA public surface, with wider CUDA backend kernels below it, and now a landed Parameter Golf CUDA training coverage report that keeps BF16, RoPE or GQA attention, RMSNorm, residual-mix, Muon, and quantized-export requirement families explicit with a stable blocker list | retire the remaining CUDA train-time blockers by widening the public runtime or kernel path needed for competitive small-decoder throughput without overclaiming the public array surface |
| packaging and compatibility | `implemented_early` for a real non-record runtime package | keep the shipped folder contract honest, carry the explicit counted-runtime and build-dependency posture forward to stronger payloads, and verify that the exported folder compiles and runs offline inside a local `parameter-golf` clone |

## Strategy

The lane should be delivered in two separate answers:

- Deliverable A: a Psionic-native research and training lane that owns model
  execution, training, evaluation, artifact export, and reporting
- Deliverable B: a challenge-compliant submission lane whose code accounting,
  wrapper shape, and record-folder output are acceptable under the public
  challenge rules and that can be copied as-is into
  `parameter-golf/records/...` for a real PR

We should not assume A automatically solves B.

## Current Sprint Assumption

This section is historical only.

Before the 2026-03-19 stop decision, the continuation sprint was no longer
aimed at another review package.

It was aimed at a real record-track candidate path.

That meant the next queue should optimize for:

- a generated folder that can actually be staged into `~/code/parameter-golf`
  as a submission candidate
- the real exported entrypoint and runtime path, not another receipt-review
  shim
- real `8xH100` evidence from that exported folder

For this sprint, a top-level `train_gpt.py` launcher into a shipped Rust
payload is an acceptable engineering candidate path so long as:

- the folder remains self-contained and runnable
- the shipped runtime, helper, and build-dependency bytes remain explicit
- the verifier and replay evidence stay machine-readable

## GitHub Issue Queue

The bootstrap issue queue for this roadmap now exists and is complete:

- `PGOLF-000` / [#159](https://github.com/OpenAgentsInc/psionic/issues/159)
  through `PGOLF-403` /
  [#174](https://github.com/OpenAgentsInc/psionic/issues/174) are closed
- that issue block closed the honest bootstrap lane: oracle parity, bounded
  trainer parity, distributed receipts, non-record review packaging, research
  harness, and explicit blocked record-track contract

The follow-on real-record sprint queue now exists under
`PGOLF-500` / [#183](https://github.com/OpenAgentsInc/psionic/issues/183).

That queue covers `PGOLF-501` / [#184](https://github.com/OpenAgentsInc/psionic/issues/184)
through `PGOLF-703` / [#193](https://github.com/OpenAgentsInc/psionic/issues/193)
and turns the remaining work from "reviewable package" to "record-track PR
candidate" into an explicit dependency-ordered program.

That historical sprint queue is no longer active.

On 2026-03-19, the remaining open issues in that queue were closed not planned
under the stop decision recorded in `docs/PARAMETER_GOLF_AFTER_ACTION.md`.

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
| `PGOLF-102` / [#164](https://github.com/OpenAgentsInc/psionic/issues/164) | done (2026-03-18) | `Psionic Parameter Golf: add SentencePiece byte-accounting and exact val_bpb reproduction` | `psionic-data::parameter_golf` now exposes explicit SentencePiece token-role metadata, a LUT builder that mirrors the current `build_sentencepiece_luts(...)` behavior for normal, byte, control, unknown, and unused tokens, plus exact leading-space byte counting and `bits per byte` helpers for the challenge oracle. |
| `PGOLF-103` / [#165](https://github.com/OpenAgentsInc/psionic/issues/165) | done (2026-03-18) | `Psionic Parameter Golf: add frozen parity fixtures against train_gpt.py and train_gpt_mlx.py` | `psionic-data` now ships a committed oracle-parity fixture, a repo-local generator that extracts the current public Python and MLX reference functions, and a frozen test that proves shard loading, validation token slicing, SentencePiece LUT parity, `val_loss`, and `val_bpb` against both public scripts. |

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
| `PGOLF-201` / [#166](https://github.com/OpenAgentsInc/psionic/issues/166) | done (2026-03-18) | `Psionic Parameter Golf: add the compact challenge decoder family in psionic-models` | `psionic-models` now ships a dedicated Parameter Golf reference family with the public `9x512` baseline config, stable tensor naming and parameter accounting, a CPU-reference forward path covering tied embeddings, GQA, RoPE, RMSNorm-without-weight, learned residual mixing, skip weights, and tanh logit softcap, plus a frozen `train_gpt.py` parity fixture and generator that prove baseline parameter-count, tensor-layout, logits, and loss parity under a deterministic weight recipe. |
| `PGOLF-202` / [#167](https://github.com/OpenAgentsInc/psionic/issues/167) | done (2026-03-18) | `Psionic Parameter Golf: add challenge optimizer grouping and Muon parity in psionic-train` | `psionic-train` now ships a dedicated Parameter Golf lane module that reproduces the public optimizer split, baseline warmdown and Muon-momentum schedule helpers, and exact Muon reference-step math, plus a frozen `train_gpt.py` optimizer fixture and generator that prove token, matrix, scalar, and head grouping plus Muon-update parity against the current public script. |
| `PGOLF-203` / [#168](https://github.com/OpenAgentsInc/psionic/issues/168) | done (2026-03-18) | `Psionic Parameter Golf: add a single-device reference trainer with int8+zlib roundtrip validation` | `psionic-train` now ships a bounded local-reference trainer for the public baseline family with explicit single-device challenge batch geometry, grad accumulation, checkpoint and restart state, raw safetensors export, int8+zlib export or restore, and post-roundtrip validation reports in `psionic-eval`, all under Psionic ownership. |

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
| `PGOLF-301` / [#169](https://github.com/OpenAgentsInc/psionic/issues/169) | done (2026-03-18) | `Psionic Parameter Golf: add challenge benchmark packages, eval receipts, and leaderboard-facing reports` | `psionic-eval` now ships a dedicated Parameter Golf benchmark package plus score, wallclock, memory, artifact-size, and bundle-root receipt contracts, and `psionic-train` now ships a bounded local-reference benchmark bundle builder that emits those review artifacts from the Psionic-owned trainer instead of relying on ad hoc logs. |
| `PGOLF-302` / [#170](https://github.com/OpenAgentsInc/psionic/issues/170) | done (2026-03-18) | `Psionic Parameter Golf: add the real distributed 8xH100 training lane and throughput receipts` | `psionic-train` plus `psionic-eval` now ship an exact `8xH100` DDP-style receipt lane aligned to the public `train_gpt.py` posture, including explicit H100 admission gates, replicated topology, NCCL-style all-reduce communication stages, measured wallclock receipts, analytic memory planning, and typed refusal when the lane cannot honestly clear the declared challenge bar. |
| `PGOLF-303` / [#171](https://github.com/OpenAgentsInc/psionic/issues/171) | done (2026-03-18) | `Psionic Parameter Golf: widen CUDA runtime and kernel coverage required by the challenge baseline` | `psionic-train` now ships a machine-readable Parameter Golf CUDA training coverage report that keeps the required BF16, RoPE or GQA attention, RMSNorm, residual-mix, Muon, and quantized-export families explicit, publishes an honest challenge-readiness refusal plus stable blocker list, and threads that report digest and blocker set directly into the distributed `8xH100` receipt lane instead of hiding the remaining CUDA train-path gaps behind a green throughput headline. |

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
| `PGOLF-401` / [#172](https://github.com/OpenAgentsInc/psionic/issues/172) | done (2026-03-18) | `Psionic Parameter Golf: add a non-record submission wrapper and record-folder output contract` | `psionic-train` now ships a typed non-record submission package builder plus folder writer that emit `README.md`, `submission.json`, `train.log`, a runnable `train_gpt.py` surface, preserved benchmark artifacts, and a machine-readable counted-byte accounting receipt without pretending a hidden Rust runtime is free. |
| `PGOLF-402` / [#173](https://github.com/OpenAgentsInc/psionic/issues/173) | done (2026-03-18) | `Psionic Parameter Golf: add a research harness for post-parity architecture and compression variants` | `psionic-research` now ships a committed Parameter Golf research-harness report that freezes one measured baseline control plus explicit shared-depth or recurrent, stronger parameter-tying, and compression or quantization candidate families against the same oracle digests, submission metric, and counted-byte vocabulary used by the landed non-record package. |
| `PGOLF-403` / [#174](https://github.com/OpenAgentsInc/psionic/issues/174) | done (2026-03-18) | `Psionic Parameter Golf: add a record-track submission contract once public accounting is explicit` | `psionic-train` now ships a committed blocked record-track contract report that binds the acceptance report, non-record package surface, research harness, and distributed benchmark reference together while explicitly refusing promotion beyond `non_record_submission` until reproducible `8xH100` evidence is retired as the remaining blocker. |

## Current Blocking Delta To A Real Record-Track PR

The challenge repo now defines a narrower target than this roadmap originally
spelled out.

To be truly "recordable" in the public challenge sense, Psionic must be able
to open a PR that only adds one new `records/...` folder to
`~/code/parameter-golf`, where that folder's `train_gpt.py` and dependencies
actually execute the real submission path in place.

The current delta is:

| Public challenge requirement | Current Psionic status | Why this still blocks a real PR |
| --- | --- | --- |
| PR adds only one new `records/...` folder | Psionic now has a deterministic final PR-bundle generator plus a verifier-backed dry-run against the live `parameter-golf` records contract | This closes the first honest maintainer-facing non-record PR surface, but it is still not a record-track runtime or promotion claim. |
| `train_gpt.py` and any shipped dependencies compile and run within that folder | The current `train_gpt.py` launches a shipped runtime payload and the exported folder now passes end-to-end dry-runs against the live local challenge repo | This proves folder-local Psionic execution, but not yet a record-track training runtime or challenge-speed execution claim. |
| code bytes plus compressed model bytes stay under `16,000,000` | Psionic now counts the shipped launcher, runtime payload, and model artifact against the real exported folder and keeps the current shipped runtime/build-dependency posture explicit | This requirement is closed for the current shipped payload; it no longer blocks the lane by itself. |
| record-track runs reproduce under `10` minutes on `8xH100` | Psionic has an explicit distributed receipt lane and explicit blocker list | The repo still lacks a promoted submission entrypoint whose folder-local execution has actually produced reproducible record-track evidence on real `8xH100` hardware. |
| SOTA record PRs carry delta/significance evidence or a systems-only waiver posture | `psionic-eval` now ships a typed maintainer-facing promotion receipt and `psionic-train` now carries it in the final PR bundle | This closes the promotion-evidence mechanics, but the current receipt still refuses record promotion until stronger record-track evidence exists. |

That is the difference between the current truthful posture
`non_record_submission` and the stronger claim "we can generate a real
record-track Parameter Golf PR."

## Epic 5: Challenge-Repo Submission Surface

### Goal

Turn the current honest runtime package into a real challenge-compatible
submission folder.

### Exit Criteria

- the repo emits the exact `records/...` folder shape expected by the public
  challenge repo
- the shipped `train_gpt.py` and dependencies execute the real Psionic
  training, export, and eval path inside that folder
- one exported folder can be staged into a local `parameter-golf` clone and
  verified offline without hidden repo dependencies

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-501` / [#184](https://github.com/OpenAgentsInc/psionic/issues/184) | done (2026-03-18) | `Psionic Parameter Golf: add a challenge-repo record-folder compatibility matrix and external-repo verifier` | `psionic-train` now ships a typed record-folder compatibility report plus `scripts/check-parameter-golf-record-folder-compatibility.sh`, which validate the canonical `README.md` / `submission.json` / `train.log` / `train_gpt.py` contract, keep folder-local dependency posture explicit, and dry-run one exported folder against a local `parameter-golf` clone instead of assuming the Psionic package root is already challenge-compatible. |
| `PGOLF-502` / [#185](https://github.com/OpenAgentsInc/psionic/issues/185) | done (2026-03-18) | `Psionic Parameter Golf: replace the non-record review wrapper with a true Psionic submission entrypoint` | `psionic-train` now ships a root-local `train_gpt.py` launcher plus a committed Psionic runtime payload, runtime manifest, and runtime fixture so the exported folder executes a real Psionic-owned restore-and-eval path and writes its own runtime receipt instead of relying on a receipt-review shim. |
| `PGOLF-503` / [#186](https://github.com/OpenAgentsInc/psionic/issues/186) | done (2026-03-18) | `Psionic Parameter Golf: add counted-runtime and build-dependency accounting for the real submission payload` | The exported non-record folder now carries a machine-readable accounting receipt tied to the real shipped launcher and runtime payload bytes, keeps helper and build-dependency posture explicit, and computes the artifact-cap verdict from the actual exported payload instead of the earlier review-wrapper surface. |
| `PGOLF-504` / [#187](https://github.com/OpenAgentsInc/psionic/issues/187) | done (2026-03-18) | `Psionic Parameter Golf: add a PR-ready non-record export path that runs in the public repo` | `psionic-train` now ships `examples/parameter_golf_non_record_submission_bundle.rs`, which exports the canonical non-record folder directly into `parameter-golf/records/track_non_record_16mb/...`, and the exported folder has already passed a live local-clone dry-run through `scripts/check-parameter-golf-record-folder-compatibility.sh`. |

## Epic 6: Record-Track Runtime And 8xH100 Closure

### Goal

Turn the PR-ready submission surface into a genuine record-candidate runtime.

### Exit Criteria

- the remaining Parameter Golf CUDA training blockers are retired on the
  public submission path rather than hidden behind receipt-only evidence
- one exported folder produces real `8xH100` run evidence from the shipped
  entrypoint itself
- folder-local replay verifies the recorded metric, wallclock, and artifact
  facts well enough for challenge review

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-601` / [#188](https://github.com/OpenAgentsInc/psionic/issues/188) | done (2026-03-19) | `Psionic Parameter Golf: retire the remaining baseline CUDA blockers on the public submission path` | The canonical CUDA coverage report no longer carries any family-level blocker ids on the public submission path. Broader end-to-end trainer, hardware-evidence, and promotion work remains tracked separately under `PGOLF-604`, `PGOLF-602`, and `PGOLF-609`. |
| `PGOLF-615` / [#259](https://github.com/OpenAgentsInc/psionic/issues/259) | done (2026-03-19) | `Psionic Parameter Golf: widen public CUDA execution for mul and RMSNorm` | `psionic-backend-cuda` now executes dense contiguous `f32` `mul` and backend-specialized `rms_norm` through the public `ExecutionBackend` path, validates those plan steps explicitly, and advertises runtime RMSNorm extension support honestly. This widens the real forward CUDA surface for residual-control tensors and RMSNorm without claiming backward or full train-path closure. |
| `PGOLF-616` / [#260](https://github.com/OpenAgentsInc/psionic/issues/260) | done (2026-03-19) | `Psionic Parameter Golf: add RMSNorm autodiff and reference-eval support` | `psionic-ir` now evaluates `rms_norm` through the dense `f32` reference path, builds bounded reverse-mode RMSNorm backward graphs instead of refusing them, and keeps the other backend-extension gradient families explicitly unsupported. The repo now owns real train-visible RMSNorm graph semantics above the backend while leaving public CUDA backward execution for `PGOLF-617`. |
| `PGOLF-617` / [#261](https://github.com/OpenAgentsInc/psionic/issues/261) | done (2026-03-19) | `Psionic Parameter Golf: widen public CUDA execution for RMSNorm backward graphs` | `psionic-backend-cuda` now executes bounded dense contiguous `f32` RMSNorm forward plus the dedicated RMSNorm backward graph ops across batched rows through the public runtime path, with end-to-end CUDA backward-graph tests and updated coverage truth that retire the explicit RMSNorm blocker without implying BF16, decoder-block, residual-mix, or optimizer closure. |
| `PGOLF-618` / [#262](https://github.com/OpenAgentsInc/psionic/issues/262) | done (2026-03-19) | `Psionic Parameter Golf: add public CUDA train-time RoPE/GQA decoder-block execution` | `psionic-ir` now accepts grouped-query attention program shapes, and `psionic-backend-cuda` now executes one bounded dense `f32` non-interleaved RoPE plus causal grouped-query decoder block through the public runtime path by packing baseline token slices into the existing decode-kernel surface. This retires the old "no public decoder-block execution" gap while leaving reverse-mode closure explicit in `PGOLF-622`. |
| `PGOLF-622` / [#266](https://github.com/OpenAgentsInc/psionic/issues/266) | done (2026-03-19) | `Psionic Parameter Golf: add public reverse-mode support for the RoPE/GQA decoder block` | `psionic-ir` now lowers bounded decoder reverse-mode gradients through dedicated `rotary_embedding_backward` plus `scaled_dot_product_attention_{query,key,value}_backward` ops, evaluates that path on the dense `f32` reference surface, and keeps RoPE table gradients explicitly unsupported. The canonical coverage truth now retires the old reverse-mode blocker and narrows the remaining attention blocker to public CUDA backward-runtime execution. |
| `PGOLF-624` / [#268](https://github.com/OpenAgentsInc/psionic/issues/268) | done (2026-03-19) | `Psionic Parameter Golf: add public CUDA execution for decoder-block backward graphs` | `psionic-backend-cuda` now executes bounded decoder backward graphs for `rotary_embedding_backward` and `scaled_dot_product_attention_{query,key,value}_backward` on CUDA device buffers through a host-orchestrated dense `f32` runtime, with end-to-end backward-graph parity tests against the dense reference path. The canonical coverage truth now retires the explicit decoder backward-runtime blocker without claiming fused, fully device-resident, or challenge-speed decoder training closure. |
| `PGOLF-619` / [#263](https://github.com/OpenAgentsInc/psionic/issues/263) | done (2026-03-19) | `Psionic Parameter Golf: add public CUDA BF16 train-visible execution posture` | `psionic-backend-cuda` now owns dense BF16 buffer residency plus bounded row-major BF16xBF16-to-F32 matmul execution through the public CUDA submission path, and the canonical coverage truth now narrows the BF16 blocker from a vague precision-policy contract to the remaining train-graph and optimizer surface tracked by `PGOLF-623`. |
| `PGOLF-623` / [#267](https://github.com/OpenAgentsInc/psionic/issues/267) | done (2026-03-19) | `Psionic Parameter Golf: widen public CUDA BF16 train graph and optimizer execution` | `psionic-train` now owns one bounded host-orchestrated CUDA BF16 master-weight optimizer step over BF16 train-visible parameter and gradient buffers with FP32 master weights and FP32 optimizer state. The canonical coverage truth now retires the last family-level BF16 blocker without claiming generic BF16 graph execution, fused optimizer kernels, or challenge-speed mixed-precision trainer closure. |
| `PGOLF-620` / [#264](https://github.com/OpenAgentsInc/psionic/issues/264) | done (2026-03-19) | `Psionic Parameter Golf: add public CUDA Muon optimizer execution` | `psionic-train` now owns one bounded public CUDA Muon step for the baseline matrix-shaped parameter groups, offloading the Newton-Schulz BF16 matmul family to the public CUDA dense surface while keeping transpose, norm, and scalar orchestration explicit in Rust. The canonical coverage truth now retires the explicit Muon blocker without claiming fused or fully device-resident trainer closure. |
| `PGOLF-621` / [#265](https://github.com/OpenAgentsInc/psionic/issues/265) | done (2026-03-19) | `Psionic Parameter Golf: add public CUDA residual-mix train-path execution` | `psionic-backend-cuda` now proves one bounded Parameter Golf residual-mix training graph end to end on CUDA, including reverse-mode gradients over dense contiguous full-shape residual-control tensors, and the canonical coverage truth now retires the explicit residual-mix blocker without claiming generic broadcast or fused decoder closure. |
| `PGOLF-625` / [#270](https://github.com/OpenAgentsInc/psionic/issues/270) | done (2026-03-19) | `Psionic Parameter Golf: add Parameter Golf train-time SiLU graph and CUDA runtime support` | `psionic-core`, `psionic-ir`, `psionic-backend-cpu`, and `psionic-backend-cuda` now expose bounded `silu` plus `silu_backward` backend-extension ops, register them in the built-in operator surface, evaluate them on the dense reference path, materialize reverse-mode SiLU gradients, and execute the forward plus seeded backward graph on CUDA through a bounded host-orchestrated `f32` runtime. This widens the pointwise train-time graph surface, but the actual baseline MLP activation blocker is tracked separately under `PGOLF-627`. |
| `PGOLF-627` / [#271](https://github.com/OpenAgentsInc/psionic/issues/271) | done (2026-03-19) | `Psionic Parameter Golf: add Parameter Golf train-time ReLU^2 graph and runtime support` | `psionic-core`, `psionic-ir`, `psionic-backend-cpu`, and `psionic-backend-cuda` now expose exact bounded `relu_squared` plus `relu_squared_backward` backend-extension ops, register them in the built-in operator surface, evaluate them on the dense reference path, materialize reverse-mode ReLU^2 gradients, and execute the forward plus seeded backward graph on CUDA through a bounded host-orchestrated `f32` runtime. This retires the real baseline MLP activation gap on the honest Rust-native train graph without claiming logits, loss, or end-to-end trainer closure. |
| `PGOLF-628` / [#272](https://github.com/OpenAgentsInc/psionic/issues/272) | done (2026-03-19) | `Psionic Parameter Golf: add Parameter Golf tanh-softcap projection-loss graph and CUDA runtime support` | `psionic-core`, `psionic-ir`, `psionic-backend-cpu`, and `psionic-backend-cuda` now expose a bounded `parameter_golf_projection_loss` plus `parameter_golf_projection_loss_backward` backend-extension family, register it in the built-in operator surface, evaluate it on the dense reference path, materialize reverse-mode logits gradients without differentiating target ids, execute the forward and seeded backward graph on CUDA, and refuse the same op explicitly on the CPU runtime. This retires the remaining host-logits roundtrip dependency for the honest Rust-native Parameter Golf trainer path without claiming the trainer loop itself is done. |
| `PGOLF-626` / [#269](https://github.com/OpenAgentsInc/psionic/issues/269) | done (2026-03-19) | `Psionic Parameter Golf: lower the full baseline forward-logits and seeded backward path` | `psionic-train` now ships a trainer-owned `parameter_golf_baseline_graph` module that lowers the full bounded baseline path from dense embedded inputs through pre-softcap logits, binds the stable Parameter Golf parameter surface explicitly, applies the host-owned tanh softcap plus cross-entropy seed honestly, and materializes seeded backward gradients including the tied token-embedding scatter-add path. The bounded tests prove forward-logit parity against the committed baseline fixture and finite-difference gradient agreement across representative parameter families, while the same change also retires the rotary-sign mismatch and attention-backward seed-layout bug that were blocking that parity. |
| `PGOLF-602` / [#189](https://github.com/OpenAgentsInc/psionic/issues/189) | closed_not_planned (2026-03-19) | `Psionic Parameter Golf: capture real 8xH100 run bundles from the exported submission entrypoint` | This remained real unfinished work, but the lane stopped before real exported-folder `8xH100` evidence was captured. The repo now preserves the gap in `docs/PARAMETER_GOLF_AFTER_ACTION.md` instead of tracking it as active work. |
| `PGOLF-603` / [#190](https://github.com/OpenAgentsInc/psionic/issues/190) | done (2026-03-18) | `Psionic Parameter Golf: add record-folder-local replay verification for metric, wallclock, and artifact bytes` | `psionic-train` now ships exported-folder run-evidence and replay-verification reports plus `scripts/check-parameter-golf-record-folder-replay.sh`, validating `submission.json`, `train.log`, runtime receipts, benchmark receipts, and counted bytes directly from the generated folder while keeping the current measured-or-refused `8xH100` posture explicit. |
| `PGOLF-604` / [#194](https://github.com/OpenAgentsInc/psionic/issues/194) | closed_not_planned (2026-03-19) | `Psionic Parameter Golf: add a Rust-native 1xH100 baseline trainer entrypoint` | The repo landed the real Rust-native single-H100 bring-up seam and bounded baseline graph lane, but not the full Psionic-owned trainer closure before the lane stopped. The final honest status now lives in `docs/PARAMETER_GOLF_AFTER_ACTION.md`. |
| `PGOLF-605` / [#195](https://github.com/OpenAgentsInc/psionic/issues/195) | done (2026-03-18) | `Psionic Parameter Golf: add Rust-only 1xH100 parity and bring-up evidence for the public baseline` | The repo now ships a committed `parameter_golf_single_h100_bringup.json` report plus `scripts/check-parameter-golf-single-h100-bringup.sh`, both generated from the real downloaded FineWeb `sp1024` cache and tokenizer. The current evidence is an honest refusal with exact dataset, tokenizer, geometry, wallclock, and blocker facts, while preserving explicit absence of final training metrics and artifact bytes because the command still stops at contract validation. |

## Epic 7: Promotion Gate And PR Submission

### Goal

Turn runtime closure into a maintainer-facing PR path for non-record and
record-track submissions.

### Exit Criteria

- one generated folder is PR-ready for `records/track_non_record_16mb`
- one generated folder is PR-ready for `records/track_10min_16mb` once the
  metric bar is actually met
- record promotion carries the public README-required delta/significance
  evidence or a systems-only waiver justification

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-701` / [#191](https://github.com/OpenAgentsInc/psionic/issues/191) | done (2026-03-18) | `Psionic Parameter Golf: add submission-promotion receipts for SOTA delta, significance, and systems-only waiver posture` | `psionic-eval` now ships a typed promotion receipt that captures the compared baseline, delta in bits or nats per byte, significance posture, systems-only waiver posture, and explicit refusal when the evidence does not support record promotion. |
| `PGOLF-702` / [#192](https://github.com/OpenAgentsInc/psionic/issues/192) | done (2026-03-18) | `Psionic Parameter Golf: add a final PR bundle generator and checklist for parameter-golf/records` | `psionic-train` now emits one deterministic final PR bundle that contains the exact `records/...` folder, exported-folder evidence and replay reports, promotion receipt, and maintainer-facing checklist text needed to open a public non-record challenge PR directly from Psionic-owned artifacts. |
| `PGOLF-703` / [#193](https://github.com/OpenAgentsInc/psionic/issues/193) | done (2026-03-18) | `Psionic Parameter Golf: dry-run a full Psionic submission against the local parameter-golf clone and preserve the verifier report` | The repo now stages the generated folder into the live local `~/code/parameter-golf` clone, reruns the compatibility and replay verifiers there, removes the staged folder, checks the clone returns to the original `git status --short --branch`, and preserves the committed dry-run report. |

## Epic 8: Concrete Post-Parity Architecture Queue

### Goal

Turn the research harness into one concrete, bounded architecture queue without
weakening baseline or record-track claim posture.

### Exit Criteria

- one committed queue report exists under the canonical research harness
- each queue row has one explicit mechanism and one bounded claim posture
- implemented rows preserve metric, artifact-byte, and runtime facts
- locality or restricted-attention promotion stays blocked on challenge-like
  evidence rather than toy-window intuition

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-610` / [#254](https://github.com/OpenAgentsInc/psionic/issues/254) | done (2026-03-19) | `Psionic Parameter Golf: create the first concrete post-parity architecture experiment queue` | `psionic-research` now ships `parameter_golf_architecture_experiment_queue_report.json` plus `docs/PARAMETER_GOLF_ARCHITECTURE_EXPERIMENT_QUEUE.md`, turning the earlier generic variant bucket into one explicit queue with implemented shared-depth, stronger parameter-tying, and restricted-attention evidence rows. |
| `PGOLF-611` / [#255](https://github.com/OpenAgentsInc/psionic/issues/255) | done (2026-03-19) | `Psionic Parameter Golf: add a shared-depth or recurrent compact decoder variant report` | The first concrete queue now includes a measured `shared_depth_decoder_value_tying_proxy` row, preserving `val_loss`, `val_bpb`, compressed-model bytes, counted total bytes, and explicit unchanged-dense-runtime facts for a decoder-depth value-reuse proxy on the frozen baseline family. |
| `PGOLF-612` / [#256](https://github.com/OpenAgentsInc/psionic/issues/256) | done (2026-03-19) | `Psionic Parameter Golf: add a stronger parameter-tying compact variant report` | The same queue now includes a measured `mirrored_block_pair_tying_proxy` row, preserving metric, artifact-byte, and runtime facts for one stronger block-level parameter-tying probe on the frozen baseline family under unchanged code bytes and unchanged dense runtime posture. |
| `PGOLF-613` / [#257](https://github.com/OpenAgentsInc/psionic/issues/257) | done (2026-03-19) | `Psionic Parameter Golf: add a locality or restricted-attention compact variant report` | `psionic-eval` now exposes a bounded windowed-validation surface, `psionic-research` now commits `parameter_golf_restricted_attention_report.json` plus `docs/PARAMETER_GOLF_RESTRICTED_ATTENTION_REPORT.md`, and Psionic now has public-safe restricted-attention evidence on one real `seq_len=1024` challenge-format slice. The first fixed `256`-token window is explicit negative evidence: attention-score terms drop to about `43.7%` of dense, but `val_bpb` worsens by about `0.05037`, so locality is not promoted. |

## Follow-On External And Promotion Issues

### Goal

Carry the historical external non-record PR record, the local blocked
accounting posture, and the later record-candidate campaign as explicit tracked
work instead of leaving those steps implicit after the local dry-run lane.

### Issues

| ID | Status | Proposed GitHub issue title | Description |
| --- | --- | --- | --- |
| `PGOLF-606` / [#250](https://github.com/OpenAgentsInc/psionic/issues/250) | closed_not_planned (2026-03-19) | `Psionic Parameter Golf: promote the acceptance matrix and claim posture after the public runtime path closes` | This never became honest because the real runtime and hardware-evidence issues stopped first. The retained claim posture remains `non_record_submission`, and the stop record now freezes that boundary explicitly. |
| `PGOLF-607` / [#251](https://github.com/OpenAgentsInc/psionic/issues/251) | done (2026-03-19) | `Psionic Parameter Golf: submit and track the first real external non-record PR against openai/parameter-golf` | Psionic now preserves the first real external non-record PR as historical evidence only: `https://github.com/openai/parameter-golf/pull/119` was opened and then closed on 2026-03-19, and the exact fork commit, record-folder tree hash, and verifier digests remain frozen in `fixtures/parameter_golf/reports/parameter_golf_external_non_record_pr.json`. |
| `PGOLF-608` / [#252](https://github.com/OpenAgentsInc/psionic/issues/252) | done (2026-03-19) | `Psionic Parameter Golf: capture maintainer-facing resolution on counted Rust runtime and build-dependency posture` | This issue is intentionally retired under the outbound-contribution ban. Psionic no longer tracks a live maintainer-facing counted-runtime question inside this repo; the earlier unanswered upstream question is preserved as historical evidence while the repo now owns a local counted-runtime and zero-build-dependency contract for the shipped runtime payload. |
| `PGOLF-609` / [#253](https://github.com/OpenAgentsInc/psionic/issues/253) | closed_not_planned (2026-03-19) | `Psionic Parameter Golf: freeze one real record-candidate config and run campaign` | This remained blocked on the unfinished public runtime path and missing real `8xH100` evidence when the lane stopped. No real record-candidate campaign was run before closure. |
| `PGOLF-614` / [#258](https://github.com/OpenAgentsInc/psionic/issues/258) | done (2026-03-19) | `Psionic Parameter Golf: replace stale external-PR claims with repo-local blocked posture` | Psionic now updates the canonical docs and machine-readable receipt to reflect that the first external non-record PR is closed historical evidence, not a live workflow, and that the repo-local counted-runtime posture is defended without requiring a live outbound clarification lane. |

## Current Execution Order

### Phase 1: freeze challenge posture and acceptance (done 2026-03-18)

- `PGOLF-001` -> roadmap and issue queue
- `PGOLF-002` -> accounting and claim-language contract
- `PGOLF-003` -> acceptance matrix, schema, report, and checker

### Phase 2: rebuild the challenge oracle

- `PGOLF-101` -> landed FineWeb shard ABI, manifest builder, and deterministic token-stream contract
- `PGOLF-102` -> landed SentencePiece byte-accounting LUTs and `val_bpb` byte-count helpers
- `PGOLF-103` -> landed frozen Python/MLX parity fixtures plus shard, validation-slice, `val_loss`, and `val_bpb` oracle tests

### Phase 3: port the public baseline exactly

- `PGOLF-201` -> landed compact decoder family, stable tensor layout, baseline parameter accounting, and frozen `train_gpt.py` logits/loss parity fixture at the public `9x512` shape
- `PGOLF-202` -> landed challenge optimizer grouping, baseline schedule helpers, exact Muon reference-step parity, and a frozen `train_gpt.py` optimizer fixture in `psionic-train`
- `PGOLF-203` -> landed bounded local-reference trainer, checkpoint or restart posture, raw safetensors export, int8+zlib roundtrip restore, and validation re-eval in `psionic-train` plus `psionic-eval`

### Phase 4: close measured 8xH100 execution truth

- `PGOLF-301` -> landed local-reference benchmark package, challenge score report, wallclock or memory or artifact-size receipts, and train or eval bundle roots for later leaderboard-facing review
- `PGOLF-302` -> landed exact `8xH100` DDP-style topology, NCCL-style collective receipts, analytic memory planning, and measured-or-refused challenge-bar outcomes
- `PGOLF-303` -> landed the machine-readable CUDA training coverage report, stable blocker set, challenge-readiness refusal posture, and distributed receipt linkage for the remaining train-path gaps

### Phase 5: package the lane honestly, then widen research

- `PGOLF-401` -> landed the first honest non-record submission package, counted-byte accounting receipt, runnable entrypoint surface, and record-folder writer
- `PGOLF-402` -> landed the committed post-parity research harness report with one measured baseline control and three guarded candidate families on the same oracle and accounting surface
- `PGOLF-403` -> landed the explicit blocked record-track contract so the remaining runtime, accounting, and `8xH100` blockers are machine-readable

### Phase 6: turn the runtime package into a real challenge folder

- `PGOLF-501` -> landed the compatibility matrix and dry-run verifier against the live `parameter-golf/records/...` contract
- `PGOLF-502` -> landed the true Psionic non-record submission entrypoint path with a shipped runtime payload and runtime receipt
- `PGOLF-503` -> landed machine-readable counted-runtime and build-dependency accounting for the real exported payload
- `PGOLF-504` -> landed the first PR-ready non-record export path that runs inside the public repo

### Phase 7: close real record-track runtime and `8xH100` evidence

- `PGOLF-601` -> retired the remaining family-level public CUDA baseline blockers on the exported submission path
- `PGOLF-615` -> landed dense `f32` `mul` and backend-specialized `rms_norm` forward execution on the public CUDA path
- `PGOLF-616` -> landed RMSNorm autodiff and reference-eval closure so the train-visible graph now exists above the backend
- `PGOLF-617` -> landed bounded public CUDA RMSNorm backward-graph execution and retired the explicit RMSNorm blocker
- `PGOLF-618` -> landed the bounded public CUDA decoder-core forward path for train-time RoPE plus GQA execution
- `PGOLF-622` -> landed bounded decoder reverse-mode graph closure above the backend and narrowed the remaining attention blocker to CUDA backward runtime
- `PGOLF-624` -> landed the bounded host-orchestrated CUDA decoder backward path and retired the explicit attention blocker
- `PGOLF-619` -> landed the first honest public CUDA BF16 runtime seam and narrowed the remaining BF16 blocker to train-graph and optimizer closure
- `PGOLF-623` -> landed the bounded host-orchestrated CUDA BF16 master-weight optimizer step and retired the last family-level BF16 blocker
- `PGOLF-620` -> landed the bounded public CUDA Muon step and retired the explicit optimizer blocker
- `PGOLF-621` -> landed the bounded public CUDA residual-mix train path and retired the explicit residual blocker
- `PGOLF-625` -> landed bounded train-time SiLU forward and backward graph/runtime support across the core IR, CPU reference backend, and CUDA runtime, but that was extra pointwise coverage rather than the actual baseline MLP blocker
- `PGOLF-627` -> landed exact bounded train-time ReLU^2 forward and backward graph/runtime support across the core IR, CPU reference backend, and CUDA runtime
- `PGOLF-628` -> landed bounded Parameter Golf tanh-softcap projection-loss forward and backward graph/runtime support across the core IR, CPU reference backend refusal path, and CUDA runtime
- `PGOLF-626` -> landed the bounded full-baseline forward-logits and seeded backward path, including explicit host-owned softcap or seed handling and tied-embedding gradient reconstruction for the real Parameter Golf parameter surface
- `PGOLF-603` -> landed folder-local replay verification plus exported-folder run-evidence reports for metrics, wallclock, and artifact bytes
- `PGOLF-604` -> landed the first Rust-native single-H100 bring-up command and report seam plus the bounded baseline graph lane; the remaining trainer closure was never completed before the lane stopped
- `PGOLF-605` -> landed the committed single-H100 parity or refusal report plus local-cache verifier for the Rust-native bring-up path
- `PGOLF-602` -> real exported-folder `8xH100` run bundles were never captured before the lane stopped

### Phase 8: promote and submit

- `PGOLF-701` -> landed the maintainer-facing promotion receipt for delta, significance, and systems-waiver posture
- `PGOLF-702` -> landed the deterministic final PR bundle and checklist for `parameter-golf/records/...`
- `PGOLF-703` -> landed the full local challenge-clone dry run plus committed verifier report

### Phase 9: widen the concrete post-parity architecture queue

- `PGOLF-610` -> landed the first committed concrete queue for post-parity architecture work under the canonical research harness
- `PGOLF-611` -> landed one measured shared-depth decoder value-tying proxy row with explicit metric, artifact-byte, and unchanged-runtime facts
- `PGOLF-612` -> landed one measured mirrored block-pair tying proxy row with explicit metric, artifact-byte, and unchanged-runtime facts
- `PGOLF-613` -> locality or restricted-attention stays open until the repo has challenge-like `seq_len=1024` evidence

### Phase 10: run the first real external submission loop

- `PGOLF-607` -> landed the first external non-record PR receipt; that PR is now closed and preserved as historical evidence only
- `PGOLF-608` -> retired the live-upstream accounting-question lane under the outbound-contribution ban; the ambiguity now stays blocked locally instead
- `PGOLF-614` -> landed the stale external-PR cleanup and repo-local blocked posture in the canonical docs and receipt
- `PGOLF-606` -> acceptance-matrix promotion never became honest before the lane stopped
- `PGOLF-609` -> no real record-candidate campaign was frozen and run before the lane stopped

## Bottom Line

Parameter Golf is a good forcing function for compact-model training, exact
tokenizer-byte accounting, artifact compression, and honest benchmark
packaging.

Psionic is now strong enough to justify a real lane, but the repo should stay
precise about what was true before the 2026-03-19 stop decision:

- Psionic already has strong reusable train, eval, model, tokenizer, and
  distributed substrate
- Psionic now owns the challenge oracle, the public baseline path, and the
  bounded local-reference plus distributed receipt lanes, and the canonical
  CUDA coverage report no longer carries family-level blocker ids
- Psionic now also has a first honest non-record runtime package and a
  maintainer-facing export path for the public non-record repo surface, plus
  exported-folder replay evidence, typed promotion posture, one deterministic
  PR bundle, and one live local-clone dry run, but that is still not the same
  thing as a record-track-ready folder because the exported path is a bounded
  local-reference replay and the stronger shipped runtime-byte story is not yet
  defended for record-track promotion
- post-parity architecture and compression work now has a committed research
  harness plus a first concrete architecture queue, but those candidates remain
  explicitly research-only until new results are measured
- the first truthful result should be parity against `train_gpt.py`, not a new
  architecture
- the next real work is no longer "more bootstrap closure"; it is turning the
  Rust-native single-H100 seam into the actual trainer path and then capturing
  reproducible `8xH100` evidence from the exported folder path that now exists
- record-track claims still stay blocked until that path is implemented,
  measured, and promoted, even though the blocked contract is now documented
  and machine-readable
