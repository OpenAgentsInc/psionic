# Psionic Tassadar Roadmap Bridge

> Status: repo-local bridge refreshed 2026-03-19 after reviewing the live
> external Tassadar roadmap, `README.md`, `docs/ARCHITECTURE.md`,
> `docs/ROADMAP_TASSADAR_INDEX.md`, `docs/ROADMAP_TASSADAR_TAS_SYNC.md`,
> `docs/TASSADAR_WASM_RUNBOOK.md`, and the open `TAS-*` GitHub issue queue.

This file stays in the repo because a large set of GitHub issues and repo docs
already link to `docs/ROADMAP_TASSADAR.md`.

It is no longer the full living Tassadar roadmap.

## Canonical Live Roadmap

The active Tassadar roadmap now lives outside this repo at:

- `/Users/christopherdavid/code/alpha/tassadar/tassadar-llm-as-computer-roadmap.md`

Use that file for:

- tranche definitions
- issue sequencing after `TAS-102`
- terminal-contract language
- the current widening path from bounded named profiles to `TCM.v1`

This repo-local file remains a stable bridge so existing issue links keep
landing on something truthful instead of a stale copied roadmap.

## Current Repo-Local Summary

As of 2026-03-19, the live external roadmap and the current GitHub `tassadar`
issue queue are aligned.

Current honest posture:

- benchmarked, bounded internal computation under named profiles with explicit
  refusal surfaces
- the current served profile remains
  `tassadar.internal_compute.article_closeout.v1`
- the frozen core-Wasm lane now has a declared semantic window plus a committed
  closure gate, and the current closure verdict remains `not_closed` with
  `served_publication_allowed = false`
- the repo now also has a bounded scalar-`f32` semantics matrix with canonical
  quiet-NaN normalization, ordered Wasm-style comparisons, CPU-reference-only
  execution posture, and explicit refusal on `f64`, NaN-payload preservation,
  and non-CPU fast-math regimes
- the repo now also has a staged mixed-numeric ladder over exact scalar-`f32`,
  exact mixed `i32`/`f32`, and bounded-approximate `f64 -> f32` conversion
  profiles, with malformed and out-of-envelope conversions kept on explicit
  typed refusal paths
- the repo now also has a numeric portability matrix over backend, toolchain,
  and machine-class envelopes for the bounded float and mixed-numeric lanes,
  keeping exact cpu-reference publication separate from suppressed non-CPU and
  bounded-approximate numeric regimes
- the repo now also has a float-profile acceptance gate plus route policy for
  exact numeric profiles, allowing bounded cpu-reference public named-profile
  posture without widening those profiles into the default served lane
- the repo now also has one bounded exceptions proposal profile over typed
  throw/catch/rethrow semantics with explicit trap-stack parity and malformed
  handler refusal truth, allowing named public profile posture only on the
  current-host cpu-reference envelope while keeping the default served
  exceptions lane empty
- the repo now also has one bounded `memory64` continuation profile over
  sparse single-memory checkpoints above the 4GiB boundary, with committed
  resume artifacts, typed datastream locators, and explicit backend-limit
  refusal truth on the current-host cpu-reference envelope
- the repo now also has one bounded multi-memory routing profile over two
  explicit topology families, with committed per-memory checkpoint artifacts
  and typed malformed-topology refusal truth on the current-host cpu-reference
  envelope
- the repo now also has one bounded component/linking proposal profile over
  two explicit component-pair topologies, with committed interface-type
  lineage artifacts and typed incompatible-interface refusal truth on the
  current-host cpu-reference envelope
- the repo now also has one bounded SIMD deterministic profile with committed
  backend rows for cpu-reference exactness, metal and cuda scalar fallback,
  and typed accelerator-specific refusal truth, carried through served
  publication without creating a default served SIMD lane
- the repo now also has one research-only shared-memory and threads profile
  with a deterministic scheduler envelope, explicit sandbox boundary truth,
  and typed refusal on host-nondeterministic scheduling plus relaxed
  memory-ordering regimes; it remains non-served and non-promoted by design
- the repo now also has one durable process-object family
  `tassadar.internal_compute.process_objects.v1` over the committed
  checkpoint-backed workloads, with first-class snapshot, tape, and work-queue
  objects, typed datastream locators, provider receipts, and environment
  bindings while keeping stale snapshots, out-of-range tape cursors, and
  profile-mismatched queues on explicit refusal paths
- the repo now also has one bounded interactive session-process profile
  `tassadar.internal_compute.session_process.v1` over deterministic echo and
  stateful counter message-loop surfaces, with committed runtime, eval, and
  route-policy reports plus served-publication and provider-envelope bindings,
  while keeping open-ended external event streams on explicit refusal paths and
  leaving the default served session-process lane empty
- the repo now also has one bounded spill-aware continuation profile
  `tassadar.internal_compute.spill_tape_store.v1` over spill-backed memory
  segments and external tape-store artifacts, with committed runtime, eval,
  provider, and environment bindings on the current-host cpu-reference
  portability envelope while keeping oversize state, missing tape segments, and
  non-cpu portability widening on explicit refusal paths
- the repo now also has one bounded preemptive-job profile
  `tassadar.internal_compute.preemptive_jobs.v1` over deterministic
  round-robin and weighted-fair slice schedulers, with committed runtime,
  cluster, and eval reports plus served-publication and provider-envelope
  bindings that keep the lane operator-visible but non-served while refusing
  host-nondeterministic scheduling explicitly
- the repo now also has one bounded virtual-filesystem and artifact-mount
  profile `tassadar.effect_profile.virtual_fs_mounts.v1`, with committed
  runtime, sandbox, and eval artifacts plus provider receipts that keep the
  lane challengeable, replay-safe, and non-served while leaving canonical
  mount authority explicitly owned by `world-mounts` outside standalone psionic
- the repo now also has one bounded simulator-backed effect profile
  `tassadar.effect_profile.simulator_backed_io.v1`, with committed runtime,
  sandbox, and eval artifacts plus provider receipts and environment bindings
  that admit seeded clock, randomness, and loopback-network rows while keeping
  ambient clock, OS entropy, and socket I/O on explicit refusal paths and
  leaving the lane non-served by design
- the repo now also has one bounded async-lifecycle profile
  `tassadar.internal_compute.async_lifecycle.v1`, with committed runtime, eval,
  and router artifacts plus served-publication and provider-envelope bindings
  that admit deterministic interrupt, bounded retry, and safe-boundary
  cancellation rows while keeping open-ended callbacks, mid-effect
  cancellation, and unbounded retry on explicit refusal paths and leaving the
  default served async lane empty
- the repo now also has one bounded effectful replay-and-challenge lane
  `tassadar.effect_profile.replay_challenge_receipts.v1`, with committed
  runtime and eval artifacts plus provider-facing receipts that bind seeded
  simulator, virtual-filesystem, and async-safe-cancel replay rows to explicit
  effect receipts, replay digests, and challenge receipts while keeping
  missing-effect evidence, missing challenge evidence, and unsafe effect
  families on explicit refusal paths; named `kernel-policy` and `nexus`
  follow-ons remain explicit dependency markers outside standalone psionic and
  the lane stays non-served by design
- the repo now also has one bounded internal component-model ABI lane
  `tassadar.internal_compute.component_model_abi.v1`, with committed IR and
  compiler contracts, runtime and eval artifacts, and provider-facing receipts
  that bind session-checkpoint, artifact-reader-retry, and spill-resume
  component graphs to explicit interface manifests while keeping cross-profile
  handle mismatches and unsupported variant unions on typed refusal paths; the
  lane remains benchmark-only with `served_publication_allowed = false`
- the repo now also has one bounded internal-compute package-manager lane with
  committed compiler, router, eval, served-publication, and provider-facing
  surfaces that publish the named public packages
  `package.clrs_shortest_path_stack.v1`,
  `package.hungarian_matching_stack.v1`, and
  `package.verifier_search_stack.v1`, while keeping ambiguous solver,
  insufficient-evidence, and portability-mismatch requests on explicit refusal
  paths and leaving the default served package lane empty
- the repo now also has one bounded cross-profile link-compatibility lane with
  committed compiler, runtime, router, eval, and provider-facing surfaces that
  preserve one exact session-process -> spill-tape link, one explicit
  generalized-ABI -> component-model downgrade plan, and typed refusal on
  portability-envelope and effect-boundary mismatches while keeping
  `served_publication_allowed = false`
- broader profiles above that remain benchmarked, suppressed, refused, or
  unpromoted unless their explicit gates are green

Still unclaimed:

- arbitrary Wasm execution
- full frozen core-Wasm closure
- broad practical internal computation as a generally served capability
- Turing-complete support

Current dependency spine:

1. `TAS-103` to `TAS-106`: real program execution
2. `TAS-107` to `TAS-110`: resumable and effect-safe execution
3. `TAS-111` to `TAS-112`: portable and publishable execution
4. `TAS-113` to `TAS-114`: frozen core-Wasm closure
5. `TAS-115` to `TAS-124`: numeric and proposal-family widening
6. `TAS-125` to `TAS-136`: process and internal-platform closure
7. `TAS-137` to `TAS-140`: learned and hybrid broad-compute comparison
8. `TAS-141` to `TAS-150`: public-claim, economic, and governance closure
9. `TAS-151` to `TAS-156`: universal-substrate and Turing-completeness
   closeout

Issue-state note:

- `TAS-103` through `TAS-124` are already implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- `TAS-125` through `TAS-135` are now implemented and tracked in
  `docs/ROADMAP_TASSADAR_TAS_SYNC.md`
- the current open GitHub backlog begins at `TAS-138` and runs through
  `TAS-156`

## Repo-Local Companion Docs

Use the repo-local docs for the questions they actually answer:

- `README.md` for the current Tassadar inventory and claim surface inside this
  repo
- `docs/ARCHITECTURE.md` for canonical Psionic-wide status vocabulary and
  machine-facing contract boundaries
- `docs/ROADMAP_TASSADAR_INDEX.md` for landed phase-to-artifact mapping
- `docs/ROADMAP_TASSADAR_TAS_SYNC.md` for issue-to-implementation closure
  notes
- `docs/TASSADAR_WASM_RUNBOOK.md` for the current bounded Wasm operator path
- `fixtures/tassadar/reports/tassadar_acceptance_report.json` and
  `scripts/check-tassadar-acceptance.sh` for the current machine-readable claim
  gate

If you only have the public `psionic` checkout and not the external Tassadar
workspace, use this bridge plus the current `TAS-*` issue bodies, the TAS sync
doc, and the artifact index together instead of relying on the old copied
roadmap text.

## Maintenance Rule

- update the external roadmap when tranche order, issue sequencing, or
  terminal-contract language changes
- update this repo-local bridge when the pointer, current truthful posture, or
  repo-visible issue alignment changes
- do not reintroduce a copied issue-by-issue backlog here unless the canonical
  roadmap moves back into this repo
