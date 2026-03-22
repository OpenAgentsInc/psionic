# PSION Plugin Program Map

> Status: canonical proposed convergence map for the next `Psion x Tassadar`
> plugin-conditioned training tranche, written 2026-03-22 from the historical
> alpha planning note in `~/code/alpha/tassadar/` and adjusted to current repo
> truth.

## Why This Doc Exists

The repo now has two mature but still separate systems:

- the `Psion` learned-model training lane
- the bounded `Tassadar` starter-plugin and controller substrate

The next meaningful tranche is not more generic `Psion` pretraining and not
more standalone plugin-runtime work. It is the convergence lane that turns real
plugin traces, receipts, schemas, and admissibility posture into training and
evaluation truth for a learned plugin-conditioned model.

This doc exists so the repo can point at one durable dependency-ordered map for
that convergence tranche instead of reconstructing it from the alpha planning
note or scattered issue titles.

## Current Starting Point

The current repo truth is:

1. `Psion`
   - `PSION-1` through `PSION-45` are closed
   - the reference corpus, reference pilot, checkpoint recovery, route/refusal,
     and Google single-node operator lane are all real
   - the current Google proof is still CPU-bound on the L4 host, so broader
     accelerator-backed pretraining remains later
2. `Tassadar` plugin substrate
   - the repo now has five host-native starter plugins, including the first
     user-added capability-free plugin
   - the shared starter-plugin bridge, deterministic workflow controller,
     router-owned tool loop, Apple FM tool lane, and lane-neutral multi-plugin
     trace corpus are all implemented
   - the only fully proved user-authoring path today is still host-native,
     capability-free, and local deterministic
   - the `networked_read_only` class is documented but still intentionally
     narrower and more manual than the capability-free authoring path
   - secret-backed and stateful starter-plugin authoring remain later bounded
     substrate work, not already-closed platform facts
   - weighted-controller and publication posture remain bounded and internal
   - the old guest-artifact / user-provided Wasm path is still not a live
     runtime lane
3. Convergence gap
   - the plugin runtime now emits enough truth to bootstrap learned plugin-use
     training
   - the learned lane is not yet trained on canonical plugin-conditioned task
     records, plugin-use benchmarks, or guest-artifact-aware capability
     boundaries

## Claim Boundary

This tranche is for a **plugin-conditioned learned lane**.

It is not:

- plugin publication closure
- arbitrary software capability closure
- executor-claim widening
- broad GPU-scale pretraining closure
- a claim that the model runs software internally

The learned lane may own:

- plugin discovery
- plugin choice
- plugin argument planning
- plugin sequencing
- request-for-structure behavior
- refusal on unsupported capability
- post-plugin result interpretation

The runtime still owns:

- plugin admission
- loading
- capability mediation
- mount policy
- execution
- receipts
- replay posture
- runtime-side refusal

## Canonical Inputs

The convergence tranche should consume the existing repo truth directly:

- `docs/PSION_PROGRAM_MAP.md`
- `docs/TRAIN_SYSTEM.md`
- `docs/PSION_PLUGIN_CLAIM_BOUNDARY_AND_CAPABILITY_POSTURE.md`
- `docs/PSION_PLUGIN_TRAINING_RECORD_SCHEMA.md`
- `docs/PSION_PLUGIN_TRACE_DERIVATION.md`
- `docs/PSION_PLUGIN_CONDITIONED_DATASET.md`
- `docs/PSION_PLUGIN_CONTAMINATION_CONTROLS.md`
- `docs/PSION_PLUGIN_BENCHMARK_PACKAGES.md`
- `docs/TASSADAR_MULTI_PLUGIN_ORCHESTRATION_WAVE.md`
- `docs/TASSADAR_STARTER_PLUGIN_AUTHORING.md`
- `docs/TASSADAR_STARTER_PLUGIN_CATALOG.md`
- `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md`
- `docs/TASSADAR_STARTER_PLUGIN_TOOL_BRIDGE.md`
- `docs/TASSADAR_STARTER_PLUGIN_WORKFLOW_CONTROLLER.md`
- `docs/TASSADAR_STARTER_PLUGIN_USER_AUTHORING_WAVE.md`
- `docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`
- `docs/TASSADAR_APPLE_FM_PLUGIN_SESSION.md`
- `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md`
- `docs/audits/2026-03-22-tassadar-full-plugin-system-state-audit.md`
- `docs/audits/2026-03-22-psion-training-system-full-state-audit.md`
- the bounded guest packet-ABI / Rust-first guest surface already frozen in:
  `docs/ARCHITECTURE.md` and the `TAS-199` / `TAS-200` era plugin docs

No plugin-program issue should invent a second plugin API, a second trace
format, or a
second tool-call notation just for training.

## Live Substrate Preconditions

The current plugin-system audit already narrowed a few things that later
`PSION_PLUGIN-*` work must not accidentally overstate:

- the live first-class authoring lane is still the capability-free local
  deterministic host-native path
- the `networked_read_only` lane should be treated as an immediate substrate
  proof target, not as something already closed by the current repo state
- secret-backed and stateful plugin classes stay separate bounded follow-on
  substrate work until their host-mediated secret and durable-state contracts
  are explicit
- guest-artifact restoration is still a deliberate later decision wave, not an
  implied property of the current starter-plugin platform

That means the Psion plugin convergence tranche is allowed to build learned
plugin-use training on top of
today's bounded substrate, but it must preserve the current operator-internal,
publication-blocked, non-universal claim boundary while doing so.

## Dependency Order

The Psion plugin convergence program is dependency-ordered in nine tracks:

1. program and claim discipline: `PSION_PLUGIN-1`, `PSION_PLUGIN-2`
2. plugin-training data substrate:
   `PSION_PLUGIN-3`, `PSION_PLUGIN-4`, `PSION_PLUGIN-5`, `PSION_PLUGIN-6`
3. plugin-aware benchmark and eval surfaces:
   `PSION_PLUGIN-7`, `PSION_PLUGIN-8`, `PSION_PLUGIN-9`, `PSION_PLUGIN-10`, `PSION_PLUGIN-11`,
   `PSION_PLUGIN-12`
4. learned plugin-use reference lane:
   `PSION_PLUGIN-13`, `PSION_PLUGIN-14`, `PSION_PLUGIN-15`, `PSION_PLUGIN-16`
5. `networked_read_only` substrate proof: `PSION_PLUGIN-17`
6. narrow guest-artifact Wasm restoration:
   `PSION_PLUGIN-18`, `PSION_PLUGIN-19`, `PSION_PLUGIN-20`, `PSION_PLUGIN-21`, `PSION_PLUGIN-22`,
   `PSION_PLUGIN-23`
7. combined plugin-conditioned lane with guest-artifact coverage:
   `PSION_PLUGIN-24`, `PSION_PLUGIN-25`, `PSION_PLUGIN-26`, `PSION_PLUGIN-27`, `PSION_PLUGIN-28`
8. real operator proof: `PSION_PLUGIN-29`, `PSION_PLUGIN-30`
9. follow-on scale and hardening:
   `PSION_PLUGIN-31`, `PSION_PLUGIN-32`, `PSION_PLUGIN-33`

## Canonical Track Map

### Track 1: Program and claim discipline

- `PSION_PLUGIN-1`: publish the convergence umbrella issue
- `PSION_PLUGIN-2`: freeze the combined claim boundary and capability-matrix posture

### Track 2: Plugin-training data substrate

- `PSION_PLUGIN-3`: canonical plugin-training record schema
- `PSION_PLUGIN-4`: plugin-trace normalization and derivation pipeline
- `PSION_PLUGIN-5`: plugin-conditioned dataset builder
- `PSION_PLUGIN-6`: plugin-aware contamination and derivation controls

### Track 3: Plugin-aware benchmark and eval surfaces

- `PSION_PLUGIN-7`: shared plugin-use benchmark schema and grader interfaces
- `PSION_PLUGIN-8`: plugin discovery and selection benchmark package
- `PSION_PLUGIN-9`: plugin argument construction benchmark package
- `PSION_PLUGIN-10`: plugin sequencing and multi-call benchmark package
- `PSION_PLUGIN-11`: plugin refusal and request-for-structure benchmark package
- `PSION_PLUGIN-12`: plugin result interpretation benchmark package

### Track 4: Learned plugin-use reference lane

- `PSION_PLUGIN-13`: plugin-conditioned SFT stage in `psionic-train`
- `PSION_PLUGIN-14`: plugin-conditioned compact-decoder reference model config
- `PSION_PLUGIN-15`: first host-native starter-plugin-conditioned reference lane
  for the currently fully proved authoring class: host-native, capability-free,
  local deterministic starter plugins
- `PSION_PLUGIN-16`: capability matrix and served posture v1 for the host-native
  plugin-conditioned model, explicitly distinguishing supported host-native
  capability-free behavior from not-yet-proved `networked_read_only` behavior
  and unsupported secret-backed, stateful, and guest-artifact classes

### Track 5: `networked_read_only` substrate proof

- `PSION_PLUGIN-17`: prove the first end-to-end `networked_read_only`
  user-authored plugin lane across runtime, bridge, catalog, controller, and
  weighted-controller truth without widening publication or authoring claims

### Track 6: Narrow guest-artifact Wasm restoration

- `PSION_PLUGIN-18`: make an explicit product decision that guest-artifact
  restoration is a later separate bounded lane and not present-tense
  starter-plugin truth
- `PSION_PLUGIN-19`: digest-bound guest-artifact manifest and identity contract
- `PSION_PLUGIN-20`: bounded guest-artifact runtime loading path
- `PSION_PLUGIN-21`: receipt-equivalent guest-artifact invocation path
- `PSION_PLUGIN-22`: one user-provided Wasm starter plugin end to end
- `PSION_PLUGIN-23`: guest-artifact trust-tier and authority-gate rows

### Track 7: Combined plugin-conditioned lane with guest-artifact coverage

- `PSION_PLUGIN-24`: extend derivation pipeline for guest-artifact traces
- `PSION_PLUGIN-25`: mixed host-native plus guest-artifact dataset v1
- `PSION_PLUGIN-26`: mixed plugin-conditioned model v1
- `PSION_PLUGIN-27`: guest-plugin benchmark package
- `PSION_PLUGIN-28`: mixed capability matrix v2

### Track 8: Real operator proof

- `PSION_PLUGIN-29`: first real plugin-conditioned Google single-node training
  audit
- `PSION_PLUGIN-30`: first real mixed guest-artifact plugin-conditioned Google
  audit

### Track 9: Follow-on scale and hardening

- `PSION_PLUGIN-31`: plugin-conditioned route/refusal hardening tranche
- `PSION_PLUGIN-32`: expand weighted-controller corpus only for new plugin classes
- `PSION_PLUGIN-33`: decide whether cluster-scale plugin-conditioned training is
  warranted

## Operating Rules

Every `PSION_PLUGIN-*` issue must preserve all of the following:

- training examples come from real plugin-runtime truth, not mock string tool
  calls
- plugin-use traces preserve receipt linkage and replay class
- issue language and capability matrices distinguish proved, not-yet-proved,
  and unsupported plugin classes instead of flattening them into one platform
  claim
- route and refusal remain explicit and benchmarked
- the learned lane never implies execution without receipts
- guest-artifact restoration stays digest-bound, trust-tiered, and publication-
  blocked unless a later explicit tranche changes that
- no issue widens this lane into arbitrary plugin marketplace or arbitrary
  software capability language
