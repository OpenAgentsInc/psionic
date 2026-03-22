# TASSION Program Map

> Status: canonical proposed convergence map for the next `Psion x Tassadar`
> plugin-conditioned training tranche, written 2026-03-22 from
> `~/code/alpha/tassadar/TASSION.md` and adjusted to current repo truth.

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
- `docs/TASSADAR_MULTI_PLUGIN_ORCHESTRATION_WAVE.md`
- `docs/TASSADAR_STARTER_PLUGIN_RUNTIME.md`
- `docs/TASSADAR_STARTER_PLUGIN_TOOL_BRIDGE.md`
- `docs/TASSADAR_ROUTER_PLUGIN_TOOL_LOOP.md`
- `docs/TASSADAR_APPLE_FM_PLUGIN_SESSION.md`
- `docs/TASSADAR_MULTI_PLUGIN_TRACE_CORPUS.md`
- `docs/audits/2026-03-22-tassadar-full-plugin-system-state-audit.md`
- `docs/audits/2026-03-22-psion-training-system-full-state-audit.md`
- the bounded guest packet-ABI / Rust-first guest surface already frozen in:
  `docs/ARCHITECTURE.md` and the `TAS-199` / `TAS-200` era plugin docs

No TASSION issue should invent a second plugin API, a second trace format, or a
second tool-call notation just for training.

## Dependency Order

The `TASSION` convergence program is dependency-ordered in eight tracks:

1. program and claim discipline: `TASSION-1`, `TASSION-2`
2. plugin-training data substrate:
   `TASSION-3`, `TASSION-4`, `TASSION-5`, `TASSION-6`
3. plugin-aware benchmark and eval surfaces:
   `TASSION-7`, `TASSION-8`, `TASSION-9`, `TASSION-10`, `TASSION-11`,
   `TASSION-12`
4. learned plugin-use reference lane:
   `TASSION-13`, `TASSION-14`, `TASSION-15`, `TASSION-16`
5. narrow guest-artifact Wasm restoration:
   `TASSION-17`, `TASSION-18`, `TASSION-19`, `TASSION-20`, `TASSION-21`,
   `TASSION-22`
6. combined plugin-conditioned lane with guest-artifact coverage:
   `TASSION-23`, `TASSION-24`, `TASSION-25`, `TASSION-26`, `TASSION-27`
7. real operator proof: `TASSION-28`, `TASSION-29`
8. follow-on scale and hardening:
   `TASSION-30`, `TASSION-31`, `TASSION-32`

## Canonical Track Map

### Track 1: Program and claim discipline

- `TASSION-1`: publish the convergence umbrella issue
- `TASSION-2`: freeze the combined claim boundary and capability-matrix posture

### Track 2: Plugin-training data substrate

- `TASSION-3`: canonical plugin-training record schema
- `TASSION-4`: plugin-trace normalization and derivation pipeline
- `TASSION-5`: plugin-conditioned dataset builder
- `TASSION-6`: plugin-aware contamination and derivation controls

### Track 3: Plugin-aware benchmark and eval surfaces

- `TASSION-7`: shared plugin-use benchmark schema and grader interfaces
- `TASSION-8`: plugin discovery and selection benchmark package
- `TASSION-9`: plugin argument construction benchmark package
- `TASSION-10`: plugin sequencing and multi-call benchmark package
- `TASSION-11`: plugin refusal and request-for-structure benchmark package
- `TASSION-12`: plugin result interpretation benchmark package

### Track 4: Learned plugin-use reference lane

- `TASSION-13`: plugin-conditioned SFT stage in `psionic-train`
- `TASSION-14`: plugin-conditioned compact-decoder reference model config
- `TASSION-15`: first host-native starter-plugin-conditioned reference lane
- `TASSION-16`: capability matrix and served posture v1 for the host-native
  plugin-conditioned model

### Track 5: Narrow guest-artifact Wasm restoration

- `TASSION-17`: freeze guest-artifact product direction
- `TASSION-18`: digest-bound guest-artifact manifest and identity contract
- `TASSION-19`: bounded guest-artifact runtime loading path
- `TASSION-20`: receipt-equivalent guest-artifact invocation path
- `TASSION-21`: one user-provided Wasm starter plugin end to end
- `TASSION-22`: guest-artifact trust-tier and authority-gate rows

### Track 6: Combined plugin-conditioned lane with guest-artifact coverage

- `TASSION-23`: extend derivation pipeline for guest-artifact traces
- `TASSION-24`: mixed host-native plus guest-artifact dataset v1
- `TASSION-25`: mixed plugin-conditioned model v1
- `TASSION-26`: guest-plugin benchmark package
- `TASSION-27`: mixed capability matrix v2

### Track 7: Real operator proof

- `TASSION-28`: first real plugin-conditioned Google single-node training audit
- `TASSION-29`: first real mixed guest-artifact plugin-conditioned Google audit

### Track 8: Follow-on scale and hardening

- `TASSION-30`: plugin-conditioned route/refusal hardening tranche
- `TASSION-31`: expand weighted-controller corpus only for new plugin classes
- `TASSION-32`: decide whether cluster-scale plugin-conditioned training is
  warranted

## Operating Rules

Every `TASSION-*` issue must preserve all of the following:

- training examples come from real plugin-runtime truth, not mock string tool
  calls
- plugin-use traces preserve receipt linkage and replay class
- route and refusal remain explicit and benchmarked
- the learned lane never implies execution without receipts
- guest-artifact restoration stays digest-bound, trust-tiered, and publication-
  blocked unless a later explicit tranche changes that
- no issue widens this lane into arbitrary plugin marketplace or arbitrary
  software capability language
