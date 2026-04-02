# Psion Actual Pretraining Recipe

> Status: canonical actual-lane recipe and topology/storage bundles, written
> 2026-04-02 after freezing one repo-owned recipe id and one admitted
> topology/storage bundle for `psion_actual_pretraining_v1`.

This document freezes the machine-readable recipe authority for the canonical
actual `Psion` pretraining lane.

It does not claim that the launcher, backup, or auto-eval work is already
finished. It does fix one recipe bundle, one topology/storage bundle, and one
bounded continuation path that later launcher and hardening work must consume.

## Canonical Artifacts

- `crates/psionic-train/src/psion_actual_pretraining_recipe_bundle.rs` owns the
  recipe and topology/storage bundle contracts.
- `crates/psionic-train/examples/psion_actual_pretraining_recipe_bundle_fixtures.rs`
  regenerates the committed bundles.
- `fixtures/psion/pretrain/psion_actual_pretraining_recipe_bundle_v1.json`
  carries the canonical actual-lane recipe bundle.
- `fixtures/psion/pretrain/psion_actual_pretraining_topology_storage_bundle_v1.json`
  carries the canonical actual-lane topology/storage bundle.

Stable schema versions:

- `psion.actual_pretraining_recipe_bundle.v1`
- `psion.actual_pretraining_topology_storage_bundle.v1`

## Frozen Recipe Authority

The actual-lane recipe is:

- recipe id: `psion_actual_pretraining_recipe_v1`
- lane id: `psion_actual_pretraining_v1`
- model id: `psion-compact-decoder-internal-v1`
- tokenizer id: `psion_sentencepiece_seed`
- dataset identity: `psion_corpus_tokenized@v1`
- sampling policy: `psion_pretrain_mix@v1`
- base stage kinds: `pretrain`
- bounded token budget: `1073741824` train / `33554432` validation /
  `8388608` held-out
- bounded optimizer steps: `16384`
- max context tokens: `8192`

The bundle is anchored directly to the broader-pretraining trusted-cluster run
bundle instead of inventing a second synthetic recipe.

## Frozen Topology And Storage Authority

The admitted topology/storage bundle is:

- bundle id: `psion_actual_pretraining_topology_storage_bundle_v1`
- topology label: `homogeneous_four_node_h100_tensor_parallel`
- backend: `cuda`
- worker count: `4`
- placement shape: `tensor_parallel(world_size=4,ranks_per_node=1)`
- remote run root template:
  `${PSION_ACTUAL_PRETRAINING_BUCKET_URL}/psion_actual_pretraining_runs/<run_id>`

The storage bundle declares durable checkpoint and manifest tiers plus a
transient log tier.

The storage and remote-backend credential sources are declared by name only:

- `PSION_ACTUAL_PRETRAINING_GCP_PROJECT_ID`
- `PSION_ACTUAL_PRETRAINING_BUCKET_URL`
- `GOOGLE_APPLICATION_CREDENTIALS`

The bundle records only source names and redaction posture. It does not embed
credentials or raw bucket secrets.

## Bounded Continuation Path

The actual-lane recipe now declares one bounded continuation path:

- `pretrain -> general_sft -> agentic_sft`

That path is anchored to:

- `fixtures/psion/sft/psion_reasoning_sft_run_bundle_v1.json`
- `fixtures/psion/plugins/training/psion_plugin_conditioned_sft_v1/psion_plugin_conditioned_sft_stage_manifest.json`
- `fixtures/psion/plugins/training/psion_plugin_conditioned_sft_v1/psion_plugin_conditioned_sft_run_bundle.json`

This keeps the plugin-conditioned target explicit without pretending the actual
pretraining lane already proves cluster-scale plugin-conditioned continuation.

## Why This Matters

Without one frozen recipe bundle and one admitted topology/storage bundle,
later launcher, resume, eval, and CS336-port work would keep reconstructing
what the actual lane is supposed to run.

This document prevents that drift by fixing the model, tokenizer, mixture,
token budget, topology, credential-source declaration, and bounded
continuation target in committed machine-readable form.
