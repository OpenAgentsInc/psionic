# Probe GEPA Candidate Manifests

Status: implemented_early

Psionic now owns the first manifest contract for GEPA text-bundle candidates
that improve Probe and Blueprint behavior without changing model weights.
This is a distributed benchmark-driven optimization surface, not distributed
neural-network training.

The current implementation lives in
`crates/psionic-train/src/probe_gepa_candidate_manifest.rs`. It defines
`psionic.probe_gepa_candidate_manifest.v1` and a retained Stage 0/1 seed
fixture at
`fixtures/probe/gepa/probe_gepa_candidate_manifest_stage_0_1_seed_v1.json`.

## Contract

The manifest content-addresses these text components:

- `probe_system_prompt`
- `terminal_bench_global_playbook`
- `signature_selection_policy`
- `tool_menu_policy`
- `patch_and_test_policy`
- `failure_family_playbooks`
- `closeout_policy`

Each component has a stable SHA-256 hash. The candidate hash is computed from
component hashes plus campaign, split, trace, import, and safety refs. The
manifest hash is computed over the complete manifest including the candidate
text.

## Promotion Boundary

Optimizer acceptance is separate from runtime promotion.

- `optimizer_acceptance_state` records whether GEPA accepted the text bundle.
- `runtime_promotion_state` records whether Probe/Omega has admitted the
  candidate into shadow, release-candidate, active, or reverted runtime state.
- `promotion_state` is the combined lifecycle label used by downstream
  importers.

An optimizer-accepted candidate is still `not_promoted` until external release
gates pass. An `active` candidate must carry `policy_gate_state = passed`.

## Import Boundary

The manifest carries refs that Probe and benchmark-cloud can import:

- Probe import refs for prompt, Blueprint, tool-menu, and loop-policy
  candidate objects.
- benchmark-cloud import refs for split manifests, run manifests, and artifact
  contract refs.

These are refs only. The candidate cannot grant new runtime authority, bypass
release gates, carry raw secrets, or upgrade public benchmark claims.

## Regeneration

Regenerate the retained seed fixture with:

```bash
cargo run -q -p psionic-train --example probe_gepa_candidate_manifest_fixture -- \
  fixtures/probe/gepa/probe_gepa_candidate_manifest_stage_0_1_seed_v1.json
```

Verify the contract with:

```bash
cargo test -p psionic-train probe_gepa_candidate_manifest --lib
```
