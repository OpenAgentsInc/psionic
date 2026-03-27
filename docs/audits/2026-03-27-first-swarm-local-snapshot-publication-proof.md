# 2026-03-27 First Swarm Local Snapshot Publication Proof

This audit records the new retained publication proof for the frozen
first-swarm local snapshot target.

It is intentionally separate from the retained mixed-hardware live run:

- the live run still truthfully ends with `publish=refused`
- the live run still truthfully ends with `promotion=held`
- this proof only answers the narrower missing question:
  can the repo materialize one honest local snapshot directory for the frozen
  first-swarm publish target using the existing publish surface

## Final Verdict

Yes.

The repo now retains one truthful local snapshot publication proof for:

- publish id: `first-swarm-local-snapshot`
- target: `hugging_face_snapshot`
- repo id: `openagents/swarm-local-open-adapter`
- snapshot root:
  `local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot`

The proof is machine-checkable, drift-detectable, and tied directly to the
same first-swarm publish config already named in the historical closeout and
workflow-plan artifacts.

## Artifacts Reviewed

- `fixtures/swarm/publications/first_swarm_local_snapshot_publication_v1.json`
- `fixtures/swarm/publications/local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot/model.safetensors`
- `fixtures/swarm/publications/local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot/psionic_bundle_manifest.json`
- `fixtures/swarm/publications/local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot/tokenizer_contract.json`
- `fixtures/swarm/publications/local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot/compatibility_contract.json`
- `fixtures/swarm/publications/local_publish/openagents_swarm_local_open_adapter/first-swarm-local-snapshot/publish_manifest.json`
- `fixtures/swarm/first_swarm_live_workflow_plan_v1.json`
- `scripts/check-first-swarm-local-snapshot-publication.sh`
- `crates/psionic-mlx-workflows/src/bin/first_swarm_local_snapshot_publication.rs`
- `crates/psionic-mlx-workflows/src/swarm_live_plan.rs`

For contrast, this proof remains explicitly downstream of the retained real
mixed-hardware run artifacts:

- `fixtures/swarm/runs/first-swarm-live-20260327-real-2/first_swarm_real_run_bundle.json`
- `docs/audits/2026-03-27-first-swarm-trusted-lan-real-run-audit.md`

## What Actually Ran

The proof path does four bounded things:

1. rebuild the canonical first-swarm live workflow plan in-process
2. construct one deterministic portable merge pair for the first-swarm
   open-adapter family
3. merge that pair through
   `psionic-mlx-workflows::MlxWorkflowWorkspace::merge_adapter`
4. publish the merged portable bundle through
   `psionic-mlx-workflows::MlxWorkflowWorkspace::publish_bundle`

That produces one retained local Hugging Face-style snapshot directory and one
machine-readable publication report.

## Retained Output

The retained publication report records:

- workflow-plan digest:
  `491bb0db695cfd4b32f7789e94139de908918c0b711e411f53d8318fc7e62386`
- publish manifest digest:
  `sha256:26bc09e78a04f120ac06c028042b8cb97a22bd7d4ef9dc591efbf227205b4275`
- merged state-dict digest:
  `a49be837fd798cea27d25b25a1ba9550490bfc778fe6865f439aa30228c38824`
- merged artifact digest:
  `b43355e310a667a80f27266195d0e54a5a046dcc42d415b2e206a8db15154541`
- report digest:
  `06093a6fe2e1b061444ee373b385263ef5d12be908e1a6c058a6a83bfbc6e735`

The retained snapshot directory contains:

- `model.safetensors`
- `psionic_bundle_manifest.json`
- `tokenizer_contract.json`
- `compatibility_contract.json`
- `README.md`
- `publish_manifest.json`

## What Was Tested

The publication proof was validated three ways:

1. targeted build:
   `cargo build -q -p psionic-mlx-workflows --bin first_swarm_local_snapshot_publication`
2. targeted Rust tests:
   `cargo test -q -p psionic-mlx-workflows first_swarm_local_snapshot_publication -- --nocapture`
3. retained proof drift check:
   `scripts/check-first-swarm-local-snapshot-publication.sh`

The third command rebuilds the proof into a temp directory and compares:

- the generated report against the committed report
- the generated publication tree against the committed publication tree

That means the proof is not just a retained directory; it is generator-backed.

## What This Proof Now Proves

This proof now proves:

- the frozen first-swarm publish config is executable
- the repo can write one real local Hugging Face-style snapshot directory for
  the first-swarm target
- the publish path retains a machine-readable manifest, tokenizer contract,
  compatibility contract, and portable bundle manifest
- the publication proof is stable enough to treat as retained repo truth

## What This Proof Still Does Not Prove

This proof still does not prove:

- that the retained mixed-hardware live run earned publication
- that publication happens automatically after a future promoted live run
- that any served-model registry or remote hosting step has occurred
- that the lane now supports internet discovery, elastic membership, or
  full-model mixed-backend dense training

Those remain separate goals.

## Bottom Line

The old first-swarm “publication path exists only on paper” gap is now closed.

The repo still truthfully keeps the retained mixed-hardware run at
`publish=refused`, but it also now retains one separate honest publication
proof for the frozen local snapshot target. That is the right claim boundary.
