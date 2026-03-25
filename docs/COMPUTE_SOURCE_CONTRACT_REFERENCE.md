# Cross-Provider Compute-Source Contract Reference

> Status: canonical `XTRAIN-2` / `#518` record, updated 2026-03-25 after
> landing the first typed provider-neutral compute-source contract family in
> `crates/psionic-train/src/cross_provider_compute_source_contract.rs`.

This document records the shared training-facing machine contract above the
current Google, RunPod, local NVIDIA, and local Apple lane artifacts.

## Canonical Runner

Run the checker from the repo root:

```bash
scripts/check-cross-provider-compute-source-contracts.sh
```

## What Landed

`psionic-train` now owns one typed compute-source contract family that can
admit or refuse the same execution classes across different providers and local
machines.

The landed surface includes:

- `CrossProviderComputeSourceContract`
- `CrossProviderComputeProviderKind`
- `CrossProviderLocalityKind`
- `CrossProviderBackendFamily`
- `CrossProviderComputeSourceNetworkPosture`
- `CrossProviderComputeSourceStoragePosture`
- `CrossProviderComputeSourceCostPosture`
- `CrossProviderExecutionClassAdmissionRefusal`
- `CrossProviderPlannerInput`
- `CrossProviderLaunchInput`
- the binary `cross_provider_compute_source_contracts`
- the checker `scripts/check-cross-provider-compute-source-contracts.sh`
- the canonical example fixtures in `fixtures/training/compute_sources/`

## Canonical Execution Classes

The contract keeps the execution-class vocabulary frozen to the same six values
already admitted by the root program manifest:

- `dense_full_model_rank`
- `validated_contributor_window`
- `validator`
- `checkpoint_writer`
- `eval_worker`
- `data_builder`

Every unsupported class now needs one typed refusal example in the source
contract. That keeps local Apple, local NVIDIA, Google, and RunPod source
comparisons explicit instead of letting operator scripts silently widen what a
machine can claim.

## Canonical Example Sources

The first example family is intentionally grounded in retained repo evidence:

- `google_l4_validator_node_v1.json`
  Built from the Google two-node swarm contract, launch profile, network
  posture, and identity profile.
- `runpod_8xh100_dense_node_v1.json`
  Built from the retained RunPod 8xH100 launch profile, operator preflight, and
  cost guardrails.
- `local_rtx4080_workstation_v1.json`
  Built from the retained local RTX 4080 bring-up report.
- `local_mlx_mac_workstation_v1.json`
  Built from the retained local Mac MLX bring-up report.

## Planner And Launcher Binding

This issue does not add provider APIs or runtime execution changes. It does add
the first shared binding layer above provider-specific details:

- `planner_input_v1.json`
  One provider-neutral planner input that uses the same contract family to
  admit a Google node, a RunPod pod, a local RTX host, and a local Mac, while
  also retaining one refused `dense_full_model_rank` request for the Mac.
- `launch_inputs_v1.json`
  One provider-neutral launch-input family that projects the root program
  manifest plus one admitted source into run id, environment package key,
  artifact roots, and optional cluster-port binding.

This keeps provider-specific launchers on the hook for resource creation only.
Training-facing machine truth now sits in one typed contract family instead of
being split between Google-only, RunPod-only, and local-only artifacts.

## Current Limits

This issue intentionally does not claim:

- shared provider-neutral launch scripts
- dense-rank runtime closure
- remote artifact backend closure
- mixed-backend dense admission
- public or adversarial swarm admission

This issue closes the machine-contract layer first.
