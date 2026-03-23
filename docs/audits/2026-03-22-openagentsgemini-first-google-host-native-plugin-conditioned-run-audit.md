# OpenAgentsGemini First Google Host-Native Plugin-Conditioned Run Audit

> Status: follow-up `PSION_PLUGIN-29` audit written 2026-03-22 after the first
> real Google-hosted host-native plugin-conditioned Psion run completed on
> `openagentsgemini` and the instance was deleted.

## Scope

This audit covers one bounded real Google single-node execution of the
host-native plugin-conditioned Psion reference lane.

It claims:

- one real Google-hosted host-native plugin-conditioned run exists
- the run preserved launch truth, input-package truth, stage truth,
  plugin-evaluation truth, archive truth, and teardown truth
- the run preserved the host-native capability boundary instead of widening into
  mixed or guest-artifact claims

It does not claim:

- mixed guest-artifact plugin-conditioned Google proof
- dense accelerator-backed training throughput proof
- trusted-cluster readiness
- plugin publication enablement
- broad cost truth for this run via billing-export query data

## Typed Outcome

- result classification: `bounded_success`
- run id: `psion-plugin-host-native-g2-l4-20260323t015231z`
- lane id: `psion_plugin_conditioned_host_native_reference`
- repo revision: `44590827a0e729cbbcee1186356d92d322c44791`

## Topology

- project: `openagentsgemini`
- zone: `us-central1-a`
- machine type: `g2-standard-8`
- accelerator: `nvidia-l4`
- accelerator count: `1`
- observed GPU name: `NVIDIA L4`
- driver version: `570.211.01`
- boot disk: `pd-balanced`, `200 GB`
- external IP: `false`

## Timeline

UTC timestamps from the retained run timeline:

- launch created: `2026-03-23T01:52:41Z`
- bootstrap started: `2026-03-23T01:53:39Z`
- bootstrap finished: `2026-03-23T01:54:38Z`
- training started: `2026-03-23T01:54:38Z`
- training finished: `2026-03-23T02:02:45Z`
- archive completed: `2026-03-23T02:03:11Z`
- teardown started: `2026-03-23T02:03:11Z`
- teardown finished: `2026-03-23T02:03:14Z`
- final manifest written: `2026-03-23T02:04:16Z`

The instance was deleted after finalization via:

- `bash scripts/psion-google-delete-single-node.sh --run-id psion-plugin-host-native-g2-l4-20260323t015231z --force`

## Input Truth

The run used the committed plugin host-native descriptor:

- descriptor URI:
  `gs://openagentsgemini-psion-train-us-central1/manifests/psion_google_plugin_host_native_input_package_v1.json`
- package id:
  `psion-google-plugin-host-native-inputs-1a5177c34d2d-20260323t015150z`
- archive SHA-256:
  `d831593e3e487efefeef3f2c3d91a08d1391ce5b58ae146f7f72301e5107c95f`
- manifest SHA-256:
  `3df6d43b1b5ace076cef0dd1d6c2321137ef67ab96836ed47da307033ab3f339`

The launch manifest bound the training command explicitly:

```bash
cargo run -p psionic-train --example psion_google_plugin_host_native_reference_run -- "$PSION_OUTPUT_DIR"
```

The launch manifest also bound the archive posture explicitly:

```bash
bash "$PSION_REPO_DIR/scripts/psion-google-archive-plugin-conditioned-run.sh" \
  --stem psion_plugin_host_native_reference \
  --manifest-out "$PSION_SCRATCH_DIR/psion_google_checkpoint_archive_manifest.json" \
  "$PSION_OUTPUT_DIR"
```

The default restore path was explicitly disabled for this lane with
`--post-training-restore-command __none__` because this bounded host-native
lane retains logical checkpoint evidence only and does not claim the
reference-pilot dense-checkpoint cold-restore surface.

## Host-Native Run Facts

Retained run summary:

- run id: `run-psion-plugin-host-native-reference`
- dataset ref:
  `dataset://openagents/psion/plugin_conditioned_host_native_reference`
- stable dataset identity:
  `dataset://openagents/psion/plugin_conditioned_host_native_reference@2026.03.22.v1`
- training example count: `3`
- learned plugin ids:
  - `plugin.feed.rss_atom_parse`
  - `plugin.html.extract_readable`
  - `plugin.text.stats`
  - `plugin.text.url_extract`
- benchmark family count: `5`
- model artifact digest:
  `dfb6e226a35b32cc0f5a3ec66a9213d3bff9080fdd035fb40797c7a4b7a9c0d1`
- evaluation receipt digest:
  `bc2290f007274265cf4f5876e5817a6a81580be3cc045fc7396fea7a896052b2`
- run bundle digest:
  `97e269df9724248e18a8f08b2066721193a09397ed7c550e8983a5d18031a797`

The evaluation receipt stayed explicitly inside the proved host-native class:

- baseline label: `non_plugin_conditioned_baseline_v1`
- limited to proved authoring class: `true`
- proved authoring class label:
  `host_native_capability_free_local_deterministic`

Benchmark deltas:

- `discovery_selection`: baseline `6666`, trained `10000`, delta `3334`
- `argument_construction`: baseline `0`, trained `10000`, delta `10000`
- `sequencing_multi_call`: eligible `0`, trained `0`, delta `0`
- `refusal_request_structure`: baseline `6000`, trained `10000`, delta `4000`
- `result_interpretation`: baseline `0`, trained `10000`, delta `10000`

## Archive Truth

This run did retain checkpoint-linked archive truth, but only in the bounded
logical form this lane honestly supports.

Archive manifest:

- archive manifest URI:
  `gs://openagentsgemini-psion-train-us-central1/checkpoints/plugin_conditioned/psion-plugin-conditioned-host-native-reference/run-psion-plugin-host-native-reference/checkpoint-psion-plugin-host-native-reference-1/archive/psion_google_plugin_conditioned_archive_manifest.json`
- archive manifest SHA-256:
  `313b3de629e5f921965db12a56d83560b2c6e97d7bedcb6277a859ab192ab98e`
- archive mode: `logical_checkpoint_evidence_only`
- checkpoint family: `train.psion.plugin_host_native_reference`
- latest checkpoint ref:
  `checkpoint://psion/plugin_host_native_reference/1`
- latest checkpoint step: `1`
- checkpoint ref count: `1`
- stage receipt digest:
  `4fc7ae658c40fa0ab1184965b701a65021705772a6d6446787aa3aff2c9f704a`

Archived objects:

- bounded run bundle
- bounded stage bundle
- stage receipt
- model artifact
- evaluation receipt
- logical checkpoint evidence manifest
- bounded run summary

This is enough to preserve checkpoint-lineage and stage-lineage truth for the
host-native lane without claiming a dense tensor checkpoint that the
plugin-conditioned reference lane does not actually emit.

## GPU Reality

The run used a real L4 host, but the retained GPU samples prove that this lane
was still CPU-bound:

- sample count: `102`
- average GPU utilization: `0%`
- max GPU utilization: `0%`
- average GPU memory utilization: `0%`
- max observed GPU memory used: `0 MiB`

That means the operator path is real, but this is still not evidence of
accelerator-backed plugin-conditioned training throughput.

## Cost Truth

Cost truth for this specific run remains partial.

What is real:

- the launch profile ceiling was explicit at `15 USD`
- quota, budget-topic, and price-profile preflight were all retained in the
  launch manifest

What is still missing:

- the expected machine-queryable billing-export table was not present in
  `openagentsgemini:psion_training_finops`
- the dataset currently exposed only `single_node_price_profiles_v1`

So this audit does not claim query-backed realized cost for the run itself.

## Evidence Locations

Primary retained objects:

- final manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-host-native-g2-l4-20260323t015231z/final/psion_google_run_final_manifest.json`
- manifest of manifests:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-host-native-g2-l4-20260323t015231z/final/psion_google_run_manifest_of_manifests.json`
- launch manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-host-native-g2-l4-20260323t015231z/launch/psion_google_single_node_launch_manifest.json`
- run summary:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-host-native-g2-l4-20260323t015231z/receipts/psion_plugin_host_native_reference_run_summary.json`
- evaluation receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-host-native-g2-l4-20260323t015231z/receipts/psion_plugin_host_native_reference_evaluation_receipt.json`
- stage receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-host-native-g2-l4-20260323t015231z/receipts/psion_plugin_host_native_reference_stage_receipt.json`
- checkpoint evidence:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-host-native-g2-l4-20260323t015231z/receipts/psion_plugin_host_native_reference_checkpoint_evidence.json`
- archive manifest:
  `gs://openagentsgemini-psion-train-us-central1/checkpoints/plugin_conditioned/psion-plugin-conditioned-host-native-reference/run-psion-plugin-host-native-reference/checkpoint-psion-plugin-host-native-reference-1/archive/psion_google_plugin_conditioned_archive_manifest.json`

## Conclusion

`PSION_PLUGIN-29` is now closed in the narrow form it was supposed to prove.

The repo now has:

- one real Google-hosted host-native plugin-conditioned run
- retained launch, host, input, stage, eval, archive, and teardown evidence
- an explicit `bounded_success` audit classification

The repo still does not have:

- mixed guest-artifact Google proof
- dense accelerator-backed plugin-conditioned training proof
- query-backed realized cost truth for this run

That is the correct boundary.

## Next Steps

- run the first real mixed guest-artifact plugin-conditioned Google audit
- harden the plugin-conditioned route/refusal tranche against overdelegation and
  execution implication
- fix the missing billing-export surface so later Google audits can preserve
  machine-queryable realized cost truth
