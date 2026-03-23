# OpenAgentsGemini First Google Mixed Plugin-Conditioned Run Audit

> Status: follow-up `PSION_PLUGIN-30` audit written 2026-03-22 after the first
> real Google-hosted mixed host-native plus guest-artifact plugin-conditioned
> Psion run completed on `openagentsgemini` and the instance was deleted.

## Scope

This audit covers one bounded real Google single-node execution of the mixed
plugin-conditioned Psion reference lane.

It claims:

- one real Google-hosted mixed plugin-conditioned run exists
- the run preserved launch truth, input-package truth, stage truth, mixed
  comparison truth, guest capability-boundary truth, archive truth, and
  teardown truth
- the run preserved the explicit mixed capability boundary rather than
  flattening host-native and guest-artifact evidence into a vague plugin claim

It does not claim:

- dense accelerator-backed mixed training throughput proof
- trusted-cluster readiness
- plugin publication enablement
- generic guest-artifact loading support
- query-backed realized cost truth for this run

## Typed Outcome

- result classification: `bounded_success`
- Google run id: `psion-plugin-mixed-g2-l4-20260323t021022z`
- lane id: `psion_plugin_conditioned_mixed_reference`
- repo revision: `1322953f353d32b618a58914bd91ddba655622ea`

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

- launch created: `2026-03-23T02:10:32Z`
- bootstrap started: `2026-03-23T02:11:21Z`
- bootstrap finished: `2026-03-23T02:12:25Z`
- training started: `2026-03-23T02:12:25Z`
- training finished: `2026-03-23T02:20:14Z`
- archive completed: `2026-03-23T02:20:40Z`
- teardown started: `2026-03-23T02:20:40Z`
- teardown finished: `2026-03-23T02:20:43Z`
- final manifest written: `2026-03-23T02:21:53Z`

The instance was deleted after finalization via:

- `bash scripts/psion-google-delete-single-node.sh --run-id psion-plugin-mixed-g2-l4-20260323t021022z --force`

## Input Truth

The run used the committed mixed plugin descriptor:

- descriptor URI:
  `gs://openagentsgemini-psion-train-us-central1/manifests/psion_google_plugin_mixed_input_package_v1.json`
- package id:
  `psion-google-plugin-mixed-inputs-1322953f353d-20260323t021007z`
- archive SHA-256:
  `c6e589c125f52601d7189a393139a88e416df2c69e97e3646aa70a1af17c95a6`
- manifest SHA-256:
  `728e4f6292fa4af15cda15add72486f20ebc9ecfe61b264191cbf7c73e148343`

The launch manifest bound the mixed training command explicitly:

```bash
cargo run -p psionic-train --example psion_google_plugin_mixed_reference_run -- "$PSION_OUTPUT_DIR"
```

The launch manifest bound the mixed archive posture explicitly:

```bash
bash "$PSION_REPO_DIR/scripts/psion-google-archive-plugin-conditioned-run.sh" \
  --stem psion_plugin_mixed_reference \
  --manifest-out "$PSION_SCRATCH_DIR/psion_google_checkpoint_archive_manifest.json" \
  "$PSION_OUTPUT_DIR"
```

The default restore path was explicitly disabled with
`--post-training-restore-command __none__` because the mixed lane still retains
logical checkpoint evidence only and does not prove the reference-pilot
dense-checkpoint cold-restore surface.

## Mixed Run Facts

Retained run summary:

- run id: `run-psion-plugin-mixed-reference`
- dataset ref:
  `dataset://openagents/psion/plugin_conditioned_mixed_reference`
- stable dataset identity:
  `dataset://openagents/psion/plugin_conditioned_mixed_reference@2026.03.22.v1`
- training example count: `4`
- guest-artifact training example count: `1`
- learned plugin ids:
  - `plugin.example.echo_guest`
  - `plugin.feed.rss_atom_parse`
  - `plugin.html.extract_readable`
  - `plugin.http.fetch_text`
  - `plugin.text.stats`
  - `plugin.text.url_extract`
- guest benchmark receipt digest:
  `439d120ebba64177a9b59cc18b962633748f441d518d861ffdc2bedae057973b`
- model artifact digest:
  `dc3bf10c619fa6b9ae980890844923c7adf166ddf79d17cf8da6e99604083703`
- mixed evaluation receipt digest:
  `50540012758271d92490b7c4f93b51900f34338fa4ce6e498730aab15ebca8e7`
- mixed capability matrix digest:
  `4681ddf8469e375e4e4012677962769159111548c1fe8df17d818ae7907e36c7`
- mixed served posture digest:
  `f9a68bebf735192f4ca09faa05df1381c6e29e01b37ae2366735e97a3997516c`
- run bundle digest:
  `4a174051e64af60958ad5cd23c40532b7ad35aeeff1799646e223ebc0697f451`

The mixed evaluation receipt kept the host-native comparison explicit:

- comparison label: `psion_plugin_host_native_reference`
- guest-artifact training example count: `1`

Benchmark comparisons:

- `discovery_selection`: host-native `10000`, mixed `10000`, delta `0`
- `argument_construction`: host-native `10000`, mixed `10000`, delta `0`
- `sequencing_multi_call`: host-native `0`, mixed `10000`, delta `10000`
- `refusal_request_structure`: host-native `10000`, mixed `10000`, delta `0`
- `result_interpretation`: host-native `10000`, mixed `10000`, delta `0`

## Guest-Artifact Evidence

This run retained class-specific mixed evidence that the host-native run did
not:

- guest benchmark package id: `psion.plugin.guest_capability_boundary.v1`
- guest benchmark package digest:
  `6d8820a5d3d320e11ff5034de169ea3e839221e14958d7262091f68cba311c12`
- guest benchmark receipt id:
  `receipt.psion.plugin.guest_capability_boundary.reference.v1`
- guest benchmark receipt digest:
  `439d120ebba64177a9b59cc18b962633748f441d518d861ffdc2bedae057973b`

Retained guest capability-boundary metric kinds:

- `guest_plugin_admitted_use_accuracy_bps`
- `guest_plugin_unsupported_load_refusal_accuracy_bps`
- `guest_plugin_publication_boundary_accuracy_bps`
- `guest_plugin_arbitrary_binary_boundary_accuracy_bps`
- `guest_plugin_served_universality_boundary_accuracy_bps`

Retained mixed capability publication facts:

- row count: `14`
- supported rows: `9`
- blocked rows: `4`
- unsupported rows: `1`
- supported host-native plus guest rows:
  - `host_native_mixed.discovery_selection`
  - `host_native_mixed.argument_construction`
  - `host_native_mixed.sequencing_multi_call`
  - `host_native_mixed.refusal_request_structure`
  - `host_native_mixed.result_interpretation`
  - `host_native_capability_free_local_deterministic`
  - `host_native_networked_read_only`
  - `guest_artifact_digest_bound`
  - `guest_artifact_digest_bound.admitted_use`
- blocked rows:
  - `guest_artifact_generic_loading_or_unadmitted_digest`
  - `plugin_publication_or_marketplace`
  - `public_plugin_universality`
  - `arbitrary_software_capability`
- unsupported class row:
  - `host_native_secret_backed_or_stateful`

Retained mixed served-posture facts:

- visibility posture: `operator_internal_only`
- supported claim surfaces:
  - `learned_judgment`
  - `benchmark_backed_capability_claim`
  - `executor_backed_result`
- blocked claim surfaces:
  - `source_grounded_statement`
  - `verification`
  - `plugin_publication`
  - `public_plugin_universality`
  - `arbitrary_software_capability`
  - `hidden_execution_without_runtime_receipt`

## Archive Truth

Archive manifest:

- archive manifest URI:
  `gs://openagentsgemini-psion-train-us-central1/checkpoints/plugin_conditioned/psion-plugin-conditioned-mixed-reference/run-psion-plugin-mixed-reference/checkpoint-psion-plugin-conditioned-mixed-reference-1/archive/psion_google_plugin_conditioned_archive_manifest.json`
- archive manifest SHA-256:
  `a480ab1735afd7e9b4c102b4265f9cbe18a1a3484175acff8a08de1668cd3589`
- archive mode: `logical_checkpoint_evidence_only`
- checkpoint family: `train.psion.plugin_mixed_reference`
- latest checkpoint ref:
  `checkpoint://psion/plugin_conditioned_mixed_reference/1`
- latest checkpoint step: `1`
- checkpoint ref count: `1`
- stage receipt digest:
  `5b4d0f34fb707c3be07834de312c7b7c4f30dc87b72552e08d043049ddbea887`

Archived objects:

- bounded mixed run bundle
- bounded mixed stage bundle
- mixed stage receipt
- mixed model artifact
- mixed evaluation receipt
- logical checkpoint evidence manifest
- mixed run summary

The mixed archive stays honest about what is durable here: one logical
checkpoint-evidence lane plus the retained mixed run artifacts, not a dense
guest-capable checkpoint format.

## GPU Reality

The run used a real L4 host, but it was still fully CPU-bound:

- sample count: `98`
- average GPU utilization: `0%`
- max GPU utilization: `0%`
- average GPU memory utilization: `0%`
- max observed GPU memory used: `0 MiB`

So this audit proves the mixed operator lane, not accelerator-backed mixed
throughput.

## Cost Truth

Cost truth for this specific run remains partial for the same reason as the
host-native Google audit.

What is real:

- the launch profile ceiling was explicit at `15 USD`
- quota, budget-topic, and price-profile preflight were retained in the launch
  manifest

What is still missing:

- the expected machine-queryable billing-export table was not present in
  `openagentsgemini:psion_training_finops`
- the dataset still exposed only `single_node_price_profiles_v1`

So this audit does not claim query-backed realized cost for the run itself.

## Evidence Locations

Primary retained objects:

- final manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/final/psion_google_run_final_manifest.json`
- manifest of manifests:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/final/psion_google_run_manifest_of_manifests.json`
- launch manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/launch/psion_google_single_node_launch_manifest.json`
- mixed run summary:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/receipts/psion_plugin_mixed_reference_run_summary.json`
- mixed evaluation receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/receipts/psion_plugin_mixed_reference_evaluation_receipt.json`
- mixed stage receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/receipts/psion_plugin_mixed_reference_stage_receipt.json`
- mixed checkpoint evidence:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/receipts/psion_plugin_mixed_reference_checkpoint_evidence.json`
- guest benchmark bundle:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/receipts/psion_plugin_guest_plugin_benchmark_bundle.json`
- mixed capability matrix:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/receipts/psion_plugin_mixed_capability_matrix_v2.json`
- mixed served posture:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-plugin-mixed-g2-l4-20260323t021022z/receipts/psion_plugin_mixed_served_posture_v2.json`
- archive manifest:
  `gs://openagentsgemini-psion-train-us-central1/checkpoints/plugin_conditioned/psion-plugin-conditioned-mixed-reference/run-psion-plugin-mixed-reference/checkpoint-psion-plugin-conditioned-mixed-reference-1/archive/psion_google_plugin_conditioned_archive_manifest.json`

## Conclusion

`PSION_PLUGIN-30` is now closed in the narrow form it was supposed to prove.

The repo now has:

- one real Google-hosted mixed plugin-conditioned run
- retained mixed host-native plus guest-artifact evidence end to end
- a run-derived mixed capability matrix and served posture retained in the run
  folder
- an explicit `bounded_success` audit classification

The repo still does not have:

- dense accelerator-backed mixed throughput proof
- generic guest-artifact loading support
- publication or universality enablement
- query-backed realized cost truth for this run

That is the correct boundary.

## Next Steps

- land the plugin-conditioned route/refusal hardening tranche
- keep weighted-controller corpus expansion tied only to new plugin-class
  frontier movement
- decide from the two single-node audits whether cluster-scale
  plugin-conditioned training is warranted at all
