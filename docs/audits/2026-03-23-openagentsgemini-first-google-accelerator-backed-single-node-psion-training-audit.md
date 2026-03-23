# OpenAgentsGemini First Google Accelerator-Backed Single-Node Psion Training Audit

> Status: follow-up `PSION_ACCEL-6` audit written on March 23, 2026 after the
> first real Google-hosted accelerator-backed single-node Psion training run
> completed on `openagentsgemini`, retained its evidence bundle, and the VM was
> deleted.

## Scope

This audit covers one bounded Google single-node execution of the canonical
accelerated Psion trainer.

It claims:

- one real Google-hosted accelerator-backed Psion training run now exists
- the run used the committed CUDA trainer path instead of the CPU reference
  bundle path
- the run preserved launch truth, backend truth, GPU-sample truth, throughput
  truth, checkpoint truth, archive truth, and teardown truth

It does not claim:

- plugin-conditioned accelerator-backed training
- cluster-scale readiness
- broad cost optimality
- production-scale pretraining throughput

## Typed Outcome

- result classification: `bounded_success`
- Google run id: `psion-g2-l4-accelerated-20260323t074419z`
- trainer lane id: `psion_accelerated_reference_pilot`
- repo revision: `08b208a16c5f2282244847e1b17488ad67f70a66`

## Topology

- project: `openagentsgemini`
- zone: `us-central1-a`
- machine type: `g2-standard-8`
- accelerator: `nvidia-l4`
- accelerator count: `1`
- image family: `common-cu128-ubuntu-2204-nvidia-570`
- resolved image: `common-cu128-ubuntu-2204-nvidia-570-v20260320`
- boot disk: `pd-balanced`, `200 GB`
- external IP: `false`

## Launch Truth

The retained launch manifest bound the accelerated lane explicitly:

- profile id: `g2_l4_single_node_accelerated`
- expected execution backend: `cuda`
- pre-training command:
  `cargo build -p psionic-train --example psion_accelerated_reference_pilot`
- training command:
  `"$CARGO_TARGET_DIR/debug/examples/psion_accelerated_reference_pilot" "$PSION_OUTPUT_DIR"`
- post-training archive command:
  `bash "$PSION_REPO_DIR/scripts/psion-google-archive-reference-pilot-checkpoint.sh" --manifest-out "$PSION_SCRATCH_DIR/psion_google_checkpoint_archive_manifest.json" "$PSION_OUTPUT_DIR"`
- post-training restore command: disabled for this launch profile
- declared run cost ceiling: `20 USD`

Quota preflight at launch time reported:

- CPU quota available: `186`
- L4 quota available: `8`
- instance quota available: `713`
- disk quota available: `16868 GB`
- quota result: `ready`

## Timeline

UTC timestamps from the retained run timeline:

- launch created: `2026-03-23T07:44:30Z`
- bootstrap started: `2026-03-23T07:45:21Z`
- bootstrap finished: `2026-03-23T07:54:07Z`
- training started: `2026-03-23T07:54:07Z`
- training finished: `2026-03-23T07:54:12Z`
- checkpoint completed: `2026-03-23T07:54:32Z`
- teardown started: `2026-03-23T07:54:32Z`
- teardown finished: `2026-03-23T07:54:36Z`
- final manifest written: `2026-03-23T07:56:01Z`

The instance was deleted after evidence retention with:

- `bash scripts/psion-google-delete-single-node.sh --run-id psion-g2-l4-accelerated-20260323t074419z --force`

## Backend And GPU Evidence

This run is the first truthful accelerator-backed Google proof because all four
required evidence surfaces aligned:

- stage receipt backend: `cuda`
- observability receipt backend: `cuda`
- accelerator-backed declared: `true`
- accelerator-backed pass: `true`

Retained GPU summary:

- sample count: `5`
- average GPU utilization: `0.6%`
- max GPU utilization: `2%`
- average memory utilization: `0%`
- max observed GPU memory used: `214 MiB`
- observed total GPU memory: `23034 MiB`

Retained post-warmup accelerator validation facts:

- considered post-warmup samples: `4`
- non-zero utilization samples: `2`
- non-zero memory-residency samples: `4`
- average post-warmup GPU utilization: `0.75%`
- peak post-warmup GPU utilization: `2%`
- peak post-warmup GPU memory used: `214 MiB`

That is still a very small bounded run, but it is no longer a CPU-executed run
on a GPU host. The trainer touched CUDA, retained non-zero utilization, and
retained non-zero device memory residency past warmup.

## Training And Throughput Facts

Retained stage receipt:

- run id: `psion-accelerated-reference-pilot-run`
- stage id: `psion-accelerated-reference-pretrain-stage`
- model id: `psion-compact-decoder-pilot-v1`
- dataset identity: `psion_reference_tokenized@v1`
- optimizer steps completed: `4`
- mean step latency: `1000 ms`
- promoted checkpoint label: `psion-reference-pilot-step-4`

Retained observability receipt:

- train tokens processed: `379028`
- validation tokens processed: `530`
- held-out tokens scored: `161`
- optimizer steps completed: `4`
- wall clock: `4000 ms`
- mean tokens per second: `94929`
- peak tokens per second: `94993`
- checkpoint write throughput: `385144 bytes/sec`

## Cost Envelope

This run preserves bounded cost truth, not full billing-export realized cost.

What is explicit and retained:

- launch ceiling: `20 USD`
- observability compute estimate: `0.0144 USD`
- observability storage estimate: `0.00032 USD`
- observability network estimate: `0.00008 USD`
- observability total estimate: `0.0148 USD`

What is still missing:

- machine-queryable realized cost from billing export for this exact run

## Evidence Locations

Primary retained objects:

- final manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-accelerated-20260323t074419z/final/psion_google_run_final_manifest.json`
- manifest of manifests:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-accelerated-20260323t074419z/final/psion_google_run_manifest_of_manifests.json`
- launch manifest:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-accelerated-20260323t074419z/launch/psion_google_single_node_launch_manifest.json`
- stage receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-accelerated-20260323t074419z/receipts/psion_accelerated_reference_pilot_stage_receipt.json`
- observability receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-accelerated-20260323t074419z/receipts/psion_accelerated_reference_pilot_observability_receipt.json`
- accelerator validation receipt:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-accelerated-20260323t074419z/receipts/psion_google_accelerator_validation_receipt.json`
- GPU samples:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-accelerated-20260323t074419z/logs/psion_google_gpu_samples.csv`
- GPU summary:
  `gs://openagentsgemini-psion-train-us-central1/runs/psion-g2-l4-accelerated-20260323t074419z/host/psion_google_gpu_summary.json`
- archive manifest:
  `gs://openagentsgemini-psion-train-us-central1/checkpoints/reference_pilot/psion-accelerated-reference-pilot-run/psion-reference-pilot-step-4/archive/psion_google_reference_checkpoint_archive_manifest.json`

## Conclusion

`PSION_ACCEL-6` is now closed in the narrow form it was supposed to prove.

The repo now has:

- one real Google-hosted accelerator-backed single-node Psion training lane
- explicit `cuda` backend truth in the retained stage and observability receipts
- non-zero post-warmup GPU utilization and device-memory residency
- retained throughput, checkpoint, archive, and final-manifest truth

The repo still does not have:

- plugin-conditioned accelerator-backed training proof
- cluster-scale readiness
- query-backed realized cost truth

That is the correct claim boundary after March 23, 2026.
