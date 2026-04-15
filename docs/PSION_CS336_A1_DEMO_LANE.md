# Psion CS336 A1 Demo Lane

> Status: packaged bounded machine-runtime lane, updated 2026-04-13 after
> wiring the existing CS336 A1 port into the zero-touch `psionic-train`
> manifest path for Episode 223 rehearsal and Pylon/Nexus execution.

This document records the packaged A1 demo lane that sits between the bounded
reference-lane implementation and the actual `Psion` pretraining lane.

Its job is simple:

- keep the Stanford CS336 A1 work honest and bounded
- let `Pylon` receive one named lane through the normal machine manifest path
- let `Nexus` consume retained status, checkpoint, and closeout outputs without
  inventing a second parser

## Canonical Identity

The packaged demo lane is:

- lane id: `psion_cs336_a1_demo_v1`
- request schema:
  `psion.cs336_a1_demo_automatic_execution_request.v1`
- output schema:
  `psion.cs336_a1_demo_automatic_execution_outputs.v1`
- work class: `small_model_local_training`
- release id: `psionic-train.psion_cs336_a1_demo.release.v1`
- environment ref:
  `psionic.environment.psion_cs336_a1_demo.host_cpu.operator@v1`

The lane is intentionally fixed to:

- the admitted tiny corpus at
  `fixtures/training/cs336_a1_reference_tiny_corpus.txt`
- the `Cs336A1ReferenceTrainingConfig::tiny()` configuration
- a four-step training budget
- one accepted checkpoint labeled `bounded_step_000004`

## Canonical Machine Contract

The current packaged demo lane is explicitly a CPU reference lane.

The exported `psionic-train` lane contract freezes it as:

- release id: `psionic-train.psion_cs336_a1_demo.release.v1`
- environment ref:
  `psionic.environment.psion_cs336_a1_demo.host_cpu.operator@v1`
- backend family: `cpu`
- topology class: `single_host_cpu_reference`
- minimum machine class: `cross_platform_cpu_compatible_operator`

The lane still executes the bounded host-CPU reference trainer. It is
cross-platform because a Mac or Linux operator host can run that CPU path even
when the machine also has Apple-silicon or CUDA accelerators present. Those
accelerators are not the claim boundary for this lane.

There is still no separate Apple-native or CUDA-native A1 variant. If a later
live demo needs one of those accelerated paths, it must land as a separate
explicit lane/environment identity instead of silently stretching this bounded
host-CPU lane.

## Operator Entry Points

Local rehearsal now uses the same packaged path that the machine runtime uses:

```bash
./TRAIN --lane cs336_a1_demo start
./TRAIN --lane cs336_a1_demo rehearse-base-lane
./TRAIN --lane cs336_a1_demo status --run-root <path>
./TRAIN --lane cs336_a1_demo verify --run-root <path>
bash scripts/check-psion-cs336-a1-demo-lane.sh
```

The lower-level machine boundary is the shared manifest entrypoint:

```bash
cargo run -q -p psionic-train --bin psionic-train -- manifest --manifest <path>
```

That means the A1 demo lane no longer depends on ad hoc shell glue when one
Mac or Linux operator host receives a packaged assignment for the bounded
host-CPU path.

The new `verify` surface is the bounded acceptance gate for a fresh local run.
It refuses to call a run demo-valid unless the retained runtime packets, the
generic checkpoint surface, the accepted checkpoint pointer, the closeout
bundle, and the descending loss pair all exist together under one run root.
The checker script wraps that path and can either validate an existing run root
or launch one fresh rehearsal run and verify it immediately.

## Retained Outputs

The retained path set is frozen:

- `manifests/launch_manifest.json`
- `status/current_run_status.json`
- `status/retained_summary.json`
- `checkpoints/latest_accepted_checkpoint_pointer.json`
- `checkpoints/manifests/checkpoint_manifest_step-000004.json`
- `checkpoints/step-000004/cs336_a1_reference_checkpoint.json`
- `closeout/closeout_bundle.json`
- `logs/launcher.log`

The shared machine runtime then projects the normal `Pylon`/`Nexus` packet
surfaces above those files:

- `status/psionic_train_run_status_packet.json`
- `status/psionic_train_window_status_packet.json`
- `status/checkpoint_surface.json`

That is the key distinction from the older reference bundle. The underlying A1
math is still the same bounded tiny lane, but the retained outputs now fit the
normal machine supervision path.

## Demo-Valid Success Gate

For Episode 224, one bounded run is demo-valid only when all of the following
are true:

- `status/current_run_status.json` ends in phase `completed` or `rehearsed`
- `status/retained_summary.json` carries both `initial_loss` and `final_loss`
  with `final_loss < initial_loss`
- `status/psionic_train_run_status_packet.json` exists
- `status/psionic_train_window_status_packet.json` exists
- `status/checkpoint_surface.json` exists and reports pointer state `accepted`
- `checkpoints/latest_accepted_checkpoint_pointer.json` exists
- `closeout/closeout_bundle.json` exists with outcome `accepted`

The bounded checker intentionally does not claim fresh multi-host proof by
itself. Multi-host proof still requires live `Pylon`/`Nexus` assignment intake
above this single-host lane contract.

## Fixtures

The canonical packaged fixtures are:

- `crates/psionic-train/examples/psion_cs336_a1_demo_fixtures.rs`
- `fixtures/training/psion_cs336_a1_demo_automatic_execution_request_v1.json`
- `fixtures/training/psion_cs336_a1_demo_automatic_execution_outputs_v1.json`
- `fixtures/training/psion_cs336_a1_demo_example/run-psion-cs336-a1-demo-fixture/`

Those fixtures keep one deterministic request/output pair and one real retained
run-root example checked into the repo.

## Claim Boundary

This lane honestly claims only that:

- the bounded CS336 A1 port can now be launched through the normal machine
  manifest path
- one Mac or Linux operator host can execute the tiny four-step run zero-touch
  through the bounded host-CPU path
- the retained outputs now look like normal machine-run artifacts instead of a
  special local bundle

It does not claim:

- broader-pretraining scale
- production autograd or distributed training closure
- that this lane replaces `psion_actual_pretraining_v1`
