# 2026-04-12 Tailnet 4080 Accelerated Reference Training Run Log

This log records the multidevice reference-pilot run executed from the local
M5 Mac against the admitted Tailnet RTX 4080 host `archlinux`.

## Goal

Repeat the bounded `reference_pilot` run in multidevice mode:

- local M5 Mac as control plane
- `archlinux` over Tailnet as the single remote CUDA worker
- copied-back retained artifacts stored under the local run root

## Remote preflight

Confirmed Tailnet reachability and remote GPU visibility:

```bash
tailscale status
ssh christopherdavid@archlinux 'nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader'
```

Initial remote compute state on the 4080:

- `llama.cpp` resident compute process holding about `14 GiB`
- Chromium GPU process present

Per operator instruction, both were force-closed before launch so the admitted
CUDA device was actually idle.

## First successful accelerated run

Command:

```bash
./TRAIN --lane reference_pilot --mode accelerated_reference --cleanup-remote \
  --run-id tailnet-4080-reference-20260412t165601Z
```

Result:

- completed successfully with `remote_git_worktree`
- local run root:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-reference-20260412t165601Z`
- delivered backend: `cuda`
- checkpoint ref: `psion-reference-pilot-step-4`

Post-run issues discovered from this first success:

1. `--cleanup-remote` removed the remote `repo` and `output` paths but left the
   top-level remote run directory behind.
2. `psion_reference_pilot_resume_probe` failed against the accelerated artifact
   bundle on the local Mac with a `GradientTensorMismatch` because the retained
   optimizer-state parameter groups still carried `cuda:0` device metadata.

## Fixes applied

### 1. Remote cleanup root removal

Updated `scripts/train-psion-local-first.sh` so cleanup now also removes the
parent remote run directory when it becomes empty.

### 2. Resume probe CPU rebinding

Updated `crates/psionic-train/src/psion_reference_pilot.rs` so
`probe_psion_reference_pilot_resume` rebinds retained parameter-group tensor
specs to `Device::cpu()` before replaying the resumed optimizer step.

Added a regression test that mutates retained optimizer-state device metadata to
`cuda:0` and verifies the local resume probe still succeeds.

## Second issue exposed by live verification

After committing those fixes, the next accelerated rerun exercised
`archive_tarball` staging because the remote seed clone did not yet contain the
new local commit.

Run:

```bash
./TRAIN --lane reference_pilot --mode accelerated_reference --cleanup-remote \
  --run-id tailnet-4080-reference-20260412t170644Z
```

Failure:

- `scp: .../repo.tar: No such file or directory`

Cause:

- archive fallback tried to upload the tarball to a remote parent directory
  that had not been created yet

Fix:

- `train-psion-local-first.sh` now creates `$(dirname "$remote_worktree_dir")`
  before uploading the tarball

## Third issue exposed by live verification

The next archive-fallback rerun got past `scp` but failed during remote Rust
compilation:

Run:

```bash
./TRAIN --lane reference_pilot --mode accelerated_reference --cleanup-remote \
  --run-id tailnet-4080-reference-20260412t170718Z
```

Failure:

- remote `rustc` crashed with `SIGSEGV` while compiling `psionic-runtime`
- compiler hint suggested increasing `RUST_MIN_STACK`

Fix:

- `train-psion-local-first.sh` now exports `RUST_MIN_STACK=16777216` in the
  remote `cargo run` environment

## Final verified multidevice run

Command:

```bash
./TRAIN --lane reference_pilot --mode accelerated_reference --cleanup-remote \
  --run-id tailnet-4080-reference-20260412t170956Z
```

Result:

- status: `completed`
- stage strategy: `archive_tarball`
- local run root:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-reference-20260412t170956Z`
- local artifact dir:
  `/Users/christopherdavid/scratch/psion_reference_pilot_runs/tailnet-4080-reference-20260412t170956Z/reference_pilot_artifacts`
- checkpoint ref: `psion-reference-pilot-step-4`
- checkpoint parameter digest:
  `d70c4e25cb1cfbfb3f1d6c0aa1bca3e1abc431a000fbca7fc3b8ae9d43f5889a`
- estimated total cost: `14800` microusd
- worker host: `archlinux`
- worker Tailnet IP: `100.108.56.85`

## Resume verification on the copied-back accelerated artifact bundle

Command:

```bash
cargo run -q -p psionic-train --example psion_reference_pilot_resume_probe -- \
  ~/scratch/psion_reference_pilot_runs/tailnet-4080-reference-20260412t170956Z/reference_pilot_artifacts \
  /tmp/tailnet-4080-reference-resume-probe-20260412t170956Z
```

Result:

- completed successfully
- output:
  `/tmp/tailnet-4080-reference-resume-probe-20260412t170956Z/psion_reference_pilot_resume_probe.json`

## Remote cleanup verification

After the final successful run:

```bash
ssh christopherdavid@archlinux \
  'test -d ~/code/psion-reference-pilot/tailnet-4080-reference-20260412t170956Z && echo present || echo removed'
```

Result:

- remote run directory removed

## Focused verification run during this session

Code-level checks:

```bash
cargo test -q -p psionic-train reference_pilot_resume_probe_restores_from_last_stable_checkpoint
cargo test -q -p psionic-train reference_pilot_resume_probe_rebinds_accelerated_parameter_groups_to_cpu
```

Live checks:

- successful copied-back accelerated run over Tailnet using the RTX 4080
- successful local resume probe against the accelerated artifact bundle
- successful remote cleanup of the archive-staged run root

## Landed fix commit

- `ee6eb3f98bee0049c04ff88177214d475b362891`
- subject: `Fix accelerated reference cleanup and resume probe`
