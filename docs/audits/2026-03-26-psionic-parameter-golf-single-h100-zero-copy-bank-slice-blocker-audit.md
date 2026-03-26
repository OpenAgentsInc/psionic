# 2026-03-26 Psionic Parameter Golf Single-H100 Zero-Copy Bank-Slice Blocker Audit

This audit records the first fresh exact public-shape same-node H100 rerun
after `df70fcea` landed the zero-copy direct-banked CUDA bank-slice path in
`crates/psionic-backend-cuda/src/lib.rs`.

## Conclusion

Current `main` does not yet have a usable same-node H100 train-step proof for
the zero-copy bank-slice slice.

The fresh exact public-shape rerun on a real RunPod `NVIDIA H100 80GB HBM3`
node never emitted `train_runtime_receipt`, never wrote a JSON report, and
never wrote any readable stdout or stderr. During live inspection the trainer
degenerated into a long CPU-hot, low-GPU-memory posture instead of the earlier
same-node direct-banked receipt posture.

That means:

- `#546` stays open
- `#466` stays open
- the zero-copy bank-slice code slice is still only functionally validated by
  targeted local tests
- fresh same-node H100 wallclock evidence for this slice remains blocked until
  the reportless CPU-hot posture is root-caused

## Run Identity

- pod device: `NVIDIA H100 80GB HBM3`
- clean proof source tree on pod:
  `/root/psionic-h100-proof-df70fcea`
- build target dir:
  `/root/psionic-target-proof-df70fcea`
- live run root:
  `/workspace/parameter-golf-runpod/single-h100-proof-20260326T083858Z-zero-copy-bank-slices`
- exact trainer command shape:

```bash
PSIONIC_PARAMETER_GOLF_MATRIX_EXECUTION_MODE=direct_banked \
  /root/psionic-target-proof-df70fcea/release/parameter_golf_single_h100_train \
  /workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
  /workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
  /workspace/parameter-golf-runpod/single-h100-proof-20260326T083858Z-zero-copy-bank-slices/parameter_golf_single_h100_training.json \
  1 \
  live_only \
  non_overlapping
```

## Observed Outcome

The fresh rerun never produced any of the expected proof artifacts:

- no `train_runtime_receipt`
- no `parameter_golf_single_h100_training.json`
- no `exit_code`
- no readable `run.log`

After the remote trainer was interrupted, the visible run root contained only an
empty log file:

```text
total 2.0M
drwxrwxrwx  2 root root    1 Mar 26 08:38 .
drwxrwxrwx 28 root root 2.0M Mar 26 08:38 ..
-rw-rw-rw-  1 root root    0 Mar 26 08:58 run.log
```

During live inspection, while the trainer was still running, the process
posture was:

```text
PID     ELAPSED %CPU %MEM CMD
72384   17:44   99.3  0.0 /root/psionic-target-proof-df70fcea/release/parameter_golf_single_h100_train ...

72384, /root/psionic-target-proof-df70fcea/release/parameter_golf_single_h100_train, 4630
```

That means:

- the trainer stayed alive for at least `17m44s`
- the main thread burned one host CPU core at about `99%`
- GPU residency stayed around `4630 MiB`, which is far below the earlier
  direct-banked same-node proof posture

The live process file-descriptor state was also wrong for a normal proof run:

```text
1 -> /workspace/parameter-golf-runpod/single-h100-proof-20260326T083858Z-zero-copy-bank-slices/run.log (deleted)
2 -> /workspace/parameter-golf-runpod/single-h100-proof-20260326T083858Z-zero-copy-bank-slices/run.log (deleted)
```

Reading the live stdout descriptor returned only NUL bytes. The visible `run.log`
path later reappeared as a zero-byte file after the interrupted run exited.

## Comparison Against The Retained Same-Node Baseline

The retained direct-banked same-node H100 proof in
`docs/audits/2026-03-26-psionic-parameter-golf-single-h100-banked-vs-split-audit.md`
reached `train_runtime_receipt` in `478.511s`.

This zero-copy bank-slice rerun did not produce a comparable receipt at all. It
crossed that prior proof bar and continued into a reportless CPU-hot posture.

So this slice does not yet have a clean same-node H100 wallclock delta. The
current honest statement is narrower:

- the zero-copy bank-slice implementation is covered by targeted local CUDA and
  train-path tests
- the first exact public-shape same-node H100 proof attempt after landing it
  failed to produce a usable receipt

## Issue Impact

This audit does not close any open issue.

It narrows the next work item for `#546`:

- root-cause the reportless CPU-hot direct-banked posture on the same-node H100
  exact-shape lane
- rerun the exact public-shape proof after that fix
- only then compare the resulting receipt against the retained
  `478.511s` direct-banked baseline
