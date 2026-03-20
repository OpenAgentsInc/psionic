# Psionic Parameter Golf After-Action

> Status: canonical Parameter Golf stop record, written 2026-03-19 after the
> user directed Psionic to pause the lane, stop active H100 work, and close
> the remaining open PGOLF issues as `wontfix` / not planned.

This document records the final honest state of the Parameter Golf lane at the
moment active work stopped.

It is not a new promotion claim.

It is the repo-local record of what landed, what did not land, what the last
real hardware evidence showed, and what would still be required if the lane is
ever resumed later under explicit user direction.

The machine-readable companion report lives at:

- `fixtures/parameter_golf/reports/parameter_golf_after_action_report.json`

## Stop Decision

On 2026-03-19, the user directed Psionic to:

- stop active Parameter Golf implementation work
- stop the live H100 run
- write final after-action records in the repo
- comment on the remaining open PGOLF issues and close them as `wontfix`

That means Parameter Golf is now preserved here as historical repo truth, not
as an active sprint queue.

## What Landed Before The Stop

The repo keeps the work that already became real and reviewable:

- challenge-oracle parity for FineWeb shard loading, tokenizer-byte accounting,
  `val_loss`, and `val_bpb`
- a bounded Psionic-owned baseline decoder and single-device CPU-reference
  trainer lane
- explicit distributed receipt, topology, and refusal surfaces for `8xH100`
- a real non-record exported-folder contract, compatibility checker, replay
  verification, and machine-readable accounting receipts
- a concrete post-parity research queue with negative evidence for one fixed
  `256`-token restricted-attention window
- a historical record of the closed external non-record PR and the outbound
  contribution ban
- a Rust-native single-H100 bring-up seam and report path

That landed work remains useful as historical substrate truth for:

- tokenizer-byte accounting
- compact-decoder graph work
- non-record packaging
- offline compatibility verification
- bounded CUDA/runtime closure evidence

## Final Honest Posture

At the moment the lane stopped, the canonical acceptance categories remained:

| Category | Final repo posture |
| --- | --- |
| `challenge-oracle-parity` | `implemented` |
| `single-device-trainer-parity` | `implemented_early` |
| `distributed-throughput-closure` | `partial` |
| `packaging-readiness` | `implemented` |
| `record-track-readiness` | `partial_outside_psionic` |

So the strongest honest retained claim posture is still:

- `non_record_submission`

What never became true:

- a fully Psionic-owned Rust-only single-H100 baseline training run
- real exported-folder `8xH100` run bundles from the shipped entrypoint
- acceptance-matrix promotion beyond the existing non-record posture
- a real record-candidate campaign

## Last H100 Findings

Before the stop, Psionic did get to real H100 hardware and the lane advanced
beyond earlier machine-admission refusal.

The most important facts from the last bounded single-H100 smoke were:

- device admission succeeded on one non-MIG `NVIDIA H100 PCIe`
- the Rust-native bring-up path reached the actual CUDA microbatch path
- the traced forward path progressed well into the baseline graph instead of
  failing immediately at launch
- the dominant observed costs were not only attention itself; host-materialized
  view-style ops were also expensive in the public CUDA runtime

Representative traced timings from the last bounded H100 run:

- `scaled_dot_product_attention` at about `7.7s`
- `rotary_embedding` at about `2.0s` to `3.9s`
- `permute` at about `1.1s` to `2.3s`
- `expand` at about `2.0s`
- `reshape` at about `0.2s` to `0.8s`

The practical conclusion from that run was:

- the next engineering move would have been zero-copy or lower-copy view
  handling plus continued forward-path profiling, not more reporting

That work did not land because the user stopped the lane before the next
trainer/runtime optimization cycle was completed.

## Why The Remaining Queue Closed Not Planned

The remaining open issues were all real, but they were open for unfinished
runtime or hardware-evidence work rather than because the queue was wrong.

They closed as not planned because the lane itself was stopped:

- `#183` `PGOLF-500`: parent sprint queue closed because the active sprint was
  ended
- `#189` `PGOLF-602`: no real `8xH100` exported-folder run evidence was
  captured before the stop
- `#194` `PGOLF-604`: the Rust-native single-H100 seam existed, but the real
  trainer path was not finished
- `#250` `PGOLF-606`: acceptance-matrix promotion could not happen honestly
  without `#189` and `#194`
- `#253` `PGOLF-609`: no real record-candidate campaign was frozen and run

This is a stop record, not a claim that those tasks were impossible in
principle.

It only records that Psionic is no longer pursuing them in this repo unless
the user later reopens the lane explicitly.

## If The Lane Is Resumed Later

The repo should resume from the last concrete blocker, not from roadmap prose.

The first practical restart point would be:

1. restore the bounded H100 trace workflow
2. reduce host-materialized view-op cost in the CUDA runtime
3. finish the actual Rust-only single-H100 baseline trainer path
4. only then spend on real exported-folder `8xH100` evidence

Any future outbound PR, issue, or maintainer-facing comment to external repos
still requires explicit user direction first.
