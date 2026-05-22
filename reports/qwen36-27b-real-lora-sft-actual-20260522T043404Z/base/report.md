# Legal Benchmark Report: harvey_public_three_deterministic_replay_v1.base

## Global

- runs: 3
- all-pass rate: 3333 bps
- criterion pass rate: 3333 bps
- document coverage: 0 bps
- cost: 0 micro-usd
- wall time: 452 ms
- tokens: 960 input / 180 output

## Runs

### harvey.public.lease_notice / run.base.harvey.public.lease_notice

- all pass: false
- criterion pass rate: 0 bps
- document coverage: 0 bps
- run hash: 04e5e4bb304be6e5de7b9fafbee1cdf098e0a193e3423d346a718bbf1ad4e738
- output manifest hash: 70a4470c97bc3940f26506a21702d6d44ba86b35aee3c823b49fa3e03cc8c8dd
- cost: 0 micro-usd
- wall time: 150 ms

Missed criteria:

- harvey.public.lease_notice.answer_file: Fail; coverage=DraftingGap; required deliverable failed deterministic precheck
- harvey.public.lease_notice.legal_work_product: Fail; coverage=CoverageGap; required deliverable failed deterministic precheck

### harvey.public.purchase_indemnity / run.base.harvey.public.purchase_indemnity

- all pass: false
- criterion pass rate: 0 bps
- document coverage: 0 bps
- run hash: e9c8bd89fd17fd0c7c60dca57645aa52e30829b205c893128e2593767491d4c5
- output manifest hash: d497662fffedeb94dd3fbb6bb65bb752917e05169944c1bc62a7de26bdc29634
- cost: 0 micro-usd
- wall time: 150 ms

Missed criteria:

- harvey.public.purchase_indemnity.answer_file: Fail; coverage=DraftingGap; required deliverable failed deterministic precheck
- harvey.public.purchase_indemnity.legal_work_product: Fail; coverage=CoverageGap; required deliverable failed deterministic precheck

### harvey.public.privilege_log / run.base.harvey.public.privilege_log

- all pass: true
- criterion pass rate: 10000 bps
- document coverage: 0 bps
- run hash: ec71ed5b2045fdf339908e4c8442af15a3b9c7bd55cb297617d218e816629b46
- output manifest hash: 076061e43d32914a1f3c12de953002f309add2d5dde14401603ec64e946f6d2a
- cost: 0 micro-usd
- wall time: 152 ms

All criteria passed.

## Failure Clusters

- deterministic_precheck: 4 failures across 2 tasks; repro `cargo test -p psionic-eval --no-default-features --lib legal_benchmark_reports`

## Comparisons

- comparison.harvey_public_three_deterministic_replay_v1.base.run.base.harvey.public.lease_notice.run.base.harvey.public.purchase_indemnity: all-pass delta 0 bps, criterion delta 0 bps, cost delta 0 micro-usd, wall-time delta 0 ms
- comparison.harvey_public_three_deterministic_replay_v1.base.run.base.harvey.public.purchase_indemnity.run.base.harvey.public.privilege_log: all-pass delta 10000 bps, criterion delta 10000 bps, cost delta 0 micro-usd, wall-time delta 2 ms


Export hash: `62938448ec558c2ace6b8b465b6d600da1df33caaa85a48adb443b7cee930604`
