# W3 Student Sweep 2026-06-12

This fixture bundle preserves the completed W3 four-baseline sweep for
OpenAgents issue `#4749`.

Corpus snapshot:

- `corpus_id`: `corpus.tassadar_trace.v0_2.w3_100m`
- `dataset_snapshot_digest`:
  `d045a53d0cecbe6ffb1b4f0c1522ab76b02014491842f1770d34c12a885c8c3a`
- Train prep SHA-256:
  `8095588b05ff1bc3b8a723431c35015882a25566f74d895b514071f5e1734350`
- Eval prep SHA-256:
  `512830dcbdd4f8e4842adbf1960522c70e8609475581aa4936f6424b4981102b`

Artifact layout:

| Baseline | Directory | Artifacts |
| --- | --- | --- |
| `baseline_a_next_token` | `a/` | `weights.bin`, `receipt.json`, `eval-report.json` |
| `baseline_b_aux_state` | `b/` | `weights.bin`, `receipt.json`, `eval-report.json` |
| `baseline_c_lookup_analytic` | `c/` | `weights.bin`, `receipt.json`, `eval-report.json` |
| `baseline_d_frozen_executor_learned_interface` | `d/` | `interface.json`, `receipt.json`, `eval-report.json` |

Overall replay metrics:

| Baseline | pass@1 | replay acceptance | median divergence step | p90 divergence step | median valid-prefix tokens | Top divergence cause |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| A next-token | `0.0` | `0.0` | `0` | `0` | `0` | `wrong_fetch=499` |
| B auxiliary-state | `0.0` | `0.0` | `0` | `0` | `0` | `wrong_fetch=316`, `memory_read=306` |
| C lookup analytic | `0.0` | `0.0` | `0` | `0` | `0` | `wrong_fetch=499` |
| D frozen-interface | `1.0` | `1.0` | `512` | `4096` | `10240` | none |

Interpretation:

- H1 is supported: pure next-token trace learning did not produce replay-safe
  rollouts on this corpus.
- H2 is supported: the frozen analytic executor plus learned interface is the
  only successful route in this sweep.
- H3 is falsified in this exact setup: analytic lookup initialization solves
  the lookup auxiliary during training, but the backbone still diverges at
  rollout step zero.

