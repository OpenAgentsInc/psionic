# Tassadar Default Train Lane

This document freezes the default meaning of `train Tassadar` inside this repo.

The canonical default is the bounded trace-bound article-transformer
weight-production lane that yields
`tassadar-article-transformer-trace-bound-trained-v0`.

Run it with:

```bash
./TRAIN_TASSADAR
```

Read the retained contract at:

- `fixtures/tassadar/operator/tassadar_default_train_lane_contract_v1.json`

## Default Contract

- lane id:
  `tassadar_article_transformer_trace_bound_trained_v0`
- launcher:
  `./TRAIN_TASSADAR`
- launcher surface id:
  `tassadar_train_default_start`
- stage program id:
  `tassadar_article_transformer_weight_production_v1`
- training profile:
  `bounded_article_weight_production`
- hardware profile:
  `cpu_reference`
- writer node:
  `psionic.local.cpu_reference`
- output-root family:
  `fixtures/tassadar/runs/tassadar_article_transformer_weight_production_v1`
- evidence family:
  `train.tassadar.article_transformer.weight_production`
- checker bundle:
  `scripts/check-tassadar-default-train-lane.sh`
  and `scripts/check-tassadar-acceptance.sh`
- restart posture:
  `restart_from_trace_bound_base_v0`
- promotion target model id:
  `tassadar-article-transformer-trace-bound-trained-v0`
- promotion target descriptor:
  `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v0_descriptor.json`
- promotion target artifact:
  `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v0.safetensors`
- promotion target lineage:
  `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v0_lineage_contract.json`

## Why This Lane

This is the current default because it is the trained Tassadar family that the
rest of the repo already reuses as incumbent truth.

- The retained weight-production bundle lives at
  `fixtures/tassadar/runs/tassadar_article_transformer_weight_production_v1/article_transformer_weight_production_bundle.json`.
- The retained lineage contract binds that bundle into
  `fixtures/tassadar/models/tassadar_article_transformer_trace_bound_trained_v0_lineage_contract.json`.
- The post-article machine and promotion docs already cite
  `tassadar-article-transformer-trace-bound-trained-v0` as the incumbent
  model family.
- The later `trained-v1` executor replacement and 4080 decision-grade work are
  explicitly follow-on candidate tracks above this incumbent, not the default
  starting meaning of `train Tassadar`.

## What This Default Is Not

This default does not silently alias other existing Tassadar lanes.

- It is not the older 4x4 learned promotion bundle under
  `fixtures/tassadar/runs/sudoku_v0_promotion_v3`.
- It is not the bounded 9x9 learned reference lane under
  `fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0`.
- It is not the Hungarian-10x10 exact learned benchmark lane under
  `fixtures/tassadar/runs/hungarian_10x10_v0_learned_article_executor_v0`.
- It is not the 4080 decision-grade executor or later replacement program.

Those lanes remain valid evidence. They are not the canonical operator default.

## Claim Boundary

This default lane proves one bounded training path that produces the retained
`trained-v0` article-transformer family with checkpoint restore and lineage.
It does not claim full unification of all historical Tassadar training lanes,
held-out generalization, or that later replacement candidates have become the
new incumbent automatically.
