# Tassadar Default Train Rehearsal

`Tassadar` now has one bounded default-lane rehearsal bundle above the frozen
default lane and the canonical operator launcher.

It is intentionally narrow. The rehearsal proves that the incumbent default
lane can be cited with:

- one explicit `./TRAIN_TASSADAR` start-surface run root
- one focused default-lane checker receipt
- one broader acceptance-checker receipt
- one retained promotion-target evidence packet for the incumbent
  `tassadar-article-transformer-trace-bound-trained-v0` family

The canonical retained artifacts are:

- `fixtures/tassadar/operator/tassadar_default_train_lane_contract_checker_receipt_v1.json`
- `fixtures/tassadar/operator/tassadar_default_train_acceptance_checker_receipt_v1.json`
- `fixtures/tassadar/operator/tassadar_default_train_promotion_evidence_v1.json`
- `fixtures/tassadar/operator/tassadar_default_train_rehearsal_bundle_v1.json`

The committed example run root is:

- `fixtures/tassadar/operator/tassadar_default_train_rehearsal_example/run-tassadar-default-rehearsal-20260402t220000z`

That run root keeps the same launcher-level vocabulary the Psion actual lane
uses where it fits:

- `manifests/launch_manifest.json`
- `status/current_run_status.json`
- `status/retained_summary.json`
- `checker/default_train_lane_contract_check.json`
- `checker/acceptance_check.json`
- `promotion/promotion_target_evidence.json`
- `closeout/rehearsal_bundle.json`

Run the bounded checker from the repo root:

```bash
bash scripts/check-tassadar-default-train-rehearsal.sh
```

## Claim Boundary

This rehearsal proves one operator-readable closeout for the incumbent default
lane only. It does not claim that every historical Tassadar lane has equal
launcher parity, and it does not promote any later 4080 executor candidate
above the retained article-transformer trained-v0 family.
