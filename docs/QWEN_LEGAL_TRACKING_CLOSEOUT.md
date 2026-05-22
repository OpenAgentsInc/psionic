# Qwen Legal Tracking Closeout

This closes the QWEN-LEGAL implementation tracker as an engineering control
plane. It is not a claim that the current retained adapter is already a strong
legal model.

The current strong-claim boundary is:

- current canonical hillclimb claim: `holdout_improvement`
- retained baseline: `5260` bps on `public_heldout`
- canonical candidate: `5380` bps on `public_heldout`
- strong-model threshold: `7000` bps on `public_heldout`
- near-perfect threshold: `9500` bps on `public_heldout`
- runner-added answer text allowed: `0`
- critical regression limit: `0` bps unless an issue explicitly changes it

## Implemented Surfaces

| Tracker Area | Repo Surface |
| --- | --- |
| Dense target path | Qwen3.6-27B load/admission, local prompt smoke, forward admission, and target-path docs in `docs/QWEN_LEGAL_FINETUNE_LANE.md`. |
| Adapter training path | Qwen legal SFT/DPO/GRPO smoke commands, Pylon-network SFT fixture, and distributed LoRA merge command. |
| Corpus and traces | Locked legal corpus bundle with train/dev/holdout splits, trace store, leakage boundaries, and runner-added answer-text exclusion. |
| Measurement ladder | Public training fixture, public heldout, model-only, Blueprint-assisted, private-gate boundary, and no-cheat runner docs. |
| Remote Pylon work | Pylon worker job, dispatch, loopback/tailnet/production dispatch modes, worker receipts, and Pylon-network SFT aggregation. |
| Promotion and rollback | Artifact promotion, strict merge validation, quarantine, route admission, canary, rollback, and hillclimb registry/feed outputs. |
| RL hillclimb | DPO/RL rollout extraction, reward traces, improvement rows, bad-completion preservation, and hillclimb controller. |
| Autopilot Blueprint path | Blueprint-assisted benchmark serving path and required acceptance report mode. |
| Settlement proof | Pylon treasury handoff and attached Bitcoin/Lightning settlement proof validation. |
| Model ladder | `docs/QWEN_LEGAL_MODEL_LADDER.md`. |
| Acceptance targets | `docs/QWEN_LEGAL_ACCEPTANCE_TARGETS.md`. |

## Done-Criteria Mapping

The tracker's done criteria now map to enforceable repo surfaces:

- Real Pylon work: represented by Pylon job dispatch, Pylon-network SFT, merge,
  worker receipts, and settlement proof gates.
- Harvey improvement: represented by the hillclimb plan and registry/feed; the
  current canonical candidate is a holdout improvement, not a strong-model
  threshold pass.
- Model-only and Blueprint-assisted views: required as separate
  `acceptance_reports` in the hillclimb plan.
- Retained evidence: every scored baseline, candidate, regression check, and
  acceptance report must carry a report path, replay command, retained trace,
  and zero runner-added answer text.
- Settlement: accepted Pylon work must be settled or explicitly
  operator-deferred.
- Promotion/rollback: the hillclimb controller exports `promotion_decision`
  and `stop_decision`; critical regressions force rollback.

## Next Real Run Rule

Any future candidate may be described as a strong legal model only when the
controller can record `score_claim_level = strong_legal_model` without refusal.
That requires a holdout split, a score at or above the strong-model threshold,
clean model-only and Blueprint-assisted reporting, retained traces, replayable
reports, zero runner-added answer text, settled or operator-deferred Pylon
work, and no critical task/failure regression breach.
