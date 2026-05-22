# Qwen Legal Acceptance Targets

This policy defines when Psionic may call a Qwen legal run plumbing proof,
a public fixture win, a holdout improvement, or a strong legal model. A run
that wins only on trainable fixtures is not a strong legal model.

## Score Targets

Scores are basis points out of `10000`.

| Split / Mode | Baseline | First Credible Improvement | Strong-Model Threshold | Near-Perfect Threshold | Critical Regression Limit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `trainable_public_fixture` | 10000 | 10000 | 10000 | 10000 | 0 |
| `public_heldout` | 5260 | 5310 | 7000 | 9500 | 0 |
| `model_only` | 5260 | 5310 | 6500 | 9000 | 0 |
| `blueprint_assisted` | 5260 | 5310 | 7500 | 9500 | 0 |

The trainable public fixture can prove the runner, corpus, receipt, replay,
and scoring pipe. It cannot prove strong legal reasoning, even at `10000` bps.
The first credible strong-model evidence starts with `public_heldout`, then
must stay clean in both `model_only` and `blueprint_assisted` reporting.

## Plain Definitions

`plumbing proof` means the trainer, runner, scoring, trace retention, report
writing, and registry/feed export work on a cheap or trainable fixture. It is
not a quality claim.

`public fixture win` means the candidate wins on a public fixture that may be
trainable or known during iteration. It can justify more training or a smoke
release, but not a strong-model claim.

`holdout improvement` means the candidate beats the retained baseline on a
public held-out split by at least the declared first-credible target, with a
replayable report, retained scored traces, zero runner-added answer text, and
no critical regression breach.

`strong legal model` means the candidate beats baseline on a holdout split,
clears the strong-model threshold, reports the required model-only and
Blueprint-assisted views separately, links every score to a replayable report,
retains every scored trace, and stays within all critical task/failure
regression limits.

## Required Reporting

Every hillclimb plan must report these modes separately:

| Report Mode | Purpose |
| --- | --- |
| `trainable_public_fixture` | Proves the pipe and catches runner/scorer drift. |
| `public_holdout` | Anchors first credible improvement and strong-model claims. |
| `model_only` | Measures the adapter without Blueprint scaffold help. |
| `blueprint_assisted` | Measures the intended legal workflow with the scaffold. |

Every baseline, candidate, regression check, and required report must include
a replay command, a report path, retained scored traces, and
`runner_added_answer_text_count = 0`. Runner-added answer text is a hard stop:
the runner may ask the model, execute model-written tools, validate files, and
score outputs, but it may not write answer text for the model.

Regression checks must name the task type and failure category. Critical
categories use a zero-regression limit unless a later issue changes this
policy explicitly.

## Stop Conditions

Continue training when the candidate is clean but below the first-credible or
strong-model threshold.

Roll back when the candidate fails, a critical regression exceeds its declared
limit, trace retention is missing, or runner-added answer text appears.

Hold for operator review when evidence is incomplete, a payment state is not
settled or operator-deferred, or the claim/report mode does not match the
declared split.

Promote only when the candidate clears the declared score target, every score
claim links to a replayable report, all scored traces are retained, runner-added
answer text is zero, and critical regressions stay within limit.

The `qwen-legal-hillclimb` controller enforces these fields in the plan and
exports the resulting `score_claim_level` and `stop_decision` into registry and
feed records.
