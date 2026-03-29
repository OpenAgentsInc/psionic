use std::{
    fs,
    path::{Path, PathBuf},
};

use psionic_eval::{
    CompiledAgentModuleEvalReport, CompiledAgentModuleKind, CompiledAgentModuleRevisionSet,
    CompiledAgentPublicOutcomeKind, CompiledAgentRoute, CompiledAgentToolResult,
    build_compiled_agent_module_eval_report, compiled_agent_baseline_revision_set,
    evaluate_compiled_agent_grounded_answer, evaluate_compiled_agent_route,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    CompiledAgentReceiptError, CompiledAgentReplayBundle, canonical_compiled_agent_replay_bundle,
    repo_relative_path,
};

pub const COMPILED_AGENT_ROUTE_CANDIDATE_REPORT_FIXTURE_PATH: &str =
    "fixtures/compiled_agent/compiled_agent_route_candidate_module_eval_report_v1.json";
pub const COMPILED_AGENT_GROUNDED_CANDIDATE_REPORT_FIXTURE_PATH: &str =
    "fixtures/compiled_agent/compiled_agent_grounded_candidate_module_eval_report_v1.json";
pub const COMPILED_AGENT_XTRAIN_CYCLE_RECEIPT_FIXTURE_PATH: &str =
    "fixtures/compiled_agent/compiled_agent_xtrain_cycle_receipt_v1.json";

const XTRAIN_CYCLE_SCHEMA_VERSION: &str = "psionic.compiled_agent.xtrain_cycle.v1";

#[derive(Debug, Error)]
pub enum CompiledAgentXtrainError {
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("module summary for `{module}` missing from eval report")]
    MissingModuleSummary { module: String },
    #[error("route replay sample `{sample_id}` is missing `user_request`")]
    MissingRoutePrompt { sample_id: String },
    #[error("grounded replay sample `{sample_id}` is missing `route`")]
    MissingGroundedRoute { sample_id: String },
    #[error("grounded replay sample `{sample_id}` has invalid tool results")]
    InvalidGroundedToolResults { sample_id: String },
    #[error(transparent)]
    Receipts(#[from] CompiledAgentReceiptError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompiledAgentPromotionDecision {
    Promote,
    Hold,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentCandidateDelta {
    pub module: CompiledAgentModuleKind,
    pub base_revision_id: String,
    pub candidate_revision_id: String,
    pub replay_bundle_digest: String,
    pub targeted_failure_classes: Vec<String>,
    pub delta_summary: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentModuleValidation {
    pub module: CompiledAgentModuleKind,
    pub baseline_passed_cases: u32,
    pub candidate_passed_cases: u32,
    pub improvement_case_ids: Vec<String>,
    pub regression_case_ids: Vec<String>,
    pub baseline_replay_match_count: u32,
    pub candidate_replay_match_count: u32,
    pub replay_improvement_sample_ids: Vec<String>,
    pub replay_regression_sample_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentXtrainModuleOutcome {
    pub module: CompiledAgentModuleKind,
    pub decision: CompiledAgentPromotionDecision,
    pub reason: String,
    pub delta: CompiledAgentCandidateDelta,
    pub validation: CompiledAgentModuleValidation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompiledAgentXtrainCycleReceipt {
    pub schema_version: String,
    pub cycle_id: String,
    pub row_id: String,
    pub source_ledger_digest: String,
    pub replay_bundle_digest: String,
    pub route_outcome: CompiledAgentXtrainModuleOutcome,
    pub grounded_answer_outcome: CompiledAgentXtrainModuleOutcome,
    pub summary: String,
    pub receipt_digest: String,
}

#[must_use]
pub fn compiled_agent_route_candidate_report_fixture_path() -> PathBuf {
    repo_relative_path(COMPILED_AGENT_ROUTE_CANDIDATE_REPORT_FIXTURE_PATH)
}

#[must_use]
pub fn compiled_agent_grounded_candidate_report_fixture_path() -> PathBuf {
    repo_relative_path(COMPILED_AGENT_GROUNDED_CANDIDATE_REPORT_FIXTURE_PATH)
}

#[must_use]
pub fn compiled_agent_xtrain_cycle_receipt_fixture_path() -> PathBuf {
    repo_relative_path(COMPILED_AGENT_XTRAIN_CYCLE_RECEIPT_FIXTURE_PATH)
}

#[must_use]
pub fn compiled_agent_route_candidate_revision() -> CompiledAgentModuleRevisionSet {
    let mut candidate = compiled_agent_baseline_revision_set();
    candidate.revision_id = String::from("compiled_agent.route.rule_v2.negation_guard");
    candidate.negation_keywords = vec![
        String::from("not"),
        String::from("dont"),
        String::from("don't"),
    ];
    candidate.unsupported_route_keywords = vec![
        String::from("poem"),
        String::from("gpu"),
        String::from("gpus"),
    ];
    candidate
}

#[must_use]
pub fn compiled_agent_grounded_candidate_revision() -> CompiledAgentModuleRevisionSet {
    let mut candidate = compiled_agent_baseline_revision_set();
    candidate.revision_id = String::from("compiled_agent.grounded_answer.rule_v2.recent_earnings");
    candidate.include_recent_earnings = true;
    candidate
}

#[must_use]
pub fn canonical_compiled_agent_route_candidate_report() -> CompiledAgentModuleEvalReport {
    build_compiled_agent_module_eval_report(&compiled_agent_route_candidate_revision())
}

#[must_use]
pub fn canonical_compiled_agent_grounded_candidate_report() -> CompiledAgentModuleEvalReport {
    build_compiled_agent_module_eval_report(&compiled_agent_grounded_candidate_revision())
}

pub fn canonical_compiled_agent_xtrain_cycle_receipt()
-> Result<CompiledAgentXtrainCycleReceipt, CompiledAgentXtrainError> {
    let baseline = compiled_agent_baseline_revision_set();
    let replay_bundle = canonical_compiled_agent_replay_bundle()?;
    let baseline_report = build_compiled_agent_module_eval_report(&baseline);
    let route_candidate = compiled_agent_route_candidate_revision();
    let route_candidate_report = build_compiled_agent_module_eval_report(&route_candidate);
    let grounded_candidate = compiled_agent_grounded_candidate_revision();
    let grounded_candidate_report = build_compiled_agent_module_eval_report(&grounded_candidate);

    let route_outcome = build_route_outcome(
        &replay_bundle,
        &baseline_report,
        &route_candidate_report,
        &baseline,
        &route_candidate,
    )?;
    let grounded_answer_outcome = build_grounded_answer_outcome(
        &replay_bundle,
        &baseline_report,
        &grounded_candidate_report,
        &baseline,
        &grounded_candidate,
    )?;

    let mut receipt = CompiledAgentXtrainCycleReceipt {
        schema_version: String::from(XTRAIN_CYCLE_SCHEMA_VERSION),
        cycle_id: String::from("compiled_agent.xtrain.cycle.v1"),
        row_id: baseline_report.row_id.clone(),
        source_ledger_digest: replay_bundle.source_ledger_digest.clone(),
        replay_bundle_digest: replay_bundle.bundle_digest.clone(),
        route_outcome,
        grounded_answer_outcome,
        summary: String::new(),
        receipt_digest: String::new(),
    };
    receipt.summary = format!(
        "Compiled-agent XTRAIN cycle retained route decision={:?} and grounded_answer decision={:?} on replay bundle {}.",
        receipt.route_outcome.decision,
        receipt.grounded_answer_outcome.decision,
        receipt.replay_bundle_digest
    );
    receipt.receipt_digest = stable_digest(b"compiled_agent_xtrain_cycle_receipt|", &receipt);
    Ok(receipt)
}

pub fn write_compiled_agent_route_candidate_report(
    output_path: impl AsRef<Path>,
) -> Result<CompiledAgentModuleEvalReport, CompiledAgentXtrainError> {
    write_report(
        output_path,
        &canonical_compiled_agent_route_candidate_report(),
    )
}

pub fn write_compiled_agent_grounded_candidate_report(
    output_path: impl AsRef<Path>,
) -> Result<CompiledAgentModuleEvalReport, CompiledAgentXtrainError> {
    write_report(
        output_path,
        &canonical_compiled_agent_grounded_candidate_report(),
    )
}

pub fn write_compiled_agent_xtrain_cycle_receipt(
    output_path: impl AsRef<Path>,
) -> Result<CompiledAgentXtrainCycleReceipt, CompiledAgentXtrainError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| CompiledAgentXtrainError::CreateDir {
            path: parent.display().to_string(),
            error,
        })?;
    }
    let receipt = canonical_compiled_agent_xtrain_cycle_receipt()?;
    let json = serde_json::to_string_pretty(&receipt)?;
    fs::write(output_path, format!("{json}\n")).map_err(|error| {
        CompiledAgentXtrainError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(receipt)
}

pub fn verify_compiled_agent_xtrain_fixtures() -> Result<(), CompiledAgentXtrainError> {
    let expected_route = canonical_compiled_agent_route_candidate_report();
    let committed_route: CompiledAgentModuleEvalReport = serde_json::from_slice(
        &fs::read(compiled_agent_route_candidate_report_fixture_path()).map_err(|error| {
            CompiledAgentXtrainError::Read {
                path: compiled_agent_route_candidate_report_fixture_path()
                    .display()
                    .to_string(),
                error,
            }
        })?,
    )?;
    if committed_route != expected_route {
        return Err(CompiledAgentXtrainError::Read {
            path: compiled_agent_route_candidate_report_fixture_path()
                .display()
                .to_string(),
            error: std::io::Error::other("route candidate eval report drift"),
        });
    }

    let expected_grounded = canonical_compiled_agent_grounded_candidate_report();
    let committed_grounded: CompiledAgentModuleEvalReport = serde_json::from_slice(
        &fs::read(compiled_agent_grounded_candidate_report_fixture_path()).map_err(|error| {
            CompiledAgentXtrainError::Read {
                path: compiled_agent_grounded_candidate_report_fixture_path()
                    .display()
                    .to_string(),
                error,
            }
        })?,
    )?;
    if committed_grounded != expected_grounded {
        return Err(CompiledAgentXtrainError::Read {
            path: compiled_agent_grounded_candidate_report_fixture_path()
                .display()
                .to_string(),
            error: std::io::Error::other("grounded candidate eval report drift"),
        });
    }

    let expected_cycle = canonical_compiled_agent_xtrain_cycle_receipt()?;
    let committed_cycle: CompiledAgentXtrainCycleReceipt = serde_json::from_slice(
        &fs::read(compiled_agent_xtrain_cycle_receipt_fixture_path()).map_err(|error| {
            CompiledAgentXtrainError::Read {
                path: compiled_agent_xtrain_cycle_receipt_fixture_path()
                    .display()
                    .to_string(),
                error,
            }
        })?,
    )?;
    if committed_cycle != expected_cycle {
        return Err(CompiledAgentXtrainError::Read {
            path: compiled_agent_xtrain_cycle_receipt_fixture_path()
                .display()
                .to_string(),
            error: std::io::Error::other("xtrain cycle receipt drift"),
        });
    }
    Ok(())
}

fn build_route_outcome(
    replay_bundle: &CompiledAgentReplayBundle,
    baseline_report: &CompiledAgentModuleEvalReport,
    candidate_report: &CompiledAgentModuleEvalReport,
    baseline_revision: &CompiledAgentModuleRevisionSet,
    candidate_revision: &CompiledAgentModuleRevisionSet,
) -> Result<CompiledAgentXtrainModuleOutcome, CompiledAgentXtrainError> {
    let validation = validate_route_candidate(
        replay_bundle,
        baseline_report,
        candidate_report,
        baseline_revision,
        candidate_revision,
    )?;
    let decision = if validation.regression_case_ids.is_empty()
        && validation.replay_regression_sample_ids.is_empty()
        && validation.candidate_passed_cases > validation.baseline_passed_cases
        && validation.candidate_replay_match_count > validation.baseline_replay_match_count
    {
        CompiledAgentPromotionDecision::Promote
    } else {
        CompiledAgentPromotionDecision::Hold
    };
    let reason = match decision {
        CompiledAgentPromotionDecision::Promote => String::from(
            "Candidate fixes the retained negated-route false positive with no module-eval or replay regressions.",
        ),
        CompiledAgentPromotionDecision::Hold => String::from(
            "Candidate did not clear the route validator gate cleanly enough to promote.",
        ),
    };
    Ok(CompiledAgentXtrainModuleOutcome {
        module: CompiledAgentModuleKind::Route,
        decision,
        reason,
        delta: CompiledAgentCandidateDelta {
            module: CompiledAgentModuleKind::Route,
            base_revision_id: baseline_revision.revision_id.clone(),
            candidate_revision_id: candidate_revision.revision_id.clone(),
            replay_bundle_digest: replay_bundle.bundle_digest.clone(),
            targeted_failure_classes: vec![String::from("negated_route_false_positive")],
            delta_summary: String::from(
                "Adds a bounded negation plus unsupported-context guard so `do not ... wallet balance ... poem` no longer routes to the wallet lane.",
            ),
        },
        validation,
    })
}

fn build_grounded_answer_outcome(
    replay_bundle: &CompiledAgentReplayBundle,
    baseline_report: &CompiledAgentModuleEvalReport,
    candidate_report: &CompiledAgentModuleEvalReport,
    baseline_revision: &CompiledAgentModuleRevisionSet,
    candidate_revision: &CompiledAgentModuleRevisionSet,
) -> Result<CompiledAgentXtrainModuleOutcome, CompiledAgentXtrainError> {
    let validation = validate_grounded_candidate(
        replay_bundle,
        baseline_report,
        candidate_report,
        baseline_revision,
        candidate_revision,
    )?;
    let decision = if validation.regression_case_ids.is_empty()
        && validation.replay_regression_sample_ids.is_empty()
        && validation.candidate_replay_match_count > validation.baseline_replay_match_count
        && validation.candidate_passed_cases >= validation.baseline_passed_cases
    {
        CompiledAgentPromotionDecision::Promote
    } else {
        CompiledAgentPromotionDecision::Hold
    };
    let reason = match decision {
        CompiledAgentPromotionDecision::Promote => String::from(
            "Candidate improves replay-target grounding on the wallet answer while keeping the independent module eval surface non-regressing.",
        ),
        CompiledAgentPromotionDecision::Hold => String::from(
            "Candidate did not clear the grounded-answer validator gate cleanly enough to promote.",
        ),
    };
    Ok(CompiledAgentXtrainModuleOutcome {
        module: CompiledAgentModuleKind::GroundedAnswer,
        decision,
        reason,
        delta: CompiledAgentCandidateDelta {
            module: CompiledAgentModuleKind::GroundedAnswer,
            base_revision_id: baseline_revision.revision_id.clone(),
            candidate_revision_id: candidate_revision.revision_id.clone(),
            replay_bundle_digest: replay_bundle.bundle_digest.clone(),
            targeted_failure_classes: vec![String::from("grounded_answer_mismatch")],
            delta_summary: String::from(
                "Extends the bounded wallet grounding template to include recent earnings when that fact is present in the receipt-backed tool result.",
            ),
        },
        validation,
    })
}

fn validate_route_candidate(
    replay_bundle: &CompiledAgentReplayBundle,
    baseline_report: &CompiledAgentModuleEvalReport,
    candidate_report: &CompiledAgentModuleEvalReport,
    baseline_revision: &CompiledAgentModuleRevisionSet,
    candidate_revision: &CompiledAgentModuleRevisionSet,
) -> Result<CompiledAgentModuleValidation, CompiledAgentXtrainError> {
    let baseline_summary = module_summary(baseline_report, CompiledAgentModuleKind::Route)?;
    let candidate_summary = module_summary(candidate_report, CompiledAgentModuleKind::Route)?;
    let improvement_case_ids = improvement_case_ids(
        baseline_report,
        candidate_report,
        CompiledAgentModuleKind::Route,
    );
    let regression_case_ids = regression_case_ids(
        baseline_report,
        candidate_report,
        CompiledAgentModuleKind::Route,
    );
    let (baseline_replay_match_count, baseline_replay_matches) =
        route_replay_matches(replay_bundle, baseline_revision)?;
    let (candidate_replay_match_count, candidate_replay_matches) =
        route_replay_matches(replay_bundle, candidate_revision)?;
    Ok(CompiledAgentModuleValidation {
        module: CompiledAgentModuleKind::Route,
        baseline_passed_cases: baseline_summary.passed_cases,
        candidate_passed_cases: candidate_summary.passed_cases,
        improvement_case_ids,
        regression_case_ids,
        baseline_replay_match_count,
        candidate_replay_match_count,
        replay_improvement_sample_ids: candidate_replay_matches
            .iter()
            .filter(|sample_id| !baseline_replay_matches.contains(sample_id))
            .cloned()
            .collect(),
        replay_regression_sample_ids: baseline_replay_matches
            .iter()
            .filter(|sample_id| !candidate_replay_matches.contains(sample_id))
            .cloned()
            .collect(),
    })
}

fn validate_grounded_candidate(
    replay_bundle: &CompiledAgentReplayBundle,
    baseline_report: &CompiledAgentModuleEvalReport,
    candidate_report: &CompiledAgentModuleEvalReport,
    baseline_revision: &CompiledAgentModuleRevisionSet,
    candidate_revision: &CompiledAgentModuleRevisionSet,
) -> Result<CompiledAgentModuleValidation, CompiledAgentXtrainError> {
    let baseline_summary =
        module_summary(baseline_report, CompiledAgentModuleKind::GroundedAnswer)?;
    let candidate_summary =
        module_summary(candidate_report, CompiledAgentModuleKind::GroundedAnswer)?;
    let improvement_case_ids = improvement_case_ids(
        baseline_report,
        candidate_report,
        CompiledAgentModuleKind::GroundedAnswer,
    );
    let regression_case_ids = regression_case_ids(
        baseline_report,
        candidate_report,
        CompiledAgentModuleKind::GroundedAnswer,
    );
    let (baseline_replay_match_count, baseline_replay_matches) =
        grounded_replay_matches(replay_bundle, baseline_revision)?;
    let (candidate_replay_match_count, candidate_replay_matches) =
        grounded_replay_matches(replay_bundle, candidate_revision)?;
    Ok(CompiledAgentModuleValidation {
        module: CompiledAgentModuleKind::GroundedAnswer,
        baseline_passed_cases: baseline_summary.passed_cases,
        candidate_passed_cases: candidate_summary.passed_cases,
        improvement_case_ids,
        regression_case_ids,
        baseline_replay_match_count,
        candidate_replay_match_count,
        replay_improvement_sample_ids: candidate_replay_matches
            .iter()
            .filter(|sample_id| !baseline_replay_matches.contains(sample_id))
            .cloned()
            .collect(),
        replay_regression_sample_ids: baseline_replay_matches
            .iter()
            .filter(|sample_id| !candidate_replay_matches.contains(sample_id))
            .cloned()
            .collect(),
    })
}

fn route_replay_matches(
    replay_bundle: &CompiledAgentReplayBundle,
    revision: &CompiledAgentModuleRevisionSet,
) -> Result<(u32, Vec<String>), CompiledAgentXtrainError> {
    let mut matches = Vec::new();
    for sample in replay_bundle
        .samples
        .iter()
        .filter(|sample| sample.module == CompiledAgentModuleKind::Route)
    {
        let Some(prompt) = sample.input.get("user_request").and_then(Value::as_str) else {
            return Err(CompiledAgentXtrainError::MissingRoutePrompt {
                sample_id: sample.sample_id.clone(),
            });
        };
        let observed = evaluate_compiled_agent_route(prompt, revision);
        let expected = parse_route(sample.expected_output.get("route"), &sample.sample_id)?;
        if observed == expected {
            matches.push(sample.sample_id.clone());
        }
    }
    Ok((matches.len() as u32, matches))
}

fn grounded_replay_matches(
    replay_bundle: &CompiledAgentReplayBundle,
    revision: &CompiledAgentModuleRevisionSet,
) -> Result<(u32, Vec<String>), CompiledAgentXtrainError> {
    let mut matches = Vec::new();
    for sample in replay_bundle
        .samples
        .iter()
        .filter(|sample| sample.module == CompiledAgentModuleKind::GroundedAnswer)
    {
        let route = parse_route(sample.input.get("route"), &sample.sample_id)?;
        let tool_results = parse_tool_results(sample.input.get("tool_results"), &sample.sample_id)?;
        let observed = evaluate_compiled_agent_grounded_answer(route, &tool_results, revision);
        let expected_kind =
            parse_public_kind(sample.expected_output.get("kind"), &sample.sample_id)?;
        let expected_response = sample
            .expected_output
            .get("response")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let observed_kind = public_kind_for_route(route);
        if observed_kind == expected_kind && observed == expected_response {
            matches.push(sample.sample_id.clone());
        }
    }
    Ok((matches.len() as u32, matches))
}

fn write_report(
    output_path: impl AsRef<Path>,
    report: &CompiledAgentModuleEvalReport,
) -> Result<CompiledAgentModuleEvalReport, CompiledAgentXtrainError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| CompiledAgentXtrainError::CreateDir {
            path: parent.display().to_string(),
            error,
        })?;
    }
    let json = serde_json::to_string_pretty(report)?;
    fs::write(output_path, format!("{json}\n")).map_err(|error| {
        CompiledAgentXtrainError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(report.clone())
}

fn module_summary<'a>(
    report: &'a CompiledAgentModuleEvalReport,
    module: CompiledAgentModuleKind,
) -> Result<&'a psionic_eval::CompiledAgentModuleEvalSummary, CompiledAgentXtrainError> {
    report
        .module_summaries
        .iter()
        .find(|summary| summary.module == module)
        .ok_or_else(|| CompiledAgentXtrainError::MissingModuleSummary {
            module: format!("{module:?}"),
        })
}

fn improvement_case_ids(
    baseline_report: &CompiledAgentModuleEvalReport,
    candidate_report: &CompiledAgentModuleEvalReport,
    module: CompiledAgentModuleKind,
) -> Vec<String> {
    candidate_report
        .case_reports
        .iter()
        .filter(|case| case.module == module && case.pass)
        .filter(|candidate_case| {
            baseline_report
                .case_reports
                .iter()
                .find(|baseline_case| baseline_case.case_id == candidate_case.case_id)
                .map(|baseline_case| !baseline_case.pass)
                .unwrap_or(false)
        })
        .map(|case| case.case_id.clone())
        .collect()
}

fn regression_case_ids(
    baseline_report: &CompiledAgentModuleEvalReport,
    candidate_report: &CompiledAgentModuleEvalReport,
    module: CompiledAgentModuleKind,
) -> Vec<String> {
    baseline_report
        .case_reports
        .iter()
        .filter(|case| case.module == module && case.pass)
        .filter(|baseline_case| {
            candidate_report
                .case_reports
                .iter()
                .find(|candidate_case| candidate_case.case_id == baseline_case.case_id)
                .map(|candidate_case| !candidate_case.pass)
                .unwrap_or(false)
        })
        .map(|case| case.case_id.clone())
        .collect()
}

fn parse_route(
    value: Option<&Value>,
    sample_id: &str,
) -> Result<CompiledAgentRoute, CompiledAgentXtrainError> {
    serde_json::from_value(value.cloned().unwrap_or(Value::Null)).map_err(|_| {
        CompiledAgentXtrainError::MissingGroundedRoute {
            sample_id: sample_id.to_string(),
        }
    })
}

fn parse_tool_results(
    value: Option<&Value>,
    sample_id: &str,
) -> Result<Vec<CompiledAgentToolResult>, CompiledAgentXtrainError> {
    serde_json::from_value(value.cloned().unwrap_or(Value::Array(Vec::new()))).map_err(|_| {
        CompiledAgentXtrainError::InvalidGroundedToolResults {
            sample_id: sample_id.to_string(),
        }
    })
}

fn parse_public_kind(
    value: Option<&Value>,
    sample_id: &str,
) -> Result<CompiledAgentPublicOutcomeKind, CompiledAgentXtrainError> {
    serde_json::from_value(value.cloned().unwrap_or(Value::Null)).map_err(|_| {
        CompiledAgentXtrainError::MissingGroundedRoute {
            sample_id: sample_id.to_string(),
        }
    })
}

fn public_kind_for_route(route: CompiledAgentRoute) -> CompiledAgentPublicOutcomeKind {
    match route {
        CompiledAgentRoute::Unsupported => CompiledAgentPublicOutcomeKind::UnsupportedRefusal,
        CompiledAgentRoute::ProviderStatus | CompiledAgentRoute::WalletStatus => {
            CompiledAgentPublicOutcomeKind::GroundedAnswer
        }
    }
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    hex::encode(hasher.finalize())
}
