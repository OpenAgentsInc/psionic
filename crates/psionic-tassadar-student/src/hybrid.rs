//! H1 hybrid ring gate: frozen compiled core + learned interface.
//!
//! This module keeps the W3 Baseline D result machine-checkable for the
//! later hybrid lane. It validates the retained sweep artifacts and the
//! frozen-core contract: gradients may train the learned marshaling
//! interface, but they must not target or mutate the compiled exact core.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const BASELINE_D_FROZEN_EXECUTOR_LEARNED_INTERFACE: &str =
    "baseline_d_frozen_executor_learned_interface";
pub const H1_HYBRID_REPORT_VERSION: &str = "tassadar_h1_frozen_core_learned_interface_report.v1";

const DOCUMENTED_BASELINE_D_PASS_AT_1: f64 = 1.0;
const DOCUMENTED_BASELINE_D_REPLAY_ACCEPTANCE: f64 = 1.0;
const DOCUMENTED_BASELINE_D_OUTPUT_DIGEST_MATCH: f64 = 1.0;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum H1HybridGateError {
    #[error("{artifact} json did not decode: {reason}")]
    Json {
        artifact: &'static str,
        reason: String,
    },
    #[error("baseline D artifact is missing from the W3 manifest")]
    MissingBaselineD,
    #[error("{field} does not match the retained W3 Baseline D artifact")]
    ArtifactMismatch { field: &'static str },
    #[error("{metric} is below the documented W3 Baseline D value")]
    MetricBelowBaselineD { metric: &'static str },
    #[error("frozen compiled core evidence digest changed across the window")]
    FrozenCoreMutated,
    #[error("gradient target scope includes the compiled exact core")]
    GradientTargetsFrozenCore,
    #[error("{field} must be public-safe")]
    UnsafeRef { field: &'static str },
    #[error("the learned interface trace must remain part of the forward pass")]
    TraceNotInForwardPass,
    #[error("compiled exact core must be listed as frozen")]
    MissingFrozenCoreScope,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenCoreLearnedInterfaceConfig {
    pub frozen_core_ref: String,
    pub frozen_core_digest_before: String,
    pub frozen_core_digest_after: String,
    pub frozen_parameter_scopes: Vec<String>,
    pub trainable_parameter_scopes: Vec<String>,
    pub gradient_updates_target_frozen_core: bool,
    pub trace_part_of_forward_pass: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenCoreLearnedInterfaceAuthority {
    pub canonical_checkpoint_mutation_authority: bool,
    pub compiled_core_gradient_mutation_allowed: bool,
    pub learned_interface_training_allowed: bool,
    pub public_gradient_window_promotion_required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrozenCoreLearnedInterfaceReport {
    pub report_version: String,
    pub baseline: String,
    pub corpus_id: String,
    pub dataset_snapshot_digest: String,
    pub eval_report_sha256: String,
    pub receipt_sha256: String,
    pub interface_sha256: String,
    pub interface_digest: String,
    pub frozen_core_ref: String,
    pub frozen_core_digest_before: String,
    pub frozen_core_digest_after: String,
    pub frozen_core_unchanged: bool,
    pub exact_rollout_pass_at_1: f64,
    pub output_digest_match_rate: f64,
    pub replay_verifier_acceptance_rate: f64,
    pub first_divergence_step_median: f64,
    pub first_divergence_step_p90: f64,
    pub valid_prefix_tokens_median: f64,
    pub trainable_parameter_scopes: Vec<String>,
    pub frozen_parameter_scopes: Vec<String>,
    pub authority: FrozenCoreLearnedInterfaceAuthority,
    pub source_artifact_refs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct W3SweepManifest {
    corpus_id: String,
    dataset_snapshot_digest: String,
    train_prep_sha256: String,
    eval_prep_sha256: String,
    baselines: Vec<W3SweepBaseline>,
}

#[derive(Clone, Debug, Deserialize)]
struct W3SweepBaseline {
    baseline: String,
    directory: String,
    eval_report_sha256: String,
    receipt_sha256: String,
    interface_sha256: Option<String>,
    checkpoint_sha256: String,
    config_digest: String,
    exact_rollout_pass_at_1: f64,
    replay_verifier_acceptance_rate: f64,
    first_divergence_step_median: f64,
    first_divergence_step_p90: f64,
    valid_prefix_tokens_median: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct W3EvalReport {
    baseline: String,
    corpus_id: String,
    dataset_snapshot_digest: String,
    eval_prep_sha256: String,
    checkpoint_sha256: String,
    config_digest: String,
    overall: W3EvalOverall,
}

#[derive(Clone, Debug, Deserialize)]
struct W3EvalOverall {
    exact_rollout_pass_at_1: f64,
    first_divergence_step_median: f64,
    first_divergence_step_p90: f64,
    valid_prefix_tokens_median: f64,
    output_digest_match_rate: f64,
    replay_verifier_acceptance_rate: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct W3InterfaceReceipt {
    baseline: String,
    corpus_id: String,
    dataset_snapshot_digest: String,
    train_prep_sha256: String,
    interface_digest: String,
}

pub fn baseline_d_frozen_core_config_from_manifest_json(
    manifest_json: &str,
) -> Result<FrozenCoreLearnedInterfaceConfig, H1HybridGateError> {
    let manifest: W3SweepManifest = parse_json("manifest", manifest_json)?;
    let _ = baseline_d_from_manifest(&manifest)?;
    let frozen_core_digest = stable_digest(
        "tassadar_h1_frozen_core.v1",
        &[
            manifest.corpus_id.as_str(),
            manifest.dataset_snapshot_digest.as_str(),
            manifest.eval_prep_sha256.as_str(),
            "psionic_compiler::tassadar_alm_numeric_execute",
            "digest_pinned_tassadar_alm_numeric_model_graphs",
        ],
    );

    Ok(FrozenCoreLearnedInterfaceConfig {
        frozen_core_ref: String::from("core.public.psionic.tassadar_alm_numeric_executor.v1"),
        frozen_core_digest_before: frozen_core_digest.clone(),
        frozen_core_digest_after: frozen_core_digest,
        frozen_parameter_scopes: vec![
            String::from("compiled_exact_core.tassadar_alm_numeric_executor"),
            String::from("compiled_exact_core.digest_pinned_numeric_models"),
        ],
        trainable_parameter_scopes: vec![
            String::from("learned_interface.input_limb_assignment"),
            String::from("learned_interface.output_limb_assignment"),
            String::from("learned_interface.output_routing"),
        ],
        gradient_updates_target_frozen_core: false,
        trace_part_of_forward_pass: true,
    })
}

pub fn verify_w3_baseline_d_hybrid_ring(
    manifest_json: &str,
    eval_report_json: &str,
    receipt_json: &str,
    config: FrozenCoreLearnedInterfaceConfig,
) -> Result<FrozenCoreLearnedInterfaceReport, H1HybridGateError> {
    let manifest: W3SweepManifest = parse_json("manifest", manifest_json)?;
    let eval_report: W3EvalReport = parse_json("eval report", eval_report_json)?;
    let receipt: W3InterfaceReceipt = parse_json("receipt", receipt_json)?;
    let baseline = baseline_d_from_manifest(&manifest)?;

    assert_artifact_match(
        baseline.baseline == BASELINE_D_FROZEN_EXECUTOR_LEARNED_INTERFACE,
        "manifest.baseline",
    )?;
    assert_artifact_match(baseline.directory == "d", "manifest.directory")?;
    assert_artifact_match(
        eval_report.baseline == BASELINE_D_FROZEN_EXECUTOR_LEARNED_INTERFACE,
        "eval.baseline",
    )?;
    assert_artifact_match(
        receipt.baseline == BASELINE_D_FROZEN_EXECUTOR_LEARNED_INTERFACE,
        "receipt.baseline",
    )?;
    assert_artifact_match(
        eval_report.corpus_id == manifest.corpus_id,
        "eval.corpus_id",
    )?;
    assert_artifact_match(receipt.corpus_id == manifest.corpus_id, "receipt.corpus_id")?;
    assert_artifact_match(
        eval_report.dataset_snapshot_digest == manifest.dataset_snapshot_digest,
        "eval.dataset_snapshot_digest",
    )?;
    assert_artifact_match(
        receipt.dataset_snapshot_digest == manifest.dataset_snapshot_digest,
        "receipt.dataset_snapshot_digest",
    )?;
    assert_artifact_match(
        eval_report.eval_prep_sha256 == manifest.eval_prep_sha256,
        "eval.eval_prep_sha256",
    )?;
    assert_artifact_match(
        receipt.train_prep_sha256 == manifest.train_prep_sha256,
        "receipt.train_prep_sha256",
    )?;
    assert_artifact_match(
        eval_report.checkpoint_sha256 == baseline.checkpoint_sha256,
        "eval.checkpoint_sha256",
    )?;
    assert_artifact_match(
        eval_report.config_digest == baseline.config_digest,
        "eval.config_digest",
    )?;
    assert_artifact_match(
        receipt.interface_digest == baseline.checkpoint_sha256,
        "receipt.interface_digest",
    )?;
    assert_artifact_match(
        same_metric(
            baseline.first_divergence_step_median,
            eval_report.overall.first_divergence_step_median,
        ),
        "eval.first_divergence_step_median",
    )?;
    assert_artifact_match(
        same_metric(
            baseline.first_divergence_step_p90,
            eval_report.overall.first_divergence_step_p90,
        ),
        "eval.first_divergence_step_p90",
    )?;
    assert_artifact_match(
        same_metric(
            baseline.valid_prefix_tokens_median,
            eval_report.overall.valid_prefix_tokens_median,
        ),
        "eval.valid_prefix_tokens_median",
    )?;

    metric_at_least(
        "manifest exact_rollout_pass_at_1",
        baseline.exact_rollout_pass_at_1,
        DOCUMENTED_BASELINE_D_PASS_AT_1,
    )?;
    metric_at_least(
        "eval exact_rollout_pass_at_1",
        eval_report.overall.exact_rollout_pass_at_1,
        DOCUMENTED_BASELINE_D_PASS_AT_1,
    )?;
    metric_at_least(
        "manifest replay_verifier_acceptance_rate",
        baseline.replay_verifier_acceptance_rate,
        DOCUMENTED_BASELINE_D_REPLAY_ACCEPTANCE,
    )?;
    metric_at_least(
        "eval replay_verifier_acceptance_rate",
        eval_report.overall.replay_verifier_acceptance_rate,
        DOCUMENTED_BASELINE_D_REPLAY_ACCEPTANCE,
    )?;
    metric_at_least(
        "eval output_digest_match_rate",
        eval_report.overall.output_digest_match_rate,
        DOCUMENTED_BASELINE_D_OUTPUT_DIGEST_MATCH,
    )?;

    validate_config(&config)?;

    Ok(FrozenCoreLearnedInterfaceReport {
        report_version: String::from(H1_HYBRID_REPORT_VERSION),
        baseline: String::from(BASELINE_D_FROZEN_EXECUTOR_LEARNED_INTERFACE),
        corpus_id: manifest.corpus_id.clone(),
        dataset_snapshot_digest: manifest.dataset_snapshot_digest.clone(),
        eval_report_sha256: baseline.eval_report_sha256.clone(),
        receipt_sha256: baseline.receipt_sha256.clone(),
        interface_sha256: baseline.interface_sha256.clone().unwrap_or_default(),
        interface_digest: receipt.interface_digest,
        frozen_core_ref: config.frozen_core_ref,
        frozen_core_digest_before: config.frozen_core_digest_before,
        frozen_core_digest_after: config.frozen_core_digest_after,
        frozen_core_unchanged: true,
        exact_rollout_pass_at_1: eval_report.overall.exact_rollout_pass_at_1,
        output_digest_match_rate: eval_report.overall.output_digest_match_rate,
        replay_verifier_acceptance_rate: eval_report.overall.replay_verifier_acceptance_rate,
        first_divergence_step_median: eval_report.overall.first_divergence_step_median,
        first_divergence_step_p90: eval_report.overall.first_divergence_step_p90,
        valid_prefix_tokens_median: eval_report.overall.valid_prefix_tokens_median,
        trainable_parameter_scopes: config.trainable_parameter_scopes,
        frozen_parameter_scopes: config.frozen_parameter_scopes,
        authority: FrozenCoreLearnedInterfaceAuthority {
            canonical_checkpoint_mutation_authority: false,
            compiled_core_gradient_mutation_allowed: false,
            learned_interface_training_allowed: true,
            public_gradient_window_promotion_required: true,
        },
        source_artifact_refs: vec![
            String::from("fixtures/tassadar/w3_student_sweep_20260612/manifest.json"),
            String::from("fixtures/tassadar/w3_student_sweep_20260612/d/eval-report.json"),
            String::from("fixtures/tassadar/w3_student_sweep_20260612/d/receipt.json"),
            String::from("fixtures/tassadar/w3_student_sweep_20260612/d/interface.json"),
        ],
    })
}

fn parse_json<T: for<'de> Deserialize<'de>>(
    artifact: &'static str,
    value: &str,
) -> Result<T, H1HybridGateError> {
    serde_json::from_str(value).map_err(|error| H1HybridGateError::Json {
        artifact,
        reason: error.to_string(),
    })
}

fn baseline_d_from_manifest(
    manifest: &W3SweepManifest,
) -> Result<&W3SweepBaseline, H1HybridGateError> {
    manifest
        .baselines
        .iter()
        .find(|baseline| baseline.baseline == BASELINE_D_FROZEN_EXECUTOR_LEARNED_INTERFACE)
        .ok_or(H1HybridGateError::MissingBaselineD)
}

fn assert_artifact_match(condition: bool, field: &'static str) -> Result<(), H1HybridGateError> {
    if condition {
        Ok(())
    } else {
        Err(H1HybridGateError::ArtifactMismatch { field })
    }
}

fn metric_at_least(
    metric: &'static str,
    actual: f64,
    expected: f64,
) -> Result<(), H1HybridGateError> {
    if actual.is_finite() && actual >= expected {
        Ok(())
    } else {
        Err(H1HybridGateError::MetricBelowBaselineD { metric })
    }
}

fn same_metric(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= f64::EPSILON
}

fn stable_digest(label: &str, parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(label.as_bytes());
    for part in parts {
        hasher.update(b"|");
        hasher.update(part.as_bytes());
    }
    hex::encode(hasher.finalize())
}

fn validate_config(config: &FrozenCoreLearnedInterfaceConfig) -> Result<(), H1HybridGateError> {
    validate_safe_ref("frozen_core_ref", &config.frozen_core_ref)?;
    validate_safe_ref(
        "frozen_core_digest_before",
        &config.frozen_core_digest_before,
    )?;
    validate_safe_ref("frozen_core_digest_after", &config.frozen_core_digest_after)?;
    for scope in &config.trainable_parameter_scopes {
        validate_safe_ref("trainable_parameter_scopes", scope)?;
    }
    for scope in &config.frozen_parameter_scopes {
        validate_safe_ref("frozen_parameter_scopes", scope)?;
    }

    if config.frozen_core_digest_before != config.frozen_core_digest_after {
        return Err(H1HybridGateError::FrozenCoreMutated);
    }
    if config.gradient_updates_target_frozen_core
        || config
            .trainable_parameter_scopes
            .iter()
            .any(|scope| scope_targets_frozen_core(scope))
    {
        return Err(H1HybridGateError::GradientTargetsFrozenCore);
    }
    if !config.trace_part_of_forward_pass {
        return Err(H1HybridGateError::TraceNotInForwardPass);
    }
    if !config
        .frozen_parameter_scopes
        .iter()
        .any(|scope| scope_targets_frozen_core(scope))
    {
        return Err(H1HybridGateError::MissingFrozenCoreScope);
    }

    Ok(())
}

fn scope_targets_frozen_core(scope: &str) -> bool {
    let lower = scope.to_ascii_lowercase();
    lower.contains("compiled_exact_core")
        || lower.contains("compiled_core")
        || lower.contains("exact_core")
        || lower.contains("frozen_core")
        || lower.contains("analytic_executor")
        || lower.contains("tassadar_alm_numeric_execute")
        || lower.contains("tassadar_alm_numeric_executor")
}

fn validate_safe_ref(field: &'static str, value: &str) -> Result<(), H1HybridGateError> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err(H1HybridGateError::UnsafeRef { field });
    };
    if !first.is_ascii_alphanumeric()
        || value.len() > 260
        || !value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | ':' | '/' | '-'))
        || contains_private_marker(value)
    {
        return Err(H1HybridGateError::UnsafeRef { field });
    }
    Ok(())
}

fn contains_private_marker(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    [
        "/users/",
        "/home/",
        "access_token",
        "auth.json",
        "bearer",
        "customer",
        "invoice",
        "mnemonic",
        "payment",
        "payout",
        "preimage",
        "private",
        "raw",
        "secret",
        "token",
        "trace_payload",
        "wallet",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str =
        include_str!("../../../fixtures/tassadar/w3_student_sweep_20260612/manifest.json");
    const EVAL_REPORT: &str =
        include_str!("../../../fixtures/tassadar/w3_student_sweep_20260612/d/eval-report.json");
    const RECEIPT: &str =
        include_str!("../../../fixtures/tassadar/w3_student_sweep_20260612/d/receipt.json");

    #[test]
    fn retained_baseline_d_fixture_reproduces_h2_with_frozen_core() {
        let config =
            baseline_d_frozen_core_config_from_manifest_json(MANIFEST).expect("baseline D config");
        let report = verify_w3_baseline_d_hybrid_ring(MANIFEST, EVAL_REPORT, RECEIPT, config)
            .expect("baseline D H1 report");

        assert_eq!(report.report_version, H1_HYBRID_REPORT_VERSION);
        assert_eq!(report.exact_rollout_pass_at_1, 1.0);
        assert_eq!(report.replay_verifier_acceptance_rate, 1.0);
        assert_eq!(report.output_digest_match_rate, 1.0);
        assert!(report.frozen_core_unchanged);
        assert!(!report.authority.compiled_core_gradient_mutation_allowed);
        assert!(report.authority.learned_interface_training_allowed);
        assert!(!report.authority.canonical_checkpoint_mutation_authority);
    }

    #[test]
    fn frozen_core_digest_change_is_rejected() {
        let mut config =
            baseline_d_frozen_core_config_from_manifest_json(MANIFEST).expect("baseline D config");
        config.frozen_core_digest_after = String::from("digest.sha256.changed");

        let error = verify_w3_baseline_d_hybrid_ring(MANIFEST, EVAL_REPORT, RECEIPT, config)
            .expect_err("mutated core must reject");

        assert_eq!(error, H1HybridGateError::FrozenCoreMutated);
    }

    #[test]
    fn trainable_compiled_core_scope_is_rejected() {
        let mut config =
            baseline_d_frozen_core_config_from_manifest_json(MANIFEST).expect("baseline D config");
        config
            .trainable_parameter_scopes
            .push(String::from("compiled_exact_core.ffn_bank"));

        let error = verify_w3_baseline_d_hybrid_ring(MANIFEST, EVAL_REPORT, RECEIPT, config)
            .expect_err("trainable core scope must reject");

        assert_eq!(error, H1HybridGateError::GradientTargetsFrozenCore);
    }

    #[test]
    fn unsafe_core_ref_is_rejected() {
        let mut config =
            baseline_d_frozen_core_config_from_manifest_json(MANIFEST).expect("baseline D config");
        config.frozen_core_ref = String::from("/Users/example/private_core");

        let error = verify_w3_baseline_d_hybrid_ring(MANIFEST, EVAL_REPORT, RECEIPT, config)
            .expect_err("unsafe ref must reject");

        assert_eq!(
            error,
            H1HybridGateError::UnsafeRef {
                field: "frozen_core_ref",
            },
        );
    }
}
