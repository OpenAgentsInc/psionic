use std::{
    fs,
    path::{Path, PathBuf},
};

#[cfg(test)]
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use psionic_models::TassadarExecutorFixture;
use psionic_router::{
    TassadarDirectModelWeightRouteBindingError, bind_tassadar_direct_model_weight_route,
};
use psionic_runtime::{
    TassadarDirectModelWeightExecutionProofError, TassadarDirectModelWeightExecutionProofInput,
    TassadarDirectModelWeightExecutionProofReceipt, TassadarExecutorDecodeMode,
};

use crate::{
    LocalTassadarArticleExecutorSessionService, LocalTassadarExecutorService,
    LocalTassadarPlannerRouter, TassadarArticleExecutorSessionOutcome,
    TassadarArticleExecutorSessionRequest, TassadarArticleExecutorSessionResponse,
    TassadarPlannerRouteDescriptorError,
};

pub const TASSADAR_DIRECT_MODEL_WEIGHT_EXECUTION_PROOF_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_direct_model_weight_execution_proof_report.json";

const REPORT_SCHEMA_VERSION: u16 = 1;
const CANONICAL_CASE_IDS: [&str; 3] =
    ["long_loop_kernel", "sudoku_v0_test_a", "hungarian_matching"];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarDirectModelWeightExecutionProofReport {
    pub schema_version: u16,
    pub report_id: String,
    pub product_id: String,
    pub model_id: String,
    pub benchmark_report_ref: String,
    pub route_descriptor_digest: String,
    pub case_ids: Vec<String>,
    pub receipts: Vec<TassadarDirectModelWeightExecutionProofReceipt>,
    pub direct_case_count: u32,
    pub fallback_free_case_count: u32,
    pub zero_external_call_case_count: u32,
    pub claim_boundary: String,
    pub summary: String,
    pub report_digest: String,
}

impl TassadarDirectModelWeightExecutionProofReport {
    fn new(
        route_descriptor_digest: String,
        receipts: Vec<TassadarDirectModelWeightExecutionProofReceipt>,
    ) -> Self {
        let model_id = receipts
            .first()
            .map(|receipt| receipt.model_id.clone())
            .unwrap_or_default();
        let benchmark_report_ref = receipts
            .first()
            .map(|receipt| receipt.benchmark_report_ref.clone())
            .unwrap_or_default();
        let case_ids = receipts
            .iter()
            .map(|receipt| receipt.article_case_id.clone())
            .collect::<Vec<_>>();
        let direct_case_count = receipts
            .iter()
            .filter(|receipt| {
                receipt.selection_state == psionic_runtime::TassadarExecutorSelectionState::Direct
            })
            .count() as u32;
        let fallback_free_case_count = receipts
            .iter()
            .filter(|receipt| !receipt.fallback_observed)
            .count() as u32;
        let zero_external_call_case_count = receipts
            .iter()
            .filter(|receipt| receipt.external_call_count == 0)
            .count() as u32;
        let mut report = Self {
            schema_version: REPORT_SCHEMA_VERSION,
            report_id: String::from("tassadar.direct_model_weight_execution_proof.v1"),
            product_id: String::from(crate::ARTICLE_EXECUTOR_SESSION_PRODUCT_ID),
            model_id,
            benchmark_report_ref,
            route_descriptor_digest,
            case_ids,
            receipts,
            direct_case_count,
            fallback_free_case_count,
            zero_external_call_case_count,
            claim_boundary: String::from(
                "this report closes the direct model-weight execution claim only for the committed article reproduction workloads named here on the direct executor lane with explicit route, trace, proof, and zero-external-call posture. It does not imply later routes or undeclared workloads inherit the same proof without this report family",
            ),
            summary: String::new(),
            report_digest: String::new(),
        };
        report.summary = format!(
            "Direct model-weight execution proof now freezes {} canonical article workloads on route `{}` with direct_cases={}, fallback_free_cases={}, and zero_external_call_cases={}.",
            report.receipts.len(),
            report.route_descriptor_digest,
            report.direct_case_count,
            report.fallback_free_case_count,
            report.zero_external_call_case_count,
        );
        report.report_digest = stable_digest(
            b"psionic_tassadar_direct_model_weight_execution_proof_report|",
            &report,
        );
        report
    }
}

#[derive(Debug, Error)]
pub enum TassadarDirectModelWeightExecutionProofReportError {
    #[error(transparent)]
    ArticleSessionService(#[from] crate::TassadarArticleExecutorSessionServiceError),
    #[error(transparent)]
    RouteDescriptor(#[from] TassadarPlannerRouteDescriptorError),
    #[error(transparent)]
    RouteBinding(#[from] TassadarDirectModelWeightRouteBindingError),
    #[error(transparent)]
    ProofReceipt(#[from] TassadarDirectModelWeightExecutionProofError),
    #[error("article session `{case_id}` did not complete: {detail}")]
    CaseDidNotComplete { case_id: String, detail: String },
    #[error("article session `{case_id}` completed without a direct model-weight proof receipt")]
    MissingReceipt { case_id: String },
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to decode `{path}`: {error}")]
    Decode {
        path: String,
        error: serde_json::Error,
    },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub fn build_tassadar_direct_model_weight_execution_proof_receipt_for_article_session(
    executor_service: &LocalTassadarExecutorService,
    request: &TassadarArticleExecutorSessionRequest,
    response: &TassadarArticleExecutorSessionResponse,
) -> Result<
    TassadarDirectModelWeightExecutionProofReceipt,
    TassadarDirectModelWeightExecutionProofReportError,
> {
    let route_descriptor = LocalTassadarPlannerRouter::new()
        .with_executor_service(executor_service.clone())
        .route_capability_descriptor(Some(
            response
                .executor_response
                .model_descriptor
                .model
                .model_id
                .as_str(),
        ))?;
    let route_binding =
        bind_tassadar_direct_model_weight_route(&route_descriptor, request.requested_decode_mode)?;
    let selection = &response.executor_response.execution_report.selection;
    let compiled_backend_features = response
        .executor_response
        .evidence_bundle
        .proof_bundle
        .runtime_identity
        .backend_toolchain
        .compiled_backend_features
        .clone();
    let external_call_count = compiled_backend_features
        .iter()
        .filter(|feature| is_external_tool_marker(feature))
        .count() as u32;
    let external_tool_surface_observed = external_call_count > 0;
    let cpu_result_substitution_observed = compiled_backend_features
        .iter()
        .any(|feature| is_cpu_substitution_marker(feature));
    let fallback_observed = selection.selection_state
        != psionic_runtime::TassadarExecutorSelectionState::Direct
        || selection.effective_decode_mode != Some(request.requested_decode_mode);
    Ok(TassadarDirectModelWeightExecutionProofReceipt::new(
        TassadarDirectModelWeightExecutionProofInput {
            receipt_id: format!(
                "direct_model_weight_proof.{}",
                response.benchmark_identity.case_id
            ),
            benchmark_ref: response.benchmark_identity.benchmark_ref.clone(),
            benchmark_environment_ref: response
                .benchmark_identity
                .benchmark_environment_ref
                .clone(),
            benchmark_report_ref: response.benchmark_identity.benchmark_report_ref.clone(),
            workload_family_id: response.benchmark_identity.workload_family.clone(),
            article_case_id: response.benchmark_identity.case_id.clone(),
            article_case_summary: response.benchmark_identity.case_summary.clone(),
            executor_product_id: response.executor_response.product_id.clone(),
            model_id: response
                .executor_response
                .model_descriptor
                .model
                .model_id
                .clone(),
            model_descriptor_digest: response.executor_response.model_descriptor.stable_digest(),
            model_weight_bundle_digest: response
                .executor_response
                .model_descriptor
                .weights
                .digest
                .clone(),
            model_primary_artifact_digest: response
                .executor_response
                .model_descriptor
                .weights
                .primary_artifact_digest()
                .map(String::from),
            requested_decode_mode: request.requested_decode_mode,
            effective_decode_mode: selection.effective_decode_mode,
            selection_state: selection.selection_state,
            fallback_observed,
            external_call_count,
            external_tool_surface_observed,
            cpu_result_substitution_observed,
            compiled_backend_features,
            program_artifact_digest: response.proof_identity.program_artifact_digest.clone(),
            trace_artifact_digest: response.proof_identity.trace_artifact_digest.clone(),
            trace_digest: response.proof_identity.trace_digest.clone(),
            trace_proof_digest: response.proof_identity.trace_proof_digest.clone(),
            runtime_manifest_identity_digest: response
                .proof_identity
                .runtime_manifest_identity_digest
                .clone(),
            runtime_manifest_digest: response.proof_identity.runtime_manifest_digest.clone(),
            proof_bundle_request_digest: response
                .proof_identity
                .proof_bundle_request_digest
                .clone(),
            proof_bundle_model_id: response.proof_identity.proof_bundle_model_id.clone(),
            route_binding,
        },
    )?)
}

pub fn build_tassadar_direct_model_weight_execution_proof_report() -> Result<
    TassadarDirectModelWeightExecutionProofReport,
    TassadarDirectModelWeightExecutionProofReportError,
> {
    let executor_service = LocalTassadarExecutorService::new()
        .with_fixture(TassadarExecutorFixture::article_i32_compute_v1());
    let session_service = LocalTassadarArticleExecutorSessionService::new()
        .with_executor_service(executor_service.clone());
    let route_descriptor = LocalTassadarPlannerRouter::new()
        .with_executor_service(executor_service.clone())
        .route_capability_descriptor(Some(TassadarExecutorFixture::ARTICLE_I32_COMPUTE_MODEL_ID))?;
    let mut receipts = Vec::new();
    for case_id in CANONICAL_CASE_IDS {
        let request = TassadarArticleExecutorSessionRequest::new(
            format!("direct-proof-{case_id}"),
            case_id,
            TassadarExecutorDecodeMode::ReferenceLinear,
        )
        .with_requested_model_id(TassadarExecutorFixture::ARTICLE_I32_COMPUTE_MODEL_ID)
        .require_direct_model_weight_proof();
        match session_service.execute(&request)? {
            TassadarArticleExecutorSessionOutcome::Completed { response } => {
                let receipt = response
                    .direct_model_weight_execution_proof_receipt
                    .clone()
                    .ok_or_else(|| {
                        TassadarDirectModelWeightExecutionProofReportError::MissingReceipt {
                            case_id: String::from(case_id),
                        }
                    })?;
                receipts.push(receipt);
            }
            TassadarArticleExecutorSessionOutcome::Refused { refusal } => {
                return Err(
                    TassadarDirectModelWeightExecutionProofReportError::CaseDidNotComplete {
                        case_id: String::from(case_id),
                        detail: refusal.detail,
                    },
                );
            }
        }
    }
    Ok(TassadarDirectModelWeightExecutionProofReport::new(
        route_descriptor.descriptor_digest,
        receipts,
    ))
}

#[must_use]
pub fn tassadar_direct_model_weight_execution_proof_report_path() -> PathBuf {
    repo_root().join(TASSADAR_DIRECT_MODEL_WEIGHT_EXECUTION_PROOF_REPORT_REF)
}

pub fn write_tassadar_direct_model_weight_execution_proof_report(
    output_path: impl AsRef<Path>,
) -> Result<
    TassadarDirectModelWeightExecutionProofReport,
    TassadarDirectModelWeightExecutionProofReportError,
> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            TassadarDirectModelWeightExecutionProofReportError::CreateDir {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    let report = build_tassadar_direct_model_weight_execution_proof_report()?;
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(output_path, format!("{json}\n")).map_err(|error| {
        TassadarDirectModelWeightExecutionProofReportError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(report)
}

fn is_external_tool_marker(feature: &str) -> bool {
    feature.contains("external_tool")
        || feature.contains("tool_call")
        || feature.contains("external_call")
}

fn is_cpu_substitution_marker(feature: &str) -> bool {
    feature.contains("cpu_result_substitution") || feature.contains("result_substitution")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("psionic-serve should live under <repo>/crates/psionic-serve")
        .to_path_buf()
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
fn read_repo_json<T: DeserializeOwned>(
    relative_path: &str,
) -> Result<T, TassadarDirectModelWeightExecutionProofReportError> {
    let path = repo_root().join(relative_path);
    let bytes = fs::read(&path).map_err(|error| {
        TassadarDirectModelWeightExecutionProofReportError::Read {
            path: path.display().to_string(),
            error,
        }
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        TassadarDirectModelWeightExecutionProofReportError::Decode {
            path: path.display().to_string(),
            error,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        TASSADAR_DIRECT_MODEL_WEIGHT_EXECUTION_PROOF_REPORT_REF,
        TassadarDirectModelWeightExecutionProofReport,
        build_tassadar_direct_model_weight_execution_proof_report, read_repo_json,
        tassadar_direct_model_weight_execution_proof_report_path,
        write_tassadar_direct_model_weight_execution_proof_report,
    };

    #[test]
    fn direct_model_weight_execution_proof_report_is_machine_legible() {
        let report = build_tassadar_direct_model_weight_execution_proof_report().expect("report");

        assert_eq!(report.receipts.len(), 3);
        assert_eq!(report.direct_case_count, 3);
        assert_eq!(report.fallback_free_case_count, 3);
        assert_eq!(report.zero_external_call_case_count, 3);
        assert!(report.case_ids.contains(&String::from("long_loop_kernel")));
        assert!(report.case_ids.contains(&String::from("sudoku_v0_test_a")));
        assert!(
            report
                .case_ids
                .contains(&String::from("hungarian_matching"))
        );
        assert!(
            report
                .receipts
                .iter()
                .all(|receipt| receipt.route_binding.route_descriptor_digest
                    == report.route_descriptor_digest)
        );
    }

    #[test]
    fn direct_model_weight_execution_proof_report_matches_committed_truth() {
        let generated =
            build_tassadar_direct_model_weight_execution_proof_report().expect("report");
        let committed: TassadarDirectModelWeightExecutionProofReport =
            read_repo_json(TASSADAR_DIRECT_MODEL_WEIGHT_EXECUTION_PROOF_REPORT_REF)
                .expect("committed report");
        assert_eq!(generated, committed);
    }

    #[test]
    fn write_direct_model_weight_execution_proof_report_persists_current_truth() {
        let directory = tempfile::tempdir().expect("tempdir");
        let output_path = directory
            .path()
            .join("tassadar_direct_model_weight_execution_proof_report.json");
        let written = write_tassadar_direct_model_weight_execution_proof_report(&output_path)
            .expect("write report");
        let persisted: TassadarDirectModelWeightExecutionProofReport =
            serde_json::from_slice(&std::fs::read(&output_path).expect("read")).expect("decode");
        assert_eq!(written, persisted);
        assert_eq!(
            tassadar_direct_model_weight_execution_proof_report_path()
                .file_name()
                .and_then(|value| value.to_str()),
            Some("tassadar_direct_model_weight_execution_proof_report.json")
        );
    }
}
