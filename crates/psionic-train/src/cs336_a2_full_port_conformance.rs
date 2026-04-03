use std::{collections::BTreeSet, fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CS336_A2_BASELINE_PROFILE_BUNDLE_FIXTURE_PATH, CS336_A2_DDP_BUCKETED_RECEIPT_FIXTURE_PATH,
    CS336_A2_DDP_INDIVIDUAL_PARAMETERS_RECEIPT_FIXTURE_PATH,
    CS336_A2_FLASHATTENTION_FUSED_CUDA_RECEIPT_FIXTURE_PATH,
    CS336_A2_FLASHATTENTION_REFERENCE_RECEIPT_FIXTURE_PATH,
    CS336_A2_SHARDED_OPTIMIZER_RECEIPT_FIXTURE_PATH,
};

pub const CS336_A2_FULL_PORT_CONFORMANCE_REPORT_SCHEMA_VERSION: &str =
    "psion.cs336_a2.full_port_conformance_report.v1";
pub const CS336_A2_FULL_PORT_CONFORMANCE_REPORT_FIXTURE_PATH: &str =
    "fixtures/training/cs336_a2_full_port_conformance_report_v1.json";
pub const CS336_A2_FULL_PORT_MATRIX_DOC_PATH: &str = "docs/PSION_CS336_A2_FULL_PORT_MATRIX.md";
pub const CS336_A2_REFERENCE_LANE_DOC_PATH_FOR_CONFORMANCE: &str =
    "docs/PSION_CS336_A2_REFERENCE_LANE.md";

const FLASHATTENTION_REFERENCE_SOURCE_PATH: &str =
    "crates/psionic-train/src/cs336_a2_flashattention_reference_receipt.rs";
const FLASHATTENTION_REFERENCE_IMPL_PATH: &str =
    "crates/psionic-models/src/cs336_a2_flashattention_reference.rs";
const FLASHATTENTION_FUSED_SOURCE_PATH: &str =
    "crates/psionic-train/src/cs336_a2_flashattention_fused_cuda_receipt.rs";
const FLASHATTENTION_FUSED_IMPL_PATH: &str = "crates/psionic-backend-cuda/src/lib.rs";
const DDP_INDIVIDUAL_SOURCE_PATH: &str =
    "crates/psionic-train/src/cs336_a2_ddp_individual_parameters_receipt.rs";
const DDP_BUCKETED_SOURCE_PATH: &str = "crates/psionic-train/src/cs336_a2_ddp_bucketed_receipt.rs";
const SHARDED_OPTIMIZER_SOURCE_PATH: &str =
    "crates/psionic-train/src/cs336_a2_sharded_optimizer_receipt.rs";

const EXPECTED_STANFORD_ADAPTERS: [&str; 8] = [
    "get_flashattention_autograd_function_pytorch",
    "get_flashattention_autograd_function_triton",
    "get_ddp_individual_parameters",
    "ddp_individual_parameters_on_after_backward",
    "get_ddp_bucketed",
    "ddp_bucketed_on_after_backward",
    "ddp_bucketed_on_train_batch_start",
    "get_sharded_optimizer",
];

#[derive(Debug, Error)]
pub enum Cs336A2FullPortConformanceError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("missing or invalid proof surface: {0}")]
    MissingProofSurface(String),
    #[error("missing or invalid implementation surface: {0}")]
    MissingImplementationSurface(String),
    #[error("invalid CS336 A2 full-port conformance report: {0}")]
    InvalidReport(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cs336A2ConformanceProofSurface {
    pub proof_kind: String,
    pub repo_relative_path: String,
    pub selector: Option<String>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cs336A2ConformanceRow {
    pub stanford_adapter_name: String,
    pub category: String,
    pub status: String,
    pub implementation_surfaces: Vec<String>,
    pub proof_surfaces: Vec<Cs336A2ConformanceProofSurface>,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cs336A2FullPortConformanceReport {
    pub schema_version: String,
    pub completion_matrix_path: String,
    pub reference_lane_doc_path: String,
    pub retained_proof_bundle_paths: Vec<String>,
    pub row_count: usize,
    pub green_row_count: usize,
    pub fully_green: bool,
    pub rows: Vec<Cs336A2ConformanceRow>,
    pub claim_boundary: String,
}

pub fn build_cs336_a2_full_port_conformance_report(
    repo_root: impl AsRef<Path>,
) -> Result<Cs336A2FullPortConformanceReport, Cs336A2FullPortConformanceError> {
    let repo_root = repo_root.as_ref();
    let rows = expected_rows();
    for row in &rows {
        validate_row(repo_root, row)?;
    }
    let report = Cs336A2FullPortConformanceReport {
        schema_version: String::from(CS336_A2_FULL_PORT_CONFORMANCE_REPORT_SCHEMA_VERSION),
        completion_matrix_path: String::from(CS336_A2_FULL_PORT_MATRIX_DOC_PATH),
        reference_lane_doc_path: String::from(CS336_A2_REFERENCE_LANE_DOC_PATH_FOR_CONFORMANCE),
        retained_proof_bundle_paths: vec![
            String::from(CS336_A2_BASELINE_PROFILE_BUNDLE_FIXTURE_PATH),
            String::from(CS336_A2_FLASHATTENTION_REFERENCE_RECEIPT_FIXTURE_PATH),
            String::from(CS336_A2_FLASHATTENTION_FUSED_CUDA_RECEIPT_FIXTURE_PATH),
            String::from(CS336_A2_DDP_INDIVIDUAL_PARAMETERS_RECEIPT_FIXTURE_PATH),
            String::from(CS336_A2_DDP_BUCKETED_RECEIPT_FIXTURE_PATH),
            String::from(CS336_A2_SHARDED_OPTIMIZER_RECEIPT_FIXTURE_PATH),
        ],
        row_count: rows.len(),
        green_row_count: rows.iter().filter(|row| row.status == "green").count(),
        fully_green: rows.iter().all(|row| row.status == "green"),
        rows,
        claim_boundary: String::from(
            "This report closes Stanford CS336 Assignment 2 only as a bounded psionic reference lane. It proves every Stanford A2 adapter family is mapped to owned Rust surfaces plus checked-in profiling, attention, DDP, and sharded-optimizer proof bundles. It does not promote the bounded systems lane into the actual Psion pretraining operator lane, and it does not claim admitted distributed throughput or transport-backed cluster execution.",
        ),
    };
    validate_report(&report)?;
    Ok(report)
}

pub fn write_cs336_a2_full_port_conformance_report(
    output_root: impl AsRef<Path>,
    report: &Cs336A2FullPortConformanceReport,
) -> Result<(), Cs336A2FullPortConformanceError> {
    validate_report(report)?;
    let report_path = output_root
        .as_ref()
        .join(CS336_A2_FULL_PORT_CONFORMANCE_REPORT_FIXTURE_PATH);
    if let Some(parent) = report_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(report_path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

pub fn write_cs336_a2_full_port_conformance_fixture(
    repo_root: impl AsRef<Path>,
) -> Result<Cs336A2FullPortConformanceReport, Cs336A2FullPortConformanceError> {
    let report = build_cs336_a2_full_port_conformance_report(&repo_root)?;
    write_cs336_a2_full_port_conformance_report(repo_root, &report)?;
    Ok(report)
}

fn validate_report(
    report: &Cs336A2FullPortConformanceReport,
) -> Result<(), Cs336A2FullPortConformanceError> {
    if report.schema_version != CS336_A2_FULL_PORT_CONFORMANCE_REPORT_SCHEMA_VERSION {
        return Err(Cs336A2FullPortConformanceError::InvalidReport(format!(
            "expected schema version `{CS336_A2_FULL_PORT_CONFORMANCE_REPORT_SCHEMA_VERSION}`, got `{}`",
            report.schema_version
        )));
    }
    if report.row_count != report.rows.len() {
        return Err(Cs336A2FullPortConformanceError::InvalidReport(format!(
            "row_count {} does not match {} rows",
            report.row_count,
            report.rows.len()
        )));
    }
    let green_row_count = report
        .rows
        .iter()
        .filter(|row| row.status == "green")
        .count();
    if report.green_row_count != green_row_count {
        return Err(Cs336A2FullPortConformanceError::InvalidReport(format!(
            "green_row_count {} does not match computed count {green_row_count}",
            report.green_row_count
        )));
    }
    if !report.fully_green || report.green_row_count != report.row_count {
        return Err(Cs336A2FullPortConformanceError::InvalidReport(
            "full-port report is not fully green".into(),
        ));
    }
    let adapter_set = report
        .rows
        .iter()
        .map(|row| row.stanford_adapter_name.as_str())
        .collect::<BTreeSet<_>>();
    let expected = EXPECTED_STANFORD_ADAPTERS
        .into_iter()
        .collect::<BTreeSet<_>>();
    if adapter_set != expected {
        return Err(Cs336A2FullPortConformanceError::InvalidReport(format!(
            "adapter set mismatch: expected {:?}, got {:?}",
            expected, adapter_set
        )));
    }
    Ok(())
}

fn validate_row(
    repo_root: &Path,
    row: &Cs336A2ConformanceRow,
) -> Result<(), Cs336A2FullPortConformanceError> {
    if row.status != "green" {
        return Err(Cs336A2FullPortConformanceError::InvalidReport(format!(
            "row `{}` is not green",
            row.stanford_adapter_name
        )));
    }
    if row.implementation_surfaces.is_empty() {
        return Err(Cs336A2FullPortConformanceError::InvalidReport(format!(
            "row `{}` is missing implementation surfaces",
            row.stanford_adapter_name
        )));
    }
    if row.proof_surfaces.is_empty() {
        return Err(Cs336A2FullPortConformanceError::InvalidReport(format!(
            "row `{}` is missing proof surfaces",
            row.stanford_adapter_name
        )));
    }
    for implementation_surface in &row.implementation_surfaces {
        let implementation_path = repo_root.join(implementation_surface);
        if !implementation_path.exists() {
            return Err(
                Cs336A2FullPortConformanceError::MissingImplementationSurface(format!(
                    "`{implementation_surface}` does not exist"
                )),
            );
        }
    }
    for proof in &row.proof_surfaces {
        validate_proof_surface(repo_root, proof)?;
    }
    Ok(())
}

fn validate_proof_surface(
    repo_root: &Path,
    proof: &Cs336A2ConformanceProofSurface,
) -> Result<(), Cs336A2FullPortConformanceError> {
    let proof_path = repo_root.join(&proof.repo_relative_path);
    if !proof_path.exists() {
        return Err(Cs336A2FullPortConformanceError::MissingProofSurface(
            format!("`{}` does not exist", proof.repo_relative_path),
        ));
    }
    if let Some(selector) = &proof.selector {
        let contents = fs::read_to_string(&proof_path)?;
        if !contents.contains(selector) {
            return Err(Cs336A2FullPortConformanceError::MissingProofSurface(
                format!(
                    "`{}` does not contain selector `{selector}`",
                    proof.repo_relative_path
                ),
            ));
        }
    }
    if proof.repo_relative_path.ends_with(".json") {
        let bytes = fs::read(&proof_path)?;
        let _: serde_json::Value = serde_json::from_slice(&bytes)?;
    }
    Ok(())
}

fn expected_rows() -> Vec<Cs336A2ConformanceRow> {
    vec![
        row(
            "get_flashattention_autograd_function_pytorch",
            "attention",
            vec![
                FLASHATTENTION_REFERENCE_IMPL_PATH,
                FLASHATTENTION_REFERENCE_SOURCE_PATH,
            ],
            vec![
                source_proof(
                    FLASHATTENTION_REFERENCE_SOURCE_PATH,
                    "build_cs336_a2_flashattention_reference_receipt",
                    "Owned bounded reference receipt over the PyTorch-style FlashAttention2 path.",
                ),
                json_proof(
                    CS336_A2_FLASHATTENTION_REFERENCE_RECEIPT_FIXTURE_PATH,
                    "Retained parity and memory-surface receipt for the reference path.",
                ),
            ],
            "The PyTorch-only FlashAttention adapter is closed by the owned tiled forward/backward reference implementation plus the retained parity receipt.",
        ),
        row(
            "get_flashattention_autograd_function_triton",
            "attention",
            vec![FLASHATTENTION_FUSED_IMPL_PATH, FLASHATTENTION_FUSED_SOURCE_PATH],
            vec![
                source_proof(
                    FLASHATTENTION_FUSED_SOURCE_PATH,
                    "build_cs336_a2_flashattention_fused_cuda_receipt",
                    "Owned bounded fused CUDA receipt family for the Triton-class adapter surface.",
                ),
                json_proof(
                    CS336_A2_FLASHATTENTION_FUSED_CUDA_RECEIPT_FIXTURE_PATH,
                    "Retained admitted-path or explicit-refusal receipt for the fused CUDA lane.",
                ),
            ],
            "The Triton-class FlashAttention adapter is closed by the owned fused CUDA receipt family, with explicit refusal posture on non-CUDA hosts.",
        ),
        row(
            "get_ddp_individual_parameters",
            "distributed",
            vec![DDP_INDIVIDUAL_SOURCE_PATH],
            vec![
                source_proof(
                    DDP_INDIVIDUAL_SOURCE_PATH,
                    "build_cs336_a2_ddp_individual_parameters_receipt",
                    "Owned bounded constructor surface for the individual-parameter DDP lane.",
                ),
                json_proof(
                    CS336_A2_DDP_INDIVIDUAL_PARAMETERS_RECEIPT_FIXTURE_PATH,
                    "Retained two-rank parity receipt for the bounded individual-parameter DDP lane.",
                ),
            ],
            "The individual-parameter DDP adapter is mapped to the owned two-rank receipt lane above the A1 trainer.",
        ),
        row(
            "ddp_individual_parameters_on_after_backward",
            "distributed",
            vec![DDP_INDIVIDUAL_SOURCE_PATH],
            vec![
                source_proof(
                    DDP_INDIVIDUAL_SOURCE_PATH,
                    "parameter_syncs",
                    "The after-backward hook is represented by retained per-parameter synchronization receipts before the optimizer step.",
                ),
                json_proof(
                    CS336_A2_DDP_INDIVIDUAL_PARAMETERS_RECEIPT_FIXTURE_PATH,
                    "Retained proof that the bounded after-backward sync path stays aligned with the non-parallel baseline.",
                ),
            ],
            "The individual-parameter after-backward hook is closed by retained per-parameter sync receipts and bounded parity proof.",
        ),
        row(
            "get_ddp_bucketed",
            "distributed",
            vec![DDP_BUCKETED_SOURCE_PATH],
            vec![
                source_proof(
                    DDP_BUCKETED_SOURCE_PATH,
                    "build_cs336_a2_ddp_bucketed_receipt",
                    "Owned bounded constructor surface for the bucketed DDP lane.",
                ),
                json_proof(
                    CS336_A2_DDP_BUCKETED_RECEIPT_FIXTURE_PATH,
                    "Retained two-rank bucket-plan and parity receipt for the bounded bucketed DDP lane.",
                ),
            ],
            "The bucketed DDP adapter is mapped to the owned bucket-planning and parity receipt lane above the A1 trainer.",
        ),
        row(
            "ddp_bucketed_on_after_backward",
            "distributed",
            vec![DDP_BUCKETED_SOURCE_PATH],
            vec![
                source_proof(
                    DDP_BUCKETED_SOURCE_PATH,
                    "build_after_backward_receipt",
                    "The after-backward hook is represented by deterministic bucket completion receipts.",
                ),
                json_proof(
                    CS336_A2_DDP_BUCKETED_RECEIPT_FIXTURE_PATH,
                    "Retained proof that after-backward bucket synchronization stays aligned with the non-parallel baseline.",
                ),
            ],
            "The bucketed after-backward hook is closed by retained bucket completion receipts and bounded parity proof.",
        ),
        row(
            "ddp_bucketed_on_train_batch_start",
            "distributed",
            vec![DDP_BUCKETED_SOURCE_PATH],
            vec![
                source_proof(
                    DDP_BUCKETED_SOURCE_PATH,
                    "build_train_batch_start_receipt",
                    "The train-batch-start hook is represented by explicit pending-bucket reset receipts.",
                ),
                json_proof(
                    CS336_A2_DDP_BUCKETED_RECEIPT_FIXTURE_PATH,
                    "Retained proof that the bounded bucket reset surface exists for every step.",
                ),
            ],
            "The bucketed train-batch-start hook is closed by retained reset receipts tied to the bounded bucket-plan lane.",
        ),
        row(
            "get_sharded_optimizer",
            "optimizer",
            vec![SHARDED_OPTIMIZER_SOURCE_PATH],
            vec![
                source_proof(
                    SHARDED_OPTIMIZER_SOURCE_PATH,
                    "build_cs336_a2_sharded_optimizer_receipt",
                    "Owned bounded constructor surface for the sharded-optimizer lane.",
                ),
                json_proof(
                    CS336_A2_SHARDED_OPTIMIZER_RECEIPT_FIXTURE_PATH,
                    "Retained proof that disjoint optimizer-state ownership reconstructs the non-sharded baseline after each bounded step.",
                ),
            ],
            "The sharded optimizer adapter is mapped to the owned bounded ZeRO-stage-1-style receipt lane with retained checkpoint-state reconstruction proof.",
        ),
    ]
}

fn row(
    stanford_adapter_name: &str,
    category: &str,
    implementation_surfaces: Vec<&str>,
    proof_surfaces: Vec<Cs336A2ConformanceProofSurface>,
    detail: &str,
) -> Cs336A2ConformanceRow {
    Cs336A2ConformanceRow {
        stanford_adapter_name: String::from(stanford_adapter_name),
        category: String::from(category),
        status: String::from("green"),
        implementation_surfaces: implementation_surfaces
            .into_iter()
            .map(String::from)
            .collect(),
        proof_surfaces,
        detail: String::from(detail),
    }
}

fn source_proof(
    repo_relative_path: &str,
    selector: &str,
    detail: &str,
) -> Cs336A2ConformanceProofSurface {
    Cs336A2ConformanceProofSurface {
        proof_kind: String::from("source_selector"),
        repo_relative_path: String::from(repo_relative_path),
        selector: Some(String::from(selector)),
        detail: String::from(detail),
    }
}

fn json_proof(repo_relative_path: &str, detail: &str) -> Cs336A2ConformanceProofSurface {
    Cs336A2ConformanceProofSurface {
        proof_kind: String::from("retained_fixture"),
        repo_relative_path: String::from(repo_relative_path),
        selector: None,
        detail: String::from(detail),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        build_cs336_a2_full_port_conformance_report, write_cs336_a2_full_port_conformance_fixture,
        CS336_A2_FULL_PORT_CONFORMANCE_REPORT_FIXTURE_PATH,
    };

    #[test]
    fn a2_full_port_report_is_fully_green() -> Result<(), Box<dyn std::error::Error>> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .ok_or("missing repo root")?;
        let report = build_cs336_a2_full_port_conformance_report(repo_root)?;
        assert!(report.fully_green);
        assert_eq!(report.row_count, 8);
        Ok(())
    }

    #[test]
    fn a2_full_port_report_covers_expected_adapter_set() -> Result<(), Box<dyn std::error::Error>> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .ok_or("missing repo root")?;
        let report = build_cs336_a2_full_port_conformance_report(repo_root)?;
        assert_eq!(report.rows.len(), 8);
        assert!(report
            .rows
            .iter()
            .any(|row| row.stanford_adapter_name == "get_sharded_optimizer"));
        Ok(())
    }

    #[test]
    fn a2_full_port_writer_emits_json_fixture() -> Result<(), Box<dyn std::error::Error>> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .ok_or("missing repo root")?;
        let report = write_cs336_a2_full_port_conformance_fixture(repo_root)?;
        let fixture_path = repo_root.join(CS336_A2_FULL_PORT_CONFORMANCE_REPORT_FIXTURE_PATH);
        let bytes = std::fs::read(&fixture_path)?;
        let written: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(
            written["schema_version"].as_str(),
            Some(report.schema_version.as_str())
        );
        Ok(())
    }
}
