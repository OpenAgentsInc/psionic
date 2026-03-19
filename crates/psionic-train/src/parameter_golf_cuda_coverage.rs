use psionic_core::{
    builtin_quantization_capability_semantics_report, PsionicRefusal, PsionicRefusalCode,
    PsionicRefusalScope,
};
use psionic_ir::{builtin_advanced_operator_program_matrix_report, GraphError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::PARAMETER_GOLF_DISTRIBUTED_8XH100_VERSION;

/// One Parameter Golf CUDA training requirement family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfCudaTrainingFamily {
    /// BF16 or FP32 precision-policy requirements for the public baseline lane.
    Precision,
    /// RoPE plus GQA attention requirements for the decoder core.
    Attention,
    /// RMSNorm execution requirements for the decoder core.
    RmsNorm,
    /// Residual addition and residual-mix requirements for the decoder core.
    Residual,
    /// Optimizer requirements such as Muon on matrix parameters.
    Optimizer,
    /// Post-train export or quantization requirements.
    Quantization,
}

/// Status vocabulary for one Parameter Golf CUDA training requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfCudaTrainingCoverageStatus {
    /// The current repo has an early but real answer for this requirement.
    ImplementedEarly,
    /// The current repo has some substrate or semantics, but not full direct closure.
    Partial,
    /// The current repo still lacks a credible implementation answer.
    Planned,
}

/// One machine-readable CUDA training coverage case for Parameter Golf.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfCudaTrainingCoverageCase {
    /// Stable case identifier.
    pub case_id: String,
    /// Requirement family covered by the case.
    pub family: ParameterGolfCudaTrainingFamily,
    /// Current status for this requirement.
    pub status: ParameterGolfCudaTrainingCoverageStatus,
    /// Stable statement of what the public Parameter Golf baseline requires.
    pub required_scope: String,
    /// Stable statement of what the repo currently owns.
    pub current_surface: String,
    /// Honest boundary note when the public CUDA surface remains narrower.
    pub boundary_note: String,
}

/// Aggregate machine-readable CUDA training coverage report for Parameter Golf.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfCudaTrainingCapabilityReport {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable scope window for the report.
    pub scope_window: String,
    /// Distributed-lane version this report is aligned to.
    pub distributed_lane_version: String,
    /// Stable digest for the advanced-operator matrix reused as evidence.
    pub advanced_operator_matrix_digest: String,
    /// Stable digest for the quantization semantics report reused as evidence.
    pub quantization_semantics_digest: String,
    /// Ordered coverage cases.
    pub cases: Vec<ParameterGolfCudaTrainingCoverageCase>,
    /// Stable blocker case identifiers for full challenge-speed CUDA closure.
    pub blocking_case_ids: Vec<String>,
    /// Stable digest over the report contents.
    pub report_digest: String,
}

impl ParameterGolfCudaTrainingCapabilityReport {
    fn new(
        distributed_lane_version: impl Into<String>,
        advanced_operator_matrix_digest: impl Into<String>,
        quantization_semantics_digest: impl Into<String>,
        cases: Vec<ParameterGolfCudaTrainingCoverageCase>,
    ) -> Self {
        let distributed_lane_version = distributed_lane_version.into();
        let advanced_operator_matrix_digest = advanced_operator_matrix_digest.into();
        let quantization_semantics_digest = quantization_semantics_digest.into();
        let blocking_case_ids = cases
            .iter()
            .filter(|case| case.status != ParameterGolfCudaTrainingCoverageStatus::ImplementedEarly)
            .map(|case| case.case_id.clone())
            .collect::<Vec<_>>();
        let report_digest = stable_digest(
            b"psionic_parameter_golf_cuda_training_capability_report|",
            &(
                &distributed_lane_version,
                &advanced_operator_matrix_digest,
                &quantization_semantics_digest,
                &cases,
                &blocking_case_ids,
            ),
        );
        Self {
            schema_version: 1,
            scope_window: String::from("parameter_golf_cuda_training_v1"),
            distributed_lane_version,
            advanced_operator_matrix_digest,
            quantization_semantics_digest,
            cases,
            blocking_case_ids,
            report_digest,
        }
    }

    /// Returns the current blocker case identifiers.
    #[must_use]
    pub fn challenge_kernel_blockers(&self) -> &[String] {
        self.blocking_case_ids.as_slice()
    }

    /// Returns honest boundary-note lines for the remaining blockers.
    #[must_use]
    pub fn boundary_notes(&self) -> Vec<String> {
        self.cases
            .iter()
            .filter(|case| case.status != ParameterGolfCudaTrainingCoverageStatus::ImplementedEarly)
            .map(|case| format!("{}: {}", case.case_id, case.boundary_note))
            .collect()
    }

    /// Returns the canonical refusal for full challenge-speed CUDA closure.
    #[must_use]
    pub fn challenge_readiness_refusal(&self) -> Option<PsionicRefusal> {
        if self.blocking_case_ids.is_empty() {
            return None;
        }
        Some(
            PsionicRefusal::new(
                PsionicRefusalCode::UnsupportedBackendCapability,
                PsionicRefusalScope::Runtime,
                format!(
                    "parameter golf CUDA challenge closure still has explicit blockers: {}",
                    self.blocking_case_ids.join(", ")
                ),
            )
            .with_subject(String::from("parameter_golf_cuda_training")),
        )
    }
}

/// Builds the canonical Parameter Golf CUDA training coverage report.
pub fn builtin_parameter_golf_cuda_training_capability_report(
) -> Result<ParameterGolfCudaTrainingCapabilityReport, GraphError> {
    let advanced_operator_report = builtin_advanced_operator_program_matrix_report()?;
    let quantization_report = builtin_quantization_capability_semantics_report();
    Ok(ParameterGolfCudaTrainingCapabilityReport::new(
        PARAMETER_GOLF_DISTRIBUTED_8XH100_VERSION,
        advanced_operator_report.matrix_digest,
        quantization_report.report_digest,
        vec![
            ParameterGolfCudaTrainingCoverageCase {
                case_id: String::from("cuda_bf16_train_precision_contract"),
                family: ParameterGolfCudaTrainingFamily::Precision,
                status: ParameterGolfCudaTrainingCoverageStatus::Partial,
                required_scope: String::from(
                    "the public 8xH100 baseline requires BF16 train-visible parameters and gradients with FP32 optimizer or master-weight posture",
                ),
                current_surface: String::from(
                    "the distributed 8xH100 receipt lane now encodes an explicit BF16-forward or FP32-master precision policy, but the public CUDA array surface still advertises only bounded dense f32 execution instead of broad BF16 train-time closure",
                ),
                boundary_note: String::from(
                    "Do not treat the BF16 policy contract as proof that the public CUDA array surface already owns broad BF16 tensor, backward, and optimizer execution.",
                ),
            },
            ParameterGolfCudaTrainingCoverageCase {
                case_id: String::from("cuda_rope_gqa_attention_block"),
                family: ParameterGolfCudaTrainingFamily::Attention,
                status: ParameterGolfCudaTrainingCoverageStatus::Partial,
                required_scope: String::from(
                    "the compact decoder baseline requires train-time RoPE plus grouped-query attention over the public 9x512 family",
                ),
                current_surface: String::from(
                    "psionic-ir now proves one attention_rotary_residual_block program and an explicit backend-capability refusal when the required rotary or attention kernels are missing, but this remains an IR or meta-program contract rather than a widened public CUDA training kernel lane",
                ),
                boundary_note: String::from(
                    "IR-level attention or RoPE coverage is now explicit, but the public CUDA train path still lacks a declared direct decoder-block execution surface.",
                ),
            },
            ParameterGolfCudaTrainingCoverageCase {
                case_id: String::from("cuda_rms_norm_train_path"),
                family: ParameterGolfCudaTrainingFamily::RmsNorm,
                status: ParameterGolfCudaTrainingCoverageStatus::Partial,
                required_scope: String::from(
                    "the compact decoder baseline requires RMSNorm in the train-time forward or backward path",
                ),
                current_surface: String::from(
                    "the public CUDA execution backend now declares and executes backend-specialized dense f32 RMSNorm forward plans, and psionic-ir now owns bounded dense f32 reference evaluation plus reverse-mode support for RMSNorm train-visible graphs, but the public CUDA backward execution contract is still missing on the train path",
                ),
                boundary_note: String::from(
                    "RMSNorm train-visible graph semantics are now real, but public CUDA backward execution is still missing on the train path.",
                ),
            },
            ParameterGolfCudaTrainingCoverageCase {
                case_id: String::from("cuda_residual_mix_train_path"),
                family: ParameterGolfCudaTrainingFamily::Residual,
                status: ParameterGolfCudaTrainingCoverageStatus::Partial,
                required_scope: String::from(
                    "the compact decoder baseline requires residual addition plus learned residual-mix control tensors across the train-time path",
                ),
                current_surface: String::from(
                    "simple add is inside the public builtin operator matrix, and the public CUDA execution backend now also executes dense f32 pointwise mul for residual-control tensors, but the coupled decoder path still rides on CPU reference and IR evidence rather than a widened direct CUDA train path",
                ),
                boundary_note: String::from(
                    "Treat residual support as forward operator evidence only until the fused decoder path and backward lane are widened on CUDA.",
                ),
            },
            ParameterGolfCudaTrainingCoverageCase {
                case_id: String::from("cuda_muon_optimizer_path"),
                family: ParameterGolfCudaTrainingFamily::Optimizer,
                status: ParameterGolfCudaTrainingCoverageStatus::Partial,
                required_scope: String::from(
                    "the public baseline uses Muon on matrix-shaped transformer parameters under distributed all-reduce",
                ),
                current_surface: String::from(
                    "psionic-train now owns exact CPU Muon reference-step parity and the distributed matrix-update all-reduce contract, but there is still no public CUDA Muon optimizer kernel or runtime path",
                ),
                boundary_note: String::from(
                    "Muon semantics are explicit and distributed communication is explicit, but the train-time CUDA optimizer surface is still narrower than the lane contract.",
                ),
            },
            ParameterGolfCudaTrainingCoverageCase {
                case_id: String::from("cuda_quantized_export_roundtrip"),
                family: ParameterGolfCudaTrainingFamily::Quantization,
                status: ParameterGolfCudaTrainingCoverageStatus::ImplementedEarly,
                required_scope: String::from(
                    "the public baseline reports post-train int8 plus zlib artifact size and roundtrip validation",
                ),
                current_surface: String::from(
                    "psionic-train now owns raw safetensors plus int8_zlib export or restore for Parameter Golf, and psionic-core now owns a bounded quantization capability semantics report above raw decode",
                ),
                boundary_note: String::from(
                    "This closes the post-train artifact quantization lane only; it does not imply broader train-time QAT or low-precision optimizer closure.",
                ),
            },
        ],
    ))
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded = match serde_json::to_vec(value) {
        Ok(encoded) => encoded,
        Err(error) => error.to_string().into_bytes(),
    };
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use psionic_core::{builtin_quantization_capability_semantics_report, PsionicRefusalCode};
    use psionic_ir::builtin_advanced_operator_program_matrix_report;

    use crate::{
        builtin_parameter_golf_cuda_training_capability_report,
        ParameterGolfCudaTrainingCoverageStatus,
    };

    #[test]
    fn parameter_golf_cuda_training_capability_report_tracks_required_families(
    ) -> Result<(), Box<dyn Error>> {
        let report = builtin_parameter_golf_cuda_training_capability_report()?;
        assert_eq!(report.schema_version, 1);
        assert_eq!(report.cases.len(), 6);
        assert_eq!(
            report.advanced_operator_matrix_digest,
            builtin_advanced_operator_program_matrix_report()?.matrix_digest
        );
        assert_eq!(
            report.quantization_semantics_digest,
            builtin_quantization_capability_semantics_report().report_digest
        );
        assert_eq!(
            report.cases.last().expect("quantization case").status,
            ParameterGolfCudaTrainingCoverageStatus::ImplementedEarly
        );
        assert_eq!(report.blocking_case_ids.len(), 5);
        Ok(())
    }

    #[test]
    fn parameter_golf_cuda_training_capability_report_refuses_full_challenge_closure(
    ) -> Result<(), Box<dyn Error>> {
        let report = builtin_parameter_golf_cuda_training_capability_report()?;
        let refusal = report
            .challenge_readiness_refusal()
            .expect("current report should stay blocked");
        assert_eq!(
            refusal.code,
            PsionicRefusalCode::UnsupportedBackendCapability
        );
        assert!(refusal
            .detail
            .contains("cuda_bf16_train_precision_contract"));
        Ok(())
    }
}
