use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use psionic_data::TassadarSequenceSplit;
use psionic_eval::{
    TassadarSequenceDatasetBundle, TassadarSequenceEvalError, TassadarSequenceWorkload,
    build_tassadar_sequence_dataset_with_trace_family,
};
use psionic_models::{
    TassadarExecutorLongTraceContract, TassadarExecutorTrainableSurface,
    TassadarSequenceTraceFamily, TassadarTraceTokenizer, TokenId,
};
use psionic_runtime::{
    tassadar_hungarian_10x10_corpus, tassadar_hungarian_v0_corpus, tassadar_sudoku_9x9_corpus,
    tassadar_sudoku_v0_corpus,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    TassadarExecutorStructuralSupervisionConfig, TassadarExecutorTeacherForcedTrainingStrategy,
    TassadarSequenceTrainingError, TassadarSequenceTrainingManifest,
    build_tassadar_sequence_training_manifest_from_bundle,
};

const DATASET_MANIFEST_FILE: &str = "dataset_manifest.json";
const TRAINING_MANIFEST_FILE: &str = "training_manifest.json";
const TRACE_FAMILY_DATASET_VERSION: &str = "trace-family-v1";
const CURRENT_LEARNED_MODEL_CONTEXT_LIMIT: u32 = 524_288;

/// Canonical output root for the first sequential-vs-wavefront research comparison.
pub const TASSADAR_TRACE_FAMILY_COMPARISON_OUTPUT_DIR: &str =
    "fixtures/tassadar/runs/tassadar_trace_family_comparison_v1";
/// Canonical machine-readable report for the first sequential-vs-wavefront comparison.
pub const TASSADAR_TRACE_FAMILY_COMPARISON_REPORT_FILE: &str =
    "trace_family_comparison_report.json";

/// One persisted dataset-family summary inside the comparison root.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarTraceFamilyDatasetSummary {
    /// Stable workload label.
    pub workload: String,
    /// Stable symbolic trace family label.
    pub trace_family: TassadarSequenceTraceFamily,
    /// Honest claim boundary for the family artifact.
    pub claim_boundary: String,
    /// Honest reconstruction scope for the family artifact.
    pub reconstruction_scope: String,
    /// Dataset storage key for the family.
    pub dataset_storage_key: String,
    /// Stable dataset digest.
    pub dataset_digest: String,
    /// Stable dataset-manifest digest.
    pub dataset_manifest_digest: String,
    /// Stable training-manifest digest.
    pub training_manifest_digest: String,
    /// Example count across all splits.
    pub example_count: u32,
    /// Training example count.
    pub train_example_count: u32,
    /// Validation example count.
    pub validation_example_count: u32,
    /// Test example count.
    pub test_example_count: u32,
    /// Minimum target token count.
    pub target_token_count_min: u32,
    /// Maximum target token count.
    pub target_token_count_max: u32,
    /// Mean target token count rounded down.
    pub target_token_count_mean: u32,
    /// Maximum total token count across the family.
    pub total_token_count_max: u32,
    /// Maximum source CPU trace step count backing the family.
    pub source_trace_step_count_max: u32,
    /// Current learned model context limit used for fit comparisons.
    pub current_model_context_limit: u32,
    /// Case count that fits the current learned model context.
    pub fits_current_model_context_case_count: u32,
    /// Case count whose reconstructed outputs exactly match the source outputs.
    pub final_output_exact_case_count: u32,
    /// Final-output exactness in basis points.
    pub final_output_exactness_bps: u32,
    /// Whether the family is a full CPU-trace authority rather than a final-output target.
    pub full_cpu_trace_authority: bool,
    /// Relative dataset-manifest path.
    pub dataset_manifest_ref: String,
    /// Relative training-manifest path.
    pub training_manifest_ref: String,
    /// Honest note carried into the report.
    pub note: String,
}

/// Same-corpus sequential-vs-alternate comparison for one workload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarTraceFamilyWorkloadComparison {
    /// Stable workload label.
    pub workload: String,
    /// Sequential CPU-style authority summary.
    pub sequential_cpu_reference: TassadarTraceFamilyDatasetSummary,
    /// Alternate wavefront or frontier family summary.
    pub alternate_trace_family: TassadarTraceFamilyDatasetSummary,
    /// Whether the alternate family reduces the maximum total token count.
    pub alternate_reduces_max_total_tokens: bool,
    /// Whether the alternate family preserves final-output exactness.
    pub alternate_matches_final_output_exactness: bool,
    /// Plain-language workload summary.
    pub summary: String,
}

/// Top-level report for the first sequential-vs-wavefront dataset comparison.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarTraceFamilyComparisonReport {
    /// Shared dataset version used for all persisted manifests.
    pub dataset_version: String,
    /// Current learned model context limit used for fit comparisons.
    pub current_model_context_limit: u32,
    /// Same-corpus comparisons per workload.
    pub workload_comparisons: Vec<TassadarTraceFamilyWorkloadComparison>,
    /// Plain-language top-level summary.
    pub summary: String,
    /// Stable report digest.
    pub report_digest: String,
}

/// Errors while materializing the sequential-vs-wavefront comparison root.
#[derive(Debug, Error)]
pub enum TassadarTraceFamilyComparisonError {
    /// Dataset generation failed.
    #[error(transparent)]
    SequenceEval(#[from] TassadarSequenceEvalError),
    /// Training-manifest generation failed.
    #[error(transparent)]
    SequenceTraining(#[from] TassadarSequenceTrainingError),
    /// Writing one comparison artifact failed.
    #[error("failed to write `{path}`: {error}")]
    Write {
        /// File path.
        path: String,
        /// Source error.
        error: std::io::Error,
    },
    /// One comparison artifact failed to serialize.
    #[error("failed to serialize `{artifact_kind}` for `{path}`: {error}")]
    Serialize {
        /// Artifact kind.
        artifact_kind: String,
        /// File path.
        path: String,
        /// Source error.
        error: serde_json::Error,
    },
}

/// Executes the first honest sequential-vs-wavefront comparison and writes the report.
pub fn execute_tassadar_trace_family_comparison(
    output_dir: &Path,
) -> Result<TassadarTraceFamilyComparisonReport, TassadarTraceFamilyComparisonError> {
    fs::create_dir_all(output_dir).map_err(|error| TassadarTraceFamilyComparisonError::Write {
        path: output_dir.display().to_string(),
        error,
    })?;

    let workload_comparisons = [
        TassadarSequenceWorkload::SudokuV0,
        TassadarSequenceWorkload::Sudoku9x9,
        TassadarSequenceWorkload::HungarianV0,
        TassadarSequenceWorkload::Hungarian10x10,
    ]
    .into_iter()
    .map(|workload| build_workload_comparison(output_dir, workload))
    .collect::<Result<Vec<_>, _>>()?;

    let mut report = TassadarTraceFamilyComparisonReport {
        dataset_version: String::from(TRACE_FAMILY_DATASET_VERSION),
        current_model_context_limit: CURRENT_LEARNED_MODEL_CONTEXT_LIMIT,
        workload_comparisons,
        summary: String::new(),
        report_digest: String::new(),
    };
    let alternate_fit_wins = report
        .workload_comparisons
        .iter()
        .filter(|comparison| {
            comparison
                .alternate_trace_family
                .fits_current_model_context_case_count
                > comparison
                    .sequential_cpu_reference
                    .fits_current_model_context_case_count
        })
        .count();
    report.summary = format!(
        "Sequential-vs-wavefront research comparison is now artifact-backed across {} workloads at context_limit={}: alternate families preserve final-output exactness on all cases and improve fit on {} workload groups while the sequential CPU trace remains the only full-trace authority.",
        report.workload_comparisons.len(),
        report.current_model_context_limit,
        alternate_fit_wins,
    );
    report.report_digest =
        stable_digest(b"psionic_tassadar_trace_family_comparison_report|", &report);
    write_json(
        &output_dir.join(TASSADAR_TRACE_FAMILY_COMPARISON_REPORT_FILE),
        "trace-family comparison report",
        &report,
    )?;
    Ok(report)
}

fn build_workload_comparison(
    output_dir: &Path,
    workload: TassadarSequenceWorkload,
) -> Result<TassadarTraceFamilyWorkloadComparison, TassadarTraceFamilyComparisonError> {
    let alternate_trace_family = match workload {
        TassadarSequenceWorkload::SudokuV0 | TassadarSequenceWorkload::Sudoku9x9 => {
            TassadarSequenceTraceFamily::SudokuDiagonalWavefront
        }
        TassadarSequenceWorkload::HungarianV0 | TassadarSequenceWorkload::Hungarian10x10 => {
            TassadarSequenceTraceFamily::HungarianAssignmentFrontier
        }
    };
    let workload_dir = output_dir.join(workload_directory_label(workload));
    let sequential_dir =
        workload_dir.join(TassadarSequenceTraceFamily::SequentialCpuReference.label());
    let alternate_dir = workload_dir.join(alternate_trace_family.label());

    let sequential_bundle = build_tassadar_sequence_dataset_with_trace_family(
        workload,
        TRACE_FAMILY_DATASET_VERSION,
        TassadarSequenceTraceFamily::SequentialCpuReference,
    )?;
    let alternate_bundle = build_tassadar_sequence_dataset_with_trace_family(
        workload,
        TRACE_FAMILY_DATASET_VERSION,
        alternate_trace_family,
    )?;
    let sequential_manifest = build_trace_family_training_manifest(&sequential_bundle)?;
    let alternate_manifest = build_trace_family_training_manifest(&alternate_bundle)?;

    let sequential_summary = persist_and_summarize_family(
        workload,
        &sequential_bundle,
        &sequential_manifest,
        &sequential_dir,
    )?;
    let alternate_summary = persist_and_summarize_family(
        workload,
        &alternate_bundle,
        &alternate_manifest,
        &alternate_dir,
    )?;

    let alternate_reduces_max_total_tokens =
        alternate_summary.total_token_count_max < sequential_summary.total_token_count_max;
    let alternate_matches_final_output_exactness = alternate_summary.final_output_exact_case_count
        == alternate_summary.example_count
        && alternate_summary.final_output_exact_case_count
            == sequential_summary.final_output_exact_case_count;
    let summary = format!(
        "{}: sequential_max_total_tokens={}, alternate_max_total_tokens={}, sequential_fit_cases={}, alternate_fit_cases={}, sequential_output_exact_bps={}, alternate_output_exact_bps={}.",
        workload_directory_label(workload),
        sequential_summary.total_token_count_max,
        alternate_summary.total_token_count_max,
        sequential_summary.fits_current_model_context_case_count,
        alternate_summary.fits_current_model_context_case_count,
        sequential_summary.final_output_exactness_bps,
        alternate_summary.final_output_exactness_bps,
    );

    Ok(TassadarTraceFamilyWorkloadComparison {
        workload: workload.dataset_ref().to_string(),
        sequential_cpu_reference: sequential_summary,
        alternate_trace_family: alternate_summary,
        alternate_reduces_max_total_tokens,
        alternate_matches_final_output_exactness,
        summary,
    })
}

fn build_trace_family_training_manifest(
    bundle: &TassadarSequenceDatasetBundle,
) -> Result<TassadarSequenceTrainingManifest, TassadarTraceFamilyComparisonError> {
    Ok(build_tassadar_sequence_training_manifest_from_bundle(
        bundle,
        TassadarExecutorTrainableSurface::OutputHeadOnly,
        TassadarExecutorTeacherForcedTrainingStrategy::FullForwardWindow,
        TassadarExecutorLongTraceContract::FlatPrefixFullForward,
        TassadarExecutorStructuralSupervisionConfig::next_token_only(),
    )?)
}

fn persist_and_summarize_family(
    workload: TassadarSequenceWorkload,
    bundle: &TassadarSequenceDatasetBundle,
    training_manifest: &TassadarSequenceTrainingManifest,
    family_dir: &Path,
) -> Result<TassadarTraceFamilyDatasetSummary, TassadarTraceFamilyComparisonError> {
    fs::create_dir_all(family_dir).map_err(|error| TassadarTraceFamilyComparisonError::Write {
        path: family_dir.display().to_string(),
        error,
    })?;
    let dataset_manifest_path = family_dir.join(DATASET_MANIFEST_FILE);
    let training_manifest_path = family_dir.join(TRAINING_MANIFEST_FILE);
    write_json(
        &dataset_manifest_path,
        "dataset manifest",
        &bundle.dataset.manifest,
    )?;
    write_json(
        &training_manifest_path,
        "training manifest",
        training_manifest,
    )?;

    summarize_family(
        workload,
        bundle,
        training_manifest,
        dataset_manifest_path
            .strip_prefix(repo_root())
            .unwrap_or(&dataset_manifest_path),
        training_manifest_path
            .strip_prefix(repo_root())
            .unwrap_or(&training_manifest_path),
    )
}

fn summarize_family(
    workload: TassadarSequenceWorkload,
    bundle: &TassadarSequenceDatasetBundle,
    training_manifest: &TassadarSequenceTrainingManifest,
    dataset_manifest_ref: &Path,
    training_manifest_ref: &Path,
) -> Result<TassadarTraceFamilyDatasetSummary, TassadarTraceFamilyComparisonError> {
    let tokenizer = TassadarTraceTokenizer::new();
    let expected_outputs = expected_outputs_by_case_id(workload);
    let example_count = bundle.dataset.examples.len() as u32;
    let target_counts = bundle
        .dataset
        .examples
        .iter()
        .map(|example| example.metadata.target_token_count)
        .collect::<Vec<_>>();
    let total_counts = bundle
        .dataset
        .examples
        .iter()
        .map(|example| example.metadata.total_token_count)
        .collect::<Vec<_>>();
    let source_trace_step_count_max = bundle
        .dataset
        .examples
        .iter()
        .map(|example| example.metadata.trace_step_count)
        .max()
        .unwrap_or(0);
    let fits_current_model_context_case_count = total_counts
        .iter()
        .filter(|count| **count <= CURRENT_LEARNED_MODEL_CONTEXT_LIMIT)
        .count() as u32;
    let final_output_exact_case_count = bundle
        .dataset
        .examples
        .iter()
        .filter(|example| {
            let Some(expected) = expected_outputs.get(example.metadata.case_id.as_str()) else {
                return false;
            };
            let tokens = example
                .token_ids
                .iter()
                .map(|token| TokenId(*token))
                .collect::<Vec<_>>();
            reconstruct_outputs(
                &tokenizer,
                bundle.trace_family,
                workload,
                tokens.as_slice(),
                example.metadata.prompt_token_count as usize,
            ) == *expected
        })
        .count() as u32;
    let final_output_exactness_bps = if example_count == 0 {
        0
    } else {
        final_output_exact_case_count.saturating_mul(10_000) / example_count
    };
    let full_cpu_trace_authority =
        bundle.trace_family == TassadarSequenceTraceFamily::SequentialCpuReference;
    let note = match bundle.trace_family {
        TassadarSequenceTraceFamily::SequentialCpuReference => String::from(
            "full CPU-trace authority remains sequential; this is the only trace family that preserves the reference execution structure step by step",
        ),
        TassadarSequenceTraceFamily::SudokuDiagonalWavefront => String::from(
            "research-only anti-diagonal Sudoku output target; it preserves final solved outputs exactly but is not a CPU-trace authority",
        ),
        TassadarSequenceTraceFamily::HungarianAssignmentFrontier => String::from(
            "research-only parallel Hungarian assignment frontier; it preserves final assignment-plus-cost outputs exactly but is not a CPU-trace authority",
        ),
    };

    Ok(TassadarTraceFamilyDatasetSummary {
        workload: workload.dataset_ref().to_string(),
        trace_family: bundle.trace_family,
        claim_boundary: trace_family_claim_boundary(workload, bundle.trace_family).to_string(),
        reconstruction_scope: bundle.trace_family.reconstruction_scope().to_string(),
        dataset_storage_key: bundle.dataset.storage_key(),
        dataset_digest: bundle.dataset.stable_digest(),
        dataset_manifest_digest: stable_digest(
            b"psionic_tassadar_trace_family_dataset_manifest|",
            &bundle.dataset.manifest,
        ),
        training_manifest_digest: training_manifest.manifest_digest.clone(),
        example_count,
        train_example_count: bundle
            .dataset
            .split_examples(TassadarSequenceSplit::Train)
            .len() as u32,
        validation_example_count: bundle
            .dataset
            .split_examples(TassadarSequenceSplit::Validation)
            .len() as u32,
        test_example_count: bundle
            .dataset
            .split_examples(TassadarSequenceSplit::Test)
            .len() as u32,
        target_token_count_min: target_counts.iter().copied().min().unwrap_or(0),
        target_token_count_max: target_counts.iter().copied().max().unwrap_or(0),
        target_token_count_mean: if target_counts.is_empty() {
            0
        } else {
            target_counts.iter().copied().map(u64::from).sum::<u64>() as u32
                / target_counts.len() as u32
        },
        total_token_count_max: total_counts.iter().copied().max().unwrap_or(0),
        source_trace_step_count_max,
        current_model_context_limit: CURRENT_LEARNED_MODEL_CONTEXT_LIMIT,
        fits_current_model_context_case_count,
        final_output_exact_case_count,
        final_output_exactness_bps,
        full_cpu_trace_authority,
        dataset_manifest_ref: dataset_manifest_ref.display().to_string(),
        training_manifest_ref: training_manifest_ref.display().to_string(),
        note,
    })
}

fn reconstruct_outputs(
    tokenizer: &TassadarTraceTokenizer,
    trace_family: TassadarSequenceTraceFamily,
    workload: TassadarSequenceWorkload,
    tokens: &[TokenId],
    prompt_token_count: usize,
) -> Vec<i32> {
    match trace_family {
        TassadarSequenceTraceFamily::SequentialCpuReference => {
            tokenizer.extract_output_values(tokens)
        }
        TassadarSequenceTraceFamily::SudokuDiagonalWavefront => tokenizer
            .extract_sudoku_diagonal_wavefront_outputs(
                tokens,
                prompt_token_count,
                sudoku_cell_count(workload),
            ),
        TassadarSequenceTraceFamily::HungarianAssignmentFrontier => tokenizer
            .extract_hungarian_assignment_frontier_outputs(
                tokens,
                prompt_token_count,
                hungarian_dimension(workload),
            ),
    }
}

fn expected_outputs_by_case_id(workload: TassadarSequenceWorkload) -> BTreeMap<String, Vec<i32>> {
    match workload {
        TassadarSequenceWorkload::SudokuV0 => tassadar_sudoku_v0_corpus()
            .into_iter()
            .map(|case| (case.case_id, case.validation_case.expected_outputs))
            .collect(),
        TassadarSequenceWorkload::Sudoku9x9 => tassadar_sudoku_9x9_corpus()
            .into_iter()
            .map(|case| (case.case_id, case.validation_case.expected_outputs))
            .collect(),
        TassadarSequenceWorkload::HungarianV0 => tassadar_hungarian_v0_corpus()
            .into_iter()
            .map(|case| (case.case_id, case.validation_case.expected_outputs))
            .collect(),
        TassadarSequenceWorkload::Hungarian10x10 => tassadar_hungarian_10x10_corpus()
            .into_iter()
            .map(|case| (case.case_id, case.validation_case.expected_outputs))
            .collect(),
    }
}

fn trace_family_claim_boundary(
    workload: TassadarSequenceWorkload,
    trace_family: TassadarSequenceTraceFamily,
) -> &'static str {
    match (workload, trace_family) {
        (
            TassadarSequenceWorkload::SudokuV0 | TassadarSequenceWorkload::Sudoku9x9,
            TassadarSequenceTraceFamily::SequentialCpuReference,
        ) => "learned_bounded",
        _ => "research_only",
    }
}

fn workload_directory_label(workload: TassadarSequenceWorkload) -> &'static str {
    match workload {
        TassadarSequenceWorkload::SudokuV0 => "sudoku_v0",
        TassadarSequenceWorkload::Sudoku9x9 => "sudoku_9x9",
        TassadarSequenceWorkload::HungarianV0 => "hungarian_v0",
        TassadarSequenceWorkload::Hungarian10x10 => "hungarian_10x10",
    }
}

fn sudoku_cell_count(workload: TassadarSequenceWorkload) -> usize {
    match workload {
        TassadarSequenceWorkload::SudokuV0 => 16,
        TassadarSequenceWorkload::Sudoku9x9 => 81,
        TassadarSequenceWorkload::HungarianV0 | TassadarSequenceWorkload::Hungarian10x10 => 0,
    }
}

fn hungarian_dimension(workload: TassadarSequenceWorkload) -> usize {
    match workload {
        TassadarSequenceWorkload::HungarianV0 => 4,
        TassadarSequenceWorkload::Hungarian10x10 => 10,
        TassadarSequenceWorkload::SudokuV0 | TassadarSequenceWorkload::Sudoku9x9 => 0,
    }
}

fn write_json(
    path: &Path,
    artifact_kind: &str,
    value: &impl Serialize,
) -> Result<(), TassadarTraceFamilyComparisonError> {
    let encoded = serde_json::to_vec_pretty(value).map_err(|error| {
        TassadarTraceFamilyComparisonError::Serialize {
            artifact_kind: artifact_kind.to_string(),
            path: path.display().to_string(),
            error,
        }
    })?;
    fs::write(path, encoded).map_err(|error| TassadarTraceFamilyComparisonError::Write {
        path: path.display().to_string(),
        error,
    })
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root should exist")
        .to_path_buf()
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded =
        serde_json::to_vec(value).expect("Tassadar trace-family comparison value should serialize");
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        TASSADAR_TRACE_FAMILY_COMPARISON_OUTPUT_DIR, execute_tassadar_trace_family_comparison,
    };

    #[test]
    fn trace_family_comparison_preserves_output_exactness() -> Result<(), Box<dyn std::error::Error>>
    {
        let temp_dir = tempfile::tempdir()?;
        let report = execute_tassadar_trace_family_comparison(temp_dir.path())?;

        assert_eq!(report.workload_comparisons.len(), 4);
        assert!(
            report
                .workload_comparisons
                .iter()
                .all(|comparison| comparison.alternate_matches_final_output_exactness)
        );
        assert!(TASSADAR_TRACE_FAMILY_COMPARISON_OUTPUT_DIR.contains("trace_family_comparison"));
        Ok(())
    }
}
