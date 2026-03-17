use std::{collections::BTreeMap, time::Instant};

use psionic_data::DatasetKey;
use psionic_environments::{
    EnvironmentDatasetBinding, EnvironmentPolicyKind, EnvironmentPolicyReference,
    TassadarEnvironmentBundle, TassadarEnvironmentError, TassadarEnvironmentPackageRefs,
    TassadarEnvironmentSpec, TassadarExactnessContract, TassadarIoContract, TassadarProgramBinding,
    TassadarWorkloadTarget,
};
use psionic_models::{TassadarExecutorContractError, TassadarExecutorFixture};
use psionic_runtime::{
    build_tassadar_execution_evidence_bundle, run_tassadar_exact_equivalence,
    tassadar_article_class_corpus, tassadar_hungarian_v0_corpus, tassadar_sudoku_v0_corpus,
    tassadar_validation_corpus, TassadarCpuReferenceRunner, TassadarExecutionRefusal,
    TassadarExecutorDecodeMode, TassadarExecutorSelectionReason, TassadarExecutorSelectionState,
    TassadarFixtureRunner, TassadarHullCacheRunner, TassadarHungarianV0CorpusCase,
    TassadarProgramArtifact, TassadarProgramArtifactError, TassadarSparseTopKRunner,
    TassadarSudokuV0CorpusCase, TassadarSudokuV0CorpusSplit, TassadarTraceAbi, TassadarWasmProfile,
    TASSADAR_ARTICLE_CLASS_BENCHMARK_REPORT_REF,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Digest;
use thiserror::Error;

use crate::{
    BenchmarkAggregationKind, BenchmarkCase, BenchmarkExecutionMode, BenchmarkPackage,
    BenchmarkPackageKey, BenchmarkVerificationPolicy, EvalArtifact, EvalExecutionStrategyFacts,
    EvalFinalStateCapture, EvalMetric, EvalRunContract, EvalRunMode, EvalRunState,
    EvalRuntimeError, EvalSampleRecord, EvalSampleStatus, EvalTimerIntegrityFacts,
    EvalVerificationFacts,
};

/// Stable environment ref for the Tassadar eval package.
pub const TASSADAR_EVAL_ENVIRONMENT_REF: &str = "env.openagents.tassadar.eval";
/// Stable environment ref for the Tassadar benchmark package.
pub const TASSADAR_BENCHMARK_ENVIRONMENT_REF: &str = "env.openagents.tassadar.benchmark";
/// Stable benchmark ref for the Tassadar validation-corpus suite.
pub const TASSADAR_REFERENCE_FIXTURE_BENCHMARK_REF: &str =
    "benchmark://openagents/tassadar/reference_fixture/validation_corpus";
/// Stable dataset ref for the current validation corpus.
pub const TASSADAR_VALIDATION_CORPUS_DATASET_REF: &str =
    "dataset://openagents/tassadar/validation_corpus";
/// Stable environment ref for the widened article-class eval package.
pub const TASSADAR_ARTICLE_CLASS_EVAL_ENVIRONMENT_REF: &str =
    "env.openagents.tassadar.article_class.eval";
/// Stable environment ref for the widened article-class benchmark package.
pub const TASSADAR_ARTICLE_CLASS_BENCHMARK_ENVIRONMENT_REF: &str =
    "env.openagents.tassadar.article_class.benchmark";
/// Stable benchmark ref for the widened article-class suite.
pub const TASSADAR_ARTICLE_CLASS_BENCHMARK_REF: &str =
    "benchmark://openagents/tassadar/article_class/reference_fixture";
/// Stable dataset ref for the widened article-class corpus.
pub const TASSADAR_ARTICLE_CLASS_DATASET_REF: &str =
    "dataset://openagents/tassadar/article_class_corpus";
/// Stable environment ref for the bounded Hungarian-v0 eval package.
pub const TASSADAR_HUNGARIAN_V0_EVAL_ENVIRONMENT_REF: &str =
    "env.openagents.tassadar.hungarian_v0.eval";
/// Stable environment ref for the bounded Hungarian-v0 benchmark package.
pub const TASSADAR_HUNGARIAN_V0_BENCHMARK_ENVIRONMENT_REF: &str =
    "env.openagents.tassadar.hungarian_v0.benchmark";
/// Stable benchmark ref for the bounded Hungarian-v0 suite.
pub const TASSADAR_HUNGARIAN_V0_BENCHMARK_REF: &str =
    "benchmark://openagents/tassadar/hungarian_v0/reference_fixture";
/// Stable dataset ref for the bounded Hungarian-v0 corpus.
pub const TASSADAR_HUNGARIAN_V0_DATASET_REF: &str =
    "dataset://openagents/tassadar/hungarian_v0_corpus";
/// Stable metric id for the Phase 5 hull-cache lane.
pub const TASSADAR_HULL_CACHE_METRIC_ID: &str = "tassadar.hull_cache_steps_per_second";
/// Stable metric id for the Phase 8 sparse-top-k lane.
pub const TASSADAR_SPARSE_TOP_K_METRIC_ID: &str = "tassadar.sparse_top_k_steps_per_second";
/// Canonical machine-readable output path for the workload capability matrix report.
pub const TASSADAR_WORKLOAD_CAPABILITY_MATRIX_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_workload_capability_matrix.json";
/// Canonical machine-readable output path for the widened HullCache closure report.
pub const TASSADAR_HULL_CACHE_CLOSURE_REPORT_REF: &str =
    "fixtures/tassadar/reports/tassadar_hull_cache_closure_report.json";

const TASSADAR_OUTPUT_EXACTNESS_METRIC_ID: &str = "tassadar.final_output_exactness_bps";
const TASSADAR_STEP_EXACTNESS_METRIC_ID: &str = "tassadar.step_exactness_bps";
const TASSADAR_HALT_EXACTNESS_METRIC_ID: &str = "tassadar.halt_exactness_bps";
const TASSADAR_CPU_BASELINE_METRIC_ID: &str = "tassadar.cpu_reference_steps_per_second";
const TASSADAR_REFERENCE_LINEAR_METRIC_ID: &str = "tassadar.reference_linear_steps_per_second";
const TASSADAR_TRACE_DIGEST_EQUAL_METRIC_ID: &str = "tassadar.trace_digest_equal_bps";
const TASSADAR_HULL_CACHE_SPEEDUP_METRIC_ID: &str =
    "tassadar.hull_cache_speedup_over_reference_linear";
const TASSADAR_HULL_CACHE_CPU_GAP_METRIC_ID: &str =
    "tassadar.hull_cache_remaining_gap_vs_cpu_reference";
const TASSADAR_SPARSE_TOP_K_SPEEDUP_METRIC_ID: &str =
    "tassadar.sparse_top_k_speedup_over_reference_linear";
const TASSADAR_SPARSE_TOP_K_CPU_GAP_METRIC_ID: &str =
    "tassadar.sparse_top_k_remaining_gap_vs_cpu_reference";
const TASSADAR_TRACE_STEP_COUNT_METRIC_ID: &str = "tassadar.trace_step_count";
const TASSADAR_WORKLOAD_CAPABILITY_MATRIX_SCHEMA_VERSION: u16 = 1;
const TASSADAR_HULL_CACHE_CLOSURE_SCHEMA_VERSION: u16 = 1;

/// One packaged Tassadar Phase 3 suite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarReferenceFixtureSuite {
    /// Environment bundle shared by eval and benchmark execution.
    pub environment_bundle: TassadarEnvironmentBundle,
    /// Packaged benchmark contract for the current corpus.
    pub benchmark_package: BenchmarkPackage,
    /// Digest-bound artifacts for the benchmark corpus.
    pub artifacts: Vec<TassadarProgramArtifact>,
    /// Stable digest over the ordered artifact set.
    pub corpus_digest: String,
}

/// Per-case benchmark result for the current Tassadar suite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarBenchmarkCaseReport {
    /// Stable benchmark case id.
    pub case_id: String,
    /// Workload target for the case.
    pub workload_target: TassadarWorkloadTarget,
    /// Terminal sample status.
    pub status: EvalSampleStatus,
    /// Aggregate score in basis points.
    pub score_bps: u32,
    /// Final-output exactness score.
    pub final_output_exactness_bps: u32,
    /// Step exactness score.
    pub step_exactness_bps: u32,
    /// Halt exactness score.
    pub halt_exactness_bps: u32,
    /// Decode mode requested by the benchmark harness.
    pub requested_decode_mode: TassadarExecutorDecodeMode,
    /// Effective decode mode after runtime direct/fallback selection.
    pub effective_decode_mode: TassadarExecutorDecodeMode,
    /// Direct/fallback/refused state emitted before execution.
    pub selection_state: TassadarExecutorSelectionState,
    /// Stable reason for fallback or refusal when one existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_reason: Option<TassadarExecutorSelectionReason>,
    /// Whether execution used an explicit decode fallback.
    pub used_decode_fallback: bool,
    /// Whether trace digests matched across CPU, linear, hull-cache, and sparse-top-k paths.
    pub trace_digest_equal: bool,
    /// Whether outputs matched across CPU, linear, hull-cache, and sparse-top-k paths.
    pub outputs_equal: bool,
    /// Whether halt reasons matched across CPU, linear, hull-cache, and sparse-top-k paths.
    pub halt_equal: bool,
    /// Observed trace-step count.
    pub trace_steps: u64,
    /// Direct CPU baseline throughput.
    pub cpu_reference_steps_per_second: f64,
    /// Reference-linear executor throughput.
    pub reference_linear_steps_per_second: f64,
    /// Hull-cache executor throughput.
    pub hull_cache_steps_per_second: f64,
    /// Speedup ratio of hull-cache over reference-linear execution.
    pub hull_cache_speedup_over_reference_linear: f64,
    /// Remaining CPU-reference gap ratio, computed as `cpu / hull`.
    pub hull_cache_remaining_gap_vs_cpu_reference: f64,
    /// Direct/fallback/refused state emitted for the sparse-top-k path.
    pub sparse_top_k_selection_state: TassadarExecutorSelectionState,
    /// Stable reason for sparse-top-k fallback or refusal when one existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_top_k_selection_reason: Option<TassadarExecutorSelectionReason>,
    /// Whether sparse-top-k execution used an explicit decode fallback.
    pub sparse_top_k_used_decode_fallback: bool,
    /// Sparse-top-k executor throughput.
    pub sparse_top_k_steps_per_second: f64,
    /// Speedup ratio of sparse-top-k over reference-linear execution.
    pub sparse_top_k_speedup_over_reference_linear: f64,
    /// Remaining CPU-reference gap ratio, computed as `cpu / sparse_top_k`.
    pub sparse_top_k_remaining_gap_vs_cpu_reference: f64,
    /// Runner-independent CPU behavior digest.
    pub cpu_behavior_digest: String,
    /// Runner-independent reference-linear behavior digest.
    pub reference_linear_behavior_digest: String,
    /// Runner-independent hull-cache behavior digest.
    pub hull_cache_behavior_digest: String,
    /// Runner-independent sparse-top-k behavior digest.
    pub sparse_top_k_behavior_digest: String,
}

/// Full report for one package-driven Tassadar benchmark run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarBenchmarkReport {
    /// Packaged suite identity.
    pub suite: TassadarReferenceFixtureSuite,
    /// Finalized benchmark-mode eval run.
    pub eval_run: EvalRunState,
    /// Aggregate benchmark summary.
    pub aggregate_summary: crate::BenchmarkAggregateSummary,
    /// Per-case benchmark reports.
    pub case_reports: Vec<TassadarBenchmarkCaseReport>,
}

/// Capability posture for one workload family on one Tassadar surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarCapabilityPosture {
    /// Surface is exact on the declared workload family.
    Exact,
    /// Surface remains exact only by explicitly falling back to another mode.
    FallbackOnly,
    /// Surface exists but remains bounded or partial for this family.
    Partial,
    /// Surface is a research-only signal rather than a promoted capability.
    ResearchOnly,
    /// Surface has no landed capability for this family yet.
    NotLanded,
}

/// One capability cell in the Tassadar workload matrix.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarWorkloadCapabilityCell {
    /// Stable surface identifier.
    pub surface_id: String,
    /// Current posture on the workload family.
    pub posture: TassadarCapabilityPosture,
    /// Repo-owned artifact anchoring the posture.
    pub artifact_ref: String,
    /// Plain-language note tied to the artifact.
    pub note: String,
}

/// One workload-family row in the capability matrix.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarWorkloadCapabilityRow {
    /// Stable workload-family identifier.
    pub workload_family_id: String,
    /// Optional current benchmark workload target when one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload_target: Option<TassadarWorkloadTarget>,
    /// Short summary for the row.
    pub summary: String,
    /// Capability cells in stable surface order.
    pub capabilities: Vec<TassadarWorkloadCapabilityCell>,
}

/// Machine-readable workload capability matrix for the current Tassadar lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarWorkloadCapabilityMatrixReport {
    /// Stable schema version for the matrix report.
    pub schema_version: u16,
    /// Ordered artifact roots used to generate the matrix.
    pub generated_from_artifacts: Vec<String>,
    /// Ordered workload-family rows.
    pub rows: Vec<TassadarWorkloadCapabilityRow>,
    /// Plain-language boundary statement for the matrix.
    pub claim_boundary: String,
    /// Stable digest over the full report.
    pub report_digest: String,
}

/// One workload-family summary in the widened HullCache closure report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarHullCacheWorkloadSummary {
    /// Stable workload target for the summary.
    pub workload_target: TassadarWorkloadTarget,
    /// Current HullCache posture on the workload family.
    pub posture: TassadarCapabilityPosture,
    /// Number of cases that stayed direct on HullCache.
    pub direct_case_count: usize,
    /// Number of cases that fell back away from HullCache.
    pub fallback_case_count: usize,
    /// Average HullCache speedup over reference-linear execution.
    pub average_speedup_over_reference_linear: f64,
    /// Average remaining CPU gap ratio.
    pub average_remaining_gap_vs_cpu_reference: f64,
    /// Aggregated typed fallback reasons for the workload family.
    pub fallback_reason_counts: BTreeMap<String, u64>,
    /// Stable artifact ref anchoring the summary.
    pub artifact_ref: String,
}

/// Machine-readable widened HullCache closure report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarHullCacheClosureReport {
    /// Stable schema version.
    pub schema_version: u16,
    /// Direct exact workload families currently inside the widened closure.
    pub exact_workloads: Vec<TassadarHullCacheWorkloadSummary>,
    /// Workload families that remain fallback-only outside the direct closure.
    pub fallback_only_workloads: Vec<TassadarHullCacheWorkloadSummary>,
    /// Plain-language boundary statement for the current HullCache closure.
    pub claim_boundary: String,
    /// Stable digest over the report.
    pub report_digest: String,
}

/// Tassadar benchmark build or execution failure.
#[derive(Debug, Error)]
pub enum TassadarBenchmarkError {
    /// Environment bundle build failed.
    #[error(transparent)]
    Environment(#[from] TassadarEnvironmentError),
    /// Eval runtime failed.
    #[error(transparent)]
    EvalRuntime(#[from] EvalRuntimeError),
    /// Program artifact assembly failed.
    #[error(transparent)]
    ProgramArtifact(#[from] TassadarProgramArtifactError),
    /// Runtime execution refused one program.
    #[error(transparent)]
    ExecutionRefusal(#[from] TassadarExecutionRefusal),
    /// Executor descriptor rejected one program artifact.
    #[error(transparent)]
    ExecutorContract(#[from] TassadarExecutorContractError),
    /// Artifact count and validation-corpus count differed.
    #[error("Tassadar artifact count mismatch: expected {expected}, found {actual}")]
    ArtifactCountMismatch { expected: usize, actual: usize },
    /// One artifact targeted a different case than the current corpus ordering.
    #[error(
        "Tassadar artifact `{artifact_id}` does not match case `{case_id}` in the validation corpus"
    )]
    ArtifactCaseMismatch {
        artifact_id: String,
        case_id: String,
    },
}

/// Workload capability matrix build or persistence failure.
#[derive(Debug, Error)]
pub enum TassadarWorkloadCapabilityMatrixError {
    /// Underlying filesystem read/write failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON parsing or serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// One required workload family was missing from the article benchmark report.
    #[error("missing workload target `{workload_target:?}` in the article-class benchmark report")]
    MissingWorkloadTarget {
        /// Missing workload target.
        workload_target: TassadarWorkloadTarget,
    },
}

/// HullCache closure report build or persistence failure.
#[derive(Debug, Error)]
pub enum TassadarHullCacheClosureError {
    /// Underlying filesystem read/write failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON parsing or serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

/// Builds the packaged Phase 3 suite for the current Tassadar validation corpus.
pub fn build_tassadar_reference_fixture_suite(
    version: &str,
) -> Result<TassadarReferenceFixtureSuite, TassadarBenchmarkError> {
    let artifacts = tassadar_program_artifacts(version)?;
    let corpus_digest = stable_corpus_digest(artifacts.as_slice());
    let environment_bundle =
        build_tassadar_environment_bundle(version, &artifacts, &corpus_digest)?;
    let benchmark_package =
        build_tassadar_benchmark_package(version, &environment_bundle, artifacts.as_slice())?;
    Ok(TassadarReferenceFixtureSuite {
        environment_bundle,
        benchmark_package,
        artifacts,
        corpus_digest,
    })
}

/// Builds the widened article-class suite for the current Tassadar executor lane.
pub fn build_tassadar_article_class_suite(
    version: &str,
) -> Result<TassadarReferenceFixtureSuite, TassadarBenchmarkError> {
    let artifacts = tassadar_article_class_program_artifacts(version)?;
    let corpus_digest = stable_corpus_digest(artifacts.as_slice());
    let environment_bundle =
        build_tassadar_article_class_environment_bundle(version, &artifacts, &corpus_digest)?;
    let benchmark_package =
        build_tassadar_article_class_benchmark_package(version, &environment_bundle, &artifacts)?;
    Ok(TassadarReferenceFixtureSuite {
        environment_bundle,
        benchmark_package,
        artifacts,
        corpus_digest,
    })
}

/// Builds the bounded Hungarian-v0 suite for the current exact matching corpus.
pub fn build_tassadar_hungarian_v0_suite(
    version: &str,
) -> Result<TassadarReferenceFixtureSuite, TassadarBenchmarkError> {
    let artifacts = tassadar_hungarian_v0_program_artifacts(version)?;
    let corpus_digest = stable_corpus_digest(artifacts.as_slice());
    let environment_bundle =
        build_tassadar_hungarian_v0_environment_bundle(version, &artifacts, &corpus_digest)?;
    let benchmark_package =
        build_tassadar_hungarian_v0_benchmark_package(version, &environment_bundle, &artifacts)?;
    Ok(TassadarReferenceFixtureSuite {
        environment_bundle,
        benchmark_package,
        artifacts,
        corpus_digest,
    })
}

/// Runs the current packaged Phase 3 suite through the reference-linear executor.
pub fn run_tassadar_reference_fixture_benchmark(
    version: &str,
) -> Result<TassadarBenchmarkReport, TassadarBenchmarkError> {
    let suite = build_tassadar_reference_fixture_suite(version)?;
    let corpus = tassadar_validation_corpus();
    if suite.artifacts.len() != corpus.len() {
        return Err(TassadarBenchmarkError::ArtifactCountMismatch {
            expected: corpus.len(),
            actual: suite.artifacts.len(),
        });
    }

    let fixture = TassadarExecutorFixture::new();
    let descriptor = fixture.descriptor();
    let model_descriptor_digest = descriptor.stable_digest();
    let runtime_capability = fixture.runtime_capability_report();
    let cpu_runner = TassadarCpuReferenceRunner::new();
    let reference_linear_runner = TassadarFixtureRunner::new();
    let hull_cache_runner = TassadarHullCacheRunner::new();
    let sparse_top_k_runner = TassadarSparseTopKRunner::new();

    let mut eval_run = EvalRunState::open(
        EvalRunContract::new(
            format!("tassadar-benchmark-run-{version}"),
            EvalRunMode::Benchmark,
            suite.environment_bundle.benchmark_package.key.clone(),
        )
        .with_dataset(
            suite.environment_bundle.program_binding.dataset.clone(),
            Some(String::from("benchmark")),
        )
        .with_benchmark_package(suite.benchmark_package.key.clone())
        .with_expected_sample_count(corpus.len() as u64),
    )?;
    eval_run.start(1_000)?;

    let mut case_reports = Vec::new();
    for (ordinal, (case, artifact)) in corpus.into_iter().zip(suite.artifacts.iter()).enumerate() {
        if artifact.validated_program.program_id != case.program.program_id {
            return Err(TassadarBenchmarkError::ArtifactCaseMismatch {
                artifact_id: artifact.artifact_id.clone(),
                case_id: case.case_id,
            });
        }
        descriptor
            .validate_program_artifact(artifact, TassadarExecutorDecodeMode::ReferenceLinear)?;
        descriptor.validate_program_artifact(artifact, TassadarExecutorDecodeMode::HullCache)?;
        descriptor.validate_program_artifact(artifact, TassadarExecutorDecodeMode::SparseTopK)?;
        let selection = fixture
            .runtime_selection_diagnostic(&case.program, TassadarExecutorDecodeMode::HullCache);
        let sparse_top_k_selection = fixture
            .runtime_selection_diagnostic(&case.program, TassadarExecutorDecodeMode::SparseTopK);
        let effective_decode_mode = selection
            .effective_decode_mode
            .expect("validated benchmark corpus should not be refused");
        let equivalence_started = Instant::now();
        let equivalence_report = run_tassadar_exact_equivalence(&case.program)?;
        let equivalence_elapsed = equivalence_started.elapsed();
        let cpu_execution = &equivalence_report.cpu_reference;
        let reference_execution = &equivalence_report.reference_linear;
        let hull_cache_execution = &equivalence_report.hull_cache;
        let sparse_top_k_execution = &equivalence_report.sparse_top_k;
        let trace_steps = reference_execution.steps.len() as u64;
        let cpu_steps_per_second =
            benchmark_runner_steps_per_second(trace_steps, || cpu_runner.execute(&case.program))?;
        let reference_linear_steps_per_second =
            benchmark_runner_steps_per_second(trace_steps, || {
                reference_linear_runner.execute(&case.program)
            })?;
        let hull_cache_steps_per_second = benchmark_runner_steps_per_second(trace_steps, || {
            hull_cache_runner.execute(&case.program)
        })?;
        let sparse_top_k_steps_per_second = benchmark_runner_steps_per_second(trace_steps, || {
            sparse_top_k_runner.execute(&case.program)
        })?;
        let trace_digest_equal = equivalence_report.trace_digest_equal();
        let outputs_equal = equivalence_report.outputs_equal();
        let halt_equal = equivalence_report.halt_equal();
        let hull_cache_speedup_over_reference_linear =
            hull_cache_steps_per_second / reference_linear_steps_per_second.max(1e-9);
        let hull_cache_remaining_gap_vs_cpu_reference =
            cpu_steps_per_second / hull_cache_steps_per_second.max(1e-9);
        let sparse_top_k_speedup_over_reference_linear =
            sparse_top_k_steps_per_second / reference_linear_steps_per_second.max(1e-9);
        let sparse_top_k_remaining_gap_vs_cpu_reference =
            cpu_steps_per_second / sparse_top_k_steps_per_second.max(1e-9);

        let final_output_exactness_bps =
            u32::from(reference_execution.outputs == case.expected_outputs && outputs_equal)
                * 10_000;
        let step_exactness_bps =
            u32::from(reference_execution.steps == case.expected_trace && trace_digest_equal)
                * 10_000;
        let halt_exactness_bps = u32::from(halt_equal) * 10_000;
        let score_bps = (final_output_exactness_bps + step_exactness_bps + halt_exactness_bps) / 3;
        let status = if score_bps == 10_000 {
            EvalSampleStatus::Passed
        } else {
            EvalSampleStatus::Failed
        };

        let evidence = build_tassadar_execution_evidence_bundle(
            format!("tassadar-case-{}", case.case_id),
            stable_corpus_digest(std::slice::from_ref(artifact)),
            "tassadar_reference_fixture",
            descriptor.model.model_id.clone(),
            model_descriptor_digest.clone(),
            vec![suite.environment_bundle.benchmark_package.storage_key()],
            artifact,
            TassadarExecutorDecodeMode::ReferenceLinear,
            &reference_execution,
        );
        let sample_artifacts = build_case_artifacts(
            version,
            &case.case_id,
            artifact,
            reference_execution,
            &case,
            &evidence,
            &selection,
        )?;
        let sample = EvalSampleRecord {
            sample_id: case.case_id.clone(),
            ordinal: Some(ordinal as u64),
            environment: suite.environment_bundle.benchmark_package.key.clone(),
            status,
            input_ref: Some(format!("tassadar://input/{}/none", case.case_id)),
            output_ref: Some(format!(
                "tassadar://output/{}/reference_linear",
                case.case_id
            )),
            expected_output_ref: Some(format!("tassadar://expected_output/{}", case.case_id)),
            score_bps: Some(score_bps),
            metrics: vec![
                EvalMetric::new(
                    TASSADAR_OUTPUT_EXACTNESS_METRIC_ID,
                    f64::from(final_output_exactness_bps),
                )
                .with_unit("bps"),
                EvalMetric::new(
                    TASSADAR_STEP_EXACTNESS_METRIC_ID,
                    f64::from(step_exactness_bps),
                )
                .with_unit("bps"),
                EvalMetric::new(
                    TASSADAR_HALT_EXACTNESS_METRIC_ID,
                    f64::from(halt_exactness_bps),
                )
                .with_unit("bps"),
                EvalMetric::new(
                    TASSADAR_TRACE_DIGEST_EQUAL_METRIC_ID,
                    f64::from(u32::from(trace_digest_equal) * 10_000),
                )
                .with_unit("bps"),
                EvalMetric::new(TASSADAR_CPU_BASELINE_METRIC_ID, cpu_steps_per_second)
                    .with_unit("steps_per_second"),
                EvalMetric::new(
                    TASSADAR_REFERENCE_LINEAR_METRIC_ID,
                    reference_linear_steps_per_second,
                )
                .with_unit("steps_per_second"),
                EvalMetric::new(TASSADAR_HULL_CACHE_METRIC_ID, hull_cache_steps_per_second)
                    .with_unit("steps_per_second"),
                EvalMetric::new(
                    TASSADAR_HULL_CACHE_SPEEDUP_METRIC_ID,
                    hull_cache_speedup_over_reference_linear,
                )
                .with_unit("ratio"),
                EvalMetric::new(
                    TASSADAR_HULL_CACHE_CPU_GAP_METRIC_ID,
                    hull_cache_remaining_gap_vs_cpu_reference,
                )
                .with_unit("ratio"),
                EvalMetric::new(
                    TASSADAR_SPARSE_TOP_K_METRIC_ID,
                    sparse_top_k_steps_per_second,
                )
                .with_unit("steps_per_second"),
                EvalMetric::new(
                    TASSADAR_SPARSE_TOP_K_SPEEDUP_METRIC_ID,
                    sparse_top_k_speedup_over_reference_linear,
                )
                .with_unit("ratio"),
                EvalMetric::new(
                    TASSADAR_SPARSE_TOP_K_CPU_GAP_METRIC_ID,
                    sparse_top_k_remaining_gap_vs_cpu_reference,
                )
                .with_unit("ratio"),
                EvalMetric::new(TASSADAR_TRACE_STEP_COUNT_METRIC_ID, trace_steps as f64)
                    .with_unit("steps"),
            ],
            artifacts: sample_artifacts.clone(),
            error_reason: None,
            verification: Some(EvalVerificationFacts {
                timer_integrity: Some(EvalTimerIntegrityFacts {
                    declared_budget_ms: Some(
                        suite
                            .environment_bundle
                            .exactness_contract
                            .timeout_budget_ms,
                    ),
                    elapsed_ms: equivalence_elapsed.as_millis() as u64,
                    within_budget: equivalence_elapsed.as_millis() as u64
                        <= suite
                            .environment_bundle
                            .exactness_contract
                            .timeout_budget_ms,
                }),
                token_accounting: None,
                final_state: Some(EvalFinalStateCapture {
                    session_digest: reference_execution.behavior_digest(),
                    output_digest: Some(stable_outputs_digest(&reference_execution.outputs)),
                    artifact_digests: sample_artifacts
                        .iter()
                        .map(|artifact| artifact.artifact_digest.clone())
                        .collect(),
                }),
                execution_strategy: Some(EvalExecutionStrategyFacts {
                    strategy_label: String::from("tassadar_exact_equivalence_triplicate"),
                    runtime_family: Some(String::from("tassadar_executor")),
                    scheduler_posture: Some(String::from(
                        "cpu_reference+reference_linear+hull_cache",
                    )),
                }),
            }),
            session_digest: Some(reference_execution.behavior_digest()),
            metadata: std::collections::BTreeMap::from([
                (
                    String::from("workload_target"),
                    serde_json::to_value(classify_case(&case.case_id)).unwrap_or(Value::Null),
                ),
                (
                    String::from("cpu_behavior_digest"),
                    Value::String(cpu_execution.behavior_digest()),
                ),
                (
                    String::from("reference_linear_behavior_digest"),
                    Value::String(reference_execution.behavior_digest()),
                ),
                (
                    String::from("hull_cache_behavior_digest"),
                    Value::String(hull_cache_execution.behavior_digest()),
                ),
                (
                    String::from("sparse_top_k_behavior_digest"),
                    Value::String(sparse_top_k_execution.behavior_digest()),
                ),
                (
                    String::from("trace_digest_equal"),
                    Value::Bool(trace_digest_equal),
                ),
                (String::from("outputs_equal"), Value::Bool(outputs_equal)),
                (String::from("halt_equal"), Value::Bool(halt_equal)),
                (
                    String::from("hull_cache_speedup_over_reference_linear"),
                    serde_json::to_value(hull_cache_speedup_over_reference_linear)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("hull_cache_remaining_gap_vs_cpu_reference"),
                    serde_json::to_value(hull_cache_remaining_gap_vs_cpu_reference)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("sparse_top_k_speedup_over_reference_linear"),
                    serde_json::to_value(sparse_top_k_speedup_over_reference_linear)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("sparse_top_k_remaining_gap_vs_cpu_reference"),
                    serde_json::to_value(sparse_top_k_remaining_gap_vs_cpu_reference)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("selection_state"),
                    serde_json::to_value(selection.selection_state).unwrap_or(Value::Null),
                ),
                (
                    String::from("selection_reason"),
                    serde_json::to_value(selection.selection_reason).unwrap_or(Value::Null),
                ),
                (
                    String::from("sparse_top_k_selection_state"),
                    serde_json::to_value(sparse_top_k_selection.selection_state)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("sparse_top_k_selection_reason"),
                    serde_json::to_value(sparse_top_k_selection.selection_reason)
                        .unwrap_or(Value::Null),
                ),
            ]),
        };
        eval_run.append_sample(sample)?;
        case_reports.push(TassadarBenchmarkCaseReport {
            case_id: case.case_id,
            workload_target: classify_case(&artifact.validated_program.program_id),
            status,
            score_bps,
            final_output_exactness_bps,
            step_exactness_bps,
            halt_exactness_bps,
            requested_decode_mode: TassadarExecutorDecodeMode::HullCache,
            effective_decode_mode,
            selection_state: selection.selection_state,
            selection_reason: selection.selection_reason,
            used_decode_fallback: selection.is_fallback(),
            trace_digest_equal,
            outputs_equal,
            halt_equal,
            trace_steps,
            cpu_reference_steps_per_second: cpu_steps_per_second,
            reference_linear_steps_per_second,
            hull_cache_steps_per_second,
            hull_cache_speedup_over_reference_linear,
            hull_cache_remaining_gap_vs_cpu_reference,
            sparse_top_k_selection_state: sparse_top_k_selection.selection_state,
            sparse_top_k_selection_reason: sparse_top_k_selection.selection_reason,
            sparse_top_k_used_decode_fallback: sparse_top_k_selection.is_fallback(),
            sparse_top_k_steps_per_second,
            sparse_top_k_speedup_over_reference_linear,
            sparse_top_k_remaining_gap_vs_cpu_reference,
            cpu_behavior_digest: cpu_execution.behavior_digest(),
            reference_linear_behavior_digest: reference_execution.behavior_digest(),
            hull_cache_behavior_digest: hull_cache_execution.behavior_digest(),
            sparse_top_k_behavior_digest: sparse_top_k_execution.behavior_digest(),
        });
    }

    let run_artifacts = vec![
        EvalArtifact::new(
            "tassadar_benchmark_package.json",
            format!("artifact://tassadar/{version}/benchmark_package"),
            &serde_json::to_vec(&suite.benchmark_package).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_environment_bundle.json",
            format!("artifact://tassadar/{version}/environment_bundle"),
            &serde_json::to_vec(&suite.environment_bundle).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_runtime_capability.json",
            format!("artifact://tassadar/{version}/runtime_capability"),
            &serde_json::to_vec(&runtime_capability).unwrap_or_default(),
        ),
    ];
    eval_run.finalize(2_000, run_artifacts)?;

    let mut execution = suite
        .benchmark_package
        .clone()
        .open_execution(BenchmarkExecutionMode::OperatorSimulation)?;
    execution.record_round(&eval_run)?;
    let aggregate_summary = execution.finalize()?;

    Ok(TassadarBenchmarkReport {
        suite,
        eval_run,
        aggregate_summary,
        case_reports,
    })
}

/// Runs the widened article-class suite through the exact executor harness.
pub fn run_tassadar_article_class_benchmark(
    version: &str,
) -> Result<TassadarBenchmarkReport, TassadarBenchmarkError> {
    let suite = build_tassadar_article_class_suite(version)?;
    let corpus = tassadar_article_class_corpus();
    if suite.artifacts.len() != corpus.len() {
        return Err(TassadarBenchmarkError::ArtifactCountMismatch {
            expected: corpus.len(),
            actual: suite.artifacts.len(),
        });
    }

    let fixture = TassadarExecutorFixture::article_i32_compute_v1();
    let descriptor = fixture.descriptor();
    let model_descriptor_digest = descriptor.stable_digest();
    let runtime_capability = fixture.runtime_capability_report();

    let mut eval_run = EvalRunState::open(
        EvalRunContract::new(
            format!("tassadar-article-class-benchmark-run-{version}"),
            EvalRunMode::Benchmark,
            suite.environment_bundle.benchmark_package.key.clone(),
        )
        .with_dataset(
            suite.environment_bundle.program_binding.dataset.clone(),
            Some(String::from("benchmark")),
        )
        .with_benchmark_package(suite.benchmark_package.key.clone())
        .with_expected_sample_count(corpus.len() as u64),
    )?;
    eval_run.start(3_000)?;

    let mut case_reports = Vec::new();
    for (ordinal, (case, artifact)) in corpus.into_iter().zip(suite.artifacts.iter()).enumerate() {
        if artifact.validated_program.program_id != case.program.program_id {
            return Err(TassadarBenchmarkError::ArtifactCaseMismatch {
                artifact_id: artifact.artifact_id.clone(),
                case_id: case.case_id,
            });
        }
        descriptor
            .validate_program_artifact(artifact, TassadarExecutorDecodeMode::ReferenceLinear)?;
        descriptor.validate_program_artifact(artifact, TassadarExecutorDecodeMode::HullCache)?;
        descriptor.validate_program_artifact(artifact, TassadarExecutorDecodeMode::SparseTopK)?;
        let selection = fixture
            .runtime_selection_diagnostic(&case.program, TassadarExecutorDecodeMode::HullCache);
        let sparse_top_k_selection = fixture
            .runtime_selection_diagnostic(&case.program, TassadarExecutorDecodeMode::SparseTopK);
        let effective_decode_mode = selection
            .effective_decode_mode
            .expect("validated article-class corpus should not be refused");

        let cpu_runner = TassadarCpuReferenceRunner::for_program(&case.program)?;
        let reference_linear_runner = TassadarFixtureRunner::for_program(&case.program)?;
        let hull_cache_runner = TassadarHullCacheRunner::for_program(&case.program)?;
        let sparse_top_k_runner = TassadarSparseTopKRunner::for_program(&case.program)?;
        let equivalence_started = Instant::now();
        let equivalence_report = run_tassadar_exact_equivalence(&case.program)?;
        let equivalence_elapsed = equivalence_started.elapsed();
        let cpu_execution = &equivalence_report.cpu_reference;
        let reference_execution = &equivalence_report.reference_linear;
        let hull_cache_execution = &equivalence_report.hull_cache;
        let sparse_top_k_execution = &equivalence_report.sparse_top_k;
        let trace_steps = reference_execution.steps.len() as u64;
        let cpu_steps_per_second =
            benchmark_runner_steps_per_second(trace_steps, || cpu_runner.execute(&case.program))?;
        let reference_linear_steps_per_second =
            benchmark_runner_steps_per_second(trace_steps, || {
                reference_linear_runner.execute(&case.program)
            })?;
        let hull_cache_steps_per_second = match effective_decode_mode {
            TassadarExecutorDecodeMode::ReferenceLinear => {
                benchmark_runner_steps_per_second(trace_steps, || {
                    reference_linear_runner.execute(&case.program)
                })?
            }
            TassadarExecutorDecodeMode::HullCache => {
                benchmark_runner_steps_per_second(trace_steps, || {
                    hull_cache_runner.execute(&case.program)
                })?
            }
            TassadarExecutorDecodeMode::SparseTopK => {
                benchmark_runner_steps_per_second(trace_steps, || {
                    sparse_top_k_runner.execute(&case.program)
                })?
            }
        };
        let sparse_top_k_steps_per_second = match sparse_top_k_selection
            .effective_decode_mode
            .expect("validated article-class corpus should not be refused")
        {
            TassadarExecutorDecodeMode::ReferenceLinear => {
                benchmark_runner_steps_per_second(trace_steps, || {
                    reference_linear_runner.execute(&case.program)
                })?
            }
            TassadarExecutorDecodeMode::HullCache => {
                benchmark_runner_steps_per_second(trace_steps, || {
                    hull_cache_runner.execute(&case.program)
                })?
            }
            TassadarExecutorDecodeMode::SparseTopK => {
                benchmark_runner_steps_per_second(trace_steps, || {
                    sparse_top_k_runner.execute(&case.program)
                })?
            }
        };
        let trace_digest_equal = equivalence_report.trace_digest_equal();
        let outputs_equal = equivalence_report.outputs_equal();
        let halt_equal = equivalence_report.halt_equal();
        let hull_cache_speedup_over_reference_linear =
            hull_cache_steps_per_second / reference_linear_steps_per_second.max(1e-9);
        let hull_cache_remaining_gap_vs_cpu_reference =
            cpu_steps_per_second / hull_cache_steps_per_second.max(1e-9);
        let sparse_top_k_speedup_over_reference_linear =
            sparse_top_k_steps_per_second / reference_linear_steps_per_second.max(1e-9);
        let sparse_top_k_remaining_gap_vs_cpu_reference =
            cpu_steps_per_second / sparse_top_k_steps_per_second.max(1e-9);

        let final_output_exactness_bps =
            u32::from(reference_execution.outputs == case.expected_outputs && outputs_equal)
                * 10_000;
        let step_exactness_bps =
            u32::from(reference_execution.steps == case.expected_trace && trace_digest_equal)
                * 10_000;
        let halt_exactness_bps = u32::from(halt_equal) * 10_000;
        let score_bps = (final_output_exactness_bps + step_exactness_bps + halt_exactness_bps) / 3;
        let status = if score_bps == 10_000 {
            EvalSampleStatus::Passed
        } else {
            EvalSampleStatus::Failed
        };

        let evidence = build_tassadar_execution_evidence_bundle(
            format!("tassadar-article-class-case-{}", case.case_id),
            stable_corpus_digest(std::slice::from_ref(artifact)),
            "tassadar_article_class_fixture",
            descriptor.model.model_id.clone(),
            model_descriptor_digest.clone(),
            vec![suite.environment_bundle.benchmark_package.storage_key()],
            artifact,
            TassadarExecutorDecodeMode::ReferenceLinear,
            reference_execution,
        );
        let sample_artifacts = build_case_artifacts(
            version,
            &case.case_id,
            artifact,
            reference_execution,
            &case,
            &evidence,
            &selection,
        )?;
        let sample = EvalSampleRecord {
            sample_id: case.case_id.clone(),
            ordinal: Some(ordinal as u64),
            environment: suite.environment_bundle.benchmark_package.key.clone(),
            status,
            input_ref: Some(format!("tassadar://input/{}/none", case.case_id)),
            output_ref: Some(format!(
                "tassadar://output/{}/reference_linear",
                case.case_id
            )),
            expected_output_ref: Some(format!("tassadar://expected_output/{}", case.case_id)),
            score_bps: Some(score_bps),
            metrics: vec![
                EvalMetric::new(
                    TASSADAR_OUTPUT_EXACTNESS_METRIC_ID,
                    f64::from(final_output_exactness_bps),
                )
                .with_unit("bps"),
                EvalMetric::new(
                    TASSADAR_STEP_EXACTNESS_METRIC_ID,
                    f64::from(step_exactness_bps),
                )
                .with_unit("bps"),
                EvalMetric::new(
                    TASSADAR_HALT_EXACTNESS_METRIC_ID,
                    f64::from(halt_exactness_bps),
                )
                .with_unit("bps"),
                EvalMetric::new(TASSADAR_CPU_BASELINE_METRIC_ID, cpu_steps_per_second)
                    .with_unit("steps_per_second"),
                EvalMetric::new(
                    TASSADAR_REFERENCE_LINEAR_METRIC_ID,
                    reference_linear_steps_per_second,
                )
                .with_unit("steps_per_second"),
                EvalMetric::new(TASSADAR_HULL_CACHE_METRIC_ID, hull_cache_steps_per_second)
                    .with_unit("steps_per_second"),
                EvalMetric::new(
                    TASSADAR_HULL_CACHE_SPEEDUP_METRIC_ID,
                    hull_cache_speedup_over_reference_linear,
                )
                .with_unit("ratio"),
                EvalMetric::new(
                    TASSADAR_HULL_CACHE_CPU_GAP_METRIC_ID,
                    hull_cache_remaining_gap_vs_cpu_reference,
                )
                .with_unit("ratio"),
                EvalMetric::new(
                    TASSADAR_SPARSE_TOP_K_METRIC_ID,
                    sparse_top_k_steps_per_second,
                )
                .with_unit("steps_per_second"),
                EvalMetric::new(
                    TASSADAR_SPARSE_TOP_K_SPEEDUP_METRIC_ID,
                    sparse_top_k_speedup_over_reference_linear,
                )
                .with_unit("ratio"),
                EvalMetric::new(
                    TASSADAR_SPARSE_TOP_K_CPU_GAP_METRIC_ID,
                    sparse_top_k_remaining_gap_vs_cpu_reference,
                )
                .with_unit("ratio"),
                EvalMetric::new(
                    TASSADAR_TRACE_DIGEST_EQUAL_METRIC_ID,
                    f64::from(u32::from(trace_digest_equal) * 10_000),
                )
                .with_unit("bps"),
                EvalMetric::new(TASSADAR_TRACE_STEP_COUNT_METRIC_ID, trace_steps as f64)
                    .with_unit("steps"),
            ],
            artifacts: sample_artifacts.clone(),
            error_reason: None,
            verification: Some(EvalVerificationFacts {
                timer_integrity: Some(EvalTimerIntegrityFacts {
                    declared_budget_ms: Some(
                        suite
                            .environment_bundle
                            .exactness_contract
                            .timeout_budget_ms,
                    ),
                    elapsed_ms: equivalence_elapsed.as_millis() as u64,
                    within_budget: equivalence_elapsed.as_millis() as u64
                        <= suite
                            .environment_bundle
                            .exactness_contract
                            .timeout_budget_ms,
                }),
                token_accounting: None,
                final_state: Some(EvalFinalStateCapture {
                    session_digest: reference_execution.behavior_digest(),
                    output_digest: Some(stable_outputs_digest(&reference_execution.outputs)),
                    artifact_digests: sample_artifacts
                        .iter()
                        .map(|artifact| artifact.artifact_digest.clone())
                        .collect(),
                }),
                execution_strategy: Some(EvalExecutionStrategyFacts {
                    strategy_label: String::from(
                        "tassadar_article_class_exact_equivalence_triplicate",
                    ),
                    runtime_family: Some(String::from("tassadar_executor")),
                    scheduler_posture: Some(String::from(
                        "cpu_reference+reference_linear+hull_cache",
                    )),
                }),
            }),
            session_digest: Some(reference_execution.behavior_digest()),
            metadata: std::collections::BTreeMap::from([
                (
                    String::from("workload_target"),
                    serde_json::to_value(classify_case(&case.case_id)).unwrap_or(Value::Null),
                ),
                (
                    String::from("cpu_behavior_digest"),
                    Value::String(cpu_execution.behavior_digest()),
                ),
                (
                    String::from("reference_linear_behavior_digest"),
                    Value::String(reference_execution.behavior_digest()),
                ),
                (
                    String::from("hull_cache_behavior_digest"),
                    Value::String(hull_cache_execution.behavior_digest()),
                ),
                (
                    String::from("sparse_top_k_behavior_digest"),
                    Value::String(sparse_top_k_execution.behavior_digest()),
                ),
                (
                    String::from("trace_digest_equal"),
                    Value::Bool(trace_digest_equal),
                ),
                (String::from("outputs_equal"), Value::Bool(outputs_equal)),
                (String::from("halt_equal"), Value::Bool(halt_equal)),
                (
                    String::from("hull_cache_speedup_over_reference_linear"),
                    serde_json::to_value(hull_cache_speedup_over_reference_linear)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("hull_cache_remaining_gap_vs_cpu_reference"),
                    serde_json::to_value(hull_cache_remaining_gap_vs_cpu_reference)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("sparse_top_k_speedup_over_reference_linear"),
                    serde_json::to_value(sparse_top_k_speedup_over_reference_linear)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("sparse_top_k_remaining_gap_vs_cpu_reference"),
                    serde_json::to_value(sparse_top_k_remaining_gap_vs_cpu_reference)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("selection_state"),
                    serde_json::to_value(selection.selection_state).unwrap_or(Value::Null),
                ),
                (
                    String::from("selection_reason"),
                    serde_json::to_value(selection.selection_reason).unwrap_or(Value::Null),
                ),
                (
                    String::from("sparse_top_k_selection_state"),
                    serde_json::to_value(sparse_top_k_selection.selection_state)
                        .unwrap_or(Value::Null),
                ),
                (
                    String::from("sparse_top_k_selection_reason"),
                    serde_json::to_value(sparse_top_k_selection.selection_reason)
                        .unwrap_or(Value::Null),
                ),
            ]),
        };
        eval_run.append_sample(sample)?;

        case_reports.push(TassadarBenchmarkCaseReport {
            case_id: case.case_id,
            workload_target: classify_case(&artifact.validated_program.program_id),
            status,
            score_bps,
            final_output_exactness_bps,
            step_exactness_bps,
            halt_exactness_bps,
            requested_decode_mode: TassadarExecutorDecodeMode::HullCache,
            effective_decode_mode,
            selection_state: selection.selection_state,
            selection_reason: selection.selection_reason,
            used_decode_fallback: selection.selection_state
                == TassadarExecutorSelectionState::Fallback,
            trace_digest_equal,
            outputs_equal,
            halt_equal,
            trace_steps,
            cpu_reference_steps_per_second: cpu_steps_per_second,
            reference_linear_steps_per_second,
            hull_cache_steps_per_second,
            hull_cache_speedup_over_reference_linear,
            hull_cache_remaining_gap_vs_cpu_reference,
            sparse_top_k_selection_state: sparse_top_k_selection.selection_state,
            sparse_top_k_selection_reason: sparse_top_k_selection.selection_reason,
            sparse_top_k_used_decode_fallback: sparse_top_k_selection.selection_state
                == TassadarExecutorSelectionState::Fallback,
            sparse_top_k_steps_per_second,
            sparse_top_k_speedup_over_reference_linear,
            sparse_top_k_remaining_gap_vs_cpu_reference,
            cpu_behavior_digest: cpu_execution.behavior_digest(),
            reference_linear_behavior_digest: reference_execution.behavior_digest(),
            hull_cache_behavior_digest: hull_cache_execution.behavior_digest(),
            sparse_top_k_behavior_digest: sparse_top_k_execution.behavior_digest(),
        });
    }

    let run_artifacts = vec![
        EvalArtifact::new(
            "tassadar_article_class_benchmark_package.json",
            format!("artifact://tassadar/{version}/article_class_benchmark_package"),
            &serde_json::to_vec(&suite.benchmark_package).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_article_class_environment_bundle.json",
            format!("artifact://tassadar/{version}/article_class_environment_bundle"),
            &serde_json::to_vec(&suite.environment_bundle).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_runtime_capability.json",
            format!("artifact://tassadar/{version}/runtime_capability"),
            &serde_json::to_vec(&runtime_capability).unwrap_or_default(),
        ),
    ];
    eval_run.finalize(4_000, run_artifacts)?;

    let mut execution = suite
        .benchmark_package
        .clone()
        .open_execution(BenchmarkExecutionMode::OperatorSimulation)?;
    execution.record_round(&eval_run)?;
    let aggregate_summary = execution.finalize()?;

    Ok(TassadarBenchmarkReport {
        suite,
        eval_run,
        aggregate_summary,
        case_reports,
    })
}

/// Builds digest-bound fixture artifacts for the current validation corpus.
pub fn tassadar_program_artifacts(
    version: &str,
) -> Result<Vec<TassadarProgramArtifact>, TassadarBenchmarkError> {
    let profile = TassadarWasmProfile::core_i32_v1();
    let trace_abi = TassadarTraceAbi::core_i32_v1();
    tassadar_validation_corpus()
        .into_iter()
        .map(|case| {
            TassadarProgramArtifact::fixture_reference(
                format!("tassadar://artifact/{version}/{}", case.case_id),
                &profile,
                &trace_abi,
                case.program,
            )
            .map_err(TassadarBenchmarkError::from)
        })
        .collect()
}

fn build_tassadar_environment_bundle(
    version: &str,
    artifacts: &[TassadarProgramArtifact],
    corpus_digest: &str,
) -> Result<TassadarEnvironmentBundle, TassadarBenchmarkError> {
    let profile = TassadarWasmProfile::core_i32_v1();
    let trace_abi = TassadarTraceAbi::core_i32_v1();
    let dataset = DatasetKey::new(TASSADAR_VALIDATION_CORPUS_DATASET_REF, version);
    TassadarEnvironmentSpec {
        version: String::from(version),
        display_name: String::from("Tassadar Validation Corpus"),
        eval_environment_ref: String::from(TASSADAR_EVAL_ENVIRONMENT_REF),
        benchmark_environment_ref: String::from(TASSADAR_BENCHMARK_ENVIRONMENT_REF),
        eval_dataset: EnvironmentDatasetBinding {
            dataset: dataset.clone(),
            split: Some(String::from("validation")),
            mount_path: String::from("/datasets/tassadar/validation"),
            required: true,
        },
        benchmark_dataset: EnvironmentDatasetBinding {
            dataset: dataset.clone(),
            split: Some(String::from("benchmark")),
            mount_path: String::from("/datasets/tassadar/benchmark"),
            required: true,
        },
        package_refs: TassadarEnvironmentPackageRefs {
            group_ref: String::from("group.tassadar.validation"),
            eval_pin_alias: String::from("tassadar_eval"),
            benchmark_pin_alias: String::from("tassadar_benchmark"),
            eval_member_ref: String::from("tassadar_eval_member"),
            benchmark_member_ref: String::from("tassadar_benchmark_member"),
            program_corpus_ref: String::from("tassadar://corpus/phase1.validation"),
            io_contract_ref: String::from("tassadar://io/exact_i32_sequence"),
            rubric_binding_ref: String::from("tassadar://rubric/exactness"),
            eval_runtime_profile_ref: String::from("runtime://tassadar/eval"),
            benchmark_profile_ref: String::from("benchmark://tassadar/reference_fixture"),
            benchmark_runtime_profile_ref: String::from("runtime://tassadar/benchmark"),
        },
        program_binding: TassadarProgramBinding {
            dataset,
            program_corpus_ref: String::from("tassadar://corpus/phase1.validation"),
            corpus_digest: String::from(corpus_digest),
            wasm_profile_id: profile.profile_id.clone(),
            trace_abi_id: trace_abi.abi_id.clone(),
            trace_abi_version: trace_abi.schema_version,
            opcode_vocabulary_digest: profile.opcode_vocabulary_digest(),
            artifact_digests: artifacts
                .iter()
                .map(|artifact| artifact.artifact_digest.clone())
                .collect(),
        },
        io_contract: TassadarIoContract::exact_i32_sequence(),
        exactness_contract: TassadarExactnessContract {
            require_final_output_exactness: true,
            require_step_exactness: true,
            require_halt_exactness: true,
            timeout_budget_ms: 5_000,
            trace_budget_steps: 128,
            require_cpu_reference_baseline: true,
            require_reference_linear_baseline: true,
            future_throughput_metric_ids: vec![String::from(TASSADAR_HULL_CACHE_METRIC_ID)],
        },
        eval_policy_references: vec![EnvironmentPolicyReference {
            kind: EnvironmentPolicyKind::Verification,
            policy_ref: String::from("policy://tassadar/eval/verification"),
            required: true,
        }],
        benchmark_policy_references: vec![
            EnvironmentPolicyReference {
                kind: EnvironmentPolicyKind::Benchmark,
                policy_ref: String::from("policy://tassadar/benchmark"),
                required: true,
            },
            EnvironmentPolicyReference {
                kind: EnvironmentPolicyKind::Verification,
                policy_ref: String::from("policy://tassadar/benchmark/verification"),
                required: true,
            },
        ],
        current_workload_targets: vec![
            TassadarWorkloadTarget::ArithmeticMicroprogram,
            TassadarWorkloadTarget::MemoryLookupMicroprogram,
            TassadarWorkloadTarget::BranchControlFlowMicroprogram,
        ],
        planned_workload_targets: vec![
            TassadarWorkloadTarget::MicroWasmKernel,
            TassadarWorkloadTarget::SudokuClass,
            TassadarWorkloadTarget::HungarianMatching,
        ],
    }
    .build_bundle()
    .map_err(TassadarBenchmarkError::from)
}

fn build_tassadar_benchmark_package(
    version: &str,
    environment_bundle: &TassadarEnvironmentBundle,
    artifacts: &[TassadarProgramArtifact],
) -> Result<BenchmarkPackage, TassadarBenchmarkError> {
    let corpus = tassadar_validation_corpus();
    if artifacts.len() != corpus.len() {
        return Err(TassadarBenchmarkError::ArtifactCountMismatch {
            expected: corpus.len(),
            actual: artifacts.len(),
        });
    }

    let cases = corpus
        .into_iter()
        .zip(artifacts.iter())
        .enumerate()
        .map(|(ordinal, (case, artifact))| {
            let mut benchmark_case = BenchmarkCase::new(case.case_id.clone());
            benchmark_case.ordinal = Some(ordinal as u64);
            benchmark_case.input_ref = Some(format!("tassadar://input/{}/none", case.case_id));
            benchmark_case.expected_output_ref =
                Some(format!("tassadar://expected_output/{}", case.case_id));
            benchmark_case.metadata = json!({
                "summary": case.summary,
                "workload_target": classify_case(&case.case_id),
                "artifact_id": artifact.artifact_id,
                "artifact_digest": artifact.artifact_digest,
                "program_digest": artifact.validated_program_digest,
                "expected_outputs": case.expected_outputs,
                "expected_trace_steps": case.expected_trace.len(),
                "trace_budget_steps": environment_bundle.exactness_contract.trace_budget_steps,
                "timeout_budget_ms": environment_bundle.exactness_contract.timeout_budget_ms
            });
            benchmark_case
        })
        .collect::<Vec<_>>();

    let mut package = BenchmarkPackage::new(
        BenchmarkPackageKey::new(TASSADAR_REFERENCE_FIXTURE_BENCHMARK_REF, version),
        "Tassadar Validation Corpus Benchmark",
        environment_bundle.benchmark_package.key.clone(),
        1,
        BenchmarkAggregationKind::MedianScore,
    )
    .with_dataset(
        environment_bundle.program_binding.dataset.clone(),
        Some(String::from("benchmark")),
    )
    .with_verification_policy(BenchmarkVerificationPolicy {
        require_timer_integrity: true,
        require_token_accounting: false,
        require_final_state_capture: true,
        require_execution_strategy: true,
    })
    .with_cases(cases);
    package.metadata.insert(
        String::from("tassadar.current_workload_targets"),
        serde_json::to_value(&environment_bundle.current_workload_targets).unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.planned_workload_targets"),
        serde_json::to_value(&environment_bundle.planned_workload_targets).unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.cpu_baseline_metric_id"),
        Value::String(String::from(TASSADAR_CPU_BASELINE_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.reference_linear_metric_id"),
        Value::String(String::from(TASSADAR_REFERENCE_LINEAR_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.hull_cache_metric_id"),
        Value::String(String::from(TASSADAR_HULL_CACHE_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.sparse_top_k_metric_id"),
        Value::String(String::from(TASSADAR_SPARSE_TOP_K_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.corpus_digest"),
        Value::String(environment_bundle.program_binding.corpus_digest.clone()),
    );
    package.validate()?;
    Ok(package)
}

/// Builds digest-bound fixture artifacts for the widened article-class corpus.
pub fn tassadar_article_class_program_artifacts(
    version: &str,
) -> Result<Vec<TassadarProgramArtifact>, TassadarBenchmarkError> {
    let profile = TassadarWasmProfile::article_i32_compute_v1();
    let trace_abi = TassadarTraceAbi::article_i32_compute_v1();
    tassadar_article_class_corpus()
        .into_iter()
        .map(|case| {
            TassadarProgramArtifact::fixture_reference(
                format!(
                    "tassadar://artifact/{version}/article_class/{}",
                    case.case_id
                ),
                &profile,
                &trace_abi,
                case.program,
            )
            .map_err(TassadarBenchmarkError::from)
        })
        .collect()
}

fn build_tassadar_article_class_environment_bundle(
    version: &str,
    artifacts: &[TassadarProgramArtifact],
    corpus_digest: &str,
) -> Result<TassadarEnvironmentBundle, TassadarBenchmarkError> {
    let profile = TassadarWasmProfile::article_i32_compute_v1();
    let trace_abi = TassadarTraceAbi::article_i32_compute_v1();
    let dataset = DatasetKey::new(TASSADAR_ARTICLE_CLASS_DATASET_REF, version);
    TassadarEnvironmentSpec {
        version: String::from(version),
        display_name: String::from("Tassadar Article-Class Corpus"),
        eval_environment_ref: String::from(TASSADAR_ARTICLE_CLASS_EVAL_ENVIRONMENT_REF),
        benchmark_environment_ref: String::from(TASSADAR_ARTICLE_CLASS_BENCHMARK_ENVIRONMENT_REF),
        eval_dataset: EnvironmentDatasetBinding {
            dataset: dataset.clone(),
            split: Some(String::from("validation")),
            mount_path: String::from("/datasets/tassadar/article_class/validation"),
            required: true,
        },
        benchmark_dataset: EnvironmentDatasetBinding {
            dataset: dataset.clone(),
            split: Some(String::from("benchmark")),
            mount_path: String::from("/datasets/tassadar/article_class/benchmark"),
            required: true,
        },
        package_refs: TassadarEnvironmentPackageRefs {
            group_ref: String::from("group.tassadar.article_class"),
            eval_pin_alias: String::from("tassadar_article_class_eval"),
            benchmark_pin_alias: String::from("tassadar_article_class_benchmark"),
            eval_member_ref: String::from("tassadar_article_class_eval_member"),
            benchmark_member_ref: String::from("tassadar_article_class_benchmark_member"),
            program_corpus_ref: String::from("tassadar://corpus/article_class.validation"),
            io_contract_ref: String::from("tassadar://io/exact_i32_sequence"),
            rubric_binding_ref: String::from("tassadar://rubric/exactness"),
            eval_runtime_profile_ref: String::from("runtime://tassadar/article_class/eval"),
            benchmark_profile_ref: String::from(
                "benchmark://tassadar/article_class/reference_fixture",
            ),
            benchmark_runtime_profile_ref: String::from(
                "runtime://tassadar/article_class/benchmark",
            ),
        },
        program_binding: TassadarProgramBinding {
            dataset,
            program_corpus_ref: String::from("tassadar://corpus/article_class.validation"),
            corpus_digest: String::from(corpus_digest),
            wasm_profile_id: profile.profile_id.clone(),
            trace_abi_id: trace_abi.abi_id.clone(),
            trace_abi_version: trace_abi.schema_version,
            opcode_vocabulary_digest: profile.opcode_vocabulary_digest(),
            artifact_digests: artifacts
                .iter()
                .map(|artifact| artifact.artifact_digest.clone())
                .collect(),
        },
        io_contract: TassadarIoContract::exact_i32_sequence(),
        exactness_contract: TassadarExactnessContract {
            require_final_output_exactness: true,
            require_step_exactness: true,
            require_halt_exactness: true,
            timeout_budget_ms: 15_000,
            trace_budget_steps: 512,
            require_cpu_reference_baseline: true,
            require_reference_linear_baseline: true,
            future_throughput_metric_ids: vec![String::from(TASSADAR_HULL_CACHE_METRIC_ID)],
        },
        eval_policy_references: vec![EnvironmentPolicyReference {
            kind: EnvironmentPolicyKind::Verification,
            policy_ref: String::from("policy://tassadar/article_class/eval/verification"),
            required: true,
        }],
        benchmark_policy_references: vec![
            EnvironmentPolicyReference {
                kind: EnvironmentPolicyKind::Benchmark,
                policy_ref: String::from("policy://tassadar/article_class/benchmark"),
                required: true,
            },
            EnvironmentPolicyReference {
                kind: EnvironmentPolicyKind::Verification,
                policy_ref: String::from("policy://tassadar/article_class/benchmark/verification"),
                required: true,
            },
        ],
        current_workload_targets: vec![
            TassadarWorkloadTarget::MicroWasmKernel,
            TassadarWorkloadTarget::BranchHeavyKernel,
            TassadarWorkloadTarget::MemoryHeavyKernel,
            TassadarWorkloadTarget::LongLoopKernel,
            TassadarWorkloadTarget::SudokuClass,
            TassadarWorkloadTarget::HungarianMatching,
        ],
        planned_workload_targets: Vec::new(),
    }
    .build_bundle()
    .map_err(TassadarBenchmarkError::from)
}

fn build_tassadar_article_class_benchmark_package(
    version: &str,
    environment_bundle: &TassadarEnvironmentBundle,
    artifacts: &[TassadarProgramArtifact],
) -> Result<BenchmarkPackage, TassadarBenchmarkError> {
    let corpus = tassadar_article_class_corpus();
    let sudoku_v0_corpus = tassadar_sudoku_v0_corpus();
    let sudoku_v0_metadata = sudoku_v0_case_metadata_map(sudoku_v0_corpus.as_slice());
    if artifacts.len() != corpus.len() {
        return Err(TassadarBenchmarkError::ArtifactCountMismatch {
            expected: corpus.len(),
            actual: artifacts.len(),
        });
    }

    let cases = corpus
        .into_iter()
        .zip(artifacts.iter())
        .enumerate()
        .map(|(ordinal, (case, artifact))| {
            let mut benchmark_case = BenchmarkCase::new(case.case_id.clone());
            benchmark_case.ordinal = Some(ordinal as u64);
            benchmark_case.input_ref = Some(format!("tassadar://input/{}/none", case.case_id));
            benchmark_case.expected_output_ref =
                Some(format!("tassadar://expected_output/{}", case.case_id));
            let sudoku_v0_case = sudoku_v0_metadata.get(case.case_id.as_str());
            benchmark_case.metadata = json!({
                "summary": case.summary,
                "workload_target": classify_case(&case.case_id),
                "artifact_id": artifact.artifact_id,
                "artifact_digest": artifact.artifact_digest,
                "program_digest": artifact.validated_program_digest,
                "expected_outputs": case.expected_outputs,
                "expected_trace_steps": case.expected_trace.len(),
                "trace_budget_steps": environment_bundle.exactness_contract.trace_budget_steps,
                "timeout_budget_ms": environment_bundle.exactness_contract.timeout_budget_ms,
                "sudoku_v0_split": sudoku_v0_case.map(|case| case.split.as_str()),
                "sudoku_v0_given_count": sudoku_v0_case.map(|case| case.given_count),
                "sudoku_v0_puzzle_cells": sudoku_v0_case.map(|case| case.puzzle_cells.clone()),
            });
            benchmark_case
        })
        .collect::<Vec<_>>();

    let mut package = BenchmarkPackage::new(
        BenchmarkPackageKey::new(TASSADAR_ARTICLE_CLASS_BENCHMARK_REF, version),
        "Tassadar Article-Class Benchmark",
        environment_bundle.benchmark_package.key.clone(),
        1,
        BenchmarkAggregationKind::MedianScore,
    )
    .with_dataset(
        environment_bundle.program_binding.dataset.clone(),
        Some(String::from("benchmark")),
    )
    .with_verification_policy(BenchmarkVerificationPolicy {
        require_timer_integrity: true,
        require_token_accounting: false,
        require_final_state_capture: true,
        require_execution_strategy: true,
    })
    .with_cases(cases);
    package.metadata.insert(
        String::from("tassadar.current_workload_targets"),
        serde_json::to_value(&environment_bundle.current_workload_targets).unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.planned_workload_targets"),
        serde_json::to_value(&environment_bundle.planned_workload_targets).unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.cpu_baseline_metric_id"),
        Value::String(String::from(TASSADAR_CPU_BASELINE_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.reference_linear_metric_id"),
        Value::String(String::from(TASSADAR_REFERENCE_LINEAR_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.hull_cache_metric_id"),
        Value::String(String::from(TASSADAR_HULL_CACHE_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.sparse_top_k_metric_id"),
        Value::String(String::from(TASSADAR_SPARSE_TOP_K_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.corpus_digest"),
        Value::String(environment_bundle.program_binding.corpus_digest.clone()),
    );
    package.metadata.insert(
        String::from("tassadar.sudoku_v0_train_case_ids"),
        serde_json::to_value(sudoku_v0_case_ids_for_split(
            sudoku_v0_corpus.as_slice(),
            TassadarSudokuV0CorpusSplit::Train,
        ))
        .unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.sudoku_v0_validation_case_ids"),
        serde_json::to_value(sudoku_v0_case_ids_for_split(
            sudoku_v0_corpus.as_slice(),
            TassadarSudokuV0CorpusSplit::Validation,
        ))
        .unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.sudoku_v0_test_case_ids"),
        serde_json::to_value(sudoku_v0_case_ids_for_split(
            sudoku_v0_corpus.as_slice(),
            TassadarSudokuV0CorpusSplit::Test,
        ))
        .unwrap_or(Value::Null),
    );
    package.validate()?;
    Ok(package)
}

/// Builds digest-bound fixture artifacts for the bounded Hungarian-v0 corpus.
pub fn tassadar_hungarian_v0_program_artifacts(
    version: &str,
) -> Result<Vec<TassadarProgramArtifact>, TassadarBenchmarkError> {
    let profile = TassadarWasmProfile::hungarian_v0_matching_v1();
    let trace_abi = TassadarTraceAbi::hungarian_v0_matching_v1();
    tassadar_hungarian_v0_corpus()
        .into_iter()
        .map(|case| {
            TassadarProgramArtifact::fixture_reference(
                format!(
                    "tassadar://artifact/{version}/hungarian_v0/{}",
                    case.case_id
                ),
                &profile,
                &trace_abi,
                case.validation_case.program,
            )
            .map_err(TassadarBenchmarkError::from)
        })
        .collect()
}

fn build_tassadar_hungarian_v0_environment_bundle(
    version: &str,
    artifacts: &[TassadarProgramArtifact],
    corpus_digest: &str,
) -> Result<TassadarEnvironmentBundle, TassadarBenchmarkError> {
    let profile = TassadarWasmProfile::hungarian_v0_matching_v1();
    let trace_abi = TassadarTraceAbi::hungarian_v0_matching_v1();
    let dataset = DatasetKey::new(TASSADAR_HUNGARIAN_V0_DATASET_REF, version);
    TassadarEnvironmentSpec {
        version: String::from(version),
        display_name: String::from("Tassadar Hungarian-v0 Corpus"),
        eval_environment_ref: String::from(TASSADAR_HUNGARIAN_V0_EVAL_ENVIRONMENT_REF),
        benchmark_environment_ref: String::from(TASSADAR_HUNGARIAN_V0_BENCHMARK_ENVIRONMENT_REF),
        eval_dataset: EnvironmentDatasetBinding {
            dataset: dataset.clone(),
            split: Some(String::from("validation")),
            mount_path: String::from("/datasets/tassadar/hungarian_v0/validation"),
            required: true,
        },
        benchmark_dataset: EnvironmentDatasetBinding {
            dataset: dataset.clone(),
            split: Some(String::from("benchmark")),
            mount_path: String::from("/datasets/tassadar/hungarian_v0/benchmark"),
            required: true,
        },
        package_refs: TassadarEnvironmentPackageRefs {
            group_ref: String::from("group.tassadar.hungarian_v0"),
            eval_pin_alias: String::from("tassadar_hungarian_v0_eval"),
            benchmark_pin_alias: String::from("tassadar_hungarian_v0_benchmark"),
            eval_member_ref: String::from("tassadar_hungarian_v0_eval_member"),
            benchmark_member_ref: String::from("tassadar_hungarian_v0_benchmark_member"),
            program_corpus_ref: String::from("tassadar://corpus/hungarian_v0.validation"),
            io_contract_ref: String::from("tassadar://io/exact_i32_sequence"),
            rubric_binding_ref: String::from("tassadar://rubric/exactness"),
            eval_runtime_profile_ref: String::from("runtime://tassadar/hungarian_v0/eval"),
            benchmark_profile_ref: String::from(
                "benchmark://tassadar/hungarian_v0/reference_fixture",
            ),
            benchmark_runtime_profile_ref: String::from(
                "runtime://tassadar/hungarian_v0/benchmark",
            ),
        },
        program_binding: TassadarProgramBinding {
            dataset,
            program_corpus_ref: String::from("tassadar://corpus/hungarian_v0.validation"),
            corpus_digest: String::from(corpus_digest),
            wasm_profile_id: profile.profile_id.clone(),
            trace_abi_id: trace_abi.abi_id.clone(),
            trace_abi_version: trace_abi.schema_version,
            opcode_vocabulary_digest: profile.opcode_vocabulary_digest(),
            artifact_digests: artifacts
                .iter()
                .map(|artifact| artifact.artifact_digest.clone())
                .collect(),
        },
        io_contract: TassadarIoContract::exact_i32_sequence(),
        exactness_contract: TassadarExactnessContract {
            require_final_output_exactness: true,
            require_step_exactness: true,
            require_halt_exactness: true,
            timeout_budget_ms: 15_000,
            trace_budget_steps: 4_096,
            require_cpu_reference_baseline: true,
            require_reference_linear_baseline: true,
            future_throughput_metric_ids: vec![String::from(TASSADAR_HULL_CACHE_METRIC_ID)],
        },
        eval_policy_references: vec![EnvironmentPolicyReference {
            kind: EnvironmentPolicyKind::Verification,
            policy_ref: String::from("policy://tassadar/hungarian_v0/eval/verification"),
            required: true,
        }],
        benchmark_policy_references: vec![
            EnvironmentPolicyReference {
                kind: EnvironmentPolicyKind::Benchmark,
                policy_ref: String::from("policy://tassadar/hungarian_v0/benchmark"),
                required: true,
            },
            EnvironmentPolicyReference {
                kind: EnvironmentPolicyKind::Verification,
                policy_ref: String::from("policy://tassadar/hungarian_v0/benchmark/verification"),
                required: true,
            },
        ],
        current_workload_targets: vec![TassadarWorkloadTarget::HungarianMatching],
        planned_workload_targets: Vec::new(),
    }
    .build_bundle()
    .map_err(TassadarBenchmarkError::from)
}

fn build_tassadar_hungarian_v0_benchmark_package(
    version: &str,
    environment_bundle: &TassadarEnvironmentBundle,
    artifacts: &[TassadarProgramArtifact],
) -> Result<BenchmarkPackage, TassadarBenchmarkError> {
    let corpus = tassadar_hungarian_v0_corpus();
    let train_case_ids =
        hungarian_v0_case_ids_for_split(corpus.as_slice(), TassadarSudokuV0CorpusSplit::Train);
    let validation_case_ids =
        hungarian_v0_case_ids_for_split(corpus.as_slice(), TassadarSudokuV0CorpusSplit::Validation);
    let test_case_ids =
        hungarian_v0_case_ids_for_split(corpus.as_slice(), TassadarSudokuV0CorpusSplit::Test);
    if artifacts.len() != corpus.len() {
        return Err(TassadarBenchmarkError::ArtifactCountMismatch {
            expected: corpus.len(),
            actual: artifacts.len(),
        });
    }

    let cases = corpus
        .into_iter()
        .zip(artifacts.iter())
        .enumerate()
        .map(|(ordinal, (case, artifact))| {
            let mut benchmark_case = BenchmarkCase::new(case.case_id.clone());
            benchmark_case.ordinal = Some(ordinal as u64);
            benchmark_case.input_ref = Some(format!("tassadar://input/{}/none", case.case_id));
            benchmark_case.expected_output_ref =
                Some(format!("tassadar://expected_output/{}", case.case_id));
            benchmark_case.metadata = json!({
                "summary": case.validation_case.summary,
                "workload_target": classify_case(&case.case_id),
                "artifact_id": artifact.artifact_id,
                "artifact_digest": artifact.artifact_digest,
                "program_digest": artifact.validated_program_digest,
                "expected_outputs": case.validation_case.expected_outputs,
                "expected_trace_steps": case.validation_case.expected_trace.len(),
                "trace_budget_steps": environment_bundle.exactness_contract.trace_budget_steps,
                "timeout_budget_ms": environment_bundle.exactness_contract.timeout_budget_ms,
                "hungarian_v0_split": case.split.as_str(),
                "hungarian_v0_cost_matrix": case.cost_matrix.clone(),
                "hungarian_v0_optimal_assignment": case.optimal_assignment.clone(),
                "hungarian_v0_optimal_cost": case.optimal_cost,
            });
            benchmark_case
        })
        .collect::<Vec<_>>();

    let mut package = BenchmarkPackage::new(
        BenchmarkPackageKey::new(TASSADAR_HUNGARIAN_V0_BENCHMARK_REF, version),
        "Tassadar Hungarian-v0 Benchmark",
        environment_bundle.benchmark_package.key.clone(),
        1,
        BenchmarkAggregationKind::MedianScore,
    )
    .with_dataset(
        environment_bundle.program_binding.dataset.clone(),
        Some(String::from("benchmark")),
    )
    .with_verification_policy(BenchmarkVerificationPolicy {
        require_timer_integrity: true,
        require_token_accounting: false,
        require_final_state_capture: true,
        require_execution_strategy: true,
    })
    .with_cases(cases);
    package.metadata.insert(
        String::from("tassadar.current_workload_targets"),
        serde_json::to_value(&environment_bundle.current_workload_targets).unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.planned_workload_targets"),
        serde_json::to_value(&environment_bundle.planned_workload_targets).unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.cpu_baseline_metric_id"),
        Value::String(String::from(TASSADAR_CPU_BASELINE_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.reference_linear_metric_id"),
        Value::String(String::from(TASSADAR_REFERENCE_LINEAR_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.hull_cache_metric_id"),
        Value::String(String::from(TASSADAR_HULL_CACHE_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.sparse_top_k_metric_id"),
        Value::String(String::from(TASSADAR_SPARSE_TOP_K_METRIC_ID)),
    );
    package.metadata.insert(
        String::from("tassadar.corpus_digest"),
        Value::String(environment_bundle.program_binding.corpus_digest.clone()),
    );
    package.metadata.insert(
        String::from("tassadar.hungarian_v0_train_case_ids"),
        serde_json::to_value(train_case_ids).unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.hungarian_v0_validation_case_ids"),
        serde_json::to_value(validation_case_ids).unwrap_or(Value::Null),
    );
    package.metadata.insert(
        String::from("tassadar.hungarian_v0_test_case_ids"),
        serde_json::to_value(test_case_ids).unwrap_or(Value::Null),
    );
    package.validate()?;
    Ok(package)
}

fn classify_case(case_id: &str) -> TassadarWorkloadTarget {
    match case_id {
        "locals_add" | "tassadar.locals_add.v1" => TassadarWorkloadTarget::ArithmeticMicroprogram,
        "memory_roundtrip" | "tassadar.memory_roundtrip.v1" => {
            TassadarWorkloadTarget::MemoryLookupMicroprogram
        }
        "branch_guard" | "tassadar.branch_guard.v1" => {
            TassadarWorkloadTarget::BranchControlFlowMicroprogram
        }
        "micro_wasm_kernel" | "tassadar.micro_wasm_kernel.v2" => {
            TassadarWorkloadTarget::MicroWasmKernel
        }
        "branch_heavy_kernel" | "tassadar.branch_heavy_kernel.v1" => {
            TassadarWorkloadTarget::BranchHeavyKernel
        }
        "memory_heavy_kernel" | "tassadar.memory_heavy_kernel.v1" => {
            TassadarWorkloadTarget::MemoryHeavyKernel
        }
        "long_loop_kernel" | "tassadar.long_loop_kernel.v1" => {
            TassadarWorkloadTarget::LongLoopKernel
        }
        "sudoku_class" | "tassadar.sudoku_class.v2" => TassadarWorkloadTarget::SudokuClass,
        value if value.starts_with("sudoku_v0_") || value.starts_with("tassadar.sudoku_v0_") => {
            TassadarWorkloadTarget::SudokuClass
        }
        "hungarian_matching" | "tassadar.hungarian_matching.v2" => {
            TassadarWorkloadTarget::HungarianMatching
        }
        value
            if value.starts_with("hungarian_v0_")
                || value.starts_with("tassadar.hungarian_v0_") =>
        {
            TassadarWorkloadTarget::HungarianMatching
        }
        _ => TassadarWorkloadTarget::MicroWasmKernel,
    }
}

fn sudoku_v0_case_metadata_map(
    sudoku_v0_corpus: &[TassadarSudokuV0CorpusCase],
) -> std::collections::BTreeMap<&str, &TassadarSudokuV0CorpusCase> {
    sudoku_v0_corpus
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect()
}

fn sudoku_v0_case_ids_for_split(
    sudoku_v0_corpus: &[TassadarSudokuV0CorpusCase],
    split: TassadarSudokuV0CorpusSplit,
) -> Vec<String> {
    sudoku_v0_corpus
        .iter()
        .filter(|case| case.split == split)
        .map(|case| case.case_id.clone())
        .collect()
}

fn hungarian_v0_case_ids_for_split(
    corpus: &[TassadarHungarianV0CorpusCase],
    split: TassadarSudokuV0CorpusSplit,
) -> Vec<String> {
    corpus
        .iter()
        .filter(|case| case.split == split)
        .map(|case| case.case_id.clone())
        .collect()
}

fn eval_repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn workload_target_id(workload_target: TassadarWorkloadTarget) -> &'static str {
    match workload_target {
        TassadarWorkloadTarget::ArithmeticMicroprogram => "arithmetic_microprogram",
        TassadarWorkloadTarget::MemoryLookupMicroprogram => "memory_lookup_microprogram",
        TassadarWorkloadTarget::BranchControlFlowMicroprogram => "branch_control_flow_microprogram",
        TassadarWorkloadTarget::MicroWasmKernel => "micro_wasm_kernel",
        TassadarWorkloadTarget::BranchHeavyKernel => "branch_heavy_kernel",
        TassadarWorkloadTarget::MemoryHeavyKernel => "memory_heavy_kernel",
        TassadarWorkloadTarget::LongLoopKernel => "long_loop_kernel",
        TassadarWorkloadTarget::SudokuClass => "sudoku_class",
        TassadarWorkloadTarget::HungarianMatching => "hungarian_matching",
    }
}

fn read_repo_json<T: serde::de::DeserializeOwned>(
    relative_path: &str,
) -> Result<T, TassadarWorkloadCapabilityMatrixError> {
    let bytes = std::fs::read(eval_repo_root().join(relative_path))?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn runtime_capability_cell(
    benchmark_report: &TassadarBenchmarkReport,
    workload_target: TassadarWorkloadTarget,
    surface_id: &str,
) -> Result<TassadarWorkloadCapabilityCell, TassadarWorkloadCapabilityMatrixError> {
    let cases = benchmark_report
        .case_reports
        .iter()
        .filter(|case| case.workload_target == workload_target)
        .collect::<Vec<_>>();
    if cases.is_empty() {
        return Err(
            TassadarWorkloadCapabilityMatrixError::MissingWorkloadTarget { workload_target },
        );
    }
    let exact_count = cases.iter().filter(|case| case.score_bps == 10_000).count();
    let total = cases.len();
    let (posture, note) = match surface_id {
        "runtime.reference_linear" => (
            TassadarCapabilityPosture::Exact,
            format!(
                "reference-linear exact on {exact_count}/{total} benchmark cases for `{}`",
                workload_target_id(workload_target)
            ),
        ),
        "runtime.hull_cache" => {
            let direct_count = cases
                .iter()
                .filter(|case| case.selection_state == TassadarExecutorSelectionState::Direct)
                .count();
            let fallback_count = total.saturating_sub(direct_count);
            let posture = if direct_count == total {
                TassadarCapabilityPosture::Exact
            } else {
                TassadarCapabilityPosture::FallbackOnly
            };
            (
                posture,
                format!(
                    "hull-cache exact on {exact_count}/{total} cases with {direct_count} direct and {fallback_count} fallback selections for `{}`",
                    workload_target_id(workload_target)
                ),
            )
        }
        "runtime.sparse_top_k" => {
            let direct_count = cases
                .iter()
                .filter(|case| {
                    case.sparse_top_k_selection_state == TassadarExecutorSelectionState::Direct
                })
                .count();
            let fallback_count = total.saturating_sub(direct_count);
            let posture = if direct_count == total {
                TassadarCapabilityPosture::Exact
            } else {
                TassadarCapabilityPosture::FallbackOnly
            };
            (
                posture,
                format!(
                    "sparse-top-k exact on {exact_count}/{total} cases with {direct_count} direct and {fallback_count} fallback selections for `{}`",
                    workload_target_id(workload_target)
                ),
            )
        }
        _ => unreachable!("unsupported runtime surface"),
    };
    Ok(TassadarWorkloadCapabilityCell {
        surface_id: String::from(surface_id),
        posture,
        artifact_ref: String::from(TASSADAR_ARTICLE_CLASS_BENCHMARK_REPORT_REF),
        note,
    })
}

/// Builds the machine-readable Tassadar workload capability matrix from the
/// current committed benchmark, compiled, and learned artifacts.
pub fn build_tassadar_workload_capability_matrix_report(
) -> Result<TassadarWorkloadCapabilityMatrixReport, TassadarWorkloadCapabilityMatrixError> {
    let benchmark_report =
        read_repo_json::<TassadarBenchmarkReport>(TASSADAR_ARTICLE_CLASS_BENCHMARK_REPORT_REF)?;
    let sudoku_compiled_run_bundle = read_repo_json::<Value>(
        "fixtures/tassadar/runs/sudoku_v0_compiled_executor_v0/run_bundle.json",
    )?;
    let hungarian_compiled_run_bundle = read_repo_json::<Value>(
        "fixtures/tassadar/runs/hungarian_v0_compiled_executor_v0/run_bundle.json",
    )?;
    let sudoku_promotion_bundle = read_repo_json::<Value>(
        "fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_bundle.json",
    )?;
    let sudoku_promotion_gate = read_repo_json::<Value>(
        "fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_gate_report.json",
    )?;
    let sudoku_promotion_family = read_repo_json::<Value>(
        "fixtures/tassadar/runs/sudoku_v0_promotion_v3/family_report.json",
    )?;
    let sudoku_9x9_fit_report = read_repo_json::<Value>(
        "fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0/sequence_fit_report.json",
    )?;

    let rows = vec![
        TassadarWorkloadCapabilityRow {
            workload_family_id: String::from("micro_wasm_kernel"),
            workload_target: Some(TassadarWorkloadTarget::MicroWasmKernel),
            summary: String::from(
                "unrolled article-class micro-kernel over memory-backed inputs",
            ),
            capabilities: vec![
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::MicroWasmKernel,
                    "runtime.reference_linear",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::MicroWasmKernel,
                    "runtime.hull_cache",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::MicroWasmKernel,
                    "runtime.sparse_top_k",
                )?,
            ],
        },
        TassadarWorkloadCapabilityRow {
            workload_family_id: String::from("branch_heavy_kernel"),
            workload_target: Some(TassadarWorkloadTarget::BranchHeavyKernel),
            summary: String::from(
                "forward-branch ladder kernel that widens control-flow coverage without backward-branch fast-path erasure",
            ),
            capabilities: vec![
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::BranchHeavyKernel,
                    "runtime.reference_linear",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::BranchHeavyKernel,
                    "runtime.hull_cache",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::BranchHeavyKernel,
                    "runtime.sparse_top_k",
                )?,
            ],
        },
        TassadarWorkloadCapabilityRow {
            workload_family_id: String::from("memory_heavy_kernel"),
            workload_target: Some(TassadarWorkloadTarget::MemoryHeavyKernel),
            summary: String::from(
                "dense load/store kernel with staged accumulator writes across the article benchmark path",
            ),
            capabilities: vec![
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::MemoryHeavyKernel,
                    "runtime.reference_linear",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::MemoryHeavyKernel,
                    "runtime.hull_cache",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::MemoryHeavyKernel,
                    "runtime.sparse_top_k",
                )?,
            ],
        },
        TassadarWorkloadCapabilityRow {
            workload_family_id: String::from("long_loop_kernel"),
            workload_target: Some(TassadarWorkloadTarget::LongLoopKernel),
            summary: String::from(
                "current article-class long-horizon decrement loop and trace-ABI exemplar",
            ),
            capabilities: vec![
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::LongLoopKernel,
                    "runtime.reference_linear",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::LongLoopKernel,
                    "runtime.hull_cache",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::LongLoopKernel,
                    "runtime.sparse_top_k",
                )?,
            ],
        },
        TassadarWorkloadCapabilityRow {
            workload_family_id: String::from("sudoku_class_4x4"),
            workload_target: Some(TassadarWorkloadTarget::SudokuClass),
            summary: String::from(
                "real 4x4 search-heavy Sudoku family with runtime, compiled, and bounded learned evidence",
            ),
            capabilities: vec![
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::SudokuClass,
                    "runtime.reference_linear",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::SudokuClass,
                    "runtime.hull_cache",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::SudokuClass,
                    "runtime.sparse_top_k",
                )?,
                TassadarWorkloadCapabilityCell {
                    surface_id: String::from("compiled.proof_backed"),
                    posture: TassadarCapabilityPosture::Exact,
                    artifact_ref: String::from(
                        "fixtures/tassadar/runs/sudoku_v0_compiled_executor_v0/run_bundle.json",
                    ),
                    note: format!(
                        "{}; serve_posture={}",
                        sudoku_compiled_run_bundle["claim_boundary"]
                            .as_str()
                            .unwrap_or("compiled Sudoku boundary missing"),
                        sudoku_compiled_run_bundle["serve_posture"]
                            .as_str()
                            .unwrap_or("unknown")
                    ),
                },
                TassadarWorkloadCapabilityCell {
                    surface_id: String::from("learned.bounded"),
                    posture: TassadarCapabilityPosture::Partial,
                    artifact_ref: String::from(
                        "fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_bundle.json",
                    ),
                    note: format!(
                        "claim_class={}; first_target_exactness_bps={}; first_32_token_exactness_bps={}; exact_trace_case_count={}; claim_boundary={}",
                        sudoku_promotion_bundle["claim_class"]
                            .as_str()
                            .unwrap_or("unknown"),
                        sudoku_promotion_gate["first_target_exactness_bps"]
                            .as_u64()
                            .unwrap_or(0),
                        sudoku_promotion_gate["first_32_token_exactness_bps"]
                            .as_u64()
                            .unwrap_or(0),
                        sudoku_promotion_gate["exact_trace_case_count"]
                            .as_u64()
                            .unwrap_or(0),
                        sudoku_promotion_family["claim_boundary"]
                            .as_str()
                            .unwrap_or("unknown")
                    ),
                },
            ],
        },
        TassadarWorkloadCapabilityRow {
            workload_family_id: String::from("sudoku_search_9x9"),
            workload_target: None,
            summary: String::from(
                "real 9x9 Sudoku search workload beyond the current learned full-trace fit envelope",
            ),
            capabilities: vec![TassadarWorkloadCapabilityCell {
                surface_id: String::from("learned.long_horizon"),
                posture: TassadarCapabilityPosture::Partial,
                artifact_ref: String::from(
                    "fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0/sequence_fit_report.json",
                ),
                note: format!(
                    "{}; model_max_sequence_tokens={}; total_token_count_min={}; total_token_count_max={}; strategy={}",
                    sudoku_9x9_fit_report["outcome_statement"]
                        .as_str()
                        .unwrap_or("unknown"),
                    sudoku_9x9_fit_report["model_max_sequence_tokens"]
                        .as_u64()
                        .unwrap_or(0),
                    sudoku_9x9_fit_report["total_token_count_min"]
                        .as_u64()
                        .unwrap_or(0),
                    sudoku_9x9_fit_report["total_token_count_max"]
                        .as_u64()
                        .unwrap_or(0),
                    sudoku_9x9_fit_report["teacher_forced_training_strategy"]
                        .as_str()
                        .unwrap_or("unknown")
                ),
            }],
        },
        TassadarWorkloadCapabilityRow {
            workload_family_id: String::from("hungarian_matching_4x4"),
            workload_target: Some(TassadarWorkloadTarget::HungarianMatching),
            summary: String::from(
                "bounded exact 4x4 Hungarian matching family with runtime and compiled coverage",
            ),
            capabilities: vec![
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::HungarianMatching,
                    "runtime.reference_linear",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::HungarianMatching,
                    "runtime.hull_cache",
                )?,
                runtime_capability_cell(
                    &benchmark_report,
                    TassadarWorkloadTarget::HungarianMatching,
                    "runtime.sparse_top_k",
                )?,
                TassadarWorkloadCapabilityCell {
                    surface_id: String::from("compiled.proof_backed"),
                    posture: TassadarCapabilityPosture::Exact,
                    artifact_ref: String::from(
                        "fixtures/tassadar/runs/hungarian_v0_compiled_executor_v0/run_bundle.json",
                    ),
                    note: format!(
                        "{}; serve_posture={}",
                        hungarian_compiled_run_bundle["claim_boundary"]
                            .as_str()
                            .unwrap_or("compiled Hungarian boundary missing"),
                        hungarian_compiled_run_bundle["serve_posture"]
                            .as_str()
                            .unwrap_or("unknown")
                    ),
                },
            ],
        },
    ];

    let mut report = TassadarWorkloadCapabilityMatrixReport {
        schema_version: TASSADAR_WORKLOAD_CAPABILITY_MATRIX_SCHEMA_VERSION,
        generated_from_artifacts: vec![
            String::from(TASSADAR_ARTICLE_CLASS_BENCHMARK_REPORT_REF),
            String::from("fixtures/tassadar/runs/sudoku_v0_compiled_executor_v0/run_bundle.json"),
            String::from("fixtures/tassadar/runs/hungarian_v0_compiled_executor_v0/run_bundle.json"),
            String::from("fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_bundle.json"),
            String::from("fixtures/tassadar/runs/sudoku_v0_promotion_v3/promotion_gate_report.json"),
            String::from("fixtures/tassadar/runs/sudoku_v0_promotion_v3/family_report.json"),
            String::from(
                "fixtures/tassadar/runs/sudoku_9x9_v0_reference_run_v0/sequence_fit_report.json",
            ),
        ],
        rows,
        claim_boundary: String::from(
            "this matrix is a workload-family capability view over current committed artifacts only; exact runtime, fallback-only runtime, compiled exact, and bounded/partial learned results remain separated and this report does not imply article parity",
        ),
        report_digest: String::new(),
    };
    report.report_digest =
        stable_serialized_digest(b"tassadar_workload_capability_matrix|", &report);
    Ok(report)
}

/// Returns the canonical absolute path for the workload capability matrix report.
#[must_use]
pub fn tassadar_workload_capability_matrix_report_path() -> std::path::PathBuf {
    eval_repo_root().join(TASSADAR_WORKLOAD_CAPABILITY_MATRIX_REPORT_REF)
}

/// Writes the canonical workload capability matrix report.
pub fn write_tassadar_workload_capability_matrix_report(
    output_path: impl AsRef<std::path::Path>,
) -> Result<TassadarWorkloadCapabilityMatrixReport, TassadarWorkloadCapabilityMatrixError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let report = build_tassadar_workload_capability_matrix_report()?;
    let bytes = serde_json::to_vec_pretty(&report)?;
    std::fs::write(output_path, bytes)?;
    Ok(report)
}

fn build_hull_cache_workload_summary(
    benchmark_report: &TassadarBenchmarkReport,
    workload_target: TassadarWorkloadTarget,
) -> Option<TassadarHullCacheWorkloadSummary> {
    let cases = benchmark_report
        .case_reports
        .iter()
        .filter(|case| case.workload_target == workload_target)
        .collect::<Vec<_>>();
    if cases.is_empty() {
        return None;
    }
    let direct_case_count = cases
        .iter()
        .filter(|case| case.selection_state == TassadarExecutorSelectionState::Direct)
        .count();
    let fallback_case_count = cases.len().saturating_sub(direct_case_count);
    let posture = if fallback_case_count == 0 {
        TassadarCapabilityPosture::Exact
    } else {
        TassadarCapabilityPosture::FallbackOnly
    };
    let average_speedup_over_reference_linear = round_metric(
        cases
            .iter()
            .map(|case| case.hull_cache_speedup_over_reference_linear)
            .sum::<f64>()
            / cases.len() as f64,
    );
    let average_remaining_gap_vs_cpu_reference = round_metric(
        cases
            .iter()
            .map(|case| case.hull_cache_remaining_gap_vs_cpu_reference)
            .sum::<f64>()
            / cases.len() as f64,
    );
    let mut fallback_reason_counts = BTreeMap::new();
    for case in cases.iter().filter_map(|case| case.selection_reason) {
        let key = serde_json::to_string(&case)
            .unwrap_or_else(|_| String::from("\"unknown_selection_reason\""))
            .trim_matches('"')
            .to_string();
        *fallback_reason_counts.entry(key).or_insert(0) += 1;
    }
    Some(TassadarHullCacheWorkloadSummary {
        workload_target,
        posture,
        direct_case_count,
        fallback_case_count,
        average_speedup_over_reference_linear,
        average_remaining_gap_vs_cpu_reference,
        fallback_reason_counts,
        artifact_ref: String::from(TASSADAR_ARTICLE_CLASS_BENCHMARK_REPORT_REF),
    })
}

/// Builds the widened HullCache closure report from the committed article-class
/// benchmark artifact.
pub fn build_tassadar_hull_cache_closure_report(
) -> Result<TassadarHullCacheClosureReport, TassadarHullCacheClosureError> {
    let benchmark_report =
        read_repo_json::<TassadarBenchmarkReport>(TASSADAR_ARTICLE_CLASS_BENCHMARK_REPORT_REF)
            .map_err(|error| match error {
                TassadarWorkloadCapabilityMatrixError::Io(error) => {
                    TassadarHullCacheClosureError::Io(error)
                }
                TassadarWorkloadCapabilityMatrixError::Json(error) => {
                    TassadarHullCacheClosureError::Json(error)
                }
                TassadarWorkloadCapabilityMatrixError::MissingWorkloadTarget { .. } => {
                    unreachable!("benchmark report read cannot surface missing workload-target")
                }
            })?;
    let summaries = [
        TassadarWorkloadTarget::MicroWasmKernel,
        TassadarWorkloadTarget::BranchHeavyKernel,
        TassadarWorkloadTarget::MemoryHeavyKernel,
        TassadarWorkloadTarget::LongLoopKernel,
        TassadarWorkloadTarget::SudokuClass,
        TassadarWorkloadTarget::HungarianMatching,
    ]
    .into_iter()
    .filter_map(|workload_target| {
        build_hull_cache_workload_summary(&benchmark_report, workload_target)
    })
    .collect::<Vec<_>>();
    let (exact_workloads, fallback_only_workloads): (Vec<_>, Vec<_>) = summaries
        .into_iter()
        .partition(|summary| summary.posture == TassadarCapabilityPosture::Exact);
    let mut report = TassadarHullCacheClosureReport {
        schema_version: TASSADAR_HULL_CACHE_CLOSURE_SCHEMA_VERSION,
        exact_workloads,
        fallback_only_workloads,
        claim_boundary: String::from(
            "the widened HullCache closure is now artifact-backed for direct micro, branch-heavy, memory-heavy, and bounded Hungarian workloads, while long-loop and Sudoku search workloads remain explicit fallback-only families under the current control-flow contract",
        ),
        report_digest: String::new(),
    };
    report.report_digest = stable_serialized_digest(b"tassadar_hull_cache_closure|", &report);
    Ok(report)
}

/// Returns the canonical absolute path for the widened HullCache closure report.
#[must_use]
pub fn tassadar_hull_cache_closure_report_path() -> std::path::PathBuf {
    eval_repo_root().join(TASSADAR_HULL_CACHE_CLOSURE_REPORT_REF)
}

/// Writes the canonical widened HullCache closure report.
pub fn write_tassadar_hull_cache_closure_report(
    output_path: impl AsRef<std::path::Path>,
) -> Result<TassadarHullCacheClosureReport, TassadarHullCacheClosureError> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let report = build_tassadar_hull_cache_closure_report()?;
    let bytes = serde_json::to_vec_pretty(&report)?;
    std::fs::write(output_path, bytes)?;
    Ok(report)
}

fn stable_corpus_digest(artifacts: &[TassadarProgramArtifact]) -> String {
    let mut hasher = sha2::Sha256::new();
    hasher.update(b"tassadar_corpus|");
    for artifact in artifacts {
        hasher.update(artifact.artifact_id.as_bytes());
        hasher.update(b"|");
        hasher.update(artifact.artifact_digest.as_bytes());
        hasher.update(b"|");
    }
    hex::encode(hasher.finalize())
}

fn throughput_steps_per_second(steps: u64, elapsed_seconds: f64) -> f64 {
    steps as f64 / elapsed_seconds.max(1e-9)
}

fn round_metric(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

fn benchmark_runner_steps_per_second<F>(
    steps_per_run: u64,
    mut runner: F,
) -> Result<f64, TassadarBenchmarkError>
where
    F: FnMut() -> Result<psionic_runtime::TassadarExecution, TassadarExecutionRefusal>,
{
    let normalized_steps = steps_per_run.max(1);
    let target_steps = normalized_steps.saturating_mul(256).max(8_192);
    let minimum_runs = 16u64;
    let started = Instant::now();
    let mut run_count = 0u64;
    let mut total_steps = 0u64;

    loop {
        runner()?;
        run_count += 1;
        total_steps = total_steps.saturating_add(normalized_steps);
        let elapsed = started.elapsed().as_secs_f64();
        if run_count >= minimum_runs && (total_steps >= target_steps || elapsed >= 0.050) {
            return Ok(throughput_steps_per_second(total_steps, elapsed));
        }
    }
}

fn stable_outputs_digest(outputs: &[i32]) -> String {
    let bytes = serde_json::to_vec(outputs).unwrap_or_default();
    hex::encode(sha2::Sha256::digest(bytes))
}

fn stable_serialized_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = sha2::Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    hex::encode(hasher.finalize())
}

fn build_case_artifacts(
    version: &str,
    case_id: &str,
    artifact: &TassadarProgramArtifact,
    execution: &psionic_runtime::TassadarExecution,
    case: &psionic_runtime::TassadarValidationCase,
    evidence: &psionic_runtime::TassadarExecutionEvidenceBundle,
    selection: &psionic_runtime::TassadarExecutorSelectionDiagnostic,
) -> Result<Vec<EvalArtifact>, TassadarBenchmarkError> {
    Ok(vec![
        EvalArtifact::new(
            "tassadar_program_artifact.json",
            format!("artifact://tassadar/{version}/{case_id}/program"),
            &serde_json::to_vec(artifact).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_trace.json",
            format!("artifact://tassadar/{version}/{case_id}/trace"),
            &serde_json::to_vec(&execution.steps).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_expected_trace.json",
            format!("artifact://tassadar/{version}/{case_id}/expected_trace"),
            &serde_json::to_vec(&case.expected_trace).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_runtime_manifest.json",
            format!("artifact://tassadar/{version}/{case_id}/runtime_manifest"),
            &serde_json::to_vec(&evidence.runtime_manifest).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_trace_proof.json",
            format!("artifact://tassadar/{version}/{case_id}/trace_proof"),
            &serde_json::to_vec(&evidence.trace_proof).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_execution_proof_bundle.json",
            format!("artifact://tassadar/{version}/{case_id}/execution_proof_bundle"),
            &serde_json::to_vec(&evidence.proof_bundle).unwrap_or_default(),
        ),
        EvalArtifact::new(
            "tassadar_selection_diagnostic.json",
            format!("artifact://tassadar/{version}/{case_id}/selection_diagnostic"),
            &serde_json::to_vec(selection).unwrap_or_default(),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tassadar_reference_fixture_suite_builds_package_and_environment_contracts(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let suite = build_tassadar_reference_fixture_suite("2026.03.15")?;
        assert_eq!(suite.artifacts.len(), 3);
        assert_eq!(suite.benchmark_package.cases.len(), 3);
        assert_eq!(
            suite
                .benchmark_package
                .metadata
                .get("tassadar.hull_cache_metric_id")
                .and_then(Value::as_str),
            Some(TASSADAR_HULL_CACHE_METRIC_ID)
        );
        assert_eq!(
            suite
                .benchmark_package
                .metadata
                .get("tassadar.sparse_top_k_metric_id")
                .and_then(Value::as_str),
            Some(TASSADAR_SPARSE_TOP_K_METRIC_ID)
        );
        assert_eq!(
            suite.environment_bundle.current_workload_targets,
            vec![
                TassadarWorkloadTarget::ArithmeticMicroprogram,
                TassadarWorkloadTarget::MemoryLookupMicroprogram,
                TassadarWorkloadTarget::BranchControlFlowMicroprogram,
            ]
        );
        Ok(())
    }

    #[test]
    fn tassadar_reference_fixture_benchmark_is_exact_on_current_validation_corpus(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let report = run_tassadar_reference_fixture_benchmark("2026.03.15")?;
        assert_eq!(report.aggregate_summary.round_count, 1);
        assert_eq!(report.aggregate_summary.aggregate_score_bps, Some(10_000));
        assert_eq!(report.aggregate_summary.aggregate_pass_rate_bps, 10_000);
        assert_eq!(report.eval_run.status, crate::EvalRunStatus::Finalized);
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.status == EvalSampleStatus::Passed));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.cpu_reference_steps_per_second > 0.0));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.reference_linear_steps_per_second > 0.0));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.hull_cache_steps_per_second > 0.0));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.sparse_top_k_steps_per_second > 0.0));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.hull_cache_speedup_over_reference_linear > 1.0));
        assert!(report.case_reports.iter().all(|case| {
            case.sparse_top_k_selection_state == TassadarExecutorSelectionState::Direct
                && case.sparse_top_k_selection_reason.is_none()
                && !case.sparse_top_k_used_decode_fallback
                && case.sparse_top_k_speedup_over_reference_linear > 0.0
        }));
        assert!(report.case_reports.iter().all(|case| {
            case.requested_decode_mode == TassadarExecutorDecodeMode::HullCache
                && case.effective_decode_mode == TassadarExecutorDecodeMode::HullCache
                && case.selection_state == TassadarExecutorSelectionState::Direct
                && case.selection_reason.is_none()
                && !case.used_decode_fallback
        }));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.trace_digest_equal));
        assert!(report.case_reports.iter().all(|case| case.outputs_equal));
        assert!(report.case_reports.iter().all(|case| case.halt_equal));
        assert!(report.eval_run.samples.iter().all(|sample| {
            sample
                .artifacts
                .iter()
                .any(|artifact| artifact.artifact_kind == "tassadar_trace_proof.json")
        }));
        assert!(report.eval_run.samples.iter().all(|sample| {
            sample
                .artifacts
                .iter()
                .any(|artifact| artifact.artifact_kind == "tassadar_runtime_manifest.json")
        }));
        assert!(report.eval_run.samples.iter().all(|sample| {
            sample
                .artifacts
                .iter()
                .any(|artifact| artifact.artifact_kind == "tassadar_execution_proof_bundle.json")
        }));
        assert!(report.eval_run.samples.iter().all(|sample| {
            sample
                .artifacts
                .iter()
                .any(|artifact| artifact.artifact_kind == "tassadar_selection_diagnostic.json")
        }));
        assert!(report
            .eval_run
            .run_artifacts
            .iter()
            .any(|artifact| { artifact.artifact_kind == "tassadar_runtime_capability.json" }));
        Ok(())
    }

    #[test]
    fn tassadar_article_class_suite_builds_package_and_environment_contracts(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let suite = build_tassadar_article_class_suite("2026.03.16")?;
        let sudoku_v0_corpus = tassadar_sudoku_v0_corpus();
        assert_eq!(suite.artifacts.len(), sudoku_v0_corpus.len() + 5);
        assert_eq!(
            suite.benchmark_package.cases.len(),
            sudoku_v0_corpus.len() + 5
        );
        assert_eq!(
            suite.environment_bundle.current_workload_targets,
            vec![
                TassadarWorkloadTarget::MicroWasmKernel,
                TassadarWorkloadTarget::BranchHeavyKernel,
                TassadarWorkloadTarget::MemoryHeavyKernel,
                TassadarWorkloadTarget::LongLoopKernel,
                TassadarWorkloadTarget::SudokuClass,
                TassadarWorkloadTarget::HungarianMatching,
            ]
        );
        assert!(
            suite.environment_bundle.planned_workload_targets.is_empty(),
            "article-class targets should be current rather than planned"
        );
        assert_eq!(
            suite.environment_bundle.program_binding.wasm_profile_id,
            TassadarWasmProfile::article_i32_compute_v1().profile_id
        );
        assert_eq!(
            suite.benchmark_package.metadata["tassadar.sudoku_v0_train_case_ids"]
                .as_array()
                .map_or(0, Vec::len),
            4
        );
        assert_eq!(
            suite.benchmark_package.metadata["tassadar.sudoku_v0_validation_case_ids"]
                .as_array()
                .map_or(0, Vec::len),
            2
        );
        assert_eq!(
            suite.benchmark_package.metadata["tassadar.sudoku_v0_test_case_ids"]
                .as_array()
                .map_or(0, Vec::len),
            2
        );
        Ok(())
    }

    #[test]
    fn tassadar_article_class_benchmark_is_exact_on_widened_corpus(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let report = run_tassadar_article_class_benchmark("2026.03.16")?;
        assert_eq!(report.aggregate_summary.round_count, 1);
        assert_eq!(report.aggregate_summary.aggregate_score_bps, Some(10_000));
        assert_eq!(report.aggregate_summary.aggregate_pass_rate_bps, 10_000);
        assert_eq!(report.eval_run.status, crate::EvalRunStatus::Finalized);
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.status == EvalSampleStatus::Passed));
        assert_eq!(
            report
                .case_reports
                .iter()
                .filter(|case| case.workload_target == TassadarWorkloadTarget::SudokuClass)
                .count(),
            8
        );
        assert_eq!(
            report
                .case_reports
                .iter()
                .filter(|case| case.workload_target == TassadarWorkloadTarget::MicroWasmKernel)
                .count(),
            1
        );
        assert_eq!(
            report
                .case_reports
                .iter()
                .filter(|case| case.workload_target == TassadarWorkloadTarget::BranchHeavyKernel)
                .count(),
            1
        );
        assert_eq!(
            report
                .case_reports
                .iter()
                .filter(|case| case.workload_target == TassadarWorkloadTarget::MemoryHeavyKernel)
                .count(),
            1
        );
        assert_eq!(
            report
                .case_reports
                .iter()
                .filter(|case| case.workload_target == TassadarWorkloadTarget::LongLoopKernel)
                .count(),
            1
        );
        assert_eq!(
            report
                .case_reports
                .iter()
                .filter(|case| case.workload_target == TassadarWorkloadTarget::HungarianMatching)
                .count(),
            1
        );
        for case in &report.case_reports {
            assert_eq!(
                case.requested_decode_mode,
                TassadarExecutorDecodeMode::HullCache
            );
            match case.workload_target {
                TassadarWorkloadTarget::SudokuClass => {
                    assert_eq!(
                        case.effective_decode_mode,
                        TassadarExecutorDecodeMode::ReferenceLinear
                    );
                    assert_eq!(
                        case.selection_state,
                        TassadarExecutorSelectionState::Fallback
                    );
                    assert_eq!(
                        case.selection_reason,
                        Some(TassadarExecutorSelectionReason::HullCacheControlFlowUnsupported)
                    );
                    assert!(case.used_decode_fallback);
                    assert_eq!(
                        case.sparse_top_k_selection_state,
                        TassadarExecutorSelectionState::Fallback
                    );
                    assert_eq!(
                        case.sparse_top_k_selection_reason,
                        Some(TassadarExecutorSelectionReason::SparseTopKValidationUnsupported)
                    );
                    assert!(case.sparse_top_k_used_decode_fallback);
                }
                TassadarWorkloadTarget::LongLoopKernel => {
                    assert_eq!(
                        case.effective_decode_mode,
                        TassadarExecutorDecodeMode::ReferenceLinear
                    );
                    assert_eq!(
                        case.selection_state,
                        TassadarExecutorSelectionState::Fallback
                    );
                    assert_eq!(
                        case.selection_reason,
                        Some(TassadarExecutorSelectionReason::HullCacheControlFlowUnsupported)
                    );
                    assert!(case.used_decode_fallback);
                    assert_eq!(
                        case.sparse_top_k_selection_state,
                        TassadarExecutorSelectionState::Fallback
                    );
                    assert_eq!(
                        case.sparse_top_k_selection_reason,
                        Some(TassadarExecutorSelectionReason::SparseTopKValidationUnsupported)
                    );
                    assert!(case.sparse_top_k_used_decode_fallback);
                }
                TassadarWorkloadTarget::BranchHeavyKernel => {
                    assert_eq!(
                        case.effective_decode_mode,
                        TassadarExecutorDecodeMode::HullCache
                    );
                    assert_eq!(case.selection_state, TassadarExecutorSelectionState::Direct);
                    assert_eq!(case.selection_reason, None);
                    assert!(!case.used_decode_fallback);
                    assert_eq!(
                        case.sparse_top_k_selection_state,
                        TassadarExecutorSelectionState::Fallback
                    );
                    assert_eq!(
                        case.sparse_top_k_selection_reason,
                        Some(TassadarExecutorSelectionReason::SparseTopKValidationUnsupported)
                    );
                    assert!(case.sparse_top_k_used_decode_fallback);
                }
                TassadarWorkloadTarget::MicroWasmKernel
                | TassadarWorkloadTarget::MemoryHeavyKernel
                | TassadarWorkloadTarget::HungarianMatching => {
                    assert_eq!(
                        case.effective_decode_mode,
                        TassadarExecutorDecodeMode::HullCache
                    );
                    assert_eq!(case.selection_state, TassadarExecutorSelectionState::Direct);
                    assert_eq!(case.selection_reason, None);
                    assert!(!case.used_decode_fallback);
                    assert_eq!(
                        case.sparse_top_k_selection_state,
                        TassadarExecutorSelectionState::Direct
                    );
                    assert_eq!(case.sparse_top_k_selection_reason, None);
                    assert!(!case.sparse_top_k_used_decode_fallback);
                }
                _ => {}
            }
        }
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.trace_digest_equal));
        assert!(report.case_reports.iter().all(|case| case.outputs_equal));
        assert!(report.case_reports.iter().all(|case| case.halt_equal));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.cpu_reference_steps_per_second > 0.0));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.reference_linear_steps_per_second > 0.0));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.hull_cache_steps_per_second > 0.0));
        assert!(report
            .case_reports
            .iter()
            .all(|case| case.sparse_top_k_steps_per_second > 0.0));
        Ok(())
    }

    #[test]
    fn tassadar_workload_capability_matrix_maps_current_families(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let report = build_tassadar_workload_capability_matrix_report()?;
        assert_eq!(
            report.schema_version,
            TASSADAR_WORKLOAD_CAPABILITY_MATRIX_SCHEMA_VERSION
        );
        assert!(report.rows.iter().any(|row| {
            row.workload_family_id == "micro_wasm_kernel"
                && row.capabilities.iter().any(|cell| {
                    cell.surface_id == "runtime.hull_cache"
                        && cell.posture == TassadarCapabilityPosture::Exact
                })
        }));
        assert!(report.rows.iter().any(|row| {
            row.workload_family_id == "branch_heavy_kernel"
                && row.capabilities.iter().any(|cell| {
                    cell.surface_id == "runtime.sparse_top_k"
                        && cell.posture == TassadarCapabilityPosture::FallbackOnly
                })
        }));
        assert!(report.rows.iter().any(|row| {
            row.workload_family_id == "sudoku_class_4x4"
                && row.capabilities.iter().any(|cell| {
                    cell.surface_id == "compiled.proof_backed"
                        && cell.posture == TassadarCapabilityPosture::Exact
                })
                && row.capabilities.iter().any(|cell| {
                    cell.surface_id == "learned.bounded"
                        && cell.posture == TassadarCapabilityPosture::Partial
                })
        }));
        assert!(report.rows.iter().any(|row| {
            row.workload_family_id == "sudoku_search_9x9"
                && row.capabilities.iter().any(|cell| {
                    cell.surface_id == "learned.long_horizon"
                        && cell.posture == TassadarCapabilityPosture::Partial
                })
        }));
        assert!(report.rows.iter().any(|row| {
            row.workload_family_id == "hungarian_matching_4x4"
                && row.capabilities.iter().any(|cell| {
                    cell.surface_id == "compiled.proof_backed"
                        && cell.posture == TassadarCapabilityPosture::Exact
                })
        }));
        Ok(())
    }

    #[test]
    fn tassadar_workload_capability_matrix_report_matches_committed_truth(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let report = build_tassadar_workload_capability_matrix_report()?;
        let bytes = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join(TASSADAR_WORKLOAD_CAPABILITY_MATRIX_REPORT_REF),
        )?;
        let persisted: TassadarWorkloadCapabilityMatrixReport = serde_json::from_slice(&bytes)?;
        assert_eq!(persisted, report);
        Ok(())
    }

    #[test]
    fn write_tassadar_workload_capability_matrix_report_persists_current_truth(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = std::env::temp_dir().join(format!(
            "psionic-tassadar-eval-capability-matrix-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir)?;
        let report_path = temp_dir.join("tassadar_workload_capability_matrix.json");
        let report = write_tassadar_workload_capability_matrix_report(&report_path)?;
        let bytes = std::fs::read(&report_path)?;
        let persisted: TassadarWorkloadCapabilityMatrixReport = serde_json::from_slice(&bytes)?;
        assert_eq!(persisted, report);
        std::fs::remove_file(&report_path)?;
        std::fs::remove_dir(&temp_dir)?;
        Ok(())
    }

    #[test]
    fn tassadar_hull_cache_closure_report_tracks_widened_direct_subset(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let report = build_tassadar_hull_cache_closure_report()?;
        assert!(report.exact_workloads.iter().any(|row| {
            row.workload_target == TassadarWorkloadTarget::MicroWasmKernel
                && row.direct_case_count == 1
                && row.fallback_case_count == 0
        }));
        assert!(report.exact_workloads.iter().any(|row| {
            row.workload_target == TassadarWorkloadTarget::BranchHeavyKernel
                && row.direct_case_count == 1
                && row.fallback_case_count == 0
        }));
        assert!(report.exact_workloads.iter().any(|row| {
            row.workload_target == TassadarWorkloadTarget::MemoryHeavyKernel
                && row.direct_case_count == 1
                && row.fallback_case_count == 0
        }));
        assert!(report.exact_workloads.iter().any(|row| {
            row.workload_target == TassadarWorkloadTarget::HungarianMatching
                && row.direct_case_count == 1
                && row.fallback_case_count == 0
        }));
        assert!(report.fallback_only_workloads.iter().any(|row| {
            row.workload_target == TassadarWorkloadTarget::LongLoopKernel
                && row.fallback_case_count == 1
        }));
        assert!(report.fallback_only_workloads.iter().any(|row| {
            row.workload_target == TassadarWorkloadTarget::SudokuClass
                && row.fallback_case_count == 8
        }));
        Ok(())
    }

    #[test]
    fn tassadar_hull_cache_closure_report_matches_committed_truth(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let report = build_tassadar_hull_cache_closure_report()?;
        let bytes = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join(TASSADAR_HULL_CACHE_CLOSURE_REPORT_REF),
        )?;
        let persisted: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(persisted, serde_json::to_value(&report)?);
        Ok(())
    }

    #[test]
    fn write_tassadar_hull_cache_closure_report_persists_current_truth(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = std::env::temp_dir().join(format!(
            "psionic-tassadar-eval-hull-cache-closure-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&temp_dir)?;
        let report_path = temp_dir.join("tassadar_hull_cache_closure_report.json");
        let report = write_tassadar_hull_cache_closure_report(&report_path)?;
        let bytes = std::fs::read(&report_path)?;
        let persisted: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(persisted, serde_json::to_value(&report)?);
        std::fs::remove_file(&report_path)?;
        std::fs::remove_dir(&temp_dir)?;
        Ok(())
    }
}
