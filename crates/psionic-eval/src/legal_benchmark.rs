//! Legal-agent benchmark schema contracts.
//!
//! These types are the Psionic-owned task, artifact, run, transcript, and
//! scoring contracts for Harvey-compatible legal-agent benchmark work. The
//! structs intentionally keep upstream compatibility fields separate from the
//! immutable hashes and receipts Psionic and Autopilot consume.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Current schema version for the owned legal benchmark contracts.
pub const LEGAL_BENCHMARK_SCHEMA_VERSION: u16 = 1;

/// Stable metadata map used across benchmark contracts.
pub type Metadata = BTreeMap<String, Value>;

/// Source compatibility information for a task imported from another suite.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceCompatibility {
    /// Upstream suite name, for example `harvey_labs`.
    pub upstream_suite: String,
    /// Upstream repository commit or immutable release id.
    pub upstream_commit: String,
    /// Upstream task path relative to the suite root.
    pub upstream_task_path: String,
    /// Raw upstream ids or labels that should remain non-authoritative.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub upstream_fields: Metadata,
}

/// Benchmark task specification normalized into Psionic-owned shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkTaskSpec {
    /// Schema version for this task spec.
    pub schema_version: u16,
    /// Stable task id in the owned benchmark namespace.
    pub task_id: String,
    /// Immutable task version.
    pub task_version: String,
    /// Domain or corpus family, for example `legal`.
    pub domain: String,
    /// Practice area or benchmark lane.
    pub practice_area: String,
    /// Workflow class such as analyze, draft, review, or research.
    pub workflow: String,
    /// Human-readable title.
    pub title: String,
    /// Task instructions visible to the agent.
    pub instructions: String,
    /// Work type or original benchmark category.
    pub work_type: String,
    /// Search and reporting tags.
    pub tags: Vec<String>,
    /// Source documents and other immutable inputs.
    pub source_artifacts: Vec<SourceArtifact>,
    /// Required deliverables.
    pub deliverables: Vec<DeliverableSpec>,
    /// Scoring criteria.
    pub criteria: Vec<CriterionSpec>,
    /// Judge policy attached to this task.
    pub judge_policy: JudgePolicy,
    /// Tool policy attached to this task.
    pub tool_policy: ToolPolicy,
    /// Compatibility information when imported from an upstream suite.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_compatibility: Option<SourceCompatibility>,
    /// Additional owned metadata.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

/// Immutable manifest over source or output artifacts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    /// Schema version for this artifact manifest.
    pub schema_version: u16,
    /// Stable manifest id.
    pub manifest_id: String,
    /// Task id this manifest belongs to.
    pub task_id: String,
    /// Task version this manifest belongs to.
    pub task_version: String,
    /// Whether this manifest describes inputs, outputs, or derived artifacts.
    pub manifest_role: ArtifactManifestRole,
    /// Artifacts included in the manifest.
    pub artifacts: Vec<SourceArtifact>,
    /// Additional owned metadata.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

/// Artifact manifest role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactManifestRole {
    /// Agent-visible input artifacts.
    Input,
    /// Agent-generated output artifacts.
    Output,
    /// Derived extraction or intermediate artifacts.
    Derived,
}

/// One source or generated artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceArtifact {
    /// Stable artifact id within a task or run.
    pub artifact_id: String,
    /// Artifact kind.
    pub artifact_kind: ArtifactKind,
    /// Path relative to the task, run, or manifest root.
    pub relative_path: String,
    /// Original file name if different from the relative path leaf.
    pub original_filename: String,
    /// Media type such as `application/pdf`.
    pub media_type: String,
    /// Byte length.
    pub byte_size: u64,
    /// SHA-256 digest of the artifact bytes.
    pub sha256: String,
    /// Data classification for execution and reporting.
    pub data_classification: DataClassification,
    /// Optional provenance string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<String>,
}

/// Artifact kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Original benchmark source document.
    SourceDocument,
    /// Extracted text or structured representation.
    ExtractedText,
    /// Agent-generated deliverable.
    GeneratedDeliverable,
    /// Agent transcript.
    Transcript,
    /// Score report.
    ScoreReport,
    /// Comparison report.
    ComparisonReport,
    /// Scratch or intermediate artifact.
    Scratch,
}

/// Data classification for benchmark artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataClassification {
    /// Public reference corpus data.
    PublicReference,
    /// Benchmark-confidential data.
    BenchmarkConfidential,
    /// User-confidential data.
    UserConfidential,
    /// Restricted data.
    Restricted,
    /// Unknown classification.
    Unknown,
}

/// Required deliverable specification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliverableSpec {
    /// Stable deliverable id.
    pub deliverable_id: String,
    /// Deliverable kind.
    pub deliverable_kind: DeliverableKind,
    /// Required relative output path or path pattern.
    pub required_path: String,
    /// Human-readable description.
    pub description: String,
    /// Whether the deliverable must exist for scoring.
    pub required: bool,
}

/// Deliverable kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliverableKind {
    /// Plain text deliverable.
    Text,
    /// Markdown deliverable.
    Markdown,
    /// Word-processing document.
    Docx,
    /// Spreadsheet.
    Xlsx,
    /// PDF deliverable.
    Pdf,
    /// JSON deliverable.
    Json,
    /// Directory of output files.
    Directory,
    /// Other deliverable type.
    Other,
}

/// One scoring criterion.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriterionSpec {
    /// Stable criterion id.
    pub criterion_id: String,
    /// Criterion kind.
    pub criterion_kind: CriterionKind,
    /// Criterion text.
    pub description: String,
    /// Weight in basis points, if the suite uses weighted scoring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_bps: Option<u32>,
    /// Deliverables this criterion applies to.
    pub deliverable_ids: Vec<String>,
    /// Source artifacts this criterion expects to be considered.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_artifact_ids: Vec<String>,
}

/// Criterion family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriterionKind {
    /// Completeness requirement.
    Completeness,
    /// Legal reasoning requirement.
    LegalReasoning,
    /// Factual accuracy requirement.
    FactualAccuracy,
    /// Citation or evidence requirement.
    CitationEvidence,
    /// Output formatting requirement.
    Formatting,
    /// Deliverable presence or readability requirement.
    DeliverableValidation,
    /// Other criterion kind.
    Other,
}

/// Judge policy for scoring.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgePolicy {
    /// Judge mode.
    pub mode: JudgeMode,
    /// Provider or local judge route.
    pub provider: String,
    /// Judge model id.
    pub model: String,
    /// Stable judge prompt template id.
    pub prompt_template_id: String,
    /// Stable hash of the prompt template.
    pub prompt_template_hash: String,
    /// Whether all criteria must pass for task success.
    pub all_pass_required: bool,
    /// Number of judge samples.
    pub sample_count: u16,
}

/// Judge mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeMode {
    /// Deterministic evaluator only.
    Deterministic,
    /// Single LLM judge.
    Llm,
    /// Multiple LLM judges.
    MultiJudge,
    /// Human reviewer.
    Human,
}

/// Tool policy for agent execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicy {
    /// Allowed benchmark tools.
    pub allowed_tools: Vec<String>,
    /// Whether network access is allowed.
    pub network_allowed: bool,
    /// Whether source artifacts must be mounted read-only.
    pub source_artifacts_read_only: bool,
    /// Maximum model-tool turns.
    pub max_turns: u32,
    /// Maximum wall time in seconds.
    pub max_wall_time_seconds: u64,
}

/// Run configuration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunConfig {
    /// Schema version for this run config.
    pub schema_version: u16,
    /// Stable run config id.
    pub run_config_id: String,
    /// Provider route.
    pub provider: String,
    /// Model id.
    pub model: String,
    /// Agent protocol or module version id.
    pub agent_protocol_version: String,
    /// Tool policy for this run.
    pub tool_policy: ToolPolicy,
    /// Judge policy for this run.
    pub judge_policy: JudgePolicy,
    /// Optional random seed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
    /// Additional owned metadata.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

/// Complete record for one benchmark run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    /// Schema version for this run record.
    pub schema_version: u16,
    /// Stable run id.
    pub run_id: String,
    /// Task id.
    pub task_id: String,
    /// Task version.
    pub task_version: String,
    /// SHA-256 hash of the input artifact manifest.
    pub input_artifact_manifest_hash: String,
    /// SHA-256 hash of the run config.
    pub run_config_hash: String,
    /// SHA-256 hash of the output artifact manifest.
    pub output_artifact_manifest_hash: String,
    /// Terminal state for the run.
    pub terminal_state: RunTerminalState,
    /// Ordered transcript events.
    pub transcript: Vec<TranscriptEvent>,
    /// Tool calls extracted from the transcript.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Aggregate run metrics.
    pub metrics: RunMetrics,
    /// Additional owned metadata.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

/// Run terminal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunTerminalState {
    /// Agent submitted outputs.
    Submitted,
    /// Agent stopped without tool calls.
    NoToolCalls,
    /// Maximum turn count reached.
    MaxTurns,
    /// Context limit reached.
    ContextOverflow,
    /// Provider failure ended the run.
    ProviderFailure,
    /// Sandbox failure ended the run.
    SandboxFailure,
    /// Policy failure ended the run.
    PolicyFailure,
    /// Internal runner error ended the run.
    InternalError,
}

/// One transcript event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TranscriptEvent {
    /// Monotonic event index.
    pub event_index: u64,
    /// Event kind.
    pub event_kind: TranscriptEventKind,
    /// Optional role string such as `system`, `user`, `assistant`, or `tool`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Text content when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Structured JSON payload when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Value>,
    /// Timestamp in milliseconds from Unix epoch.
    pub timestamp_ms: u64,
}

/// Transcript event kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptEventKind {
    /// System prompt or policy event.
    System,
    /// User task message.
    User,
    /// Model assistant message.
    Assistant,
    /// Tool call emitted by the model.
    ToolCall,
    /// Tool result returned to the model.
    ToolResult,
    /// Runner event.
    Runner,
    /// Judge event.
    Judge,
}

/// One tool call record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Tool call id.
    pub tool_call_id: String,
    /// Tool name.
    pub tool_name: String,
    /// Event index of the call.
    pub call_event_index: u64,
    /// Event index of the result, when one exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_event_index: Option<u64>,
    /// Tool input payload.
    pub input: Value,
    /// Tool output payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Structured error kind when the call failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: u64,
}

/// Aggregate run metrics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunMetrics {
    /// Number of model turns.
    pub model_turns: u32,
    /// Number of tool calls.
    pub tool_call_count: u32,
    /// Prompt/input tokens.
    pub input_tokens: u64,
    /// Completion/output tokens.
    pub output_tokens: u64,
    /// Total wall time in milliseconds.
    pub wall_time_ms: u64,
    /// Estimated cost in microdollars.
    pub estimated_cost_micro_usd: u64,
}

/// Result for one criterion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CriterionResult {
    /// Criterion id.
    pub criterion_id: String,
    /// Whether the criterion passed.
    pub passed: bool,
    /// Verdict family.
    pub verdict: CriterionVerdict,
    /// Judge or deterministic reasoning summary.
    pub reasoning: String,
    /// Referenced output or source evidence ids.
    pub evidence_refs: Vec<String>,
    /// Judge model used for this result.
    pub judge_model: String,
    /// Judge prompt hash.
    pub judge_prompt_hash: String,
    /// Raw judge response hash.
    pub raw_response_hash: String,
}

/// Criterion verdict family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriterionVerdict {
    /// Criterion passed.
    Pass,
    /// Criterion failed.
    Fail,
    /// Judge result was ambiguous.
    Ambiguous,
    /// Criterion was not evaluated.
    NotEvaluated,
}

/// Score report for one run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreReport {
    /// Schema version for this score report.
    pub schema_version: u16,
    /// Stable score report id.
    pub score_report_id: String,
    /// Run id.
    pub run_id: String,
    /// Task id.
    pub task_id: String,
    /// Task version.
    pub task_version: String,
    /// Run record hash.
    pub run_record_hash: String,
    /// Output artifact manifest hash.
    pub output_artifact_manifest_hash: String,
    /// Whether every criterion passed.
    pub all_pass: bool,
    /// Criterion pass rate in basis points.
    pub criterion_pass_rate_bps: u32,
    /// Criterion results.
    pub criterion_results: Vec<CriterionResult>,
    /// Run metrics copied into the report.
    pub metrics: RunMetrics,
    /// Additional owned metadata.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

/// Comparison between two or more score reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Schema version for this comparison report.
    pub schema_version: u16,
    /// Stable comparison report id.
    pub comparison_report_id: String,
    /// Suite or campaign id.
    pub comparison_scope: String,
    /// Baseline score report hash.
    pub baseline_score_report_hash: String,
    /// Candidate score report hash.
    pub candidate_score_report_hash: String,
    /// All-pass delta in basis points.
    pub all_pass_delta_bps: i32,
    /// Criterion pass-rate delta in basis points.
    pub criterion_pass_rate_delta_bps: i32,
    /// Cost delta in microdollars.
    pub estimated_cost_delta_micro_usd: i64,
    /// Latency delta in milliseconds.
    pub wall_time_delta_ms: i64,
    /// Additional owned metadata.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

/// Computes a stable SHA-256 digest for a serializable value under a namespace.
pub fn stable_json_digest<T>(namespace: &str, value: &T) -> Result<String, serde_json::Error>
where
    T: Serialize,
{
    let mut hasher = Sha256::new();
    hasher.update(namespace.as_bytes());
    hasher.update(b"|");
    hasher.update(serde_json::to_vec(value)?);
    Ok(hex::encode(hasher.finalize()))
}

/// Computes a stable task-spec digest.
pub fn task_spec_digest(task_spec: &BenchmarkTaskSpec) -> Result<String, serde_json::Error> {
    stable_json_digest("psionic.legal_benchmark.task_spec.v1", task_spec)
}

/// Computes a stable artifact-manifest digest.
pub fn artifact_manifest_digest(
    artifact_manifest: &ArtifactManifest,
) -> Result<String, serde_json::Error> {
    stable_json_digest(
        "psionic.legal_benchmark.artifact_manifest.v1",
        artifact_manifest,
    )
}

/// Computes a stable run-config digest.
pub fn run_config_digest(run_config: &RunConfig) -> Result<String, serde_json::Error> {
    stable_json_digest("psionic.legal_benchmark.run_config.v1", run_config)
}

/// Computes a stable run-record digest.
pub fn run_record_digest(run_record: &RunRecord) -> Result<String, serde_json::Error> {
    stable_json_digest("psionic.legal_benchmark.run_record.v1", run_record)
}

/// Computes a stable transcript digest.
pub fn transcript_digest(transcript: &[TranscriptEvent]) -> Result<String, serde_json::Error> {
    stable_json_digest("psionic.legal_benchmark.transcript.v1", &transcript)
}

/// Computes a stable score-report digest.
pub fn score_report_digest(score_report: &ScoreReport) -> Result<String, serde_json::Error> {
    stable_json_digest("psionic.legal_benchmark.score_report.v1", score_report)
}

/// Computes a stable comparison-report digest.
pub fn comparison_report_digest(
    comparison_report: &ComparisonReport,
) -> Result<String, serde_json::Error> {
    stable_json_digest(
        "psionic.legal_benchmark.comparison_report.v1",
        comparison_report,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    struct MinimalTaskBundle {
        task_spec: BenchmarkTaskSpec,
        input_manifest: ArtifactManifest,
        output_manifest: ArtifactManifest,
        run_config: RunConfig,
        run_record: RunRecord,
        score_report: ScoreReport,
        comparison_report: ComparisonReport,
    }

    fn fixture_bundle() -> MinimalTaskBundle {
        serde_json::from_str(include_str!(
            "../../../fixtures/legal_benchmark/minimal_task_bundle.json"
        ))
        .expect("minimal legal benchmark fixture parses")
    }

    #[test]
    fn minimal_fixture_round_trips_all_core_schemas() {
        let fixture = fixture_bundle();
        let encoded = serde_json::to_string_pretty(&fixture).expect("fixture serializes");
        let decoded: MinimalTaskBundle =
            serde_json::from_str(&encoded).expect("fixture deserializes");
        assert_eq!(fixture, decoded);
    }

    #[test]
    fn stable_hash_helpers_are_deterministic() {
        let fixture = fixture_bundle();
        let task_digest_a = task_spec_digest(&fixture.task_spec).expect("task digest");
        let task_digest_b = task_spec_digest(&fixture.task_spec).expect("task digest repeat");
        assert_eq!(task_digest_a, task_digest_b);
        assert_eq!(task_digest_a.len(), 64);

        let input_digest_a =
            artifact_manifest_digest(&fixture.input_manifest).expect("input manifest digest");
        let input_digest_b =
            artifact_manifest_digest(&fixture.input_manifest).expect("input manifest repeat");
        assert_eq!(input_digest_a, input_digest_b);
        assert_eq!(input_digest_a.len(), 64);

        let output_digest =
            artifact_manifest_digest(&fixture.output_manifest).expect("output manifest digest");
        let run_config_hash = run_config_digest(&fixture.run_config).expect("run config digest");
        let run_record_hash = run_record_digest(&fixture.run_record).expect("run record digest");
        let transcript_hash =
            transcript_digest(&fixture.run_record.transcript).expect("transcript digest");
        let score_hash = score_report_digest(&fixture.score_report).expect("score digest");
        let comparison_hash =
            comparison_report_digest(&fixture.comparison_report).expect("comparison digest");

        assert_eq!(output_digest.len(), 64);
        assert_eq!(run_config_hash.len(), 64);
        assert_eq!(run_record_hash.len(), 64);
        assert_eq!(transcript_hash.len(), 64);
        assert_eq!(score_hash.len(), 64);
        assert_eq!(comparison_hash.len(), 64);
    }

    #[test]
    fn run_record_requires_identity_and_manifest_fields() {
        let missing_required = serde_json::json!({
            "schema_version": LEGAL_BENCHMARK_SCHEMA_VERSION,
            "run_id": "run.missing-fields",
            "terminal_state": "submitted",
            "transcript": [],
            "tool_calls": [],
            "metrics": {
                "model_turns": 0,
                "tool_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "wall_time_ms": 0,
                "estimated_cost_micro_usd": 0
            }
        });

        let error = serde_json::from_value::<RunRecord>(missing_required)
            .expect_err("required run record fields are enforced");
        let message = error.to_string();
        assert!(
            message.contains("task_id")
                || message.contains("task_version")
                || message.contains("input_artifact_manifest_hash")
                || message.contains("run_config_hash")
                || message.contains("output_artifact_manifest_hash")
        );
    }

    #[test]
    fn task_spec_requires_task_identity() {
        let error = serde_json::from_value::<BenchmarkTaskSpec>(serde_json::json!({
            "schema_version": LEGAL_BENCHMARK_SCHEMA_VERSION
        }))
        .expect_err("required task spec fields are enforced");
        let message = error.to_string();
        assert!(message.contains("task_id"));
    }
}
