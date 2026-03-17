//! Bounded MLX-style benchmark package and provider adapters above
//! `psionic-eval`.

#![cfg_attr(
    test,
    allow(clippy::expect_used, clippy::panic, clippy::panic_in_result_fn)
)]

use std::collections::{BTreeMap, BTreeSet};

use psionic_data::DatasetKey;
use psionic_environments::EnvironmentPackageKey;
use psionic_eval::{
    BenchmarkAggregateSummary, BenchmarkAggregationKind, BenchmarkCase, BenchmarkExecutionMode,
    BenchmarkPackage, BenchmarkPackageKey, BenchmarkVerificationPolicy, EvalArtifact,
    EvalExecutionStrategyFacts, EvalFinalStateCapture, EvalMetric, EvalRunContract, EvalRunMode,
    EvalRunState, EvalRuntimeError, EvalSampleRecord, EvalSampleStatus, EvalTimerIntegrityFacts,
    EvalTokenAccountingFacts, EvalVerificationFacts,
};
use psionic_mlx_lm::MlxLmTextRequest;
use psionic_mlx_vlm::{
    MlxVlmError, MlxVlmMessage, MlxVlmProjectionReport, MlxVlmServePlan, MlxVlmServedEndpoint,
    MlxVlmWorkspace,
};
use psionic_models::{PromptMessage, PromptMessageRole};
use psionic_serve::{GenerationOptions, GenerationResponse};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str =
    "bounded MLX-style benchmark package and provider adapters above psionic-eval";

/// Public provider family supported by the benchmark package.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxBenchmarkProviderKind {
    /// Local text-generation path above `psionic-mlx-lm`.
    LocalText,
    /// Served text path above `psionic-mlx-serve`.
    ServedText,
}

/// One machine-readable provider descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxBenchmarkProviderDescriptor {
    /// Stable provider identifier.
    pub provider_id: String,
    /// High-level provider family.
    pub provider_kind: MlxBenchmarkProviderKind,
    /// Served model reference when one exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_reference: Option<String>,
    /// Honest bounded notes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// Stable descriptor digest.
    pub descriptor_digest: String,
}

impl MlxBenchmarkProviderDescriptor {
    /// Creates one provider descriptor.
    #[must_use]
    pub fn new(
        provider_id: impl Into<String>,
        provider_kind: MlxBenchmarkProviderKind,
        model_reference: Option<String>,
        notes: Vec<String>,
    ) -> Self {
        let provider_id = provider_id.into();
        let descriptor_digest = stable_provider_descriptor_digest(
            provider_id.as_str(),
            provider_kind,
            model_reference.as_deref(),
            notes.as_slice(),
        );
        Self {
            provider_id,
            provider_kind,
            model_reference,
            notes,
            descriptor_digest,
        }
    }
}

/// Common benchmark expectation family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MlxBenchmarkExpectation {
    /// Exact trimmed text match.
    ExactText {
        /// Expected output text.
        expected_text: String,
    },
    /// Require all fragments and score by matched fraction.
    ContainsAll {
        /// Required fragments.
        required_fragments: Vec<String>,
    },
    /// Require exact machine-readable JSON output.
    StructuredJson {
        /// Expected structured output.
        expected: Value,
    },
}

/// One text benchmark request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxTextBenchmarkRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Ordered prompt messages.
    pub messages: Vec<PromptMessage>,
    /// Generation options forwarded into the provider lane.
    pub options: GenerationOptions,
}

impl MlxTextBenchmarkRequest {
    /// Creates one text benchmark request.
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        messages: Vec<PromptMessage>,
        options: GenerationOptions,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            messages,
            options,
        }
    }
}

/// One multimodal benchmark request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxMultimodalBenchmarkRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Processor family or alias.
    pub family: String,
    /// Ordered multimodal messages.
    pub messages: Vec<MlxVlmMessage>,
    /// Generation options forwarded into the provider lane.
    pub options: GenerationOptions,
}

impl MlxMultimodalBenchmarkRequest {
    /// Creates one multimodal benchmark request.
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        family: impl Into<String>,
        messages: Vec<MlxVlmMessage>,
        options: GenerationOptions,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            family: family.into(),
            messages,
            options,
        }
    }
}

/// Input family for one benchmark case.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MlxBenchmarkInput {
    /// Text-only messages.
    Text {
        /// Text benchmark request.
        request: MlxTextBenchmarkRequest,
    },
    /// Multimodal messages projected through `psionic-mlx-vlm`.
    Multimodal {
        /// Multimodal benchmark request.
        request: MlxMultimodalBenchmarkRequest,
    },
}

/// One benchmark case specification supplied to the workspace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxBenchmarkCaseSpec {
    /// Stable case identifier.
    pub case_id: String,
    /// Optional ordinal in the suite.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<u64>,
    /// Case input.
    pub input: MlxBenchmarkInput,
    /// Expected scoring contract.
    pub expectation: MlxBenchmarkExpectation,
    /// Optional input reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ref: Option<String>,
    /// Optional expected-output reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_output_ref: Option<String>,
    /// Extension metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl MlxBenchmarkCaseSpec {
    /// Creates one text-only case.
    #[must_use]
    pub fn text(
        case_id: impl Into<String>,
        request: MlxTextBenchmarkRequest,
        expectation: MlxBenchmarkExpectation,
    ) -> Self {
        Self {
            case_id: case_id.into(),
            ordinal: None,
            input: MlxBenchmarkInput::Text { request },
            expectation,
            input_ref: None,
            expected_output_ref: None,
            metadata: BTreeMap::new(),
        }
    }

    /// Creates one multimodal case.
    #[must_use]
    pub fn multimodal(
        case_id: impl Into<String>,
        request: MlxMultimodalBenchmarkRequest,
        expectation: MlxBenchmarkExpectation,
    ) -> Self {
        Self {
            case_id: case_id.into(),
            ordinal: None,
            input: MlxBenchmarkInput::Multimodal { request },
            expectation,
            input_ref: None,
            expected_output_ref: None,
            metadata: BTreeMap::new(),
        }
    }
}

/// One validated benchmark case carried by a built suite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxBenchmarkCase {
    /// Stable case identifier.
    pub case_id: String,
    /// Optional ordinal in the suite.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ordinal: Option<u64>,
    /// Case input.
    pub input: MlxBenchmarkInput,
    /// Expected scoring contract.
    pub expectation: MlxBenchmarkExpectation,
    /// Optional input reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ref: Option<String>,
    /// Optional expected-output reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_output_ref: Option<String>,
    /// Input digest carried by the manifest.
    pub input_digest: String,
    /// Expectation digest carried by the manifest.
    pub expectation_digest: String,
    /// Extension metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// One case manifest row inside a benchmark suite.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxBenchmarkCaseManifest {
    /// Stable case identifier.
    pub case_id: String,
    /// Input family label.
    pub input_kind: String,
    /// Expectation family label.
    pub expectation_kind: String,
    /// Stable input digest.
    pub input_digest: String,
    /// Stable expectation digest.
    pub expectation_digest: String,
}

/// One machine-readable benchmark suite manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxBenchmarkSuiteManifest {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable benchmark storage key.
    pub benchmark_storage_key: String,
    /// Human-readable benchmark name.
    pub display_name: String,
    /// Bound environment storage key.
    pub environment_storage_key: String,
    /// Stable benchmark-package digest.
    pub benchmark_package_digest: String,
    /// Stable case manifests.
    pub case_manifests: Vec<MlxBenchmarkCaseManifest>,
    /// Honest bounded notes.
    pub notes: Vec<String>,
    /// Stable manifest digest.
    pub manifest_digest: String,
}

/// Full benchmark suite request passed to the workspace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxBenchmarkSuiteSpec {
    /// Stable benchmark identity.
    pub benchmark_key: BenchmarkPackageKey,
    /// Human-readable benchmark name.
    pub display_name: String,
    /// Environment identity used by the eval runtime.
    pub environment: EnvironmentPackageKey,
    /// Optional dataset binding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset: Option<DatasetKey>,
    /// Optional dataset split.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
    /// Number of repeated rounds.
    pub repeat_count: u32,
    /// Robust aggregate mode.
    pub aggregation: BenchmarkAggregationKind,
    /// Verification policy expected by the benchmark runtime.
    pub verification_policy: BenchmarkVerificationPolicy,
    /// Cases carried by the benchmark.
    pub cases: Vec<MlxBenchmarkCaseSpec>,
    /// Honest bounded notes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// Extension metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

/// One built benchmark suite above `psionic-eval`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxBenchmarkSuite {
    benchmark_package: BenchmarkPackage,
    cases: Vec<MlxBenchmarkCase>,
    manifest: MlxBenchmarkSuiteManifest,
}

impl MlxBenchmarkSuite {
    /// Returns the benchmark package shared with `psionic-eval`.
    #[must_use]
    pub fn benchmark_package(&self) -> &BenchmarkPackage {
        &self.benchmark_package
    }

    /// Returns the validated cases.
    #[must_use]
    pub fn cases(&self) -> &[MlxBenchmarkCase] {
        &self.cases
    }

    /// Returns the benchmark suite manifest.
    #[must_use]
    pub fn manifest(&self) -> &MlxBenchmarkSuiteManifest {
        &self.manifest
    }

    /// Executes the suite through one provider adapter.
    pub fn execute<P: MlxBenchmarkProvider>(
        &self,
        provider: &mut P,
        execution_mode: BenchmarkExecutionMode,
    ) -> Result<MlxBenchmarkRunReceipt, MlxBenchError> {
        let descriptor = provider.descriptor().clone();
        let mut benchmark_session = self
            .benchmark_package
            .clone()
            .open_execution(execution_mode)?;
        let manifest_bytes =
            serde_json::to_vec(&self.manifest).map_err(|error| MlxBenchError::Serialization {
                context: "benchmark manifest export",
                message: error.to_string(),
            })?;
        let run_artifact = EvalArtifact::new(
            "mlx.benchmark.manifest",
            format!("benchmark://{}", self.manifest.benchmark_storage_key),
            manifest_bytes.as_slice(),
        );

        let mut rounds = Vec::with_capacity(self.benchmark_package.repeat_count as usize);
        for round_index in 1..=self.benchmark_package.repeat_count {
            let mut contract = EvalRunContract::new(
                format!(
                    "{}::{}::round-{}",
                    self.benchmark_package.key.storage_key(),
                    descriptor.provider_id,
                    round_index
                ),
                EvalRunMode::Benchmark,
                self.benchmark_package.environment.clone(),
            )
            .with_benchmark_package(self.benchmark_package.key.clone())
            .with_expected_sample_count(self.cases.len() as u64);
            if let Some(dataset) = &self.benchmark_package.dataset {
                contract =
                    contract.with_dataset(dataset.clone(), self.benchmark_package.split.clone());
            }
            let mut run = EvalRunState::open(contract)?;
            run.start(u64::from(round_index))?;
            for case in &self.cases {
                match provider.execute_case(case) {
                    Ok(response) => run.append_sample(score_case(
                        case,
                        &response,
                        &self.benchmark_package.environment,
                        self.manifest.benchmark_storage_key.as_str(),
                    )?)?,
                    Err(error) => run.append_sample(EvalSampleRecord::errored(
                        case.case_id.clone(),
                        self.benchmark_package.environment.clone(),
                        error.to_string(),
                    ))?,
                }
            }
            run.finalize(u64::from(round_index) + 1_000, vec![run_artifact.clone()])?;
            benchmark_session.record_round(&run)?;
            rounds.push(run);
        }

        let aggregate = benchmark_session.finalize()?;
        let notes = execution_notes(descriptor.provider_kind);
        let round_summary_digests = rounds
            .iter()
            .filter_map(|round| {
                round
                    .summary
                    .as_ref()
                    .map(|summary| summary.summary_digest.clone())
            })
            .collect::<Vec<_>>();
        Ok(MlxBenchmarkRunReceipt {
            schema_version: 1,
            benchmark_storage_key: self.benchmark_package.key.storage_key(),
            provider: descriptor,
            execution_mode,
            suite_manifest_digest: self.manifest.manifest_digest.clone(),
            rounds,
            aggregate,
            notes: notes.clone(),
            receipt_digest: stable_run_receipt_digest(
                self.manifest.manifest_digest.as_str(),
                self.benchmark_package.key.storage_key().as_str(),
                &provider.descriptor().descriptor_digest,
                execution_mode,
                &notes,
                round_summary_digests.as_slice(),
            ),
        })
    }
}

/// Machine-readable run receipt over repeated benchmark rounds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxBenchmarkRunReceipt {
    /// Stable schema version.
    pub schema_version: u32,
    /// Stable benchmark storage key.
    pub benchmark_storage_key: String,
    /// Provider descriptor used for the run.
    pub provider: MlxBenchmarkProviderDescriptor,
    /// Execution mode.
    pub execution_mode: BenchmarkExecutionMode,
    /// Stable suite-manifest digest.
    pub suite_manifest_digest: String,
    /// Finalized per-round eval runs.
    pub rounds: Vec<EvalRunState>,
    /// Aggregate benchmark summary.
    pub aggregate: BenchmarkAggregateSummary,
    /// Honest bounded notes.
    pub notes: Vec<String>,
    /// Stable receipt digest.
    pub receipt_digest: String,
}

/// Common provider observation consumed by the benchmark runtime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxProviderResponse {
    /// Rendered output text.
    pub output_text: String,
    /// Optional structured output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<Value>,
    /// Input token count.
    pub input_tokens: u32,
    /// Output token count.
    pub output_tokens: u32,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: u64,
    /// Stable final-state digest or equivalent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_digest: Option<String>,
    /// Declared execution strategy facts.
    pub execution_strategy: EvalExecutionStrategyFacts,
    /// Provider-emitted artifacts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<EvalArtifact>,
    /// Extension metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl MlxProviderResponse {
    /// Creates one provider response.
    #[must_use]
    pub fn new(
        output_text: impl Into<String>,
        input_tokens: u32,
        output_tokens: u32,
        elapsed_ms: u64,
    ) -> Self {
        Self {
            output_text: output_text.into(),
            structured_output: None,
            input_tokens,
            output_tokens,
            elapsed_ms,
            session_digest: None,
            execution_strategy: EvalExecutionStrategyFacts {
                strategy_label: String::from("local_provider"),
                runtime_family: None,
                scheduler_posture: None,
            },
            artifacts: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    /// Attaches machine-readable structured output.
    #[must_use]
    pub fn with_structured_output(mut self, structured_output: Value) -> Self {
        self.structured_output = Some(structured_output);
        self
    }

    /// Attaches an explicit final-state digest.
    #[must_use]
    pub fn with_session_digest(mut self, session_digest: impl Into<String>) -> Self {
        self.session_digest = Some(session_digest.into());
        self
    }

    /// Attaches explicit execution strategy facts.
    #[must_use]
    pub fn with_execution_strategy(
        mut self,
        execution_strategy: EvalExecutionStrategyFacts,
    ) -> Self {
        self.execution_strategy = execution_strategy;
        self
    }

    /// Attaches provider artifacts.
    #[must_use]
    pub fn with_artifacts(mut self, artifacts: Vec<EvalArtifact>) -> Self {
        self.artifacts = artifacts;
        self
    }

    /// Attaches metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Bridges one local `GenerationResponse` into the benchmark package.
    pub fn from_generation_response(
        response: &GenerationResponse,
        elapsed_ms_override: Option<u64>,
    ) -> Result<Self, MlxBenchError> {
        let structured_output = response
            .output
            .structured
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| MlxBenchError::Serialization {
                context: "structured output bridge",
                message: error.to_string(),
            })?;
        let elapsed_ms = elapsed_ms_override.unwrap_or_else(|| {
            response
                .metrics
                .total_duration_ns
                .map(|value| value / 1_000_000)
                .unwrap_or_default()
        });
        let execution_strategy = EvalExecutionStrategyFacts {
            strategy_label: if response
                .provenance
                .as_ref()
                .and_then(|value| value.scheduler.as_ref())
                .is_some()
            {
                String::from("shared_scheduler")
            } else {
                String::from("local_generation")
            },
            runtime_family: Some(String::from("psionic_serve")),
            scheduler_posture: Some(
                if response
                    .provenance
                    .as_ref()
                    .and_then(|value| value.scheduler.as_ref())
                    .is_some()
                {
                    String::from("scheduled")
                } else {
                    String::from("direct")
                },
            ),
        };
        Ok(Self {
            output_text: response.output.text.clone(),
            structured_output: structured_output.clone(),
            input_tokens: saturating_usize_to_u32(response.usage.input_tokens),
            output_tokens: saturating_usize_to_u32(response.usage.output_tokens),
            elapsed_ms,
            session_digest: Some(stable_provider_output_digest(
                response.request_id.as_str(),
                response.output.text.as_str(),
                structured_output.as_ref(),
            )),
            execution_strategy,
            artifacts: Vec::new(),
            metadata: BTreeMap::new(),
        })
    }
}

/// Request surfaced to one text provider adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxTextProviderRequest {
    /// Generated local text request.
    pub request: MlxLmTextRequest,
    /// Ordered source messages.
    pub messages: Vec<PromptMessage>,
    /// Stable rendered prompt digest.
    pub prompt_digest: String,
    /// Projection report when the request came from a multimodal case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection: Option<MlxVlmProjectionReport>,
}

/// Request surfaced to one served provider adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxServedBenchmarkRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Target served endpoint.
    pub endpoint: MlxVlmServedEndpoint,
    /// JSON request body.
    pub request_json: Value,
    /// Stable rendered prompt digest.
    pub prompt_digest: String,
    /// Projection plan when the request came from a multimodal case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection: Option<MlxVlmServePlan>,
}

/// Error returned by the benchmark package.
#[derive(Debug, Error)]
pub enum MlxBenchError {
    /// One required field was empty.
    #[error("missing required field `{field}`")]
    MissingField {
        /// Stable field label.
        field: &'static str,
    },
    /// One case id was repeated.
    #[error("benchmark suite repeated case `{case_id}`")]
    DuplicateCaseId {
        /// Repeated case id.
        case_id: String,
    },
    /// One case was structurally invalid.
    #[error("benchmark case `{case_id}` is invalid: {message}")]
    InvalidCase {
        /// Stable case id.
        case_id: String,
        /// Human-readable reason.
        message: String,
    },
    /// One provider did not support the supplied input kind.
    #[error("provider `{provider_kind:?}` does not support benchmark input `{input_kind}`")]
    UnsupportedInputForProvider {
        /// Provider family.
        provider_kind: MlxBenchmarkProviderKind,
        /// Input family label.
        input_kind: String,
    },
    /// One provider invocation failed explicitly.
    #[error("provider `{provider_id}` failed for case `{case_id}`: {message}")]
    ProviderFailure {
        /// Provider identifier.
        provider_id: String,
        /// Benchmark case id.
        case_id: String,
        /// Human-readable reason.
        message: String,
    },
    /// One lower-level eval contract failed.
    #[error(transparent)]
    Eval(#[from] EvalRuntimeError),
    /// One lower-level multimodal projection failed.
    #[error(transparent)]
    Vlm(#[from] MlxVlmError),
    /// JSON serialization failed.
    #[error("{context}: {message}")]
    Serialization {
        /// Serialization path.
        context: &'static str,
        /// Human-readable reason.
        message: String,
    },
}

/// Trait implemented by all benchmark providers.
pub trait MlxBenchmarkProvider {
    /// Returns the stable provider descriptor.
    fn descriptor(&self) -> &MlxBenchmarkProviderDescriptor;

    /// Executes one benchmark case and returns a provider observation.
    fn execute_case(
        &mut self,
        case: &MlxBenchmarkCase,
    ) -> Result<MlxProviderResponse, MlxBenchError>;
}

/// Workspace for building benchmark suites.
#[derive(Clone, Debug, Default)]
pub struct MlxBenchWorkspace {
    vlm: MlxVlmWorkspace,
}

impl MlxBenchWorkspace {
    /// Builds one validated benchmark suite.
    pub fn build_suite(
        &self,
        spec: &MlxBenchmarkSuiteSpec,
    ) -> Result<MlxBenchmarkSuite, MlxBenchError> {
        if spec.benchmark_key.benchmark_ref.trim().is_empty() {
            return Err(MlxBenchError::MissingField {
                field: "benchmark_key.benchmark_ref",
            });
        }
        if spec.benchmark_key.version.trim().is_empty() {
            return Err(MlxBenchError::MissingField {
                field: "benchmark_key.version",
            });
        }
        if spec.display_name.trim().is_empty() {
            return Err(MlxBenchError::MissingField {
                field: "display_name",
            });
        }
        if spec.environment.environment_ref.trim().is_empty() {
            return Err(MlxBenchError::MissingField {
                field: "environment.environment_ref",
            });
        }
        if spec.environment.version.trim().is_empty() {
            return Err(MlxBenchError::MissingField {
                field: "environment.version",
            });
        }
        if spec.cases.is_empty() {
            return Err(MlxBenchError::MissingField { field: "cases" });
        }

        let mut case_ids = BTreeSet::new();
        let mut cases = Vec::with_capacity(spec.cases.len());
        let mut case_manifests = Vec::with_capacity(spec.cases.len());
        for case in &spec.cases {
            if !case_ids.insert(case.case_id.clone()) {
                return Err(MlxBenchError::DuplicateCaseId {
                    case_id: case.case_id.clone(),
                });
            }
            let (input_digest, input_kind) = self.validate_case_input(case)?;
            let expectation_digest = validate_expectation(case)?;
            let expectation_kind = expectation_kind_label(&case.expectation).to_string();
            cases.push(MlxBenchmarkCase {
                case_id: case.case_id.clone(),
                ordinal: case.ordinal,
                input: case.input.clone(),
                expectation: case.expectation.clone(),
                input_ref: case.input_ref.clone(),
                expected_output_ref: case.expected_output_ref.clone(),
                input_digest: input_digest.clone(),
                expectation_digest: expectation_digest.clone(),
                metadata: case.metadata.clone(),
            });
            case_manifests.push(MlxBenchmarkCaseManifest {
                case_id: case.case_id.clone(),
                input_kind,
                expectation_kind,
                input_digest,
                expectation_digest,
            });
        }

        let mut benchmark_package = BenchmarkPackage::new(
            spec.benchmark_key.clone(),
            spec.display_name.clone(),
            spec.environment.clone(),
            spec.repeat_count,
            spec.aggregation,
        )
        .with_verification_policy(spec.verification_policy.clone())
        .with_cases(
            cases
                .iter()
                .map(|case| {
                    let mut metadata = serde_json::Map::new();
                    metadata.insert(
                        String::from("input_digest"),
                        Value::String(case.input_digest.clone()),
                    );
                    metadata.insert(
                        String::from("expectation_digest"),
                        Value::String(case.expectation_digest.clone()),
                    );
                    BenchmarkCase {
                        case_id: case.case_id.clone(),
                        ordinal: case.ordinal,
                        input_ref: case.input_ref.clone(),
                        expected_output_ref: case.expected_output_ref.clone(),
                        metadata: Value::Object(metadata),
                    }
                })
                .collect(),
        );
        if let Some(dataset) = &spec.dataset {
            benchmark_package = benchmark_package.with_dataset(dataset.clone(), spec.split.clone());
        }
        benchmark_package.metadata.insert(
            String::from("crate_role"),
            Value::String(String::from(CRATE_ROLE)),
        );
        benchmark_package.metadata.insert(
            String::from("mlx_package_family"),
            Value::String(String::from("benchmark")),
        );
        benchmark_package
            .metadata
            .insert(String::from("case_count"), Value::from(cases.len() as u64));
        for (key, value) in &spec.metadata {
            benchmark_package
                .metadata
                .insert(key.clone(), value.clone());
        }
        benchmark_package.validate()?;

        let notes = suite_notes(spec.notes.as_slice());
        let manifest = MlxBenchmarkSuiteManifest {
            schema_version: 1,
            benchmark_storage_key: benchmark_package.key.storage_key(),
            display_name: spec.display_name.clone(),
            environment_storage_key: spec.environment.storage_key(),
            benchmark_package_digest: benchmark_package.stable_digest(),
            case_manifests,
            notes: notes.clone(),
            manifest_digest: stable_suite_manifest_digest(
                benchmark_package.key.storage_key().as_str(),
                spec.display_name.as_str(),
                spec.environment.storage_key().as_str(),
                benchmark_package.stable_digest().as_str(),
                notes.as_slice(),
                cases.as_slice(),
            ),
        };
        Ok(MlxBenchmarkSuite {
            benchmark_package,
            cases,
            manifest,
        })
    }

    fn validate_case_input(
        &self,
        case: &MlxBenchmarkCaseSpec,
    ) -> Result<(String, String), MlxBenchError> {
        match &case.input {
            MlxBenchmarkInput::Text { request } => {
                if case.case_id.trim().is_empty() {
                    return Err(MlxBenchError::MissingField { field: "case_id" });
                }
                if request.request_id.trim().is_empty() {
                    return Err(MlxBenchError::InvalidCase {
                        case_id: case.case_id.clone(),
                        message: String::from("text request_id must be non-empty"),
                    });
                }
                if request.messages.is_empty() {
                    return Err(MlxBenchError::InvalidCase {
                        case_id: case.case_id.clone(),
                        message: String::from("text cases require at least one prompt message"),
                    });
                }
                Ok((
                    stable_text_input_digest(request).map_err(|error| {
                        MlxBenchError::Serialization {
                            context: "text benchmark input digest",
                            message: error.to_string(),
                        }
                    })?,
                    String::from("text"),
                ))
            }
            MlxBenchmarkInput::Multimodal { request } => {
                if case.case_id.trim().is_empty() {
                    return Err(MlxBenchError::MissingField { field: "case_id" });
                }
                if request.request_id.trim().is_empty() {
                    return Err(MlxBenchError::InvalidCase {
                        case_id: case.case_id.clone(),
                        message: String::from("multimodal request_id must be non-empty"),
                    });
                }
                if request.family.trim().is_empty() {
                    return Err(MlxBenchError::InvalidCase {
                        case_id: case.case_id.clone(),
                        message: String::from("multimodal family must be non-empty"),
                    });
                }
                if request.messages.is_empty() {
                    return Err(MlxBenchError::InvalidCase {
                        case_id: case.case_id.clone(),
                        message: String::from("multimodal cases require at least one message"),
                    });
                }
                let projection = self
                    .vlm
                    .project_messages(request.family.as_str(), request.messages.as_slice())?;
                Ok((
                    stable_multimodal_input_digest(request, &projection).map_err(|error| {
                        MlxBenchError::Serialization {
                            context: "multimodal benchmark input digest",
                            message: error.to_string(),
                        }
                    })?,
                    String::from("multimodal"),
                ))
            }
        }
    }
}

/// Provider adapter for local text-generation lanes.
pub struct MlxTextBenchmarkProvider<F>
where
    F: FnMut(&MlxTextProviderRequest) -> Result<MlxProviderResponse, String>,
{
    descriptor: MlxBenchmarkProviderDescriptor,
    vlm: MlxVlmWorkspace,
    runner: F,
}

impl<F> MlxTextBenchmarkProvider<F>
where
    F: FnMut(&MlxTextProviderRequest) -> Result<MlxProviderResponse, String>,
{
    /// Creates one local text provider adapter.
    pub fn new(
        provider_id: impl Into<String>,
        notes: Vec<String>,
        runner: F,
    ) -> Result<Self, MlxBenchError> {
        let provider_id = provider_id.into();
        if provider_id.trim().is_empty() {
            return Err(MlxBenchError::MissingField {
                field: "provider_id",
            });
        }
        Ok(Self {
            descriptor: MlxBenchmarkProviderDescriptor::new(
                provider_id,
                MlxBenchmarkProviderKind::LocalText,
                None,
                notes,
            ),
            vlm: MlxVlmWorkspace::default(),
            runner,
        })
    }
}

impl<F> MlxBenchmarkProvider for MlxTextBenchmarkProvider<F>
where
    F: FnMut(&MlxTextProviderRequest) -> Result<MlxProviderResponse, String>,
{
    fn descriptor(&self) -> &MlxBenchmarkProviderDescriptor {
        &self.descriptor
    }

    fn execute_case(
        &mut self,
        case: &MlxBenchmarkCase,
    ) -> Result<MlxProviderResponse, MlxBenchError> {
        let request = match &case.input {
            MlxBenchmarkInput::Text { request } => {
                let prompt = render_prompt_messages(request.messages.as_slice());
                MlxTextProviderRequest {
                    request: MlxLmTextRequest::new(
                        request.request_id.clone(),
                        prompt,
                        request.options.clone(),
                    ),
                    messages: request.messages.clone(),
                    prompt_digest: digest_text_messages(request.messages.as_slice())?,
                    projection: None,
                }
            }
            MlxBenchmarkInput::Multimodal { request } => {
                let projection = self
                    .vlm
                    .project_messages(request.family.as_str(), request.messages.as_slice())?;
                let prompt = render_prompt_messages(projection.projected_messages.as_slice());
                MlxTextProviderRequest {
                    request: MlxLmTextRequest::new(
                        request.request_id.clone(),
                        prompt,
                        request.options.clone(),
                    ),
                    messages: projection.projected_messages.clone(),
                    prompt_digest: digest_text_messages(projection.projected_messages.as_slice())?,
                    projection: Some(projection),
                }
            }
        };
        (self.runner)(&request).map_err(|message| MlxBenchError::ProviderFailure {
            provider_id: self.descriptor.provider_id.clone(),
            case_id: case.case_id.clone(),
            message,
        })
    }
}

/// Provider adapter for served text lanes.
pub struct MlxServedBenchmarkProvider<F>
where
    F: FnMut(&MlxServedBenchmarkRequest) -> Result<MlxProviderResponse, String>,
{
    descriptor: MlxBenchmarkProviderDescriptor,
    model_reference: String,
    endpoint: MlxVlmServedEndpoint,
    vlm: MlxVlmWorkspace,
    runner: F,
}

impl<F> MlxServedBenchmarkProvider<F>
where
    F: FnMut(&MlxServedBenchmarkRequest) -> Result<MlxProviderResponse, String>,
{
    /// Creates one served provider adapter.
    pub fn new(
        provider_id: impl Into<String>,
        model_reference: impl Into<String>,
        endpoint: MlxVlmServedEndpoint,
        notes: Vec<String>,
        runner: F,
    ) -> Result<Self, MlxBenchError> {
        let provider_id = provider_id.into();
        let model_reference = model_reference.into();
        if provider_id.trim().is_empty() {
            return Err(MlxBenchError::MissingField {
                field: "provider_id",
            });
        }
        if model_reference.trim().is_empty() {
            return Err(MlxBenchError::MissingField {
                field: "model_reference",
            });
        }
        Ok(Self {
            descriptor: MlxBenchmarkProviderDescriptor::new(
                provider_id,
                MlxBenchmarkProviderKind::ServedText,
                Some(model_reference.clone()),
                notes,
            ),
            model_reference,
            endpoint,
            vlm: MlxVlmWorkspace::default(),
            runner,
        })
    }
}

impl<F> MlxBenchmarkProvider for MlxServedBenchmarkProvider<F>
where
    F: FnMut(&MlxServedBenchmarkRequest) -> Result<MlxProviderResponse, String>,
{
    fn descriptor(&self) -> &MlxBenchmarkProviderDescriptor {
        &self.descriptor
    }

    fn execute_case(
        &mut self,
        case: &MlxBenchmarkCase,
    ) -> Result<MlxProviderResponse, MlxBenchError> {
        let request = match &case.input {
            MlxBenchmarkInput::Text { request } => MlxServedBenchmarkRequest {
                request_id: request.request_id.clone(),
                endpoint: self.endpoint,
                request_json: build_text_served_request_json(
                    self.model_reference.as_str(),
                    self.endpoint,
                    request.messages.as_slice(),
                    &request.options,
                )?,
                prompt_digest: digest_text_messages(request.messages.as_slice())?,
                projection: None,
            },
            MlxBenchmarkInput::Multimodal { request } => {
                let mut plan = self.vlm.plan_request(
                    request.family.as_str(),
                    self.model_reference.as_str(),
                    self.endpoint,
                    request.messages.as_slice(),
                )?;
                apply_generation_options(
                    &mut plan.translated_request_json,
                    self.endpoint,
                    &request.options,
                )?;
                let prompt_digest =
                    digest_text_messages(plan.projection.projected_messages.as_slice())?;
                MlxServedBenchmarkRequest {
                    request_id: request.request_id.clone(),
                    endpoint: self.endpoint,
                    request_json: plan.translated_request_json.clone(),
                    prompt_digest,
                    projection: Some(plan),
                }
            }
        };
        (self.runner)(&request).map_err(|message| MlxBenchError::ProviderFailure {
            provider_id: self.descriptor.provider_id.clone(),
            case_id: case.case_id.clone(),
            message,
        })
    }
}

fn validate_expectation(case: &MlxBenchmarkCaseSpec) -> Result<String, MlxBenchError> {
    match &case.expectation {
        MlxBenchmarkExpectation::ExactText { expected_text } => {
            if expected_text.trim().is_empty() {
                return Err(MlxBenchError::InvalidCase {
                    case_id: case.case_id.clone(),
                    message: String::from("exact_text expectation must be non-empty"),
                });
            }
        }
        MlxBenchmarkExpectation::ContainsAll { required_fragments } => {
            if required_fragments.is_empty() {
                return Err(MlxBenchError::InvalidCase {
                    case_id: case.case_id.clone(),
                    message: String::from(
                        "contains_all expectation requires at least one fragment",
                    ),
                });
            }
            if required_fragments
                .iter()
                .any(|value| value.trim().is_empty())
            {
                return Err(MlxBenchError::InvalidCase {
                    case_id: case.case_id.clone(),
                    message: String::from("contains_all fragments must be non-empty"),
                });
            }
        }
        MlxBenchmarkExpectation::StructuredJson { expected } => {
            if expected.is_null() {
                return Err(MlxBenchError::InvalidCase {
                    case_id: case.case_id.clone(),
                    message: String::from("structured_json expectation must be non-null"),
                });
            }
        }
    }
    stable_expectation_digest(&case.expectation).map_err(|error| MlxBenchError::Serialization {
        context: "benchmark expectation digest",
        message: error.to_string(),
    })
}

fn score_case(
    case: &MlxBenchmarkCase,
    response: &MlxProviderResponse,
    environment: &EnvironmentPackageKey,
    benchmark_storage_key: &str,
) -> Result<EvalSampleRecord, MlxBenchError> {
    let output_digest = stable_provider_output_digest(
        case.case_id.as_str(),
        response.output_text.as_str(),
        response.structured_output.as_ref(),
    );
    let (score_bps, passed, expectation_metadata) =
        evaluate_expectation(&case.expectation, response)?;
    let verification = EvalVerificationFacts {
        timer_integrity: Some(EvalTimerIntegrityFacts {
            declared_budget_ms: None,
            elapsed_ms: response.elapsed_ms,
            within_budget: true,
        }),
        token_accounting: Some(EvalTokenAccountingFacts::new(
            response.input_tokens,
            response.output_tokens,
            response.input_tokens.saturating_add(response.output_tokens),
        )?),
        final_state: Some(EvalFinalStateCapture {
            session_digest: response
                .session_digest
                .clone()
                .unwrap_or_else(|| output_digest.clone()),
            output_digest: Some(output_digest.clone()),
            artifact_digests: response
                .artifacts
                .iter()
                .map(|artifact| artifact.artifact_digest.clone())
                .collect(),
        }),
        execution_strategy: Some(response.execution_strategy.clone()),
    };
    let mut metrics = vec![
        EvalMetric::new(
            "mlx.openbench.primary_score",
            f64::from(score_bps) / 10_000.0,
        )
        .with_unit("fraction")
        .with_metadata(expectation_metadata),
        EvalMetric::new("mlx.openbench.elapsed_ms", response.elapsed_ms as f64).with_unit("ms"),
        EvalMetric::new("mlx.openbench.input_tokens", response.input_tokens as f64)
            .with_unit("tokens"),
        EvalMetric::new("mlx.openbench.output_tokens", response.output_tokens as f64)
            .with_unit("tokens"),
    ];
    metrics.sort_by(|left, right| left.metric_id.cmp(&right.metric_id));

    let mut artifacts = response.artifacts.clone();
    artifacts.push(EvalArtifact::new(
        "mlx.provider.output",
        format!("{benchmark_storage_key}#{}", case.case_id),
        response.output_text.as_bytes(),
    ));

    let mut metadata = case.metadata.clone();
    metadata.insert(
        String::from("input_digest"),
        Value::String(case.input_digest.clone()),
    );
    metadata.insert(
        String::from("expectation_digest"),
        Value::String(case.expectation_digest.clone()),
    );
    metadata.insert(
        String::from("output_digest"),
        Value::String(output_digest.clone()),
    );
    if let Some(structured_output) = &response.structured_output {
        metadata.insert(String::from("structured_output"), structured_output.clone());
    }
    for (key, value) in &response.metadata {
        metadata.insert(format!("provider_{key}"), value.clone());
    }

    Ok(EvalSampleRecord {
        sample_id: case.case_id.clone(),
        ordinal: case.ordinal,
        environment: environment.clone(),
        status: if passed {
            EvalSampleStatus::Passed
        } else {
            EvalSampleStatus::Failed
        },
        input_ref: case.input_ref.clone(),
        output_ref: Some(format!("output://{benchmark_storage_key}/{}", case.case_id)),
        expected_output_ref: case.expected_output_ref.clone(),
        score_bps: Some(score_bps),
        metrics,
        artifacts,
        error_reason: None,
        verification: Some(verification),
        session_digest: response
            .session_digest
            .clone()
            .or_else(|| Some(output_digest)),
        metadata,
    })
}

fn evaluate_expectation(
    expectation: &MlxBenchmarkExpectation,
    response: &MlxProviderResponse,
) -> Result<(u32, bool, Value), MlxBenchError> {
    match expectation {
        MlxBenchmarkExpectation::ExactText { expected_text } => {
            let passed =
                normalize_text(expected_text) == normalize_text(response.output_text.as_str());
            Ok((
                if passed { 10_000 } else { 0 },
                passed,
                json!({
                    "kind": "exact_text",
                    "expected_digest": digest_string(expected_text),
                    "actual_digest": digest_string(response.output_text.as_str()),
                }),
            ))
        }
        MlxBenchmarkExpectation::ContainsAll { required_fragments } => {
            let mut matched = Vec::new();
            let mut missing = Vec::new();
            let output_text = response.output_text.to_lowercase();
            for fragment in required_fragments {
                if output_text.contains(fragment.to_lowercase().as_str()) {
                    matched.push(fragment.clone());
                } else {
                    missing.push(fragment.clone());
                }
            }
            let score_bps =
                (matched.len() as u32).saturating_mul(10_000) / required_fragments.len() as u32;
            Ok((
                score_bps,
                missing.is_empty(),
                json!({
                    "kind": "contains_all",
                    "matched_fragments": matched,
                    "missing_fragments": missing,
                }),
            ))
        }
        MlxBenchmarkExpectation::StructuredJson { expected } => {
            let actual = if let Some(structured_output) = &response.structured_output {
                structured_output.clone()
            } else {
                serde_json::from_str::<Value>(response.output_text.as_str()).unwrap_or(Value::Null)
            };
            let passed = actual == *expected;
            Ok((
                if passed { 10_000 } else { 0 },
                passed,
                json!({
                    "kind": "structured_json",
                    "expected_digest": digest_value(expected)?,
                    "actual_digest": digest_value(&actual)?,
                }),
            ))
        }
    }
}

fn suite_notes(notes: &[String]) -> Vec<String> {
    let mut merged = vec![
        String::from(
            "This package reuses `psionic-eval` benchmark packages, eval runs, and aggregate summaries instead of inventing a second benchmark runtime.",
        ),
        String::from(
            "Multimodal benchmark cases stay bounded to `psionic-mlx-vlm` prompt projection; they do not claim a native image, audio, or video encoder inside this crate.",
        ),
        String::from(
            "Served provider adapters expose JSON requests plus receipt-bound output scoring, but transport ownership remains with the caller-supplied closure or harness.",
        ),
    ];
    merged.extend(notes.iter().cloned());
    merged
}

fn execution_notes(provider_kind: MlxBenchmarkProviderKind) -> Vec<String> {
    let mut notes = vec![String::from(
        "Benchmark execution records finalized `psionic-eval` runs per round and then reuses the same aggregate summary contract for repeated benchmark scoring.",
    )];
    match provider_kind {
        MlxBenchmarkProviderKind::LocalText => notes.push(String::from(
            "The local text adapter renders prompt messages into one deterministic text prompt before calling the provider-owned local generation lane.",
        )),
        MlxBenchmarkProviderKind::ServedText => notes.push(String::from(
            "The served adapter emits one JSON request per case so caller-owned harnesses can target local or private OpenAI-compatible served lanes without bypassing benchmark receipts.",
        )),
    }
    notes
}

fn stable_provider_descriptor_digest(
    provider_id: &str,
    provider_kind: MlxBenchmarkProviderKind,
    model_reference: Option<&str>,
    notes: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_bench_provider|");
    hasher.update(provider_id.as_bytes());
    hasher.update(b"|");
    hasher.update(provider_kind_label(provider_kind).as_bytes());
    if let Some(model_reference) = model_reference {
        hasher.update(b"|model|");
        hasher.update(model_reference.as_bytes());
    }
    for note in notes {
        hasher.update(b"|note|");
        hasher.update(note.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_suite_manifest_digest(
    benchmark_storage_key: &str,
    display_name: &str,
    environment_storage_key: &str,
    benchmark_package_digest: &str,
    notes: &[String],
    cases: &[MlxBenchmarkCase],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_bench_manifest|");
    hasher.update(benchmark_storage_key.as_bytes());
    hasher.update(b"|");
    hasher.update(display_name.as_bytes());
    hasher.update(b"|");
    hasher.update(environment_storage_key.as_bytes());
    hasher.update(b"|");
    hasher.update(benchmark_package_digest.as_bytes());
    for case in cases {
        hasher.update(b"|case|");
        hasher.update(case.case_id.as_bytes());
        hasher.update(b"|");
        hasher.update(case.input_digest.as_bytes());
        hasher.update(b"|");
        hasher.update(case.expectation_digest.as_bytes());
    }
    for note in notes {
        hasher.update(b"|note|");
        hasher.update(note.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_run_receipt_digest(
    suite_manifest_digest: &str,
    benchmark_storage_key: &str,
    provider_descriptor_digest: &str,
    execution_mode: BenchmarkExecutionMode,
    notes: &[String],
    round_summary_digests: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_bench_run|");
    hasher.update(suite_manifest_digest.as_bytes());
    hasher.update(b"|");
    hasher.update(benchmark_storage_key.as_bytes());
    hasher.update(b"|");
    hasher.update(provider_descriptor_digest.as_bytes());
    hasher.update(b"|");
    hasher.update(benchmark_execution_mode_label(execution_mode).as_bytes());
    for digest in round_summary_digests {
        hasher.update(b"|round|");
        hasher.update(digest.as_bytes());
    }
    for note in notes {
        hasher.update(b"|note|");
        hasher.update(note.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_text_input_digest(
    request: &MlxTextBenchmarkRequest,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(&(
        request.request_id.as_str(),
        &request.messages,
        &request.options,
    ))?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn stable_multimodal_input_digest(
    request: &MlxMultimodalBenchmarkRequest,
    projection: &MlxVlmProjectionReport,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(&(
        request.request_id.as_str(),
        request.family.as_str(),
        &request.messages,
        &request.options,
        projection,
    ))?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn stable_expectation_digest(
    expectation: &MlxBenchmarkExpectation,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(expectation)?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn stable_provider_output_digest(
    case_id: &str,
    output_text: &str,
    structured_output: Option<&Value>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_bench_output|");
    hasher.update(case_id.as_bytes());
    hasher.update(b"|");
    hasher.update(output_text.as_bytes());
    if let Some(structured_output) = structured_output {
        hasher.update(b"|structured|");
        hasher.update(
            digest_value(structured_output)
                .unwrap_or_else(|_| String::new())
                .as_bytes(),
        );
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn build_text_served_request_json(
    model_reference: &str,
    endpoint: MlxVlmServedEndpoint,
    messages: &[PromptMessage],
    options: &GenerationOptions,
) -> Result<Value, MlxBenchError> {
    let mut request_json = match endpoint {
        MlxVlmServedEndpoint::Responses => json!({
            "model": model_reference,
            "input": messages
                .iter()
                .map(|message| {
                    json!({
                        "role": prompt_message_role_label(message.role),
                        "content": [{
                            "type": "input_text",
                            "text": message.content,
                        }],
                    })
                })
                .collect::<Vec<_>>(),
        }),
        MlxVlmServedEndpoint::ChatCompletions => json!({
            "model": model_reference,
            "messages": messages
                .iter()
                .map(|message| {
                    json!({
                        "role": prompt_message_role_label(message.role),
                        "content": message.content,
                    })
                })
                .collect::<Vec<_>>(),
        }),
    };
    apply_generation_options(&mut request_json, endpoint, options)?;
    Ok(request_json)
}

fn apply_generation_options(
    request_json: &mut Value,
    endpoint: MlxVlmServedEndpoint,
    options: &GenerationOptions,
) -> Result<(), MlxBenchError> {
    let Some(map) = request_json.as_object_mut() else {
        return Err(MlxBenchError::Serialization {
            context: "served request option injection",
            message: String::from("request JSON must be an object"),
        });
    };
    map.insert(
        String::from(match endpoint {
            MlxVlmServedEndpoint::Responses => "max_output_tokens",
            MlxVlmServedEndpoint::ChatCompletions => "max_tokens",
        }),
        Value::from(options.max_output_tokens as u64),
    );
    if let Some(temperature) = options.temperature {
        map.insert(String::from("temperature"), Value::from(temperature));
    }
    if let Some(top_p) = options.top_p {
        map.insert(String::from("top_p"), Value::from(top_p));
    }
    if let Some(top_k) = options.top_k {
        map.insert(String::from("top_k"), Value::from(top_k as u64));
    }
    if let Some(seed) = options.seed {
        map.insert(String::from("seed"), Value::from(seed));
    }
    if !options.stop_sequences.is_empty() {
        map.insert(
            String::from("stop"),
            Value::Array(
                options
                    .stop_sequences
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
    }
    if let Some(structured_output) = &options.structured_output {
        map.insert(
            String::from("psionic_structured_output"),
            serde_json::to_value(structured_output).map_err(|error| {
                MlxBenchError::Serialization {
                    context: "structured output option export",
                    message: error.to_string(),
                }
            })?,
        );
    }
    Ok(())
}

fn render_prompt_messages(messages: &[PromptMessage]) -> String {
    messages
        .iter()
        .map(|message| {
            format!(
                "[{}]\n{}",
                prompt_message_role_label(message.role),
                message.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn digest_text_messages(messages: &[PromptMessage]) -> Result<String, MlxBenchError> {
    let bytes = serde_json::to_vec(messages).map_err(|error| MlxBenchError::Serialization {
        context: "prompt message digest",
        message: error.to_string(),
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn prompt_message_role_label(role: PromptMessageRole) -> &'static str {
    match role {
        PromptMessageRole::System => "system",
        PromptMessageRole::Developer => "developer",
        PromptMessageRole::User => "user",
        PromptMessageRole::Assistant => "assistant",
        PromptMessageRole::Tool => "tool",
    }
}

fn provider_kind_label(kind: MlxBenchmarkProviderKind) -> &'static str {
    match kind {
        MlxBenchmarkProviderKind::LocalText => "local_text",
        MlxBenchmarkProviderKind::ServedText => "served_text",
    }
}

fn benchmark_execution_mode_label(mode: BenchmarkExecutionMode) -> &'static str {
    match mode {
        BenchmarkExecutionMode::Validator => "validator",
        BenchmarkExecutionMode::OperatorSimulation => "operator_simulation",
    }
}

fn expectation_kind_label(expectation: &MlxBenchmarkExpectation) -> &'static str {
    match expectation {
        MlxBenchmarkExpectation::ExactText { .. } => "exact_text",
        MlxBenchmarkExpectation::ContainsAll { .. } => "contains_all",
        MlxBenchmarkExpectation::StructuredJson { .. } => "structured_json",
    }
}

fn normalize_text(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn digest_string(value: &str) -> String {
    format!("sha256:{:x}", Sha256::digest(value.as_bytes()))
}

fn digest_value(value: &Value) -> Result<String, MlxBenchError> {
    let bytes = serde_json::to_vec(value).map_err(|error| MlxBenchError::Serialization {
        context: "json digest",
        message: error.to_string(),
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn saturating_usize_to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        MlxBenchWorkspace, MlxBenchmarkCaseSpec, MlxBenchmarkExpectation, MlxBenchmarkProvider,
        MlxBenchmarkProviderKind, MlxMultimodalBenchmarkRequest, MlxProviderResponse,
        MlxServedBenchmarkProvider, MlxTextBenchmarkProvider, MlxTextBenchmarkRequest,
    };
    use psionic_data::DatasetKey;
    use psionic_environments::EnvironmentPackageKey;
    use psionic_eval::{
        BenchmarkAggregationKind, BenchmarkExecutionMode, BenchmarkPackageKey,
        BenchmarkVerificationPolicy, EvalExecutionStrategyFacts,
    };
    use psionic_mlx_vlm::{
        MlxVlmInputPart, MlxVlmMediaPart, MlxVlmMediaSource, MlxVlmMessage, MlxVlmMessageRole,
        MlxVlmServedEndpoint,
    };
    use psionic_models::{PromptMessage, PromptMessageRole};
    use psionic_serve::GenerationOptions;
    use serde_json::json;

    #[test]
    fn benchmark_suite_builds_manifest_and_package() -> Result<(), Box<dyn std::error::Error>> {
        let suite = MlxBenchWorkspace::default().build_suite(&sample_suite_spec())?;
        assert_eq!(
            suite.benchmark_package().key.storage_key(),
            String::from("benchmark://mlx/text-and-vlm@2026.03.17")
        );
        assert_eq!(suite.benchmark_package().cases.len(), 2);
        assert_eq!(suite.manifest().case_manifests.len(), 2);
        assert!(suite.manifest().manifest_digest.starts_with("sha256:"));
        assert_eq!(
            suite
                .benchmark_package()
                .dataset
                .as_ref()
                .map(DatasetKey::storage_key),
            Some(String::from("dataset://mlx/eval@2026.03.17"))
        );
        Ok(())
    }

    #[test]
    fn text_provider_executes_repeated_rounds() -> Result<(), Box<dyn std::error::Error>> {
        let suite = MlxBenchWorkspace::default().build_suite(&text_only_suite_spec())?;
        let mut prompts = Vec::new();
        let mut provider = MlxTextBenchmarkProvider::new(
            "local-text",
            vec![String::from("fixture local lane")],
            |request| {
                prompts.push(request.request.prompt.clone());
                let response = match request.request.request_id.as_str() {
                    "text-case-1" => MlxProviderResponse::new("4", 8, 1, 4),
                    "text-case-2" => MlxProviderResponse::new(
                        "Rust keeps receipts and explicit refusals visible.",
                        11,
                        8,
                        7,
                    ),
                    other => return Err(format!("unexpected request_id {other}")),
                };
                Ok(
                    response.with_execution_strategy(EvalExecutionStrategyFacts {
                        strategy_label: String::from("local_generation"),
                        runtime_family: Some(String::from("psionic_serve")),
                        scheduler_posture: Some(String::from("direct")),
                    }),
                )
            },
        )?;

        let receipt = suite.execute(&mut provider, BenchmarkExecutionMode::OperatorSimulation)?;
        assert_eq!(
            provider.descriptor().provider_kind,
            MlxBenchmarkProviderKind::LocalText
        );
        assert_eq!(receipt.rounds.len(), 2);
        assert_eq!(receipt.aggregate.round_count, 2);
        assert_eq!(receipt.aggregate.aggregate_score_bps, Some(10_000));
        assert_eq!(receipt.aggregate.aggregate_pass_rate_bps, 10_000);
        assert_eq!(prompts.len(), 4);
        assert!(prompts[0].contains("[user]"));
        Ok(())
    }

    #[test]
    fn served_provider_scores_multimodal_case() -> Result<(), Box<dyn std::error::Error>> {
        let suite = MlxBenchWorkspace::default().build_suite(&multimodal_structured_suite_spec())?;
        let mut served_requests = Vec::new();
        let mut provider = MlxServedBenchmarkProvider::new(
            "served-vlm",
            "hf:openagents/vlm",
            MlxVlmServedEndpoint::Responses,
            vec![String::from("fixture served lane")],
            |request| {
                served_requests.push(request.request_json.clone());
                Ok(MlxProviderResponse::new("cat", 14, 2, 6)
                    .with_structured_output(json!({"label": "cat"}))
                    .with_session_digest("served-session")
                    .with_execution_strategy(EvalExecutionStrategyFacts {
                        strategy_label: String::from("shared_scheduler"),
                        runtime_family: Some(String::from("psionic_serve")),
                        scheduler_posture: Some(String::from("scheduled")),
                    }))
            },
        )?;

        let receipt = suite.execute(&mut provider, BenchmarkExecutionMode::Validator)?;
        assert_eq!(receipt.aggregate.aggregate_score_bps, Some(10_000));
        assert_eq!(receipt.aggregate.aggregate_pass_rate_bps, 10_000);
        assert_eq!(served_requests.len(), 1);
        assert_eq!(served_requests[0]["model"], json!("hf:openagents/vlm"));
        assert!(
            served_requests[0]["input"][0]["content"][0]["text"]
                .as_str()
                .expect("text")
                .contains("<psionic_media")
        );
        Ok(())
    }

    fn sample_suite_spec() -> super::MlxBenchmarkSuiteSpec {
        super::MlxBenchmarkSuiteSpec {
            benchmark_key: BenchmarkPackageKey::new("benchmark://mlx/text-and-vlm", "2026.03.17"),
            display_name: String::from("MLX Text And VLM"),
            environment: EnvironmentPackageKey::new("env.psionic.mlx.openbench", "2026.03.17"),
            dataset: Some(DatasetKey::new("dataset://mlx/eval", "2026.03.17")),
            split: Some(String::from("validation")),
            repeat_count: 2,
            aggregation: BenchmarkAggregationKind::MedianScore,
            verification_policy: BenchmarkVerificationPolicy {
                require_timer_integrity: true,
                require_token_accounting: true,
                require_final_state_capture: true,
                require_execution_strategy: true,
            },
            cases: vec![
                MlxBenchmarkCaseSpec::text(
                    "text-case-1",
                    MlxTextBenchmarkRequest::new(
                        "text-case-1",
                        vec![PromptMessage::new(
                            PromptMessageRole::User,
                            "What is 2 + 2?",
                        )],
                        GenerationOptions::greedy(8),
                    ),
                    MlxBenchmarkExpectation::ExactText {
                        expected_text: String::from("4"),
                    },
                ),
                MlxBenchmarkCaseSpec::multimodal(
                    "vlm-case-1",
                    MlxMultimodalBenchmarkRequest::new(
                        "vlm-case-1",
                        "llava",
                        vec![MlxVlmMessage::new(
                            MlxVlmMessageRole::User,
                            vec![
                                MlxVlmInputPart::text("Describe the image."),
                                MlxVlmInputPart::InputImage {
                                    image: MlxVlmMediaPart::new(MlxVlmMediaSource::DataUrl {
                                        data_url: String::from("data:image/png;base64,AAAA"),
                                    }),
                                },
                            ],
                        )],
                        GenerationOptions::greedy(16),
                    ),
                    MlxBenchmarkExpectation::ContainsAll {
                        required_fragments: vec![String::from("cat")],
                    },
                ),
            ],
            notes: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    fn text_only_suite_spec() -> super::MlxBenchmarkSuiteSpec {
        super::MlxBenchmarkSuiteSpec {
            benchmark_key: BenchmarkPackageKey::new("benchmark://mlx/text", "2026.03.17"),
            display_name: String::from("MLX Text"),
            environment: EnvironmentPackageKey::new("env.psionic.mlx.openbench", "2026.03.17"),
            dataset: None,
            split: None,
            repeat_count: 2,
            aggregation: BenchmarkAggregationKind::MeanScore,
            verification_policy: BenchmarkVerificationPolicy {
                require_timer_integrity: true,
                require_token_accounting: true,
                require_final_state_capture: true,
                require_execution_strategy: true,
            },
            cases: vec![
                MlxBenchmarkCaseSpec::text(
                    "text-case-1",
                    MlxTextBenchmarkRequest::new(
                        "text-case-1",
                        vec![PromptMessage::new(PromptMessageRole::User, "2 + 2 = ?")],
                        GenerationOptions::greedy(8),
                    ),
                    MlxBenchmarkExpectation::ExactText {
                        expected_text: String::from("4"),
                    },
                ),
                MlxBenchmarkCaseSpec::text(
                    "text-case-2",
                    MlxTextBenchmarkRequest::new(
                        "text-case-2",
                        vec![
                            PromptMessage::new(
                                PromptMessageRole::System,
                                "Answer in one sentence.",
                            ),
                            PromptMessage::new(
                                PromptMessageRole::User,
                                "What matters about replay-safe inference?",
                            ),
                        ],
                        GenerationOptions::greedy(32),
                    ),
                    MlxBenchmarkExpectation::ContainsAll {
                        required_fragments: vec![
                            String::from("receipts"),
                            String::from("refusals"),
                        ],
                    },
                ),
            ],
            notes: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    fn multimodal_structured_suite_spec() -> super::MlxBenchmarkSuiteSpec {
        super::MlxBenchmarkSuiteSpec {
            benchmark_key: BenchmarkPackageKey::new("benchmark://mlx/vlm", "2026.03.17"),
            display_name: String::from("MLX VLM"),
            environment: EnvironmentPackageKey::new("env.psionic.mlx.openbench", "2026.03.17"),
            dataset: None,
            split: None,
            repeat_count: 1,
            aggregation: BenchmarkAggregationKind::MedianScore,
            verification_policy: BenchmarkVerificationPolicy {
                require_timer_integrity: true,
                require_token_accounting: true,
                require_final_state_capture: true,
                require_execution_strategy: true,
            },
            cases: vec![MlxBenchmarkCaseSpec::multimodal(
                "vlm-structured",
                MlxMultimodalBenchmarkRequest::new(
                    "vlm-structured",
                    "llava",
                    vec![MlxVlmMessage::new(
                        MlxVlmMessageRole::User,
                        vec![
                            MlxVlmInputPart::text("Name the pictured animal."),
                            MlxVlmInputPart::InputImage {
                                image: MlxVlmMediaPart::new(MlxVlmMediaSource::DataUrl {
                                    data_url: String::from("data:image/png;base64,AAAA"),
                                })
                                .with_mime_type("image/png"),
                            },
                        ],
                    )],
                    GenerationOptions::greedy(12),
                ),
                MlxBenchmarkExpectation::StructuredJson {
                    expected: json!({"label": "cat"}),
                },
            )],
            notes: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}
