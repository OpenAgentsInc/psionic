//! Bounded local MLX-lm-style text package above Psionic-native GGUF serving.

use std::{
    fs,
    path::{Path, PathBuf},
};

use psionic_catalog::{
    BlobError, BlobIntegrityPolicy, LocalBlob, LocalBlobKind, LocalBlobMetadata,
    LocalBlobOpenOptions,
};
use psionic_models::{
    ContextOverflowPolicy, DecoderModelDescriptor, GgufDecoderAdapter, GgufDecoderAdapterLoader,
    PromptMessage, PromptRenderError, PromptRenderOptions, RenderedPrompt,
};
use psionic_runtime::{
    CacheObservation, GenerationSchedulerMetrics, GenerationSchedulerRequestReceipt, KvCachePolicy,
    LocalRuntimeDiagnostic, PrefixCacheControl, PrefixCacheIdentity, PrefixCacheRefusalReason,
    PrefixCacheReusePolicy, PrefixCacheState, LocalRuntimeObservability,
};
use psionic_serve::{
    ContinuousBatchGenerationResult, CpuGgufTextGenerationService, GenerationEventStream,
    GenerationOptions, GenerationRequest, GenerationResponse, GgufDecoderRuntimeSupport,
    LoadedModelView, LoadedModelsObservation, ManagedTextGenerationRuntime,
    ReferenceTextGenerationError, SessionId, StreamingTextGenerationExecutor,
    TextGenerationExecutor,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str =
    "bounded local MLX-lm-style text package and CLI above Psionic-native GGUF serving";

/// One local text-generation request owned by the package layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxLmTextRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Optional reusable generation session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
    /// Prompt text.
    pub prompt: String,
    /// Generation options.
    pub options: GenerationOptions,
    /// Request-level shared-prefix cache policy.
    #[serde(default, skip_serializing_if = "PrefixCacheControl::is_default")]
    pub prefix_cache_control: PrefixCacheControl,
    /// Whether to reset the active session before generation.
    #[serde(default)]
    pub reset_session: bool,
}

impl MlxLmTextRequest {
    /// Creates one text-generation request.
    #[must_use]
    pub fn new(
        request_id: impl Into<String>,
        prompt: impl Into<String>,
        options: GenerationOptions,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            session_id: None,
            prompt: prompt.into(),
            options,
            prefix_cache_control: PrefixCacheControl::default(),
            reset_session: false,
        }
    }

    /// Reuses one existing session.
    #[must_use]
    pub fn with_session_id(mut self, session_id: SessionId) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Applies one explicit shared-prefix cache control.
    #[must_use]
    pub fn with_prefix_cache_control(mut self, prefix_cache_control: PrefixCacheControl) -> Self {
        self.prefix_cache_control = prefix_cache_control;
        self
    }

    /// Requests explicit context truncation or refusal posture.
    #[must_use]
    pub fn with_context_overflow_policy(mut self, policy: ContextOverflowPolicy) -> Self {
        self.options.context_overflow_policy = policy;
        self
    }

    /// Requests session reset before evaluation.
    #[must_use]
    pub fn with_reset_session(mut self, reset_session: bool) -> Self {
        self.reset_session = reset_session;
        self
    }
}

/// One load report for the current bounded MLX-lm-style package surface.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxLmLoadReport {
    /// Loaded GGUF path.
    pub model_path: PathBuf,
    /// Stable blob metadata for the opened GGUF bytes.
    pub blob_metadata: LocalBlobMetadata,
    /// Active decoder descriptor.
    pub descriptor: DecoderModelDescriptor,
    /// Active runtime-support report.
    pub runtime_support: GgufDecoderRuntimeSupport,
    /// Stable digest over all discovered chat templates.
    pub chat_template_digest: String,
    /// Whether the GGUF metadata carries a default chat template.
    pub has_default_chat_template: bool,
    /// Named GGUF chat-template keys in stable order.
    pub named_chat_templates: Vec<String>,
}

/// Persisted prompt-cache artifact for one completed text request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxLmPromptCacheArtifact {
    /// Stable schema version.
    pub schema_version: u32,
    /// Request identifier that produced the artifact.
    pub request_id: String,
    /// Model identifier used for the request.
    pub model_id: String,
    /// Stable served-artifact digest for the request path.
    pub served_artifact_digest: String,
    /// Stable execution-plan digest for the request path.
    pub execution_plan_digest: String,
    /// Prompt-token count recorded by the response.
    pub input_tokens: usize,
    /// Output-token count recorded by the response.
    pub output_tokens: usize,
    /// Retained cache-token count recorded by the response.
    pub cache_tokens: usize,
    /// Prefix tokens reused by the request, when measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_tokens_reused: Option<usize>,
    /// Context-window accounting for the request, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<psionic_models::ContextWindowAccounting>,
    /// Request-level shared-prefix cache control.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_control: Option<PrefixCacheControl>,
    /// Observable shared-prefix cache state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_state: Option<PrefixCacheState>,
    /// Explicit refusal reason when the runtime bypassed reuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_refusal_reason: Option<PrefixCacheRefusalReason>,
    /// Shared-prefix reuse policy for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_policy: Option<PrefixCacheReusePolicy>,
    /// Shared-prefix cache identity when one was used or rebuilt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache_identity: Option<PrefixCacheIdentity>,
    /// Active paged-KV policy for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_policy: Option<KvCachePolicy>,
    /// Runtime cache observations captured for the request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cache_observations: Vec<CacheObservation>,
    /// Scheduler receipt for the request, when continuous batching owned the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<GenerationSchedulerRequestReceipt>,
}

impl MlxLmPromptCacheArtifact {
    /// Builds one prompt-cache artifact from a completed response.
    #[must_use]
    pub fn from_response(response: &GenerationResponse) -> Option<Self> {
        let provenance = response.provenance.as_ref()?;
        Some(Self {
            schema_version: 1,
            request_id: response.request_id.clone(),
            model_id: response.model_id.clone(),
            served_artifact_digest: provenance.served_artifact.served_artifact_digest.clone(),
            execution_plan_digest: provenance.execution_plan_digest.clone(),
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            cache_tokens: response.usage.cache_tokens,
            prefix_tokens_reused: response.metrics.prefix_tokens_reused,
            context_window: response.metrics.context_window.clone(),
            prefix_cache_control: provenance.prefix_cache_control.clone(),
            prefix_cache_state: provenance.prefix_cache_state,
            prefix_cache_refusal_reason: provenance.prefix_cache_refusal_reason,
            prefix_cache_policy: provenance.prefix_cache_policy.clone(),
            prefix_cache_identity: provenance.prefix_cache_identity.clone(),
            kv_cache_policy: provenance.kv_cache_policy.clone(),
            cache_observations: provenance.cache_observations.clone(),
            scheduler: provenance.scheduler.clone(),
        })
    }

    /// Saves one prompt-cache artifact to JSON on disk.
    pub fn save_path(&self, path: impl AsRef<Path>) -> Result<(), MlxLmError> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, format!("{json}\n"))?;
        Ok(())
    }

    /// Loads one prompt-cache artifact from JSON on disk.
    pub fn load_path(path: impl AsRef<Path>) -> Result<Self, MlxLmError> {
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// One serializable batch result item.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum MlxLmBatchItemStatus {
    /// The request succeeded.
    Ok {
        /// Realized response.
        response: GenerationResponse,
        /// Optional persisted prompt-cache artifact view for the response.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_artifact: Option<MlxLmPromptCacheArtifact>,
    },
    /// The request failed.
    Error {
        /// Human-readable failure message.
        message: String,
        /// Typed runtime diagnostic for the failure.
        diagnostic: LocalRuntimeDiagnostic,
    },
}

/// One serializable batch result row.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxLmBatchItem {
    /// Stable request identifier.
    pub request_id: String,
    /// Realized result.
    #[serde(flatten)]
    pub status: MlxLmBatchItemStatus,
}

/// Serializable batch report for one continuous-batching run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxLmBatchReport {
    /// Responses in caller-supplied order.
    pub items: Vec<MlxLmBatchItem>,
    /// Aggregate scheduler metrics for the run.
    pub scheduler_metrics: GenerationSchedulerMetrics,
}

impl MlxLmBatchReport {
    fn from_raw(
        request_ids: Vec<String>,
        result: ContinuousBatchGenerationResult,
    ) -> Self {
        let items = result
            .responses
            .into_iter()
            .zip(request_ids)
            .map(|(response, request_id)| match response {
                Ok(response) => MlxLmBatchItem {
                    request_id,
                    status: MlxLmBatchItemStatus::Ok {
                        prompt_cache_artifact: MlxLmPromptCacheArtifact::from_response(&response),
                        response,
                    },
                },
                Err(error) => MlxLmBatchItem {
                    request_id,
                    status: MlxLmBatchItemStatus::Error {
                        message: error.to_string(),
                        diagnostic: error.diagnostic(),
                    },
                },
            })
            .collect();
        Self {
            items,
            scheduler_metrics: result.scheduler_metrics,
        }
    }
}

/// Package-local error surface.
#[derive(Debug, Error)]
pub enum MlxLmError {
    /// Opening the local GGUF blob failed.
    #[error(transparent)]
    Blob(#[from] BlobError),
    /// Loading GGUF metadata or weights failed.
    #[error(transparent)]
    Model(#[from] psionic_models::ModelLoadError),
    /// Text-generation execution failed.
    #[error(transparent)]
    Runtime(#[from] ReferenceTextGenerationError),
    /// Prompt rendering failed.
    #[error(transparent)]
    PromptRender(#[from] PromptRenderError),
    /// JSON serialization or decoding failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Reading or writing a package artifact failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The response did not carry prompt-cache provenance.
    #[error("generation response `{request_id}` carried no prompt-cache provenance")]
    MissingPromptCacheProvenance {
        /// Request identifier that lacked provenance.
        request_id: String,
    },
}

/// Bounded local MLX-lm-style text runtime over one GGUF model path.
#[derive(Debug)]
pub struct MlxLmTextRuntime {
    model_path: PathBuf,
    blob_metadata: LocalBlobMetadata,
    adapter: GgufDecoderAdapter,
    service: CpuGgufTextGenerationService,
}

impl MlxLmTextRuntime {
    /// Opens one local GGUF text package.
    pub fn from_gguf_path(path: impl AsRef<Path>) -> Result<Self, MlxLmError> {
        let model_path = path.as_ref().to_path_buf();
        let blob = LocalBlob::open_path(
            &model_path,
            LocalBlobKind::GgufFile,
            LocalBlobOpenOptions::default().with_integrity_policy(BlobIntegrityPolicy::Sha256),
        )?;
        let adapter = GgufDecoderAdapterLoader.load_path(&model_path)?;
        let service = CpuGgufTextGenerationService::from_gguf_path(&model_path)?;
        Ok(Self {
            model_path,
            blob_metadata: blob.metadata().clone(),
            adapter,
            service,
        })
    }

    /// Returns the active decoder descriptor.
    #[must_use]
    pub fn model_descriptor(&self) -> &DecoderModelDescriptor {
        self.service.model_descriptor()
    }

    /// Returns the stable local blob metadata for the loaded model.
    #[must_use]
    pub fn blob_metadata(&self) -> &LocalBlobMetadata {
        &self.blob_metadata
    }

    /// Returns one load report for the current package instance.
    #[must_use]
    pub fn load_report(&self) -> MlxLmLoadReport {
        MlxLmLoadReport {
            model_path: self.model_path.clone(),
            blob_metadata: self.blob_metadata.clone(),
            descriptor: self.model_descriptor().clone(),
            runtime_support: self.service.runtime_support(),
            chat_template_digest: self.adapter.chat_templates().digest().to_string(),
            has_default_chat_template: self.adapter.chat_templates().default_template().is_some(),
            named_chat_templates: self
                .adapter
                .chat_templates()
                .named_templates()
                .keys()
                .cloned()
                .collect(),
        }
    }

    /// Renders one chat prompt through the loaded GGUF chat-template metadata.
    pub fn render_chat_prompt(
        &self,
        template_name: Option<&str>,
        messages: &[PromptMessage],
        add_generation_prompt: bool,
        options: &PromptRenderOptions,
    ) -> Result<RenderedPrompt, MlxLmError> {
        Ok(self.adapter.render_prompt_with_options(
            template_name,
            messages,
            add_generation_prompt,
            options,
        )?)
    }

    fn build_generation_request(&self, request: MlxLmTextRequest) -> GenerationRequest {
        GenerationRequest::new_text(
            request.request_id,
            self.model_descriptor().clone(),
            request.session_id,
            request.prompt,
            request.options,
        )
        .with_prefix_cache_control(request.prefix_cache_control)
        .with_reset_session(request.reset_session)
    }

    /// Executes one text-generation request.
    pub fn generate_text(
        &mut self,
        request: MlxLmTextRequest,
    ) -> Result<GenerationResponse, MlxLmError> {
        Ok(self.service.generate(&self.build_generation_request(request))?)
    }

    /// Starts one pull-driven generation stream.
    pub fn generate_stream<'a>(
        &'a mut self,
        request: MlxLmTextRequest,
    ) -> Result<Box<dyn GenerationEventStream + 'a>, MlxLmError> {
        Ok(self
            .service
            .generate_stream(&self.build_generation_request(request))?)
    }

    /// Executes one continuous-batching run and returns a serializable report.
    #[must_use]
    pub fn generate_batch_report(&mut self, requests: Vec<MlxLmTextRequest>) -> MlxLmBatchReport {
        let request_ids = requests
            .iter()
            .map(|request| request.request_id.clone())
            .collect::<Vec<_>>();
        let requests = requests
            .into_iter()
            .map(|request| self.build_generation_request(request))
            .collect();
        MlxLmBatchReport::from_raw(request_ids, self.service.generate_continuous_batch(requests))
    }

    /// Refreshes keepalive for an already loaded model.
    pub fn warm_model(
        &mut self,
        keep_alive_millis: u64,
    ) -> Result<LoadedModelView, MlxLmError> {
        let model_id = self.model_descriptor().model.model_id.clone();
        Ok(self.service.warm_model(model_id.as_str(), keep_alive_millis)?)
    }

    /// Unloads the currently loaded model.
    pub fn unload_model(&mut self) -> Result<LoadedModelView, MlxLmError> {
        let model_id = self.model_descriptor().model.model_id.clone();
        Ok(self.service.unload_model(model_id.as_str())?)
    }

    /// Returns the currently loaded-model observation.
    #[must_use]
    pub fn loaded_models(&mut self) -> LoadedModelsObservation {
        self.service.loaded_models()
    }

    /// Returns the loaded-model lifecycle views in the current runtime order.
    #[must_use]
    pub fn loaded_model_views(&mut self) -> Vec<LoadedModelView> {
        self.service.loaded_model_views()
    }

    /// Returns runtime observability for the local package path.
    #[must_use]
    pub fn observability(&mut self) -> LocalRuntimeObservability {
        self.service.observability()
    }

    /// Saves one prompt-cache artifact for a completed response.
    pub fn save_prompt_cache_artifact(
        &self,
        response: &GenerationResponse,
        path: impl AsRef<Path>,
    ) -> Result<MlxLmPromptCacheArtifact, MlxLmError> {
        let Some(artifact) = MlxLmPromptCacheArtifact::from_response(response) else {
            return Err(MlxLmError::MissingPromptCacheProvenance {
                request_id: response.request_id.clone(),
            });
        };
        artifact.save_path(path)?;
        Ok(artifact)
    }
}

#[cfg(test)]
mod tests {
    use super::{MlxLmPromptCacheArtifact, MlxLmTextRequest, MlxLmTextRuntime};
    use std::{fs, path::Path};

    use psionic_models::{
        ContextOverflowPolicy, GgufDecoderFamily, GgufMetadataValue, GgufTensorType,
        GoldenPromptMessage, GoldenPromptRole, PromptMessage, PromptMessageRole,
        PromptRenderOptions, GgufDecoderAdapterLoader, golden_prompt_fixture,
    };
    use psionic_runtime::{PrefixCacheControl, PrefixCacheMode, PrefixCacheState};
    use psionic_serve::{GenerationOptions, GenerationStreamEvent};
    use tempfile::tempdir;

    #[test]
    fn local_package_loads_qwen_gguf_and_reports_template_truth()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = write_qwen_fixture(temp.path())?;

        let runtime = MlxLmTextRuntime::from_gguf_path(&path)?;
        let report = runtime.load_report();
        let adapter = GgufDecoderAdapterLoader.load_path(&path)?;

        assert_eq!(report.descriptor.model.family, "qwen");
        assert_eq!(report.runtime_support.family, GgufDecoderFamily::Qwen);
        assert_eq!(report.chat_template_digest, adapter.chat_templates().digest());
        assert!(report.has_default_chat_template);
        Ok(())
    }

    #[test]
    fn local_package_renders_chat_prompt_from_loaded_template()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = write_qwen_fixture(temp.path())?;
        let runtime = MlxLmTextRuntime::from_gguf_path(&path)?;

        let fixture = golden_prompt_fixture("qwen2").expect("qwen2 fixture");
        let render_case = fixture
            .template_variant("qwen2.default")
            .and_then(|variant| variant.render_case("qwen2.with_system_history"))
            .expect("qwen2 render case");
        let messages = render_case
            .messages
            .iter()
            .copied()
            .map(prompt_message_from_fixture)
            .collect::<Vec<_>>();

        let rendered = runtime.render_chat_prompt(
            None,
            messages.as_slice(),
            render_case.add_generation_prompt,
            &PromptRenderOptions::default(),
        )?;

        assert_eq!(rendered.text, render_case.expected_rendered);
        Ok(())
    }

    #[test]
    fn local_package_streams_and_persists_prompt_cache_artifact()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = write_qwen_fixture(temp.path())?;
        let mut runtime = MlxLmTextRuntime::from_gguf_path(&path)?;

        let request = || {
            MlxLmTextRequest::new("mlx-lm-cache", "hello", GenerationOptions::greedy(1))
                .with_prefix_cache_control(PrefixCacheControl {
                    mode: PrefixCacheMode::Auto,
                    tenant_id: None,
                })
        };

        let _first = runtime.generate_text(request())?;
        let response = runtime.generate_text(MlxLmTextRequest::new(
            "mlx-lm-cache-hit",
            "hello",
            GenerationOptions::greedy(1),
        )
        .with_prefix_cache_control(PrefixCacheControl {
            mode: PrefixCacheMode::Auto,
            tenant_id: None,
        }))?;

        assert_eq!(response.output.text, "world");
        assert_eq!(
            response
                .provenance
                .as_ref()
                .and_then(|value| value.prefix_cache_state),
            Some(PrefixCacheState::Hit)
        );

        let stream_request = MlxLmTextRequest::new(
            "mlx-lm-stream",
            "hello",
            GenerationOptions::greedy(1),
        );
        let mut stream = runtime.generate_stream(stream_request)?;
        let Some(GenerationStreamEvent::Chunk(chunk)) = stream.next_event() else {
            panic!("expected streamed chunk");
        };
        assert_eq!(chunk.output.text, "world");
        let Some(GenerationStreamEvent::Terminal(terminal)) = stream.next_event() else {
            panic!("expected terminal stream event");
        };
        assert_eq!(terminal.response.output.text, "world");
        drop(stream);

        let artifact_path = temp.path().join("prompt_cache.json");
        let artifact = runtime.save_prompt_cache_artifact(&response, &artifact_path)?;
        let loaded = MlxLmPromptCacheArtifact::load_path(&artifact_path)?;
        assert_eq!(artifact, loaded);
        assert_eq!(loaded.prefix_cache_state, Some(PrefixCacheState::Hit));
        Ok(())
    }

    #[test]
    fn local_package_batches_requests_and_reports_context_and_kv_cache_truth()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempdir()?;
        let path = write_qwen_fixture_with_context(temp.path(), 8)?;
        let mut runtime = MlxLmTextRuntime::from_gguf_path(&path)?;

        let rotated = runtime.generate_text(
            MlxLmTextRequest::new(
                "mlx-lm-rotate",
                "hello world",
                GenerationOptions::greedy(6),
            )
            .with_context_overflow_policy(ContextOverflowPolicy::TruncateOldest),
        )?;
        assert_eq!(
            rotated
                .metrics
                .context_window
                .as_ref()
                .map(|value| value.budget.max_context_tokens),
            Some(8)
        );
        assert_eq!(
            rotated
                .metrics
                .context_window
                .as_ref()
                .map(|value| value.budget.reserved_output_tokens),
            Some(6)
        );
        assert!(
            rotated
                .provenance
                .as_ref()
                .and_then(|value| value.kv_cache_policy.as_ref())
                .is_some()
        );
        assert!(rotated.metrics.kv_cache.is_some());

        let batch = runtime.generate_batch_report(vec![
            MlxLmTextRequest::new("mlx-lm-batch-1", "hello", GenerationOptions::greedy(1)),
            MlxLmTextRequest::new("mlx-lm-batch-2", "hello", GenerationOptions::greedy(1)),
        ]);

        assert_eq!(batch.items.len(), 2);
        assert!(batch.scheduler_metrics.max_batch_size >= 2);
        assert!(batch.items.iter().all(|item| matches!(
            item.status,
            super::MlxLmBatchItemStatus::Ok { .. }
        )));
        Ok(())
    }

    fn prompt_message_from_fixture(message: GoldenPromptMessage) -> PromptMessage {
        PromptMessage {
            role: match message.role {
                GoldenPromptRole::System => PromptMessageRole::System,
                GoldenPromptRole::Developer => PromptMessageRole::Developer,
                GoldenPromptRole::User => PromptMessageRole::User,
                GoldenPromptRole::Assistant => PromptMessageRole::Assistant,
                GoldenPromptRole::Tool => PromptMessageRole::Tool,
            },
            content: String::from(message.content),
            author_name: None,
            recipient: None,
            channel: None,
            content_type: None,
            reasoning_content: None,
        }
    }

    fn write_qwen_fixture(root: &Path) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
        write_qwen_fixture_with_context(root, 128)
    }

    fn write_qwen_fixture_with_context(
        root: &Path,
        context_length: u32,
    ) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
        let path = root.join("tiny_qwen2.gguf");
        let template = golden_prompt_fixture("qwen2")
            .and_then(|fixture| fixture.template_variant("qwen2.default"))
            .and_then(|variant| variant.raw_template)
            .expect("qwen2 raw template fixture");
        write_test_gguf(
            &path,
            qwen2_metadata("Tiny Qwen2", Some(template), context_length).as_slice(),
            dense_decoder_tensors(true, 2, 3).as_slice(),
        )?;
        Ok(path)
    }

    fn qwen2_metadata(
        name: &str,
        chat_template: Option<&str>,
        context_length: u32,
    ) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = vec![
            (
                String::from("general.architecture"),
                GgufMetadataValue::String(String::from("qwen2")),
            ),
            (
                String::from("general.name"),
                GgufMetadataValue::String(name.to_string()),
            ),
            (
                String::from("qwen2.context_length"),
                GgufMetadataValue::U32(context_length),
            ),
            (
                String::from("qwen2.embedding_length"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen2.feed_forward_length"),
                GgufMetadataValue::U32(8),
            ),
            (String::from("qwen2.block_count"), GgufMetadataValue::U32(1)),
            (
                String::from("qwen2.attention.head_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen2.attention.head_count_kv"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("qwen2.attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::F32(1e-6),
            ),
            (
                String::from("qwen2.rope.freq_base"),
                GgufMetadataValue::F32(1_000_000.0),
            ),
            (
                String::from("qwen2.attention.sliding_window"),
                GgufMetadataValue::U32(32),
            ),
            (
                String::from("tokenizer.ggml.model"),
                GgufMetadataValue::String(String::from("gpt2")),
            ),
            (
                String::from("tokenizer.ggml.pre"),
                GgufMetadataValue::String(String::from("qwen2")),
            ),
            (
                String::from("tokenizer.ggml.tokens"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String(String::from("<|bos|>")),
                    GgufMetadataValue::String(String::from("<|eos|>")),
                    GgufMetadataValue::String(String::from("hello")),
                    GgufMetadataValue::String(String::from("world")),
                    GgufMetadataValue::String(String::from("psionic")),
                    GgufMetadataValue::String(String::from("agent")),
                ]),
            ),
            (
                String::from("tokenizer.ggml.merges"),
                GgufMetadataValue::Array(vec![GgufMetadataValue::String(String::from(
                    "hello world",
                ))]),
            ),
            (
                String::from("tokenizer.ggml.bos_token_id"),
                GgufMetadataValue::U32(0),
            ),
            (
                String::from("tokenizer.ggml.eos_token_id"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("tokenizer.ggml.unknown_token_id"),
                GgufMetadataValue::U32(0),
            ),
            (
                String::from("tokenizer.ggml.add_bos_token"),
                GgufMetadataValue::Bool(false),
            ),
            (
                String::from("tokenizer.ggml.add_eos_token"),
                GgufMetadataValue::Bool(false),
            ),
        ];
        if let Some(chat_template) = chat_template {
            metadata.push((
                String::from("tokenizer.chat_template"),
                GgufMetadataValue::String(chat_template.to_string()),
            ));
        }
        metadata
    }

    fn dense_decoder_tensors(
        include_qkv_bias: bool,
        hello_token_index: usize,
        world_token_index: usize,
    ) -> Vec<TestGgufTensor> {
        let mut tensors = vec![
            dense_tensor(
                "token_embd.weight",
                vec![6, 4],
                token_embedding_values(hello_token_index),
            ),
            dense_tensor("output_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            dense_tensor(
                "output.weight",
                vec![6, 4],
                output_values(world_token_index),
            ),
            dense_tensor("blk.0.attn_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            dense_tensor("blk.0.attn_q.weight", vec![4, 4], vec![0.0; 16]),
            dense_tensor("blk.0.attn_k.weight", vec![2, 4], vec![0.0; 8]),
            dense_tensor("blk.0.attn_v.weight", vec![2, 4], vec![0.0; 8]),
            dense_tensor("blk.0.attn_output.weight", vec![4, 4], vec![0.0; 16]),
            dense_tensor("blk.0.ffn_gate.weight", vec![8, 4], vec![0.0; 32]),
            dense_tensor("blk.0.ffn_down.weight", vec![4, 8], vec![0.0; 32]),
            dense_tensor("blk.0.ffn_up.weight", vec![8, 4], vec![0.0; 32]),
            dense_tensor("blk.0.ffn_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
        ];
        if include_qkv_bias {
            tensors.push(dense_tensor("blk.0.attn_q.bias", vec![4], vec![0.0; 4]));
            tensors.push(dense_tensor("blk.0.attn_k.bias", vec![2], vec![0.0; 2]));
            tensors.push(dense_tensor("blk.0.attn_v.bias", vec![2], vec![0.0; 2]));
        }
        tensors
    }

    fn token_embedding_values(hello_token_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; 6 * 4];
        values[hello_token_index.saturating_mul(4)] = 2.0;
        values
    }

    fn output_values(world_token_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; 6 * 4];
        values[world_token_index.saturating_mul(4)] = 1.0;
        values
    }

    #[derive(Clone, Debug)]
    struct TestGgufTensor {
        name: String,
        shape: Vec<usize>,
        tensor_type: GgufTensorType,
        bytes: Vec<u8>,
    }

    impl TestGgufTensor {
        fn new(
            name: impl Into<String>,
            shape: Vec<usize>,
            tensor_type: GgufTensorType,
            bytes: Vec<u8>,
        ) -> Self {
            Self {
                name: name.into(),
                shape,
                tensor_type,
                bytes,
            }
        }
    }

    fn dense_tensor(name: &str, shape: Vec<usize>, values: Vec<f32>) -> TestGgufTensor {
        TestGgufTensor::new(name, shape, GgufTensorType::F32, encode_f32_bytes(values.as_slice()))
    }

    fn encode_f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    fn write_test_gguf(
        path: &Path,
        metadata: &[(String, GgufMetadataValue)],
        tensors: &[TestGgufTensor],
    ) -> Result<(), Box<dyn std::error::Error>> {
        fs::write(path, build_test_gguf(metadata, tensors)?)?;
        Ok(())
    }

    fn build_test_gguf(
        metadata: &[(String, GgufMetadataValue)],
        tensors: &[TestGgufTensor],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let alignment = metadata
            .iter()
            .find(|(key, _)| key == "general.alignment")
            .and_then(|(_, value)| match value {
                GgufMetadataValue::U64(value) => Some(*value as usize),
                GgufMetadataValue::U32(value) => Some(*value as usize),
                _ => None,
            })
            .unwrap_or(32)
            .max(1);

        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, u64::try_from(tensors.len())?);
        push_u64(&mut bytes, u64::try_from(metadata.len())?);

        for (key, value) in metadata {
            push_gguf_string(&mut bytes, key)?;
            push_u32(&mut bytes, gguf_metadata_value_type(value));
            push_gguf_value(&mut bytes, value)?;
        }

        let mut next_offset = 0usize;
        let mut tensor_offsets = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            tensor_offsets.push(next_offset);
            next_offset = align_usize(next_offset + tensor.bytes.len(), alignment);
        }

        for (tensor, offset) in tensors.iter().zip(&tensor_offsets) {
            push_gguf_string(&mut bytes, tensor.name.as_str())?;
            push_u32(&mut bytes, u32::try_from(tensor.shape.len())?);
            for dimension in tensor.shape.iter().rev() {
                push_u64(&mut bytes, u64::try_from(*dimension)?);
            }
            push_u32(&mut bytes, gguf_tensor_type_code(tensor.tensor_type));
            push_u64(&mut bytes, u64::try_from(*offset)?);
        }

        let tensor_data_offset = align_usize(bytes.len(), alignment);
        bytes.resize(tensor_data_offset, 0);

        for (tensor, offset) in tensors.iter().zip(&tensor_offsets) {
            let start = tensor_data_offset + offset;
            if bytes.len() < start {
                bytes.resize(start, 0);
            }
            bytes.extend_from_slice(tensor.bytes.as_slice());
            bytes.resize(align_usize(bytes.len(), alignment), 0);
        }

        Ok(bytes)
    }

    fn align_usize(value: usize, alignment: usize) -> usize {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + alignment - remainder
        }
    }

    fn gguf_metadata_value_type(value: &GgufMetadataValue) -> u32 {
        match value {
            GgufMetadataValue::U8(_) => 0,
            GgufMetadataValue::I8(_) => 1,
            GgufMetadataValue::U16(_) => 2,
            GgufMetadataValue::I16(_) => 3,
            GgufMetadataValue::U32(_) => 4,
            GgufMetadataValue::I32(_) => 5,
            GgufMetadataValue::F32(_) => 6,
            GgufMetadataValue::Bool(_) => 7,
            GgufMetadataValue::String(_) => 8,
            GgufMetadataValue::Array(_) => 9,
            GgufMetadataValue::U64(_) => 10,
            GgufMetadataValue::I64(_) => 11,
            GgufMetadataValue::F64(_) => 12,
        }
    }

    fn gguf_tensor_type_code(tensor_type: GgufTensorType) -> u32 {
        match tensor_type {
            GgufTensorType::F32 => 0,
            other => panic!("unsupported synthetic gguf tensor type: {other:?}"),
        }
    }

    fn push_gguf_string(
        bytes: &mut Vec<u8>,
        value: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        push_u64(bytes, u64::try_from(value.len())?);
        bytes.extend_from_slice(value.as_bytes());
        Ok(())
    }

    fn push_gguf_value(
        bytes: &mut Vec<u8>,
        value: &GgufMetadataValue,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match value {
            GgufMetadataValue::U8(value) => bytes.push(*value),
            GgufMetadataValue::I8(value) => bytes.push(value.to_le_bytes()[0]),
            GgufMetadataValue::U16(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::I16(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::U32(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::I32(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::U64(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::I64(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::F32(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::F64(value) => bytes.extend(value.to_le_bytes()),
            GgufMetadataValue::Bool(value) => bytes.push(u8::from(*value)),
            GgufMetadataValue::String(value) => push_gguf_string(bytes, value)?,
            GgufMetadataValue::Array(values) => {
                let value_type = values.first().map_or(4, gguf_metadata_value_type);
                push_u32(bytes, value_type);
                push_u64(bytes, u64::try_from(values.len())?);
                for value in values {
                    push_gguf_value(bytes, value)?;
                }
            }
        }
        Ok(())
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend(value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend(value.to_le_bytes());
    }
}
