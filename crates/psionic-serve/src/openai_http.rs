use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    convert::Infallible,
    env,
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::Duration,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use futures_util::stream::{self, Stream, StreamExt};
use psionic_catalog::{BlobIntegrityPolicy, LocalBlobOpenOptions};
use psionic_models::{
    GgufBlobArtifact, GgufDecoderFamily, GgufPromptTemplateRenderer, GptOssHarmonyParseOptions,
    GptOssHarmonyParsedOutput, GptOssHarmonyRenderContext, GptOssTokenizer,
    ParsedReasoningResponse, PromptChannelConfig, PromptMessage, PromptMessageRole,
    PromptReasoningEffort, PromptRenderOptions, Qwen35MultimodalProjectionConfig, ReasoningParser,
    parse_gpt_oss_harmony_text, parse_reasoning_response_text_for_decoder_family,
    reasoning_parser_for_decoder_family, render_gpt_oss_harmony_prompt,
};
use psionic_net::{
    PersistedImportedJoinBundle, PersistedJoinedMeshPreference, ServedMeshRole, ServedMeshRoleState,
};
use psionic_router::{
    FleetRouter, ResponseConversationRef, ResponseStateCapability, ResponseStateError,
    ResponseStateRecord, ResponseStateRetentionPolicy, ResponseStateStore, RouteSelection,
    RouteSelectionStrategy, RoutedModelInventory, RoutedWarmState, RoutedWorkerInventory,
    RoutingEndpoint, RoutingError, RoutingRequest, RoutingTarget,
};
use psionic_runtime::{
    ExecutionCapabilityProfile, GenerationSchedulerPolicy, GenerationSchedulerRequestReceipt,
    PrefixCacheControl, PrefixCacheRefusalReason, PrefixCacheState, StructuredGrammarSyntax,
    StructuredOutputCapability, StructuredOutputExecutionReport, StructuredOutputMatcher,
    StructuredOutputParser, StructuredOutputRequest, StructuredOutputValue,
    StructuredTaggedVariant, local_structured_output_capabilities, local_structured_output_parsers,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    net::TcpListener,
    sync::{broadcast, mpsc, oneshot},
};
use tokio_stream::iter;

use crate::{
    CpuGgufTextGenerationService, CpuModelEmbeddingsService, CudaGemma4TextGenerationService,
    CudaGgufGptOssTextGenerationService, CudaGgufQwen35TextGenerationService,
    CudaGptOssTextGenerationError, DecodeStrategy, DecoderModelDescriptor, EmbeddingMetrics,
    EmbeddingNormalization, EmbeddingProvenance, EmbeddingRequest, EmbeddingResponse,
    EmbeddingsExecutor, GenerationMetrics, GenerationOptions, GenerationRequest,
    GgufDecoderAdapterLoader, GptOssPerformanceMetrics, MetalGgufGptOssTextGenerationService,
    MetalGptOssTextGenerationError, ModelEmbeddingsError, PromptRenderError,
    ReferenceTextGenerationError, TerminationReason, TextGenerationExecutor, TokenSequence,
    continuous_batch_text_generation_execution_profile, default_embeddings_execution_profile,
    default_generation_scheduler_policy, default_text_generation_execution_profile,
    tokio_runtime_telemetry_axum::serve_with_runtime_telemetry,
};

mod tassadar_post_article_router_plugin_tool_loop_pilot;

pub use tassadar_post_article_router_plugin_tool_loop_pilot::*;

const DEFAULT_MAX_TOKENS: usize = 256;
const HARMONY_RETURN_STOP: &str = "<|return|>";
const HARMONY_CALL_STOP: &str = "<|call|>";
const CPU_SERVER_RESIDENCY_MODE: &str = "cpu_only";
const CPU_SERVER_HYBRID_OFFLOAD_MODE: &str = "unsupported";
const CPU_SERVER_FALLBACK_POLICY: &str = "refuse";
const CPU_SERVER_PERFORMANCE_CLASS: &str = "portable_cpu_degraded";
const LLAMA_CPP_PROXY_RESIDENCY_MODE: &str = "llama_cpp_proxy";
const PROXY_ONLY_FALLBACK_POLICY: &str = "proxy_only";
const CPU_PROXY_PERFORMANCE_CLASS: &str = "portable_cpu_proxy";
const BOOTSTRAP_PROXY_RESIDENCY_MODE: &str = "bootstrap_proxy";
const BOOTSTRAP_PROXY_FALLBACK_POLICY: &str = "remote_bootstrap";
const BOOTSTRAP_PROXY_PERFORMANCE_CLASS: &str = "bootstrap_proxy";
const LOCAL_SERVER_LOAD_STATUS: &str = "loaded";
const LOCAL_SERVER_WARM_CONTROL: &str = "not_implemented";
const LOCAL_SERVER_UNLOAD_CONTROL: &str = "not_implemented";
const LOCAL_SERVER_MEMORY_PRESSURE_REPORTING: &str = "not_implemented";
const OPENAI_COMPAT_WORKER_ID: &str = "local_cpu_0";
const STREAMING_TEXT_DELTA_MAX_CHARS: usize = 8;
const BOOTSTRAP_PROXY_WORKER_PREFIX: &str = "bootstrap:";
const BOOTSTRAP_PROXY_PROVENANCE: &str = "bootstrap_proxy";
const LOCAL_EXECUTION_PROVENANCE: &str = "local_execution";
const THIN_CLIENT_FALLBACK_POSTURE: &str = "thin_client_remote_only";
const WARMING_FALLBACK_POSTURE: &str = "warming_until_local_ready";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LocalServingTruth {
    residency_mode: &'static str,
    hybrid_offload: &'static str,
    hybrid_offload_layers: Option<i32>,
    fallback_policy: &'static str,
    performance_class: &'static str,
    load_status: &'static str,
    warm_control: &'static str,
    unload_control: &'static str,
    memory_pressure_reporting: &'static str,
}

impl LocalServingTruth {
    const fn cpu_reference() -> Self {
        Self {
            residency_mode: CPU_SERVER_RESIDENCY_MODE,
            hybrid_offload: CPU_SERVER_HYBRID_OFFLOAD_MODE,
            hybrid_offload_layers: None,
            fallback_policy: CPU_SERVER_FALLBACK_POLICY,
            performance_class: CPU_SERVER_PERFORMANCE_CLASS,
            load_status: LOCAL_SERVER_LOAD_STATUS,
            warm_control: LOCAL_SERVER_WARM_CONTROL,
            unload_control: LOCAL_SERVER_UNLOAD_CONTROL,
            memory_pressure_reporting: LOCAL_SERVER_MEMORY_PRESSURE_REPORTING,
        }
    }

    const fn cpu_proxy() -> Self {
        Self {
            residency_mode: LLAMA_CPP_PROXY_RESIDENCY_MODE,
            hybrid_offload: CPU_SERVER_HYBRID_OFFLOAD_MODE,
            hybrid_offload_layers: None,
            fallback_policy: PROXY_ONLY_FALLBACK_POLICY,
            performance_class: CPU_PROXY_PERFORMANCE_CLASS,
            load_status: LOCAL_SERVER_LOAD_STATUS,
            warm_control: LOCAL_SERVER_WARM_CONTROL,
            unload_control: LOCAL_SERVER_UNLOAD_CONTROL,
            memory_pressure_reporting: LOCAL_SERVER_MEMORY_PRESSURE_REPORTING,
        }
    }

    const fn bootstrap_proxy() -> Self {
        Self {
            residency_mode: BOOTSTRAP_PROXY_RESIDENCY_MODE,
            hybrid_offload: CPU_SERVER_HYBRID_OFFLOAD_MODE,
            hybrid_offload_layers: None,
            fallback_policy: BOOTSTRAP_PROXY_FALLBACK_POLICY,
            performance_class: BOOTSTRAP_PROXY_PERFORMANCE_CLASS,
            load_status: LOCAL_SERVER_LOAD_STATUS,
            warm_control: LOCAL_SERVER_WARM_CONTROL,
            unload_control: LOCAL_SERVER_UNLOAD_CONTROL,
            memory_pressure_reporting: LOCAL_SERVER_MEMORY_PRESSURE_REPORTING,
        }
    }

    const fn cuda_native() -> Self {
        Self {
            residency_mode: "cuda_accelerated",
            hybrid_offload: "unsupported",
            hybrid_offload_layers: None,
            fallback_policy: "refuse",
            performance_class: "nvidia_native",
            load_status: LOCAL_SERVER_LOAD_STATUS,
            warm_control: LOCAL_SERVER_WARM_CONTROL,
            unload_control: LOCAL_SERVER_UNLOAD_CONTROL,
            memory_pressure_reporting: LOCAL_SERVER_MEMORY_PRESSURE_REPORTING,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OpenAiCompatServingTruth {
    backend_label: &'static str,
    execution_mode_label: &'static str,
    execution_engine_label: &'static str,
    local_serving_truth: LocalServingTruth,
}

impl OpenAiCompatServingTruth {
    const fn cpu_native() -> Self {
        Self {
            backend_label: "cpu",
            execution_mode_label: "native",
            execution_engine_label: "psionic",
            local_serving_truth: LocalServingTruth::cpu_reference(),
        }
    }

    const fn cpu_llama_cpp_proxy() -> Self {
        Self {
            backend_label: "cpu",
            execution_mode_label: "proxy",
            execution_engine_label: "llama.cpp",
            local_serving_truth: LocalServingTruth::cpu_proxy(),
        }
    }

    const fn bootstrap_proxy() -> Self {
        Self {
            backend_label: "remote",
            execution_mode_label: "proxy",
            execution_engine_label: "psionic",
            local_serving_truth: LocalServingTruth::bootstrap_proxy(),
        }
    }

    const fn cuda_native() -> Self {
        Self {
            backend_label: "cuda",
            execution_mode_label: "native",
            execution_engine_label: "psionic",
            local_serving_truth: LocalServingTruth::cuda_native(),
        }
    }
}

fn structured_output_parser_labels() -> Vec<&'static str> {
    local_structured_output_parsers()
        .into_iter()
        .map(StructuredOutputParser::label)
        .collect()
}

fn unsupported_structured_output_capabilities(detail: &str) -> Vec<StructuredOutputCapability> {
    local_structured_output_capabilities()
        .into_iter()
        .map(|capability| StructuredOutputCapability::unsupported(capability.kind, detail))
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ToolCallingSupportLevel {
    Fallback,
    Unsupported,
}

impl ToolCallingSupportLevel {
    #[cfg(test)]
    fn label(self) -> &'static str {
        match self {
            Self::Fallback => "fallback",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ToolCallingCapability {
    support_level: ToolCallingSupportLevel,
    supported_modes: Vec<&'static str>,
    parser: &'static str,
    argument_validation: &'static str,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ResponseContinuationMode {
    #[default]
    AppendTurn,
    ContinueLastAssistant,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
struct PsionicResponseStateRequest {
    #[serde(
        default = "default_response_state_store",
        skip_serializing_if = "is_true"
    )]
    store: bool,
    #[serde(default)]
    continuation: ResponseContinuationMode,
    #[serde(default)]
    invalidate_references: bool,
}

impl Default for PsionicResponseStateRequest {
    fn default() -> Self {
        Self {
            store: true,
            continuation: ResponseContinuationMode::AppendTurn,
            invalidate_references: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ResponseStateReceipt {
    storage: String,
    retention_scope: String,
    cache_behavior: String,
    stored: bool,
    continuation: ResponseContinuationMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    conversation_id: Option<String>,
    replayed_prompt_messages: usize,
    input_messages_appended: usize,
    assistant_messages_recorded: usize,
    max_responses: usize,
    max_conversations: usize,
    max_items_per_conversation: usize,
    conversation_item_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    invalidated_references: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolChoiceMode {
    None,
    Auto,
    Required,
    Named,
}

#[derive(Clone, Debug)]
struct ToolCallingContract {
    tools: BTreeMap<String, ToolDefinitionRequest>,
    mode: ToolChoiceMode,
    named_tool: Option<String>,
    parallel_tool_calls: bool,
    minimum_required_tool_calls: usize,
}

impl ToolCallingContract {
    fn allows_parallel_tool_calls(&self) -> bool {
        self.parallel_tool_calls
            && matches!(self.mode, ToolChoiceMode::Auto | ToolChoiceMode::Required)
    }
}

#[derive(Clone, Debug)]
struct ToolCallOutcome {
    content: Option<String>,
    tool_calls: Vec<ResolvedToolCall>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedReasoningRequest {
    parser: ReasoningParser,
    mode: PsionicReasoningMode,
}

#[derive(Clone, Debug)]
struct ResolvedToolCall {
    id: String,
    name: String,
    arguments: serde_json::Value,
}

impl ResolvedToolCall {
    fn raw_arguments(&self) -> Result<String, OpenAiCompatHttpError> {
        serde_json::to_string(&self.arguments).map_err(|error| {
            OpenAiCompatHttpError::Internal(format!(
                "failed to serialize validated tool arguments for `{}`: {error}",
                self.name
            ))
        })
    }

    fn into_chat_tool_call(self) -> Result<ChatCompletionToolCall, OpenAiCompatHttpError> {
        let raw_arguments = self.raw_arguments()?;
        Ok(ChatCompletionToolCall {
            id: self.id,
            kind: String::from("function"),
            function: ChatCompletionToolCallFunction {
                name: self.name,
                arguments: raw_arguments,
            },
        })
    }

    fn into_psionic_tool_call(self) -> Result<PsionicToolCall, OpenAiCompatHttpError> {
        let raw_arguments = self.raw_arguments()?;
        Ok(PsionicToolCall {
            id: self.id,
            name: self.name,
            arguments: self.arguments,
            raw_arguments,
        })
    }
}

fn tool_loop_tool_call_from_resolved(call: ResolvedToolCall) -> psionic_router::ToolLoopToolCall {
    psionic_router::ToolLoopToolCall {
        id: call.id,
        name: call.name,
        arguments: call.arguments,
    }
}

fn assistant_prompt_message_for_tool_loop(content: Option<String>) -> Option<PromptMessage> {
    content
        .filter(|value| !value.trim().is_empty())
        .map(|value| PromptMessage::new(PromptMessageRole::Assistant, value))
}

#[cfg(test)]
fn tool_result_prompt_message(tool_name: &str, content: impl Into<String>) -> PromptMessage {
    PromptMessage::new(PromptMessageRole::Tool, content).with_author_name(tool_name)
}

fn gpt_oss_local_blob_open_options() -> LocalBlobOpenOptions {
    LocalBlobOpenOptions::default().with_integrity_policy(BlobIntegrityPolicy::LocalUnverifiedLabel)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptOssOpenAiCompatBackend {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

impl GptOssOpenAiCompatBackend {
    fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }

    fn resolve(self) -> Self {
        match self {
            Self::Auto => {
                if cfg!(target_os = "macos") {
                    Self::Metal
                } else {
                    Self::Cuda
                }
            }
            backend => backend,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptOssMetalExecutionMode {
    Auto,
    Native,
    ProxyLlamaCpp,
}

impl GptOssMetalExecutionMode {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Native => "native",
            Self::ProxyLlamaCpp => "proxy",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GptOssOpenAiCompatExecutionSummary {
    backend_label: &'static str,
    execution_mode_label: &'static str,
    execution_engine_label: &'static str,
}

impl GptOssOpenAiCompatExecutionSummary {
    const fn native(backend_label: &'static str) -> Self {
        Self {
            backend_label,
            execution_mode_label: "native",
            execution_engine_label: "psionic",
        }
    }

    const fn metal_proxy() -> Self {
        Self {
            backend_label: "metal",
            execution_mode_label: "proxy",
            execution_engine_label: "llama.cpp",
        }
    }

    fn uses_proxy(self) -> bool {
        matches!(self.execution_engine_label, "llama.cpp")
    }
}

fn resolve_execution_summary(
    backend: GptOssOpenAiCompatBackend,
    metal_mode: GptOssMetalExecutionMode,
    legacy_proxy_enabled: bool,
) -> Result<GptOssOpenAiCompatExecutionSummary, OpenAiCompatServerError> {
    match backend {
        GptOssOpenAiCompatBackend::Metal => match metal_mode {
            GptOssMetalExecutionMode::Auto => Ok(if legacy_proxy_enabled {
                GptOssOpenAiCompatExecutionSummary::metal_proxy()
            } else {
                GptOssOpenAiCompatExecutionSummary::native("metal")
            }),
            GptOssMetalExecutionMode::Native => {
                if legacy_proxy_enabled {
                    Err(OpenAiCompatServerError::Config(String::from(
                        "requested `--metal-mode native` while legacy PSIONIC_METAL_PROXY_LLAMA_CPP is enabled",
                    )))
                } else {
                    Ok(GptOssOpenAiCompatExecutionSummary::native("metal"))
                }
            }
            GptOssMetalExecutionMode::ProxyLlamaCpp => {
                Ok(GptOssOpenAiCompatExecutionSummary::metal_proxy())
            }
        },
        GptOssOpenAiCompatBackend::Cpu => {
            if matches!(metal_mode, GptOssMetalExecutionMode::Auto) {
                Ok(GptOssOpenAiCompatExecutionSummary::native("cpu"))
            } else {
                Err(OpenAiCompatServerError::Config(format!(
                    "requested `--metal-mode {}` but resolved backend is cpu",
                    metal_mode.label()
                )))
            }
        }
        GptOssOpenAiCompatBackend::Cuda => {
            if matches!(metal_mode, GptOssMetalExecutionMode::Auto) {
                Ok(GptOssOpenAiCompatExecutionSummary::native("cuda"))
            } else {
                Err(OpenAiCompatServerError::Config(format!(
                    "requested `--metal-mode {}` but resolved backend is cuda",
                    metal_mode.label()
                )))
            }
        }
        GptOssOpenAiCompatBackend::Auto => Err(OpenAiCompatServerError::Config(String::from(
            "auto backend must be resolved before execution mode selection",
        ))),
    }
}

fn gpt_oss_local_serving_truth(
    config: &GptOssOpenAiCompatConfig,
    summary: GptOssOpenAiCompatExecutionSummary,
) -> LocalServingTruth {
    match (summary.backend_label, summary.execution_engine_label) {
        ("metal", "psionic") => LocalServingTruth {
            residency_mode: "metal_accelerated",
            hybrid_offload: "unsupported",
            hybrid_offload_layers: None,
            fallback_policy: "refuse",
            performance_class: "apple_silicon_native",
            load_status: LOCAL_SERVER_LOAD_STATUS,
            warm_control: LOCAL_SERVER_WARM_CONTROL,
            unload_control: LOCAL_SERVER_UNLOAD_CONTROL,
            memory_pressure_reporting: LOCAL_SERVER_MEMORY_PRESSURE_REPORTING,
        },
        ("metal", "llama.cpp") => LocalServingTruth {
            residency_mode: "llama_cpp_proxy",
            hybrid_offload: "llama_cpp_gpu_layers",
            hybrid_offload_layers: Some(config.gpu_layers.unwrap_or(4)),
            fallback_policy: "proxy_only",
            performance_class: "proxy_control_plane",
            load_status: LOCAL_SERVER_LOAD_STATUS,
            warm_control: LOCAL_SERVER_WARM_CONTROL,
            unload_control: LOCAL_SERVER_UNLOAD_CONTROL,
            memory_pressure_reporting: LOCAL_SERVER_MEMORY_PRESSURE_REPORTING,
        },
        ("cuda", _) => LocalServingTruth {
            residency_mode: "cuda_accelerated",
            hybrid_offload: "host_backed_selected4",
            hybrid_offload_layers: None,
            fallback_policy: "refuse",
            performance_class: "nvidia_native",
            load_status: LOCAL_SERVER_LOAD_STATUS,
            warm_control: LOCAL_SERVER_WARM_CONTROL,
            unload_control: LOCAL_SERVER_UNLOAD_CONTROL,
            memory_pressure_reporting: LOCAL_SERVER_MEMORY_PRESSURE_REPORTING,
        },
        _ => LocalServingTruth::cpu_reference(),
    }
}

#[derive(Clone, Debug)]
pub struct GptOssOpenAiCompatConfig {
    pub model_path: PathBuf,
    pub host: String,
    pub port: u16,
    pub backend: GptOssOpenAiCompatBackend,
    pub context_length: Option<usize>,
    pub gpu_layers: Option<i32>,
    pub metal_mode: GptOssMetalExecutionMode,
    pub reasoning_budget: u8,
    pub webui_enabled: bool,
}

impl GptOssOpenAiCompatConfig {
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            host: String::from("127.0.0.1"),
            port: 8080,
            backend: GptOssOpenAiCompatBackend::Auto,
            context_length: None,
            gpu_layers: None,
            metal_mode: GptOssMetalExecutionMode::Auto,
            reasoning_budget: 0,
            webui_enabled: false,
        }
    }

    pub fn socket_addr(&self) -> Result<SocketAddr, OpenAiCompatServerError> {
        let host = self.host.parse::<IpAddr>().map_err(|error| {
            OpenAiCompatServerError::Config(format!("invalid host `{}`: {error}", self.host))
        })?;
        Ok(SocketAddr::new(host, self.port))
    }
}

#[derive(Clone)]
pub struct GptOssOpenAiCompatServer {
    state: Arc<GptOssOpenAiCompatState>,
}

#[derive(Clone)]
pub struct GptOssCudaOpenAiCompatServer {
    inner: GptOssOpenAiCompatServer,
}

struct GptOssOpenAiCompatState {
    worker: Option<GptOssWorker>,
    proxy: Option<Arc<LlamaCppProxyState>>,
    backend_label: &'static str,
    execution_mode_label: &'static str,
    execution_engine_label: &'static str,
    local_serving_truth: LocalServingTruth,
    descriptor: DecoderModelDescriptor,
    tokenizer: GptOssTokenizer,
    prompt_options: PromptRenderOptions,
    prompt_token_cache: Mutex<PromptTokenCache>,
    default_model_name: String,
    accepted_model_names: BTreeSet<String>,
    include_psionic_fields: bool,
    request_counter: AtomicU64,
}

struct LlamaCppProxyState {
    base_url: String,
    client: reqwest::Client,
    child: Mutex<Option<Child>>,
}

impl Drop for LlamaCppProxyState {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.lock().ok().and_then(|mut child| child.take()) {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[derive(Clone, Debug)]
struct PromptTokenCacheEntry {
    request_key: String,
    tokens: TokenSequence,
}

#[derive(Clone, Debug)]
struct PromptTokenCache {
    entries: VecDeque<PromptTokenCacheEntry>,
    capacity: usize,
}

impl PromptTokenCache {
    const DEFAULT_CAPACITY: usize = 16;

    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    fn lookup(&mut self, request_key: &str) -> Option<TokenSequence> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.request_key == request_key)?;
        let entry = self.entries.remove(index)?;
        let tokens = entry.tokens.clone();
        self.entries.push_front(entry);
        Some(tokens)
    }

    fn record(&mut self, request_key: String, tokens: TokenSequence) {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.request_key == request_key)
        {
            self.entries.remove(index);
        }
        self.entries.push_front(PromptTokenCacheEntry {
            request_key,
            tokens,
        });
        while self.entries.len() > self.capacity {
            self.entries.pop_back();
        }
    }
}

impl GptOssOpenAiCompatServer {
    pub fn from_config(config: &GptOssOpenAiCompatConfig) -> Result<Self, OpenAiCompatServerError> {
        let artifact =
            GgufBlobArtifact::open_path(&config.model_path, gpt_oss_local_blob_open_options())
                .map_err(|error| OpenAiCompatServerError::Config(error.to_string()))?;
        let adapter = GgufDecoderAdapterLoader
            .load_blob_artifact(&artifact)
            .map_err(|error| OpenAiCompatServerError::Config(error.to_string()))?;
        let descriptor = adapter.descriptor().clone();
        let tokenizer = GptOssTokenizer::from_gguf(adapter.tokenizer())
            .map_err(|error| OpenAiCompatServerError::Config(error.to_string()))?;
        let default_model_name =
            default_model_name(&config.model_path, descriptor.model.model_id.as_str());
        let accepted_model_names =
            accepted_model_names(&config.model_path, descriptor.model.model_id.as_str());
        let prompt_options = PromptRenderOptions {
            gpt_oss_harmony: Some(GptOssHarmonyRenderContext {
                reasoning_effort: Some(reasoning_effort(config.reasoning_budget)),
                channel_config: Some(PromptChannelConfig::default()),
                ..Default::default()
            }),
        };
        let include_psionic_fields = env::var("PSIONIC_OPENAI_INCLUDE_DEBUG_FIELDS")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let backend = config.backend.resolve();
        let execution_summary =
            resolve_execution_summary(backend, config.metal_mode, metal_proxy_llama_cpp_enabled())?;
        let local_serving_truth = gpt_oss_local_serving_truth(config, execution_summary);
        let proxy = if execution_summary.uses_proxy() {
            Some(Arc::new(LlamaCppProxyState::spawn(config)?))
        } else {
            None
        };
        Ok(Self {
            state: Arc::new(GptOssOpenAiCompatState {
                worker: if proxy.is_some() {
                    None
                } else {
                    Some(GptOssWorker::spawn(config.model_path.clone(), backend)?)
                },
                proxy,
                backend_label: execution_summary.backend_label,
                execution_mode_label: execution_summary.execution_mode_label,
                execution_engine_label: execution_summary.execution_engine_label,
                local_serving_truth,
                descriptor,
                tokenizer,
                prompt_options,
                prompt_token_cache: Mutex::new(PromptTokenCache::new(
                    PromptTokenCache::DEFAULT_CAPACITY,
                )),
                default_model_name,
                accepted_model_names,
                include_psionic_fields,
                request_counter: AtomicU64::new(1),
            }),
        })
    }

    #[must_use]
    pub fn backend_label(&self) -> &'static str {
        self.state.backend_label
    }

    #[must_use]
    pub fn execution_mode_label(&self) -> &'static str {
        self.state.execution_mode_label
    }

    #[must_use]
    pub fn execution_engine_label(&self) -> &'static str {
        self.state.execution_engine_label
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(Arc::clone(&self.state))
    }

    pub async fn serve(&self, listener: TcpListener) -> Result<(), OpenAiCompatServerError> {
        serve_with_runtime_telemetry(listener, self.router())
            .await
            .map_err(OpenAiCompatServerError::Io)
    }
}

impl GptOssCudaOpenAiCompatServer {
    pub fn from_config(config: &GptOssOpenAiCompatConfig) -> Result<Self, OpenAiCompatServerError> {
        let mut config = config.clone();
        config.backend = GptOssOpenAiCompatBackend::Cuda;
        Ok(Self {
            inner: GptOssOpenAiCompatServer::from_config(&config)?,
        })
    }

    pub fn router(&self) -> Router {
        self.inner.router()
    }

    pub async fn serve(&self, listener: TcpListener) -> Result<(), OpenAiCompatServerError> {
        self.inner.serve(listener).await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenAiCompatBackend {
    Cpu,
    Cuda,
}

#[derive(Clone, Debug)]
pub struct OpenAiCompatConfig {
    pub model_paths: Vec<PathBuf>,
    pub host: String,
    pub port: u16,
    pub backend: OpenAiCompatBackend,
    pub reasoning_budget: u8,
}

impl OpenAiCompatConfig {
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_paths: vec![model_path.into()],
            host: String::from("127.0.0.1"),
            port: 8080,
            backend: OpenAiCompatBackend::Cpu,
            reasoning_budget: 0,
        }
    }

    pub fn add_model_path(&mut self, model_path: impl Into<PathBuf>) {
        self.model_paths.push(model_path.into());
    }

    pub fn socket_addr(&self) -> Result<SocketAddr, OpenAiCompatServerError> {
        let host = self.host.parse::<IpAddr>().map_err(|error| {
            OpenAiCompatServerError::Config(format!("invalid host `{}`: {error}", self.host))
        })?;
        Ok(SocketAddr::new(host, self.port))
    }
}

#[derive(Clone)]
pub struct OpenAiCompatServer {
    state: Arc<OpenAiCompatState>,
}

struct OpenAiCompatState {
    workers: BTreeMap<String, OpenAiCompatWorker>,
    router: FleetRouter,
    backend_label: &'static str,
    execution_mode_label: &'static str,
    execution_engine_label: &'static str,
    default_model_key: String,
    default_model_name: String,
    models_by_key: BTreeMap<String, OpenAiCompatLoadedModel>,
    include_psionic_fields: bool,
    request_counter: AtomicU64,
    conversation_counter: AtomicU64,
    response_state_capability: ResponseStateCapability,
    response_state: Mutex<ResponseStateStore>,
    management_join_state: Mutex<MeshManagementJoinState>,
    management_event_counter: AtomicU64,
    management_events: broadcast::Sender<MeshManagementEventEnvelope>,
    local_management_node: Option<MeshManagementNodeStatus>,
    last_route_execution: Mutex<Option<MeshManagementRouteExecutionStatus>>,
    bootstrap_proxy: Option<Arc<BootstrapProxyState>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BootstrapProxyMode {
    ThinClient,
    Warming,
}

impl BootstrapProxyMode {
    fn from_env() -> Result<Option<Self>, OpenAiCompatServerError> {
        let Some(_) = env::var_os("PSIONIC_BOOTSTRAP_PROXY_BASE_URL") else {
            return Ok(None);
        };
        match env::var("PSIONIC_BOOTSTRAP_PROXY_MODE") {
            Ok(value) if value.eq_ignore_ascii_case("warming") => Ok(Some(Self::Warming)),
            Ok(value)
                if value.eq_ignore_ascii_case("thin_client")
                    || value.eq_ignore_ascii_case("thin-client") =>
            {
                Ok(Some(Self::ThinClient))
            }
            Ok(value) => Err(OpenAiCompatServerError::Config(format!(
                "unsupported `PSIONIC_BOOTSTRAP_PROXY_MODE` `{value}`; expected `thin_client` or `warming`",
            ))),
            Err(_) => Ok(Some(Self::ThinClient)),
        }
    }

    const fn serving_truth(self) -> OpenAiCompatServingTruth {
        OpenAiCompatServingTruth::bootstrap_proxy()
    }

    fn local_role_state(self) -> ServedMeshRoleState {
        match self {
            Self::ThinClient => ServedMeshRoleState::new(ServedMeshRole::ThinClient)
                .with_reason(psionic_net::ServedMeshRoleReason::RemoteOnly),
            Self::Warming => ServedMeshRoleState::new(ServedMeshRole::Host)
                .with_posture(psionic_net::ServedMeshRolePosture::Downgraded)
                .with_reason(psionic_net::ServedMeshRoleReason::Warming),
        }
    }

    const fn local_warm_state(self) -> RoutedWarmState {
        match self {
            Self::ThinClient => RoutedWarmState::Cold,
            Self::Warming => RoutedWarmState::Warming,
        }
    }

    const fn warm_state_reason(self) -> &'static str {
        match self {
            Self::ThinClient => "remote_only",
            Self::Warming => "warming",
        }
    }

    const fn fallback_posture(self) -> &'static str {
        match self {
            Self::ThinClient => THIN_CLIENT_FALLBACK_POSTURE,
            Self::Warming => WARMING_FALLBACK_POSTURE,
        }
    }
}

#[derive(Clone)]
struct BootstrapProxyState {
    base_url: String,
    client: reqwest::Client,
    mode: BootstrapProxyMode,
}

#[derive(Clone, Debug, Deserialize)]
struct BootstrapRemoteManagementStatus {
    default_model: String,
    nodes: Vec<BootstrapRemoteNodeStatus>,
}

#[derive(Clone, Debug, Deserialize)]
struct BootstrapRemoteNodeStatus {
    worker_id: String,
    backend_label: String,
    execution_mode_label: String,
    execution_engine_label: String,
    models: Vec<BootstrapRemoteModelStatus>,
}

#[derive(Clone, Debug, Deserialize)]
struct BootstrapRemoteModelStatus {
    model_key: String,
    canonical_name: String,
    family: String,
    supported_endpoints: Vec<String>,
    warm_state: RoutedWarmState,
    active_requests: usize,
    structured_outputs: bool,
    tool_calling: bool,
    response_state: bool,
    execution_profile: ExecutionCapabilityProfile,
    #[serde(default)]
    scheduler_policy: Option<GenerationSchedulerPolicy>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum MeshManagementJoinPosture {
    Standalone,
    PendingImport,
    Joined,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct MeshManagementJoinState {
    last_joined_mesh_preference: Option<PersistedJoinedMeshPreference>,
    last_imported_join_bundle: Option<PersistedImportedJoinBundle>,
}

impl MeshManagementJoinState {
    fn posture(&self) -> MeshManagementJoinPosture {
        if self.last_joined_mesh_preference.is_some() {
            MeshManagementJoinPosture::Joined
        } else if self.last_imported_join_bundle.is_some() {
            MeshManagementJoinPosture::PendingImport
        } else {
            MeshManagementJoinPosture::Standalone
        }
    }

    fn into_response(self) -> MeshManagementJoinStateResponse {
        MeshManagementJoinStateResponse {
            posture: self.posture(),
            last_joined_mesh_preference: self.last_joined_mesh_preference,
            last_imported_join_bundle: self.last_imported_join_bundle,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct MeshManagementJoinStateResponse {
    posture: MeshManagementJoinPosture,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_joined_mesh_preference: Option<PersistedJoinedMeshPreference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_imported_join_bundle: Option<PersistedImportedJoinBundle>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct MeshManagementModelStatus {
    model_key: String,
    canonical_name: String,
    family: String,
    supported_endpoints: Vec<&'static str>,
    warm_state: RoutedWarmState,
    active_requests: usize,
    structured_outputs: bool,
    tool_calling: bool,
    response_state: bool,
    execution_profile: ExecutionCapabilityProfile,
    #[serde(skip_serializing_if = "Option::is_none")]
    scheduler_policy: Option<GenerationSchedulerPolicy>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct MeshManagementRouteStatus {
    worker_id: String,
    model_key: String,
    canonical_name: String,
    family: String,
    endpoint: &'static str,
    warm_state: RoutedWarmState,
    active_requests: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct MeshManagementNodeStatus {
    worker_id: String,
    served_mesh_role: ServedMeshRoleState,
    backend_label: String,
    execution_mode_label: String,
    execution_engine_label: String,
    models: Vec<MeshManagementModelStatus>,
    route_inventory: Vec<MeshManagementRouteStatus>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct MeshManagementStatusDigestInput {
    join_state: MeshManagementJoinStateResponse,
    nodes: Vec<MeshManagementNodeStatus>,
    routes: Vec<MeshManagementRouteStatus>,
    last_route_execution: Option<MeshManagementRouteExecutionStatus>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct MeshManagementStatusResponse {
    status: &'static str,
    namespace: &'static str,
    topology_digest: String,
    default_model: String,
    node_count: usize,
    model_count: usize,
    event_stream_path: &'static str,
    join_state: MeshManagementJoinStateResponse,
    nodes: Vec<MeshManagementNodeStatus>,
    routes: Vec<MeshManagementRouteStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_route_execution: Option<MeshManagementRouteExecutionStatus>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum MeshManagementRouteExecutionLocality {
    Local,
    RemoteProxy,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct MeshManagementRouteExecutionStatus {
    worker_id: String,
    locality: MeshManagementRouteExecutionLocality,
    provenance: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    warm_state_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fallback_posture: Option<String>,
    executed_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct MeshManagementRoutingRequestSummary {
    endpoint: &'static str,
    target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    preferred_family: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    preferred_worker_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MeshManagementEventPayload {
    TopologySnapshot {
        status: MeshManagementStatusResponse,
    },
    RouteSelection {
        request: MeshManagementRoutingRequestSummary,
        selection: RouteSelection,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct MeshManagementEventEnvelope {
    event_id: u64,
    emitted_at_ms: u64,
    topology_digest: String,
    payload: MeshManagementEventPayload,
}

impl MeshManagementEventEnvelope {
    fn event_name(&self) -> &'static str {
        match &self.payload {
            MeshManagementEventPayload::TopologySnapshot { .. } => "topology_snapshot",
            MeshManagementEventPayload::RouteSelection { .. } => "route_selection",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenAiCompatRuntimeKind {
    GgufDecoderCpu,
    GgufDecoderCudaGemma4,
    GgufDecoderCudaQwen35,
    SafetensorsEmbeddings,
}

#[derive(Clone, Debug)]
struct OpenAiCompatModelLoadPlan {
    path: PathBuf,
    runtime_kind: OpenAiCompatRuntimeKind,
}

#[derive(Clone)]
struct OpenAiCompatLoadedModel {
    model_key: String,
    canonical_name: String,
    supported_endpoints: Vec<RoutingEndpoint>,
    serving_truth: OpenAiCompatServingTruth,
    kind: OpenAiCompatLoadedModelKind,
}

#[derive(Clone)]
enum OpenAiCompatLoadedModelKind {
    Decoder(OpenAiCompatLoadedDecoderModel),
    Embeddings(OpenAiCompatLoadedEmbeddingsModel),
}

#[derive(Clone)]
struct OpenAiCompatLoadedDecoderModel {
    descriptor: DecoderModelDescriptor,
    family: GgufDecoderFamily,
    qwen35_multimodal_projection: Option<Qwen35MultimodalProjectionConfig>,
    prompt_renderer: Option<GgufPromptTemplateRenderer>,
    prompt_options: PromptRenderOptions,
    execution_profile: ExecutionCapabilityProfile,
    scheduler_policy: Option<GenerationSchedulerPolicy>,
}

#[derive(Clone)]
struct OpenAiCompatLoadedEmbeddingsModel {
    descriptor: psionic_models::EmbeddingModelDescriptor,
    execution_profile: ExecutionCapabilityProfile,
}

impl OpenAiCompatLoadedModel {
    fn decoder(&self) -> Option<&OpenAiCompatLoadedDecoderModel> {
        match &self.kind {
            OpenAiCompatLoadedModelKind::Decoder(model) => Some(model),
            OpenAiCompatLoadedModelKind::Embeddings(_) => None,
        }
    }

    fn embeddings(&self) -> Option<&OpenAiCompatLoadedEmbeddingsModel> {
        match &self.kind {
            OpenAiCompatLoadedModelKind::Decoder(_) => None,
            OpenAiCompatLoadedModelKind::Embeddings(model) => Some(model),
        }
    }

    fn execution_profile(&self) -> &ExecutionCapabilityProfile {
        match &self.kind {
            OpenAiCompatLoadedModelKind::Decoder(model) => &model.execution_profile,
            OpenAiCompatLoadedModelKind::Embeddings(model) => &model.execution_profile,
        }
    }

    fn scheduler_policy(&self) -> Option<&GenerationSchedulerPolicy> {
        self.decoder()
            .and_then(|model| model.scheduler_policy.as_ref())
    }

    fn serving_truth(&self) -> OpenAiCompatServingTruth {
        self.serving_truth
    }

    fn backend_label(&self) -> &'static str {
        self.serving_truth.backend_label
    }

    fn execution_mode_label(&self) -> &'static str {
        self.serving_truth.execution_mode_label
    }

    fn execution_engine_label(&self) -> &'static str {
        self.serving_truth.execution_engine_label
    }

    fn local_serving_truth(&self) -> LocalServingTruth {
        self.serving_truth.local_serving_truth
    }

    fn supports_structured_outputs(&self) -> bool {
        self.decoder().is_some_and(|model| {
            !matches!(model.family, GgufDecoderFamily::Qwen35)
                || self.execution_engine_label() == "psionic"
        })
    }

    fn supports_tool_calling(&self) -> bool {
        self.decoder().is_some_and(|model| {
            if matches!(model.family, GgufDecoderFamily::Gemma4) {
                return false;
            }
            !matches!(model.family, GgufDecoderFamily::Qwen35)
                || self.execution_engine_label() == "psionic"
        })
    }

    fn supports_response_state(&self) -> bool {
        self.decoder().is_some()
            && self
                .supported_endpoints
                .contains(&RoutingEndpoint::Responses)
    }

    fn publishes_kv_cache_policies(&self) -> bool {
        self.decoder()
            .is_some_and(|model| !matches!(model.family, GgufDecoderFamily::Qwen35))
    }

    fn structured_output_labels(&self) -> Option<Vec<&'static str>> {
        self.supports_structured_outputs()
            .then(structured_output_parser_labels)
    }

    fn structured_output_capabilities(&self) -> Vec<StructuredOutputCapability> {
        match self.decoder() {
            Some(model)
                if matches!(model.family, GgufDecoderFamily::Qwen35)
                    && self.execution_engine_label() != "psionic" =>
            {
                unsupported_structured_output_capabilities(
                    qwen35_structured_output_unavailable_detail(self.execution_engine_label()),
                )
            }
            Some(_) => local_structured_output_capabilities(),
            None => unsupported_structured_output_capabilities(
                "structured outputs are unavailable on embeddings-only models",
            ),
        }
    }

    fn tool_calling_capability(&self) -> ToolCallingCapability {
        match self.decoder() {
            Some(_) if !self.supports_tool_calling() => ToolCallingCapability {
                support_level: ToolCallingSupportLevel::Unsupported,
                supported_modes: vec!["none"],
                parser: "not_available",
                argument_validation: "not_available",
            },
            Some(_) => ToolCallingCapability {
                support_level: ToolCallingSupportLevel::Fallback,
                supported_modes: vec!["none", "auto", "required", "named"],
                parser: "tagged_json_schema",
                argument_validation: "json_schema_subset",
            },
            None => ToolCallingCapability {
                support_level: ToolCallingSupportLevel::Unsupported,
                supported_modes: vec!["none"],
                parser: "not_available",
                argument_validation: "not_available",
            },
        }
    }

    fn response_state_capability(
        &self,
        state: &OpenAiCompatState,
    ) -> Option<ResponseStateCapability> {
        self.supports_response_state()
            .then(|| state.response_state_capability.clone())
    }

    fn family_label(&self) -> &str {
        match &self.kind {
            OpenAiCompatLoadedModelKind::Decoder(model) => model.descriptor.model.family.as_str(),
            OpenAiCompatLoadedModelKind::Embeddings(model) => {
                model.descriptor.model.family.as_str()
            }
        }
    }

    fn embedding_dimensions(&self) -> Option<usize> {
        self.embeddings().map(|model| model.descriptor.dimensions)
    }

    fn embedding_normalization(&self) -> Option<EmbeddingNormalization> {
        self.embeddings()
            .map(|model| model.descriptor.normalization)
    }

    fn multimodal_projection_mode(&self) -> Option<&'static str> {
        self.decoder()
            .and_then(|model| model.qwen35_multimodal_projection.as_ref())
            .map(|_| "prompt_projection_only")
    }

    fn multimodal_supported_media(&self) -> Option<Vec<&'static str>> {
        self.decoder()
            .and_then(|model| model.qwen35_multimodal_projection.as_ref())
            .map(|_| vec!["image", "video"])
    }

    fn multimodal_projection_config(&self) -> Option<Qwen35MultimodalProjectionConfig> {
        self.decoder()
            .and_then(|model| model.qwen35_multimodal_projection.clone())
    }
}

impl OpenAiCompatState {
    fn management_status(&self) -> MeshManagementStatusResponse {
        let join_state = self
            .management_join_state
            .lock()
            .expect("management join state should not be poisoned")
            .clone()
            .into_response();
        let last_route_execution = self
            .last_route_execution
            .lock()
            .expect("last route execution should not be poisoned")
            .clone();
        let inventories = self.router.inventory();
        let mut nodes = inventories
            .iter()
            .map(mesh_management_node_status)
            .collect::<Vec<_>>();
        if let Some(local_management_node) = self.local_management_node.clone() {
            nodes.push(local_management_node);
        }
        nodes.sort_by(|left, right| left.worker_id.cmp(&right.worker_id));
        let mut routes = nodes
            .iter()
            .flat_map(|node| node.route_inventory.clone())
            .collect::<Vec<_>>();
        routes.sort_by(|left, right| {
            left.worker_id
                .cmp(&right.worker_id)
                .then_with(|| left.model_key.cmp(&right.model_key))
                .then_with(|| left.endpoint.cmp(right.endpoint))
        });
        let topology_digest = mesh_management_topology_digest(&MeshManagementStatusDigestInput {
            join_state: join_state.clone(),
            nodes: nodes.clone(),
            routes: routes.clone(),
            last_route_execution: last_route_execution.clone(),
        });
        MeshManagementStatusResponse {
            status: "ok",
            namespace: "psionic_management",
            topology_digest,
            default_model: self.default_model_name.clone(),
            node_count: nodes.len(),
            model_count: self.models_by_key.len(),
            event_stream_path: "/psionic/management/events",
            join_state,
            nodes,
            routes,
            last_route_execution,
        }
    }

    fn publish_route_selection_event(&self, request: &RoutingRequest, selection: &RouteSelection) {
        let status = self.management_status();
        let event = MeshManagementEventEnvelope {
            event_id: self
                .management_event_counter
                .fetch_add(1, Ordering::Relaxed),
            emitted_at_ms: unix_timestamp_ms(),
            topology_digest: status.topology_digest.clone(),
            payload: MeshManagementEventPayload::RouteSelection {
                request: MeshManagementRoutingRequestSummary {
                    endpoint: request.endpoint.path(),
                    target: routing_target_label(&request.target),
                    preferred_family: request.preferred_family.clone(),
                    preferred_worker_ids: request.preferred_worker_ids.clone(),
                },
                selection: selection.clone(),
            },
        };
        let _ = self.management_events.send(event);
    }

    fn management_snapshot_event(&self) -> MeshManagementEventEnvelope {
        let status = self.management_status();
        MeshManagementEventEnvelope {
            event_id: self
                .management_event_counter
                .fetch_add(1, Ordering::Relaxed),
            emitted_at_ms: unix_timestamp_ms(),
            topology_digest: status.topology_digest.clone(),
            payload: MeshManagementEventPayload::TopologySnapshot { status },
        }
    }

    fn record_route_execution(&self, status: MeshManagementRouteExecutionStatus) {
        *self
            .last_route_execution
            .lock()
            .expect("last route execution should not be poisoned") = Some(status);
    }
}

fn routing_target_label(target: &RoutingTarget) -> String {
    match target {
        RoutingTarget::Default => String::from("default"),
        RoutingTarget::RequestedModel(requested) => format!("requested:{requested}"),
        RoutingTarget::ModelKey(model_key) => format!("model_key:{model_key}"),
    }
}

fn mesh_management_node_role(worker_id: &str) -> ServedMeshRoleState {
    if worker_id == OPENAI_COMPAT_WORKER_ID {
        ServedMeshRoleState::new(ServedMeshRole::Host)
    } else {
        ServedMeshRoleState::new(ServedMeshRole::Worker)
    }
}

fn mesh_management_model_status(model: &RoutedModelInventory) -> MeshManagementModelStatus {
    MeshManagementModelStatus {
        model_key: model.model_key.clone(),
        canonical_name: model.canonical_name.clone(),
        family: model.family.clone(),
        supported_endpoints: model
            .supported_endpoints
            .iter()
            .map(|endpoint| endpoint.path())
            .collect(),
        warm_state: model.runtime_state.warm_state,
        active_requests: model.runtime_state.active_requests,
        structured_outputs: model.structured_outputs,
        tool_calling: model.tool_calling,
        response_state: model.response_state,
        execution_profile: model.execution_profile.clone(),
        scheduler_policy: model.scheduler_policy.clone(),
    }
}

fn mesh_management_route_status(
    worker_id: &str,
    model: &RoutedModelInventory,
) -> Vec<MeshManagementRouteStatus> {
    model
        .supported_endpoints
        .iter()
        .map(|endpoint| MeshManagementRouteStatus {
            worker_id: worker_id.to_string(),
            model_key: model.model_key.clone(),
            canonical_name: model.canonical_name.clone(),
            family: model.family.clone(),
            endpoint: endpoint.path(),
            warm_state: model.runtime_state.warm_state,
            active_requests: model.runtime_state.active_requests,
        })
        .collect()
}

fn mesh_management_node_status(worker: &RoutedWorkerInventory) -> MeshManagementNodeStatus {
    let mut models = worker
        .models
        .iter()
        .map(mesh_management_model_status)
        .collect::<Vec<_>>();
    models.sort_by(|left, right| left.model_key.cmp(&right.model_key));
    let mut route_inventory = worker
        .models
        .iter()
        .flat_map(|model| mesh_management_route_status(worker.worker_id.as_str(), model))
        .collect::<Vec<_>>();
    route_inventory.sort_by(|left, right| {
        left.model_key
            .cmp(&right.model_key)
            .then_with(|| left.endpoint.cmp(right.endpoint))
    });
    MeshManagementNodeStatus {
        worker_id: worker.worker_id.clone(),
        served_mesh_role: mesh_management_node_role(worker.worker_id.as_str()),
        backend_label: worker.backend_label.clone(),
        execution_mode_label: worker.execution_mode_label.clone(),
        execution_engine_label: worker.execution_engine_label.clone(),
        models,
        route_inventory,
    }
}

fn mesh_management_topology_digest(input: &MeshManagementStatusDigestInput) -> String {
    let mut hasher = Sha256::new();
    let encoded =
        serde_json::to_vec(input).expect("mesh management topology digest input should serialize");
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

fn mesh_management_event_to_sse(event: &MeshManagementEventEnvelope) -> Event {
    Event::default()
        .event(event.event_name())
        .id(event.event_id.to_string())
        .json_data(event)
        .expect("mesh management event should serialize")
}

fn route_execution_status_for_local_route(
    selection: &RouteSelection,
) -> MeshManagementRouteExecutionStatus {
    MeshManagementRouteExecutionStatus {
        worker_id: selection.worker_id.clone(),
        locality: MeshManagementRouteExecutionLocality::Local,
        provenance: String::from(LOCAL_EXECUTION_PROVENANCE),
        warm_state_reason: None,
        fallback_posture: None,
        executed_at_ms: unix_timestamp_ms(),
    }
}

fn route_execution_status_for_bootstrap_proxy(
    selection: &RouteSelection,
    mode: BootstrapProxyMode,
) -> MeshManagementRouteExecutionStatus {
    MeshManagementRouteExecutionStatus {
        worker_id: selection.worker_id.clone(),
        locality: MeshManagementRouteExecutionLocality::RemoteProxy,
        provenance: String::from(BOOTSTRAP_PROXY_PROVENANCE),
        warm_state_reason: Some(String::from(mode.warm_state_reason())),
        fallback_posture: Some(String::from(mode.fallback_posture())),
        executed_at_ms: unix_timestamp_ms(),
    }
}

fn insert_route_execution_headers(
    headers: &mut HeaderMap,
    route_execution: &MeshManagementRouteExecutionStatus,
) {
    headers.insert(
        HeaderName::from_static("x-psionic-route-locality"),
        HeaderValue::from_static(match route_execution.locality {
            MeshManagementRouteExecutionLocality::Local => "local",
            MeshManagementRouteExecutionLocality::RemoteProxy => "remote_proxy",
        }),
    );
    if let Ok(value) = HeaderValue::from_str(route_execution.provenance.as_str()) {
        headers.insert(HeaderName::from_static("x-psionic-route-provenance"), value);
    }
    if let Some(warm_state_reason) = route_execution.warm_state_reason.as_deref()
        && let Ok(value) = HeaderValue::from_str(warm_state_reason)
    {
        headers.insert(
            HeaderName::from_static("x-psionic-route-warm-state-reason"),
            value,
        );
    }
    if let Some(fallback_posture) = route_execution.fallback_posture.as_deref()
        && let Ok(value) = HeaderValue::from_str(fallback_posture)
    {
        headers.insert(
            HeaderName::from_static("x-psionic-route-fallback-posture"),
            value,
        );
    }
}

fn routing_endpoint_from_path(path: &str) -> Option<RoutingEndpoint> {
    match path {
        "/v1/chat/completions" => Some(RoutingEndpoint::ChatCompletions),
        "/v1/responses" => Some(RoutingEndpoint::Responses),
        "/v1/embeddings" => Some(RoutingEndpoint::Embeddings),
        _ => None,
    }
}

fn qwen35_structured_output_unavailable_detail(execution_engine_label: &str) -> &'static str {
    if execution_engine_label == "psionic" {
        "structured outputs are unavailable on the native qwen35 text-only runtime"
    } else {
        "structured outputs are unavailable on the qwen35 llama.cpp text-only proxy runtime"
    }
}

enum OpenAiCompatGenerationService {
    Cpu(CpuGgufTextGenerationService),
    Gemma4Cuda(CudaGemma4TextGenerationService),
    Qwen35Cuda(CudaGgufQwen35TextGenerationService),
}

#[derive(Clone)]
struct OpenAiCompatWorker {
    sender: mpsc::UnboundedSender<OpenAiCompatWorkerCommand>,
}

enum OpenAiCompatWorkerCommand {
    Generate {
        model_key: String,
        request: GenerationRequest,
        reply: oneshot::Sender<Result<crate::GenerationResponse, ReferenceTextGenerationError>>,
    },
    Embed {
        model_key: String,
        request: EmbeddingRequest,
        reply: oneshot::Sender<Result<EmbeddingResponse, ModelEmbeddingsError>>,
    },
}

impl BootstrapProxyState {
    fn from_remote_status(
        mode: BootstrapProxyMode,
        models_by_key: &BTreeMap<String, OpenAiCompatLoadedModel>,
    ) -> Result<Option<(Arc<Self>, Vec<RoutedWorkerInventory>, String)>, OpenAiCompatServerError>
    {
        let Some(base_url) = env::var_os("PSIONIC_BOOTSTRAP_PROXY_BASE_URL") else {
            return Ok(None);
        };
        let base_url = base_url.to_string_lossy().trim_end_matches('/').to_string();
        let blocking_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|error| {
                OpenAiCompatServerError::Config(format!(
                    "failed to build bootstrap proxy discovery client: {error}"
                ))
            })?;
        let status = blocking_client
            .get(format!("{base_url}/psionic/management/status"))
            .send()
            .map_err(|error| {
                OpenAiCompatServerError::Config(format!(
                    "failed to fetch bootstrap proxy management status from `{base_url}`: {error}"
                ))
            })?
            .error_for_status()
            .map_err(|error| {
                OpenAiCompatServerError::Config(format!(
                    "bootstrap proxy management status request failed for `{base_url}`: {error}"
                ))
            })?
            .json::<BootstrapRemoteManagementStatus>()
            .map_err(|error| {
                OpenAiCompatServerError::Config(format!(
                    "failed to decode bootstrap proxy management status from `{base_url}`: {error}"
                ))
            })?;

        let remote_workers = status
            .nodes
            .iter()
            .filter_map(|node| bootstrap_remote_worker_inventory(node, models_by_key))
            .collect::<Vec<_>>();
        if remote_workers.is_empty() {
            return Err(OpenAiCompatServerError::Config(format!(
                "bootstrap proxy `{base_url}` did not advertise any warm models matching the local model plan",
            )));
        }

        Ok(Some((
            Arc::new(Self {
                base_url,
                client: reqwest::Client::new(),
                mode,
            }),
            remote_workers,
            status.default_model,
        )))
    }
}

fn bootstrap_remote_worker_inventory(
    node: &BootstrapRemoteNodeStatus,
    models_by_key: &BTreeMap<String, OpenAiCompatLoadedModel>,
) -> Option<RoutedWorkerInventory> {
    let models = node
        .models
        .iter()
        .filter(|model| {
            models_by_key.contains_key(model.model_key.as_str())
                && model.warm_state == RoutedWarmState::Warm
        })
        .filter_map(bootstrap_remote_model_inventory)
        .collect::<Vec<_>>();
    if models.is_empty() {
        return None;
    }
    Some(
        RoutedWorkerInventory::new(
            format!("{BOOTSTRAP_PROXY_WORKER_PREFIX}{}", node.worker_id),
            node.backend_label.clone(),
            node.execution_mode_label.clone(),
            node.execution_engine_label.clone(),
        )
        .with_model_entries(models),
    )
}

fn bootstrap_remote_model_inventory(
    model: &BootstrapRemoteModelStatus,
) -> Option<RoutedModelInventory> {
    let mut inventory = RoutedModelInventory::new(
        model.model_key.clone(),
        model.canonical_name.clone(),
        model.family.clone(),
        model.execution_profile.clone(),
    )
    .with_warm_state(model.warm_state)
    .with_active_requests(model.active_requests);
    if let Some(stem) = Path::new(model.canonical_name.as_str())
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
    {
        inventory = inventory.with_alias(stem.to_string());
    }
    for endpoint in &model.supported_endpoints {
        inventory = inventory.with_supported_endpoint(routing_endpoint_from_path(endpoint)?);
    }
    if model.structured_outputs {
        inventory = inventory.with_structured_outputs();
    }
    if model.tool_calling {
        inventory = inventory.with_tool_calling();
    }
    if model.response_state {
        inventory = inventory.with_response_state();
    }
    if let Some(policy) = model.scheduler_policy.clone() {
        inventory = inventory.with_scheduler_policy(policy);
    }
    Some(inventory)
}

fn bootstrap_management_node(
    models_by_key: &BTreeMap<String, OpenAiCompatLoadedModel>,
    mode: BootstrapProxyMode,
) -> MeshManagementNodeStatus {
    let mut models = models_by_key
        .values()
        .map(|model| MeshManagementModelStatus {
            model_key: model.model_key.clone(),
            canonical_name: model.canonical_name.clone(),
            family: model.family_label().to_string(),
            supported_endpoints: model_endpoint_paths(model),
            warm_state: mode.local_warm_state(),
            active_requests: 0,
            structured_outputs: model.supports_structured_outputs(),
            tool_calling: model.supports_tool_calling(),
            response_state: model.supports_response_state(),
            execution_profile: model.execution_profile().clone(),
            scheduler_policy: model.scheduler_policy().cloned(),
        })
        .collect::<Vec<_>>();
    models.sort_by(|left, right| left.model_key.cmp(&right.model_key));
    let mut route_inventory = models
        .iter()
        .flat_map(|model| {
            model
                .supported_endpoints
                .iter()
                .map(|endpoint| MeshManagementRouteStatus {
                    worker_id: String::from(OPENAI_COMPAT_WORKER_ID),
                    model_key: model.model_key.clone(),
                    canonical_name: model.canonical_name.clone(),
                    family: model.family.clone(),
                    endpoint,
                    warm_state: model.warm_state,
                    active_requests: model.active_requests,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    route_inventory.sort_by(|left, right| {
        left.model_key
            .cmp(&right.model_key)
            .then_with(|| left.endpoint.cmp(right.endpoint))
    });
    MeshManagementNodeStatus {
        worker_id: String::from(OPENAI_COMPAT_WORKER_ID),
        served_mesh_role: mode.local_role_state(),
        backend_label: String::from("remote"),
        execution_mode_label: String::from("proxy"),
        execution_engine_label: String::from("psionic"),
        models,
        route_inventory,
    }
}

impl OpenAiCompatServer {
    pub fn from_config(config: &OpenAiCompatConfig) -> Result<Self, OpenAiCompatServerError> {
        Self::from_config_with_response_state_store(
            config,
            ResponseStateStore::in_memory(ResponseStateRetentionPolicy::default()),
        )
    }

    pub fn from_config_with_response_state_store(
        config: &OpenAiCompatConfig,
        response_state: ResponseStateStore,
    ) -> Result<Self, OpenAiCompatServerError> {
        if config.model_paths.is_empty() {
            return Err(OpenAiCompatServerError::Config(String::from(
                "generic OpenAI server requires at least one `--model` path",
            )));
        }

        let include_psionic_fields = env::var("PSIONIC_OPENAI_INCLUDE_DEBUG_FIELDS")
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let mut models_by_key = BTreeMap::new();
        let mut routed_models = Vec::new();
        let mut default_model_key = None;
        let mut default_canonical_model_name = None;
        let mut load_plans = Vec::new();
        let bootstrap_mode = BootstrapProxyMode::from_env()?;

        for model_path in &config.model_paths {
            let decoder_attempt =
                load_generic_decoder_model(model_path, config.reasoning_budget, config.backend);
            let embeddings_attempt = if matches!(config.backend, OpenAiCompatBackend::Cpu) {
                load_generic_embeddings_model(model_path)
            } else {
                Err(String::from(
                    "generic OpenAI cuda backend does not support embeddings artifacts",
                ))
            };
            let (loaded_model, accepted_names, load_plan) = match (
                decoder_attempt,
                embeddings_attempt,
            ) {
                (Ok(result), _) => result,
                (Err(_), Ok(result)) => result,
                (Err(decoder_error), Err(embeddings_error)) => {
                    return Err(OpenAiCompatServerError::Config(format!(
                        "unsupported generic model artifact `{}`: decoder load failed: {decoder_error}; embeddings load failed: {embeddings_error}",
                        model_path.display()
                    )));
                }
            };
            if models_by_key
                .insert(loaded_model.model_key.clone(), loaded_model.clone())
                .is_some()
            {
                return Err(OpenAiCompatServerError::Config(format!(
                    "duplicate loaded model id `{}`",
                    loaded_model.model_key
                )));
            }
            routed_models.push(routed_inventory_for_loaded_model(
                &loaded_model,
                accepted_names.into_iter().collect(),
                loaded_model.backend_label(),
            ));
            if default_model_key.is_none() {
                default_model_key = Some(loaded_model.model_key.clone());
                default_canonical_model_name = Some(loaded_model.canonical_name.clone());
            }
            load_plans.push(load_plan);
        }

        let default_model_key = default_model_key.expect("validated non-empty model list");
        if let Some(mode) = bootstrap_mode {
            for model in models_by_key.values_mut() {
                model.serving_truth = mode.serving_truth();
            }
        }
        let default_model_truth = models_by_key
            .get(&default_model_key)
            .expect("default model should exist")
            .serving_truth();
        let response_state_capability = response_state.capability();
        let (management_events, _) = broadcast::channel(64);
        let (workers, router, local_management_node, bootstrap_proxy) =
            if let Some(mode) = bootstrap_mode {
                let (bootstrap_proxy, remote_workers, remote_default_model) =
                    BootstrapProxyState::from_remote_status(mode, &models_by_key)?
                        .expect("bootstrap proxy mode should require a configured base URL");
                let router_default_model = if remote_workers.iter().any(|worker| {
                    worker
                        .models
                        .iter()
                        .any(|model| model.aliases.contains(&default_model_key))
                }) {
                    default_model_key.clone()
                } else {
                    remote_default_model
                };
                (
                    BTreeMap::new(),
                    FleetRouter::new(router_default_model, remote_workers)
                        .map_err(|error| OpenAiCompatServerError::Config(error.to_string()))?,
                    Some(bootstrap_management_node(&models_by_key, mode)),
                    Some(bootstrap_proxy),
                )
            } else {
                let worker = OpenAiCompatWorker::spawn(load_plans)?;
                let router = FleetRouter::new(
                    default_model_key.clone(),
                    vec![
                        RoutedWorkerInventory::new(
                            OPENAI_COMPAT_WORKER_ID,
                            default_model_truth.backend_label,
                            default_model_truth.execution_mode_label,
                            default_model_truth.execution_engine_label,
                        )
                        .with_model_entries(routed_models),
                    ],
                )
                .map_err(|error| OpenAiCompatServerError::Config(error.to_string()))?;
                let mut workers = BTreeMap::new();
                workers.insert(String::from(OPENAI_COMPAT_WORKER_ID), worker);
                (workers, router, None, None)
            };
        Ok(Self {
            state: Arc::new(OpenAiCompatState {
                workers,
                router,
                backend_label: default_model_truth.backend_label,
                execution_mode_label: default_model_truth.execution_mode_label,
                execution_engine_label: default_model_truth.execution_engine_label,
                default_model_key,
                default_model_name: default_canonical_model_name
                    .expect("validated non-empty model list"),
                models_by_key,
                include_psionic_fields,
                request_counter: AtomicU64::new(1),
                conversation_counter: AtomicU64::new(1),
                response_state_capability,
                response_state: Mutex::new(response_state),
                management_join_state: Mutex::new(MeshManagementJoinState::default()),
                management_event_counter: AtomicU64::new(1),
                management_events,
                local_management_node,
                last_route_execution: Mutex::new(None),
                bootstrap_proxy,
            }),
        })
    }

    #[must_use]
    pub fn backend_label(&self) -> &'static str {
        self.state.backend_label
    }

    #[must_use]
    pub fn execution_mode_label(&self) -> &'static str {
        self.state.execution_mode_label
    }

    #[must_use]
    pub fn execution_engine_label(&self) -> &'static str {
        self.state.execution_engine_label
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(generic_health))
            .route("/psionic/management/status", get(generic_management_status))
            .route("/psionic/management/events", get(generic_management_events))
            .route("/v1/models", get(generic_list_models))
            .route("/v1/chat/completions", post(generic_chat_completions))
            .route("/v1/responses", post(generic_responses))
            .route("/v1/embeddings", post(generic_embeddings))
            .with_state(Arc::clone(&self.state))
    }

    pub async fn serve(&self, listener: TcpListener) -> Result<(), OpenAiCompatServerError> {
        serve_with_runtime_telemetry(listener, self.router())
            .await
            .map_err(OpenAiCompatServerError::Io)
    }
}

impl OpenAiCompatWorker {
    fn spawn(load_plans: Vec<OpenAiCompatModelLoadPlan>) -> Result<Self, OpenAiCompatServerError> {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name(String::from("psionic-openai-worker"))
            .spawn(move || {
                let mut generation_services = BTreeMap::new();
                let mut embeddings_services = BTreeMap::new();
                for load_plan in &load_plans {
                    match load_plan.runtime_kind {
                        OpenAiCompatRuntimeKind::GgufDecoderCpu => {
                            match CpuGgufTextGenerationService::from_gguf_path(&load_plan.path) {
                                Ok(service) => {
                                    let model_key =
                                        service.model_descriptor().model.model_id.clone();
                                    generation_services.insert(
                                        model_key,
                                        OpenAiCompatGenerationService::Cpu(service),
                                    );
                                }
                                Err(error) => {
                                    let _ = ready_tx.send(Err::<(), String>(error.to_string()));
                                    return;
                                }
                            }
                        }
                        OpenAiCompatRuntimeKind::GgufDecoderCudaGemma4 => {
                            match CudaGemma4TextGenerationService::from_gguf_path(&load_plan.path) {
                                Ok(service) => {
                                    let model_key =
                                        service.model_descriptor().model.model_id.clone();
                                    generation_services.insert(
                                        model_key,
                                        OpenAiCompatGenerationService::Gemma4Cuda(service),
                                    );
                                }
                                Err(error) => {
                                    let _ = ready_tx.send(Err::<(), String>(error.to_string()));
                                    return;
                                }
                            }
                        }
                        OpenAiCompatRuntimeKind::GgufDecoderCudaQwen35 => {
                            match CudaGgufQwen35TextGenerationService::from_gguf_path(
                                &load_plan.path,
                            ) {
                                Ok(service) => {
                                    let model_key =
                                        service.model_descriptor().model.model_id.clone();
                                    generation_services.insert(
                                        model_key,
                                        OpenAiCompatGenerationService::Qwen35Cuda(service),
                                    );
                                }
                                Err(error) => {
                                    let _ = ready_tx.send(Err::<(), String>(error.to_string()));
                                    return;
                                }
                            }
                        }
                        OpenAiCompatRuntimeKind::SafetensorsEmbeddings => {
                            match CpuModelEmbeddingsService::from_safetensors_artifact(
                                &load_plan.path,
                            ) {
                                Ok(service) => {
                                    let model_key =
                                        service.model_descriptor().model.model_id.clone();
                                    embeddings_services.insert(model_key, service);
                                }
                                Err(error) => {
                                    let _ = ready_tx.send(Err::<(), String>(error.to_string()));
                                    return;
                                }
                            }
                        }
                    }
                }
                let _ = ready_tx.send(Ok::<(), String>(()));
                let mut pending_commands = VecDeque::new();
                loop {
                    let Some(command) = pending_commands
                        .pop_front()
                        .or_else(|| receiver.blocking_recv())
                    else {
                        break;
                    };
                    pending_commands.push_back(command);
                    while let Ok(command) = receiver.try_recv() {
                        pending_commands.push_back(command);
                    }

                    let Some(model_key) = pending_commands.front().map(|command| match command {
                        OpenAiCompatWorkerCommand::Generate { model_key, .. } => model_key.clone(),
                        OpenAiCompatWorkerCommand::Embed { model_key, .. } => model_key.clone(),
                    }) else {
                        continue;
                    };
                    if matches!(
                        pending_commands.front(),
                        Some(OpenAiCompatWorkerCommand::Embed { .. })
                    ) {
                        let Some(OpenAiCompatWorkerCommand::Embed {
                            model_key,
                            request,
                            reply,
                        }) = pending_commands.pop_front()
                        else {
                            continue;
                        };
                        let Some(service) = embeddings_services.get_mut(model_key.as_str()) else {
                            let _ = reply.send(Err(ModelEmbeddingsError::UnsupportedModel(
                                model_key.clone(),
                            )));
                            continue;
                        };
                        let _ = reply.send(service.embed(&request));
                        continue;
                    }
                    let mut selected = Vec::new();
                    let mut remaining = VecDeque::new();
                    while let Some(command) = pending_commands.pop_front() {
                        match command {
                            OpenAiCompatWorkerCommand::Generate {
                                model_key: command_model_key,
                                request,
                                reply,
                            } if command_model_key == model_key => {
                                selected.push((request, reply));
                            }
                            OpenAiCompatWorkerCommand::Embed {
                                model_key: command_model_key,
                                request,
                                reply,
                            } if command_model_key == model_key => {
                                remaining.push_back(OpenAiCompatWorkerCommand::Embed {
                                    model_key: command_model_key,
                                    request,
                                    reply,
                                });
                            }
                            other => remaining.push_back(other),
                        }
                    }
                    pending_commands = remaining;

                    let Some(service) = generation_services.get_mut(model_key.as_str()) else {
                        for (_, reply) in selected {
                            let _ = reply.send(Err(
                                ReferenceTextGenerationError::UnsupportedModel(model_key.clone()),
                            ));
                        }
                        continue;
                    };
                    let requests = selected
                        .iter()
                        .map(|(request, _)| request.clone())
                        .collect::<Vec<_>>();
                    let results = match service {
                        OpenAiCompatGenerationService::Cpu(service) => {
                            service.generate_continuous_batch(requests)
                        }
                        OpenAiCompatGenerationService::Gemma4Cuda(service) => {
                            service.generate_continuous_batch(requests)
                        }
                        OpenAiCompatGenerationService::Qwen35Cuda(service) => {
                            service.generate_continuous_batch(requests)
                        }
                    };
                    for ((_, reply), result) in selected.into_iter().zip(results.responses) {
                        let _ = reply.send(result);
                    }
                }
            })?;
        match ready_rx.recv().map_err(|error| {
            OpenAiCompatServerError::Config(format!(
                "failed to receive generic OpenAI worker readiness: {error}"
            ))
        })? {
            Ok(()) => Ok(Self { sender }),
            Err(message) => Err(OpenAiCompatServerError::Config(message)),
        }
    }

    async fn generate(
        &self,
        model_key: String,
        request: GenerationRequest,
    ) -> Result<crate::GenerationResponse, ReferenceTextGenerationError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.sender
            .send(OpenAiCompatWorkerCommand::Generate {
                model_key,
                request,
                reply: reply_tx,
            })
            .map_err(|_| {
                ReferenceTextGenerationError::Runtime(psionic_runtime::RuntimeError::Backend(
                    String::from("generic OpenAI worker is no longer available"),
                ))
            })?;
        reply_rx.await.map_err(|_| {
            ReferenceTextGenerationError::Runtime(psionic_runtime::RuntimeError::Backend(
                String::from("generic OpenAI worker dropped the response channel"),
            ))
        })?
    }

    async fn embed(
        &self,
        model_key: String,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ModelEmbeddingsError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.sender
            .send(OpenAiCompatWorkerCommand::Embed {
                model_key,
                request,
                reply: reply_tx,
            })
            .map_err(|_| {
                ModelEmbeddingsError::Runtime(psionic_runtime::RuntimeError::Backend(String::from(
                    "generic OpenAI worker is no longer available",
                )))
            })?;
        reply_rx.await.map_err(|_| {
            ModelEmbeddingsError::Runtime(psionic_runtime::RuntimeError::Backend(String::from(
                "generic OpenAI worker dropped the response channel",
            )))
        })?
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiCompatServerError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Config(String),
}

#[derive(Debug, thiserror::Error)]
pub enum GptOssOpenAiCompatGenerationError {
    #[error("{backend} backend unavailable ({status:?}): {message}")]
    BackendUnavailable {
        backend: &'static str,
        status: psionic_runtime::HealthStatus,
        message: String,
    },
    #[error(transparent)]
    Generation(#[from] ReferenceTextGenerationError),
}

impl From<CudaGptOssTextGenerationError> for GptOssOpenAiCompatGenerationError {
    fn from(value: CudaGptOssTextGenerationError) -> Self {
        match value {
            CudaGptOssTextGenerationError::BackendUnavailable { status, message } => {
                Self::BackendUnavailable {
                    backend: "cuda",
                    status,
                    message,
                }
            }
            CudaGptOssTextGenerationError::Generation(error) => Self::Generation(error),
        }
    }
}

impl From<MetalGptOssTextGenerationError> for GptOssOpenAiCompatGenerationError {
    fn from(value: MetalGptOssTextGenerationError) -> Self {
        match value {
            MetalGptOssTextGenerationError::BackendUnavailable { status, message } => {
                Self::BackendUnavailable {
                    backend: "metal",
                    status,
                    message,
                }
            }
            MetalGptOssTextGenerationError::Generation(error) => Self::Generation(error),
        }
    }
}

#[derive(Clone)]
struct GptOssWorker {
    sender: mpsc::UnboundedSender<GptOssWorkerCommand>,
}

enum GptOssWorkerCommand {
    Generate {
        request: GenerationRequest,
        reply:
            oneshot::Sender<Result<crate::GenerationResponse, GptOssOpenAiCompatGenerationError>>,
    },
}

impl GptOssWorker {
    fn spawn(
        model_path: PathBuf,
        backend: GptOssOpenAiCompatBackend,
    ) -> Result<Self, OpenAiCompatServerError> {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name(format!("psionic-gpt-oss-{}-worker", backend.label()))
            .spawn(move || {
                let ready = match backend {
                    GptOssOpenAiCompatBackend::Cpu => {
                        Err(String::from("cpu GPT-OSS OpenAI server is not implemented"))
                    }
                    GptOssOpenAiCompatBackend::Cuda => {
                        match CudaGgufGptOssTextGenerationService::from_gguf_path(&model_path) {
                            Ok(mut service) => {
                                let _ = ready_tx.send(Ok::<(), String>(()));
                                while let Some(command) = receiver.blocking_recv() {
                                    match command {
                                        GptOssWorkerCommand::Generate { request, reply } => {
                                            let _ = reply.send(
                                                service.generate(&request).map_err(Into::into),
                                            );
                                        }
                                    }
                                }
                                return;
                            }
                            Err(error) => Err(error.to_string()),
                        }
                    }
                    GptOssOpenAiCompatBackend::Metal => {
                        match MetalGgufGptOssTextGenerationService::from_gguf_path(&model_path) {
                            Ok(mut service) => {
                                let _ = ready_tx.send(Ok::<(), String>(()));
                                while let Some(command) = receiver.blocking_recv() {
                                    match command {
                                        GptOssWorkerCommand::Generate { request, reply } => {
                                            let _ = reply.send(
                                                service.generate(&request).map_err(Into::into),
                                            );
                                        }
                                    }
                                }
                                return;
                            }
                            Err(error) => Err(error.to_string()),
                        }
                    }
                    GptOssOpenAiCompatBackend::Auto => Err(String::from(
                        "auto backend must be resolved before worker spawn",
                    )),
                };
                let _ = ready_tx.send(ready);
            })?;
        match ready_rx.recv().map_err(|error| {
            OpenAiCompatServerError::Config(format!(
                "failed to receive GPT-OSS {} worker readiness: {error}",
                backend.label()
            ))
        })? {
            Ok(()) => Ok(Self { sender }),
            Err(message) => Err(OpenAiCompatServerError::Config(message)),
        }
    }

    async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<crate::GenerationResponse, GptOssOpenAiCompatGenerationError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.sender
            .send(GptOssWorkerCommand::Generate {
                request,
                reply: reply_tx,
            })
            .map_err(|_| GptOssOpenAiCompatGenerationError::BackendUnavailable {
                backend: "worker",
                status: psionic_runtime::HealthStatus::Offline,
                message: String::from("gpt-oss worker is no longer available"),
            })?;
        reply_rx
            .await
            .map_err(|_| GptOssOpenAiCompatGenerationError::BackendUnavailable {
                backend: "worker",
                status: psionic_runtime::HealthStatus::Offline,
                message: String::from("gpt-oss worker dropped the response channel"),
            })?
    }
}

fn metal_proxy_llama_cpp_enabled() -> bool {
    env::var("PSIONIC_METAL_PROXY_LLAMA_CPP")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

impl LlamaCppProxyState {
    fn spawn(config: &GptOssOpenAiCompatConfig) -> Result<Self, OpenAiCompatServerError> {
        let llama_bin = env::var("PSIONIC_LLAMA_SERVER_BIN").unwrap_or_else(|_| {
            if cfg!(target_os = "macos") {
                String::from("/Users/christopherdavid/code/llama.cpp/build/bin/llama-server")
            } else {
                String::from("/home/christopherdavid/code/llama.cpp/build/bin/llama-server")
            }
        });
        let internal_port = reserve_local_port()?;
        let host = "127.0.0.1";
        let mut command = Command::new(&llama_bin);
        let ctx = config
            .context_length
            .unwrap_or(if cfg!(target_os = "macos") {
                1024
            } else {
                4096
            });
        let gpu_layers =
            config
                .gpu_layers
                .unwrap_or(if cfg!(target_os = "macos") { 4 } else { 999 });
        let batch_size = env::var("PSIONIC_LLAMA_BATCH_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(if cfg!(target_os = "macos") { 64 } else { 2048 });
        let ubatch_size = env::var("PSIONIC_LLAMA_UBATCH_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(if cfg!(target_os = "macos") { 64 } else { 512 });
        command
            .arg("-m")
            .arg(&config.model_path)
            .arg("--host")
            .arg(host)
            .arg("--port")
            .arg(internal_port.to_string())
            .arg("-c")
            .arg(ctx.to_string())
            .arg("-b")
            .arg(batch_size.to_string())
            .arg("-ub")
            .arg(ubatch_size.to_string())
            .arg("-ngl")
            .arg(gpu_layers.to_string())
            .arg("--reasoning-budget")
            .arg(config.reasoning_budget.to_string())
            .arg("--no-webui")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if cfg!(target_os = "macos")
            && env::var("PSIONIC_LLAMA_DISABLE_CPU_MOE")
                .ok()
                .map(|value| !matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
                .unwrap_or(true)
        {
            command.arg("--cpu-moe");
        }
        let child = command.spawn().map_err(|error| {
            OpenAiCompatServerError::Config(format!(
                "failed to spawn llama.cpp proxy backend `{llama_bin}`: {error}"
            ))
        })?;
        let base_url = format!("http://{host}:{internal_port}");
        wait_for_upstream_ready(base_url.as_str(), config.model_path.as_path())?;
        Ok(Self {
            base_url,
            client: reqwest::Client::new(),
            child: Mutex::new(Some(child)),
        })
    }
}

fn reserve_local_port() -> Result<u16, OpenAiCompatServerError> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|error| {
        OpenAiCompatServerError::Config(format!("failed to reserve local proxy port: {error}"))
    })?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|error| {
            OpenAiCompatServerError::Config(format!("failed to query reserved proxy port: {error}"))
        })
}

fn wait_for_upstream_ready(
    base_url: &str,
    model_path: &Path,
) -> Result<(), OpenAiCompatServerError> {
    const HEALTH_TIMEOUT: Duration = Duration::from_secs(1);
    const CHAT_TIMEOUT: Duration = Duration::from_secs(10);

    let health_url = format!("{base_url}/health");
    let chat_url = format!("{base_url}/v1/chat/completions");
    let model_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            OpenAiCompatServerError::Config(format!(
                "failed to derive proxy model name from {}",
                model_path.display()
            ))
        })?;
    let health_client = reqwest::blocking::Client::builder()
        .timeout(HEALTH_TIMEOUT)
        .build()
        .map_err(|error| {
            OpenAiCompatServerError::Config(format!(
                "failed to build llama.cpp proxy health client: {error}"
            ))
        })?;
    let chat_client = reqwest::blocking::Client::builder()
        .timeout(CHAT_TIMEOUT)
        .build()
        .map_err(|error| {
            OpenAiCompatServerError::Config(format!(
                "failed to build llama.cpp proxy chat client: {error}"
            ))
        })?;
    let probe = serde_json::json!({
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Say hello."
            }
        ],
        "max_tokens": 1,
        "temperature": 0
    });
    for _ in 0..300 {
        let health_ready = matches!(
            health_client.get(health_url.as_str()).send(),
            Ok(response) if response.status().is_success()
        );
        if health_ready {
            match chat_client.post(chat_url.as_str()).json(&probe).send() {
                Ok(response) if response.status().is_success() => return Ok(()),
                Ok(response) if response.status() != reqwest::StatusCode::SERVICE_UNAVAILABLE => {
                    return Err(OpenAiCompatServerError::Config(format!(
                        "llama.cpp proxy readiness probe failed with status {}",
                        response.status()
                    )));
                }
                Ok(_) | Err(_) => {}
            }
        }
        thread::sleep(Duration::from_millis(200));
    }
    Err(OpenAiCompatServerError::Config(format!(
        "llama.cpp proxy backend did not become ready for chat completions: {chat_url}"
    )))
}

#[derive(Debug, thiserror::Error)]
enum OpenAiCompatHttpError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    Internal(String),
    #[error(transparent)]
    PromptRender(Box<PromptRenderError>),
    #[error(transparent)]
    Embeddings(Box<ModelEmbeddingsError>),
    #[error(transparent)]
    Generation(Box<GptOssOpenAiCompatGenerationError>),
}

impl From<PromptRenderError> for OpenAiCompatHttpError {
    fn from(value: PromptRenderError) -> Self {
        Self::PromptRender(Box::new(value))
    }
}

impl From<GptOssOpenAiCompatGenerationError> for OpenAiCompatHttpError {
    fn from(value: GptOssOpenAiCompatGenerationError) -> Self {
        Self::Generation(Box::new(value))
    }
}

impl From<ModelEmbeddingsError> for OpenAiCompatHttpError {
    fn from(value: ModelEmbeddingsError) -> Self {
        Self::Embeddings(Box::new(value))
    }
}

impl IntoResponse for OpenAiCompatHttpError {
    fn into_response(self) -> Response {
        let (status, kind) = match &self {
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Self::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
            Self::PromptRender(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
            Self::Embeddings(error) => (
                StatusCode::from_u16(error.diagnostic().status)
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                "embeddings_error",
            ),
            Self::Generation(error) => match error.as_ref() {
                GptOssOpenAiCompatGenerationError::BackendUnavailable { .. } => {
                    (StatusCode::SERVICE_UNAVAILABLE, "backend_unavailable")
                }
                GptOssOpenAiCompatGenerationError::Generation(error) => (
                    StatusCode::from_u16(error.diagnostic().status)
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    "generation_error",
                ),
            },
        };
        (
            status,
            Json(OpenAiErrorEnvelope {
                error: OpenAiErrorBody {
                    message: self.to_string(),
                    kind: String::from(kind),
                },
            }),
        )
            .into_response()
    }
}

#[derive(Clone, Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    backend: &'static str,
    execution_mode: &'static str,
    execution_engine: &'static str,
    model: String,
    residency_mode: &'static str,
    hybrid_offload: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    hybrid_offload_layers: Option<i32>,
    fallback_policy: &'static str,
    performance_class: &'static str,
    load_status: &'static str,
    warm_control: &'static str,
    unload_control: &'static str,
    memory_pressure_reporting: &'static str,
}

async fn health(State(state): State<Arc<GptOssOpenAiCompatState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        backend: state.backend_label,
        execution_mode: state.execution_mode_label,
        execution_engine: state.execution_engine_label,
        model: state.default_model_name.clone(),
        residency_mode: state.local_serving_truth.residency_mode,
        hybrid_offload: state.local_serving_truth.hybrid_offload,
        hybrid_offload_layers: state.local_serving_truth.hybrid_offload_layers,
        fallback_policy: state.local_serving_truth.fallback_policy,
        performance_class: state.local_serving_truth.performance_class,
        load_status: state.local_serving_truth.load_status,
        warm_control: state.local_serving_truth.warm_control,
        unload_control: state.local_serving_truth.unload_control,
        memory_pressure_reporting: state.local_serving_truth.memory_pressure_reporting,
    })
}

#[derive(Clone, Debug, Serialize)]
struct ModelsResponse {
    data: Vec<ModelCard>,
}

#[derive(Clone, Debug, Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    psionic_supported_endpoints: Vec<&'static str>,
    psionic_model_family: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_served_backend: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_execution_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_execution_engine: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_residency_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_hybrid_offload: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_hybrid_offload_layers: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_fallback_policy: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_performance_class: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_structured_outputs: Option<Vec<&'static str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_structured_output_capabilities: Option<Vec<StructuredOutputCapability>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_tool_calling: Option<ToolCallingCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_response_state: Option<ResponseStateCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_execution_profile: Option<ExecutionCapabilityProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_scheduler_policy: Option<GenerationSchedulerPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_multimodal_projection_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_multimodal_supported_media: Option<Vec<&'static str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_multimodal_projection_config: Option<Qwen35MultimodalProjectionConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_embedding_dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_embedding_normalization: Option<EmbeddingNormalization>,
}

async fn list_models(State(state): State<Arc<GptOssOpenAiCompatState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        data: vec![ModelCard {
            id: state.default_model_name.clone(),
            object: "model",
            owned_by: "psionic",
            psionic_supported_endpoints: vec![RoutingEndpoint::ChatCompletions.path()],
            psionic_model_family: state.descriptor.model.family.clone(),
            psionic_served_backend: Some(state.backend_label),
            psionic_execution_mode: Some(state.execution_mode_label),
            psionic_execution_engine: Some(state.execution_engine_label),
            psionic_residency_mode: Some(state.local_serving_truth.residency_mode),
            psionic_hybrid_offload: Some(state.local_serving_truth.hybrid_offload),
            psionic_hybrid_offload_layers: state.local_serving_truth.hybrid_offload_layers,
            psionic_fallback_policy: Some(state.local_serving_truth.fallback_policy),
            psionic_performance_class: Some(state.local_serving_truth.performance_class),
            psionic_structured_outputs: None,
            psionic_structured_output_capabilities: None,
            psionic_tool_calling: None,
            psionic_response_state: None,
            psionic_execution_profile: None,
            psionic_scheduler_policy: None,
            psionic_multimodal_projection_mode: None,
            psionic_multimodal_supported_media: None,
            psionic_multimodal_projection_config: None,
            psionic_embedding_dimensions: None,
            psionic_embedding_normalization: None,
        }],
    })
}

#[derive(Clone, Debug, Serialize)]
struct GenericHealthResponse {
    status: &'static str,
    backend: &'static str,
    execution_mode: &'static str,
    execution_engine: &'static str,
    default_model: String,
    model_count: usize,
    residency_mode: &'static str,
    hybrid_offload: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    hybrid_offload_layers: Option<i32>,
    fallback_policy: &'static str,
    performance_class: &'static str,
    load_status: &'static str,
    warm_control: &'static str,
    unload_control: &'static str,
    memory_pressure_reporting: &'static str,
    default_model_supported_endpoints: Vec<&'static str>,
    supported_endpoints: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    structured_output_fallbacks: Option<Vec<&'static str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    structured_output_capabilities: Option<Vec<StructuredOutputCapability>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calling: Option<ToolCallingCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_state: Option<ResponseStateCapability>,
    execution_profile: ExecutionCapabilityProfile,
    #[serde(skip_serializing_if = "Option::is_none")]
    scheduler_policy: Option<GenerationSchedulerPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    multimodal_projection_mode: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    multimodal_supported_media: Option<Vec<&'static str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    multimodal_projection_config: Option<Qwen35MultimodalProjectionConfig>,
}

async fn generic_health(
    State(state): State<Arc<OpenAiCompatState>>,
) -> Json<GenericHealthResponse> {
    let default_model = state
        .models_by_key
        .get(&state.default_model_key)
        .expect("default model should exist");
    Json(GenericHealthResponse {
        status: "ok",
        backend: default_model.backend_label(),
        execution_mode: default_model.execution_mode_label(),
        execution_engine: default_model.execution_engine_label(),
        default_model: state.default_model_name.clone(),
        model_count: state.models_by_key.len(),
        residency_mode: default_model.local_serving_truth().residency_mode,
        hybrid_offload: default_model.local_serving_truth().hybrid_offload,
        hybrid_offload_layers: default_model.local_serving_truth().hybrid_offload_layers,
        fallback_policy: default_model.local_serving_truth().fallback_policy,
        performance_class: default_model.local_serving_truth().performance_class,
        load_status: default_model.local_serving_truth().load_status,
        warm_control: default_model.local_serving_truth().warm_control,
        unload_control: default_model.local_serving_truth().unload_control,
        memory_pressure_reporting: default_model
            .local_serving_truth()
            .memory_pressure_reporting,
        default_model_supported_endpoints: model_endpoint_paths(default_model),
        supported_endpoints: union_supported_endpoint_paths(state.as_ref()),
        structured_output_fallbacks: default_model.structured_output_labels(),
        structured_output_capabilities: Some(default_model.structured_output_capabilities()),
        tool_calling: Some(default_model.tool_calling_capability()),
        response_state: default_model.response_state_capability(state.as_ref()),
        execution_profile: default_model.execution_profile().clone(),
        scheduler_policy: default_model.scheduler_policy().cloned(),
        multimodal_projection_mode: default_model.multimodal_projection_mode(),
        multimodal_supported_media: default_model.multimodal_supported_media(),
        multimodal_projection_config: default_model.multimodal_projection_config(),
    })
}

async fn generic_management_status(
    State(state): State<Arc<OpenAiCompatState>>,
) -> Json<MeshManagementStatusResponse> {
    Json(state.management_status())
}

async fn generic_management_events(
    State(state): State<Arc<OpenAiCompatState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let snapshot = mesh_management_event_to_sse(&state.management_snapshot_event());
    let receiver = state.management_events.subscribe();
    let initial = iter(vec![Ok::<_, Infallible>(snapshot)]);
    let updates = stream::unfold(receiver, |mut receiver| async move {
        loop {
            match receiver.recv().await {
                Ok(event) => {
                    let sse = mesh_management_event_to_sse(&event);
                    return Some((Ok::<_, Infallible>(sse), receiver));
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(initial.chain(updates))
}

async fn generic_list_models(State(state): State<Arc<OpenAiCompatState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        data: state
            .models_by_key
            .values()
            .map(|model| ModelCard {
                id: model.canonical_name.clone(),
                object: "model",
                owned_by: "psionic",
                psionic_supported_endpoints: model_endpoint_paths(model),
                psionic_model_family: model.family_label().to_string(),
                psionic_served_backend: Some(model.backend_label()),
                psionic_execution_mode: Some(model.execution_mode_label()),
                psionic_execution_engine: Some(model.execution_engine_label()),
                psionic_residency_mode: Some(model.local_serving_truth().residency_mode),
                psionic_hybrid_offload: Some(model.local_serving_truth().hybrid_offload),
                psionic_hybrid_offload_layers: model.local_serving_truth().hybrid_offload_layers,
                psionic_fallback_policy: Some(model.local_serving_truth().fallback_policy),
                psionic_performance_class: Some(model.local_serving_truth().performance_class),
                psionic_structured_outputs: model.structured_output_labels(),
                psionic_structured_output_capabilities: Some(
                    model.structured_output_capabilities(),
                ),
                psionic_tool_calling: Some(model.tool_calling_capability()),
                psionic_response_state: model.response_state_capability(state.as_ref()),
                psionic_execution_profile: Some(model.execution_profile().clone()),
                psionic_scheduler_policy: model.scheduler_policy().cloned(),
                psionic_multimodal_projection_mode: model.multimodal_projection_mode(),
                psionic_multimodal_supported_media: model.multimodal_supported_media(),
                psionic_multimodal_projection_config: model.multimodal_projection_config(),
                psionic_embedding_dimensions: model.embedding_dimensions(),
                psionic_embedding_normalization: model.embedding_normalization(),
            })
            .collect(),
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatCompletionRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    messages: Vec<ChatCompletionMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    typical_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mirostat: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mirostat_tau: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mirostat_eta: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repeat_last_n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stop: Option<StopSequences>,
    #[serde(default)]
    stream: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinitionEnvelope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoiceRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    response_format: Option<ChatCompletionResponseFormatRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psionic_grammar: Option<PsionicGrammarRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psionic_structured_output: Option<StructuredOutputRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psionic_reasoning: Option<PsionicReasoningRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psionic_prefix_cache: Option<PrefixCacheControl>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: None,
            messages: Vec::new(),
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            typical_p: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            repeat_penalty: None,
            repeat_last_n: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            max_tokens: None,
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum PsionicReasoningMode {
    #[default]
    Separate,
    Suppress,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
struct PsionicReasoningRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parser: Option<ReasoningParser>,
    #[serde(default)]
    mode: PsionicReasoningMode,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ToolDefinitionEnvelope {
    #[serde(rename = "type")]
    kind: String,
    function: ToolDefinitionRequest,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ToolDefinitionRequest {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum ToolChoiceRequest {
    Mode(String),
    Named(NamedToolChoiceRequest),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct NamedToolChoiceRequest {
    #[serde(rename = "type")]
    kind: String,
    function: NamedToolChoiceFunction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct NamedToolChoiceFunction {
    name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
struct ChatCompletionMessage {
    role: String,
    content: ChatCompletionMessageContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatCompletionToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl ChatCompletionMessage {
    fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: ChatCompletionMessageContent::Text(content.into()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[cfg(test)]
    fn named_text(
        role: impl Into<String>,
        content: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            role: role.into(),
            content: ChatCompletionMessageContent::Text(content.into()),
            name: Some(name.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[cfg(test)]
    fn multimodal(role: impl Into<String>, content: Vec<ChatCompletionContentPart>) -> Self {
        Self {
            role: role.into(),
            content: ChatCompletionMessageContent::Parts(content),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
enum ChatCompletionMessageContent {
    Text(String),
    Parts(Vec<ChatCompletionContentPart>),
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ChatCompletionContentPart {
    Text { text: String },
    ImageUrl { image_url: ChatCompletionMediaUrl },
    VideoUrl { video_url: ChatCompletionMediaUrl },
}

impl ChatCompletionContentPart {
    #[cfg(test)]
    fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    #[cfg(test)]
    fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ChatCompletionMediaUrl { url: url.into() },
        }
    }

    #[cfg(test)]
    fn video_url(url: impl Into<String>) -> Self {
        Self::VideoUrl {
            video_url: ChatCompletionMediaUrl { url: url.into() },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
struct ChatCompletionMediaUrl {
    url: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatCompletionResponseFormatRequest {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    json_schema: Option<ChatCompletionJsonSchemaRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    schema: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatCompletionJsonSchemaRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    schema: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PsionicGrammarRequest {
    grammar: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    syntax: Option<StructuredGrammarSyntax>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct EmbeddingsRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    input: EmbeddingsInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

impl EmbeddingsInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ResponsesRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    conversation: Option<String>,
    input: ResponsesInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    typical_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mirostat: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mirostat_tau: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    mirostat_eta: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    repeat_last_n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stop: Option<StopSequences>,
    #[serde(default)]
    stream: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinitionEnvelope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoiceRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psionic_structured_output: Option<StructuredOutputRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psionic_reasoning: Option<PsionicReasoningRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psionic_response_state: Option<PsionicResponseStateRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psionic_prefix_cache: Option<PrefixCacheControl>,
}

impl Default for ResponsesRequest {
    fn default() -> Self {
        Self {
            model: None,
            instructions: None,
            conversation: None,
            input: ResponsesInput::Text(String::new()),
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            typical_p: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            repeat_penalty: None,
            repeat_last_n: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            max_output_tokens: None,
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_response_state: None,
            psionic_prefix_cache: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum ResponsesInput {
    Text(String),
    Messages(Vec<ChatCompletionMessage>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

fn structured_output_from_chat_request(
    request: &ChatCompletionRequest,
) -> Result<Option<StructuredOutputRequest>, OpenAiCompatHttpError> {
    let surfaces = usize::from(request.response_format.is_some())
        + usize::from(request.psionic_grammar.is_some())
        + usize::from(request.psionic_structured_output.is_some());
    if surfaces > 1 {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "structured output accepts exactly one of `psionic_structured_output`, `response_format`, or `psionic_grammar`",
        )));
    }

    if let Some(structured_output) = request.psionic_structured_output.clone() {
        return validate_structured_output_request(structured_output).map(Some);
    }

    if let Some(grammar) = &request.psionic_grammar {
        if grammar.grammar.trim().is_empty() {
            return Err(OpenAiCompatHttpError::BadRequest(String::from(
                "`psionic_grammar.grammar` must not be empty",
            )));
        }
        return validate_structured_output_request(StructuredOutputRequest::Grammar {
            syntax: grammar.syntax.unwrap_or(StructuredGrammarSyntax::Gbnf),
            grammar: grammar.grammar.clone(),
        })
        .map(Some);
    }

    let Some(response_format) = &request.response_format else {
        return Ok(None);
    };
    match response_format.kind.as_str() {
        "json_object" => {
            if let Some(schema) = response_format.schema.as_ref() {
                validate_structured_output_request(StructuredOutputRequest::JsonSchema {
                    name: None,
                    schema: schema.clone(),
                })
                .map(Some)
            } else {
                validate_structured_output_request(StructuredOutputRequest::JsonObject).map(Some)
            }
        }
        "json_schema" => {
            let Some(schema) = response_format.json_schema.as_ref() else {
                return Err(OpenAiCompatHttpError::BadRequest(String::from(
                    "`response_format.type = json_schema` requires a `json_schema` object",
                )));
            };
            validate_structured_output_request(StructuredOutputRequest::JsonSchema {
                name: schema.name.clone(),
                schema: schema.schema.clone(),
            })
            .map(Some)
        }
        other => Err(OpenAiCompatHttpError::BadRequest(format!(
            "unsupported `response_format.type` `{other}` for local structured output fallback"
        ))),
    }
}

fn structured_output_from_responses_request(
    request: &ResponsesRequest,
) -> Result<Option<StructuredOutputRequest>, OpenAiCompatHttpError> {
    let Some(structured_output) = request.psionic_structured_output.clone() else {
        return Ok(None);
    };
    validate_structured_output_request(structured_output).map(Some)
}

fn validate_structured_output_request(
    structured_output: StructuredOutputRequest,
) -> Result<StructuredOutputRequest, OpenAiCompatHttpError> {
    StructuredOutputMatcher::compile(structured_output.clone())
        .map_err(|error| OpenAiCompatHttpError::BadRequest(error.to_string()))?;
    Ok(structured_output)
}

fn reasoning_request_for_family(
    request: Option<&PsionicReasoningRequest>,
    family: GgufDecoderFamily,
) -> Result<Option<ResolvedReasoningRequest>, OpenAiCompatHttpError> {
    let Some(request) = request else {
        return Ok(None);
    };
    let Some(family_parser) = reasoning_parser_for_decoder_family(family) else {
        return Err(OpenAiCompatHttpError::BadRequest(format!(
            "model family `{}` does not expose a Psionic reasoning parser",
            decoder_family_label(family)
        )));
    };
    if let Some(parser) = request.parser
        && parser != family_parser
    {
        return Err(OpenAiCompatHttpError::BadRequest(format!(
            "requested reasoning parser `{}` does not match the `{}` parser for model family `{}`",
            parser.label(),
            family_parser.label(),
            decoder_family_label(family)
        )));
    }
    Ok(Some(ResolvedReasoningRequest {
        parser: request.parser.unwrap_or(family_parser),
        mode: request.mode,
    }))
}

fn decoder_family_label(family: GgufDecoderFamily) -> &'static str {
    match family {
        GgufDecoderFamily::Llama => "llama",
        GgufDecoderFamily::Qwen => "qwen",
        GgufDecoderFamily::Qwen35 => "qwen35",
        GgufDecoderFamily::Gemma4 => "gemma4",
        GgufDecoderFamily::Mistral => "mistral",
        GgufDecoderFamily::GptOss => "gpt_oss",
    }
}

fn default_response_state_store() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

fn resolved_response_state_request(
    request: &ResponsesRequest,
) -> Result<PsionicResponseStateRequest, OpenAiCompatHttpError> {
    if request.previous_response_id.is_some() && request.conversation.is_some() {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "`conversation` and `previous_response_id` are mutually exclusive on `/v1/responses`",
        )));
    }
    Ok(request.psionic_response_state.clone().unwrap_or_default())
}

fn next_conversation_id(state: &OpenAiCompatState) -> String {
    let next = state.conversation_counter.fetch_add(1, Ordering::Relaxed);
    format!("psionic-conv-{next}")
}

fn current_response_state_capability(state: &OpenAiCompatState) -> ResponseStateCapability {
    state.response_state_capability.clone()
}

fn response_state_error_into_http(error: ResponseStateError) -> OpenAiCompatHttpError {
    match error {
        ResponseStateError::UnknownResponseState { response_id } => {
            OpenAiCompatHttpError::BadRequest(format!(
                "response state `{response_id}` is unknown or expired"
            ))
        }
        ResponseStateError::UnknownConversationState { conversation_id } => {
            OpenAiCompatHttpError::BadRequest(format!(
                "conversation state `{conversation_id}` is unknown or expired"
            ))
        }
        ResponseStateError::ConversationTooLarge {
            max_items_per_conversation,
            ..
        } => OpenAiCompatHttpError::BadRequest(format!(
            "stateful response exceeds the bounded conversation-state limit of {max_items_per_conversation} prompt messages"
        )),
        ResponseStateError::IoRead { .. }
        | ResponseStateError::IoWrite { .. }
        | ResponseStateError::Deserialize { .. }
        | ResponseStateError::Serialize { .. } => OpenAiCompatHttpError::Internal(format!(
            "generic response-state backend failed: {error}"
        )),
    }
}

fn parse_reasoning_response_for_family(
    family: GgufDecoderFamily,
    text: &str,
) -> Result<Option<ParsedReasoningResponse>, OpenAiCompatHttpError> {
    parse_reasoning_response_text_for_decoder_family(
        family,
        text,
        GptOssHarmonyParseOptions {
            role_hint: Some(PromptMessageRole::Assistant),
            strict: false,
        },
    )
    .map_err(|error| OpenAiCompatHttpError::BadRequest(error.to_string()))
}

fn surfaced_reasoning_response(
    parsed: Option<&ParsedReasoningResponse>,
    request: Option<&ResolvedReasoningRequest>,
    include_debug_fields: bool,
) -> Option<ParsedReasoningResponse> {
    let parsed = parsed?;
    if let Some(request) = request {
        return Some(match request.mode {
            PsionicReasoningMode::Separate => parsed.clone(),
            PsionicReasoningMode::Suppress => parsed.suppress_reasoning(),
        });
    }
    include_debug_fields.then(|| parsed.clone())
}

fn tool_contract_from_chat_request(
    request: &ChatCompletionRequest,
    structured_output_requested: bool,
) -> Result<Option<ToolCallingContract>, OpenAiCompatHttpError> {
    validate_tool_contract(
        request.tools.as_slice(),
        request.tool_choice.as_ref(),
        request.parallel_tool_calls,
        structured_output_requested,
    )
}

fn tool_contract_from_responses_request(
    request: &ResponsesRequest,
    structured_output_requested: bool,
) -> Result<Option<ToolCallingContract>, OpenAiCompatHttpError> {
    validate_tool_contract(
        request.tools.as_slice(),
        request.tool_choice.as_ref(),
        request.parallel_tool_calls,
        structured_output_requested,
    )
}

fn validate_tool_contract(
    tools: &[ToolDefinitionEnvelope],
    tool_choice: Option<&ToolChoiceRequest>,
    parallel_tool_calls: Option<bool>,
    structured_output_requested: bool,
) -> Result<Option<ToolCallingContract>, OpenAiCompatHttpError> {
    if tools.is_empty() {
        if tool_choice.is_some() {
            return Err(OpenAiCompatHttpError::BadRequest(String::from(
                "`tool_choice` requires at least one declared tool",
            )));
        }
        if parallel_tool_calls.is_some() {
            return Err(OpenAiCompatHttpError::BadRequest(String::from(
                "`parallel_tool_calls` requires at least one declared tool",
            )));
        }
        return Ok(None);
    }

    let mut tool_map = BTreeMap::new();
    for tool in tools {
        if tool.kind != "function" {
            return Err(OpenAiCompatHttpError::BadRequest(format!(
                "unsupported tool type `{}`; only `function` is supported",
                tool.kind
            )));
        }
        if tool.function.name.trim().is_empty() {
            return Err(OpenAiCompatHttpError::BadRequest(String::from(
                "tool function names must not be empty",
            )));
        }
        if tool_map
            .insert(tool.function.name.clone(), tool.function.clone())
            .is_some()
        {
            return Err(OpenAiCompatHttpError::BadRequest(format!(
                "duplicate tool definition `{}`",
                tool.function.name
            )));
        }
        let _ = normalized_tool_parameters_schema(&tool.function)?;
    }

    let (mode, named_tool) = match tool_choice {
        None => (ToolChoiceMode::Auto, None),
        Some(ToolChoiceRequest::Mode(value)) => match value.as_str() {
            "none" => (ToolChoiceMode::None, None),
            "auto" => (ToolChoiceMode::Auto, None),
            "required" => (ToolChoiceMode::Required, None),
            other => {
                return Err(OpenAiCompatHttpError::BadRequest(format!(
                    "unsupported `tool_choice` mode `{other}`"
                )));
            }
        },
        Some(ToolChoiceRequest::Named(named)) => {
            if named.kind != "function" {
                return Err(OpenAiCompatHttpError::BadRequest(format!(
                    "unsupported named tool choice type `{}`",
                    named.kind
                )));
            }
            if !tool_map.contains_key(named.function.name.as_str()) {
                return Err(OpenAiCompatHttpError::BadRequest(format!(
                    "named tool choice `{}` does not match a declared tool",
                    named.function.name
                )));
            }
            (ToolChoiceMode::Named, Some(named.function.name.clone()))
        }
    };

    if structured_output_requested && !matches!(mode, ToolChoiceMode::None) {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "tool-calling modes cannot be combined with `psionic_structured_output`, `response_format`, or `psionic_grammar` on the same request",
        )));
    }

    Ok(Some(ToolCallingContract {
        tools: tool_map,
        mode,
        named_tool,
        parallel_tool_calls: parallel_tool_calls.unwrap_or(true),
        minimum_required_tool_calls: 1,
    }))
}

fn normalized_tool_parameters_schema(
    tool: &ToolDefinitionRequest,
) -> Result<serde_json::Value, OpenAiCompatHttpError> {
    let mut schema = match tool.parameters.clone() {
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => {
            return Err(OpenAiCompatHttpError::BadRequest(format!(
                "tool `{}` parameters must be a JSON object schema",
                tool.name
            )));
        }
        None => serde_json::Map::new(),
    };
    match schema.get("type") {
        Some(serde_json::Value::String(kind)) if kind == "object" => {}
        Some(_) => {
            return Err(OpenAiCompatHttpError::BadRequest(format!(
                "tool `{}` parameters must describe an object schema",
                tool.name
            )));
        }
        None => {
            schema.insert(
                String::from("type"),
                serde_json::Value::String(String::from("object")),
            );
        }
    }
    if !schema.contains_key("properties") {
        schema.insert(
            String::from("properties"),
            serde_json::Value::Object(serde_json::Map::new()),
        );
    }
    if !schema.contains_key("additionalProperties") {
        schema.insert(
            String::from("additionalProperties"),
            serde_json::Value::Bool(false),
        );
    }
    Ok(serde_json::Value::Object(schema))
}

fn required_tool_call_floor_from_chat_messages(
    messages: &[ChatCompletionMessage],
    contract: &ToolCallingContract,
    family: GgufDecoderFamily,
    qwen35_multimodal_projection: Option<&Qwen35MultimodalProjectionConfig>,
) -> Result<usize, OpenAiCompatHttpError> {
    if !matches!(contract.mode, ToolChoiceMode::Required) || !contract.allows_parallel_tool_calls()
    {
        return Ok(1);
    }

    let mut mentions = Vec::new();
    for (message_index, message) in messages.iter().enumerate() {
        let role = match message.role.as_str() {
            "system" => PromptMessageRole::System,
            "developer" => PromptMessageRole::Developer,
            "user" => PromptMessageRole::User,
            _ => continue,
        };
        let text = chat_message_content_to_text(
            &message.content,
            family,
            role,
            qwen35_multimodal_projection,
        )?;
        for tool_name in contract.tools.keys() {
            let quoted = format!("`{tool_name}`");
            for (offset, _) in text.match_indices(quoted.as_str()) {
                mentions.push(((message_index, offset), tool_name.clone()));
            }
        }
    }
    mentions.sort_by_key(|(position, _)| *position);

    let mut distinct = Vec::new();
    for (_, tool_name) in mentions {
        if !distinct.iter().any(|value| value == &tool_name) {
            distinct.push(tool_name);
        }
    }

    Ok(distinct.len().max(1))
}

fn tool_prompt_message(contract: &ToolCallingContract) -> PromptMessage {
    let mut lines = vec![String::from(
        "When tools are enabled, respond with exactly one JSON object that matches the declared Psionic tool contract.",
    )];
    match contract.mode {
        ToolChoiceMode::None => lines.push(String::from(
            "Tool use is disabled for this request. Answer normally.",
        )),
        ToolChoiceMode::Auto => lines.push(String::from(
            if contract.allows_parallel_tool_calls() {
                "Use `{ \"kind\": \"message\", \"content\": \"...\" }` for a normal answer, or `{ \"kind\": \"tool_calls\", \"tool_calls\": [{ \"name\": \"<tool>\", \"arguments\": { ... } }] }` to call one or more tools in order."
            } else {
                "Use `{ \"kind\": \"message\", \"content\": \"...\" }` for a normal answer, or `{ \"kind\": \"tool_calls\", \"tool_calls\": [{ \"name\": \"<tool>\", \"arguments\": { ... } }] }` to call exactly one tool."
            },
        )),
        ToolChoiceMode::Required => lines.push(String::from(if contract.allows_parallel_tool_calls() {
            if contract.minimum_required_tool_calls > 1 {
                format!(
                    "You must call at least {} tools using `{{ \"tool_calls\": [{{ \"name\": \"<tool>\", \"arguments\": {{ ... }} }}] }}`.",
                    contract.minimum_required_tool_calls
                )
            } else {
                String::from(
                    "You must call one or more tools using `{ \"tool_calls\": [{ \"name\": \"<tool>\", \"arguments\": { ... } }] }`.",
                )
            }
        } else {
            String::from(
                "You must call exactly one tool using `{ \"tool_calls\": [{ \"name\": \"<tool>\", \"arguments\": { ... } }] }`.",
            )
        })),
        ToolChoiceMode::Named => lines.push(format!(
            "You must call exactly one tool using `{{ \"kind\": \"tool_calls\", \"tool_calls\": [{{ \"name\": \"{}\", \"arguments\": {{ ... }} }}] }}`.",
            contract.named_tool.as_deref().unwrap_or_default()
        )),
    }
    if contract.allows_parallel_tool_calls()
        && let Some(example) = parallel_tool_call_example(contract)
    {
        lines.push(example);
        lines.push(String::from(
            "When the user explicitly asks for multiple declared tools in the same turn, include each requested tool exactly once in one `tool_calls` array before any answer. Do not omit a requested tool, split the calls across turns, or repeat one tool instead of another.",
        ));
    }
    lines.push(String::from("Declared tools:"));
    for tool in contract.tools.values() {
        let schema =
            normalized_tool_parameters_schema(tool).unwrap_or_else(|_| serde_json::json!({}));
        lines.push(format!(
            "- {}: {} | schema={}",
            tool.name,
            tool.description
                .clone()
                .unwrap_or_else(|| String::from("no description")),
            schema
        ));
    }
    PromptMessage::new(PromptMessageRole::Developer, lines.join("\n"))
}

fn parallel_tool_call_example(contract: &ToolCallingContract) -> Option<String> {
    let tool_names = contract
        .tools
        .values()
        .take(2)
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();
    if tool_names.len() < 2 {
        return None;
    }
    Some(format!(
        "If multiple tools are needed in the same turn, emit them in one `tool_calls` array like `{{ \"kind\": \"tool_calls\", \"tool_calls\": [{{ \"name\": \"{}\", \"arguments\": {{ ... }} }}, {{ \"name\": \"{}\", \"arguments\": {{ ... }} }}] }}`.",
        tool_names[0], tool_names[1]
    ))
}

fn apply_tool_contract_to_prompt_messages(
    mut messages: Vec<PromptMessage>,
    contract: Option<&ToolCallingContract>,
) -> Vec<PromptMessage> {
    if let Some(contract) = contract
        && !matches!(contract.mode, ToolChoiceMode::None)
    {
        messages.insert(0, tool_prompt_message(contract));
    }
    messages
}

fn structured_output_from_tool_contract(
    contract: Option<&ToolCallingContract>,
) -> Result<Option<StructuredOutputRequest>, OpenAiCompatHttpError> {
    let Some(contract) = contract else {
        return Ok(None);
    };
    if matches!(contract.mode, ToolChoiceMode::None) {
        return Ok(None);
    }

    if matches!(
        contract.mode,
        ToolChoiceMode::Required | ToolChoiceMode::Named
    ) {
        return validate_structured_output_request(StructuredOutputRequest::JsonSchema {
            name: Some(String::from("psionic_tool_call_batch")),
            schema: tool_calls_batch_schema(contract)?,
        })
        .map(Some);
    }

    let variants = vec![
        StructuredTaggedVariant {
            tag: String::from("message"),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string", "minLength": 1 }
                },
                "required": ["content"],
                "additionalProperties": false
            }),
        },
        StructuredTaggedVariant {
            tag: String::from("tool_calls"),
            schema: tool_calls_batch_schema(contract)?,
        },
    ];

    validate_structured_output_request(StructuredOutputRequest::TaggedStructure {
        name: Some(String::from("psionic_tool_call")),
        discriminator: String::from("kind"),
        variants,
    })
    .map(Some)
}

fn tool_calls_batch_schema(
    contract: &ToolCallingContract,
) -> Result<serde_json::Value, OpenAiCompatHttpError> {
    let items_schema = match contract.mode {
        ToolChoiceMode::Named => {
            let name = contract.named_tool.as_ref().ok_or_else(|| {
                OpenAiCompatHttpError::Internal(String::from(
                    "named tool mode is missing the selected tool",
                ))
            })?;
            let tool = contract.tools.get(name).ok_or_else(|| {
                OpenAiCompatHttpError::Internal(format!(
                    "named tool `{name}` is missing from the validated tool map"
                ))
            })?;
            tool_call_item_schema(tool)?
        }
        ToolChoiceMode::Auto | ToolChoiceMode::Required => serde_json::json!({
            "oneOf": contract
                .tools
                .values()
                .map(tool_call_item_schema)
                .collect::<Result<Vec<_>, OpenAiCompatHttpError>>()?
        }),
        ToolChoiceMode::None => {
            return Err(OpenAiCompatHttpError::Internal(String::from(
                "tool batch schema requested while tool calling is disabled",
            )));
        }
    };

    let mut tool_calls = serde_json::Map::new();
    tool_calls.insert(
        String::from("type"),
        serde_json::Value::String(String::from("array")),
    );
    tool_calls.insert(
        String::from("minItems"),
        serde_json::json!(contract.minimum_required_tool_calls.max(1)),
    );
    tool_calls.insert(String::from("items"), items_schema);
    if !contract.allows_parallel_tool_calls() {
        tool_calls.insert(String::from("maxItems"), serde_json::json!(1));
    }

    Ok(serde_json::Value::Object(serde_json::Map::from_iter([
        (
            String::from("type"),
            serde_json::Value::String(String::from("object")),
        ),
        (
            String::from("properties"),
            serde_json::json!({
                "tool_calls": serde_json::Value::Object(tool_calls)
            }),
        ),
        (String::from("required"), serde_json::json!(["tool_calls"])),
        (
            String::from("additionalProperties"),
            serde_json::Value::Bool(false),
        ),
    ])))
}

fn tool_call_item_schema(
    tool: &ToolDefinitionRequest,
) -> Result<serde_json::Value, OpenAiCompatHttpError> {
    Ok(serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "const": tool.name.clone() },
            "arguments": normalized_tool_parameters_schema(tool)?
        },
        "required": ["name", "arguments"],
        "additionalProperties": false
    }))
}

fn tool_call_outcome_from_response(
    request_id: &str,
    response: &crate::GenerationResponse,
    contract: Option<&ToolCallingContract>,
) -> Result<Option<ToolCallOutcome>, OpenAiCompatHttpError> {
    let Some(contract) = contract else {
        return Ok(None);
    };
    if matches!(contract.mode, ToolChoiceMode::None) {
        return Ok(None);
    }

    let Some(structured) = response.output.structured.clone() else {
        if matches!(contract.mode, ToolChoiceMode::Auto) {
            if response.output.text.is_empty() {
                return Err(OpenAiCompatHttpError::BadRequest(String::from(
                    "tool auto-mode response omitted both a machine-readable tool envelope and assistant text",
                )));
            }
            return Ok(Some(ToolCallOutcome {
                content: Some(response.output.text.clone()),
                tool_calls: Vec::new(),
            }));
        }
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "tool-calling request completed without a machine-readable tool envelope",
        )));
    };
    match structured {
        StructuredOutputValue::Json { value } => {
            return Ok(Some(ToolCallOutcome {
                content: None,
                tool_calls: resolved_tool_calls_from_json_value(request_id, &value, contract)?,
            }));
        }
        StructuredOutputValue::TaggedStructure {
            discriminator,
            tag,
            value,
        } => {
            if discriminator != "kind" {
                return Err(OpenAiCompatHttpError::BadRequest(format!(
                    "unexpected tool envelope discriminator `{discriminator}`"
                )));
            }

            if tag == "message" {
                let content = value
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        OpenAiCompatHttpError::BadRequest(String::from(
                            "tool auto-mode message envelope is missing string `content`",
                        ))
                    })?
                    .to_string();
                return Ok(Some(ToolCallOutcome {
                    content: Some(content),
                    tool_calls: Vec::new(),
                }));
            }

            if tag == "tool_calls" {
                return Ok(Some(ToolCallOutcome {
                    content: None,
                    tool_calls: resolved_tool_calls_from_json_value(request_id, &value, contract)?,
                }));
            }

            let Some(tool_name) = tag.strip_prefix("tool:") else {
                return Err(OpenAiCompatHttpError::BadRequest(format!(
                    "unexpected tool envelope tag `{tag}`"
                )));
            };
            let tool = contract.tools.get(tool_name).ok_or_else(|| {
                OpenAiCompatHttpError::BadRequest(format!(
                    "model selected undeclared tool `{tool_name}`"
                ))
            })?;
            let mut arguments = match value {
                serde_json::Value::Object(map) => map,
                _ => {
                    return Err(OpenAiCompatHttpError::BadRequest(format!(
                        "tool envelope for `{tool_name}` must be a JSON object"
                    )));
                }
            };
            let _ = arguments.remove("kind");
            let arguments = serde_json::Value::Object(arguments);
            validate_tool_arguments(tool, &arguments)?;
            return Ok(Some(ToolCallOutcome {
                content: None,
                tool_calls: vec![ResolvedToolCall {
                    id: format!("{request_id}-tool-0"),
                    name: tool_name.to_string(),
                    arguments,
                }],
            }));
        }
        _ => {
            return Err(OpenAiCompatHttpError::BadRequest(String::from(
                "tool-calling request completed without a supported machine-readable tool envelope",
            )));
        }
    }
}

fn resolved_tool_calls_from_json_value(
    request_id: &str,
    value: &serde_json::Value,
    contract: &ToolCallingContract,
) -> Result<Vec<ResolvedToolCall>, OpenAiCompatHttpError> {
    let tool_calls = value
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            OpenAiCompatHttpError::BadRequest(String::from(
                "tool batch envelope is missing an array `tool_calls` field",
            ))
        })?;
    if tool_calls.is_empty() {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "tool batch envelope must contain at least one tool call",
        )));
    }
    if !contract.allows_parallel_tool_calls() && tool_calls.len() != 1 {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "tool batch envelope returned more than one tool call while `parallel_tool_calls` is disabled",
        )));
    }
    tool_calls
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let item = item.as_object().ok_or_else(|| {
                OpenAiCompatHttpError::BadRequest(String::from(
                    "tool batch entries must be JSON objects",
                ))
            })?;
            let tool_name = item
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    OpenAiCompatHttpError::BadRequest(String::from(
                        "tool batch entries must include string `name`",
                    ))
                })?;
            let arguments = item.get("arguments").cloned().ok_or_else(|| {
                OpenAiCompatHttpError::BadRequest(String::from(
                    "tool batch entries must include `arguments`",
                ))
            })?;
            let tool = contract.tools.get(tool_name).ok_or_else(|| {
                OpenAiCompatHttpError::BadRequest(format!(
                    "model selected undeclared tool `{tool_name}`"
                ))
            })?;
            validate_tool_arguments(tool, &arguments)?;
            Ok(ResolvedToolCall {
                id: format!("{request_id}-tool-{index}"),
                name: tool_name.to_string(),
                arguments,
            })
        })
        .collect()
}

fn validate_tool_arguments(
    tool: &ToolDefinitionRequest,
    arguments: &serde_json::Value,
) -> Result<(), OpenAiCompatHttpError> {
    let schema = normalized_tool_parameters_schema(tool)?;
    let matcher = StructuredOutputMatcher::compile(StructuredOutputRequest::JsonSchema {
        name: Some(format!("tool:{} arguments", tool.name)),
        schema,
    })
    .map_err(|error| OpenAiCompatHttpError::BadRequest(error.to_string()))?;
    let raw = serde_json::to_string(arguments).map_err(|error| {
        OpenAiCompatHttpError::BadRequest(format!(
            "failed to serialize arguments for tool `{}`: {error}",
            tool.name
        ))
    })?;
    matcher
        .materialize(raw.as_str())
        .map_err(|error| OpenAiCompatHttpError::BadRequest(error.to_string()))?
        .ok_or_else(|| {
            OpenAiCompatHttpError::BadRequest(format!(
                "arguments for tool `{}` did not satisfy the declared schema",
                tool.name
            ))
        })?;
    Ok(())
}

async fn chat_completions(
    State(state): State<Arc<GptOssOpenAiCompatState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    match handle_chat_completions(state, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn handle_chat_completions(
    state: Arc<GptOssOpenAiCompatState>,
    request: ChatCompletionRequest,
) -> Result<Response, OpenAiCompatHttpError> {
    let reasoning_request = reasoning_request_for_family(
        request.psionic_reasoning.as_ref(),
        GgufDecoderFamily::GptOss,
    )?;
    let tool_contract = tool_contract_from_chat_request(&request, false)?;
    if tool_contract
        .as_ref()
        .is_some_and(|contract| !matches!(contract.mode, ToolChoiceMode::None))
    {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "tool-calling modes are only available on the generic Psionic server today",
        )));
    }
    if structured_output_from_chat_request(&request)?.is_some() {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "structured output fallback is only available on `psionic-openai-server` today",
        )));
    }
    if state.proxy.is_some() && reasoning_request.is_some() {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "psionic reasoning separation is unavailable while the GPT-OSS endpoint is proxying through llama.cpp",
        )));
    }
    if let Some(proxy) = state.proxy.as_ref() {
        return proxy_chat_completions(state.as_ref(), proxy, &request).await;
    }
    validate_requested_model(request.model.as_deref(), &state.accepted_model_names)?;
    let prompt_messages = chat_messages_to_prompt_messages(&request.messages)?;
    let request_prompt_key = prompt_request_cache_key(prompt_messages.as_slice());
    let request_id = next_request_id(&state);
    let response_model_name = request
        .model
        .clone()
        .unwrap_or_else(|| state.default_model_name.clone());
    let options = generation_options_from_chat_request(&request);
    let prompt_tokens = {
        let mut cache = state.prompt_token_cache.lock().map_err(|_| {
            OpenAiCompatHttpError::from(GptOssOpenAiCompatGenerationError::Generation(
                ReferenceTextGenerationError::Runtime(psionic_runtime::RuntimeError::Backend(
                    String::from("openai prompt token cache is poisoned"),
                )),
            ))
        })?;
        if let Some(tokens) = cache.lookup(request_prompt_key.as_str()) {
            tokens
        } else {
            let rendered = render_gpt_oss_harmony_prompt(
                prompt_messages.as_slice(),
                true,
                Some(&state.prompt_options),
            )
            .map_err(|error| {
                OpenAiCompatHttpError::from(PromptRenderError::HarmonyRendering {
                    message: error.to_string(),
                })
            })?;
            let tokens = state.tokenizer.encode_with_defaults(rendered.as_str());
            cache.record(request_prompt_key, tokens.clone());
            tokens
        }
    };
    let generation_request = GenerationRequest::new_tokens(
        request_id.clone(),
        state.descriptor.clone(),
        None,
        prompt_tokens,
        options,
    )
    .with_prefix_cache_control(request.psionic_prefix_cache.clone().unwrap_or_default());

    let worker = state.worker.as_ref().ok_or_else(|| {
        OpenAiCompatHttpError::from(GptOssOpenAiCompatGenerationError::BackendUnavailable {
            backend: state.backend_label,
            status: psionic_runtime::HealthStatus::Offline,
            message: String::from("gpt-oss native worker is not available"),
        })
    })?;
    let response = worker.generate(generation_request).await?;
    let parsed = parse_gpt_oss_harmony_text(
        response.output.text.as_str(),
        GptOssHarmonyParseOptions {
            role_hint: Some(PromptMessageRole::Assistant),
            strict: false,
        },
    )
    .ok();
    let parsed_reasoning = parsed
        .as_ref()
        .map(GptOssHarmonyParsedOutput::reasoning_response);
    let choice = completion_choice(
        &response,
        parsed_reasoning.as_ref(),
        reasoning_request.as_ref(),
    );
    if request.stream {
        let terminal_chunk = completion_terminal_chunk(
            request_id.as_str(),
            &response_model_name,
            response.termination,
            Some(choice.finish_reason),
            unix_timestamp_secs(),
        );
        let delta_chunk = serialize_event_data(&completion_delta_chunk_for_choice(
            request_id.as_str(),
            response_model_name.as_str(),
            &choice,
            unix_timestamp_secs(),
        ))?;
        let terminal_chunk = serialize_event_data(&terminal_chunk)?;
        let events = vec![
            Ok::<_, Infallible>(Event::default().data(delta_chunk)),
            Ok::<_, Infallible>(Event::default().data(terminal_chunk)),
            Ok::<_, Infallible>(Event::default().data("[DONE]")),
        ];
        let mut response = Sse::new(iter(events)).into_response();
        insert_execution_headers(response.headers_mut(), state.as_ref());
        return Ok(response);
    }

    let psionic_harmony = if state.include_psionic_fields {
        parsed
    } else {
        None
    };
    let full_choice = choice.into_full_choice();
    let mut response = Json(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created: unix_timestamp_secs(),
        model: response_model_name,
        choices: vec![full_choice],
        usage: ChatCompletionUsage {
            prompt_tokens: response.usage.input_tokens,
            completion_tokens: response.usage.output_tokens,
            total_tokens: response.usage.input_tokens + response.usage.output_tokens,
        },
        psionic_metrics: state
            .include_psionic_fields
            .then(|| response.metrics.clone()),
        psionic_harmony,
        psionic_reasoning: surfaced_reasoning_response(
            parsed_reasoning.as_ref(),
            reasoning_request.as_ref(),
            state.include_psionic_fields,
        ),
        psionic_perf: state
            .include_psionic_fields
            .then(|| response.metrics.gpt_oss_perf.clone())
            .flatten(),
        psionic_output_text: state
            .include_psionic_fields
            .then(|| response.output.text.clone()),
        psionic_output_tokens: state.include_psionic_fields.then(|| {
            response
                .output
                .tokens
                .as_slice()
                .iter()
                .map(|token| token.as_u32())
                .collect()
        }),
        psionic_structured_output: None,
        psionic_structured_value: None,
        psionic_tool_calls: None,
        psionic_claim_posture: None,
        psionic_scheduler: None,
    })
    .into_response();
    insert_execution_headers(response.headers_mut(), state.as_ref());
    Ok(response)
}

async fn generic_chat_completions(
    State(state): State<Arc<OpenAiCompatState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    match handle_generic_chat_completions(state, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn handle_generic_chat_completions(
    state: Arc<OpenAiCompatState>,
    request: ChatCompletionRequest,
) -> Result<Response, OpenAiCompatHttpError> {
    let structured_output = structured_output_from_chat_request(&request)?;
    let tool_contract = tool_contract_from_chat_request(&request, structured_output.is_some())?;
    let route = resolve_generic_model_for_endpoint(
        state.as_ref(),
        request.model.as_deref(),
        RoutingEndpoint::ChatCompletions,
        {
            let mut route_request = RoutingRequest::new(RoutingEndpoint::ChatCompletions);
            if structured_output.is_some() {
                route_request = route_request.require_structured_outputs();
            }
            if tool_contract.is_some() {
                route_request = route_request.require_tool_calling();
            }
            route_request
        },
    )?;
    let loaded_model = route.loaded_model;
    if let Some(proxy) = bootstrap_proxy_for_route(state.as_ref(), &route.selection) {
        let route_execution =
            route_execution_status_for_bootstrap_proxy(&route.selection, proxy.mode);
        let response =
            proxy_bootstrap_chat_completions(proxy, &route, &request, &route_execution).await?;
        state.record_route_execution(route_execution);
        return Ok(response);
    }
    let model = loaded_model.decoder().ok_or_else(|| {
        OpenAiCompatHttpError::Internal(format!(
            "loaded model `{}` is missing decoder metadata",
            loaded_model.model_key
        ))
    })?;
    let reasoning_request =
        reasoning_request_for_family(request.psionic_reasoning.as_ref(), model.family)?;
    let mut tool_contract = tool_contract;
    if let Some(contract) = tool_contract.as_mut() {
        contract.minimum_required_tool_calls = required_tool_call_floor_from_chat_messages(
            &request.messages,
            contract,
            model.family,
            model.qwen35_multimodal_projection.as_ref(),
        )?;
    }
    let prompt_messages = apply_tool_contract_to_prompt_messages(
        chat_messages_to_prompt_messages_for_decoder(&request.messages, model)?,
        tool_contract.as_ref(),
    );
    let rendered = render_prompt_for_model(loaded_model, prompt_messages.as_slice())?;
    let request_id = next_generic_request_id(&state, "psionic-chatcmpl");
    let response_model_name = request
        .model
        .clone()
        .unwrap_or_else(|| loaded_model.canonical_name.clone());
    let options = generation_options_from_chat_request_for_family(
        &request,
        model.family,
        rendered.stop_sequences.as_slice(),
    );
    let mut options = options;
    options.structured_output =
        structured_output_from_tool_contract(tool_contract.as_ref())?.or(structured_output);
    let generation_request = GenerationRequest::new_text(
        request_id.clone(),
        model.descriptor.clone(),
        None,
        rendered.text,
        options,
    )
    .with_prefix_cache_control(request.psionic_prefix_cache.clone().unwrap_or_default());

    let response = worker_for_route(state.as_ref(), &route.selection)?
        .generate(route.selection.model_key.clone(), generation_request)
        .await
        .map_err(|error| {
            OpenAiCompatHttpError::from(GptOssOpenAiCompatGenerationError::Generation(error))
        })?;
    let route_execution = route_execution_status_for_local_route(&route.selection);
    state.record_route_execution(route_execution.clone());
    let parsed_reasoning = if reasoning_parser_for_decoder_family(model.family).is_some() {
        parse_reasoning_response_for_family(model.family, response.output.text.as_str())
            .ok()
            .flatten()
    } else {
        None
    };
    let parsed =
        if state.include_psionic_fields && matches!(model.family, GgufDecoderFamily::GptOss) {
            parse_gpt_oss_harmony_text(
                response.output.text.as_str(),
                GptOssHarmonyParseOptions {
                    role_hint: Some(PromptMessageRole::Assistant),
                    strict: false,
                },
            )
            .ok()
        } else {
            None
        };
    let tool_outcome =
        tool_call_outcome_from_response(request_id.as_str(), &response, tool_contract.as_ref())?;
    let choice = completion_choice_for_family(
        model.family,
        &response,
        parsed_reasoning.as_ref(),
        reasoning_request.as_ref(),
        tool_outcome.as_ref(),
    )?;
    let psionic_tool_calls = tool_outcome
        .as_ref()
        .map(|outcome| {
            outcome
                .tool_calls
                .clone()
                .into_iter()
                .map(ResolvedToolCall::into_psionic_tool_call)
                .collect::<Result<Vec<_>, OpenAiCompatHttpError>>()
        })
        .transpose()?
        .filter(|tool_calls| !tool_calls.is_empty());
    let structured_output_report = response
        .provenance
        .as_ref()
        .and_then(|value| value.structured_output.clone());
    let structured_output_value = response.output.structured.clone();
    let scheduler_receipt = response
        .provenance
        .as_ref()
        .and_then(|value| value.scheduler.clone());
    let prefix_cache_state = response
        .provenance
        .as_ref()
        .and_then(|value| value.prefix_cache_state);
    let prefix_cache_refusal_reason = response
        .provenance
        .as_ref()
        .and_then(|value| value.prefix_cache_refusal_reason);
    let prefix_tokens_reused = response.metrics.prefix_tokens_reused;
    let prefill_decode_mode = scheduler_receipt
        .as_ref()
        .and_then(|receipt| receipt.prefill_decode_mode)
        .or_else(|| {
            response
                .provenance
                .as_ref()
                .and_then(|value| value.delivery_proof.as_ref())
                .and_then(|proof| proof.prefill_decode_handoff.as_ref())
                .map(|handoff| handoff.mode)
        });
    let time_to_first_token_ns = response.metrics.time_to_first_token_ns;
    let inter_token_latency_ns = response.metrics.inter_token_latency_ns;
    if request.stream {
        let terminal_chunk = completion_terminal_chunk(
            request_id.as_str(),
            &response_model_name,
            response.termination,
            Some(choice.finish_reason),
            unix_timestamp_secs(),
        );
        let delta_chunks = completion_delta_chunks_for_choice(
            request_id.as_str(),
            response_model_name.as_str(),
            &choice,
            unix_timestamp_secs(),
        )
        .into_iter()
        .map(|chunk| serialize_event_data(&chunk))
        .collect::<Result<Vec<_>, _>>()?;
        let terminal_chunk = serialize_event_data(&terminal_chunk)?;
        let mut events = delta_chunks
            .into_iter()
            .map(|chunk| Ok::<_, Infallible>(Event::default().data(chunk)))
            .collect::<Vec<_>>();
        events.push(Ok::<_, Infallible>(Event::default().data(terminal_chunk)));
        events.push(Ok::<_, Infallible>(Event::default().data("[DONE]")));
        let mut response = Sse::new(iter(events)).into_response();
        insert_generic_execution_headers(
            response.headers_mut(),
            loaded_model,
            &route.selection,
            &route_execution,
            structured_output_report.as_ref(),
            scheduler_receipt.as_ref(),
            prefill_decode_mode,
            time_to_first_token_ns,
            inter_token_latency_ns,
            prefix_cache_state,
            prefix_cache_refusal_reason,
            prefix_tokens_reused,
        );
        return Ok(response);
    }

    let psionic_harmony = if state.include_psionic_fields {
        parsed
    } else {
        None
    };
    let body = ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created: unix_timestamp_secs(),
        model: response_model_name,
        choices: vec![choice.into_full_choice()],
        usage: ChatCompletionUsage {
            prompt_tokens: response.usage.input_tokens,
            completion_tokens: response.usage.output_tokens,
            total_tokens: response.usage.input_tokens + response.usage.output_tokens,
        },
        psionic_metrics: state
            .include_psionic_fields
            .then(|| response.metrics.clone()),
        psionic_harmony,
        psionic_reasoning: surfaced_reasoning_response(
            parsed_reasoning.as_ref(),
            reasoning_request.as_ref(),
            state.include_psionic_fields,
        ),
        psionic_perf: state
            .include_psionic_fields
            .then(|| response.metrics.gpt_oss_perf.clone())
            .flatten(),
        psionic_output_text: state
            .include_psionic_fields
            .then(|| response.output.text.clone()),
        psionic_output_tokens: state.include_psionic_fields.then(|| {
            response
                .output
                .tokens
                .as_slice()
                .iter()
                .map(|token| token.as_u32())
                .collect()
        }),
        psionic_structured_output: response
            .provenance
            .as_ref()
            .and_then(|value| value.structured_output.clone()),
        psionic_structured_value: structured_output_value,
        psionic_tool_calls,
        psionic_claim_posture: state
            .include_psionic_fields
            .then(|| {
                response
                    .provenance
                    .as_ref()
                    .and_then(|value| value.psion_served_output_claim_posture.clone())
            })
            .flatten(),
        psionic_scheduler: state
            .include_psionic_fields
            .then(|| scheduler_receipt.clone())
            .flatten(),
    };
    let mut response = Json(body).into_response();
    insert_generic_execution_headers(
        response.headers_mut(),
        loaded_model,
        &route.selection,
        &route_execution,
        structured_output_report.as_ref(),
        scheduler_receipt.as_ref(),
        prefill_decode_mode,
        time_to_first_token_ns,
        inter_token_latency_ns,
        prefix_cache_state,
        prefix_cache_refusal_reason,
        prefix_tokens_reused,
    );
    Ok(response)
}

async fn generic_responses(
    State(state): State<Arc<OpenAiCompatState>>,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    match handle_generic_responses(state, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn handle_generic_responses(
    state: Arc<OpenAiCompatState>,
    request: ResponsesRequest,
) -> Result<Response, OpenAiCompatHttpError> {
    if request.stream {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "streaming `/v1/responses` is not implemented on the generic Psionic server yet",
        )));
    }
    let response_state_request = resolved_response_state_request(&request)?;
    if matches!(
        response_state_request.continuation,
        ResponseContinuationMode::ContinueLastAssistant
    ) {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "`psionic_response_state.continuation = continue_last_assistant` is not available on the current prompt-replay `/v1/responses` runtime",
        )));
    }
    let response_state_context = state
        .response_state
        .lock()
        .map_err(|_| {
            OpenAiCompatHttpError::Internal(String::from(
                "generic response-state store is poisoned",
            ))
        })?
        .load_context(
            request.previous_response_id.as_deref(),
            request.conversation.as_deref(),
        )
        .map_err(response_state_error_into_http)?;
    let structured_output = structured_output_from_responses_request(&request)?;
    let tool_contract =
        tool_contract_from_responses_request(&request, structured_output.is_some())?;
    let route_request = {
        let mut route_request =
            RoutingRequest::new(RoutingEndpoint::Responses).require_response_state();
        if structured_output.is_some() {
            route_request = route_request.require_structured_outputs();
        }
        if tool_contract.is_some() {
            route_request = route_request.require_tool_calling();
        }
        if let Some(worker_id) = response_state_context.worker_id.as_deref() {
            route_request = route_request.prefer_worker(worker_id.to_string());
        }
        route_request
    };
    let route = match (
        request.model.as_deref(),
        response_state_context.model_key.as_deref(),
    ) {
        (Some(requested), _) => resolve_generic_model_for_endpoint(
            state.as_ref(),
            Some(requested),
            RoutingEndpoint::Responses,
            route_request.clone(),
        )?,
        (None, Some(model_key)) => resolve_generic_model_key_for_endpoint(
            state.as_ref(),
            model_key,
            RoutingEndpoint::Responses,
            route_request.clone(),
        )?,
        (None, None) => resolve_generic_model_for_endpoint(
            state.as_ref(),
            None,
            RoutingEndpoint::Responses,
            route_request,
        )?,
    };
    let loaded_model = route.loaded_model;
    if let Some(proxy) = bootstrap_proxy_for_route(state.as_ref(), &route.selection) {
        let route_execution =
            route_execution_status_for_bootstrap_proxy(&route.selection, proxy.mode);
        let response = proxy_bootstrap_responses(proxy, &route, &request, &route_execution).await?;
        state.record_route_execution(route_execution);
        return Ok(response);
    }
    if let Some(expected_model_key) = response_state_context.model_key.as_deref()
        && loaded_model.model_key != expected_model_key
    {
        return Err(OpenAiCompatHttpError::BadRequest(format!(
            "stateful `/v1/responses` continuation must stay on model `{}`",
            loaded_model.canonical_name
        )));
    }
    let model = loaded_model.decoder().ok_or_else(|| {
        OpenAiCompatHttpError::Internal(format!(
            "loaded model `{}` is missing decoder metadata",
            loaded_model.model_key
        ))
    })?;
    if !response_state_context.prompt_history.is_empty()
        && let Some(instructions) = request.instructions.as_deref()
        && leading_response_instructions(response_state_context.prompt_history.as_slice())
            != Some(instructions)
    {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "stateful `/v1/responses` continuation cannot change `instructions`; omit it or repeat the original value exactly",
        )));
    }
    let reasoning_request =
        reasoning_request_for_family(request.psionic_reasoning.as_ref(), model.family)?;
    let appended_prompt_messages = response_input_to_prompt_messages_with_options(
        &request,
        model,
        response_state_context.prompt_history.is_empty(),
        false,
    )?;
    let mut prompt_history = response_state_context.prompt_history.clone();
    prompt_history.extend(appended_prompt_messages.clone());
    let prompt_messages =
        apply_tool_contract_to_prompt_messages(prompt_history.clone(), tool_contract.as_ref());
    let rendered = render_prompt_for_model(loaded_model, prompt_messages.as_slice())?;
    let request_id = next_generic_request_id(&state, "psionic-resp");
    let response_model_name = request
        .model
        .clone()
        .unwrap_or_else(|| loaded_model.canonical_name.clone());
    let mut options = generation_options_from_responses_request(
        &request,
        model.family,
        rendered.stop_sequences.as_slice(),
    );
    options.structured_output =
        structured_output_from_tool_contract(tool_contract.as_ref())?.or(structured_output);
    let generation_request = GenerationRequest::new_text(
        request_id.clone(),
        model.descriptor.clone(),
        None,
        rendered.text,
        options,
    )
    .with_prefix_cache_control(request.psionic_prefix_cache.clone().unwrap_or_default());

    let response = worker_for_route(state.as_ref(), &route.selection)?
        .generate(route.selection.model_key.clone(), generation_request)
        .await
        .map_err(|error| {
            OpenAiCompatHttpError::from(GptOssOpenAiCompatGenerationError::Generation(error))
        })?;
    let route_execution = route_execution_status_for_local_route(&route.selection);
    state.record_route_execution(route_execution.clone());
    let parsed_reasoning = if reasoning_parser_for_decoder_family(model.family).is_some() {
        parse_reasoning_response_for_family(model.family, response.output.text.as_str())
            .ok()
            .flatten()
    } else {
        None
    };
    let parsed =
        if state.include_psionic_fields && matches!(model.family, GgufDecoderFamily::GptOss) {
            parse_gpt_oss_harmony_text(
                response.output.text.as_str(),
                GptOssHarmonyParseOptions {
                    role_hint: Some(PromptMessageRole::Assistant),
                    strict: false,
                },
            )
            .ok()
        } else {
            None
        };
    let tool_outcome =
        tool_call_outcome_from_response(request_id.as_str(), &response, tool_contract.as_ref())?;
    let choice = completion_choice_for_family(
        model.family,
        &response,
        parsed_reasoning.as_ref(),
        reasoning_request.as_ref(),
        tool_outcome.as_ref(),
    )?;
    let content = choice.content.clone().unwrap_or_default();
    let psionic_tool_calls = tool_outcome
        .as_ref()
        .map(|outcome| {
            outcome
                .tool_calls
                .clone()
                .into_iter()
                .map(ResolvedToolCall::into_psionic_tool_call)
                .collect::<Result<Vec<_>, OpenAiCompatHttpError>>()
        })
        .transpose()?
        .filter(|tool_calls| !tool_calls.is_empty());
    let structured_output_report = response
        .provenance
        .as_ref()
        .and_then(|value| value.structured_output.clone());
    let structured_output_value = response.output.structured.clone();
    let scheduler_receipt = response
        .provenance
        .as_ref()
        .and_then(|value| value.scheduler.clone());
    let prefix_cache_state = response
        .provenance
        .as_ref()
        .and_then(|value| value.prefix_cache_state);
    let prefix_cache_refusal_reason = response
        .provenance
        .as_ref()
        .and_then(|value| value.prefix_cache_refusal_reason);
    let prefix_tokens_reused = response.metrics.prefix_tokens_reused;
    let prefill_decode_mode = scheduler_receipt
        .as_ref()
        .and_then(|receipt| receipt.prefill_decode_mode)
        .or_else(|| {
            response
                .provenance
                .as_ref()
                .and_then(|value| value.delivery_proof.as_ref())
                .and_then(|proof| proof.prefill_decode_handoff.as_ref())
                .map(|handoff| handoff.mode)
        });
    let time_to_first_token_ns = response.metrics.time_to_first_token_ns;
    let inter_token_latency_ns = response.metrics.inter_token_latency_ns;
    let assistant_history = assistant_history_from_response(
        model.family,
        response.output.text.as_str(),
        parsed.as_ref(),
    );
    let response_state_capability = current_response_state_capability(state.as_ref());
    let assigned_conversation_id = response_state_request.store.then(|| {
        if response_state_request.invalidate_references
            || response_state_context.conversation_id.is_none()
        {
            next_conversation_id(state.as_ref())
        } else {
            response_state_context
                .conversation_id
                .clone()
                .expect("checked conversation presence above")
        }
    });
    let mut stored_prompt_history = prompt_history.clone();
    stored_prompt_history.extend(assistant_history.clone());
    let (conversation, invalidated_references) = {
        let mut response_state = state.response_state.lock().map_err(|_| {
            OpenAiCompatHttpError::Internal(String::from(
                "generic response-state store is poisoned",
            ))
        })?;
        let conversation = if response_state_request.store {
            response_state
                .record_response(ResponseStateRecord {
                    response_id: request_id.clone(),
                    model_key: loaded_model.model_key.clone(),
                    worker_id: route.selection.worker_id.clone(),
                    conversation_id: assigned_conversation_id.clone(),
                    prompt_history: stored_prompt_history.clone(),
                })
                .map_err(response_state_error_into_http)?
        } else {
            None
        };
        let invalidated = if response_state_request.invalidate_references {
            let invalidated_conversation_id = response_state_context
                .conversation_id
                .as_deref()
                .filter(|candidate| Some(*candidate) != assigned_conversation_id.as_deref());
            response_state
                .invalidate_references(
                    request.previous_response_id.as_deref(),
                    invalidated_conversation_id,
                )
                .map_err(response_state_error_into_http)?
        } else {
            Vec::new()
        };
        (conversation, invalidated)
    };
    let body = ResponsesResponse {
        id: request_id.clone(),
        object: "response",
        created_at: unix_timestamp_secs(),
        status: "completed",
        model: response_model_name,
        output: responses_output_items(request_id.as_str(), &choice),
        output_text: content,
        usage: ResponsesUsage {
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            total_tokens: response.usage.input_tokens + response.usage.output_tokens,
        },
        previous_response_id: response_state_context.previous_response_id.clone(),
        conversation,
        psionic_metrics: state
            .include_psionic_fields
            .then(|| response.metrics.clone()),
        psionic_harmony: state.include_psionic_fields.then_some(parsed).flatten(),
        psionic_reasoning: surfaced_reasoning_response(
            parsed_reasoning.as_ref(),
            reasoning_request.as_ref(),
            state.include_psionic_fields,
        ),
        psionic_response_state: Some(ResponseStateReceipt {
            storage: response_state_capability.storage.clone(),
            retention_scope: response_state_capability.retention_scope.clone(),
            cache_behavior: response_state_capability.cache_behavior.clone(),
            stored: response_state_request.store,
            continuation: response_state_request.continuation,
            previous_response_id: response_state_context.previous_response_id.clone(),
            conversation_id: assigned_conversation_id.clone(),
            replayed_prompt_messages: response_state_context.replayed_prompt_messages,
            input_messages_appended: appended_prompt_messages.len(),
            assistant_messages_recorded: if response_state_request.store {
                assistant_history.len()
            } else {
                0
            },
            max_responses: response_state_capability.max_responses,
            max_conversations: response_state_capability.max_conversations,
            max_items_per_conversation: response_state_capability.max_items_per_conversation,
            conversation_item_count: if response_state_request.store {
                stored_prompt_history.len()
            } else {
                response_state_context.conversation_item_count
            },
            invalidated_references,
        }),
        psionic_perf: state
            .include_psionic_fields
            .then(|| response.metrics.gpt_oss_perf.clone())
            .flatten(),
        psionic_output_tokens: state.include_psionic_fields.then(|| {
            response
                .output
                .tokens
                .as_slice()
                .iter()
                .map(|token| token.as_u32())
                .collect()
        }),
        psionic_structured_output: response
            .provenance
            .as_ref()
            .and_then(|value| value.structured_output.clone()),
        psionic_structured_value: structured_output_value,
        psionic_tool_calls,
        psionic_claim_posture: state
            .include_psionic_fields
            .then(|| {
                response
                    .provenance
                    .as_ref()
                    .and_then(|value| value.psion_served_output_claim_posture.clone())
            })
            .flatten(),
        psionic_scheduler: state
            .include_psionic_fields
            .then(|| scheduler_receipt.clone())
            .flatten(),
    };
    let mut response = Json(body).into_response();
    insert_generic_execution_headers(
        response.headers_mut(),
        loaded_model,
        &route.selection,
        &route_execution,
        structured_output_report.as_ref(),
        scheduler_receipt.as_ref(),
        prefill_decode_mode,
        time_to_first_token_ns,
        inter_token_latency_ns,
        prefix_cache_state,
        prefix_cache_refusal_reason,
        prefix_tokens_reused,
    );
    Ok(response)
}

async fn generic_embeddings(
    State(state): State<Arc<OpenAiCompatState>>,
    Json(request): Json<EmbeddingsRequest>,
) -> Response {
    match handle_generic_embeddings(state, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn handle_generic_embeddings(
    state: Arc<OpenAiCompatState>,
    request: EmbeddingsRequest,
) -> Result<Response, OpenAiCompatHttpError> {
    if let Some(encoding_format) = request.encoding_format.as_deref()
        && encoding_format != "float"
    {
        return Err(OpenAiCompatHttpError::BadRequest(format!(
            "unsupported `encoding_format` `{encoding_format}` for `/v1/embeddings`; only `float` is supported"
        )));
    }
    let loaded_model = resolve_generic_model_for_endpoint(
        state.as_ref(),
        request.model.as_deref(),
        RoutingEndpoint::Embeddings,
        RoutingRequest::new(RoutingEndpoint::Embeddings),
    )?;
    let route = loaded_model;
    let loaded_model = route.loaded_model;
    if let Some(proxy) = bootstrap_proxy_for_route(state.as_ref(), &route.selection) {
        let route_execution =
            route_execution_status_for_bootstrap_proxy(&route.selection, proxy.mode);
        let response =
            proxy_bootstrap_embeddings(proxy, &route, &request, &route_execution).await?;
        state.record_route_execution(route_execution);
        return Ok(response);
    }
    let model = loaded_model.embeddings().ok_or_else(|| {
        OpenAiCompatHttpError::Internal(format!(
            "loaded model `{}` is missing embeddings metadata",
            loaded_model.model_key
        ))
    })?;
    let request_id = next_generic_request_id(&state, "psionic-embed");
    let response_model_name = request
        .model
        .clone()
        .unwrap_or_else(|| loaded_model.canonical_name.clone());
    let embedding_request = if let Some(dimensions) = request.dimensions {
        EmbeddingRequest::new(
            request_id.clone(),
            model.descriptor.clone(),
            request.input.into_vec(),
        )
        .with_output_dimensions(dimensions)
    } else {
        EmbeddingRequest::new(
            request_id.clone(),
            model.descriptor.clone(),
            request.input.into_vec(),
        )
    };
    let response = worker_for_route(state.as_ref(), &route.selection)?
        .embed(route.selection.model_key.clone(), embedding_request)
        .await?;
    let route_execution = route_execution_status_for_local_route(&route.selection);
    state.record_route_execution(route_execution.clone());
    let body = EmbeddingsResponse {
        object: "list",
        data: response
            .embeddings
            .iter()
            .map(|embedding| EmbeddingsResponseData {
                object: "embedding",
                index: embedding.index,
                embedding: embedding.values.clone(),
            })
            .collect(),
        model: response_model_name,
        usage: response
            .metrics
            .prompt_eval_count
            .map(|prompt_tokens| EmbeddingsUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            }),
        psionic_metrics: state
            .include_psionic_fields
            .then(|| response.metrics.clone()),
        psionic_provenance: state
            .include_psionic_fields
            .then(|| response.provenance.clone())
            .flatten(),
    };
    let mut response = Json(body).into_response();
    insert_generic_execution_headers(
        response.headers_mut(),
        loaded_model,
        &route.selection,
        &route_execution,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    Ok(response)
}

async fn proxy_chat_completions(
    state: &GptOssOpenAiCompatState,
    proxy: &LlamaCppProxyState,
    request: &ChatCompletionRequest,
) -> Result<Response, OpenAiCompatHttpError> {
    let upstream = proxy
        .client
        .post(format!("{}/v1/chat/completions", proxy.base_url))
        .json(request)
        .send()
        .await
        .map_err(|error| {
            OpenAiCompatHttpError::from(GptOssOpenAiCompatGenerationError::BackendUnavailable {
                backend: "metal-proxy",
                status: psionic_runtime::HealthStatus::Offline,
                message: format!("llama.cpp proxy request failed: {error}"),
            })
        })?;
    let status = upstream.status();
    let content_type = upstream
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .cloned();
    let body = upstream.bytes().await.map_err(|error| {
        OpenAiCompatHttpError::from(GptOssOpenAiCompatGenerationError::BackendUnavailable {
            backend: "metal-proxy",
            status: psionic_runtime::HealthStatus::Offline,
            message: format!("llama.cpp proxy response read failed: {error}"),
        })
    })?;
    let mut response = Response::builder().status(status);
    if let Some(content_type) = content_type {
        response = response.header(axum::http::header::CONTENT_TYPE, content_type);
    }
    let mut response = response
        .body(axum::body::Body::from(body))
        .map_err(|error| OpenAiCompatHttpError::BadRequest(error.to_string()))?;
    insert_execution_headers(response.headers_mut(), state);
    Ok(response)
}

async fn proxy_bootstrap_json_request<T: Serialize>(
    proxy: &BootstrapProxyState,
    endpoint: &str,
    request: &T,
) -> Result<Response, OpenAiCompatHttpError> {
    let upstream = proxy
        .client
        .post(format!(
            "{}/{}",
            proxy.base_url,
            endpoint.trim_start_matches('/')
        ))
        .json(request)
        .send()
        .await
        .map_err(|error| {
            OpenAiCompatHttpError::Internal(format!(
                "bootstrap proxy request to `{}` failed: {error}",
                proxy.base_url
            ))
        })?;
    let status = upstream.status();
    let upstream_headers = upstream.headers().clone();
    let body = upstream.bytes().await.map_err(|error| {
        OpenAiCompatHttpError::Internal(format!(
            "bootstrap proxy response read failed for `{}`: {error}",
            proxy.base_url
        ))
    })?;
    let mut response = Response::builder()
        .status(status)
        .body(axum::body::Body::from(body))
        .map_err(|error| OpenAiCompatHttpError::BadRequest(error.to_string()))?;
    for (name, value) in &upstream_headers {
        if name != axum::http::header::CONTENT_LENGTH {
            response.headers_mut().insert(name, value.clone());
        }
    }
    Ok(response)
}

async fn proxy_bootstrap_chat_completions(
    proxy: &BootstrapProxyState,
    route: &ResolvedGenericRoute<'_>,
    request: &ChatCompletionRequest,
    route_execution: &MeshManagementRouteExecutionStatus,
) -> Result<Response, OpenAiCompatHttpError> {
    let mut proxied = request.clone();
    proxied.model = Some(route.selection.canonical_name.clone());
    let mut response =
        proxy_bootstrap_json_request(proxy, RoutingEndpoint::ChatCompletions.path(), &proxied)
            .await?;
    insert_route_execution_headers(response.headers_mut(), route_execution);
    Ok(response)
}

async fn proxy_bootstrap_responses(
    proxy: &BootstrapProxyState,
    route: &ResolvedGenericRoute<'_>,
    request: &ResponsesRequest,
    route_execution: &MeshManagementRouteExecutionStatus,
) -> Result<Response, OpenAiCompatHttpError> {
    let mut proxied = request.clone();
    proxied.model = Some(route.selection.canonical_name.clone());
    let mut response =
        proxy_bootstrap_json_request(proxy, RoutingEndpoint::Responses.path(), &proxied).await?;
    insert_route_execution_headers(response.headers_mut(), route_execution);
    Ok(response)
}

async fn proxy_bootstrap_embeddings(
    proxy: &BootstrapProxyState,
    route: &ResolvedGenericRoute<'_>,
    request: &EmbeddingsRequest,
    route_execution: &MeshManagementRouteExecutionStatus,
) -> Result<Response, OpenAiCompatHttpError> {
    let mut proxied = request.clone();
    proxied.model = Some(route.selection.canonical_name.clone());
    let mut response =
        proxy_bootstrap_json_request(proxy, RoutingEndpoint::Embeddings.path(), &proxied).await?;
    insert_route_execution_headers(response.headers_mut(), route_execution);
    Ok(response)
}

fn insert_execution_headers(headers: &mut HeaderMap, state: &GptOssOpenAiCompatState) {
    headers.insert(
        HeaderName::from_static("x-psionic-backend"),
        HeaderValue::from_static(state.backend_label),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-served-backend"),
        HeaderValue::from_static(state.backend_label),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-execution-mode"),
        HeaderValue::from_static(state.execution_mode_label),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-execution-engine"),
        HeaderValue::from_static(state.execution_engine_label),
    );
    insert_local_serving_truth_headers(headers, state.local_serving_truth);
}

fn insert_local_serving_truth_headers(headers: &mut HeaderMap, truth: LocalServingTruth) {
    headers.insert(
        HeaderName::from_static("x-psionic-residency-mode"),
        HeaderValue::from_static(truth.residency_mode),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-hybrid-offload"),
        HeaderValue::from_static(truth.hybrid_offload),
    );
    if let Some(layers) = truth.hybrid_offload_layers {
        headers.insert(
            HeaderName::from_static("x-psionic-hybrid-offload-layers"),
            HeaderValue::from_str(layers.to_string().as_str())
                .unwrap_or_else(|_| HeaderValue::from_static("invalid")),
        );
    }
    headers.insert(
        HeaderName::from_static("x-psionic-fallback-policy"),
        HeaderValue::from_static(truth.fallback_policy),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-performance-class"),
        HeaderValue::from_static(truth.performance_class),
    );
}

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: ChatCompletionUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_metrics: Option<GenerationMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_harmony: Option<GptOssHarmonyParsedOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_reasoning: Option<ParsedReasoningResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_perf: Option<GptOssPerformanceMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_output_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_output_tokens: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_structured_output: Option<StructuredOutputExecutionReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_structured_value: Option<StructuredOutputValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_tool_calls: Option<Vec<PsionicToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_claim_posture: Option<crate::PsionServedOutputClaimPosture>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_scheduler: Option<GenerationSchedulerRequestReceipt>,
}

#[derive(Clone, Debug, Serialize)]
struct ResponsesResponse {
    id: String,
    object: &'static str,
    created_at: u64,
    status: &'static str,
    model: String,
    output: Vec<ResponsesOutputItem>,
    output_text: String,
    usage: ResponsesUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    conversation: Option<ResponseConversationRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_metrics: Option<GenerationMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_harmony: Option<GptOssHarmonyParsedOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_reasoning: Option<ParsedReasoningResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_response_state: Option<ResponseStateReceipt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_perf: Option<GptOssPerformanceMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_output_tokens: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_structured_output: Option<StructuredOutputExecutionReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_structured_value: Option<StructuredOutputValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_tool_calls: Option<Vec<PsionicToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_claim_posture: Option<crate::PsionServedOutputClaimPosture>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_scheduler: Option<GenerationSchedulerRequestReceipt>,
}

#[derive(Clone, Debug, Serialize)]
struct ResponsesOutputItem {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    status: &'static str,
    role: &'static str,
    content: Vec<ResponsesOutputContent>,
}

#[derive(Clone, Debug, Serialize)]
struct ResponsesOutputContent {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
}

#[derive(Clone, Debug, Serialize)]
struct ResponsesUsage {
    input_tokens: usize,
    output_tokens: usize,
    total_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionChoice {
    index: usize,
    message: ChatCompletionResponseMessage,
    finish_reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionResponseMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatCompletionToolCall>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatCompletionToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: ChatCompletionToolCallFunction,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChatCompletionToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionChunkToolCall {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function: Option<ChatCompletionChunkToolCallFunctionDelta>,
}

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionChunkToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
struct EmbeddingsResponse {
    object: &'static str,
    data: Vec<EmbeddingsResponseData>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<EmbeddingsUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_metrics: Option<EmbeddingMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_provenance: Option<EmbeddingProvenance>,
}

#[derive(Clone, Debug, Serialize)]
struct EmbeddingsResponseData {
    object: &'static str,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Clone, Debug, Serialize)]
struct EmbeddingsUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChunkChoice>,
}

#[derive(Clone, Debug, Serialize)]
struct ChatCompletionChunkChoice {
    index: usize,
    delta: ChatCompletionChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize, Default)]
struct ChatCompletionChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatCompletionChunkToolCall>>,
}

#[derive(Clone, Debug)]
struct ParsedCompletionChoice {
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Vec<ChatCompletionToolCall>,
    finish_reason: &'static str,
}

impl ParsedCompletionChoice {
    fn into_full_choice(self) -> ChatCompletionChoice {
        ChatCompletionChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                role: "assistant",
                content: self.content,
                reasoning_content: self.reasoning_content,
                tool_calls: (!self.tool_calls.is_empty()).then_some(self.tool_calls),
            },
            finish_reason: self.finish_reason,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct PsionicToolCall {
    id: String,
    name: String,
    arguments: serde_json::Value,
    raw_arguments: String,
}

fn completion_choice(
    response: &crate::GenerationResponse,
    parsed_reasoning: Option<&ParsedReasoningResponse>,
    reasoning_request: Option<&ResolvedReasoningRequest>,
) -> ParsedCompletionChoice {
    let content = parsed_reasoning
        .and_then(|parsed| parsed.final_content.clone())
        .unwrap_or_else(|| response.output.text.clone());
    ParsedCompletionChoice {
        content: Some(content),
        reasoning_content: reasoning_request.and_then(|request| match request.mode {
            PsionicReasoningMode::Separate => {
                parsed_reasoning.and_then(|parsed| parsed.reasoning_content.clone())
            }
            PsionicReasoningMode::Suppress => None,
        }),
        tool_calls: Vec::new(),
        finish_reason: finish_reason(response.termination),
    }
}

fn completion_terminal_chunk(
    request_id: &str,
    model: &str,
    termination: TerminationReason,
    finish_reason_override: Option<&'static str>,
    created: u64,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: request_id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkDelta::default(),
            finish_reason: Some(finish_reason_override.unwrap_or(finish_reason(termination))),
        }],
    }
}

fn completion_stream_tool_calls(
    tool_calls: &[ChatCompletionToolCall],
) -> Vec<ChatCompletionChunkToolCall> {
    tool_calls
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, tool_call)| ChatCompletionChunkToolCall {
            index,
            id: Some(tool_call.id),
            kind: Some("function"),
            function: Some(ChatCompletionChunkToolCallFunctionDelta {
                name: Some(tool_call.function.name),
                arguments: Some(tool_call.function.arguments),
            }),
        })
        .collect()
}

fn completion_delta_chunk(
    request_id: &str,
    model: &str,
    role: Option<&'static str>,
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionChunkToolCall>>,
    created: u64,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: request_id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkDelta {
                role,
                content,
                reasoning_content,
                tool_calls,
            },
            finish_reason: None,
        }],
    }
}

fn completion_delta_chunk_for_choice(
    request_id: &str,
    model: &str,
    choice: &ParsedCompletionChoice,
    created: u64,
) -> ChatCompletionChunk {
    completion_delta_chunk(
        request_id,
        model,
        Some("assistant"),
        choice.content.clone(),
        choice.reasoning_content.clone(),
        (!choice.tool_calls.is_empty())
            .then(|| completion_stream_tool_calls(choice.tool_calls.as_slice())),
        created,
    )
}

fn split_streaming_text_segments(text: &str, max_chars: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        let mut char_count = 0usize;
        let mut end = text.len();
        let mut last_break = None;
        for (offset, ch) in text[start..].char_indices() {
            char_count = char_count.saturating_add(1);
            let next = start + offset + ch.len_utf8();
            if ch.is_whitespace() {
                last_break = Some(next);
            }
            if char_count >= max_chars {
                end = last_break.filter(|value| *value > start).unwrap_or(next);
                break;
            }
        }
        if char_count < max_chars {
            end = text.len();
        }
        if end == start {
            end = text[start..]
                .chars()
                .next()
                .map(|ch| start + ch.len_utf8())
                .unwrap_or(text.len());
        }
        segments.push(text[start..end].to_string());
        start = end;
    }
    segments
}

fn completion_delta_chunks_for_choice(
    request_id: &str,
    model: &str,
    choice: &ParsedCompletionChoice,
    created: u64,
) -> Vec<ChatCompletionChunk> {
    let mut chunks = Vec::new();
    let mut next_role = Some("assistant");
    let mut push_chunk =
        |chunks: &mut Vec<ChatCompletionChunk>,
         content: Option<String>,
         reasoning_content: Option<String>,
         tool_calls: Option<Vec<ChatCompletionChunkToolCall>>| {
            chunks.push(completion_delta_chunk(
                request_id,
                model,
                next_role.take(),
                content,
                reasoning_content,
                tool_calls,
                created,
            ));
        };

    for segment in split_streaming_text_segments(
        choice.reasoning_content.as_deref().unwrap_or_default(),
        STREAMING_TEXT_DELTA_MAX_CHARS,
    ) {
        push_chunk(&mut chunks, None, Some(segment), None);
    }
    if !choice.tool_calls.is_empty() {
        push_chunk(
            &mut chunks,
            None,
            None,
            Some(completion_stream_tool_calls(choice.tool_calls.as_slice())),
        );
        return chunks;
    }
    for segment in split_streaming_text_segments(
        choice.content.as_deref().unwrap_or_default(),
        STREAMING_TEXT_DELTA_MAX_CHARS,
    ) {
        push_chunk(&mut chunks, Some(segment), None, None);
    }
    if chunks.is_empty() {
        push_chunk(&mut chunks, None, None, None);
    }
    chunks
}

fn responses_output_items(
    request_id: &str,
    choice: &ParsedCompletionChoice,
) -> Vec<ResponsesOutputItem> {
    let mut content_items = Vec::new();
    if let Some(reasoning) = choice.reasoning_content.clone() {
        content_items.push(ResponsesOutputContent {
            kind: "reasoning_text",
            text: reasoning,
        });
    }
    if let Some(content) = choice.content.clone()
        && !content.is_empty()
    {
        content_items.push(ResponsesOutputContent {
            kind: "output_text",
            text: content,
        });
    }
    if content_items.is_empty() {
        return Vec::new();
    }
    vec![ResponsesOutputItem {
        id: format!("{request_id}-msg-0"),
        kind: "message",
        status: "completed",
        role: "assistant",
        content: content_items,
    }]
}

fn serialize_event_data(value: &impl Serialize) -> Result<String, OpenAiCompatHttpError> {
    serde_json::to_string(value).map_err(|error| {
        OpenAiCompatHttpError::Internal(format!("failed to serialize OpenAI stream event: {error}"))
    })
}

fn finish_reason(termination: TerminationReason) -> &'static str {
    match termination {
        TerminationReason::EndOfSequence => "stop",
        TerminationReason::MaxOutputTokens | TerminationReason::ContextLimit => "length",
        TerminationReason::Cancelled
        | TerminationReason::Disconnected
        | TerminationReason::Error => "stop",
    }
}

fn next_request_id(state: &GptOssOpenAiCompatState) -> String {
    let next = state.request_counter.fetch_add(1, Ordering::Relaxed);
    format!("psionic-chatcmpl-{next}")
}

fn next_generic_request_id(state: &OpenAiCompatState, prefix: &str) -> String {
    let next = state.request_counter.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{next}")
}

fn insert_generic_execution_headers(
    headers: &mut HeaderMap,
    loaded_model: &OpenAiCompatLoadedModel,
    route_selection: &RouteSelection,
    route_execution: &MeshManagementRouteExecutionStatus,
    structured_output: Option<&StructuredOutputExecutionReport>,
    scheduler: Option<&GenerationSchedulerRequestReceipt>,
    prefill_decode_mode: Option<psionic_runtime::PrefillDecodeExecutionMode>,
    time_to_first_token_ns: Option<u64>,
    inter_token_latency_ns: Option<u64>,
    prefix_cache_state: Option<PrefixCacheState>,
    prefix_cache_refusal_reason: Option<PrefixCacheRefusalReason>,
    prefix_tokens_reused: Option<usize>,
) {
    headers.insert(
        HeaderName::from_static("x-psionic-backend"),
        HeaderValue::from_str(route_selection.backend_label.as_str())
            .unwrap_or_else(|_| HeaderValue::from_static("invalid")),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-served-backend"),
        HeaderValue::from_str(route_selection.backend_label.as_str())
            .unwrap_or_else(|_| HeaderValue::from_static("invalid")),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-execution-mode"),
        HeaderValue::from_str(route_selection.execution_mode_label.as_str())
            .unwrap_or_else(|_| HeaderValue::from_static("invalid")),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-execution-engine"),
        HeaderValue::from_str(route_selection.execution_engine_label.as_str())
            .unwrap_or_else(|_| HeaderValue::from_static("invalid")),
    );
    insert_local_serving_truth_headers(headers, loaded_model.local_serving_truth());
    insert_route_execution_headers(headers, route_execution);
    headers.insert(
        HeaderName::from_static("x-psionic-route-worker"),
        HeaderValue::from_str(route_selection.worker_id.as_str())
            .unwrap_or_else(|_| HeaderValue::from_static("invalid")),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-route-strategy"),
        HeaderValue::from_static(match route_selection.metrics.strategy {
            RouteSelectionStrategy::FirstReady => "first_ready",
            RouteSelectionStrategy::CacheAware => "cache_aware",
            RouteSelectionStrategy::WarmAware => "warm_aware",
            RouteSelectionStrategy::PowerOfTwoLeastLoaded => "power_of_two_least_loaded",
        }),
    );
    insert_usize_header(
        headers,
        "x-psionic-route-eligible-workers",
        route_selection.metrics.eligible_workers,
    );
    insert_usize_header(
        headers,
        "x-psionic-route-warm-workers",
        route_selection.metrics.warm_workers,
    );
    insert_usize_header(
        headers,
        "x-psionic-route-cache-matches",
        route_selection.metrics.cache_matches,
    );
    insert_usize_header(
        headers,
        "x-psionic-route-sampled-workers",
        route_selection.metrics.sampled_workers,
    );
    insert_usize_header(
        headers,
        "x-psionic-route-active-requests",
        route_selection.metrics.selected_active_requests,
    );
    if let Some(fallback_reason) = route_selection.metrics.fallback_reason.as_deref()
        && let Ok(value) = HeaderValue::from_str(fallback_reason)
    {
        headers.insert(HeaderName::from_static("x-psionic-route-fallback"), value);
    }
    headers.insert(
        HeaderName::from_static("x-psionic-batch-posture"),
        HeaderValue::from_static(batch_posture_label(
            route_selection.execution_profile.batch_posture,
        )),
    );
    if let Some(scheduler) = scheduler {
        headers.insert(
            HeaderName::from_static("x-psionic-scheduling-class"),
            HeaderValue::from_static(match scheduler.scheduling_class {
                psionic_runtime::GenerationSchedulingClass::Prefill => "prefill",
                psionic_runtime::GenerationSchedulingClass::Decode => "decode",
                psionic_runtime::GenerationSchedulingClass::MixedPrefillDecode => {
                    "mixed_prefill_decode"
                }
                psionic_runtime::GenerationSchedulingClass::FallbackSingleRequest => {
                    "fallback_single_request"
                }
            }),
        );
    }
    if let Some(prefill_decode_mode) = prefill_decode_mode {
        headers.insert(
            HeaderName::from_static("x-psionic-prefill-decode-mode"),
            HeaderValue::from_static(prefill_decode_mode.as_str()),
        );
    }
    if let Some(time_to_first_token_ns) = time_to_first_token_ns
        && let Ok(value) = HeaderValue::from_str(time_to_first_token_ns.to_string().as_str())
    {
        headers.insert(HeaderName::from_static("x-psionic-ttft-ns"), value);
    }
    if let Some(inter_token_latency_ns) = inter_token_latency_ns
        && let Ok(value) = HeaderValue::from_str(inter_token_latency_ns.to_string().as_str())
    {
        headers.insert(HeaderName::from_static("x-psionic-itl-ns"), value);
    }
    if let Some(prefix_cache_state) = prefix_cache_state {
        headers.insert(
            HeaderName::from_static("x-psionic-prefix-cache-state"),
            HeaderValue::from_static(match prefix_cache_state {
                PrefixCacheState::None => "none",
                PrefixCacheState::Hit => "hit",
                PrefixCacheState::Miss => "miss",
                PrefixCacheState::Bypassed => "bypassed",
                PrefixCacheState::Rebuilt => "rebuilt",
            }),
        );
    }
    if let Some(prefix_cache_refusal_reason) = prefix_cache_refusal_reason {
        headers.insert(
            HeaderName::from_static("x-psionic-prefix-cache-refusal"),
            HeaderValue::from_static(match prefix_cache_refusal_reason {
                PrefixCacheRefusalReason::RequestOptOut => "request_opt_out",
                PrefixCacheRefusalReason::ForcedInvalidation => "forced_invalidation",
                PrefixCacheRefusalReason::TenantBoundary => "tenant_boundary",
                PrefixCacheRefusalReason::SamplerBoundary => "sampler_boundary",
                PrefixCacheRefusalReason::SessionBoundState => "session_bound_state",
            }),
        );
    }
    if let Some(prefix_tokens_reused) = prefix_tokens_reused {
        if let Ok(value) = HeaderValue::from_str(prefix_tokens_reused.to_string().as_str()) {
            headers.insert(
                HeaderName::from_static("x-psionic-prefix-cache-reused-tokens"),
                value,
            );
        }
    }
    insert_structured_output_headers(headers, structured_output);
}

fn batch_posture_label(batch_posture: psionic_runtime::BatchExecutionPosture) -> &'static str {
    match batch_posture {
        psionic_runtime::BatchExecutionPosture::SingleRequestOnly => "single_request_only",
        psionic_runtime::BatchExecutionPosture::CallerStaticBatch => "caller_static_batch",
        psionic_runtime::BatchExecutionPosture::SchedulerStaticBatch => "scheduler_static_batch",
        psionic_runtime::BatchExecutionPosture::ContinuousBatch => "continuous_batch",
    }
}

fn insert_usize_header(headers: &mut HeaderMap, name: &'static str, value: usize) {
    if let Ok(value) = HeaderValue::from_str(value.to_string().as_str()) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

fn insert_structured_output_headers(
    headers: &mut HeaderMap,
    structured_output: Option<&StructuredOutputExecutionReport>,
) {
    let Some(structured_output) = structured_output else {
        return;
    };
    headers.insert(
        HeaderName::from_static("x-psionic-structured-output-mode"),
        HeaderValue::from_static(structured_output.mode.label()),
    );
    headers.insert(
        HeaderName::from_static("x-psionic-structured-output-parser"),
        HeaderValue::from_static(structured_output.parser.label()),
    );
}

#[derive(Clone, Debug)]
struct GenericRenderedPrompt {
    text: String,
    stop_sequences: Vec<String>,
}

struct ResolvedGenericRoute<'a> {
    selection: RouteSelection,
    loaded_model: &'a OpenAiCompatLoadedModel,
}

#[cfg(test)]
fn resolve_generic_model<'a>(
    state: &'a OpenAiCompatState,
    requested: Option<&str>,
) -> Result<&'a OpenAiCompatLoadedModel, OpenAiCompatHttpError> {
    Ok(resolve_generic_route(
        state,
        match requested {
            Some(requested) => RoutingTarget::RequestedModel(requested.to_string()),
            None => RoutingTarget::Default,
        },
        None,
    )?
    .loaded_model)
}

fn resolve_generic_route<'a>(
    state: &'a OpenAiCompatState,
    target: RoutingTarget,
    request: Option<RoutingRequest>,
) -> Result<ResolvedGenericRoute<'a>, OpenAiCompatHttpError> {
    let request = match request {
        Some(mut request) => {
            request.target = target;
            request
        }
        None => {
            let mut request = RoutingRequest::new(RoutingEndpoint::ChatCompletions);
            request.target = target;
            request
        }
    };
    let selection = state
        .router
        .resolve(&request)
        .map_err(openai_http_error_from_routing)?;
    let loaded_model = state
        .models_by_key
        .get(selection.model_key.as_str())
        .ok_or_else(|| {
            OpenAiCompatHttpError::Internal(format!(
                "loaded model `{}` selected by router is missing",
                selection.model_key
            ))
        })?;
    state.publish_route_selection_event(&request, &selection);
    Ok(ResolvedGenericRoute {
        selection,
        loaded_model,
    })
}

fn resolve_generic_model_for_endpoint<'a>(
    state: &'a OpenAiCompatState,
    requested: Option<&str>,
    endpoint: RoutingEndpoint,
    request: RoutingRequest,
) -> Result<ResolvedGenericRoute<'a>, OpenAiCompatHttpError> {
    let route = resolve_generic_route(
        state,
        match requested {
            Some(requested) => RoutingTarget::RequestedModel(requested.to_string()),
            None => RoutingTarget::Default,
        },
        Some(request),
    )?;
    if route.loaded_model.supported_endpoints.contains(&endpoint) {
        Ok(route)
    } else {
        Err(OpenAiCompatHttpError::BadRequest(format!(
            "model `{}` does not support `{}`; supported endpoints: {}",
            requested.unwrap_or(route.loaded_model.canonical_name.as_str()),
            endpoint.path(),
            model_endpoint_paths(route.loaded_model).join(", ")
        )))
    }
}

fn resolve_generic_model_key_for_endpoint<'a>(
    state: &'a OpenAiCompatState,
    model_key: &str,
    endpoint: RoutingEndpoint,
    request: RoutingRequest,
) -> Result<ResolvedGenericRoute<'a>, OpenAiCompatHttpError> {
    let route = resolve_generic_route(
        state,
        RoutingTarget::ModelKey(model_key.to_string()),
        Some(request.with_model_key(model_key.to_string())),
    )?;
    if route.loaded_model.supported_endpoints.contains(&endpoint) {
        Ok(route)
    } else {
        Err(OpenAiCompatHttpError::BadRequest(format!(
            "model `{}` does not support `{}`; supported endpoints: {}",
            route.loaded_model.canonical_name,
            endpoint.path(),
            model_endpoint_paths(route.loaded_model).join(", ")
        )))
    }
}

fn openai_http_error_from_routing(error: RoutingError) -> OpenAiCompatHttpError {
    match error {
        RoutingError::UnknownRequestedModel { requested } => OpenAiCompatHttpError::BadRequest(
            format!("requested model `{requested}` is not loaded"),
        ),
        RoutingError::UnknownModelKey { model_key } => OpenAiCompatHttpError::BadRequest(format!(
            "requested model key `{model_key}` is not loaded"
        )),
        RoutingError::NoEligibleRoute { reason, .. } => OpenAiCompatHttpError::BadRequest(reason),
        RoutingError::EmptyWorkerInventory
        | RoutingError::DuplicateWorkerId { .. }
        | RoutingError::UnknownDefaultModel { .. }
        | RoutingError::InconsistentInventory { .. } => {
            OpenAiCompatHttpError::Internal(error.to_string())
        }
    }
}

fn worker_for_route<'a>(
    state: &'a OpenAiCompatState,
    selection: &RouteSelection,
) -> Result<&'a OpenAiCompatWorker, OpenAiCompatHttpError> {
    state
        .workers
        .get(selection.worker_id.as_str())
        .ok_or_else(|| {
            OpenAiCompatHttpError::Internal(format!(
                "worker `{}` selected by router is missing",
                selection.worker_id
            ))
        })
}

fn bootstrap_proxy_for_route<'a>(
    state: &'a OpenAiCompatState,
    selection: &RouteSelection,
) -> Option<&'a BootstrapProxyState> {
    selection
        .worker_id
        .starts_with(BOOTSTRAP_PROXY_WORKER_PREFIX)
        .then_some(state.bootstrap_proxy.as_deref())
        .flatten()
}

fn model_endpoint_paths(model: &OpenAiCompatLoadedModel) -> Vec<&'static str> {
    model
        .supported_endpoints
        .iter()
        .map(|endpoint| endpoint.path())
        .collect()
}

fn union_supported_endpoint_paths(state: &OpenAiCompatState) -> Vec<&'static str> {
    let mut endpoints = BTreeSet::new();
    for model in state.models_by_key.values() {
        for endpoint in &model.supported_endpoints {
            endpoints.insert(endpoint.path());
        }
    }
    endpoints.into_iter().collect()
}

fn routed_inventory_for_loaded_model(
    model: &OpenAiCompatLoadedModel,
    accepted_names: Vec<String>,
    runtime_backend: &str,
) -> RoutedModelInventory {
    let mut inventory = RoutedModelInventory::new(
        model.model_key.clone(),
        model.canonical_name.clone(),
        model.family_label().to_string(),
        model.execution_profile().clone(),
    );
    for alias in accepted_names {
        inventory = inventory.with_alias(alias);
    }
    for endpoint in &model.supported_endpoints {
        inventory = inventory.with_supported_endpoint(*endpoint);
    }
    if let Some(policy) = model.scheduler_policy() {
        inventory = inventory.with_scheduler_policy(policy.clone());
    }
    inventory = inventory.with_warm_state(RoutedWarmState::Warm);
    if let Some(decoder) = model.decoder()
        && model.publishes_kv_cache_policies()
    {
        inventory = inventory.with_kv_cache_encoding_policy(
            super::default_decoder_kv_cache_encoding_policy(&decoder.descriptor, runtime_backend),
        );
        for policy in super::supported_decoder_kv_cache_encoding_policies(
            &decoder.descriptor,
            runtime_backend,
        ) {
            inventory = inventory.with_supported_kv_cache_encoding_policy(policy);
        }
    }
    if model.supports_structured_outputs() {
        inventory = inventory.with_structured_outputs();
    }
    if model.supports_tool_calling() {
        inventory = inventory.with_tool_calling();
    }
    if model.supports_response_state() {
        inventory = inventory.with_response_state();
    }
    inventory
}

fn prompt_options_for_family(
    family: GgufDecoderFamily,
    reasoning_budget: u8,
) -> PromptRenderOptions {
    if matches!(family, GgufDecoderFamily::GptOss) {
        PromptRenderOptions {
            gpt_oss_harmony: Some(GptOssHarmonyRenderContext {
                reasoning_effort: Some(reasoning_effort(reasoning_budget)),
                channel_config: Some(PromptChannelConfig::default()),
                ..Default::default()
            }),
        }
    } else {
        PromptRenderOptions::default()
    }
}

fn load_generic_decoder_model(
    model_path: &Path,
    reasoning_budget: u8,
    backend: OpenAiCompatBackend,
) -> Result<
    (
        OpenAiCompatLoadedModel,
        BTreeSet<String>,
        OpenAiCompatModelLoadPlan,
    ),
    String,
> {
    let artifact = GgufBlobArtifact::open_path(model_path, gpt_oss_local_blob_open_options())
        .map_err(|error| error.to_string())?;
    let adapter = GgufDecoderAdapterLoader
        .load_blob_artifact(&artifact)
        .map_err(|error| error.to_string())?;
    let descriptor = adapter.descriptor().clone();
    let family = adapter.family_metadata().family;
    let runtime_kind = match (backend, family) {
        (OpenAiCompatBackend::Cpu, _) => OpenAiCompatRuntimeKind::GgufDecoderCpu,
        (OpenAiCompatBackend::Cuda, GgufDecoderFamily::Gemma4) => {
            OpenAiCompatRuntimeKind::GgufDecoderCudaGemma4
        }
        (OpenAiCompatBackend::Cuda, GgufDecoderFamily::Qwen35) => {
            OpenAiCompatRuntimeKind::GgufDecoderCudaQwen35
        }
        (OpenAiCompatBackend::Cuda, _) => {
            return Err(format!(
                "generic OpenAI cuda backend currently supports only gemma4 and qwen35 GGUF decoders; `{}` resolved to `{}`",
                model_path.display(),
                decoder_family_label(family),
            ));
        }
    };
    let supported_endpoints = if matches!(
        (backend, family),
        (OpenAiCompatBackend::Cuda, GgufDecoderFamily::Gemma4)
    ) {
        vec![RoutingEndpoint::ChatCompletions]
    } else {
        vec![RoutingEndpoint::ChatCompletions, RoutingEndpoint::Responses]
    };
    let loaded_model = OpenAiCompatLoadedModel {
        model_key: descriptor.model.model_id.clone(),
        canonical_name: default_model_name(model_path, descriptor.model.model_id.as_str()),
        supported_endpoints,
        serving_truth: generic_decoder_serving_truth(family, backend),
        kind: OpenAiCompatLoadedModelKind::Decoder(OpenAiCompatLoadedDecoderModel {
            descriptor: descriptor.clone(),
            family,
            qwen35_multimodal_projection: adapter
                .family_metadata()
                .qwen35_multimodal_projection_config(),
            prompt_renderer: (!matches!(family, GgufDecoderFamily::GptOss))
                .then(|| adapter.prompt_renderer()),
            prompt_options: prompt_options_for_family(family, reasoning_budget),
            execution_profile: generic_decoder_execution_profile(family, backend),
            scheduler_policy: generic_decoder_scheduler_policy(family, backend),
        }),
    };
    Ok((
        loaded_model,
        accepted_model_names(model_path, descriptor.model.model_id.as_str()),
        OpenAiCompatModelLoadPlan {
            path: model_path.to_path_buf(),
            runtime_kind,
        },
    ))
}

fn load_generic_embeddings_model(
    model_path: &Path,
) -> Result<
    (
        OpenAiCompatLoadedModel,
        BTreeSet<String>,
        OpenAiCompatModelLoadPlan,
    ),
    String,
> {
    let service = CpuModelEmbeddingsService::from_safetensors_artifact(model_path)
        .map_err(|error| error.to_string())?;
    let descriptor = service.model_descriptor().clone();
    let loaded_model = OpenAiCompatLoadedModel {
        model_key: descriptor.model.model_id.clone(),
        canonical_name: default_model_name(model_path, descriptor.model.model_id.as_str()),
        supported_endpoints: vec![RoutingEndpoint::Embeddings],
        serving_truth: OpenAiCompatServingTruth::cpu_native(),
        kind: OpenAiCompatLoadedModelKind::Embeddings(OpenAiCompatLoadedEmbeddingsModel {
            descriptor: descriptor.clone(),
            execution_profile: default_embeddings_execution_profile(),
        }),
    };
    Ok((
        loaded_model,
        accepted_model_names(model_path, descriptor.model.model_id.as_str()),
        OpenAiCompatModelLoadPlan {
            path: model_path.to_path_buf(),
            runtime_kind: OpenAiCompatRuntimeKind::SafetensorsEmbeddings,
        },
    ))
}

fn generic_decoder_serving_truth(
    family: GgufDecoderFamily,
    backend: OpenAiCompatBackend,
) -> OpenAiCompatServingTruth {
    if matches!(
        (backend, family),
        (OpenAiCompatBackend::Cuda, GgufDecoderFamily::Gemma4)
            | (OpenAiCompatBackend::Cuda, GgufDecoderFamily::Qwen35)
    ) {
        OpenAiCompatServingTruth::cuda_native()
    } else if matches!(family, GgufDecoderFamily::Qwen35) {
        OpenAiCompatServingTruth::cpu_llama_cpp_proxy()
    } else {
        OpenAiCompatServingTruth::cpu_native()
    }
}

fn generic_decoder_execution_profile(
    family: GgufDecoderFamily,
    backend: OpenAiCompatBackend,
) -> ExecutionCapabilityProfile {
    if matches!(
        (backend, family),
        (OpenAiCompatBackend::Cuda, GgufDecoderFamily::Gemma4)
            | (OpenAiCompatBackend::Cuda, GgufDecoderFamily::Qwen35)
    ) {
        default_text_generation_execution_profile()
    } else {
        continuous_batch_text_generation_execution_profile()
    }
}

fn generic_decoder_scheduler_policy(
    family: GgufDecoderFamily,
    backend: OpenAiCompatBackend,
) -> Option<GenerationSchedulerPolicy> {
    (matches!(backend, OpenAiCompatBackend::Cpu) && !matches!(family, GgufDecoderFamily::Qwen35))
        .then(default_generation_scheduler_policy)
}

fn render_prompt_for_model(
    model: &OpenAiCompatLoadedModel,
    messages: &[PromptMessage],
) -> Result<GenericRenderedPrompt, OpenAiCompatHttpError> {
    let decoder = model.decoder().ok_or_else(|| {
        OpenAiCompatHttpError::BadRequest(format!(
            "model `{}` does not support text-generation prompts",
            model.canonical_name
        ))
    })?;
    if matches!(decoder.family, GgufDecoderFamily::GptOss) {
        let text = render_gpt_oss_harmony_prompt(messages, true, Some(&decoder.prompt_options))
            .map_err(|error| {
                OpenAiCompatHttpError::from(PromptRenderError::HarmonyRendering {
                    message: error.to_string(),
                })
            })?;
        return Ok(GenericRenderedPrompt {
            text,
            stop_sequences: vec![
                String::from(HARMONY_RETURN_STOP),
                String::from(HARMONY_CALL_STOP),
            ],
        });
    }
    let renderer = decoder.prompt_renderer.as_ref().ok_or_else(|| {
        OpenAiCompatHttpError::Internal(format!(
            "model `{}` is missing a generic prompt renderer",
            model.model_key
        ))
    })?;
    let rendered = match renderer.render_with_options(None, messages, true, &decoder.prompt_options)
    {
        Ok(rendered) => rendered,
        Err(PromptRenderError::MissingDefaultTemplate)
            if messages
                .iter()
                .all(|message| message.role != PromptMessageRole::Tool) =>
        {
            return Ok(GenericRenderedPrompt {
                text: fallback_prompt_text(messages),
                stop_sequences: Vec::new(),
            });
        }
        Err(error) => return Err(error.into()),
    };
    Ok(GenericRenderedPrompt {
        text: rendered.text,
        stop_sequences: rendered.stop_sequences,
    })
}

fn fallback_prompt_text(messages: &[PromptMessage]) -> String {
    if messages.len() == 1 && messages[0].role == PromptMessageRole::User {
        return messages[0].content.clone();
    }
    messages
        .iter()
        .map(|message| {
            let role = match message.role {
                PromptMessageRole::System => "system",
                PromptMessageRole::Developer => "developer",
                PromptMessageRole::User => "user",
                PromptMessageRole::Assistant => "assistant",
                PromptMessageRole::Tool => "tool",
            };
            format!("{role}:\n{}", message.content)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn completion_choice_for_family(
    family: GgufDecoderFamily,
    response: &crate::GenerationResponse,
    parsed_reasoning: Option<&ParsedReasoningResponse>,
    reasoning_request: Option<&ResolvedReasoningRequest>,
    tool_outcome: Option<&ToolCallOutcome>,
) -> Result<ParsedCompletionChoice, OpenAiCompatHttpError> {
    if let Some(tool_outcome) = tool_outcome {
        let tool_calls = tool_outcome
            .tool_calls
            .clone()
            .into_iter()
            .map(ResolvedToolCall::into_chat_tool_call)
            .collect::<Result<Vec<_>, OpenAiCompatHttpError>>()?;
        return Ok(ParsedCompletionChoice {
            content: tool_outcome.content.clone(),
            reasoning_content: reasoning_request.and_then(|request| match request.mode {
                PsionicReasoningMode::Separate => {
                    parsed_reasoning.and_then(|parsed| parsed.reasoning_content.clone())
                }
                PsionicReasoningMode::Suppress => None,
            }),
            finish_reason: if tool_calls.is_empty() {
                finish_reason(response.termination)
            } else {
                "tool_calls"
            },
            tool_calls,
        });
    }
    if matches!(family, GgufDecoderFamily::GptOss) {
        return Ok(completion_choice(
            response,
            parsed_reasoning,
            reasoning_request,
        ));
    }
    Ok(ParsedCompletionChoice {
        content: Some(response.output.text.clone()),
        reasoning_content: None,
        tool_calls: Vec::new(),
        finish_reason: finish_reason(response.termination),
    })
}

fn prompt_request_cache_key(messages: &[PromptMessage]) -> String {
    let mut hasher = Sha256::new();
    for message in messages {
        hasher.update(prompt_message_role_cache_key(message.role).as_bytes());
        hasher.update([0xff]);
        hasher.update(message.content.as_bytes());
        hasher.update([0xff]);
        if let Some(name) = message.author_name.as_deref() {
            hasher.update(name.as_bytes());
        }
        hasher.update([0x00]);
    }
    format!("{:x}", hasher.finalize())
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn unix_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn generation_options_from_chat_request(request: &ChatCompletionRequest) -> GenerationOptions {
    generation_options_from_chat_request_for_family(request, GgufDecoderFamily::GptOss, &[])
}

fn generation_options_from_chat_request_for_family(
    request: &ChatCompletionRequest,
    family: GgufDecoderFamily,
    default_stop_sequences: &[String],
) -> GenerationOptions {
    let max_output_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let mut options = if request_uses_sample_decode(
        request.temperature,
        request.top_k,
        request.top_p,
        request.min_p,
        request.typical_p,
        request.mirostat,
    ) {
        GenerationOptions::sample(max_output_tokens)
    } else {
        GenerationOptions::greedy(max_output_tokens)
    };
    options.decode_strategy = if request_uses_sample_decode(
        request.temperature,
        request.top_k,
        request.top_p,
        request.min_p,
        request.typical_p,
        request.mirostat,
    ) {
        DecodeStrategy::Sample
    } else {
        DecodeStrategy::Greedy
    };
    apply_sampling_controls(
        &mut options,
        request.temperature,
        request.top_k,
        request.top_p,
        request.min_p,
        request.typical_p,
        request.mirostat,
        request.mirostat_tau,
        request.mirostat_eta,
        request.repeat_penalty,
        request.repeat_last_n,
        request.presence_penalty,
        request.frequency_penalty,
        request.seed,
    );
    if let Some(stop) = &request.stop {
        options.stop_sequences.extend(stop.clone().into_vec());
    }
    for stop in default_stop_sequences {
        if !options.stop_sequences.iter().any(|value| value == stop) {
            options.stop_sequences.push(stop.clone());
        }
    }
    if matches!(family, GgufDecoderFamily::GptOss) {
        ensure_harmony_stop_sequences(&mut options.stop_sequences);
    }
    options
}

fn generation_options_from_responses_request(
    request: &ResponsesRequest,
    family: GgufDecoderFamily,
    default_stop_sequences: &[String],
) -> GenerationOptions {
    let max_output_tokens = request.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    let mut options = if request_uses_sample_decode(
        request.temperature,
        request.top_k,
        request.top_p,
        request.min_p,
        request.typical_p,
        request.mirostat,
    ) {
        GenerationOptions::sample(max_output_tokens)
    } else {
        GenerationOptions::greedy(max_output_tokens)
    };
    options.decode_strategy = if request_uses_sample_decode(
        request.temperature,
        request.top_k,
        request.top_p,
        request.min_p,
        request.typical_p,
        request.mirostat,
    ) {
        DecodeStrategy::Sample
    } else {
        DecodeStrategy::Greedy
    };
    apply_sampling_controls(
        &mut options,
        request.temperature,
        request.top_k,
        request.top_p,
        request.min_p,
        request.typical_p,
        request.mirostat,
        request.mirostat_tau,
        request.mirostat_eta,
        request.repeat_penalty,
        request.repeat_last_n,
        request.presence_penalty,
        request.frequency_penalty,
        request.seed,
    );
    if let Some(stop) = &request.stop {
        options.stop_sequences.extend(stop.clone().into_vec());
    }
    for stop in default_stop_sequences {
        if !options.stop_sequences.iter().any(|value| value == stop) {
            options.stop_sequences.push(stop.clone());
        }
    }
    if matches!(family, GgufDecoderFamily::GptOss) {
        ensure_harmony_stop_sequences(&mut options.stop_sequences);
    }
    options
}

fn request_uses_sample_decode(
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    typical_p: Option<f32>,
    mirostat: Option<u8>,
) -> bool {
    temperature.is_some_and(|value| value > f32::EPSILON)
        || top_k.is_some_and(|value| value > 1)
        || top_p.is_some_and(|value| value.is_finite() && value > 0.0 && value < 1.0)
        || min_p.is_some_and(|value| value.is_finite() && value > 0.0 && value <= 1.0)
        || typical_p.is_some_and(|value| value.is_finite() && value > 0.0 && value < 1.0)
        || mirostat.is_some_and(|value| matches!(value, 1 | 2))
}

fn apply_sampling_controls(
    options: &mut GenerationOptions,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    typical_p: Option<f32>,
    mirostat: Option<u8>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
) {
    options.temperature = temperature;
    options.top_k = top_k;
    options.top_p = top_p;
    options.min_p = min_p;
    options.typical_p = typical_p;
    options.mirostat = mirostat;
    options.mirostat_tau = mirostat_tau;
    options.mirostat_eta = mirostat_eta;
    options.repeat_penalty = repeat_penalty;
    options.repeat_last_n = repeat_last_n;
    options.presence_penalty = presence_penalty;
    options.frequency_penalty = frequency_penalty;
    options.seed = seed;
}

fn response_input_to_prompt_messages_with_options(
    request: &ResponsesRequest,
    model: &OpenAiCompatLoadedDecoderModel,
    include_instructions: bool,
    allow_empty_input: bool,
) -> Result<Vec<PromptMessage>, OpenAiCompatHttpError> {
    let mut messages = Vec::new();
    if include_instructions && let Some(instructions) = request.instructions.as_ref() {
        messages.push(ChatCompletionMessage::text(
            "developer",
            instructions.clone(),
        ));
    }
    match &request.input {
        ResponsesInput::Text(text) => {
            if allow_empty_input && text.is_empty() {
            } else {
                messages.push(ChatCompletionMessage::text("user", text.clone()));
            }
        }
        ResponsesInput::Messages(input_messages) => {
            if allow_empty_input && input_messages.is_empty() {
            } else {
                messages.extend(input_messages.clone());
            }
        }
    }
    chat_messages_to_prompt_messages_for_decoder(messages.as_slice(), model)
}

fn assistant_history_from_response(
    family: GgufDecoderFamily,
    raw_output: &str,
    parsed_harmony: Option<&GptOssHarmonyParsedOutput>,
) -> Vec<PromptMessage> {
    if matches!(family, GgufDecoderFamily::GptOss)
        && let Some(parsed_harmony) = parsed_harmony
        && !parsed_harmony.messages.is_empty()
    {
        return parsed_harmony.messages.clone();
    }
    vec![PromptMessage::new(PromptMessageRole::Assistant, raw_output)]
}

fn leading_response_instructions(prompt_history: &[PromptMessage]) -> Option<&str> {
    prompt_history
        .first()
        .filter(|message| {
            message.role == PromptMessageRole::Developer
                && message.author_name.is_none()
                && message.recipient.is_none()
                && message.channel.is_none()
                && message.content_type.is_none()
        })
        .map(|message| message.content.as_str())
}

fn ensure_harmony_stop_sequences(stop_sequences: &mut Vec<String>) {
    for stop in [HARMONY_RETURN_STOP, HARMONY_CALL_STOP] {
        if !stop_sequences.iter().any(|value| value == stop) {
            stop_sequences.push(String::from(stop));
        }
    }
}

fn chat_messages_to_prompt_messages(
    messages: &[ChatCompletionMessage],
) -> Result<Vec<PromptMessage>, OpenAiCompatHttpError> {
    chat_messages_to_prompt_messages_for_family(messages, GgufDecoderFamily::GptOss)
}

fn chat_messages_to_prompt_messages_for_decoder(
    messages: &[ChatCompletionMessage],
    model: &OpenAiCompatLoadedDecoderModel,
) -> Result<Vec<PromptMessage>, OpenAiCompatHttpError> {
    if matches!(model.family, GgufDecoderFamily::GptOss) {
        return chat_messages_to_prompt_messages_gpt_oss(messages);
    }
    chat_messages_to_prompt_messages_generic(
        messages,
        model.family,
        model.qwen35_multimodal_projection.as_ref(),
    )
}

fn chat_messages_to_prompt_messages_for_family(
    messages: &[ChatCompletionMessage],
    family: GgufDecoderFamily,
) -> Result<Vec<PromptMessage>, OpenAiCompatHttpError> {
    if matches!(family, GgufDecoderFamily::GptOss) {
        return chat_messages_to_prompt_messages_gpt_oss(messages);
    }
    chat_messages_to_prompt_messages_generic(messages, family, None)
}

fn chat_messages_to_prompt_messages_gpt_oss(
    messages: &[ChatCompletionMessage],
) -> Result<Vec<PromptMessage>, OpenAiCompatHttpError> {
    if messages.is_empty() {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "chat completions require at least one message",
        )));
    }
    let mut prompt_messages = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let role = match message.role.as_str() {
            "system" => PromptMessageRole::System,
            "developer" => PromptMessageRole::Developer,
            "user" => PromptMessageRole::User,
            "assistant" => PromptMessageRole::Assistant,
            "tool" => PromptMessageRole::Tool,
            other => {
                return Err(OpenAiCompatHttpError::BadRequest(format!(
                    "unsupported chat message role `{other}`"
                )));
            }
        };
        // Mirror the GPT-OSS llama.cpp OpenAI template so native and proxy
        // backends tokenize the same public request contract.
        let normalized_role = match (index, role) {
            (0, PromptMessageRole::System | PromptMessageRole::Developer) => {
                PromptMessageRole::Developer
            }
            (_, PromptMessageRole::System | PromptMessageRole::Developer) => continue,
            _ => role,
        };
        let mut prompt = PromptMessage::new(
            normalized_role,
            chat_message_content_to_text(
                &message.content,
                GgufDecoderFamily::GptOss,
                normalized_role,
                None,
            )?,
        );
        if normalized_role == PromptMessageRole::Tool {
            let Some(name) = message.name.as_ref() else {
                return Err(OpenAiCompatHttpError::BadRequest(String::from(
                    "tool messages require a `name` field",
                )));
            };
            prompt = prompt.with_author_name(name.clone());
        }
        prompt_messages.push(prompt);
    }
    Ok(prompt_messages)
}

fn chat_messages_to_prompt_messages_generic(
    messages: &[ChatCompletionMessage],
    family: GgufDecoderFamily,
    qwen35_multimodal_projection: Option<&Qwen35MultimodalProjectionConfig>,
) -> Result<Vec<PromptMessage>, OpenAiCompatHttpError> {
    if messages.is_empty() {
        return Err(OpenAiCompatHttpError::BadRequest(String::from(
            "chat completions require at least one message",
        )));
    }
    let mut prompt_messages = Vec::new();
    let mut tool_names_by_id = std::collections::HashMap::new();
    for message in messages {
        let role = match message.role.as_str() {
            "system" => PromptMessageRole::System,
            "developer" => PromptMessageRole::Developer,
            "user" => PromptMessageRole::User,
            "assistant" => PromptMessageRole::Assistant,
            "tool" => PromptMessageRole::Tool,
            other => {
                return Err(OpenAiCompatHttpError::BadRequest(format!(
                    "unsupported chat message role `{other}`"
                )));
            }
        };
        if role == PromptMessageRole::Assistant
            && let Some(tool_calls) = message.tool_calls.as_ref()
            && !tool_calls.is_empty()
        {
            for tool_call in tool_calls {
                tool_names_by_id.insert(tool_call.id.clone(), tool_call.function.name.clone());
            }
            prompt_messages.push(PromptMessage::new(
                PromptMessageRole::Assistant,
                assistant_tool_call_envelope_json(tool_calls)?,
            ));
            continue;
        }
        let mut prompt = PromptMessage::new(
            role,
            chat_message_content_to_text(
                &message.content,
                family,
                role,
                qwen35_multimodal_projection,
            )?,
        );
        if role == PromptMessageRole::Tool {
            let tool_name = message.name.clone().or_else(|| {
                message
                    .tool_call_id
                    .as_ref()
                    .and_then(|tool_call_id| tool_names_by_id.get(tool_call_id).cloned())
            });
            let Some(name) = tool_name else {
                return Err(OpenAiCompatHttpError::BadRequest(String::from(
                    "tool messages require a `name` field or a `tool_call_id` that matches an earlier assistant tool call",
                )));
            };
            prompt = prompt.with_author_name(name);
        }
        prompt_messages.push(prompt);
    }
    Ok(prompt_messages)
}

fn assistant_tool_call_envelope_json(
    tool_calls: &[ChatCompletionToolCall],
) -> Result<String, OpenAiCompatHttpError> {
    let tool_calls = tool_calls
        .iter()
        .map(|tool_call| {
            let arguments = serde_json::from_str::<serde_json::Value>(
                &tool_call.function.arguments,
            )
            .map_err(|error| {
                OpenAiCompatHttpError::BadRequest(format!(
                    "assistant tool call `{}` arguments are not valid JSON: {error}",
                    tool_call.function.name
                ))
            })?;
            Ok(serde_json::json!({
                "name": tool_call.function.name,
                "arguments": arguments,
            }))
        })
        .collect::<Result<Vec<_>, OpenAiCompatHttpError>>()?;
    serde_json::to_string(&serde_json::json!({
        "kind": "tool_calls",
        "tool_calls": tool_calls,
    }))
    .map_err(|error| {
        OpenAiCompatHttpError::Internal(format!(
            "failed to serialize assistant tool-call envelope: {error}"
        ))
    })
}

fn chat_message_content_to_text(
    content: &ChatCompletionMessageContent,
    family: GgufDecoderFamily,
    role: PromptMessageRole,
    qwen35_multimodal_projection: Option<&Qwen35MultimodalProjectionConfig>,
) -> Result<String, OpenAiCompatHttpError> {
    match content {
        ChatCompletionMessageContent::Text(text) => Ok(text.clone()),
        ChatCompletionMessageContent::Parts(parts) => {
            if matches!(family, GgufDecoderFamily::Qwen35)
                && let Some(config) = qwen35_multimodal_projection
            {
                return project_qwen35_multimodal_content(parts.as_slice(), role, config);
            }
            let mut text = String::new();
            for part in parts {
                match part {
                    ChatCompletionContentPart::Text { text: part_text } => {
                        text.push_str(part_text);
                    }
                    ChatCompletionContentPart::ImageUrl { .. }
                    | ChatCompletionContentPart::VideoUrl { .. } => {
                        return Err(unsupported_multimodal_content_error(family));
                    }
                }
            }
            Ok(text)
        }
    }
}

fn project_qwen35_multimodal_content(
    parts: &[ChatCompletionContentPart],
    role: PromptMessageRole,
    config: &Qwen35MultimodalProjectionConfig,
) -> Result<String, OpenAiCompatHttpError> {
    let mut text = String::new();
    for part in parts {
        match part {
            ChatCompletionContentPart::Text { text: part_text } => {
                text.push_str(part_text);
            }
            ChatCompletionContentPart::ImageUrl { .. }
            | ChatCompletionContentPart::VideoUrl { .. }
                if matches!(role, PromptMessageRole::System) =>
            {
                return Err(OpenAiCompatHttpError::BadRequest(String::from(
                    "qwen35 system messages cannot contain image or video parts",
                )));
            }
            ChatCompletionContentPart::ImageUrl { .. } => {
                text.push_str(config.image_marker());
            }
            ChatCompletionContentPart::VideoUrl { .. } => {
                text.push_str(config.video_marker());
            }
        }
    }
    Ok(text)
}

fn unsupported_multimodal_content_error(family: GgufDecoderFamily) -> OpenAiCompatHttpError {
    if matches!(family, GgufDecoderFamily::Qwen35) {
        OpenAiCompatHttpError::BadRequest(String::from(
            "multimodal inputs are unavailable because the loaded qwen35 artifact lacks multimodal projection facts",
        ))
    } else {
        OpenAiCompatHttpError::BadRequest(format!(
            "multimodal inputs are unavailable on the current `{}` generic prompt-render path",
            decoder_family_label(family)
        ))
    }
}

fn prompt_message_role_cache_key(role: PromptMessageRole) -> &'static str {
    match role {
        PromptMessageRole::System => "system",
        PromptMessageRole::Developer => "developer",
        PromptMessageRole::User => "user",
        PromptMessageRole::Assistant => "assistant",
        PromptMessageRole::Tool => "tool",
    }
}

fn validate_requested_model(
    requested: Option<&str>,
    accepted_model_names: &BTreeSet<String>,
) -> Result<(), OpenAiCompatHttpError> {
    let Some(requested) = requested else {
        return Ok(());
    };
    if accepted_model_names.contains(requested) {
        return Ok(());
    }
    Err(OpenAiCompatHttpError::BadRequest(format!(
        "requested model `{requested}` is not loaded"
    )))
}

fn default_model_name(path: &Path, model_id: &str) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(String::from)
        .unwrap_or_else(|| model_id.to_string())
}

fn accepted_model_names(path: &Path, model_id: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert(model_id.to_string());
    if let Some(file_name) = path.file_name().and_then(|value| value.to_str()) {
        names.insert(file_name.to_string());
    }
    if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
        names.insert(stem.to_string());
    }
    names
}

fn reasoning_effort(reasoning_budget: u8) -> PromptReasoningEffort {
    match reasoning_budget {
        0 => PromptReasoningEffort::Low,
        1 => PromptReasoningEffort::Medium,
        _ => PromptReasoningEffort::High,
    }
}

#[derive(Clone, Debug, Serialize)]
struct OpenAiErrorEnvelope {
    error: OpenAiErrorBody,
}

#[derive(Clone, Debug, Serialize)]
struct OpenAiErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: String,
}

#[cfg(test)]
mod tests {
    use super::{
        BOOTSTRAP_PROXY_RESIDENCY_MODE, CPU_SERVER_FALLBACK_POLICY, CPU_SERVER_HYBRID_OFFLOAD_MODE,
        CPU_SERVER_RESIDENCY_MODE, ChatCompletionContentPart, ChatCompletionJsonSchemaRequest,
        ChatCompletionMessage, ChatCompletionMessageContent, ChatCompletionRequest,
        ChatCompletionResponseFormatRequest, ChatCompletionToolCall,
        ChatCompletionToolCallFunction, EmbeddingsInput, EmbeddingsRequest,
        GptOssMetalExecutionMode, GptOssOpenAiCompatBackend, GptOssOpenAiCompatConfig,
        HARMONY_CALL_STOP, HARMONY_RETURN_STOP, LOCAL_SERVER_LOAD_STATUS,
        LOCAL_SERVER_MEMORY_PRESSURE_REPORTING, LOCAL_SERVER_UNLOAD_CONTROL,
        LOCAL_SERVER_WARM_CONTROL, LocalServingTruth, NamedToolChoiceFunction,
        NamedToolChoiceRequest, OPENAI_COMPAT_WORKER_ID, OpenAiCompatConfig,
        OpenAiCompatRuntimeKind, OpenAiCompatServer, PromptTokenCache, PsionicGrammarRequest,
        PsionicReasoningMode, PsionicReasoningRequest, PsionicResponseStateRequest,
        ResolvedReasoningRequest, ResolvedToolCall, ResponseContinuationMode, ResponsesInput,
        ResponsesRequest, RoutingEndpoint, RoutingRequest, StopSequences,
        THIN_CLIENT_FALLBACK_POSTURE, ToolCallingContract, ToolChoiceMode, ToolChoiceRequest,
        ToolDefinitionEnvelope, ToolDefinitionRequest, WARMING_FALLBACK_POSTURE,
        apply_tool_contract_to_prompt_messages, assistant_prompt_message_for_tool_loop,
        chat_messages_to_prompt_messages, chat_messages_to_prompt_messages_for_family,
        chat_messages_to_prompt_messages_generic, completion_choice, ensure_harmony_stop_sequences,
        generation_options_from_chat_request, generation_options_from_chat_request_for_family,
        generation_options_from_responses_request, generic_embeddings, generic_health,
        generic_list_models, generic_management_status, gpt_oss_local_serving_truth,
        handle_generic_chat_completions, handle_generic_embeddings, handle_generic_responses,
        insert_local_serving_truth_headers, load_generic_decoder_model, model_endpoint_paths,
        prompt_request_cache_key, render_prompt_for_model,
        required_tool_call_floor_from_chat_messages, resolve_execution_summary,
        resolve_generic_model, resolve_generic_model_for_endpoint,
        response_input_to_prompt_messages_with_options, responses_output_items,
        structured_output_from_tool_contract, surfaced_reasoning_response,
        tool_call_outcome_from_response, tool_contract_from_chat_request,
        tool_loop_tool_call_from_resolved, tool_prompt_message, tool_result_prompt_message,
    };
    use crate::conformance::{
        ConformanceSuite, GenerateConformanceCase, GenerateObservation, RecordedConformanceSubject,
        SubjectObservation, run_conformance_suite,
    };
    use crate::{
        DecodeStrategy, GenerationMetrics, GenerationOptions, GenerationOutput, GenerationRequest,
        GenerationResponse, GenerationUsage, OpenAiCompatBackend, TerminationReason,
    };
    use axum::{
        Json,
        body::{Body, to_bytes},
        extract::State,
        http::{HeaderMap, Request, StatusCode},
        response::{IntoResponse, Response},
    };
    use psionic_models::{
        ByteProjectionEmbedder, GgufContent, GgufDecoderFamily, GgufMetadataValue,
        GgufPromptTemplateRenderer, GgufTensorType, GptOssHarmonyParseOptions,
        GptOssHarmonyRenderContext, PromptChannelConfig, PromptMessage, PromptMessageRole,
        PromptReasoningEffort, PromptRenderOptions, Qwen35MultimodalProjectionConfig,
        ReasoningParser, TokenId, TokenSequence, golden_prompt_fixture, parse_gpt_oss_harmony_text,
        render_gpt_oss_harmony_prompt,
    };
    use psionic_net::{ServedMeshRole, ServedMeshRolePosture, ServedMeshRoleReason};
    use psionic_router::{
        ResponseStateRecord, ResponseStateRetentionPolicy, ResponseStateStore,
        ToolExecutionRequest, ToolGateway, ToolHistoryVisibility, ToolLoopController,
        ToolLoopError, ToolLoopModelRunner, ToolLoopModelTurn, ToolLoopRequest,
        ToolLoopToolExecutor, ToolLoopToolResult, ToolProviderDescriptor, ToolResultVisibility,
    };
    use psionic_runtime::{
        BatchExecutionPosture, PrefixCacheControl, PrefixCacheMode, QueueDiscipline,
        StructuredGrammarSyntax, StructuredOutputRequest, StructuredOutputValue,
        StructuredTaggedVariant,
    };
    use std::{
        collections::BTreeMap,
        path::Path,
        sync::{Mutex, OnceLock},
    };
    use tower::util::ServiceExt;

    #[test]
    fn chat_messages_map_to_prompt_messages() {
        let prompt = chat_messages_to_prompt_messages(&[
            ChatCompletionMessage::text("system", "sys"),
            ChatCompletionMessage::named_text("tool", "{\"ok\":true}", "functions.lookup_weather"),
        ])
        .expect("prompt messages");

        assert_eq!(prompt[0].role, PromptMessageRole::Developer);
        assert_eq!(prompt[1].role, PromptMessageRole::Tool);
        assert_eq!(
            prompt[1].author_name.as_deref(),
            Some("functions.lookup_weather")
        );
    }

    #[test]
    fn chat_messages_ignore_non_initial_instruction_turns_for_gpt_oss_parity() {
        let prompt = chat_messages_to_prompt_messages(&[
            ChatCompletionMessage::text("system", "first instruction"),
            ChatCompletionMessage::text("developer", "ignored instruction"),
            ChatCompletionMessage::text("user", "hello"),
        ])
        .expect("prompt messages");

        assert_eq!(prompt.len(), 2);
        assert_eq!(prompt[0].role, PromptMessageRole::Developer);
        assert_eq!(prompt[0].content, "first instruction");
        assert_eq!(prompt[1].role, PromptMessageRole::User);
        assert_eq!(prompt[1].content, "hello");
    }

    #[test]
    fn generic_chat_qwen35_tool_result_replay_infers_name_from_prior_tool_call() {
        let prompt = chat_messages_to_prompt_messages_generic(
            &[
                ChatCompletionMessage::text("user", "use the tool"),
                ChatCompletionMessage {
                    role: String::from("assistant"),
                    content: ChatCompletionMessageContent::Text(String::new()),
                    name: None,
                    tool_calls: Some(vec![ChatCompletionToolCall {
                        id: String::from("call-1"),
                        kind: String::from("function"),
                        function: ChatCompletionToolCallFunction {
                            name: String::from("get_weather"),
                            arguments: String::from("{\"city\":\"Paris\"}"),
                        },
                    }]),
                    tool_call_id: None,
                },
                ChatCompletionMessage {
                    role: String::from("tool"),
                    content: ChatCompletionMessageContent::Text(String::from(
                        "{\"condition\":\"sunny\"}",
                    )),
                    name: None,
                    tool_calls: None,
                    tool_call_id: Some(String::from("call-1")),
                },
            ],
            GgufDecoderFamily::Qwen35,
            None,
        )
        .expect("prompt messages");

        assert_eq!(prompt.len(), 3);
        assert_eq!(prompt[0].role, PromptMessageRole::User);
        assert_eq!(prompt[1].role, PromptMessageRole::Assistant);
        assert_eq!(
            prompt[1].content,
            "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]}"
        );
        assert_eq!(prompt[2].role, PromptMessageRole::Tool);
        assert_eq!(prompt[2].author_name.as_deref(), Some("get_weather"));
        assert_eq!(prompt[2].content, "{\"condition\":\"sunny\"}");
    }

    #[test]
    fn rendered_prompt_matches_llama_cpp_gpt_oss_openai_contract() {
        let prompt_messages = chat_messages_to_prompt_messages(&[
            ChatCompletionMessage::text(
                "system",
                "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2026-03-09\n\nReasoning: low\n\n# Valid channels: analysis, final. Channel must be included for every message.",
            ),
            ChatCompletionMessage::text(
                "developer",
                "Be concise. Output exactly one sentence.",
            ),
            ChatCompletionMessage::text(
                "user",
                "Reply with exactly this sentence and nothing else: HTTPS protects users by encrypting traffic, preventing tampering, and confirming they are connected to the right website.",
            ),
        ])
        .expect("prompt messages");
        let prompt_options = PromptRenderOptions {
            gpt_oss_harmony: Some(GptOssHarmonyRenderContext {
                reasoning_effort: Some(PromptReasoningEffort::Low),
                conversation_start_date: Some(String::from("2026-03-09")),
                knowledge_cutoff: Some(String::from("2024-06")),
                channel_config: Some(PromptChannelConfig::default()),
                ..Default::default()
            }),
        };

        let rendered =
            render_gpt_oss_harmony_prompt(prompt_messages.as_slice(), true, Some(&prompt_options))
                .expect("rendered prompt");

        assert_eq!(
            rendered,
            concat!(
                "<|start|>system<|message|>",
                "You are ChatGPT, a large language model trained by OpenAI.\n",
                "Knowledge cutoff: 2024-06\n",
                "Current date: 2026-03-09\n\n",
                "Reasoning: low\n\n",
                "# Valid channels: analysis, commentary, final. Channel must be included for every message.",
                "<|end|>",
                "<|start|>developer<|message|>",
                "# Instructions\n\n",
                "You are ChatGPT, a large language model trained by OpenAI.\n",
                "Knowledge cutoff: 2024-06\n",
                "Current date: 2026-03-09\n\n",
                "Reasoning: low\n\n",
                "# Valid channels: analysis, final. Channel must be included for every message.",
                "<|end|>",
                "<|start|>user<|message|>",
                "Reply with exactly this sentence and nothing else: HTTPS protects users by encrypting traffic, preventing tampering, and confirming they are connected to the right website.",
                "<|end|>",
                "<|start|>assistant",
            )
        );
    }

    #[test]
    fn generation_options_force_harmony_stop_sequences() {
        let options = generation_options_from_chat_request(&ChatCompletionRequest {
            model: None,
            messages: vec![ChatCompletionMessage::text("user", "hi")],
            temperature: Some(0.0),
            max_tokens: Some(64),
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        });

        assert!(
            options
                .stop_sequences
                .iter()
                .any(|value| value == HARMONY_RETURN_STOP)
        );
        assert!(
            options
                .stop_sequences
                .iter()
                .any(|value| value == HARMONY_CALL_STOP)
        );
    }

    #[test]
    fn generation_options_from_chat_request_forward_sampling_controls() {
        let options = generation_options_from_chat_request_for_family(
            &ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35")),
                messages: vec![ChatCompletionMessage::text("user", "sample")],
                temperature: Some(0.7),
                top_k: Some(23),
                top_p: Some(0.85),
                min_p: Some(0.1),
                typical_p: Some(0.72),
                mirostat: Some(1),
                mirostat_tau: Some(5.5),
                mirostat_eta: Some(0.15),
                repeat_penalty: Some(1.15),
                repeat_last_n: Some(32),
                presence_penalty: Some(0.25),
                frequency_penalty: Some(0.5),
                seed: Some(42),
                max_tokens: Some(17),
                stop: Some(StopSequences::Many(vec![String::from("done")])),
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
            GgufDecoderFamily::Qwen35,
            &[],
        );

        assert_eq!(options.decode_strategy, DecodeStrategy::Sample);
        assert_eq!(options.max_output_tokens, 17);
        assert_eq!(options.temperature, Some(0.7));
        assert_eq!(options.top_k, Some(23));
        assert_eq!(options.top_p, Some(0.85));
        assert_eq!(options.min_p, Some(0.1));
        assert_eq!(options.typical_p, Some(0.72));
        assert_eq!(options.mirostat, Some(1));
        assert_eq!(options.mirostat_tau, Some(5.5));
        assert_eq!(options.mirostat_eta, Some(0.15));
        assert_eq!(options.repeat_penalty, Some(1.15));
        assert_eq!(options.repeat_last_n, Some(32));
        assert_eq!(options.presence_penalty, Some(0.25));
        assert_eq!(options.frequency_penalty, Some(0.5));
        assert_eq!(options.seed, Some(42));
        assert_eq!(options.stop_sequences, vec![String::from("done")]);
    }

    #[test]
    fn generation_options_from_responses_request_forward_sampling_controls() {
        let options = generation_options_from_responses_request(
            &ResponsesRequest {
                model: Some(String::from("tiny-qwen35")),
                instructions: None,
                conversation: None,
                input: ResponsesInput::Text(String::from("sample")),
                temperature: Some(0.65),
                top_k: Some(19),
                top_p: Some(0.92),
                min_p: Some(0.05),
                typical_p: Some(0.61),
                mirostat: Some(2),
                mirostat_tau: Some(6.0),
                mirostat_eta: Some(0.12),
                repeat_penalty: Some(1.2),
                repeat_last_n: Some(-1),
                presence_penalty: Some(0.2),
                frequency_penalty: Some(0.4),
                seed: Some(7),
                max_output_tokens: Some(29),
                stop: Some(StopSequences::One(String::from("END"))),
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
            GgufDecoderFamily::Qwen35,
            &[],
        );

        assert_eq!(options.decode_strategy, DecodeStrategy::Sample);
        assert_eq!(options.max_output_tokens, 29);
        assert_eq!(options.temperature, Some(0.65));
        assert_eq!(options.top_k, Some(19));
        assert_eq!(options.top_p, Some(0.92));
        assert_eq!(options.min_p, Some(0.05));
        assert_eq!(options.typical_p, Some(0.61));
        assert_eq!(options.mirostat, Some(2));
        assert_eq!(options.mirostat_tau, Some(6.0));
        assert_eq!(options.mirostat_eta, Some(0.12));
        assert_eq!(options.repeat_penalty, Some(1.2));
        assert_eq!(options.repeat_last_n, Some(-1));
        assert_eq!(options.presence_penalty, Some(0.2));
        assert_eq!(options.frequency_penalty, Some(0.4));
        assert_eq!(options.seed, Some(7));
        assert_eq!(options.stop_sequences, vec![String::from("END")]);
    }

    #[test]
    fn auto_metal_mode_resolves_to_native_without_legacy_proxy() {
        let summary = resolve_execution_summary(
            GptOssOpenAiCompatBackend::Metal,
            GptOssMetalExecutionMode::Auto,
            false,
        )
        .expect("metal summary");

        assert_eq!(summary.backend_label, "metal");
        assert_eq!(summary.execution_mode_label, "native");
        assert_eq!(summary.execution_engine_label, "psionic");
    }

    #[test]
    fn explicit_native_metal_mode_rejects_legacy_proxy_env() {
        let error = resolve_execution_summary(
            GptOssOpenAiCompatBackend::Metal,
            GptOssMetalExecutionMode::Native,
            true,
        )
        .expect_err("native metal should reject legacy proxy env");

        assert!(error.to_string().contains("PSIONIC_METAL_PROXY_LLAMA_CPP"));
    }

    #[test]
    fn explicit_metal_mode_is_rejected_when_backend_is_not_metal() {
        let error = resolve_execution_summary(
            GptOssOpenAiCompatBackend::Cuda,
            GptOssMetalExecutionMode::ProxyLlamaCpp,
            false,
        )
        .expect_err("non-metal backend should reject explicit metal mode");

        assert!(error.to_string().contains("resolved backend is cuda"));
    }

    #[test]
    fn gpt_oss_local_backend_truth_is_explicit_across_native_and_proxy_modes() {
        let mut config = GptOssOpenAiCompatConfig::new("/tmp/tiny-gpt-oss.gguf");
        config.gpu_layers = Some(12);

        let metal_native = gpt_oss_local_serving_truth(
            &config,
            resolve_execution_summary(
                GptOssOpenAiCompatBackend::Metal,
                GptOssMetalExecutionMode::Native,
                false,
            )
            .expect("metal native summary"),
        );
        assert_eq!(metal_native.residency_mode, "metal_accelerated");
        assert_eq!(metal_native.hybrid_offload, "unsupported");
        assert_eq!(metal_native.performance_class, "apple_silicon_native");

        let metal_proxy = gpt_oss_local_serving_truth(
            &config,
            resolve_execution_summary(
                GptOssOpenAiCompatBackend::Metal,
                GptOssMetalExecutionMode::ProxyLlamaCpp,
                false,
            )
            .expect("metal proxy summary"),
        );
        assert_eq!(metal_proxy.residency_mode, "llama_cpp_proxy");
        assert_eq!(metal_proxy.hybrid_offload, "llama_cpp_gpu_layers");
        assert_eq!(metal_proxy.hybrid_offload_layers, Some(12));
        assert_eq!(metal_proxy.fallback_policy, "proxy_only");

        let cuda_native = gpt_oss_local_serving_truth(
            &config,
            resolve_execution_summary(
                GptOssOpenAiCompatBackend::Cuda,
                GptOssMetalExecutionMode::Auto,
                false,
            )
            .expect("cuda summary"),
        );
        assert_eq!(cuda_native.residency_mode, "cuda_accelerated");
        assert_eq!(cuda_native.hybrid_offload, "host_backed_selected4");
        assert_eq!(cuda_native.performance_class, "nvidia_native");
    }

    #[test]
    fn local_serving_truth_headers_include_optional_hybrid_layers() {
        let mut headers = HeaderMap::new();
        insert_local_serving_truth_headers(
            &mut headers,
            LocalServingTruth {
                residency_mode: "llama_cpp_proxy",
                hybrid_offload: "llama_cpp_gpu_layers",
                hybrid_offload_layers: Some(7),
                fallback_policy: "proxy_only",
                performance_class: "proxy_control_plane",
                load_status: LOCAL_SERVER_LOAD_STATUS,
                warm_control: LOCAL_SERVER_WARM_CONTROL,
                unload_control: LOCAL_SERVER_UNLOAD_CONTROL,
                memory_pressure_reporting: LOCAL_SERVER_MEMORY_PRESSURE_REPORTING,
            },
        );
        assert_eq!(
            headers
                .get("x-psionic-hybrid-offload-layers")
                .and_then(|value| value.to_str().ok()),
            Some("7")
        );
        assert_eq!(
            headers
                .get("x-psionic-residency-mode")
                .and_then(|value| value.to_str().ok()),
            Some("llama_cpp_proxy")
        );
    }

    #[test]
    fn gpt_oss_completion_choice_can_surface_reasoning_contracts()
    -> Result<(), Box<dyn std::error::Error>> {
        let raw = "<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>323";
        let parsed = parse_gpt_oss_harmony_text(
            raw,
            GptOssHarmonyParseOptions {
                role_hint: Some(PromptMessageRole::Assistant),
                strict: false,
            },
        )?
        .reasoning_response();
        let response = test_generation_response(raw);
        let reasoning_request = ResolvedReasoningRequest {
            parser: ReasoningParser::GptOssHarmony,
            mode: PsionicReasoningMode::Separate,
        };

        let choice = completion_choice(&response, Some(&parsed), Some(&reasoning_request));
        let serialized_choice = serde_json::to_value(choice.clone().into_full_choice())?;

        assert_eq!(choice.content.as_deref(), Some("323"));
        assert_eq!(choice.reasoning_content.as_deref(), Some("thinking"));
        assert_eq!(
            serialized_choice["message"]["reasoning_content"],
            serde_json::json!("thinking")
        );
        let surfaced = surfaced_reasoning_response(Some(&parsed), Some(&reasoning_request), false)
            .expect("typed reasoning should surface");
        assert_eq!(surfaced.final_content.as_deref(), Some("323"));
        assert_eq!(surfaced.reasoning_content.as_deref(), Some("thinking"));
        Ok(())
    }

    #[test]
    fn responses_output_items_keep_reasoning_and_final_text_in_order() {
        let items = responses_output_items(
            "resp-1",
            &super::ParsedCompletionChoice {
                content: Some(String::from("323")),
                reasoning_content: Some(String::from("thinking")),
                tool_calls: Vec::new(),
                finish_reason: "stop",
            },
        );

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].content.len(), 2);
        assert_eq!(items[0].content[0].kind, "reasoning_text");
        assert_eq!(items[0].content[0].text, "thinking");
        assert_eq!(items[0].content[1].kind, "output_text");
        assert_eq!(items[0].content[1].text, "323");
    }

    #[test]
    fn ensure_harmony_stop_sequences_is_idempotent() {
        let mut stops = vec![String::from(HARMONY_RETURN_STOP)];
        ensure_harmony_stop_sequences(&mut stops);
        ensure_harmony_stop_sequences(&mut stops);

        assert_eq!(
            stops
                .iter()
                .filter(|value| value.as_str() == HARMONY_RETURN_STOP)
                .count(),
            1
        );
        assert_eq!(
            stops
                .iter()
                .filter(|value| value.as_str() == HARMONY_CALL_STOP)
                .count(),
            1
        );
    }

    #[test]
    fn prompt_token_cache_is_lru() {
        let mut cache = PromptTokenCache::new(2);
        cache.record(
            String::from("key-one"),
            TokenSequence::new(vec![TokenId(1), TokenId(2)]),
        );
        cache.record(
            String::from("key-two"),
            TokenSequence::new(vec![TokenId(3)]),
        );

        assert_eq!(
            cache.lookup("key-one").expect("cached prompt").as_slice(),
            &[TokenId(1), TokenId(2)]
        );

        cache.record(
            String::from("key-three"),
            TokenSequence::new(vec![TokenId(4)]),
        );

        assert!(cache.lookup("key-two").is_none());
        assert_eq!(
            cache.lookup("key-three").expect("cached prompt").as_slice(),
            &[TokenId(4)]
        );
    }

    #[test]
    fn prompt_request_cache_key_is_stable_for_identical_messages() {
        let messages = vec![PromptMessage::new(PromptMessageRole::User, "hello")];

        assert_eq!(
            prompt_request_cache_key(messages.as_slice()),
            prompt_request_cache_key(messages.as_slice())
        );
    }

    #[test]
    fn prompt_request_cache_key_uses_normalized_prompt_messages() {
        let first = chat_messages_to_prompt_messages(&[
            ChatCompletionMessage::text("system", "first instruction"),
            ChatCompletionMessage::text("developer", "ignored instruction"),
            ChatCompletionMessage::text("user", "hello"),
        ])
        .expect("first normalized prompt");
        let second = chat_messages_to_prompt_messages(&[
            ChatCompletionMessage::text("system", "first instruction"),
            ChatCompletionMessage::text("user", "hello"),
        ])
        .expect("second normalized prompt");

        assert_eq!(
            prompt_request_cache_key(first.as_slice()),
            prompt_request_cache_key(second.as_slice())
        );
    }

    #[test]
    fn generic_server_routes_multiple_dense_model_families()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let llama_path = temp.path().join("tiny-llama.gguf");
        let qwen_path = temp.path().join("tiny-qwen.gguf");
        write_test_gguf(
            &llama_path,
            dense_llama_metadata("tiny server llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;
        write_test_gguf(
            &qwen_path,
            dense_qwen_metadata("tiny server qwen").as_slice(),
            dense_decoder_tensors(true, 2, 3).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&llama_path);
        config.add_model_path(&qwen_path);
        let server = OpenAiCompatServer::from_config(&config)?;

        let llama_model = resolve_generic_model(server.state.as_ref(), Some("tiny-llama"))
            .expect("llama model should resolve");
        let qwen_model = resolve_generic_model(server.state.as_ref(), Some("tiny-qwen"))
            .expect("qwen model should resolve");
        let llama_decoder = llama_model.decoder().expect("llama decoder model");
        let qwen_decoder = qwen_model.decoder().expect("qwen decoder model");

        assert_eq!(llama_decoder.family, GgufDecoderFamily::Llama);
        assert_eq!(qwen_decoder.family, GgufDecoderFamily::Qwen);
        assert_eq!(server.state.models_by_key.len(), 2);
        let health = tokio::runtime::Runtime::new()?
            .block_on(generic_health(State(std::sync::Arc::clone(&server.state))));
        assert_eq!(health.0.residency_mode, CPU_SERVER_RESIDENCY_MODE);
        assert_eq!(health.0.fallback_policy, CPU_SERVER_FALLBACK_POLICY);
        assert_eq!(health.0.hybrid_offload, CPU_SERVER_HYBRID_OFFLOAD_MODE);
        assert_eq!(
            health.0.structured_output_fallbacks,
            Some(vec![
                "choice_set",
                "regex_subset",
                "gbnf_subset",
                "json_schema_subset",
                "json_object",
                "tagged_json_schema",
            ])
        );
        assert_eq!(
            health
                .0
                .structured_output_capabilities
                .as_ref()
                .map(|capabilities| {
                    capabilities
                        .iter()
                        .map(|capability| capability.kind.label())
                        .collect::<Vec<_>>()
                }),
            Some(vec![
                "choice",
                "regex",
                "grammar",
                "json_schema",
                "json_object",
                "tagged_structure",
            ])
        );
        assert_eq!(
            health.0.tool_calling.as_ref().map(|capability| (
                capability.support_level.label(),
                capability.supported_modes.clone(),
                capability.parser,
                capability.argument_validation,
            )),
            Some((
                "fallback",
                vec!["none", "auto", "required", "named"],
                "tagged_json_schema",
                "json_schema_subset",
            ))
        );
        assert_eq!(
            health.0.execution_profile.batch_posture,
            BatchExecutionPosture::ContinuousBatch
        );
        assert_eq!(
            health.0.execution_profile.queue_policy.discipline,
            QueueDiscipline::Fifo
        );
        assert!(
            health
                .0
                .scheduler_policy
                .as_ref()
                .is_some_and(|policy| policy.max_active_requests > 0)
        );
        let models = tokio::runtime::Runtime::new()?.block_on(generic_list_models(State(
            std::sync::Arc::clone(&server.state),
        )));
        assert_eq!(models.0.data.len(), 2);
        assert!(
            models
                .0
                .data
                .iter()
                .all(|model| model.psionic_residency_mode == Some(CPU_SERVER_RESIDENCY_MODE))
        );
        assert!(models.0.data.iter().all(|model| {
            model.psionic_structured_outputs.as_deref()
                == Some(
                    [
                        "choice_set",
                        "regex_subset",
                        "gbnf_subset",
                        "json_schema_subset",
                        "json_object",
                        "tagged_json_schema",
                    ]
                    .as_slice(),
                )
        }));
        assert!(models.0.data.iter().all(|model| {
            model
                .psionic_structured_output_capabilities
                .as_ref()
                .is_some_and(|capabilities| {
                    capabilities
                        .iter()
                        .all(|capability| capability.support_level.label() == "fallback")
                })
        }));
        assert!(models.0.data.iter().all(|model| {
            model
                .psionic_tool_calling
                .as_ref()
                .is_some_and(|capability| {
                    capability.support_level.label() == "fallback"
                        && capability.supported_modes == vec!["none", "auto", "required", "named"]
                        && capability.parser == "tagged_json_schema"
                })
        }));
        assert!(models.0.data.iter().all(|model| {
            model
                .psionic_execution_profile
                .as_ref()
                .map(|profile| profile.batch_posture)
                == Some(BatchExecutionPosture::ContinuousBatch)
        }));
        assert!(
            models
                .0
                .data
                .iter()
                .all(|model| model.psionic_scheduler_policy.is_some())
        );

        let request = ChatCompletionRequest {
            model: Some(String::from("tiny-qwen")),
            messages: vec![ChatCompletionMessage::text("user", "hello")],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        };
        let prompt_messages =
            chat_messages_to_prompt_messages_for_family(&request.messages, qwen_decoder.family)?;
        let rendered = render_prompt_for_model(qwen_model, prompt_messages.as_slice())?;
        let generation_request = GenerationRequest::new_text(
            String::from("generic-server-qwen"),
            qwen_decoder.descriptor.clone(),
            None,
            rendered.text,
            generation_options_from_chat_request_for_family(
                &request,
                qwen_decoder.family,
                rendered.stop_sequences.as_slice(),
            ),
        );
        let response = tokio::runtime::Runtime::new()?.block_on(
            server
                .state
                .workers
                .get(super::OPENAI_COMPAT_WORKER_ID)
                .expect("generic test worker should exist")
                .generate(qwen_model.model_key.clone(), generation_request),
        )?;
        assert_eq!(response.output.text, "world");
        Ok(())
    }

    #[test]
    fn generic_management_status_reports_join_state_and_routes()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let llama_path = temp.path().join("tiny-llama.gguf");
        let qwen_path = temp.path().join("tiny-qwen.gguf");
        write_test_gguf(
            &llama_path,
            dense_llama_metadata("tiny server llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;
        write_test_gguf(
            &qwen_path,
            dense_qwen_metadata("tiny server qwen").as_slice(),
            dense_decoder_tensors(true, 2, 3).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&llama_path);
        config.add_model_path(&qwen_path);
        let server = OpenAiCompatServer::from_config(&config)?;

        let response = tokio::runtime::Runtime::new()?.block_on(async {
            server
                .router()
                .oneshot(
                    Request::builder()
                        .uri("/psionic/management/status")
                        .body(Body::empty())
                        .expect("management status request"),
                )
                .await
        })?;
        assert_eq!(response.status(), StatusCode::OK);
        let body = tokio::runtime::Runtime::new()?
            .block_on(async { to_bytes(response.into_body(), usize::MAX).await })?;
        let payload: serde_json::Value = serde_json::from_slice(&body)?;

        assert_eq!(
            payload["namespace"],
            serde_json::json!("psionic_management")
        );
        assert_eq!(
            payload["join_state"]["posture"],
            serde_json::json!("standalone")
        );
        assert_eq!(payload["node_count"], serde_json::json!(1));
        assert_eq!(payload["model_count"], serde_json::json!(2));
        assert_eq!(
            payload["nodes"][0]["worker_id"],
            serde_json::json!(OPENAI_COMPAT_WORKER_ID)
        );
        assert_eq!(
            payload["nodes"][0]["served_mesh_role"]["role"],
            serde_json::json!("host")
        );
        assert!(
            payload["nodes"][0]["models"]
                .as_array()
                .is_some_and(|models| models.iter().all(|model| model["warm_state"] == "warm"))
        );
        assert!(payload["routes"].as_array().is_some_and(|routes| {
            routes
                .iter()
                .any(|route| route["endpoint"] == RoutingEndpoint::ChatCompletions.path())
        }));

        let direct = tokio::runtime::Runtime::new()?.block_on(generic_management_status(State(
            std::sync::Arc::clone(&server.state),
        )));
        assert_eq!(direct.0.node_count, 1);
        assert_eq!(
            direct.0.join_state.posture,
            super::MeshManagementJoinPosture::Standalone
        );
        Ok(())
    }

    #[test]
    fn generic_management_events_publish_route_selection_updates()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let llama_path = temp.path().join("tiny-llama.gguf");
        write_test_gguf(
            &llama_path,
            dense_llama_metadata("tiny management llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;
        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&llama_path))?;

        let event_response = tokio::runtime::Runtime::new()?.block_on(async {
            server
                .router()
                .oneshot(
                    Request::builder()
                        .uri("/psionic/management/events")
                        .body(Body::empty())
                        .expect("management events request"),
                )
                .await
        })?;
        assert_eq!(event_response.status(), StatusCode::OK);
        assert_eq!(
            event_response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );

        let mut receiver = server.state.management_events.subscribe();
        let route = super::resolve_generic_route(
            server.state.as_ref(),
            psionic_router::RoutingTarget::Default,
            None,
        )?;
        let event = receiver
            .try_recv()
            .expect("route selection event should publish");

        match event.payload {
            super::MeshManagementEventPayload::RouteSelection { request, selection } => {
                assert_eq!(request.endpoint, RoutingEndpoint::ChatCompletions.path());
                assert_eq!(request.target, "default");
                assert_eq!(selection.worker_id, OPENAI_COMPAT_WORKER_ID);
                assert_eq!(selection.model_key, route.selection.model_key);
            }
            other => panic!("unexpected management event payload: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn generic_server_bootstrap_thin_client_proxies_chat_and_management_reports_remote_execution()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = bootstrap_proxy_test_lock()
            .lock()
            .expect("bootstrap proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let temp = tempfile::tempdir()?;
        let llama_path = temp.path().join("tiny-bootstrap-llama.gguf");
        write_test_gguf(
            &llama_path,
            dense_llama_metadata("tiny bootstrap llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let remote_server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&llama_path))?;
        let (base_url, shutdown_tx) =
            runtime.block_on(start_openai_compat_test_server(remote_server))?;

        let bootstrap_env =
            ScopedEnvVar::set("PSIONIC_BOOTSTRAP_PROXY_BASE_URL", base_url.as_str());
        let mode_env = ScopedEnvVar::set("PSIONIC_BOOTSTRAP_PROXY_MODE", "thin_client");
        let local_server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&llama_path))?;
        drop(mode_env);
        drop(bootstrap_env);

        let health = runtime.block_on(generic_health(State(std::sync::Arc::clone(
            &local_server.state,
        ))));
        assert_eq!(health.0.execution_mode, "proxy");
        assert_eq!(health.0.residency_mode, BOOTSTRAP_PROXY_RESIDENCY_MODE);

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&local_server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-bootstrap-llama")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(1),
                ..Default::default()
            },
        ))?;
        let headers = response.headers().clone();
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(payload["object"], serde_json::json!("chat.completion"));
        assert_eq!(
            header_value(&headers, "x-psionic-route-locality"),
            Some(String::from("remote_proxy"))
        );
        assert_eq!(
            header_value(&headers, "x-psionic-route-provenance"),
            Some(String::from("bootstrap_proxy"))
        );
        assert_eq!(
            header_value(&headers, "x-psionic-route-warm-state-reason"),
            Some(String::from("remote_only"))
        );
        assert_eq!(
            header_value(&headers, "x-psionic-route-fallback-posture"),
            Some(String::from(THIN_CLIENT_FALLBACK_POSTURE))
        );

        let management = runtime.block_on(generic_management_status(State(std::sync::Arc::clone(
            &local_server.state,
        ))));
        let local_node = management
            .0
            .nodes
            .iter()
            .find(|node| node.worker_id == OPENAI_COMPAT_WORKER_ID)
            .expect("local thin-client node should be present");
        assert_eq!(local_node.served_mesh_role.role, ServedMeshRole::ThinClient);
        assert_eq!(
            local_node.served_mesh_role.reasons,
            vec![ServedMeshRoleReason::RemoteOnly]
        );
        assert_eq!(
            management
                .0
                .last_route_execution
                .as_ref()
                .map(|status| status.locality),
            Some(super::MeshManagementRouteExecutionLocality::RemoteProxy)
        );

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_server_bootstrap_thin_client_proxies_responses_and_embeddings()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = bootstrap_proxy_test_lock()
            .lock()
            .expect("bootstrap proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let temp = tempfile::tempdir()?;
        let llama_path = temp.path().join("tiny-bootstrap-llama.gguf");
        let embeddings_path = temp.path().join("tiny-bootstrap-embed.safetensors");
        write_test_gguf(
            &llama_path,
            dense_llama_metadata("tiny bootstrap llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;
        ByteProjectionEmbedder::write_default_safetensors_artifact(&embeddings_path)?;

        let mut remote_config = OpenAiCompatConfig::new(&llama_path);
        remote_config.add_model_path(&embeddings_path);
        let remote_server = OpenAiCompatServer::from_config(&remote_config)?;
        let (base_url, shutdown_tx) =
            runtime.block_on(start_openai_compat_test_server(remote_server))?;

        let bootstrap_env =
            ScopedEnvVar::set("PSIONIC_BOOTSTRAP_PROXY_BASE_URL", base_url.as_str());
        let mode_env = ScopedEnvVar::set("PSIONIC_BOOTSTRAP_PROXY_MODE", "thin_client");
        let mut local_config = OpenAiCompatConfig::new(&llama_path);
        local_config.add_model_path(&embeddings_path);
        let local_server = OpenAiCompatServer::from_config(&local_config)?;
        drop(mode_env);
        drop(bootstrap_env);

        let responses = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&local_server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-bootstrap-llama")),
                instructions: Some(String::from("Be brief.")),
                conversation: None,
                input: ResponsesInput::Text(String::from("hello")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                ..Default::default()
            },
        ))?;
        let responses_headers = responses.headers().clone();
        let responses_payload = runtime.block_on(response_json(responses))?;
        assert_eq!(responses_payload["object"], serde_json::json!("response"));
        assert_eq!(
            header_value(&responses_headers, "x-psionic-route-locality"),
            Some(String::from("remote_proxy"))
        );
        assert_eq!(
            header_value(&responses_headers, "x-psionic-route-provenance"),
            Some(String::from("bootstrap_proxy"))
        );

        let embeddings = runtime.block_on(handle_generic_embeddings(
            std::sync::Arc::clone(&local_server.state),
            EmbeddingsRequest {
                model: Some(String::from("tiny-bootstrap-embed")),
                input: EmbeddingsInput::One(String::from("hello")),
                dimensions: Some(4),
                encoding_format: Some(String::from("float")),
            },
        ))?;
        let embeddings_headers = embeddings.headers().clone();
        let embeddings_payload = runtime.block_on(response_json(embeddings))?;
        assert_eq!(embeddings_payload["object"], serde_json::json!("list"));
        assert_eq!(
            embeddings_payload["data"][0]["embedding"]
                .as_array()
                .map(Vec::len),
            Some(4)
        );
        assert_eq!(
            header_value(&embeddings_headers, "x-psionic-route-locality"),
            Some(String::from("remote_proxy"))
        );
        assert_eq!(
            header_value(&embeddings_headers, "x-psionic-route-fallback-posture"),
            Some(String::from(THIN_CLIENT_FALLBACK_POSTURE))
        );

        let management = runtime.block_on(generic_management_status(State(std::sync::Arc::clone(
            &local_server.state,
        ))));
        assert_eq!(
            management
                .0
                .last_route_execution
                .as_ref()
                .map(|status| status.provenance.as_str()),
            Some("bootstrap_proxy")
        );

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_server_bootstrap_warming_reports_host_role_and_remote_execution()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = bootstrap_proxy_test_lock()
            .lock()
            .expect("bootstrap proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let temp = tempfile::tempdir()?;
        let llama_path = temp.path().join("tiny-bootstrap-llama.gguf");
        write_test_gguf(
            &llama_path,
            dense_llama_metadata("tiny bootstrap llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let remote_server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&llama_path))?;
        let (base_url, shutdown_tx) =
            runtime.block_on(start_openai_compat_test_server(remote_server))?;

        let bootstrap_env =
            ScopedEnvVar::set("PSIONIC_BOOTSTRAP_PROXY_BASE_URL", base_url.as_str());
        let mode_env = ScopedEnvVar::set("PSIONIC_BOOTSTRAP_PROXY_MODE", "warming");
        let local_server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&llama_path))?;
        drop(mode_env);
        drop(bootstrap_env);

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&local_server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-bootstrap-llama")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(1),
                ..Default::default()
            },
        ))?;
        let headers = response.headers().clone();
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(payload["object"], serde_json::json!("chat.completion"));
        assert_eq!(
            header_value(&headers, "x-psionic-route-locality"),
            Some(String::from("remote_proxy"))
        );
        assert_eq!(
            header_value(&headers, "x-psionic-route-warm-state-reason"),
            Some(String::from("warming"))
        );
        assert_eq!(
            header_value(&headers, "x-psionic-route-fallback-posture"),
            Some(String::from(WARMING_FALLBACK_POSTURE))
        );

        let management = runtime.block_on(generic_management_status(State(std::sync::Arc::clone(
            &local_server.state,
        ))));
        let local_node = management
            .0
            .nodes
            .iter()
            .find(|node| node.worker_id == OPENAI_COMPAT_WORKER_ID)
            .expect("local warming node should be present");
        assert_eq!(local_node.served_mesh_role.role, ServedMeshRole::Host);
        assert_eq!(
            local_node.served_mesh_role.posture,
            ServedMeshRolePosture::Downgraded
        );
        assert_eq!(
            local_node.served_mesh_role.reasons,
            vec![ServedMeshRoleReason::Warming]
        );
        assert_eq!(
            management
                .0
                .last_route_execution
                .as_ref()
                .map(|status| status.locality),
            Some(super::MeshManagementRouteExecutionLocality::RemoteProxy)
        );
        assert_eq!(
            management
                .0
                .last_route_execution
                .as_ref()
                .and_then(|status| status.fallback_posture.as_deref()),
            Some(WARMING_FALLBACK_POSTURE)
        );

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_server_qwen_pilot_is_end_to_end_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let qwen_path = temp.path().join("tiny-qwen-pilot.gguf");
        write_test_gguf(
            &qwen_path,
            dense_qwen_metadata("tiny pilot qwen").as_slice(),
            dense_decoder_tensors(true, 2, 3).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&qwen_path))?;

        let health = tokio::runtime::Runtime::new()?
            .block_on(generic_health(State(std::sync::Arc::clone(&server.state))));
        let pilot_model_id = health.0.default_model.clone();
        assert_eq!(
            health.0.default_model_supported_endpoints,
            vec!["/v1/chat/completions", "/v1/responses"]
        );
        assert_eq!(
            health.0.execution_profile.batch_posture,
            BatchExecutionPosture::ContinuousBatch
        );
        assert!(health.0.scheduler_policy.is_some());

        let models = tokio::runtime::Runtime::new()?.block_on(generic_list_models(State(
            std::sync::Arc::clone(&server.state),
        )));
        let model = models
            .0
            .data
            .iter()
            .find(|model| model.id == pilot_model_id)
            .expect("pilot qwen model should be listed");
        assert_eq!(model.psionic_model_family, "qwen");
        assert_eq!(
            model.psionic_supported_endpoints,
            vec!["/v1/chat/completions", "/v1/responses"]
        );
        assert_eq!(
            model.psionic_residency_mode,
            Some(CPU_SERVER_RESIDENCY_MODE)
        );
        assert_eq!(
            model
                .psionic_execution_profile
                .as_ref()
                .map(|profile| profile.batch_posture),
            Some(BatchExecutionPosture::ContinuousBatch)
        );
        assert!(model.psionic_scheduler_policy.is_some());

        let response =
            tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(pilot_model_id.clone()),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-batch-posture"),
            Some(String::from("continuous_batch"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-scheduling-class"),
            Some(String::from("mixed_prefill_decode"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-prefill-decode-mode"),
            Some(String::from("disaggregated_colocated"))
        );

        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(payload["model"], serde_json::json!(pilot_model_id));
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("world")
        );
        assert_eq!(payload["usage"]["completion_tokens"], serde_json::json!(1));
        Ok(())
    }

    #[test]
    fn generic_cuda_gemma4_load_plan_keeps_the_first_claim_bounded()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let gemma_path = temp.path().join("tiny-gemma4.gguf");
        write_test_gguf(
            &gemma_path,
            dense_gemma4_metadata("tiny pilot gemma4").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let (loaded_model, accepted_names, load_plan) =
            load_generic_decoder_model(&gemma_path, 0, OpenAiCompatBackend::Cuda)?;

        assert_eq!(
            load_plan.runtime_kind,
            OpenAiCompatRuntimeKind::GgufDecoderCudaGemma4
        );
        assert_eq!(loaded_model.backend_label(), "cuda");
        assert_eq!(loaded_model.execution_mode_label(), "native");
        assert_eq!(loaded_model.execution_engine_label(), "psionic");
        assert_eq!(
            model_endpoint_paths(&loaded_model),
            vec!["/v1/chat/completions"]
        );
        assert!(!loaded_model.supports_tool_calling());
        assert!(!loaded_model.supports_response_state());
        assert_eq!(
            loaded_model.decoder().map(|decoder| decoder.family),
            Some(GgufDecoderFamily::Gemma4)
        );
        assert!(accepted_names.contains("tiny-gemma4.gguf"));
        Ok(())
    }

    #[test]
    fn generic_server_gemma4_prompt_render_keeps_system_on_the_instruction_surface()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let gemma_path = temp.path().join("tiny-gemma4-render.gguf");
        write_test_gguf(
            &gemma_path,
            dense_gemma4_metadata_with_chat_template("tiny gemma4 render").as_slice(),
            dense_gemma4_cuda_decoder_tensors_with_vocab(7, 5).as_slice(),
        )?;

        let prompt_messages = chat_messages_to_prompt_messages_for_family(
            &[
                ChatCompletionMessage::text("system", "Be terse."),
                ChatCompletionMessage::text("user", "hello"),
            ],
            GgufDecoderFamily::Gemma4,
        )?;
        let content = GgufContent::read_path(&gemma_path)?;
        let renderer = GgufPromptTemplateRenderer::new(
            content.load_tokenizer()?,
            content.load_chat_templates()?,
        );
        let rendered = renderer.render(None, prompt_messages.as_slice(), true)?;

        assert_eq!(prompt_messages[0].role, PromptMessageRole::System);
        assert_eq!(prompt_messages[1].role, PromptMessageRole::User);
        assert_eq!(
            rendered.text,
            "<bos><|turn>developer\nBe terse.<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n"
        );
        Ok(())
    }

    #[test]
    fn generic_server_gemma4_cuda_publication_and_generation_are_honest_when_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let gemma_path = temp.path().join("tiny-gemma4-cuda.gguf");
        write_test_gguf(
            &gemma_path,
            dense_gemma4_metadata_with_chat_template("tiny gemma4 cuda").as_slice(),
            dense_gemma4_cuda_decoder_tensors_with_vocab(7, 5).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&gemma_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let health = runtime.block_on(generic_health(State(std::sync::Arc::clone(&server.state))));
        let model_id = health.0.default_model.clone();
        assert_eq!(health.0.backend, "cuda");
        assert_eq!(health.0.execution_mode, "native");
        assert_eq!(health.0.execution_engine, "psionic");
        assert_eq!(
            health.0.default_model_supported_endpoints,
            vec!["/v1/chat/completions"]
        );
        assert_eq!(health.0.supported_endpoints, vec!["/v1/chat/completions"]);
        assert_eq!(
            health.0.execution_profile.batch_posture,
            BatchExecutionPosture::SingleRequestOnly
        );
        assert!(health.0.scheduler_policy.is_none());
        assert_eq!(health.0.response_state, None);
        assert_eq!(
            health.0.tool_calling.as_ref().map(|capability| (
                capability.support_level.label(),
                capability.supported_modes.clone(),
                capability.parser,
                capability.argument_validation,
            )),
            Some((
                "unsupported",
                vec!["none"],
                "not_available",
                "not_available",
            ))
        );
        assert!(
            health
                .0
                .structured_output_capabilities
                .as_ref()
                .is_some_and(|capabilities| {
                    capabilities
                        .iter()
                        .all(|capability| capability.support_level.label() == "unsupported")
                })
        );

        let models = runtime.block_on(generic_list_models(State(std::sync::Arc::clone(
            &server.state,
        ))));
        let model = models
            .0
            .data
            .iter()
            .find(|candidate| candidate.id == model_id)
            .expect("gemma4 cuda model should be listed");
        assert_eq!(model.psionic_model_family, "gemma4");
        assert_eq!(
            model.psionic_supported_endpoints,
            vec!["/v1/chat/completions"]
        );
        assert_eq!(model.psionic_served_backend, Some("cuda"));
        assert_eq!(model.psionic_execution_mode, Some("native"));
        assert_eq!(model.psionic_execution_engine, Some("psionic"));
        assert_eq!(model.psionic_response_state, None);
        assert!(
            model
                .psionic_structured_output_capabilities
                .as_ref()
                .is_some_and(|capabilities| {
                    capabilities
                        .iter()
                        .all(|capability| capability.support_level.label() == "unsupported")
                })
        );

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(model_id.clone()),
                messages: vec![
                    ChatCompletionMessage::text("system", "Be terse."),
                    ChatCompletionMessage::text("user", "hello"),
                ],
                temperature: Some(0.0),
                max_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-served-backend"),
            Some(String::from("cuda"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-execution-mode"),
            Some(String::from("native"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-execution-engine"),
            Some(String::from("psionic"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-batch-posture"),
            Some(String::from("single_request_only"))
        );

        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(payload["model"], serde_json::json!(model_id));
        assert!(payload["choices"][0]["message"]["content"].is_string());
        assert_eq!(payload["usage"]["completion_tokens"], serde_json::json!(1));
        Ok(())
    }

    #[test]
    fn generic_server_gemma4_cuda_fails_closed_for_unsupported_surfaces_when_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let gemma_path = temp.path().join("tiny-gemma4-refusal.gguf");
        write_test_gguf(
            &gemma_path,
            dense_gemma4_metadata_with_chat_template("tiny gemma4 refusal").as_slice(),
            dense_gemma4_cuda_decoder_tensors_with_vocab(7, 5).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&gemma_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;
        let model_id = runtime
            .block_on(generic_health(State(std::sync::Arc::clone(&server.state))))
            .0
            .default_model;

        let tool_error = runtime
            .block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(model_id.clone()),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: vec![weather_tool_definition()],
                    tool_choice: None,
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("gemma4 tool calling should fail closed");
        let tool_payload = runtime.block_on(response_json(tool_error.into_response()))?;
        assert!(
            tool_payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("lacks tool-calling support")
        );

        let structured_output_error = runtime
            .block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(model_id.clone()),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: Some(ChatCompletionResponseFormatRequest {
                        kind: String::from("json_object"),
                        json_schema: None,
                        schema: None,
                    }),
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("gemma4 structured output should fail closed");
        let structured_output_payload =
            runtime.block_on(response_json(structured_output_error.into_response()))?;
        assert!(
            structured_output_payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("lacks structured-output support")
        );

        let multimodal_error = runtime
            .block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(model_id.clone()),
                    messages: vec![ChatCompletionMessage::multimodal(
                        "user",
                        vec![
                            ChatCompletionContentPart::text("hello "),
                            ChatCompletionContentPart::image_url("https://example.invalid/cat.png"),
                        ],
                    )],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("gemma4 multimodal input should fail closed");
        let multimodal_payload =
            runtime.block_on(response_json(multimodal_error.into_response()))?;
        assert_eq!(
            multimodal_payload["error"]["message"],
            serde_json::json!(
                "multimodal inputs are unavailable on the current `gemma4` generic prompt-render path"
            )
        );

        let responses_error = runtime
            .block_on(handle_generic_responses(
                std::sync::Arc::clone(&server.state),
                ResponsesRequest {
                    model: Some(model_id),
                    instructions: None,
                    conversation: None,
                    input: ResponsesInput::Text(String::from("hello")),
                    temperature: Some(0.0),
                    max_output_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    previous_response_id: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_response_state: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("gemma4 should refuse responses");
        let response = responses_error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let payload = runtime.block_on(response_json(response))?;
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("/v1/chat/completions")
        );
        Ok(())
    }

    #[test]
    fn gemma4_e4b_cuda_conformance_repeat_is_machine_checkable_when_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let fixture = golden_prompt_fixture("gemma4_e4b").expect("gemma4 prompt fixture");
        let path = gemma4_pilot_fixture_path(fixture.source_path);
        if !Path::new(path.as_str()).exists() {
            return Ok(());
        }

        let mut config = OpenAiCompatConfig::new(path.as_str());
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;
        let model_id = runtime
            .block_on(generic_health(State(std::sync::Arc::clone(&server.state))))
            .0
            .default_model;
        let case = GenerateConformanceCase::from_generate_compatible_prompt_fixture(
            "gemma4-e4b-repeat",
            model_id.as_str(),
            "gemma4_e4b",
            "gemma4_e4b.default",
            "gemma4_e4b.default_developer",
        )?;

        let first = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(model_id.clone()),
                messages: vec![
                    ChatCompletionMessage::text("developer", "Be terse."),
                    ChatCompletionMessage::text("user", "Summarize the lane."),
                ],
                temperature: Some(0.0),
                max_tokens: Some(8),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        assert_eq!(
            header_value(first.headers(), "x-psionic-served-backend"),
            Some(String::from("cuda"))
        );
        let first_payload = runtime.block_on(response_json(first))?;

        let second = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(model_id.clone()),
                messages: vec![
                    ChatCompletionMessage::text("developer", "Be terse."),
                    ChatCompletionMessage::text("user", "Summarize the lane."),
                ],
                temperature: Some(0.0),
                max_tokens: Some(8),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let second_payload = runtime.block_on(response_json(second))?;

        let third = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(model_id.clone()),
                messages: vec![
                    ChatCompletionMessage::text("developer", "Be terse."),
                    ChatCompletionMessage::text("user", "Summarize the lane."),
                ],
                temperature: Some(0.0),
                max_tokens: Some(8),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let third_payload = runtime.block_on(response_json(third))?;

        let expected_rendered_prompt = case.expected_rendered_prompt.clone();
        let first_observation = generate_observation_from_chat_payload(
            &first_payload,
            expected_rendered_prompt.as_deref(),
        );
        let second_observation = generate_observation_from_chat_payload(
            &second_payload,
            expected_rendered_prompt.as_deref(),
        );
        let third_observation = generate_observation_from_chat_payload(
            &third_payload,
            expected_rendered_prompt.as_deref(),
        );

        let suite = ConformanceSuite {
            id: String::from("gemma4-e4b-cuda-repeat"),
            compare_tags: false,
            compare_ps: false,
            show_cases: Vec::new(),
            generate_cases: vec![case],
            embed_cases: Vec::new(),
        };
        let baseline = RecordedConformanceSubject::new("gemma4-e4b-cuda-run-1").with_generate_case(
            "gemma4-e4b-repeat",
            SubjectObservation::Supported(first_observation),
        );
        let candidate = RecordedConformanceSubject::new("gemma4-e4b-cuda-run-2")
            .with_generate_case(
                "gemma4-e4b-repeat",
                SubjectObservation::Supported(second_observation.clone()),
            );

        let mut baseline = baseline;
        let mut candidate = candidate;
        let report = run_conformance_suite(&suite, &mut baseline, &mut candidate)?;

        assert_eq!(report.summary.passed, 1);
        assert_eq!(report.summary.failed, 0);
        assert_eq!(report.summary.unsupported, 0);
        assert_eq!(report.summary.intentional_differences, 0);
        assert!(report.cutover_ready());
        assert_eq!(
            second_observation.output_text,
            third_observation.output_text
        );
        assert_eq!(
            second_observation.done_reason,
            third_observation.done_reason
        );
        assert_eq!(second_observation.eval_count, third_observation.eval_count);
        Ok(())
    }

    #[test]
    fn generic_server_qwen35_proxy_publication_and_generation_are_honest()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = qwen35_proxy_test_lock()
            .lock()
            .expect("qwen35 proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let (base_url, shutdown_tx, observed_requests) =
            runtime.block_on(start_qwen35_proxy_test_server())?;
        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_decoder_metadata("tiny qwen35 proxy").as_slice(),
            qwen35_decoder_tensors().as_slice(),
        )?;

        let _proxy_env = ScopedEnvVar::set("PSIONIC_QWEN35_PROXY_BASE_URL", base_url.as_str());
        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&qwen35_path))?;
        drop(_proxy_env);

        let health = runtime.block_on(generic_health(State(std::sync::Arc::clone(&server.state))));
        assert_eq!(health.0.backend, "cpu");
        assert_eq!(health.0.execution_mode, "proxy");
        assert_eq!(health.0.execution_engine, "llama.cpp");
        assert_eq!(health.0.residency_mode, "llama_cpp_proxy");
        assert_eq!(health.0.hybrid_offload, "unsupported");
        assert_eq!(health.0.fallback_policy, "proxy_only");
        assert_eq!(
            health.0.execution_profile.batch_posture,
            BatchExecutionPosture::SingleRequestOnly
        );
        assert!(health.0.scheduler_policy.is_none());
        assert_eq!(health.0.structured_output_fallbacks, None);
        assert!(
            health
                .0
                .structured_output_capabilities
                .as_ref()
                .is_some_and(|capabilities| {
                    capabilities
                        .iter()
                        .all(|capability| capability.support_level.label() == "unsupported")
                })
        );
        assert_eq!(
            health.0.tool_calling.as_ref().map(|capability| (
                capability.support_level.label(),
                capability.supported_modes.clone(),
                capability.parser,
                capability.argument_validation,
            )),
            Some((
                "unsupported",
                vec!["none"],
                "not_available",
                "not_available",
            ))
        );
        assert!(health.0.response_state.is_some());
        assert_eq!(
            health.0.multimodal_projection_mode,
            Some("prompt_projection_only")
        );
        assert_eq!(
            health.0.multimodal_supported_media,
            Some(vec!["image", "video"])
        );
        assert_eq!(
            health.0.multimodal_projection_config,
            Some(Qwen35MultimodalProjectionConfig {
                vision_block_count: 2,
                vision_embedding_length: 6,
                vision_start_token_id: TokenId(900),
                vision_end_token_id: TokenId(901),
                image_token_id: TokenId(902),
            })
        );

        let models = runtime.block_on(generic_list_models(State(std::sync::Arc::clone(
            &server.state,
        ))));
        let model = models
            .0
            .data
            .iter()
            .find(|model| model.id == "tiny-qwen35.gguf")
            .expect("qwen35 proxy model should be listed");
        assert_eq!(model.psionic_model_family, "qwen35");
        assert_eq!(model.psionic_served_backend, Some("cpu"));
        assert_eq!(model.psionic_execution_mode, Some("proxy"));
        assert_eq!(model.psionic_execution_engine, Some("llama.cpp"));
        assert_eq!(model.psionic_residency_mode, Some("llama_cpp_proxy"));
        assert_eq!(model.psionic_hybrid_offload, Some("unsupported"));
        assert_eq!(model.psionic_fallback_policy, Some("proxy_only"));
        assert_eq!(model.psionic_structured_outputs, None);
        assert!(
            model
                .psionic_structured_output_capabilities
                .as_ref()
                .is_some_and(|capabilities| {
                    capabilities
                        .iter()
                        .all(|capability| capability.support_level.label() == "unsupported")
                })
        );
        assert!(
            model
                .psionic_tool_calling
                .as_ref()
                .is_some_and(|capability| capability.support_level.label() == "unsupported")
        );
        assert_eq!(
            model
                .psionic_execution_profile
                .as_ref()
                .map(|profile| profile.batch_posture),
            Some(BatchExecutionPosture::SingleRequestOnly)
        );
        assert!(model.psionic_scheduler_policy.is_none());
        assert!(model.psionic_response_state.is_some());
        assert_eq!(
            model.psionic_multimodal_projection_mode,
            Some("prompt_projection_only")
        );
        assert_eq!(
            model.psionic_multimodal_supported_media,
            Some(vec!["image", "video"])
        );
        assert_eq!(
            model.psionic_multimodal_projection_config,
            Some(Qwen35MultimodalProjectionConfig {
                vision_block_count: 2,
                vision_embedding_length: 6,
                vision_start_token_id: TokenId(900),
                vision_end_token_id: TokenId(901),
                image_token_id: TokenId(902),
            })
        );

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(2),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-execution-mode"),
            Some(String::from("proxy"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-execution-engine"),
            Some(String::from("llama.cpp"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-residency-mode"),
            Some(String::from("llama_cpp_proxy"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-fallback-policy"),
            Some(String::from("proxy_only"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-batch-posture"),
            Some(String::from("single_request_only"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-scheduling-class"),
            None
        );
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("proxy world")
        );
        assert_eq!(payload["usage"]["completion_tokens"], serde_json::json!(2));

        let observed_requests = observed_requests
            .lock()
            .expect("observed qwen35 proxy requests should be readable");
        assert!(observed_requests.iter().any(|body| {
            body.get("n_predict") == Some(&serde_json::json!(2))
                && body["prompt"]
                    .as_str()
                    .is_some_and(|prompt| prompt.contains("hello"))
        }));

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_server_qwen35_proxy_forwards_sampling_controls()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = qwen35_proxy_test_lock()
            .lock()
            .expect("qwen35 proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let (base_url, shutdown_tx, observed_requests) =
            runtime.block_on(start_qwen35_proxy_test_server())?;
        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_decoder_metadata("tiny qwen35 proxy").as_slice(),
            qwen35_decoder_tensors().as_slice(),
        )?;

        let _proxy_env = ScopedEnvVar::set("PSIONIC_QWEN35_PROXY_BASE_URL", base_url.as_str());
        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&qwen35_path))?;
        drop(_proxy_env);

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35")),
                messages: vec![ChatCompletionMessage::text("user", "sample with controls")],
                temperature: None,
                top_k: Some(23),
                top_p: Some(0.85),
                min_p: Some(0.1),
                typical_p: Some(0.72),
                mirostat: Some(1),
                mirostat_tau: Some(5.5),
                mirostat_eta: Some(0.15),
                repeat_penalty: Some(1.15),
                repeat_last_n: Some(32),
                presence_penalty: Some(0.25),
                frequency_penalty: Some(0.5),
                seed: Some(42),
                max_tokens: Some(5),
                stop: Some(StopSequences::One(String::from("done"))),
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("proxy world")
        );

        let observed_requests = observed_requests
            .lock()
            .expect("observed qwen35 proxy requests should be readable");
        assert!(observed_requests.iter().any(|body| {
            body.get("n_predict") == Some(&serde_json::json!(5))
                && body.get("temperature").is_none()
                && body.get("top_k") == Some(&serde_json::json!(23))
                && body
                    .get("top_p")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|value| (value - 0.85).abs() < 1e-6)
                && body
                    .get("min_p")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|value| (value - 0.1).abs() < 1e-6)
                && body
                    .get("typical_p")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|value| (value - 0.72).abs() < 1e-6)
                && body.get("mirostat") == Some(&serde_json::json!(1))
                && body
                    .get("mirostat_tau")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|value| (value - 5.5).abs() < 1e-6)
                && body
                    .get("mirostat_eta")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|value| (value - 0.15).abs() < 1e-6)
                && body
                    .get("repeat_penalty")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|value| (value - 1.15).abs() < 1e-6)
                && body.get("repeat_last_n") == Some(&serde_json::json!(32))
                && body
                    .get("presence_penalty")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|value| (value - 0.25).abs() < 1e-6)
                && body
                    .get("frequency_penalty")
                    .and_then(serde_json::Value::as_f64)
                    .is_some_and(|value| (value - 0.5).abs() < 1e-6)
                && body.get("seed") == Some(&serde_json::json!(42))
                && body
                    .get("stop")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|values| values.contains(&serde_json::json!("done")))
                && body["prompt"]
                    .as_str()
                    .is_some_and(|prompt| prompt.contains("sample with controls"))
        }));

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_server_qwen35_proxy_streaming_plain_text_emits_multiple_delta_chunks()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = qwen35_proxy_test_lock()
            .lock()
            .expect("qwen35 proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let (base_url, shutdown_tx, _) = runtime.block_on(start_qwen35_proxy_test_server())?;
        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-streaming.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_decoder_metadata("tiny qwen35 proxy stream").as_slice(),
            qwen35_decoder_tensors().as_slice(),
        )?;

        let _proxy_env = ScopedEnvVar::set("PSIONIC_QWEN35_PROXY_BASE_URL", base_url.as_str());
        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&qwen35_path))?;
        drop(_proxy_env);

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35-streaming")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(2),
                stop: None,
                stream: true,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let body = runtime.block_on(response_text(response))?;
        let events = sse_json_events(body.as_str())?;

        assert_eq!(events.len(), 3);
        assert_eq!(
            events[0]["choices"][0]["delta"]["content"],
            serde_json::json!("proxy ")
        );
        assert_eq!(
            events[1]["choices"][0]["delta"]["content"],
            serde_json::json!("world")
        );
        assert_eq!(
            events[2]["choices"][0]["finish_reason"],
            serde_json::json!("stop")
        );
        assert!(body.contains("[DONE]"));

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_server_qwen35_headers_remain_model_specific_when_default_model_is_native()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = qwen35_proxy_test_lock()
            .lock()
            .expect("qwen35 proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let (base_url, shutdown_tx, _) = runtime.block_on(start_qwen35_proxy_test_server())?;
        let temp = tempfile::tempdir()?;
        let llama_path = temp.path().join("tiny-llama.gguf");
        let qwen35_path = temp.path().join("tiny-qwen35.gguf");
        write_test_gguf(
            &llama_path,
            dense_llama_metadata("tiny server llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;
        write_test_gguf(
            &qwen35_path,
            qwen35_decoder_metadata("tiny qwen35 proxy").as_slice(),
            qwen35_decoder_tensors().as_slice(),
        )?;

        let _proxy_env = ScopedEnvVar::set("PSIONIC_QWEN35_PROXY_BASE_URL", base_url.as_str());
        let mut config = OpenAiCompatConfig::new(&llama_path);
        config.add_model_path(&qwen35_path);
        let server = OpenAiCompatServer::from_config(&config)?;
        drop(_proxy_env);

        let health = runtime.block_on(generic_health(State(std::sync::Arc::clone(&server.state))));
        assert_eq!(health.0.execution_mode, "native");
        assert_eq!(health.0.execution_engine, "psionic");

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(2),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-execution-mode"),
            Some(String::from("proxy"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-execution-engine"),
            Some(String::from("llama.cpp"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-batch-posture"),
            Some(String::from("single_request_only"))
        );
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("proxy world")
        );

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_server_qwen35_fails_closed_for_tools_and_structured_outputs()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = qwen35_proxy_test_lock()
            .lock()
            .expect("qwen35 proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let (base_url, shutdown_tx, _) = runtime.block_on(start_qwen35_proxy_test_server())?;
        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_decoder_metadata("tiny qwen35 proxy").as_slice(),
            qwen35_decoder_tensors().as_slice(),
        )?;

        let _proxy_env = ScopedEnvVar::set("PSIONIC_QWEN35_PROXY_BASE_URL", base_url.as_str());
        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&qwen35_path))?;
        drop(_proxy_env);

        let tool_error = runtime
            .block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-qwen35")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(2),
                    stop: None,
                    stream: false,
                    tools: vec![weather_tool_definition()],
                    tool_choice: None,
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("qwen35 tool calling should fail closed");
        let tool_payload = runtime.block_on(response_json(tool_error.into_response()))?;
        assert!(
            tool_payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("lacks tool-calling support")
        );

        let structured_output_error = runtime
            .block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-qwen35")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(2),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: Some(ChatCompletionResponseFormatRequest {
                        kind: String::from("json_object"),
                        json_schema: None,
                        schema: None,
                    }),
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("qwen35 structured output should fail closed");
        let structured_output_payload =
            runtime.block_on(response_json(structured_output_error.into_response()))?;
        assert!(
            structured_output_payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("lacks structured-output support")
        );

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_grammar_fallback_is_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata("tiny native qwen35").as_slice(),
            qwen35_native_full_attention_decoder_tensors().as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let health = runtime.block_on(generic_health(State(std::sync::Arc::clone(&server.state))));
        assert_eq!(health.0.backend, "cuda");
        assert_eq!(health.0.execution_mode, "native");
        assert_eq!(health.0.execution_engine, "psionic");
        assert_eq!(
            health.0.structured_output_fallbacks,
            Some(vec![
                "choice_set",
                "regex_subset",
                "gbnf_subset",
                "json_schema_subset",
                "json_object",
                "tagged_json_schema",
            ])
        );
        assert_eq!(
            health.0.tool_calling.as_ref().map(|capability| (
                capability.support_level.label(),
                capability.supported_modes.clone(),
                capability.parser,
                capability.argument_validation,
            )),
            Some((
                "fallback",
                vec!["none", "auto", "required", "named"],
                "tagged_json_schema",
                "json_schema_subset",
            ))
        );
        assert!(
            health
                .0
                .structured_output_capabilities
                .as_ref()
                .is_some_and(|capabilities| {
                    capabilities
                        .iter()
                        .all(|capability| capability.support_level.label() != "unsupported")
                })
        );

        let models = runtime.block_on(generic_list_models(State(std::sync::Arc::clone(
            &server.state,
        ))));
        let model = models
            .0
            .data
            .iter()
            .find(|model| model.id == "tiny-qwen35.gguf")
            .expect("native qwen35 model should be listed");
        assert_eq!(
            model.psionic_structured_outputs,
            Some(vec![
                "choice_set",
                "regex_subset",
                "gbnf_subset",
                "json_schema_subset",
                "json_object",
                "tagged_json_schema",
            ])
        );
        assert!(
            model
                .psionic_tool_calling
                .as_ref()
                .is_some_and(|capability| {
                    capability.support_level.label() == "fallback"
                        && capability.supported_modes == vec!["none", "auto", "required", "named"]
                        && capability.parser == "tagged_json_schema"
                        && capability.argument_validation == "json_schema_subset"
                })
        );
        assert!(
            model
                .psionic_structured_output_capabilities
                .as_ref()
                .is_some_and(|capabilities| {
                    capabilities
                        .iter()
                        .all(|capability| capability.support_level.label() != "unsupported")
                })
        );

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: Some(PsionicGrammarRequest {
                    grammar: String::from("root ::= \"world\"\n"),
                    syntax: Some(StructuredGrammarSyntax::Gbnf),
                }),
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-execution-mode"),
            Some(String::from("native"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-execution-engine"),
            Some(String::from("psionic"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-mode"),
            Some(String::from("fallback_grammar"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-parser"),
            Some(String::from("gbnf_subset"))
        );
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(payload["choices"][0]["message"]["content"], "world");
        assert_eq!(
            payload["psionic_structured_output"]["mode"],
            serde_json::json!("fallback_grammar")
        );
        assert_eq!(
            payload["psionic_structured_output"]["kind"],
            serde_json::json!("grammar")
        );
        assert_eq!(
            payload["psionic_structured_output"]["parser"],
            serde_json::json!("gbnf_subset")
        );
        assert_eq!(
            payload["psionic_structured_value"],
            serde_json::json!({
                "kind": "grammar",
                "value": "world"
            })
        );
        Ok(())
    }

    fn qwen35_cuda_debug_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn run_native_qwen35_cuda_generation(
        request_id: &str,
        prompt: &str,
        options: GenerationOptions,
        debug_attention: bool,
    ) -> Result<GenerationResponse, Box<dyn std::error::Error>> {
        let _env_lock = qwen35_cuda_debug_env_lock()
            .lock()
            .expect("qwen35 cuda env lock should not be poisoned");
        let _debug_guard = ScopedEnvVar::set_optional(
            "PSIONIC_QWEN35_DEBUG_ATTENTION",
            debug_attention.then_some("1"),
        );

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join(format!("{request_id}.gguf"));
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata("tiny native qwen35 parity").as_slice(),
            qwen35_native_full_attention_decoder_tensors().as_slice(),
        )?;

        let mut service = crate::CudaGgufQwen35TextGenerationService::from_gguf_path(&qwen35_path)?;
        let request = GenerationRequest::new_text(
            String::from(request_id),
            service.model_descriptor().clone(),
            None,
            String::from(prompt),
            options,
        );
        Ok(crate::TextGenerationExecutor::generate(
            &mut service,
            &request,
        )?)
    }

    fn qwen35_cuda_metrics(response: &GenerationResponse) -> &crate::Qwen35CudaDecodeOutputMetrics {
        response
            .metrics
            .qwen35_cuda_decode
            .as_ref()
            .expect("native qwen35 response should expose cuda decode metrics")
    }

    fn qwen35_cuda_graph_metrics(response: &GenerationResponse) -> &crate::CudaGraphReplayMetrics {
        qwen35_cuda_metrics(response)
            .graph_replay
            .as_ref()
            .expect("fused qwen35 response should expose graph replay metrics")
    }

    #[test]
    fn native_qwen35_cuda_argmax_graph_fast_path_matches_debug_attention_when_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let options = GenerationOptions::greedy(3);
        let fused = run_native_qwen35_cuda_generation(
            "qwen35-cuda-argmax-fused",
            "hello",
            options.clone(),
            false,
        )?;
        let debug =
            run_native_qwen35_cuda_generation("qwen35-cuda-argmax-debug", "hello", options, true)?;

        assert_eq!(fused.output.tokens, debug.output.tokens);
        assert_eq!(fused.output.text, debug.output.text);
        assert_eq!(fused.termination, debug.termination);

        let fused_metrics = qwen35_cuda_metrics(&fused);
        let debug_metrics = qwen35_cuda_metrics(&debug);
        assert_eq!(fused_metrics.output_modes, debug_metrics.output_modes);
        assert_eq!(fused_metrics.readback_bytes, debug_metrics.readback_bytes);
        assert_eq!(
            fused_metrics.raw_logits_materialized,
            debug_metrics.raw_logits_materialized
        );
        assert!(debug_metrics.graph_replay.is_none());
        assert!(
            fused_metrics
                .output_modes
                .contains(&crate::Qwen35CudaDecodeOutputMode::ArgmaxOnly)
        );
        assert!(!fused_metrics.raw_logits_materialized);
        assert!(fused_metrics.readback_bytes > 0);

        let graph_metrics = qwen35_cuda_graph_metrics(&fused);
        assert!(
            graph_metrics
                .output_modes
                .contains(&crate::CudaGraphReplayMode::ArgmaxOnly)
        );
        assert!(graph_metrics.step_count >= 2);
        assert!(graph_metrics.capture_count >= 1);
        assert!(graph_metrics.replay_hit_count >= 1);
        Ok(())
    }

    #[test]
    fn native_qwen35_cuda_top_k_graph_fast_path_matches_debug_attention_when_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let mut options = GenerationOptions::sample(3);
        options.temperature = Some(0.8);
        options.top_k = Some(4);
        options.seed = Some(7);
        let fused = run_native_qwen35_cuda_generation(
            "qwen35-cuda-top-k-fused",
            "hello",
            options.clone(),
            false,
        )?;
        let debug =
            run_native_qwen35_cuda_generation("qwen35-cuda-top-k-debug", "hello", options, true)?;

        assert_eq!(fused.output.tokens, debug.output.tokens);
        assert_eq!(fused.output.text, debug.output.text);
        assert_eq!(fused.termination, debug.termination);

        let expected_mode = crate::Qwen35CudaDecodeOutputMode::TopKCandidates { top_k: 4 };
        let expected_graph_mode = crate::CudaGraphReplayMode::TopKCandidates { top_k: 4 };
        let fused_metrics = qwen35_cuda_metrics(&fused);
        let debug_metrics = qwen35_cuda_metrics(&debug);
        assert_eq!(fused_metrics.output_modes, debug_metrics.output_modes);
        assert_eq!(fused_metrics.readback_bytes, debug_metrics.readback_bytes);
        assert_eq!(
            fused_metrics.raw_logits_materialized,
            debug_metrics.raw_logits_materialized
        );
        assert!(debug_metrics.graph_replay.is_none());
        assert!(fused_metrics.output_modes.contains(&expected_mode));
        assert!(!fused_metrics.raw_logits_materialized);
        assert!(fused_metrics.readback_bytes > 0);

        let graph_metrics = qwen35_cuda_graph_metrics(&fused);
        assert!(graph_metrics.output_modes.contains(&expected_graph_mode));
        assert!(graph_metrics.step_count >= 2);
        assert!(graph_metrics.capture_count >= 1);
        assert!(graph_metrics.replay_hit_count >= 1);
        Ok(())
    }

    #[test]
    fn native_qwen35_cuda_full_logits_graph_fast_path_matches_debug_attention_when_available()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let mut options = GenerationOptions::sample(3);
        options.temperature = Some(0.8);
        options.seed = Some(11);
        let fused = run_native_qwen35_cuda_generation(
            "qwen35-cuda-full-logits-fused",
            "hello",
            options.clone(),
            false,
        )?;
        let debug = run_native_qwen35_cuda_generation(
            "qwen35-cuda-full-logits-debug",
            "hello",
            options,
            true,
        )?;

        assert_eq!(fused.output.tokens, debug.output.tokens);
        assert_eq!(fused.output.text, debug.output.text);
        assert_eq!(fused.termination, debug.termination);

        let fused_metrics = qwen35_cuda_metrics(&fused);
        let debug_metrics = qwen35_cuda_metrics(&debug);
        assert_eq!(fused_metrics.output_modes, debug_metrics.output_modes);
        assert_eq!(fused_metrics.readback_bytes, debug_metrics.readback_bytes);
        assert_eq!(
            fused_metrics.raw_logits_materialized,
            debug_metrics.raw_logits_materialized
        );
        assert!(debug_metrics.graph_replay.is_none());
        assert!(
            fused_metrics
                .output_modes
                .contains(&crate::Qwen35CudaDecodeOutputMode::RawLogits)
        );
        assert!(fused_metrics.raw_logits_materialized);
        assert!(fused_metrics.readback_bytes > 0);

        let graph_metrics = qwen35_cuda_graph_metrics(&fused);
        assert!(
            graph_metrics
                .output_modes
                .contains(&crate::CudaGraphReplayMode::RawLogits)
        );
        assert!(graph_metrics.step_count >= 2);
        assert!(graph_metrics.capture_count >= 1);
        assert!(graph_metrics.replay_hit_count >= 1);
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_tool_choice_auto_can_return_message_envelope()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-auto-tool.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 auto tool",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    "{\"kind\":\"message\",\"content\":\"world\"}",
                    "world",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35-auto-tool")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(1),
                stop: None,
                stream: false,
                tools: vec![weather_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Mode(String::from("auto"))),
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("world")
        );
        assert!(payload["choices"][0]["message"]["tool_calls"].is_null());
        assert!(payload["psionic_tool_calls"].is_null());
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_tool_contract_merges_with_system_instruction()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-render-tool-contract.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata(
                "tiny native qwen35 render tool contract",
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors().as_slice(),
        )?;

        let request = ChatCompletionRequest {
            model: Some(String::from("tiny-qwen35-render-tool-contract")),
            messages: vec![
                ChatCompletionMessage::text("system", "You are Hermes."),
                ChatCompletionMessage::text("user", "hello"),
            ],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: vec![weather_tool_definition()],
            tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
            parallel_tool_calls: Some(false),
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        };
        let prompt_messages = apply_tool_contract_to_prompt_messages(
            chat_messages_to_prompt_messages_for_family(
                &request.messages,
                GgufDecoderFamily::Qwen35,
            )?,
            tool_contract_from_chat_request(&request, false)?.as_ref(),
        );
        let content = GgufContent::read_path(&qwen35_path)?;
        let renderer = GgufPromptTemplateRenderer::new(
            content.load_tokenizer()?,
            content.load_chat_templates()?,
        );
        let rendered = renderer.render(None, prompt_messages.as_slice(), true)?;

        assert!(rendered.text.starts_with("<|im_start|>system\n"));
        assert!(rendered.text.contains("When tools are enabled"));
        assert!(rendered.text.contains("You are Hermes."));
        assert!(!rendered.text.contains("developer\n"));
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_parallel_tool_contract_includes_batched_example()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let qwen35_path = temp
            .path()
            .join("tiny-qwen35-render-parallel-tool-contract.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata(
                "tiny native qwen35 render parallel tool contract",
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors().as_slice(),
        )?;

        let request = ChatCompletionRequest {
            model: Some(String::from("tiny-qwen35-render-parallel-tool-contract")),
            messages: vec![
                ChatCompletionMessage::text("system", "You are Hermes."),
                ChatCompletionMessage::text("user", "call both tools"),
            ],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: vec![weather_tool_definition(), time_tool_definition()],
            tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
            parallel_tool_calls: Some(true),
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        };
        let prompt_messages = apply_tool_contract_to_prompt_messages(
            chat_messages_to_prompt_messages_for_family(
                &request.messages,
                GgufDecoderFamily::Qwen35,
            )?,
            tool_contract_from_chat_request(&request, false)?.as_ref(),
        );
        let content = GgufContent::read_path(&qwen35_path)?;
        let renderer = GgufPromptTemplateRenderer::new(
            content.load_tokenizer()?,
            content.load_chat_templates()?,
        );
        let rendered = renderer.render(None, prompt_messages.as_slice(), true)?;

        assert!(
            rendered
                .text
                .contains("If multiple tools are needed in the same turn")
        );
        assert!(rendered.text.contains("\"name\": \"get_time\""));
        assert!(rendered.text.contains("\"name\": \"get_weather\""));
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_required_tool_call_is_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-required-tool.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 required tool",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}",
                    "world",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35-required-tool")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(1),
                stop: None,
                stream: false,
                tools: vec![weather_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                parallel_tool_calls: Some(false),
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["finish_reason"],
            serde_json::json!("tool_calls")
        );
        assert_eq!(
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            payload["psionic_tool_calls"][0]["arguments"],
            serde_json::json!({
                "latitude": 48.8566,
                "longitude": 2.3522
            })
        );
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_tool_prompt_prefix_cache_is_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp
            .path()
            .join("tiny-qwen35-required-tool-prefix-cache.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 required tool prefix cache",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}",
                    "world",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;
        let tenant = String::from("hermes-tool-loop");
        let build_request = || ChatCompletionRequest {
            model: Some(String::from("tiny-qwen35-required-tool-prefix-cache")),
            messages: vec![ChatCompletionMessage::text("user", "hello")],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: vec![weather_tool_definition()],
            tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
            parallel_tool_calls: Some(false),
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: Some(PrefixCacheControl {
                mode: PrefixCacheMode::Auto,
                tenant_id: Some(tenant.clone()),
            }),
            ..Default::default()
        };

        let seeded = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            build_request(),
        ))?;
        assert_eq!(
            header_value(seeded.headers(), "x-psionic-prefix-cache-state"),
            Some(String::from("none"))
        );
        assert_eq!(
            header_value(seeded.headers(), "x-psionic-prefix-cache-reused-tokens"),
            Some(String::from("0"))
        );

        let cached = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            build_request(),
        ))?;
        assert_eq!(
            header_value(cached.headers(), "x-psionic-prefix-cache-state"),
            Some(String::from("hit"))
        );
        assert!(
            header_value(cached.headers(), "x-psionic-prefix-cache-reused-tokens")
                .is_some_and(|value| value != "0"),
            "cached qwen35 tool request should reuse prompt tokens"
        );
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_tool_result_prefix_cache_preserves_final_answer()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let tool_call_json = "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}";
        let final_answer = "Tomorrow will also be sunny.";

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp
            .path()
            .join("tiny-qwen35-tool-result-prefix-cache.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 tool result prefix cache",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    tool_call_json,
                    final_answer,
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;
        let tenant = String::from("hermes-tool-result-loop");
        let build_request = || ChatCompletionRequest {
            model: Some(String::from("tiny-qwen35-tool-result-prefix-cache")),
            messages: vec![
                ChatCompletionMessage::text("user", "hello"),
                ChatCompletionMessage {
                    role: String::from("assistant"),
                    content: ChatCompletionMessageContent::Text(String::new()),
                    name: None,
                    tool_calls: Some(vec![ChatCompletionToolCall {
                        id: String::from("call-1"),
                        kind: String::from("function"),
                        function: ChatCompletionToolCallFunction {
                            name: String::from("get_weather"),
                            arguments: String::from("{\"latitude\":48.8566,\"longitude\":2.3522}"),
                        },
                    }]),
                    tool_call_id: None,
                },
                ChatCompletionMessage {
                    role: String::from("tool"),
                    content: ChatCompletionMessageContent::Text(String::from(
                        "{\"forecast\":\"sunny\",\"tomorrow\":\"sunny\"}",
                    )),
                    name: None,
                    tool_calls: None,
                    tool_call_id: Some(String::from("call-1")),
                },
            ],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: vec![weather_tool_definition()],
            tool_choice: Some(ToolChoiceRequest::Mode(String::from("auto"))),
            parallel_tool_calls: Some(false),
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: Some(PrefixCacheControl {
                mode: PrefixCacheMode::Auto,
                tenant_id: Some(tenant.clone()),
            }),
            ..Default::default()
        };

        let seeded = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            build_request(),
        ))?;
        assert_eq!(
            header_value(seeded.headers(), "x-psionic-prefix-cache-state"),
            Some(String::from("none"))
        );
        let seeded_payload = runtime.block_on(response_json(seeded))?;
        assert_eq!(
            seeded_payload["choices"][0]["message"]["content"],
            serde_json::json!(final_answer)
        );

        let cached = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            build_request(),
        ))?;
        assert_eq!(
            header_value(cached.headers(), "x-psionic-prefix-cache-state"),
            Some(String::from("hit"))
        );
        assert!(
            header_value(cached.headers(), "x-psionic-prefix-cache-reused-tokens")
                .is_some_and(|value| value != "0"),
            "cached qwen35 tool-result request should reuse prompt tokens"
        );
        let cached_payload = runtime.block_on(response_json(cached))?;
        assert_eq!(
            cached_payload["choices"][0]["message"]["content"],
            serde_json::json!(final_answer)
        );
        assert_eq!(
            cached_payload["choices"][0]["finish_reason"],
            serde_json::json!("stop")
        );
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_named_tool_choice_surfaces_tool_call()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-named-tool.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 named tool",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}",
                    "world",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35-named-tool")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(1),
                stop: None,
                stream: false,
                tools: vec![weather_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Named(NamedToolChoiceRequest {
                    kind: String::from("function"),
                    function: NamedToolChoiceFunction {
                        name: String::from("get_weather"),
                    },
                })),
                parallel_tool_calls: Some(true),
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            payload["psionic_tool_calls"][0]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            payload["psionic_tool_calls"].as_array().map(Vec::len),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_tool_call_validation_refuses_invalid_arguments()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-invalid-tool.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 invalid tool",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":\"oops\",\"longitude\":2.3522}}]}",
                    "world",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let error = runtime
            .block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-qwen35-invalid-tool")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: vec![weather_tool_definition()],
                    tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                    parallel_tool_calls: Some(false),
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("native qwen35 invalid tool arguments should be refused");
        let payload = runtime.block_on(response_json(error.into_response()))?;
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("did not satisfy the declared schema")
        );
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_streaming_tool_calls_emit_delta_tool_calls()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-stream-tool.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 stream tool",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}",
                    "world",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35-stream-tool")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(1),
                stop: None,
                stream: true,
                tools: vec![weather_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                parallel_tool_calls: Some(false),
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let body = runtime.block_on(response_text(response))?;
        let events = sse_json_events(body.as_str())?;
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][0]["index"],
            serde_json::json!(0)
        );
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            serde_json::json!("{\"latitude\":48.8566,\"longitude\":2.3522}")
        );
        assert_eq!(
            events[1]["choices"][0]["finish_reason"],
            serde_json::json!("tool_calls")
        );
        assert!(body.contains("[DONE]"));
        Ok(())
    }

    #[test]
    fn generic_server_native_qwen35_streaming_parallel_tool_calls_preserve_order()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-stream-tool-batch.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 stream tool batch",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}},{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}]}",
                    "world",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let response = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            ChatCompletionRequest {
                model: Some(String::from("tiny-qwen35-stream-tool-batch")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(1),
                stop: None,
                stream: true,
                tools: vec![weather_tool_definition(), time_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                parallel_tool_calls: Some(true),
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let body = runtime.block_on(response_text(response))?;
        let events = sse_json_events(body.as_str())?;
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][0]["index"],
            serde_json::json!(0)
        );
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][1]["index"],
            serde_json::json!(1)
        );
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][1]["function"]["name"],
            serde_json::json!("get_time")
        );
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][1]["function"]["arguments"],
            serde_json::json!("{\"timezone\":\"UTC\"}")
        );
        assert_eq!(
            events[1]["choices"][0]["finish_reason"],
            serde_json::json!("tool_calls")
        );
        assert!(body.contains("[DONE]"));
        Ok(())
    }

    #[test]
    fn generic_server_qwen35_projects_multimodal_inputs_through_real_template_markers()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = qwen35_proxy_test_lock()
            .lock()
            .expect("qwen35 proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let (base_url, shutdown_tx, observed_requests) =
            runtime.block_on(start_qwen35_proxy_test_server())?;
        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_decoder_metadata("tiny qwen35 proxy").as_slice(),
            qwen35_decoder_tensors().as_slice(),
        )?;

        let _proxy_env = ScopedEnvVar::set("PSIONIC_QWEN35_PROXY_BASE_URL", base_url.as_str());
        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&qwen35_path))?;
        drop(_proxy_env);

        let response = runtime
            .block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-qwen35")),
                    messages: vec![ChatCompletionMessage::multimodal(
                        "user",
                        vec![
                            ChatCompletionContentPart::text("hello "),
                            ChatCompletionContentPart::image_url(
                                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB",
                            ),
                            ChatCompletionContentPart::text(" compare "),
                            ChatCompletionContentPart::video_url(
                                "https://example.invalid/pilot.mp4",
                            ),
                        ],
                    )],
                    temperature: Some(0.0),
                    max_tokens: Some(2),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect("qwen35 multimodal input should project through the prompt surface");
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("proxy world")
        );

        let observed_requests = observed_requests
            .lock()
            .expect("observed qwen35 proxy requests should be readable");
        assert!(observed_requests.iter().any(|body| {
            body.get("n_predict") == Some(&serde_json::json!(2))
                && body["prompt"].as_str().is_some_and(|prompt| {
                    prompt.contains(
                        "hello <|vision_start|><|image_pad|><|vision_end|> compare <|vision_start|><|video_pad|><|vision_end|>"
                    )
                })
        }));

        let system_error = runtime
            .block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-qwen35")),
                    messages: vec![ChatCompletionMessage::multimodal(
                        "system",
                        vec![
                            ChatCompletionContentPart::text("look"),
                            ChatCompletionContentPart::image_url("https://example.invalid/cat.png"),
                        ],
                    )],
                    temperature: Some(0.0),
                    max_tokens: Some(2),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("qwen35 system multimodal input should follow template refusal");
        let system_payload = runtime.block_on(response_json(system_error.into_response()))?;
        assert_eq!(
            system_payload["error"]["message"],
            serde_json::json!("qwen35 system messages cannot contain image or video parts")
        );

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_responses_qwen35_projects_multimodal_message_input()
    -> Result<(), Box<dyn std::error::Error>> {
        let _proxy_lock = qwen35_proxy_test_lock()
            .lock()
            .expect("qwen35 proxy test lock should not be poisoned");
        let runtime = tokio::runtime::Runtime::new()?;
        let (base_url, shutdown_tx, observed_requests) =
            runtime.block_on(start_qwen35_proxy_test_server())?;
        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_decoder_metadata("tiny qwen35 proxy").as_slice(),
            qwen35_decoder_tensors().as_slice(),
        )?;

        let _proxy_env = ScopedEnvVar::set("PSIONIC_QWEN35_PROXY_BASE_URL", base_url.as_str());
        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&qwen35_path))?;
        drop(_proxy_env);

        let response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-qwen35")),
                instructions: None,
                conversation: None,
                input: ResponsesInput::Messages(vec![ChatCompletionMessage::multimodal(
                    "user",
                    vec![
                        ChatCompletionContentPart::text("describe "),
                        ChatCompletionContentPart::image_url("https://example.invalid/dog.png"),
                    ],
                )]),
                temperature: Some(0.0),
                max_output_tokens: Some(2),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(payload["output_text"], serde_json::json!("proxy world"));

        let observed_requests = observed_requests
            .lock()
            .expect("observed qwen35 proxy requests should be readable");
        assert!(observed_requests.iter().any(|body| {
            body["prompt"].as_str().is_some_and(|prompt| {
                prompt.contains("describe <|vision_start|><|image_pad|><|vision_end|>")
            })
        }));

        let _ = shutdown_tx.send(());
        Ok(())
    }

    #[test]
    fn generic_responses_qwen35_tool_result_messages_preserve_role_and_name_through_prompt_conversion()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-response-prompt-replay.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 response prompt replay",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}",
                    "Tomorrow will also be sunny.",
                    "{\"forecast\":\"sunny\",\"tomorrow\":\"sunny\"}",
                    "what about tomorrow?",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(11).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let route = resolve_generic_model_for_endpoint(
            server.state.as_ref(),
            Some("tiny-qwen35-response-prompt-replay"),
            RoutingEndpoint::Responses,
            RoutingRequest::new(RoutingEndpoint::Responses).require_response_state(),
        )?;
        let model = route
            .loaded_model
            .decoder()
            .expect("response route should resolve a decoder");
        let prompt = response_input_to_prompt_messages_with_options(
            &ResponsesRequest {
                model: Some(String::from("tiny-qwen35-response-prompt-replay")),
                instructions: None,
                conversation: None,
                input: ResponsesInput::Messages(vec![
                    ChatCompletionMessage::text(
                        "assistant",
                        "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}",
                    ),
                    ChatCompletionMessage::named_text(
                        "tool",
                        "{\"forecast\":\"sunny\",\"tomorrow\":\"sunny\"}",
                        "get_weather",
                    ),
                    ChatCompletionMessage::text("user", "what about tomorrow?"),
                ]),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
            model,
            false,
            false,
        )?;
        assert_eq!(prompt.len(), 3);
        assert_eq!(prompt[0].role, PromptMessageRole::Assistant);
        assert_eq!(
            prompt[0].content,
            "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}"
        );
        assert_eq!(prompt[1].role, PromptMessageRole::Tool);
        assert_eq!(prompt[1].author_name.as_deref(), Some("get_weather"));
        assert_eq!(
            prompt[1].content,
            "{\"forecast\":\"sunny\",\"tomorrow\":\"sunny\"}"
        );
        assert_eq!(prompt[2].role, PromptMessageRole::User);
        assert_eq!(prompt[2].content, "what about tomorrow?");
        Ok(())
    }

    #[test]
    fn generic_responses_native_qwen35_tool_turn_stores_replayable_response_state()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let tool_call_json = "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}";
        let temp = tempfile::tempdir()?;
        let qwen35_path = temp.path().join("tiny-qwen35-response-tool-turn.gguf");
        write_test_gguf(
            &qwen35_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 response tool turn",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    tool_call_json,
                    "Tomorrow will also be sunny.",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&qwen35_path);
        config.backend = OpenAiCompatBackend::Cuda;
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-qwen35-response-tool-turn")),
                instructions: None,
                conversation: None,
                input: ResponsesInput::Text(String::from("hello")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: vec![weather_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                parallel_tool_calls: Some(false),
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let payload = runtime.block_on(response_json(response))?;
        assert_eq!(payload["output_text"], serde_json::json!(""));
        assert_eq!(
            payload["psionic_tool_calls"][0]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            payload["psionic_response_state"]["stored"],
            serde_json::json!(true)
        );
        let response_id = payload["id"]
            .as_str()
            .expect("stored qwen response id should be present")
            .to_string();
        let stored_context = server
            .state
            .response_state
            .lock()
            .expect("response-state store should be readable")
            .load_context(Some(response_id.as_str()), None)?;
        assert_eq!(
            stored_context.model_key.as_deref(),
            Some(server.state.default_model_key.as_str())
        );
        assert_eq!(stored_context.prompt_history.len(), 2);
        assert_eq!(
            stored_context.prompt_history[0].role,
            PromptMessageRole::User
        );
        assert_eq!(stored_context.prompt_history[0].content, "hello");
        assert_eq!(
            stored_context.prompt_history[1].role,
            PromptMessageRole::Assistant
        );
        assert_eq!(stored_context.prompt_history[1].content, tool_call_json);
        Ok(())
    }

    #[test]
    fn generic_responses_native_qwen35_tool_result_replay_reaches_final_assistant_answer()
    -> Result<(), Box<dyn std::error::Error>> {
        let backend = psionic_backend_cuda::CudaBackend::new();
        if !backend.quantized_kernels_available() {
            return Ok(());
        }

        let tool_call_json = "{\"kind\":\"tool_calls\",\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}";
        let tool_result_json = "{\"forecast\":\"sunny\",\"tomorrow\":\"sunny\"}";
        let final_answer = "Tomorrow will also be sunny.";

        let temp = tempfile::tempdir()?;
        let qwen35_tool_path = temp.path().join("tiny-qwen35-response-tool-source.gguf");
        write_test_gguf(
            &qwen35_tool_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 response tool source",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    tool_call_json,
                    final_answer,
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(9).as_slice(),
        )?;

        let mut tool_config = OpenAiCompatConfig::new(&qwen35_tool_path);
        tool_config.backend = OpenAiCompatBackend::Cuda;
        let tool_server = OpenAiCompatServer::from_config(&tool_config)?;
        let runtime = tokio::runtime::Runtime::new()?;

        let first_response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&tool_server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-qwen35-response-tool-source")),
                instructions: None,
                conversation: None,
                input: ResponsesInput::Text(String::from("hello")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: vec![weather_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                parallel_tool_calls: Some(false),
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let first_payload = runtime.block_on(response_json(first_response))?;
        let first_response_id = first_payload["id"]
            .as_str()
            .expect("tool-turn qwen response id should be present")
            .to_string();
        let first_context = tool_server
            .state
            .response_state
            .lock()
            .expect("source response-state store should be readable")
            .load_context(Some(first_response_id.as_str()), None)?;

        let qwen35_final_path = temp.path().join("tiny-qwen35-response-final-answer.gguf");
        write_test_gguf(
            &qwen35_final_path,
            qwen35_native_full_attention_decoder_metadata_with_tokens(
                "tiny native qwen35 response final answer",
                vec![
                    "<|bos|>",
                    "<|eos|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<think>",
                    "</think>",
                    "hello",
                    final_answer,
                    tool_result_json,
                    "what about tomorrow?",
                ],
            )
            .as_slice(),
            qwen35_native_full_attention_decoder_tensors_with_vocab(10).as_slice(),
        )?;

        let mut final_config = OpenAiCompatConfig::new(&qwen35_final_path);
        final_config.backend = OpenAiCompatBackend::Cuda;
        let final_server = OpenAiCompatServer::from_config(&final_config)?;
        let seeded_conversation_id = String::from("conv-qwen35-tool-loop");
        let seeded_response_id = String::from("resp-qwen35-tool-loop-seeded");
        let mut seeded_prompt_history = first_context.prompt_history.clone();
        seeded_prompt_history.push(tool_result_prompt_message("get_weather", tool_result_json));
        final_server
            .state
            .response_state
            .lock()
            .expect("final response-state store should be writable")
            .record_response(ResponseStateRecord {
                response_id: seeded_response_id.clone(),
                model_key: final_server.state.default_model_key.clone(),
                worker_id: String::from(super::OPENAI_COMPAT_WORKER_ID),
                conversation_id: Some(seeded_conversation_id.clone()),
                prompt_history: seeded_prompt_history,
            })?;

        let continued_response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&final_server.state),
            ResponsesRequest {
                model: None,
                instructions: None,
                conversation: Some(seeded_conversation_id.clone()),
                input: ResponsesInput::Text(String::from("what about tomorrow?")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let continued_payload = runtime.block_on(response_json(continued_response))?;
        assert_eq!(
            continued_payload["output_text"],
            serde_json::json!(final_answer)
        );
        assert_eq!(
            continued_payload["previous_response_id"],
            serde_json::json!(seeded_response_id)
        );
        assert_eq!(
            continued_payload["conversation"]["id"],
            serde_json::json!(seeded_conversation_id)
        );
        assert_eq!(
            continued_payload["conversation"]["revision"],
            serde_json::json!(2)
        );
        assert!(
            continued_payload["psionic_response_state"]["replayed_prompt_messages"]
                .as_u64()
                .is_some_and(|count| count >= 3)
        );

        let continued_response_id = continued_payload["id"]
            .as_str()
            .expect("continued qwen response id should be present")
            .to_string();
        let continued_context = final_server
            .state
            .response_state
            .lock()
            .expect("continued response-state store should be readable")
            .load_context(Some(continued_response_id.as_str()), None)?;
        assert!(continued_context.prompt_history.iter().any(|message| {
            message.role == PromptMessageRole::Tool
                && message.author_name.as_deref() == Some("get_weather")
                && message.content == tool_result_json
        }));
        assert_eq!(
            continued_context
                .prompt_history
                .last()
                .expect("continued history should end with an assistant turn")
                .content,
            final_answer
        );
        Ok(())
    }

    #[test]
    fn generic_server_boots_and_generates_for_gpt_oss() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-gpt-oss.gguf");
        write_test_gguf(
            &path,
            gpt_oss_metadata().as_slice(),
            gpt_oss_tensors().as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let model = resolve_generic_model(server.state.as_ref(), None)
            .expect("default model should resolve");
        let decoder = model.decoder().expect("gpt-oss decoder model");
        assert_eq!(decoder.family, GgufDecoderFamily::GptOss);

        let request = ChatCompletionRequest {
            model: None,
            messages: vec![ChatCompletionMessage::text("user", "hello")],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        };
        let prompt_messages =
            chat_messages_to_prompt_messages_for_family(&request.messages, decoder.family)?;
        let rendered = render_prompt_for_model(model, prompt_messages.as_slice())?;
        let generation_request = GenerationRequest::new_text(
            String::from("generic-server-gpt-oss"),
            decoder.descriptor.clone(),
            None,
            rendered.text,
            generation_options_from_chat_request_for_family(
                &request,
                decoder.family,
                rendered.stop_sequences.as_slice(),
            ),
        );
        let response = tokio::runtime::Runtime::new()?.block_on(
            server
                .state
                .workers
                .get(super::OPENAI_COMPAT_WORKER_ID)
                .expect("generic test worker should exist")
                .generate(model.model_key.clone(), generation_request),
        )?;
        assert_eq!(response.usage.output_tokens, 1);
        Ok(())
    }

    #[test]
    fn generic_server_refuses_reasoning_request_for_unsupported_family()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny reasoning llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let error = tokio::runtime::Runtime::new()?
            .block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-llama")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: Some(PsionicReasoningRequest {
                        parser: None,
                        mode: PsionicReasoningMode::Separate,
                    }),
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("llama family should refuse the reasoning parser contract");
        let payload =
            tokio::runtime::Runtime::new()?.block_on(response_json(error.into_response()))?;
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("does not expose a Psionic reasoning parser")
        );
        Ok(())
    }

    #[test]
    fn generic_server_surfaces_embeddings_truthfully() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let llama_path = temp.path().join("tiny-llama.gguf");
        let embeddings_path = temp.path().join("tiny-embed.safetensors");
        write_test_gguf(
            &llama_path,
            dense_llama_metadata("tiny server llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;
        ByteProjectionEmbedder::write_default_safetensors_artifact(&embeddings_path)?;

        let mut config = OpenAiCompatConfig::new(&llama_path);
        config.add_model_path(&embeddings_path);
        let server = OpenAiCompatServer::from_config(&config)?;

        let health = tokio::runtime::Runtime::new()?
            .block_on(generic_health(State(std::sync::Arc::clone(&server.state))));
        assert_eq!(
            health.0.supported_endpoints,
            vec!["/v1/chat/completions", "/v1/embeddings", "/v1/responses"]
        );
        assert_eq!(
            health
                .0
                .response_state
                .as_ref()
                .map(|capability| capability.continuation_modes.clone()),
            Some(vec![String::from("append_turn")])
        );

        let models = tokio::runtime::Runtime::new()?.block_on(generic_list_models(State(
            std::sync::Arc::clone(&server.state),
        )));
        let decoder_model = models
            .0
            .data
            .iter()
            .find(|model| {
                model.psionic_supported_endpoints.contains(&"/v1/responses")
                    && model.psionic_response_state.is_some()
            })
            .expect("decoder model should be listed");
        assert_eq!(
            decoder_model
                .psionic_response_state
                .as_ref()
                .map(|capability| capability.cache_behavior.clone()),
            Some(String::from("prompt_replay_only"))
        );
        let embeddings_model = models
            .0
            .data
            .iter()
            .find(|model| model.psionic_supported_endpoints == vec!["/v1/embeddings"])
            .expect("embeddings model should be listed");
        assert_eq!(embeddings_model.psionic_embedding_dimensions, Some(8));
        assert_eq!(embeddings_model.psionic_response_state, None);

        let response = tokio::runtime::Runtime::new()?.block_on(generic_embeddings(
            State(std::sync::Arc::clone(&server.state)),
            Json(EmbeddingsRequest {
                model: Some(String::from("tiny-embed")),
                input: EmbeddingsInput::Many(vec![String::from("hello"), String::from("world")]),
                dimensions: Some(4),
                encoding_format: Some(String::from("float")),
            }),
        ));
        assert_eq!(response.status(), StatusCode::OK);
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(payload["object"], serde_json::json!("list"));
        assert_eq!(payload["data"].as_array().map(Vec::len), Some(2));
        assert_eq!(
            payload["data"][0]["embedding"].as_array().map(Vec::len),
            Some(4)
        );
        Ok(())
    }

    #[test]
    fn generic_responses_surface_runs_real_generation() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny response llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response = tokio::runtime::Runtime::new()?.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-llama")),
                instructions: Some(String::from("Be brief.")),
                conversation: None,
                input: ResponsesInput::Text(String::from("hello")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        assert_eq!(response.status(), StatusCode::OK);
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(payload["object"], serde_json::json!("response"));
        assert_eq!(payload["status"], serde_json::json!("completed"));
        assert_eq!(payload["output_text"], serde_json::json!("world"));
        assert_eq!(payload["previous_response_id"], serde_json::Value::Null);
        assert_eq!(
            payload["conversation"]["id"],
            serde_json::json!("psionic-conv-1")
        );
        assert_eq!(
            payload["psionic_response_state"]["stored"],
            serde_json::json!(true)
        );
        assert_eq!(payload["output"][0]["type"], serde_json::json!("message"));
        Ok(())
    }

    #[test]
    fn generic_responses_conversation_state_replays_and_updates()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-stateful-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny stateful llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let runtime = tokio::runtime::Runtime::new()?;
        let first_response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-stateful-llama")),
                instructions: Some(String::from("Be brief.")),
                conversation: None,
                input: ResponsesInput::Text(String::from("hello")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let first_payload = runtime.block_on(response_json(first_response))?;
        assert_eq!(
            first_payload["psionic_response_state"]["replayed_prompt_messages"],
            serde_json::json!(0)
        );
        assert_eq!(
            first_payload["psionic_response_state"]["input_messages_appended"],
            serde_json::json!(2)
        );
        assert_eq!(
            first_payload["psionic_response_state"]["assistant_messages_recorded"],
            serde_json::json!(1)
        );
        assert_eq!(
            first_payload["psionic_response_state"]["conversation_item_count"],
            serde_json::json!(3)
        );
        let first_response_id = first_payload["id"]
            .as_str()
            .expect("response id")
            .to_string();
        let conversation_id = first_payload["conversation"]["id"]
            .as_str()
            .expect("conversation id")
            .to_string();

        let second_response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: None,
                instructions: None,
                conversation: Some(conversation_id.clone()),
                input: ResponsesInput::Text(String::from("again")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let second_payload = runtime.block_on(response_json(second_response))?;
        assert_eq!(
            second_payload["previous_response_id"],
            serde_json::json!(first_response_id)
        );
        assert_eq!(
            second_payload["conversation"]["id"],
            serde_json::json!(conversation_id)
        );
        assert_eq!(
            second_payload["conversation"]["revision"],
            serde_json::json!(2)
        );
        assert_eq!(
            second_payload["psionic_response_state"]["replayed_prompt_messages"],
            serde_json::json!(3)
        );
        assert_eq!(
            second_payload["psionic_response_state"]["input_messages_appended"],
            serde_json::json!(1)
        );
        assert_eq!(
            second_payload["psionic_response_state"]["conversation_item_count"],
            serde_json::json!(5)
        );
        Ok(())
    }

    #[test]
    fn generic_responses_file_backed_state_survives_server_restart()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let model_path = temp.path().join("tiny-durable-stateful-llama.gguf");
        let state_path = temp.path().join("response-state.json");
        write_test_gguf(
            &model_path,
            dense_llama_metadata("tiny durable stateful llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let config = OpenAiCompatConfig::new(&model_path);
        let runtime = tokio::runtime::Runtime::new()?;
        let first_server = OpenAiCompatServer::from_config_with_response_state_store(
            &config,
            ResponseStateStore::file_backed(&state_path, ResponseStateRetentionPolicy::default())?,
        )?;
        let first_response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&first_server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-durable-stateful-llama")),
                instructions: Some(String::from("Be brief.")),
                conversation: None,
                input: ResponsesInput::Text(String::from("hello")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let first_payload = runtime.block_on(response_json(first_response))?;
        let conversation_id = first_payload["conversation"]["id"]
            .as_str()
            .expect("conversation id")
            .to_string();
        let first_response_id = first_payload["id"]
            .as_str()
            .expect("response id")
            .to_string();

        let second_server = OpenAiCompatServer::from_config_with_response_state_store(
            &config,
            ResponseStateStore::file_backed(&state_path, ResponseStateRetentionPolicy::default())?,
        )?;
        let second_response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&second_server.state),
            ResponsesRequest {
                model: None,
                instructions: None,
                conversation: Some(conversation_id.clone()),
                input: ResponsesInput::Text(String::from("again")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let second_payload = runtime.block_on(response_json(second_response))?;
        assert_eq!(
            second_payload["previous_response_id"],
            serde_json::json!(first_response_id)
        );
        assert_eq!(
            second_payload["conversation"]["id"],
            serde_json::json!(conversation_id)
        );
        assert_eq!(
            second_payload["psionic_response_state"]["storage"],
            serde_json::json!("json_file")
        );
        assert_eq!(
            second_payload["psionic_response_state"]["retention_scope"],
            serde_json::json!("best_effort_local_durable")
        );
        assert_eq!(
            second_payload["psionic_response_state"]["replayed_prompt_messages"],
            serde_json::json!(3)
        );
        Ok(())
    }

    #[test]
    fn generic_responses_refuse_unknown_state_references() -> Result<(), Box<dyn std::error::Error>>
    {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-unknown-state-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny unknown state llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let error = tokio::runtime::Runtime::new()?
            .block_on(handle_generic_responses(
                std::sync::Arc::clone(&server.state),
                ResponsesRequest {
                    model: Some(String::from("tiny-unknown-state-llama")),
                    instructions: None,
                    conversation: None,
                    input: ResponsesInput::Text(String::from("hello")),
                    temperature: Some(0.0),
                    max_output_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    previous_response_id: Some(String::from("resp-missing")),
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_response_state: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("unknown response state should be refused");
        let payload =
            tokio::runtime::Runtime::new()?.block_on(response_json(error.into_response()))?;
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("unknown or expired")
        );
        Ok(())
    }

    #[test]
    fn generic_responses_refuse_instruction_changes_on_continuation()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-instruction-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny instruction llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let runtime = tokio::runtime::Runtime::new()?;
        let first_response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-instruction-llama")),
                instructions: Some(String::from("Be brief.")),
                conversation: None,
                input: ResponsesInput::Text(String::from("hello")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let first_payload = runtime.block_on(response_json(first_response))?;
        let error = runtime
            .block_on(handle_generic_responses(
                std::sync::Arc::clone(&server.state),
                ResponsesRequest {
                    model: None,
                    instructions: Some(String::from("Be verbose.")),
                    conversation: Some(
                        first_payload["conversation"]["id"]
                            .as_str()
                            .expect("conversation id")
                            .to_string(),
                    ),
                    input: ResponsesInput::Text(String::from("again")),
                    temperature: Some(0.0),
                    max_output_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    previous_response_id: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_response_state: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("instruction drift should be refused");
        let payload = runtime.block_on(response_json(error.into_response()))?;
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("cannot change `instructions`")
        );
        Ok(())
    }

    #[test]
    fn generic_responses_refuse_unsupported_continue_last_assistant()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-continue-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny continue llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let error = tokio::runtime::Runtime::new()?
            .block_on(handle_generic_responses(
                std::sync::Arc::clone(&server.state),
                ResponsesRequest {
                    model: Some(String::from("tiny-continue-llama")),
                    instructions: None,
                    conversation: None,
                    input: ResponsesInput::Text(String::from("hello")),
                    temperature: Some(0.0),
                    max_output_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    previous_response_id: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_response_state: Some(PsionicResponseStateRequest {
                        store: true,
                        continuation: ResponseContinuationMode::ContinueLastAssistant,
                        invalidate_references: false,
                    }),
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("continue_last_assistant should be refused on the current runtime");
        let payload =
            tokio::runtime::Runtime::new()?.block_on(response_json(error.into_response()))?;
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("continue_last_assistant")
        );
        Ok(())
    }

    #[test]
    fn generic_server_refuses_model_endpoint_mismatches() -> Result<(), Box<dyn std::error::Error>>
    {
        let temp = tempfile::tempdir()?;
        let llama_path = temp.path().join("tiny-llama.gguf");
        let embeddings_path = temp.path().join("tiny-embed.safetensors");
        write_test_gguf(
            &llama_path,
            dense_llama_metadata("tiny server llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;
        ByteProjectionEmbedder::write_default_safetensors_artifact(&embeddings_path)?;

        let mut config = OpenAiCompatConfig::new(&llama_path);
        config.add_model_path(&embeddings_path);
        let server = OpenAiCompatServer::from_config(&config)?;

        let embeddings_response = tokio::runtime::Runtime::new()?.block_on(generic_embeddings(
            State(std::sync::Arc::clone(&server.state)),
            Json(EmbeddingsRequest {
                model: Some(String::from("tiny-llama")),
                input: EmbeddingsInput::One(String::from("hello")),
                dimensions: None,
                encoding_format: None,
            }),
        ));
        assert_eq!(embeddings_response.status(), StatusCode::BAD_REQUEST);
        let embeddings_payload =
            tokio::runtime::Runtime::new()?.block_on(response_json(embeddings_response))?;
        assert!(
            embeddings_payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("/v1/chat/completions"),
            "unsupported endpoint error should describe supported surfaces"
        );

        let responses_response = tokio::runtime::Runtime::new()?
            .block_on(handle_generic_responses(
                std::sync::Arc::clone(&server.state),
                ResponsesRequest {
                    model: Some(String::from("tiny-embed")),
                    instructions: None,
                    conversation: None,
                    input: ResponsesInput::Text(String::from("hello")),
                    temperature: Some(0.0),
                    max_output_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    previous_response_id: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_response_state: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))
            .expect_err("embeddings-only model should refuse responses");
        let response = responses_response.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("/v1/embeddings"),
            "unsupported endpoint error should describe supported surfaces"
        );
        Ok(())
    }

    #[test]
    fn generic_server_grammar_fallback_is_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny grammar llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let request = ChatCompletionRequest {
            model: Some(String::from("tiny-llama")),
            messages: vec![ChatCompletionMessage::text("user", "hello")],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            psionic_grammar: Some(PsionicGrammarRequest {
                grammar: String::from("root ::= \"psionic\"\n"),
                syntax: Some(StructuredGrammarSyntax::Gbnf),
            }),
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        };

        let response = tokio::runtime::Runtime::new()?.block_on(
            handle_generic_chat_completions(std::sync::Arc::clone(&server.state), request),
        )?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-mode"),
            Some(String::from("fallback_grammar"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-batch-posture"),
            Some(String::from("continuous_batch"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-scheduling-class"),
            Some(String::from("mixed_prefill_decode"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-prefill-decode-mode"),
            Some(String::from("disaggregated_colocated"))
        );
        assert!(
            header_value(response.headers(), "x-psionic-ttft-ns")
                .is_some_and(|value| !value.is_empty()),
            "TTFT header should be surfaced when measured"
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-parser"),
            Some(String::from("gbnf_subset"))
        );
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(payload["choices"][0]["message"]["content"], "psionic");
        assert_eq!(
            payload["psionic_structured_output"]["mode"],
            serde_json::json!("fallback_grammar")
        );
        assert_eq!(
            payload["psionic_structured_output"]["kind"],
            serde_json::json!("grammar")
        );
        assert_eq!(
            payload["psionic_structured_output"]["parser"],
            serde_json::json!("gbnf_subset")
        );
        assert_eq!(
            payload["psionic_structured_value"],
            serde_json::json!({
                "kind": "grammar",
                "value": "psionic"
            })
        );
        Ok(())
    }

    #[test]
    fn generic_server_json_schema_fallback_is_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-json-llama.gguf");
        write_test_gguf(
            &path,
            json_llama_metadata("tiny json llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 8, 3, 6).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let request = ChatCompletionRequest {
            model: Some(String::from("tiny-json-llama")),
            messages: vec![ChatCompletionMessage::text("user", "hello")],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            response_format: Some(ChatCompletionResponseFormatRequest {
                kind: String::from("json_schema"),
                json_schema: Some(ChatCompletionJsonSchemaRequest {
                    name: Some(String::from("ok_object")),
                    schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "ok": { "type": "boolean" }
                        },
                        "required": ["ok"],
                        "additionalProperties": false
                    }),
                    strict: Some(true),
                }),
                schema: None,
            }),
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        };

        let response = tokio::runtime::Runtime::new()?.block_on(
            handle_generic_chat_completions(std::sync::Arc::clone(&server.state), request),
        )?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-mode"),
            Some(String::from("fallback_json_schema"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-batch-posture"),
            Some(String::from("continuous_batch"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-scheduling-class"),
            Some(String::from("mixed_prefill_decode"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-prefill-decode-mode"),
            Some(String::from("disaggregated_colocated"))
        );
        assert!(
            header_value(response.headers(), "x-psionic-ttft-ns")
                .is_some_and(|value| !value.is_empty()),
            "TTFT header should be surfaced when measured"
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-parser"),
            Some(String::from("json_schema_subset"))
        );
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(payload["choices"][0]["message"]["content"], "{\"ok\":true}");
        assert_eq!(
            payload["psionic_structured_output"]["mode"],
            serde_json::json!("fallback_json_schema")
        );
        assert_eq!(
            payload["psionic_structured_output"]["kind"],
            serde_json::json!("json_schema")
        );
        assert_eq!(
            payload["psionic_structured_output"]["schema_name"],
            serde_json::json!("ok_object")
        );
        assert_eq!(
            payload["psionic_structured_value"],
            serde_json::json!({
                "kind": "json",
                "value": { "ok": true }
            })
        );
        Ok(())
    }

    #[test]
    fn generic_server_choice_structured_output_is_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-choice-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny choice llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let request = ChatCompletionRequest {
            model: Some(String::from("tiny-choice-llama")),
            messages: vec![ChatCompletionMessage::text("user", "hello")],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: Some(StructuredOutputRequest::Choice {
                values: vec![String::from("world"), String::from("psionic")],
            }),
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        };

        let response = tokio::runtime::Runtime::new()?.block_on(
            handle_generic_chat_completions(std::sync::Arc::clone(&server.state), request),
        )?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-mode"),
            Some(String::from("fallback_choice"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-parser"),
            Some(String::from("choice_set"))
        );
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("world")
        );
        assert_eq!(
            payload["psionic_structured_output"]["kind"],
            serde_json::json!("choice")
        );
        assert_eq!(
            payload["psionic_structured_value"],
            serde_json::json!({
                "kind": "choice",
                "value": "world"
            })
        );
        Ok(())
    }

    #[test]
    fn generic_responses_regex_structured_output_is_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-regex-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny regex llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response = tokio::runtime::Runtime::new()?.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-regex-llama")),
                instructions: None,
                conversation: None,
                input: ResponsesInput::Text(String::from("hello")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: Some(StructuredOutputRequest::Regex {
                    pattern: String::from("w[a-z]{4}"),
                }),
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-mode"),
            Some(String::from("fallback_regex"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-parser"),
            Some(String::from("regex_subset"))
        );
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(payload["output_text"], serde_json::json!("world"));
        assert_eq!(
            payload["psionic_structured_output"]["kind"],
            serde_json::json!("regex")
        );
        assert_eq!(
            payload["psionic_structured_value"],
            serde_json::json!({
                "kind": "regex",
                "value": "world"
            })
        );
        Ok(())
    }

    #[test]
    fn generic_server_tagged_structure_survives_as_machine_value()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tagged-llama.gguf");
        write_test_gguf(
            &path,
            tagged_llama_metadata("tiny tagged llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 6, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let request = ChatCompletionRequest {
            model: Some(String::from("tiny-tagged-llama")),
            messages: vec![ChatCompletionMessage::text("user", "hello")],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            response_format: None,
            psionic_grammar: None,
            psionic_structured_output: Some(StructuredOutputRequest::TaggedStructure {
                name: Some(String::from("decision")),
                discriminator: String::from("kind"),
                variants: vec![StructuredTaggedVariant {
                    tag: String::from("approve"),
                    schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "reason": { "type": "string", "minLength": 1 }
                        },
                        "required": ["reason"],
                        "additionalProperties": false
                    }),
                }],
            }),
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        };

        let response = tokio::runtime::Runtime::new()?.block_on(
            handle_generic_chat_completions(std::sync::Arc::clone(&server.state), request),
        )?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-mode"),
            Some(String::from("fallback_tagged_structure"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-structured-output-parser"),
            Some(String::from("tagged_json_schema"))
        );
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("{\"kind\":\"approve\",\"reason\":\"ok\"}")
        );
        assert_eq!(
            payload["psionic_structured_output"]["kind"],
            serde_json::json!("tagged_structure")
        );
        assert_eq!(
            payload["psionic_structured_value"],
            serde_json::json!({
                "kind": "tagged_structure",
                "discriminator": "kind",
                "tag": "approve",
                "value": {
                    "kind": "approve",
                    "reason": "ok"
                }
            })
        );
        Ok(())
    }

    #[test]
    fn generic_server_tool_choice_none_preserves_plain_text_generation()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tool-none-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny tool none llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response =
            tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-tool-none-llama")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: vec![weather_tool_definition()],
                    tool_choice: Some(ToolChoiceRequest::Mode(String::from("none"))),
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))?;
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("world")
        );
        assert!(payload["choices"][0]["message"]["tool_calls"].is_null());
        assert!(payload["psionic_tool_calls"].is_null());
        Ok(())
    }

    #[test]
    fn generic_server_tool_choice_auto_can_return_message_envelope()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tool-auto-llama.gguf");
        write_test_gguf(
            &path,
            auto_tool_message_llama_metadata("tiny tool auto llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 6, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response =
            tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-tool-auto-llama")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: vec![weather_tool_definition()],
                    tool_choice: Some(ToolChoiceRequest::Mode(String::from("auto"))),
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))?;
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["message"]["content"],
            serde_json::json!("world")
        );
        assert_eq!(
            payload["psionic_structured_value"],
            serde_json::json!({
                "kind": "tagged_structure",
                "discriminator": "kind",
                "tag": "message",
                "value": {
                    "kind": "message",
                    "content": "world"
                }
            })
        );
        Ok(())
    }

    #[test]
    fn generic_server_required_tool_call_is_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tool-call-llama.gguf");
        write_test_gguf(
            &path,
            tool_call_llama_metadata("tiny tool call llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 6, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response =
            tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-tool-call-llama")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: vec![weather_tool_definition()],
                    tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))?;
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["finish_reason"],
            serde_json::json!("tool_calls")
        );
        assert!(payload["choices"][0]["message"]["content"].is_null());
        assert_eq!(
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            payload["psionic_tool_calls"][0]["arguments"],
            serde_json::json!({
                "latitude": 48.8566,
                "longitude": 2.3522
            })
        );
        Ok(())
    }

    #[test]
    fn generic_server_parallel_tool_calls_surface_ordered_batch()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tool-batch-llama.gguf");
        write_test_gguf(
            &path,
            multi_tool_call_llama_metadata("tiny tool batch llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 6, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response =
            tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-tool-batch-llama")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: vec![weather_tool_definition(), time_tool_definition()],
                    tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                    parallel_tool_calls: Some(true),
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))?;
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(
            payload["choices"][0]["finish_reason"],
            serde_json::json!("tool_calls")
        );
        assert_eq!(
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            payload["choices"][0]["message"]["tool_calls"][1]["function"]["name"],
            serde_json::json!("get_time")
        );
        assert!(
            payload["choices"][0]["message"]["tool_calls"][0]["id"]
                .as_str()
                .is_some_and(|id| id.ends_with("-tool-0"))
        );
        assert!(
            payload["choices"][0]["message"]["tool_calls"][1]["id"]
                .as_str()
                .is_some_and(|id| id.ends_with("-tool-1"))
        );
        assert_eq!(
            payload["psionic_tool_calls"][0]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            payload["psionic_tool_calls"][1]["name"],
            serde_json::json!("get_time")
        );
        Ok(())
    }

    #[test]
    fn generic_responses_named_tool_choice_surfaces_tool_call()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tool-response-llama.gguf");
        write_test_gguf(
            &path,
            tool_call_llama_metadata("tiny tool response llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 6, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response = tokio::runtime::Runtime::new()?.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-tool-response-llama")),
                instructions: None,
                conversation: None,
                input: ResponsesInput::Text(String::from("hello")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: vec![weather_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Named(NamedToolChoiceRequest {
                    kind: String::from("function"),
                    function: NamedToolChoiceFunction {
                        name: String::from("get_weather"),
                    },
                })),
                parallel_tool_calls: Some(true),
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert_eq!(payload["output_text"], serde_json::json!(""));
        assert!(
            payload["output"]
                .as_array()
                .is_some_and(|items| items.is_empty())
        );
        assert_eq!(
            payload["psionic_tool_calls"][0]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            payload["psionic_tool_calls"].as_array().map(Vec::len),
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn generic_server_tool_call_validation_refuses_invalid_arguments()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tool-invalid-llama.gguf");
        write_test_gguf(
            &path,
            invalid_tool_call_llama_metadata("tiny tool invalid llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 6, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response = tokio::runtime::Runtime::new()?.block_on(super::generic_chat_completions(
            State(std::sync::Arc::clone(&server.state)),
            Json(ChatCompletionRequest {
                model: Some(String::from("tiny-tool-invalid-llama")),
                messages: vec![ChatCompletionMessage::text("user", "hello")],
                temperature: Some(0.0),
                max_tokens: Some(1),
                stop: None,
                stream: false,
                tools: vec![weather_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                parallel_tool_calls: Some(false),
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: None,
                ..Default::default()
            }),
        ));
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("structured output fallback could not find a valid continuation"),
            "validation failures should surface through parser-backed refusal"
        );
        Ok(())
    }

    #[test]
    fn generic_server_streaming_tool_calls_preserve_machine_envelope()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tool-stream-llama.gguf");
        write_test_gguf(
            &path,
            tool_call_llama_metadata("tiny tool stream llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 6, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response =
            tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-tool-stream-llama")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: true,
                    tools: vec![weather_tool_definition()],
                    tool_choice: Some(ToolChoiceRequest::Mode(String::from("required"))),
                    parallel_tool_calls: Some(false),
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))?;
        let body = tokio::runtime::Runtime::new()?.block_on(response_text(response))?;
        let events = sse_json_events(body.as_str())?;
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][0]["index"],
            serde_json::json!(0)
        );
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            serde_json::json!("{\"latitude\":48.8566,\"longitude\":2.3522}")
        );
        assert_eq!(
            events[1]["choices"][0]["finish_reason"],
            serde_json::json!("tool_calls")
        );
        assert!(body.contains("[DONE]"));
        Ok(())
    }

    #[test]
    fn generic_server_router_tool_loop_boundary_executes_multi_step_flow()
    -> Result<(), Box<dyn std::error::Error>> {
        struct ScriptedServeToolLoopRunner {
            turns: Vec<(Option<String>, Vec<ResolvedToolCall>)>,
            observed_history_lens: Vec<usize>,
        }

        impl ToolLoopModelRunner for ScriptedServeToolLoopRunner {
            fn run_turn(
                &mut self,
                request: psionic_router::ToolLoopTurnRequest,
            ) -> Result<ToolLoopModelTurn, ToolLoopError> {
                self.observed_history_lens
                    .push(request.prompt_history.len());
                let (content, tool_calls) =
                    self.turns.get(request.step_index).cloned().ok_or_else(|| {
                        ToolLoopError::Execution(String::from("missing scripted turn"))
                    })?;
                Ok(ToolLoopModelTurn {
                    assistant_message: assistant_prompt_message_for_tool_loop(content),
                    tool_calls: tool_calls
                        .into_iter()
                        .map(tool_loop_tool_call_from_resolved)
                        .collect(),
                })
            }
        }

        struct ScriptedToolExecutor {
            descriptor: ToolProviderDescriptor,
            observed_history_lens: std::sync::Mutex<Vec<usize>>,
        }

        impl ToolLoopToolExecutor for ScriptedToolExecutor {
            fn descriptor(&self) -> &ToolProviderDescriptor {
                &self.descriptor
            }

            fn execute(
                &self,
                request: ToolExecutionRequest,
            ) -> Result<ToolLoopToolResult, ToolLoopError> {
                self.observed_history_lens
                    .lock()
                    .expect("history mutex")
                    .push(request.prompt_history.len());
                Ok(ToolLoopToolResult {
                    tool_call_id: request.tool_call.id,
                    tool_name: request.tool_call.name.clone(),
                    provider: self.descriptor.clone(),
                    visibility: self.descriptor.result_visibility,
                    message: tool_result_prompt_message(
                        request.tool_call.name.as_str(),
                        "72f and sunny",
                    ),
                    structured: Some(serde_json::json!({
                        "forecast": "sunny",
                        "temperature_f": 72
                    })),
                })
            }
        }

        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tool-loop-llama.gguf");
        write_test_gguf(
            &path,
            tool_call_llama_metadata("tiny tool loop llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 6, 3, 4).as_slice(),
        )?;
        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let mut gateway = ToolGateway::new();
        gateway.register(
            "get_weather",
            ScriptedToolExecutor {
                descriptor: ToolProviderDescriptor::mcp("weather-provider", "weather", "sse")
                    .with_history_visibility(ToolHistoryVisibility::PromptHistory)
                    .with_result_visibility(ToolResultVisibility::InjectIntoModel),
                observed_history_lens: std::sync::Mutex::new(Vec::new()),
            },
        );
        let controller = ToolLoopController::new(&server.state.router, &gateway);
        let mut runner = ScriptedServeToolLoopRunner {
            turns: vec![
                (
                    Some(String::from("Calling weather tool")),
                    vec![ResolvedToolCall {
                        id: String::from("tool-0"),
                        name: String::from("get_weather"),
                        arguments: serde_json::json!({"city": "Paris"}),
                    }],
                ),
                (Some(String::from("Paris is 72F and sunny.")), Vec::new()),
            ],
            observed_history_lens: Vec::new(),
        };
        let outcome = controller.run(
            ToolLoopRequest::new(
                RoutingRequest::new(RoutingEndpoint::Responses).require_tool_calling(),
                vec![PromptMessage::new(
                    PromptMessageRole::User,
                    "How is the weather?",
                )],
            ),
            &mut runner,
        )?;

        assert_eq!(outcome.steps.len(), 2);
        assert_eq!(
            outcome
                .final_message
                .as_ref()
                .map(|message| message.content.as_str()),
            Some("Paris is 72F and sunny.")
        );
        assert_eq!(runner.observed_history_lens, vec![1, 4]);
        assert!(matches!(
            outcome.steps[0].tool_results[0].provider.interface,
            psionic_router::ToolProviderInterface::Mcp { .. }
        ));
        assert_eq!(
            outcome.steps[0].tool_results[0]
                .message
                .author_name
                .as_deref(),
            Some("get_weather")
        );
        Ok(())
    }

    #[test]
    fn generic_server_router_tool_loop_boundary_replays_parallel_tool_results_in_order()
    -> Result<(), Box<dyn std::error::Error>> {
        struct ScriptedServeToolLoopRunner {
            turns: Vec<(Option<String>, Vec<ResolvedToolCall>)>,
        }

        impl ToolLoopModelRunner for ScriptedServeToolLoopRunner {
            fn run_turn(
                &mut self,
                request: psionic_router::ToolLoopTurnRequest,
            ) -> Result<ToolLoopModelTurn, ToolLoopError> {
                let (content, tool_calls) =
                    self.turns.get(request.step_index).cloned().ok_or_else(|| {
                        ToolLoopError::Execution(String::from("missing scripted turn"))
                    })?;
                Ok(ToolLoopModelTurn {
                    assistant_message: assistant_prompt_message_for_tool_loop(content),
                    tool_calls: tool_calls
                        .into_iter()
                        .map(tool_loop_tool_call_from_resolved)
                        .collect(),
                })
            }
        }

        struct ScriptedToolExecutor {
            descriptor: ToolProviderDescriptor,
            observed_tool_call_ids: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
            result_text: &'static str,
        }

        impl ToolLoopToolExecutor for ScriptedToolExecutor {
            fn descriptor(&self) -> &ToolProviderDescriptor {
                &self.descriptor
            }

            fn execute(
                &self,
                request: ToolExecutionRequest,
            ) -> Result<ToolLoopToolResult, ToolLoopError> {
                self.observed_tool_call_ids
                    .lock()
                    .expect("tool call id mutex")
                    .push(request.tool_call.id.clone());
                Ok(ToolLoopToolResult {
                    tool_call_id: request.tool_call.id,
                    tool_name: request.tool_call.name.clone(),
                    provider: self.descriptor.clone(),
                    visibility: self.descriptor.result_visibility,
                    message: tool_result_prompt_message(
                        request.tool_call.name.as_str(),
                        self.result_text,
                    ),
                    structured: None,
                })
            }
        }

        let observed_tool_call_ids = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut gateway = ToolGateway::new();
        gateway.register(
            "get_weather",
            ScriptedToolExecutor {
                descriptor: ToolProviderDescriptor::mcp("weather-provider", "weather", "sse")
                    .with_history_visibility(ToolHistoryVisibility::PromptHistory)
                    .with_result_visibility(ToolResultVisibility::InjectIntoModel),
                observed_tool_call_ids: std::sync::Arc::clone(&observed_tool_call_ids),
                result_text: "72f and sunny",
            },
        );
        gateway.register(
            "get_time",
            ScriptedToolExecutor {
                descriptor: ToolProviderDescriptor::mcp("clock-provider", "clock", "sse")
                    .with_history_visibility(ToolHistoryVisibility::PromptHistory)
                    .with_result_visibility(ToolResultVisibility::InjectIntoModel),
                observed_tool_call_ids: std::sync::Arc::clone(&observed_tool_call_ids),
                result_text: "13:00 UTC",
            },
        );

        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-tool-loop-batch-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny tool loop batch llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;
        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let controller = ToolLoopController::new(&server.state.router, &gateway);
        let mut runner = ScriptedServeToolLoopRunner {
            turns: vec![
                (
                    Some(String::from("Calling weather and time tools")),
                    vec![
                        ResolvedToolCall {
                            id: String::from("tool-0"),
                            name: String::from("get_weather"),
                            arguments: serde_json::json!({"city": "Paris"}),
                        },
                        ResolvedToolCall {
                            id: String::from("tool-1"),
                            name: String::from("get_time"),
                            arguments: serde_json::json!({"timezone": "UTC"}),
                        },
                    ],
                ),
                (
                    Some(String::from("Paris is sunny and it is 13:00 UTC.")),
                    Vec::new(),
                ),
            ],
        };
        let outcome = controller.run(
            ToolLoopRequest::new(
                RoutingRequest::new(RoutingEndpoint::Responses).require_tool_calling(),
                vec![PromptMessage::new(
                    PromptMessageRole::User,
                    "What is the weather in Paris and the current UTC time?",
                )],
            ),
            &mut runner,
        )?;

        assert_eq!(outcome.steps.len(), 2);
        assert_eq!(
            outcome
                .final_message
                .as_ref()
                .map(|message| message.content.as_str()),
            Some("Paris is sunny and it is 13:00 UTC.")
        );
        assert_eq!(outcome.steps[0].tool_results.len(), 2);
        assert_eq!(
            outcome.steps[0]
                .tool_results
                .iter()
                .map(|result| result.tool_call_id.as_str())
                .collect::<Vec<_>>(),
            vec!["tool-0", "tool-1"]
        );
        assert_eq!(
            outcome.steps[0]
                .tool_results
                .iter()
                .map(|result| result.tool_name.as_str())
                .collect::<Vec<_>>(),
            vec!["get_weather", "get_time"]
        );
        assert_eq!(
            observed_tool_call_ids
                .lock()
                .expect("tool call ids should be readable")
                .as_slice(),
            ["tool-0", "tool-1"]
        );
        Ok(())
    }

    #[test]
    fn generic_server_weather_agent_pilot_is_end_to_end_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        struct ScriptedServeToolLoopRunner {
            turns: Vec<(Option<String>, Vec<ResolvedToolCall>)>,
            observed_history_lens: Vec<usize>,
        }

        impl ToolLoopModelRunner for ScriptedServeToolLoopRunner {
            fn run_turn(
                &mut self,
                request: psionic_router::ToolLoopTurnRequest,
            ) -> Result<ToolLoopModelTurn, ToolLoopError> {
                self.observed_history_lens
                    .push(request.prompt_history.len());
                let (content, tool_calls) =
                    self.turns.get(request.step_index).cloned().ok_or_else(|| {
                        ToolLoopError::Execution(String::from("missing scripted turn"))
                    })?;
                Ok(ToolLoopModelTurn {
                    assistant_message: assistant_prompt_message_for_tool_loop(content),
                    tool_calls: tool_calls
                        .into_iter()
                        .map(tool_loop_tool_call_from_resolved)
                        .collect(),
                })
            }
        }

        struct ScriptedToolExecutor {
            descriptor: ToolProviderDescriptor,
        }

        impl ToolLoopToolExecutor for ScriptedToolExecutor {
            fn descriptor(&self) -> &ToolProviderDescriptor {
                &self.descriptor
            }

            fn execute(
                &self,
                request: ToolExecutionRequest,
            ) -> Result<ToolLoopToolResult, ToolLoopError> {
                Ok(ToolLoopToolResult {
                    tool_call_id: request.tool_call.id,
                    tool_name: request.tool_call.name.clone(),
                    provider: self.descriptor.clone(),
                    visibility: self.descriptor.result_visibility,
                    message: tool_result_prompt_message(
                        request.tool_call.name.as_str(),
                        "72f and sunny",
                    ),
                    structured: Some(serde_json::json!({
                        "forecast": "sunny",
                        "temperature_f": 72
                    })),
                })
            }
        }

        let temp = tempfile::tempdir()?;
        let tool_path = temp.path().join("tiny-agent-tool-llama.gguf");
        let structured_path = temp.path().join("tiny-agent-structured-llama.gguf");
        write_test_gguf(
            &tool_path,
            tool_call_llama_metadata("tiny agent tool llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 6, 3, 4).as_slice(),
        )?;
        write_test_gguf(
            &structured_path,
            json_llama_metadata("tiny agent structured llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 8, 3, 6).as_slice(),
        )?;

        let mut config = OpenAiCompatConfig::new(&tool_path);
        config.add_model_path(&structured_path);
        let server = OpenAiCompatServer::from_config(&config)?;
        let runtime = tokio::runtime::Runtime::new()?;
        let tenant = String::from("agent-pilot");

        let build_structured_request = |prompt: &str, tenant_id: &str| ChatCompletionRequest {
            model: Some(String::from("tiny-agent-structured-llama")),
            messages: vec![ChatCompletionMessage::text("user", prompt)],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            response_format: Some(ChatCompletionResponseFormatRequest {
                kind: String::from("json_schema"),
                json_schema: Some(ChatCompletionJsonSchemaRequest {
                    name: Some(String::from("weather_summary")),
                    schema: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "ok": { "type": "boolean" }
                        },
                        "required": ["ok"],
                        "additionalProperties": false
                    }),
                    strict: Some(true),
                }),
                schema: None,
            }),
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: Some(PrefixCacheControl {
                mode: PrefixCacheMode::Auto,
                tenant_id: Some(String::from(tenant_id)),
            }),
            ..Default::default()
        };

        let seeded_summary = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            build_structured_request("Paris weather tomorrow", tenant.as_str()),
        ))?;
        assert_eq!(
            header_value(seeded_summary.headers(), "x-psionic-structured-output-mode"),
            Some(String::from("fallback_json_schema"))
        );
        assert_eq!(
            header_value(seeded_summary.headers(), "x-psionic-route-worker"),
            Some(String::from(super::OPENAI_COMPAT_WORKER_ID))
        );
        assert_eq!(
            header_value(seeded_summary.headers(), "x-psionic-route-strategy"),
            Some(String::from("warm_aware"))
        );
        assert_eq!(
            header_value(seeded_summary.headers(), "x-psionic-prefix-cache-state"),
            Some(String::from("none"))
        );
        let seeded_payload = runtime.block_on(response_json(seeded_summary))?;
        assert_eq!(
            seeded_payload["psionic_structured_output"]["kind"],
            serde_json::json!("json_schema")
        );
        assert_eq!(
            seeded_payload["psionic_structured_value"]["kind"],
            serde_json::json!("json")
        );
        assert_eq!(
            seeded_payload["psionic_structured_output"]["schema_name"],
            serde_json::json!("weather_summary")
        );
        assert!(
            seeded_payload["psionic_structured_value"]["value"]["ok"].is_boolean(),
            "weather summary should remain machine-checkable JSON"
        );

        let cached_summary = runtime.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            build_structured_request("Paris weather", tenant.as_str()),
        ))?;
        assert_eq!(
            header_value(cached_summary.headers(), "x-psionic-prefix-cache-state"),
            Some(String::from("hit"))
        );
        assert!(
            header_value(
                cached_summary.headers(),
                "x-psionic-prefix-cache-reused-tokens"
            )
            .is_some_and(|value| value != "0"),
            "cached summary should reuse at least one prompt token"
        );

        let first_response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: Some(String::from("tiny-agent-tool-llama")),
                instructions: None,
                conversation: None,
                input: ResponsesInput::Text(String::from("What's the weather in Paris?")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: vec![weather_tool_definition()],
                tool_choice: Some(ToolChoiceRequest::Named(NamedToolChoiceRequest {
                    kind: String::from("function"),
                    function: NamedToolChoiceFunction {
                        name: String::from("get_weather"),
                    },
                })),
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let first_payload = runtime.block_on(response_json(first_response))?;
        assert_eq!(
            first_payload["psionic_tool_calls"][0]["name"],
            serde_json::json!("get_weather")
        );
        assert_eq!(
            first_payload["psionic_response_state"]["stored"],
            serde_json::json!(true)
        );
        let first_response_id = first_payload["id"]
            .as_str()
            .expect("first response id")
            .to_string();
        let conversation_id = first_payload["conversation"]["id"]
            .as_str()
            .expect("conversation id")
            .to_string();

        let mut gateway = ToolGateway::new();
        gateway.register(
            "get_weather",
            ScriptedToolExecutor {
                descriptor: ToolProviderDescriptor::mcp("weather-provider", "weather", "sse")
                    .with_history_visibility(ToolHistoryVisibility::PromptHistory)
                    .with_result_visibility(ToolResultVisibility::InjectIntoModel),
            },
        );
        let controller = ToolLoopController::new(&server.state.router, &gateway);
        let mut runner = ScriptedServeToolLoopRunner {
            turns: vec![
                (
                    Some(String::from("Calling weather tool")),
                    vec![ResolvedToolCall {
                        id: String::from("tool-0"),
                        name: String::from("get_weather"),
                        arguments: serde_json::json!({"city": "Paris"}),
                    }],
                ),
                (Some(String::from("Paris is 72F and sunny.")), Vec::new()),
            ],
            observed_history_lens: Vec::new(),
        };
        let outcome = controller.run(
            ToolLoopRequest::new(
                RoutingRequest::new(RoutingEndpoint::Responses).require_tool_calling(),
                vec![PromptMessage::new(
                    PromptMessageRole::User,
                    "What's the weather in Paris?",
                )],
            ),
            &mut runner,
        )?;
        assert_eq!(outcome.steps.len(), 2);
        assert_eq!(
            outcome
                .final_message
                .as_ref()
                .map(|message| message.content.as_str()),
            Some("Paris is 72F and sunny.")
        );
        assert_eq!(
            outcome.steps[0].route_selection.worker_id,
            super::OPENAI_COMPAT_WORKER_ID
        );
        assert_eq!(
            outcome.steps[0].tool_results[0].structured.as_ref(),
            Some(&serde_json::json!({
                "forecast": "sunny",
                "temperature_f": 72
            }))
        );

        let continued_response = runtime.block_on(handle_generic_responses(
            std::sync::Arc::clone(&server.state),
            ResponsesRequest {
                model: None,
                instructions: None,
                conversation: Some(conversation_id.clone()),
                input: ResponsesInput::Text(String::from("and tomorrow?")),
                temperature: Some(0.0),
                max_output_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                previous_response_id: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_response_state: None,
                psionic_prefix_cache: None,
                ..Default::default()
            },
        ))?;
        let continued_payload = runtime.block_on(response_json(continued_response))?;
        assert_eq!(
            continued_payload["previous_response_id"],
            serde_json::json!(first_response_id)
        );
        assert_eq!(
            continued_payload["conversation"]["id"],
            serde_json::json!(conversation_id)
        );
        assert_eq!(
            continued_payload["conversation"]["revision"],
            serde_json::json!(2)
        );
        assert!(
            continued_payload["psionic_response_state"]["replayed_prompt_messages"]
                .as_u64()
                .is_some_and(|count| count > 0)
        );
        Ok(())
    }

    #[test]
    fn generic_server_prefix_cache_headers_are_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-prefix-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny prefix llama").as_slice(),
            dense_decoder_tensors(false, 3, 5).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let tenant = String::from("tenant-a");
        let build_request =
            |prompt: &str, prefix_cache: PrefixCacheControl| ChatCompletionRequest {
                model: Some(String::from("tiny-prefix-llama")),
                messages: vec![ChatCompletionMessage::text("user", prompt)],
                temperature: Some(0.0),
                max_tokens: Some(1),
                stop: None,
                stream: false,
                tools: Vec::new(),
                tool_choice: None,
                response_format: None,
                psionic_grammar: None,
                psionic_structured_output: None,
                psionic_reasoning: None,
                psionic_prefix_cache: Some(prefix_cache),
                ..Default::default()
            };

        let seeded = tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            build_request(
                "hello world",
                PrefixCacheControl {
                    mode: PrefixCacheMode::Auto,
                    tenant_id: Some(tenant.clone()),
                },
            ),
        ))?;
        assert_eq!(
            header_value(seeded.headers(), "x-psionic-prefix-cache-state"),
            Some(String::from("none"))
        );
        assert_eq!(
            header_value(seeded.headers(), "x-psionic-prefix-cache-reused-tokens"),
            Some(String::from("0"))
        );

        let hit = tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
            std::sync::Arc::clone(&server.state),
            build_request(
                "hello",
                PrefixCacheControl {
                    mode: PrefixCacheMode::Auto,
                    tenant_id: Some(tenant.clone()),
                },
            ),
        ))?;
        assert_eq!(
            header_value(hit.headers(), "x-psionic-prefix-cache-state"),
            Some(String::from("hit"))
        );
        assert_eq!(
            header_value(hit.headers(), "x-psionic-prefix-cache-reused-tokens"),
            Some(String::from("1"))
        );
        assert_eq!(
            header_value(hit.headers(), "x-psionic-prefix-cache-refusal"),
            None
        );

        let bypassed =
            tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                build_request(
                    "hello",
                    PrefixCacheControl {
                        mode: PrefixCacheMode::Bypass,
                        tenant_id: Some(tenant),
                    },
                ),
            ))?;
        assert_eq!(
            header_value(bypassed.headers(), "x-psionic-prefix-cache-state"),
            Some(String::from("bypassed"))
        );
        assert_eq!(
            header_value(bypassed.headers(), "x-psionic-prefix-cache-refusal"),
            Some(String::from("request_opt_out"))
        );
        assert_eq!(
            header_value(bypassed.headers(), "x-psionic-prefix-cache-reused-tokens"),
            Some(String::from("0"))
        );
        Ok(())
    }

    #[test]
    fn generic_server_route_headers_are_machine_checkable() -> Result<(), Box<dyn std::error::Error>>
    {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-route-llama.gguf");
        write_test_gguf(
            &path,
            dense_llama_metadata("tiny route llama").as_slice(),
            dense_decoder_tensors(false, 3, 4).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let response =
            tokio::runtime::Runtime::new()?.block_on(handle_generic_chat_completions(
                std::sync::Arc::clone(&server.state),
                ChatCompletionRequest {
                    model: Some(String::from("tiny-route-llama")),
                    messages: vec![ChatCompletionMessage::text("user", "hello")],
                    temperature: Some(0.0),
                    max_tokens: Some(1),
                    stop: None,
                    stream: false,
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: None,
                    psionic_grammar: None,
                    psionic_structured_output: None,
                    psionic_reasoning: None,
                    psionic_prefix_cache: None,
                    ..Default::default()
                },
            ))?;
        assert_eq!(
            header_value(response.headers(), "x-psionic-route-worker"),
            Some(String::from(super::OPENAI_COMPAT_WORKER_ID))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-route-strategy"),
            Some(String::from("warm_aware"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-route-eligible-workers"),
            Some(String::from("1"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-route-warm-workers"),
            Some(String::from("1"))
        );
        assert_eq!(
            header_value(response.headers(), "x-psionic-route-cache-matches"),
            Some(String::from("0"))
        );
        Ok(())
    }

    #[test]
    fn generic_server_refuses_unsupported_json_schema_features()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("tiny-json-llama.gguf");
        write_test_gguf(
            &path,
            json_llama_metadata("tiny json llama").as_slice(),
            dense_decoder_tensors_with_vocab(false, 8, 3, 6).as_slice(),
        )?;

        let server = OpenAiCompatServer::from_config(&OpenAiCompatConfig::new(&path))?;
        let request = ChatCompletionRequest {
            model: Some(String::from("tiny-json-llama")),
            messages: vec![ChatCompletionMessage::text("user", "hello")],
            temperature: Some(0.0),
            max_tokens: Some(1),
            stop: None,
            stream: false,
            tools: Vec::new(),
            tool_choice: None,
            response_format: Some(ChatCompletionResponseFormatRequest {
                kind: String::from("json_schema"),
                json_schema: Some(ChatCompletionJsonSchemaRequest {
                    name: None,
                    schema: serde_json::json!({
                        "type": "string",
                        "format": "uuid"
                    }),
                    strict: Some(true),
                }),
                schema: None,
            }),
            psionic_grammar: None,
            psionic_structured_output: None,
            psionic_reasoning: None,
            psionic_prefix_cache: None,
            ..Default::default()
        };

        let response = tokio::runtime::Runtime::new()?.block_on(super::generic_chat_completions(
            State(std::sync::Arc::clone(&server.state)),
            Json(request),
        ));
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let payload = tokio::runtime::Runtime::new()?.block_on(response_json(response))?;
        assert!(
            payload["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("format"),
            "unsupported schema feature should be reported explicitly"
        );
        Ok(())
    }

    async fn response_json(
        response: Response,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let body = to_bytes(response.into_body(), usize::MAX).await?;
        Ok(serde_json::from_slice(body.as_ref())?)
    }

    async fn response_text(response: Response) -> Result<String, Box<dyn std::error::Error>> {
        let body = to_bytes(response.into_body(), usize::MAX).await?;
        Ok(String::from_utf8(body.to_vec())?)
    }

    fn sse_json_events(body: &str) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
            .map(|line| Ok(serde_json::from_str(line)?))
            .collect()
    }

    #[test]
    fn chat_completion_response_serializes_psion_claim_posture()
    -> Result<(), Box<dyn std::error::Error>> {
        let psion_claim_posture: crate::PsionServedOutputClaimPosture = serde_json::from_str(
            include_str!("../../../fixtures/psion/serve/psion_served_output_claim_direct_v1.json"),
        )?;
        let payload = serde_json::to_value(super::ChatCompletionResponse {
            id: String::from("chatcmpl-test"),
            object: "chat.completion",
            created: 0,
            model: String::from("tiny-gpt-oss"),
            choices: vec![super::ChatCompletionChoice {
                index: 0,
                message: super::ChatCompletionResponseMessage {
                    role: "assistant",
                    content: Some(String::from("ok")),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: "stop",
            }],
            usage: super::ChatCompletionUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            psionic_metrics: None,
            psionic_harmony: None,
            psionic_reasoning: None,
            psionic_perf: None,
            psionic_output_text: Some(String::from("ok")),
            psionic_output_tokens: Some(vec![1]),
            psionic_structured_output: None,
            psionic_structured_value: None,
            psionic_tool_calls: None,
            psionic_claim_posture: Some(psion_claim_posture),
            psionic_scheduler: None,
        })?;

        assert_eq!(
            payload["psionic_claim_posture"]["posture_id"],
            serde_json::json!("psion-served-output-claim-direct-v1")
        );
        assert_eq!(
            payload["psionic_claim_posture"]["visible_claims"]["benchmark_backing_visible"],
            serde_json::json!(true)
        );
        Ok(())
    }

    fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(String::from)
    }

    struct ScopedEnvVar {
        key: &'static str,
        previous: Option<String>,
    }

    impl ScopedEnvVar {
        fn set(key: &'static str, value: &str) -> Self {
            Self::set_optional(key, Some(value))
        }

        fn set_optional(key: &'static str, value: Option<&str>) -> Self {
            let previous = std::env::var(key).ok();
            // Safety: test env overrides are serialized behind a per-suite mutex and restored
            // before that lock is released.
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
            Self { key, previous }
        }
    }

    impl Drop for ScopedEnvVar {
        fn drop(&mut self) {
            if let Some(previous) = self.previous.as_deref() {
                // Safety: this restores the serialized test-local env override established in `set`.
                unsafe {
                    std::env::set_var(self.key, previous);
                }
            } else {
                // Safety: this clears the serialized test-local env override established in `set`.
                unsafe {
                    std::env::remove_var(self.key);
                }
            }
        }
    }

    fn qwen35_proxy_test_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    fn bootstrap_proxy_test_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    async fn start_qwen35_proxy_test_server() -> Result<
        (
            String,
            tokio::sync::oneshot::Sender<()>,
            std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
        ),
        Box<dyn std::error::Error>,
    > {
        let observed_requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let observed_for_route = std::sync::Arc::clone(&observed_requests);
        let router = axum::Router::new()
            .route("/health", axum::routing::get(|| async { StatusCode::OK }))
            .route(
                "/completion",
                axum::routing::post(move |Json(body): Json<serde_json::Value>| {
                    let observed_requests = std::sync::Arc::clone(&observed_for_route);
                    async move {
                        observed_requests
                            .lock()
                            .expect("observed qwen35 proxy requests should not be poisoned")
                            .push(body);
                        Json(serde_json::json!({
                            "content": "proxy world",
                            "tokens": [7, 8],
                            "stop_type": "eos",
                            "truncated": false,
                            "tokens_evaluated": 3
                        }))
                    }
                }),
            );
        tokio::spawn(async move {
            let _ = axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        Ok((format!("http://{address}"), shutdown_tx, observed_requests))
    }

    async fn start_openai_compat_test_server(
        server: OpenAiCompatServer,
    ) -> Result<(String, tokio::sync::oneshot::Sender<()>), Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let base_url = format!("http://{address}");
        let router = server.router();
        tokio::spawn(async move {
            let _ = axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await;
        });
        let client = reqwest::Client::new();
        for _ in 0..50 {
            if client
                .get(format!("{base_url}/health"))
                .send()
                .await
                .is_ok()
            {
                return Ok((base_url, shutdown_tx));
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        Err(format!("OpenAI compat test server did not become ready at {base_url}").into())
    }

    fn test_generation_response(text: &str) -> GenerationResponse {
        GenerationResponse {
            request_id: String::from("req-test"),
            product_id: String::from("psionic.text_generation"),
            model_id: String::from("tiny-gpt-oss"),
            session_id: None,
            output: GenerationOutput {
                tokens: TokenSequence::new(Vec::new()),
                text: String::from(text),
                structured: None,
                harmony: None,
            },
            usage: GenerationUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_tokens: 0,
            },
            metrics: GenerationMetrics::default(),
            provenance: None,
            termination: TerminationReason::EndOfSequence,
        }
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

    fn dense_llama_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("llama", name);
        metadata.extend(sentencepiece_tokenizer_metadata_entries());
        metadata
    }

    fn dense_gemma4_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("gemma4", name);
        metadata.push((
            String::from("tokenizer.ggml.pre"),
            GgufMetadataValue::String(String::from("gemma4")),
        ));
        metadata.extend(sentencepiece_tokenizer_metadata_entries());
        metadata
    }

    fn dense_gemma4_metadata_with_chat_template(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("gemma4", name);
        metadata.push((
            String::from("tokenizer.ggml.pre"),
            GgufMetadataValue::String(String::from("gemma4")),
        ));
        metadata.push((
            String::from("tokenizer.chat_template"),
            GgufMetadataValue::String(gemma4_chat_template().to_string()),
        ));
        metadata.extend(sentencepiece_tokenizer_metadata_entries_with_tokens(vec![
            "<unk>", "<bos>", "<eos>", "<|turn>", "<turn|>", "hello", "world",
        ]));
        metadata
    }

    fn json_llama_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("llama", name);
        metadata.extend(sentencepiece_tokenizer_metadata_entries_with_tokens(vec![
            "<unk>",
            "<s>",
            "</s>",
            "hello",
            "world",
            "psionic",
            "{\"ok\":true}",
            "{\"ok\":false}",
        ]));
        metadata
    }

    fn tagged_llama_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("llama", name);
        metadata.extend(sentencepiece_tokenizer_metadata_entries_with_tokens(vec![
            "<unk>",
            "<s>",
            "</s>",
            "hello",
            "{\"kind\":\"approve\",\"reason\":\"ok\"}",
            "{\"kind\":\"reject\",\"code\":7}",
        ]));
        metadata
    }

    fn auto_tool_message_llama_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("llama", name);
        set_context_length(&mut metadata, "llama", 256);
        metadata.extend(sentencepiece_tokenizer_metadata_entries_with_tokens(vec![
            "<unk>",
            "<s>",
            "</s>",
            "hello",
            "{\"kind\":\"message\",\"content\":\"world\"}",
            "world",
        ]));
        metadata
    }

    fn tool_call_llama_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("llama", name);
        set_context_length(&mut metadata, "llama", 256);
        metadata.extend(sentencepiece_tokenizer_metadata_entries_with_tokens(vec![
            "<unk>",
            "<s>",
            "</s>",
            "hello",
            "{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}}]}",
            "world",
        ]));
        metadata
    }

    fn invalid_tool_call_llama_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("llama", name);
        set_context_length(&mut metadata, "llama", 256);
        metadata.extend(sentencepiece_tokenizer_metadata_entries_with_tokens(vec![
            "<unk>",
            "<s>",
            "</s>",
            "hello",
            "{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":\"oops\",\"longitude\":2.3522}}]}",
            "world",
        ]));
        metadata
    }

    fn multi_tool_call_llama_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("llama", name);
        set_context_length(&mut metadata, "llama", 256);
        metadata.extend(sentencepiece_tokenizer_metadata_entries_with_tokens(vec![
            "<unk>",
            "<s>",
            "</s>",
            "hello",
            "{\"tool_calls\":[{\"name\":\"get_weather\",\"arguments\":{\"latitude\":48.8566,\"longitude\":2.3522}},{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}]}",
            "world",
        ]));
        metadata
    }

    #[test]
    fn required_tool_contract_prefers_plain_json_schema_batch() {
        let mut tools = BTreeMap::new();
        tools.insert(
            String::from("get_weather"),
            weather_tool_definition().function,
        );
        tools.insert(String::from("get_time"), time_tool_definition().function);

        let structured = structured_output_from_tool_contract(Some(&ToolCallingContract {
            tools,
            mode: ToolChoiceMode::Required,
            named_tool: None,
            parallel_tool_calls: true,
            minimum_required_tool_calls: 1,
        }))
        .expect("tool contract should compile")
        .expect("required tool contract should surface a schema");

        match structured {
            StructuredOutputRequest::JsonSchema { name, schema } => {
                assert_eq!(name.as_deref(), Some("psionic_tool_call_batch"));
                assert_eq!(schema["type"], serde_json::json!("object"));
                assert!(schema["properties"]["tool_calls"].is_object());
            }
            other => panic!("expected json schema tool batch, got {other:?}"),
        }
    }

    #[test]
    fn json_schema_tool_batch_outcome_surfaces_multiple_ordered_tool_calls() {
        let mut tools = BTreeMap::new();
        tools.insert(
            String::from("get_weather"),
            weather_tool_definition().function,
        );
        tools.insert(String::from("get_time"), time_tool_definition().function);
        let contract = ToolCallingContract {
            tools,
            mode: ToolChoiceMode::Required,
            named_tool: None,
            parallel_tool_calls: true,
            minimum_required_tool_calls: 1,
        };
        let mut response = test_generation_response("");
        response.output.structured = Some(StructuredOutputValue::Json {
            value: serde_json::json!({
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": {
                            "latitude": 48.8566,
                            "longitude": 2.3522
                        }
                    },
                    {
                        "name": "get_time",
                        "arguments": {
                            "timezone": "UTC"
                        }
                    }
                ]
            }),
        });

        let outcome = tool_call_outcome_from_response("req-test", &response, Some(&contract))
            .expect("json schema tool batch should parse")
            .expect("tool outcome should exist");

        assert_eq!(outcome.tool_calls.len(), 2);
        assert_eq!(outcome.tool_calls[0].name, "get_weather");
        assert_eq!(outcome.tool_calls[1].name, "get_time");
        assert_eq!(outcome.tool_calls[0].id, "req-test-tool-0");
        assert_eq!(outcome.tool_calls[1].id, "req-test-tool-1");
    }

    #[test]
    fn auto_tool_outcome_accepts_plain_text_when_structured_output_is_missing() {
        let mut tools = BTreeMap::new();
        tools.insert(
            String::from("get_weather"),
            weather_tool_definition().function,
        );
        let contract = ToolCallingContract {
            tools,
            mode: ToolChoiceMode::Auto,
            named_tool: None,
            parallel_tool_calls: false,
            minimum_required_tool_calls: 0,
        };
        let response = test_generation_response("plain assistant answer");

        let outcome = tool_call_outcome_from_response("req-test", &response, Some(&contract))
            .expect("auto tool mode should accept plain assistant text")
            .expect("tool outcome should exist");

        assert_eq!(outcome.content.as_deref(), Some("plain assistant answer"));
        assert!(outcome.tool_calls.is_empty());
    }

    fn weather_tool_definition() -> ToolDefinitionEnvelope {
        ToolDefinitionEnvelope {
            kind: String::from("function"),
            function: ToolDefinitionRequest {
                name: String::from("get_weather"),
                description: Some(String::from("Get the weather for one coordinate pair.")),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "latitude": { "type": "number" },
                        "longitude": { "type": "number" }
                    },
                    "required": ["latitude", "longitude"],
                    "additionalProperties": false
                })),
            },
        }
    }

    fn time_tool_definition() -> ToolDefinitionEnvelope {
        ToolDefinitionEnvelope {
            kind: String::from("function"),
            function: ToolDefinitionRequest {
                name: String::from("get_time"),
                description: Some(String::from("Get the current time for one timezone.")),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "timezone": { "type": "string", "minLength": 1 }
                    },
                    "required": ["timezone"],
                    "additionalProperties": false
                })),
            },
        }
    }

    #[test]
    fn parallel_tool_prompt_forbids_partial_or_duplicate_batches() {
        let mut tools = BTreeMap::new();
        tools.insert(
            String::from("get_weather"),
            weather_tool_definition().function,
        );
        tools.insert(String::from("get_time"), time_tool_definition().function);
        let prompt = tool_prompt_message(&ToolCallingContract {
            tools,
            mode: ToolChoiceMode::Required,
            named_tool: None,
            parallel_tool_calls: true,
            minimum_required_tool_calls: 1,
        });
        assert_eq!(prompt.role, PromptMessageRole::Developer);
        assert!(
            prompt
                .content
                .contains("include each requested tool exactly once")
        );
        assert!(prompt.content.contains("Do not omit a requested tool"));
        assert!(
            prompt
                .content
                .contains("repeat one tool instead of another")
        );
    }

    #[test]
    fn required_tool_floor_uses_backticked_declared_tool_mentions() {
        let mut tools = BTreeMap::new();
        tools.insert(
            String::from("get_paris_weather"),
            ToolDefinitionRequest {
                name: String::from("get_paris_weather"),
                description: Some(String::from("Paris only")),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                })),
            },
        );
        tools.insert(
            String::from("get_tokyo_weather"),
            ToolDefinitionRequest {
                name: String::from("get_tokyo_weather"),
                description: Some(String::from("Tokyo only")),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                })),
            },
        );
        let contract = ToolCallingContract {
            tools,
            mode: ToolChoiceMode::Required,
            named_tool: None,
            parallel_tool_calls: true,
            minimum_required_tool_calls: 1,
        };
        let floor = required_tool_call_floor_from_chat_messages(
            &[
                ChatCompletionMessage::text(
                    "system",
                    "Call `get_paris_weather` first, then `get_tokyo_weather`.",
                ),
                ChatCompletionMessage::text("user", "Use both weather tools now."),
            ],
            &contract,
            GgufDecoderFamily::Qwen35,
            None,
        )
        .expect("backticked tool floor should parse");

        assert_eq!(floor, 2);
    }

    fn dense_qwen_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = dense_family_header("qwen2", name);
        metadata.extend(qwen_tokenizer_metadata_entries());
        metadata
    }

    fn qwen35_chat_template() -> &'static str {
        include_str!("../../psionic-models/src/testdata/qwen35_chat_template.jinja")
            .trim_end_matches('\n')
    }

    fn gemma4_chat_template() -> &'static str {
        include_str!("../../psionic-models/src/testdata/gemma4_chat_template.jinja")
    }

    fn gemma4_pilot_fixture_path(default_path: &str) -> String {
        std::env::var("PSIONIC_GEMMA4_PILOT_GGUF_PATH")
            .unwrap_or_else(|_| String::from(default_path))
    }

    fn qwen35_decoder_metadata(name: &str) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = vec![
            (
                String::from("general.architecture"),
                GgufMetadataValue::String(String::from("qwen35")),
            ),
            (
                String::from("general.name"),
                GgufMetadataValue::String(name.to_string()),
            ),
            (
                String::from("qwen35.context_length"),
                GgufMetadataValue::U32(256),
            ),
            (
                String::from("qwen35.embedding_length"),
                GgufMetadataValue::U32(8),
            ),
            (
                String::from("qwen35.feed_forward_length"),
                GgufMetadataValue::U32(16),
            ),
            (
                String::from("qwen35.block_count"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.attention.head_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen35.attention.head_count_kv"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::U32(0),
                    GgufMetadataValue::U32(0),
                    GgufMetadataValue::U32(0),
                    GgufMetadataValue::U32(1),
                ]),
            ),
            (
                String::from("qwen35.attention.key_length"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.attention.value_length"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::F32(1e-6),
            ),
            (
                String::from("qwen35.rope.dimension_count"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.rope.freq_base"),
                GgufMetadataValue::F32(10_000_000.0),
            ),
            (
                String::from("qwen35.full_attention_interval"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.ssm.conv_kernel"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.ssm.group_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen35.ssm.inner_size"),
                GgufMetadataValue::U32(8),
            ),
            (
                String::from("qwen35.ssm.state_size"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.ssm.time_step_rank"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen35.vision.block_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen35.vision.embedding_length"),
                GgufMetadataValue::U32(6),
            ),
            (
                String::from("qwen35.vision_start_token_id"),
                GgufMetadataValue::U32(900),
            ),
            (
                String::from("qwen35.vision_end_token_id"),
                GgufMetadataValue::U32(901),
            ),
            (
                String::from("qwen35.image_token_id"),
                GgufMetadataValue::U32(902),
            ),
            (
                String::from("tokenizer.chat_template"),
                GgufMetadataValue::String(qwen35_chat_template().to_string()),
            ),
        ];
        metadata.extend(qwen35_tokenizer_metadata_entries());
        metadata
    }

    fn qwen35_native_full_attention_decoder_metadata(
        name: &str,
    ) -> Vec<(String, GgufMetadataValue)> {
        qwen35_native_full_attention_decoder_metadata_with_tokens(
            name,
            vec![
                "<|bos|>",
                "<|eos|>",
                "<|im_start|>",
                "<|im_end|>",
                "<think>",
                "</think>",
                "hello",
                "world",
                "proxy",
                "qwen35",
            ],
        )
    }

    fn qwen35_native_full_attention_decoder_metadata_with_tokens(
        name: &str,
        tokens: Vec<&str>,
    ) -> Vec<(String, GgufMetadataValue)> {
        let mut metadata = vec![
            (
                String::from("general.architecture"),
                GgufMetadataValue::String(String::from("qwen35")),
            ),
            (
                String::from("general.name"),
                GgufMetadataValue::String(name.to_string()),
            ),
            (
                String::from("qwen35.context_length"),
                GgufMetadataValue::U32(256),
            ),
            (
                String::from("qwen35.embedding_length"),
                GgufMetadataValue::U32(32),
            ),
            (
                String::from("qwen35.feed_forward_length"),
                GgufMetadataValue::U32(32),
            ),
            (
                String::from("qwen35.block_count"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("qwen35.attention.head_count"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.attention.head_count_kv"),
                GgufMetadataValue::Array(vec![GgufMetadataValue::U32(2)]),
            ),
            (
                String::from("qwen35.attention.key_length"),
                GgufMetadataValue::U32(8),
            ),
            (
                String::from("qwen35.attention.value_length"),
                GgufMetadataValue::U32(8),
            ),
            (
                String::from("qwen35.attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::F32(1e-6),
            ),
            (
                String::from("qwen35.rope.dimension_count"),
                GgufMetadataValue::U32(8),
            ),
            (
                String::from("qwen35.rope.freq_base"),
                GgufMetadataValue::F32(10_000_000.0),
            ),
            (
                String::from("qwen35.full_attention_interval"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("qwen35.ssm.conv_kernel"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("qwen35.ssm.group_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("qwen35.ssm.inner_size"),
                GgufMetadataValue::U32(32),
            ),
            (
                String::from("qwen35.ssm.state_size"),
                GgufMetadataValue::U32(8),
            ),
            (
                String::from("qwen35.ssm.time_step_rank"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("tokenizer.chat_template"),
                GgufMetadataValue::String(qwen35_chat_template().to_string()),
            ),
        ];
        metadata.extend(qwen35_tokenizer_metadata_entries_with_tokens(tokens));
        metadata
    }

    fn set_context_length(
        metadata: &mut [(String, GgufMetadataValue)],
        architecture: &str,
        context_length: u32,
    ) {
        let key = format!("{architecture}.context_length");
        if let Some((_, value)) = metadata.iter_mut().find(|(candidate, _)| candidate == &key) {
            *value = GgufMetadataValue::U32(context_length);
        }
    }

    fn dense_family_header(architecture: &str, name: &str) -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                String::from("general.architecture"),
                GgufMetadataValue::String(architecture.to_string()),
            ),
            (
                String::from("general.name"),
                GgufMetadataValue::String(name.to_string()),
            ),
            (
                format!("{architecture}.context_length"),
                GgufMetadataValue::U32(32),
            ),
            (
                format!("{architecture}.embedding_length"),
                GgufMetadataValue::U32(4),
            ),
            (
                format!("{architecture}.feed_forward_length"),
                GgufMetadataValue::U32(8),
            ),
            (
                format!("{architecture}.block_count"),
                GgufMetadataValue::U32(1),
            ),
            (
                format!("{architecture}.attention.head_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                format!("{architecture}.attention.head_count_kv"),
                GgufMetadataValue::U32(1),
            ),
            (
                format!("{architecture}.attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::F32(1e-5),
            ),
            (
                format!("{architecture}.rope.freq_base"),
                GgufMetadataValue::F32(10_000.0),
            ),
        ]
    }

    fn sentencepiece_tokenizer_metadata_entries() -> Vec<(String, GgufMetadataValue)> {
        sentencepiece_tokenizer_metadata_entries_with_tokens(vec![
            "<unk>", "<s>", "</s>", "hello", "world", "psionic",
        ])
    }

    fn sentencepiece_tokenizer_metadata_entries_with_tokens(
        tokens: Vec<&str>,
    ) -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                String::from("tokenizer.ggml.model"),
                GgufMetadataValue::String(String::from("llama")),
            ),
            (
                String::from("tokenizer.ggml.tokens"),
                GgufMetadataValue::Array(
                    tokens
                        .into_iter()
                        .map(|token| GgufMetadataValue::String(String::from(token)))
                        .collect(),
                ),
            ),
            (
                String::from("tokenizer.ggml.bos_token_id"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("tokenizer.ggml.eos_token_id"),
                GgufMetadataValue::U32(2),
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
        ]
    }

    fn qwen_tokenizer_metadata_entries() -> Vec<(String, GgufMetadataValue)> {
        vec![
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
        ]
    }

    fn qwen35_tokenizer_metadata_entries() -> Vec<(String, GgufMetadataValue)> {
        qwen35_tokenizer_metadata_entries_with_tokens(vec![
            "<|bos|>",
            "<|eos|>",
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
            "hello",
            "world",
            "proxy",
            "qwen35",
        ])
    }

    fn qwen35_tokenizer_metadata_entries_with_tokens(
        tokens: Vec<&str>,
    ) -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                String::from("tokenizer.ggml.model"),
                GgufMetadataValue::String(String::from("gpt2")),
            ),
            (
                String::from("tokenizer.ggml.pre"),
                GgufMetadataValue::String(String::from("qwen35")),
            ),
            (
                String::from("tokenizer.ggml.tokens"),
                GgufMetadataValue::Array(
                    tokens
                        .into_iter()
                        .map(|token| GgufMetadataValue::String(String::from(token)))
                        .collect(),
                ),
            ),
            (
                String::from("tokenizer.ggml.merges"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String(String::from("hello world")),
                    GgufMetadataValue::String(String::from("proxy qwen35")),
                ]),
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
                String::from("tokenizer.ggml.add_bos_token"),
                GgufMetadataValue::Bool(false),
            ),
            (
                String::from("tokenizer.ggml.add_eos_token"),
                GgufMetadataValue::Bool(false),
            ),
        ]
    }

    fn dense_decoder_tensors(
        include_qkv_bias: bool,
        hello_token_index: usize,
        world_token_index: usize,
    ) -> Vec<TestGgufTensor> {
        dense_decoder_tensors_with_vocab(include_qkv_bias, 6, hello_token_index, world_token_index)
    }

    fn dense_gemma4_cuda_decoder_tensors_with_vocab(
        vocab_size: usize,
        hello_token_index: usize,
    ) -> Vec<TestGgufTensor> {
        vec![
            dense_tensor(
                "token_embd.weight",
                vec![vocab_size, 4],
                token_embedding_values(vocab_size, hello_token_index),
            ),
            dense_tensor("output_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            quantized_q8_0_tensor("output.weight", vec![vocab_size, 4]),
            dense_tensor("blk.0.attn_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            quantized_q8_0_tensor("blk.0.attn_q.weight", vec![4, 4]),
            quantized_q8_0_tensor("blk.0.attn_k.weight", vec![2, 4]),
            quantized_q8_0_tensor("blk.0.attn_v.weight", vec![2, 4]),
            quantized_q8_0_tensor("blk.0.attn_output.weight", vec![4, 4]),
            quantized_q8_0_tensor("blk.0.ffn_gate.weight", vec![8, 4]),
            quantized_q8_0_tensor("blk.0.ffn_down.weight", vec![4, 8]),
            quantized_q8_0_tensor("blk.0.ffn_up.weight", vec![8, 4]),
            dense_tensor("blk.0.ffn_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
        ]
    }

    fn dense_decoder_tensors_with_vocab(
        include_qkv_bias: bool,
        vocab_size: usize,
        hello_token_index: usize,
        output_token_index: usize,
    ) -> Vec<TestGgufTensor> {
        let mut tensors = vec![
            dense_tensor(
                "token_embd.weight",
                vec![vocab_size, 4],
                token_embedding_values(vocab_size, hello_token_index),
            ),
            dense_tensor("output_norm.weight", vec![4], vec![1.0, 1.0, 1.0, 1.0]),
            dense_tensor(
                "output.weight",
                vec![vocab_size, 4],
                output_values(vocab_size, output_token_index),
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

    fn qwen35_decoder_tensors() -> Vec<TestGgufTensor> {
        let mut tensors = vec![
            dense_f32_tensor("token_embd.weight", vec![10, 8]),
            dense_f32_tensor("output_norm.weight", vec![8]),
            dense_f32_tensor("output.weight", vec![10, 8]),
        ];

        for layer_index in 0..4 {
            let prefix = format!("blk.{layer_index}");
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.attn_norm.weight"),
                vec![8],
            ));
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.ffn_gate.weight"),
                vec![16, 8],
            ));
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.ffn_up.weight"),
                vec![16, 8],
            ));
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.ffn_down.weight"),
                vec![8, 16],
            ));
            tensors.push(dense_f32_tensor(
                &format!("{prefix}.post_attention_norm.weight"),
                vec![8],
            ));

            if layer_index < 3 {
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_qkv.weight"),
                    vec![24, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_gate.weight"),
                    vec![8, 8],
                ));
                tensors.push(dense_f32_tensor(&format!("{prefix}.ssm_a"), vec![2]));
                tensors.push(dense_f32_tensor(&format!("{prefix}.ssm_dt"), vec![2]));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_alpha.weight"),
                    vec![2, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_beta.weight"),
                    vec![2, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_conv1d.weight"),
                    vec![24, 4],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_norm.weight"),
                    vec![4],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.ssm_out.weight"),
                    vec![8, 8],
                ));
            } else {
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_q.weight"),
                    vec![16, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_k.weight"),
                    vec![4, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_v.weight"),
                    vec![4, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_output.weight"),
                    vec![8, 8],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_q_norm.weight"),
                    vec![4],
                ));
                tensors.push(dense_f32_tensor(
                    &format!("{prefix}.attn_k_norm.weight"),
                    vec![4],
                ));
            }
        }

        tensors
    }

    fn qwen35_native_full_attention_decoder_tensors() -> Vec<TestGgufTensor> {
        qwen35_native_full_attention_decoder_tensors_with_vocab(10)
    }

    fn qwen35_native_full_attention_decoder_tensors_with_vocab(
        vocab_size: usize,
    ) -> Vec<TestGgufTensor> {
        vec![
            dense_f32_tensor("token_embd.weight", vec![vocab_size, 32]),
            dense_f32_tensor("output_norm.weight", vec![32]),
            quantized_q8_0_tensor("output.weight", vec![vocab_size, 32]),
            dense_f32_tensor("blk.0.attn_norm.weight", vec![32]),
            quantized_q8_0_tensor("blk.0.ffn_gate.weight", vec![32, 32]),
            quantized_q8_0_tensor("blk.0.ffn_up.weight", vec![32, 32]),
            quantized_q8_0_tensor("blk.0.ffn_down.weight", vec![32, 32]),
            dense_f32_tensor("blk.0.post_attention_norm.weight", vec![32]),
            quantized_q8_0_tensor("blk.0.attn_q.weight", vec![64, 32]),
            quantized_q8_0_tensor("blk.0.attn_k.weight", vec![16, 32]),
            quantized_q8_0_tensor("blk.0.attn_v.weight", vec![16, 32]),
            quantized_q8_0_tensor("blk.0.attn_output.weight", vec![32, 32]),
            dense_f32_tensor("blk.0.attn_q_norm.weight", vec![8]),
            dense_f32_tensor("blk.0.attn_k_norm.weight", vec![8]),
        ]
    }

    fn token_embedding_values(vocab_size: usize, hello_token_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; vocab_size * 4];
        values[hello_token_index.saturating_mul(4)] = 2.0;
        values
    }

    fn output_values(vocab_size: usize, output_token_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; vocab_size * 4];
        values[output_token_index.saturating_mul(4)] = 1.0;
        values
    }

    fn gpt_oss_metadata() -> Vec<(String, GgufMetadataValue)> {
        vec![
            (
                String::from("general.architecture"),
                GgufMetadataValue::String(String::from("gpt-oss")),
            ),
            (
                String::from("general.name"),
                GgufMetadataValue::String(String::from("tiny psionic gpt-oss")),
            ),
            (
                String::from("general.alignment"),
                GgufMetadataValue::U32(32),
            ),
            (
                String::from("gpt-oss.context_length"),
                GgufMetadataValue::U32(128),
            ),
            (
                String::from("gpt-oss.embedding_length"),
                GgufMetadataValue::U32(32),
            ),
            (
                String::from("gpt-oss.feed_forward_length"),
                GgufMetadataValue::U32(32),
            ),
            (
                String::from("gpt-oss.expert_feed_forward_length"),
                GgufMetadataValue::U32(32),
            ),
            (
                String::from("gpt-oss.block_count"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("gpt-oss.attention.head_count"),
                GgufMetadataValue::U32(4),
            ),
            (
                String::from("gpt-oss.attention.head_count_kv"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("gpt-oss.attention.key_length"),
                GgufMetadataValue::U32(16),
            ),
            (
                String::from("gpt-oss.attention.value_length"),
                GgufMetadataValue::U32(16),
            ),
            (
                String::from("gpt-oss.attention.layer_norm_rms_epsilon"),
                GgufMetadataValue::F32(1e-5),
            ),
            (
                String::from("gpt-oss.rope.dimension_count"),
                GgufMetadataValue::U32(16),
            ),
            (
                String::from("gpt-oss.rope.freq_base"),
                GgufMetadataValue::F32(10_000.0),
            ),
            (
                String::from("gpt-oss.rope.scaling.factor"),
                GgufMetadataValue::F32(32.0),
            ),
            (
                String::from("gpt-oss.rope.scaling.original_context_length"),
                GgufMetadataValue::U32(4096),
            ),
            (
                String::from("gpt-oss.expert_count"),
                GgufMetadataValue::U32(3),
            ),
            (
                String::from("gpt-oss.expert_used_count"),
                GgufMetadataValue::U32(2),
            ),
            (
                String::from("tokenizer.ggml.model"),
                GgufMetadataValue::String(String::from("gpt2")),
            ),
            (
                String::from("tokenizer.ggml.pre"),
                GgufMetadataValue::String(String::from("gpt-4o")),
            ),
            (
                String::from("tokenizer.ggml.tokens"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String(String::from("<|start|>")),
                    GgufMetadataValue::String(String::from("<|end|>")),
                    GgufMetadataValue::String(String::from("hello")),
                    GgufMetadataValue::String(String::from("world")),
                    GgufMetadataValue::String(String::from("psionic")),
                    GgufMetadataValue::String(String::from("gpt-oss")),
                ]),
            ),
            (
                String::from("tokenizer.ggml.merges"),
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String(String::from("hello world")),
                    GgufMetadataValue::String(String::from("psionic gpt-oss")),
                ]),
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
                String::from("tokenizer.ggml.padding_token_id"),
                GgufMetadataValue::U32(1),
            ),
            (
                String::from("tokenizer.ggml.add_bos_token"),
                GgufMetadataValue::Bool(false),
            ),
            (
                String::from("tokenizer.ggml.add_eos_token"),
                GgufMetadataValue::Bool(false),
            ),
        ]
    }

    fn gpt_oss_tensors() -> Vec<TestGgufTensor> {
        let expert_blocks = 3 * 32;
        vec![
            quantized_q8_0_tensor("token_embd.weight", vec![6, 32]),
            dense_f32_tensor("output_norm.weight", vec![32]),
            quantized_q8_0_tensor("output.weight", vec![6, 32]),
            dense_f32_tensor("blk.0.attn_norm.weight", vec![32]),
            quantized_q8_0_tensor("blk.0.attn_q.weight", vec![64, 32]),
            dense_f32_tensor("blk.0.attn_q.bias", vec![64]),
            quantized_q8_0_tensor("blk.0.attn_k.weight", vec![16, 32]),
            dense_f32_tensor("blk.0.attn_k.bias", vec![16]),
            quantized_q8_0_tensor("blk.0.attn_v.weight", vec![16, 32]),
            dense_f32_tensor("blk.0.attn_v.bias", vec![16]),
            quantized_q8_0_tensor("blk.0.attn_output.weight", vec![32, 64]),
            dense_f32_tensor("blk.0.attn_output.bias", vec![32]),
            dense_f32_tensor("blk.0.post_attention_norm.weight", vec![32]),
            dense_f32_tensor("blk.0.attn_sinks.weight", vec![16]),
            dense_f32_tensor("blk.0.ffn_gate_inp.weight", vec![3, 32]),
            dense_f32_tensor("blk.0.ffn_gate_inp.bias", vec![3]),
            quantized_mxfp4_tensor(
                "blk.0.ffn_gate_exps.weight",
                vec![3, 32, 32],
                repeated_mxfp4_bytes(expert_blocks),
            ),
            dense_f32_tensor("blk.0.ffn_gate_exps.bias", vec![3, 32]),
            quantized_mxfp4_tensor(
                "blk.0.ffn_up_exps.weight",
                vec![3, 32, 32],
                repeated_mxfp4_bytes(expert_blocks),
            ),
            dense_f32_tensor("blk.0.ffn_up_exps.bias", vec![3, 32]),
            quantized_mxfp4_tensor(
                "blk.0.ffn_down_exps.weight",
                vec![3, 32, 32],
                repeated_mxfp4_bytes(expert_blocks),
            ),
            dense_f32_tensor("blk.0.ffn_down_exps.bias", vec![3, 32]),
        ]
    }

    fn dense_tensor(name: &str, shape: Vec<usize>, values: Vec<f32>) -> TestGgufTensor {
        TestGgufTensor::new(
            name,
            shape,
            GgufTensorType::F32,
            encode_f32_bytes(values.as_slice()),
        )
    }

    fn dense_f32_tensor(name: &str, shape: Vec<usize>) -> TestGgufTensor {
        let elements = shape.iter().product::<usize>();
        TestGgufTensor::new(
            name,
            shape,
            GgufTensorType::F32,
            encode_f32_bytes(&vec![0.0; elements]),
        )
    }

    fn quantized_q8_0_tensor(name: &str, shape: Vec<usize>) -> TestGgufTensor {
        let rows = shape
            .iter()
            .take(shape.len().saturating_sub(1))
            .product::<usize>();
        TestGgufTensor::new(name, shape, GgufTensorType::Q8_0, repeated_q8_0_bytes(rows))
    }

    fn quantized_mxfp4_tensor(name: &str, shape: Vec<usize>, bytes: Vec<u8>) -> TestGgufTensor {
        TestGgufTensor::new(name, shape, GgufTensorType::MXFP4, bytes)
    }

    fn repeated_q8_0_bytes(row_count: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(row_count * 34);
        for _ in 0..row_count {
            bytes.extend([0x00, 0x3c]);
            bytes.extend([0_u8; 32]);
        }
        bytes
    }

    fn repeated_mxfp4_bytes(block_count: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(block_count * 17);
        for _ in 0..block_count {
            bytes.push(128_u8);
            bytes.extend([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]);
            bytes.extend([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]);
        }
        bytes
    }

    fn write_test_gguf(
        path: &std::path::Path,
        metadata: &[(String, GgufMetadataValue)],
        tensors: &[TestGgufTensor],
    ) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(path, build_test_gguf(metadata, tensors)?)?;
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

    fn generate_observation_from_chat_payload(
        payload: &serde_json::Value,
        rendered_prompt: Option<&str>,
    ) -> GenerateObservation {
        GenerateObservation {
            rendered_prompt: rendered_prompt.map(String::from),
            output_text: payload["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            done_reason: payload["choices"][0]["finish_reason"]
                .as_str()
                .map(String::from),
            prompt_eval_count: payload["usage"]["prompt_tokens"]
                .as_u64()
                .map(|value| value as usize),
            eval_count: payload["usage"]["completion_tokens"]
                .as_u64()
                .map(|value| value as usize),
            performance: None,
            error: None,
        }
    }

    fn align_usize(value: usize, alignment: usize) -> usize {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + alignment - remainder
        }
    }

    fn encode_f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
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
            GgufTensorType::Q8_0 => 8,
            GgufTensorType::MXFP4 => 39,
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
