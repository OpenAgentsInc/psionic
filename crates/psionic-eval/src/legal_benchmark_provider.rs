//! Provider-neutral model adapter contracts for legal benchmark agents.
//!
//! The adapter layer normalizes hosted providers, OpenAI-compatible local
//! serving endpoints, and deterministic CI mocks into one benchmark model
//! surface. It deliberately records route and secret reference identity without
//! carrying raw credentials into run artifacts.

use std::collections::{BTreeMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{LegalBenchmarkToolName, Metadata, stable_json_digest};

pub const LEGAL_BENCHMARK_PROVIDER_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelProviderFamily {
    OpenAiCompatible,
    Anthropic,
    PsionicCompatible,
    Mock,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelProviderRoute {
    pub schema_version: u16,
    pub route_id: String,
    pub family: ModelProviderFamily,
    pub base_url: String,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub secret_reference_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub redacted_headers: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

impl ModelProviderRoute {
    pub fn openai_compatible(
        route_id: impl Into<String>,
        base_url: impl Into<String>,
        model_id: impl Into<String>,
        secret_reference_id: Option<String>,
    ) -> Self {
        let secret_reference_id = secret_reference_id.filter(|value| !value.trim().is_empty());
        let mut redacted_headers = BTreeMap::new();
        if let Some(secret_ref) = &secret_reference_id {
            redacted_headers.insert(
                String::from("authorization"),
                format!("Bearer <secret_ref:{secret_ref}>"),
            );
        }
        Self {
            schema_version: LEGAL_BENCHMARK_PROVIDER_SCHEMA_VERSION,
            route_id: route_id.into(),
            family: ModelProviderFamily::OpenAiCompatible,
            base_url: trim_url(base_url.into()),
            model_id: model_id.into(),
            endpoint_path: Some(String::from("/chat/completions")),
            secret_reference_id,
            redacted_headers,
            metadata: Metadata::new(),
        }
    }

    pub fn anthropic(
        route_id: impl Into<String>,
        base_url: impl Into<String>,
        model_id: impl Into<String>,
        secret_reference_id: Option<String>,
    ) -> Self {
        let secret_reference_id = secret_reference_id.filter(|value| !value.trim().is_empty());
        let mut redacted_headers = BTreeMap::new();
        redacted_headers.insert(
            String::from("anthropic-version"),
            String::from("2023-06-01"),
        );
        if let Some(secret_ref) = &secret_reference_id {
            redacted_headers.insert(
                String::from("x-api-key"),
                format!("<secret_ref:{secret_ref}>"),
            );
        }
        Self {
            schema_version: LEGAL_BENCHMARK_PROVIDER_SCHEMA_VERSION,
            route_id: route_id.into(),
            family: ModelProviderFamily::Anthropic,
            base_url: trim_url(base_url.into()),
            model_id: model_id.into(),
            endpoint_path: Some(String::from("/messages")),
            secret_reference_id,
            redacted_headers,
            metadata: Metadata::new(),
        }
    }

    pub fn mock(route_id: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            schema_version: LEGAL_BENCHMARK_PROVIDER_SCHEMA_VERSION,
            route_id: route_id.into(),
            family: ModelProviderFamily::Mock,
            base_url: String::from("mock://legal-benchmark"),
            model_id: model_id.into(),
            endpoint_path: None,
            secret_reference_id: None,
            redacted_headers: BTreeMap::new(),
            metadata: Metadata::new(),
        }
    }

    pub fn endpoint_url(&self) -> String {
        match &self.endpoint_path {
            Some(path) if path.starts_with('/') => format!("{}{}", self.base_url, path),
            Some(path) => format!("{}/{}", self.base_url, path),
            None => self.base_url.clone(),
        }
    }

    pub fn config_hash(&self) -> Result<String, ModelAdapterError> {
        stable_json_digest("psionic.legal_benchmark.provider_route.v1", self).map_err(|err| {
            ModelAdapterError::new(
                ModelAdapterFailureKind::InternalError,
                format!("failed to hash provider route: {err}"),
            )
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelMessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ModelMessageRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelMessage {
    pub role: ModelMessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ModelToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

impl ModelMessage {
    pub fn new(role: ModelMessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
            metadata: Metadata::new(),
        }
    }

    pub fn assistant_response(content: Option<String>, tool_calls: Vec<ModelToolCall>) -> Self {
        Self {
            role: ModelMessageRole::Assistant,
            content,
            tool_calls,
            tool_call_id: None,
            tool_name: None,
            metadata: Metadata::new(),
        }
    }

    pub fn tool_result(result: ToolResultMessage) -> Self {
        Self {
            role: ModelMessageRole::Tool,
            content: Some(result.content),
            tool_calls: Vec::new(),
            tool_call_id: Some(result.tool_call_id),
            tool_name: Some(result.tool_name),
            metadata: result.metadata,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: String,
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSamplingConfig {
    pub max_output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
}

impl Default for ModelSamplingConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: 4096,
            temperature: Some(0.0),
            top_p: None,
            random_seed: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelRetryPolicy {
    pub max_retries: u32,
    pub timeout_ms: u64,
    pub retry_on_rate_limit: bool,
    pub retry_on_timeout: bool,
    pub retry_on_server_error: bool,
}

impl Default for ModelRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            timeout_ms: 120_000,
            retry_on_rate_limit: true,
            retry_on_timeout: true,
            retry_on_server_error: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    pub schema_version: u16,
    pub request_id: String,
    pub messages: Vec<ModelMessage>,
    pub tools: Vec<ModelToolSpec>,
    pub sampling: ModelSamplingConfig,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

impl ModelRequest {
    pub fn new(request_id: impl Into<String>, messages: Vec<ModelMessage>) -> Self {
        Self {
            schema_version: LEGAL_BENCHMARK_PROVIDER_SCHEMA_VERSION,
            request_id: request_id.into(),
            messages,
            tools: legal_benchmark_model_tool_specs(),
            sampling: ModelSamplingConfig::default(),
            metadata: Metadata::new(),
        }
    }

    pub fn config_hash(&self, route: &ModelProviderRoute) -> Result<String, ModelAdapterError> {
        let value = json!({
            "route": route,
            "sampling": self.sampling,
            "tools": self.tools,
        });
        stable_json_digest("psionic.legal_benchmark.model_config.v1", &value).map_err(|err| {
            ModelAdapterError::new(
                ModelAdapterFailureKind::InternalError,
                format!("failed to hash model config: {err}"),
            )
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelToolCall {
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStopReason {
    Stop,
    ToolCalls,
    MaxTokens,
    SafetyRefusal,
    ProviderError,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cached_input_tokens: u64,
    pub estimated_cost_micro_usd: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    pub schema_version: u16,
    pub response_id: String,
    pub request_id: String,
    pub route_id: String,
    pub provider_family: ModelProviderFamily,
    pub model_id: String,
    pub model_config_hash: String,
    pub secret_reference_id: Option<String>,
    pub final_text: Option<String>,
    pub tool_calls: Vec<ModelToolCall>,
    pub stop_reason: ModelStopReason,
    pub usage: ModelUsage,
    pub elapsed_ms: u64,
    pub retry_count: u32,
    pub raw_response_hash: String,
    pub created_at_ms: u64,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAdapterFailureKind {
    Timeout,
    RateLimited,
    ContextOverflow,
    SafetyRefusal,
    ProviderError,
    TransportError,
    ParseError,
    InvalidRequest,
    InternalError,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelAdapterError {
    pub kind: ModelAdapterFailureKind,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_code: Option<u16>,
    pub retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_response_hash: Option<String>,
}

impl ModelAdapterError {
    pub fn new(kind: ModelAdapterFailureKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
            status_code: None,
            retryable: false,
            route_id: None,
            model_id: None,
            elapsed_ms: None,
            raw_response_hash: None,
        }
    }

    fn with_route(mut self, route: &ModelProviderRoute) -> Self {
        self.route_id = Some(route.route_id.clone());
        self.model_id = Some(route.model_id.clone());
        self
    }
}

pub type ModelAdapterResult<T> = Result<T, ModelAdapterError>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderHttpRequest {
    pub method: String,
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Value,
    pub timeout_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProviderHttpResponse {
    pub status: u16,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    pub body: Value,
    pub elapsed_ms: u64,
}

pub trait ProviderHttpTransport {
    fn send(&mut self, request: ProviderHttpRequest) -> ModelAdapterResult<ProviderHttpResponse>;
}

pub trait ModelAdapter {
    fn route(&self) -> &ModelProviderRoute;

    fn complete(&mut self, request: &ModelRequest) -> ModelAdapterResult<ModelResponse>;
}

#[derive(Clone, Debug)]
pub struct OpenAiCompatibleAdapter<T> {
    route: ModelProviderRoute,
    retry_policy: ModelRetryPolicy,
    transport: T,
}

impl<T> OpenAiCompatibleAdapter<T> {
    pub fn new(route: ModelProviderRoute, transport: T, retry_policy: ModelRetryPolicy) -> Self {
        Self {
            route,
            retry_policy,
            transport,
        }
    }

    pub fn into_transport(self) -> T {
        self.transport
    }
}

impl<T> ModelAdapter for OpenAiCompatibleAdapter<T>
where
    T: ProviderHttpTransport,
{
    fn route(&self) -> &ModelProviderRoute {
        &self.route
    }

    fn complete(&mut self, request: &ModelRequest) -> ModelAdapterResult<ModelResponse> {
        let http_request = build_openai_request(&self.route, &self.retry_policy, request);
        let (http_response, retry_count) = send_with_retries(
            &mut self.transport,
            http_request,
            &self.route,
            &self.retry_policy,
        )?;
        parse_openai_response(&self.route, request, &http_response, retry_count)
    }
}

#[derive(Clone, Debug)]
pub struct AnthropicAdapter<T> {
    route: ModelProviderRoute,
    retry_policy: ModelRetryPolicy,
    transport: T,
}

impl<T> AnthropicAdapter<T> {
    pub fn new(route: ModelProviderRoute, transport: T, retry_policy: ModelRetryPolicy) -> Self {
        Self {
            route,
            retry_policy,
            transport,
        }
    }

    pub fn into_transport(self) -> T {
        self.transport
    }
}

impl<T> ModelAdapter for AnthropicAdapter<T>
where
    T: ProviderHttpTransport,
{
    fn route(&self) -> &ModelProviderRoute {
        &self.route
    }

    fn complete(&mut self, request: &ModelRequest) -> ModelAdapterResult<ModelResponse> {
        let http_request = build_anthropic_request(&self.route, &self.retry_policy, request);
        let (http_response, retry_count) = send_with_retries(
            &mut self.transport,
            http_request,
            &self.route,
            &self.retry_policy,
        )?;
        parse_anthropic_response(&self.route, request, &http_response, retry_count)
    }
}

#[derive(Clone, Debug)]
pub struct MockModelAdapter {
    route: ModelProviderRoute,
    responses: VecDeque<ModelAdapterResult<ModelResponse>>,
}

impl MockModelAdapter {
    pub fn new(
        route: ModelProviderRoute,
        responses: Vec<ModelAdapterResult<ModelResponse>>,
    ) -> Self {
        Self {
            route,
            responses: responses.into(),
        }
    }
}

impl ModelAdapter for MockModelAdapter {
    fn route(&self) -> &ModelProviderRoute {
        &self.route
    }

    fn complete(&mut self, _request: &ModelRequest) -> ModelAdapterResult<ModelResponse> {
        match self.responses.pop_front() {
            Some(response) => response,
            None => Err(ModelAdapterError::new(
                ModelAdapterFailureKind::ProviderError,
                "mock adapter has no queued response",
            )
            .with_route(&self.route)),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MockHttpTransport {
    pub requests: Vec<ProviderHttpRequest>,
    pub responses: VecDeque<ModelAdapterResult<ProviderHttpResponse>>,
}

impl MockHttpTransport {
    pub fn new(responses: Vec<ModelAdapterResult<ProviderHttpResponse>>) -> Self {
        Self {
            requests: Vec::new(),
            responses: responses.into(),
        }
    }
}

impl ProviderHttpTransport for MockHttpTransport {
    fn send(&mut self, request: ProviderHttpRequest) -> ModelAdapterResult<ProviderHttpResponse> {
        self.requests.push(request);
        match self.responses.pop_front() {
            Some(response) => response,
            None => Err(ModelAdapterError::new(
                ModelAdapterFailureKind::TransportError,
                "mock HTTP transport has no queued response",
            )),
        }
    }
}

pub fn legal_benchmark_model_tool_specs() -> Vec<ModelToolSpec> {
    vec![
        tool_spec(
            LegalBenchmarkToolName::Shell,
            "Run a sandboxed command with argv, timeout, stdout, stderr, and exit-code receipt.",
            json!({
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "timeout_ms": {"type": "integer", "minimum": 1}
                },
                "additionalProperties": false
            }),
        ),
        tool_spec(
            LegalBenchmarkToolName::Read,
            "Read a file from documents, workspace, or output roots, preferring extracted text when requested.",
            json!({
                "type": "object",
                "required": ["root", "relative_path", "prefer_extracted"],
                "properties": {
                    "root": {"type": "string", "enum": ["documents", "workspace", "output"]},
                    "relative_path": {"type": "string"},
                    "prefer_extracted": {"type": "boolean"}
                },
                "additionalProperties": false
            }),
        ),
        tool_spec(
            LegalBenchmarkToolName::Write,
            "Write a UTF-8 output file under the workspace or output root.",
            json!({
                "type": "object",
                "required": ["root", "relative_path", "content", "overwrite"],
                "properties": {
                    "root": {"type": "string", "enum": ["workspace", "output"]},
                    "relative_path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"}
                },
                "additionalProperties": false
            }),
        ),
        tool_spec(
            LegalBenchmarkToolName::Edit,
            "Replace exact text in an existing workspace or output file.",
            json!({
                "type": "object",
                "required": ["root", "relative_path", "find", "replace"],
                "properties": {
                    "root": {"type": "string", "enum": ["workspace", "output"]},
                    "relative_path": {"type": "string"},
                    "find": {"type": "string"},
                    "replace": {"type": "string"},
                    "expected_replacements": {"type": "integer", "minimum": 0}
                },
                "additionalProperties": false
            }),
        ),
        tool_spec(
            LegalBenchmarkToolName::Glob,
            "List files matching a simple wildcard pattern under one allowed root.",
            json!({
                "type": "object",
                "required": ["root", "pattern", "max_results", "include_hidden"],
                "properties": {
                    "root": {"type": "string", "enum": ["documents", "workspace", "output"]},
                    "pattern": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1},
                    "include_hidden": {"type": "boolean"}
                },
                "additionalProperties": false
            }),
        ),
        tool_spec(
            LegalBenchmarkToolName::Grep,
            "Search text files under one allowed root and return deterministic line matches.",
            json!({
                "type": "object",
                "required": ["root", "pattern", "case_sensitive", "max_results", "include_hidden"],
                "properties": {
                    "root": {"type": "string", "enum": ["documents", "workspace", "output"]},
                    "pattern": {"type": "string"},
                    "case_sensitive": {"type": "boolean"},
                    "max_results": {"type": "integer", "minimum": 1},
                    "include_hidden": {"type": "boolean"}
                },
                "additionalProperties": false
            }),
        ),
    ]
}

fn tool_spec(
    name: LegalBenchmarkToolName,
    description: &str,
    input_schema: Value,
) -> ModelToolSpec {
    ModelToolSpec {
        name: name.as_str().to_owned(),
        description: description.to_owned(),
        input_schema,
    }
}

fn build_openai_request(
    route: &ModelProviderRoute,
    retry_policy: &ModelRetryPolicy,
    request: &ModelRequest,
) -> ProviderHttpRequest {
    let messages = request
        .messages
        .iter()
        .map(openai_message_json)
        .collect::<Vec<_>>();
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        })
        .collect::<Vec<_>>();
    let mut body = json!({
        "model": route.model_id,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_completion_tokens": request.sampling.max_output_tokens,
    });
    if let Some(temperature) = request.sampling.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.sampling.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(seed) = request.sampling.random_seed {
        body["seed"] = json!(seed);
    }
    ProviderHttpRequest {
        method: String::from("POST"),
        url: route.endpoint_url(),
        headers: route.redacted_headers.clone(),
        body,
        timeout_ms: retry_policy.timeout_ms,
    }
}

fn openai_message_json(message: &ModelMessage) -> Value {
    match message.role {
        ModelMessageRole::Tool => json!({
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content.clone().unwrap_or_default(),
        }),
        ModelMessageRole::Assistant if !message.tool_calls.is_empty() => json!({
            "role": "assistant",
            "content": message.content.clone().unwrap_or_default(),
            "tool_calls": message.tool_calls.iter().map(openai_tool_call_message_json).collect::<Vec<_>>(),
        }),
        _ => json!({
            "role": message.role.as_str(),
            "content": message.content.clone().unwrap_or_default(),
        }),
    }
}

fn openai_tool_call_message_json(tool_call: &ModelToolCall) -> Value {
    json!({
        "id": tool_call.tool_call_id.clone(),
        "type": "function",
        "function": {
            "name": tool_call.tool_name.clone(),
            "arguments": tool_call.arguments.to_string(),
        }
    })
}

fn build_anthropic_request(
    route: &ModelProviderRoute,
    retry_policy: &ModelRetryPolicy,
    request: &ModelRequest,
) -> ProviderHttpRequest {
    let system = request
        .messages
        .iter()
        .filter(|message| message.role == ModelMessageRole::System)
        .filter_map(|message| message.content.clone())
        .collect::<Vec<_>>()
        .join("\n\n");
    let messages = request
        .messages
        .iter()
        .filter(|message| message.role != ModelMessageRole::System)
        .map(anthropic_message_json)
        .collect::<Vec<_>>();
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            })
        })
        .collect::<Vec<_>>();
    let mut body = json!({
        "model": route.model_id,
        "max_tokens": request.sampling.max_output_tokens,
        "messages": messages,
        "tools": tools,
    });
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if let Some(temperature) = request.sampling.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.sampling.top_p {
        body["top_p"] = json!(top_p);
    }
    ProviderHttpRequest {
        method: String::from("POST"),
        url: route.endpoint_url(),
        headers: route.redacted_headers.clone(),
        body,
        timeout_ms: retry_policy.timeout_ms,
    }
}

fn anthropic_message_json(message: &ModelMessage) -> Value {
    match message.role {
        ModelMessageRole::Tool => json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content.clone().unwrap_or_default(),
                "is_error": message
                    .metadata
                    .get("is_error")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            }]
        }),
        ModelMessageRole::Assistant if !message.tool_calls.is_empty() => json!({
            "role": "assistant",
            "content": message.tool_calls.iter().map(anthropic_tool_use_message_json).collect::<Vec<_>>(),
        }),
        ModelMessageRole::Assistant => json!({
            "role": "assistant",
            "content": message.content.clone().unwrap_or_default(),
        }),
        _ => json!({
            "role": "user",
            "content": message.content.clone().unwrap_or_default(),
        }),
    }
}

fn anthropic_tool_use_message_json(tool_call: &ModelToolCall) -> Value {
    json!({
        "type": "tool_use",
        "id": tool_call.tool_call_id.clone(),
        "name": tool_call.tool_name.clone(),
        "input": tool_call.arguments.clone(),
    })
}

fn send_with_retries<T>(
    transport: &mut T,
    request: ProviderHttpRequest,
    route: &ModelProviderRoute,
    retry_policy: &ModelRetryPolicy,
) -> ModelAdapterResult<(ProviderHttpResponse, u32)>
where
    T: ProviderHttpTransport,
{
    let mut attempts = 0;
    loop {
        let response = transport.send(request.clone());
        match response {
            Ok(http_response) if http_response.status < 400 => {
                return Ok((http_response, attempts));
            }
            Ok(http_response) => {
                let mut err = classify_http_failure(route, &http_response)?;
                let retryable = should_retry(err.kind, retry_policy);
                err.retryable = retryable;
                if retryable && attempts < retry_policy.max_retries {
                    attempts += 1;
                    continue;
                }
                return Err(err);
            }
            Err(mut err) => {
                err = err.with_route(route);
                let retryable = should_retry(err.kind, retry_policy);
                err.retryable = retryable;
                if retryable && attempts < retry_policy.max_retries {
                    attempts += 1;
                    continue;
                }
                return Err(err);
            }
        }
    }
}

fn classify_http_failure(
    route: &ModelProviderRoute,
    response: &ProviderHttpResponse,
) -> ModelAdapterResult<ModelAdapterError> {
    let raw_response_hash = raw_response_hash(&response.body)?;
    let kind = match response.status {
        408 | 504 => ModelAdapterFailureKind::Timeout,
        429 => ModelAdapterFailureKind::RateLimited,
        400 if json_contains_context_overflow(&response.body) => {
            ModelAdapterFailureKind::ContextOverflow
        }
        400 | 401 | 403 | 404 => ModelAdapterFailureKind::ProviderError,
        500..=599 => ModelAdapterFailureKind::ProviderError,
        _ => ModelAdapterFailureKind::ProviderError,
    };
    Ok(ModelAdapterError {
        kind,
        detail: response
            .body
            .get("error")
            .and_then(|value| {
                value
                    .get("message")
                    .and_then(Value::as_str)
                    .or_else(|| value.as_str())
            })
            .unwrap_or("provider returned an error status")
            .to_owned(),
        status_code: Some(response.status),
        retryable: false,
        route_id: Some(route.route_id.clone()),
        model_id: Some(route.model_id.clone()),
        elapsed_ms: Some(response.elapsed_ms),
        raw_response_hash: Some(raw_response_hash),
    })
}

fn should_retry(kind: ModelAdapterFailureKind, retry_policy: &ModelRetryPolicy) -> bool {
    match kind {
        ModelAdapterFailureKind::Timeout => retry_policy.retry_on_timeout,
        ModelAdapterFailureKind::RateLimited => retry_policy.retry_on_rate_limit,
        ModelAdapterFailureKind::ProviderError => retry_policy.retry_on_server_error,
        _ => false,
    }
}

fn parse_openai_response(
    route: &ModelProviderRoute,
    request: &ModelRequest,
    response: &ProviderHttpResponse,
    retry_count: u32,
) -> ModelAdapterResult<ModelResponse> {
    let choice = response
        .body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| parse_error(route, "OpenAI-compatible response missing first choice"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| parse_error(route, "OpenAI-compatible response missing message"))?;
    let final_text = message
        .get("content")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(openai_tool_call)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let finish_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("stop");
    let stop_reason = match finish_reason {
        "tool_calls" => ModelStopReason::ToolCalls,
        "length" => ModelStopReason::MaxTokens,
        "content_filter" => ModelStopReason::SafetyRefusal,
        "stop" => ModelStopReason::Stop,
        _ if !tool_calls.is_empty() => ModelStopReason::ToolCalls,
        _ => ModelStopReason::ProviderError,
    };
    Ok(ModelResponse {
        schema_version: LEGAL_BENCHMARK_PROVIDER_SCHEMA_VERSION,
        response_id: response
            .body
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("openai-compatible-response")
            .to_owned(),
        request_id: request.request_id.clone(),
        route_id: route.route_id.clone(),
        provider_family: route.family,
        model_id: route.model_id.clone(),
        model_config_hash: request.config_hash(route)?,
        secret_reference_id: route.secret_reference_id.clone(),
        final_text,
        tool_calls,
        stop_reason,
        usage: parse_openai_usage(response),
        elapsed_ms: response.elapsed_ms,
        retry_count,
        raw_response_hash: raw_response_hash(&response.body)?,
        created_at_ms: now_ms(),
        metadata: Metadata::new(),
    })
}

fn openai_tool_call(value: &Value) -> Option<ModelToolCall> {
    let function = value.get("function")?;
    Some(ModelToolCall {
        tool_call_id: value.get("id")?.as_str()?.to_owned(),
        tool_name: function.get("name")?.as_str()?.to_owned(),
        arguments: parse_tool_arguments(function.get("arguments")),
    })
}

fn parse_openai_usage(response: &ProviderHttpResponse) -> ModelUsage {
    let usage = response.body.get("usage").unwrap_or(&Value::Null);
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(input_tokens + output_tokens);
    let cached_input_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    ModelUsage {
        input_tokens,
        output_tokens,
        total_tokens,
        cached_input_tokens,
        estimated_cost_micro_usd: 0,
    }
}

fn parse_anthropic_response(
    route: &ModelProviderRoute,
    request: &ModelRequest,
    response: &ProviderHttpResponse,
    retry_count: u32,
) -> ModelAdapterResult<ModelResponse> {
    let content = response
        .body
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| parse_error(route, "Anthropic response missing content blocks"))?;
    let mut text_blocks = Vec::new();
    let mut tool_calls = Vec::new();
    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    text_blocks.push(text.to_owned());
                }
            }
            Some("tool_use") => {
                let Some(tool_call_id) = block.get("id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(tool_name) = block.get("name").and_then(Value::as_str) else {
                    continue;
                };
                tool_calls.push(ModelToolCall {
                    tool_call_id: tool_call_id.to_owned(),
                    tool_name: tool_name.to_owned(),
                    arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
                });
            }
            _ => {}
        }
    }
    let stop_reason_raw = response
        .body
        .get("stop_reason")
        .and_then(Value::as_str)
        .unwrap_or("end_turn");
    let stop_reason = match stop_reason_raw {
        "tool_use" => ModelStopReason::ToolCalls,
        "max_tokens" => ModelStopReason::MaxTokens,
        "refusal" => ModelStopReason::SafetyRefusal,
        "end_turn" | "stop_sequence" => ModelStopReason::Stop,
        _ if !tool_calls.is_empty() => ModelStopReason::ToolCalls,
        _ => ModelStopReason::ProviderError,
    };
    Ok(ModelResponse {
        schema_version: LEGAL_BENCHMARK_PROVIDER_SCHEMA_VERSION,
        response_id: response
            .body
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("anthropic-response")
            .to_owned(),
        request_id: request.request_id.clone(),
        route_id: route.route_id.clone(),
        provider_family: route.family,
        model_id: route.model_id.clone(),
        model_config_hash: request.config_hash(route)?,
        secret_reference_id: route.secret_reference_id.clone(),
        final_text: (!text_blocks.is_empty()).then(|| text_blocks.join("\n")),
        tool_calls,
        stop_reason,
        usage: parse_anthropic_usage(response),
        elapsed_ms: response.elapsed_ms,
        retry_count,
        raw_response_hash: raw_response_hash(&response.body)?,
        created_at_ms: now_ms(),
        metadata: Metadata::new(),
    })
}

fn parse_anthropic_usage(response: &ProviderHttpResponse) -> ModelUsage {
    let usage = response.body.get("usage").unwrap_or(&Value::Null);
    let input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached_input_tokens = usage
        .get("cache_read_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    ModelUsage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        cached_input_tokens,
        estimated_cost_micro_usd: 0,
    }
}

fn parse_tool_arguments(value: Option<&Value>) -> Value {
    match value {
        Some(Value::String(arguments)) if arguments.trim().is_empty() => json!({}),
        Some(Value::String(arguments)) => {
            serde_json::from_str(arguments).unwrap_or_else(|_| json!({"raw_arguments": arguments}))
        }
        Some(other) => other.clone(),
        None => json!({}),
    }
}

fn parse_error(route: &ModelProviderRoute, detail: &str) -> ModelAdapterError {
    ModelAdapterError::new(ModelAdapterFailureKind::ParseError, detail).with_route(route)
}

fn json_contains_context_overflow(value: &Value) -> bool {
    let needle = value.to_string().to_ascii_lowercase();
    needle.contains("context") && (needle.contains("overflow") || needle.contains("length"))
}

fn raw_response_hash(value: &Value) -> ModelAdapterResult<String> {
    stable_json_digest("psionic.legal_benchmark.provider_raw_response.v1", value).map_err(|err| {
        ModelAdapterError::new(
            ModelAdapterFailureKind::InternalError,
            format!("failed to hash provider response: {err}"),
        )
    })
}

fn trim_url(url: String) -> String {
    url.trim_end_matches('/').to_owned()
}

fn now_ms() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ModelRequest {
        ModelRequest::new(
            "req.legal.provider.1",
            vec![
                ModelMessage::new(ModelMessageRole::System, "Use only allowed tools."),
                ModelMessage::new(ModelMessageRole::User, "Draft a short memo."),
            ],
        )
    }

    #[test]
    fn legal_tool_specs_cover_the_closed_tool_surface() {
        let specs = legal_benchmark_model_tool_specs();
        let names = specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec!["shell", "read", "write", "edit", "glob", "grep"]
        );
        assert!(specs.iter().all(|spec| spec.input_schema.is_object()));
    }

    #[test]
    fn openai_adapter_normalizes_tool_calls_and_usage() {
        let route = ModelProviderRoute::openai_compatible(
            "openai.local",
            "http://127.0.0.1:8000/v1",
            "psionic-local",
            Some(String::from("secret.openai.local")),
        );
        let transport = MockHttpTransport::new(vec![Ok(ProviderHttpResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: json!({
                "id": "chatcmpl_1",
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": "{\"root\":\"documents\",\"relative_path\":\"case.txt\",\"prefer_extracted\":true}"
                            }
                        }]
                    }
                }],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 5,
                    "total_tokens": 16,
                    "prompt_tokens_details": {"cached_tokens": 3}
                }
            }),
            elapsed_ms: 42,
        })]);
        let mut adapter =
            OpenAiCompatibleAdapter::new(route, transport, ModelRetryPolicy::default());
        let response = adapter.complete(&request()).expect("openai response");
        assert_eq!(response.stop_reason, ModelStopReason::ToolCalls);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].tool_name, "read");
        assert_eq!(response.usage.input_tokens, 11);
        assert_eq!(response.usage.cached_input_tokens, 3);

        let transport = adapter.into_transport();
        let sent = &transport.requests[0];
        assert_eq!(sent.url, "http://127.0.0.1:8000/v1/chat/completions");
        assert_eq!(
            sent.headers.get("authorization").map(String::as_str),
            Some("Bearer <secret_ref:secret.openai.local>")
        );
        assert!(!sent.body.to_string().contains("sk-"));
    }

    #[test]
    fn anthropic_adapter_normalizes_tool_calls_and_usage() {
        let route = ModelProviderRoute::anthropic(
            "anthropic.hosted",
            "https://api.anthropic.com/v1",
            "claude-sonnet-legal",
            Some(String::from("secret.anthropic.prod")),
        );
        let transport = MockHttpTransport::new(vec![Ok(ProviderHttpResponse {
            status: 200,
            headers: BTreeMap::new(),
            body: json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "stop_reason": "tool_use",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "grep",
                    "input": {
                        "root": "documents",
                        "pattern": "indemnity",
                        "case_sensitive": false,
                        "max_results": 20,
                        "include_hidden": false
                    }
                }],
                "usage": {
                    "input_tokens": 21,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 4
                }
            }),
            elapsed_ms: 77,
        })]);
        let mut adapter = AnthropicAdapter::new(route, transport, ModelRetryPolicy::default());
        let response = adapter.complete(&request()).expect("anthropic response");
        assert_eq!(response.stop_reason, ModelStopReason::ToolCalls);
        assert_eq!(response.tool_calls[0].tool_name, "grep");
        assert_eq!(
            response.tool_calls[0].arguments["pattern"],
            Value::String(String::from("indemnity"))
        );
        assert_eq!(response.usage.total_tokens, 28);

        let transport = adapter.into_transport();
        let sent = &transport.requests[0];
        assert_eq!(sent.url, "https://api.anthropic.com/v1/messages");
        assert_eq!(
            sent.headers.get("x-api-key").map(String::as_str),
            Some("<secret_ref:secret.anthropic.prod>")
        );
        assert!(sent.body["system"].as_str().is_some());
        assert!(!sent.body.to_string().contains("sk-"));
    }

    #[test]
    fn retryable_rate_limit_is_retried_and_counted() {
        let route = ModelProviderRoute::openai_compatible(
            "openai.retry",
            "http://127.0.0.1:8000/v1",
            "retry-model",
            None,
        );
        let transport = MockHttpTransport::new(vec![
            Ok(ProviderHttpResponse {
                status: 429,
                headers: BTreeMap::new(),
                body: json!({"error": {"message": "rate limited"}}),
                elapsed_ms: 5,
            }),
            Ok(ProviderHttpResponse {
                status: 200,
                headers: BTreeMap::new(),
                body: json!({
                    "id": "chatcmpl_2",
                    "choices": [{
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Done."}
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }),
                elapsed_ms: 9,
            }),
        ]);
        let mut adapter =
            OpenAiCompatibleAdapter::new(route, transport, ModelRetryPolicy::default());
        let response = adapter.complete(&request()).expect("retried response");
        assert_eq!(response.retry_count, 1);
        assert_eq!(response.final_text.as_deref(), Some("Done."));
        assert_eq!(adapter.into_transport().requests.len(), 2);
    }

    #[test]
    fn context_overflow_is_structured_provider_error() {
        let route = ModelProviderRoute::openai_compatible(
            "openai.context",
            "http://127.0.0.1:8000/v1",
            "small-context",
            None,
        );
        let transport = MockHttpTransport::new(vec![Ok(ProviderHttpResponse {
            status: 400,
            headers: BTreeMap::new(),
            body: json!({"error": {"message": "context length exceeded"}}),
            elapsed_ms: 3,
        })]);
        let mut adapter =
            OpenAiCompatibleAdapter::new(route, transport, ModelRetryPolicy::default());
        let err = adapter.complete(&request()).expect_err("context overflow");
        assert_eq!(err.kind, ModelAdapterFailureKind::ContextOverflow);
        assert_eq!(err.status_code, Some(400));
        assert!(!err.retryable);
        assert!(err.raw_response_hash.is_some());
    }

    #[test]
    fn mock_adapter_supports_ci_without_live_keys() {
        let route = ModelProviderRoute::mock("mock.ci", "deterministic-legal-model");
        let response = ModelResponse {
            schema_version: LEGAL_BENCHMARK_PROVIDER_SCHEMA_VERSION,
            response_id: String::from("mock.response.1"),
            request_id: String::from("req.legal.provider.1"),
            route_id: route.route_id.clone(),
            provider_family: route.family,
            model_id: route.model_id.clone(),
            model_config_hash: String::from("mock-config-hash"),
            secret_reference_id: None,
            final_text: Some(String::from("submit")),
            tool_calls: Vec::new(),
            stop_reason: ModelStopReason::Stop,
            usage: ModelUsage {
                input_tokens: 10,
                output_tokens: 1,
                total_tokens: 11,
                cached_input_tokens: 0,
                estimated_cost_micro_usd: 0,
            },
            elapsed_ms: 1,
            retry_count: 0,
            raw_response_hash: String::from("mock-raw-hash"),
            created_at_ms: 1,
            metadata: Metadata::new(),
        };
        let mut adapter = MockModelAdapter::new(route, vec![Ok(response.clone())]);
        assert_eq!(
            adapter.complete(&request()).expect("mock response"),
            response
        );
    }
}
