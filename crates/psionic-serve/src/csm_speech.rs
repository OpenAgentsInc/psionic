use std::{
    env,
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use psionic_models::{CsmPythonParityFixture, csm_python_parity_fixture};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::tokio_runtime_telemetry_axum::serve_with_runtime_telemetry;

pub const CSM_SPEECH_MODEL_ID: &str = "sesame/csm-1b";
pub const CSM_SPEECH_PRODUCT_ID: &str = "psionic.csm_speech";
pub const CSM_SPEECH_ROUTE_OPENAI: &str = "/v1/audio/speech";
pub const CSM_SPEECH_ROUTE_PSIONIC: &str = "/psionic/csm/speech";
pub const CSM_SPEECH_RESPONSE_FORMAT_WAV: &str = "wav";
pub const CSM_SPEECH_SERVED_BACKEND: &str = "cpu";
pub const CSM_SPEECH_EXECUTION_MODE: &str = "native";
pub const CSM_SPEECH_EXECUTION_ENGINE: &str = "rust_csm_pending";

#[derive(Clone, Debug)]
pub struct CsmSpeechServerConfig {
    pub host: String,
    pub port: u16,
    pub model_id: String,
}

impl Default for CsmSpeechServerConfig {
    fn default() -> Self {
        Self {
            host: String::from("127.0.0.1"),
            port: 8081,
            model_id: String::from(CSM_SPEECH_MODEL_ID),
        }
    }
}

impl CsmSpeechServerConfig {
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(host) = env::var("PSIONIC_CSM_HOST") {
            config.host = host;
        }
        if let Ok(port) = env::var("PSIONIC_CSM_PORT")
            && let Ok(port) = port.parse::<u16>()
        {
            config.port = port;
        }
        if let Ok(model_id) = env::var("PSIONIC_CSM_MODEL_ID") {
            config.model_id = model_id;
        }
        config
    }

    pub fn socket_addr(&self) -> Result<SocketAddr, CsmSpeechServerError> {
        let host = self.host.parse::<IpAddr>().map_err(|error| {
            CsmSpeechServerError::Config(format!("invalid CSM host `{}`: {error}", self.host))
        })?;
        Ok(SocketAddr::new(host, self.port))
    }
}

#[derive(Clone)]
pub struct CsmSpeechServer {
    state: Arc<CsmSpeechState>,
}

#[derive(Clone)]
struct CsmSpeechState {
    model_id: String,
    fixture: CsmPythonParityFixture,
}

impl CsmSpeechServer {
    pub fn from_config(config: CsmSpeechServerConfig) -> Result<Self, CsmSpeechServerError> {
        let fixture = csm_python_parity_fixture().map_err(|error| {
            CsmSpeechServerError::Config(format!("failed to load CSM parity fixture: {error}"))
        })?;
        Ok(Self {
            state: Arc::new(CsmSpeechState {
                model_id: config.model_id,
                fixture,
            }),
        })
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(csm_health))
            .route("/v1/models", get(csm_models))
            .route(CSM_SPEECH_ROUTE_OPENAI, post(csm_audio_speech))
            .route(CSM_SPEECH_ROUTE_PSIONIC, post(csm_audio_speech))
            .with_state(Arc::clone(&self.state))
    }

    pub async fn serve(&self, listener: TcpListener) -> Result<(), CsmSpeechServerError> {
        serve_with_runtime_telemetry(listener, self.router())
            .await
            .map_err(CsmSpeechServerError::Io)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CsmSpeechServerError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Config(String),
}

#[derive(Clone, Debug, Serialize)]
struct CsmSpeechHealthResponse {
    status: &'static str,
    model: String,
    capability: &'static str,
    served_backend: &'static str,
    execution_mode: &'static str,
    execution_engine: &'static str,
    supported_endpoints: Vec<&'static str>,
    supported_response_formats: Vec<&'static str>,
    voice_profiles: Vec<CsmVoiceProfilePublication>,
    artifact_digests: CsmArtifactDigestPublication,
    execution_refusal: CsmSpeechRefusalPublication,
}

#[derive(Clone, Debug, Serialize)]
struct CsmSpeechModelsResponse {
    object: &'static str,
    data: Vec<CsmSpeechModelCard>,
}

#[derive(Clone, Debug, Serialize)]
struct CsmSpeechModelCard {
    id: String,
    object: &'static str,
    owned_by: &'static str,
    psionic_capability: &'static str,
    psionic_supported_endpoints: Vec<&'static str>,
    psionic_supported_response_formats: Vec<&'static str>,
    psionic_served_backend: &'static str,
    psionic_execution_mode: &'static str,
    psionic_execution_engine: &'static str,
    psionic_voice_profiles: Vec<CsmVoiceProfilePublication>,
    psionic_artifact_digests: CsmArtifactDigestPublication,
    psionic_execution_refusal: CsmSpeechRefusalPublication,
}

#[derive(Clone, Debug, Serialize)]
struct CsmVoiceProfilePublication {
    id: String,
    speaker: u32,
    source: &'static str,
    prompt_audio_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct CsmArtifactDigestPublication {
    csm_config_digest: String,
    csm_model_digest: String,
    llama_tokenizer_digest: String,
    mimi_weight_digest: String,
}

#[derive(Clone, Debug, Serialize)]
struct CsmSpeechRefusalPublication {
    code: &'static str,
    reason: &'static str,
    pending_phases: Vec<&'static str>,
}

fn pending_execution_refusal() -> CsmSpeechRefusalPublication {
    CsmSpeechRefusalPublication {
        code: "rust_csm_generation_not_implemented",
        reason: "Rust CSM tokenizer, Mimi codec, safetensors binding, and generation loop are still pending",
        pending_phases: vec![
            "rust_tokenizer_prompt_framing",
            "rust_mimi_codec",
            "rust_csm_generation_loop",
        ],
    }
}

fn voice_profiles(fixture: &CsmPythonParityFixture) -> Vec<CsmVoiceProfilePublication> {
    fixture
        .prompts
        .iter()
        .map(|prompt| CsmVoiceProfilePublication {
            id: prompt.profile_id.clone(),
            speaker: prompt.speaker,
            source: "committed_parity_fixture",
            prompt_audio_sha256: prompt.audio_sha256.clone(),
        })
        .collect()
}

fn artifact_digests(fixture: &CsmPythonParityFixture) -> CsmArtifactDigestPublication {
    CsmArtifactDigestPublication {
        csm_config_digest: fixture.model.csm_config_digest.clone(),
        csm_model_digest: fixture.model.csm_model_digest.clone(),
        llama_tokenizer_digest: fixture.model.llama_tokenizer_digest.clone(),
        mimi_weight_digest: fixture.model.mimi_weight_digest.clone(),
    }
}

async fn csm_health(State(state): State<Arc<CsmSpeechState>>) -> Json<CsmSpeechHealthResponse> {
    Json(CsmSpeechHealthResponse {
        status: "degraded",
        model: state.model_id.clone(),
        capability: "speech_generation",
        served_backend: CSM_SPEECH_SERVED_BACKEND,
        execution_mode: CSM_SPEECH_EXECUTION_MODE,
        execution_engine: CSM_SPEECH_EXECUTION_ENGINE,
        supported_endpoints: vec![CSM_SPEECH_ROUTE_OPENAI, CSM_SPEECH_ROUTE_PSIONIC],
        supported_response_formats: vec![CSM_SPEECH_RESPONSE_FORMAT_WAV],
        voice_profiles: voice_profiles(&state.fixture),
        artifact_digests: artifact_digests(&state.fixture),
        execution_refusal: pending_execution_refusal(),
    })
}

async fn csm_models(State(state): State<Arc<CsmSpeechState>>) -> Json<CsmSpeechModelsResponse> {
    Json(CsmSpeechModelsResponse {
        object: "list",
        data: vec![CsmSpeechModelCard {
            id: state.model_id.clone(),
            object: "model",
            owned_by: "psionic",
            psionic_capability: "speech_generation",
            psionic_supported_endpoints: vec![CSM_SPEECH_ROUTE_OPENAI, CSM_SPEECH_ROUTE_PSIONIC],
            psionic_supported_response_formats: vec![CSM_SPEECH_RESPONSE_FORMAT_WAV],
            psionic_served_backend: CSM_SPEECH_SERVED_BACKEND,
            psionic_execution_mode: CSM_SPEECH_EXECUTION_MODE,
            psionic_execution_engine: CSM_SPEECH_EXECUTION_ENGINE,
            psionic_voice_profiles: voice_profiles(&state.fixture),
            psionic_artifact_digests: artifact_digests(&state.fixture),
            psionic_execution_refusal: pending_execution_refusal(),
        }],
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CsmSpeechRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    voice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    voice_profile_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    psionic_csm: Option<CsmGenerationParams>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct CsmGenerationParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_audio_length_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context_policy: Option<String>,
}

#[derive(Clone, Debug)]
struct ValidatedCsmSpeechRequest {
    model: String,
    voice_profile_id: String,
}

impl ValidatedCsmSpeechRequest {
    fn from_request(
        state: &CsmSpeechState,
        request: CsmSpeechRequest,
    ) -> Result<Self, CsmSpeechHttpError> {
        let model = request.model.unwrap_or_else(|| state.model_id.clone());
        if model != state.model_id {
            return Err(CsmSpeechHttpError::model_unavailable());
        }
        if request.input.trim().is_empty() {
            return Err(CsmSpeechHttpError::invalid_request(
                "input text must not be empty",
                "empty_input",
            ));
        }
        if request.stream {
            return Err(CsmSpeechHttpError::unsupported_stream());
        }
        let response_format = request
            .response_format
            .unwrap_or_else(|| String::from(CSM_SPEECH_RESPONSE_FORMAT_WAV));
        if response_format != CSM_SPEECH_RESPONSE_FORMAT_WAV {
            return Err(CsmSpeechHttpError::unsupported_format());
        }
        let voice_profile_id = request
            .voice_profile_id
            .or(request.voice)
            .unwrap_or_else(|| String::from("conversational_a"));
        if !state
            .fixture
            .prompts
            .iter()
            .any(|prompt| prompt.profile_id == voice_profile_id)
        {
            return Err(CsmSpeechHttpError::missing_voice_profile());
        }
        if let Some(params) = request.psionic_csm {
            validate_params(&params)?;
        }
        Ok(Self {
            model,
            voice_profile_id,
        })
    }
}

fn validate_params(params: &CsmGenerationParams) -> Result<(), CsmSpeechHttpError> {
    if let Some(temperature) = params.temperature
        && !(0.0..=2.0).contains(&temperature)
    {
        return Err(CsmSpeechHttpError::invalid_request(
            "temperature must be between 0 and 2",
            "invalid_temperature",
        ));
    }
    if let Some(top_k) = params.top_k
        && top_k == 0
    {
        return Err(CsmSpeechHttpError::invalid_request(
            "top_k must be greater than zero",
            "invalid_top_k",
        ));
    }
    if let Some(max_audio_length_ms) = params.max_audio_length_ms
        && !(80..=90_000).contains(&max_audio_length_ms)
    {
        return Err(CsmSpeechHttpError::invalid_request(
            "max_audio_length_ms must be between 80 and 90000",
            "invalid_max_audio_length_ms",
        ));
    }
    if let Some(context_policy) = params.context_policy.as_deref()
        && !matches!(context_policy, "prompt_profile_only" | "none")
    {
        return Err(CsmSpeechHttpError::invalid_request(
            "context_policy must be prompt_profile_only or none",
            "invalid_context_policy",
        ));
    }
    Ok(())
}

async fn csm_audio_speech(
    State(state): State<Arc<CsmSpeechState>>,
    Json(request): Json<CsmSpeechRequest>,
) -> Result<Response, CsmSpeechHttpError> {
    let validated = ValidatedCsmSpeechRequest::from_request(&state, request)?;
    let mut response = CsmSpeechHttpError::rust_generation_not_implemented().into_response();
    insert_csm_execution_headers(response.headers_mut(), &state, &validated);
    Ok(response)
}

fn insert_csm_execution_headers(
    headers: &mut HeaderMap,
    state: &CsmSpeechState,
    request: &ValidatedCsmSpeechRequest,
) {
    insert_header(headers, "x-psionic-model-id", request.model.as_str());
    insert_header(
        headers,
        "x-psionic-served-backend",
        CSM_SPEECH_SERVED_BACKEND,
    );
    insert_header(
        headers,
        "x-psionic-execution-mode",
        CSM_SPEECH_EXECUTION_MODE,
    );
    insert_header(
        headers,
        "x-psionic-execution-engine",
        CSM_SPEECH_EXECUTION_ENGINE,
    );
    insert_header(
        headers,
        "x-psionic-csm-voice-profile-id",
        request.voice_profile_id.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-artifact-csm-config-digest",
        state.fixture.model.csm_config_digest.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-artifact-csm-model-digest",
        state.fixture.model.csm_model_digest.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-artifact-llama-tokenizer-digest",
        state.fixture.model.llama_tokenizer_digest.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-artifact-mimi-weight-digest",
        state.fixture.model.mimi_weight_digest.as_str(),
    );
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

#[derive(Clone, Debug, Serialize)]
struct CsmSpeechErrorEnvelope {
    error: CsmSpeechErrorBody,
}

#[derive(Clone, Debug, Serialize)]
struct CsmSpeechErrorBody {
    message: &'static str,
    #[serde(rename = "type")]
    kind: &'static str,
    code: &'static str,
    served_backend: &'static str,
    execution_mode: &'static str,
    execution_engine: &'static str,
}

#[derive(Clone, Debug)]
struct CsmSpeechHttpError {
    status: StatusCode,
    message: &'static str,
    kind: &'static str,
    code: &'static str,
}

impl CsmSpeechHttpError {
    const fn invalid_request(message: &'static str, code: &'static str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message,
            kind: "invalid_request_error",
            code,
        }
    }

    const fn unsupported_format() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: "CSM speech currently supports response_format=wav only",
            kind: "invalid_request_error",
            code: "unsupported_response_format",
        }
    }

    const fn unsupported_stream() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: "CSM speech streaming is pending the Rust streaming runtime",
            kind: "invalid_request_error",
            code: "unsupported_stream",
        }
    }

    const fn missing_voice_profile() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: "requested CSM voice profile is not available",
            kind: "invalid_request_error",
            code: "voice_profile_unavailable",
        }
    }

    const fn model_unavailable() -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: "requested CSM model is not available on this Psionic speech server",
            kind: "not_found_error",
            code: "model_unavailable",
        }
    }

    const fn rust_generation_not_implemented() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "Rust CSM generation is not implemented yet; tokenizer, Mimi, and model execution phases are required before audio bytes can be served",
            kind: "backend_unavailable",
            code: "rust_csm_generation_not_implemented",
        }
    }
}

impl IntoResponse for CsmSpeechHttpError {
    fn into_response(self) -> Response {
        let mut response = (
            self.status,
            Json(CsmSpeechErrorEnvelope {
                error: CsmSpeechErrorBody {
                    message: self.message,
                    kind: self.kind,
                    code: self.code,
                    served_backend: CSM_SPEECH_SERVED_BACKEND,
                    execution_mode: CSM_SPEECH_EXECUTION_MODE,
                    execution_engine: CSM_SPEECH_EXECUTION_ENGINE,
                },
            }),
        )
            .into_response();
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn csm_speech_route_refuses_until_rust_generation_lands()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = CsmSpeechServer::from_config(CsmSpeechServerConfig::default())?;
        let response = server
            .router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(CSM_SPEECH_ROUTE_OPENAI)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&serde_json::json!({
                        "model": CSM_SPEECH_MODEL_ID,
                        "input": "hello from psionic",
                        "voice_profile_id": "conversational_a",
                        "response_format": "wav",
                        "psionic_csm": {
                            "temperature": 0.1,
                            "top_k": 1,
                            "max_audio_length_ms": 250
                        }
                    }))?))?,
            )
            .await?;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get("x-psionic-execution-engine")
                .and_then(|value| value.to_str().ok()),
            Some(CSM_SPEECH_EXECUTION_ENGINE)
        );
        assert_eq!(
            response
                .headers()
                .get("x-psionic-csm-voice-profile-id")
                .and_then(|value| value.to_str().ok()),
            Some("conversational_a")
        );
        let body = to_bytes(response.into_body(), usize::MAX).await?;
        let payload: serde_json::Value = serde_json::from_slice(&body)?;
        assert_eq!(
            payload["error"]["code"],
            serde_json::json!("rust_csm_generation_not_implemented")
        );
        assert!(!String::from_utf8(body.to_vec())?.contains("/Users/"));
        Ok(())
    }

    #[tokio::test]
    async fn csm_speech_route_refuses_missing_voice_profile()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = CsmSpeechServer::from_config(CsmSpeechServerConfig::default())?;
        let response = server
            .router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(CSM_SPEECH_ROUTE_PSIONIC)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&serde_json::json!({
                        "input": "hello",
                        "voice_profile_id": "missing"
                    }))?))?,
            )
            .await?;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await?;
        let payload: serde_json::Value = serde_json::from_slice(&body)?;
        assert_eq!(
            payload["error"]["code"],
            serde_json::json!("voice_profile_unavailable")
        );
        Ok(())
    }

    #[tokio::test]
    async fn csm_health_publishes_voice_profiles_artifacts_and_refusal_truth()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = CsmSpeechServer::from_config(CsmSpeechServerConfig::default())?;
        let response = server
            .router()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .expect("health request"),
            )
            .await?;

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await?;
        let payload: serde_json::Value = serde_json::from_slice(&body)?;
        assert_eq!(payload["execution_engine"], CSM_SPEECH_EXECUTION_ENGINE);
        assert_eq!(
            payload["execution_refusal"]["code"],
            serde_json::json!("rust_csm_generation_not_implemented")
        );
        assert!(
            payload["voice_profiles"]
                .as_array()
                .is_some_and(|profiles| profiles
                    .iter()
                    .any(|profile| { profile["id"] == serde_json::json!("conversational_a") }))
        );
        Ok(())
    }

    #[tokio::test]
    async fn csm_models_publish_openai_and_psionic_speech_routes()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = CsmSpeechServer::from_config(CsmSpeechServerConfig::default())?;
        let response = server
            .router()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .expect("models request"),
            )
            .await?;

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await?;
        let payload: serde_json::Value = serde_json::from_slice(&body)?;
        let endpoints = payload["data"][0]["psionic_supported_endpoints"]
            .as_array()
            .expect("endpoints");
        assert!(endpoints.contains(&serde_json::json!(CSM_SPEECH_ROUTE_OPENAI)));
        assert!(endpoints.contains(&serde_json::json!(CSM_SPEECH_ROUTE_PSIONIC)));
        Ok(())
    }
}
