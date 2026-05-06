use std::{
    env,
    net::{IpAddr, SocketAddr},
    sync::{Arc, Mutex},
    time::Instant,
};

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use psionic_models::{
    CSM_CPU_EXECUTION_ENGINE, CSM_LYRA_DEFAULT_FEMALE_PROFILE_ID, CSM_WATERMARK_OPERATOR_DOGFOOD,
    CsmCapabilityRefusal, CsmContextWindowPolicy, CsmCpuGenerationRequest, CsmCpuGenerator,
    CsmFrontendError, CsmGenerationWindow, CsmLlamaTextTokenizer, CsmMimiDecoder,
    CsmModelArtifactDescriptor, CsmModelConfig, CsmPromptSegment, CsmPythonParityFixture,
    CsmSamplingStrategy, CsmVoiceProfileGovernanceManifest, csm_build_prompt_frame_plan,
    csm_default_config_candidates, csm_default_mimi_weight_candidates,
    csm_default_model_weight_candidates, csm_python_parity_fixture,
    csm_reference_audio_encoding_refusal, csm_voice_profile_governance_manifest,
};
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
pub const CSM_SPEECH_EXECUTION_ENGINE: &str = CSM_CPU_EXECUTION_ENGINE;
pub const CSM_SPEECH_RESIDENCY_MODE: &str = "warm_cpu";
const CSM_SPEECH_DEFAULT_AUDIO_LENGTH_MS: u64 = 240;
const CSM_SPEECH_MAX_AUDIO_LENGTH_MS: u64 = 2_000;
const CSM_SPEECH_STREAM_BOUNDARY: &str = "psionic-csm-stream";
const CSM_SPEECH_STREAM_CHUNK_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug)]
pub struct CsmSpeechServerConfig {
    pub host: String,
    pub port: u16,
    pub model_id: String,
    pub runtime_enabled: bool,
    pub backend: String,
}

impl Default for CsmSpeechServerConfig {
    fn default() -> Self {
        Self {
            host: String::from("127.0.0.1"),
            port: 8081,
            model_id: String::from(CSM_SPEECH_MODEL_ID),
            runtime_enabled: true,
            backend: String::from(CSM_SPEECH_SERVED_BACKEND),
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
        if let Ok(runtime) = env::var("PSIONIC_CSM_RUNTIME") {
            config.runtime_enabled = !matches!(
                runtime.to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "disabled"
            );
        }
        if let Ok(backend) = env::var("PSIONIC_CSM_BACKEND") {
            config.backend = backend;
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

struct CsmSpeechState {
    model_id: String,
    fixture: CsmPythonParityFixture,
    descriptor: CsmModelArtifactDescriptor,
    governance: CsmVoiceProfileGovernanceManifest,
    runtime: Mutex<CsmSpeechRuntimeSlot>,
}

impl CsmSpeechServer {
    pub fn from_config(config: CsmSpeechServerConfig) -> Result<Self, CsmSpeechServerError> {
        let fixture = csm_python_parity_fixture().map_err(|error| {
            CsmSpeechServerError::Config(format!("failed to load CSM parity fixture: {error}"))
        })?;
        let descriptor = CsmModelArtifactDescriptor::from_fixture(&fixture).map_err(|error| {
            CsmSpeechServerError::Config(format!("failed to build CSM descriptor: {error}"))
        })?;
        let governance = csm_voice_profile_governance_manifest().map_err(|error| {
            CsmSpeechServerError::Config(format!(
                "failed to load CSM voice-profile governance manifest: {error}"
            ))
        })?;
        governance
            .validate_against_descriptor(&descriptor)
            .map_err(|error| {
                CsmSpeechServerError::Config(format!(
                    "failed to validate CSM voice-profile governance manifest: {error}"
                ))
            })?;
        let runtime = CsmSpeechRuntimeSlot::load(&config, &fixture, &descriptor);
        Ok(Self {
            state: Arc::new(CsmSpeechState {
                model_id: config.model_id,
                fixture,
                descriptor,
                governance,
                runtime: Mutex::new(runtime),
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

struct CsmSpeechRuntimeSlot {
    status: CsmRuntimeStatus,
    engine: Option<CsmResidentSpeechEngine>,
}

struct CsmResidentSpeechEngine {
    tokenizer: CsmLlamaTextTokenizer,
    generator: CsmCpuGenerator,
    mimi: CsmMimiDecoder,
}

#[derive(Clone, Debug, Serialize)]
struct CsmRuntimeStatus {
    residency: &'static str,
    backend: &'static str,
    execution_engine: &'static str,
    state: &'static str,
    loaded_at_unix_ms: Option<u128>,
    load_latency_ms: Option<u128>,
    tokenizer_loaded: bool,
    csm_config_loaded: bool,
    csm_model_loaded: bool,
    mimi_loaded: bool,
    accelerated_backend: &'static str,
    refusal: Option<CsmSpeechRefusalPublication>,
}

impl CsmSpeechRuntimeSlot {
    fn load(
        config: &CsmSpeechServerConfig,
        fixture: &CsmPythonParityFixture,
        descriptor: &CsmModelArtifactDescriptor,
    ) -> Self {
        if !config.runtime_enabled {
            return Self::unavailable(
                "runtime_disabled",
                "CSM runtime loading is disabled by server configuration",
                vec!["enable_rust_csm_runtime"],
            );
        }
        if config.backend != CSM_SPEECH_SERVED_BACKEND {
            return Self::unavailable(
                "unsupported_backend",
                "CSM speech currently supports the Rust CPU backend only",
                vec!["cpu_backend_or_future_accelerated_backend"],
            );
        }

        let started = Instant::now();
        let loaded_at_unix_ms = current_unix_ms();

        let tokenizer = match CsmLlamaTextTokenizer::from_default_hf_cache(Some(
            &descriptor.digests.llama_tokenizer_digest,
        )) {
            Ok(tokenizer) => tokenizer,
            Err(_) => {
                return Self::unavailable(
                    "llama_tokenizer_unavailable",
                    "matching Llama tokenizer artifact is unavailable in the local Hugging Face cache",
                    vec!["hydrate_gated_hf_artifacts"],
                );
            }
        };

        let Some(config_path) =
            csm_default_config_candidates(Some(&descriptor.digests.csm_config_digest))
                .into_iter()
                .next()
        else {
            return Self::unavailable(
                "csm_config_unavailable",
                "matching CSM config artifact is unavailable in the local Hugging Face cache",
                vec!["hydrate_gated_hf_artifacts"],
            );
        };
        let (model_config, config_digest) = match CsmModelConfig::from_json_file_with_digest(
            &config_path,
            Some(&descriptor.digests.csm_config_digest),
        ) {
            Ok(config) => config,
            Err(_) => {
                return Self::unavailable(
                    "csm_config_invalid",
                    "matching CSM config artifact failed Rust digest or contract validation",
                    vec!["refresh_csm_artifact_cache"],
                );
            }
        };

        let Some(model_path) =
            csm_default_model_weight_candidates(Some(&descriptor.digests.csm_model_digest))
                .into_iter()
                .next()
        else {
            return Self::unavailable(
                "csm_model_unavailable",
                "matching CSM model artifact is unavailable in the local Hugging Face cache",
                vec!["hydrate_gated_hf_artifacts"],
            );
        };
        let generator = match CsmCpuGenerator::from_safetensors_file(
            &model_config,
            config_digest,
            &model_path,
            Some(&descriptor.digests.csm_model_digest),
        ) {
            Ok(generator) => generator,
            Err(_) => {
                return Self::unavailable(
                    "csm_model_load_failed",
                    "matching CSM model artifact failed Rust loading or digest validation",
                    vec!["refresh_csm_artifact_cache"],
                );
            }
        };

        let Some(mimi_path) =
            csm_default_mimi_weight_candidates(Some(&fixture.model.mimi_weight_digest))
                .into_iter()
                .next()
        else {
            return Self::unavailable(
                "mimi_model_unavailable",
                "matching Mimi artifact is unavailable in the local Hugging Face cache",
                vec!["hydrate_gated_hf_artifacts"],
            );
        };
        let mimi = match CsmMimiDecoder::from_safetensors_file(
            &mimi_path,
            Some(&fixture.model.mimi_weight_digest),
        ) {
            Ok(mimi) => mimi,
            Err(_) => {
                return Self::unavailable(
                    "mimi_model_load_failed",
                    "matching Mimi artifact failed Rust loading or digest validation",
                    vec!["refresh_mimi_artifact_cache"],
                );
            }
        };

        let load_latency_ms = started.elapsed().as_millis();
        Self {
            status: CsmRuntimeStatus {
                residency: CSM_SPEECH_RESIDENCY_MODE,
                backend: CSM_SPEECH_SERVED_BACKEND,
                execution_engine: CSM_SPEECH_EXECUTION_ENGINE,
                state: "ready",
                loaded_at_unix_ms: Some(loaded_at_unix_ms),
                load_latency_ms: Some(load_latency_ms),
                tokenizer_loaded: true,
                csm_config_loaded: true,
                csm_model_loaded: true,
                mimi_loaded: true,
                accelerated_backend: "unavailable_fail_closed",
                refusal: None,
            },
            engine: Some(CsmResidentSpeechEngine {
                tokenizer,
                generator,
                mimi,
            }),
        }
    }

    fn unavailable(
        code: &'static str,
        reason: &'static str,
        pending_phases: Vec<&'static str>,
    ) -> Self {
        Self {
            status: CsmRuntimeStatus {
                residency: "unavailable",
                backend: CSM_SPEECH_SERVED_BACKEND,
                execution_engine: CSM_SPEECH_EXECUTION_ENGINE,
                state: "unavailable",
                loaded_at_unix_ms: None,
                load_latency_ms: None,
                tokenizer_loaded: false,
                csm_config_loaded: false,
                csm_model_loaded: false,
                mimi_loaded: false,
                accelerated_backend: "unavailable_fail_closed",
                refusal: Some(CsmSpeechRefusalPublication {
                    code,
                    reason,
                    pending_phases,
                }),
            },
            engine: None,
        }
    }

    #[cfg(test)]
    fn is_ready(&self) -> bool {
        self.engine.is_some()
    }
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
    artifact_descriptor: CsmArtifactDescriptorPublication,
    runtime: CsmRuntimeStatus,
    codec_capabilities: CsmCodecCapabilityPublication,
    safety_capabilities: CsmSafetyCapabilityPublication,
    #[serde(skip_serializing_if = "Option::is_none")]
    execution_refusal: Option<CsmSpeechRefusalPublication>,
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
    psionic_artifact_descriptor: CsmArtifactDescriptorPublication,
    psionic_runtime: CsmRuntimeStatus,
    psionic_codec_capabilities: CsmCodecCapabilityPublication,
    psionic_safety_capabilities: CsmSafetyCapabilityPublication,
    #[serde(skip_serializing_if = "Option::is_none")]
    psionic_execution_refusal: Option<CsmSpeechRefusalPublication>,
}

#[derive(Clone, Debug, Serialize)]
struct CsmVoiceProfilePublication {
    id: String,
    display_name: String,
    approval_status: String,
    runtime_admission: String,
    source_prompt_profile_id: String,
    speaker: u32,
    source: &'static str,
    prompt_transcript_sha256: String,
    prompt_audio_sha256: String,
    mimi_tokens_sha256: Option<String>,
    prompt_codebooks: Option<CsmPromptCodebookPublication>,
    consent_posture: String,
    allowed_product_surfaces: Vec<String>,
    disallowed_product_surfaces: Vec<String>,
    retention_policy: String,
    redaction_policy: String,
    watermarking: String,
    watermarking_refusal_code: String,
}

#[derive(Clone, Debug, Serialize)]
struct CsmPromptCodebookPublication {
    source: String,
    sample_rate_hz: u32,
    codebook_count: usize,
    frame_count: usize,
    prefix_frame_count: usize,
    prefix_codebook_count: usize,
    tokens_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct CsmArtifactDigestPublication {
    csm_config_digest: String,
    csm_model_digest: String,
    llama_tokenizer_digest: String,
    mimi_weight_digest: String,
}

#[derive(Clone, Debug, Serialize)]
struct CsmFrameContractPublication {
    frame_lanes: usize,
    audio_codebook_lanes: usize,
    text_lane_index: usize,
    max_seq_len: usize,
    generation_frame_ms: u64,
    sample_rate_hz: u32,
}

#[derive(Clone, Debug, Serialize)]
struct CsmArtifactDescriptorPublication {
    model_id: String,
    csm_repo: String,
    llama_tokenizer_repo: String,
    mimi_repo: String,
    mimi_weight: String,
    frame_contract: CsmFrameContractPublication,
}

#[derive(Clone, Debug, Serialize)]
struct CsmCodecCapabilityPublication {
    mimi_decode: &'static str,
    mimi_decode_engine: &'static str,
    reference_audio_encode: &'static str,
    reference_audio_encode_refusal: CsmCapabilityRefusal,
}

#[derive(Clone, Debug, Serialize)]
struct CsmSafetyCapabilityPublication {
    watermarking: &'static str,
    watermarking_refusal: CsmCapabilityRefusal,
}

#[derive(Clone, Debug, Serialize)]
struct CsmSpeechRefusalPublication {
    code: &'static str,
    reason: &'static str,
    pending_phases: Vec<&'static str>,
}

fn voice_profiles(
    governance: &CsmVoiceProfileGovernanceManifest,
    descriptor: &CsmModelArtifactDescriptor,
) -> Vec<CsmVoiceProfilePublication> {
    governance
        .profiles
        .iter()
        .filter_map(|profile| {
            let source = descriptor
                .voice_profiles
                .iter()
                .find(|candidate| candidate.profile_id == profile.source_prompt_profile_id)?;
            Some(CsmVoiceProfilePublication {
                id: profile.profile_id.clone(),
                display_name: profile.display_name.clone(),
                approval_status: profile.approval_status.clone(),
                runtime_admission: profile.runtime_admission.clone(),
                source_prompt_profile_id: profile.source_prompt_profile_id.clone(),
                speaker: profile.speaker,
                source: "committed_voice_governance_manifest",
                prompt_transcript_sha256: profile.prompt_transcript_sha256.clone(),
                prompt_audio_sha256: profile.prompt_audio_sha256.clone(),
                mimi_tokens_sha256: profile.prompt_codebook_tokens_sha256.clone(),
                prompt_codebooks: source.prompt_codebooks.as_ref().map(|codebooks| {
                    CsmPromptCodebookPublication {
                        source: codebooks.source.clone(),
                        sample_rate_hz: codebooks.sample_rate_hz,
                        codebook_count: codebooks.codebook_count,
                        frame_count: codebooks.frame_count,
                        prefix_frame_count: codebooks.prefix_frame_count,
                        prefix_codebook_count: codebooks.prefix_codebook_count,
                        tokens_sha256: codebooks.tokens_sha256.clone(),
                    }
                }),
                consent_posture: profile.consent_posture.clone(),
                allowed_product_surfaces: profile.allowed_product_surfaces.clone(),
                disallowed_product_surfaces: profile.disallowed_product_surfaces.clone(),
                retention_policy: profile.retention_policy.clone(),
                redaction_policy: profile.redaction_policy.clone(),
                watermarking: profile.watermark_policy.status.clone(),
                watermarking_refusal_code: profile.watermark_policy.refusal_code.clone(),
            })
        })
        .collect()
}

fn artifact_digests(descriptor: &CsmModelArtifactDescriptor) -> CsmArtifactDigestPublication {
    CsmArtifactDigestPublication {
        csm_config_digest: descriptor.digests.csm_config_digest.clone(),
        csm_model_digest: descriptor.digests.csm_model_digest.clone(),
        llama_tokenizer_digest: descriptor.digests.llama_tokenizer_digest.clone(),
        mimi_weight_digest: descriptor.digests.mimi_weight_digest.clone(),
    }
}

fn artifact_descriptor(
    descriptor: &CsmModelArtifactDescriptor,
) -> CsmArtifactDescriptorPublication {
    CsmArtifactDescriptorPublication {
        model_id: descriptor.model_id.clone(),
        csm_repo: descriptor.csm_repo.clone(),
        llama_tokenizer_repo: descriptor.llama_tokenizer_repo.clone(),
        mimi_repo: descriptor.mimi_repo.clone(),
        mimi_weight: descriptor.mimi_weight.clone(),
        frame_contract: CsmFrameContractPublication {
            frame_lanes: descriptor.frame_contract.frame_lanes,
            audio_codebook_lanes: descriptor.frame_contract.audio_codebook_lanes,
            text_lane_index: descriptor.frame_contract.text_lane_index,
            max_seq_len: descriptor.frame_contract.max_seq_len,
            generation_frame_ms: descriptor.frame_contract.generation_frame_ms,
            sample_rate_hz: descriptor.frame_contract.sample_rate_hz,
        },
    }
}

fn codec_capabilities() -> CsmCodecCapabilityPublication {
    CsmCodecCapabilityPublication {
        mimi_decode: "implemented",
        mimi_decode_engine: "rust_moshi_mimi_cpu",
        reference_audio_encode: "refused",
        reference_audio_encode_refusal: csm_reference_audio_encoding_refusal(),
    }
}

fn safety_capabilities() -> CsmSafetyCapabilityPublication {
    CsmSafetyCapabilityPublication {
        watermarking: CSM_WATERMARK_OPERATOR_DOGFOOD,
        watermarking_refusal: CsmCapabilityRefusal {
            code: "csm_watermarking_unavailable".to_string(),
            reason: "CSM speech watermarking is not implemented in the Rust serving path; output is admitted only for OpenAgents-operated Lyra dogfood and remains unavailable for arbitrary public voice cloning".to_string(),
            required_phase: "private_watermark_or_equivalent_voice_safety_control_before_public_voice_clone".to_string(),
        },
    }
}

fn runtime_snapshot(state: &CsmSpeechState) -> CsmRuntimeStatus {
    match state.runtime.lock() {
        Ok(runtime) => runtime.status.clone(),
        Err(_) => {
            CsmSpeechRuntimeSlot::unavailable(
                "runtime_lock_poisoned",
                "CSM runtime lock is poisoned and requests fail closed",
                vec!["restart_server"],
            )
            .status
        }
    }
}

async fn csm_health(State(state): State<Arc<CsmSpeechState>>) -> Json<CsmSpeechHealthResponse> {
    let runtime = runtime_snapshot(&state);
    Json(CsmSpeechHealthResponse {
        status: if runtime.state == "ready" {
            "ok"
        } else {
            "degraded"
        },
        model: state.model_id.clone(),
        capability: "speech_generation",
        served_backend: CSM_SPEECH_SERVED_BACKEND,
        execution_mode: CSM_SPEECH_EXECUTION_MODE,
        execution_engine: CSM_SPEECH_EXECUTION_ENGINE,
        supported_endpoints: vec![CSM_SPEECH_ROUTE_OPENAI, CSM_SPEECH_ROUTE_PSIONIC],
        supported_response_formats: vec![CSM_SPEECH_RESPONSE_FORMAT_WAV],
        voice_profiles: voice_profiles(&state.governance, &state.descriptor),
        artifact_digests: artifact_digests(&state.descriptor),
        artifact_descriptor: artifact_descriptor(&state.descriptor),
        runtime: runtime.clone(),
        codec_capabilities: codec_capabilities(),
        safety_capabilities: safety_capabilities(),
        execution_refusal: runtime.refusal,
    })
}

async fn csm_models(State(state): State<Arc<CsmSpeechState>>) -> Json<CsmSpeechModelsResponse> {
    let runtime = runtime_snapshot(&state);
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
            psionic_voice_profiles: voice_profiles(&state.governance, &state.descriptor),
            psionic_artifact_digests: artifact_digests(&state.descriptor),
            psionic_artifact_descriptor: artifact_descriptor(&state.descriptor),
            psionic_runtime: runtime.clone(),
            psionic_codec_capabilities: codec_capabilities(),
            psionic_safety_capabilities: safety_capabilities(),
            psionic_execution_refusal: runtime.refusal,
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
    input: String,
    voice_profile_id: String,
    source_prompt_profile_id: String,
    voice_approval_status: String,
    watermarking: String,
    speaker: u32,
    stream: bool,
    max_audio_length_ms: u64,
    sampling: CsmSamplingStrategy,
    context_policy: CsmServedContextPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CsmServedContextPolicy {
    None,
    PromptProfileOnly,
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
        let stream = request.stream;
        let response_format = request
            .response_format
            .unwrap_or_else(|| String::from(CSM_SPEECH_RESPONSE_FORMAT_WAV));
        if response_format != CSM_SPEECH_RESPONSE_FORMAT_WAV {
            return Err(CsmSpeechHttpError::unsupported_format());
        }
        let voice_profile_id = request
            .voice_profile_id
            .or(request.voice)
            .unwrap_or_else(|| String::from(CSM_LYRA_DEFAULT_FEMALE_PROFILE_ID));
        let Some(governed_profile) = state.governance.find_profile(&voice_profile_id) else {
            return Err(CsmSpeechHttpError::missing_voice_profile());
        };
        if !governed_profile.is_runtime_admitted() {
            return Err(CsmSpeechHttpError::unapproved_voice_profile());
        }
        let Some(prompt) = state.fixture.prompts.iter().find(|prompt| {
            prompt.profile_id == governed_profile.source_prompt_profile_id
                && prompt.speaker == governed_profile.speaker
        }) else {
            return Err(CsmSpeechHttpError::runtime_unavailable(
                "voice_profile_source_unavailable",
                "governed CSM voice profile references a missing source prompt",
            ));
        };
        let Some(source_descriptor) = state
            .descriptor
            .voice_profiles
            .iter()
            .find(|profile| profile.profile_id == governed_profile.source_prompt_profile_id)
        else {
            return Err(CsmSpeechHttpError::runtime_unavailable(
                "voice_profile_descriptor_unavailable",
                "governed CSM voice profile references a missing descriptor",
            ));
        };
        if source_descriptor.prompt_audio_sha256 != governed_profile.prompt_audio_sha256 {
            return Err(CsmSpeechHttpError::runtime_unavailable(
                "voice_profile_digest_mismatch",
                "governed CSM voice profile digest does not match descriptor",
            ));
        }
        let params = request.psionic_csm.unwrap_or_default();
        validate_params(&params)?;
        let max_audio_length_ms = params
            .max_audio_length_ms
            .unwrap_or(CSM_SPEECH_DEFAULT_AUDIO_LENGTH_MS);
        let context_policy = match params.context_policy.as_deref().unwrap_or("none") {
            "none" => CsmServedContextPolicy::None,
            "prompt_profile_only" => CsmServedContextPolicy::PromptProfileOnly,
            _ => unreachable!("context policy validated above"),
        };
        let sampling = csm_sampling_strategy(&params)?;
        Ok(Self {
            model,
            input: request.input,
            voice_profile_id,
            source_prompt_profile_id: governed_profile.source_prompt_profile_id.clone(),
            voice_approval_status: governed_profile.approval_status.clone(),
            watermarking: governed_profile.watermark_policy.status.clone(),
            speaker: prompt.speaker,
            stream,
            max_audio_length_ms,
            sampling,
            context_policy,
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
        && !(80..=CSM_SPEECH_MAX_AUDIO_LENGTH_MS).contains(&max_audio_length_ms)
    {
        return Err(CsmSpeechHttpError::invalid_request(
            "max_audio_length_ms must be between 80 and 2000 on the warm Rust CPU CSM server",
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

fn csm_sampling_strategy(
    params: &CsmGenerationParams,
) -> Result<CsmSamplingStrategy, CsmSpeechHttpError> {
    let top_k = params.top_k.unwrap_or(1);
    let temperature = f64::from(params.temperature.unwrap_or(0.0));
    if top_k <= 1 || temperature == 0.0 {
        return Ok(CsmSamplingStrategy::Greedy);
    }
    Ok(CsmSamplingStrategy::TopK {
        top_k,
        temperature,
        seed: 0,
    })
}

#[derive(Debug)]
struct CsmSpeechSynthesis {
    wav_bytes: Vec<u8>,
    codebook_frames_sha256: String,
    wav_pcm16_digest: String,
    generated_frame_count: usize,
    prompt_frame_count: usize,
    hit_eos: bool,
    first_audio_latency_ms: u128,
    full_generation_latency_ms: u128,
    output_duration_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
struct CsmSpeechStreamTerminalMetadata {
    event: &'static str,
    model: String,
    voice_profile_id: String,
    backend: &'static str,
    execution_engine: &'static str,
    residency: &'static str,
    generated_frame_count: usize,
    prompt_frame_count: usize,
    hit_eos: bool,
    first_audio_latency_ms: u128,
    full_generation_latency_ms: u128,
    output_duration_ms: u64,
    wav_bytes: usize,
    chunk_count: usize,
    codebook_frames_sha256: String,
    wav_pcm16_digest: String,
}

impl CsmSpeechState {
    fn synthesize_blocking(
        &self,
        request: &ValidatedCsmSpeechRequest,
    ) -> Result<CsmSpeechSynthesis, CsmSpeechHttpError> {
        if request.context_policy == CsmServedContextPolicy::PromptProfileOnly {
            return Err(CsmSpeechHttpError::invalid_request(
                "prompt_profile_only context requires full prompt codebooks or Rust Mimi encode; this server currently supports context_policy=none",
                "prompt_profile_context_unavailable",
            ));
        }
        let started = Instant::now();
        let mut runtime = self.runtime.lock().map_err(|_| {
            CsmSpeechHttpError::runtime_unavailable(
                "runtime_lock_poisoned",
                "CSM runtime lock is poisoned and requests fail closed",
            )
        })?;
        let Some(engine) = runtime.engine.as_mut() else {
            let (code, reason) = runtime
                .status
                .refusal
                .as_ref()
                .map(|refusal| (refusal.code, refusal.reason))
                .unwrap_or((
                    "runtime_unavailable",
                    "CSM runtime is unavailable and requests fail closed",
                ));
            return Err(CsmSpeechHttpError::runtime_unavailable(code, reason));
        };

        let target =
            CsmPromptSegment::encode_text_only(&engine.tokenizer, request.speaker, &request.input)
                .map_err(runtime_generation_error)?;
        let plan = csm_build_prompt_frame_plan(
            &[],
            &target,
            CsmGenerationWindow::new(request.max_audio_length_ms),
            CsmContextWindowPolicy::Reject,
        )
        .map_err(runtime_generation_error)?;
        let report = engine
            .generator
            .generate_and_decode(
                &CsmCpuGenerationRequest {
                    prompt: plan,
                    sampling: request.sampling.clone(),
                },
                &mut engine.mimi,
            )
            .map_err(runtime_generation_error)?;
        let wav_bytes = report
            .decode
            .clip
            .to_wav_pcm16()
            .map_err(runtime_generation_error)?;
        let output_duration_ms = csm_output_duration_ms(
            report.decode.clip.samples.len(),
            report.decode.clip.sample_rate_hz,
        );
        let full_generation_latency_ms = started.elapsed().as_millis();

        Ok(CsmSpeechSynthesis {
            wav_bytes,
            codebook_frames_sha256: report.generation.frames_sha256,
            wav_pcm16_digest: report.decode.wav_pcm16_digest,
            generated_frame_count: report.generation.generated_frame_count,
            prompt_frame_count: report.generation.prompt_frame_count,
            hit_eos: report.generation.hit_eos,
            first_audio_latency_ms: full_generation_latency_ms,
            full_generation_latency_ms,
            output_duration_ms,
        })
    }
}

fn runtime_generation_error(error: CsmFrontendError) -> CsmSpeechHttpError {
    match error {
        CsmFrontendError::PromptContextOverflow { .. } => CsmSpeechHttpError::invalid_request(
            "CSM prompt exceeds the admitted context window",
            "context_overflow",
        ),
        CsmFrontendError::GenerationWindow { .. } => CsmSpeechHttpError::invalid_request(
            "CSM generation window is invalid",
            "invalid_generation_window",
        ),
        CsmFrontendError::UnsupportedSampling { .. } => CsmSpeechHttpError::invalid_request(
            "requested CSM sampling mode is unsupported",
            "unsupported_sampling",
        ),
        _ => CsmSpeechHttpError::runtime_unavailable(
            "csm_synthesis_failed",
            "CSM speech synthesis failed inside the Rust runtime",
        ),
    }
}

fn csm_output_duration_ms(sample_count: usize, sample_rate_hz: u32) -> u64 {
    if sample_rate_hz == 0 {
        return 0;
    }
    ((sample_count as u128 * 1_000) / u128::from(sample_rate_hz)) as u64
}

fn current_unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

async fn csm_audio_speech(
    State(state): State<Arc<CsmSpeechState>>,
    Json(request): Json<CsmSpeechRequest>,
) -> Result<Response, CsmSpeechHttpError> {
    let validated = ValidatedCsmSpeechRequest::from_request(&state, request)?;
    let state_for_runtime = Arc::clone(&state);
    let request_for_runtime = validated.clone();
    let synthesis = match tokio::task::spawn_blocking(move || {
        state_for_runtime.synthesize_blocking(&request_for_runtime)
    })
    .await
    {
        Ok(Ok(synthesis)) => synthesis,
        Ok(Err(error)) => {
            let mut response = error.into_response();
            insert_csm_execution_headers(response.headers_mut(), &state, &validated);
            return Ok(response);
        }
        Err(_) => {
            let mut response = CsmSpeechHttpError::runtime_unavailable(
                "runtime_join_failed",
                "CSM runtime blocking task failed before returning a speech response",
            )
            .into_response();
            insert_csm_execution_headers(response.headers_mut(), &state, &validated);
            return Ok(response);
        }
    };

    let mut response = if validated.stream {
        csm_streaming_response(&validated, synthesis)?
    } else {
        csm_wav_response(synthesis)?
    };
    insert_csm_execution_headers(response.headers_mut(), &state, &validated);
    insert_csm_success_headers(response.headers_mut());
    Ok(response)
}

fn csm_wav_response(synthesis: CsmSpeechSynthesis) -> Result<Response, CsmSpeechHttpError> {
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .body(Body::from(synthesis.wav_bytes.clone()))
        .map_err(|_| {
            CsmSpeechHttpError::runtime_unavailable(
                "response_build_failed",
                "failed to build CSM WAV response",
            )
        })?;
    insert_synthesis_headers(response.headers_mut(), &synthesis, 1);
    Ok(response)
}

fn csm_streaming_response(
    request: &ValidatedCsmSpeechRequest,
    synthesis: CsmSpeechSynthesis,
) -> Result<Response, CsmSpeechHttpError> {
    let chunks = synthesis
        .wav_bytes
        .chunks(CSM_SPEECH_STREAM_CHUNK_BYTES)
        .collect::<Vec<_>>();
    let chunk_count = chunks.len();
    let terminal = CsmSpeechStreamTerminalMetadata {
        event: "terminal",
        model: request.model.clone(),
        voice_profile_id: request.voice_profile_id.clone(),
        backend: CSM_SPEECH_SERVED_BACKEND,
        execution_engine: CSM_SPEECH_EXECUTION_ENGINE,
        residency: CSM_SPEECH_RESIDENCY_MODE,
        generated_frame_count: synthesis.generated_frame_count,
        prompt_frame_count: synthesis.prompt_frame_count,
        hit_eos: synthesis.hit_eos,
        first_audio_latency_ms: synthesis.first_audio_latency_ms,
        full_generation_latency_ms: synthesis.full_generation_latency_ms,
        output_duration_ms: synthesis.output_duration_ms,
        wav_bytes: synthesis.wav_bytes.len(),
        chunk_count,
        codebook_frames_sha256: synthesis.codebook_frames_sha256.clone(),
        wav_pcm16_digest: synthesis.wav_pcm16_digest.clone(),
    };
    let body = csm_multipart_stream_body(chunks, &terminal)?;
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/mixed; boundary={CSM_SPEECH_STREAM_BOUNDARY}"),
        )
        .body(Body::from(body))
        .map_err(|_| {
            CsmSpeechHttpError::runtime_unavailable(
                "response_build_failed",
                "failed to build CSM stream response",
            )
        })?;
    insert_synthesis_headers(response.headers_mut(), &synthesis, chunk_count);
    Ok(response)
}

fn csm_multipart_stream_body(
    chunks: Vec<&[u8]>,
    terminal: &CsmSpeechStreamTerminalMetadata,
) -> Result<Vec<u8>, CsmSpeechHttpError> {
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{CSM_SPEECH_STREAM_BOUNDARY}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Type: application/json\r\n\r\n");
    body.extend_from_slice(
        br#"{"event":"start","encoding":"wav","chunk_transport":"multipart_mixed"}"#,
    );
    body.extend_from_slice(b"\r\n");
    for (index, chunk) in chunks.iter().enumerate() {
        body.extend_from_slice(format!("--{CSM_SPEECH_STREAM_BOUNDARY}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Type: audio/wav\r\n");
        body.extend_from_slice(format!("X-Psionic-Chunk-Index: {index}\r\n").as_bytes());
        body.extend_from_slice(
            format!("X-Psionic-Chunk-Bytes: {}\r\n\r\n", chunk.len()).as_bytes(),
        );
        body.extend_from_slice(chunk);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{CSM_SPEECH_STREAM_BOUNDARY}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Type: application/json\r\n\r\n");
    let terminal_json = serde_json::to_vec(terminal).map_err(|_| {
        CsmSpeechHttpError::runtime_unavailable(
            "terminal_metadata_failed",
            "failed to serialize CSM stream terminal metadata",
        )
    })?;
    body.extend_from_slice(&terminal_json);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{CSM_SPEECH_STREAM_BOUNDARY}--\r\n").as_bytes());
    Ok(body)
}

fn insert_synthesis_headers(
    headers: &mut HeaderMap,
    synthesis: &CsmSpeechSynthesis,
    chunk_count: usize,
) {
    insert_header(
        headers,
        "x-psionic-csm-generated-frame-count",
        synthesis.generated_frame_count.to_string().as_str(),
    );
    insert_header(
        headers,
        "x-psionic-csm-prompt-frame-count",
        synthesis.prompt_frame_count.to_string().as_str(),
    );
    insert_header(
        headers,
        "x-psionic-csm-hit-eos",
        if synthesis.hit_eos { "true" } else { "false" },
    );
    insert_header(
        headers,
        "x-psionic-csm-frames-sha256",
        synthesis.codebook_frames_sha256.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-csm-wav-pcm16-digest",
        synthesis.wav_pcm16_digest.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-first-audio-latency-ms",
        synthesis.first_audio_latency_ms.to_string().as_str(),
    );
    insert_header(
        headers,
        "x-psionic-full-generation-latency-ms",
        synthesis.full_generation_latency_ms.to_string().as_str(),
    );
    insert_header(
        headers,
        "x-psionic-output-duration-ms",
        synthesis.output_duration_ms.to_string().as_str(),
    );
    insert_header(
        headers,
        "x-psionic-stream-chunk-count",
        chunk_count.to_string().as_str(),
    );
}

fn insert_csm_success_headers(headers: &mut HeaderMap) {
    insert_header(
        headers,
        "x-psionic-residency-mode",
        CSM_SPEECH_RESIDENCY_MODE,
    );
    insert_header(
        headers,
        "x-psionic-accelerated-backend",
        "unavailable_fail_closed",
    );
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
        "x-psionic-csm-source-prompt-profile-id",
        request.source_prompt_profile_id.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-csm-voice-approval-status",
        request.voice_approval_status.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-csm-watermarking",
        request.watermarking.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-artifact-csm-config-digest",
        state.descriptor.digests.csm_config_digest.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-artifact-csm-model-digest",
        state.descriptor.digests.csm_model_digest.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-artifact-llama-tokenizer-digest",
        state.descriptor.digests.llama_tokenizer_digest.as_str(),
    );
    insert_header(
        headers,
        "x-psionic-artifact-mimi-weight-digest",
        state.descriptor.digests.mimi_weight_digest.as_str(),
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
    message: String,
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
    message: String,
    kind: &'static str,
    code: &'static str,
}

impl CsmSpeechHttpError {
    fn invalid_request(message: impl Into<String>, code: &'static str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
            code,
        }
    }

    fn unsupported_format() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: "CSM speech currently supports response_format=wav only".to_string(),
            kind: "invalid_request_error",
            code: "unsupported_response_format",
        }
    }

    fn missing_voice_profile() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: "requested CSM voice profile is not available".to_string(),
            kind: "invalid_request_error",
            code: "voice_profile_unavailable",
        }
    }

    fn unapproved_voice_profile() -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: "requested CSM voice profile is not approved for this runtime".to_string(),
            kind: "invalid_request_error",
            code: "voice_profile_unapproved",
        }
    }

    fn model_unavailable() -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: "requested CSM model is not available on this Psionic speech server"
                .to_string(),
            kind: "not_found_error",
            code: "model_unavailable",
        }
    }

    fn runtime_unavailable(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            kind: "backend_unavailable",
            code,
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

    fn disabled_runtime_config() -> CsmSpeechServerConfig {
        CsmSpeechServerConfig {
            runtime_enabled: false,
            ..CsmSpeechServerConfig::default()
        }
    }

    #[tokio::test]
    async fn csm_speech_route_refuses_when_runtime_is_disabled()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = CsmSpeechServer::from_config(disabled_runtime_config())?;
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
                        "voice_profile_id": CSM_LYRA_DEFAULT_FEMALE_PROFILE_ID,
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
            Some(CSM_LYRA_DEFAULT_FEMALE_PROFILE_ID)
        );
        assert_eq!(
            response
                .headers()
                .get("x-psionic-csm-source-prompt-profile-id")
                .and_then(|value| value.to_str().ok()),
            Some("conversational_a")
        );
        let body = to_bytes(response.into_body(), usize::MAX).await?;
        let payload: serde_json::Value = serde_json::from_slice(&body)?;
        assert_eq!(
            payload["error"]["code"],
            serde_json::json!("runtime_disabled")
        );
        assert!(!String::from_utf8(body.to_vec())?.contains("/Users/"));
        Ok(())
    }

    #[tokio::test]
    async fn csm_speech_route_refuses_missing_voice_profile()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = CsmSpeechServer::from_config(disabled_runtime_config())?;
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
    async fn csm_speech_route_refuses_raw_source_prompt_profile()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = CsmSpeechServer::from_config(disabled_runtime_config())?;
        let response = server
            .router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(CSM_SPEECH_ROUTE_PSIONIC)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&serde_json::json!({
                        "input": "hello",
                        "voice_profile_id": "conversational_a"
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
        let server = CsmSpeechServer::from_config(disabled_runtime_config())?;
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
        assert_eq!(payload["status"], serde_json::json!("degraded"));
        assert_eq!(payload["execution_engine"], CSM_SPEECH_EXECUTION_ENGINE);
        assert_eq!(
            payload["runtime"]["state"],
            serde_json::json!("unavailable")
        );
        assert_eq!(
            payload["execution_refusal"]["code"],
            serde_json::json!("runtime_disabled")
        );
        assert_eq!(
            payload["artifact_descriptor"]["frame_contract"]["frame_lanes"],
            serde_json::json!(33)
        );
        assert_eq!(
            payload["codec_capabilities"]["mimi_decode"],
            serde_json::json!("implemented")
        );
        assert_eq!(
            payload["codec_capabilities"]["reference_audio_encode_refusal"]["code"],
            serde_json::json!("rust_mimi_encode_not_implemented")
        );
        assert_eq!(
            payload["safety_capabilities"]["watermarking"],
            serde_json::json!(CSM_WATERMARK_OPERATOR_DOGFOOD)
        );
        assert_eq!(
            payload["safety_capabilities"]["watermarking_refusal"]["code"],
            serde_json::json!("csm_watermarking_unavailable")
        );
        assert!(
            payload["voice_profiles"]
                .as_array()
                .is_some_and(|profiles| profiles.iter().any(|profile| {
                    profile["id"] == serde_json::json!(CSM_LYRA_DEFAULT_FEMALE_PROFILE_ID)
                        && profile["source_prompt_profile_id"]
                            == serde_json::json!("conversational_a")
                        && profile["approval_status"]
                            == serde_json::json!("approved_openagents_operated_dogfood")
                        && profile["watermarking"]
                            == serde_json::json!(CSM_WATERMARK_OPERATOR_DOGFOOD)
                }))
        );
        assert_eq!(
            payload["voice_profiles"][0]["prompt_codebooks"]["codebook_count"],
            serde_json::json!(32)
        );
        Ok(())
    }

    #[tokio::test]
    async fn csm_models_publish_openai_and_psionic_speech_routes()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = CsmSpeechServer::from_config(disabled_runtime_config())?;
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
        assert_eq!(
            payload["data"][0]["psionic_voice_profiles"][0]["id"],
            serde_json::json!(CSM_LYRA_DEFAULT_FEMALE_PROFILE_ID)
        );
        assert_eq!(
            payload["data"][0]["psionic_safety_capabilities"]["watermarking_refusal"]["code"],
            serde_json::json!("csm_watermarking_unavailable")
        );
        Ok(())
    }

    #[tokio::test]
    async fn csm_speech_route_serves_wav_when_runtime_artifacts_are_present()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = CsmSpeechServer::from_config(CsmSpeechServerConfig::default())?;
        if !server
            .state
            .runtime
            .lock()
            .expect("runtime lock")
            .is_ready()
        {
            eprintln!("skipping CSM serving test because gated runtime artifacts are unavailable");
            return Ok(());
        }

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
                        "voice_profile_id": CSM_LYRA_DEFAULT_FEMALE_PROFILE_ID,
                        "response_format": "wav",
                        "psionic_csm": {
                            "max_audio_length_ms": 160,
                            "context_policy": "none"
                        }
                    }))?))?,
            )
            .await?;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("audio/wav")
        );
        assert_eq!(
            response
                .headers()
                .get("x-psionic-residency-mode")
                .and_then(|value| value.to_str().ok()),
            Some(CSM_SPEECH_RESIDENCY_MODE)
        );
        assert_eq!(
            response
                .headers()
                .get("x-psionic-csm-watermarking")
                .and_then(|value| value.to_str().ok()),
            Some(CSM_WATERMARK_OPERATOR_DOGFOOD)
        );
        assert!(
            response
                .headers()
                .get("x-psionic-full-generation-latency-ms")
                .is_some()
        );
        let body = to_bytes(response.into_body(), usize::MAX).await?;
        assert!(body.starts_with(b"RIFF"));
        assert!(body.len() > 44);
        Ok(())
    }

    #[test]
    fn csm_multipart_stream_body_carries_binary_chunks_and_terminal_metadata()
    -> Result<(), Box<dyn std::error::Error>> {
        let terminal = CsmSpeechStreamTerminalMetadata {
            event: "terminal",
            model: CSM_SPEECH_MODEL_ID.to_string(),
            voice_profile_id: CSM_LYRA_DEFAULT_FEMALE_PROFILE_ID.to_string(),
            backend: CSM_SPEECH_SERVED_BACKEND,
            execution_engine: CSM_SPEECH_EXECUTION_ENGINE,
            residency: CSM_SPEECH_RESIDENCY_MODE,
            generated_frame_count: 2,
            prompt_frame_count: 6,
            hit_eos: false,
            first_audio_latency_ms: 12,
            full_generation_latency_ms: 34,
            output_duration_ms: 160,
            wav_bytes: 8,
            chunk_count: 2,
            codebook_frames_sha256: "sha256:frames".to_string(),
            wav_pcm16_digest: "sha256:wav".to_string(),
        };
        let body =
            csm_multipart_stream_body(vec![b"RIFF".as_slice(), b"WAVE".as_slice()], &terminal)
                .expect("stream body");
        let text = String::from_utf8_lossy(&body);

        assert!(text.contains("Content-Type: audio/wav"));
        assert!(text.contains("X-Psionic-Chunk-Index: 0"));
        assert!(text.contains("\"event\":\"terminal\""));
        assert!(body.windows(4).any(|window| window == b"RIFF"));
        Ok(())
    }
}
