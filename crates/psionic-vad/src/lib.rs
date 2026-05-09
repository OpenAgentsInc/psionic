//! Psionic-owned streaming voice activity detection worker.
//!
//! This crate intentionally keeps the model-backed VAD boundary inside
//! Psionic. Autopilot should consume this crate through a Psionic worker API
//! rather than embedding Silero, Candle, ONNX, or browser-side VAD logic as its
//! product path.

use std::{collections::BTreeMap, fs, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Current Psionic VAD worker protocol version.
pub const PSIONIC_VAD_PROTOCOL_VERSION: &str = "psionic.vad.v1";

/// Current execution engine label.
pub const PSIONIC_VAD_EXECUTION_ENGINE: &str = "psionic_silero_style_vad_mvp";

/// Current built-in model artifact id.
pub const PSIONIC_VAD_MODEL_ARTIFACT_ID: &str = "psionic-vad/silero-style-mvp-v0";

/// Current artifact manifest schema version.
pub const PSIONIC_VAD_ARTIFACT_SCHEMA_VERSION: &str = "psionic.vad.artifact.v1";

/// Default sample rate used by the Silero reference stream shape.
pub const DEFAULT_INFERENCE_SAMPLE_RATE_HZ: u32 = 16_000;

/// Default Silero-style frame size at 16 kHz, about 32 ms.
pub const DEFAULT_FRAME_SIZE_SAMPLES: usize = 512;

/// Default Silero-style recurrent context size at 16 kHz.
pub const DEFAULT_CONTEXT_SIZE_SAMPLES: usize = 64;

/// Runtime configuration for the VAD worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VadWorkerConfig {
    /// Model artifact identifier.
    pub model_artifact_id: String,
    /// Worker execution engine label.
    pub execution_engine: String,
    /// Inference sample rate.
    pub inference_sample_rate_hz: u32,
    /// Samples per inference frame.
    pub frame_size_samples: usize,
    /// Samples of context carried from the previous frame.
    pub context_size_samples: usize,
    /// Speech probability threshold.
    pub threshold: f32,
    /// Exit threshold while speech is active.
    pub negative_threshold: f32,
    /// Minimum accepted speech duration.
    pub min_speech_duration_ms: u32,
    /// Minimum silence before ending one active speech segment.
    pub min_silence_duration_ms: u32,
    /// Pad speech starts and ends by this duration.
    pub speech_pad_ms: u32,
    /// Maximum speech duration before forced endpoint.
    pub max_speech_duration_ms: u32,
    /// Maximum queued input samples per session at inference rate.
    pub max_buffered_samples: usize,
}

impl Default for VadWorkerConfig {
    fn default() -> Self {
        Self {
            model_artifact_id: PSIONIC_VAD_MODEL_ARTIFACT_ID.to_string(),
            execution_engine: PSIONIC_VAD_EXECUTION_ENGINE.to_string(),
            inference_sample_rate_hz: DEFAULT_INFERENCE_SAMPLE_RATE_HZ,
            frame_size_samples: DEFAULT_FRAME_SIZE_SAMPLES,
            context_size_samples: DEFAULT_CONTEXT_SIZE_SAMPLES,
            threshold: 0.5,
            negative_threshold: 0.35,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            max_speech_duration_ms: 30_000,
            max_buffered_samples: DEFAULT_INFERENCE_SAMPLE_RATE_HZ as usize * 10,
        }
    }
}

impl VadWorkerConfig {
    /// Stable digest for config-sensitive release comparisons.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.model_artifact_id.as_bytes());
        hasher.update(self.execution_engine.as_bytes());
        hasher.update(self.inference_sample_rate_hz.to_le_bytes());
        hasher.update((self.frame_size_samples as u64).to_le_bytes());
        hasher.update((self.context_size_samples as u64).to_le_bytes());
        hasher.update(self.threshold.to_le_bytes());
        hasher.update(self.negative_threshold.to_le_bytes());
        hasher.update(self.min_speech_duration_ms.to_le_bytes());
        hasher.update(self.min_silence_duration_ms.to_le_bytes());
        hasher.update(self.speech_pad_ms.to_le_bytes());
        hasher.update(self.max_speech_duration_ms.to_le_bytes());
        format!("sha256:{:x}", hasher.finalize())
    }

    fn validate(&self) -> Result<(), VadError> {
        if self.inference_sample_rate_hz != DEFAULT_INFERENCE_SAMPLE_RATE_HZ {
            return Err(VadError::UnsupportedInferenceSampleRate {
                sample_rate_hz: self.inference_sample_rate_hz,
            });
        }
        if self.frame_size_samples == 0 {
            return Err(VadError::InvalidConfig {
                field: "frame_size_samples",
                reason: "must be greater than zero",
            });
        }
        if self.context_size_samples > self.frame_size_samples {
            return Err(VadError::InvalidConfig {
                field: "context_size_samples",
                reason: "must be less than or equal to frame_size_samples",
            });
        }
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(VadError::InvalidConfig {
                field: "threshold",
                reason: "must be between 0.0 and 1.0",
            });
        }
        if !(0.0..=self.threshold).contains(&self.negative_threshold) {
            return Err(VadError::InvalidConfig {
                field: "negative_threshold",
                reason: "must be between 0.0 and threshold",
            });
        }
        Ok(())
    }
}

/// Model/reference artifact manifest used to validate Psionic VAD startup.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VadModelArtifactManifest {
    /// Manifest schema version.
    pub schema_version: String,
    /// Stable artifact id.
    pub artifact_id: String,
    /// Human-readable artifact version.
    pub artifact_version: String,
    /// Artifact kind.
    pub artifact_kind: String,
    /// Execution engine this artifact admits.
    pub execution_engine: String,
    /// License posture.
    pub license: VadArtifactLicense,
    /// Source references used to build the artifact.
    pub source_references: Vec<VadArtifactSourceReference>,
    /// Inference sample rate.
    pub inference_sample_rate_hz: u32,
    /// Frame size.
    pub frame_size_samples: usize,
    /// Context size.
    pub context_size_samples: usize,
    /// Digest over the admitted runtime parameters.
    pub parameter_digest: String,
    /// Whether provider keys are required.
    pub provider_keys_required: bool,
    /// Whether private raw audio retention is required.
    pub raw_audio_retention_required: bool,
    /// Runtime dependencies exposed to callers.
    pub runtime_dependencies: Vec<String>,
    /// Artifact update policy.
    pub update_policy: String,
}

impl VadModelArtifactManifest {
    /// Returns the built-in MVP artifact manifest.
    #[must_use]
    pub fn builtin() -> Self {
        let config = VadWorkerConfig::default();
        Self {
            schema_version: PSIONIC_VAD_ARTIFACT_SCHEMA_VERSION.to_string(),
            artifact_id: PSIONIC_VAD_MODEL_ARTIFACT_ID.to_string(),
            artifact_version: "v0".to_string(),
            artifact_kind: "owned_silero_style_signal_vad_mvp".to_string(),
            execution_engine: PSIONIC_VAD_EXECUTION_ENGINE.to_string(),
            license: VadArtifactLicense {
                spdx_expression: "Apache-2.0 AND MIT-reference-notice".to_string(),
                notice: "Silero VAD reference material is MIT licensed by the Silero Team. Psionic does not vendor the full Silero repository and does not expose Silero, Candle, or ONNX as the Autopilot product runtime.".to_string(),
                required_attribution: vec![
                    "Silero VAD reference: Copyright (c) 2020-present Silero Team, MIT License".to_string(),
                ],
            },
            source_references: vec![
                VadArtifactSourceReference {
                    name: "silero-vad".to_string(),
                    reference_type: "architecture_and_threshold_reference".to_string(),
                    path_or_url: "../competition/repos/silero-vad".to_string(),
                    license: "MIT".to_string(),
                    used_for: "VAD threshold, hysteresis, frame cadence, context/state shape, min speech, min silence, and speech padding reference".to_string(),
                },
                VadArtifactSourceReference {
                    name: "candle silero-vad example".to_string(),
                    reference_type: "stream_state_reference_only".to_string(),
                    path_or_url: "../competition/repos/candle/candle-examples/examples/silero-vad".to_string(),
                    license: "Apache-2.0".to_string(),
                    used_for: "Rust stream-state pattern reference only; not a product runtime dependency".to_string(),
                },
            ],
            inference_sample_rate_hz: config.inference_sample_rate_hz,
            frame_size_samples: config.frame_size_samples,
            context_size_samples: config.context_size_samples,
            parameter_digest: config.digest(),
            provider_keys_required: false,
            raw_audio_retention_required: false,
            runtime_dependencies: vec!["psionic-vad".to_string()],
            update_policy: "changes require corpus, replay, shadow, fallback, and release gates before Autopilot primary endpointing".to_string(),
        }
    }

    /// Stable digest over the manifest itself.
    #[must_use]
    pub fn digest(&self) -> String {
        let encoded = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(encoded);
        format!("sha256:{:x}", hasher.finalize())
    }

    /// Validates this manifest against a worker config.
    pub fn validate_for_config(&self, config: &VadWorkerConfig) -> Result<(), VadError> {
        if self.schema_version != PSIONIC_VAD_ARTIFACT_SCHEMA_VERSION {
            return Err(VadError::ArtifactManifest {
                reason: "unsupported schema version",
            });
        }
        if self.artifact_id != config.model_artifact_id {
            return Err(VadError::ArtifactManifest {
                reason: "artifact id does not match worker config",
            });
        }
        if self.execution_engine != config.execution_engine {
            return Err(VadError::ArtifactManifest {
                reason: "execution engine does not match worker config",
            });
        }
        if self.inference_sample_rate_hz != config.inference_sample_rate_hz {
            return Err(VadError::ArtifactManifest {
                reason: "inference sample rate does not match worker config",
            });
        }
        if self.frame_size_samples != config.frame_size_samples {
            return Err(VadError::ArtifactManifest {
                reason: "frame size does not match worker config",
            });
        }
        if self.context_size_samples != config.context_size_samples {
            return Err(VadError::ArtifactManifest {
                reason: "context size does not match worker config",
            });
        }
        if self.parameter_digest != config.digest() {
            return Err(VadError::ArtifactManifest {
                reason: "parameter digest does not match worker config",
            });
        }
        if self.provider_keys_required {
            return Err(VadError::ArtifactManifest {
                reason: "VAD artifacts must not require provider keys",
            });
        }
        if self.raw_audio_retention_required {
            return Err(VadError::ArtifactManifest {
                reason: "VAD artifacts must not require raw audio retention",
            });
        }
        Ok(())
    }
}

/// License metadata for one VAD artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VadArtifactLicense {
    /// SPDX expression or bounded internal expression.
    pub spdx_expression: String,
    /// Human-readable notice.
    pub notice: String,
    /// Required attribution lines.
    pub required_attribution: Vec<String>,
}

/// Source reference used by one VAD artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VadArtifactSourceReference {
    /// Reference name.
    pub name: String,
    /// Reference type.
    pub reference_type: String,
    /// Local path or URL.
    pub path_or_url: String,
    /// Source license.
    pub license: String,
    /// What this source was used for.
    pub used_for: String,
}

/// Loads one artifact manifest from JSON.
pub fn load_model_artifact_manifest(
    path: impl AsRef<Path>,
) -> Result<VadModelArtifactManifest, VadError> {
    let bytes = fs::read(path).map_err(|_| VadError::ArtifactManifest {
        reason: "artifact manifest could not be read",
    })?;
    serde_json::from_slice(bytes.as_slice()).map_err(|_| VadError::ArtifactManifest {
        reason: "artifact manifest could not be parsed",
    })
}

/// Per-session controls.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VadSessionConfig {
    /// Stable session id owned by the caller.
    pub session_id: String,
    /// Input sample rate in Hz.
    pub input_sample_rate_hz: u32,
    /// Number of input channels.
    pub channels: u16,
}

impl VadSessionConfig {
    /// Creates a mono session config.
    #[must_use]
    pub fn mono(session_id: impl Into<String>, input_sample_rate_hz: u32) -> Self {
        Self {
            session_id: session_id.into(),
            input_sample_rate_hz,
            channels: 1,
        }
    }
}

/// One chunk submitted to the worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VadChunkRequest {
    /// Stable session id.
    pub session_id: String,
    /// Monotonic chunk index assigned by the caller.
    pub chunk_index: u64,
    /// Input sample rate.
    pub input_sample_rate_hz: u32,
    /// Number of channels in the interleaved input.
    pub channels: u16,
    /// Interleaved signed 16-bit PCM samples.
    pub pcm_s16le: Vec<i16>,
    /// Whether this chunk is the final chunk for the turn.
    pub final_chunk: bool,
}

impl VadChunkRequest {
    /// Creates one mono request.
    #[must_use]
    pub fn mono(
        session_id: impl Into<String>,
        chunk_index: u64,
        input_sample_rate_hz: u32,
        pcm_s16le: Vec<i16>,
        final_chunk: bool,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            chunk_index,
            input_sample_rate_hz,
            channels: 1,
            pcm_s16le,
            final_chunk,
        }
    }
}

/// Worker health status.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VadWorkerHealth {
    /// Protocol version.
    pub protocol_version: String,
    /// Execution engine.
    pub execution_engine: String,
    /// Artifact id.
    pub model_artifact_id: String,
    /// Model/load readiness.
    pub ready: bool,
    /// Number of active sessions.
    pub active_sessions: usize,
    /// Config digest.
    pub config_digest: String,
}

/// State returned for one chunk.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VadChunkResponse {
    /// Protocol version.
    pub protocol_version: String,
    /// Session id.
    pub session_id: String,
    /// Chunk index.
    pub chunk_index: u64,
    /// Worker execution engine.
    pub execution_engine: String,
    /// Model artifact id.
    pub model_artifact_id: String,
    /// Input samples accepted.
    pub input_samples: usize,
    /// Inference samples accepted after channel reduction and resampling.
    pub inference_samples: usize,
    /// Number of full frames processed.
    pub processed_frames: usize,
    /// Latest speech probability.
    pub speech_probability: f32,
    /// Smoothed speech probability.
    pub smoothed_probability: f32,
    /// Whether speech is currently active after this chunk.
    pub speech_active: bool,
    /// Current endpoint event, if any.
    pub event: Option<VadEndpointEvent>,
    /// Number of samples buffered at inference rate.
    pub buffered_inference_samples: usize,
    /// State digest for replay comparisons.
    pub state_digest: String,
}

/// Endpoint events produced by the worker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VadEndpointEvent {
    /// Speech start candidate became active.
    SpeechStart {
        /// Start sample at inference rate after speech padding.
        start_sample: u64,
        /// Reason for the event.
        reason: String,
    },
    /// Speech ended.
    SpeechEnd {
        /// Start sample at inference rate after speech padding.
        start_sample: u64,
        /// End sample at inference rate after speech padding.
        end_sample: u64,
        /// Reason for the event.
        reason: String,
    },
    /// Turn finalized without enough accepted speech.
    NoSpeech {
        /// Reason for the event.
        reason: String,
    },
}

/// Psionic VAD worker error.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VadError {
    /// Unsupported inference sample rate.
    #[error("unsupported inference sample rate {sample_rate_hz}")]
    UnsupportedInferenceSampleRate {
        /// Requested sample rate.
        sample_rate_hz: u32,
    },
    /// Unsupported input sample rate.
    #[error("unsupported input sample rate {sample_rate_hz}")]
    UnsupportedInputSampleRate {
        /// Requested sample rate.
        sample_rate_hz: u32,
    },
    /// Invalid config.
    #[error("invalid VAD config field `{field}`: {reason}")]
    InvalidConfig {
        /// Field name.
        field: &'static str,
        /// Refusal reason.
        reason: &'static str,
    },
    /// Invalid request.
    #[error("invalid VAD request: {reason}")]
    InvalidRequest {
        /// Refusal reason.
        reason: &'static str,
    },
    /// Unknown session.
    #[error("unknown VAD session `{session_id}`")]
    UnknownSession {
        /// Session id.
        session_id: String,
    },
    /// Session buffer exceeded bounded limit.
    #[error("VAD session `{session_id}` exceeded max buffered samples")]
    BufferLimit {
        /// Session id.
        session_id: String,
    },
    /// Artifact manifest validation failed.
    #[error("VAD artifact manifest validation failed: {reason}")]
    ArtifactManifest {
        /// Refusal reason.
        reason: &'static str,
    },
}

/// Psionic-owned VAD worker.
#[derive(Debug)]
pub struct PsionicVadWorker {
    config: VadWorkerConfig,
    sessions: BTreeMap<String, VadSessionState>,
}

impl PsionicVadWorker {
    /// Creates a worker.
    pub fn new(config: VadWorkerConfig) -> Result<Self, VadError> {
        config.validate()?;
        VadModelArtifactManifest::builtin().validate_for_config(&config)?;
        Ok(Self {
            config,
            sessions: BTreeMap::new(),
        })
    }

    /// Creates a worker after validating an explicit artifact manifest.
    pub fn new_with_artifact(
        config: VadWorkerConfig,
        artifact_manifest: &VadModelArtifactManifest,
    ) -> Result<Self, VadError> {
        config.validate()?;
        artifact_manifest.validate_for_config(&config)?;
        Ok(Self {
            config,
            sessions: BTreeMap::new(),
        })
    }

    /// Creates a worker with default config.
    pub fn default_worker() -> Result<Self, VadError> {
        Self::new(VadWorkerConfig::default())
    }

    /// Returns current health.
    #[must_use]
    pub fn health(&self) -> VadWorkerHealth {
        VadWorkerHealth {
            protocol_version: PSIONIC_VAD_PROTOCOL_VERSION.to_string(),
            execution_engine: self.config.execution_engine.clone(),
            model_artifact_id: self.config.model_artifact_id.clone(),
            ready: true,
            active_sessions: self.sessions.len(),
            config_digest: self.config.digest(),
        }
    }

    /// Starts or replaces a session.
    pub fn start_session(
        &mut self,
        session: VadSessionConfig,
    ) -> Result<VadWorkerHealth, VadError> {
        validate_input_rate(session.input_sample_rate_hz)?;
        if session.channels == 0 {
            return Err(VadError::InvalidRequest {
                reason: "channels must be greater than zero",
            });
        }
        let state = VadSessionState::new(&self.config, session);
        self.sessions.insert(state.session_id.clone(), state);
        Ok(self.health())
    }

    /// Resets an existing session.
    pub fn reset_session(&mut self, session_id: &str) -> Result<(), VadError> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| VadError::UnknownSession {
                session_id: session_id.to_string(),
            })?
            .session_config
            .clone();
        self.sessions.insert(
            session_id.to_string(),
            VadSessionState::new(&self.config, session),
        );
        Ok(())
    }

    /// Processes one chunk.
    pub fn infer_chunk(&mut self, request: VadChunkRequest) -> Result<VadChunkResponse, VadError> {
        validate_input_rate(request.input_sample_rate_hz)?;
        if request.channels == 0 {
            return Err(VadError::InvalidRequest {
                reason: "channels must be greater than zero",
            });
        }
        if request.pcm_s16le.is_empty() && !request.final_chunk {
            return Err(VadError::InvalidRequest {
                reason: "chunk has no samples",
            });
        }
        if !self.sessions.contains_key(request.session_id.as_str()) {
            self.start_session(VadSessionConfig {
                session_id: request.session_id.clone(),
                input_sample_rate_hz: request.input_sample_rate_hz,
                channels: request.channels,
            })?;
        }
        let state = self
            .sessions
            .get_mut(request.session_id.as_str())
            .ok_or_else(|| VadError::UnknownSession {
                session_id: request.session_id.clone(),
            })?;
        state.accept_chunk(&self.config, request)
    }

    /// Flushes one session and finalizes silence/no-speech behavior.
    pub fn flush_session(
        &mut self,
        session_id: &str,
        chunk_index: u64,
    ) -> Result<VadChunkResponse, VadError> {
        self.infer_chunk(VadChunkRequest {
            session_id: session_id.to_string(),
            chunk_index,
            input_sample_rate_hz: DEFAULT_INFERENCE_SAMPLE_RATE_HZ,
            channels: 1,
            pcm_s16le: Vec::new(),
            final_chunk: true,
        })
    }
}

#[derive(Debug)]
struct VadSessionState {
    session_id: String,
    session_config: VadSessionConfig,
    buffered: Vec<f32>,
    context: Vec<f32>,
    sample_cursor: u64,
    speech_start_sample: Option<u64>,
    temp_end_sample: Option<u64>,
    triggered: bool,
    smoothed_probability: f32,
    noise_floor: f32,
    latest_probability: f32,
}

impl VadSessionState {
    fn new(config: &VadWorkerConfig, session_config: VadSessionConfig) -> Self {
        Self {
            session_id: session_config.session_id.clone(),
            session_config,
            buffered: Vec::new(),
            context: vec![0.0; config.context_size_samples],
            sample_cursor: 0,
            speech_start_sample: None,
            temp_end_sample: None,
            triggered: false,
            smoothed_probability: 0.0,
            noise_floor: 0.004,
            latest_probability: 0.0,
        }
    }

    fn accept_chunk(
        &mut self,
        config: &VadWorkerConfig,
        request: VadChunkRequest,
    ) -> Result<VadChunkResponse, VadError> {
        let input_samples = request.pcm_s16le.len();
        let mono = downmix_to_mono(&request.pcm_s16le, request.channels)?;
        let resampled = resample_to_inference_rate(
            mono.as_slice(),
            request.input_sample_rate_hz,
            config.inference_sample_rate_hz,
        )?;
        self.buffered.extend(resampled.iter().copied());
        if self.buffered.len() > config.max_buffered_samples {
            return Err(VadError::BufferLimit {
                session_id: self.session_id.clone(),
            });
        }

        let mut processed_frames = 0;
        let mut event = None;
        while self.buffered.len() >= config.frame_size_samples {
            let frame: Vec<f32> = self.buffered.drain(..config.frame_size_samples).collect();
            let candidate = self.process_frame(config, frame.as_slice());
            processed_frames += 1;
            if candidate.is_some() {
                event = candidate;
            }
            let context_start = frame.len().saturating_sub(config.context_size_samples);
            self.context = frame[context_start..].to_vec();
        }

        if request.final_chunk {
            let final_event = self.finalize(config);
            if final_event.is_some() {
                event = final_event;
            }
        }

        Ok(VadChunkResponse {
            protocol_version: PSIONIC_VAD_PROTOCOL_VERSION.to_string(),
            session_id: request.session_id,
            chunk_index: request.chunk_index,
            execution_engine: config.execution_engine.clone(),
            model_artifact_id: config.model_artifact_id.clone(),
            input_samples,
            inference_samples: resampled.len(),
            processed_frames,
            speech_probability: self.latest_probability,
            smoothed_probability: self.smoothed_probability,
            speech_active: self.triggered,
            event,
            buffered_inference_samples: self.buffered.len(),
            state_digest: self.state_digest(config),
        })
    }

    fn process_frame(
        &mut self,
        config: &VadWorkerConfig,
        frame: &[f32],
    ) -> Option<VadEndpointEvent> {
        let probability = silero_style_probability(frame, self.noise_floor);
        self.latest_probability = probability;
        self.smoothed_probability = (0.72 * self.smoothed_probability) + (0.28 * probability);

        if !self.triggered {
            self.noise_floor = (0.995 * self.noise_floor) + (0.005 * frame_rms(frame));
        }

        let frame_start = self.sample_cursor;
        self.sample_cursor = self
            .sample_cursor
            .saturating_add(config.frame_size_samples as u64);

        if self.smoothed_probability >= config.threshold {
            self.temp_end_sample = None;
            if !self.triggered {
                self.triggered = true;
                let pad = samples_from_ms(config.speech_pad_ms, config.inference_sample_rate_hz);
                let start = frame_start.saturating_sub(pad);
                self.speech_start_sample = Some(start);
                return Some(VadEndpointEvent::SpeechStart {
                    start_sample: start,
                    reason: "probability_above_threshold".to_string(),
                });
            }
        }

        if self.triggered
            && self.speech_start_sample.is_some_and(|start| {
                self.sample_cursor.saturating_sub(start)
                    >= u64::from(config.max_speech_duration_ms)
                        * u64::from(config.inference_sample_rate_hz)
                        / 1000
            })
        {
            return self.end_speech(config, "max_speech_duration");
        }

        if self.triggered && self.smoothed_probability < config.negative_threshold {
            if self.temp_end_sample.is_none() {
                self.temp_end_sample = Some(self.sample_cursor);
            }
            let min_silence = samples_from_ms(
                config.min_silence_duration_ms,
                config.inference_sample_rate_hz,
            );
            if self
                .temp_end_sample
                .is_some_and(|temp_end| self.sample_cursor.saturating_sub(temp_end) >= min_silence)
            {
                return self.end_speech(config, "min_silence_duration");
            }
        }

        None
    }

    fn end_speech(&mut self, config: &VadWorkerConfig, reason: &str) -> Option<VadEndpointEvent> {
        let start = self.speech_start_sample.unwrap_or(self.sample_cursor);
        let pad = samples_from_ms(config.speech_pad_ms, config.inference_sample_rate_hz);
        let end_base = self.temp_end_sample.unwrap_or(self.sample_cursor);
        let end = end_base.saturating_add(pad);
        self.triggered = false;
        self.temp_end_sample = None;
        self.speech_start_sample = None;
        let min_speech = samples_from_ms(
            config.min_speech_duration_ms,
            config.inference_sample_rate_hz,
        );
        if end.saturating_sub(start) < min_speech {
            return Some(VadEndpointEvent::NoSpeech {
                reason: "speech_shorter_than_min_speech_duration".to_string(),
            });
        }
        Some(VadEndpointEvent::SpeechEnd {
            start_sample: start,
            end_sample: end,
            reason: reason.to_string(),
        })
    }

    fn finalize(&mut self, config: &VadWorkerConfig) -> Option<VadEndpointEvent> {
        if self.triggered {
            return self.end_speech(config, "final_chunk");
        }
        if self.sample_cursor == 0 {
            return Some(VadEndpointEvent::NoSpeech {
                reason: "no_audio_frames".to_string(),
            });
        }
        None
    }

    fn state_digest(&self, config: &VadWorkerConfig) -> String {
        let mut hasher = Sha256::new();
        hasher.update(PSIONIC_VAD_PROTOCOL_VERSION.as_bytes());
        hasher.update(config.digest().as_bytes());
        hasher.update(self.session_id.as_bytes());
        hasher.update(self.sample_cursor.to_le_bytes());
        hasher.update(self.smoothed_probability.to_le_bytes());
        hasher.update(self.latest_probability.to_le_bytes());
        hasher.update([u8::from(self.triggered)]);
        if let Some(start) = self.speech_start_sample {
            hasher.update(start.to_le_bytes());
        }
        format!("sha256:{:x}", hasher.finalize())
    }
}

fn validate_input_rate(sample_rate_hz: u32) -> Result<(), VadError> {
    match sample_rate_hz {
        8_000 | 16_000 | 24_000 | 48_000 => Ok(()),
        _ => Err(VadError::UnsupportedInputSampleRate { sample_rate_hz }),
    }
}

fn downmix_to_mono(input: &[i16], channels: u16) -> Result<Vec<f32>, VadError> {
    if channels == 0 {
        return Err(VadError::InvalidRequest {
            reason: "channels must be greater than zero",
        });
    }
    let channels = usize::from(channels);
    if channels == 1 {
        return Ok(input
            .iter()
            .map(|sample| f32::from(*sample) / f32::from(i16::MAX))
            .collect());
    }
    if !input.len().is_multiple_of(channels) {
        return Err(VadError::InvalidRequest {
            reason: "interleaved sample count is not divisible by channels",
        });
    }
    Ok(input
        .chunks_exact(channels)
        .map(|frame| {
            let sum: f32 = frame
                .iter()
                .map(|sample| f32::from(*sample) / f32::from(i16::MAX))
                .sum();
            sum / channels as f32
        })
        .collect())
}

fn resample_to_inference_rate(
    input: &[f32],
    input_sample_rate_hz: u32,
    inference_sample_rate_hz: u32,
) -> Result<Vec<f32>, VadError> {
    validate_input_rate(input_sample_rate_hz)?;
    if input_sample_rate_hz == inference_sample_rate_hz {
        return Ok(input.to_vec());
    }
    let ratio = f64::from(input_sample_rate_hz) / f64::from(inference_sample_rate_hz);
    let output_len = ((input.len() as f64) / ratio).floor() as usize;
    let mut output = Vec::with_capacity(output_len);
    for output_index in 0..output_len {
        let source = output_index as f64 * ratio;
        let left = source.floor() as usize;
        let right = (left + 1).min(input.len().saturating_sub(1));
        let fraction = (source - left as f64) as f32;
        let sample = input[left] * (1.0 - fraction) + input[right] * fraction;
        output.push(sample);
    }
    Ok(output)
}

fn silero_style_probability(frame: &[f32], noise_floor: f32) -> f32 {
    let rms = frame_rms(frame);
    let peak = frame
        .iter()
        .copied()
        .fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
    let zcr = zero_crossing_rate(frame);
    let snr = rms / noise_floor.max(0.000_5);
    let energy_score = sigmoid((snr - 2.4) * 1.35);
    let peak_score = sigmoid((peak - 0.06) * 22.0);
    let zcr_penalty = if zcr > 0.32 { 0.74 } else { 1.0 };
    ((0.78 * energy_score + 0.22 * peak_score) * zcr_penalty).clamp(0.0, 1.0)
}

fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let energy: f32 = frame.iter().map(|sample| sample * sample).sum();
    (energy / frame.len() as f32).sqrt()
}

fn zero_crossing_rate(frame: &[f32]) -> f32 {
    if frame.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0_u32;
    for pair in frame.windows(2) {
        if pair[0].is_sign_positive() != pair[1].is_sign_positive() {
            crossings += 1;
        }
    }
    crossings as f32 / (frame.len() - 1) as f32
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn samples_from_ms(ms: u32, sample_rate_hz: u32) -> u64 {
    (u64::from(ms) * u64::from(sample_rate_hz)) / 1000
}

/// Builds one synthetic speech fixture for tests and local smoke runs.
#[must_use]
pub fn synthetic_speech_fixture(sample_rate_hz: u32) -> Vec<i16> {
    let silence_a = vec![0_i16; (sample_rate_hz as usize * 120) / 1000];
    let speech_len = (sample_rate_hz as usize * 520) / 1000;
    let silence_b = vec![0_i16; (sample_rate_hz as usize * 260) / 1000];
    let mut samples = silence_a;
    for index in 0..speech_len {
        let phase = (index as f32 / sample_rate_hz as f32) * 2.0 * std::f32::consts::PI * 180.0;
        let envelope = if index < sample_rate_hz as usize / 20 {
            index as f32 / (sample_rate_hz as f32 / 20.0)
        } else {
            1.0
        };
        samples.push((phase.sin() * envelope * 10_000.0) as i16);
    }
    samples.extend(silence_b);
    samples
}

/// Runs the built-in fixture through the worker and returns the responses.
pub fn run_builtin_fixture_smoke() -> Result<Vec<VadChunkResponse>, VadError> {
    let sample_rate = 48_000;
    let mut worker = PsionicVadWorker::default_worker()?;
    worker.start_session(VadSessionConfig::mono("fixture_speech", sample_rate))?;
    let samples = synthetic_speech_fixture(sample_rate);
    let chunk_size = (sample_rate as usize * 20) / 1000;
    let mut responses = Vec::new();
    for (chunk_index, chunk) in samples.chunks(chunk_size).enumerate() {
        let response = worker.infer_chunk(VadChunkRequest::mono(
            "fixture_speech",
            chunk_index as u64,
            sample_rate,
            chunk.to_vec(),
            false,
        ))?;
        responses.push(response);
    }
    responses.push(worker.flush_session("fixture_speech", responses.len() as u64)?);
    Ok(responses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_smoke_detects_speech_end() {
        let responses = match run_builtin_fixture_smoke() {
            Ok(responses) => responses,
            Err(error) => panic!("fixture smoke failed: {error}"),
        };
        assert!(responses.iter().any(|response| {
            matches!(
                response.event,
                Some(VadEndpointEvent::SpeechStart { .. })
                    | Some(VadEndpointEvent::SpeechEnd { .. })
            )
        }));
        assert!(responses.iter().any(|response| {
            matches!(response.event, Some(VadEndpointEvent::SpeechEnd { .. }))
        }));
    }

    #[test]
    fn silent_final_chunk_returns_no_speech() {
        let mut worker = match PsionicVadWorker::default_worker() {
            Ok(worker) => worker,
            Err(error) => panic!("worker init failed: {error}"),
        };
        let response = match worker.infer_chunk(VadChunkRequest::mono(
            "silent",
            0,
            DEFAULT_INFERENCE_SAMPLE_RATE_HZ,
            Vec::new(),
            true,
        )) {
            Ok(response) => response,
            Err(error) => panic!("silent final failed: {error}"),
        };
        assert!(matches!(
            response.event,
            Some(VadEndpointEvent::NoSpeech {
                reason
            }) if reason == "no_audio_frames"
        ));
    }

    #[test]
    fn unsupported_sample_rate_fails_closed() {
        let mut worker = match PsionicVadWorker::default_worker() {
            Ok(worker) => worker,
            Err(error) => panic!("worker init failed: {error}"),
        };
        let error = match worker.infer_chunk(VadChunkRequest::mono(
            "bad_rate",
            0,
            44_100,
            vec![0; 128],
            false,
        )) {
            Ok(_) => panic!("bad rate should fail"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            VadError::UnsupportedInputSampleRate {
                sample_rate_hz: 44_100
            }
        );
    }

    #[test]
    fn fixture_manifest_validates_default_config() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/vad/model_artifacts/silero_style_vad_mvp_manifest.v1.json");
        let manifest = match load_model_artifact_manifest(path) {
            Ok(manifest) => manifest,
            Err(error) => panic!("manifest load failed: {error}"),
        };
        let config = VadWorkerConfig::default();
        if let Err(error) = manifest.validate_for_config(&config) {
            panic!("manifest validation failed: {error}");
        }
        assert_eq!(manifest.artifact_id, PSIONIC_VAD_MODEL_ARTIFACT_ID);
        assert!(!manifest.provider_keys_required);
        assert!(!manifest.raw_audio_retention_required);
    }
}
