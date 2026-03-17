//! Bounded MLX-style audio package with CPU-reference synthesis and codec I/O.

use std::{f32::consts::PI, fs, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str =
    "bounded MLX-style audio package with CPU-reference synthesis, WAV IO, codec helpers, and server-facing request contracts";

/// Supported audio task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxAudioTask {
    /// Text-to-speech generation.
    TextToSpeech,
    /// Speech-to-speech transformation.
    SpeechToSpeech,
    /// Codec/transcode IO helper.
    Codec,
}

/// Supported conditioning mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxAudioConditioningMode {
    /// No explicit conditioning.
    None,
    /// Stable voice label only.
    VoiceLabel,
    /// Reference audio clip conditioning.
    ReferenceAudio,
}

/// Supported checkpoint quantization descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxAudioCheckpointDescriptor {
    /// Canonical quantization label.
    pub label: String,
    /// Storage format for the checkpoint.
    pub format: String,
}

/// One registered audio model family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxAudioModelRegistration {
    /// Canonical family label.
    pub canonical_family: String,
    /// Accepted aliases.
    pub aliases: Vec<String>,
    /// Supported task set.
    pub tasks: Vec<MlxAudioTask>,
    /// Supported conditioning modes.
    pub conditioning_modes: Vec<MlxAudioConditioningMode>,
    /// Supported quantized checkpoint descriptors.
    pub quantized_checkpoints: Vec<MlxAudioCheckpointDescriptor>,
}

/// Bounded registry for MLX-style audio families.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxAudioModelRegistry {
    registrations: std::collections::BTreeMap<String, MlxAudioModelRegistration>,
}

impl Default for MlxAudioModelRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl MlxAudioModelRegistry {
    /// Returns the builtin registry.
    #[must_use]
    pub fn builtin() -> Self {
        let mut registry = Self {
            registrations: std::collections::BTreeMap::new(),
        };
        registry.register(MlxAudioModelRegistration {
            canonical_family: String::from("kokoro"),
            aliases: vec![String::from("kokoro"), String::from("kokoro_tts")],
            tasks: vec![MlxAudioTask::TextToSpeech],
            conditioning_modes: vec![
                MlxAudioConditioningMode::None,
                MlxAudioConditioningMode::VoiceLabel,
            ],
            quantized_checkpoints: vec![
                MlxAudioCheckpointDescriptor {
                    label: String::from("q4_k"),
                    format: String::from("gguf"),
                },
                MlxAudioCheckpointDescriptor {
                    label: String::from("q8_0"),
                    format: String::from("gguf"),
                },
            ],
        });
        registry.register(MlxAudioModelRegistration {
            canonical_family: String::from("xtts"),
            aliases: vec![String::from("xtts"), String::from("xtts_v2")],
            tasks: vec![MlxAudioTask::TextToSpeech, MlxAudioTask::SpeechToSpeech],
            conditioning_modes: vec![
                MlxAudioConditioningMode::VoiceLabel,
                MlxAudioConditioningMode::ReferenceAudio,
            ],
            quantized_checkpoints: vec![
                MlxAudioCheckpointDescriptor {
                    label: String::from("q4_k"),
                    format: String::from("gguf"),
                },
                MlxAudioCheckpointDescriptor {
                    label: String::from("q6_k"),
                    format: String::from("gguf"),
                },
            ],
        });
        registry.register(MlxAudioModelRegistration {
            canonical_family: String::from("encodec"),
            aliases: vec![String::from("encodec"), String::from("codec")],
            tasks: vec![MlxAudioTask::Codec],
            conditioning_modes: vec![MlxAudioConditioningMode::None],
            quantized_checkpoints: vec![MlxAudioCheckpointDescriptor {
                label: String::from("q8_0"),
                format: String::from("gguf"),
            }],
        });
        registry
    }

    /// Registers one audio model family and aliases.
    pub fn register(&mut self, registration: MlxAudioModelRegistration) {
        let canonical = normalize_family_key(registration.canonical_family.as_str());
        self.registrations
            .insert(canonical, registration.clone());
        for alias in &registration.aliases {
            self.registrations
                .insert(normalize_family_key(alias.as_str()), registration.clone());
        }
    }

    /// Resolves one family or alias.
    #[must_use]
    pub fn resolve(&self, family: &str) -> Option<&MlxAudioModelRegistration> {
        self.registrations.get(&normalize_family_key(family))
    }
}

/// Stable audio clip held in native f32 samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxAudioClip {
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Channel count.
    pub channels: u16,
    /// Interleaved PCM samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
}

impl MlxAudioClip {
    /// Creates one audio clip.
    #[must_use]
    pub fn new(sample_rate_hz: u32, channels: u16, samples: Vec<f32>) -> Self {
        Self {
            sample_rate_hz,
            channels: channels.max(1),
            samples,
        }
    }

    /// Returns clip length in frames.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels)
    }

    /// Returns clip length in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        let frames = self.frames() as f64;
        let sample_rate = f64::from(self.sample_rate_hz.max(1));
        ((frames / sample_rate) * 1000.0).round() as u64
    }

    /// Returns one stable digest over clip metadata and samples.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.sample_rate_hz.to_le_bytes());
        hasher.update(self.channels.to_le_bytes());
        for sample in &self.samples {
            hasher.update(sample.to_le_bytes());
        }
        format!("sha256:{:x}", hasher.finalize())
    }

    /// Saves one clip as PCM16 WAV.
    pub fn save_wav(&self, path: impl AsRef<Path>) -> Result<(), MlxAudioError> {
        fs::write(path, encode_wav_pcm16(self)?)?;
        Ok(())
    }

    /// Loads one PCM16 WAV clip.
    pub fn load_wav(path: impl AsRef<Path>) -> Result<Self, MlxAudioError> {
        let bytes = fs::read(path)?;
        decode_wav_pcm16(bytes.as_slice())
    }
}

/// One requested voice-conditioning surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MlxAudioConditioning {
    /// Stable voice label only.
    VoiceLabel {
        /// Stable voice label.
        voice: String,
    },
    /// Reference audio clip conditioning.
    ReferenceAudio {
        /// Reference audio clip.
        clip: MlxAudioClip,
    },
}

impl MlxAudioConditioning {
    fn mode(&self) -> MlxAudioConditioningMode {
        match self {
            Self::VoiceLabel { .. } => MlxAudioConditioningMode::VoiceLabel,
            Self::ReferenceAudio { .. } => MlxAudioConditioningMode::ReferenceAudio,
        }
    }
}

/// Request for text-to-speech generation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxTextToSpeechRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Input text.
    pub text: String,
    /// Optional conditioning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditioning: Option<MlxAudioConditioning>,
    /// Requested output sample rate.
    pub sample_rate_hz: u32,
    /// Chunk size for streaming views.
    pub stream_chunk_frames: usize,
}

impl MlxTextToSpeechRequest {
    /// Creates one text-to-speech request.
    #[must_use]
    pub fn new(request_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            text: text.into(),
            conditioning: None,
            sample_rate_hz: 16_000,
            stream_chunk_frames: 1_024,
        }
    }
}

/// Request for speech-to-speech transformation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxSpeechToSpeechRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Input clip.
    pub input_clip: MlxAudioClip,
    /// Optional transcript hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    /// Optional target conditioning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditioning: Option<MlxAudioConditioning>,
    /// Chunk size for streaming views.
    pub stream_chunk_frames: usize,
}

impl MlxSpeechToSpeechRequest {
    /// Creates one speech-to-speech request.
    #[must_use]
    pub fn new(request_id: impl Into<String>, input_clip: MlxAudioClip) -> Self {
        Self {
            request_id: request_id.into(),
            input_clip,
            transcript: None,
            conditioning: None,
            stream_chunk_frames: 1_024,
        }
    }
}

/// Request for codec transcode/inspection workflows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxCodecRequest {
    /// Stable request identifier.
    pub request_id: String,
    /// Input clip to transcode or inspect.
    pub input_clip: MlxAudioClip,
}

/// OpenAI-compatible audio speech request shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxAudioSpeechRequest {
    /// Model family or model id.
    pub model: String,
    /// Input text.
    pub input: String,
    /// Optional voice label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// Requested response format.
    pub response_format: String,
    /// Whether streaming chunks are requested.
    pub stream: bool,
}

/// One streamed output chunk.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxAudioStreamChunk {
    /// Stable chunk index.
    pub chunk_index: usize,
    /// Starting frame index within the rendered clip.
    pub start_frame: usize,
    /// Chunk-local samples.
    pub samples: Vec<f32>,
}

/// Response for one audio generation request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxAudioSynthesisReport {
    /// Model family used by the reference runtime.
    pub family: String,
    /// Executed task.
    pub task: MlxAudioTask,
    /// Conditioning mode that influenced synthesis.
    pub conditioning_mode: MlxAudioConditioningMode,
    /// Output clip.
    pub output_clip: MlxAudioClip,
    /// Stream chunks in stable order.
    pub stream_chunks: Vec<MlxAudioStreamChunk>,
    /// Stable digest for the output clip.
    pub clip_digest: String,
    /// Honest notes for the current bounded lane.
    pub notes: Vec<String>,
}

/// Response for one server-facing speech request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxAudioSpeechResponse {
    /// MIME content type.
    pub content_type: String,
    /// Clip digest.
    pub clip_digest: String,
    /// Output clip.
    pub clip: MlxAudioClip,
    /// Optional streaming chunks when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_chunks: Option<Vec<MlxAudioStreamChunk>>,
}

/// Error returned by the bounded audio package.
#[derive(Debug, Error)]
pub enum MlxAudioError {
    /// The requested family is unknown.
    #[error("unknown MLX audio family `{family}`")]
    UnknownFamily {
        /// Requested family label.
        family: String,
    },
    /// The requested task is unsupported for the selected family.
    #[error("audio family `{family}` does not support `{task:?}`")]
    UnsupportedTask {
        /// Canonical family label.
        family: String,
        /// Requested task.
        task: MlxAudioTask,
    },
    /// The selected conditioning mode is unsupported for the selected family.
    #[error("audio family `{family}` does not support conditioning mode `{mode:?}`")]
    UnsupportedConditioning {
        /// Canonical family label.
        family: String,
        /// Requested conditioning mode.
        mode: MlxAudioConditioningMode,
    },
    /// Reading or writing an audio file failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The WAV container is invalid or unsupported.
    #[error("{0}")]
    InvalidWav(String),
}

/// CPU-reference runtime for the bounded audio package.
#[derive(Clone, Debug)]
pub struct MlxAudioReferenceRuntime {
    registry: MlxAudioModelRegistry,
}

impl Default for MlxAudioReferenceRuntime {
    fn default() -> Self {
        Self::new(MlxAudioModelRegistry::builtin())
    }
}

impl MlxAudioReferenceRuntime {
    /// Creates one reference runtime over the provided registry.
    #[must_use]
    pub fn new(registry: MlxAudioModelRegistry) -> Self {
        Self { registry }
    }

    /// Returns the model registry.
    #[must_use]
    pub fn registry(&self) -> &MlxAudioModelRegistry {
        &self.registry
    }

    /// Runs one text-to-speech request.
    pub fn synthesize_text(
        &self,
        family: &str,
        request: &MlxTextToSpeechRequest,
    ) -> Result<MlxAudioSynthesisReport, MlxAudioError> {
        let registration = self.require_family_support(family, MlxAudioTask::TextToSpeech)?;
        let conditioning_mode = validate_conditioning(registration, request.conditioning.as_ref())?;
        let clip = synthesize_reference_clip(
            request.text.as_str(),
            request.sample_rate_hz,
            conditioning_seed(request.conditioning.as_ref()),
        );
        Ok(build_report(
            registration.canonical_family.as_str(),
            MlxAudioTask::TextToSpeech,
            conditioning_mode,
            clip,
            request.stream_chunk_frames,
            String::from(
                "Current MLX audio support is a CPU-reference contract lane: it owns the request/response, streaming, WAV IO, conditioning, and checkpoint metadata surfaces, but it does not claim human-quality TTS.",
            ),
        ))
    }

    /// Runs one speech-to-speech request.
    pub fn synthesize_speech_to_speech(
        &self,
        family: &str,
        request: &MlxSpeechToSpeechRequest,
    ) -> Result<MlxAudioSynthesisReport, MlxAudioError> {
        let registration = self.require_family_support(family, MlxAudioTask::SpeechToSpeech)?;
        let conditioning_mode = validate_conditioning(registration, request.conditioning.as_ref())?;
        let clip = transform_reference_clip(
            &request.input_clip,
            request.transcript.as_deref(),
            conditioning_seed(request.conditioning.as_ref()),
        );
        Ok(build_report(
            registration.canonical_family.as_str(),
            MlxAudioTask::SpeechToSpeech,
            conditioning_mode,
            clip,
            request.stream_chunk_frames,
            String::from(
                "Speech-to-speech remains a bounded reference transform over the input waveform plus explicit conditioning; it does not claim a production speech model.",
            ),
        ))
    }

    /// Runs one codec helper request.
    pub fn run_codec(
        &self,
        family: &str,
        request: &MlxCodecRequest,
    ) -> Result<MlxAudioSynthesisReport, MlxAudioError> {
        let registration = self.require_family_support(family, MlxAudioTask::Codec)?;
        let clip = normalize_clip(&request.input_clip);
        Ok(build_report(
            registration.canonical_family.as_str(),
            MlxAudioTask::Codec,
            MlxAudioConditioningMode::None,
            clip,
            1_024,
            String::from(
                "Codec mode is a bounded clip-normalization and container-IO helper, not a learned neural codec claim.",
            ),
        ))
    }

    /// Handles one server-facing speech request through the reference lane.
    pub fn handle_speech_request(
        &self,
        request: &MlxAudioSpeechRequest,
    ) -> Result<MlxAudioSpeechResponse, MlxAudioError> {
        let report = self.synthesize_text(
            request.model.as_str(),
            &MlxTextToSpeechRequest {
                request_id: String::from("audio-speech"),
                text: request.input.clone(),
                conditioning: request
                    .voice
                    .as_ref()
                    .map(|voice| MlxAudioConditioning::VoiceLabel {
                        voice: voice.clone(),
                    }),
                sample_rate_hz: 16_000,
                stream_chunk_frames: 1_024,
            },
        )?;
        Ok(MlxAudioSpeechResponse {
            content_type: content_type_for_format(request.response_format.as_str()),
            clip_digest: report.clip_digest,
            clip: report.output_clip,
            stream_chunks: request.stream.then_some(report.stream_chunks),
        })
    }

    fn require_family_support(
        &self,
        family: &str,
        task: MlxAudioTask,
    ) -> Result<&MlxAudioModelRegistration, MlxAudioError> {
        let registration = self
            .registry
            .resolve(family)
            .ok_or_else(|| MlxAudioError::UnknownFamily {
                family: family.to_string(),
            })?;
        if !registration.tasks.contains(&task) {
            return Err(MlxAudioError::UnsupportedTask {
                family: registration.canonical_family.clone(),
                task,
            });
        }
        Ok(registration)
    }
}

fn validate_conditioning(
    registration: &MlxAudioModelRegistration,
    conditioning: Option<&MlxAudioConditioning>,
) -> Result<MlxAudioConditioningMode, MlxAudioError> {
    let mode = conditioning
        .map(MlxAudioConditioning::mode)
        .unwrap_or(MlxAudioConditioningMode::None);
    if !registration.conditioning_modes.contains(&mode) {
        return Err(MlxAudioError::UnsupportedConditioning {
            family: registration.canonical_family.clone(),
            mode,
        });
    }
    Ok(mode)
}

fn synthesize_reference_clip(
    text: &str,
    sample_rate_hz: u32,
    conditioning_seed: u64,
) -> MlxAudioClip {
    let effective_rate = sample_rate_hz.max(8_000);
    let mut samples = Vec::new();
    for (index, byte) in text.bytes().enumerate() {
        let frequency = 180.0
            + (f32::from(byte % 32) * 12.0)
            + ((conditioning_seed % 29) as f32);
        let frame_count = effective_rate / 30;
        for frame in 0..frame_count {
            let phase = 2.0 * PI * frequency * (frame as f32 / effective_rate as f32);
            let envelope = (frame as f32 / frame_count.max(1) as f32).min(1.0);
            let amplitude = 0.18 + (index % 3) as f32 * 0.03;
            samples.push((phase.sin() * amplitude * envelope).clamp(-1.0, 1.0));
        }
        samples.extend(std::iter::repeat_n(0.0, usize::try_from(effective_rate / 200).unwrap_or(0)));
    }
    if samples.is_empty() {
        samples.extend(std::iter::repeat_n(0.0, usize::try_from(effective_rate / 10).unwrap_or(0)));
    }
    MlxAudioClip::new(effective_rate, 1, samples)
}

fn transform_reference_clip(
    input: &MlxAudioClip,
    transcript: Option<&str>,
    conditioning_seed: u64,
) -> MlxAudioClip {
    let transcript_bias = transcript
        .map(|value| value.bytes().fold(0_u64, |acc, byte| acc + u64::from(byte)) % 17)
        .unwrap_or(0) as f32;
    let scale = 0.85 + transcript_bias * 0.01 + (conditioning_seed % 13) as f32 * 0.005;
    let samples = input
        .samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let phase = ((index % 64) as f32 / 64.0) * 2.0 * PI;
            let wobble = 0.05 * phase.sin();
            (sample * scale + wobble).clamp(-1.0, 1.0)
        })
        .collect::<Vec<_>>();
    MlxAudioClip::new(input.sample_rate_hz, input.channels, samples)
}

fn normalize_clip(input: &MlxAudioClip) -> MlxAudioClip {
    let peak = input
        .samples
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
        .max(1e-6);
    let samples = input
        .samples
        .iter()
        .map(|sample| (sample / peak).clamp(-1.0, 1.0))
        .collect::<Vec<_>>();
    MlxAudioClip::new(input.sample_rate_hz, input.channels, samples)
}

fn conditioning_seed(conditioning: Option<&MlxAudioConditioning>) -> u64 {
    match conditioning {
        Some(MlxAudioConditioning::VoiceLabel { voice }) => voice
            .bytes()
            .fold(0_u64, |acc, byte| acc.wrapping_mul(131).wrapping_add(u64::from(byte))),
        Some(MlxAudioConditioning::ReferenceAudio { clip }) => clip
            .digest()
            .bytes()
            .fold(0_u64, |acc, byte| acc.wrapping_mul(109).wrapping_add(u64::from(byte))),
        None => 0,
    }
}

fn build_report(
    family: &str,
    task: MlxAudioTask,
    conditioning_mode: MlxAudioConditioningMode,
    clip: MlxAudioClip,
    chunk_frames: usize,
    note: String,
) -> MlxAudioSynthesisReport {
    let clip_digest = clip.digest();
    let stream_chunks = chunk_clip(&clip, chunk_frames.max(1));
    MlxAudioSynthesisReport {
        family: family.to_string(),
        task,
        conditioning_mode,
        output_clip: clip,
        stream_chunks,
        clip_digest,
        notes: vec![note],
    }
}

fn chunk_clip(clip: &MlxAudioClip, chunk_frames: usize) -> Vec<MlxAudioStreamChunk> {
    let mut chunks = Vec::new();
    let frame_width = usize::from(clip.channels);
    let total_frames = clip.frames();
    let mut frame = 0usize;
    while frame < total_frames {
        let end_frame = (frame + chunk_frames).min(total_frames);
        let start_sample = frame * frame_width;
        let end_sample = end_frame * frame_width;
        chunks.push(MlxAudioStreamChunk {
            chunk_index: chunks.len(),
            start_frame: frame,
            samples: clip.samples[start_sample..end_sample].to_vec(),
        });
        frame = end_frame;
    }
    chunks
}

fn content_type_for_format(format: &str) -> String {
    match format {
        "wav" | "pcm16" => String::from("audio/wav"),
        "json" => String::from("application/json"),
        _ => String::from("audio/wav"),
    }
}

fn normalize_family_key(value: &str) -> String {
    value
        .chars()
        .filter(|character| *character != '_' && *character != '-')
        .flat_map(char::to_lowercase)
        .collect()
}

fn encode_wav_pcm16(clip: &MlxAudioClip) -> Result<Vec<u8>, MlxAudioError> {
    let data_len = clip
        .samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| MlxAudioError::InvalidWav(String::from("wav data too large")))?;
    let riff_len = 36usize
        .checked_add(data_len)
        .ok_or_else(|| MlxAudioError::InvalidWav(String::from("wav riff too large")))?;
    let byte_rate = clip
        .sample_rate_hz
        .checked_mul(u32::from(clip.channels))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| MlxAudioError::InvalidWav(String::from("wav byte rate overflow")))?;
    let block_align = clip.channels.saturating_mul(2);

    let mut bytes = Vec::with_capacity(riff_len + 8);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(riff_len as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&clip.channels.to_le_bytes());
    bytes.extend_from_slice(&clip.sample_rate_hz.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_len as u32).to_le_bytes());
    for sample in &clip.samples {
        let value = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_wav_pcm16(bytes: &[u8]) -> Result<MlxAudioClip, MlxAudioError> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(MlxAudioError::InvalidWav(String::from(
            "invalid RIFF/WAVE header",
        )));
    }
    if &bytes[12..16] != b"fmt " {
        return Err(MlxAudioError::InvalidWav(String::from(
            "unsupported wav layout: missing fmt chunk",
        )));
    }
    let audio_format = u16::from_le_bytes([bytes[20], bytes[21]]);
    if audio_format != 1 {
        return Err(MlxAudioError::InvalidWav(String::from(
            "only PCM16 WAV is supported",
        )));
    }
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
    let sample_rate_hz = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    if &bytes[36..40] != b"data" {
        return Err(MlxAudioError::InvalidWav(String::from(
            "unsupported wav layout: missing data chunk",
        )));
    }
    let data_len = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize;
    let data_end = 44usize
        .checked_add(data_len)
        .ok_or_else(|| MlxAudioError::InvalidWav(String::from("wav data length overflow")))?;
    if data_end > bytes.len() || data_len % 2 != 0 {
        return Err(MlxAudioError::InvalidWav(String::from(
            "wav data chunk is truncated",
        )));
    }
    let mut samples = Vec::with_capacity(data_len / 2);
    for chunk in bytes[44..data_end].chunks_exact(2) {
        let value = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(f32::from(value) / f32::from(i16::MAX));
    }
    Ok(MlxAudioClip::new(sample_rate_hz, channels, samples))
}

#[cfg(test)]
mod tests {
    use super::{
        MlxAudioClip, MlxAudioConditioning, MlxAudioModelRegistry, MlxAudioReferenceRuntime,
        MlxAudioSpeechRequest, MlxAudioTask, MlxCodecRequest, MlxSpeechToSpeechRequest,
        MlxTextToSpeechRequest,
    };

    #[test]
    fn builtin_registry_tracks_tasks_and_quantized_checkpoints() {
        let registry = MlxAudioModelRegistry::builtin();
        let xtts = registry.resolve("xtts_v2").expect("xtts alias");
        assert!(xtts.tasks.contains(&MlxAudioTask::SpeechToSpeech));
        assert!(
            xtts.quantized_checkpoints
                .iter()
                .any(|descriptor| descriptor.label == "q6_k")
        );
    }

    #[test]
    fn wav_round_trip_preserves_shape() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("clip.wav");
        let clip = MlxAudioClip::new(16_000, 1, vec![0.0, 0.5, -0.5, 0.25]);
        clip.save_wav(&path)?;
        let loaded = MlxAudioClip::load_wav(&path)?;
        assert_eq!(loaded.sample_rate_hz, 16_000);
        assert_eq!(loaded.channels, 1);
        assert_eq!(loaded.samples.len(), clip.samples.len());
        Ok(())
    }

    #[test]
    fn text_to_speech_and_stream_chunks_are_machine_checkable()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = MlxAudioReferenceRuntime::default();
        let report = runtime.synthesize_text(
            "kokoro",
            &MlxTextToSpeechRequest {
                request_id: String::from("tts"),
                text: String::from("hello psionic"),
                conditioning: Some(MlxAudioConditioning::VoiceLabel {
                    voice: String::from("alto"),
                }),
                sample_rate_hz: 12_000,
                stream_chunk_frames: 256,
            },
        )?;

        assert!(!report.output_clip.samples.is_empty());
        assert!(!report.stream_chunks.is_empty());
        assert!(report.clip_digest.starts_with("sha256:"));
        Ok(())
    }

    #[test]
    fn speech_to_speech_and_server_request_surfaces_work()
    -> Result<(), Box<dyn std::error::Error>> {
        let runtime = MlxAudioReferenceRuntime::default();
        let input = MlxAudioClip::new(16_000, 1, vec![0.1; 1_024]);
        let report = runtime.synthesize_speech_to_speech(
            "xtts",
            &MlxSpeechToSpeechRequest {
                request_id: String::from("s2s"),
                input_clip: input.clone(),
                transcript: Some(String::from("rewrite this")),
                conditioning: Some(MlxAudioConditioning::ReferenceAudio { clip: input }),
                stream_chunk_frames: 128,
            },
        )?;
        assert_eq!(report.output_clip.sample_rate_hz, 16_000);

        let response = runtime.handle_speech_request(&MlxAudioSpeechRequest {
            model: String::from("kokoro"),
            input: String::from("speak"),
            voice: Some(String::from("alto")),
            response_format: String::from("wav"),
            stream: true,
        })?;
        assert_eq!(response.content_type, "audio/wav");
        assert!(response.stream_chunks.is_some());
        Ok(())
    }

    #[test]
    fn codec_mode_normalizes_input() -> Result<(), Box<dyn std::error::Error>> {
        let runtime = MlxAudioReferenceRuntime::default();
        let report = runtime.run_codec(
            "codec",
            &MlxCodecRequest {
                request_id: String::from("codec"),
                input_clip: MlxAudioClip::new(16_000, 1, vec![0.2, 0.4, -0.8]),
            },
        )?;
        let peak = report
            .output_clip
            .samples
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
        assert!((peak - 1.0).abs() < 1e-3);
        Ok(())
    }
}
