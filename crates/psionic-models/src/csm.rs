//! Rust-native CSM prompt, tokenizer, and artifact descriptors.

use std::{
    env, fmt, fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokenizers::{Tokenizer, processors::template::TemplateProcessing};

use crate::{
    CsmAudioMetadata, CsmMimiCodebookPrefix, CsmParityPrompt, CsmPythonParityFixture,
    validate_csm_python_parity_fixture,
};

/// Psionic-owned CSM model id.
pub const CSM_MODEL_ID: &str = "sesame/csm-1b";
/// CSM uses one 32-codebook audio lane group and one text lane.
pub const CSM_FRAME_LANES: usize = 33;
/// CSM audio codebook lane count.
pub const CSM_AUDIO_CODEBOOK_LANES: usize = 32;
/// CSM text lane index inside each 33-lane frame.
pub const CSM_TEXT_LANE_INDEX: usize = 32;
/// Python reference max sequence length.
pub const CSM_MAX_SEQ_LEN: usize = 2048;
/// One generated Mimi frame covers 80 ms at the reference frame rate.
pub const CSM_GENERATION_FRAME_MS: u64 = 80;
/// Native CSM/Mimi sample rate.
pub const CSM_SAMPLE_RATE_HZ: u32 = 24_000;
/// Source prompt WAV sample rate from Sesame's prompt assets.
pub const CSM_PROMPT_SOURCE_SAMPLE_RATE_HZ: u32 = 44_100;
/// CSM codebook EOS token id.
pub const CSM_CODEBOOK_EOS_TOKEN_ID: u32 = 0;
/// CSM codebook pad token id.
pub const CSM_CODEBOOK_PAD_TOKEN_ID: u32 = 2050;
/// Llama 3.2 BOS token used by the CSM text encoder.
pub const CSM_LLAMA_BOS_TOKEN_ID: u32 = 128_000;
/// Llama 3.2 EOS token used by the CSM text encoder.
pub const CSM_LLAMA_EOS_TOKEN_ID: u32 = 128_001;
/// Llama BOS token string in Hugging Face tokenizer JSON.
pub const CSM_LLAMA_BOS_TOKEN: &str = "<|begin_of_text|>";
/// Llama EOS token string in Hugging Face tokenizer JSON.
pub const CSM_LLAMA_EOS_TOKEN: &str = "<|end_of_text|>";
/// Llama tokenizer repo used by CSM.
pub const CSM_LLAMA_TOKENIZER_REPO: &str = "meta-llama/Llama-3.2-1B";

/// CSM frontend, descriptor, and prompt-building errors.
#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum CsmFrontendError {
    /// Artifact digest did not match the expected fixture digest.
    #[error("CSM artifact digest mismatch for {artifact}: expected {expected}, got {actual}")]
    ArtifactDigestMismatch {
        /// Artifact label.
        artifact: String,
        /// Expected sha256 digest.
        expected: String,
        /// Actual sha256 digest.
        actual: String,
    },
    /// Artifact bytes could not be read.
    #[error("failed to read CSM artifact {artifact}: {message}")]
    ArtifactRead {
        /// Artifact label.
        artifact: String,
        /// Error detail.
        message: String,
    },
    /// Tokenizer JSON could not be loaded or configured.
    #[error("failed to load CSM tokenizer: {message}")]
    TokenizerLoad {
        /// Error detail.
        message: String,
    },
    /// Tokenizer encode failed.
    #[error("failed to encode CSM text: {message}")]
    TokenizerEncode {
        /// Error detail.
        message: String,
    },
    /// CSM config JSON could not be parsed.
    #[error("failed to parse CSM config: {message}")]
    ConfigParse {
        /// Error detail.
        message: String,
    },
    /// CSM config values are not compatible with the known CSM frame contract.
    #[error("invalid CSM config: {message}")]
    ConfigContract {
        /// Error detail.
        message: String,
    },
    /// Fixture or descriptor contract is invalid.
    #[error("invalid CSM descriptor fixture: {message}")]
    DescriptorContract {
        /// Error detail.
        message: String,
    },
    /// Audio codebooks are malformed.
    #[error("invalid CSM audio codebooks: {message}")]
    AudioCodebooks {
        /// Error detail.
        message: String,
    },
    /// Prompt plus requested generation exceeds the admitted context window.
    #[error(
        "CSM prompt has {prompt_frames} frames, but must be below max_context_len={max_context_len}"
    )]
    PromptContextOverflow {
        /// Prompt frame count.
        prompt_frames: usize,
        /// Allowed context length.
        max_context_len: usize,
        /// Requested generated frame budget.
        max_generation_len: usize,
    },
    /// Generation duration cannot be converted into a valid generation window.
    #[error("invalid CSM generation window: {message}")]
    GenerationWindow {
        /// Error detail.
        message: String,
    },
}

/// Complete artifact identity needed before loading CSM weights.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsmArtifactDigestSet {
    /// CSM config digest.
    pub csm_config_digest: String,
    /// CSM safetensors digest.
    pub csm_model_digest: String,
    /// Llama tokenizer JSON digest.
    pub llama_tokenizer_digest: String,
    /// Mimi safetensors digest.
    pub mimi_weight_digest: String,
}

/// Stable 33-lane frame contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsmFrameContract {
    /// Total frame width.
    pub frame_lanes: usize,
    /// Number of audio codebook lanes.
    pub audio_codebook_lanes: usize,
    /// Text token lane.
    pub text_lane_index: usize,
    /// Maximum sequence length admitted by CSM.
    pub max_seq_len: usize,
    /// Generated-frame duration.
    pub generation_frame_ms: u64,
    /// CSM/Mimi runtime sample rate.
    pub sample_rate_hz: u32,
    /// Codebook EOS token.
    pub codebook_eos_token_id: u32,
    /// Codebook pad token.
    pub codebook_pad_token_id: u32,
}

impl Default for CsmFrameContract {
    fn default() -> Self {
        Self {
            frame_lanes: CSM_FRAME_LANES,
            audio_codebook_lanes: CSM_AUDIO_CODEBOOK_LANES,
            text_lane_index: CSM_TEXT_LANE_INDEX,
            max_seq_len: CSM_MAX_SEQ_LEN,
            generation_frame_ms: CSM_GENERATION_FRAME_MS,
            sample_rate_hz: CSM_SAMPLE_RATE_HZ,
            codebook_eos_token_id: CSM_CODEBOOK_EOS_TOKEN_ID,
            codebook_pad_token_id: CSM_CODEBOOK_PAD_TOKEN_ID,
        }
    }
}

/// Prompt voice-profile descriptor from the committed CSM fixture.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsmVoiceProfileDescriptor {
    /// Stable profile id.
    pub profile_id: String,
    /// Numeric CSM speaker id.
    pub speaker: u32,
    /// Prompt transcript.
    pub text: String,
    /// Prompt WAV digest.
    pub prompt_audio_sha256: String,
    /// Prompt WAV metadata.
    pub audio: CsmAudioMetadata,
    /// Optional compact Mimi codebook digest for this prompt.
    pub mimi_tokens_sha256: Option<String>,
}

/// CSM model descriptor exported without loading neural weights.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsmModelArtifactDescriptor {
    /// Psionic model id.
    pub model_id: String,
    /// CSM HF repo.
    pub csm_repo: String,
    /// Llama tokenizer repo.
    pub llama_tokenizer_repo: String,
    /// Mimi HF repo.
    pub mimi_repo: String,
    /// Mimi weight file.
    pub mimi_weight: String,
    /// Artifact digests.
    pub digests: CsmArtifactDigestSet,
    /// 33-lane framing contract.
    pub frame_contract: CsmFrameContract,
    /// Prompt voice profiles admitted by this descriptor.
    pub voice_profiles: Vec<CsmVoiceProfileDescriptor>,
}

impl CsmModelArtifactDescriptor {
    /// Builds the descriptor from the committed CSM fixture.
    pub fn from_fixture(fixture: &CsmPythonParityFixture) -> Result<Self, CsmFrontendError> {
        validate_csm_python_parity_fixture(fixture).map_err(|message| {
            CsmFrontendError::DescriptorContract {
                message: message.to_string(),
            }
        })?;

        let voice_profiles = fixture
            .prompts
            .iter()
            .map(|prompt| voice_profile_descriptor(prompt, &fixture.mimi_codebook_prefixes))
            .collect();

        Ok(Self {
            model_id: CSM_MODEL_ID.to_string(),
            csm_repo: fixture.model.csm_repo.clone(),
            llama_tokenizer_repo: fixture.model.llama_tokenizer_repo.clone(),
            mimi_repo: fixture.model.mimi_repo.clone(),
            mimi_weight: fixture.model.mimi_weight.clone(),
            digests: CsmArtifactDigestSet {
                csm_config_digest: fixture.model.csm_config_digest.clone(),
                csm_model_digest: fixture.model.csm_model_digest.clone(),
                llama_tokenizer_digest: fixture.model.llama_tokenizer_digest.clone(),
                mimi_weight_digest: fixture.model.mimi_weight_digest.clone(),
            },
            frame_contract: CsmFrameContract::default(),
            voice_profiles,
        })
    }
}

fn voice_profile_descriptor(
    prompt: &CsmParityPrompt,
    prefixes: &[CsmMimiCodebookPrefix],
) -> CsmVoiceProfileDescriptor {
    CsmVoiceProfileDescriptor {
        profile_id: prompt.profile_id.clone(),
        speaker: prompt.speaker,
        text: prompt.text.clone(),
        prompt_audio_sha256: prompt.audio_sha256.clone(),
        audio: prompt.audio.clone(),
        mimi_tokens_sha256: prefixes
            .iter()
            .find(|prefix| prefix.profile_id == prompt.profile_id)
            .map(|prefix| prefix.tokens_sha256.clone()),
    }
}

/// CSM config root parsed from `config.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CsmModelConfig {
    /// HF architecture labels.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Model type.
    pub model_type: String,
    /// Maximum position embeddings.
    pub max_position_embeddings: usize,
    /// Text vocabulary size.
    pub text_vocab_size: usize,
    /// Audio vocabulary size.
    pub audio_vocab_size: usize,
    /// Main vocab size.
    pub vocab_size: usize,
    /// Audio codebook count.
    pub num_codebooks: usize,
    /// Audio codebook count duplicate from Transformers config.
    pub audio_num_codebooks: usize,
    /// BOS token.
    pub bos_token_id: u32,
    /// Audio token.
    pub audio_token_id: u32,
    /// Audio EOS token.
    pub audio_eos_token_id: u32,
    /// Pad token.
    pub pad_token_id: u32,
    /// Codebook EOS token.
    pub codebook_eos_token_id: u32,
    /// Codebook pad token.
    pub codebook_pad_token_id: u32,
    /// Codec/Mimi nested config.
    pub codec_config: CsmCodecConfig,
    /// Depth decoder nested config.
    pub depth_decoder_config: CsmDepthDecoderConfig,
}

impl CsmModelConfig {
    /// Parses and validates CSM config JSON.
    pub fn from_json_str(json: &str) -> Result<Self, CsmFrontendError> {
        let config: Self =
            serde_json::from_str(json).map_err(|error| CsmFrontendError::ConfigParse {
                message: error.to_string(),
            })?;
        config.validate()?;
        Ok(config)
    }

    /// Parses and validates a CSM config JSON file.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CsmFrontendError> {
        let path = path.as_ref();
        let json = fs::read_to_string(path).map_err(|error| CsmFrontendError::ArtifactRead {
            artifact: path.display().to_string(),
            message: error.to_string(),
        })?;
        Self::from_json_str(&json)
    }

    /// Validates the config against the CSM 33-lane contract.
    pub fn validate(&self) -> Result<(), CsmFrontendError> {
        if self.model_type != "csm" {
            return Err(CsmFrontendError::ConfigContract {
                message: format!("expected model_type=csm, got {}", self.model_type),
            });
        }
        if self.max_position_embeddings != CSM_MAX_SEQ_LEN {
            return Err(CsmFrontendError::ConfigContract {
                message: format!(
                    "expected max_position_embeddings={}, got {}",
                    CSM_MAX_SEQ_LEN, self.max_position_embeddings
                ),
            });
        }
        if self.num_codebooks != CSM_AUDIO_CODEBOOK_LANES
            || self.audio_num_codebooks != CSM_AUDIO_CODEBOOK_LANES
            || self.depth_decoder_config.num_codebooks != CSM_AUDIO_CODEBOOK_LANES
            || self.codec_config.num_quantizers != CSM_AUDIO_CODEBOOK_LANES
        {
            return Err(CsmFrontendError::ConfigContract {
                message: "config does not declare 32 audio codebooks".to_string(),
            });
        }
        if self.codec_config.sampling_rate != CSM_SAMPLE_RATE_HZ {
            return Err(CsmFrontendError::ConfigContract {
                message: format!(
                    "expected codec sampling_rate={}, got {}",
                    CSM_SAMPLE_RATE_HZ, self.codec_config.sampling_rate
                ),
            });
        }
        if self.codebook_eos_token_id != CSM_CODEBOOK_EOS_TOKEN_ID {
            return Err(CsmFrontendError::ConfigContract {
                message: format!(
                    "expected codebook_eos_token_id={}, got {}",
                    CSM_CODEBOOK_EOS_TOKEN_ID, self.codebook_eos_token_id
                ),
            });
        }
        if self.codebook_pad_token_id != CSM_CODEBOOK_PAD_TOKEN_ID {
            return Err(CsmFrontendError::ConfigContract {
                message: format!(
                    "expected codebook_pad_token_id={}, got {}",
                    CSM_CODEBOOK_PAD_TOKEN_ID, self.codebook_pad_token_id
                ),
            });
        }
        Ok(())
    }
}

/// CSM nested Mimi config fields used by the Rust contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CsmCodecConfig {
    /// Codec model type.
    pub model_type: String,
    /// Audio channel count.
    pub audio_channels: usize,
    /// Codebook size.
    pub codebook_size: usize,
    /// Quantizer count.
    pub num_quantizers: usize,
    /// Runtime sample rate.
    pub sampling_rate: u32,
    /// Codec frame rate.
    pub frame_rate: f32,
}

/// CSM depth decoder config fields used by the Rust contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CsmDepthDecoderConfig {
    /// Decoder model type.
    pub model_type: String,
    /// Maximum decoder positions.
    pub max_position_embeddings: usize,
    /// Decoder codebook count.
    pub num_codebooks: usize,
    /// Decoder vocab size.
    pub vocab_size: usize,
}

/// Rust wrapper around Hugging Face's `tokenizer.json` for CSM text.
#[derive(Clone)]
pub struct CsmLlamaTextTokenizer {
    tokenizer: Tokenizer,
    tokenizer_sha256: String,
}

impl fmt::Debug for CsmLlamaTextTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CsmLlamaTextTokenizer")
            .field("tokenizer_sha256", &self.tokenizer_sha256)
            .finish_non_exhaustive()
    }
}

impl CsmLlamaTextTokenizer {
    /// Loads a tokenizer JSON file, validates its digest when supplied, and
    /// installs the BOS/EOS template used by the Python CSM reference.
    pub fn from_tokenizer_json_file(
        path: impl AsRef<Path>,
        expected_sha256: Option<&str>,
    ) -> Result<Self, CsmFrontendError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| CsmFrontendError::ArtifactRead {
            artifact: path.display().to_string(),
            message: error.to_string(),
        })?;
        Self::from_tokenizer_json_bytes(&bytes, expected_sha256)
    }

    /// Loads a tokenizer JSON byte slice.
    pub fn from_tokenizer_json_bytes(
        bytes: &[u8],
        expected_sha256: Option<&str>,
    ) -> Result<Self, CsmFrontendError> {
        let tokenizer_sha256 = sha256_digest(bytes);
        if let Some(expected) = expected_sha256
            && tokenizer_sha256 != expected
        {
            return Err(CsmFrontendError::ArtifactDigestMismatch {
                artifact: "llama_tokenizer_json".to_string(),
                expected: expected.to_string(),
                actual: tokenizer_sha256,
            });
        }

        let mut tokenizer =
            Tokenizer::from_bytes(bytes).map_err(|error| CsmFrontendError::TokenizerLoad {
                message: error.to_string(),
            })?;
        install_csm_llama_template(&mut tokenizer)?;
        Ok(Self {
            tokenizer,
            tokenizer_sha256,
        })
    }

    /// Loads the first matching tokenizer JSON from the local Hugging Face cache.
    pub fn from_default_hf_cache(expected_sha256: Option<&str>) -> Result<Self, CsmFrontendError> {
        let candidates = csm_default_llama_tokenizer_json_candidates();
        let mut errors = Vec::new();
        for candidate in candidates {
            match Self::from_tokenizer_json_file(&candidate, expected_sha256) {
                Ok(tokenizer) => return Ok(tokenizer),
                Err(error) => errors.push(format!("{}: {error}", candidate.display())),
            }
        }
        Err(CsmFrontendError::TokenizerLoad {
            message: format!(
                "no matching {CSM_LLAMA_TOKENIZER_REPO} tokenizer.json found in local HF cache ({})",
                errors.join("; ")
            ),
        })
    }

    /// Returns the tokenizer JSON digest.
    #[must_use]
    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    /// Encodes raw text using the installed CSM BOS/EOS template.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>, CsmFrontendError> {
        self.tokenizer
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| CsmFrontendError::TokenizerEncode {
                message: error.to_string(),
            })
    }

    /// Encodes one CSM speaker-tagged text segment as `[{speaker}]{text}`.
    pub fn encode_segment_text(
        &self,
        speaker: u32,
        text: &str,
    ) -> Result<Vec<u32>, CsmFrontendError> {
        self.encode_raw(&csm_format_segment_text(speaker, text))
    }
}

fn install_csm_llama_template(tokenizer: &mut Tokenizer) -> Result<(), CsmFrontendError> {
    let bos_id = tokenizer.token_to_id(CSM_LLAMA_BOS_TOKEN).ok_or_else(|| {
        CsmFrontendError::TokenizerLoad {
            message: format!("missing BOS token {CSM_LLAMA_BOS_TOKEN}"),
        }
    })?;
    let eos_id = tokenizer.token_to_id(CSM_LLAMA_EOS_TOKEN).ok_or_else(|| {
        CsmFrontendError::TokenizerLoad {
            message: format!("missing EOS token {CSM_LLAMA_EOS_TOKEN}"),
        }
    })?;
    if bos_id != CSM_LLAMA_BOS_TOKEN_ID || eos_id != CSM_LLAMA_EOS_TOKEN_ID {
        return Err(CsmFrontendError::TokenizerLoad {
            message: format!(
                "unexpected BOS/EOS ids: bos={bos_id}, eos={eos_id}; expected {CSM_LLAMA_BOS_TOKEN_ID}/{CSM_LLAMA_EOS_TOKEN_ID}"
            ),
        });
    }

    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single(format!(
                "{CSM_LLAMA_BOS_TOKEN}:0 $A:0 {CSM_LLAMA_EOS_TOKEN}:0"
            ))
            .map_err(|error| CsmFrontendError::TokenizerLoad {
                message: error.to_string(),
            })?
            .try_pair(format!(
                "{CSM_LLAMA_BOS_TOKEN}:0 $A:0 {CSM_LLAMA_EOS_TOKEN}:0 {CSM_LLAMA_BOS_TOKEN}:1 $B:1 {CSM_LLAMA_EOS_TOKEN}:1"
            ))
            .map_err(|error| CsmFrontendError::TokenizerLoad {
                message: error.to_string(),
            })?
            .special_tokens(vec![
                (CSM_LLAMA_BOS_TOKEN, CSM_LLAMA_BOS_TOKEN_ID),
                (CSM_LLAMA_EOS_TOKEN, CSM_LLAMA_EOS_TOKEN_ID),
            ])
            .build()
            .map_err(|error| CsmFrontendError::TokenizerLoad {
                message: error.to_string(),
            })?,
    ));
    Ok(())
}

/// Returns local HF cache tokenizer candidates for Llama 3.2 1B.
#[must_use]
pub fn csm_default_llama_tokenizer_json_candidates() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(hf_home) = env::var("HF_HOME") {
        roots.push(PathBuf::from(hf_home).join("hub"));
    }
    if let Ok(home) = env::var("HOME") {
        roots.push(
            PathBuf::from(home)
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }

    let mut candidates = Vec::new();
    for root in roots {
        let snapshots = root
            .join("models--meta-llama--Llama-3.2-1B")
            .join("snapshots");
        let Ok(entries) = fs::read_dir(snapshots) else {
            continue;
        };
        for entry in entries.flatten() {
            let candidate = entry.path().join("tokenizer.json");
            if candidate.is_file() {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort();
    candidates.dedup();
    candidates
}

/// Formats one CSM text segment exactly as the reference generator does.
#[must_use]
pub fn csm_format_segment_text(speaker: u32, text: &str) -> String {
    format!("[{speaker}]{text}")
}

/// One 33-lane CSM frame.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsmTokenFrame {
    /// Token ids for each lane.
    pub tokens: Vec<u32>,
    /// Active-lane mask.
    pub mask: Vec<bool>,
}

impl CsmTokenFrame {
    /// Creates an empty frame.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            tokens: vec![0; CSM_FRAME_LANES],
            mask: vec![false; CSM_FRAME_LANES],
        }
    }

    /// Creates a text-lane frame.
    #[must_use]
    pub fn text(token_id: u32) -> Self {
        let mut frame = Self::empty();
        frame.tokens[CSM_TEXT_LANE_INDEX] = token_id;
        frame.mask[CSM_TEXT_LANE_INDEX] = true;
        frame
    }

    /// Creates an audio frame from 32 codebook tokens.
    #[must_use]
    pub fn audio(codebook_tokens: [u32; CSM_AUDIO_CODEBOOK_LANES]) -> Self {
        let mut frame = Self::empty();
        frame.tokens[..CSM_AUDIO_CODEBOOK_LANES].copy_from_slice(&codebook_tokens);
        frame.mask[..CSM_AUDIO_CODEBOOK_LANES].fill(true);
        frame
    }
}

/// Contiguous CSM frame block.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsmFrameBlock {
    /// Ordered frames.
    pub frames: Vec<CsmTokenFrame>,
}

impl CsmFrameBlock {
    /// Creates a frame block.
    #[must_use]
    pub fn new(frames: Vec<CsmTokenFrame>) -> Self {
        Self { frames }
    }

    /// Returns frame count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns whether the block is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns `[frames, lanes]`.
    #[must_use]
    pub fn shape(&self) -> [usize; 2] {
        [self.frames.len(), CSM_FRAME_LANES]
    }

    /// Appends another block.
    pub fn extend(&mut self, other: Self) {
        self.frames.extend(other.frames);
    }
}

/// Builds CSM text frames from token ids.
#[must_use]
pub fn csm_text_frame_block(token_ids: &[u32]) -> CsmFrameBlock {
    CsmFrameBlock::new(token_ids.iter().copied().map(CsmTokenFrame::text).collect())
}

/// Builds CSM audio frames and appends the reference all-zero codebook EOS frame.
pub fn csm_audio_frame_block(
    frames: &[[u32; CSM_AUDIO_CODEBOOK_LANES]],
) -> Result<CsmFrameBlock, CsmFrontendError> {
    let mut output = Vec::with_capacity(frames.len() + 1);
    for frame in frames {
        if let Some(token) = frame
            .iter()
            .find(|token| **token > CSM_CODEBOOK_PAD_TOKEN_ID)
        {
            return Err(CsmFrontendError::AudioCodebooks {
                message: format!("audio token {token} exceeds CSM codebook vocab"),
            });
        }
        output.push(CsmTokenFrame::audio(*frame));
    }
    output.push(CsmTokenFrame::audio(
        [CSM_CODEBOOK_EOS_TOKEN_ID; CSM_AUDIO_CODEBOOK_LANES],
    ));
    Ok(CsmFrameBlock::new(output))
}

/// One prepared context or generation segment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsmPromptSegment {
    /// CSM speaker id.
    pub speaker: u32,
    /// Original text.
    pub text: String,
    /// Llama token ids for `[{speaker}]{text}`.
    pub text_token_ids: Vec<u32>,
    /// Optional 32-codebook Mimi frames. Context segments have this; target
    /// generation segments do not.
    pub audio_codebook_frames: Option<Vec<[u32; CSM_AUDIO_CODEBOOK_LANES]>>,
}

impl CsmPromptSegment {
    /// Creates a text-only segment from already encoded token ids.
    #[must_use]
    pub fn text_only(speaker: u32, text: impl Into<String>, text_token_ids: Vec<u32>) -> Self {
        Self {
            speaker,
            text: text.into(),
            text_token_ids,
            audio_codebook_frames: None,
        }
    }

    /// Creates a context segment with audio codebooks.
    #[must_use]
    pub fn with_audio_codebooks(
        speaker: u32,
        text: impl Into<String>,
        text_token_ids: Vec<u32>,
        audio_codebook_frames: Vec<[u32; CSM_AUDIO_CODEBOOK_LANES]>,
    ) -> Self {
        Self {
            speaker,
            text: text.into(),
            text_token_ids,
            audio_codebook_frames: Some(audio_codebook_frames),
        }
    }

    /// Encodes a text-only segment with a Rust CSM tokenizer.
    pub fn encode_text_only(
        tokenizer: &CsmLlamaTextTokenizer,
        speaker: u32,
        text: impl Into<String>,
    ) -> Result<Self, CsmFrontendError> {
        let text = text.into();
        let text_token_ids = tokenizer.encode_segment_text(speaker, &text)?;
        Ok(Self::text_only(speaker, text, text_token_ids))
    }

    /// Builds the segment's text and optional audio frame block.
    pub fn frame_block(&self) -> Result<CsmFrameBlock, CsmFrontendError> {
        let mut block = csm_text_frame_block(&self.text_token_ids);
        if let Some(audio_codebook_frames) = &self.audio_codebook_frames {
            block.extend(csm_audio_frame_block(audio_codebook_frames)?);
        }
        Ok(block)
    }
}

/// Prompt overflow policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CsmContextWindowPolicy {
    /// Refuse prompts that exceed the window.
    Reject,
    /// Drop oldest context segments until the target text fits or refusal is unavoidable.
    DropOldestSegments,
}

/// Prompt window settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsmGenerationWindow {
    /// Max sequence length.
    pub max_seq_len: usize,
    /// Requested generation duration.
    pub max_audio_length_ms: u64,
    /// Frame duration.
    pub generation_frame_ms: u64,
}

impl CsmGenerationWindow {
    /// Creates the standard CSM generation window.
    #[must_use]
    pub const fn new(max_audio_length_ms: u64) -> Self {
        Self {
            max_seq_len: CSM_MAX_SEQ_LEN,
            max_audio_length_ms,
            generation_frame_ms: CSM_GENERATION_FRAME_MS,
        }
    }

    /// Returns the generated frame budget.
    pub fn max_generation_len(self) -> Result<usize, CsmFrontendError> {
        if self.generation_frame_ms == 0 {
            return Err(CsmFrontendError::GenerationWindow {
                message: "generation_frame_ms must be positive".to_string(),
            });
        }
        let frames = self.max_audio_length_ms / self.generation_frame_ms;
        if frames == 0 {
            return Err(CsmFrontendError::GenerationWindow {
                message: "max_audio_length_ms is below one CSM frame".to_string(),
            });
        }
        usize::try_from(frames).map_err(|error| CsmFrontendError::GenerationWindow {
            message: error.to_string(),
        })
    }

    /// Returns the prompt context limit.
    pub fn max_context_len(self) -> Result<usize, CsmFrontendError> {
        let max_generation_len = self.max_generation_len()?;
        if max_generation_len >= self.max_seq_len {
            return Err(CsmFrontendError::GenerationWindow {
                message: format!(
                    "max_generation_len={max_generation_len} leaves no prompt context"
                ),
            });
        }
        Ok(self.max_seq_len - max_generation_len)
    }
}

/// Prompt-frame plan after optional segment-boundary truncation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsmPromptFramePlan {
    /// Final frames sent to the model prompt.
    pub frames: CsmFrameBlock,
    /// Number of original context segments retained.
    pub retained_context_segments: usize,
    /// Number of original context segments dropped.
    pub dropped_context_segments: usize,
    /// Generated frame budget.
    pub max_generation_len: usize,
    /// Prompt context limit.
    pub max_context_len: usize,
}

/// Builds a CSM prompt from context and target text.
pub fn csm_build_prompt_frame_plan(
    context: &[CsmPromptSegment],
    target_text: &CsmPromptSegment,
    window: CsmGenerationWindow,
    policy: CsmContextWindowPolicy,
) -> Result<CsmPromptFramePlan, CsmFrontendError> {
    let max_generation_len = window.max_generation_len()?;
    let max_context_len = window.max_context_len()?;
    let mut first_context_index = 0;

    loop {
        let mut frames = CsmFrameBlock::default();
        for segment in &context[first_context_index..] {
            frames.extend(segment.frame_block()?);
        }
        frames.extend(csm_text_frame_block(&target_text.text_token_ids));

        if frames.len() < max_context_len {
            return Ok(CsmPromptFramePlan {
                frames,
                retained_context_segments: context.len() - first_context_index,
                dropped_context_segments: first_context_index,
                max_generation_len,
                max_context_len,
            });
        }

        if policy == CsmContextWindowPolicy::Reject || first_context_index >= context.len() {
            return Err(CsmFrontendError::PromptContextOverflow {
                prompt_frames: frames.len(),
                max_context_len,
                max_generation_len,
            });
        }
        first_context_index += 1;
    }
}

/// Computes a `sha256:<hex>` digest.
#[must_use]
pub fn sha256_digest(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csm_python_parity_fixture;

    const CONFIG_JSON: &str = r#"{
      "architectures": ["CsmForConditionalGeneration"],
      "audio_eos_token_id": 128003,
      "audio_token_id": 128002,
      "bos_token_id": 128000,
      "codebook_eos_token_id": 0,
      "codebook_pad_token_id": 2050,
      "codec_config": {
        "audio_channels": 1,
        "codebook_size": 2048,
        "frame_rate": 12.5,
        "model_type": "mimi",
        "num_quantizers": 32,
        "sampling_rate": 24000
      },
      "depth_decoder_config": {
        "max_position_embeddings": 33,
        "model_type": "csm_depth_decoder_model",
        "num_codebooks": 32,
        "vocab_size": 2051
      },
      "max_position_embeddings": 2048,
      "model_type": "csm",
      "num_codebooks": 32,
      "pad_token_id": 128002,
      "text_vocab_size": 128256,
      "vocab_size": 2051,
      "audio_num_codebooks": 32,
      "audio_vocab_size": 2051
    }"#;

    #[test]
    fn csm_descriptor_from_fixture_exports_artifacts_and_voice_profiles() {
        let fixture = csm_python_parity_fixture().expect("fixture should parse");
        let descriptor =
            CsmModelArtifactDescriptor::from_fixture(&fixture).expect("descriptor should build");

        assert_eq!(descriptor.model_id, CSM_MODEL_ID);
        assert_eq!(
            descriptor.frame_contract.text_lane_index,
            CSM_TEXT_LANE_INDEX
        );
        assert_eq!(descriptor.voice_profiles.len(), 2);
        assert!(
            descriptor
                .voice_profiles
                .iter()
                .any(|profile| profile.profile_id == "conversational_a"
                    && profile.mimi_tokens_sha256.is_some())
        );
    }

    #[test]
    fn csm_config_parser_validates_frame_contract() {
        let config = CsmModelConfig::from_json_str(CONFIG_JSON).expect("config should parse");

        assert_eq!(config.model_type, "csm");
        assert_eq!(config.max_position_embeddings, CSM_MAX_SEQ_LEN);
        assert_eq!(config.codec_config.sampling_rate, CSM_SAMPLE_RATE_HZ);
    }

    #[test]
    fn csm_speaker_tag_format_matches_reference() {
        assert_eq!(
            csm_format_segment_text(0, "Hey how are you doing?"),
            "[0]Hey how are you doing?"
        );
        assert_eq!(csm_format_segment_text(1, ""), "[1]");
    }

    #[test]
    fn csm_text_frame_block_matches_fixture_example() {
        let fixture = csm_python_parity_fixture().expect("fixture should parse");
        let example = &fixture.tokenizer_examples[0];

        let block = csm_text_frame_block(&example.encoded_token_ids);

        assert_eq!(block.shape(), example.frame_shape);
        assert_eq!(block.frames[0].tokens, example.first_frame_tokens);
        assert_eq!(block.frames[0].mask, example.first_frame_mask);
    }

    #[test]
    fn csm_empty_text_segment_builds_text_frames() {
        let segment = CsmPromptSegment::text_only(
            1,
            "",
            vec![CSM_LLAMA_BOS_TOKEN_ID, 58, 16, 60, CSM_LLAMA_EOS_TOKEN_ID],
        );

        let block = segment.frame_block().expect("empty text frame block");

        assert_eq!(segment.text, "");
        assert_eq!(block.shape(), [5, CSM_FRAME_LANES]);
        assert_eq!(
            block.frames[0].tokens[CSM_TEXT_LANE_INDEX],
            CSM_LLAMA_BOS_TOKEN_ID
        );
        assert_eq!(
            block.frames[4].tokens[CSM_TEXT_LANE_INDEX],
            CSM_LLAMA_EOS_TOKEN_ID
        );
    }

    #[test]
    fn csm_audio_frame_block_adds_codebook_eos_frame() {
        let frames = vec![[7; CSM_AUDIO_CODEBOOK_LANES], [8; CSM_AUDIO_CODEBOOK_LANES]];

        let block = csm_audio_frame_block(&frames).expect("audio block");

        assert_eq!(block.shape(), [3, CSM_FRAME_LANES]);
        assert_eq!(block.frames[0].tokens[0], 7);
        assert_eq!(
            &block.frames[2].tokens[..CSM_AUDIO_CODEBOOK_LANES],
            &[CSM_CODEBOOK_EOS_TOKEN_ID; CSM_AUDIO_CODEBOOK_LANES]
        );
        assert!(
            block.frames[2].mask[..CSM_AUDIO_CODEBOOK_LANES]
                .iter()
                .all(|value| *value)
        );
        assert!(!block.frames[2].mask[CSM_TEXT_LANE_INDEX]);
    }

    #[test]
    fn csm_prompt_plan_supports_multi_segment_context() {
        let segment_a = CsmPromptSegment::with_audio_codebooks(
            0,
            "a",
            vec![CSM_LLAMA_BOS_TOKEN_ID, 11, CSM_LLAMA_EOS_TOKEN_ID],
            vec![[1; CSM_AUDIO_CODEBOOK_LANES]; 2],
        );
        let segment_b = CsmPromptSegment::with_audio_codebooks(
            1,
            "b",
            vec![CSM_LLAMA_BOS_TOKEN_ID, 12, CSM_LLAMA_EOS_TOKEN_ID],
            vec![[2; CSM_AUDIO_CODEBOOK_LANES]; 3],
        );
        let target = CsmPromptSegment::text_only(
            0,
            "target",
            vec![CSM_LLAMA_BOS_TOKEN_ID, 13, CSM_LLAMA_EOS_TOKEN_ID],
        );

        let plan = csm_build_prompt_frame_plan(
            &[segment_a, segment_b],
            &target,
            CsmGenerationWindow::new(1_000),
            CsmContextWindowPolicy::Reject,
        )
        .expect("prompt plan");

        assert_eq!(plan.retained_context_segments, 2);
        assert_eq!(plan.dropped_context_segments, 0);
        assert_eq!(plan.frames.len(), 3 + 3 + 3 + 4 + 3);
    }

    #[test]
    fn csm_prompt_overflow_refuses_like_reference() {
        let long_context = CsmPromptSegment::with_audio_codebooks(
            0,
            "long",
            vec![CSM_LLAMA_BOS_TOKEN_ID],
            vec![[1; CSM_AUDIO_CODEBOOK_LANES]; 2042],
        );
        let target = CsmPromptSegment::text_only(
            0,
            "target",
            vec![CSM_LLAMA_BOS_TOKEN_ID, 13, CSM_LLAMA_EOS_TOKEN_ID],
        );

        let result = csm_build_prompt_frame_plan(
            &[long_context],
            &target,
            CsmGenerationWindow::new(160),
            CsmContextWindowPolicy::Reject,
        );

        assert!(matches!(
            result,
            Err(CsmFrontendError::PromptContextOverflow {
                max_context_len: 2046,
                max_generation_len: 2,
                ..
            })
        ));
    }

    #[test]
    fn csm_prompt_policy_drops_oldest_segments_at_boundaries() {
        let too_large = CsmPromptSegment::with_audio_codebooks(
            0,
            "too large",
            vec![CSM_LLAMA_BOS_TOKEN_ID],
            vec![[1; CSM_AUDIO_CODEBOOK_LANES]; 2042],
        );
        let retained = CsmPromptSegment::with_audio_codebooks(
            1,
            "retained",
            vec![CSM_LLAMA_BOS_TOKEN_ID, 12, CSM_LLAMA_EOS_TOKEN_ID],
            vec![[2; CSM_AUDIO_CODEBOOK_LANES]; 2],
        );
        let target = CsmPromptSegment::text_only(
            1,
            "target",
            vec![CSM_LLAMA_BOS_TOKEN_ID, 13, CSM_LLAMA_EOS_TOKEN_ID],
        );

        let plan = csm_build_prompt_frame_plan(
            &[too_large, retained],
            &target,
            CsmGenerationWindow::new(160),
            CsmContextWindowPolicy::DropOldestSegments,
        )
        .expect("prompt should fit after dropping oldest segment");

        assert_eq!(plan.dropped_context_segments, 1);
        assert_eq!(plan.retained_context_segments, 1);
    }

    #[test]
    fn csm_llama_tokenizer_matches_python_fixture_when_cache_is_present() {
        let fixture = csm_python_parity_fixture().expect("fixture should parse");
        let tokenizer = match CsmLlamaTextTokenizer::from_default_hf_cache(Some(
            &fixture.model.llama_tokenizer_digest,
        )) {
            Ok(tokenizer) => tokenizer,
            Err(error) => {
                eprintln!("skipping local tokenizer parity test: {error}");
                return;
            }
        };

        for example in &fixture.tokenizer_examples {
            let actual = tokenizer
                .encode_segment_text(example.speaker, &example.text)
                .expect("encode segment");
            assert_eq!(actual, example.encoded_token_ids);
        }
    }
}
