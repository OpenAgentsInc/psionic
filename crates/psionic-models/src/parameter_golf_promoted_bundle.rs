use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Stable schema version for the promoted PGOLF tokenizer asset.
pub const PARAMETER_GOLF_PROMOTED_TOKENIZER_ASSET_SCHEMA_VERSION: &str =
    "psionic.parameter_golf_promoted_tokenizer_asset.v1";
/// Stable schema version for the promoted PGOLF generation config.
pub const PARAMETER_GOLF_PROMOTED_GENERATION_CONFIG_SCHEMA_VERSION: &str =
    "psionic.parameter_golf_promoted_generation_config.v1";
/// Stable schema version for the promoted PGOLF bundle manifest.
pub const PARAMETER_GOLF_PROMOTED_BUNDLE_MANIFEST_SCHEMA_VERSION: &str =
    "psionic.parameter_golf_promoted_bundle_manifest.v1";

/// Runtime-loadable tokenizer asset format admitted by the promoted PGOLF bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfPromotedTokenizerAssetFormat {
    /// One JSON piece table carrying the full runtime vocabulary.
    SentencePiecePieceTableJson,
}

/// Tokenizer family admitted by the promoted PGOLF bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfPromotedTokenizerFamily {
    /// SentencePiece-style tokenization.
    SentencePiece,
}

/// Token role admitted by the promoted PGOLF runtime tokenizer asset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfPromotedTokenizerTokenKind {
    Normal,
    Byte,
    Control,
    Unknown,
    Unused,
}

/// One ordered tokenizer piece inside the promoted PGOLF runtime asset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfPromotedTokenizerToken {
    /// Stable token id.
    pub token_id: u32,
    /// Stable token piece string.
    pub piece: String,
    /// Runtime token role.
    pub kind: ParameterGolfPromotedTokenizerTokenKind,
}

/// Runtime-loadable tokenizer asset emitted beside one promoted PGOLF model bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfPromotedTokenizerAsset {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable promoted profile id.
    pub profile_id: String,
    /// Stable tokenizer identifier.
    pub tokenizer_id: String,
    /// Stable tokenizer revision or version.
    pub tokenizer_version: String,
    /// Tokenizer family.
    pub family: ParameterGolfPromotedTokenizerFamily,
    /// Runtime asset format.
    pub asset_format: ParameterGolfPromotedTokenizerAssetFormat,
    /// Vocabulary size for the emitted runtime tokenizer.
    pub vocab_size: u32,
    /// Whether BOS should be injected by default.
    pub add_bos: bool,
    /// Whether EOS should be injected by default.
    pub add_eos: bool,
    /// Optional BOS token id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token_id: Option<u32>,
    /// Ordered EOS token ids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub eos_token_ids: Vec<u32>,
    /// Optional PAD token id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_token_id: Option<u32>,
    /// Optional unknown-token id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unknown_token_id: Option<u32>,
    /// Full runtime vocabulary in stable token-id order.
    pub pieces: Vec<ParameterGolfPromotedTokenizerToken>,
    /// Stable digest over the tokenizer contract itself.
    pub tokenizer_digest: String,
    /// Stable digest over the full asset payload.
    pub asset_digest: String,
    /// Human-readable detail for operators and audits.
    pub detail: String,
}

impl ParameterGolfPromotedTokenizerAsset {
    /// Returns the stable digest over the logical tokenizer contract.
    #[must_use]
    pub fn tokenizer_contract_digest(&self) -> String {
        stable_digest(
            b"psionic_parameter_golf_promoted_tokenizer_contract|",
            &(
                self.profile_id.as_str(),
                self.tokenizer_id.as_str(),
                self.tokenizer_version.as_str(),
                self.family,
                self.vocab_size,
                self.add_bos,
                self.add_eos,
                self.bos_token_id,
                self.eos_token_ids.as_slice(),
                self.pad_token_id,
                self.unknown_token_id,
                self.pieces.as_slice(),
            ),
        )
    }

    /// Returns the stable digest over the full asset payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.asset_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_promoted_tokenizer_asset|",
            &canonical,
        )
    }

    /// Validates the emitted tokenizer asset.
    pub fn validate(&self) -> Result<(), ParameterGolfPromotedBundleError> {
        require_exact_schema(
            self.schema_version.as_str(),
            PARAMETER_GOLF_PROMOTED_TOKENIZER_ASSET_SCHEMA_VERSION,
            "tokenizer_asset.schema_version",
        )?;
        require_nonempty(self.profile_id.as_str(), "tokenizer_asset.profile_id")?;
        require_nonempty(self.tokenizer_id.as_str(), "tokenizer_asset.tokenizer_id")?;
        require_nonempty(
            self.tokenizer_version.as_str(),
            "tokenizer_asset.tokenizer_version",
        )?;
        require_nonempty(
            self.tokenizer_digest.as_str(),
            "tokenizer_asset.tokenizer_digest",
        )?;
        require_nonempty(self.asset_digest.as_str(), "tokenizer_asset.asset_digest")?;
        require_nonempty(self.detail.as_str(), "tokenizer_asset.detail")?;
        if self.vocab_size == 0 {
            return Err(ParameterGolfPromotedBundleError::InvalidValue {
                field: String::from("tokenizer_asset.vocab_size"),
                detail: String::from("vocab_size must be positive"),
            });
        }
        if self.pieces.len() != self.vocab_size as usize {
            return Err(ParameterGolfPromotedBundleError::InvalidValue {
                field: String::from("tokenizer_asset.pieces"),
                detail: format!(
                    "piece count {} must exactly match vocab size {}",
                    self.pieces.len(),
                    self.vocab_size
                ),
            });
        }
        for (expected_id, piece) in self.pieces.iter().enumerate() {
            if piece.token_id != expected_id as u32 {
                return Err(ParameterGolfPromotedBundleError::InvalidValue {
                    field: String::from("tokenizer_asset.pieces"),
                    detail: format!(
                        "piece at position {} carried token id {}",
                        expected_id, piece.token_id
                    ),
                });
            }
        }
        for (field, token_id) in [
            ("tokenizer_asset.bos_token_id", self.bos_token_id),
            ("tokenizer_asset.pad_token_id", self.pad_token_id),
            ("tokenizer_asset.unknown_token_id", self.unknown_token_id),
        ] {
            if let Some(token_id) = token_id {
                if token_id >= self.vocab_size {
                    return Err(ParameterGolfPromotedBundleError::InvalidValue {
                        field: String::from(field),
                        detail: format!(
                            "token id {} exceeds vocab size {}",
                            token_id, self.vocab_size
                        ),
                    });
                }
            }
        }
        for token_id in &self.eos_token_ids {
            if *token_id >= self.vocab_size {
                return Err(ParameterGolfPromotedBundleError::InvalidValue {
                    field: String::from("tokenizer_asset.eos_token_ids"),
                    detail: format!(
                        "token id {} exceeds vocab size {}",
                        token_id, self.vocab_size
                    ),
                });
            }
        }
        if self.tokenizer_digest != self.tokenizer_contract_digest() {
            return Err(ParameterGolfPromotedBundleError::DigestMismatch {
                field: String::from("tokenizer_asset.tokenizer_digest"),
                expected: self.tokenizer_contract_digest(),
                actual: self.tokenizer_digest.clone(),
            });
        }
        if self.asset_digest != self.stable_digest() {
            return Err(ParameterGolfPromotedBundleError::DigestMismatch {
                field: String::from("tokenizer_asset.asset_digest"),
                expected: self.stable_digest(),
                actual: self.asset_digest.clone(),
            });
        }
        Ok(())
    }
}

/// Default generation config emitted with one promoted PGOLF model bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfPromotedGenerationConfig {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable promoted profile id.
    pub profile_id: String,
    /// Stable prompt-format identifier.
    pub prompt_format: String,
    /// Maximum supported context length.
    pub max_context: usize,
    /// Default max new-token budget.
    pub default_max_new_tokens: usize,
    /// Default sampling mode label.
    pub default_sampling_mode: String,
    /// Default temperature for seeded sampling paths.
    pub default_temperature: f32,
    /// Optional top-k cap for seeded sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_top_k: Option<usize>,
    /// Whether the runtime should stop on EOS when one is configured.
    pub stop_on_eos: bool,
    /// Stable digest over the config payload.
    pub config_digest: String,
    /// Human-readable detail for operators and audits.
    pub detail: String,
}

impl ParameterGolfPromotedGenerationConfig {
    /// Returns the stable digest over the config payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.config_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_promoted_generation_config|",
            &canonical,
        )
    }

    /// Validates the emitted generation config.
    pub fn validate(&self) -> Result<(), ParameterGolfPromotedBundleError> {
        require_exact_schema(
            self.schema_version.as_str(),
            PARAMETER_GOLF_PROMOTED_GENERATION_CONFIG_SCHEMA_VERSION,
            "generation_config.schema_version",
        )?;
        require_nonempty(self.profile_id.as_str(), "generation_config.profile_id")?;
        require_nonempty(
            self.prompt_format.as_str(),
            "generation_config.prompt_format",
        )?;
        require_nonempty(
            self.default_sampling_mode.as_str(),
            "generation_config.default_sampling_mode",
        )?;
        require_nonempty(
            self.config_digest.as_str(),
            "generation_config.config_digest",
        )?;
        require_nonempty(self.detail.as_str(), "generation_config.detail")?;
        if self.max_context == 0 {
            return Err(ParameterGolfPromotedBundleError::InvalidValue {
                field: String::from("generation_config.max_context"),
                detail: String::from("max_context must be positive"),
            });
        }
        if self.default_max_new_tokens == 0 {
            return Err(ParameterGolfPromotedBundleError::InvalidValue {
                field: String::from("generation_config.default_max_new_tokens"),
                detail: String::from("default_max_new_tokens must be positive"),
            });
        }
        if !self.default_temperature.is_finite() || self.default_temperature < 0.0 {
            return Err(ParameterGolfPromotedBundleError::InvalidValue {
                field: String::from("generation_config.default_temperature"),
                detail: String::from("default_temperature must be finite and non-negative"),
            });
        }
        if let Some(top_k) = self.default_top_k {
            if top_k == 0 {
                return Err(ParameterGolfPromotedBundleError::InvalidValue {
                    field: String::from("generation_config.default_top_k"),
                    detail: String::from("default_top_k must be positive when present"),
                });
            }
        }
        if self.config_digest != self.stable_digest() {
            return Err(ParameterGolfPromotedBundleError::DigestMismatch {
                field: String::from("generation_config.config_digest"),
                expected: self.stable_digest(),
                actual: self.config_digest.clone(),
            });
        }
        Ok(())
    }
}

/// One file artifact referenced by the promoted PGOLF bundle manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfPromotedBundleArtifactRef {
    /// Relative file path inside the bundle directory.
    pub relative_path: String,
    /// Raw SHA-256 over the referenced file bytes.
    pub sha256: String,
}

impl ParameterGolfPromotedBundleArtifactRef {
    fn validate(&self, field: &str) -> Result<(), ParameterGolfPromotedBundleError> {
        require_relative_path(self.relative_path.as_str(), field)?;
        require_nonempty(self.sha256.as_str(), &format!("{field}.sha256"))?;
        Ok(())
    }
}

/// File inventory for the promoted PGOLF model bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfPromotedBundleArtifacts {
    pub descriptor: ParameterGolfPromotedBundleArtifactRef,
    pub model: ParameterGolfPromotedBundleArtifactRef,
    pub tokenizer_asset: ParameterGolfPromotedBundleArtifactRef,
    pub generation_config: ParameterGolfPromotedBundleArtifactRef,
    pub profile_contract: ParameterGolfPromotedBundleArtifactRef,
    pub training_config: ParameterGolfPromotedBundleArtifactRef,
    pub summary: ParameterGolfPromotedBundleArtifactRef,
    pub checkpoint_manifest: ParameterGolfPromotedBundleArtifactRef,
    pub checkpoint_surface_report: ParameterGolfPromotedBundleArtifactRef,
    pub resume_proof: ParameterGolfPromotedBundleArtifactRef,
}

impl ParameterGolfPromotedBundleArtifacts {
    fn validate(&self) -> Result<(), ParameterGolfPromotedBundleError> {
        self.descriptor
            .validate("bundle_manifest.artifacts.descriptor")?;
        self.model.validate("bundle_manifest.artifacts.model")?;
        self.tokenizer_asset
            .validate("bundle_manifest.artifacts.tokenizer_asset")?;
        self.generation_config
            .validate("bundle_manifest.artifacts.generation_config")?;
        self.profile_contract
            .validate("bundle_manifest.artifacts.profile_contract")?;
        self.training_config
            .validate("bundle_manifest.artifacts.training_config")?;
        self.summary.validate("bundle_manifest.artifacts.summary")?;
        self.checkpoint_manifest
            .validate("bundle_manifest.artifacts.checkpoint_manifest")?;
        self.checkpoint_surface_report
            .validate("bundle_manifest.artifacts.checkpoint_surface_report")?;
        self.resume_proof
            .validate("bundle_manifest.artifacts.resume_proof")?;
        Ok(())
    }
}

/// Training lineage and provenance carried by the promoted PGOLF bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfPromotedBundleLineage {
    /// Stable promoted run id.
    pub run_id: String,
    /// Stable checkpoint family.
    pub checkpoint_family: String,
    /// Stable final checkpoint ref.
    pub final_checkpoint_ref: String,
    /// Stable final checkpoint manifest digest.
    pub final_checkpoint_manifest_digest: String,
    /// Stable emitted checkpoint artifact digest.
    pub checkpoint_artifact_digest: String,
    /// Stable emitted descriptor digest.
    pub descriptor_digest: String,
    /// Stable training dataset digest.
    pub training_dataset_digest: String,
    /// Stable validation dataset digest.
    pub validation_dataset_digest: String,
    /// Stable promoted profile id.
    pub profile_id: String,
    /// Stable promoted profile kind label.
    pub profile_kind: String,
    /// Human-readable lineage detail.
    pub detail: String,
}

impl ParameterGolfPromotedBundleLineage {
    fn validate(&self) -> Result<(), ParameterGolfPromotedBundleError> {
        require_nonempty(self.run_id.as_str(), "bundle_manifest.lineage.run_id")?;
        require_nonempty(
            self.checkpoint_family.as_str(),
            "bundle_manifest.lineage.checkpoint_family",
        )?;
        require_nonempty(
            self.final_checkpoint_ref.as_str(),
            "bundle_manifest.lineage.final_checkpoint_ref",
        )?;
        require_nonempty(
            self.final_checkpoint_manifest_digest.as_str(),
            "bundle_manifest.lineage.final_checkpoint_manifest_digest",
        )?;
        require_nonempty(
            self.checkpoint_artifact_digest.as_str(),
            "bundle_manifest.lineage.checkpoint_artifact_digest",
        )?;
        require_nonempty(
            self.descriptor_digest.as_str(),
            "bundle_manifest.lineage.descriptor_digest",
        )?;
        require_nonempty(
            self.training_dataset_digest.as_str(),
            "bundle_manifest.lineage.training_dataset_digest",
        )?;
        require_nonempty(
            self.validation_dataset_digest.as_str(),
            "bundle_manifest.lineage.validation_dataset_digest",
        )?;
        require_nonempty(
            self.profile_id.as_str(),
            "bundle_manifest.lineage.profile_id",
        )?;
        require_nonempty(
            self.profile_kind.as_str(),
            "bundle_manifest.lineage.profile_kind",
        )?;
        require_nonempty(self.detail.as_str(), "bundle_manifest.lineage.detail")?;
        Ok(())
    }
}

/// Canonical promoted PGOLF bundle manifest emitted by training and consumed by
/// later runtime or serve loaders.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfPromotedBundleManifest {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable bundle identifier.
    pub bundle_id: String,
    /// Stable promoted family id.
    pub family_id: String,
    /// Stable model family.
    pub model_family: String,
    /// Stable model id.
    pub model_id: String,
    /// Stable model revision.
    pub model_revision: String,
    /// Stable promoted profile id.
    pub profile_id: String,
    /// Stable promoted profile kind label.
    pub profile_kind: String,
    /// File inventory for the bundle.
    pub artifacts: ParameterGolfPromotedBundleArtifacts,
    /// Training lineage and provenance.
    pub lineage: ParameterGolfPromotedBundleLineage,
    /// Human-readable manifest detail.
    pub detail: String,
    /// Stable digest over the bundle manifest payload.
    pub bundle_digest: String,
}

impl ParameterGolfPromotedBundleManifest {
    /// Returns the stable digest over the bundle manifest payload.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.bundle_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_promoted_bundle_manifest|",
            &canonical,
        )
    }

    /// Validates the promoted PGOLF bundle manifest.
    pub fn validate(&self) -> Result<(), ParameterGolfPromotedBundleError> {
        require_exact_schema(
            self.schema_version.as_str(),
            PARAMETER_GOLF_PROMOTED_BUNDLE_MANIFEST_SCHEMA_VERSION,
            "bundle_manifest.schema_version",
        )?;
        require_nonempty(self.bundle_id.as_str(), "bundle_manifest.bundle_id")?;
        require_nonempty(self.family_id.as_str(), "bundle_manifest.family_id")?;
        require_nonempty(self.model_family.as_str(), "bundle_manifest.model_family")?;
        require_nonempty(self.model_id.as_str(), "bundle_manifest.model_id")?;
        require_nonempty(
            self.model_revision.as_str(),
            "bundle_manifest.model_revision",
        )?;
        require_nonempty(self.profile_id.as_str(), "bundle_manifest.profile_id")?;
        require_nonempty(self.profile_kind.as_str(), "bundle_manifest.profile_kind")?;
        self.artifacts.validate()?;
        self.lineage.validate()?;
        require_nonempty(self.detail.as_str(), "bundle_manifest.detail")?;
        require_nonempty(self.bundle_digest.as_str(), "bundle_manifest.bundle_digest")?;
        if self.bundle_digest != self.stable_digest() {
            return Err(ParameterGolfPromotedBundleError::DigestMismatch {
                field: String::from("bundle_manifest.bundle_digest"),
                expected: self.stable_digest(),
                actual: self.bundle_digest.clone(),
            });
        }
        Ok(())
    }
}

/// Validation error for the promoted PGOLF bundle contract.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ParameterGolfPromotedBundleError {
    #[error("promoted PGOLF bundle field `{field}` is missing")]
    MissingField { field: String },
    #[error(
        "promoted PGOLF bundle field `{field}` expected schema `{expected}`, found `{actual}`"
    )]
    SchemaVersionMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    #[error("promoted PGOLF bundle digest mismatch for `{field}`: expected `{expected}`, found `{actual}`")]
    DigestMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    #[error("promoted PGOLF bundle field `{field}` is invalid: {detail}")]
    InvalidValue { field: String, detail: String },
}

fn require_nonempty(value: &str, field: &str) -> Result<(), ParameterGolfPromotedBundleError> {
    if value.trim().is_empty() {
        return Err(ParameterGolfPromotedBundleError::MissingField {
            field: String::from(field),
        });
    }
    Ok(())
}

fn require_exact_schema(
    actual: &str,
    expected: &str,
    field: &str,
) -> Result<(), ParameterGolfPromotedBundleError> {
    require_nonempty(actual, field)?;
    if actual != expected {
        return Err(ParameterGolfPromotedBundleError::SchemaVersionMismatch {
            field: String::from(field),
            expected: String::from(expected),
            actual: String::from(actual),
        });
    }
    Ok(())
}

fn require_relative_path(path: &str, field: &str) -> Result<(), ParameterGolfPromotedBundleError> {
    require_nonempty(path, field)?;
    if path.starts_with('/') || path.split('/').any(|component| component == "..") {
        return Err(ParameterGolfPromotedBundleError::InvalidValue {
            field: String::from(field),
            detail: format!("path `{path}` must stay relative to the bundle directory"),
        });
    }
    Ok(())
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).expect("promoted PGOLF bundle value should serialize"));
    hex::encode(hasher.finalize())
}
