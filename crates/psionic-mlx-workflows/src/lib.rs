//! Bounded MLX-style workflow package above `psionic-data`,
//! `psionic-mlx-recipes`, and `psionic-train`.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use psionic_data::{
    DatasetContractError, DatasetKey, DatasetManifest, DatasetRecordEncoding, DatasetShardManifest,
    DatasetSplitDeclaration, DatasetSplitKind, TokenizerDigest,
};
use psionic_datastream::{DatastreamEncoding, DatastreamManifest, DatastreamSubjectKind};
use psionic_environments::{
    EnvironmentPackageKey, EnvironmentPolicyKind, EnvironmentPolicyReference,
};
use psionic_mlx_recipes::{
    MlxRecipeConfig, MlxRecipeError, MlxRecipeMethod, MlxRecipePlan, MlxRecipeWorkspace,
};
use psionic_train::{
    ModelAdapterDelta, ModelArtifactFormat, ModelInteropStatus, ModelIoArtifactReceipt,
    ModelIoError, PortableModelBundle,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

mod swarm_live_plan;

pub use swarm_live_plan::*;

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str =
    "bounded MLX-style synthetic-data, supervision-helper, adapter-merge, and publish workflows";

/// Supported synthetic dataset family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxSyntheticDatasetKind {
    /// Synthetic supervised fine-tuning records.
    Sft,
    /// Synthetic preference or ranking records.
    Preference,
}

/// One deterministic SFT-style synthetic record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSyntheticSftSample {
    /// Stable sample identifier.
    pub sample_id: String,
    /// Prompt or instruction text.
    pub prompt: String,
    /// Expected assistant response.
    pub response: String,
    /// Optional tags.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

/// One declared split for a synthetic SFT bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSyntheticSftSplit {
    /// Stable split name.
    pub split_name: String,
    /// High-level split role.
    pub kind: DatasetSplitKind,
    /// Split-local records.
    pub samples: Vec<MlxSyntheticSftSample>,
}

/// Full synthetic SFT dataset request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSyntheticSftDatasetSpec {
    /// Stable workflow identifier.
    pub workflow_id: String,
    /// Versioned dataset identity.
    pub dataset: DatasetKey,
    /// Human-readable dataset name.
    pub display_name: String,
    /// Tokenizer contract for the dataset.
    pub tokenizer: TokenizerDigest,
    /// Optional context window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window_tokens: Option<u32>,
    /// Declared splits.
    pub splits: Vec<MlxSyntheticSftSplit>,
}

/// One deterministic preference-style synthetic record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSyntheticPreferenceSample {
    /// Stable sample identifier.
    pub sample_id: String,
    /// Shared prompt or conversation prefix.
    pub prompt: String,
    /// Preferred response.
    pub chosen: String,
    /// Non-preferred response.
    pub rejected: String,
    /// Optional rubric or preference rationale.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

/// One declared split for a synthetic preference bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSyntheticPreferenceSplit {
    /// Stable split name.
    pub split_name: String,
    /// High-level split role.
    pub kind: DatasetSplitKind,
    /// Split-local records.
    pub samples: Vec<MlxSyntheticPreferenceSample>,
}

/// Full synthetic preference dataset request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSyntheticPreferenceDatasetSpec {
    /// Stable workflow identifier.
    pub workflow_id: String,
    /// Versioned dataset identity.
    pub dataset: DatasetKey,
    /// Human-readable dataset name.
    pub display_name: String,
    /// Tokenizer contract for the dataset.
    pub tokenizer: TokenizerDigest,
    /// Optional context window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window_tokens: Option<u32>,
    /// Declared splits.
    pub splits: Vec<MlxSyntheticPreferenceSplit>,
}

/// One materialized split artifact inside a synthetic dataset bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSyntheticSplitArtifact {
    /// Stable split name.
    pub split_name: String,
    /// High-level split role.
    pub kind: DatasetSplitKind,
    /// Datastream manifest for the JSONL payload.
    pub datastream_manifest: DatastreamManifest,
    /// Exact JSONL payload bytes.
    pub jsonl_bytes: Vec<u8>,
    /// Number of records carried by the split.
    pub record_count: usize,
    /// Approximate token count carried by the split.
    pub approx_token_count: u64,
}

/// One machine-readable split summary inside a synthetic dataset report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSyntheticSplitReceipt {
    /// Stable split name.
    pub split_name: String,
    /// High-level split role.
    pub kind: DatasetSplitKind,
    /// Number of records carried by the split.
    pub record_count: usize,
    /// Approximate token count carried by the split.
    pub approx_token_count: u64,
    /// Datastream manifest digest for the split payload.
    pub datastream_manifest_digest: String,
    /// Object digest for the split JSONL payload.
    pub object_digest: String,
}

/// One machine-readable synthetic dataset report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSyntheticDatasetReport {
    /// Stable workflow identifier.
    pub workflow_id: String,
    /// Synthetic dataset family.
    pub dataset_kind: MlxSyntheticDatasetKind,
    /// Versioned dataset identity.
    pub dataset_storage_key: String,
    /// Stable dataset-manifest digest.
    pub dataset_manifest_digest: String,
    /// Ordered split receipts.
    pub splits: Vec<MlxSyntheticSplitReceipt>,
    /// Stable report digest.
    pub report_digest: String,
}

/// Complete synthetic dataset artifact bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxSyntheticDatasetArtifact {
    /// Synthetic dataset family.
    pub dataset_kind: MlxSyntheticDatasetKind,
    /// Canonical dataset manifest.
    pub dataset_manifest: DatasetManifest,
    /// Materialized split artifacts.
    pub split_artifacts: Vec<MlxSyntheticSplitArtifact>,
    /// Machine-readable report.
    pub report: MlxSyntheticDatasetReport,
}

impl MlxSyntheticDatasetArtifact {
    /// Writes the dataset bundle into one local directory.
    pub fn write_to_directory(
        &self,
        root: impl AsRef<Path>,
    ) -> Result<MlxWorkflowDirectoryWrite, MlxWorkflowError> {
        let root = root.as_ref();
        fs::create_dir_all(root).map_err(|error| MlxWorkflowError::Io {
            path: root.display().to_string(),
            message: error.to_string(),
        })?;

        let mut files = Vec::new();
        files.push(write_json_file(
            root.join("dataset_manifest.json"),
            &self.dataset_manifest,
            "dataset manifest export",
        )?);
        files.push(write_json_file(
            root.join("synthetic_dataset_report.json"),
            &self.report,
            "synthetic dataset report export",
        )?);
        for split in &self.split_artifacts {
            files.push(write_bytes_file(
                root.join(format!("{}.jsonl", split.split_name)),
                split.jsonl_bytes.as_slice(),
            )?);
            files.push(write_json_file(
                root.join(format!("{}.datastream_manifest.json", split.split_name)),
                &split.datastream_manifest,
                "synthetic split datastream export",
            )?);
        }

        Ok(MlxWorkflowDirectoryWrite {
            root_path: root.display().to_string(),
            files,
            write_digest: stable_directory_write_digest(root, &self.report.report_digest),
        })
    }
}

/// Supported supervision-helper family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxSupervisionHelperKind {
    /// Judge-model training helper.
    JudgeModel,
    /// Reward-model training helper.
    RewardModel,
}

/// Label surface used by a supervision helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxSupervisionLabelKind {
    /// Binary or pass/fail labels.
    Binary,
    /// Scalar or rubric scores.
    Scalar,
}

/// Machine-readable label schema for a supervision helper.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxSupervisionLabelSchema {
    /// Stable label field.
    pub label_field: String,
    /// Label family.
    pub label_kind: MlxSupervisionLabelKind,
    /// Optional minimum score for scalar labels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_score: Option<f32>,
    /// Optional maximum score for scalar labels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_score: Option<f32>,
}

/// One supervision-helper planning request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxSupervisionHelperConfig {
    /// Stable helper identifier.
    pub helper_id: String,
    /// Helper family.
    pub helper_kind: MlxSupervisionHelperKind,
    /// Dataset storage key consumed by the helper.
    pub dataset_storage_key: String,
    /// Recipe config reused from `psionic-mlx-recipes`.
    pub recipe: MlxRecipeConfig,
    /// Label schema consumed by the helper.
    pub label_schema: MlxSupervisionLabelSchema,
}

/// One machine-readable supervision-helper plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxSupervisionHelperPlan {
    /// Stable helper identifier.
    pub helper_id: String,
    /// Helper family.
    pub helper_kind: MlxSupervisionHelperKind,
    /// Dataset storage key consumed by the helper.
    pub dataset_storage_key: String,
    /// Environment package key used by the helper.
    pub environment: EnvironmentPackageKey,
    /// Reused MLX recipe plan.
    pub recipe_plan: MlxRecipePlan,
    /// Label schema used by the helper.
    pub label_schema: MlxSupervisionLabelSchema,
    /// Typed policy references surfaced into runtime packages.
    pub policy_references: Vec<EnvironmentPolicyReference>,
    /// Stable output artifact family label.
    pub output_artifact_family: String,
    /// Honest bounded notes.
    pub notes: Vec<String>,
    /// Stable plan digest.
    pub plan_digest: String,
}

/// One adapter-merge request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxAdapterMergeConfig {
    /// Stable merge identifier.
    pub merge_id: String,
    /// Stable adapter identifier.
    pub adapter_id: String,
}

/// Machine-readable adapter-merge report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxAdapterMergeReport {
    /// Stable merge identifier.
    pub merge_id: String,
    /// Stable adapter identifier.
    pub adapter_id: String,
    /// Base bundle state-dict digest.
    pub base_state_dict_digest: String,
    /// Target bundle state-dict digest.
    pub target_state_dict_digest: String,
    /// Number of parameter tensors inside the delta.
    pub delta_tensor_count: usize,
    /// Safetensors export receipt for the merged bundle.
    pub merged_artifact_receipt: ModelIoArtifactReceipt,
    /// Honest bounded notes.
    pub notes: Vec<String>,
    /// Stable report digest.
    pub report_digest: String,
}

/// Complete adapter-merge artifact bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxAdapterMergeArtifact {
    /// Derived additive delta.
    pub adapter_delta: ModelAdapterDelta,
    /// Merged bundle after applying the delta.
    pub merged_bundle: PortableModelBundle,
    /// Exported merged safetensors bytes.
    pub merged_safetensors: Vec<u8>,
    /// Machine-readable merge report.
    pub report: MlxAdapterMergeReport,
}

/// Publish target family admitted by the workflow package.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxPublishTarget {
    /// Write a local Hugging Face-style snapshot directory.
    HuggingFaceSnapshot,
    /// Attempt a direct GGUF artifact export.
    GgufArtifact,
}

/// One publish request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxPublishConfig {
    /// Stable publish identifier.
    pub publish_id: String,
    /// Publish target family.
    pub target: MlxPublishTarget,
    /// Local or logical repository identifier.
    pub repo_id: String,
}

/// One file emitted by a workflow write.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxWorkflowFile {
    /// Package-relative file path.
    pub relative_path: String,
    /// Byte length.
    pub byte_length: u64,
    /// Stable payload digest.
    pub sha256: String,
}

/// Publish manifest emitted by a local snapshot write.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MlxPublishManifest {
    /// Stable publish identifier.
    pub publish_id: String,
    /// Publish target family.
    pub target: MlxPublishTarget,
    /// Repository identifier.
    pub repo_id: String,
    /// Export receipt for the primary model artifact.
    pub source_artifact_receipt: ModelIoArtifactReceipt,
    /// Ordered payload files emitted by the snapshot.
    pub files: Vec<MlxWorkflowFile>,
    /// Honest bounded notes.
    pub notes: Vec<String>,
    /// Stable publish-manifest digest.
    pub manifest_digest: String,
}

/// Result of writing one workflow bundle to disk.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxWorkflowDirectoryWrite {
    /// Root directory written by the workflow.
    pub root_path: String,
    /// Ordered payload files.
    pub files: Vec<MlxWorkflowFile>,
    /// Stable write digest.
    pub write_digest: String,
}

/// Errors returned by the workflow package.
#[derive(Debug, Error)]
pub enum MlxWorkflowError {
    /// One workflow field was required but empty.
    #[error("missing required field `{field}`")]
    MissingField {
        /// Stable field label.
        field: &'static str,
    },
    /// One split was declared without any samples.
    #[error("synthetic split `{split_name}` requires at least one sample")]
    EmptySplit {
        /// Stable split name.
        split_name: String,
    },
    /// A sample identifier was repeated inside one split.
    #[error("synthetic split `{split_name}` repeated sample `{sample_id}`")]
    DuplicateSampleId {
        /// Stable split name.
        split_name: String,
        /// Stable sample identifier.
        sample_id: String,
    },
    /// A split name was repeated inside one workflow bundle.
    #[error("synthetic dataset repeated split `{split_name}`")]
    DuplicateSplitName {
        /// Stable split name.
        split_name: String,
    },
    /// One sample was structurally invalid.
    #[error("synthetic split `{split_name}` sample `{sample_id}` is invalid: {message}")]
    InvalidSyntheticSample {
        /// Stable split name.
        split_name: String,
        /// Stable sample identifier.
        sample_id: String,
        /// Human-readable failure reason.
        message: String,
    },
    /// One supervision helper chose an unsupported recipe method.
    #[error("supervision helper `{helper_kind:?}` does not support recipe method `{method:?}`")]
    UnsupportedHelperMethod {
        /// Helper family.
        helper_kind: MlxSupervisionHelperKind,
        /// Selected recipe method.
        method: MlxRecipeMethod,
    },
    /// The adapter merge request used incompatible bundle metadata.
    #[error("adapter merge requires matching bundle metadata: {message}")]
    IncompatibleMergePair {
        /// Human-readable reason.
        message: String,
    },
    /// The requested publish target is not supported by the current portable surface.
    #[error("publish target `{target:?}` is unsupported: {reason}")]
    UnsupportedPublishTarget {
        /// Publish target family.
        target: MlxPublishTarget,
        /// Human-readable refusal reason.
        reason: String,
    },
    /// Local IO failed.
    #[error("io failure at `{path}`: {message}")]
    Io {
        /// Local path that failed.
        path: String,
        /// Human-readable reason.
        message: String,
    },
    /// The first live swarm planner refused the provided contract or inputs.
    #[error("first swarm live planner refused: {message}")]
    FirstSwarmPlan {
        /// Human-readable refusal reason.
        message: String,
    },
    /// JSON serialization or parsing failed.
    #[error("{context}: {message}")]
    Serialization {
        /// Which serialization path failed.
        context: &'static str,
        /// Human-readable reason.
        message: String,
    },
    /// Lower-level dataset validation failed.
    #[error(transparent)]
    Dataset(#[from] DatasetContractError),
    /// Lower-level recipe planning failed.
    #[error(transparent)]
    Recipe(#[from] MlxRecipeError),
    /// Lower-level model-IO failed.
    #[error(transparent)]
    ModelIo(#[from] ModelIoError),
}

/// Workspace for the workflow package.
#[derive(Clone, Debug, Default)]
pub struct MlxWorkflowWorkspace {
    recipe_workspace: MlxRecipeWorkspace,
}

impl MlxWorkflowWorkspace {
    /// Builds a synthetic SFT dataset bundle.
    pub fn build_synthetic_sft_dataset(
        &self,
        spec: &MlxSyntheticSftDatasetSpec,
    ) -> Result<MlxSyntheticDatasetArtifact, MlxWorkflowError> {
        validate_dataset_spec_header(
            spec.workflow_id.as_str(),
            &spec.dataset,
            spec.display_name.as_str(),
            &spec.tokenizer,
        )?;
        let mut split_records = Vec::with_capacity(spec.splits.len());
        let mut split_names = BTreeSet::new();
        for split in &spec.splits {
            if !split_names.insert(split.split_name.clone()) {
                return Err(MlxWorkflowError::DuplicateSplitName {
                    split_name: split.split_name.clone(),
                });
            }
            if split.samples.is_empty() {
                return Err(MlxWorkflowError::EmptySplit {
                    split_name: split.split_name.clone(),
                });
            }
            let mut sample_ids = BTreeSet::new();
            let mut records = Vec::with_capacity(split.samples.len());
            for sample in &split.samples {
                validate_sft_sample(sample, split.split_name.as_str(), &mut sample_ids)?;
                records.push(SyntheticRecordLine {
                    json_value: json!({
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "response": sample.response,
                        "tags": sample.tags,
                    }),
                    approx_token_count: approx_text_tokens(sample.prompt.as_str())
                        .saturating_add(approx_text_tokens(sample.response.as_str())),
                });
            }
            split_records.push(SyntheticSplitRecords {
                split_name: split.split_name.clone(),
                kind: split.kind,
                records,
            });
        }
        build_dataset_artifact(
            spec.workflow_id.as_str(),
            MlxSyntheticDatasetKind::Sft,
            &spec.dataset,
            spec.display_name.as_str(),
            DatasetRecordEncoding::JsonlText,
            spec.tokenizer.clone(),
            spec.context_window_tokens,
            split_records,
        )
    }

    /// Builds a synthetic preference dataset bundle.
    pub fn build_synthetic_preference_dataset(
        &self,
        spec: &MlxSyntheticPreferenceDatasetSpec,
    ) -> Result<MlxSyntheticDatasetArtifact, MlxWorkflowError> {
        validate_dataset_spec_header(
            spec.workflow_id.as_str(),
            &spec.dataset,
            spec.display_name.as_str(),
            &spec.tokenizer,
        )?;
        let mut split_records = Vec::with_capacity(spec.splits.len());
        let mut split_names = BTreeSet::new();
        for split in &spec.splits {
            if !split_names.insert(split.split_name.clone()) {
                return Err(MlxWorkflowError::DuplicateSplitName {
                    split_name: split.split_name.clone(),
                });
            }
            if split.samples.is_empty() {
                return Err(MlxWorkflowError::EmptySplit {
                    split_name: split.split_name.clone(),
                });
            }
            let mut sample_ids = BTreeSet::new();
            let mut records = Vec::with_capacity(split.samples.len());
            for sample in &split.samples {
                validate_preference_sample(sample, split.split_name.as_str(), &mut sample_ids)?;
                records.push(SyntheticRecordLine {
                    json_value: json!({
                        "sample_id": sample.sample_id,
                        "prompt": sample.prompt,
                        "chosen": sample.chosen,
                        "rejected": sample.rejected,
                        "rationale": sample.rationale,
                    }),
                    approx_token_count: approx_text_tokens(sample.prompt.as_str())
                        .saturating_add(approx_text_tokens(sample.chosen.as_str()))
                        .saturating_add(approx_text_tokens(sample.rejected.as_str())),
                });
            }
            split_records.push(SyntheticSplitRecords {
                split_name: split.split_name.clone(),
                kind: split.kind,
                records,
            });
        }
        build_dataset_artifact(
            spec.workflow_id.as_str(),
            MlxSyntheticDatasetKind::Preference,
            &spec.dataset,
            spec.display_name.as_str(),
            DatasetRecordEncoding::PreferenceJsonl,
            spec.tokenizer.clone(),
            spec.context_window_tokens,
            split_records,
        )
    }

    /// Plans one reward-model or judge-model helper over the shared recipe lane.
    pub fn plan_supervision_helper(
        &self,
        config: &MlxSupervisionHelperConfig,
    ) -> Result<MlxSupervisionHelperPlan, MlxWorkflowError> {
        if config.helper_id.trim().is_empty() {
            return Err(MlxWorkflowError::MissingField { field: "helper_id" });
        }
        if config.dataset_storage_key.trim().is_empty() {
            return Err(MlxWorkflowError::MissingField {
                field: "dataset_storage_key",
            });
        }
        if config.label_schema.label_field.trim().is_empty() {
            return Err(MlxWorkflowError::MissingField {
                field: "label_field",
            });
        }
        validate_helper_method(config.helper_kind, config.recipe.method)?;

        let recipe_plan = self.recipe_workspace.plan(&config.recipe)?;
        let policy_references =
            helper_policy_references(config.helper_id.as_str(), config.helper_kind);
        let notes = helper_notes(config.helper_kind, config.recipe.method);
        let output_artifact_family = match config.helper_kind {
            MlxSupervisionHelperKind::JudgeModel => String::from("mlx.judge.snapshot"),
            MlxSupervisionHelperKind::RewardModel => String::from("mlx.reward.snapshot"),
        };
        let plan_digest = stable_supervision_helper_digest(
            config.helper_id.as_str(),
            config.helper_kind,
            config.dataset_storage_key.as_str(),
            &recipe_plan,
            &config.label_schema,
            &policy_references,
            output_artifact_family.as_str(),
        );
        Ok(MlxSupervisionHelperPlan {
            helper_id: config.helper_id.clone(),
            helper_kind: config.helper_kind,
            dataset_storage_key: config.dataset_storage_key.clone(),
            environment: config.recipe.environment.clone(),
            recipe_plan,
            label_schema: config.label_schema.clone(),
            policy_references,
            output_artifact_family,
            notes,
            plan_digest,
        })
    }

    /// Derives, verifies, and exports a merged adapter bundle above portable model IO.
    pub fn merge_adapter(
        &self,
        config: &MlxAdapterMergeConfig,
        base: &PortableModelBundle,
        tuned: &PortableModelBundle,
    ) -> Result<MlxAdapterMergeArtifact, MlxWorkflowError> {
        if config.merge_id.trim().is_empty() {
            return Err(MlxWorkflowError::MissingField { field: "merge_id" });
        }
        if config.adapter_id.trim().is_empty() {
            return Err(MlxWorkflowError::MissingField {
                field: "adapter_id",
            });
        }
        validate_merge_pair(base, tuned)?;

        let adapter_delta = psionic_train::PortableModelStateDict::derive_adapter_delta(
            &base.state_dict,
            &tuned.state_dict,
            config.adapter_id.clone(),
        )?;
        let merged_state = base.state_dict.apply_adapter_delta(&adapter_delta)?;
        let merged_bundle = PortableModelBundle {
            state_dict: merged_state,
            tokenizer: base.tokenizer.clone(),
            chat_template_digest: base.chat_template_digest.clone(),
            preferred_serving_formats: tuned.preferred_serving_formats.clone(),
        };
        let roundtrip = merged_bundle
            .state_dict
            .remove_adapter_delta(&adapter_delta)?;
        if roundtrip.digest != base.state_dict.digest {
            return Err(MlxWorkflowError::IncompatibleMergePair {
                message: String::from("adapter delta roundtrip did not return to the base digest"),
            });
        }
        let (merged_safetensors, merged_artifact_receipt) = merged_bundle.export_safetensors()?;
        let notes = vec![
            String::from(
                "This workflow reuses portable model-IO adapter-delta derivation plus apply/remove verification instead of inventing a second merge path.",
            ),
            String::from(
                "The merged artifact is exported through the existing safetensors manifest surface so downstream publish flows stay bound to Psionic-native model-IO receipts.",
            ),
        ];
        let report = MlxAdapterMergeReport {
            merge_id: config.merge_id.clone(),
            adapter_id: config.adapter_id.clone(),
            base_state_dict_digest: base.state_dict.digest.clone(),
            target_state_dict_digest: tuned.state_dict.digest.clone(),
            delta_tensor_count: adapter_delta.tensors.len(),
            merged_artifact_receipt,
            report_digest: stable_adapter_merge_report_digest(
                config.merge_id.as_str(),
                config.adapter_id.as_str(),
                base.state_dict.digest.as_str(),
                tuned.state_dict.digest.as_str(),
                adapter_delta.tensors.len(),
                &notes,
            ),
            notes,
        };
        Ok(MlxAdapterMergeArtifact {
            adapter_delta,
            merged_bundle,
            merged_safetensors,
            report,
        })
    }

    /// Publishes one portable bundle into a local snapshot directory.
    pub fn publish_bundle(
        &self,
        config: &MlxPublishConfig,
        bundle: &PortableModelBundle,
        output_dir: impl AsRef<Path>,
    ) -> Result<MlxPublishManifest, MlxWorkflowError> {
        if config.publish_id.trim().is_empty() {
            return Err(MlxWorkflowError::MissingField {
                field: "publish_id",
            });
        }
        if config.repo_id.trim().is_empty() {
            return Err(MlxWorkflowError::MissingField { field: "repo_id" });
        }
        let output_dir = output_dir.as_ref();
        match config.target {
            MlxPublishTarget::HuggingFaceSnapshot => {
                publish_hugging_face_snapshot(config, bundle, output_dir)
            }
            MlxPublishTarget::GgufArtifact => Err(MlxWorkflowError::UnsupportedPublishTarget {
                target: config.target,
                reason: String::from(
                    "the current portable model-IO boundary imports GGUF but does not export GGUF; use the Hugging Face-style safetensors snapshot instead",
                ),
            }),
        }
    }
}

#[derive(Clone, Debug)]
struct SyntheticRecordLine {
    json_value: Value,
    approx_token_count: u64,
}

#[derive(Clone, Debug)]
struct SyntheticSplitRecords {
    split_name: String,
    kind: DatasetSplitKind,
    records: Vec<SyntheticRecordLine>,
}

fn validate_dataset_spec_header(
    workflow_id: &str,
    dataset: &DatasetKey,
    display_name: &str,
    tokenizer: &TokenizerDigest,
) -> Result<(), MlxWorkflowError> {
    if workflow_id.trim().is_empty() {
        return Err(MlxWorkflowError::MissingField {
            field: "workflow_id",
        });
    }
    if dataset.dataset_ref.trim().is_empty() {
        return Err(MlxWorkflowError::MissingField {
            field: "dataset.dataset_ref",
        });
    }
    if dataset.version.trim().is_empty() {
        return Err(MlxWorkflowError::MissingField {
            field: "dataset.version",
        });
    }
    if display_name.trim().is_empty() {
        return Err(MlxWorkflowError::MissingField {
            field: "display_name",
        });
    }
    if tokenizer.tokenizer_digest.trim().is_empty() {
        return Err(MlxWorkflowError::MissingField {
            field: "tokenizer.tokenizer_digest",
        });
    }
    Ok(())
}

fn validate_sft_sample(
    sample: &MlxSyntheticSftSample,
    split_name: &str,
    seen: &mut BTreeSet<String>,
) -> Result<(), MlxWorkflowError> {
    validate_sample_id(sample.sample_id.as_str(), split_name, seen)?;
    if sample.prompt.trim().is_empty() {
        return Err(MlxWorkflowError::InvalidSyntheticSample {
            split_name: String::from(split_name),
            sample_id: sample.sample_id.clone(),
            message: String::from("prompt must be non-empty"),
        });
    }
    if sample.response.trim().is_empty() {
        return Err(MlxWorkflowError::InvalidSyntheticSample {
            split_name: String::from(split_name),
            sample_id: sample.sample_id.clone(),
            message: String::from("response must be non-empty"),
        });
    }
    Ok(())
}

fn validate_preference_sample(
    sample: &MlxSyntheticPreferenceSample,
    split_name: &str,
    seen: &mut BTreeSet<String>,
) -> Result<(), MlxWorkflowError> {
    validate_sample_id(sample.sample_id.as_str(), split_name, seen)?;
    if sample.prompt.trim().is_empty() {
        return Err(MlxWorkflowError::InvalidSyntheticSample {
            split_name: String::from(split_name),
            sample_id: sample.sample_id.clone(),
            message: String::from("prompt must be non-empty"),
        });
    }
    if sample.chosen.trim().is_empty() || sample.rejected.trim().is_empty() {
        return Err(MlxWorkflowError::InvalidSyntheticSample {
            split_name: String::from(split_name),
            sample_id: sample.sample_id.clone(),
            message: String::from("chosen and rejected responses must be non-empty"),
        });
    }
    if sample.chosen == sample.rejected {
        return Err(MlxWorkflowError::InvalidSyntheticSample {
            split_name: String::from(split_name),
            sample_id: sample.sample_id.clone(),
            message: String::from("chosen and rejected responses must differ"),
        });
    }
    Ok(())
}

fn validate_sample_id(
    sample_id: &str,
    split_name: &str,
    seen: &mut BTreeSet<String>,
) -> Result<(), MlxWorkflowError> {
    if sample_id.trim().is_empty() {
        return Err(MlxWorkflowError::InvalidSyntheticSample {
            split_name: String::from(split_name),
            sample_id: String::from("<missing>"),
            message: String::from("sample_id must be non-empty"),
        });
    }
    if !seen.insert(String::from(sample_id)) {
        return Err(MlxWorkflowError::DuplicateSampleId {
            split_name: String::from(split_name),
            sample_id: String::from(sample_id),
        });
    }
    Ok(())
}

fn build_dataset_artifact(
    workflow_id: &str,
    dataset_kind: MlxSyntheticDatasetKind,
    dataset: &DatasetKey,
    display_name: &str,
    record_encoding: DatasetRecordEncoding,
    tokenizer: TokenizerDigest,
    context_window_tokens: Option<u32>,
    split_records: Vec<SyntheticSplitRecords>,
) -> Result<MlxSyntheticDatasetArtifact, MlxWorkflowError> {
    if split_records.is_empty() {
        return Err(MlxWorkflowError::MissingField { field: "splits" });
    }

    let mut split_artifacts = Vec::with_capacity(split_records.len());
    let mut split_declarations = Vec::with_capacity(split_records.len());
    for split in split_records {
        let jsonl_bytes = encode_jsonl_records(split.records.as_slice())?;
        let manifest = DatastreamManifest::from_bytes(
            format!("{}::{}", dataset.storage_key(), split.split_name),
            DatastreamSubjectKind::TokenizedCorpus,
            jsonl_bytes.as_slice(),
            64 * 1024,
            DatastreamEncoding::Jsonl,
        )
        .with_dataset_binding(dataset.datastream_binding(split.split_name.clone(), "shard-0000"));
        let approx_token_count = split
            .records
            .iter()
            .map(|record| record.approx_token_count)
            .sum();
        let min_sequence_tokens = split
            .records
            .iter()
            .map(|record| saturating_u64_to_u32(record.approx_token_count))
            .min()
            .unwrap_or(0);
        let max_sequence_tokens = split
            .records
            .iter()
            .map(|record| saturating_u64_to_u32(record.approx_token_count))
            .max()
            .unwrap_or(0);
        let shard = DatasetShardManifest::new(
            dataset,
            split.split_name.clone(),
            "shard-0000",
            manifest.manifest_ref(),
            split.records.len() as u64,
            approx_token_count,
            min_sequence_tokens,
            max_sequence_tokens,
        )?;
        split_declarations.push(DatasetSplitDeclaration::new(
            dataset,
            split.split_name.clone(),
            split.kind,
            vec![shard],
        )?);
        split_artifacts.push(MlxSyntheticSplitArtifact {
            split_name: split.split_name,
            kind: split.kind,
            datastream_manifest: manifest,
            jsonl_bytes,
            record_count: split.records.len(),
            approx_token_count,
        });
    }

    let mut dataset_manifest =
        DatasetManifest::new(dataset.clone(), display_name, record_encoding, tokenizer)
            .with_splits(split_declarations);
    if let Some(context_window_tokens) = context_window_tokens {
        dataset_manifest = dataset_manifest.with_context_window_tokens(context_window_tokens);
    }
    dataset_manifest = dataset_manifest.with_provenance_digest(stable_synthetic_provenance_digest(
        workflow_id,
        dataset_kind,
        split_artifacts.as_slice(),
    ));
    dataset_manifest.validate()?;

    let splits = split_artifacts
        .iter()
        .map(|split| MlxSyntheticSplitReceipt {
            split_name: split.split_name.clone(),
            kind: split.kind,
            record_count: split.record_count,
            approx_token_count: split.approx_token_count,
            datastream_manifest_digest: split.datastream_manifest.stable_digest(),
            object_digest: split.datastream_manifest.object_digest.clone(),
        })
        .collect::<Vec<_>>();
    let report = MlxSyntheticDatasetReport {
        workflow_id: String::from(workflow_id),
        dataset_kind,
        dataset_storage_key: dataset.storage_key(),
        dataset_manifest_digest: dataset_manifest.stable_digest(),
        report_digest: stable_synthetic_report_digest(
            workflow_id,
            dataset_kind,
            dataset_manifest.stable_digest().as_str(),
            splits.as_slice(),
        ),
        splits,
    };
    Ok(MlxSyntheticDatasetArtifact {
        dataset_kind,
        dataset_manifest,
        split_artifacts,
        report,
    })
}

fn validate_helper_method(
    helper_kind: MlxSupervisionHelperKind,
    method: MlxRecipeMethod,
) -> Result<(), MlxWorkflowError> {
    let supported = match helper_kind {
        MlxSupervisionHelperKind::JudgeModel => !matches!(
            method,
            MlxRecipeMethod::Grpo
                | MlxRecipeMethod::OnlineDpo
                | MlxRecipeMethod::Xpo
                | MlxRecipeMethod::Ppo
        ),
        MlxSupervisionHelperKind::RewardModel => matches!(
            method,
            MlxRecipeMethod::Sft
                | MlxRecipeMethod::Lora
                | MlxRecipeMethod::Dora
                | MlxRecipeMethod::Qlora
        ),
    };
    supported
        .then_some(())
        .ok_or(MlxWorkflowError::UnsupportedHelperMethod {
            helper_kind,
            method,
        })
}

fn helper_policy_references(
    helper_id: &str,
    helper_kind: MlxSupervisionHelperKind,
) -> Vec<EnvironmentPolicyReference> {
    let mut references = vec![EnvironmentPolicyReference {
        kind: EnvironmentPolicyKind::Training,
        policy_ref: format!("{helper_id}.training"),
        required: true,
    }];
    references.push(EnvironmentPolicyReference {
        kind: match helper_kind {
            MlxSupervisionHelperKind::JudgeModel => EnvironmentPolicyKind::Verification,
            MlxSupervisionHelperKind::RewardModel => EnvironmentPolicyKind::Reward,
        },
        policy_ref: format!(
            "{helper_id}.{}",
            match helper_kind {
                MlxSupervisionHelperKind::JudgeModel => "judge",
                MlxSupervisionHelperKind::RewardModel => "reward",
            }
        ),
        required: true,
    });
    references
}

fn helper_notes(helper_kind: MlxSupervisionHelperKind, method: MlxRecipeMethod) -> Vec<String> {
    let mut notes = vec![String::from(
        "This helper reuses the `psionic-mlx-recipes` planner and the existing train substrate instead of introducing a second reward/judge trainer architecture.",
    )];
    match helper_kind {
        MlxSupervisionHelperKind::JudgeModel => notes.push(String::from(
            "Judge helpers publish one verification-facing policy reference plus the selected recipe plan so rubric-style supervision remains machine-legible.",
        )),
        MlxSupervisionHelperKind::RewardModel => notes.push(String::from(
            "Reward helpers publish one reward-facing policy reference plus the selected recipe plan so score-model training remains bound to explicit policy ids.",
        )),
    }
    if method.requires_adapter() {
        notes.push(String::from(
            "Adapter-capable supervision helpers continue to target the open adapter lane through the selected MLX recipe plan.",
        ));
    }
    notes
}

fn validate_merge_pair(
    base: &PortableModelBundle,
    tuned: &PortableModelBundle,
) -> Result<(), MlxWorkflowError> {
    if base.state_dict.model_family != tuned.state_dict.model_family {
        return Err(MlxWorkflowError::IncompatibleMergePair {
            message: String::from("model_family must match"),
        });
    }
    if base.state_dict.revision != tuned.state_dict.revision {
        return Err(MlxWorkflowError::IncompatibleMergePair {
            message: String::from("revision must match"),
        });
    }
    if base.state_dict.checkpoint_family != tuned.state_dict.checkpoint_family {
        return Err(MlxWorkflowError::IncompatibleMergePair {
            message: String::from("checkpoint_family must match"),
        });
    }
    if base.state_dict.checkpoint_ref != tuned.state_dict.checkpoint_ref {
        return Err(MlxWorkflowError::IncompatibleMergePair {
            message: String::from("checkpoint_ref must match"),
        });
    }
    if base.tokenizer != tuned.tokenizer {
        return Err(MlxWorkflowError::IncompatibleMergePair {
            message: String::from("tokenizer binding must match"),
        });
    }
    if base.chat_template_digest != tuned.chat_template_digest {
        return Err(MlxWorkflowError::IncompatibleMergePair {
            message: String::from("chat_template_digest must match"),
        });
    }
    Ok(())
}

fn publish_hugging_face_snapshot(
    config: &MlxPublishConfig,
    bundle: &PortableModelBundle,
    output_dir: &Path,
) -> Result<MlxPublishManifest, MlxWorkflowError> {
    fs::create_dir_all(output_dir).map_err(|error| MlxWorkflowError::Io {
        path: output_dir.display().to_string(),
        message: error.to_string(),
    })?;

    let (model_bytes, source_artifact_receipt) = bundle.export_safetensors()?;
    if source_artifact_receipt.format != ModelArtifactFormat::Safetensors {
        return Err(MlxWorkflowError::UnsupportedPublishTarget {
            target: config.target,
            reason: String::from("publish snapshots currently require safetensors export"),
        });
    }

    let bundle_manifest_bytes = to_pretty_json_bytes(&bundle.manifest(), "bundle manifest export")?;
    let tokenizer_contract_bytes =
        to_pretty_json_bytes(&bundle.tokenizer, "tokenizer contract export")?;
    let compatibility_contract = bundle.compatibility_contract();
    let compatibility_contract_bytes =
        to_pretty_json_bytes(&compatibility_contract, "compatibility contract export")?;
    let readme_bytes = build_publish_readme(config, bundle, &source_artifact_receipt).into_bytes();

    let mut files = vec![
        write_bytes_file(output_dir.join("model.safetensors"), model_bytes.as_slice())?,
        write_bytes_file(
            output_dir.join("psionic_bundle_manifest.json"),
            bundle_manifest_bytes.as_slice(),
        )?,
        write_bytes_file(
            output_dir.join("tokenizer_contract.json"),
            tokenizer_contract_bytes.as_slice(),
        )?,
        write_bytes_file(
            output_dir.join("compatibility_contract.json"),
            compatibility_contract_bytes.as_slice(),
        )?,
        write_bytes_file(output_dir.join("README.md"), readme_bytes.as_slice())?,
    ];
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

    let notes = vec![
        String::from(
            "The current publish workflow writes one local Hugging Face-style snapshot directory over the existing safetensors export boundary.",
        ),
        String::from(
            "Direct GGUF export remains an explicit refusal because the current portable model-IO layer imports GGUF but does not emit GGUF.",
        ),
        if compatibility_contract.surfaces.iter().any(|surface| {
            surface.surface == psionic_train::ModelInteropSurface::Gguf
                && surface.export_status == ModelInteropStatus::Unsupported
        }) {
            String::from(
                "The emitted compatibility contract keeps the GGUF export boundary explicit instead of pretending the snapshot is a direct GGUF artifact.",
            )
        } else {
            String::from(
                "The emitted compatibility contract preserves the portable model-IO export boundary for downstream review.",
            )
        },
    ];
    let manifest = MlxPublishManifest {
        publish_id: config.publish_id.clone(),
        target: config.target,
        repo_id: config.repo_id.clone(),
        source_artifact_receipt,
        manifest_digest: stable_publish_manifest_digest(
            config.publish_id.as_str(),
            config.target,
            config.repo_id.as_str(),
            &files,
            &notes,
        ),
        files,
        notes,
    };
    let _ = write_json_file(
        output_dir.join("publish_manifest.json"),
        &manifest,
        "publish manifest export",
    )?;
    Ok(manifest)
}

fn build_publish_readme(
    config: &MlxPublishConfig,
    bundle: &PortableModelBundle,
    receipt: &ModelIoArtifactReceipt,
) -> String {
    format!(
        "# {}\n\nPublished by `psionic-mlx-workflows`.\n\n- publish_id: `{}`\n- state_dict_digest: `{}`\n- tokenizer_contract_digest: `{}`\n- artifact_format: `{:?}`\n- artifact_digest: `{}`\n- preferred_serving_formats: `{}`\n",
        config.repo_id,
        config.publish_id,
        bundle.state_dict.digest,
        bundle.tokenizer.contract_digest(),
        receipt.format,
        receipt.artifact_digest,
        bundle
            .preferred_serving_formats
            .iter()
            .map(|format| format!("{format:?}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn write_json_file<T: Serialize>(
    path: PathBuf,
    value: &T,
    context: &'static str,
) -> Result<MlxWorkflowFile, MlxWorkflowError> {
    let bytes = to_pretty_json_bytes(value, context)?;
    write_bytes_file(path, bytes.as_slice())
}

fn to_pretty_json_bytes<T: Serialize>(
    value: &T,
    context: &'static str,
) -> Result<Vec<u8>, MlxWorkflowError> {
    serde_json::to_vec_pretty(value).map_err(|error| MlxWorkflowError::Serialization {
        context,
        message: error.to_string(),
    })
}

fn write_bytes_file(path: PathBuf, bytes: &[u8]) -> Result<MlxWorkflowFile, MlxWorkflowError> {
    fs::write(&path, bytes).map_err(|error| MlxWorkflowError::Io {
        path: path.display().to_string(),
        message: error.to_string(),
    })?;
    Ok(MlxWorkflowFile {
        relative_path: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string(),
        byte_length: bytes.len() as u64,
        sha256: digest_bytes(bytes),
    })
}

fn encode_jsonl_records(records: &[SyntheticRecordLine]) -> Result<Vec<u8>, MlxWorkflowError> {
    let mut encoded = Vec::new();
    for record in records {
        let line = serde_json::to_vec(&record.json_value).map_err(|error| {
            MlxWorkflowError::Serialization {
                context: "synthetic jsonl export",
                message: error.to_string(),
            }
        })?;
        encoded.extend_from_slice(&line);
        encoded.push(b'\n');
    }
    Ok(encoded)
}

fn approx_text_tokens(text: &str) -> u64 {
    text.split_whitespace().count().max(1) as u64
}

fn saturating_u64_to_u32(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn stable_synthetic_provenance_digest(
    workflow_id: &str,
    dataset_kind: MlxSyntheticDatasetKind,
    split_artifacts: &[MlxSyntheticSplitArtifact],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_synthetic_provenance|");
    hasher.update(workflow_id.as_bytes());
    hasher.update(b"|");
    hasher.update(match dataset_kind {
        MlxSyntheticDatasetKind::Sft => b"sft".as_slice(),
        MlxSyntheticDatasetKind::Preference => b"preference".as_slice(),
    });
    for split in split_artifacts {
        hasher.update(b"|split|");
        hasher.update(split.split_name.as_bytes());
        hasher.update(b"|");
        hasher.update(split.datastream_manifest.object_digest.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_synthetic_report_digest(
    workflow_id: &str,
    dataset_kind: MlxSyntheticDatasetKind,
    dataset_manifest_digest: &str,
    splits: &[MlxSyntheticSplitReceipt],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_synthetic_report|");
    hasher.update(workflow_id.as_bytes());
    hasher.update(b"|");
    hasher.update(match dataset_kind {
        MlxSyntheticDatasetKind::Sft => b"sft".as_slice(),
        MlxSyntheticDatasetKind::Preference => b"preference".as_slice(),
    });
    hasher.update(b"|");
    hasher.update(dataset_manifest_digest.as_bytes());
    for split in splits {
        hasher.update(b"|split|");
        hasher.update(split.split_name.as_bytes());
        hasher.update(b"|");
        hasher.update(split.datastream_manifest_digest.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_supervision_helper_digest(
    helper_id: &str,
    helper_kind: MlxSupervisionHelperKind,
    dataset_storage_key: &str,
    recipe_plan: &MlxRecipePlan,
    label_schema: &MlxSupervisionLabelSchema,
    policy_references: &[EnvironmentPolicyReference],
    output_artifact_family: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_supervision_helper|");
    hasher.update(helper_id.as_bytes());
    hasher.update(b"|");
    hasher.update(match helper_kind {
        MlxSupervisionHelperKind::JudgeModel => b"judge".as_slice(),
        MlxSupervisionHelperKind::RewardModel => b"reward".as_slice(),
    });
    hasher.update(b"|");
    hasher.update(dataset_storage_key.as_bytes());
    hasher.update(b"|");
    hasher.update(recipe_plan.config_digest.as_bytes());
    hasher.update(b"|");
    hasher.update(label_schema.label_field.as_bytes());
    hasher.update(b"|");
    hasher.update(match label_schema.label_kind {
        MlxSupervisionLabelKind::Binary => b"binary".as_slice(),
        MlxSupervisionLabelKind::Scalar => b"scalar".as_slice(),
    });
    if let Some(min_score) = label_schema.min_score {
        hasher.update(b"|min|");
        hasher.update(min_score.to_bits().to_le_bytes());
    }
    if let Some(max_score) = label_schema.max_score {
        hasher.update(b"|max|");
        hasher.update(max_score.to_bits().to_le_bytes());
    }
    hasher.update(b"|");
    hasher.update(output_artifact_family.as_bytes());
    for reference in policy_references {
        hasher.update(b"|policy|");
        hasher.update(reference.kind.to_string().as_bytes());
        hasher.update(b"|");
        hasher.update(reference.policy_ref.as_bytes());
        hasher.update(b"|");
        hasher.update(if reference.required { b"1" } else { b"0" });
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_adapter_merge_report_digest(
    merge_id: &str,
    adapter_id: &str,
    base_state_dict_digest: &str,
    target_state_dict_digest: &str,
    delta_tensor_count: usize,
    notes: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_adapter_merge_report|");
    hasher.update(merge_id.as_bytes());
    hasher.update(b"|");
    hasher.update(adapter_id.as_bytes());
    hasher.update(b"|");
    hasher.update(base_state_dict_digest.as_bytes());
    hasher.update(b"|");
    hasher.update(target_state_dict_digest.as_bytes());
    hasher.update(b"|");
    hasher.update(delta_tensor_count.to_string().as_bytes());
    for note in notes {
        hasher.update(b"|note|");
        hasher.update(note.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_publish_manifest_digest(
    publish_id: &str,
    target: MlxPublishTarget,
    repo_id: &str,
    files: &[MlxWorkflowFile],
    notes: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_publish_manifest|");
    hasher.update(publish_id.as_bytes());
    hasher.update(b"|");
    hasher.update(match target {
        MlxPublishTarget::HuggingFaceSnapshot => b"hf_snapshot".as_slice(),
        MlxPublishTarget::GgufArtifact => b"gguf_artifact".as_slice(),
    });
    hasher.update(b"|");
    hasher.update(repo_id.as_bytes());
    for file in files {
        hasher.update(b"|file|");
        hasher.update(file.relative_path.as_bytes());
        hasher.update(b"|");
        hasher.update(file.byte_length.to_string().as_bytes());
        hasher.update(b"|");
        hasher.update(file.sha256.as_bytes());
    }
    for note in notes {
        hasher.update(b"|note|");
        hasher.update(note.as_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn stable_directory_write_digest(root: &Path, report_digest: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"psionic_mlx_directory_write|");
    hasher.update(root.display().to_string().as_bytes());
    hasher.update(b"|");
    hasher.update(report_digest.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::{
        MlxAdapterMergeConfig, MlxPublishConfig, MlxPublishTarget, MlxSupervisionHelperConfig,
        MlxSupervisionHelperKind, MlxSupervisionLabelKind, MlxSupervisionLabelSchema,
        MlxSyntheticPreferenceDatasetSpec, MlxSyntheticPreferenceSample,
        MlxSyntheticPreferenceSplit, MlxSyntheticSftDatasetSpec, MlxSyntheticSftSample,
        MlxSyntheticSftSplit, MlxWorkflowError, MlxWorkflowWorkspace,
    };
    use psionic_core::{DType, Device, Shape};
    use psionic_data::{DatasetKey, DatasetSplitKind, TokenizerDigest, TokenizerFamily};
    use psionic_environments::EnvironmentPackageKey;
    use psionic_mlx_recipes::{MlxAdapterRecipe, MlxRecipeConfig, MlxRecipeMethod};
    use psionic_train::{
        OptimizerStateResidency, PortableModelBundle, PortableTokenizerAssetFormat,
        PortableTokenizerBinding, TrainingOptimizerConfig, TrainingOptimizerResidencyPolicy,
        TrainingOptimizerState, TrainingParameterClass, TrainingParameterGroupState,
        TrainingTensorBuffer,
    };
    use tempfile::tempdir;

    #[test]
    fn synthetic_sft_dataset_writes_manifest_and_jsonl() -> Result<(), Box<dyn std::error::Error>> {
        let artifact = MlxWorkflowWorkspace::default().build_synthetic_sft_dataset(
            &MlxSyntheticSftDatasetSpec {
                workflow_id: String::from("workflow.sft"),
                dataset: DatasetKey::new("mlx/synthetic-sft", "v1"),
                display_name: String::from("Synthetic SFT"),
                tokenizer: sample_tokenizer(),
                context_window_tokens: Some(2048),
                splits: vec![
                    MlxSyntheticSftSplit {
                        split_name: String::from("train"),
                        kind: DatasetSplitKind::Train,
                        samples: vec![MlxSyntheticSftSample {
                            sample_id: String::from("sft-train-1"),
                            prompt: String::from("Explain grouped all-reduce."),
                            response: String::from("It reduces gradients across a group."),
                            tags: vec![String::from("collectives")],
                        }],
                    },
                    MlxSyntheticSftSplit {
                        split_name: String::from("validation"),
                        kind: DatasetSplitKind::Validation,
                        samples: vec![MlxSyntheticSftSample {
                            sample_id: String::from("sft-val-1"),
                            prompt: String::from("What does typed refusal mean?"),
                            response: String::from("Unsupported behavior must stay explicit."),
                            tags: Vec::new(),
                        }],
                    },
                ],
            },
        )?;

        assert_eq!(
            artifact.dataset_manifest.record_encoding,
            psionic_data::DatasetRecordEncoding::JsonlText
        );
        assert_eq!(artifact.dataset_manifest.splits.len(), 2);
        let temp = tempdir()?;
        let write = artifact.write_to_directory(temp.path())?;
        assert!(
            write
                .files
                .iter()
                .any(|file| file.relative_path == "train.jsonl")
        );
        assert!(temp.path().join("dataset_manifest.json").exists());
        assert!(temp.path().join("synthetic_dataset_report.json").exists());
        Ok(())
    }

    #[test]
    fn synthetic_preference_dataset_uses_preference_encoding()
    -> Result<(), Box<dyn std::error::Error>> {
        let artifact = MlxWorkflowWorkspace::default().build_synthetic_preference_dataset(
            &MlxSyntheticPreferenceDatasetSpec {
                workflow_id: String::from("workflow.pref"),
                dataset: DatasetKey::new("mlx/synthetic-preference", "v1"),
                display_name: String::from("Synthetic Preference"),
                tokenizer: sample_tokenizer(),
                context_window_tokens: None,
                splits: vec![MlxSyntheticPreferenceSplit {
                    split_name: String::from("preference"),
                    kind: DatasetSplitKind::Preference,
                    samples: vec![MlxSyntheticPreferenceSample {
                        sample_id: String::from("pref-1"),
                        prompt: String::from("Summarize replay truth."),
                        chosen: String::from("Tie the report to stable receipts and seeds."),
                        rejected: String::from("Just say it is deterministic."),
                        rationale: Some(String::from("chosen answer preserves receipt truth")),
                    }],
                }],
            },
        )?;

        assert_eq!(
            artifact.dataset_manifest.record_encoding,
            psionic_data::DatasetRecordEncoding::PreferenceJsonl
        );
        assert_eq!(artifact.split_artifacts[0].record_count, 1);
        Ok(())
    }

    #[test]
    fn synthetic_dataset_refuses_duplicate_split_names() {
        let error = MlxWorkflowWorkspace::default()
            .build_synthetic_sft_dataset(&MlxSyntheticSftDatasetSpec {
                workflow_id: String::from("workflow.duplicate-split"),
                dataset: DatasetKey::new("mlx/synthetic-duplicate", "v1"),
                display_name: String::from("Duplicate Split"),
                tokenizer: sample_tokenizer(),
                context_window_tokens: None,
                splits: vec![
                    MlxSyntheticSftSplit {
                        split_name: String::from("train"),
                        kind: DatasetSplitKind::Train,
                        samples: vec![MlxSyntheticSftSample {
                            sample_id: String::from("train-1"),
                            prompt: String::from("first"),
                            response: String::from("one"),
                            tags: Vec::new(),
                        }],
                    },
                    MlxSyntheticSftSplit {
                        split_name: String::from("train"),
                        kind: DatasetSplitKind::Validation,
                        samples: vec![MlxSyntheticSftSample {
                            sample_id: String::from("validation-1"),
                            prompt: String::from("second"),
                            response: String::from("two"),
                            tags: Vec::new(),
                        }],
                    },
                ],
            })
            .expect_err("expected duplicate split refusal");
        assert!(matches!(
            error,
            MlxWorkflowError::DuplicateSplitName { ref split_name } if split_name == "train"
        ));
    }

    #[test]
    fn reward_helper_reuses_recipe_plan_and_policy_refs() -> Result<(), Box<dyn std::error::Error>>
    {
        let config = MlxRecipeConfig::new(
            "reward-run",
            "cluster-a",
            "reward-checkpoints",
            EnvironmentPackageKey::new("env.reward", "v1"),
            MlxRecipeMethod::Qlora,
        )?
        .with_adapter(MlxAdapterRecipe {
            method: MlxRecipeMethod::Qlora,
            rank: 16,
            alpha: 32.0,
            quantization: Some(String::from("q4_k")),
        });
        let plan = MlxWorkflowWorkspace::default().plan_supervision_helper(
            &MlxSupervisionHelperConfig {
                helper_id: String::from("reward-helper"),
                helper_kind: MlxSupervisionHelperKind::RewardModel,
                dataset_storage_key: String::from("mlx/synthetic-preference@v1"),
                recipe: config,
                label_schema: MlxSupervisionLabelSchema {
                    label_field: String::from("score"),
                    label_kind: MlxSupervisionLabelKind::Scalar,
                    min_score: Some(0.0),
                    max_score: Some(1.0),
                },
            },
        )?;

        assert_eq!(plan.recipe_plan.method, MlxRecipeMethod::Qlora);
        assert_eq!(plan.policy_references.len(), 2);
        assert!(
            plan.policy_references
                .iter()
                .any(|reference| reference.kind
                    == psionic_environments::EnvironmentPolicyKind::Reward)
        );
        Ok(())
    }

    #[test]
    fn judge_helper_refuses_rl_only_method() {
        let config = MlxRecipeConfig::new(
            "judge-run",
            "cluster-a",
            "judge-checkpoints",
            EnvironmentPackageKey::new("env.judge", "v1"),
            MlxRecipeMethod::Ppo,
        )
        .expect("config");
        let error = MlxWorkflowWorkspace::default()
            .plan_supervision_helper(&MlxSupervisionHelperConfig {
                helper_id: String::from("judge-helper"),
                helper_kind: MlxSupervisionHelperKind::JudgeModel,
                dataset_storage_key: String::from("mlx/synthetic-sft@v1"),
                recipe: config,
                label_schema: MlxSupervisionLabelSchema {
                    label_field: String::from("pass"),
                    label_kind: MlxSupervisionLabelKind::Binary,
                    min_score: None,
                    max_score: None,
                },
            })
            .expect_err("expected refusal");
        assert!(matches!(
            error,
            MlxWorkflowError::UnsupportedHelperMethod {
                helper_kind: MlxSupervisionHelperKind::JudgeModel,
                method: MlxRecipeMethod::Ppo
            }
        ));
    }

    #[test]
    fn adapter_merge_exports_merged_bundle_and_publish_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let base = sample_bundle()?;
        let mut tuned = sample_bundle()?;
        let psionic_core::TensorData::F32(values) = &mut tuned
            .state_dict
            .tensors
            .get_mut("model.decoder.head.parameter")
            .expect("head")
            .data
        else {
            panic!("expected dense tensor");
        };
        values[0] += 0.25;
        values[1] -= 0.5;
        tuned.state_dict = psionic_train::PortableModelStateDict::new(
            tuned.state_dict.model_family.clone(),
            tuned.state_dict.revision.clone(),
            tuned.state_dict.checkpoint_family.clone(),
            tuned.state_dict.checkpoint_ref.clone(),
            tuned.state_dict.source_format,
            tuned.state_dict.groups.clone(),
            tuned.state_dict.tensors.clone(),
        )?;

        let workspace = MlxWorkflowWorkspace::default();
        let merged = workspace.merge_adapter(
            &MlxAdapterMergeConfig {
                merge_id: String::from("merge-1"),
                adapter_id: String::from("adapter-weather"),
            },
            &base,
            &tuned,
        )?;
        assert_eq!(
            merged.report.target_state_dict_digest,
            tuned.state_dict.digest
        );
        assert!(!merged.merged_safetensors.is_empty());

        let publish_dir = tempdir()?;
        let manifest = workspace.publish_bundle(
            &MlxPublishConfig {
                publish_id: String::from("publish-1"),
                target: MlxPublishTarget::HuggingFaceSnapshot,
                repo_id: String::from("openagents/adapter-weather"),
            },
            &merged.merged_bundle,
            publish_dir.path(),
        )?;
        assert!(publish_dir.path().join("model.safetensors").exists());
        assert!(publish_dir.path().join("publish_manifest.json").exists());
        assert_eq!(manifest.target, MlxPublishTarget::HuggingFaceSnapshot);
        Ok(())
    }

    #[test]
    fn gguf_publish_target_refuses_explicitly() -> Result<(), Box<dyn std::error::Error>> {
        let bundle = sample_bundle()?;
        let error = MlxWorkflowWorkspace::default()
            .publish_bundle(
                &MlxPublishConfig {
                    publish_id: String::from("publish-gguf"),
                    target: MlxPublishTarget::GgufArtifact,
                    repo_id: String::from("openagents/weather"),
                },
                &bundle,
                tempdir()?.path(),
            )
            .expect_err("expected refusal");
        assert!(matches!(
            error,
            MlxWorkflowError::UnsupportedPublishTarget {
                target: MlxPublishTarget::GgufArtifact,
                ..
            }
        ));
        Ok(())
    }

    fn sample_bundle() -> Result<PortableModelBundle, Box<dyn std::error::Error>> {
        let mut embedding = TrainingParameterGroupState::new(
            "embedding",
            TrainingParameterClass::Embedding,
            TrainingTensorBuffer::from_f32(
                "embedding",
                psionic_core::TensorSpec::new(Shape::new(vec![2, 2]), DType::F32, Device::cpu()),
                vec![1.0, 2.0, 3.0, 4.0],
            )?,
            TrainingOptimizerConfig::sgd(0.1).with_momentum(0.9),
            TrainingOptimizerResidencyPolicy::new(
                OptimizerStateResidency::DeviceResident,
                OptimizerStateResidency::HostResident,
            ),
        )?;
        embedding.optimizer_state = TrainingOptimizerState::Sgd {
            momentum_buffer: Some(vec![0.1, 0.2, 0.3, 0.4]),
        };
        embedding.optimizer_residency = OptimizerStateResidency::DeviceResident;
        embedding.applied_steps = 3;

        let mut decoder_head = TrainingParameterGroupState::new(
            "decoder.head",
            TrainingParameterClass::Head,
            TrainingTensorBuffer::from_f32(
                "decoder.head",
                psionic_core::TensorSpec::new(Shape::new(vec![2]), DType::F32, Device::cpu()),
                vec![0.5, -0.5],
            )?,
            TrainingOptimizerConfig::adamw(0.01, 0.9, 0.999, 1e-8).with_weight_decay(0.01),
            TrainingOptimizerResidencyPolicy::new(
                OptimizerStateResidency::DeviceResident,
                OptimizerStateResidency::Offloaded,
            ),
        )?;
        decoder_head.optimizer_state = TrainingOptimizerState::AdamW {
            first_moment: vec![0.01, -0.02],
            second_moment: vec![0.03, 0.04],
        };
        decoder_head.optimizer_residency = OptimizerStateResidency::Offloaded;
        decoder_head.applied_steps = 7;

        Ok(PortableModelBundle::from_training_groups(
            "weather-agent",
            "r1",
            "weather-checkpoints",
            Some(String::from("checkpoint://weather/7")),
            &[embedding, decoder_head],
            PortableTokenizerBinding::new(
                TokenizerDigest::new(
                    TokenizerFamily::BytePairEncoding,
                    "weather-tokenizer-digest",
                    32_000,
                )
                .with_special_tokens_digest("weather-tokenizer-specials")
                .with_template_digest("chat-template-digest"),
                PortableTokenizerAssetFormat::TokenizerJson,
                "2026-03-17",
            )
            .with_special_tokens(Some(1), vec![2], Some(0), Some(3), true, false),
            Some(String::from("chat-template-digest")),
        )?)
    }

    fn sample_tokenizer() -> TokenizerDigest {
        TokenizerDigest::new(
            TokenizerFamily::BytePairEncoding,
            "workflow-tokenizer",
            32_000,
        )
        .with_special_tokens_digest("workflow-tokenizer-specials")
        .with_template_digest("workflow-template")
    }
}
