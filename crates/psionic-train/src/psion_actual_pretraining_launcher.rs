use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF_PATH,
    PSION_ACTUAL_PRETRAINING_DRY_RUN_SURFACE_ID, PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT_ID,
    PSION_ACTUAL_PRETRAINING_LANE_ID, PSION_ACTUAL_PRETRAINING_RECIPE_ID,
    PSION_ACTUAL_PRETRAINING_RESUME_SURFACE_ID, PSION_ACTUAL_PRETRAINING_START_SURFACE_ID,
    PSION_ACTUAL_PRETRAINING_TOPOLOGY_STORAGE_BUNDLE_ID, PsionActualPretrainingArtifactRef,
    PsionActualPretrainingLauncherSurfaces,
};

/// Stable schema version for the canonical actual-lane launch manifest.
pub const PSION_ACTUAL_PRETRAINING_LAUNCH_MANIFEST_SCHEMA_VERSION: &str =
    "psion.actual_pretraining_launch_manifest.v1";

/// Stable schema version for the canonical actual-lane resume manifest.
pub const PSION_ACTUAL_PRETRAINING_RESUME_MANIFEST_SCHEMA_VERSION: &str =
    "psion.actual_pretraining_resume_manifest.v1";

/// Stable schema version for the canonical actual-lane checkpoint pointer.
pub const PSION_ACTUAL_PRETRAINING_CHECKPOINT_POINTER_SCHEMA_VERSION: &str =
    "psion.actual_pretraining_checkpoint_pointer.v1";

/// Stable schema version for the canonical actual-lane closeout bundle.
pub const PSION_ACTUAL_PRETRAINING_CLOSEOUT_BUNDLE_SCHEMA_VERSION: &str =
    "psion.actual_pretraining_closeout_bundle.v1";

/// Fixed retained paths used by the actual-lane launcher contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionActualPretrainingRetainedPathSet {
    /// Relative launch manifest path.
    pub launch_manifest_path: String,
    /// Relative resume manifest path.
    pub resume_manifest_path: String,
    /// Relative current status path.
    pub current_status_path: String,
    /// Relative retained summary path.
    pub retained_summary_path: String,
    /// Relative latest-checkpoint pointer path.
    pub latest_checkpoint_pointer_path: String,
    /// Relative continuation handoff path.
    pub continuation_handoff_path: String,
    /// Relative closeout bundle path.
    pub closeout_bundle_path: String,
    /// Relative launcher log path.
    pub launcher_log_path: String,
}

/// Contract refs consumed by the actual-lane launcher.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionActualPretrainingLauncherContractRefs {
    /// Actual-lane spec fixture.
    pub lane_spec: PsionActualPretrainingArtifactRef,
    /// Recipe bundle fixture.
    pub recipe_bundle: PsionActualPretrainingArtifactRef,
    /// Topology/storage bundle fixture.
    pub topology_storage_bundle: PsionActualPretrainingArtifactRef,
    /// Evidence contract fixture.
    pub evidence_contract: PsionActualPretrainingArtifactRef,
}

/// Redacted credential-source binding retained by the launcher.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionActualPretrainingCredentialBinding {
    /// Credential-source kind.
    pub kind: String,
    /// Declared source name.
    pub source_name: String,
    /// Redaction policy retained by artifacts.
    pub retained_redaction: String,
}

/// Actual run roots selected by the launcher.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionActualPretrainingRunRoots {
    /// Local run root.
    pub local_run_root: String,
    /// Remote durable run root.
    pub remote_run_root: String,
    /// Remote durable checkpoint root.
    pub remote_checkpoint_root: String,
    /// Remote durable manifest root.
    pub remote_manifest_root: String,
    /// Remote transient log root.
    pub remote_log_root: String,
}

/// Launch manifest for the actual pretraining lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionActualPretrainingLaunchManifest {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable launcher surface id.
    pub surface_id: String,
    /// Stable actual-lane identifier.
    pub lane_id: String,
    /// Stable recipe identifier.
    pub recipe_id: String,
    /// Stable topology/storage bundle identifier.
    pub topology_storage_bundle_id: String,
    /// Stable evidence-contract identifier.
    pub evidence_contract_id: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Fixed retained paths under the run root.
    pub retained_paths: PsionActualPretrainingRetainedPathSet,
    /// Reserved launcher surface ids.
    pub launcher_surfaces: PsionActualPretrainingLauncherSurfaces,
    /// Selected local and remote run roots.
    pub run_roots: PsionActualPretrainingRunRoots,
    /// Committed contract refs this launch consumes directly.
    pub contract_refs: PsionActualPretrainingLauncherContractRefs,
    /// Selected git ref.
    pub selected_git_ref: String,
    /// Exact resolved git commit SHA.
    pub git_commit_sha: String,
    /// Dirty-tree admission posture.
    pub dirty_tree_admission: String,
    /// Optional digest of the status snapshot when dirty-tree override is used.
    pub workspace_status_sha256: Option<String>,
    /// Redacted credential-source bindings.
    pub credential_sources: Vec<PsionActualPretrainingCredentialBinding>,
    /// Narrow claim boundary.
    pub claim_boundary: String,
    /// Short detail.
    pub detail: String,
}

/// Resume manifest for the actual pretraining lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionActualPretrainingResumeManifest {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable launcher surface id.
    pub surface_id: String,
    /// Stable actual-lane identifier.
    pub lane_id: String,
    /// Stable recipe identifier.
    pub recipe_id: String,
    /// Stable topology/storage bundle identifier.
    pub topology_storage_bundle_id: String,
    /// Stable evidence-contract identifier.
    pub evidence_contract_id: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Fixed retained paths under the run root.
    pub retained_paths: PsionActualPretrainingRetainedPathSet,
    /// Reserved launcher surface ids.
    pub launcher_surfaces: PsionActualPretrainingLauncherSurfaces,
    /// Selected local and remote run roots.
    pub run_roots: PsionActualPretrainingRunRoots,
    /// Committed contract refs this resume consumes directly.
    pub contract_refs: PsionActualPretrainingLauncherContractRefs,
    /// Selected git ref.
    pub selected_git_ref: String,
    /// Exact resolved git commit SHA.
    pub git_commit_sha: String,
    /// Dirty-tree admission posture.
    pub dirty_tree_admission: String,
    /// Optional digest of the status snapshot when dirty-tree override is used.
    pub workspace_status_sha256: Option<String>,
    /// Relative path to the canonical checkpoint pointer.
    pub latest_checkpoint_pointer_path: String,
    /// Checkpoint label chosen for resume.
    pub checkpoint_label: String,
    /// Optimizer step chosen for resume.
    pub optimizer_step: u64,
    /// Checkpoint ref chosen for resume.
    pub checkpoint_ref: String,
    /// Narrow claim boundary.
    pub claim_boundary: String,
    /// Short detail.
    pub detail: String,
}

/// Canonical latest-checkpoint pointer for the actual lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionActualPretrainingCheckpointPointer {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable actual-lane identifier.
    pub lane_id: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Pointer state.
    pub pointer_state: String,
    /// Last known checkpoint label.
    pub checkpoint_label: String,
    /// Last known accepted step.
    pub optimizer_step: u64,
    /// Optional checkpoint ref for an accepted pointer.
    pub checkpoint_ref: Option<String>,
    /// Optional checkpoint-manifest path for an accepted pointer.
    pub checkpoint_manifest_relative_path: Option<String>,
    /// Short detail.
    pub detail: String,
}

/// Provisional closeout bundle emitted by the actual-lane launcher.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionActualPretrainingCloseoutBundle {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable actual-lane identifier.
    pub lane_id: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Current closeout state.
    pub closeout_state: String,
    /// Fixed retained paths under the run root.
    pub retained_paths: PsionActualPretrainingRetainedPathSet,
    /// Selected git ref.
    pub selected_git_ref: String,
    /// Exact resolved git commit SHA.
    pub git_commit_sha: String,
    /// Dirty-tree admission posture.
    pub dirty_tree_admission: String,
    /// Optional digest of the status snapshot when dirty-tree override is used.
    pub workspace_status_sha256: Option<String>,
    /// Narrow claim boundary.
    pub claim_boundary: String,
    /// Short detail.
    pub detail: String,
}

impl PsionActualPretrainingRetainedPathSet {
    /// Validates the fixed retained paths.
    pub fn validate(&self) -> Result<(), PsionActualPretrainingLauncherError> {
        ensure_exact(
            self.launch_manifest_path.as_str(),
            "retained_paths.launch_manifest_path",
            "manifests/launch_manifest.json",
        )?;
        ensure_exact(
            self.resume_manifest_path.as_str(),
            "retained_paths.resume_manifest_path",
            "manifests/resume_manifest.json",
        )?;
        ensure_exact(
            self.current_status_path.as_str(),
            "retained_paths.current_status_path",
            "status/current_run_status.json",
        )?;
        ensure_exact(
            self.retained_summary_path.as_str(),
            "retained_paths.retained_summary_path",
            "status/retained_summary.json",
        )?;
        ensure_exact(
            self.latest_checkpoint_pointer_path.as_str(),
            "retained_paths.latest_checkpoint_pointer_path",
            "checkpoints/latest_accepted_checkpoint_pointer.json",
        )?;
        ensure_exact(
            self.continuation_handoff_path.as_str(),
            "retained_paths.continuation_handoff_path",
            PSION_ACTUAL_PRETRAINING_CONTINUATION_HANDOFF_PATH,
        )?;
        ensure_exact(
            self.closeout_bundle_path.as_str(),
            "retained_paths.closeout_bundle_path",
            "closeout/closeout_bundle.json",
        )?;
        ensure_exact(
            self.launcher_log_path.as_str(),
            "retained_paths.launcher_log_path",
            "logs/launcher.log",
        )?;
        Ok(())
    }
}

impl PsionActualPretrainingLauncherContractRefs {
    /// Validates the committed contract refs.
    pub fn validate(&self) -> Result<(), PsionActualPretrainingLauncherError> {
        ensure_artifact_ref(&self.lane_spec, "contract_refs.lane_spec")?;
        ensure_artifact_ref(&self.recipe_bundle, "contract_refs.recipe_bundle")?;
        ensure_artifact_ref(
            &self.topology_storage_bundle,
            "contract_refs.topology_storage_bundle",
        )?;
        ensure_artifact_ref(&self.evidence_contract, "contract_refs.evidence_contract")?;
        Ok(())
    }
}

impl PsionActualPretrainingRunRoots {
    /// Validates the selected run roots.
    pub fn validate(&self) -> Result<(), PsionActualPretrainingLauncherError> {
        ensure_nonempty(self.local_run_root.as_str(), "run_roots.local_run_root")?;
        ensure_nonempty(self.remote_run_root.as_str(), "run_roots.remote_run_root")?;
        ensure_nonempty(
            self.remote_checkpoint_root.as_str(),
            "run_roots.remote_checkpoint_root",
        )?;
        ensure_nonempty(
            self.remote_manifest_root.as_str(),
            "run_roots.remote_manifest_root",
        )?;
        ensure_nonempty(self.remote_log_root.as_str(), "run_roots.remote_log_root")?;
        Ok(())
    }
}

impl PsionActualPretrainingLaunchManifest {
    /// Validates the actual-lane launch manifest.
    pub fn validate(&self) -> Result<(), PsionActualPretrainingLauncherError> {
        ensure_exact(
            self.schema_version.as_str(),
            "launch_manifest.schema_version",
            PSION_ACTUAL_PRETRAINING_LAUNCH_MANIFEST_SCHEMA_VERSION,
        )?;
        if self.surface_id != PSION_ACTUAL_PRETRAINING_START_SURFACE_ID
            && self.surface_id != PSION_ACTUAL_PRETRAINING_DRY_RUN_SURFACE_ID
        {
            return Err(PsionActualPretrainingLauncherError::UnsupportedValue {
                field: String::from("launch_manifest.surface_id"),
                detail: String::from(
                    "launch manifest surface_id must be the start or dry-run surface",
                ),
            });
        }
        validate_launcher_common(
            self.lane_id.as_str(),
            self.recipe_id.as_str(),
            self.topology_storage_bundle_id.as_str(),
            self.evidence_contract_id.as_str(),
            self.run_id.as_str(),
            &self.retained_paths,
            &self.launcher_surfaces,
            &self.run_roots,
            &self.contract_refs,
            self.selected_git_ref.as_str(),
            self.git_commit_sha.as_str(),
            self.dirty_tree_admission.as_str(),
            self.workspace_status_sha256.as_deref(),
            &self.credential_sources,
            self.claim_boundary.as_str(),
            self.detail.as_str(),
        )?;
        Ok(())
    }
}

impl PsionActualPretrainingResumeManifest {
    /// Validates the actual-lane resume manifest.
    pub fn validate(&self) -> Result<(), PsionActualPretrainingLauncherError> {
        ensure_exact(
            self.schema_version.as_str(),
            "resume_manifest.schema_version",
            PSION_ACTUAL_PRETRAINING_RESUME_MANIFEST_SCHEMA_VERSION,
        )?;
        ensure_exact(
            self.surface_id.as_str(),
            "resume_manifest.surface_id",
            PSION_ACTUAL_PRETRAINING_RESUME_SURFACE_ID,
        )?;
        validate_launcher_common(
            self.lane_id.as_str(),
            self.recipe_id.as_str(),
            self.topology_storage_bundle_id.as_str(),
            self.evidence_contract_id.as_str(),
            self.run_id.as_str(),
            &self.retained_paths,
            &self.launcher_surfaces,
            &self.run_roots,
            &self.contract_refs,
            self.selected_git_ref.as_str(),
            self.git_commit_sha.as_str(),
            self.dirty_tree_admission.as_str(),
            self.workspace_status_sha256.as_deref(),
            &[],
            self.claim_boundary.as_str(),
            self.detail.as_str(),
        )?;
        ensure_exact(
            self.latest_checkpoint_pointer_path.as_str(),
            "resume_manifest.latest_checkpoint_pointer_path",
            "checkpoints/latest_accepted_checkpoint_pointer.json",
        )?;
        ensure_nonempty(
            self.checkpoint_label.as_str(),
            "resume_manifest.checkpoint_label",
        )?;
        if self.optimizer_step == 0 {
            return Err(PsionActualPretrainingLauncherError::MissingField {
                field: String::from("resume_manifest.optimizer_step"),
            });
        }
        ensure_nonempty(
            self.checkpoint_ref.as_str(),
            "resume_manifest.checkpoint_ref",
        )?;
        Ok(())
    }
}

impl PsionActualPretrainingCheckpointPointer {
    /// Validates the actual-lane checkpoint pointer.
    pub fn validate(&self) -> Result<(), PsionActualPretrainingLauncherError> {
        ensure_exact(
            self.schema_version.as_str(),
            "checkpoint_pointer.schema_version",
            PSION_ACTUAL_PRETRAINING_CHECKPOINT_POINTER_SCHEMA_VERSION,
        )?;
        ensure_exact(
            self.lane_id.as_str(),
            "checkpoint_pointer.lane_id",
            PSION_ACTUAL_PRETRAINING_LANE_ID,
        )?;
        ensure_nonempty(self.run_id.as_str(), "checkpoint_pointer.run_id")?;
        ensure_nonempty(
            self.checkpoint_label.as_str(),
            "checkpoint_pointer.checkpoint_label",
        )?;
        ensure_nonempty(self.detail.as_str(), "checkpoint_pointer.detail")?;
        match self.pointer_state.as_str() {
            "pending_first_checkpoint" => {
                if self.optimizer_step != 0 {
                    return Err(PsionActualPretrainingLauncherError::UnsupportedValue {
                        field: String::from("checkpoint_pointer.optimizer_step"),
                        detail: String::from(
                            "pending_first_checkpoint pointers must retain optimizer_step 0",
                        ),
                    });
                }
                if self.checkpoint_ref.is_some() {
                    return Err(PsionActualPretrainingLauncherError::UnsupportedValue {
                        field: String::from("checkpoint_pointer.checkpoint_ref"),
                        detail: String::from(
                            "pending_first_checkpoint pointers must not retain a checkpoint ref",
                        ),
                    });
                }
                if self.checkpoint_manifest_relative_path.is_some() {
                    return Err(PsionActualPretrainingLauncherError::UnsupportedValue {
                        field: String::from("checkpoint_pointer.checkpoint_manifest_relative_path"),
                        detail: String::from(
                            "pending_first_checkpoint pointers must not retain a checkpoint manifest path",
                        ),
                    });
                }
            }
            "accepted" => {
                if self.optimizer_step == 0 {
                    return Err(PsionActualPretrainingLauncherError::MissingField {
                        field: String::from("checkpoint_pointer.optimizer_step"),
                    });
                }
                ensure_nonempty_option(
                    self.checkpoint_ref.as_deref(),
                    "checkpoint_pointer.checkpoint_ref",
                )?;
                ensure_nonempty_option(
                    self.checkpoint_manifest_relative_path.as_deref(),
                    "checkpoint_pointer.checkpoint_manifest_relative_path",
                )?;
            }
            _ => {
                return Err(PsionActualPretrainingLauncherError::UnsupportedValue {
                    field: String::from("checkpoint_pointer.pointer_state"),
                    detail: String::from(
                        "checkpoint pointer state must be pending_first_checkpoint or accepted",
                    ),
                });
            }
        }
        Ok(())
    }
}

impl PsionActualPretrainingCloseoutBundle {
    /// Validates the actual-lane provisional closeout bundle.
    pub fn validate(&self) -> Result<(), PsionActualPretrainingLauncherError> {
        ensure_exact(
            self.schema_version.as_str(),
            "closeout_bundle.schema_version",
            PSION_ACTUAL_PRETRAINING_CLOSEOUT_BUNDLE_SCHEMA_VERSION,
        )?;
        ensure_exact(
            self.lane_id.as_str(),
            "closeout_bundle.lane_id",
            PSION_ACTUAL_PRETRAINING_LANE_ID,
        )?;
        ensure_nonempty(self.run_id.as_str(), "closeout_bundle.run_id")?;
        ensure_nonempty(
            self.closeout_state.as_str(),
            "closeout_bundle.closeout_state",
        )?;
        self.retained_paths.validate()?;
        ensure_nonempty(
            self.selected_git_ref.as_str(),
            "closeout_bundle.selected_git_ref",
        )?;
        ensure_git_sha(
            self.git_commit_sha.as_str(),
            "closeout_bundle.git_commit_sha",
        )?;
        ensure_dirty_tree_admission(
            self.dirty_tree_admission.as_str(),
            self.workspace_status_sha256.as_deref(),
            "closeout_bundle",
        )?;
        ensure_nonempty(
            self.claim_boundary.as_str(),
            "closeout_bundle.claim_boundary",
        )?;
        ensure_nonempty(self.detail.as_str(), "closeout_bundle.detail")?;
        Ok(())
    }
}

fn validate_launcher_common(
    lane_id: &str,
    recipe_id: &str,
    topology_storage_bundle_id: &str,
    evidence_contract_id: &str,
    run_id: &str,
    retained_paths: &PsionActualPretrainingRetainedPathSet,
    launcher_surfaces: &PsionActualPretrainingLauncherSurfaces,
    run_roots: &PsionActualPretrainingRunRoots,
    contract_refs: &PsionActualPretrainingLauncherContractRefs,
    selected_git_ref: &str,
    git_commit_sha: &str,
    dirty_tree_admission: &str,
    workspace_status_sha256: Option<&str>,
    credential_sources: &[PsionActualPretrainingCredentialBinding],
    claim_boundary: &str,
    detail: &str,
) -> Result<(), PsionActualPretrainingLauncherError> {
    ensure_exact(lane_id, "lane_id", PSION_ACTUAL_PRETRAINING_LANE_ID)?;
    ensure_exact(recipe_id, "recipe_id", PSION_ACTUAL_PRETRAINING_RECIPE_ID)?;
    ensure_exact(
        topology_storage_bundle_id,
        "topology_storage_bundle_id",
        PSION_ACTUAL_PRETRAINING_TOPOLOGY_STORAGE_BUNDLE_ID,
    )?;
    ensure_exact(
        evidence_contract_id,
        "evidence_contract_id",
        PSION_ACTUAL_PRETRAINING_EVIDENCE_CONTRACT_ID,
    )?;
    ensure_nonempty(run_id, "run_id")?;
    retained_paths.validate()?;
    launcher_surfaces.validate().map_err(|error| {
        PsionActualPretrainingLauncherError::NestedValidation {
            field: String::from("launcher_surfaces"),
            detail: error.to_string(),
        }
    })?;
    run_roots.validate()?;
    contract_refs.validate()?;
    ensure_nonempty(selected_git_ref, "selected_git_ref")?;
    ensure_git_sha(git_commit_sha, "git_commit_sha")?;
    ensure_dirty_tree_admission(
        dirty_tree_admission,
        workspace_status_sha256,
        "dirty_tree_admission",
    )?;
    if !credential_sources.is_empty() {
        for credential in credential_sources {
            ensure_nonempty(credential.kind.as_str(), "credential_source.kind")?;
            ensure_nonempty(
                credential.source_name.as_str(),
                "credential_source.source_name",
            )?;
            ensure_nonempty(
                credential.retained_redaction.as_str(),
                "credential_source.retained_redaction",
            )?;
        }
    }
    ensure_nonempty(claim_boundary, "claim_boundary")?;
    ensure_nonempty(detail, "detail")?;
    Ok(())
}

fn ensure_artifact_ref(
    artifact: &PsionActualPretrainingArtifactRef,
    field_prefix: &str,
) -> Result<(), PsionActualPretrainingLauncherError> {
    ensure_nonempty(artifact.path.as_str(), &format!("{field_prefix}.path"))?;
    ensure_nonempty(artifact.sha256.as_str(), &format!("{field_prefix}.sha256"))?;
    Ok(())
}

fn ensure_exact(
    actual: &str,
    field: &str,
    expected: &str,
) -> Result<(), PsionActualPretrainingLauncherError> {
    if actual != expected {
        return Err(PsionActualPretrainingLauncherError::FieldMismatch {
            field: String::from(field),
            expected: String::from(expected),
            actual: String::from(actual),
        });
    }
    Ok(())
}

fn ensure_nonempty(value: &str, field: &str) -> Result<(), PsionActualPretrainingLauncherError> {
    if value.trim().is_empty() {
        return Err(PsionActualPretrainingLauncherError::MissingField {
            field: String::from(field),
        });
    }
    Ok(())
}

fn ensure_nonempty_option(
    value: Option<&str>,
    field: &str,
) -> Result<(), PsionActualPretrainingLauncherError> {
    match value {
        Some(value) => ensure_nonempty(value, field),
        None => Err(PsionActualPretrainingLauncherError::MissingField {
            field: String::from(field),
        }),
    }
}

fn ensure_git_sha(value: &str, field: &str) -> Result<(), PsionActualPretrainingLauncherError> {
    ensure_nonempty(value, field)?;
    if value.len() != 40 || !value.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Err(PsionActualPretrainingLauncherError::UnsupportedValue {
            field: String::from(field),
            detail: String::from("git commit SHA must be a 40-character hex string"),
        });
    }
    Ok(())
}

fn ensure_dirty_tree_admission(
    dirty_tree_admission: &str,
    workspace_status_sha256: Option<&str>,
    field_prefix: &str,
) -> Result<(), PsionActualPretrainingLauncherError> {
    match dirty_tree_admission {
        "refuse_by_default" => Ok(()),
        "allowed_by_operator_override" => ensure_nonempty_option(
            workspace_status_sha256,
            &format!("{field_prefix}.workspace_status_sha256"),
        ),
        _ => Err(PsionActualPretrainingLauncherError::UnsupportedValue {
            field: String::from(field_prefix),
            detail: String::from(
                "dirty-tree admission must be refuse_by_default or allowed_by_operator_override",
            ),
        }),
    }
}

/// Validation errors for the actual-lane launcher surfaces.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PsionActualPretrainingLauncherError {
    #[error("psion actual-pretraining launcher field `{field}` must not be empty")]
    MissingField { field: String },
    #[error(
        "psion actual-pretraining launcher field `{field}` mismatch: expected `{expected}`, got `{actual}`"
    )]
    FieldMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    #[error("psion actual-pretraining launcher field `{field}` is unsupported: {detail}")]
    UnsupportedValue { field: String, detail: String },
    #[error("psion actual-pretraining launcher nested validation for `{field}` failed: {detail}")]
    NestedValidation { field: String, detail: String },
}

#[cfg(test)]
mod tests {
    use super::{
        PsionActualPretrainingCheckpointPointer, PsionActualPretrainingCloseoutBundle,
        PsionActualPretrainingLaunchManifest, PsionActualPretrainingResumeManifest,
    };

    fn launch_manifest() -> PsionActualPretrainingLaunchManifest {
        serde_json::from_str(include_str!(
            "../../../fixtures/psion/pretrain/psion_actual_pretraining_launch_manifest_v1.json"
        ))
        .expect("actual pretraining launch manifest fixture should parse")
    }

    fn resume_manifest() -> PsionActualPretrainingResumeManifest {
        serde_json::from_str(include_str!(
            "../../../fixtures/psion/pretrain/psion_actual_pretraining_resume_manifest_v1.json"
        ))
        .expect("actual pretraining resume manifest fixture should parse")
    }

    fn checkpoint_pointer() -> PsionActualPretrainingCheckpointPointer {
        serde_json::from_str(include_str!(
            "../../../fixtures/psion/pretrain/psion_actual_pretraining_checkpoint_pointer_v1.json"
        ))
        .expect("actual pretraining checkpoint pointer fixture should parse")
    }

    fn closeout_bundle() -> PsionActualPretrainingCloseoutBundle {
        serde_json::from_str(include_str!(
            "../../../fixtures/psion/pretrain/psion_actual_pretraining_closeout_bundle_v1.json"
        ))
        .expect("actual pretraining closeout bundle fixture should parse")
    }

    #[test]
    fn actual_pretraining_launch_manifest_fixture_validates() {
        launch_manifest()
            .validate()
            .expect("actual pretraining launch manifest fixture should validate");
    }

    #[test]
    fn actual_pretraining_resume_manifest_fixture_validates() {
        resume_manifest()
            .validate()
            .expect("actual pretraining resume manifest fixture should validate");
    }

    #[test]
    fn actual_pretraining_checkpoint_pointer_fixture_validates() {
        checkpoint_pointer()
            .validate()
            .expect("actual pretraining checkpoint pointer fixture should validate");
    }

    #[test]
    fn actual_pretraining_closeout_bundle_fixture_validates() {
        closeout_bundle()
            .validate()
            .expect("actual pretraining closeout bundle fixture should validate");
    }

    #[test]
    fn actual_pretraining_resume_manifest_rejects_pending_pointer_step() {
        let mut manifest = resume_manifest();
        manifest.optimizer_step = 0;
        let error = manifest
            .validate()
            .expect_err("resume manifest must reject zero-step resume");
        assert_eq!(
            error,
            super::PsionActualPretrainingLauncherError::MissingField {
                field: String::from("resume_manifest.optimizer_step"),
            }
        );
    }
}
