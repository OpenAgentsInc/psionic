use std::{fs, path::Path};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{
    PSION_ACTUAL_PRETRAINING_LANE_ID, PsionActualPretrainingAutoResumeReceipt,
    PsionActualPretrainingCheckpointBackupReceipt, PsionActualPretrainingCheckpointManifest,
    PsionActualPretrainingCheckpointPointer, PsionActualPretrainingCurrentRunStatus,
    PsionicTrainOperation, PsionicTrainRole,
};

/// Stable schema version for the retained machine-readable checkpoint surface.
pub const PSIONIC_TRAIN_CHECKPOINT_SURFACE_SCHEMA_VERSION: &str =
    "psionic.train.checkpoint_surface.v1";

/// Shared retained artifact paths for the latest local checkpoint state.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainCheckpointArtifactPaths {
    /// Absolute path to the canonical latest-checkpoint pointer when present.
    pub checkpoint_pointer_path: Option<String>,
    /// Absolute path to the canonical latest checkpoint manifest when present.
    pub checkpoint_manifest_path: Option<String>,
    /// Absolute path to the latest durable-backup receipt when present.
    pub checkpoint_backup_receipt_path: Option<String>,
    /// Absolute path to the latest backup pointer copy when present.
    pub checkpoint_backup_pointer_path: Option<String>,
    /// Absolute path to the latest backup manifest copy when present.
    pub checkpoint_backup_manifest_path: Option<String>,
    /// Absolute path to the latest auto-resume receipt when present.
    pub auto_resume_receipt_path: Option<String>,
    /// Absolute path to the latest retained checkpoint failure drill when present.
    pub checkpoint_failure_drill_path: Option<String>,
}

/// Machine-readable summary of the retained checkpoint and recovery state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainCheckpointSurface {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable lane identifier.
    pub lane_id: String,
    /// Runtime role that produced the latest state transition.
    pub role: PsionicTrainRole,
    /// Runtime operation that produced the latest state transition.
    pub operation: PsionicTrainOperation,
    /// Stable run identifier.
    pub run_id: String,
    /// Absolute local run root.
    pub run_root: String,
    /// Latest retained actual-lane phase when present.
    pub current_phase: Option<String>,
    /// Pointer state for the latest checkpoint pointer when present.
    pub pointer_state: Option<String>,
    /// Latest checkpoint label when present.
    pub checkpoint_label: Option<String>,
    /// Latest checkpoint step when present.
    pub optimizer_step: Option<u64>,
    /// Latest checkpoint ref when present.
    pub checkpoint_ref: Option<String>,
    /// Stable checkpoint-manifest digest when present.
    pub checkpoint_manifest_digest: Option<String>,
    /// Stable checkpoint-object digest when present.
    pub checkpoint_object_digest: Option<String>,
    /// Stable checkpoint byte count when present.
    pub checkpoint_total_bytes: Option<u64>,
    /// Backup state when the durable-backup receipt exists.
    pub backup_state: Option<String>,
    /// Upload outcome when the durable-backup receipt exists.
    pub upload_outcome: Option<String>,
    /// Upload failure reason when the durable-backup receipt was refused.
    pub upload_failure_reason: Option<String>,
    /// Auto-resume resolution state when the recovery receipt exists.
    pub recovery_resolution_state: Option<String>,
    /// Auto-resume source kind when the recovery receipt exists.
    pub recovery_source_kind: Option<String>,
    /// Whether the recovery receipt restored the primary pointer.
    pub restored_primary_pointer: Option<bool>,
    /// Shared retained artifact paths for the latest checkpoint family.
    pub artifacts: PsionicTrainCheckpointArtifactPaths,
}

/// Errors while deriving the retained checkpoint surface from a run root.
#[derive(Debug, Error)]
pub enum PsionicTrainCheckpointSurfaceError {
    #[error("failed to read retained checkpoint artifact `{path}`: {detail}")]
    Read { path: String, detail: String },
    #[error("failed to parse retained checkpoint artifact `{path}`: {detail}")]
    Parse { path: String, detail: String },
    #[error("retained checkpoint artifact `{path}` is invalid: {detail}")]
    Invalid { path: String, detail: String },
}

/// Loads the latest retained checkpoint surface from a run root when checkpoint state exists.
pub fn inspect_psionic_train_checkpoint_surface(
    run_root: &Path,
    role: PsionicTrainRole,
    operation: PsionicTrainOperation,
) -> Result<Option<PsionicTrainCheckpointSurface>, PsionicTrainCheckpointSurfaceError> {
    let current_status_path = run_root.join("status/current_run_status.json");
    let checkpoint_pointer_path =
        run_root.join("checkpoints/latest_accepted_checkpoint_pointer.json");
    let checkpoint_backup_receipt_path =
        run_root.join("checkpoints/latest_accepted_checkpoint_backup_receipt.json");
    let auto_resume_receipt_path = run_root.join("checkpoints/auto_resume_receipt.json");

    let current_status: Option<PsionActualPretrainingCurrentRunStatus> =
        load_optional_json(current_status_path.as_path())?;
    if let Some(current_status) = &current_status {
        current_status
            .validate()
            .map_err(|error| PsionicTrainCheckpointSurfaceError::Invalid {
                path: current_status_path.display().to_string(),
                detail: error.to_string(),
            })?;
    }

    let checkpoint_pointer: Option<PsionActualPretrainingCheckpointPointer> =
        load_optional_json(checkpoint_pointer_path.as_path())?;
    if let Some(checkpoint_pointer) = &checkpoint_pointer {
        checkpoint_pointer.validate().map_err(|error| {
            PsionicTrainCheckpointSurfaceError::Invalid {
                path: checkpoint_pointer_path.display().to_string(),
                detail: error.to_string(),
            }
        })?;
    }

    let checkpoint_manifest_path = checkpoint_pointer
        .as_ref()
        .and_then(|value| value.checkpoint_manifest_relative_path.as_ref())
        .map(|value| run_root.join(value));
    let checkpoint_manifest: Option<PsionActualPretrainingCheckpointManifest> =
        match checkpoint_manifest_path.as_ref() {
            Some(path) => {
                let manifest: PsionActualPretrainingCheckpointManifest = load_json(path.as_path())?;
                manifest.validate().map_err(|error| {
                    PsionicTrainCheckpointSurfaceError::Invalid {
                        path: path.display().to_string(),
                        detail: error.to_string(),
                    }
                })?;
                Some(manifest)
            }
            None => None,
        };

    let checkpoint_backup_receipt: Option<PsionActualPretrainingCheckpointBackupReceipt> =
        load_optional_json(checkpoint_backup_receipt_path.as_path())?;
    if let Some(checkpoint_backup_receipt) = &checkpoint_backup_receipt {
        checkpoint_backup_receipt.validate().map_err(|error| {
            PsionicTrainCheckpointSurfaceError::Invalid {
                path: checkpoint_backup_receipt_path.display().to_string(),
                detail: error.to_string(),
            }
        })?;
    }

    let checkpoint_backup_pointer_path = checkpoint_backup_receipt
        .as_ref()
        .map(|receipt| run_root.join(receipt.backup_pointer.path.as_str()));
    let checkpoint_backup_manifest_path = checkpoint_backup_receipt
        .as_ref()
        .map(|receipt| run_root.join(receipt.backup_checkpoint_manifest.path.as_str()));

    let auto_resume_receipt: Option<PsionActualPretrainingAutoResumeReceipt> =
        load_optional_json(auto_resume_receipt_path.as_path())?;
    if let Some(auto_resume_receipt) = &auto_resume_receipt {
        auto_resume_receipt.validate().map_err(|error| {
            PsionicTrainCheckpointSurfaceError::Invalid {
                path: auto_resume_receipt_path.display().to_string(),
                detail: error.to_string(),
            }
        })?;
    }

    let checkpoint_failure_drill_path = checkpoint_failure_drill_path(run_root, operation);

    if current_status.is_none()
        && checkpoint_pointer.is_none()
        && checkpoint_backup_receipt.is_none()
        && auto_resume_receipt.is_none()
    {
        return Ok(None);
    }

    let run_id = current_status
        .as_ref()
        .map(|value| value.run_id.clone())
        .or_else(|| {
            checkpoint_pointer
                .as_ref()
                .map(|value| value.run_id.clone())
        })
        .or_else(|| {
            checkpoint_backup_receipt
                .as_ref()
                .map(|value| value.run_id.clone())
        })
        .or_else(|| {
            auto_resume_receipt
                .as_ref()
                .map(|value| value.run_id.clone())
        })
        .unwrap_or_else(|| String::from("unknown_run"));

    let checkpoint_label = checkpoint_pointer
        .as_ref()
        .map(|value| value.checkpoint_label.clone())
        .or_else(|| {
            checkpoint_backup_receipt
                .as_ref()
                .map(|value| value.checkpoint_label.clone())
        })
        .or_else(|| {
            auto_resume_receipt
                .as_ref()
                .and_then(|value| value.chosen_checkpoint_label.clone())
        });

    let optimizer_step = checkpoint_pointer
        .as_ref()
        .map(|value| value.optimizer_step)
        .filter(|value| *value > 0)
        .or_else(|| {
            checkpoint_backup_receipt
                .as_ref()
                .map(|value| value.optimizer_step)
        })
        .or_else(|| {
            auto_resume_receipt
                .as_ref()
                .and_then(|value| value.chosen_optimizer_step)
        });

    let checkpoint_ref = checkpoint_pointer
        .as_ref()
        .and_then(|value| value.checkpoint_ref.clone())
        .or_else(|| {
            checkpoint_backup_receipt
                .as_ref()
                .map(|value| value.checkpoint_ref.clone())
        })
        .or_else(|| {
            auto_resume_receipt
                .as_ref()
                .and_then(|value| value.chosen_checkpoint_ref.clone())
        });

    Ok(Some(PsionicTrainCheckpointSurface {
        schema_version: String::from(PSIONIC_TRAIN_CHECKPOINT_SURFACE_SCHEMA_VERSION),
        lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
        role,
        operation,
        run_id,
        run_root: run_root.display().to_string(),
        current_phase: current_status.as_ref().map(|value| value.phase.clone()),
        pointer_state: checkpoint_pointer
            .as_ref()
            .map(|value| value.pointer_state.clone()),
        checkpoint_label,
        optimizer_step,
        checkpoint_ref,
        checkpoint_manifest_digest: checkpoint_manifest
            .as_ref()
            .map(|value| value.manifest_digest.clone()),
        checkpoint_object_digest: checkpoint_manifest
            .as_ref()
            .map(|value| value.checkpoint_object_digest.clone()),
        checkpoint_total_bytes: checkpoint_manifest
            .as_ref()
            .map(|value| value.checkpoint_total_bytes),
        backup_state: checkpoint_backup_receipt
            .as_ref()
            .map(|value| value.backup_state.clone()),
        upload_outcome: checkpoint_backup_receipt
            .as_ref()
            .map(|value| value.upload_outcome.clone()),
        upload_failure_reason: checkpoint_backup_receipt
            .as_ref()
            .and_then(|value| value.upload_failure_reason.clone()),
        recovery_resolution_state: auto_resume_receipt
            .as_ref()
            .map(|value| value.resolution_state.clone()),
        recovery_source_kind: auto_resume_receipt
            .as_ref()
            .map(|value| value.resume_source_kind.clone()),
        restored_primary_pointer: auto_resume_receipt
            .as_ref()
            .map(|value| value.restored_primary_pointer),
        artifacts: PsionicTrainCheckpointArtifactPaths {
            checkpoint_pointer_path: checkpoint_pointer_path
                .is_file()
                .then(|| checkpoint_pointer_path.display().to_string()),
            checkpoint_manifest_path: checkpoint_manifest_path
                .as_ref()
                .filter(|value| value.is_file())
                .map(|value| value.display().to_string()),
            checkpoint_backup_receipt_path: checkpoint_backup_receipt_path
                .is_file()
                .then(|| checkpoint_backup_receipt_path.display().to_string()),
            checkpoint_backup_pointer_path: checkpoint_backup_pointer_path
                .as_ref()
                .filter(|value| value.is_file())
                .map(|value| value.display().to_string()),
            checkpoint_backup_manifest_path: checkpoint_backup_manifest_path
                .as_ref()
                .filter(|value| value.is_file())
                .map(|value| value.display().to_string()),
            auto_resume_receipt_path: auto_resume_receipt_path
                .is_file()
                .then(|| auto_resume_receipt_path.display().to_string()),
            checkpoint_failure_drill_path: checkpoint_failure_drill_path
                .as_ref()
                .filter(|value| value.is_file())
                .map(|value| value.display().to_string()),
        },
    }))
}

fn checkpoint_failure_drill_path(
    run_root: &Path,
    operation: PsionicTrainOperation,
) -> Option<std::path::PathBuf> {
    let candidates: &[&str] = match operation {
        PsionicTrainOperation::Backup => &[
            "checkpoints/failures/failed_upload_drill.json",
            "checkpoints/failures/corrupt_pointer_drill.json",
            "checkpoints/failures/stale_pointer_drill.json",
        ],
        PsionicTrainOperation::Resume => &[
            "checkpoints/failures/corrupt_pointer_drill.json",
            "checkpoints/failures/stale_pointer_drill.json",
            "checkpoints/failures/failed_upload_drill.json",
        ],
        _ => &[
            "checkpoints/failures/failed_upload_drill.json",
            "checkpoints/failures/corrupt_pointer_drill.json",
            "checkpoints/failures/stale_pointer_drill.json",
        ],
    };
    candidates
        .iter()
        .map(|value| run_root.join(value))
        .find(|value| value.is_file())
}

fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T, PsionicTrainCheckpointSurfaceError> {
    let bytes = fs::read(path).map_err(|error| PsionicTrainCheckpointSurfaceError::Read {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| PsionicTrainCheckpointSurfaceError::Parse {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}

fn load_optional_json<T: DeserializeOwned>(
    path: &Path,
) -> Result<Option<T>, PsionicTrainCheckpointSurfaceError> {
    if !path.is_file() {
        return Ok(None);
    }
    load_json(path).map(Some)
}
