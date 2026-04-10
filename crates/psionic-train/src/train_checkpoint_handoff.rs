use std::{fs, path::Path};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    inspect_psionic_train_checkpoint_surface, PsionActualPretrainingCheckpointManifest,
    PsionActualPretrainingCheckpointPointer, PsionicTrainOperation, PsionicTrainRole,
    PSION_ACTUAL_PRETRAINING_LANE_ID,
};

/// Stable schema version for the machine-readable peer checkpoint handoff receipt.
pub const PSIONIC_TRAIN_CHECKPOINT_HANDOFF_RECEIPT_SCHEMA_VERSION: &str =
    "psionic.train.checkpoint_handoff_receipt.v1";

/// Canonical retained current handoff path under a run root.
pub const PSIONIC_TRAIN_CHECKPOINT_HANDOFF_RECEIPT_RELATIVE_PATH: &str =
    "status/peer_checkpoint_handoff_receipt.json";

/// Canonical retained handoff history directory under a run root.
pub const PSIONIC_TRAIN_CHECKPOINT_HANDOFF_HISTORY_RELATIVE_DIR: &str =
    "status/peer_checkpoint_handoffs";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionicTrainCheckpointHandoffSourceKind {
    LivePrimaryPointer,
    DurableBackupCopy,
}

/// Typed receipt describing one peer-readable checkpoint handoff source.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainCheckpointHandoffReceipt {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable lane identifier.
    pub lane_id: String,
    /// Admitted node pubkey serving the checkpoint state.
    pub serving_node_pubkey: String,
    /// Admitted node pubkey the handoff targets.
    pub peer_node_pubkey: String,
    /// Stable source run identifier.
    pub source_run_id: String,
    /// Absolute source run root.
    pub source_run_root: String,
    /// Whether the source is the live primary pointer or a durable backup copy.
    pub source_kind: PsionicTrainCheckpointHandoffSourceKind,
    /// Latest checkpoint label.
    pub checkpoint_label: String,
    /// Latest checkpoint step.
    pub optimizer_step: u64,
    /// Latest checkpoint ref.
    pub checkpoint_ref: String,
    /// Stable checkpoint-manifest digest.
    pub checkpoint_manifest_digest: String,
    /// Stable checkpoint-object digest.
    pub checkpoint_object_digest: String,
    /// Stable checkpoint byte count.
    pub checkpoint_total_bytes: u64,
    /// Absolute path to the source checkpoint pointer.
    pub source_checkpoint_pointer_path: String,
    /// Absolute path to the source checkpoint manifest.
    pub source_checkpoint_manifest_path: String,
    /// Whether the source handoff had to fall back to the durable backup copy.
    pub restored_from_backup: bool,
    /// Short claim boundary.
    pub detail: String,
    /// Stable digest over the receipt payload.
    pub receipt_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PsionicTrainCheckpointHandoffMaterialization {
    pub local_receipt_path: String,
    pub local_checkpoint_pointer_path: String,
    pub local_checkpoint_manifest_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PsionicTrainCheckpointHandoffRetention {
    pub current_receipt_path: String,
    pub history_receipt_path: String,
}

#[derive(Debug, Error)]
pub enum PsionicTrainCheckpointHandoffError {
    #[error("checkpoint handoff is unavailable: {detail}")]
    MissingCheckpoint { detail: String },
    #[error("checkpoint handoff is invalid: {detail}")]
    Invalid { detail: String },
    #[error("failed to read `{path}`: {detail}")]
    Read { path: String, detail: String },
    #[error("failed to parse `{path}`: {detail}")]
    Parse { path: String, detail: String },
    #[error("failed to write `{path}`: {detail}")]
    Write { path: String, detail: String },
}

impl PsionicTrainCheckpointHandoffReceipt {
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.receipt_digest.clear();
        stable_digest(&clone)
    }

    pub fn validate(&self) -> Result<(), PsionicTrainCheckpointHandoffError> {
        if self.schema_version != PSIONIC_TRAIN_CHECKPOINT_HANDOFF_RECEIPT_SCHEMA_VERSION {
            return Err(PsionicTrainCheckpointHandoffError::Invalid {
                detail: format!(
                    "checkpoint handoff schema version must stay `{}` but was `{}`",
                    PSIONIC_TRAIN_CHECKPOINT_HANDOFF_RECEIPT_SCHEMA_VERSION, self.schema_version
                ),
            });
        }
        if self.lane_id != PSION_ACTUAL_PRETRAINING_LANE_ID {
            return Err(PsionicTrainCheckpointHandoffError::Invalid {
                detail: format!(
                    "checkpoint handoff lane must stay `{}` but was `{}`",
                    PSION_ACTUAL_PRETRAINING_LANE_ID, self.lane_id
                ),
            });
        }
        require_nonempty(
            self.serving_node_pubkey.as_str(),
            "checkpoint handoff serving_node_pubkey",
        )?;
        require_nonempty(
            self.peer_node_pubkey.as_str(),
            "checkpoint handoff peer_node_pubkey",
        )?;
        require_nonempty(
            self.source_run_id.as_str(),
            "checkpoint handoff source_run_id",
        )?;
        require_nonempty(
            self.source_run_root.as_str(),
            "checkpoint handoff source_run_root",
        )?;
        require_nonempty(
            self.checkpoint_label.as_str(),
            "checkpoint handoff checkpoint_label",
        )?;
        if self.optimizer_step == 0 {
            return Err(PsionicTrainCheckpointHandoffError::Invalid {
                detail: String::from("checkpoint handoff optimizer_step must be non-zero"),
            });
        }
        require_nonempty(
            self.checkpoint_ref.as_str(),
            "checkpoint handoff checkpoint_ref",
        )?;
        require_nonempty(
            self.checkpoint_manifest_digest.as_str(),
            "checkpoint handoff checkpoint_manifest_digest",
        )?;
        require_nonempty(
            self.checkpoint_object_digest.as_str(),
            "checkpoint handoff checkpoint_object_digest",
        )?;
        if self.checkpoint_total_bytes == 0 {
            return Err(PsionicTrainCheckpointHandoffError::Invalid {
                detail: String::from("checkpoint handoff checkpoint_total_bytes must be non-zero"),
            });
        }
        require_nonempty(
            self.source_checkpoint_pointer_path.as_str(),
            "checkpoint handoff source_checkpoint_pointer_path",
        )?;
        require_nonempty(
            self.source_checkpoint_manifest_path.as_str(),
            "checkpoint handoff source_checkpoint_manifest_path",
        )?;
        require_nonempty(self.detail.as_str(), "checkpoint handoff detail")?;
        if self.receipt_digest != self.stable_digest() {
            return Err(PsionicTrainCheckpointHandoffError::Invalid {
                detail: String::from("checkpoint handoff receipt digest drifted"),
            });
        }
        Ok(())
    }
}

pub fn build_psionic_train_checkpoint_handoff_receipt(
    run_root: &Path,
    serving_node_pubkey: &str,
    peer_node_pubkey: &str,
) -> Result<PsionicTrainCheckpointHandoffReceipt, PsionicTrainCheckpointHandoffError> {
    let checkpoint_surface = inspect_psionic_train_checkpoint_surface(
        run_root,
        PsionicTrainRole::RecoverySource,
        PsionicTrainOperation::ServeCheckpoint,
    )
    .map_err(|error| PsionicTrainCheckpointHandoffError::Invalid {
        detail: error.to_string(),
    })?
    .ok_or_else(|| PsionicTrainCheckpointHandoffError::MissingCheckpoint {
        detail: String::from(
            "checkpoint handoff requires retained checkpoint state under the source run root",
        ),
    })?;

    let checkpoint_label = checkpoint_surface.checkpoint_label.clone().ok_or_else(|| {
        PsionicTrainCheckpointHandoffError::MissingCheckpoint {
            detail: String::from(
                "checkpoint handoff requires an accepted checkpoint label from the source run root",
            ),
        }
    })?;
    let optimizer_step = checkpoint_surface.optimizer_step.ok_or_else(|| {
        PsionicTrainCheckpointHandoffError::MissingCheckpoint {
            detail: String::from(
                "checkpoint handoff requires an accepted checkpoint step from the source run root",
            ),
        }
    })?;
    let checkpoint_ref = checkpoint_surface.checkpoint_ref.clone().ok_or_else(|| {
        PsionicTrainCheckpointHandoffError::MissingCheckpoint {
            detail: String::from(
                "checkpoint handoff requires an accepted checkpoint ref from the source run root",
            ),
        }
    })?;
    let checkpoint_manifest_digest = checkpoint_surface
        .checkpoint_manifest_digest
        .clone()
        .ok_or_else(|| PsionicTrainCheckpointHandoffError::MissingCheckpoint {
            detail: String::from(
                "checkpoint handoff requires an accepted checkpoint manifest digest",
            ),
        })?;
    let checkpoint_object_digest = checkpoint_surface
        .checkpoint_object_digest
        .clone()
        .ok_or_else(|| PsionicTrainCheckpointHandoffError::MissingCheckpoint {
            detail: String::from(
                "checkpoint handoff requires an accepted checkpoint object digest",
            ),
        })?;
    let checkpoint_total_bytes = checkpoint_surface.checkpoint_total_bytes.ok_or_else(|| {
        PsionicTrainCheckpointHandoffError::MissingCheckpoint {
            detail: String::from("checkpoint handoff requires an accepted checkpoint byte count"),
        }
    })?;

    let (
        source_kind,
        source_checkpoint_pointer_path,
        source_checkpoint_manifest_path,
        detail,
        restored_from_backup,
    ) = if checkpoint_surface.pointer_state.as_deref() == Some("accepted") {
        (
                PsionicTrainCheckpointHandoffSourceKind::LivePrimaryPointer,
                checkpoint_surface
                    .artifacts
                    .checkpoint_pointer_path
                    .clone()
                    .ok_or_else(|| PsionicTrainCheckpointHandoffError::MissingCheckpoint {
                        detail: String::from(
                            "checkpoint handoff requires the live primary pointer path",
                        ),
                    })?,
                checkpoint_surface
                    .artifacts
                    .checkpoint_manifest_path
                    .clone()
                    .ok_or_else(|| PsionicTrainCheckpointHandoffError::MissingCheckpoint {
                        detail: String::from(
                            "checkpoint handoff requires the live checkpoint manifest path",
                        ),
                    })?,
                String::from(
                    "Recovery-source handoff points the joiner at the live accepted checkpoint pointer and manifest from the serving run root.",
                ),
                false,
            )
    } else if checkpoint_surface.backup_state.as_deref() == Some("backed_up") {
        (
                PsionicTrainCheckpointHandoffSourceKind::DurableBackupCopy,
                checkpoint_surface
                    .artifacts
                    .checkpoint_backup_pointer_path
                    .clone()
                    .ok_or_else(|| PsionicTrainCheckpointHandoffError::MissingCheckpoint {
                        detail: String::from(
                            "checkpoint handoff requires the durable backup pointer path",
                        ),
                    })?,
                checkpoint_surface
                    .artifacts
                    .checkpoint_backup_manifest_path
                    .clone()
                    .ok_or_else(|| PsionicTrainCheckpointHandoffError::MissingCheckpoint {
                        detail: String::from(
                            "checkpoint handoff requires the durable backup manifest path",
                        ),
                    })?,
                String::from(
                    "Recovery-source handoff falls back to the durable backup checkpoint copy because the live primary pointer is not presently usable.",
                ),
                true,
            )
    } else {
        return Err(PsionicTrainCheckpointHandoffError::MissingCheckpoint {
                detail: String::from(
                    "checkpoint handoff requires an accepted primary checkpoint or a durable backup copy",
                ),
            });
    };

    let mut receipt = PsionicTrainCheckpointHandoffReceipt {
        schema_version: String::from(PSIONIC_TRAIN_CHECKPOINT_HANDOFF_RECEIPT_SCHEMA_VERSION),
        lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
        serving_node_pubkey: String::from(serving_node_pubkey),
        peer_node_pubkey: String::from(peer_node_pubkey),
        source_run_id: checkpoint_surface.run_id,
        source_run_root: run_root.display().to_string(),
        source_kind,
        checkpoint_label,
        optimizer_step,
        checkpoint_ref,
        checkpoint_manifest_digest,
        checkpoint_object_digest,
        checkpoint_total_bytes,
        source_checkpoint_pointer_path,
        source_checkpoint_manifest_path,
        restored_from_backup,
        detail,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = receipt.stable_digest();
    receipt.validate()?;
    Ok(receipt)
}

pub fn materialize_psionic_train_checkpoint_handoff(
    joiner_run_root: &Path,
    handoff_receipt_path: &Path,
    joiner_node_pubkey: &str,
) -> Result<PsionicTrainCheckpointHandoffMaterialization, PsionicTrainCheckpointHandoffError> {
    let receipt: PsionicTrainCheckpointHandoffReceipt = load_json(handoff_receipt_path)?;
    receipt.validate()?;
    if receipt.peer_node_pubkey != joiner_node_pubkey {
        return Err(PsionicTrainCheckpointHandoffError::Invalid {
            detail: format!(
                "checkpoint handoff receipt targets peer `{}` but the joiner is `{}`",
                receipt.peer_node_pubkey, joiner_node_pubkey
            ),
        });
    }

    let checkpoint_pointer: PsionActualPretrainingCheckpointPointer =
        load_json(Path::new(receipt.source_checkpoint_pointer_path.as_str()))?;
    checkpoint_pointer
        .validate()
        .map_err(|error| PsionicTrainCheckpointHandoffError::Invalid {
            detail: format!("source checkpoint pointer is invalid: {error}"),
        })?;
    let checkpoint_manifest: PsionActualPretrainingCheckpointManifest =
        load_json(Path::new(receipt.source_checkpoint_manifest_path.as_str()))?;
    checkpoint_manifest.validate().map_err(|error| {
        PsionicTrainCheckpointHandoffError::Invalid {
            detail: format!("source checkpoint manifest is invalid: {error}"),
        }
    })?;

    if checkpoint_pointer.run_id != receipt.source_run_id
        || checkpoint_manifest.run_id != receipt.source_run_id
        || checkpoint_pointer
            .checkpoint_manifest_relative_path
            .as_deref()
            != Some(checkpoint_manifest.relative_manifest_path.as_str())
        || checkpoint_manifest.manifest_digest != receipt.checkpoint_manifest_digest
        || checkpoint_manifest.checkpoint_object_digest != receipt.checkpoint_object_digest
        || checkpoint_manifest.checkpoint_total_bytes != receipt.checkpoint_total_bytes
    {
        return Err(PsionicTrainCheckpointHandoffError::Invalid {
            detail: String::from("checkpoint handoff receipt drifted from the source checkpoint"),
        });
    }

    let local_pointer_path =
        joiner_run_root.join("checkpoints/latest_accepted_checkpoint_pointer.json");
    let local_manifest_path = joiner_run_root.join(&checkpoint_manifest.relative_manifest_path);
    let retention = retain_psionic_train_checkpoint_handoff_receipt(joiner_run_root, &receipt)?;

    write_json(local_pointer_path.as_path(), &checkpoint_pointer)?;
    write_json(local_manifest_path.as_path(), &checkpoint_manifest)?;

    Ok(PsionicTrainCheckpointHandoffMaterialization {
        local_receipt_path: retention.current_receipt_path,
        local_checkpoint_pointer_path: local_pointer_path.display().to_string(),
        local_checkpoint_manifest_path: local_manifest_path.display().to_string(),
    })
}

pub fn retain_psionic_train_checkpoint_handoff_receipt(
    run_root: &Path,
    receipt: &PsionicTrainCheckpointHandoffReceipt,
) -> Result<PsionicTrainCheckpointHandoffRetention, PsionicTrainCheckpointHandoffError> {
    receipt.validate()?;
    fs::create_dir_all(run_root.join("status")).map_err(|error| {
        PsionicTrainCheckpointHandoffError::Write {
            path: run_root.join("status").display().to_string(),
            detail: error.to_string(),
        }
    })?;
    fs::create_dir_all(run_root.join(PSIONIC_TRAIN_CHECKPOINT_HANDOFF_HISTORY_RELATIVE_DIR))
        .map_err(|error| PsionicTrainCheckpointHandoffError::Write {
            path: run_root
                .join(PSIONIC_TRAIN_CHECKPOINT_HANDOFF_HISTORY_RELATIVE_DIR)
                .display()
                .to_string(),
            detail: error.to_string(),
        })?;

    let current_receipt_path =
        run_root.join(PSIONIC_TRAIN_CHECKPOINT_HANDOFF_RECEIPT_RELATIVE_PATH);
    let history_receipt_path = run_root.join(format!(
        "{}/step-{:06}-{}.json",
        PSIONIC_TRAIN_CHECKPOINT_HANDOFF_HISTORY_RELATIVE_DIR,
        receipt.optimizer_step,
        short_digest(receipt.receipt_digest.as_str()),
    ));
    write_json(current_receipt_path.as_path(), receipt)?;
    write_json(history_receipt_path.as_path(), receipt)?;

    Ok(PsionicTrainCheckpointHandoffRetention {
        current_receipt_path: current_receipt_path.display().to_string(),
        history_receipt_path: history_receipt_path.display().to_string(),
    })
}

fn require_nonempty(value: &str, field: &str) -> Result<(), PsionicTrainCheckpointHandoffError> {
    if value.trim().is_empty() {
        return Err(PsionicTrainCheckpointHandoffError::Invalid {
            detail: format!("{field} must not be empty"),
        });
    }
    Ok(())
}

fn short_digest(value: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(value.as_bytes());
    let hex = format!("{:x}", digest.finalize());
    hex[..12].to_string()
}

fn stable_digest(receipt: &PsionicTrainCheckpointHandoffReceipt) -> String {
    let encoded = serde_json::to_vec(receipt)
        .expect("checkpoint handoff receipt serialization should not fail");
    let mut digest = Sha256::new();
    digest.update(b"psionic_train_checkpoint_handoff|");
    digest.update(&encoded);
    format!("{:x}", digest.finalize())
}

fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T, PsionicTrainCheckpointHandoffError> {
    let bytes = fs::read(path).map_err(|error| PsionicTrainCheckpointHandoffError::Read {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| PsionicTrainCheckpointHandoffError::Parse {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}

fn write_json<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), PsionicTrainCheckpointHandoffError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| PsionicTrainCheckpointHandoffError::Write {
            path: parent.display().to_string(),
            detail: error.to_string(),
        })?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|error| {
            PsionicTrainCheckpointHandoffError::Write {
                path: path.display().to_string(),
                detail: error.to_string(),
            }
        })?,
    )
    .map_err(|error| PsionicTrainCheckpointHandoffError::Write {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}
