use std::{fs, path::Path};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    PsionicTrainCheckpointSurface, PsionicTrainContributionArtifactManifest,
    PsionicTrainContributionReceipt, PsionicTrainInvocationManifest, PsionicTrainOutcomeKind,
    TrainingExecutionValidatorDisposition,
};

pub const PSIONIC_TRAIN_VALIDATOR_SCORE_ARTIFACT_SCHEMA_VERSION: &str =
    "psionic.train.validator_score_artifact.v1";
pub const PSIONIC_TRAIN_VALIDATOR_SCORE_RECEIPT_SCHEMA_VERSION: &str =
    "psionic.train.validator_score_receipt.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionicTrainValidatorReplayReasonCode {
    ContributionOutcomeRefused,
    PrimaryCheckpointAccepted,
    CheckpointRecovered,
    CheckpointReplayRequired,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainValidatorScoreArtifact {
    pub schema_version: String,
    pub lane_id: String,
    pub network_id: Option<String>,
    pub validator_run_id: String,
    pub challenged_run_id: String,
    pub window_id: String,
    pub assignment_id: String,
    pub challenge_id: String,
    pub validator_node_pubkey: String,
    pub challenged_node_pubkey: String,
    pub contribution_id: String,
    pub contribution_digest: String,
    pub artifact_manifest_digest: String,
    pub artifact_count: usize,
    pub checkpoint_pointer_state: Option<String>,
    pub checkpoint_manifest_digest: Option<String>,
    pub checkpoint_object_digest: Option<String>,
    pub disposition: TrainingExecutionValidatorDisposition,
    pub reason_codes: Vec<PsionicTrainValidatorReplayReasonCode>,
    pub score_bps: u16,
    pub detail: String,
    pub score_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainValidatorScoreReceipt {
    pub schema_version: String,
    pub lane_id: String,
    pub network_id: Option<String>,
    pub validator_run_id: String,
    pub challenged_run_id: String,
    pub window_id: String,
    pub assignment_id: String,
    pub challenge_id: String,
    pub validator_node_pubkey: String,
    pub challenged_node_pubkey: String,
    pub contribution_id: String,
    pub contribution_digest: String,
    pub artifact_manifest_digest: String,
    pub disposition: TrainingExecutionValidatorDisposition,
    pub reason_codes: Vec<PsionicTrainValidatorReplayReasonCode>,
    pub score_bps: u16,
    pub score_artifact_path: String,
    pub score_artifact_digest: String,
    pub detail: String,
    pub score_receipt_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainValidatorArtifactOutputs {
    pub validator_score_receipt_path: String,
}

#[derive(Clone, Debug)]
pub struct PsionicTrainValidatorReplayExecution {
    pub artifacts: PsionicTrainValidatorArtifactOutputs,
    pub score_receipt: PsionicTrainValidatorScoreReceipt,
    pub detail: String,
}

#[derive(Debug, Error)]
pub enum PsionicTrainValidatorReplayError {
    #[error("failed to read `{path}`: {detail}")]
    Read { path: String, detail: String },
    #[error("failed to write `{path}`: {detail}")]
    Write { path: String, detail: String },
    #[error("failed to parse `{path}`: {detail}")]
    Parse { path: String, detail: String },
    #[error("validator replay input is stale: {detail}")]
    StaleAssignment { detail: String },
    #[error("validator replay input drifted: {detail}")]
    ArtifactDigestMismatch { detail: String },
    #[error("validator replay is missing checkpoint state: {detail}")]
    CheckpointMissing { detail: String },
}

pub fn execute_psionic_train_validator_replay(
    manifest: &PsionicTrainInvocationManifest,
    run_root: &Path,
) -> Result<PsionicTrainValidatorReplayExecution, PsionicTrainValidatorReplayError> {
    let contribution_receipt_path = Path::new(
        manifest
            .validator_target_contribution_receipt_path
            .as_deref()
            .expect("validated validator manifests always carry target contribution receipt path"),
    );
    let contribution_receipt: PsionicTrainContributionReceipt =
        read_json(contribution_receipt_path)?;
    if contribution_receipt.contribution_digest != contribution_receipt.stable_contribution_digest()
    {
        return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
            detail: format!(
                "contribution receipt `{}` digest does not match its canonical contents",
                contribution_receipt_path.display()
            ),
        });
    }

    let contribution_artifact_manifest_path = Path::new(
        manifest
            .validator_target_contribution_artifact_manifest_path
            .as_deref()
            .expect(
                "validated validator manifests always carry target contribution artifact manifest path",
            ),
    );
    let contribution_artifact_manifest: PsionicTrainContributionArtifactManifest =
        read_json(contribution_artifact_manifest_path)?;
    if contribution_artifact_manifest.artifact_manifest_digest
        != contribution_artifact_manifest.stable_artifact_manifest_digest()
    {
        return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
            detail: format!(
                "contribution artifact manifest `{}` digest does not match its canonical contents",
                contribution_artifact_manifest_path.display()
            ),
        });
    }
    if contribution_receipt.artifact_manifest_path
        != contribution_artifact_manifest_path.display().to_string()
    {
        return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
            detail: format!(
                "contribution receipt references `{}` but validator replay opened `{}`",
                contribution_receipt.artifact_manifest_path,
                contribution_artifact_manifest_path.display()
            ),
        });
    }
    if contribution_receipt.artifact_manifest_digest
        != contribution_artifact_manifest.artifact_manifest_digest
    {
        return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
            detail: String::from(
                "contribution receipt artifact-manifest digest does not match the replayed manifest",
            ),
        });
    }
    if contribution_receipt.lane_id != manifest.lane_id
        || contribution_artifact_manifest.lane_id != manifest.lane_id
    {
        return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
            detail: String::from(
                "validator replay lane id does not match the challenged contribution lane",
            ),
        });
    }
    if contribution_receipt.window_id != contribution_artifact_manifest.window_id
        || contribution_receipt.assignment_id != contribution_artifact_manifest.assignment_id
        || contribution_receipt.contribution_id != contribution_artifact_manifest.contribution_id
    {
        return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
            detail: String::from(
                "challenged contribution receipt drifted from the challenged artifact manifest",
            ),
        });
    }
    if manifest.coordination.window_id.as_deref() != Some(contribution_receipt.window_id.as_str())
        || manifest.coordination.assignment_id.as_deref()
            != Some(contribution_receipt.assignment_id.as_str())
    {
        return Err(PsionicTrainValidatorReplayError::StaleAssignment {
            detail: format!(
                "validator replay targets window `{}` assignment `{}` but the challenged contribution belongs to window `{}` assignment `{}`",
                manifest.coordination.window_id.as_deref().unwrap_or(""),
                manifest.coordination.assignment_id.as_deref().unwrap_or(""),
                contribution_receipt.window_id,
                contribution_receipt.assignment_id
            ),
        });
    }

    let checkpoint_surface = load_checkpoint_surface(&contribution_artifact_manifest)?;
    let (disposition, reason_codes, score_bps, detail, checkpoint_pointer_state) =
        classify_validator_result(&contribution_receipt, checkpoint_surface.as_ref())?;

    let challenge_id = manifest
        .coordination
        .challenge_id
        .as_deref()
        .expect("validated validator manifests always carry challenge_id");
    let validator_node_pubkey = manifest
        .coordination
        .node_pubkey
        .as_deref()
        .expect("validated validator manifests always carry validator node_pubkey");
    let validator_run_id = manifest
        .run_id
        .clone()
        .or_else(|| {
            run_root
                .file_name()
                .map(|value| value.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| String::from("validator_run"));
    let validator_root = run_root
        .join("windows")
        .join(contribution_receipt.window_id.as_str())
        .join("validators")
        .join(challenge_id);
    fs::create_dir_all(&validator_root).map_err(|error| {
        PsionicTrainValidatorReplayError::Write {
            path: validator_root.display().to_string(),
            detail: error.to_string(),
        }
    })?;
    let score_artifact_path = validator_root.join("validator_score_artifact.json");
    let mut score_artifact = PsionicTrainValidatorScoreArtifact {
        schema_version: String::from(PSIONIC_TRAIN_VALIDATOR_SCORE_ARTIFACT_SCHEMA_VERSION),
        lane_id: manifest.lane_id.clone(),
        network_id: manifest.coordination.network_id.clone(),
        validator_run_id: validator_run_id.clone(),
        challenged_run_id: contribution_receipt.run_id.clone(),
        window_id: contribution_receipt.window_id.clone(),
        assignment_id: contribution_receipt.assignment_id.clone(),
        challenge_id: String::from(challenge_id),
        validator_node_pubkey: String::from(validator_node_pubkey),
        challenged_node_pubkey: contribution_receipt.node_pubkey.clone(),
        contribution_id: contribution_receipt.contribution_id.clone(),
        contribution_digest: contribution_receipt.contribution_digest.clone(),
        artifact_manifest_digest: contribution_artifact_manifest
            .artifact_manifest_digest
            .clone(),
        artifact_count: contribution_artifact_manifest.artifact_count,
        checkpoint_pointer_state,
        checkpoint_manifest_digest: checkpoint_surface
            .as_ref()
            .and_then(|value| value.checkpoint_manifest_digest.clone()),
        checkpoint_object_digest: checkpoint_surface
            .as_ref()
            .and_then(|value| value.checkpoint_object_digest.clone()),
        disposition,
        reason_codes: reason_codes.clone(),
        score_bps,
        detail: detail.clone(),
        score_digest: String::new(),
    };
    score_artifact.score_digest =
        stable_digest(b"psionic_train_validator_score_artifact|", &score_artifact);
    write_json(score_artifact_path.as_path(), &score_artifact)?;

    let score_receipt_path = validator_root.join("validator_score_receipt.json");
    let mut score_receipt = PsionicTrainValidatorScoreReceipt {
        schema_version: String::from(PSIONIC_TRAIN_VALIDATOR_SCORE_RECEIPT_SCHEMA_VERSION),
        lane_id: manifest.lane_id.clone(),
        network_id: manifest.coordination.network_id.clone(),
        validator_run_id,
        challenged_run_id: contribution_receipt.run_id.clone(),
        window_id: contribution_receipt.window_id.clone(),
        assignment_id: contribution_receipt.assignment_id.clone(),
        challenge_id: String::from(challenge_id),
        validator_node_pubkey: String::from(validator_node_pubkey),
        challenged_node_pubkey: contribution_receipt.node_pubkey.clone(),
        contribution_id: contribution_receipt.contribution_id.clone(),
        contribution_digest: contribution_receipt.contribution_digest.clone(),
        artifact_manifest_digest: contribution_artifact_manifest
            .artifact_manifest_digest
            .clone(),
        disposition,
        reason_codes,
        score_bps,
        score_artifact_path: score_artifact_path.display().to_string(),
        score_artifact_digest: score_artifact.score_digest.clone(),
        detail: detail.clone(),
        score_receipt_digest: String::new(),
    };
    score_receipt.score_receipt_digest =
        stable_digest(b"psionic_train_validator_score_receipt|", &score_receipt);
    write_json(score_receipt_path.as_path(), &score_receipt)?;

    Ok(PsionicTrainValidatorReplayExecution {
        artifacts: PsionicTrainValidatorArtifactOutputs {
            validator_score_receipt_path: score_receipt_path.display().to_string(),
        },
        score_receipt,
        detail,
    })
}

fn load_checkpoint_surface(
    contribution_artifact_manifest: &PsionicTrainContributionArtifactManifest,
) -> Result<Option<PsionicTrainCheckpointSurface>, PsionicTrainValidatorReplayError> {
    let Some(checkpoint_surface_artifact) = contribution_artifact_manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.artifact_kind == "checkpoint_surface")
    else {
        return Ok(None);
    };
    read_json(Path::new(
        checkpoint_surface_artifact.artifact_path.as_str(),
    ))
    .map(Some)
}

fn classify_validator_result(
    contribution_receipt: &PsionicTrainContributionReceipt,
    checkpoint_surface: Option<&PsionicTrainCheckpointSurface>,
) -> Result<
    (
        TrainingExecutionValidatorDisposition,
        Vec<PsionicTrainValidatorReplayReasonCode>,
        u16,
        String,
        Option<String>,
    ),
    PsionicTrainValidatorReplayError,
> {
    if contribution_receipt.outcome == PsionicTrainOutcomeKind::Refused {
        return Ok((
            TrainingExecutionValidatorDisposition::Rejected,
            vec![PsionicTrainValidatorReplayReasonCode::ContributionOutcomeRefused],
            0,
            String::from(
                "validator replay rejected the challenged contribution because the worker contribution itself ended in refused posture",
            ),
            None,
        ));
    }

    let checkpoint_surface = checkpoint_surface.ok_or_else(|| {
        PsionicTrainValidatorReplayError::CheckpointMissing {
            detail: String::from(
                "successful challenged contribution did not retain a checkpoint surface artifact for replay",
            ),
        }
    })?;
    let checkpoint_pointer_state = checkpoint_surface.pointer_state.clone();
    let clean_recovery_posture = matches!(
        checkpoint_surface.recovery_source_kind.as_deref(),
        None | Some("none")
    );
    let accepted_pointer_state = matches!(
        checkpoint_surface.pointer_state.as_deref(),
        Some("accepted") | Some("accepted_primary")
    );

    if checkpoint_surface.upload_outcome.as_deref() == Some("refused")
        || checkpoint_surface.recovery_resolution_state.as_deref() == Some("refused")
    {
        return Ok((
            TrainingExecutionValidatorDisposition::ReplayRequired,
            vec![PsionicTrainValidatorReplayReasonCode::CheckpointReplayRequired],
            5_000,
            String::from(
                "validator replay kept the challenged contribution in replay-required posture because the retained checkpoint surface still records a refused upload or refused recovery path",
            ),
            checkpoint_pointer_state,
        ));
    }

    if checkpoint_surface.restored_primary_pointer == Some(true)
        || !clean_recovery_posture
        || !accepted_pointer_state
    {
        return Ok((
            TrainingExecutionValidatorDisposition::Quarantined,
            vec![PsionicTrainValidatorReplayReasonCode::CheckpointRecovered],
            7_500,
            String::from(
                "validator replay quarantined the challenged contribution because it depends on recovered or non-primary checkpoint posture rather than a clean accepted-primary pointer",
            ),
            checkpoint_pointer_state,
        ));
    }

    Ok((
        TrainingExecutionValidatorDisposition::Accepted,
        vec![PsionicTrainValidatorReplayReasonCode::PrimaryCheckpointAccepted],
        10_000,
        String::from(
            "validator replay accepted the challenged contribution because the retained artifact manifest, contribution receipt, and accepted-primary checkpoint surface all match the declared replay target",
        ),
        checkpoint_pointer_state,
    ))
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, PsionicTrainValidatorReplayError> {
    let bytes = fs::read(path).map_err(|error| PsionicTrainValidatorReplayError::Read {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| PsionicTrainValidatorReplayError::Parse {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}

fn write_json<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), PsionicTrainValidatorReplayError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| PsionicTrainValidatorReplayError::Write {
            path: parent.display().to_string(),
            detail: error.to_string(),
        })?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        PsionicTrainValidatorReplayError::Write {
            path: path.display().to_string(),
            detail: error.to_string(),
        }
    })?;
    fs::write(path, bytes).map_err(|error| PsionicTrainValidatorReplayError::Write {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(
        serde_json::to_vec(value)
            .expect("validator replay score surfaces must serialize for stable digest"),
    );
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        PsionicTrainValidatorReplayReasonCode, TrainingExecutionValidatorDisposition,
        classify_validator_result,
    };
    use crate::{
        PSIONIC_TRAIN_CHECKPOINT_SURFACE_SCHEMA_VERSION,
        PSIONIC_TRAIN_CONTRIBUTION_RECEIPT_SCHEMA_VERSION, PsionicTrainAuthorityOwner,
        PsionicTrainCheckpointArtifactPaths, PsionicTrainCheckpointSurface,
        PsionicTrainContributionReceipt, PsionicTrainOutcomeKind, PsionicTrainRole,
    };

    fn succeeded_contribution() -> PsionicTrainContributionReceipt {
        let mut receipt = PsionicTrainContributionReceipt {
            schema_version: String::from(PSIONIC_TRAIN_CONTRIBUTION_RECEIPT_SCHEMA_VERSION),
            lane_id: String::from(crate::PSION_ACTUAL_PRETRAINING_LANE_ID),
            run_id: String::from("worker-run"),
            window_id: String::from("window-0001"),
            window_execution_id: String::from("window-execution-1"),
            assignment_id: String::from("assignment-0001"),
            contribution_id: String::from("contribution-0001"),
            node_pubkey: String::from("npub1-worker"),
            grouped_stage_assignment: None,
            role: PsionicTrainRole::Worker,
            operation: String::from("record-checkpoint"),
            outcome: PsionicTrainOutcomeKind::Succeeded,
            exit_code: 0,
            retryable: false,
            authority_owner: PsionicTrainAuthorityOwner::Pylon,
            refusal_class: None,
            artifact_manifest_path: String::from("/tmp/artifact_manifest.json"),
            artifact_manifest_digest: String::from("artifact-digest-1"),
            artifact_count: 4,
            contribution_digest: String::new(),
            detail: String::from("successful contribution"),
        };
        receipt.contribution_digest = receipt.stable_contribution_digest();
        receipt
    }

    fn refused_contribution() -> PsionicTrainContributionReceipt {
        let mut receipt = succeeded_contribution();
        receipt.outcome = PsionicTrainOutcomeKind::Refused;
        receipt.detail = String::from("refused contribution");
        receipt.contribution_digest = receipt.stable_contribution_digest();
        receipt
    }

    fn checkpoint_surface(
        pointer_state: Option<&str>,
        upload_outcome: Option<&str>,
        recovery_source_kind: Option<&str>,
        restored_primary_pointer: Option<bool>,
        recovery_resolution_state: Option<&str>,
    ) -> PsionicTrainCheckpointSurface {
        PsionicTrainCheckpointSurface {
            schema_version: String::from(PSIONIC_TRAIN_CHECKPOINT_SURFACE_SCHEMA_VERSION),
            lane_id: String::from(crate::PSION_ACTUAL_PRETRAINING_LANE_ID),
            role: PsionicTrainRole::Worker,
            operation: crate::PsionicTrainOperation::RecordCheckpoint,
            run_id: String::from("worker-run"),
            run_root: String::from("/tmp/worker-run"),
            current_phase: Some(String::from("checkpoint_evaluated")),
            pointer_state: pointer_state.map(String::from),
            checkpoint_label: Some(String::from("accepted")),
            optimizer_step: Some(4_096),
            checkpoint_ref: Some(String::from("checkpoint://psion/test")),
            checkpoint_manifest_digest: Some(String::from("manifest-digest")),
            checkpoint_object_digest: Some(String::from("object-digest")),
            checkpoint_total_bytes: Some(2_048),
            backup_state: Some(String::from("backed_up")),
            upload_outcome: upload_outcome.map(String::from),
            upload_failure_reason: None,
            recovery_resolution_state: recovery_resolution_state.map(String::from),
            recovery_source_kind: recovery_source_kind.map(String::from),
            restored_primary_pointer,
            artifacts: PsionicTrainCheckpointArtifactPaths::default(),
        }
    }

    #[test]
    fn clean_accepted_checkpoint_is_accepted() {
        let receipt = succeeded_contribution();
        let surface =
            checkpoint_surface(Some("accepted"), Some("succeeded"), None, Some(false), None);
        let (disposition, reason_codes, score_bps, _, _) =
            classify_validator_result(&receipt, Some(&surface))
                .expect("classification should work");
        assert_eq!(disposition, TrainingExecutionValidatorDisposition::Accepted);
        assert_eq!(
            reason_codes,
            vec![PsionicTrainValidatorReplayReasonCode::PrimaryCheckpointAccepted]
        );
        assert_eq!(score_bps, 10_000);
    }

    #[test]
    fn refused_upload_requires_replay() {
        let receipt = succeeded_contribution();
        let surface =
            checkpoint_surface(Some("accepted"), Some("refused"), None, Some(false), None);
        let (disposition, reason_codes, score_bps, _, _) =
            classify_validator_result(&receipt, Some(&surface))
                .expect("classification should work");
        assert_eq!(
            disposition,
            TrainingExecutionValidatorDisposition::ReplayRequired
        );
        assert_eq!(
            reason_codes,
            vec![PsionicTrainValidatorReplayReasonCode::CheckpointReplayRequired]
        );
        assert_eq!(score_bps, 5_000);
    }

    #[test]
    fn recovered_checkpoint_is_quarantined() {
        let receipt = succeeded_contribution();
        let surface = checkpoint_surface(
            Some("accepted"),
            Some("succeeded"),
            Some("backup_receipt"),
            Some(true),
            Some("restored"),
        );
        let (disposition, reason_codes, score_bps, _, _) =
            classify_validator_result(&receipt, Some(&surface))
                .expect("classification should work");
        assert_eq!(
            disposition,
            TrainingExecutionValidatorDisposition::Quarantined
        );
        assert_eq!(
            reason_codes,
            vec![PsionicTrainValidatorReplayReasonCode::CheckpointRecovered]
        );
        assert_eq!(score_bps, 7_500);
    }

    #[test]
    fn refused_worker_contribution_is_rejected_without_checkpoint() {
        let receipt = refused_contribution();
        let (disposition, reason_codes, score_bps, _, _) =
            classify_validator_result(&receipt, None).expect("classification should work");
        assert_eq!(disposition, TrainingExecutionValidatorDisposition::Rejected);
        assert_eq!(
            reason_codes,
            vec![PsionicTrainValidatorReplayReasonCode::ContributionOutcomeRefused]
        );
        assert_eq!(score_bps, 0);
    }
}
