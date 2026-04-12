use std::{fs, path::Path};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    PsionicTrainCheckpointSurface, PsionicTrainContributionArtifact,
    PsionicTrainContributionArtifactManifest, PsionicTrainContributionReceipt,
    PsionicTrainGroupedReplicaEvidenceError, PsionicTrainGroupedReplicaStageExecutionSummary,
    PsionicTrainGroupedReplicaStageReplayEvidence, PsionicTrainInvocationManifest,
    PsionicTrainOutcomeKind, TrainingExecutionValidatorDisposition,
    load_psionic_train_grouped_stage_execution_summary, load_psionic_train_grouped_stage_transport,
    persist_psionic_train_grouped_stage_replay_evidence,
};

pub const PSIONIC_TRAIN_VALIDATOR_SCORE_ARTIFACT_SCHEMA_VERSION: &str =
    "psionic.train.validator_score_artifact.v1";
pub const PSIONIC_TRAIN_VALIDATOR_SCORE_RECEIPT_SCHEMA_VERSION: &str =
    "psionic.train.validator_score_receipt.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionicTrainValidatorReplayReasonCode {
    ContributionOutcomeRefused,
    PrimaryCheckpointAccepted,
    CheckpointRecovered,
    CheckpointReplayRequired,
    GroupedStageEvidenceVerified,
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
    pub grouped_stage_execution_summary_path: Option<String>,
    pub grouped_stage_execution_summary_digest: Option<String>,
    pub grouped_stage_replay_evidence_path: Option<String>,
    pub grouped_stage_replay_evidence_digest: Option<String>,
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
    pub grouped_stage_execution_summary_path: Option<String>,
    pub grouped_stage_execution_summary_digest: Option<String>,
    pub grouped_stage_replay_evidence_path: Option<String>,
    pub grouped_stage_replay_evidence_digest: Option<String>,
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
    pub grouped_stage_replay_evidence_path: Option<String>,
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
    let grouped_stage_execution_summary = load_grouped_stage_execution_summary(
        manifest,
        &contribution_receipt,
        &contribution_artifact_manifest,
    )?;
    let grouped_stage_execution_summary_path = grouped_stage_execution_summary
        .as_ref()
        .map(|_| {
            require_artifact(
                &contribution_artifact_manifest,
                "grouped_stage_execution_summary",
            )
            .map(|artifact| artifact.artifact_path.clone())
        })
        .transpose()?;

    let checkpoint_surface = load_checkpoint_surface(&contribution_artifact_manifest)?;
    let (disposition, reason_codes, score_bps, detail, checkpoint_pointer_state) =
        classify_validator_result(
            &contribution_receipt,
            checkpoint_surface.as_ref(),
            grouped_stage_execution_summary.is_some(),
        )?;

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
    let grouped_stage_replay_evidence = grouped_stage_execution_summary
        .as_ref()
        .map(|execution_summary| {
            persist_grouped_stage_replay_evidence(
                manifest,
                &contribution_receipt,
                &contribution_artifact_manifest,
                execution_summary,
                &validator_root,
                challenge_id,
                validator_node_pubkey,
                validator_run_id.as_str(),
                disposition,
                &reason_codes,
                score_bps,
                detail.as_str(),
            )
        })
        .transpose()?;
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
        grouped_stage_execution_summary_path: grouped_stage_execution_summary_path.clone(),
        grouped_stage_execution_summary_digest: grouped_stage_execution_summary
            .as_ref()
            .map(|value| value.execution_summary_digest.clone()),
        grouped_stage_replay_evidence_path: grouped_stage_replay_evidence
            .as_ref()
            .map(|value| value.grouped_stage_replay_evidence_path.clone()),
        grouped_stage_replay_evidence_digest: grouped_stage_replay_evidence
            .as_ref()
            .map(|value| value.grouped_stage_replay_evidence_digest.clone()),
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
        grouped_stage_execution_summary_path: grouped_stage_execution_summary_path,
        grouped_stage_execution_summary_digest: grouped_stage_execution_summary
            .as_ref()
            .map(|value| value.execution_summary_digest.clone()),
        grouped_stage_replay_evidence_path: grouped_stage_replay_evidence
            .as_ref()
            .map(|value| value.grouped_stage_replay_evidence_path.clone()),
        grouped_stage_replay_evidence_digest: grouped_stage_replay_evidence
            .as_ref()
            .map(|value| value.grouped_stage_replay_evidence_digest.clone()),
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
            grouped_stage_replay_evidence_path: grouped_stage_replay_evidence
                .map(|value| value.grouped_stage_replay_evidence_path),
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

fn load_grouped_stage_execution_summary(
    manifest: &PsionicTrainInvocationManifest,
    contribution_receipt: &PsionicTrainContributionReceipt,
    contribution_artifact_manifest: &PsionicTrainContributionArtifactManifest,
) -> Result<Option<PsionicTrainGroupedReplicaStageExecutionSummary>, PsionicTrainValidatorReplayError>
{
    let Some(grouped_stage_assignment) = contribution_receipt.grouped_stage_assignment.as_ref()
    else {
        return Ok(None);
    };
    let execution_summary_artifact = require_artifact(
        contribution_artifact_manifest,
        "grouped_stage_execution_summary",
    )?;
    let execution_summary = load_psionic_train_grouped_stage_execution_summary(Path::new(
        execution_summary_artifact.artifact_path.as_str(),
    ))
    .map_err(map_grouped_stage_evidence_error)?;
    if execution_summary.lane_id != manifest.lane_id
        || execution_summary.run_id != contribution_receipt.run_id
        || execution_summary.window_id != contribution_receipt.window_id
        || execution_summary.assignment_id != contribution_receipt.assignment_id
        || execution_summary.contribution_id != contribution_receipt.contribution_id
        || execution_summary.node_pubkey != contribution_receipt.node_pubkey
        || execution_summary.grouped_stage_assignment != *grouped_stage_assignment
    {
        return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
            detail: String::from(
                "grouped stage execution summary drifted from the challenged contribution receipt",
            ),
        });
    }
    validate_grouped_stage_execution_summary_artifacts(
        contribution_artifact_manifest,
        &execution_summary,
    )?;
    Ok(Some(execution_summary))
}

fn validate_grouped_stage_execution_summary_artifacts(
    contribution_artifact_manifest: &PsionicTrainContributionArtifactManifest,
    execution_summary: &PsionicTrainGroupedReplicaStageExecutionSummary,
) -> Result<(), PsionicTrainValidatorReplayError> {
    if let Some(path) = execution_summary.input_transport_path.as_deref() {
        let input_transport_artifact = require_artifact(
            contribution_artifact_manifest,
            "grouped_stage_input_transport",
        )?;
        if input_transport_artifact.artifact_path != path {
            return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
                detail: String::from(
                    "grouped stage execution summary input transport path drifted from the artifact manifest",
                ),
            });
        }
        let loaded_transport = load_psionic_train_grouped_stage_transport(Path::new(path))
            .map_err(map_grouped_stage_transport_error)?;
        if execution_summary.input_transport_digest.as_deref()
            != Some(loaded_transport.envelope.transport_digest.as_str())
            || execution_summary.input_payload_digest.as_deref()
                != Some(loaded_transport.payload.payload_digest.as_str())
        {
            return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
                detail: String::from(
                    "grouped stage execution summary input transport digests drifted from the retained transport envelope",
                ),
            });
        }
    }
    if let Some(path) = execution_summary.output_transport_path.as_deref() {
        let output_transport_artifact = require_artifact(
            contribution_artifact_manifest,
            "grouped_stage_output_transport",
        )?;
        if output_transport_artifact.artifact_path != path {
            return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
                detail: String::from(
                    "grouped stage execution summary output transport path drifted from the artifact manifest",
                ),
            });
        }
        let output_payload_path = execution_summary.output_payload_path.as_deref().ok_or_else(|| {
            PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
                detail: String::from(
                    "grouped stage execution summary declared one output transport without one output payload path",
                ),
            }
        })?;
        let output_payload_artifact = require_artifact(
            contribution_artifact_manifest,
            "grouped_stage_output_payload",
        )?;
        if output_payload_artifact.artifact_path != output_payload_path {
            return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
                detail: String::from(
                    "grouped stage execution summary output payload path drifted from the artifact manifest",
                ),
            });
        }
        let loaded_transport = load_psionic_train_grouped_stage_transport(Path::new(path))
            .map_err(map_grouped_stage_transport_error)?;
        if execution_summary.output_transport_digest.as_deref()
            != Some(loaded_transport.envelope.transport_digest.as_str())
            || execution_summary.output_payload_digest.as_deref()
                != Some(loaded_transport.payload.payload_digest.as_str())
        {
            return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
                detail: String::from(
                    "grouped stage execution summary output transport digests drifted from the retained transport envelope",
                ),
            });
        }
        if loaded_transport.envelope.payload_path != output_payload_path {
            return Err(PsionicTrainValidatorReplayError::ArtifactDigestMismatch {
                detail: String::from(
                    "grouped stage execution summary output payload path drifted from the retained transport envelope",
                ),
            });
        }
    }
    Ok(())
}

fn persist_grouped_stage_replay_evidence(
    manifest: &PsionicTrainInvocationManifest,
    contribution_receipt: &PsionicTrainContributionReceipt,
    contribution_artifact_manifest: &PsionicTrainContributionArtifactManifest,
    execution_summary: &PsionicTrainGroupedReplicaStageExecutionSummary,
    validator_root: &Path,
    challenge_id: &str,
    validator_node_pubkey: &str,
    validator_run_id: &str,
    disposition: TrainingExecutionValidatorDisposition,
    reason_codes: &[PsionicTrainValidatorReplayReasonCode],
    score_bps: u16,
    detail: &str,
) -> Result<
    crate::PsionicTrainGroupedReplicaStageReplayEvidenceArtifacts,
    PsionicTrainValidatorReplayError,
> {
    let path = validator_root.join("grouped_stage_replay_evidence.json");
    persist_psionic_train_grouped_stage_replay_evidence(
        path.as_path(),
        PsionicTrainGroupedReplicaStageReplayEvidence {
            schema_version: String::from(
                crate::PSIONIC_TRAIN_GROUPED_STAGE_REPLAY_EVIDENCE_SCHEMA_VERSION,
            ),
            lane_id: manifest.lane_id.clone(),
            network_id: manifest.coordination.network_id.clone(),
            validator_run_id: String::from(validator_run_id),
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
            grouped_stage_assignment: execution_summary.grouped_stage_assignment.clone(),
            execution_summary_path: require_artifact(
                contribution_artifact_manifest,
                "grouped_stage_execution_summary",
            )?
            .artifact_path
            .clone(),
            execution_summary_digest: execution_summary.execution_summary_digest.clone(),
            input_transport_digest: execution_summary.input_transport_digest.clone(),
            output_transport_digest: execution_summary.output_transport_digest.clone(),
            disposition,
            reason_codes: reason_codes
                .iter()
                .map(|reason_code| validator_reason_code_label(*reason_code).to_string())
                .collect(),
            score_bps,
            detail: detail.to_string(),
            replay_evidence_digest: String::new(),
        },
    )
    .map_err(map_grouped_stage_evidence_error)
}

fn require_artifact<'a>(
    contribution_artifact_manifest: &'a PsionicTrainContributionArtifactManifest,
    artifact_kind: &str,
) -> Result<&'a PsionicTrainContributionArtifact, PsionicTrainValidatorReplayError> {
    contribution_artifact_manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.artifact_kind == artifact_kind)
        .ok_or_else(|| PsionicTrainValidatorReplayError::Read {
            path: contribution_artifact_manifest
                .artifacts
                .first()
                .map(|artifact| artifact.artifact_path.clone())
                .unwrap_or_else(|| String::from("contribution_artifact_manifest")),
            detail: format!("missing required artifact kind `{artifact_kind}`"),
        })
}

fn map_grouped_stage_evidence_error(
    error: PsionicTrainGroupedReplicaEvidenceError,
) -> PsionicTrainValidatorReplayError {
    match error {
        PsionicTrainGroupedReplicaEvidenceError::Read { path, detail } => {
            PsionicTrainValidatorReplayError::Read { path, detail }
        }
        PsionicTrainGroupedReplicaEvidenceError::Write { path, detail } => {
            PsionicTrainValidatorReplayError::Write { path, detail }
        }
        PsionicTrainGroupedReplicaEvidenceError::Parse { path, detail } => {
            PsionicTrainValidatorReplayError::Parse { path, detail }
        }
        PsionicTrainGroupedReplicaEvidenceError::Invalid { detail }
        | PsionicTrainGroupedReplicaEvidenceError::ArtifactDigestMismatch { detail } => {
            PsionicTrainValidatorReplayError::ArtifactDigestMismatch { detail }
        }
        PsionicTrainGroupedReplicaEvidenceError::StaleAssignment { detail } => {
            PsionicTrainValidatorReplayError::StaleAssignment { detail }
        }
    }
}

fn map_grouped_stage_transport_error(
    error: crate::PsionicTrainGroupedReplicaTransportError,
) -> PsionicTrainValidatorReplayError {
    match error {
        crate::PsionicTrainGroupedReplicaTransportError::Read { path, detail } => {
            PsionicTrainValidatorReplayError::Read { path, detail }
        }
        crate::PsionicTrainGroupedReplicaTransportError::Write { path, detail } => {
            PsionicTrainValidatorReplayError::Write { path, detail }
        }
        crate::PsionicTrainGroupedReplicaTransportError::Parse { path, detail } => {
            PsionicTrainValidatorReplayError::Parse { path, detail }
        }
        crate::PsionicTrainGroupedReplicaTransportError::Invalid { detail }
        | crate::PsionicTrainGroupedReplicaTransportError::ArtifactDigestMismatch { detail } => {
            PsionicTrainValidatorReplayError::ArtifactDigestMismatch { detail }
        }
        crate::PsionicTrainGroupedReplicaTransportError::StaleAssignment { detail } => {
            PsionicTrainValidatorReplayError::StaleAssignment { detail }
        }
    }
}

fn validator_reason_code_label(reason_code: PsionicTrainValidatorReplayReasonCode) -> &'static str {
    match reason_code {
        PsionicTrainValidatorReplayReasonCode::ContributionOutcomeRefused => {
            "contribution_outcome_refused"
        }
        PsionicTrainValidatorReplayReasonCode::PrimaryCheckpointAccepted => {
            "primary_checkpoint_accepted"
        }
        PsionicTrainValidatorReplayReasonCode::CheckpointRecovered => "checkpoint_recovered",
        PsionicTrainValidatorReplayReasonCode::CheckpointReplayRequired => {
            "checkpoint_replay_required"
        }
        PsionicTrainValidatorReplayReasonCode::GroupedStageEvidenceVerified => {
            "grouped_stage_evidence_verified"
        }
    }
}

fn classify_validator_result(
    contribution_receipt: &PsionicTrainContributionReceipt,
    checkpoint_surface: Option<&PsionicTrainCheckpointSurface>,
    grouped_stage_evidence_verified: bool,
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
        let mut reason_codes =
            vec![PsionicTrainValidatorReplayReasonCode::CheckpointReplayRequired];
        if grouped_stage_evidence_verified {
            reason_codes.push(PsionicTrainValidatorReplayReasonCode::GroupedStageEvidenceVerified);
        }
        return Ok((
            TrainingExecutionValidatorDisposition::ReplayRequired,
            reason_codes,
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
        let mut reason_codes = vec![PsionicTrainValidatorReplayReasonCode::CheckpointRecovered];
        if grouped_stage_evidence_verified {
            reason_codes.push(PsionicTrainValidatorReplayReasonCode::GroupedStageEvidenceVerified);
        }
        return Ok((
            TrainingExecutionValidatorDisposition::Quarantined,
            reason_codes,
            7_500,
            String::from(
                "validator replay quarantined the challenged contribution because it depends on recovered or non-primary checkpoint posture rather than a clean accepted-primary pointer",
            ),
            checkpoint_pointer_state,
        ));
    }

    let mut reason_codes = vec![PsionicTrainValidatorReplayReasonCode::PrimaryCheckpointAccepted];
    if grouped_stage_evidence_verified {
        reason_codes.push(PsionicTrainValidatorReplayReasonCode::GroupedStageEvidenceVerified);
    }
    Ok((
        TrainingExecutionValidatorDisposition::Accepted,
        reason_codes,
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
    use std::{
        env, fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{
        PsionicTrainValidatorReplayReasonCode, TrainingExecutionValidatorDisposition,
        classify_validator_result, execute_psionic_train_validator_replay,
    };
    use crate::{
        PSIONIC_TRAIN_ACTUAL_PRETRAINING_BACKEND_FAMILY,
        PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF,
        PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID,
        PSIONIC_TRAIN_ACTUAL_PRETRAINING_TOPOLOGY_CLASS,
        PSIONIC_TRAIN_CHECKPOINT_SURFACE_SCHEMA_VERSION,
        PSIONIC_TRAIN_CONTRIBUTION_RECEIPT_SCHEMA_VERSION,
        PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION, PSIONIC_TRAIN_RUNTIME_SURFACE_ID,
        PsionicTrainAdmissionIdentity, PsionicTrainAuthorityOwner,
        PsionicTrainCapabilityProjection, PsionicTrainCheckpointArtifactPaths,
        PsionicTrainCheckpointSurface, PsionicTrainContributionReceipt,
        PsionicTrainCoordinationContext, PsionicTrainGroupedReplicaStageAssignment,
        PsionicTrainGroupedReplicaStageRole, PsionicTrainInvocationManifest, PsionicTrainOperation,
        PsionicTrainOutcomeKind, PsionicTrainRole, PsionicTrainRuntimeAttestation,
        PsionicTrainWindowArtifactInputRefs, load_psionic_train_grouped_stage_replay_evidence,
        persist_psionic_train_window_artifacts,
    };

    fn temp_root(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        let path = env::temp_dir().join(format!("psionic-validator-{label}-{unique}"));
        if path.exists() {
            fs::remove_dir_all(&path).expect("temp dir should clear");
        }
        fs::create_dir_all(&path).expect("temp dir should create");
        path
    }

    fn write_json<T: serde::Serialize>(path: &Path, value: &T) {
        fs::create_dir_all(path.parent().expect("parent should exist"))
            .expect("parent dir should create");
        fs::write(
            path,
            serde_json::to_vec_pretty(value).expect("json should serialize"),
        )
        .expect("json should write");
    }

    fn runtime_attestation() -> PsionicTrainRuntimeAttestation {
        PsionicTrainRuntimeAttestation::new(
            PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID,
            "sha256:test-build",
            "1111222233334444555566667777888899990000",
            "clean",
            None,
            PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF,
        )
    }

    fn capability_projection() -> PsionicTrainCapabilityProjection {
        PsionicTrainCapabilityProjection {
            lane_id: String::from(crate::PSION_ACTUAL_PRETRAINING_LANE_ID),
            role: PsionicTrainRole::Worker,
            backend_family: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_BACKEND_FAMILY),
            topology_class: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_TOPOLOGY_CLASS),
            environment_ref: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF),
        }
    }

    fn grouped_worker_manifest(run_root: &Path) -> PsionicTrainInvocationManifest {
        PsionicTrainInvocationManifest {
            schema_version: String::from(PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION),
            runtime_surface_id: String::from(PSIONIC_TRAIN_RUNTIME_SURFACE_ID),
            lane_id: String::from(crate::PSION_ACTUAL_PRETRAINING_LANE_ID),
            role: PsionicTrainRole::Worker,
            operation: PsionicTrainOperation::Start,
            coordination: PsionicTrainCoordinationContext {
                network_id: Some(String::from("network.psionic.validator-test")),
                window_id: Some(String::from("window-0001")),
                assignment_id: Some(String::from("assignment-0001")),
                challenge_id: None,
                node_pubkey: Some(String::from("npub1-grouped-worker")),
                membership_revision: Some(7),
            },
            grouped_stage_assignment: Some(
                PsionicTrainGroupedReplicaStageAssignment::new(
                    "replica-01",
                    "stage-01",
                    0,
                    2,
                    PsionicTrainGroupedReplicaStageRole::Ingress,
                    None,
                    Some(String::from("stage-02")),
                )
                .expect("grouped stage assignment should build"),
            ),
            admission_identity: PsionicTrainAdmissionIdentity {
                release_id: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID),
                build_digest: String::from("sha256:test-build"),
                environment_ref: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF),
            },
            run_id: Some(String::from("worker-run")),
            output_root: Some(run_root.display().to_string()),
            run_root: None,
            peer_node_pubkey: None,
            peer_checkpoint_handoff_receipt_path: None,
            validator_target_contribution_receipt_path: None,
            validator_target_contribution_artifact_manifest_path: None,
            grouped_stage_input_transport_path: None,
            selected_git_ref: Some(String::from("HEAD")),
            hardware_observation_path: None,
            run_shape_observation_path: None,
            allow_dirty_tree: false,
            dry_run: true,
            checkpoint_label: None,
            optimizer_step: None,
            checkpoint_ref: None,
            checkpoint_object_digest: None,
            checkpoint_total_bytes: None,
            inject_failed_upload: false,
            inject_eval_worker_unavailable: false,
            manifest_digest: None,
        }
    }

    fn validator_manifest(
        run_root: &Path,
        contribution_receipt_path: String,
        artifact_manifest_path: String,
    ) -> PsionicTrainInvocationManifest {
        PsionicTrainInvocationManifest {
            schema_version: String::from(PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION),
            runtime_surface_id: String::from(PSIONIC_TRAIN_RUNTIME_SURFACE_ID),
            lane_id: String::from(crate::PSION_ACTUAL_PRETRAINING_LANE_ID),
            role: PsionicTrainRole::Validator,
            operation: PsionicTrainOperation::ValidateContribution,
            coordination: PsionicTrainCoordinationContext {
                network_id: Some(String::from("network.psionic.validator-test")),
                window_id: Some(String::from("window-0001")),
                assignment_id: Some(String::from("assignment-0001")),
                challenge_id: Some(String::from("challenge-0001")),
                node_pubkey: Some(String::from("npub1-validator")),
                membership_revision: Some(9),
            },
            grouped_stage_assignment: None,
            admission_identity: PsionicTrainAdmissionIdentity {
                release_id: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID),
                build_digest: String::from("sha256:test-build"),
                environment_ref: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF),
            },
            run_id: Some(String::from("validator-run")),
            output_root: None,
            run_root: Some(run_root.display().to_string()),
            peer_node_pubkey: None,
            peer_checkpoint_handoff_receipt_path: None,
            validator_target_contribution_receipt_path: Some(contribution_receipt_path),
            validator_target_contribution_artifact_manifest_path: Some(artifact_manifest_path),
            grouped_stage_input_transport_path: None,
            selected_git_ref: Some(String::from("HEAD")),
            hardware_observation_path: None,
            run_shape_observation_path: None,
            allow_dirty_tree: false,
            dry_run: true,
            checkpoint_label: None,
            optimizer_step: None,
            checkpoint_ref: None,
            checkpoint_object_digest: None,
            checkpoint_total_bytes: None,
            inject_failed_upload: false,
            inject_eval_worker_unavailable: false,
            manifest_digest: None,
        }
    }

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
            classify_validator_result(&receipt, Some(&surface), false)
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
            classify_validator_result(&receipt, Some(&surface), false)
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
            classify_validator_result(&receipt, Some(&surface), false)
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
            classify_validator_result(&receipt, None, false).expect("classification should work");
        assert_eq!(disposition, TrainingExecutionValidatorDisposition::Rejected);
        assert_eq!(
            reason_codes,
            vec![PsionicTrainValidatorReplayReasonCode::ContributionOutcomeRefused]
        );
        assert_eq!(score_bps, 0);
    }

    #[test]
    fn grouped_stage_validator_replay_emits_replay_evidence() {
        let run_root = temp_root("grouped-stage-replay");
        let mut worker_manifest = grouped_worker_manifest(&run_root);
        worker_manifest
            .populate_manifest_digest()
            .expect("worker manifest digest should populate");
        let invocation_manifest_path = run_root.join("manifest/worker_invocation_manifest.json");
        write_json(invocation_manifest_path.as_path(), &worker_manifest);

        let checkpoint_surface_path = run_root.join("status/checkpoint_surface.json");
        write_json(
            checkpoint_surface_path.as_path(),
            &checkpoint_surface(Some("accepted"), Some("succeeded"), None, Some(false), None),
        );

        let window_artifacts = persist_psionic_train_window_artifacts(
            &worker_manifest,
            &runtime_attestation(),
            &capability_projection(),
            "worker-run",
            &run_root,
            &PsionicTrainWindowArtifactInputRefs {
                invocation_manifest_path: invocation_manifest_path.display().to_string(),
                launch_manifest_path: None,
                membership_revision_path: None,
                grouped_stage_input_transport_path: None,
                checkpoint_surface_path: Some(checkpoint_surface_path.display().to_string()),
                checkpoint_pointer_path: None,
                checkpoint_manifest_path: None,
                checkpoint_backup_receipt_path: None,
                checkpoint_handoff_receipt_path: None,
                recovery_receipt_path: None,
                current_status_path: None,
                retained_summary_path: None,
                launcher_log_path: None,
                final_closeout_bundle_path: None,
            },
            PsionicTrainOutcomeKind::Succeeded,
            0,
            false,
            PsionicTrainAuthorityOwner::Pylon,
            None,
            "grouped stage worker contribution completed",
        )
        .expect("worker window artifacts should persist")
        .expect("worker should emit window artifacts");
        let validator_manifest = validator_manifest(
            &run_root,
            window_artifacts.contribution_receipt_path.clone(),
            window_artifacts.contribution_artifact_manifest_path.clone(),
        );
        let execution = execute_psionic_train_validator_replay(&validator_manifest, &run_root)
            .expect("validator replay should succeed");
        let contribution_receipt: PsionicTrainContributionReceipt = serde_json::from_slice(
            &fs::read(&window_artifacts.contribution_receipt_path)
                .expect("contribution receipt should read"),
        )
        .expect("contribution receipt should parse");
        let replay_evidence_path = execution
            .artifacts
            .grouped_stage_replay_evidence_path
            .as_deref()
            .expect("grouped stage replay evidence should be emitted");
        let replay_evidence =
            load_psionic_train_grouped_stage_replay_evidence(Path::new(replay_evidence_path))
                .expect("grouped stage replay evidence should parse");
        assert_eq!(
            replay_evidence.contribution_id,
            contribution_receipt.contribution_id
        );
        assert_eq!(
            replay_evidence.grouped_stage_assignment.stage_id,
            "stage-01"
        );
        assert!(
            replay_evidence
                .reason_codes
                .iter()
                .any(|reason_code| reason_code == "grouped_stage_evidence_verified")
        );
        assert_eq!(
            execution
                .score_receipt
                .grouped_stage_execution_summary_path
                .as_deref(),
            window_artifacts
                .grouped_stage_execution_summary_path
                .as_deref()
        );
        assert_eq!(
            execution
                .score_receipt
                .grouped_stage_replay_evidence_path
                .as_deref(),
            Some(replay_evidence_path)
        );
    }
}
