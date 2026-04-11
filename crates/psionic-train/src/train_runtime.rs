use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::PSION_ACTUAL_PRETRAINING_LANE_ID;
use crate::PsionicTrainGroupedReplicaStageAssignment;

/// Stable admitted lane id for the first Apple-homogeneous machine training lane.
pub const PSION_APPLE_WINDOWED_TRAINING_LANE_ID: &str = "psion_apple_windowed_training_v1";

/// Stable schema version for the machine-consumable `psionic-train` invocation manifest.
pub const PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION: &str =
    "psionic.train.invocation_manifest.v1";

/// Stable schema version for the machine-consumable `psionic-train` status packet.
pub const PSIONIC_TRAIN_STATUS_PACKET_SCHEMA_VERSION: &str = "psionic.train.status_packet.v1";

/// Stable schema version for one retained runtime attestation packet.
pub const PSIONIC_TRAIN_RUNTIME_ATTESTATION_SCHEMA_VERSION: &str =
    "psionic.train.runtime_attestation.v1";

/// Stable schema version for the machine-readable run-status packet.
pub const PSIONIC_TRAIN_RUN_STATUS_PACKET_SCHEMA_VERSION: &str =
    "psionic.train.run_status_packet.v1";

/// Stable schema version for the machine-readable window-status packet.
pub const PSIONIC_TRAIN_WINDOW_STATUS_PACKET_SCHEMA_VERSION: &str =
    "psionic.train.window_status_packet.v1";

/// Stable runtime surface identifier for the first machine-consumable `psionic-train` CLI.
pub const PSIONIC_TRAIN_RUNTIME_SURFACE_ID: &str = "psionic-train.runtime.v1";

/// Stable admitted release id for the actual pretraining lane on the first machine runtime.
pub const PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID: &str =
    "psionic-train.psion_actual_pretraining.release.v1";

/// Stable admitted environment ref for the actual pretraining lane on the first machine runtime.
pub const PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF: &str =
    "psionic.environment.psion_actual_pretraining.cuda_h100.operator@v1";

/// Stable admitted release id for the first Apple machine lane on the runtime surface.
pub const PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_RELEASE_ID: &str =
    "psionic-train.psion_apple_windowed_training.release.v1";

/// Stable admitted environment ref for the first Apple machine lane on the runtime surface.
pub const PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_ENVIRONMENT_REF: &str =
    "psionic.environment.psion_apple_windowed_training.metal_mlx.operator@v1";

/// Stable backend family projected by the first admitted actual-lane runtime.
pub const PSIONIC_TRAIN_ACTUAL_PRETRAINING_BACKEND_FAMILY: &str = "cuda";

/// Stable topology class projected by the first admitted actual-lane runtime.
pub const PSIONIC_TRAIN_ACTUAL_PRETRAINING_TOPOLOGY_CLASS: &str =
    "homogeneous_four_node_h100_tensor_parallel";

/// Stable backend family projected by the first admitted Apple machine runtime.
pub const PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_BACKEND_FAMILY: &str = "metal";

/// Stable topology class projected by the first admitted Apple machine runtime.
pub const PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_TOPOLOGY_CLASS: &str =
    "homogeneous_apple_silicon_data_parallel";

/// One stable machine role consumed by `psionic-train`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionicTrainRole {
    Worker,
    Validator,
    RecoverySource,
}

/// One stable machine operation consumed by `psionic-train`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionicTrainOperation {
    Start,
    Resume,
    ServeCheckpoint,
    ValidateContribution,
    RecordCheckpoint,
    Backup,
    DecideContinueRestart,
    RehearseBaseLane,
}

/// Stable outcome class for the machine-readable status packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionicTrainOutcomeKind {
    Succeeded,
    Refused,
}

/// Shared refusal taxonomy for the machine-facing `psionic-train` process boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionicTrainRefusalClass {
    BadConfig,
    StaleAssignment,
    LeaseExpired,
    UnsupportedTopology,
    GroupedStageAssignmentInvalid,
    CheckpointMissing,
    CheckpointDigestMismatch,
    ArtifactIncomplete,
    ArtifactDigestMismatch,
    ValidatorTimeout,
    ValidatorDisagreement,
    EnvironmentMismatch,
    BuildRevoked,
    InternalError,
}

/// Authority that owns the next durable transition after the refusal leaves the local process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionicTrainAuthorityOwner {
    Pylon,
    Nexus,
}

/// Shared coordination envelope frozen for telemetry, receipts, and machine status packets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainCoordinationContext {
    /// Stable network identifier when the caller binds the run to one network.
    pub network_id: Option<String>,
    /// Stable window identifier when the caller binds the run to one window.
    pub window_id: Option<String>,
    /// Stable assignment identifier when the caller binds the run to one assignment.
    pub assignment_id: Option<String>,
    /// Stable challenge identifier when the caller binds the run to one validator challenge.
    pub challenge_id: Option<String>,
    /// Stable node pubkey when the caller binds the run to one admitted node identity.
    pub node_pubkey: Option<String>,
    /// Stable membership revision when one distributed membership revision already exists.
    pub membership_revision: Option<u64>,
}

/// Admitted identity that the machine caller expects the runtime to satisfy before launch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainAdmissionIdentity {
    /// Stable admitted release id.
    pub release_id: String,
    /// Stable admitted build digest.
    pub build_digest: String,
    /// Stable admitted environment ref.
    pub environment_ref: String,
}

/// Runtime attestation the machine process emits after resolving the local executable context.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainRuntimeAttestation {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable admitted release id.
    pub release_id: String,
    /// Stable resolved build digest.
    pub build_digest: String,
    /// Stable resolved git commit SHA.
    pub git_commit_sha: String,
    /// Stable dirty-tree posture used to derive the build identity.
    pub dirty_tree_admission: String,
    /// Optional status digest when dirty-tree override is active.
    pub workspace_status_sha256: Option<String>,
    /// Stable admitted environment ref.
    pub environment_ref: String,
}

/// Minimal capability projection frozen for the first machine runtime lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainCapabilityProjection {
    /// Stable lane identifier.
    pub lane_id: String,
    /// Stable admitted role.
    pub role: PsionicTrainRole,
    /// Stable backend family.
    pub backend_family: String,
    /// Stable topology class.
    pub topology_class: String,
    /// Stable admitted environment ref.
    pub environment_ref: String,
}

/// One shared set of retained artifact refs carried by run-status and window-status packets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainArtifactSurfaceRefs {
    /// Optional retained launch-manifest path.
    pub launch_manifest_path: Option<String>,
    /// Optional retained membership-revision receipt path.
    pub membership_revision_path: Option<String>,
    /// Optional retained runtime-window execution path.
    pub window_execution_path: Option<String>,
    /// Optional retained contribution receipt path.
    pub contribution_receipt_path: Option<String>,
    /// Optional retained contribution artifact-manifest path.
    pub contribution_artifact_manifest_path: Option<String>,
    /// Optional retained checkpoint-surface path.
    pub checkpoint_surface_path: Option<String>,
    /// Optional retained checkpoint-pointer path.
    pub checkpoint_pointer_path: Option<String>,
    /// Optional retained checkpoint-manifest path.
    pub checkpoint_manifest_path: Option<String>,
    /// Optional retained checkpoint-backup receipt path.
    pub checkpoint_backup_receipt_path: Option<String>,
    /// Optional retained peer checkpoint-handoff receipt path.
    pub checkpoint_handoff_receipt_path: Option<String>,
    /// Optional retained recovery receipt path.
    pub recovery_receipt_path: Option<String>,
    /// Optional retained validator score receipt path.
    pub validator_score_receipt_path: Option<String>,
    /// Optional retained sealed-window bundle path.
    pub sealed_window_bundle_path: Option<String>,
    /// Optional retained final closeout bundle path.
    pub final_closeout_bundle_path: Option<String>,
}

/// Stable machine-consumable invocation manifest for `psionic-train`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainInvocationManifest {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable runtime surface id.
    pub runtime_surface_id: String,
    /// Stable lane identifier.
    pub lane_id: String,
    /// Stable runtime role.
    pub role: PsionicTrainRole,
    /// Stable runtime operation.
    pub operation: PsionicTrainOperation,
    /// Shared coordination envelope for the run when the caller already knows those ids.
    #[serde(default)]
    pub coordination: PsionicTrainCoordinationContext,
    /// Optional grouped-replica stage assignment when one worker participates in a staged replica.
    #[serde(default)]
    pub grouped_stage_assignment: Option<PsionicTrainGroupedReplicaStageAssignment>,
    /// Admitted release, build, and environment identity expected before launch.
    pub admission_identity: PsionicTrainAdmissionIdentity,
    /// Stable run identifier when the machine caller wants deterministic run roots.
    pub run_id: Option<String>,
    /// Explicit output root for launch-style commands.
    pub output_root: Option<String>,
    /// Explicit run root for commands that operate on retained state.
    pub run_root: Option<String>,
    /// Optional admitted peer node pubkey for recovery-source checkpoint serving.
    pub peer_node_pubkey: Option<String>,
    /// Optional retained peer checkpoint-handoff receipt path consumed before resume.
    pub peer_checkpoint_handoff_receipt_path: Option<String>,
    /// Optional challenged contribution receipt path consumed by validator replay.
    pub validator_target_contribution_receipt_path: Option<String>,
    /// Optional challenged contribution artifact-manifest path consumed by validator replay.
    pub validator_target_contribution_artifact_manifest_path: Option<String>,
    /// Explicit git ref selection. Defaults are not allowed for machine mode.
    pub selected_git_ref: Option<String>,
    /// Optional retained hardware observation path.
    pub hardware_observation_path: Option<String>,
    /// Optional retained run-shape observation path.
    pub run_shape_observation_path: Option<String>,
    /// Whether dirty-tree override is enabled.
    pub allow_dirty_tree: bool,
    /// Whether the execution remains in dry-run posture.
    pub dry_run: bool,
    /// Optional checkpoint label for record-checkpoint.
    pub checkpoint_label: Option<String>,
    /// Optional optimizer step for record-checkpoint.
    pub optimizer_step: Option<u64>,
    /// Optional checkpoint ref for record-checkpoint.
    pub checkpoint_ref: Option<String>,
    /// Optional checkpoint object digest override.
    pub checkpoint_object_digest: Option<String>,
    /// Optional checkpoint object size override.
    pub checkpoint_total_bytes: Option<u64>,
    /// Optional injected failed-upload drill.
    pub inject_failed_upload: bool,
    /// Optional injected eval-worker-unavailable drill.
    pub inject_eval_worker_unavailable: bool,
    /// Canonical digest with this field omitted from the digest basis.
    pub manifest_digest: Option<String>,
}

/// Stable machine-readable status packet emitted by `psionic-train`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainStatusPacket {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable runtime surface id.
    pub runtime_surface_id: String,
    /// Stable lane identifier.
    pub lane_id: String,
    /// Stable runtime role.
    pub role: PsionicTrainRole,
    /// Stable runtime operation.
    pub operation: PsionicTrainOperation,
    /// Stable outcome kind.
    pub outcome: PsionicTrainOutcomeKind,
    /// Stable numeric exit code.
    pub exit_code: u8,
    /// Retryability bit for supervisors.
    pub retryable: bool,
    /// Authority that owns the next durable transition.
    pub authority_owner: PsionicTrainAuthorityOwner,
    /// Optional refusal class when the run is refused.
    pub refusal_class: Option<PsionicTrainRefusalClass>,
    /// Optional invocation manifest path.
    pub manifest_path: Option<String>,
    /// Optional invocation manifest digest.
    pub manifest_digest: Option<String>,
    /// Shared coordination envelope.
    pub coordination: PsionicTrainCoordinationContext,
    /// Optional grouped-replica stage assignment projected by the admitted manifest.
    pub grouped_stage_assignment: Option<PsionicTrainGroupedReplicaStageAssignment>,
    /// Runtime attestation resolved by the local machine before or during execution.
    pub runtime_attestation: Option<PsionicTrainRuntimeAttestation>,
    /// Minimal runtime capability projection.
    pub capability_projection: Option<PsionicTrainCapabilityProjection>,
    /// Optional run identifier.
    pub run_id: Option<String>,
    /// Optional run root.
    pub run_root: Option<String>,
    /// Optional retained run-status packet path.
    pub run_status_packet_path: Option<String>,
    /// Optional retained window-status packet path.
    pub window_status_packet_path: Option<String>,
    /// Optional current-status path for actual-lane runs.
    pub current_status_path: Option<String>,
    /// Optional retained-summary path for actual-lane runs.
    pub retained_summary_path: Option<String>,
    /// Optional latest-checkpoint-pointer path for actual-lane runs.
    pub latest_checkpoint_pointer_path: Option<String>,
    /// Optional launcher log path for actual-lane runs.
    pub launcher_log_path: Option<String>,
    /// Short human-readable detail for logs and operator inspection.
    pub detail: String,
}

/// Stable machine-readable run-status packet for `Pylon`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainRunStatusPacket {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable runtime surface id.
    pub runtime_surface_id: String,
    /// Stable lane identifier.
    pub lane_id: String,
    /// Stable runtime role.
    pub role: PsionicTrainRole,
    /// Stable runtime operation.
    pub operation: PsionicTrainOperation,
    /// Stable outcome kind.
    pub outcome: PsionicTrainOutcomeKind,
    /// Stable numeric exit code.
    pub exit_code: u8,
    /// Retryability bit for supervisors.
    pub retryable: bool,
    /// Authority that owns the next durable transition.
    pub authority_owner: PsionicTrainAuthorityOwner,
    /// Optional refusal class when the run is refused.
    pub refusal_class: Option<PsionicTrainRefusalClass>,
    /// Shared coordination envelope.
    pub coordination: PsionicTrainCoordinationContext,
    /// Optional grouped-replica stage assignment projected by the admitted manifest.
    pub grouped_stage_assignment: Option<PsionicTrainGroupedReplicaStageAssignment>,
    /// Optional invocation manifest path.
    pub manifest_path: Option<String>,
    /// Optional invocation manifest digest.
    pub manifest_digest: Option<String>,
    /// Optional run identifier.
    pub run_id: Option<String>,
    /// Optional run root.
    pub run_root: Option<String>,
    /// Optional retained phase for the live run.
    pub phase: Option<String>,
    /// Resolved runtime attestation.
    pub runtime_attestation: PsionicTrainRuntimeAttestation,
    /// Minimal runtime capability projection.
    pub capability_projection: PsionicTrainCapabilityProjection,
    /// Shared retained artifact refs.
    pub artifacts: PsionicTrainArtifactSurfaceRefs,
    /// Optional current-status path for actual-lane runs.
    pub current_status_path: Option<String>,
    /// Optional retained-summary path for actual-lane runs.
    pub retained_summary_path: Option<String>,
    /// Optional launcher log path for actual-lane runs.
    pub launcher_log_path: Option<String>,
    /// Short human-readable detail for logs and operator inspection.
    pub detail: String,
}

/// Stable machine-readable window-status packet for `Nexus`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainWindowStatusPacket {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable runtime surface id.
    pub runtime_surface_id: String,
    /// Stable lane identifier.
    pub lane_id: String,
    /// Stable runtime role.
    pub role: PsionicTrainRole,
    /// Stable runtime operation.
    pub operation: PsionicTrainOperation,
    /// Stable outcome kind.
    pub outcome: PsionicTrainOutcomeKind,
    /// Stable numeric exit code.
    pub exit_code: u8,
    /// Retryability bit for supervisors.
    pub retryable: bool,
    /// Authority that owns the next durable transition.
    pub authority_owner: PsionicTrainAuthorityOwner,
    /// Optional refusal class when the run is refused.
    pub refusal_class: Option<PsionicTrainRefusalClass>,
    /// Shared coordination envelope.
    pub coordination: PsionicTrainCoordinationContext,
    /// Optional grouped-replica stage assignment projected by the admitted manifest.
    pub grouped_stage_assignment: Option<PsionicTrainGroupedReplicaStageAssignment>,
    /// Optional invocation manifest digest.
    pub manifest_digest: Option<String>,
    /// Optional run identifier.
    pub run_id: Option<String>,
    /// Optional run root.
    pub run_root: Option<String>,
    /// Optional current window state.
    pub window_state: Option<String>,
    /// Resolved runtime attestation.
    pub runtime_attestation: PsionicTrainRuntimeAttestation,
    /// Minimal runtime capability projection.
    pub capability_projection: PsionicTrainCapabilityProjection,
    /// Shared retained artifact refs.
    pub artifacts: PsionicTrainArtifactSurfaceRefs,
    /// Short human-readable detail for logs and operator inspection.
    pub detail: String,
}

/// Validation errors for the machine-consumable runtime contract.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PsionicTrainRuntimeContractError {
    #[error("psionic-train runtime field `{field}` must not be empty")]
    MissingField { field: String },
    #[error("psionic-train runtime field `{field}` expected `{expected}` but found `{actual}`")]
    FieldMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    #[error("psionic-train runtime field `{field}` is invalid: {detail}")]
    InvalidValue { field: String, detail: String },
}

impl PsionicTrainInvocationManifest {
    /// Computes the canonical manifest digest with `manifest_digest` omitted from the digest basis.
    pub fn canonical_manifest_digest(&self) -> Result<String, PsionicTrainRuntimeContractError> {
        let mut digest_basis = self.clone();
        digest_basis.manifest_digest = None;
        let serialized = serde_json::to_vec(&digest_basis).map_err(|error| {
            PsionicTrainRuntimeContractError::InvalidValue {
                field: String::from("manifest_digest"),
                detail: error.to_string(),
            }
        })?;
        Ok(sha256_hex(&serialized))
    }

    /// Populates the canonical manifest digest in place.
    pub fn populate_manifest_digest(&mut self) -> Result<(), PsionicTrainRuntimeContractError> {
        self.manifest_digest = Some(self.canonical_manifest_digest()?);
        Ok(())
    }

    /// Validates the machine-consumable runtime contract.
    pub fn validate_machine_contract(&self) -> Result<(), PsionicTrainRuntimeContractError> {
        require_nonempty(
            self.schema_version.as_str(),
            "invocation_manifest.schema_version",
        )?;
        if self.schema_version != PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION {
            return Err(PsionicTrainRuntimeContractError::FieldMismatch {
                field: String::from("invocation_manifest.schema_version"),
                expected: String::from(PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION),
                actual: self.schema_version.clone(),
            });
        }
        require_nonempty(
            self.runtime_surface_id.as_str(),
            "invocation_manifest.runtime_surface_id",
        )?;
        if self.runtime_surface_id != PSIONIC_TRAIN_RUNTIME_SURFACE_ID {
            return Err(PsionicTrainRuntimeContractError::FieldMismatch {
                field: String::from("invocation_manifest.runtime_surface_id"),
                expected: String::from(PSIONIC_TRAIN_RUNTIME_SURFACE_ID),
                actual: self.runtime_surface_id.clone(),
            });
        }
        require_nonempty(self.lane_id.as_str(), "invocation_manifest.lane_id")?;
        if !is_machine_admitted_lane(self.lane_id.as_str()) {
            return Err(PsionicTrainRuntimeContractError::InvalidValue {
                field: String::from("invocation_manifest.lane_id"),
                detail: format!(
                    "lane `{}` is not yet admitted by the machine runtime surface",
                    self.lane_id
                ),
            });
        }
        if let Some(manifest_digest) = self.manifest_digest.as_deref() {
            require_nonempty(manifest_digest, "invocation_manifest.manifest_digest")?;
            let expected = self.canonical_manifest_digest()?;
            if manifest_digest != expected {
                return Err(PsionicTrainRuntimeContractError::FieldMismatch {
                    field: String::from("invocation_manifest.manifest_digest"),
                    expected,
                    actual: String::from(manifest_digest),
                });
            }
        }
        self.coordination
            .validate("invocation_manifest.coordination")?;
        if let Some(stage_assignment) = &self.grouped_stage_assignment {
            stage_assignment.validate("invocation_manifest.grouped_stage_assignment")?;
            require_nonempty_option(
                self.coordination.window_id.as_deref(),
                "invocation_manifest.coordination.window_id",
            )?;
            require_nonempty_option(
                self.coordination.assignment_id.as_deref(),
                "invocation_manifest.coordination.assignment_id",
            )?;
            if self.role != PsionicTrainRole::Worker {
                return Err(PsionicTrainRuntimeContractError::InvalidValue {
                    field: String::from("invocation_manifest.grouped_stage_assignment"),
                    detail: String::from(
                        "grouped stage assignment is currently admitted only for worker role on the machine runtime surface",
                    ),
                });
            }
        }
        self.admission_identity
            .validate("invocation_manifest.admission_identity")?;
        require_nonempty_option(
            self.coordination.node_pubkey.as_deref(),
            "invocation_manifest.coordination.node_pubkey",
        )?;
        validate_optional_field(
            self.peer_node_pubkey.as_deref(),
            "invocation_manifest",
            "peer_node_pubkey",
        )?;
        validate_optional_field(
            self.peer_checkpoint_handoff_receipt_path.as_deref(),
            "invocation_manifest",
            "peer_checkpoint_handoff_receipt_path",
        )?;
        validate_optional_field(
            self.validator_target_contribution_receipt_path.as_deref(),
            "invocation_manifest",
            "validator_target_contribution_receipt_path",
        )?;
        validate_optional_field(
            self.validator_target_contribution_artifact_manifest_path
                .as_deref(),
            "invocation_manifest",
            "validator_target_contribution_artifact_manifest_path",
        )?;

        if self.selected_git_ref.is_none() {
            return Err(PsionicTrainRuntimeContractError::MissingField {
                field: String::from("invocation_manifest.selected_git_ref"),
            });
        }
        require_nonempty_option(
            self.selected_git_ref.as_deref(),
            "invocation_manifest.selected_git_ref",
        )?;

        match self.role {
            PsionicTrainRole::Worker => match self.operation {
                PsionicTrainOperation::Start
                | PsionicTrainOperation::RecordCheckpoint
                | PsionicTrainOperation::Backup
                | PsionicTrainOperation::RehearseBaseLane => {}
                PsionicTrainOperation::Resume
                | PsionicTrainOperation::ServeCheckpoint
                | PsionicTrainOperation::ValidateContribution
                | PsionicTrainOperation::DecideContinueRestart => {
                    return Err(PsionicTrainRuntimeContractError::InvalidValue {
                        field: String::from("invocation_manifest.role"),
                        detail: String::from(
                            "worker role cannot claim recovery-source operations on the machine runtime surface",
                        ),
                    });
                }
            },
            PsionicTrainRole::RecoverySource => match self.operation {
                PsionicTrainOperation::Resume
                | PsionicTrainOperation::ServeCheckpoint
                | PsionicTrainOperation::DecideContinueRestart => {}
                _ => {
                    return Err(PsionicTrainRuntimeContractError::InvalidValue {
                        field: String::from("invocation_manifest.role"),
                        detail: String::from(
                            "recovery_source role is limited to resume, serve_checkpoint, and decide_continue_restart on the machine runtime surface",
                        ),
                    });
                }
            },
            PsionicTrainRole::Validator => match self.operation {
                PsionicTrainOperation::ValidateContribution => {}
                _ => {
                    return Err(PsionicTrainRuntimeContractError::InvalidValue {
                        field: String::from("invocation_manifest.role"),
                        detail: String::from(
                            "validator role is limited to validate_contribution on the machine runtime surface",
                        ),
                    });
                }
            },
        }

        match self.operation {
            PsionicTrainOperation::Start | PsionicTrainOperation::RehearseBaseLane => {
                require_nonempty_option(self.run_id.as_deref(), "invocation_manifest.run_id")?;
                require_nonempty_option(
                    self.output_root.as_deref(),
                    "invocation_manifest.output_root",
                )?;
                if self.run_root.is_some() {
                    return Err(PsionicTrainRuntimeContractError::InvalidValue {
                        field: String::from("invocation_manifest.run_root"),
                        detail: String::from(
                            "launch-style operations must use output_root instead of run_root",
                        ),
                    });
                }
            }
            PsionicTrainOperation::Resume
            | PsionicTrainOperation::ServeCheckpoint
            | PsionicTrainOperation::ValidateContribution
            | PsionicTrainOperation::Backup
            | PsionicTrainOperation::DecideContinueRestart => {
                require_nonempty_option(self.run_root.as_deref(), "invocation_manifest.run_root")?;
            }
            PsionicTrainOperation::RecordCheckpoint => {
                require_nonempty_option(self.run_root.as_deref(), "invocation_manifest.run_root")?;
                require_nonempty_option(
                    self.checkpoint_label.as_deref(),
                    "invocation_manifest.checkpoint_label",
                )?;
                if self.optimizer_step.is_none() {
                    return Err(PsionicTrainRuntimeContractError::MissingField {
                        field: String::from("invocation_manifest.optimizer_step"),
                    });
                }
                require_nonempty_option(
                    self.checkpoint_ref.as_deref(),
                    "invocation_manifest.checkpoint_ref",
                )?;
            }
        }
        if self.operation == PsionicTrainOperation::ServeCheckpoint {
            require_nonempty_option(
                self.peer_node_pubkey.as_deref(),
                "invocation_manifest.peer_node_pubkey",
            )?;
        } else if self.peer_node_pubkey.is_some() {
            return Err(PsionicTrainRuntimeContractError::InvalidValue {
                field: String::from("invocation_manifest.peer_node_pubkey"),
                detail: String::from(
                    "peer_node_pubkey is only admitted for serve_checkpoint on the machine runtime surface",
                ),
            });
        }
        if self.operation != PsionicTrainOperation::Resume
            && self.peer_checkpoint_handoff_receipt_path.is_some()
        {
            return Err(PsionicTrainRuntimeContractError::InvalidValue {
                field: String::from("invocation_manifest.peer_checkpoint_handoff_receipt_path"),
                detail: String::from(
                    "peer_checkpoint_handoff_receipt_path is only admitted for resume on the machine runtime surface",
                ),
            });
        }
        if self.operation == PsionicTrainOperation::ValidateContribution {
            require_nonempty_option(
                self.coordination.window_id.as_deref(),
                "invocation_manifest.coordination.window_id",
            )?;
            require_nonempty_option(
                self.coordination.assignment_id.as_deref(),
                "invocation_manifest.coordination.assignment_id",
            )?;
            require_nonempty_option(
                self.coordination.challenge_id.as_deref(),
                "invocation_manifest.coordination.challenge_id",
            )?;
            require_nonempty_option(
                self.validator_target_contribution_receipt_path.as_deref(),
                "invocation_manifest.validator_target_contribution_receipt_path",
            )?;
            require_nonempty_option(
                self.validator_target_contribution_artifact_manifest_path
                    .as_deref(),
                "invocation_manifest.validator_target_contribution_artifact_manifest_path",
            )?;
        } else if self.validator_target_contribution_receipt_path.is_some()
            || self
                .validator_target_contribution_artifact_manifest_path
                .is_some()
        {
            return Err(PsionicTrainRuntimeContractError::InvalidValue {
                field: String::from(
                    "invocation_manifest.validator_target_contribution_receipt_path",
                ),
                detail: String::from(
                    "validator target contribution inputs are only admitted for validate_contribution on the machine runtime surface",
                ),
            });
        }

        Ok(())
    }
}

impl PsionicTrainStatusPacket {
    /// Creates a success status packet.
    #[allow(clippy::too_many_arguments)]
    pub fn success(
        manifest: &PsionicTrainInvocationManifest,
        manifest_path: Option<String>,
        runtime_attestation: PsionicTrainRuntimeAttestation,
        capability_projection: PsionicTrainCapabilityProjection,
        run_id: Option<String>,
        run_root: Option<String>,
        run_status_packet_path: Option<String>,
        window_status_packet_path: Option<String>,
        current_status_path: Option<String>,
        retained_summary_path: Option<String>,
        latest_checkpoint_pointer_path: Option<String>,
        launcher_log_path: Option<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: String::from(PSIONIC_TRAIN_STATUS_PACKET_SCHEMA_VERSION),
            runtime_surface_id: String::from(PSIONIC_TRAIN_RUNTIME_SURFACE_ID),
            lane_id: manifest.lane_id.clone(),
            role: manifest.role,
            operation: manifest.operation,
            outcome: PsionicTrainOutcomeKind::Succeeded,
            exit_code: 0,
            retryable: false,
            authority_owner: PsionicTrainAuthorityOwner::Pylon,
            refusal_class: None,
            manifest_path,
            manifest_digest: manifest.manifest_digest.clone(),
            coordination: manifest.coordination.clone(),
            grouped_stage_assignment: manifest.grouped_stage_assignment.clone(),
            runtime_attestation: Some(runtime_attestation),
            capability_projection: Some(capability_projection),
            run_id,
            run_root,
            run_status_packet_path,
            window_status_packet_path,
            current_status_path,
            retained_summary_path,
            latest_checkpoint_pointer_path,
            launcher_log_path,
            detail: detail.into(),
        }
    }

    /// Creates a refusal status packet from the shared refusal taxonomy.
    pub fn refusal(
        manifest: Option<&PsionicTrainInvocationManifest>,
        refusal_class: PsionicTrainRefusalClass,
        manifest_path: Option<String>,
        runtime_attestation: Option<PsionicTrainRuntimeAttestation>,
        capability_projection: Option<PsionicTrainCapabilityProjection>,
        run_id: Option<String>,
        run_root: Option<String>,
        run_status_packet_path: Option<String>,
        window_status_packet_path: Option<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: String::from(PSIONIC_TRAIN_STATUS_PACKET_SCHEMA_VERSION),
            runtime_surface_id: String::from(PSIONIC_TRAIN_RUNTIME_SURFACE_ID),
            lane_id: manifest
                .map(|value| value.lane_id.clone())
                .unwrap_or_else(|| String::from(PSION_ACTUAL_PRETRAINING_LANE_ID)),
            role: manifest
                .map(|value| value.role)
                .unwrap_or(PsionicTrainRole::Worker),
            operation: manifest
                .map(|value| value.operation)
                .unwrap_or(PsionicTrainOperation::Start),
            outcome: PsionicTrainOutcomeKind::Refused,
            exit_code: refusal_class.exit_code(),
            retryable: refusal_class.retryable(),
            authority_owner: refusal_class.authority_owner(),
            refusal_class: Some(refusal_class),
            manifest_path,
            manifest_digest: manifest.and_then(|value| value.manifest_digest.clone()),
            coordination: manifest
                .map(|value| value.coordination.clone())
                .unwrap_or_default(),
            grouped_stage_assignment: manifest
                .and_then(|value| value.grouped_stage_assignment.clone()),
            runtime_attestation,
            capability_projection,
            run_id,
            run_root,
            run_status_packet_path,
            window_status_packet_path,
            current_status_path: None,
            retained_summary_path: None,
            latest_checkpoint_pointer_path: None,
            launcher_log_path: None,
            detail: detail.into(),
        }
    }
}

impl PsionicTrainCoordinationContext {
    /// Validates optional coordination fields when present.
    pub fn validate(&self, field_prefix: &str) -> Result<(), PsionicTrainRuntimeContractError> {
        validate_optional_field(self.network_id.as_deref(), field_prefix, "network_id")?;
        validate_optional_field(self.window_id.as_deref(), field_prefix, "window_id")?;
        validate_optional_field(self.assignment_id.as_deref(), field_prefix, "assignment_id")?;
        validate_optional_field(self.challenge_id.as_deref(), field_prefix, "challenge_id")?;
        validate_optional_field(self.node_pubkey.as_deref(), field_prefix, "node_pubkey")?;
        Ok(())
    }
}

impl PsionicTrainAdmissionIdentity {
    /// Validates the admitted runtime identity fields.
    pub fn validate(&self, field_prefix: &str) -> Result<(), PsionicTrainRuntimeContractError> {
        require_nonempty(
            self.release_id.as_str(),
            Box::leak(format!("{field_prefix}.release_id").into_boxed_str()),
        )?;
        require_nonempty(
            self.build_digest.as_str(),
            Box::leak(format!("{field_prefix}.build_digest").into_boxed_str()),
        )?;
        require_nonempty(
            self.environment_ref.as_str(),
            Box::leak(format!("{field_prefix}.environment_ref").into_boxed_str()),
        )?;
        Ok(())
    }
}

impl PsionicTrainRuntimeAttestation {
    /// Creates one resolved runtime attestation packet.
    pub fn new(
        release_id: impl Into<String>,
        build_digest: impl Into<String>,
        git_commit_sha: impl Into<String>,
        dirty_tree_admission: impl Into<String>,
        workspace_status_sha256: Option<String>,
        environment_ref: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: String::from(PSIONIC_TRAIN_RUNTIME_ATTESTATION_SCHEMA_VERSION),
            release_id: release_id.into(),
            build_digest: build_digest.into(),
            git_commit_sha: git_commit_sha.into(),
            dirty_tree_admission: dirty_tree_admission.into(),
            workspace_status_sha256,
            environment_ref: environment_ref.into(),
        }
    }
}

impl PsionicTrainCapabilityProjection {
    /// Builds the frozen capability projection for one admitted lane and role.
    pub fn for_lane(
        lane_id: &str,
        role: PsionicTrainRole,
        environment_ref: impl Into<String>,
    ) -> Result<Self, PsionicTrainRuntimeContractError> {
        match lane_id {
            PSION_ACTUAL_PRETRAINING_LANE_ID => Ok(Self {
                lane_id: String::from(lane_id),
                role,
                backend_family: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_BACKEND_FAMILY),
                topology_class: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_TOPOLOGY_CLASS),
                environment_ref: environment_ref.into(),
            }),
            PSION_APPLE_WINDOWED_TRAINING_LANE_ID => Ok(Self {
                lane_id: String::from(lane_id),
                role,
                backend_family: String::from(PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_BACKEND_FAMILY),
                topology_class: String::from(PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_TOPOLOGY_CLASS),
                environment_ref: environment_ref.into(),
            }),
            other => Err(PsionicTrainRuntimeContractError::InvalidValue {
                field: String::from("invocation_manifest.lane_id"),
                detail: format!("lane `{other}` has no admitted machine capability projection"),
            }),
        }
    }
}

/// Returns the admitted release id for one machine-supported lane.
pub fn admitted_release_id_for_lane(
    lane_id: &str,
) -> Result<&'static str, PsionicTrainRuntimeContractError> {
    match lane_id {
        PSION_ACTUAL_PRETRAINING_LANE_ID => Ok(PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID),
        PSION_APPLE_WINDOWED_TRAINING_LANE_ID => {
            Ok(PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_RELEASE_ID)
        }
        other => Err(PsionicTrainRuntimeContractError::InvalidValue {
            field: String::from("invocation_manifest.lane_id"),
            detail: format!("lane `{other}` has no admitted release id on the machine runtime"),
        }),
    }
}

/// Returns the admitted environment ref for one machine-supported lane.
pub fn admitted_environment_ref_for_lane(
    lane_id: &str,
) -> Result<&'static str, PsionicTrainRuntimeContractError> {
    match lane_id {
        PSION_ACTUAL_PRETRAINING_LANE_ID => Ok(PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF),
        PSION_APPLE_WINDOWED_TRAINING_LANE_ID => {
            Ok(PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_ENVIRONMENT_REF)
        }
        other => Err(PsionicTrainRuntimeContractError::InvalidValue {
            field: String::from("invocation_manifest.lane_id"),
            detail: format!(
                "lane `{other}` has no admitted environment ref on the machine runtime"
            ),
        }),
    }
}

/// Whether the lane is admitted by the machine runtime surface.
pub fn is_machine_admitted_lane(lane_id: &str) -> bool {
    matches!(
        lane_id,
        PSION_ACTUAL_PRETRAINING_LANE_ID | PSION_APPLE_WINDOWED_TRAINING_LANE_ID
    )
}

/// Computes the stable runtime build digest from the resolved executable posture.
pub fn runtime_build_digest(
    release_id: &str,
    runtime_surface_id: &str,
    lane_id: &str,
    git_commit_sha: &str,
    dirty_tree_admission: &str,
    workspace_status_sha256: Option<&str>,
    environment_ref: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"psionic-train-runtime-build|release_id|");
    digest.update(release_id.as_bytes());
    digest.update(b"|runtime_surface_id|");
    digest.update(runtime_surface_id.as_bytes());
    digest.update(b"|lane_id|");
    digest.update(lane_id.as_bytes());
    digest.update(b"|git_commit_sha|");
    digest.update(git_commit_sha.as_bytes());
    digest.update(b"|dirty_tree_admission|");
    digest.update(dirty_tree_admission.as_bytes());
    digest.update(b"|workspace_status_sha256|");
    digest.update(workspace_status_sha256.unwrap_or(""));
    digest.update(b"|environment_ref|");
    digest.update(environment_ref.as_bytes());
    format!("{:x}", digest.finalize())
}

impl PsionicTrainOperation {
    /// Stable CLI subcommand used by the actual-lane operator surface.
    pub fn cli_subcommand(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Resume => "resume",
            Self::ServeCheckpoint => "serve-checkpoint",
            Self::ValidateContribution => "validate-contribution",
            Self::RecordCheckpoint => "record-checkpoint",
            Self::Backup => "backup",
            Self::DecideContinueRestart => "decide-continue-restart",
            Self::RehearseBaseLane => "rehearse-base-lane",
        }
    }
}

impl PsionicTrainRefusalClass {
    /// Stable exit code for the refusal class.
    pub fn exit_code(self) -> u8 {
        match self {
            Self::BadConfig => 10,
            Self::StaleAssignment => 11,
            Self::LeaseExpired => 12,
            Self::UnsupportedTopology => 13,
            Self::GroupedStageAssignmentInvalid => 14,
            Self::CheckpointMissing => 15,
            Self::CheckpointDigestMismatch => 16,
            Self::ArtifactIncomplete => 17,
            Self::ArtifactDigestMismatch => 18,
            Self::ValidatorTimeout => 19,
            Self::ValidatorDisagreement => 20,
            Self::EnvironmentMismatch => 21,
            Self::BuildRevoked => 22,
            Self::InternalError => 70,
        }
    }

    /// Whether the caller should treat the refusal as retryable.
    pub fn retryable(self) -> bool {
        match self {
            Self::BadConfig => false,
            Self::StaleAssignment => true,
            Self::LeaseExpired => true,
            Self::UnsupportedTopology => false,
            Self::GroupedStageAssignmentInvalid => false,
            Self::CheckpointMissing => true,
            Self::CheckpointDigestMismatch => false,
            Self::ArtifactIncomplete => true,
            Self::ArtifactDigestMismatch => false,
            Self::ValidatorTimeout => true,
            Self::ValidatorDisagreement => true,
            Self::EnvironmentMismatch => false,
            Self::BuildRevoked => false,
            Self::InternalError => false,
        }
    }

    /// Which authority owns the next durable transition once the refusal leaves the local process.
    pub fn authority_owner(self) -> PsionicTrainAuthorityOwner {
        match self {
            Self::StaleAssignment
            | Self::LeaseExpired
            | Self::ValidatorTimeout
            | Self::ValidatorDisagreement => PsionicTrainAuthorityOwner::Nexus,
            Self::BadConfig
            | Self::UnsupportedTopology
            | Self::GroupedStageAssignmentInvalid
            | Self::CheckpointMissing
            | Self::CheckpointDigestMismatch
            | Self::ArtifactIncomplete
            | Self::ArtifactDigestMismatch
            | Self::EnvironmentMismatch
            | Self::BuildRevoked
            | Self::InternalError => PsionicTrainAuthorityOwner::Pylon,
        }
    }
}

fn require_nonempty(
    value: &str,
    field: &'static str,
) -> Result<(), PsionicTrainRuntimeContractError> {
    if value.trim().is_empty() {
        return Err(PsionicTrainRuntimeContractError::MissingField {
            field: String::from(field),
        });
    }
    Ok(())
}

fn require_nonempty_option(
    value: Option<&str>,
    field: &'static str,
) -> Result<(), PsionicTrainRuntimeContractError> {
    let value = value.ok_or_else(|| PsionicTrainRuntimeContractError::MissingField {
        field: String::from(field),
    })?;
    require_nonempty(value, field)
}

fn validate_optional_field(
    value: Option<&str>,
    field_prefix: &str,
    field_name: &str,
) -> Result<(), PsionicTrainRuntimeContractError> {
    if let Some(value) = value {
        require_nonempty(
            value,
            Box::leak(format!("{field_prefix}.{field_name}").into_boxed_str()),
        )?;
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PsionicTrainGroupedReplicaStageRole;

    fn base_manifest() -> PsionicTrainInvocationManifest {
        PsionicTrainInvocationManifest {
            schema_version: String::from(PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION),
            runtime_surface_id: String::from(PSIONIC_TRAIN_RUNTIME_SURFACE_ID),
            lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
            role: PsionicTrainRole::Worker,
            operation: PsionicTrainOperation::Start,
            coordination: PsionicTrainCoordinationContext {
                network_id: Some(String::from("network.psionic.contract-test")),
                window_id: None,
                assignment_id: None,
                challenge_id: None,
                node_pubkey: Some(String::from("npub1-psionic-contract-test")),
                membership_revision: None,
            },
            grouped_stage_assignment: None,
            admission_identity: PsionicTrainAdmissionIdentity {
                release_id: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID),
                build_digest: String::from("sha256:test-build"),
                environment_ref: String::from(PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF),
            },
            run_id: Some(String::from("psion-train-contract-test")),
            output_root: Some(String::from("/tmp/psion-train-contract-test")),
            run_root: None,
            peer_node_pubkey: None,
            peer_checkpoint_handoff_receipt_path: None,
            validator_target_contribution_receipt_path: None,
            validator_target_contribution_artifact_manifest_path: None,
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

    #[test]
    fn invocation_manifest_digest_round_trips() {
        let mut manifest = base_manifest();
        manifest
            .populate_manifest_digest()
            .expect("digest should populate");
        manifest
            .validate_machine_contract()
            .expect("manifest should validate after digest population");
    }

    #[test]
    fn launch_manifest_requires_explicit_output_root() {
        let mut manifest = base_manifest();
        manifest.output_root = None;
        let error = manifest
            .validate_machine_contract()
            .expect_err("machine launch manifest should require output_root");
        assert_eq!(
            error,
            PsionicTrainRuntimeContractError::MissingField {
                field: String::from("invocation_manifest.output_root"),
            }
        );
    }

    #[test]
    fn validator_role_requires_validate_contribution_operation() {
        let mut manifest = base_manifest();
        manifest.role = PsionicTrainRole::Validator;
        let error = manifest
            .validate_machine_contract()
            .expect_err("validator role should reject worker operations");
        assert_eq!(
            error,
            PsionicTrainRuntimeContractError::InvalidValue {
                field: String::from("invocation_manifest.role"),
                detail: String::from(
                    "validator role is limited to validate_contribution on the machine runtime surface",
                ),
            }
        );
    }

    #[test]
    fn validator_manifest_requires_replay_target_paths() {
        let mut manifest = base_manifest();
        manifest.role = PsionicTrainRole::Validator;
        manifest.operation = PsionicTrainOperation::ValidateContribution;
        manifest.output_root = None;
        manifest.run_root = Some(String::from("/tmp/validator-run"));
        manifest.coordination.window_id = Some(String::from("window-0001"));
        manifest.coordination.assignment_id = Some(String::from("assignment-0001"));
        manifest.coordination.challenge_id = Some(String::from("challenge-0001"));
        let error = manifest
            .validate_machine_contract()
            .expect_err("validator replay should require target contribution paths");
        assert_eq!(
            error,
            PsionicTrainRuntimeContractError::MissingField {
                field: String::from(
                    "invocation_manifest.validator_target_contribution_receipt_path"
                ),
            }
        );
    }

    #[test]
    fn refusal_codes_and_authority_owners_are_stable() {
        assert_eq!(PsionicTrainRefusalClass::BadConfig.exit_code(), 10);
        assert_eq!(PsionicTrainRefusalClass::LeaseExpired.exit_code(), 12);
        assert_eq!(
            PsionicTrainRefusalClass::GroupedStageAssignmentInvalid.exit_code(),
            14
        );
        assert_eq!(
            PsionicTrainRefusalClass::ValidatorTimeout.authority_owner(),
            PsionicTrainAuthorityOwner::Nexus
        );
        assert!(!PsionicTrainRefusalClass::BuildRevoked.retryable());
    }

    #[test]
    fn worker_manifest_requires_node_pubkey() {
        let mut manifest = base_manifest();
        manifest.coordination.node_pubkey = None;
        let error = manifest
            .validate_machine_contract()
            .expect_err("worker manifest should require node_pubkey");
        assert_eq!(
            error,
            PsionicTrainRuntimeContractError::MissingField {
                field: String::from("invocation_manifest.coordination.node_pubkey"),
            }
        );
    }

    #[test]
    fn runtime_build_digest_is_stable() {
        let digest = runtime_build_digest(
            PSIONIC_TRAIN_ACTUAL_PRETRAINING_RELEASE_ID,
            PSIONIC_TRAIN_RUNTIME_SURFACE_ID,
            PSION_ACTUAL_PRETRAINING_LANE_ID,
            "1111222233334444555566667777888899990000",
            "refuse_by_default",
            None,
            PSIONIC_TRAIN_ACTUAL_PRETRAINING_ENVIRONMENT_REF,
        );
        assert_eq!(
            digest,
            "2ed54b617f887b00bcb7f65caecf75a40058e5cea8c7b12631948f199c8281b7"
        );
    }

    #[test]
    fn apple_lane_is_admitted_by_machine_contract() {
        let mut manifest = base_manifest();
        manifest.lane_id = String::from(PSION_APPLE_WINDOWED_TRAINING_LANE_ID);
        manifest.admission_identity = PsionicTrainAdmissionIdentity {
            release_id: String::from(PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_RELEASE_ID),
            build_digest: String::from("sha256:apple-build"),
            environment_ref: String::from(PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_ENVIRONMENT_REF),
        };
        manifest
            .validate_machine_contract()
            .expect("apple machine manifests should validate");
        let projection = PsionicTrainCapabilityProjection::for_lane(
            PSION_APPLE_WINDOWED_TRAINING_LANE_ID,
            PsionicTrainRole::Worker,
            PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_ENVIRONMENT_REF,
        )
        .expect("apple capability projection should exist");
        assert_eq!(
            projection.backend_family,
            PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_BACKEND_FAMILY
        );
        assert_eq!(
            projection.topology_class,
            PSIONIC_TRAIN_APPLE_WINDOWED_TRAINING_TOPOLOGY_CLASS
        );
    }

    #[test]
    fn grouped_stage_assignment_requires_window_and_assignment_context() {
        let mut manifest = base_manifest();
        manifest.coordination.window_id = None;
        manifest.grouped_stage_assignment = Some(
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
        );
        let error = manifest
            .validate_machine_contract()
            .expect_err("grouped stage assignment should require window context");
        assert_eq!(
            error,
            PsionicTrainRuntimeContractError::MissingField {
                field: String::from("invocation_manifest.coordination.window_id"),
            }
        );
    }

    #[test]
    fn validator_manifest_rejects_grouped_stage_assignment() {
        let mut manifest = base_manifest();
        manifest.role = PsionicTrainRole::Validator;
        manifest.operation = PsionicTrainOperation::ValidateContribution;
        manifest.output_root = None;
        manifest.run_root = Some(String::from("/tmp/validator-run"));
        manifest.coordination.window_id = Some(String::from("window-0001"));
        manifest.coordination.assignment_id = Some(String::from("assignment-0001"));
        manifest.coordination.challenge_id = Some(String::from("challenge-0001"));
        manifest.validator_target_contribution_receipt_path =
            Some(String::from("/tmp/contribution_receipt.json"));
        manifest.validator_target_contribution_artifact_manifest_path =
            Some(String::from("/tmp/artifact_manifest.json"));
        manifest.grouped_stage_assignment = Some(
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
        );
        let error = manifest
            .validate_machine_contract()
            .expect_err("validator should reject grouped-stage worker assignment");
        assert_eq!(
            error,
            PsionicTrainRuntimeContractError::InvalidValue {
                field: String::from("invocation_manifest.grouped_stage_assignment"),
                detail: String::from(
                    "grouped stage assignment is currently admitted only for worker role on the machine runtime surface",
                ),
            }
        );
    }
}
