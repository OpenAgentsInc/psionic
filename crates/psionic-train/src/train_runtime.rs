use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::PSION_ACTUAL_PRETRAINING_LANE_ID;

/// Stable schema version for the machine-consumable `psionic-train` invocation manifest.
pub const PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION: &str =
    "psionic.train.invocation_manifest.v1";

/// Stable schema version for the machine-consumable `psionic-train` status packet.
pub const PSIONIC_TRAIN_STATUS_PACKET_SCHEMA_VERSION: &str = "psionic.train.status_packet.v1";

/// Stable runtime surface identifier for the first machine-consumable `psionic-train` CLI.
pub const PSIONIC_TRAIN_RUNTIME_SURFACE_ID: &str = "psionic-train.runtime.v1";

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
    /// Stable run identifier when the machine caller wants deterministic run roots.
    pub run_id: Option<String>,
    /// Explicit output root for launch-style commands.
    pub output_root: Option<String>,
    /// Explicit run root for commands that operate on retained state.
    pub run_root: Option<String>,
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
    /// Optional run identifier.
    pub run_id: Option<String>,
    /// Optional run root.
    pub run_root: Option<String>,
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
        if self.lane_id != PSION_ACTUAL_PRETRAINING_LANE_ID {
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
                PsionicTrainOperation::Resume | PsionicTrainOperation::DecideContinueRestart => {
                    return Err(PsionicTrainRuntimeContractError::InvalidValue {
                        field: String::from("invocation_manifest.role"),
                        detail: String::from(
                            "worker role cannot claim recovery-source operations on the machine runtime surface",
                        ),
                    });
                }
            },
            PsionicTrainRole::RecoverySource => match self.operation {
                PsionicTrainOperation::Resume | PsionicTrainOperation::DecideContinueRestart => {}
                _ => {
                    return Err(PsionicTrainRuntimeContractError::InvalidValue {
                        field: String::from("invocation_manifest.role"),
                        detail: String::from(
                            "recovery_source role is limited to resume and decide_continue_restart on the machine runtime surface",
                        ),
                    });
                }
            },
            PsionicTrainRole::Validator => {
                return Err(PsionicTrainRuntimeContractError::InvalidValue {
                    field: String::from("invocation_manifest.role"),
                    detail: String::from(
                        "validator role is not yet admitted by the first machine runtime surface",
                    ),
                });
            }
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

        Ok(())
    }
}

impl PsionicTrainStatusPacket {
    /// Creates a success status packet.
    #[allow(clippy::too_many_arguments)]
    pub fn success(
        manifest: &PsionicTrainInvocationManifest,
        manifest_path: Option<String>,
        run_id: Option<String>,
        run_root: Option<String>,
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
            run_id,
            run_root,
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
        run_id: Option<String>,
        run_root: Option<String>,
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
            run_id,
            run_root,
            current_status_path: None,
            retained_summary_path: None,
            latest_checkpoint_pointer_path: None,
            launcher_log_path: None,
            detail: detail.into(),
        }
    }
}

impl PsionicTrainOperation {
    /// Stable CLI subcommand used by the actual-lane operator surface.
    pub fn cli_subcommand(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::Resume => "resume",
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
            Self::CheckpointMissing => 14,
            Self::CheckpointDigestMismatch => 15,
            Self::ArtifactIncomplete => 16,
            Self::ArtifactDigestMismatch => 17,
            Self::ValidatorTimeout => 18,
            Self::ValidatorDisagreement => 19,
            Self::EnvironmentMismatch => 20,
            Self::BuildRevoked => 21,
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

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_manifest() -> PsionicTrainInvocationManifest {
        PsionicTrainInvocationManifest {
            schema_version: String::from(PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION),
            runtime_surface_id: String::from(PSIONIC_TRAIN_RUNTIME_SURFACE_ID),
            lane_id: String::from(PSION_ACTUAL_PRETRAINING_LANE_ID),
            role: PsionicTrainRole::Worker,
            operation: PsionicTrainOperation::Start,
            run_id: Some(String::from("psion-train-contract-test")),
            output_root: Some(String::from("/tmp/psion-train-contract-test")),
            run_root: None,
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
    fn validator_role_is_not_yet_admitted() {
        let mut manifest = base_manifest();
        manifest.role = PsionicTrainRole::Validator;
        let error = manifest
            .validate_machine_contract()
            .expect_err("validator role should be refused");
        assert_eq!(
            error,
            PsionicTrainRuntimeContractError::InvalidValue {
                field: String::from("invocation_manifest.role"),
                detail: String::from(
                    "validator role is not yet admitted by the first machine runtime surface",
                ),
            }
        );
    }

    #[test]
    fn refusal_codes_and_authority_owners_are_stable() {
        assert_eq!(PsionicTrainRefusalClass::BadConfig.exit_code(), 10);
        assert_eq!(PsionicTrainRefusalClass::LeaseExpired.exit_code(), 12);
        assert_eq!(
            PsionicTrainRefusalClass::ValidatorTimeout.authority_owner(),
            PsionicTrainAuthorityOwner::Nexus
        );
        assert!(!PsionicTrainRefusalClass::BuildRevoked.retryable());
    }
}
