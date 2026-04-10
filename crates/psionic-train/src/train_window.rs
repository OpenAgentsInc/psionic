use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    PsionicTrainAuthorityOwner, PsionicTrainCapabilityProjection,
    PsionicTrainInvocationManifest, PsionicTrainOutcomeKind, PsionicTrainRefusalClass,
    PsionicTrainRole, PsionicTrainRuntimeAttestation,
};

pub const PSIONIC_TRAIN_WINDOW_EXECUTION_SCHEMA_VERSION: &str =
    "psionic.train.window_execution.v1";
pub const PSIONIC_TRAIN_CONTRIBUTION_RECEIPT_SCHEMA_VERSION: &str =
    "psionic.train.contribution_receipt.v1";
pub const PSIONIC_TRAIN_CONTRIBUTION_ARTIFACT_MANIFEST_SCHEMA_VERSION: &str =
    "psionic.train.contribution_artifact_manifest.v1";
pub const PSIONIC_TRAIN_SEALED_WINDOW_BUNDLE_SCHEMA_VERSION: &str =
    "psionic.train.sealed_window_bundle.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainWindowArtifactInputRefs {
    pub invocation_manifest_path: String,
    pub launch_manifest_path: Option<String>,
    pub membership_revision_path: Option<String>,
    pub checkpoint_surface_path: Option<String>,
    pub checkpoint_pointer_path: Option<String>,
    pub checkpoint_manifest_path: Option<String>,
    pub checkpoint_backup_receipt_path: Option<String>,
    pub checkpoint_handoff_receipt_path: Option<String>,
    pub recovery_receipt_path: Option<String>,
    pub current_status_path: Option<String>,
    pub retained_summary_path: Option<String>,
    pub launcher_log_path: Option<String>,
    pub final_closeout_bundle_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainWindowArtifactOutputs {
    pub window_execution_path: String,
    pub contribution_receipt_path: String,
    pub contribution_artifact_manifest_path: String,
    pub sealed_window_bundle_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainWindowAssignmentMaterialization {
    pub assignment_id: String,
    pub contribution_id: String,
    pub node_pubkey: String,
    pub role: PsionicTrainRole,
    pub membership_revision: Option<u64>,
    pub assignment_materialization_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainWindowExecution {
    pub schema_version: String,
    pub lane_id: String,
    pub network_id: Option<String>,
    pub run_id: String,
    pub window_id: String,
    pub challenge_id: Option<String>,
    pub window_execution_id: String,
    pub current_assignment: PsionicTrainWindowAssignmentMaterialization,
    pub runtime_build_digest: String,
    pub capability_backend_family: String,
    pub capability_topology_class: String,
    pub detail: String,
    pub window_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainContributionArtifact {
    pub artifact_kind: String,
    pub artifact_path: String,
    pub artifact_sha256: String,
    pub artifact_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainContributionArtifactManifest {
    pub schema_version: String,
    pub lane_id: String,
    pub run_id: String,
    pub window_id: String,
    pub assignment_id: String,
    pub contribution_id: String,
    pub node_pubkey: String,
    pub artifact_count: usize,
    pub artifacts: Vec<PsionicTrainContributionArtifact>,
    pub artifact_manifest_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainContributionReceipt {
    pub schema_version: String,
    pub lane_id: String,
    pub run_id: String,
    pub window_id: String,
    pub window_execution_id: String,
    pub assignment_id: String,
    pub contribution_id: String,
    pub node_pubkey: String,
    pub role: PsionicTrainRole,
    pub operation: String,
    pub outcome: PsionicTrainOutcomeKind,
    pub exit_code: u8,
    pub retryable: bool,
    pub authority_owner: PsionicTrainAuthorityOwner,
    pub refusal_class: Option<PsionicTrainRefusalClass>,
    pub artifact_manifest_path: String,
    pub artifact_manifest_digest: String,
    pub artifact_count: usize,
    pub contribution_digest: String,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainSealedWindowContribution {
    pub assignment_id: String,
    pub contribution_id: String,
    pub node_pubkey: String,
    pub outcome: PsionicTrainOutcomeKind,
    pub artifact_manifest_digest: String,
    pub contribution_digest: String,
    pub contribution_receipt_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionicTrainSealedWindowBundle {
    pub schema_version: String,
    pub lane_id: String,
    pub network_id: Option<String>,
    pub run_id: String,
    pub window_id: String,
    pub window_execution_path: String,
    pub contribution_count: usize,
    pub artifact_manifest_count: usize,
    pub contributions: Vec<PsionicTrainSealedWindowContribution>,
    pub contribution_rollup_digest: String,
    pub detail: String,
    pub sealed_window_digest: String,
}

#[derive(Debug, Error)]
pub enum PsionicTrainWindowArtifactError {
    #[error("failed to read `{path}`: {detail}")]
    Read { path: String, detail: String },
    #[error("failed to write `{path}`: {detail}")]
    Write { path: String, detail: String },
    #[error("failed to parse `{path}`: {detail}")]
    Parse { path: String, detail: String },
}

impl PsionicTrainContributionArtifactManifest {
    #[must_use]
    pub fn stable_artifact_manifest_digest(&self) -> String {
        let mut digest_basis = self.clone();
        digest_basis.artifact_manifest_digest.clear();
        stable_digest(
            b"psionic_train_contribution_artifact_manifest|",
            &digest_basis,
        )
    }
}

impl PsionicTrainContributionReceipt {
    #[must_use]
    pub fn stable_contribution_digest(&self) -> String {
        let mut digest_basis = self.clone();
        digest_basis.contribution_digest.clear();
        stable_digest(b"psionic_train_contribution_receipt|", &digest_basis)
    }
}

pub fn persist_psionic_train_window_artifacts(
    manifest: &PsionicTrainInvocationManifest,
    runtime_attestation: &PsionicTrainRuntimeAttestation,
    capability_projection: &PsionicTrainCapabilityProjection,
    run_id: &str,
    run_root: &Path,
    artifact_inputs: &PsionicTrainWindowArtifactInputRefs,
    outcome: PsionicTrainOutcomeKind,
    exit_code: u8,
    retryable: bool,
    authority_owner: PsionicTrainAuthorityOwner,
    refusal_class: Option<PsionicTrainRefusalClass>,
    detail: &str,
) -> Result<Option<PsionicTrainWindowArtifactOutputs>, PsionicTrainWindowArtifactError> {
    if manifest.role == PsionicTrainRole::Validator {
        return Ok(None);
    }
    let Some(window_id) = manifest.coordination.window_id.as_ref() else {
        return Ok(None);
    };
    let Some(assignment_id) = manifest.coordination.assignment_id.as_ref() else {
        return Ok(None);
    };
    let Some(node_pubkey) = manifest.coordination.node_pubkey.as_ref() else {
        return Ok(None);
    };

    let window_execution_id = stable_id(
        b"psionic_train_window_execution_id|",
        &[
            manifest.lane_id.as_str(),
            manifest.coordination.network_id.as_deref().unwrap_or(""),
            run_id,
            window_id,
        ],
    );
    let contribution_id = stable_id(
        b"psionic_train_contribution_id|",
        &[
            manifest.lane_id.as_str(),
            run_id,
            window_id,
            assignment_id,
            node_pubkey,
        ],
    );
    let assignment_materialization_digest = stable_id(
        b"psionic_train_assignment_materialization|",
        &[
            &window_execution_id,
            assignment_id,
            node_pubkey,
            manifest
                .coordination
                .membership_revision
                .map_or(String::new(), |value| value.to_string())
                .as_str(),
            role_label(manifest.role),
            manifest.operation.cli_subcommand(),
        ],
    );

    let window_root = run_root.join("windows").join(window_id);
    let contribution_root = window_root.join("contributions").join(&contribution_id);
    fs::create_dir_all(&contribution_root).map_err(|error| PsionicTrainWindowArtifactError::Write {
        path: contribution_root.display().to_string(),
        detail: error.to_string(),
    })?;

    let current_assignment = PsionicTrainWindowAssignmentMaterialization {
        assignment_id: assignment_id.clone(),
        contribution_id: contribution_id.clone(),
        node_pubkey: node_pubkey.clone(),
        role: manifest.role,
        membership_revision: manifest.coordination.membership_revision,
        assignment_materialization_digest,
    };
    let mut window_execution = PsionicTrainWindowExecution {
        schema_version: String::from(PSIONIC_TRAIN_WINDOW_EXECUTION_SCHEMA_VERSION),
        lane_id: manifest.lane_id.clone(),
        network_id: manifest.coordination.network_id.clone(),
        run_id: String::from(run_id),
        window_id: window_id.clone(),
        challenge_id: manifest.coordination.challenge_id.clone(),
        window_execution_id,
        current_assignment,
        runtime_build_digest: runtime_attestation.build_digest.clone(),
        capability_backend_family: capability_projection.backend_family.clone(),
        capability_topology_class: capability_projection.topology_class.clone(),
        detail: String::from(
            "Machine runtime window execution binds one deterministic window context and assignment materialization to the retained contribution artifact family.",
        ),
        window_digest: String::new(),
    };
    window_execution.window_digest =
        stable_digest(b"psionic_train_window_execution|", &window_execution);
    let window_execution_path = window_root.join("window_execution.json");
    write_json(window_execution_path.as_path(), &window_execution)?;

    let mut artifact_manifest = PsionicTrainContributionArtifactManifest {
        schema_version: String::from(PSIONIC_TRAIN_CONTRIBUTION_ARTIFACT_MANIFEST_SCHEMA_VERSION),
        lane_id: manifest.lane_id.clone(),
        run_id: String::from(run_id),
        window_id: window_id.clone(),
        assignment_id: assignment_id.clone(),
        contribution_id: contribution_id.clone(),
        node_pubkey: node_pubkey.clone(),
        artifact_count: 0,
        artifacts: collect_artifacts(artifact_inputs)?,
        artifact_manifest_digest: String::new(),
    };
    artifact_manifest.artifact_count = artifact_manifest.artifacts.len();
    artifact_manifest.artifact_manifest_digest = stable_digest(
        b"psionic_train_contribution_artifact_manifest|",
        &artifact_manifest,
    );
    let contribution_artifact_manifest_path = contribution_root.join("artifact_manifest.json");
    write_json(
        contribution_artifact_manifest_path.as_path(),
        &artifact_manifest,
    )?;

    let mut contribution_receipt = PsionicTrainContributionReceipt {
        schema_version: String::from(PSIONIC_TRAIN_CONTRIBUTION_RECEIPT_SCHEMA_VERSION),
        lane_id: manifest.lane_id.clone(),
        run_id: String::from(run_id),
        window_id: window_id.clone(),
        window_execution_id: window_execution.window_execution_id.clone(),
        assignment_id: assignment_id.clone(),
        contribution_id: contribution_id.clone(),
        node_pubkey: node_pubkey.clone(),
        role: manifest.role,
        operation: String::from(manifest.operation.cli_subcommand()),
        outcome,
        exit_code,
        retryable,
        authority_owner,
        refusal_class,
        artifact_manifest_path: contribution_artifact_manifest_path.display().to_string(),
        artifact_manifest_digest: artifact_manifest.artifact_manifest_digest.clone(),
        artifact_count: artifact_manifest.artifact_count,
        contribution_digest: String::new(),
        detail: detail.to_string(),
    };
    contribution_receipt.contribution_digest =
        stable_digest(b"psionic_train_contribution_receipt|", &contribution_receipt);
    let contribution_receipt_path = contribution_root.join("contribution_receipt.json");
    write_json(contribution_receipt_path.as_path(), &contribution_receipt)?;

    let mut contributions = load_window_contributions(&window_root)?;
    contributions.sort_by(|left, right| {
        left.assignment_id
            .cmp(&right.assignment_id)
            .then_with(|| left.contribution_id.cmp(&right.contribution_id))
    });
    let sealed_window_contributions = contributions
        .iter()
        .map(|entry| PsionicTrainSealedWindowContribution {
            assignment_id: entry.assignment_id.clone(),
            contribution_id: entry.contribution_id.clone(),
            node_pubkey: entry.node_pubkey.clone(),
            outcome: entry.outcome,
            artifact_manifest_digest: entry.artifact_manifest_digest.clone(),
            contribution_digest: entry.contribution_digest.clone(),
            contribution_receipt_path: entry.receipt_path.display().to_string(),
        })
        .collect::<Vec<_>>();
    let contribution_rollup_digest = stable_id(
        b"psionic_train_sealed_window_rollup|",
        &sealed_window_contributions
            .iter()
            .flat_map(|entry| {
                [
                    entry.assignment_id.as_str(),
                    entry.contribution_id.as_str(),
                    entry.artifact_manifest_digest.as_str(),
                    entry.contribution_digest.as_str(),
                ]
            })
            .collect::<Vec<_>>(),
    );
    let mut sealed_window_bundle = PsionicTrainSealedWindowBundle {
        schema_version: String::from(PSIONIC_TRAIN_SEALED_WINDOW_BUNDLE_SCHEMA_VERSION),
        lane_id: manifest.lane_id.clone(),
        network_id: manifest.coordination.network_id.clone(),
        run_id: String::from(run_id),
        window_id: window_id.clone(),
        window_execution_path: window_execution_path.display().to_string(),
        contribution_count: sealed_window_contributions.len(),
        artifact_manifest_count: sealed_window_contributions.len(),
        contributions: sealed_window_contributions,
        contribution_rollup_digest,
        detail: String::from(
            "Sealed-window bundle rolls up the current contribution receipt set into one deterministic count-and-digest summary for the machine runtime kernel model.",
        ),
        sealed_window_digest: String::new(),
    };
    sealed_window_bundle.sealed_window_digest =
        stable_digest(b"psionic_train_sealed_window_bundle|", &sealed_window_bundle);
    let sealed_window_bundle_path = window_root.join("sealed_window_bundle.json");
    write_json(sealed_window_bundle_path.as_path(), &sealed_window_bundle)?;

    Ok(Some(PsionicTrainWindowArtifactOutputs {
        window_execution_path: window_execution_path.display().to_string(),
        contribution_receipt_path: contribution_receipt_path.display().to_string(),
        contribution_artifact_manifest_path: contribution_artifact_manifest_path
            .display()
            .to_string(),
        sealed_window_bundle_path: sealed_window_bundle_path.display().to_string(),
    }))
}

#[derive(Clone, Debug)]
struct RetainedContributionEntry {
    assignment_id: String,
    contribution_id: String,
    node_pubkey: String,
    outcome: PsionicTrainOutcomeKind,
    artifact_manifest_digest: String,
    contribution_digest: String,
    receipt_path: PathBuf,
}

fn collect_artifacts(
    refs: &PsionicTrainWindowArtifactInputRefs,
) -> Result<Vec<PsionicTrainContributionArtifact>, PsionicTrainWindowArtifactError> {
    let candidates = [
        ("invocation_manifest", Some(refs.invocation_manifest_path.as_str())),
        ("launch_manifest", refs.launch_manifest_path.as_deref()),
        ("membership_revision", refs.membership_revision_path.as_deref()),
        ("checkpoint_surface", refs.checkpoint_surface_path.as_deref()),
        ("checkpoint_pointer", refs.checkpoint_pointer_path.as_deref()),
        ("checkpoint_manifest", refs.checkpoint_manifest_path.as_deref()),
        (
            "checkpoint_backup_receipt",
            refs.checkpoint_backup_receipt_path.as_deref(),
        ),
        (
            "checkpoint_handoff_receipt",
            refs.checkpoint_handoff_receipt_path.as_deref(),
        ),
        ("recovery_receipt", refs.recovery_receipt_path.as_deref()),
        ("current_status", refs.current_status_path.as_deref()),
        ("retained_summary", refs.retained_summary_path.as_deref()),
        ("launcher_log", refs.launcher_log_path.as_deref()),
        (
            "final_closeout_bundle",
            refs.final_closeout_bundle_path.as_deref(),
        ),
    ];
    let mut artifacts = Vec::new();
    for (kind, path) in candidates {
        let Some(path) = path else {
            continue;
        };
        let path = Path::new(path);
        if !path.is_file() {
            continue;
        }
        let bytes = fs::read(path).map_err(|error| PsionicTrainWindowArtifactError::Read {
            path: path.display().to_string(),
            detail: error.to_string(),
        })?;
        let artifact_bytes = u64::try_from(bytes.len()).map_err(|error| {
            PsionicTrainWindowArtifactError::Read {
                path: path.display().to_string(),
                detail: error.to_string(),
            }
        })?;
        artifacts.push(PsionicTrainContributionArtifact {
            artifact_kind: String::from(kind),
            artifact_path: path.display().to_string(),
            artifact_sha256: sha256_hex(bytes.as_slice()),
            artifact_bytes,
        });
    }
    artifacts.sort_by(|left, right| left.artifact_kind.cmp(&right.artifact_kind));
    Ok(artifacts)
}

fn load_window_contributions(
    window_root: &Path,
) -> Result<Vec<RetainedContributionEntry>, PsionicTrainWindowArtifactError> {
    let contributions_root = window_root.join("contributions");
    if !contributions_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for entry in fs::read_dir(&contributions_root).map_err(|error| {
        PsionicTrainWindowArtifactError::Read {
            path: contributions_root.display().to_string(),
            detail: error.to_string(),
        }
    })? {
        let entry = entry.map_err(|error| PsionicTrainWindowArtifactError::Read {
            path: contributions_root.display().to_string(),
            detail: error.to_string(),
        })?;
        let receipt_path = entry.path().join("contribution_receipt.json");
        if !receipt_path.is_file() {
            continue;
        }
        let contribution_receipt: PsionicTrainContributionReceipt =
            read_json(receipt_path.as_path())?;
        entries.push(RetainedContributionEntry {
            assignment_id: contribution_receipt.assignment_id,
            contribution_id: contribution_receipt.contribution_id,
            node_pubkey: contribution_receipt.node_pubkey,
            outcome: contribution_receipt.outcome,
            artifact_manifest_digest: contribution_receipt.artifact_manifest_digest,
            contribution_digest: contribution_receipt.contribution_digest,
            receipt_path,
        });
    }
    Ok(entries)
}

fn role_label(role: PsionicTrainRole) -> &'static str {
    match role {
        PsionicTrainRole::Worker => "worker",
        PsionicTrainRole::Validator => "validator",
        PsionicTrainRole::RecoverySource => "recovery_source",
    }
}

fn stable_id(prefix: &[u8], parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(prefix);
    for part in parts {
        digest.update(part.as_bytes());
        digest.update(b"|");
    }
    format!("{:x}", digest.finalize())
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let bytes =
        serde_json::to_vec(value).expect("psionic train window artifacts should serialize");
    let mut digest = Sha256::new();
    digest.update(prefix);
    digest.update(&bytes);
    format!("{:x}", digest.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn write_json<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), PsionicTrainWindowArtifactError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| PsionicTrainWindowArtifactError::Write {
            path: parent.display().to_string(),
            detail: error.to_string(),
        })?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|error| {
            PsionicTrainWindowArtifactError::Write {
                path: path.display().to_string(),
                detail: error.to_string(),
            }
        })?,
    )
    .map_err(|error| PsionicTrainWindowArtifactError::Write {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}

fn read_json<T: for<'de> Deserialize<'de>>(
    path: &Path,
) -> Result<T, PsionicTrainWindowArtifactError> {
    let bytes = fs::read(path).map_err(|error| PsionicTrainWindowArtifactError::Read {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| PsionicTrainWindowArtifactError::Parse {
        path: path.display().to_string(),
        detail: error.to_string(),
    })
}
