use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{OPEN_ADAPTER_MLX_METAL_BACKEND_LABEL, SWARM_MAC_MLX_BRINGUP_FIXTURE_PATH};

/// Stable schema version for the executor admitted-profile catalog.
pub const PSION_EXECUTOR_ADMITTED_PROFILE_CATALOG_SCHEMA_VERSION: &str =
    "psion.executor_admitted_profile_catalog.v1";
/// Canonical fixture path for the admitted-profile catalog.
pub const PSION_EXECUTOR_ADMITTED_PROFILE_CATALOG_FIXTURE_PATH: &str =
    "fixtures/psion/executor/psion_executor_admitted_profiles_v1.json";
/// Canonical doc path for the admitted local profiles.
pub const PSION_EXECUTOR_LOCAL_PROFILE_DOC_PATH: &str =
    "docs/PSION_EXECUTOR_LOCAL_PROFILE_REFERENCE.md";

const CROSS_PROVIDER_LOCAL_MLX_MAC_COMPUTE_SOURCE_FIXTURE_PATH: &str =
    "fixtures/training/compute_sources/local_mlx_mac_workstation_v1.json";

/// Run types admitted by the phase-one executor lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionExecutorRunType {
    MlxSmoke,
    MlxDecisionGrade,
    Cuda4080Smoke,
    Cuda4080DecisionGrade,
    Cuda4080Confirmation,
    CpuValidation,
    H100Escalation,
}

/// Admission posture for one run type on one admitted profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsionExecutorRunTypeAdmissionPosture {
    Primary,
    Allowed,
    NotAdmitted,
}

/// One run-type admission row for one profile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionExecutorRunTypeAdmission {
    /// Stable run type id.
    pub run_type: PsionExecutorRunType,
    /// Final admission posture on this profile.
    pub posture: PsionExecutorRunTypeAdmissionPosture,
    /// Short operator-facing detail.
    pub detail: String,
}

/// Repo-owned authority artifact used to admit one executor profile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionExecutorAuthorityArtifact {
    /// Repo-local path.
    pub path: String,
    /// Stable SHA256 over the artifact bytes.
    pub sha256: String,
    /// Why the artifact matters to this profile.
    pub detail: String,
}

/// One admitted executor profile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionExecutorAdmittedProfile {
    /// Stable profile id.
    pub profile_id: String,
    /// Short purpose statement.
    pub purpose: String,
    /// Runtime backend label if the profile maps to one backend directly.
    pub runtime_backend_label: Option<String>,
    /// Admitted run-type posture on this profile.
    pub run_type_admissions: Vec<PsionExecutorRunTypeAdmission>,
    /// Local operator requirements that must remain true before a run counts.
    pub local_requirements: Vec<String>,
    /// Checkpoint expectations for this profile.
    pub checkpoint_expectations: String,
    /// Connectivity expectations for this profile.
    pub connectivity_expectations: Vec<String>,
    /// Shipped entrypoints this profile is allowed to use.
    pub shipped_entrypoints: Vec<String>,
    /// Authority artifacts proving the profile boundary.
    pub authority_artifacts: Vec<PsionExecutorAuthorityArtifact>,
    /// Explicit claim boundary.
    pub claim_boundary: String,
    /// Stable digest over the profile.
    pub profile_digest: String,
}

/// Catalog of admitted phase-one executor profiles.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PsionExecutorAdmittedProfileCatalog {
    /// Stable schema version.
    pub schema_version: String,
    /// Stable catalog id.
    pub catalog_id: String,
    /// Ordered admitted profiles.
    pub profiles: Vec<PsionExecutorAdmittedProfile>,
    /// Short explanation of the catalog.
    pub summary: String,
    /// Stable digest over the catalog.
    pub catalog_digest: String,
}

impl PsionExecutorAdmittedProfileCatalog {
    /// Validate catalog structure and digests.
    pub fn validate(&self) -> Result<(), PsionExecutorAdmittedProfileError> {
        ensure_nonempty(
            self.schema_version.as_str(),
            "psion_executor_admitted_profile_catalog.schema_version",
        )?;
        if self.schema_version != PSION_EXECUTOR_ADMITTED_PROFILE_CATALOG_SCHEMA_VERSION {
            return Err(PsionExecutorAdmittedProfileError::SchemaVersionMismatch {
                expected: String::from(PSION_EXECUTOR_ADMITTED_PROFILE_CATALOG_SCHEMA_VERSION),
                actual: self.schema_version.clone(),
            });
        }
        ensure_nonempty(
            self.catalog_id.as_str(),
            "psion_executor_admitted_profile_catalog.catalog_id",
        )?;
        ensure_nonempty(
            self.summary.as_str(),
            "psion_executor_admitted_profile_catalog.summary",
        )?;
        if self.profiles.is_empty() {
            return Err(PsionExecutorAdmittedProfileError::MissingField {
                field: String::from("psion_executor_admitted_profile_catalog.profiles"),
            });
        }
        let mut seen_profiles = BTreeSet::new();
        for profile in &self.profiles {
            profile.validate()?;
            if !seen_profiles.insert(profile.profile_id.as_str()) {
                return Err(PsionExecutorAdmittedProfileError::DuplicateProfile {
                    profile_id: profile.profile_id.clone(),
                });
            }
        }
        if self.catalog_digest != stable_executor_profile_catalog_digest(self) {
            return Err(PsionExecutorAdmittedProfileError::DigestMismatch {
                kind: String::from("psion_executor_admitted_profile_catalog"),
            });
        }
        Ok(())
    }
}

impl PsionExecutorAdmittedProfile {
    fn validate(&self) -> Result<(), PsionExecutorAdmittedProfileError> {
        ensure_nonempty(
            self.profile_id.as_str(),
            "psion_executor_admitted_profile.profile_id",
        )?;
        ensure_nonempty(
            self.purpose.as_str(),
            "psion_executor_admitted_profile.purpose",
        )?;
        if self.run_type_admissions.is_empty() {
            return Err(PsionExecutorAdmittedProfileError::MissingField {
                field: format!(
                    "psion_executor_admitted_profile[{}].run_type_admissions",
                    self.profile_id
                ),
            });
        }
        let mut seen_run_types = BTreeSet::new();
        for admission in &self.run_type_admissions {
            ensure_nonempty(
                admission.detail.as_str(),
                "psion_executor_admitted_profile.run_type_admissions[].detail",
            )?;
            if !seen_run_types.insert(admission.run_type) {
                return Err(
                    PsionExecutorAdmittedProfileError::DuplicateRunTypeAdmission {
                        profile_id: self.profile_id.clone(),
                        run_type: format!("{:?}", admission.run_type),
                    },
                );
            }
        }
        if self.local_requirements.is_empty() {
            return Err(PsionExecutorAdmittedProfileError::MissingField {
                field: format!(
                    "psion_executor_admitted_profile[{}].local_requirements",
                    self.profile_id
                ),
            });
        }
        for requirement in &self.local_requirements {
            ensure_nonempty(
                requirement.as_str(),
                "psion_executor_admitted_profile.local_requirements[]",
            )?;
        }
        ensure_nonempty(
            self.checkpoint_expectations.as_str(),
            "psion_executor_admitted_profile.checkpoint_expectations",
        )?;
        if self.connectivity_expectations.is_empty() {
            return Err(PsionExecutorAdmittedProfileError::MissingField {
                field: format!(
                    "psion_executor_admitted_profile[{}].connectivity_expectations",
                    self.profile_id
                ),
            });
        }
        for expectation in &self.connectivity_expectations {
            ensure_nonempty(
                expectation.as_str(),
                "psion_executor_admitted_profile.connectivity_expectations[]",
            )?;
        }
        if self.shipped_entrypoints.is_empty() {
            return Err(PsionExecutorAdmittedProfileError::MissingField {
                field: format!(
                    "psion_executor_admitted_profile[{}].shipped_entrypoints",
                    self.profile_id
                ),
            });
        }
        for entrypoint in &self.shipped_entrypoints {
            ensure_nonempty(
                entrypoint.as_str(),
                "psion_executor_admitted_profile.shipped_entrypoints[]",
            )?;
        }
        if self.authority_artifacts.is_empty() {
            return Err(PsionExecutorAdmittedProfileError::MissingField {
                field: format!(
                    "psion_executor_admitted_profile[{}].authority_artifacts",
                    self.profile_id
                ),
            });
        }
        for artifact in &self.authority_artifacts {
            ensure_nonempty(
                artifact.path.as_str(),
                "psion_executor_admitted_profile.authority_artifacts[].path",
            )?;
            ensure_nonempty(
                artifact.sha256.as_str(),
                "psion_executor_admitted_profile.authority_artifacts[].sha256",
            )?;
            ensure_nonempty(
                artifact.detail.as_str(),
                "psion_executor_admitted_profile.authority_artifacts[].detail",
            )?;
        }
        ensure_nonempty(
            self.claim_boundary.as_str(),
            "psion_executor_admitted_profile.claim_boundary",
        )?;
        if self.profile_digest != stable_executor_profile_digest(self) {
            return Err(PsionExecutorAdmittedProfileError::DigestMismatch {
                kind: format!("psion_executor_admitted_profile.{}", self.profile_id),
            });
        }
        Ok(())
    }
}

/// Errors surfaced while building or validating executor admitted profiles.
#[derive(Debug, Error)]
pub enum PsionExecutorAdmittedProfileError {
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error(transparent)]
    Serialize(#[from] serde_json::Error),
    #[error("missing required field `{field}`")]
    MissingField { field: String },
    #[error("schema version mismatch: expected `{expected}`, got `{actual}`")]
    SchemaVersionMismatch { expected: String, actual: String },
    #[error("duplicate profile `{profile_id}`")]
    DuplicateProfile { profile_id: String },
    #[error("duplicate run-type admission `{run_type}` in profile `{profile_id}`")]
    DuplicateRunTypeAdmission {
        profile_id: String,
        run_type: String,
    },
    #[error("digest mismatch for `{kind}`")]
    DigestMismatch { kind: String },
}

/// Build the current canonical executor admitted-profile catalog.
pub fn builtin_executor_admitted_profile_catalog(
    workspace_root: &Path,
) -> Result<PsionExecutorAdmittedProfileCatalog, PsionExecutorAdmittedProfileError> {
    let profiles = vec![builtin_local_mac_mlx_profile(workspace_root)?];
    let mut catalog = PsionExecutorAdmittedProfileCatalog {
        schema_version: String::from(PSION_EXECUTOR_ADMITTED_PROFILE_CATALOG_SCHEMA_VERSION),
        catalog_id: String::from("psion_executor_admitted_profiles_v1"),
        profiles,
        summary: String::from(
            "Phase-one executor admitted-profile catalog freezing the local Mac MLX lane before 4080 and control-plane profiles widen the same catalog.",
        ),
        catalog_digest: String::new(),
    };
    catalog.catalog_digest = stable_executor_profile_catalog_digest(&catalog);
    catalog.validate()?;
    Ok(catalog)
}

/// Write the current admitted-profile catalog fixture.
pub fn write_builtin_executor_admitted_profile_catalog(
    workspace_root: &Path,
) -> Result<PsionExecutorAdmittedProfileCatalog, PsionExecutorAdmittedProfileError> {
    let catalog = builtin_executor_admitted_profile_catalog(workspace_root)?;
    let fixture_path = workspace_root.join(PSION_EXECUTOR_ADMITTED_PROFILE_CATALOG_FIXTURE_PATH);
    if let Some(parent) = fixture_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            PsionExecutorAdmittedProfileError::CreateDir {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    fs::write(&fixture_path, serde_json::to_vec_pretty(&catalog)?).map_err(|error| {
        PsionExecutorAdmittedProfileError::Write {
            path: fixture_path.display().to_string(),
            error,
        }
    })?;
    Ok(catalog)
}

fn builtin_local_mac_mlx_profile(
    workspace_root: &Path,
) -> Result<PsionExecutorAdmittedProfile, PsionExecutorAdmittedProfileError> {
    let authority_artifacts = vec![
        authority_artifact(
            workspace_root,
            SWARM_MAC_MLX_BRINGUP_FIXTURE_PATH,
            "Retained MLX bring-up report proving the bounded Mac Metal backend, machine envelope, and same-node bring-up gate.",
        )?,
        authority_artifact(
            workspace_root,
            CROSS_PROVIDER_LOCAL_MLX_MAC_COMPUTE_SOURCE_FIXTURE_PATH,
            "Shared compute-source contract proving the Mac can be admitted under the existing cross-provider train substrate without inventing a second launcher or machine vocabulary.",
        )?,
    ];
    let mut profile = PsionExecutorAdmittedProfile {
        profile_id: String::from("local_mac_mlx_aarch64"),
        purpose: String::from(
            "Local Apple Silicon MLX machine for executor smoke training, short training runs, eval-pack execution, checkpoint restore rehearsal, export inspection, and CPU-validation ownership.",
        ),
        runtime_backend_label: Some(String::from(OPEN_ADAPTER_MLX_METAL_BACKEND_LABEL)),
        run_type_admissions: vec![
            PsionExecutorRunTypeAdmission {
                run_type: PsionExecutorRunType::MlxSmoke,
                posture: PsionExecutorRunTypeAdmissionPosture::Primary,
                detail: String::from(
                    "This is the primary admitted machine for MLX smoke runs and local bring-up validation.",
                ),
            },
            PsionExecutorRunTypeAdmission {
                run_type: PsionExecutorRunType::MlxDecisionGrade,
                posture: PsionExecutorRunTypeAdmissionPosture::Allowed,
                detail: String::from(
                    "Decision-grade MLX runs are allowed only when the question is explicitly MLX-local and the frozen pack or subset is declared before launch.",
                ),
            },
            PsionExecutorRunTypeAdmission {
                run_type: PsionExecutorRunType::Cuda4080Smoke,
                posture: PsionExecutorRunTypeAdmissionPosture::NotAdmitted,
                detail: String::from(
                    "The Mac profile does not count as the CUDA smoke lane.",
                ),
            },
            PsionExecutorRunTypeAdmission {
                run_type: PsionExecutorRunType::Cuda4080DecisionGrade,
                posture: PsionExecutorRunTypeAdmissionPosture::NotAdmitted,
                detail: String::from(
                    "The Mac profile does not count as the 4080 decision-grade lane.",
                ),
            },
            PsionExecutorRunTypeAdmission {
                run_type: PsionExecutorRunType::Cuda4080Confirmation,
                posture: PsionExecutorRunTypeAdmissionPosture::NotAdmitted,
                detail: String::from(
                    "The Mac profile does not count as the 4080 confirmation lane.",
                ),
            },
            PsionExecutorRunTypeAdmission {
                run_type: PsionExecutorRunType::CpuValidation,
                posture: PsionExecutorRunTypeAdmissionPosture::Primary,
                detail: String::from(
                    "The Mac profile is a primary owner for CPU-matrix replacement validation, export inspection, and restore rehearsal.",
                ),
            },
            PsionExecutorRunTypeAdmission {
                run_type: PsionExecutorRunType::H100Escalation,
                posture: PsionExecutorRunTypeAdmissionPosture::NotAdmitted,
                detail: String::from(
                    "The local Mac profile does not count as the H100 escalation lane.",
                ),
            },
        ],
        local_requirements: vec![
            String::from(
                "Run the retained MLX bring-up gate before roadmap-tracked smoke or decision-grade work.",
            ),
            String::from(
                "Keep the run local-first: this profile does not claim remote checkpoint-writer or remote cluster-port authority by itself.",
            ),
            String::from(
                "Use only shipped Psionic entrypoints when the profile is cited in roadmap-tracked evidence.",
            ),
        ],
        checkpoint_expectations: String::from(
            "Checkpoint restore rehearsal and export inspection must stay inside the local operator-owned workspace; this profile does not claim shared checkpoint-writer authority by itself.",
        ),
        connectivity_expectations: vec![
            String::from(
                "No remote launch or cluster-port claim is inherited from the Mac profile alone.",
            ),
            String::from(
                "The profile remains valid for local eval and control-plane ownership even when the later Tailnet worker is offline.",
            ),
        ],
        shipped_entrypoints: vec![
            String::from("scripts/check-swarm-mac-mlx-bringup.sh"),
            String::from("crates/psionic-train/src/swarm_mlx_bringup.rs"),
            String::from("crates/psionic-train/src/bin/swarm_mac_mlx_bringup.rs"),
        ],
        authority_artifacts,
        claim_boundary: String::from(
            "This profile admits the local Apple Silicon MLX machine as a real roadmap-tracked executor development and eval host. It proves local MLX smoke, short-run, eval, restore, export, and CPU-validation posture. It does not by itself claim remote launch, shared checkpoint authority, or cross-device training closure.",
        ),
        profile_digest: String::new(),
    };
    profile.profile_digest = stable_executor_profile_digest(&profile);
    profile.validate()?;
    Ok(profile)
}

fn authority_artifact(
    workspace_root: &Path,
    rel_path: &str,
    detail: &str,
) -> Result<PsionExecutorAuthorityArtifact, PsionExecutorAdmittedProfileError> {
    Ok(PsionExecutorAuthorityArtifact {
        path: String::from(rel_path),
        sha256: sha256_for_path(workspace_root.join(rel_path))?,
        detail: String::from(detail),
    })
}

fn sha256_for_path(path: PathBuf) -> Result<String, PsionExecutorAdmittedProfileError> {
    let bytes = fs::read(&path).map_err(|error| PsionExecutorAdmittedProfileError::Read {
        path: path.display().to_string(),
        error,
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn ensure_nonempty(value: &str, field: &str) -> Result<(), PsionExecutorAdmittedProfileError> {
    if value.trim().is_empty() {
        return Err(PsionExecutorAdmittedProfileError::MissingField {
            field: String::from(field),
        });
    }
    Ok(())
}

fn stable_executor_profile_digest(profile: &PsionExecutorAdmittedProfile) -> String {
    let mut clone = profile.clone();
    clone.profile_digest.clear();
    stable_json_digest(&clone)
}

fn stable_executor_profile_catalog_digest(catalog: &PsionExecutorAdmittedProfileCatalog) -> String {
    let mut clone = catalog.clone();
    clone.catalog_digest.clear();
    stable_json_digest(&clone)
}

fn stable_json_digest<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("executor admitted-profile digest serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .map(PathBuf::from)
            .expect("workspace root")
    }

    #[test]
    fn builtin_executor_profile_catalog_matches_committed_fixture() {
        let root = workspace_root();
        let built = builtin_executor_admitted_profile_catalog(&root).expect("built catalog");
        let fixture: PsionExecutorAdmittedProfileCatalog = serde_json::from_slice(
            &fs::read(root.join(PSION_EXECUTOR_ADMITTED_PROFILE_CATALOG_FIXTURE_PATH))
                .expect("fixture bytes"),
        )
        .expect("fixture json");
        assert_eq!(built, fixture);
    }

    #[test]
    fn builtin_executor_profile_catalog_is_valid() {
        let root = workspace_root();
        let catalog = builtin_executor_admitted_profile_catalog(&root).expect("catalog");
        catalog.validate().expect("catalog should validate");
        assert_eq!(catalog.profiles.len(), 1);
        assert_eq!(catalog.profiles[0].profile_id, "local_mac_mlx_aarch64");
    }
}
