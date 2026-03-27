use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const PARAMETER_GOLF_HOMEGOLF_CLUSTERED_RUN_SURFACE_FIXTURE_PATH: &str =
    "fixtures/parameter_golf/reports/parameter_golf_homegolf_clustered_run_surface.json";
pub const PARAMETER_GOLF_HOMEGOLF_CLUSTERED_RUN_SURFACE_CHECKER: &str =
    "scripts/check-parameter-golf-homegolf-clustered-run-surface.sh";
pub const PARAMETER_GOLF_HOMEGOLF_CLUSTERED_RUN_SURFACE_AUDIT: &str =
    "docs/audits/2026-03-27-homegolf-clustered-run-surface.md";

const PARAMETER_GOLF_HOMEGOLF_TRACK_CONTRACT_REF: &str =
    "fixtures/parameter_golf/reports/parameter_golf_homegolf_track_contract.json";
const TAILRUN_ADMITTED_HOME_SUMMARY_REF: &str =
    "fixtures/swarm/runs/tailrun-home-admitted-20260327e/tailrun_admitted_home_run_summary.json";
const PARAMETER_GOLF_HOMEGOLF_DENSE_BUNDLE_PROOF_REF: &str =
    "fixtures/parameter_golf/reports/parameter_golf_homegolf_dense_bundle_proof.json";

#[derive(Debug, Error)]
pub enum ParameterGolfHomegolfClusteredRunSurfaceError {
    #[error("failed to read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("failed to create `{path}`: {error}")]
    CreateDir { path: String, error: std::io::Error },
    #[error("failed to write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("invalid clustered HOMEGOLF surface: {detail}")]
    InvalidSurface { detail: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGolfHomegolfClusteredRunSurfaceStatus {
    BoundedComposedSurface,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfHomegolfClusteredRunContribution {
    pub node_id: String,
    pub runtime_role: String,
    pub role_id: String,
    pub execution_backend_label: String,
    pub endpoint: String,
    pub observed_wallclock_ms: u64,
    pub local_execution_wallclock_ms: u64,
    pub executed_steps: u64,
    pub batch_count: u64,
    pub sample_count: u64,
    pub payload_bytes: u64,
    pub final_mean_loss: f64,
    pub contributor_receipt_digest: String,
    pub estimated_steps_per_second: f64,
    pub estimated_samples_per_second: f64,
    pub contribution_share: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParameterGolfHomegolfClusteredRunSurfaceReport {
    pub schema_version: u16,
    pub report_id: String,
    pub track_id: String,
    pub run_id: String,
    pub wallclock_cap_seconds: u64,
    pub observed_cluster_wallclock_ms: u64,
    pub source_track_contract_ref: String,
    pub source_admitted_home_run_summary_ref: String,
    pub source_dense_bundle_proof_ref: String,
    pub admitted_device_set: Vec<String>,
    pub per_device_contributions: Vec<ParameterGolfHomegolfClusteredRunContribution>,
    pub merge_disposition: String,
    pub publish_disposition: String,
    pub promotion_disposition: String,
    pub merged_bundle_descriptor_digest: String,
    pub merged_bundle_tokenizer_digest: String,
    pub final_validation_mean_loss: f64,
    pub final_validation_bits_per_byte: f64,
    pub model_artifact_bytes: u64,
    pub prompt_text: String,
    pub direct_generated_tokens: Vec<u32>,
    pub served_generated_tokens: Vec<u32>,
    pub direct_and_served_match: bool,
    pub surface_status: ParameterGolfHomegolfClusteredRunSurfaceStatus,
    pub claim_boundary: String,
    pub summary: String,
    pub report_digest: String,
}

#[derive(Debug, Deserialize)]
struct TailrunAdmittedHomeSummary {
    run_id: String,
    admitted_device_set: Vec<String>,
    per_device_contributions: Vec<ParameterGolfHomegolfClusteredRunContribution>,
    merge_disposition: String,
    publish_disposition: String,
    promotion_disposition: String,
}

#[derive(Debug, Deserialize)]
struct HomegolfDenseBundleProof {
    track_id: String,
    run_id: String,
    descriptor_digest: String,
    tokenizer_digest: String,
    final_validation_mean_loss: f64,
    final_validation_bits_per_byte: f64,
    model_artifact_bytes: u64,
    prompt_text: String,
    direct_generated_tokens: Vec<u32>,
    served_generated_tokens: Vec<u32>,
    direct_and_served_match: bool,
}

impl ParameterGolfHomegolfClusteredRunSurfaceReport {
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut clone = self.clone();
        clone.report_digest.clear();
        stable_digest(
            b"psionic_parameter_golf_homegolf_clustered_run_surface|",
            &clone,
        )
    }

    pub fn validate(&self) -> Result<(), ParameterGolfHomegolfClusteredRunSurfaceError> {
        if self.schema_version != 1 {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: format!("schema_version must stay 1 but was {}", self.schema_version),
            });
        }
        if self.wallclock_cap_seconds != 600 {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: String::from("wallclock_cap_seconds must stay 600"),
            });
        }
        if self.observed_cluster_wallclock_ms == 0
            || self.observed_cluster_wallclock_ms > self.wallclock_cap_seconds * 1_000
        {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: String::from(
                    "observed_cluster_wallclock_ms must stay positive and within the 600s cap",
                ),
            });
        }
        if self.admitted_device_set
            != vec![
                String::from("local_m5_mlx"),
                String::from("archlinux_rtx4080_cuda"),
            ]
        {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: String::from(
                    "admitted_device_set must retain the current two-device home admission set",
                ),
            });
        }
        if self.per_device_contributions.len() != 2 {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: String::from(
                    "per_device_contributions must retain exactly two admitted-device receipts",
                ),
            });
        }
        if self.merge_disposition != "merged"
            || self.publish_disposition != "refused"
            || self.promotion_disposition != "held"
        {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: String::from(
                    "merge, publish, and promotion dispositions drifted from retained clustered-home truth",
                ),
            });
        }
        if self.model_artifact_bytes == 0 || self.final_validation_bits_per_byte <= 0.0 {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: String::from(
                    "model_artifact_bytes and final_validation_bits_per_byte must stay positive",
                ),
            });
        }
        if !self.direct_and_served_match
            || self.direct_generated_tokens != self.served_generated_tokens
        {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: String::from(
                    "direct and served generation must stay matched in the retained dense bundle proof",
                ),
            });
        }
        if self.surface_status
            != ParameterGolfHomegolfClusteredRunSurfaceStatus::BoundedComposedSurface
        {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: String::from("surface_status drifted"),
            });
        }
        if self.report_digest != self.stable_digest() {
            return Err(ParameterGolfHomegolfClusteredRunSurfaceError::InvalidSurface {
                detail: String::from("report_digest drifted"),
            });
        }
        Ok(())
    }
}

pub fn build_parameter_golf_homegolf_clustered_run_surface_report(
) -> Result<ParameterGolfHomegolfClusteredRunSurfaceReport, ParameterGolfHomegolfClusteredRunSurfaceError>
{
    let tailrun_summary: TailrunAdmittedHomeSummary = serde_json::from_slice(
        &fs::read(resolve_repo_path(TAILRUN_ADMITTED_HOME_SUMMARY_REF)).map_err(|error| {
            ParameterGolfHomegolfClusteredRunSurfaceError::Read {
                path: String::from(TAILRUN_ADMITTED_HOME_SUMMARY_REF),
                error,
            }
        })?,
    )?;
    let bundle_proof: HomegolfDenseBundleProof = serde_json::from_slice(
        &fs::read(resolve_repo_path(PARAMETER_GOLF_HOMEGOLF_DENSE_BUNDLE_PROOF_REF)).map_err(
            |error| {
            ParameterGolfHomegolfClusteredRunSurfaceError::Read {
                path: String::from(PARAMETER_GOLF_HOMEGOLF_DENSE_BUNDLE_PROOF_REF),
                error,
            }
        })?,
    )?;

    let observed_cluster_wallclock_ms = tailrun_summary
        .per_device_contributions
        .iter()
        .map(|contribution| contribution.observed_wallclock_ms)
        .max()
        .unwrap_or(0);

    let mut report = ParameterGolfHomegolfClusteredRunSurfaceReport {
        schema_version: 1,
        report_id: String::from("parameter_golf.homegolf_clustered_run_surface.v1"),
        track_id: bundle_proof.track_id,
        run_id: format!(
            "{}+{}",
            tailrun_summary.run_id, bundle_proof.run_id
        ),
        wallclock_cap_seconds: 600,
        observed_cluster_wallclock_ms,
        source_track_contract_ref: String::from(PARAMETER_GOLF_HOMEGOLF_TRACK_CONTRACT_REF),
        source_admitted_home_run_summary_ref: String::from(TAILRUN_ADMITTED_HOME_SUMMARY_REF),
        source_dense_bundle_proof_ref: String::from(PARAMETER_GOLF_HOMEGOLF_DENSE_BUNDLE_PROOF_REF),
        admitted_device_set: tailrun_summary.admitted_device_set,
        per_device_contributions: tailrun_summary.per_device_contributions,
        merge_disposition: tailrun_summary.merge_disposition,
        publish_disposition: tailrun_summary.publish_disposition,
        promotion_disposition: tailrun_summary.promotion_disposition,
        merged_bundle_descriptor_digest: bundle_proof.descriptor_digest,
        merged_bundle_tokenizer_digest: bundle_proof.tokenizer_digest,
        final_validation_mean_loss: bundle_proof.final_validation_mean_loss,
        final_validation_bits_per_byte: bundle_proof.final_validation_bits_per_byte,
        model_artifact_bytes: bundle_proof.model_artifact_bytes,
        prompt_text: bundle_proof.prompt_text,
        direct_generated_tokens: bundle_proof.direct_generated_tokens,
        served_generated_tokens: bundle_proof.served_generated_tokens,
        direct_and_served_match: bundle_proof.direct_and_served_match,
        surface_status: ParameterGolfHomegolfClusteredRunSurfaceStatus::BoundedComposedSurface,
        claim_boundary: String::from(
            "This is the first honest clustered HOMEGOLF score surface, not a fake claim that one exact dense mixed-device home-cluster run already produced the scored bundle directly. The admitted-device Tailnet run provides real home-cluster contribution truth under the 600s cap, while the exact 9x512 HOMEGOLF-compatible bundle proof provides the current merged inferable artifact and final val_bpb. The report composes those adjacent retained surfaces explicitly so operators can compare clustered-device work and exact-family score progress without overstating live dense mixed-device closure.",
        ),
        summary: String::from(
            "The current HOMEGOLF clustered surface combines one real two-device home-Tailnet admitted run and one real exact-family HOMEGOLF train-to-infer proof. It freezes the current admitted device inventory, contribution receipts, merged bundle digests, and final val_bpb into one explicit bounded-composed report while live exact dense mixed-device execution remains follow-on work.",
        ),
        report_digest: String::new(),
    };
    report.report_digest = report.stable_digest();
    report.validate()?;
    Ok(report)
}

pub fn write_parameter_golf_homegolf_clustered_run_surface_report(
    output_path: &Path,
) -> Result<ParameterGolfHomegolfClusteredRunSurfaceReport, ParameterGolfHomegolfClusteredRunSurfaceError>
{
    let report = build_parameter_golf_homegolf_clustered_run_surface_report()?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ParameterGolfHomegolfClusteredRunSurfaceError::CreateDir {
                path: parent.display().to_string(),
                error,
            }
        })?;
    }
    let bytes = serde_json::to_vec_pretty(&report)?;
    fs::write(output_path, bytes).map_err(|error| {
        ParameterGolfHomegolfClusteredRunSurfaceError::Write {
            path: output_path.display().to_string(),
            error,
        }
    })?;
    Ok(report)
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(
        serde_json::to_vec(value).expect("clustered HOMEGOLF surface should serialize"),
    );
    format!("{:x}", hasher.finalize())
}

fn resolve_repo_path(relpath: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(relpath)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        build_parameter_golf_homegolf_clustered_run_surface_report,
        write_parameter_golf_homegolf_clustered_run_surface_report,
        ParameterGolfHomegolfClusteredRunSurfaceStatus,
        PARAMETER_GOLF_HOMEGOLF_CLUSTERED_RUN_SURFACE_FIXTURE_PATH,
    };

    #[test]
    fn clustered_homegolf_surface_keeps_current_admitted_home_and_bundle_truth() {
        let report =
            build_parameter_golf_homegolf_clustered_run_surface_report().expect("build report");
        assert_eq!(report.wallclock_cap_seconds, 600);
        assert_eq!(
            report.surface_status,
            ParameterGolfHomegolfClusteredRunSurfaceStatus::BoundedComposedSurface
        );
        assert_eq!(report.admitted_device_set.len(), 2);
        assert!(report.observed_cluster_wallclock_ms <= 600_000);
        assert!(report.final_validation_bits_per_byte > 0.0);
        assert!(report.model_artifact_bytes > 0);
        assert!(report.direct_and_served_match);
    }

    #[test]
    fn write_clustered_homegolf_surface_roundtrips() {
        let output = tempfile::tempdir().expect("tempdir");
        let path = output
            .path()
            .join("parameter_golf_homegolf_clustered_run_surface.json");
        let written = write_parameter_golf_homegolf_clustered_run_surface_report(path.as_path())
            .expect("write report");
        let encoded = std::fs::read(path.as_path()).expect("read report");
        let decoded: super::ParameterGolfHomegolfClusteredRunSurfaceReport =
            serde_json::from_slice(&encoded).expect("decode report");
        assert_eq!(written, decoded);
    }

    #[test]
    fn committed_clustered_homegolf_surface_fixture_roundtrips() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(PARAMETER_GOLF_HOMEGOLF_CLUSTERED_RUN_SURFACE_FIXTURE_PATH);
        let encoded = std::fs::read(fixture).expect("read fixture");
        let decoded: super::ParameterGolfHomegolfClusteredRunSurfaceReport =
            serde_json::from_slice(&encoded).expect("decode fixture");
        let rebuilt =
            build_parameter_golf_homegolf_clustered_run_surface_report().expect("rebuild report");
        assert_eq!(decoded, rebuilt);
    }
}
