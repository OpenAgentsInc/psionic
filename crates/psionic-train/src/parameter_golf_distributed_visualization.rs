use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use psionic_eval::{
    ParameterGolfDistributedLaneDisposition, ParameterGolfDistributedThroughputReceipt,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    build_remote_training_run_index, record_remote_training_visualization_bundle,
    RemoteTrainingArtifactSourceKind, RemoteTrainingDistributedSample, RemoteTrainingEmissionMode,
    RemoteTrainingEventSample, RemoteTrainingEventSeverity, RemoteTrainingGpuSample,
    RemoteTrainingHeartbeatSample, RemoteTrainingProvider, RemoteTrainingRefreshContract,
    RemoteTrainingResultClassification, RemoteTrainingRunIndex, RemoteTrainingRunIndexEntry,
    RemoteTrainingRuntimeSample, RemoteTrainingSeriesStatus, RemoteTrainingSourceArtifact,
    RemoteTrainingTimelineEntry, RemoteTrainingVisualizationBundle,
    RemoteTrainingVisualizationError, RemoteTrainingVisualizationSummary,
    REMOTE_TRAINING_TARGET_UI_UPDATE_INTERVAL_MS,
};

pub const PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_DIR_NAME: &str = "training_visualization";
pub const PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_BUNDLE_NAME: &str =
    "parameter_golf_distributed_8xh100_remote_training_visualization_bundle_v1.json";
pub const PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_RUN_INDEX_NAME: &str =
    "remote_training_run_index_v1.json";
pub const PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_SNAPSHOT_DIR_NAME: &str = "snapshots";

#[derive(Debug, Error)]
pub enum ParameterGolfDistributedVisualizationError {
    #[error("parameter golf distributed visualization could not read `{path}`: {error}")]
    Read { path: String, error: std::io::Error },
    #[error("parameter golf distributed visualization could not write `{path}`: {error}")]
    Write { path: String, error: std::io::Error },
    #[error("parameter golf distributed visualization could not decode `{path}`: {error}")]
    Deserialize {
        path: String,
        error: serde_json::Error,
    },
    #[error("parameter golf distributed visualization could not encode `{path}`: {error}")]
    Serialize {
        path: String,
        error: serde_json::Error,
    },
    #[error(transparent)]
    Contract(#[from] RemoteTrainingVisualizationError),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfRunPod8xH100FinalizerReport {
    pub schema_version: String,
    pub runner: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at_utc: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trainer_lane_id: Option<String>,
    pub run_root: String,
    pub submission_dir: String,
    pub world_size: u64,
    pub grad_accum_steps: u64,
    pub accelerator_evidence: ParameterGolfRunPod8xH100AcceleratorEvidence,
    pub exported_folder_evidence: ParameterGolfRunPod8xH100ExportedFolderEvidence,
    pub claim_boundary: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfRunPod8xH100AcceleratorEvidence {
    pub inventory_path: String,
    pub topology_path: String,
    pub inventory_line_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterGolfRunPod8xH100ExportedFolderEvidence {
    pub entrypoint_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entrypoint_sha256: Option<String>,
    pub submission_manifest_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submission_manifest_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submission_run_evidence_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub submission_run_evidence_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distributed_receipt_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distributed_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParameterGolfDistributedVisualizationPaths {
    pub bundle_path: PathBuf,
    pub run_index_path: PathBuf,
    pub snapshot_path: PathBuf,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParameterGolfDistributedVisualizationWriteOutcome {
    pub bundle: RemoteTrainingVisualizationBundle,
    pub run_index: RemoteTrainingRunIndex,
    pub paths: ParameterGolfDistributedVisualizationPaths,
}

pub fn write_parameter_golf_distributed_visualization_from_finalizer_report(
    report_path: &Path,
    repo_revision: impl Into<String>,
) -> Result<
    ParameterGolfDistributedVisualizationWriteOutcome,
    ParameterGolfDistributedVisualizationError,
> {
    let report = read_finalizer_report(report_path)?;
    let run_root = PathBuf::from(report.run_root.as_str());
    let receipt = report
        .exported_folder_evidence
        .distributed_receipt_path
        .as_deref()
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .map(|path| read_distributed_receipt(path.as_path()))
        .transpose()?;
    let observed_at_ms = unix_time_ms();
    let gpu_series = read_gpu_inventory(
        PathBuf::from(report.accelerator_evidence.inventory_path.as_str()).as_path(),
        observed_at_ms,
    )?;
    let bundle = build_bundle(
        &report,
        report_path,
        repo_revision.into(),
        observed_at_ms,
        receipt.as_ref(),
        gpu_series,
    )?;
    let run_index = build_run_index(&report, &bundle)?;
    let paths = visualization_paths(run_root.as_path());
    fs::create_dir_all(
        paths
            .bundle_path
            .parent()
            .expect("bundle path should have a parent"),
    )
    .map_err(|error| ParameterGolfDistributedVisualizationError::Write {
        path: run_root.display().to_string(),
        error,
    })?;
    write_atomically_json(paths.bundle_path.as_path(), &bundle)?;
    write_atomically_json(paths.run_index_path.as_path(), &run_index)?;
    write_atomically_json(paths.snapshot_path.as_path(), &bundle)?;
    Ok(ParameterGolfDistributedVisualizationWriteOutcome {
        bundle,
        run_index,
        paths,
    })
}

fn read_finalizer_report(
    report_path: &Path,
) -> Result<ParameterGolfRunPod8xH100FinalizerReport, ParameterGolfDistributedVisualizationError> {
    let raw = fs::read_to_string(report_path).map_err(|error| {
        ParameterGolfDistributedVisualizationError::Read {
            path: report_path.display().to_string(),
            error,
        }
    })?;
    serde_json::from_str::<ParameterGolfRunPod8xH100FinalizerReport>(raw.as_str()).map_err(
        |error| ParameterGolfDistributedVisualizationError::Deserialize {
            path: report_path.display().to_string(),
            error,
        },
    )
}

fn read_distributed_receipt(
    receipt_path: &Path,
) -> Result<ParameterGolfDistributedThroughputReceipt, ParameterGolfDistributedVisualizationError> {
    let raw = fs::read_to_string(receipt_path).map_err(|error| {
        ParameterGolfDistributedVisualizationError::Read {
            path: receipt_path.display().to_string(),
            error,
        }
    })?;
    serde_json::from_str::<ParameterGolfDistributedThroughputReceipt>(raw.as_str()).map_err(
        |error| ParameterGolfDistributedVisualizationError::Deserialize {
            path: receipt_path.display().to_string(),
            error,
        },
    )
}

fn build_bundle(
    report: &ParameterGolfRunPod8xH100FinalizerReport,
    report_path: &Path,
    repo_revision: String,
    observed_at_ms: u64,
    receipt: Option<&ParameterGolfDistributedThroughputReceipt>,
    gpu_series: Vec<RemoteTrainingGpuSample>,
) -> Result<RemoteTrainingVisualizationBundle, ParameterGolfDistributedVisualizationError> {
    let run_root = PathBuf::from(report.run_root.as_str());
    let profile_id = report
        .profile_id
        .clone()
        .unwrap_or_else(|| String::from("runpod_8xh100_parameter_golf"));
    let lane_id = report
        .trainer_lane_id
        .clone()
        .unwrap_or_else(|| String::from("parameter_golf_distributed_8xh100"));
    let run_id = report.run_id.clone().unwrap_or_else(|| {
        run_root
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| String::from("parameter-golf-runpod-8xh100"))
    });
    let total_steps_completed = receipt
        .and_then(|receipt| receipt.timing.as_ref())
        .map_or(0, |timing| timing.step_count);
    let latest_global_step = (total_steps_completed > 0).then_some(total_steps_completed);
    let latest_tokens_per_second = receipt
        .and_then(|receipt| receipt.timing.as_ref())
        .map(|timing| timing.train_tokens_per_second);
    let runtime_series = receipt
        .and_then(|receipt| receipt.timing.as_ref())
        .map(|timing| {
            vec![RemoteTrainingRuntimeSample {
                observed_at_ms,
                data_wait_ms: None,
                forward_ms: None,
                backward_ms: None,
                optimizer_ms: Some(timing.mean_step_duration_ms),
                checkpoint_ms: Some(timing.export_observed_ms),
                evaluation_ms: Some(timing.validation_observed_ms),
                tokens_per_second: Some(timing.train_tokens_per_second),
                samples_per_second_milli: None,
            }]
        })
        .unwrap_or_default();
    let distributed_series = receipt
        .and_then(|receipt| receipt.timing.as_ref())
        .map(|timing| {
            vec![RemoteTrainingDistributedSample {
                observed_at_ms,
                participating_rank_count: report.world_size.min(u64::from(u16::MAX)) as u16,
                rank_skew_ms: None,
                slowest_rank_ms: Some(timing.tail_step_duration_ms),
                collective_ms: None,
                stalled_rank_count: 0,
            }]
        })
        .unwrap_or_default();
    let result_classification = match receipt.map(|receipt| receipt.disposition) {
        Some(ParameterGolfDistributedLaneDisposition::Measured) => {
            RemoteTrainingResultClassification::CompletedSuccess
        }
        Some(ParameterGolfDistributedLaneDisposition::Refused) => {
            RemoteTrainingResultClassification::Refused
        }
        None => RemoteTrainingResultClassification::CompletedFailure,
    };
    let series_unavailable_reason = if receipt.is_some() {
        String::from(
            "the RunPod 8xH100 lane retained distributed topology, timing, and provenance receipts, but it did not retain a coordinator-owned loss curve or live rank-skew stream",
        )
    } else {
        String::from(
            "the RunPod 8xH100 lane retained exported-folder, topology, and provenance evidence only; no distributed receipt or live loss curve was retained",
        )
    };
    let timeline = vec![
        RemoteTrainingTimelineEntry {
            observed_at_ms: observed_at_ms.saturating_sub(1_000),
            phase: String::from("training"),
            subphase: Some(String::from("distributed_runpod_8xh100")),
            detail: if receipt.is_some() {
                String::from(
                    "The RunPod 8xH100 lane retained its distributed receipt posture and exported-folder evidence.",
                )
            } else {
                String::from(
                    "The RunPod 8xH100 lane retained exported-folder evidence without a coordinator-owned distributed receipt stream.",
                )
            },
        },
        RemoteTrainingTimelineEntry {
            observed_at_ms,
            phase: String::from("complete"),
            subphase: Some(String::from("finalizer_sealed")),
            detail: String::from(
                "The RunPod 8xH100 finalizer sealed the provider-neutral visualization bundle.",
            ),
        },
    ];
    let mut event_series = vec![
        RemoteTrainingEventSample {
            observed_at_ms,
            severity: RemoteTrainingEventSeverity::Warning,
            event_kind: String::from("series_unavailable"),
            detail: series_unavailable_reason.clone(),
        },
        RemoteTrainingEventSample {
            observed_at_ms,
            severity: RemoteTrainingEventSeverity::Info,
            event_kind: String::from("finalizer_report_sealed"),
            detail: String::from(
                "The RunPod 8xH100 finalizer sealed the provider-neutral bundle and run index.",
            ),
        },
    ];
    if let Some(refusal) = receipt.and_then(|receipt| receipt.refusal.as_ref()) {
        event_series.push(RemoteTrainingEventSample {
            observed_at_ms,
            severity: RemoteTrainingEventSeverity::Error,
            event_kind: String::from("distributed_lane_refused"),
            detail: refusal.reason.clone(),
        });
    }
    let heartbeat_series = vec![RemoteTrainingHeartbeatSample {
        observed_at_ms,
        phase: String::from("complete"),
        subphase: Some(String::from("finalizer_sealed")),
        step_in_progress: None,
        microbatch_in_progress: None,
        active_subsystems: vec![
            String::from("finalizer"),
            String::from("exported_submission"),
            String::from("topology_capture"),
        ],
        stale_after_ms: 2_500,
    }];
    Ok(record_remote_training_visualization_bundle(
        RemoteTrainingVisualizationBundle {
            schema_version: String::new(),
            bundle_id: format!("{run_id}-distributed-8xh100-remote-training-v1"),
            provider: RemoteTrainingProvider::RunPod,
            profile_id,
            lane_id,
            run_id,
            repo_revision,
            result_classification,
            refresh_contract: RemoteTrainingRefreshContract {
                target_ui_update_interval_ms: REMOTE_TRAINING_TARGET_UI_UPDATE_INTERVAL_MS,
                emission_mode: RemoteTrainingEmissionMode::PostRunOnly,
                last_heartbeat_at_ms: Some(observed_at_ms),
                heartbeat_seq: 1,
            },
            series_status: RemoteTrainingSeriesStatus::Unavailable,
            series_unavailable_reason: Some(series_unavailable_reason.clone()),
            timeline,
            summary: RemoteTrainingVisualizationSummary {
                total_steps_completed,
                latest_global_step,
                latest_train_loss: None,
                latest_ema_loss: None,
                latest_validation_loss: None,
                latest_tokens_per_second,
                latest_samples_per_second_milli: None,
                accumulated_cost_microusd: None,
                latest_checkpoint_ref: None,
                detail: if receipt.is_some() {
                    String::from(
                        "The RunPod 8xH100 lane retains distributed topology, timing, GPU, and provenance truth, but it still does not retain a live loss curve.",
                    )
                } else {
                    String::from(
                        "The RunPod 8xH100 lane retains GPU inventory, exported-folder, and provenance truth while remaining explicit that no live distributed trainer telemetry was retained.",
                    )
                },
            },
            heartbeat_series,
            loss_series: Vec::new(),
            math_series: Vec::new(),
            runtime_series,
            gpu_series,
            distributed_series,
            event_series,
            source_artifacts: build_source_artifacts(report, report_path, receipt),
            bundle_digest: String::new(),
        },
    )?)
}

fn build_run_index(
    report: &ParameterGolfRunPod8xH100FinalizerReport,
    bundle: &RemoteTrainingVisualizationBundle,
) -> Result<RemoteTrainingRunIndex, ParameterGolfDistributedVisualizationError> {
    Ok(build_remote_training_run_index(RemoteTrainingRunIndex {
        schema_version: String::new(),
        index_id: format!("{}-remote-training-index-v1", bundle.run_id),
        generated_at_ms: unix_time_ms(),
        entries: vec![RemoteTrainingRunIndexEntry {
            provider: bundle.provider,
            profile_id: bundle.profile_id.clone(),
            lane_id: bundle.lane_id.clone(),
            run_id: bundle.run_id.clone(),
            repo_revision: bundle.repo_revision.clone(),
            result_classification: bundle.result_classification,
            series_status: bundle.series_status,
            series_unavailable_reason: bundle.series_unavailable_reason.clone(),
            last_heartbeat_at_ms: bundle.refresh_contract.last_heartbeat_at_ms,
            bundle_artifact_uri: Some(format!(
                "{}/{}",
                PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_DIR_NAME,
                PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_BUNDLE_NAME
            )),
            bundle_digest: Some(bundle.bundle_digest.clone()),
            summary_label: format!(
                "RunPod {} distributed 8xH100",
                report
                    .profile_id
                    .as_deref()
                    .unwrap_or("runpod_8xh100_parameter_golf")
            ),
            detail: bundle.summary.detail.clone(),
        }],
        detail: String::from(
            "This run index enumerates the RunPod distributed 8xH100 Parameter Golf visualization bundle for normalized app discovery.",
        ),
        index_digest: String::new(),
    })?)
}

fn build_source_artifacts(
    report: &ParameterGolfRunPod8xH100FinalizerReport,
    report_path: &Path,
    receipt: Option<&ParameterGolfDistributedThroughputReceipt>,
) -> Vec<RemoteTrainingSourceArtifact> {
    let run_root = PathBuf::from(report.run_root.as_str());
    let mut artifacts = vec![
        RemoteTrainingSourceArtifact {
            artifact_role: String::from("live_bundle"),
            artifact_uri: format!(
                "{}/{}",
                PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_DIR_NAME,
                PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_BUNDLE_NAME
            ),
            artifact_digest: None,
            source_kind: RemoteTrainingArtifactSourceKind::LocalMirror,
            authoritative: true,
            source_receipt_ids: vec![String::from(
                "parameter_golf_distributed_8xh100.remote_training_bundle.v1",
            )],
            detail: String::from(
                "The provider-neutral local mirror is the app-facing source for the RunPod distributed 8xH100 lane.",
            ),
        },
        RemoteTrainingSourceArtifact {
            artifact_role: String::from("finalizer_report"),
            artifact_uri: relative_path_string(run_root.as_path(), report_path)
                .unwrap_or_else(|| report_path.display().to_string()),
            artifact_digest: None,
            source_kind: RemoteTrainingArtifactSourceKind::FinalizerOwned,
            authoritative: true,
            source_receipt_ids: vec![String::from("parameter_golf.runpod_8xh100_finalizer.v1")],
            detail: String::from(
                "The finalizer report remains authoritative for exported-folder, topology, and inventory provenance.",
            ),
        },
        RemoteTrainingSourceArtifact {
            artifact_role: String::from("gpu_inventory"),
            artifact_uri: relative_path_string(
                run_root.as_path(),
                Path::new(report.accelerator_evidence.inventory_path.as_str()),
            )
            .unwrap_or_else(|| report.accelerator_evidence.inventory_path.clone()),
            artifact_digest: None,
            source_kind: RemoteTrainingArtifactSourceKind::ProviderGenerated,
            authoritative: true,
            source_receipt_ids: Vec::new(),
            detail: String::from(
                "The `nvidia-smi` inventory snapshot remains authoritative for the retained per-device GPU state.",
            ),
        },
        RemoteTrainingSourceArtifact {
            artifact_role: String::from("gpu_topology"),
            artifact_uri: relative_path_string(
                run_root.as_path(),
                Path::new(report.accelerator_evidence.topology_path.as_str()),
            )
            .unwrap_or_else(|| report.accelerator_evidence.topology_path.clone()),
            artifact_digest: None,
            source_kind: RemoteTrainingArtifactSourceKind::ProviderGenerated,
            authoritative: true,
            source_receipt_ids: Vec::new(),
            detail: String::from(
                "The `nvidia-smi topo -m` capture remains authoritative for the retained fabric topology snapshot.",
            ),
        },
    ];
    artifacts.push(RemoteTrainingSourceArtifact {
        artifact_role: String::from("submission_manifest"),
        artifact_uri: relative_path_string(
            run_root.as_path(),
            Path::new(report.exported_folder_evidence.submission_manifest_path.as_str()),
        )
        .unwrap_or_else(|| report.exported_folder_evidence.submission_manifest_path.clone()),
        artifact_digest: report
            .exported_folder_evidence
            .submission_manifest_sha256
            .clone(),
        source_kind: RemoteTrainingArtifactSourceKind::RuntimeOwned,
        authoritative: true,
        source_receipt_ids: vec![String::from("parameter_golf.submission_manifest.v1")],
        detail: String::from(
            "The exported submission manifest remains authoritative for the retained folder contract.",
        ),
    });
    if let Some(path) = &report.exported_folder_evidence.submission_run_evidence_path {
        artifacts.push(RemoteTrainingSourceArtifact {
            artifact_role: String::from("submission_run_evidence"),
            artifact_uri: relative_path_string(run_root.as_path(), Path::new(path.as_str()))
                .unwrap_or_else(|| path.clone()),
            artifact_digest: report
                .exported_folder_evidence
                .submission_run_evidence_sha256
                .clone(),
            source_kind: RemoteTrainingArtifactSourceKind::RuntimeOwned,
            authoritative: true,
            source_receipt_ids: vec![String::from("parameter_golf.submission_run_evidence.v1")],
            detail: String::from(
                "The submission run evidence report remains authoritative for the exported-folder execution posture and digests.",
            ),
        });
    }
    if let Some(receipt) = receipt {
        artifacts.push(RemoteTrainingSourceArtifact {
            artifact_role: String::from("distributed_receipt"),
            artifact_uri: report
                .exported_folder_evidence
                .distributed_receipt_path
                .clone()
                .and_then(|path| relative_path_string(run_root.as_path(), Path::new(path.as_str())))
                .unwrap_or_else(|| String::from("parameter_golf_distributed_8xh100_receipt.json")),
            artifact_digest: Some(receipt.receipt_digest.clone()),
            source_kind: RemoteTrainingArtifactSourceKind::RuntimeOwned,
            authoritative: true,
            source_receipt_ids: vec![String::from(
                "parameter_golf_distributed_throughput_receipt",
            )],
            detail: String::from(
                "The distributed throughput receipt remains authoritative for retained topology, communication, timing, and memory posture.",
            ),
        });
    }
    artifacts
}

fn read_gpu_inventory(
    inventory_path: &Path,
    observed_at_ms: u64,
) -> Result<Vec<RemoteTrainingGpuSample>, ParameterGolfDistributedVisualizationError> {
    if !inventory_path.is_file() {
        return Ok(Vec::new());
    }
    let raw = fs::read_to_string(inventory_path).map_err(|error| {
        ParameterGolfDistributedVisualizationError::Read {
            path: inventory_path.display().to_string(),
            error,
        }
    })?;
    Ok(raw
        .lines()
        .enumerate()
        .filter_map(|(index, line)| parse_gpu_inventory_row(index, line, observed_at_ms))
        .collect())
}

fn parse_gpu_inventory_row(
    _index: usize,
    line: &str,
    observed_at_ms: u64,
) -> Option<RemoteTrainingGpuSample> {
    let parts = line.split(',').map(|part| part.trim()).collect::<Vec<_>>();
    if parts.len() < 5 {
        return None;
    }
    let device_id = format!("cuda:{}", parse_u64(parts.first()?)?);
    let device_label = parts.get(1)?.to_string();
    let memory_total_mib = parse_u64(parts.get(2)?)?;
    let memory_used_mib = parse_u64(parts.get(3)?)?;
    let utilization_percent = parse_u64(parts.get(4)?)?;
    Some(RemoteTrainingGpuSample {
        observed_at_ms,
        device_id,
        device_label,
        utilization_bps: (utilization_percent.min(100) as u32).saturating_mul(100),
        memory_used_bytes: memory_used_mib.saturating_mul(1024 * 1024),
        memory_total_bytes: memory_total_mib.saturating_mul(1024 * 1024),
        temperature_celsius: None,
        power_watts: None,
    })
}

fn parse_u64(raw: &str) -> Option<u64> {
    raw.chars()
        .filter(|character| character.is_ascii_digit())
        .collect::<String>()
        .parse::<u64>()
        .ok()
}

fn visualization_paths(run_root: &Path) -> ParameterGolfDistributedVisualizationPaths {
    let visualization_dir = run_root.join(PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_DIR_NAME);
    ParameterGolfDistributedVisualizationPaths {
        bundle_path: visualization_dir.join(PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_BUNDLE_NAME),
        run_index_path: visualization_dir
            .join(PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_RUN_INDEX_NAME),
        snapshot_path: visualization_dir
            .join(PARAMETER_GOLF_DISTRIBUTED_VISUALIZATION_SNAPSHOT_DIR_NAME)
            .join("finalized_bundle.json"),
    }
}

fn write_atomically_json<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), ParameterGolfDistributedVisualizationError> {
    let parent = path
        .parent()
        .expect("json output path should have a parent");
    fs::create_dir_all(parent).map_err(|error| {
        ParameterGolfDistributedVisualizationError::Write {
            path: parent.display().to_string(),
            error,
        }
    })?;
    let tmp_path = path.with_extension("tmp");
    let encoded = serde_json::to_string_pretty(value).map_err(|error| {
        ParameterGolfDistributedVisualizationError::Serialize {
            path: path.display().to_string(),
            error,
        }
    })?;
    fs::write(tmp_path.as_path(), format!("{encoded}\n")).map_err(|error| {
        ParameterGolfDistributedVisualizationError::Write {
            path: tmp_path.display().to_string(),
            error,
        }
    })?;
    fs::rename(tmp_path.as_path(), path).map_err(|error| {
        ParameterGolfDistributedVisualizationError::Write {
            path: path.display().to_string(),
            error,
        }
    })?;
    Ok(())
}

fn relative_path_string(root: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(root)
        .ok()
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use tempfile::tempdir;

    use super::*;
    use psionic_eval::{
        ParameterGolfDistributedChallengeThresholds, ParameterGolfDistributedCommunicationReceipt,
        ParameterGolfDistributedCommunicationStageReceipt, ParameterGolfDistributedTimingReceipt,
        ParameterGolfDistributedTopologyReceipt, PARAMETER_GOLF_DISTRIBUTED_8XH100_BENCHMARK_REF,
        PARAMETER_GOLF_DISTRIBUTED_8XH100_CLAIM_BOUNDARY,
    };
    use psionic_runtime::{
        BackendSelection, ClusterCommunicationClass, ClusterTransportClass, TrainingCollectiveKind,
        TrainingCollectiveQuantization, TrainingDeviceMeshAxis, TrainingDeviceMeshAxisKind,
    };

    #[test]
    fn finalizer_report_writes_unavailable_bundle_without_distributed_receipt(
    ) -> Result<(), Box<dyn Error>> {
        let tempdir = tempdir()?;
        let run_root = tempdir.path().join("parameter-golf-runpod-test");
        fs::create_dir_all(run_root.as_path())?;
        let inventory_path = run_root.join("nvidia_smi_inventory.txt");
        fs::write(
            inventory_path.as_path(),
            "0, NVIDIA H100 80GB HBM3, 81559 MiB, 1024 MiB, 76 %\n1, NVIDIA H100 80GB HBM3, 81559 MiB, 2048 MiB, 71 %\n",
        )?;
        let topology_path = run_root.join("nvidia_smi_topology.txt");
        fs::write(topology_path.as_path(), "GPU0 GPU1\n")?;
        let report_path = run_root.join("finalizer_report.json");
        let report = ParameterGolfRunPod8xH100FinalizerReport {
            schema_version: String::from("parameter_golf.runpod_8xh100_finalizer.v1"),
            runner: String::from("scripts/parameter-golf-runpod-finalize-8xh100.sh"),
            created_at_utc: Some(String::from("2026-03-24T22:00:00Z")),
            run_id: Some(String::from("parameter-golf-runpod-test")),
            profile_id: Some(String::from("runpod_8xh100_parameter_golf")),
            trainer_lane_id: Some(String::from("parameter_golf_distributed_8xh100")),
            run_root: run_root.display().to_string(),
            submission_dir: run_root.join("exported_submission").display().to_string(),
            world_size: 8,
            grad_accum_steps: 1,
            accelerator_evidence: ParameterGolfRunPod8xH100AcceleratorEvidence {
                inventory_path: inventory_path.display().to_string(),
                topology_path: topology_path.display().to_string(),
                inventory_line_count: 2,
            },
            exported_folder_evidence: ParameterGolfRunPod8xH100ExportedFolderEvidence {
                entrypoint_path: String::from("train_gpt.py"),
                entrypoint_sha256: Some(String::from("entrypoint-sha")),
                submission_manifest_path: String::from("submission.json"),
                submission_manifest_sha256: Some(String::from("manifest-sha")),
                submission_run_evidence_path: None,
                submission_run_evidence_sha256: None,
                distributed_receipt_path: None,
                distributed_receipt_sha256: None,
            },
            claim_boundary: String::from("claim boundary"),
        };
        fs::write(
            report_path.as_path(),
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
        let outcome = write_parameter_golf_distributed_visualization_from_finalizer_report(
            report_path.as_path(),
            "workspace@test",
        )?;
        outcome.bundle.validate()?;
        outcome.run_index.validate()?;
        assert_eq!(
            outcome.bundle.series_status,
            RemoteTrainingSeriesStatus::Unavailable
        );
        assert_eq!(outcome.bundle.gpu_series.len(), 2);
        Ok(())
    }

    #[test]
    fn finalizer_report_uses_distributed_receipt_when_present() -> Result<(), Box<dyn Error>> {
        let tempdir = tempdir()?;
        let run_root = tempdir.path().join("parameter-golf-runpod-test");
        fs::create_dir_all(run_root.as_path())?;
        let inventory_path = run_root.join("nvidia_smi_inventory.txt");
        fs::write(
            inventory_path.as_path(),
            "0, NVIDIA H100 80GB HBM3, 81559 MiB, 1024 MiB, 76 %\n",
        )?;
        let topology_path = run_root.join("nvidia_smi_topology.txt");
        fs::write(topology_path.as_path(), "GPU0\n")?;
        let receipt_path = run_root.join("parameter_golf_distributed_8xh100_receipt.json");
        let receipt = ParameterGolfDistributedThroughputReceipt {
            benchmark_ref: String::from(PARAMETER_GOLF_DISTRIBUTED_8XH100_BENCHMARK_REF),
            run_id: String::from("parameter-golf-runpod-test"),
            model_descriptor_digest: String::from("model-digest"),
            optimizer_plan_digest: String::from("optimizer-digest"),
            thresholds: ParameterGolfDistributedChallengeThresholds::challenge_8xh100(),
            topology: ParameterGolfDistributedTopologyReceipt {
                backend_selection: BackendSelection::direct(
                    "cuda",
                    None,
                    vec![String::from("parameter_golf_distributed_train")],
                ),
                topology_digest: String::from("topology-digest"),
                selected_device_names: vec![String::from("NVIDIA H100 80GB HBM3"); 8],
                all_devices_match_required_model: true,
            },
            communication: ParameterGolfDistributedCommunicationReceipt {
                communication_class: ClusterCommunicationClass::TensorCollectiveMesh,
                transport: ClusterTransportClass::Loopback,
                mesh_id: String::from("mesh.parameter_golf.8xh100"),
                axes: vec![TrainingDeviceMeshAxis::new(
                    "dp",
                    TrainingDeviceMeshAxisKind::DataParallel,
                    8,
                )
                .with_collective_group_size(8)],
                stages: vec![ParameterGolfDistributedCommunicationStageReceipt {
                    stage_id: String::from("ddp_gradient_all_reduce"),
                    collective_kind: TrainingCollectiveKind::AllReduce,
                    quantization: TrainingCollectiveQuantization::None,
                    payload_bytes: 1024,
                    estimated_wire_bytes: 2048,
                    worker_count: 8,
                    detail: String::from("DDP gradient synchronization"),
                }],
            },
            training_capability_report_digest: String::from("coverage-digest"),
            challenge_kernel_blockers: vec![String::from("collective.rank_skew_missing")],
            disposition: ParameterGolfDistributedLaneDisposition::Measured,
            timing: Some(ParameterGolfDistributedTimingReceipt {
                measurement_posture: String::from("observed_step_wallclock"),
                step_count: 4,
                total_train_tokens: 1_048_576,
                training_step_observed_ms: 400,
                validation_observed_ms: 20,
                export_observed_ms: 10,
                total_observed_ms: 430,
                mean_step_duration_ms: 100,
                tail_step_duration_ms: 112,
                train_tokens_per_second: 2_621_440,
                wallclock_cap_ms: 600_000,
                within_wallclock_cap: true,
            }),
            memory: None,
            refusal: None,
            boundary_notes: vec![String::from("boundary")],
            claim_boundary: String::from(PARAMETER_GOLF_DISTRIBUTED_8XH100_CLAIM_BOUNDARY),
            receipt_digest: String::new(),
        }
        .with_stable_digest();
        fs::write(
            receipt_path.as_path(),
            format!("{}\n", serde_json::to_string_pretty(&receipt)?),
        )?;
        let report_path = run_root.join("finalizer_report.json");
        let report = ParameterGolfRunPod8xH100FinalizerReport {
            schema_version: String::from("parameter_golf.runpod_8xh100_finalizer.v1"),
            runner: String::from("scripts/parameter-golf-runpod-finalize-8xh100.sh"),
            created_at_utc: Some(String::from("2026-03-24T22:00:00Z")),
            run_id: Some(String::from("parameter-golf-runpod-test")),
            profile_id: Some(String::from("runpod_8xh100_parameter_golf")),
            trainer_lane_id: Some(String::from("parameter_golf_distributed_8xh100")),
            run_root: run_root.display().to_string(),
            submission_dir: run_root.join("exported_submission").display().to_string(),
            world_size: 8,
            grad_accum_steps: 1,
            accelerator_evidence: ParameterGolfRunPod8xH100AcceleratorEvidence {
                inventory_path: inventory_path.display().to_string(),
                topology_path: topology_path.display().to_string(),
                inventory_line_count: 1,
            },
            exported_folder_evidence: ParameterGolfRunPod8xH100ExportedFolderEvidence {
                entrypoint_path: String::from("train_gpt.py"),
                entrypoint_sha256: Some(String::from("entrypoint-sha")),
                submission_manifest_path: String::from("submission.json"),
                submission_manifest_sha256: Some(String::from("manifest-sha")),
                submission_run_evidence_path: None,
                submission_run_evidence_sha256: None,
                distributed_receipt_path: Some(receipt_path.display().to_string()),
                distributed_receipt_sha256: Some(receipt.receipt_digest.clone()),
            },
            claim_boundary: String::from("claim boundary"),
        };
        fs::write(
            report_path.as_path(),
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
        let outcome = write_parameter_golf_distributed_visualization_from_finalizer_report(
            report_path.as_path(),
            "workspace@test",
        )?;
        assert_eq!(outcome.bundle.summary.total_steps_completed, 4);
        assert_eq!(
            outcome.bundle.summary.latest_tokens_per_second,
            Some(2_621_440)
        );
        assert_eq!(outcome.bundle.distributed_series.len(), 1);
        outcome.bundle.validate()?;
        Ok(())
    }
}
