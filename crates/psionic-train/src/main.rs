use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

use psionic_train::{
    PsionActualPretrainingCurrentRunStatus, PsionicTrainInvocationManifest, PsionicTrainOperation,
    PsionicTrainRefusalClass, PsionicTrainRuntimeContractError, PsionicTrainStatusPacket,
};

#[allow(dead_code)]
#[path = "../examples/psion_actual_pretraining_operator.rs"]
mod psion_actual_pretraining_operator;

fn main() -> ExitCode {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || matches!(args[0].as_str(), "--help" | "-h" | "help") {
        print_usage();
        return ExitCode::SUCCESS;
    }

    match args[0].as_str() {
        "manifest" => run_manifest_mode(&args[1..]),
        "actual-pretraining" => run_actual_pretraining_passthrough(&args[1..]),
        other => {
            eprintln!(
                "error: unsupported psionic-train subcommand `{other}`\n\nsupported subcommands: manifest, actual-pretraining"
            );
            ExitCode::from(PsionicTrainRefusalClass::BadConfig.exit_code())
        }
    }
}

fn run_manifest_mode(args: &[String]) -> ExitCode {
    let manifest_path = match parse_manifest_path(args) {
        Ok(path) => path,
        Err(error) => {
            emit_refusal_packet(PsionicTrainStatusPacket::refusal(
                None,
                PsionicTrainRefusalClass::BadConfig,
                None,
                None,
                None,
                error,
            ));
            return ExitCode::from(PsionicTrainRefusalClass::BadConfig.exit_code());
        }
    };

    let manifest = match load_manifest(&manifest_path) {
        Ok(manifest) => manifest,
        Err(packet) => {
            emit_refusal_packet(packet.clone());
            return ExitCode::from(packet.exit_code);
        }
    };

    let operator_args = match operator_args_from_manifest(&manifest) {
        Ok(args) => args,
        Err(packet) => {
            emit_refusal_packet(packet.clone());
            return ExitCode::from(packet.exit_code);
        }
    };

    let run_root = manifest_run_root(&manifest);
    let manifest_path_text = Some(manifest_path.display().to_string());
    let initial_summary = run_root
        .as_deref()
        .and_then(actual_pretraining_run_surface_summary);

    psion_actual_pretraining_operator::set_human_output_enabled(false);
    let execution = psion_actual_pretraining_operator::run_with_args(operator_args);
    psion_actual_pretraining_operator::set_human_output_enabled(true);

    match execution {
        Ok(()) => {
            let summary = run_root
                .as_deref()
                .and_then(actual_pretraining_run_surface_summary)
                .or(initial_summary);
            let packet = PsionicTrainStatusPacket::success(
                &manifest,
                manifest_path_text,
                summary
                    .as_ref()
                    .and_then(|value| value.run_id.clone())
                    .or_else(|| manifest.run_id.clone()),
                run_root.as_ref().map(|value| value.display().to_string()),
                summary
                    .as_ref()
                    .and_then(|value| value.current_status_path.clone()),
                summary
                    .as_ref()
                    .and_then(|value| value.retained_summary_path.clone()),
                summary
                    .as_ref()
                    .and_then(|value| value.latest_checkpoint_pointer_path.clone()),
                summary
                    .as_ref()
                    .and_then(|value| value.launcher_log_path.clone()),
                success_detail(&manifest, summary.as_ref()),
            );
            emit_success_packet(&packet);
            ExitCode::SUCCESS
        }
        Err(error) => {
            let packet = refusal_packet_for_error(
                Some(&manifest),
                manifest_path.display().to_string(),
                run_root.as_deref(),
                error.as_ref(),
            );
            let code = packet.exit_code;
            emit_refusal_packet(packet);
            ExitCode::from(code)
        }
    }
}

fn run_actual_pretraining_passthrough(args: &[String]) -> ExitCode {
    match psion_actual_pretraining_operator::run_with_args(args.to_vec()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(1)
        }
    }
}

fn parse_manifest_path(args: &[String]) -> Result<PathBuf, String> {
    let mut args = args.iter();
    let Some(flag) = args.next() else {
        return Err(String::from(
            "manifest mode requires --manifest <path> to a machine-consumable invocation manifest",
        ));
    };
    if flag != "--manifest" {
        return Err(format!(
            "manifest mode expected `--manifest <path>` but found `{flag}`"
        ));
    }
    let Some(path) = args.next() else {
        return Err(String::from(
            "manifest mode requires one path after --manifest",
        ));
    };
    if args.next().is_some() {
        return Err(String::from(
            "manifest mode accepts only `--manifest <path>` and no trailing arguments",
        ));
    }
    Ok(PathBuf::from(path))
}

fn load_manifest(path: &Path) -> Result<PsionicTrainInvocationManifest, PsionicTrainStatusPacket> {
    let bytes = fs::read(path).map_err(|error| {
        PsionicTrainStatusPacket::refusal(
            None,
            PsionicTrainRefusalClass::BadConfig,
            Some(path.display().to_string()),
            None,
            None,
            format!(
                "failed to read psionic-train invocation manifest `{}`: {error}",
                path.display()
            ),
        )
    })?;
    let manifest: PsionicTrainInvocationManifest =
        serde_json::from_slice(&bytes).map_err(|error| {
            PsionicTrainStatusPacket::refusal(
                None,
                PsionicTrainRefusalClass::BadConfig,
                Some(path.display().to_string()),
                None,
                None,
                format!(
                    "failed to parse psionic-train invocation manifest `{}`: {error}",
                    path.display()
                ),
            )
        })?;
    manifest.validate_machine_contract().map_err(|error| {
        PsionicTrainStatusPacket::refusal(
            Some(&manifest),
            refusal_for_contract_error(&error),
            Some(path.display().to_string()),
            manifest.run_id.clone(),
            manifest_run_root(&manifest).map(|value| value.display().to_string()),
            format!(
                "psionic-train invocation manifest `{}` failed validation: {error}",
                path.display()
            ),
        )
    })?;
    Ok(manifest)
}

fn operator_args_from_manifest(
    manifest: &PsionicTrainInvocationManifest,
) -> Result<Vec<String>, PsionicTrainStatusPacket> {
    let mut args = vec![String::from(manifest.operation.cli_subcommand())];
    match manifest.operation {
        PsionicTrainOperation::Start | PsionicTrainOperation::RehearseBaseLane => {
            args.push(String::from("--run-id"));
            args.push(manifest.run_id.clone().ok_or_else(|| {
                PsionicTrainStatusPacket::refusal(
                    Some(manifest),
                    PsionicTrainRefusalClass::BadConfig,
                    None,
                    None,
                    None,
                    "machine runtime manifest is missing run_id for launch-style operation",
                )
            })?);
            args.push(String::from("--output-root"));
            args.push(manifest.output_root.clone().ok_or_else(|| {
                PsionicTrainStatusPacket::refusal(
                    Some(manifest),
                    PsionicTrainRefusalClass::BadConfig,
                    None,
                    None,
                    None,
                    "machine runtime manifest is missing output_root for launch-style operation",
                )
            })?);
        }
        _ => {
            args.push(String::from("--run-root"));
            args.push(manifest.run_root.clone().ok_or_else(|| {
                PsionicTrainStatusPacket::refusal(
                    Some(manifest),
                    PsionicTrainRefusalClass::BadConfig,
                    None,
                    None,
                    None,
                    "machine runtime manifest is missing run_root for retained-state operation",
                )
            })?);
        }
    }

    args.push(String::from("--git-ref"));
    args.push(manifest.selected_git_ref.clone().ok_or_else(|| {
        PsionicTrainStatusPacket::refusal(
            Some(manifest),
            PsionicTrainRefusalClass::BadConfig,
            None,
            None,
            None,
            "machine runtime manifest is missing selected_git_ref",
        )
    })?);

    if let Some(path) = manifest.hardware_observation_path.clone() {
        args.push(String::from("--hardware-observation"));
        args.push(path);
    }
    if let Some(path) = manifest.run_shape_observation_path.clone() {
        args.push(String::from("--run-shape-observation"));
        args.push(path);
    }
    if manifest.allow_dirty_tree {
        args.push(String::from("--allow-dirty-tree"));
    }
    if manifest.dry_run {
        args.push(String::from("--dry-run"));
    }
    if let Some(label) = manifest.checkpoint_label.clone() {
        args.push(String::from("--checkpoint-label"));
        args.push(label);
    }
    if let Some(step) = manifest.optimizer_step {
        args.push(String::from("--optimizer-step"));
        args.push(step.to_string());
    }
    if let Some(checkpoint_ref) = manifest.checkpoint_ref.clone() {
        args.push(String::from("--checkpoint-ref"));
        args.push(checkpoint_ref);
    }
    if let Some(digest) = manifest.checkpoint_object_digest.clone() {
        args.push(String::from("--checkpoint-object-digest"));
        args.push(digest);
    }
    if let Some(total_bytes) = manifest.checkpoint_total_bytes {
        args.push(String::from("--checkpoint-total-bytes"));
        args.push(total_bytes.to_string());
    }
    if manifest.inject_failed_upload {
        args.push(String::from("--inject-failed-upload"));
    }
    if manifest.inject_eval_worker_unavailable {
        args.push(String::from("--inject-eval-worker-unavailable"));
    }

    Ok(args)
}

fn manifest_run_root(manifest: &PsionicTrainInvocationManifest) -> Option<PathBuf> {
    match manifest.operation {
        PsionicTrainOperation::Start | PsionicTrainOperation::RehearseBaseLane => {
            manifest.output_root.as_ref().map(PathBuf::from)
        }
        _ => manifest.run_root.as_ref().map(PathBuf::from),
    }
}

fn success_detail(
    manifest: &PsionicTrainInvocationManifest,
    summary: Option<&ActualPretrainingRunSurfaceSummary>,
) -> String {
    match summary.and_then(|value| value.phase.as_deref()) {
        Some(phase) => format!(
            "psionic-train completed {} for lane `{}` with retained phase `{phase}`",
            manifest.operation.cli_subcommand(),
            manifest.lane_id
        ),
        None => format!(
            "psionic-train completed {} for lane `{}`",
            manifest.operation.cli_subcommand(),
            manifest.lane_id
        ),
    }
}

fn refusal_packet_for_error(
    manifest: Option<&PsionicTrainInvocationManifest>,
    manifest_path: String,
    run_root: Option<&Path>,
    error: &dyn Error,
) -> PsionicTrainStatusPacket {
    let detail = error.to_string();
    let refusal_class = classify_operator_error(&detail);
    let summary = run_root.and_then(actual_pretraining_run_surface_summary);
    PsionicTrainStatusPacket::refusal(
        manifest,
        refusal_class,
        Some(manifest_path),
        summary
            .as_ref()
            .and_then(|value| value.run_id.clone())
            .or_else(|| manifest.and_then(|value| value.run_id.clone())),
        run_root.map(|value| value.display().to_string()),
        detail,
    )
}

fn classify_operator_error(detail: &str) -> PsionicTrainRefusalClass {
    let lower = detail.to_ascii_lowercase();
    if lower.contains("requires --")
        || lower.contains("missing subcommand")
        || lower.contains("unsupported subcommand")
        || lower.contains("unknown argument")
        || lower.contains("dirty working trees are refused by default")
    {
        PsionicTrainRefusalClass::BadConfig
    } else if lower.contains("launch refused preflight admission") {
        PsionicTrainRefusalClass::UnsupportedTopology
    } else if lower.contains("accepted checkpoint pointer")
        || lower.contains("checkpoint manifest path")
        || lower.contains("no admitted backup receipt was available")
        || lower.contains("latest checkpoint backup receipt is not durable enough")
    {
        PsionicTrainRefusalClass::CheckpointMissing
    } else if lower.contains("drifted from the accepted checkpoint pointer")
        || lower.contains("checkpoint manifest ref drifted")
    {
        PsionicTrainRefusalClass::CheckpointDigestMismatch
    } else if lower.contains("failed-upload drill") || lower.contains("durable backup confirmation")
    {
        PsionicTrainRefusalClass::ArtifactIncomplete
    } else {
        PsionicTrainRefusalClass::InternalError
    }
}

fn refusal_for_contract_error(
    error: &PsionicTrainRuntimeContractError,
) -> PsionicTrainRefusalClass {
    match error {
        PsionicTrainRuntimeContractError::InvalidValue { field, .. }
            if field == "invocation_manifest.lane_id" =>
        {
            PsionicTrainRefusalClass::UnsupportedTopology
        }
        PsionicTrainRuntimeContractError::FieldMismatch { field, .. }
            if field == "invocation_manifest.manifest_digest" =>
        {
            PsionicTrainRefusalClass::ArtifactDigestMismatch
        }
        _ => PsionicTrainRefusalClass::BadConfig,
    }
}

#[derive(Clone, Debug)]
struct ActualPretrainingRunSurfaceSummary {
    run_id: Option<String>,
    phase: Option<String>,
    current_status_path: Option<String>,
    retained_summary_path: Option<String>,
    latest_checkpoint_pointer_path: Option<String>,
    launcher_log_path: Option<String>,
}

fn actual_pretraining_run_surface_summary(
    run_root: &Path,
) -> Option<ActualPretrainingRunSurfaceSummary> {
    let current_status_path = run_root.join("status/current_run_status.json");
    let current_status = if current_status_path.is_file() {
        fs::read_to_string(&current_status_path)
            .ok()
            .and_then(|value| {
                serde_json::from_str::<PsionActualPretrainingCurrentRunStatus>(&value).ok()
            })
    } else {
        None
    };
    Some(ActualPretrainingRunSurfaceSummary {
        run_id: current_status.as_ref().map(|value| value.run_id.clone()),
        phase: current_status.as_ref().map(|value| value.phase.clone()),
        current_status_path: current_status_path
            .is_file()
            .then(|| current_status_path.display().to_string()),
        retained_summary_path: {
            let path = run_root.join("status/retained_summary.json");
            path.is_file().then(|| path.display().to_string())
        },
        latest_checkpoint_pointer_path: {
            let path = run_root.join("checkpoints/latest_accepted_checkpoint_pointer.json");
            path.is_file().then(|| path.display().to_string())
        },
        launcher_log_path: {
            let path = run_root.join("logs/launcher.log");
            path.is_file().then(|| path.display().to_string())
        },
    })
}

fn emit_success_packet(packet: &PsionicTrainStatusPacket) {
    println!(
        "{}",
        serde_json::to_string_pretty(packet).expect("status packet should serialize")
    );
}

fn emit_refusal_packet(packet: PsionicTrainStatusPacket) {
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&packet).expect("status packet should serialize")
    );
}

fn print_usage() {
    eprintln!(
        "Usage:\n  psionic-train manifest --manifest <path>\n  psionic-train actual-pretraining <operator-args>\n\nMachine mode requires a `{}` JSON manifest and emits one `{}` packet on completion.",
        psionic_train::PSIONIC_TRAIN_INVOCATION_MANIFEST_SCHEMA_VERSION,
        psionic_train::PSIONIC_TRAIN_STATUS_PACKET_SCHEMA_VERSION
    );
}
