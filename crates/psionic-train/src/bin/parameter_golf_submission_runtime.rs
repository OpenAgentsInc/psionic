use std::{env, path::PathBuf};

use psionic_train::{
    execute_parameter_golf_submission_runtime_entrypoint, ParameterGolfSubmissionRuntimeError,
    ParameterGolfSubmissionRuntimeOutcome,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), ParameterGolfSubmissionRuntimeError> {
    let manifest_arg = env::args()
        .nth(1)
        .unwrap_or_else(|| String::from("runtime/parameter_golf_submission_runtime.json"));
    let root = env::current_dir().map_err(|error| ParameterGolfSubmissionRuntimeError::Read {
        path: String::from("."),
        error,
    })?;
    let manifest_path = if PathBuf::from(&manifest_arg).is_absolute() {
        PathBuf::from(&manifest_arg)
    } else {
        root.join(&manifest_arg)
    };
    match execute_parameter_golf_submission_runtime_entrypoint(&root, &manifest_path)? {
        ParameterGolfSubmissionRuntimeOutcome::LocalReference(receipt) => {
            println!(
                "psionic_non_record_submission_runtime run_id={} runtime_posture={}",
                receipt.run_id, receipt.runtime_posture
            );
            println!(
                "final_int8_zlib_roundtrip_exact val_loss:{:.8} val_bpb:{:.8}",
                receipt.executed_validation.mean_loss, receipt.executed_validation.bits_per_byte
            );
            println!(
                "runtime_consistency bytes_code={} bytes_total={} model_bytes_match_submission={} model_bytes_match_accounting={}",
                receipt.matches_accounting_code_bytes,
                receipt.matches_accounting_total_bytes,
                receipt.matches_submission_model_bytes,
                receipt.matches_accounting_model_bytes,
            );
            Ok(())
        }
        ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100Bringup {
            report_path,
            report,
        } => {
            eprintln!(
                "parameter golf submission runtime does not ship a distributed 8xH100 trainer payload yet; wrote distributed bring-up report to {} with disposition {:?}",
                report_path,
                report.disposition
            );
            eprintln!(
                "psionic_parameter_golf_distributed_8xh100_bringup matching_h100_device_count={} machine_contract_satisfied={}",
                report.matching_h100_device_count, report.machine_contract_satisfied
            );
            if let Some(refusal) = report.refusal {
                eprintln!(
                    "refusal subject={:?} detail={}",
                    refusal.subject, refusal.detail
                );
            }
            Err(ParameterGolfSubmissionRuntimeError::ExecutionMode {
                message: String::from(
                    "the exported folder now ships a Rust-owned distributed 8xH100 bring-up path, but the real distributed trainer payload still has not landed",
                ),
            })
        }
        ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100Bootstrap {
            report_path,
            report,
            receipt_path,
            receipt,
        } => {
            eprintln!(
                "psionic_parameter_golf_distributed_8xh100_bringup matching_h100_device_count={} machine_contract_satisfied={} report_path={}",
                report.matching_h100_device_count, report.machine_contract_satisfied, report_path
            );
            eprintln!(
                "psionic_parameter_golf_distributed_8xh100_runtime_bootstrap disposition={:?} successful_rank_count={} receipt_path={}",
                receipt.disposition, receipt.successful_rank_count, receipt_path
            );
            if let Some(refusal) = receipt.refusal.as_ref() {
                eprintln!(
                    "runtime_bootstrap_refusal subject={:?} detail={}",
                    refusal.subject, refusal.detail
                );
            }
            Err(ParameterGolfSubmissionRuntimeError::ExecutionMode {
                message: String::from(
                    "the exported folder now ships a real multi-rank distributed 8xH100 runtime bootstrap path, but the later distributed train step still has not landed",
                ),
            })
        }
        ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100TrainStep {
            report_path,
            report,
            bootstrap_receipt_path,
            bootstrap_receipt,
            train_step_receipt_path,
            train_step_receipt,
        } => {
            eprintln!(
                "psionic_parameter_golf_distributed_8xh100_bringup matching_h100_device_count={} machine_contract_satisfied={} report_path={}",
                report.matching_h100_device_count, report.machine_contract_satisfied, report_path
            );
            eprintln!(
                "psionic_parameter_golf_distributed_8xh100_runtime_bootstrap disposition={:?} successful_rank_count={} receipt_path={}",
                bootstrap_receipt.disposition,
                bootstrap_receipt.successful_rank_count,
                bootstrap_receipt_path
            );
            println!(
                "step:1/1 train_loss:{:.4} train_time:{}ms step_avg:{:.2}ms",
                train_step_receipt.mean_train_loss,
                train_step_receipt.observed_step_ms,
                train_step_receipt.observed_step_ms as f64,
            );
            eprintln!(
                "psionic_parameter_golf_distributed_8xh100_train_step receipt_path={} distributed_receipt_path={} gradient_sync_ms={} optimizer_step_ms={}",
                train_step_receipt_path,
                train_step_receipt.distributed_receipt_path,
                train_step_receipt.gradient_sync_ms,
                train_step_receipt.optimizer_step_ms,
            );
            Err(ParameterGolfSubmissionRuntimeError::ExecutionMode {
                message: String::from(
                    "the exported folder now ships one real distributed 8xH100 train step and a measured distributed receipt, but the later distributed validation and final execution-closure path still have not landed",
                ),
            })
        }
        ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100BootstrapChild {
            receipt_path,
            receipt,
        } => {
            println!(
                "distributed_8xh100_runtime_bootstrap_rank_ready rank={} local_rank={} world_size={} runtime_prerequisites_satisfied={} receipt_path={}",
                receipt.rank,
                receipt.local_rank,
                receipt.world_size,
                receipt.runtime_prerequisites_satisfied,
                receipt_path
            );
            if let Some(group) = receipt.distributed_group.as_ref() {
                println!(
                    "distributed_8xh100_runtime_bootstrap_group group_id={} rank={} size={} backend={} communication_class={:?}",
                    group.group_id,
                    group.rank,
                    group.size,
                    group.effective_backend,
                    group.communication_class
                );
            }
            if !receipt.runtime_prerequisites_satisfied {
                if let Some(refusal) = receipt.refusal.as_ref() {
                    eprintln!(
                        "runtime_bootstrap_refusal subject={:?} detail={}",
                        refusal.subject, refusal.detail
                    );
                }
                return Err(ParameterGolfSubmissionRuntimeError::ExecutionMode {
                    message: format!(
                        "distributed 8xH100 runtime bootstrap child rank {} refused runtime prerequisites",
                        receipt.rank
                    ),
                });
            }
            Ok(())
        }
        ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100TrainStepChild {
            receipt_path,
            receipt,
        } => {
            println!(
                "distributed_8xh100_train_step_rank_complete rank={} local_rank={} world_size={} window_id={} loss={:.8} train_time:{}ms receipt_path={} gradient_artifact_path={}",
                receipt.rank,
                receipt.local_rank,
                receipt.world_size,
                receipt.window_id,
                receipt.loss,
                receipt.observed_wallclock_ms,
                receipt_path,
                receipt.gradient_artifact_path,
            );
            Ok(())
        }
    }
}
