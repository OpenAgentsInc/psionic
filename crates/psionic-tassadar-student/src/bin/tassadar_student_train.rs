//! Trains one W3 student baseline on a `student_prep.v0.1` file.
//!
//! Usage:
//!   tassadar-student-train --prep train.tsprep --baseline a|b|c|c-random|d \
//!     --out <dir> [--max-steps N] [--cpu-budget N] [--host <label>]
//!
//! CPU budget (psionic#1123): defaults to ONE core. Widening requires an
//! explicit owner opt-in via `--cpu-budget N` (alias: legacy `--threads N`)
//! or `PSIONIC_TRAIN_CPU_BUDGET=N`; the launch banner records the
//! effective budget and its source.

use std::io::Write as _;
use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};

use psionic_tassadar_student::budget::{CPU_BUDGET_ENV, resolve_cpu_budget};
use psionic_tassadar_student::interface::{InterfaceTrainStats, train_interface};
use psionic_tassadar_student::prep::parse_prep;
use psionic_tassadar_student::train::{Baseline, TrainConfig, train};

#[derive(Serialize)]
struct InterfaceReceipt {
    receipt_version: String,
    baseline: String,
    corpus_id: String,
    dataset_snapshot_digest: String,
    train_prep_sha256: String,
    interface_digest: String,
    learning_rate: f32,
    stats: InterfaceTrainStats,
    wall_seconds: f64,
    host: String,
}

fn main() {
    if let Err(error) = run() {
        let mut err = std::io::stderr().lock();
        let _ = writeln!(err, "error: {error}");
        std::process::abort();
    }
}

fn run() -> Result<(), String> {
    let mut prep_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut baseline: Option<String> = None;
    let mut max_steps = 0_usize;
    let mut cpu_budget_flag: Option<usize> = None;
    let mut host = String::from("unspecified");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let take_value = |index: &mut usize| -> Result<String, String> {
            *index += 1;
            args.get(*index)
                .cloned()
                .ok_or_else(|| String::from("missing argument value"))
        };
        match args[index].as_str() {
            "--prep" => prep_path = Some(PathBuf::from(take_value(&mut index)?)),
            "--out" => out_dir = Some(PathBuf::from(take_value(&mut index)?)),
            "--baseline" => baseline = Some(take_value(&mut index)?),
            "--max-steps" => {
                max_steps = take_value(&mut index)?
                    .parse()
                    .map_err(|error| format!("bad --max-steps: {error}"))?;
            }
            // `--threads` is the legacy alias; both are explicit opt-ins.
            flag @ ("--cpu-budget" | "--threads") => {
                cpu_budget_flag = Some(
                    take_value(&mut index)?
                        .parse()
                        .map_err(|error| format!("bad {flag}: {error}"))?,
                );
            }
            "--host" => host = take_value(&mut index)?,
            other => return Err(format!("unknown argument {other}")),
        }
        index += 1;
    }
    let prep_path = prep_path.ok_or("missing --prep")?;
    let out_dir = out_dir.ok_or("missing --out")?;
    let baseline = baseline.ok_or("missing --baseline")?;
    let budget = resolve_cpu_budget(
        cpu_budget_flag,
        std::env::var(CPU_BUDGET_ENV).ok().as_deref(),
    )?;
    {
        let mut log = std::io::stderr().lock();
        let _ = writeln!(log, "{}", budget.banner());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(budget.cores)
        .build_global()
        .map_err(|error| error.to_string())?;
    let bytes = std::fs::read(&prep_path).map_err(|error| error.to_string())?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let prep_sha256 = hex::encode(hasher.finalize());
    let prep = parse_prep(&bytes).map_err(|error| error.to_string())?;
    drop(bytes);
    let mut log = std::io::stderr().lock();
    let _ = writeln!(
        log,
        "prep {} records={} corpus={} snapshot={}",
        prep_path.display(),
        prep.records.len(),
        prep.corpus_id,
        prep.snapshot_digest
    );
    drop(log);
    if baseline == "d" {
        let started = std::time::Instant::now();
        let lr = 0.5_f32;
        let (interface, stats) = train_interface(&prep.records, lr);
        std::fs::create_dir_all(&out_dir).map_err(|error| error.to_string())?;
        let interface_json =
            serde_json::to_string_pretty(&interface).map_err(|error| error.to_string())?;
        std::fs::write(
            out_dir.join("interface.json"),
            format!("{interface_json}\n"),
        )
        .map_err(|error| error.to_string())?;
        let receipt = InterfaceReceipt {
            baseline: String::from("baseline_d_frozen_executor_learned_interface"),
            corpus_id: prep.corpus_id.clone(),
            dataset_snapshot_digest: prep.snapshot_digest.clone(),
            host,
            interface_digest: interface.stable_digest(),
            learning_rate: lr,
            receipt_version: String::from("tassadar_student_interface_receipt.v0.1"),
            stats,
            train_prep_sha256: prep_sha256,
            wall_seconds: started.elapsed().as_secs_f64(),
        };
        let receipt_json =
            serde_json::to_string_pretty(&receipt).map_err(|error| error.to_string())?;
        std::fs::write(out_dir.join("receipt.json"), format!("{receipt_json}\n"))
            .map_err(|error| error.to_string())?;
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{receipt_json}");
        return Ok(());
    }
    let kind = match baseline.as_str() {
        "a" => Baseline::A,
        "b" => Baseline::B,
        "c" => Baseline::C,
        "c-random" => Baseline::CRandom,
        other => return Err(format!("unknown baseline {other}")),
    };
    let mut cfg = TrainConfig::w3_default(kind);
    cfg.max_steps = max_steps;
    let receipt = train(&prep, &prep_sha256, &cfg, &out_dir, &host)?;
    let receipt_json = serde_json::to_string_pretty(&receipt).map_err(|error| error.to_string())?;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{receipt_json}");
    Ok(())
}
