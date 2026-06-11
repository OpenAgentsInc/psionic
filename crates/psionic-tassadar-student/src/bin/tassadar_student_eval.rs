//! Evaluates one trained W3 student baseline by first divergence behind
//! replay over a `student_prep.v0.1` eval file.
//!
//! Usage:
//!   tassadar-student-eval --prep eval.tsprep --checkpoint <dir> \
//!     --out report.json [--threads N]
//!
//! `<dir>` must contain either `weights.bin` + `receipt.json`
//! (baselines a/b/c) or `interface.json` + `receipt.json` (baseline d).

use std::io::Write as _;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

use psionic_tassadar_student::evalrun::{Student, run_eval};
use psionic_tassadar_student::interface::InterfaceModel;
use psionic_tassadar_student::model::Backbone;
use psionic_tassadar_student::prep::parse_prep;
use psionic_tassadar_student::train::TrainReceipt;

fn main() {
    if let Err(error) = run() {
        let mut err = std::io::stderr().lock();
        let _ = writeln!(err, "error: {error}");
        std::process::abort();
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

fn run() -> Result<(), String> {
    let mut prep_path: Option<PathBuf> = None;
    let mut checkpoint: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut threads = 0_usize;
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
            "--checkpoint" => checkpoint = Some(PathBuf::from(take_value(&mut index)?)),
            "--out" => out_path = Some(PathBuf::from(take_value(&mut index)?)),
            "--threads" => {
                threads = take_value(&mut index)?
                    .parse()
                    .map_err(|error| format!("bad --threads: {error}"))?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
        index += 1;
    }
    let prep_path = prep_path.ok_or("missing --prep")?;
    let checkpoint = checkpoint.ok_or("missing --checkpoint")?;
    let out_path = out_path.ok_or("missing --out")?;
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| error.to_string())?;
    }
    let bytes = std::fs::read(&prep_path).map_err(|error| error.to_string())?;
    let eval_prep_sha256 = sha256_hex(&bytes);
    let prep = parse_prep(&bytes).map_err(|error| error.to_string())?;
    drop(bytes);
    let weights_path = checkpoint.join("weights.bin");
    let interface_path = checkpoint.join("interface.json");
    let (student, baseline, checkpoint_sha256, config_digest) = if weights_path.exists() {
        let receipt_text = std::fs::read_to_string(checkpoint.join("receipt.json"))
            .map_err(|error| error.to_string())?;
        let receipt: TrainReceipt =
            serde_json::from_str(&receipt_text).map_err(|error| error.to_string())?;
        let weights = std::fs::read(&weights_path).map_err(|error| error.to_string())?;
        let weights_sha = sha256_hex(&weights);
        if weights_sha != receipt.weights_sha256 {
            return Err(format!(
                "weights sha {weights_sha} does not match receipt {}",
                receipt.weights_sha256
            ));
        }
        let mut model = Backbone::init(&receipt.config.model, receipt.config.seed);
        model.load_weights_bytes(&weights)?;
        (
            Student::Backbone(Box::new(model)),
            receipt.baseline.clone(),
            weights_sha,
            receipt.config_digest.clone(),
        )
    } else if interface_path.exists() {
        let interface_text =
            std::fs::read_to_string(&interface_path).map_err(|error| error.to_string())?;
        let interface: InterfaceModel =
            serde_json::from_str(&interface_text).map_err(|error| error.to_string())?;
        let digest = interface.stable_digest();
        (
            Student::Interface(Box::new(interface)),
            String::from("baseline_d_frozen_executor_learned_interface"),
            digest.clone(),
            digest,
        )
    } else {
        return Err(format!(
            "checkpoint {} has neither weights.bin nor interface.json",
            checkpoint.display()
        ));
    };
    let eval_records: Vec<_> = prep
        .records
        .iter()
        .filter(|record| record.split != psionic_tassadar_student::prep::Split::Train)
        .cloned()
        .collect();
    let report = run_eval(
        &student,
        &eval_records,
        &baseline,
        &prep.corpus_id,
        &prep.snapshot_digest,
        &eval_prep_sha256,
        &checkpoint_sha256,
        &config_digest,
    );
    let report_json =
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    std::fs::write(&out_path, format!("{report_json}\n"))
        .map_err(|error| error.to_string())?;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{report_json}");
    Ok(())
}
