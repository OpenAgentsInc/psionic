use std::{env, path::PathBuf};

use psionic_train::{
    ParameterGolfSingleH100TrainingConfig, write_parameter_golf_single_h100_training_report,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_root = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(default_dataset_root);
    let tokenizer_path = env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(default_tokenizer_path);
    let output_path = env::args()
        .nth(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/parameter_golf_single_h100_training.json"));
    let max_steps = env::args()
        .nth(4)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(1);
    let mut config =
        ParameterGolfSingleH100TrainingConfig::challenge_defaults(dataset_root, tokenizer_path);
    config.max_steps = max_steps;
    let report = write_parameter_golf_single_h100_training_report(&output_path, &config)?;
    println!(
        "wrote {} with disposition {:?} executed_steps={}",
        output_path.display(),
        report.disposition,
        report.executed_steps
    );
    for step in &report.step_metrics {
        println!(
            "train_step step:{} train_loss:{:.8} lr_mult:{:.8} muon_momentum:{:.8} windows:{}",
            step.global_step,
            step.mean_microbatch_loss,
            step.learning_rate_multiplier,
            step.muon_momentum,
            step.train_window_ids.join(",")
        );
    }
    if let Some(final_validation) = report.final_validation {
        println!(
            "final_validation val_loss:{:.8} val_bpb:{:.8}",
            final_validation.mean_loss, final_validation.bits_per_byte
        );
    }
    if let Some(bytes) = report.compressed_model_bytes {
        println!("compressed_model_bytes={bytes}");
    }
    if let Some(ref artifact_ref) = report.compressed_model_artifact_ref {
        println!("compressed_model_artifact_ref={artifact_ref}");
    }
    if let Some(ref artifact_digest) = report.compressed_model_artifact_digest {
        println!("compressed_model_artifact_digest={artifact_digest}");
    }
    if let Some(refusal) = report.refusal {
        println!(
            "refusal subject={:?} detail={}",
            refusal.subject, refusal.detail
        );
    }
    Ok(())
}

fn default_dataset_root() -> PathBuf {
    PathBuf::from(
        env::var("HOME").expect("HOME should exist for the default parameter-golf dataset root"),
    )
    .join("code/parameter-golf/data/datasets/fineweb10B_sp1024")
}

fn default_tokenizer_path() -> PathBuf {
    PathBuf::from(
        env::var("HOME").expect("HOME should exist for the default parameter-golf tokenizer path"),
    )
    .join("code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
}
