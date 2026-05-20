use std::env;
use std::error::Error;
use std::io;
use std::path::PathBuf;

use psionic_eval::{LegalBenchmarkEvalSuiteRunConfig, run_legal_benchmark_eval_suite};

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().collect::<Vec<_>>();
    let suite = required_flag(&args, "--suite")?;
    let model = required_flag(&args, "--model")?;
    let adapter = required_flag(&args, "--adapter")?;
    let out = required_flag(&args, "--out")?;
    let report = run_legal_benchmark_eval_suite(&LegalBenchmarkEvalSuiteRunConfig {
        suite_path: PathBuf::from(suite),
        base_model: model,
        adapter,
        output_dir: PathBuf::from(out),
        replay_command: args,
    })?;
    println!(
        "suite={} base_score_bps={} adapter_score_bps={} delta_bps={} report_hash={}",
        report.suite_id,
        report.base_model_result.legal_score_bps,
        report.adapter_result.legal_score_bps,
        report.comparison.score_delta_bps,
        report.replay_receipt.report_hash
    );
    Ok(())
}

fn required_flag(args: &[String], flag: &str) -> Result<String, io::Error> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].clone()))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "usage: legal_benchmark_eval_suite --suite <suite.json> --model <base> --adapter <adapter> --out <dir>",
            )
        })
}
