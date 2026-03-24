use std::{env, fs, path::PathBuf};

use psionic_train::{
    build_parameter_golf_submission_run_evidence_report,
    ParameterGolfSubmissionChallengeExecutionPosture,
};

fn main() {
    let mut submission_dir = None;
    let mut output_path = None;
    let mut posture =
        ParameterGolfSubmissionChallengeExecutionPosture::local_review_host_defaults();
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--posture" => {
                let posture_id = args.next().expect("missing value for --posture");
                posture = match posture_id.as_str() {
                    "local_review_host" => {
                        ParameterGolfSubmissionChallengeExecutionPosture::local_review_host_defaults(
                        )
                    }
                    "runpod_8xh100" => {
                        ParameterGolfSubmissionChallengeExecutionPosture::runpod_8xh100_defaults()
                    }
                    _ => panic!("unsupported posture `{posture_id}`"),
                };
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: parameter_golf_submission_run_evidence [submission_dir] [output_path] [--posture local_review_host|runpod_8xh100]"
                );
                std::process::exit(0);
            }
            value => {
                if submission_dir.is_none() {
                    submission_dir = Some(PathBuf::from(value));
                } else if output_path.is_none() {
                    output_path = Some(PathBuf::from(value));
                } else {
                    panic!("unexpected extra argument `{value}`");
                }
            }
        }
    }
    let submission_dir = submission_dir
        .unwrap_or_else(|| std::env::current_dir().expect("resolve current directory"));
    let report = build_parameter_golf_submission_run_evidence_report(&submission_dir, &posture)
        .expect("build Parameter Golf submission run evidence report");
    let json = serde_json::to_string_pretty(&report).expect("serialize run evidence report");
    if let Some(path) = output_path {
        fs::write(&path, format!("{json}\n")).expect("write run evidence report");
        println!("wrote {}", path.display());
    } else {
        println!("{json}");
    }
}
