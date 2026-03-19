use std::{env, fs, path::PathBuf};

use psionic_train::{
    ParameterGolfSubmissionChallengeExecutionPosture,
    build_parameter_golf_submission_run_evidence_report,
};

fn main() {
    let submission_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("resolve current directory"));
    let output_path = env::args().nth(2).map(PathBuf::from);
    let report = build_parameter_golf_submission_run_evidence_report(
        &submission_dir,
        &ParameterGolfSubmissionChallengeExecutionPosture::local_review_host_defaults(),
    )
    .expect("build Parameter Golf submission run evidence report");
    let json = serde_json::to_string_pretty(&report).expect("serialize run evidence report");
    if let Some(path) = output_path {
        fs::write(&path, format!("{json}\n")).expect("write run evidence report");
        println!("wrote {}", path.display());
    } else {
        println!("{json}");
    }
}
