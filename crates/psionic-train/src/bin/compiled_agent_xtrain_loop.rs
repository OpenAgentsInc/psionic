use psionic_train::{
    compiled_agent_grounded_candidate_report_fixture_path,
    compiled_agent_route_candidate_report_fixture_path,
    compiled_agent_xtrain_cycle_receipt_fixture_path,
    write_compiled_agent_grounded_candidate_report, write_compiled_agent_route_candidate_report,
    write_compiled_agent_xtrain_cycle_receipt,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let route_path = compiled_agent_route_candidate_report_fixture_path();
    let route_report = write_compiled_agent_route_candidate_report(&route_path)?;
    let grounded_path = compiled_agent_grounded_candidate_report_fixture_path();
    let grounded_report = write_compiled_agent_grounded_candidate_report(&grounded_path)?;
    let cycle_path = compiled_agent_xtrain_cycle_receipt_fixture_path();
    let cycle = write_compiled_agent_xtrain_cycle_receipt(&cycle_path)?;

    println!(
        "wrote route candidate module eval report={} digest={}",
        route_path.display(),
        route_report.report_digest
    );
    println!(
        "wrote grounded candidate module eval report={} digest={}",
        grounded_path.display(),
        grounded_report.report_digest
    );
    println!(
        "wrote compiled-agent xtrain cycle receipt={} digest={}",
        cycle_path.display(),
        cycle.receipt_digest
    );
    Ok(())
}
