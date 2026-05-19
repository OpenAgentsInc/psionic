//! Resumable legal benchmark sweep planning and manifests.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{Metadata, stable_json_digest};

pub const LEGAL_BENCHMARK_SWEEP_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegalBenchmarkSweepScope {
    TaskIds(Vec<String>),
    PracticeArea(String),
    Workflow(String),
    Slice { name: String, task_ids: Vec<String> },
    All,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkSweepBudget {
    pub max_cost_micro_usd: Option<u64>,
    pub max_wall_time_ms: Option<u64>,
    pub max_tokens: Option<u64>,
    pub max_failures: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkSweepConfig {
    pub schema_version: u16,
    pub sweep_id: String,
    pub scope: LegalBenchmarkSweepScope,
    pub task_ids: Vec<String>,
    pub run_config_hashes: Vec<String>,
    pub max_parallelism: u16,
    pub budget: LegalBenchmarkSweepBudget,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegalBenchmarkSweepJobStatus {
    Pending,
    Skipped,
    Resumed,
    Succeeded,
    Failed,
    Blocked,
    BudgetExhausted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkSweepJob {
    pub job_id: String,
    pub task_id: String,
    pub run_config_hash: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkSweepJobRecord {
    pub job_id: String,
    pub task_id: String,
    pub run_config_hash: String,
    pub status: LegalBenchmarkSweepJobStatus,
    pub run_id: Option<String>,
    pub score_report_id: Option<String>,
    pub score_report_hash: Option<String>,
    pub cost_micro_usd: u64,
    pub wall_time_ms: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub failure_kind: Option<String>,
    pub failure_detail: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkSweepResumeState {
    pub schema_version: u16,
    pub sweep_id: String,
    pub completed_jobs: Vec<LegalBenchmarkSweepJobRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegalBenchmarkSweepManifest {
    pub schema_version: u16,
    pub sweep_id: String,
    pub config_hash: String,
    pub generated_at_ms: u64,
    pub total_jobs: u64,
    pub skipped_jobs: u64,
    pub resumed_jobs: u64,
    pub succeeded_jobs: u64,
    pub failed_jobs: u64,
    pub blocked_jobs: u64,
    pub budget_exhausted_jobs: u64,
    pub total_cost_micro_usd: u64,
    pub total_wall_time_ms: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub jobs: Vec<LegalBenchmarkSweepJobRecord>,
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegalBenchmarkSweepJobOutcome {
    pub status: LegalBenchmarkSweepJobStatus,
    pub run_id: Option<String>,
    pub score_report_id: Option<String>,
    pub score_report_hash: Option<String>,
    pub cost_micro_usd: u64,
    pub wall_time_ms: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub failure_kind: Option<String>,
    pub failure_detail: Option<String>,
}

pub trait LegalBenchmarkSweepExecutor {
    fn execute(&mut self, job: &LegalBenchmarkSweepJob) -> LegalBenchmarkSweepJobOutcome;
}

#[derive(Clone, Debug, Default)]
pub struct MockLegalBenchmarkSweepExecutor {
    outcomes: BTreeMap<String, LegalBenchmarkSweepJobOutcome>,
}

impl MockLegalBenchmarkSweepExecutor {
    pub fn new(outcomes: BTreeMap<String, LegalBenchmarkSweepJobOutcome>) -> Self {
        Self { outcomes }
    }
}

impl LegalBenchmarkSweepExecutor for MockLegalBenchmarkSweepExecutor {
    fn execute(&mut self, job: &LegalBenchmarkSweepJob) -> LegalBenchmarkSweepJobOutcome {
        self.outcomes
            .remove(&job.job_id)
            .unwrap_or_else(|| LegalBenchmarkSweepJobOutcome {
                status: LegalBenchmarkSweepJobStatus::Succeeded,
                run_id: Some(format!("run.{}", job.job_id)),
                score_report_id: Some(format!("score.{}", job.job_id)),
                score_report_hash: Some(String::from(
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                )),
                cost_micro_usd: 1,
                wall_time_ms: 1,
                input_tokens: 1,
                output_tokens: 1,
                failure_kind: None,
                failure_detail: None,
            })
    }
}

pub fn plan_legal_benchmark_sweep_jobs(
    config: &LegalBenchmarkSweepConfig,
) -> Vec<LegalBenchmarkSweepJob> {
    let mut jobs = Vec::new();
    let task_ids = scoped_task_ids(config);
    for task_id in task_ids {
        for run_config_hash in &config.run_config_hashes {
            jobs.push(LegalBenchmarkSweepJob {
                job_id: format!(
                    "job.{}.{}",
                    stable_label(&task_id),
                    stable_label(run_config_hash)
                ),
                task_id: task_id.clone(),
                run_config_hash: run_config_hash.clone(),
            });
        }
    }
    jobs
}

pub fn run_legal_benchmark_sweep<E>(
    config: &LegalBenchmarkSweepConfig,
    resume_state: Option<&LegalBenchmarkSweepResumeState>,
    executor: &mut E,
) -> Result<LegalBenchmarkSweepManifest, serde_json::Error>
where
    E: LegalBenchmarkSweepExecutor,
{
    let config_hash = legal_benchmark_sweep_config_hash(config)?;
    let resume_by_job = resume_state
        .map(|state| {
            state
                .completed_jobs
                .iter()
                .map(|record| (record.job_id.clone(), record.clone()))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let mut jobs = Vec::new();
    let mut totals = SweepTotals::default();
    let mut exhausted = false;
    for job in plan_legal_benchmark_sweep_jobs(config) {
        if let Some(resumed) = resume_by_job.get(&job.job_id) {
            let mut record = resumed.clone();
            record.status = LegalBenchmarkSweepJobStatus::Resumed;
            totals.add(&record);
            jobs.push(record);
            continue;
        }
        if exhausted || totals.exceeds_before_start(&config.budget) {
            let record = blocked_job_record(&job, LegalBenchmarkSweepJobStatus::BudgetExhausted);
            totals.add(&record);
            jobs.push(record);
            exhausted = true;
            continue;
        }
        let outcome = executor.execute(&job);
        let record = job_record_from_outcome(&job, outcome);
        totals.add(&record);
        if totals.exceeds_after_finish(&config.budget) {
            exhausted = true;
        }
        jobs.push(record);
    }
    Ok(manifest_from_jobs(config, config_hash, jobs, totals))
}

pub fn legal_benchmark_sweep_config_hash(
    config: &LegalBenchmarkSweepConfig,
) -> Result<String, serde_json::Error> {
    stable_json_digest("psionic.legal_benchmark.sweep_config.v1", config)
}

pub fn legal_benchmark_sweep_manifest_hash(
    manifest: &LegalBenchmarkSweepManifest,
) -> Result<String, serde_json::Error> {
    stable_json_digest("psionic.legal_benchmark.sweep_manifest.v1", manifest)
}

fn scoped_task_ids(config: &LegalBenchmarkSweepConfig) -> Vec<String> {
    match &config.scope {
        LegalBenchmarkSweepScope::TaskIds(task_ids) => task_ids.clone(),
        LegalBenchmarkSweepScope::Slice { task_ids, .. } => task_ids.clone(),
        _ => config.task_ids.clone(),
    }
}

fn job_record_from_outcome(
    job: &LegalBenchmarkSweepJob,
    outcome: LegalBenchmarkSweepJobOutcome,
) -> LegalBenchmarkSweepJobRecord {
    LegalBenchmarkSweepJobRecord {
        job_id: job.job_id.clone(),
        task_id: job.task_id.clone(),
        run_config_hash: job.run_config_hash.clone(),
        status: outcome.status,
        run_id: outcome.run_id,
        score_report_id: outcome.score_report_id,
        score_report_hash: outcome.score_report_hash,
        cost_micro_usd: outcome.cost_micro_usd,
        wall_time_ms: outcome.wall_time_ms,
        input_tokens: outcome.input_tokens,
        output_tokens: outcome.output_tokens,
        failure_kind: outcome.failure_kind,
        failure_detail: outcome.failure_detail,
    }
}

fn blocked_job_record(
    job: &LegalBenchmarkSweepJob,
    status: LegalBenchmarkSweepJobStatus,
) -> LegalBenchmarkSweepJobRecord {
    LegalBenchmarkSweepJobRecord {
        job_id: job.job_id.clone(),
        task_id: job.task_id.clone(),
        run_config_hash: job.run_config_hash.clone(),
        status,
        run_id: None,
        score_report_id: None,
        score_report_hash: None,
        cost_micro_usd: 0,
        wall_time_ms: 0,
        input_tokens: 0,
        output_tokens: 0,
        failure_kind: Some(String::from("budget")),
        failure_detail: Some(String::from("budget exhausted before starting job")),
    }
}

fn manifest_from_jobs(
    config: &LegalBenchmarkSweepConfig,
    config_hash: String,
    jobs: Vec<LegalBenchmarkSweepJobRecord>,
    totals: SweepTotals,
) -> LegalBenchmarkSweepManifest {
    LegalBenchmarkSweepManifest {
        schema_version: LEGAL_BENCHMARK_SWEEP_SCHEMA_VERSION,
        sweep_id: config.sweep_id.clone(),
        config_hash,
        generated_at_ms: now_ms(),
        total_jobs: u64::try_from(jobs.len()).unwrap_or(u64::MAX),
        skipped_jobs: totals.status_count(LegalBenchmarkSweepJobStatus::Skipped),
        resumed_jobs: totals.status_count(LegalBenchmarkSweepJobStatus::Resumed),
        succeeded_jobs: totals.status_count(LegalBenchmarkSweepJobStatus::Succeeded),
        failed_jobs: totals.status_count(LegalBenchmarkSweepJobStatus::Failed),
        blocked_jobs: totals.status_count(LegalBenchmarkSweepJobStatus::Blocked),
        budget_exhausted_jobs: totals.status_count(LegalBenchmarkSweepJobStatus::BudgetExhausted),
        total_cost_micro_usd: totals.cost_micro_usd,
        total_wall_time_ms: totals.wall_time_ms,
        total_input_tokens: totals.input_tokens,
        total_output_tokens: totals.output_tokens,
        jobs,
        metadata: Metadata::new(),
    }
}

#[derive(Default)]
struct SweepTotals {
    cost_micro_usd: u64,
    wall_time_ms: u64,
    input_tokens: u64,
    output_tokens: u64,
    failures: u64,
    statuses: BTreeMap<LegalBenchmarkSweepJobStatus, u64>,
}

impl SweepTotals {
    fn add(&mut self, record: &LegalBenchmarkSweepJobRecord) {
        self.cost_micro_usd = self.cost_micro_usd.saturating_add(record.cost_micro_usd);
        self.wall_time_ms = self.wall_time_ms.saturating_add(record.wall_time_ms);
        self.input_tokens = self.input_tokens.saturating_add(record.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(record.output_tokens);
        if record.status == LegalBenchmarkSweepJobStatus::Failed {
            self.failures = self.failures.saturating_add(1);
        }
        *self.statuses.entry(record.status).or_default() += 1;
    }

    fn status_count(&self, status: LegalBenchmarkSweepJobStatus) -> u64 {
        self.statuses.get(&status).copied().unwrap_or(0)
    }

    fn exceeds_before_start(&self, budget: &LegalBenchmarkSweepBudget) -> bool {
        budget.max_failures.is_some_and(|max| self.failures >= max)
            || budget
                .max_cost_micro_usd
                .is_some_and(|max| self.cost_micro_usd >= max)
            || budget
                .max_wall_time_ms
                .is_some_and(|max| self.wall_time_ms >= max)
            || budget
                .max_tokens
                .is_some_and(|max| self.input_tokens.saturating_add(self.output_tokens) >= max)
    }

    fn exceeds_after_finish(&self, budget: &LegalBenchmarkSweepBudget) -> bool {
        self.exceeds_before_start(budget)
    }
}

fn stable_label(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '.'
            }
        })
        .collect()
}

fn now_ms() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
        Err(_) => 0,
    }
}

impl Ord for LegalBenchmarkSweepJobStatus {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl PartialOrd for LegalBenchmarkSweepJobStatus {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> LegalBenchmarkSweepConfig {
        LegalBenchmarkSweepConfig {
            schema_version: LEGAL_BENCHMARK_SWEEP_SCHEMA_VERSION,
            sweep_id: String::from("sweep.mock"),
            scope: LegalBenchmarkSweepScope::Slice {
                name: String::from("smoke"),
                task_ids: vec![String::from("task.a"), String::from("task.b")],
            },
            task_ids: Vec::new(),
            run_config_hashes: vec![String::from("config.1"), String::from("config.2")],
            max_parallelism: 2,
            budget: LegalBenchmarkSweepBudget {
                max_cost_micro_usd: Some(100),
                max_wall_time_ms: Some(1000),
                max_tokens: Some(1000),
                max_failures: Some(2),
            },
            metadata: Metadata::new(),
        }
    }

    #[test]
    fn plans_cross_product_jobs_for_slice_scope() {
        let jobs = plan_legal_benchmark_sweep_jobs(&config());
        assert_eq!(jobs.len(), 4);
        assert_eq!(jobs[0].job_id, "job.task.a.config.1");
        assert_eq!(jobs[3].job_id, "job.task.b.config.2");
    }

    #[test]
    fn sweep_resumes_completed_jobs_and_keeps_running_failures() {
        let cfg = config();
        let resumed = LegalBenchmarkSweepJobRecord {
            job_id: String::from("job.task.a.config.1"),
            task_id: String::from("task.a"),
            run_config_hash: String::from("config.1"),
            status: LegalBenchmarkSweepJobStatus::Succeeded,
            run_id: Some(String::from("run.old")),
            score_report_id: Some(String::from("score.old")),
            score_report_hash: Some(String::from("hash.old")),
            cost_micro_usd: 3,
            wall_time_ms: 4,
            input_tokens: 5,
            output_tokens: 6,
            failure_kind: None,
            failure_detail: None,
        };
        let mut outcomes = BTreeMap::new();
        outcomes.insert(
            String::from("job.task.a.config.2"),
            LegalBenchmarkSweepJobOutcome {
                status: LegalBenchmarkSweepJobStatus::Failed,
                run_id: Some(String::from("run.failed")),
                score_report_id: None,
                score_report_hash: None,
                cost_micro_usd: 2,
                wall_time_ms: 3,
                input_tokens: 4,
                output_tokens: 5,
                failure_kind: Some(String::from("provider_failure")),
                failure_detail: Some(String::from("mock failure")),
            },
        );
        let resume = LegalBenchmarkSweepResumeState {
            schema_version: LEGAL_BENCHMARK_SWEEP_SCHEMA_VERSION,
            sweep_id: cfg.sweep_id.clone(),
            completed_jobs: vec![resumed],
        };
        let mut executor = MockLegalBenchmarkSweepExecutor::new(outcomes);
        let manifest =
            run_legal_benchmark_sweep(&cfg, Some(&resume), &mut executor).expect("sweep");
        assert_eq!(manifest.total_jobs, 4);
        assert_eq!(manifest.resumed_jobs, 1);
        assert_eq!(manifest.failed_jobs, 1);
        assert_eq!(manifest.succeeded_jobs, 2);
        assert_eq!(manifest.budget_exhausted_jobs, 0);
        assert!(legal_benchmark_sweep_manifest_hash(&manifest).is_ok());
    }

    #[test]
    fn budget_exhaustion_blocks_new_work() {
        let mut cfg = config();
        cfg.budget.max_cost_micro_usd = Some(1);
        let mut executor = MockLegalBenchmarkSweepExecutor::default();
        let manifest = run_legal_benchmark_sweep(&cfg, None, &mut executor).expect("sweep");
        assert_eq!(manifest.succeeded_jobs, 1);
        assert!(manifest.budget_exhausted_jobs >= 1);
    }
}
