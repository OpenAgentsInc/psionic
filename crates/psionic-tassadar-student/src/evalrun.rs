//! W3 student evaluation harness (openagents#4749): first divergence
//! behind replay, never perplexity.
//!
//! Per eval record and baseline:
//!   * a teacher-forced pass measuring per-slot prediction accuracy
//!     (branch accuracy, memory-read accuracy);
//!   * a free rollout: seeds and step inputs are teacher-forced
//!     (environment input), every output limb is the student's own
//!     emission fed back as context; the first emitted token differing
//!     from the verified trace is the first divergence, classified by
//!     the slot's cause;
//!   * replay acceptance: the shipped psionic executor
//!     (`tassadar_alm_numeric_execute`) re-executes the record's
//!     digest-pinned numeric model on the true inputs and the student's
//!     emitted output rows are compared bitwise — the replay verifier
//!     pointed at student rollouts.
//!
//! Report schema is the plan's, verbatim: exact_rollout_pass@1,
//! median/p90 first-divergence step, valid prefix length, branch
//! accuracy, memory-read accuracy, output digest match rate,
//! replay-verifier acceptance rate, tokens/sec.

use std::collections::BTreeMap;
use std::time::Instant;

use psionic_compiler::{TassadarAlmNumericModel, tassadar_alm_numeric_execute};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::interface::InterfaceModel;
use crate::model::{Backbone, DecodeState, TokenFeatures};
use crate::prep::{
    Family, LIMBS_PER_VALUE, OperandSrc, StudentRecord, TokenRole, build_sequence, family_spec,
    limbs_of, value_of_limbs,
};

/// The student being evaluated.
pub enum Student {
    /// Backbone checkpoint (baselines a/b/c).
    Backbone(Box<Backbone>),
    /// Learned interface + frozen executor (baseline d).
    Interface(Box<InterfaceModel>),
}

/// Evaluation suite of one record.
fn suite_of(record: &StudentRecord) -> String {
    let length = match record.step_count {
        0..=512 => "short",
        513..=1024 => "2x",
        1025..=2048 => "4x",
        _ => "8x",
    };
    match record.family {
        Family::ApplicationStateMachine => format!("heldout_economic_{length}"),
        Family::StackLoopSum => String::from("heldout_anchor"),
        Family::NearMissLookup => format!("adversarial_{length}"),
        Family::MemoryLoadStore => format!("stress_memory_{length}"),
        Family::BranchGatedControl => format!("stress_branch_{length}"),
        Family::ArithmeticCarry => format!("long_arithmetic_{length}"),
    }
}

/// Per-record evaluation result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordEval {
    /// Corpus record id.
    pub record_id: String,
    /// Family id.
    pub family: String,
    /// Suite id.
    pub suite: String,
    /// Steps.
    pub step_count: usize,
    /// Output (corpus) tokens in the record.
    pub output_tokens: usize,
    /// Exact rollout: all output tokens matched.
    pub exact_rollout: bool,
    /// First divergence step (== step_count when exact, censored).
    pub first_divergence_step: usize,
    /// Output tokens before the first divergence.
    pub valid_prefix_tokens: usize,
    /// Divergence cause label (None when exact).
    pub divergence_cause: Option<String>,
    /// Teacher-forced output-token accuracy.
    pub tf_token_accuracy: f64,
    /// Teacher-forced branch accuracy (branch slots; None if n/a).
    pub tf_branch_accuracy: Option<f64>,
    /// Teacher-forced memory-read accuracy (read slots; None if n/a).
    pub tf_memory_read_accuracy: Option<f64>,
    /// Final-output digest matched.
    pub digest_match: bool,
    /// Replay verifier accepted the full student stream.
    pub replay_accepted: bool,
    /// Replay refusal/rejection detail (None when accepted).
    pub replay_detail: Option<String>,
    /// Output tokens emitted per second (rollout, single record).
    pub tokens_per_sec: f64,
}

/// One suite's aggregate row, schema per the plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuiteReport {
    /// Suite id.
    pub suite: String,
    /// Records evaluated.
    pub records: usize,
    /// exact_rollout_pass@1.
    pub exact_rollout_pass_at_1: f64,
    /// Median first-divergence step (censored at step_count for passes).
    pub first_divergence_step_median: f64,
    /// p90 first-divergence step.
    pub first_divergence_step_p90: f64,
    /// Median first-divergence step over diverged records only.
    pub first_divergence_step_median_diverged: Option<f64>,
    /// Median valid prefix length in output tokens.
    pub valid_prefix_tokens_median: f64,
    /// Branch accuracy (teacher-forced; None if no branch slots).
    pub branch_accuracy: Option<f64>,
    /// Memory-read accuracy (teacher-forced; None if no read slots).
    pub memory_read_accuracy: Option<f64>,
    /// Output digest match rate.
    pub output_digest_match_rate: f64,
    /// Replay-verifier acceptance rate.
    pub replay_verifier_acceptance_rate: f64,
    /// Mean rollout output tokens/sec per record.
    pub tokens_per_sec: f64,
    /// Divergence causes histogram.
    pub divergence_causes: BTreeMap<String, usize>,
    /// First-divergence step histogram (bucket -> count).
    pub divergence_step_histogram: BTreeMap<String, usize>,
}

/// Full eval report for one baseline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalReport {
    /// Report schema version.
    pub report_version: String,
    /// Baseline label.
    pub baseline: String,
    /// Corpus id.
    pub corpus_id: String,
    /// Dataset snapshot digest.
    pub dataset_snapshot_digest: String,
    /// Eval prep file sha256.
    pub eval_prep_sha256: String,
    /// Checkpoint weights sha256 (or interface digest for baseline d).
    pub checkpoint_sha256: String,
    /// Config digest of the trained model.
    pub config_digest: String,
    /// Suites.
    pub suites: Vec<SuiteReport>,
    /// Overall aggregate across all eval records.
    pub overall: SuiteReport,
    /// Wall seconds for the whole eval.
    pub wall_seconds: f64,
    /// Eval threads.
    pub threads: usize,
}

fn final_output_digest(program_hash: &str, row: &[i64]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"tassadar_alm_trace_final|");
    hasher.update(program_hash.as_bytes());
    hasher.update(b"|row|");
    for value in row {
        hasher.update(value.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

/// Rollout candidate state for the lookup module (baseline c).
struct LookupState {
    keys: Vec<f64>,
    values: Vec<i64>,
}

#[allow(clippy::too_many_lines)]
fn eval_record(student: &Student, record: &StudentRecord) -> RecordEval {
    let spec = family_spec(record.family);
    let seq = build_sequence(record);
    let suite = suite_of(record);
    let output_tokens = record.step_count * record.s * LIMBS_PER_VALUE;
    // ---------- teacher-forced pass ----------
    let mut tf_correct = 0_usize;
    let mut tf_branch = (0_usize, 0_usize);
    let mut tf_read = (0_usize, 0_usize);
    if let Student::Backbone(model) = student {
        let mut state = DecodeState::new(&model.cfg);
        let mut tf_lookup = LookupState {
            keys: record
                .seed_writes
                .iter()
                .filter(|(channel, _, _)| *channel == 0)
                .map(|(_, key, _)| *key as f64)
                .collect(),
            values: record
                .seed_writes
                .iter()
                .filter(|(channel, _, _)| *channel == 0)
                .map(|(_, _, value)| *value)
                .collect(),
        };
        let mut pending_lookup: Option<[u16; LIMBS_PER_VALUE]> = None;
        for pos in 0..seq.tokens.len() {
            let feat = TokenFeatures {
                feats: seq.feats[pos],
                limb: seq.limb[pos],
                role: match seq.roles[pos] {
                    TokenRole::Seed => 0,
                    TokenRole::Input => 1,
                    TokenRole::Output => 2,
                },
                token: seq.tokens[pos],
                vidx: seq.value_idx[pos],
            };
            let hidden = model.decode_step(&mut state, &feat);
            let next = pos + 1;
            if next >= seq.tokens.len() || seq.roles[next] != TokenRole::Output {
                // Maintain lookup candidates at end of each step block.
                if model.cfg.use_lookup {
                    maintain_candidates_tf(record, &seq, pos, &mut tf_lookup);
                }
                continue;
            }
            let step = seq.step_of[next];
            let out_idx = seq.out_idx_of[next] as usize;
            let limb = seq.limb[next] as usize;
            let mut predicted = model.predict_token(&hidden);
            if model.cfg.use_lookup {
                if let Some(read) = spec.read {
                    if out_idx == read.result_out_idx {
                        if limb == 0 {
                            let query = record.inputs[step * record.f + read.query_input];
                            pending_lookup = lookup_select(
                                model,
                                &hidden,
                                query as f64,
                                &tf_lookup,
                            );
                        }
                        if let Some(value_limbs) = pending_lookup {
                            predicted = value_limbs[limb];
                        }
                    }
                }
            }
            let truth = seq.tokens[next];
            if predicted == truth {
                tf_correct += 1;
            }
            if spec.branch_selected_out_idxs.contains(&out_idx) {
                tf_branch.1 += 1;
                if predicted == truth {
                    tf_branch.0 += 1;
                }
            }
            if let Some(read) = spec.read {
                if out_idx == read.result_out_idx {
                    tf_read.1 += 1;
                    if predicted == truth {
                        tf_read.0 += 1;
                    }
                }
            }
            if model.cfg.use_lookup {
                maintain_candidates_tf(record, &seq, pos, &mut tf_lookup);
            }
        }
    }
    // ---------- free rollout ----------
    let started = Instant::now();
    let mut emitted: Vec<u16> = Vec::with_capacity(output_tokens);
    let mut student_rows: Vec<Vec<i64>> = Vec::with_capacity(record.step_count);
    match student {
        Student::Backbone(model) => {
            let mut state = DecodeState::new(&model.cfg);
            let mut lk = LookupState {
                keys: record
                    .seed_writes
                    .iter()
                    .filter(|(channel, _, _)| *channel == 0)
                    .map(|(_, key, _)| *key as f64)
                    .collect(),
                values: record
                    .seed_writes
                    .iter()
                    .filter(|(channel, _, _)| *channel == 0)
                    .map(|(_, _, value)| *value)
                    .collect(),
            };
            // Seed preamble (teacher-forced).
            let mut hidden = vec![0.0_f32; model.cfg.d_model];
            let seed_tokens = record.seed_writes.len() * 2 * LIMBS_PER_VALUE;
            for pos in 0..seed_tokens {
                let feat = TokenFeatures {
                    feats: seq.feats[pos],
                    limb: seq.limb[pos],
                    role: 0,
                    token: seq.tokens[pos],
                    vidx: seq.value_idx[pos],
                };
                hidden = model.decode_step(&mut state, &feat);
            }
            let mut last_completed: i64 = record
                .seed_writes
                .last()
                .map_or(0, |(_, _, value)| *value);
            for step in 0..record.step_count {
                // Inputs: teacher-forced.
                for field in 0..record.f {
                    let value = record.inputs[step * record.f + field];
                    let limbs = limbs_of(value);
                    for (limb_idx, limb) in limbs.iter().enumerate() {
                        let feat = TokenFeatures {
                            feats: feats_of(last_completed),
                            limb: limb_idx as u8,
                            role: 1,
                            token: *limb,
                            vidx: field as u8,
                        };
                        hidden = model.decode_step(&mut state, &feat);
                    }
                    last_completed = value;
                }
                // Outputs: student emission fed back.
                let mut row: Vec<i64> = Vec::with_capacity(record.s);
                let mut pending_lookup: Option<[u16; LIMBS_PER_VALUE]> = None;
                for out in 0..record.s {
                    let mut value_limbs = [0_u16; LIMBS_PER_VALUE];
                    for limb_idx in 0..LIMBS_PER_VALUE {
                        let mut predicted = model.predict_token(&hidden);
                        if model.cfg.use_lookup {
                            if let Some(read) = spec.read {
                                if out == read.result_out_idx {
                                    if limb_idx == 0 {
                                        let query =
                                            record.inputs[step * record.f + read.query_input];
                                        pending_lookup =
                                            lookup_select(model, &hidden, query as f64, &lk);
                                    }
                                    if let Some(sel) = pending_lookup {
                                        predicted = sel[limb_idx];
                                    }
                                }
                            }
                        }
                        value_limbs[limb_idx] = predicted;
                        emitted.push(predicted);
                        let feat = TokenFeatures {
                            feats: feats_of(last_completed),
                            limb: limb_idx as u8,
                            role: 2,
                            token: predicted,
                            vidx: (record.f + out) as u8,
                        };
                        hidden = model.decode_step(&mut state, &feat);
                    }
                    let value = value_of_limbs(&value_limbs);
                    last_completed = value;
                    row.push(value);
                }
                // Candidate maintenance from the ROLLOUT's view: write
                // operands come from true inputs or the student's own
                // emitted outputs.
                if model.cfg.use_lookup {
                    if let Some(read) = spec.read {
                        let write_key = match read.write_key {
                            OperandSrc::Input(index) => record.inputs[step * record.f + index],
                            OperandSrc::Output(index) => row[index],
                        };
                        let write_value = match read.write_value {
                            OperandSrc::Input(index) => record.inputs[step * record.f + index],
                            OperandSrc::Output(index) => row[index],
                        };
                        lk.keys.push(write_key as f64);
                        lk.values.push(write_value);
                    }
                }
                student_rows.push(row);
            }
        }
        Student::Interface(interface) => {
            // Parse inputs through the learned assignment, run the
            // frozen executor once, emit through the learned routing.
            let input_assign = interface.input_assignment();
            let output_assign = interface.output_assignment();
            let routing = interface.routing(record.s);
            let mut steps: Vec<Vec<i64>> = Vec::with_capacity(record.step_count);
            for step in 0..record.step_count {
                let mut fields = Vec::with_capacity(record.f);
                for field in 0..record.f {
                    let value = record.inputs[step * record.f + field];
                    let stream_limbs = limbs_of(value); // protocol emission
                    fields.push(interface.assemble(&stream_limbs, &input_assign));
                }
                steps.push(fields);
            }
            let model: Result<TassadarAlmNumericModel, _> = record
                .model_json
                .as_deref()
                .map(serde_json::from_str)
                .unwrap_or_else(|| serde_json::from_str(""));
            if let Ok(model) = model { match tassadar_alm_numeric_execute(&model, &steps) {
                Ok(trace) => {
                    for row in &trace.step_outputs {
                        let mut stream_row = Vec::with_capacity(record.s);
                        for stream_idx in 0..record.s {
                            let executor_idx = routing[stream_idx].min(row.len() - 1);
                            let value = row[executor_idx];
                            let limbs = interface.emit(value, &output_assign);
                            emitted.extend_from_slice(&limbs);
                            stream_row.push(value_of_limbs(&limbs));
                        }
                        student_rows.push(stream_row);
                    }
                }
                Err(error) => {
                    // Typed executor refusal: rollout ends with no
                    // emissions beyond this point.
                    let _ = error;
                }
            } }
        }
    }
    let rollout_seconds = started.elapsed().as_secs_f64().max(1e-9);
    // ---------- divergence vs verified ----------
    let mut first_div_token: Option<usize> = None;
    for (index, token) in emitted.iter().enumerate() {
        let step = index / (record.s * LIMBS_PER_VALUE);
        let within = index % (record.s * LIMBS_PER_VALUE);
        let out_idx = within / LIMBS_PER_VALUE;
        let truth = limbs_of(record.outputs[step * record.s + out_idx])
            [within % LIMBS_PER_VALUE];
        if *token != truth {
            first_div_token = Some(index);
            break;
        }
    }
    if emitted.len() < output_tokens && first_div_token.is_none() {
        // Interface refusal or truncation: divergence at truncation point.
        first_div_token = Some(emitted.len());
    }
    let (exact_rollout, first_divergence_step, valid_prefix_tokens, divergence_cause) =
        match first_div_token {
            None => (true, record.step_count, output_tokens, None),
            Some(token_index) => {
                let step = token_index / (record.s * LIMBS_PER_VALUE);
                let out_idx =
                    (token_index % (record.s * LIMBS_PER_VALUE)) / LIMBS_PER_VALUE;
                let cause = spec
                    .causes
                    .get(out_idx)
                    .map_or("output", |cause| cause.label());
                (false, step, token_index, Some(cause.to_string()))
            }
        };
    // ---------- digest + replay acceptance ----------
    let digest_match = student_rows.last().is_some_and(|row| {
        final_output_digest(&record.program_hash, row) == record.final_output_digest
    });
    let (replay_accepted, replay_detail) = replay_acceptance(record, &student_rows);
    RecordEval {
        digest_match,
        divergence_cause,
        exact_rollout,
        family: record.family.id().to_string(),
        first_divergence_step,
        output_tokens,
        record_id: record.record_id.clone(),
        replay_accepted,
        replay_detail,
        step_count: record.step_count,
        suite,
        tf_branch_accuracy: (tf_branch.1 > 0)
            .then(|| tf_branch.0 as f64 / tf_branch.1 as f64),
        tf_memory_read_accuracy: (tf_read.1 > 0)
            .then(|| tf_read.0 as f64 / tf_read.1 as f64),
        tf_token_accuracy: if output_tokens > 0 {
            tf_correct as f64 / output_tokens as f64
        } else {
            0.0
        },
        tokens_per_sec: emitted.len() as f64 / rollout_seconds,
        valid_prefix_tokens,
    }
}

fn feats_of(value: i64) -> [f32; 2] {
    let v = value as f64;
    [
        ((v / f64::from(1 << 20)).tanh()) as f32,
        ((v / (2_f64).powi(40)).tanh()) as f32,
    ]
}

/// Teacher-forced candidate maintenance: when `pos` is the last token of
/// a step block, push that step's write (from true values).
fn maintain_candidates_tf(
    record: &StudentRecord,
    seq: &crate::prep::StudentSequence,
    pos: usize,
    lk: &mut LookupState,
) {
    let spec = family_spec(record.family);
    let Some(read) = spec.read else { return };
    let step = seq.step_of.get(pos).copied().unwrap_or(usize::MAX);
    if step == usize::MAX {
        return;
    }
    let step_end = seq.step_starts[step] + (record.f + record.s) * LIMBS_PER_VALUE - 1;
    if pos != step_end {
        return;
    }
    let write_key = match read.write_key {
        OperandSrc::Input(index) => record.inputs[step * record.f + index],
        OperandSrc::Output(index) => record.outputs[step * record.s + index],
    };
    let write_value = match read.write_value {
        OperandSrc::Input(index) => record.inputs[step * record.f + index],
        OperandSrc::Output(index) => record.outputs[step * record.s + index],
    };
    lk.keys.push(write_key as f64);
    lk.values.push(write_value);
}

/// Hard-max parabolic selection; ties break to the latest write,
/// mirroring the executor.
fn lookup_select(
    model: &Backbone,
    hidden: &[f32],
    query: f64,
    lk: &LookupState,
) -> Option<[u16; LIMBS_PER_VALUE]> {
    if lk.keys.is_empty() {
        return None;
    }
    let beta_q = f64::from(model.lk_scalars.w[0]);
    let b_q = f64::from(model.lk_scalars.w[1]);
    let beta_k = f64::from(model.lk_scalars.w[2]);
    let b_k = f64::from(model.lk_scalars.w[3]);
    let score_scale = f64::from(model.lk_scalars.w[4]);
    let mut wq_dot = 0.0_f64;
    for (h_val, w_val) in hidden.iter().zip(model.lk_wq.w.iter()) {
        wq_dot += f64::from(*h_val) * f64::from(*w_val);
    }
    let q_prime = beta_q * query + b_q + wq_dot;
    let mut best = 0_usize;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, key) in lk.keys.iter().enumerate() {
        let delta = beta_k * key + b_k - q_prime;
        let score = -score_scale * delta * delta;
        if score >= best_score {
            best_score = score;
            best = idx;
        }
    }
    Some(limbs_of(lk.values[best]))
}

/// Replay acceptance: re-execute the digest-pinned model on the true
/// inputs with the shipped psionic executor and compare the student's
/// rows bitwise.
fn replay_acceptance(
    record: &StudentRecord,
    student_rows: &[Vec<i64>],
) -> (bool, Option<String>) {
    let Some(model_json) = record.model_json.as_deref() else {
        return (false, Some(String::from("missing model json")));
    };
    let model: TassadarAlmNumericModel = match serde_json::from_str(model_json) {
        Ok(model) => model,
        Err(error) => return (false, Some(format!("model parse: {error}"))),
    };
    if model.graph_digest != record.program_hash {
        return (
            false,
            Some(String::from("model digest does not match record")),
        );
    }
    let steps: Vec<Vec<i64>> = (0..record.step_count)
        .map(|step| record.inputs[step * record.f..(step + 1) * record.f].to_vec())
        .collect();
    match tassadar_alm_numeric_execute(&model, &steps) {
        Err(error) => (false, Some(format!("replay refused: {error:?}"))),
        Ok(trace) => {
            if trace.step_outputs.len() != student_rows.len() {
                return (
                    false,
                    Some(format!(
                        "student emitted {} rows, replay produced {}",
                        student_rows.len(),
                        trace.step_outputs.len()
                    )),
                );
            }
            for (step, (replay_row, student_row)) in trace
                .step_outputs
                .iter()
                .zip(student_rows.iter())
                .enumerate()
            {
                if replay_row != student_row {
                    return (
                        false,
                        Some(format!("first row mismatch at step {step}")),
                    );
                }
            }
            (true, None)
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn step_bucket(step: usize) -> String {
    let bounds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    for bound in bounds {
        if step < bound {
            return format!("<{bound}");
        }
    }
    String::from(">=4096")
}

fn aggregate(suite: &str, evals: &[&RecordEval]) -> SuiteReport {
    let records = evals.len();
    let mut div_steps: Vec<f64> = evals
        .iter()
        .map(|e| e.first_divergence_step as f64)
        .collect();
    div_steps.sort_by(f64::total_cmp);
    let mut diverged: Vec<f64> = evals
        .iter()
        .filter(|e| !e.exact_rollout)
        .map(|e| e.first_divergence_step as f64)
        .collect();
    diverged.sort_by(f64::total_cmp);
    let mut prefixes: Vec<f64> = evals
        .iter()
        .map(|e| e.valid_prefix_tokens as f64)
        .collect();
    prefixes.sort_by(f64::total_cmp);
    let branch: Vec<f64> = evals
        .iter()
        .filter_map(|e| e.tf_branch_accuracy)
        .collect();
    let reads: Vec<f64> = evals
        .iter()
        .filter_map(|e| e.tf_memory_read_accuracy)
        .collect();
    let mut causes = BTreeMap::new();
    let mut histogram = BTreeMap::new();
    for eval in evals {
        if let Some(cause) = &eval.divergence_cause {
            *causes.entry(cause.clone()).or_insert(0) += 1;
            *histogram
                .entry(step_bucket(eval.first_divergence_step))
                .or_insert(0) += 1;
        }
    }
    SuiteReport {
        branch_accuracy: (!branch.is_empty())
            .then(|| branch.iter().sum::<f64>() / branch.len() as f64),
        divergence_causes: causes,
        divergence_step_histogram: histogram,
        exact_rollout_pass_at_1: evals.iter().filter(|e| e.exact_rollout).count() as f64
            / records.max(1) as f64,
        first_divergence_step_median: percentile(&div_steps, 0.5),
        first_divergence_step_median_diverged: (!diverged.is_empty())
            .then(|| percentile(&diverged, 0.5)),
        first_divergence_step_p90: percentile(&div_steps, 0.9),
        memory_read_accuracy: (!reads.is_empty())
            .then(|| reads.iter().sum::<f64>() / reads.len() as f64),
        output_digest_match_rate: evals.iter().filter(|e| e.digest_match).count() as f64
            / records.max(1) as f64,
        records,
        replay_verifier_acceptance_rate: evals.iter().filter(|e| e.replay_accepted).count()
            as f64
            / records.max(1) as f64,
        suite: suite.to_string(),
        tokens_per_sec: evals.iter().map(|e| e.tokens_per_sec).sum::<f64>()
            / records.max(1) as f64,
        valid_prefix_tokens_median: percentile(&prefixes, 0.5),
    }
}

/// Runs the full evaluation over the eval records.
pub fn run_eval(
    student: &Student,
    records: &[StudentRecord],
    baseline: &str,
    corpus_id: &str,
    snapshot_digest: &str,
    eval_prep_sha256: &str,
    checkpoint_sha256: &str,
    config_digest: &str,
) -> EvalReport {
    let started = Instant::now();
    let evals: Vec<RecordEval> = records
        .par_iter()
        .map(|record| eval_record(student, record))
        .collect();
    let mut by_suite: BTreeMap<String, Vec<&RecordEval>> = BTreeMap::new();
    for eval in &evals {
        by_suite.entry(eval.suite.clone()).or_default().push(eval);
    }
    let suites: Vec<SuiteReport> = by_suite
        .iter()
        .map(|(suite, members)| aggregate(suite, members))
        .collect();
    let all: Vec<&RecordEval> = evals.iter().collect();
    let overall = aggregate("overall", &all);
    EvalReport {
        baseline: baseline.to_string(),
        checkpoint_sha256: checkpoint_sha256.to_string(),
        config_digest: config_digest.to_string(),
        corpus_id: corpus_id.to_string(),
        dataset_snapshot_digest: snapshot_digest.to_string(),
        eval_prep_sha256: eval_prep_sha256.to_string(),
        overall,
        report_version: String::from("tassadar_student_eval_report.v0.1"),
        suites,
        threads: rayon::current_num_threads(),
        wall_seconds: started.elapsed().as_secs_f64(),
    }
}
