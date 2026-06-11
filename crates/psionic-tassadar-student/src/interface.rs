//! Baseline (d): frozen analytic executor + learned interface
//! (openagents#4749, RESEARCH_PLAN.md W3 — the H2 experiment).
//!
//! The frozen core is the digest-pinned Tassadar ALM numeric executor
//! (psionic-compiler `tassadar_alm_numeric_execute`); nothing in it is
//! trained. The learned interface is the marshaling layer between the
//! student token protocol and the executor ABI, trained from the same
//! verified traces as the other baselines:
//!
//!   * input limb assembly: which 16-bit slice of an i64 each of the 4
//!     token limbs of an input value carries (a 4x4 assignment learned
//!     by gradient on a matching likelihood);
//!   * output limb slicing: the same assignment for emitted values;
//!   * output routing: how the executor's output row indexes map onto
//!     stream value positions (identity-vs-permutation logits over a
//!     16x16 table, trained on value matches).
//!
//! If the interface learns the marshaling exactly, every emission is an
//! executor output and rollouts are exact at every length — that is the
//! H2 bet, and the eval harness checks it with the same first-divergence
//! metric as everything else.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::prep::{LIMBS_PER_VALUE, StudentRecord, limbs_of};

/// Maximum output row width supported by the routing table.
pub const MAX_OUT: usize = 16;

/// Learned interface parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterfaceModel {
    /// Input limb-position -> exponent-slot assignment logits (4x4).
    pub a_logits: Vec<Vec<f32>>,
    /// Output limb-position -> exponent-slot assignment logits (4x4).
    pub b_logits: Vec<Vec<f32>>,
    /// Stream output index -> executor row index logits (16x16).
    pub p_logits: Vec<Vec<f32>>,
}

impl InterfaceModel {
    /// Zero (uniform) init: no marshaling knowledge.
    pub fn new() -> Self {
        Self {
            a_logits: vec![vec![0.0; LIMBS_PER_VALUE]; LIMBS_PER_VALUE],
            b_logits: vec![vec![0.0; LIMBS_PER_VALUE]; LIMBS_PER_VALUE],
            p_logits: vec![vec![0.0; MAX_OUT]; MAX_OUT],
        }
    }

    /// Stable digest over the parameter encoding.
    pub fn stable_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"tassadar_student_interface|");
        hasher.update(serde_json::to_vec(self).unwrap_or_default());
        hex::encode(hasher.finalize())
    }

    /// Limb assignment for input assembly, decoded under an
    /// injectivity constraint: an ABI maps each limb position to a
    /// distinct 16-bit slice, so ties the training distribution never
    /// disambiguates (e.g. all-positive values below 2^32 leave the two
    /// high limbs interchangeable) resolve to a permutation instead of
    /// a collision.
    pub fn input_assignment(&self) -> [usize; LIMBS_PER_VALUE] {
        injective_assignment(&self.a_logits)
    }

    /// Limb assignment for output slicing (injective, see above).
    pub fn output_assignment(&self) -> [usize; LIMBS_PER_VALUE] {
        injective_assignment(&self.b_logits)
    }

    /// Argmax routing for one stream width.
    pub fn routing(&self, s: usize) -> Vec<usize> {
        (0..s)
            .map(|stream_idx| {
                let row = &self.p_logits[stream_idx.min(MAX_OUT - 1)];
                let mut best = 0;
                let mut best_val = f32::NEG_INFINITY;
                for (idx, value) in row.iter().enumerate().take(s) {
                    if *value > best_val {
                        best_val = *value;
                        best = idx;
                    }
                }
                best
            })
            .collect()
    }

    /// Parses one i64 from 4 limb tokens under the learned assignment.
    pub fn assemble(&self, limbs: &[u16], assignment: &[usize; LIMBS_PER_VALUE]) -> i64 {
        let mut unsigned = 0_u64;
        for (pos, exp_slot) in assignment.iter().enumerate() {
            unsigned |= u64::from(limbs[pos]) << (16 * exp_slot);
        }
        unsigned as i64
    }

    /// Emits the limb tokens of one value under the learned assignment.
    pub fn emit(&self, value: i64, assignment: &[usize; LIMBS_PER_VALUE]) -> [u16; LIMBS_PER_VALUE] {
        let canonical = limbs_of(value);
        let mut out = [0_u16; LIMBS_PER_VALUE];
        for (pos, exp_slot) in assignment.iter().enumerate() {
            out[pos] = canonical[*exp_slot];
        }
        out
    }
}

impl Default for InterfaceModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Greedy injective decode: repeatedly take the highest-confidence
/// unassigned (position, exponent) pair.
fn injective_assignment(logits: &[Vec<f32>]) -> [usize; LIMBS_PER_VALUE] {
    let mut out = [usize::MAX; LIMBS_PER_VALUE];
    let mut used = [false; LIMBS_PER_VALUE];
    for _ in 0..LIMBS_PER_VALUE {
        let mut best: Option<(usize, usize, f32)> = None;
        for (pos, row) in logits.iter().enumerate().take(LIMBS_PER_VALUE) {
            if out[pos] != usize::MAX {
                continue;
            }
            for (exp_slot, value) in row.iter().enumerate().take(LIMBS_PER_VALUE) {
                if used[exp_slot] {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some((_, _, best_val)) => *value > best_val,
                };
                if better {
                    best = Some((pos, exp_slot, *value));
                }
            }
        }
        if let Some((pos, exp_slot, _)) = best {
            out[pos] = exp_slot;
            used[exp_slot] = true;
        }
    }
    for (pos, slot) in out.iter_mut().enumerate() {
        if *slot == usize::MAX {
            *slot = pos;
        }
    }
    out
}

/// Interface training statistics.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct InterfaceTrainStats {
    /// Limb-assignment training instances seen.
    pub assembly_instances: u64,
    /// Routing training instances seen.
    pub routing_instances: u64,
    /// Verified corpus (output) tokens consumed.
    pub corpus_tokens_seen: u64,
    /// Final mean assembly CE.
    pub assembly_ce: f64,
    /// Final mean routing CE.
    pub routing_ce: f64,
}

fn softmax_grad_step(logits: &mut [f32], matches: &[bool], lr: f32) -> f64 {
    // CE against the normalized indicator of matching slots.
    let count = matches.iter().filter(|m| **m).count();
    if count == 0 {
        return 0.0;
    }
    let mut max = f32::NEG_INFINITY;
    for value in logits.iter() {
        if *value > max {
            max = *value;
        }
    }
    let mut probs: Vec<f32> = logits.iter().map(|v| (*v - max).exp()).collect();
    let total: f32 = probs.iter().sum();
    for value in &mut probs {
        *value /= total;
    }
    let target_weight = 1.0 / count as f32;
    let mut loss = 0.0_f64;
    for (idx, is_match) in matches.iter().enumerate() {
        let target = if *is_match { target_weight } else { 0.0 };
        if *is_match {
            loss -= f64::from(target_weight) * f64::from(probs[idx].max(1e-30).ln());
        }
        logits[idx] -= lr * (probs[idx] - target);
    }
    loss
}

/// Trains the interface over the train-split records of the prep file
/// (single pass, same snapshot as the other baselines).
pub fn train_interface(
    records: &[StudentRecord],
    lr: f32,
) -> (InterfaceModel, InterfaceTrainStats) {
    let mut model = InterfaceModel::new();
    let mut stats = InterfaceTrainStats::default();
    let mut assembly_loss = 0.0_f64;
    let mut routing_loss = 0.0_f64;
    for record in records {
        if record.split != crate::prep::Split::Train {
            continue;
        }
        for step in 0..record.step_count {
            // Input + output limb assembly supervision.
            for field in 0..record.f {
                let value = record.inputs[step * record.f + field];
                let canonical = limbs_of(value);
                for pos in 0..LIMBS_PER_VALUE {
                    let observed = canonical[pos]; // stream is LE order
                    let matches: Vec<bool> =
                        (0..LIMBS_PER_VALUE).map(|e| canonical[e] == observed).collect();
                    assembly_loss +=
                        softmax_grad_step(&mut model.a_logits[pos], &matches, lr);
                    stats.assembly_instances += 1;
                }
            }
            for out in 0..record.s {
                let value = record.outputs[step * record.s + out];
                let canonical = limbs_of(value);
                for pos in 0..LIMBS_PER_VALUE {
                    let observed = canonical[pos];
                    let matches: Vec<bool> =
                        (0..LIMBS_PER_VALUE).map(|e| canonical[e] == observed).collect();
                    assembly_loss +=
                        softmax_grad_step(&mut model.b_logits[pos], &matches, lr);
                    stats.assembly_instances += 1;
                }
                stats.corpus_tokens_seen += LIMBS_PER_VALUE as u64;
            }
            // Routing supervision: stream index -> executor row index.
            let row = &record.outputs[step * record.s..(step + 1) * record.s];
            for (stream_idx, stream_value) in row.iter().enumerate() {
                let matches: Vec<bool> = (0..MAX_OUT)
                    .map(|j| j < record.s && row[j] == *stream_value)
                    .collect();
                routing_loss += softmax_grad_step(
                    &mut model.p_logits[stream_idx.min(MAX_OUT - 1)],
                    &matches,
                    lr,
                );
                stats.routing_instances += 1;
            }
        }
    }
    if stats.assembly_instances > 0 {
        stats.assembly_ce = assembly_loss / stats.assembly_instances as f64;
    }
    if stats.routing_instances > 0 {
        stats.routing_ce = routing_loss / stats.routing_instances as f64;
    }
    (model, stats)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::prep::{Family, Split};

    #[test]
    fn interface_learns_le_assembly_and_identity_routing() {
        let record = StudentRecord {
            f: 2,
            family: Family::ArithmeticCarry,
            final_output_digest: String::new(),
            full_trace_digest: String::new(),
            inputs: vec![65_539, 0x0001_0002_0003_0004, 123_456_789, 42],
            model_json: None,
            outputs: vec![1, 65_540, 0x0002_0003_0004_0005, 2, 70_000, 3],
            program_hash: String::new(),
            record_id: String::from("trace_iface"),
            s: 3,
            seed_writes: vec![],
            split: Split::Train,
            step_count: 2,
        };
        let records = vec![record; 50];
        let (model, stats) = train_interface(&records, 0.5);
        assert!(stats.assembly_instances > 0);
        // Distinct limb values force the LE assignment.
        assert_eq!(model.input_assignment(), [0, 1, 2, 3]);
        assert_eq!(model.output_assignment(), [0, 1, 2, 3]);
        assert_eq!(model.routing(3), vec![0, 1, 2]);
        let assembled = model.assemble(&[3, 1, 0, 0], &model.input_assignment());
        assert_eq!(assembled, 65_539);
    }
}
