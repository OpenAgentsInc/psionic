//! Training loop for the W3 student baselines (openagents#4749).
//!
//! Baselines share one backbone and differ only in configured losses:
//!   (a) next-token byte CE on verified trace tokens;
//!   (b) (a) + auxiliary predict-ahead state heads (keyed-read value,
//!       accumulator state, gated branch outputs) + output-digest-prefix
//!       head;
//!   (c) (a) + the parabolic lookup module with analytic init and the
//!       max-margin lookup loss from the plan:
//!       `max(0, margin - score(correct) + max(score(incorrect)))`.
//!
//! The trainer streams the prep file in a deterministic shuffled record
//! order for a configured number of passes (W3 default: one pass over
//! the full train split, recorded honestly in the receipt).

use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model::{
    AUX_BYTES, AUX_VALUES, Acts, Backbone, DIGEST_BYTES, ModelConfig, TokenFeatures,
};
use crate::prep::{
    LIMBS_PER_VALUE, LookupInstance, PrepFile, StudentRecord, TokenRole, build_sequence,
    family_spec, hex_bytes, lookup_instances,
};
use crate::tensor::{SplitMix, adamw, gemm, gemm_at, gemm_bt, grad_norm, softmax_ce_row};

/// Baseline identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Baseline {
    /// (a) next-token distillation only.
    A,
    /// (b) + auxiliary state losses.
    B,
    /// (c) 2D-head / hard-max-regularized lookup variant.
    C,
    /// (c) with random (non-analytic) lookup init — the H3 control.
    CRandom,
}

impl Baseline {
    /// Stable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::A => "baseline_a_next_token",
            Self::B => "baseline_b_aux_state",
            Self::C => "baseline_c_lookup_analytic",
            Self::CRandom => "baseline_c_lookup_random_init",
        }
    }
}

/// Full training configuration, hashed into the receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Baseline.
    pub baseline: Baseline,
    /// Backbone shape.
    pub model: ModelConfig,
    /// Chunks per optimizer step.
    pub batch_size: usize,
    /// Passes over the train split.
    pub epochs: usize,
    /// Peak learning rate.
    pub lr_max: f32,
    /// Final learning rate.
    pub lr_min: f32,
    /// Warmup steps.
    pub warmup_steps: usize,
    /// AdamW beta1.
    pub beta1: f32,
    /// AdamW beta2.
    pub beta2: f32,
    /// AdamW weight decay (excluded for lookup scalars).
    pub weight_decay: f32,
    /// Global gradient-clip norm.
    pub grad_clip: f32,
    /// Aux-loss weight (baseline b).
    pub aux_weight: f32,
    /// Digest-prefix loss weight (baseline b).
    pub digest_weight: f32,
    /// Lookup margin (baseline c).
    pub lookup_margin: f32,
    /// Lookup loss weight (baseline c).
    pub lookup_weight: f32,
    /// Init / shuffle seed.
    pub seed: u64,
    /// Optional cap on optimizer steps (0 = full schedule). Used only
    /// for smoke runs; production receipts must show 0.
    pub max_steps: usize,
}

impl TrainConfig {
    /// W3 defaults for one baseline.
    pub fn w3_default(baseline: Baseline) -> Self {
        Self {
            aux_weight: if baseline == Baseline::B { 0.25 } else { 0.0 },
            baseline,
            batch_size: 8,
            beta1: 0.9,
            beta2: 0.95,
            digest_weight: if baseline == Baseline::B { 0.1 } else { 0.0 },
            epochs: 1,
            grad_clip: 1.0,
            lookup_margin: 1.0,
            lookup_weight: matches!(baseline, Baseline::C | Baseline::CRandom)
                .then_some(1.0)
                .unwrap_or(0.0),
            lr_max: 3e-3,
            lr_min: 3e-4,
            max_steps: 0,
            model: ModelConfig {
                context: 512,
                d_ff: 256,
                d_model: 64,
                n_heads: 4,
                n_layers: 2,
                use_aux: baseline == Baseline::B,
                use_lookup: matches!(baseline, Baseline::C | Baseline::CRandom),
            },
            seed: 0x4749,
            warmup_steps: 200,
            weight_decay: 0.01,
        }
    }

    /// Stable digest over the canonical JSON encoding.
    pub fn stable_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"tassadar_student_train_config|");
        hasher.update(serde_json::to_vec(self).unwrap_or_default());
        hex::encode(hasher.finalize())
    }
}

/// One chunk of one record, ready for batching.
pub struct Chunk {
    /// Token features (length = config.context, padded at the tail).
    pub features: Vec<TokenFeatures>,
    /// Next-token targets per local position (u16), masked positions only.
    pub targets: Vec<(usize, u16)>,
    /// Aux items: (local pos, present mask, target bytes).
    pub aux: Vec<(usize, u8, [u8; AUX_VALUES * AUX_BYTES])>,
    /// Digest items: (local pos, prefix bytes).
    pub digest: Vec<(usize, [u8; DIGEST_BYTES])>,
    /// Lookup items: (local query pos, query, candidate keys, correct).
    pub lookups: Vec<(usize, f64, Vec<f64>, usize)>,
    /// Real (unpadded) token count.
    pub real_len: usize,
}

const ROLE_IDX: [u8; 3] = [0, 1, 2];

fn role_index(role: TokenRole) -> u8 {
    match role {
        TokenRole::Seed => ROLE_IDX[0],
        TokenRole::Input => ROLE_IDX[1],
        TokenRole::Output => ROLE_IDX[2],
    }
}

/// Cuts one record's sequence into training chunks.
pub fn chunks_of_record(record: &StudentRecord, cfg: &TrainConfig) -> Vec<Chunk> {
    let seq = build_sequence(record);
    let spec = family_spec(record.family);
    let t = cfg.model.context;
    let total = seq.tokens.len();
    let lookups: Vec<LookupInstance> = if cfg.model.use_lookup {
        lookup_instances(record, &seq)
    } else {
        Vec::new()
    };
    let digest_prefix: Option<[u8; DIGEST_BYTES]> = hex_bytes(&record.final_output_digest)
        .and_then(|bytes| {
            bytes.get(..DIGEST_BYTES).map(|head| {
                let mut prefix = [0_u8; DIGEST_BYTES];
                prefix.copy_from_slice(head);
                prefix
            })
        });
    let mut chunks = Vec::with_capacity(total.div_ceil(t));
    let mut start = 0_usize;
    while start < total {
        let end = (start + t).min(total);
        let real_len = end - start;
        let mut features = Vec::with_capacity(t);
        for pos in start..end {
            features.push(TokenFeatures {
                feats: seq.feats[pos],
                limb: seq.limb[pos],
                role: role_index(seq.roles[pos]),
                token: seq.tokens[pos],
                vidx: seq.value_idx[pos],
            });
        }
        while features.len() < t {
            features.push(TokenFeatures {
                feats: [0.0, 0.0],
                limb: 0,
                role: 0,
                token: 0,
                vidx: 0,
            });
        }
        let mut targets = Vec::new();
        for pos in start..end {
            let next = pos + 1;
            if next < total && seq.roles[next] == TokenRole::Output {
                targets.push((pos - start, seq.tokens[next]));
            }
        }
        let mut aux = Vec::new();
        if cfg.model.use_aux {
            for (step, first_out) in seq.first_output_pos.iter().enumerate() {
                let aux_pos = first_out.wrapping_sub(1);
                if aux_pos < start || aux_pos >= end {
                    continue;
                }
                let mut bytes = [0_u8; AUX_VALUES * AUX_BYTES];
                let mut mask = 0_u8;
                for (slot, aux_idx) in spec.aux_out_idxs.iter().enumerate().take(AUX_VALUES) {
                    let value = record.outputs[step * record.s + aux_idx];
                    bytes[slot * AUX_BYTES..(slot + 1) * AUX_BYTES]
                        .copy_from_slice(&value.to_le_bytes());
                    mask |= 1 << slot;
                }
                if mask != 0 {
                    aux.push((aux_pos - start, mask, bytes));
                }
            }
        }
        let mut digest = Vec::new();
        if cfg.model.use_aux {
            if let Some(prefix) = digest_prefix {
                if end == total && total >= 1 && total > start {
                    digest.push((total - 1 - start, prefix));
                }
            }
        }
        let mut chunk_lookups = Vec::new();
        if cfg.model.use_lookup {
            for instance in &lookups {
                // Query row: last limb of the query input value.
                let spec_read = spec.read;
                let Some(read) = spec_read else { continue };
                let qpos =
                    seq.step_starts[instance.step] + (read.query_input + 1) * LIMBS_PER_VALUE - 1;
                if qpos < start || qpos >= end {
                    continue;
                }
                chunk_lookups.push((
                    qpos - start,
                    instance.query as f64,
                    instance.candidate_keys.iter().map(|k| *k as f64).collect(),
                    instance.correct,
                ));
            }
        }
        chunks.push(Chunk {
            aux,
            digest,
            features,
            lookups: chunk_lookups,
            real_len,
            targets,
        });
        start = end;
    }
    chunks
}

/// Loss summary of one optimizer step.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct StepLosses {
    /// Next-token byte CE (mean per byte head).
    pub ce: f64,
    /// Aux CE.
    pub aux: f64,
    /// Digest CE.
    pub digest: f64,
    /// Lookup margin loss.
    pub lookup: f64,
    /// Lookup top-1 accuracy in batch.
    pub lookup_acc: f64,
    /// Lookup instances in batch.
    pub lookup_count: usize,
    /// Targets in batch.
    pub targets: usize,
}

#[allow(clippy::too_many_lines)]
fn head_losses_and_backward(
    model: &mut Backbone,
    acts: &Acts,
    batch: &[Chunk],
    cfg: &TrainConfig,
    d_hidden: &mut [f32],
) -> StepLosses {
    let d = model.cfg.d_model;
    let t = model.cfg.context;
    let mut losses = StepLosses::default();
    // ---- next-token byte heads ----
    let mut rows: Vec<usize> = Vec::new();
    let mut targets: Vec<u16> = Vec::new();
    for (chunk_idx, chunk) in batch.iter().enumerate() {
        for (local, target) in &chunk.targets {
            rows.push(chunk_idx * t + local);
            targets.push(*target);
        }
    }
    let p = rows.len();
    losses.targets = p;
    if p > 0 {
        let mut h_sel = vec![0.0_f32; p * d];
        for (sel, row) in rows.iter().enumerate() {
            h_sel[sel * d..(sel + 1) * d].copy_from_slice(&acts.hidden[row * d..(row + 1) * d]);
        }
        let mut logits_lo = vec![0.0_f32; p * 256];
        let mut logits_hi = vec![0.0_f32; p * 256];
        gemm(p, d, 256, &h_sel, &model.head_lo.w, &mut logits_lo);
        gemm(p, d, 256, &h_sel, &model.head_hi.w, &mut logits_hi);
        let inv_p = 1.0 / p as f32;
        let mut ce_total = 0.0_f64;
        for sel in 0..p {
            let lo = (targets[sel] & 0xff) as usize;
            let hi = (targets[sel] >> 8) as usize;
            let row_lo = &mut logits_lo[sel * 256..(sel + 1) * 256];
            for (logit, bias) in row_lo.iter_mut().zip(model.head_lo_b.w.iter()) {
                *logit += *bias;
            }
            ce_total += f64::from(softmax_ce_row(row_lo, lo));
            row_lo[lo] -= 1.0;
            for value in row_lo.iter_mut() {
                *value *= 0.5 * inv_p;
            }
            let row_hi = &mut logits_hi[sel * 256..(sel + 1) * 256];
            for (logit, bias) in row_hi.iter_mut().zip(model.head_hi_b.w.iter()) {
                *logit += *bias;
            }
            ce_total += f64::from(softmax_ce_row(row_hi, hi));
            row_hi[hi] -= 1.0;
            for value in row_hi.iter_mut() {
                *value *= 0.5 * inv_p;
            }
        }
        losses.ce = ce_total / (2.0 * p as f64);
        let mut d_h_sel = vec![0.0_f32; p * d];
        gemm_bt(p, d, 256, &logits_lo, &model.head_lo.w, &mut d_h_sel);
        gemm_bt(p, d, 256, &logits_hi, &model.head_hi.w, &mut d_h_sel);
        gemm_at(p, d, 256, &h_sel, &logits_lo, &mut model.head_lo.g);
        gemm_at(p, d, 256, &h_sel, &logits_hi, &mut model.head_hi.g);
        for row in logits_lo.chunks(256) {
            for (b, value) in model.head_lo_b.g.iter_mut().zip(row.iter()) {
                *b += *value;
            }
        }
        for row in logits_hi.chunks(256) {
            for (b, value) in model.head_hi_b.g.iter_mut().zip(row.iter()) {
                *b += *value;
            }
        }
        for (sel, row) in rows.iter().enumerate() {
            let dst = &mut d_hidden[row * d..(row + 1) * d];
            let src = &d_h_sel[sel * d..(sel + 1) * d];
            for (d_val, s_val) in dst.iter_mut().zip(src.iter()) {
                *d_val += *s_val;
            }
        }
    }
    // ---- aux heads (baseline b) ----
    if cfg.model.use_aux {
        let mut aux_rows: Vec<usize> = Vec::new();
        let mut aux_items: Vec<(u8, [u8; AUX_VALUES * AUX_BYTES])> = Vec::new();
        for (chunk_idx, chunk) in batch.iter().enumerate() {
            for (local, mask, bytes) in &chunk.aux {
                aux_rows.push(chunk_idx * t + local);
                aux_items.push((*mask, *bytes));
            }
        }
        let a = aux_rows.len();
        if a > 0 {
            let width = AUX_VALUES * AUX_BYTES * 256;
            let mut h_sel = vec![0.0_f32; a * d];
            for (sel, row) in aux_rows.iter().enumerate() {
                h_sel[sel * d..(sel + 1) * d].copy_from_slice(&acts.hidden[row * d..(row + 1) * d]);
            }
            let mut logits = vec![0.0_f32; a * width];
            gemm(a, d, width, &h_sel, &model.w_aux.w, &mut logits);
            let mut total = 0.0_f64;
            let mut count = 0_usize;
            for (sel, (mask, bytes)) in aux_items.iter().enumerate() {
                let row = &mut logits[sel * width..(sel + 1) * width];
                for (col, bias) in row.iter_mut().zip(model.b_aux.w.iter()) {
                    *col += *bias;
                }
                for value_slot in 0..AUX_VALUES {
                    if mask & (1 << value_slot) == 0 {
                        // Zero gradient for absent targets.
                        for byte_slot in 0..AUX_BYTES {
                            let off = (value_slot * AUX_BYTES + byte_slot) * 256;
                            for logit in &mut row[off..off + 256] {
                                *logit = 0.0;
                            }
                        }
                        continue;
                    }
                    for byte_slot in 0..AUX_BYTES {
                        let off = (value_slot * AUX_BYTES + byte_slot) * 256;
                        let target = bytes[value_slot * AUX_BYTES + byte_slot] as usize;
                        total += f64::from(softmax_ce_row(&mut row[off..off + 256], target));
                        row[off + target] -= 1.0;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                losses.aux = total / count as f64;
                let scale = cfg.aux_weight / count as f32;
                for value in &mut logits {
                    *value *= scale;
                }
                let mut d_h_sel = vec![0.0_f32; a * d];
                gemm_bt(a, d, width, &logits, &model.w_aux.w, &mut d_h_sel);
                gemm_at(a, d, width, &h_sel, &logits, &mut model.w_aux.g);
                for row in logits.chunks(width) {
                    for (b, value) in model.b_aux.g.iter_mut().zip(row.iter()) {
                        *b += *value;
                    }
                }
                for (sel, row) in aux_rows.iter().enumerate() {
                    let dst = &mut d_hidden[row * d..(row + 1) * d];
                    let src = &d_h_sel[sel * d..(sel + 1) * d];
                    for (d_val, s_val) in dst.iter_mut().zip(src.iter()) {
                        *d_val += *s_val;
                    }
                }
            }
        }
        // ---- digest-prefix head ----
        let mut dig_rows: Vec<usize> = Vec::new();
        let mut dig_targets: Vec<[u8; DIGEST_BYTES]> = Vec::new();
        for (chunk_idx, chunk) in batch.iter().enumerate() {
            for (local, prefix) in &chunk.digest {
                dig_rows.push(chunk_idx * t + local);
                dig_targets.push(*prefix);
            }
        }
        let g = dig_rows.len();
        if g > 0 {
            let width = DIGEST_BYTES * 256;
            let mut h_sel = vec![0.0_f32; g * d];
            for (sel, row) in dig_rows.iter().enumerate() {
                h_sel[sel * d..(sel + 1) * d].copy_from_slice(&acts.hidden[row * d..(row + 1) * d]);
            }
            let mut logits = vec![0.0_f32; g * width];
            gemm(g, d, width, &h_sel, &model.w_dig.w, &mut logits);
            let mut total = 0.0_f64;
            let mut count = 0_usize;
            for (sel, prefix) in dig_targets.iter().enumerate() {
                let row = &mut logits[sel * width..(sel + 1) * width];
                for (col, bias) in row.iter_mut().zip(model.b_dig.w.iter()) {
                    *col += *bias;
                }
                for byte_slot in 0..DIGEST_BYTES {
                    let off = byte_slot * 256;
                    let target = prefix[byte_slot] as usize;
                    total += f64::from(softmax_ce_row(&mut row[off..off + 256], target));
                    row[off + target] -= 1.0;
                    count += 1;
                }
            }
            losses.digest = total / count as f64;
            let scale = cfg.digest_weight / count as f32;
            for value in &mut logits {
                *value *= scale;
            }
            let mut d_h_sel = vec![0.0_f32; g * d];
            gemm_bt(g, d, width, &logits, &model.w_dig.w, &mut d_h_sel);
            gemm_at(g, d, width, &h_sel, &logits, &mut model.w_dig.g);
            for row in logits.chunks(width) {
                for (b, value) in model.b_dig.g.iter_mut().zip(row.iter()) {
                    *b += *value;
                }
            }
            for (sel, row) in dig_rows.iter().enumerate() {
                let dst = &mut d_hidden[row * d..(row + 1) * d];
                let src = &d_h_sel[sel * d..(sel + 1) * d];
                for (d_val, s_val) in dst.iter_mut().zip(src.iter()) {
                    *d_val += *s_val;
                }
            }
        }
    }
    // ---- lookup module (baseline c): max-margin over parabolic scores ----
    if cfg.model.use_lookup {
        let beta_q = f64::from(model.lk_scalars.w[0]);
        let b_q = f64::from(model.lk_scalars.w[1]);
        let beta_k = f64::from(model.lk_scalars.w[2]);
        let b_k = f64::from(model.lk_scalars.w[3]);
        let score_scale = f64::from(model.lk_scalars.w[4]);
        let mut total = 0.0_f64;
        let mut hits = 0_usize;
        let mut count = 0_usize;
        for (chunk_idx, chunk) in batch.iter().enumerate() {
            for (local, query, cand_keys, correct) in &chunk.lookups {
                if *correct >= cand_keys.len() {
                    continue;
                }
                let row = chunk_idx * t + local;
                let h_row = &acts.hidden[row * d..(row + 1) * d];
                let mut wq_dot = 0.0_f64;
                for (h_val, w_val) in h_row.iter().zip(model.lk_wq.w.iter()) {
                    wq_dot += f64::from(*h_val) * f64::from(*w_val);
                }
                let q_prime = beta_q * query + b_q + wq_dot;
                let score_of = |key: f64| -> f64 {
                    let k_prime = beta_k * key + b_k;
                    let delta = k_prime - q_prime;
                    -score_scale * delta * delta
                };
                let s_correct = score_of(cand_keys[*correct]);
                let mut best_wrong = f64::NEG_INFINITY;
                let mut best_wrong_idx = usize::MAX;
                let mut best_any = f64::NEG_INFINITY;
                let mut best_any_idx = 0_usize;
                for (idx, key) in cand_keys.iter().enumerate() {
                    let score = score_of(*key);
                    if score >= best_any {
                        best_any = score;
                        best_any_idx = idx;
                    }
                    if idx != *correct
                        && cand_keys[idx] != cand_keys[*correct]
                        && score > best_wrong
                    {
                        best_wrong = score;
                        best_wrong_idx = idx;
                    }
                }
                count += 1;
                if cand_keys[best_any_idx] == cand_keys[*correct] {
                    hits += 1;
                }
                if best_wrong_idx == usize::MAX {
                    continue;
                }
                let violation = f64::from(cfg.lookup_margin) - s_correct + best_wrong;
                if violation > 0.0 {
                    total += violation;
                    // dV/ds_correct = -1, dV/ds_wrong = +1, with
                    // s_j = -ss * delta_j^2, delta_j = beta_k k_j + b_k - q',
                    // q' = beta_q q + b_q + w_q . h.
                    let h_row_vec: Vec<f32> = h_row.to_vec();
                    let mut dh_coeff = 0.0_f64;
                    for (key, sign) in [
                        (cand_keys[*correct], -1.0_f64),
                        (cand_keys[best_wrong_idx], 1.0),
                    ] {
                        let k_prime = beta_k * key + b_k;
                        let delta = k_prime - q_prime;
                        let ds_dss = -(delta * delta);
                        let ds_ddelta = -2.0 * score_scale * delta;
                        let w = f64::from(cfg.lookup_weight) * sign;
                        model.lk_scalars.g[4] += (w * ds_dss) as f32;
                        model.lk_scalars.g[2] += (w * ds_ddelta * key) as f32;
                        model.lk_scalars.g[3] += (w * ds_ddelta) as f32;
                        model.lk_scalars.g[0] += (w * ds_ddelta * -query) as f32;
                        model.lk_scalars.g[1] += -(w * ds_ddelta) as f32;
                        for (col, h_val) in h_row_vec.iter().enumerate() {
                            model.lk_wq.g[col] += (w * ds_ddelta * -f64::from(*h_val)) as f32;
                        }
                        dh_coeff += -(w * ds_ddelta);
                    }
                    let dst = &mut d_hidden[row * d..(row + 1) * d];
                    for (col, d_val) in dst.iter_mut().enumerate() {
                        *d_val += (dh_coeff * f64::from(model.lk_wq.w[col])) as f32;
                    }
                }
            }
        }
        if count > 0 {
            losses.lookup = total / count as f64;
            losses.lookup_acc = hits as f64 / count as f64;
            losses.lookup_count = count;
        }
    }
    losses
}

/// Training progress receipt, written next to the checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainReceipt {
    /// Receipt schema.
    pub receipt_version: String,
    /// Baseline label.
    pub baseline: String,
    /// Full config.
    pub config: TrainConfig,
    /// Config digest.
    pub config_digest: String,
    /// Corpus id of the training prep file.
    pub corpus_id: String,
    /// Snapshot digest of the corpus.
    pub dataset_snapshot_digest: String,
    /// sha256 of the train prep file bytes.
    pub train_prep_sha256: String,
    /// sha256 of the final weights file.
    pub weights_sha256: String,
    /// Total optimizer steps run.
    pub steps: u64,
    /// Sequence positions consumed (inputs + outputs + seeds).
    pub positions_seen: u64,
    /// Verified corpus (output) tokens consumed.
    pub corpus_tokens_seen: u64,
    /// Wall-clock seconds.
    pub wall_seconds: f64,
    /// Final loss summary (averaged over the last 200 steps).
    pub final_losses: StepLosses,
    /// Host description.
    pub host: String,
    /// Trainer threads.
    pub threads: usize,
}

/// Trains one baseline over the prep file; writes weights + receipt.
#[allow(clippy::too_many_lines)]
pub fn train(
    prep: &PrepFile,
    train_prep_sha256: &str,
    cfg: &TrainConfig,
    out_dir: &Path,
    host: &str,
) -> Result<TrainReceipt, String> {
    let mut model = Backbone::init(&cfg.model, cfg.seed);
    if cfg.baseline == Baseline::CRandom {
        // H3 control: random lookup init instead of analytic.
        let mut rng = SplitMix::new(cfg.seed ^ 0xc0ffee);
        for value in &mut model.lk_scalars.w {
            *value = rng.normal() * 0.1;
        }
    }
    let stderr = std::io::stderr();
    let mut log = stderr.lock();
    let mut order: Vec<usize> = (0..prep.records.len())
        .filter(|index| prep.records[*index].split == crate::prep::Split::Train)
        .collect();
    let mut rng = SplitMix::new(cfg.seed.wrapping_add(99));
    // Pre-compute total schedule length.
    let t = cfg.model.context;
    let mut total_chunks = 0_usize;
    for index in &order {
        let record = &prep.records[*index];
        let seq_len = (record.seed_writes.len() * 2 + (record.f + record.s) * record.step_count)
            * LIMBS_PER_VALUE;
        total_chunks += seq_len.div_ceil(t);
    }
    let total_steps = ((total_chunks * cfg.epochs) / cfg.batch_size).max(1) as u64;
    let started = Instant::now();
    let mut step: u64 = 0;
    let mut positions_seen: u64 = 0;
    let mut corpus_tokens_seen: u64 = 0;
    let mut recent: Vec<StepLosses> = Vec::new();
    let mut pending: Vec<Chunk> = Vec::new();
    let _ = writeln!(
        log,
        "train {} records={} chunks={} planned_steps={}",
        cfg.baseline.label(),
        order.len(),
        total_chunks,
        total_steps
    );
    'epochs: for _epoch in 0..cfg.epochs {
        rng.shuffle(&mut order);
        for record_index in &order {
            let record = &prep.records[*record_index];
            pending.extend(chunks_of_record(record, cfg));
            while pending.len() >= cfg.batch_size {
                let batch: Vec<Chunk> = pending.drain(..cfg.batch_size).collect();
                let losses = train_step(&mut model, &batch, cfg, step, total_steps);
                for chunk in &batch {
                    positions_seen += chunk.real_len as u64;
                    corpus_tokens_seen += chunk.targets.len() as u64;
                }
                step += 1;
                recent.push(losses);
                if recent.len() > 200 {
                    recent.remove(0);
                }
                if step.is_multiple_of(200) {
                    let avg = average(&recent);
                    let elapsed = started.elapsed().as_secs_f64();
                    let _ = writeln!(
                        log,
                        "step {step}/{total_steps} ce={:.4} aux={:.4} lookup={:.5} lk_acc={:.4} pos/s={:.0} elapsed={:.0}s",
                        avg.ce,
                        avg.aux,
                        avg.lookup,
                        avg.lookup_acc,
                        positions_seen as f64 / elapsed,
                        elapsed
                    );
                }
                if cfg.max_steps > 0 && step >= cfg.max_steps as u64 {
                    break 'epochs;
                }
            }
        }
        // Final partial batch of the epoch.
        if !pending.is_empty() && (cfg.max_steps == 0 || step < cfg.max_steps as u64) {
            let batch: Vec<Chunk> = std::mem::take(&mut pending);
            let losses = train_step(&mut model, &batch, cfg, step, total_steps);
            for chunk in &batch {
                positions_seen += chunk.real_len as u64;
                corpus_tokens_seen += chunk.targets.len() as u64;
            }
            step += 1;
            recent.push(losses);
        }
    }
    let wall_seconds = started.elapsed().as_secs_f64();
    std::fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let weights = model.weights_bytes();
    let mut hasher = Sha256::new();
    hasher.update(&weights);
    let weights_sha256 = hex::encode(hasher.finalize());
    std::fs::write(out_dir.join("weights.bin"), &weights).map_err(|error| error.to_string())?;
    let receipt = TrainReceipt {
        baseline: cfg.baseline.label().to_string(),
        config: cfg.clone(),
        config_digest: cfg.stable_digest(),
        corpus_id: prep.corpus_id.clone(),
        corpus_tokens_seen,
        dataset_snapshot_digest: prep.snapshot_digest.clone(),
        final_losses: average(&recent),
        host: host.to_string(),
        positions_seen,
        receipt_version: String::from("tassadar_student_train_receipt.v0.1"),
        steps: step,
        threads: rayon::current_num_threads(),
        train_prep_sha256: train_prep_sha256.to_string(),
        wall_seconds,
        weights_sha256,
    };
    let receipt_json = serde_json::to_string_pretty(&receipt).map_err(|error| error.to_string())?;
    std::fs::write(out_dir.join("receipt.json"), format!("{receipt_json}\n"))
        .map_err(|error| error.to_string())?;
    Ok(receipt)
}

fn average(losses: &[StepLosses]) -> StepLosses {
    if losses.is_empty() {
        return StepLosses::default();
    }
    let n = losses.len() as f64;
    // Lookup stats are weighted by instance count: most batches carry
    // no lookup instances (only keyed-read families produce them) and
    // must not dilute the average.
    let lookup_total: usize = losses.iter().map(|l| l.lookup_count).sum();
    let weighted = |of: fn(&StepLosses) -> f64| -> f64 {
        if lookup_total == 0 {
            return 0.0;
        }
        losses
            .iter()
            .map(|l| of(l) * l.lookup_count as f64)
            .sum::<f64>()
            / lookup_total as f64
    };
    StepLosses {
        aux: losses.iter().map(|l| l.aux).sum::<f64>() / n,
        ce: losses.iter().map(|l| l.ce).sum::<f64>() / n,
        digest: losses.iter().map(|l| l.digest).sum::<f64>() / n,
        lookup: weighted(|l| l.lookup),
        lookup_acc: weighted(|l| l.lookup_acc),
        lookup_count: lookup_total,
        targets: losses.iter().map(|l| l.targets).sum(),
    }
}

fn lr_at(cfg: &TrainConfig, step: u64, total_steps: u64) -> f32 {
    let warm = cfg.warmup_steps as u64;
    if step < warm {
        return cfg.lr_max * (step + 1) as f32 / warm as f32;
    }
    let progress =
        ((step - warm) as f32 / (total_steps.saturating_sub(warm)).max(1) as f32).min(1.0);
    cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
}

fn train_step(
    model: &mut Backbone,
    batch: &[Chunk],
    cfg: &TrainConfig,
    step: u64,
    total_steps: u64,
) -> StepLosses {
    let t = cfg.model.context;
    let b = batch.len();
    let mut features: Vec<TokenFeatures> = Vec::with_capacity(b * t);
    for chunk in batch {
        features.extend_from_slice(&chunk.features);
    }
    let acts = model.forward_train(&features, b, t);
    let mut d_hidden = vec![0.0_f32; b * t * cfg.model.d_model];
    let losses = head_losses_and_backward(model, &acts, batch, cfg, &mut d_hidden);
    model.backward(&features, &acts, &d_hidden);
    // Clip + AdamW.
    let lookup_param_count = 2;
    let norm = {
        let params = model.params();
        grad_norm(&params[..params.len() - lookup_param_count])
    };
    let grad_scale = if norm > cfg.grad_clip {
        cfg.grad_clip / norm
    } else {
        1.0
    };
    let lr = lr_at(cfg, step, total_steps);
    let param_total = model.params().len();
    for (index, param) in model.params_mut().into_iter().enumerate() {
        let is_lookup = index >= param_total - lookup_param_count;
        // Lookup scalars: no weight decay (analytic-geometry maintenance
        // is part of the H3 experiment) and their own gradient clip.
        let (wd, scale) = if is_lookup {
            let lk_norm: f32 = param.g.iter().map(|g| g * g).sum::<f32>().sqrt();
            let lk_scale = if lk_norm > 1.0 { 1.0 / lk_norm } else { 1.0 };
            (0.0, lk_scale)
        } else {
            (cfg.weight_decay, grad_scale)
        };
        adamw(param, lr, cfg.beta1, cfg.beta2, 1e-8, wd, step + 1, scale);
    }
    losses
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::prep::{Family, Split, StudentRecord};

    fn synthetic_record(step_count: usize) -> StudentRecord {
        let f = 3;
        let s = 5;
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut keys = vec![17_i64, 34];
        let mut values = vec![100_i64, 200];
        let mut rng = SplitMix::new(5);
        for step in 0..step_count {
            let write_key = 1000 + step as i64;
            let write_value = (rng.next_u64() % 1000) as i64;
            let read_idx = (rng.next_u64() % keys.len() as u64) as usize;
            let read_key = keys[read_idx];
            let read_value = values[read_idx];
            inputs.extend_from_slice(&[write_key, write_value, read_key]);
            outputs.extend_from_slice(&[
                write_key,
                write_value,
                read_key,
                read_value,
                read_value + write_value,
            ]);
            keys.push(write_key);
            values.push(write_value);
        }
        StudentRecord {
            f,
            family: Family::MemoryLoadStore,
            final_output_digest: String::from(
                "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
            ),
            full_trace_digest: String::new(),
            inputs,
            model_json: None,
            outputs,
            program_hash: String::new(),
            record_id: String::from("trace_synth"),
            s,
            seed_writes: vec![(0, 17, 100), (0, 34, 200)],
            split: Split::Train,
            step_count,
        }
    }

    #[test]
    fn chunking_covers_sequence_and_masks_outputs() {
        let cfg = TrainConfig {
            model: ModelConfig {
                context: 64,
                d_ff: 16,
                d_model: 16,
                n_heads: 2,
                n_layers: 1,
                use_aux: true,
                use_lookup: true,
            },
            ..TrainConfig::w3_default(Baseline::B)
        };
        let record = synthetic_record(6);
        let chunks = chunks_of_record(&record, &cfg);
        assert!(!chunks.is_empty());
        let total_targets: usize = chunks.iter().map(|c| c.targets.len()).sum();
        // All output tokens except possibly the first one of the
        // sequence are predicted (the very first output token of the
        // record IS predicted because an input limb precedes it).
        assert_eq!(total_targets, 6 * 5 * LIMBS_PER_VALUE);
    }

    #[test]
    fn lookup_margin_is_zero_at_analytic_init() {
        let mut cfg = TrainConfig::w3_default(Baseline::C);
        cfg.model.context = 128;
        cfg.model.d_model = 16;
        cfg.model.d_ff = 16;
        cfg.model.n_heads = 2;
        cfg.model.n_layers = 1;
        let record = synthetic_record(4);
        let chunks = chunks_of_record(&record, &cfg);
        let mut model = Backbone::init(&cfg.model, 3);
        let t = cfg.model.context;
        let mut features = Vec::new();
        for chunk in &chunks {
            features.extend_from_slice(&chunk.features);
        }
        let acts = model.forward_train(&features, chunks.len(), t);
        let mut d_hidden = vec![0.0_f32; chunks.len() * t * cfg.model.d_model];
        let losses = head_losses_and_backward(&mut model, &acts, &chunks, &cfg, &mut d_hidden);
        // Analytic init: scores are exact; margin 1.0 between the
        // correct key (score 0) and nearest distinct key (<= -1).
        assert!(
            losses.lookup_acc > 0.999,
            "lookup_acc {}",
            losses.lookup_acc
        );
        assert!(losses.lookup.abs() < 1e-9, "lookup loss {}", losses.lookup);
    }
}
