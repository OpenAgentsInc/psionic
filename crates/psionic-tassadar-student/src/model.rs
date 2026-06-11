//! Small causal-transformer backbone with hand-derived backprop for the
//! W3 student sweep (openagents#4749).
//!
//! Architecture (shared by baselines a/b/c): byte-factorized uint16
//! token embeddings plus structural lane embeddings (limb index, value
//! index, role) and two assembled-scalar features; pre-LN blocks with
//! ALiBi causal attention; ReLU FFN; two 256-way byte heads predicting
//! the next token's low and high bytes. Baseline (b) adds predict-ahead
//! auxiliary state heads and an output-digest-prefix head. Baseline (c)
//! adds the parabolic lookup module in `lookup.rs`.
//!
//! Claim boundary: Psion lane. Everything this model produces is a
//! bounded statistic checked by replay; nothing here is a proof.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::tensor::{Param, SplitMix, gemm, gemm_at, gemm_bt};

/// Number of structural roles (seed/input/output).
pub const ROLE_COUNT: usize = 3;
/// Value-index embedding bucket count.
pub const VIDX_BUCKETS: usize = 16;
/// Auxiliary head targets (values) and bytes per value.
pub const AUX_VALUES: usize = 2;
/// Bytes per auxiliary value (i64).
pub const AUX_BYTES: usize = 8;
/// Digest-prefix head bytes.
pub const DIGEST_BYTES: usize = 4;

/// Backbone configuration (frozen into every checkpoint receipt).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// Residual width.
    pub d_model: usize,
    /// Attention heads.
    pub n_heads: usize,
    /// Transformer layers.
    pub n_layers: usize,
    /// FFN hidden width.
    pub d_ff: usize,
    /// Training chunk length and decode attention window, in tokens.
    pub context: usize,
    /// Auxiliary state heads enabled (baseline b).
    pub use_aux: bool,
    /// Parabolic lookup module enabled (baseline c).
    pub use_lookup: bool,
}

impl ModelConfig {
    fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

/// Layer parameters.
#[derive(Clone, Debug)]
pub struct LayerParams {
    /// Pre-attention LayerNorm gain.
    pub ln1_g: Param,
    /// Pre-attention LayerNorm bias.
    pub ln1_b: Param,
    /// Query projection.
    pub wq: Param,
    /// Key projection.
    pub wk: Param,
    /// Value projection.
    pub wv: Param,
    /// Output projection.
    pub wo: Param,
    /// Pre-FFN LayerNorm gain.
    pub ln2_g: Param,
    /// Pre-FFN LayerNorm bias.
    pub ln2_b: Param,
    /// FFN in.
    pub w1: Param,
    /// FFN in bias.
    pub b1: Param,
    /// FFN out.
    pub w2: Param,
    /// FFN out bias.
    pub b2: Param,
}

/// All trainable parameters.
#[derive(Clone, Debug)]
pub struct Backbone {
    /// Config.
    pub cfg: ModelConfig,
    /// Low-byte token embedding (256 x d).
    pub e_lo: Param,
    /// High-byte token embedding (256 x d).
    pub e_hi: Param,
    /// Limb-index embedding (4 x d).
    pub e_limb: Param,
    /// Value-index embedding (16 x d).
    pub e_vidx: Param,
    /// Role embedding (3 x d).
    pub e_role: Param,
    /// Scalar-feature projection (2 x d).
    pub w_feat: Param,
    /// Transformer layers.
    pub layers: Vec<LayerParams>,
    /// Final LayerNorm gain.
    pub lnf_g: Param,
    /// Final LayerNorm bias.
    pub lnf_b: Param,
    /// Next-token low-byte head (d x 256).
    pub head_lo: Param,
    /// Next-token high-byte head (d x 256).
    pub head_hi: Param,
    /// Low head bias.
    pub head_lo_b: Param,
    /// High head bias.
    pub head_hi_b: Param,
    /// Auxiliary state head (d x AUX_VALUES*AUX_BYTES*256), baseline b.
    pub w_aux: Param,
    /// Aux bias.
    pub b_aux: Param,
    /// Digest-prefix head (d x DIGEST_BYTES*256), baseline b.
    pub w_dig: Param,
    /// Digest bias.
    pub b_dig: Param,
    /// Lookup module scalars and hidden query projection, baseline c:
    /// row 0 = [beta_q, b_q, beta_k, b_k, score_scale, reserved].
    pub lk_scalars: Param,
    /// Lookup hidden-state query projection (d x 1).
    pub lk_wq: Param,
}

fn ones(rows: usize, cols: usize) -> Param {
    let mut param = Param::zeros(rows, cols);
    for value in &mut param.w {
        *value = 1.0;
    }
    param
}

impl Backbone {
    /// Fresh deterministic init. The lookup module gets the analytic
    /// parabolic initialization: beta = 1, biases = 0, hidden query
    /// projection = 0, score scale = 1 — exact at step zero.
    pub fn init(cfg: &ModelConfig, seed: u64) -> Self {
        let mut rng = SplitMix::new(seed);
        let d = cfg.d_model;
        let std = 0.02_f32;
        let layers = (0..cfg.n_layers)
            .map(|_| LayerParams {
                b1: Param::zeros(1, cfg.d_ff),
                b2: Param::zeros(1, d),
                ln1_b: Param::zeros(1, d),
                ln1_g: ones(1, d),
                ln2_b: Param::zeros(1, d),
                ln2_g: ones(1, d),
                w1: Param::randn(d, cfg.d_ff, std, &mut rng),
                w2: Param::randn(cfg.d_ff, d, std, &mut rng),
                wk: Param::randn(d, d, std, &mut rng),
                wo: Param::randn(d, d, std, &mut rng),
                wq: Param::randn(d, d, std, &mut rng),
                wv: Param::randn(d, d, std, &mut rng),
            })
            .collect();
        let mut lk_scalars = Param::zeros(1, 6);
        lk_scalars.w[0] = 1.0; // beta_q
        lk_scalars.w[2] = 1.0; // beta_k
        lk_scalars.w[4] = 1.0; // score scale
        Self {
            b_aux: Param::zeros(1, AUX_VALUES * AUX_BYTES * 256),
            b_dig: Param::zeros(1, DIGEST_BYTES * 256),
            cfg: cfg.clone(),
            e_hi: Param::randn(256, d, std, &mut rng),
            e_limb: Param::randn(4, d, std, &mut rng),
            e_lo: Param::randn(256, d, std, &mut rng),
            e_role: Param::randn(ROLE_COUNT, d, std, &mut rng),
            e_vidx: Param::randn(VIDX_BUCKETS, d, std, &mut rng),
            head_hi: Param::randn(d, 256, std, &mut rng),
            head_hi_b: Param::zeros(1, 256),
            head_lo: Param::randn(d, 256, std, &mut rng),
            head_lo_b: Param::zeros(1, 256),
            layers,
            lk_scalars,
            lk_wq: Param::zeros(d, 1),
            lnf_b: Param::zeros(1, d),
            lnf_g: ones(1, d),
            w_aux: Param::randn(d, AUX_VALUES * AUX_BYTES * 256, std, &mut rng),
            w_dig: Param::randn(d, DIGEST_BYTES * 256, std, &mut rng),
            w_feat: Param::randn(2, d, std, &mut rng),
        }
    }

    /// All parameters in stable order (for AdamW / clip / serialization).
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut params: Vec<&mut Param> = vec![
            &mut self.e_lo,
            &mut self.e_hi,
            &mut self.e_limb,
            &mut self.e_vidx,
            &mut self.e_role,
            &mut self.w_feat,
        ];
        for layer in &mut self.layers {
            params.push(&mut layer.ln1_g);
            params.push(&mut layer.ln1_b);
            params.push(&mut layer.wq);
            params.push(&mut layer.wk);
            params.push(&mut layer.wv);
            params.push(&mut layer.wo);
            params.push(&mut layer.ln2_g);
            params.push(&mut layer.ln2_b);
            params.push(&mut layer.w1);
            params.push(&mut layer.b1);
            params.push(&mut layer.w2);
            params.push(&mut layer.b2);
        }
        params.push(&mut self.lnf_g);
        params.push(&mut self.lnf_b);
        params.push(&mut self.head_lo);
        params.push(&mut self.head_hi);
        params.push(&mut self.head_lo_b);
        params.push(&mut self.head_hi_b);
        params.push(&mut self.w_aux);
        params.push(&mut self.b_aux);
        params.push(&mut self.w_dig);
        params.push(&mut self.b_dig);
        params.push(&mut self.lk_scalars);
        params.push(&mut self.lk_wq);
        params
    }

    /// Read-only parameter list in the same stable order.
    pub fn params(&self) -> Vec<&Param> {
        let mut params: Vec<&Param> = vec![
            &self.e_lo,
            &self.e_hi,
            &self.e_limb,
            &self.e_vidx,
            &self.e_role,
            &self.w_feat,
        ];
        for layer in &self.layers {
            params.push(&layer.ln1_g);
            params.push(&layer.ln1_b);
            params.push(&layer.wq);
            params.push(&layer.wk);
            params.push(&layer.wv);
            params.push(&layer.wo);
            params.push(&layer.ln2_g);
            params.push(&layer.ln2_b);
            params.push(&layer.w1);
            params.push(&layer.b1);
            params.push(&layer.w2);
            params.push(&layer.b2);
        }
        params.push(&self.lnf_g);
        params.push(&self.lnf_b);
        params.push(&self.head_lo);
        params.push(&self.head_hi);
        params.push(&self.head_lo_b);
        params.push(&self.head_hi_b);
        params.push(&self.w_aux);
        params.push(&self.b_aux);
        params.push(&self.w_dig);
        params.push(&self.b_dig);
        params.push(&self.lk_scalars);
        params.push(&self.lk_wq);
        params
    }

    /// Serializes all weights as little-endian f32 bytes.
    pub fn weights_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for param in self.params() {
            for value in &param.w {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    /// Loads weights from the byte layout written by `weights_bytes`.
    pub fn load_weights_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let mut offset = 0_usize;
        for param in self.params_mut() {
            let need = param.w.len() * 4;
            if offset + need > bytes.len() {
                return Err(format!(
                    "weights file truncated at offset {offset}, need {need} more bytes"
                ));
            }
            for value in &mut param.w {
                let mut raw = [0_u8; 4];
                raw.copy_from_slice(&bytes[offset..offset + 4]);
                *value = f32::from_le_bytes(raw);
                offset += 4;
            }
        }
        if offset != bytes.len() {
            return Err(format!(
                "weights file has {} trailing bytes",
                bytes.len() - offset
            ));
        }
        Ok(())
    }
}

/// ALiBi slope for one head.
pub fn alibi_slope(head: usize, n_heads: usize) -> f32 {
    2.0_f32.powf(-8.0 * (head as f32 + 1.0) / n_heads as f32)
}

/// One token's structural features for embedding.
#[derive(Clone, Copy, Debug)]
pub struct TokenFeatures {
    /// uint16 token.
    pub token: u16,
    /// Limb index 0-3.
    pub limb: u8,
    /// Value index bucket.
    pub vidx: u8,
    /// Role index (0 seed, 1 input, 2 output).
    pub role: u8,
    /// Assembled-scalar features.
    pub feats: [f32; 2],
}

fn embed_into(model: &Backbone, features: &TokenFeatures, out: &mut [f32]) {
    let d = model.cfg.d_model;
    let lo = (features.token & 0xff) as usize;
    let hi = (features.token >> 8) as usize;
    let limb = (features.limb as usize).min(3);
    let vidx = (features.vidx as usize).min(VIDX_BUCKETS - 1);
    let role = (features.role as usize).min(ROLE_COUNT - 1);
    for col in 0..d {
        out[col] = model.e_lo.w[lo * d + col]
            + model.e_hi.w[hi * d + col]
            + model.e_limb.w[limb * d + col]
            + model.e_vidx.w[vidx * d + col]
            + model.e_role.w[role * d + col]
            + features.feats[0] * model.w_feat.w[col]
            + features.feats[1] * model.w_feat.w[d + col];
    }
}

fn layer_norm_forward(
    x: &[f32],
    gain: &[f32],
    bias: &[f32],
    d: usize,
    y: &mut [f32],
    mean_out: &mut [f32],
    rstd_out: &mut [f32],
) {
    y.par_chunks_mut(d)
        .zip(mean_out.par_iter_mut().zip(rstd_out.par_iter_mut()))
        .enumerate()
        .for_each(|(row, (y_row, (mean_val, rstd_val)))| {
            let x_row = &x[row * d..(row + 1) * d];
            let mean = x_row.iter().sum::<f32>() / d as f32;
            let var =
                x_row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let rstd = 1.0 / (var + 1e-5).sqrt();
            for col in 0..d {
                y_row[col] = (x_row[col] - mean) * rstd * gain[col] + bias[col];
            }
            *mean_val = mean;
            *rstd_val = rstd;
        });
}

#[allow(clippy::too_many_arguments)]
fn layer_norm_backward(
    x: &[f32],
    gain: &[f32],
    mean: &[f32],
    rstd: &[f32],
    dy: &[f32],
    d: usize,
    dx: &mut [f32],
    dgain: &mut [f32],
    dbias: &mut [f32],
) {
    // Parameter grads (sequential over rows for determinism).
    let rows = mean.len();
    for row in 0..rows {
        let x_row = &x[row * d..(row + 1) * d];
        let dy_row = &dy[row * d..(row + 1) * d];
        for col in 0..d {
            let xhat = (x_row[col] - mean[row]) * rstd[row];
            dgain[col] += dy_row[col] * xhat;
            dbias[col] += dy_row[col];
        }
    }
    dx.par_chunks_mut(d).enumerate().for_each(|(row, dx_row)| {
        let x_row = &x[row * d..(row + 1) * d];
        let dy_row = &dy[row * d..(row + 1) * d];
        let mut sum_dxhat = 0.0_f32;
        let mut sum_dxhat_xhat = 0.0_f32;
        for col in 0..d {
            let xhat = (x_row[col] - mean[row]) * rstd[row];
            let dxhat = dy_row[col] * gain[col];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * xhat;
        }
        let inv_d = 1.0 / d as f32;
        for col in 0..d {
            let xhat = (x_row[col] - mean[row]) * rstd[row];
            let dxhat = dy_row[col] * gain[col];
            dx_row[col] +=
                rstd[row] * (dxhat - inv_d * sum_dxhat - xhat * inv_d * sum_dxhat_xhat);
        }
    });
}

fn add_bias(x: &mut [f32], bias: &[f32], n: usize) {
    x.par_chunks_mut(n).for_each(|row| {
        for (value, b) in row.iter_mut().zip(bias.iter()) {
            *value += *b;
        }
    });
}

fn bias_grad(dy: &[f32], n: usize, db: &mut [f32]) {
    for row in dy.chunks(n) {
        for (b, value) in db.iter_mut().zip(row.iter()) {
            *b += *value;
        }
    }
}

/// Saved activations for one training chunk batch.
pub struct Acts {
    /// Batch rows.
    pub b: usize,
    /// Chunk length.
    pub t: usize,
    /// Embedding output (input to layer 0).
    pub x0: Vec<f32>,
    /// Per layer: residual input, ln1 stats/out, q/k/v, probs, attn-cat,
    /// post-attention residual, ln2 stats/out, ffn pre-activation,
    /// ffn hidden (post relu).
    pub layers: Vec<LayerActs>,
    /// Final LN input.
    pub xf_in: Vec<f32>,
    /// Final LN stats.
    pub lnf_mean: Vec<f32>,
    /// Final LN stats.
    pub lnf_rstd: Vec<f32>,
    /// Final hidden states (after final LN).
    pub hidden: Vec<f32>,
}

/// Per-layer saved activations.
pub struct LayerActs {
    /// Residual input.
    pub x_in: Vec<f32>,
    /// LN1 mean.
    pub ln1_mean: Vec<f32>,
    /// LN1 rstd.
    pub ln1_rstd: Vec<f32>,
    /// LN1 output.
    pub y1: Vec<f32>,
    /// Queries.
    pub q: Vec<f32>,
    /// Keys.
    pub k: Vec<f32>,
    /// Values.
    pub v: Vec<f32>,
    /// Attention probabilities (b*h*t*t).
    pub probs: Vec<f32>,
    /// Concatenated head outputs.
    pub ao: Vec<f32>,
    /// Residual after attention (input to LN2).
    pub x_mid: Vec<f32>,
    /// LN2 mean.
    pub ln2_mean: Vec<f32>,
    /// LN2 rstd.
    pub ln2_rstd: Vec<f32>,
    /// LN2 output.
    pub y2: Vec<f32>,
    /// FFN hidden pre-activation.
    pub h_pre: Vec<f32>,
    /// FFN hidden post-relu.
    pub h_act: Vec<f32>,
}

impl Backbone {
    /// Training forward over a batch of `b` chunks of length `t`.
    /// Padded tail positions (token 0 role seed) are safe because
    /// padding only ever appears at the end of a chunk and attention is
    /// causal.
    pub fn forward_train(&self, features: &[TokenFeatures], b: usize, t: usize) -> Acts {
        let d = self.cfg.d_model;
        let m = b * t;
        let mut x0 = vec![0.0_f32; m * d];
        x0.par_chunks_mut(d).enumerate().for_each(|(row, out)| {
            embed_into(self, &features[row], out);
        });
        let mut layers_acts = Vec::with_capacity(self.cfg.n_layers);
        let mut x = x0.clone();
        for layer in &self.layers {
            let x_in = x.clone();
            let mut y1 = vec![0.0_f32; m * d];
            let mut ln1_mean = vec![0.0_f32; m];
            let mut ln1_rstd = vec![0.0_f32; m];
            layer_norm_forward(
                &x_in,
                &layer.ln1_g.w,
                &layer.ln1_b.w,
                d,
                &mut y1,
                &mut ln1_mean,
                &mut ln1_rstd,
            );
            let mut q = vec![0.0_f32; m * d];
            let mut k = vec![0.0_f32; m * d];
            let mut v = vec![0.0_f32; m * d];
            gemm(m, d, d, &y1, &layer.wq.w, &mut q);
            gemm(m, d, d, &y1, &layer.wk.w, &mut k);
            gemm(m, d, d, &y1, &layer.wv.w, &mut v);
            let h = self.cfg.n_heads;
            let dh = self.cfg.head_dim();
            let scale = 1.0 / (dh as f32).sqrt();
            let mut probs = vec![0.0_f32; b * h * t * t];
            let mut ao = vec![0.0_f32; m * d];
            // Parallel over (batch, head) pairs; each writes disjoint
            // probs blocks and disjoint ao column slices per batch row.
            let chunks: Vec<(usize, usize)> = (0..b)
                .flat_map(|bi| (0..h).map(move |hi| (bi, hi)))
                .collect();
            let ao_ptr = SyncPtr(ao.as_mut_ptr());
            let probs_ptr = SyncPtr(probs.as_mut_ptr());
            chunks.par_iter().for_each(|&(bi, hi)| {
                let slope = alibi_slope(hi, h);
                let base = bi * t;
                let col0 = hi * dh;
                let pblock = unsafe {
                    std::slice::from_raw_parts_mut(
                        probs_ptr.get().add((bi * h + hi) * t * t),
                        t * t,
                    )
                };
                let ao_slice = unsafe { std::slice::from_raw_parts_mut(ao_ptr.get(), m * d) };
                for i in 0..t {
                    let q_row = &q[(base + i) * d + col0..(base + i) * d + col0 + dh];
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..=i {
                        let k_row =
                            &k[(base + j) * d + col0..(base + j) * d + col0 + dh];
                        let mut dot = 0.0_f32;
                        for c in 0..dh {
                            dot = q_row[c].mul_add(k_row[c], dot);
                        }
                        let score = dot * scale - slope * (i - j) as f32;
                        pblock[i * t + j] = score;
                        if score > max {
                            max = score;
                        }
                    }
                    let mut total = 0.0_f32;
                    for j in 0..=i {
                        let e = (pblock[i * t + j] - max).exp();
                        pblock[i * t + j] = e;
                        total += e;
                    }
                    let inv = 1.0 / total;
                    let out_row = &mut ao_slice
                        [(base + i) * d + col0..(base + i) * d + col0 + dh];
                    for c in 0..dh {
                        out_row[c] = 0.0;
                    }
                    for j in 0..=i {
                        let p = pblock[i * t + j] * inv;
                        pblock[i * t + j] = p;
                        let v_row =
                            &v[(base + j) * d + col0..(base + j) * d + col0 + dh];
                        for c in 0..dh {
                            out_row[c] = p.mul_add(v_row[c], out_row[c]);
                        }
                    }
                }
            });
            let mut attn_out = vec![0.0_f32; m * d];
            gemm(m, d, d, &ao, &layer.wo.w, &mut attn_out);
            let mut x_mid = x_in.clone();
            x_mid.par_iter_mut()
                .zip(attn_out.par_iter())
                .for_each(|(x_val, a_val)| *x_val += *a_val);
            let mut y2 = vec![0.0_f32; m * d];
            let mut ln2_mean = vec![0.0_f32; m];
            let mut ln2_rstd = vec![0.0_f32; m];
            layer_norm_forward(
                &x_mid,
                &layer.ln2_g.w,
                &layer.ln2_b.w,
                d,
                &mut y2,
                &mut ln2_mean,
                &mut ln2_rstd,
            );
            let mut h_pre = vec![0.0_f32; m * self.cfg.d_ff];
            gemm(m, d, self.cfg.d_ff, &y2, &layer.w1.w, &mut h_pre);
            add_bias(&mut h_pre, &layer.b1.w, self.cfg.d_ff);
            let mut h_act = h_pre.clone();
            h_act.par_iter_mut().for_each(|value| {
                if *value < 0.0 {
                    *value = 0.0;
                }
            });
            let mut ffn_out = vec![0.0_f32; m * d];
            gemm(m, self.cfg.d_ff, d, &h_act, &layer.w2.w, &mut ffn_out);
            add_bias(&mut ffn_out, &layer.b2.w, d);
            let mut x_next = x_mid.clone();
            x_next
                .par_iter_mut()
                .zip(ffn_out.par_iter())
                .for_each(|(x_val, f_val)| *x_val += *f_val);
            layers_acts.push(LayerActs {
                x_in,
                ln1_mean,
                ln1_rstd,
                y1,
                q,
                k,
                v,
                probs,
                ao,
                x_mid,
                ln2_mean,
                ln2_rstd,
                y2,
                h_pre,
                h_act,
            });
            x = x_next;
        }
        let xf_in = x;
        let mut hidden = vec![0.0_f32; m * d];
        let mut lnf_mean = vec![0.0_f32; m];
        let mut lnf_rstd = vec![0.0_f32; m];
        layer_norm_forward(
            &xf_in,
            &self.lnf_g.w,
            &self.lnf_b.w,
            d,
            &mut hidden,
            &mut lnf_mean,
            &mut lnf_rstd,
        );
        Acts {
            b,
            hidden,
            layers: layers_acts,
            lnf_mean,
            lnf_rstd,
            t,
            x0,
            xf_in,
        }
    }

    /// Full backward from `d_hidden` (gradient at the final-LN output).
    /// Accumulates into parameter `.g` buffers.
    pub fn backward(
        &mut self,
        features: &[TokenFeatures],
        acts: &Acts,
        d_hidden: &[f32],
    ) {
        let d = self.cfg.d_model;
        let m = acts.b * acts.t;
        let mut dx = vec![0.0_f32; m * d];
        layer_norm_backward(
            &acts.xf_in,
            &self.lnf_g.w,
            &acts.lnf_mean,
            &acts.lnf_rstd,
            d_hidden,
            d,
            &mut dx,
            &mut self.lnf_g.g,
            &mut self.lnf_b.g,
        );
        let h = self.cfg.n_heads;
        let dh = self.cfg.head_dim();
        let t = acts.t;
        let b = acts.b;
        for (layer, layer_acts) in self.layers.iter_mut().zip(acts.layers.iter()).rev() {
            // FFN backward.
            let d_ffn_out = dx.clone(); // residual: dx flows to both
            bias_grad(&d_ffn_out, d, &mut layer.b2.g);
            let mut d_h_act = vec![0.0_f32; m * self.cfg.d_ff];
            gemm_bt(m, self.cfg.d_ff, d, &d_ffn_out, &layer.w2.w, &mut d_h_act);
            gemm_at(m, self.cfg.d_ff, d, &layer_acts.h_act, &d_ffn_out, &mut layer.w2.g);
            d_h_act
                .par_iter_mut()
                .zip(layer_acts.h_pre.par_iter())
                .for_each(|(g_val, pre)| {
                    if *pre <= 0.0 {
                        *g_val = 0.0;
                    }
                });
            bias_grad(&d_h_act, self.cfg.d_ff, &mut layer.b1.g);
            let mut d_y2 = vec![0.0_f32; m * d];
            gemm_bt(m, d, self.cfg.d_ff, &d_h_act, &layer.w1.w, &mut d_y2);
            gemm_at(m, d, self.cfg.d_ff, &layer_acts.y2, &d_h_act, &mut layer.w1.g);
            // dx currently holds gradient at x_next; LN2 backward adds
            // the FFN path's contribution at x_mid.
            let mut d_x_mid = dx.clone();
            layer_norm_backward(
                &layer_acts.x_mid,
                &layer.ln2_g.w,
                &layer_acts.ln2_mean,
                &layer_acts.ln2_rstd,
                &d_y2,
                d,
                &mut d_x_mid,
                &mut layer.ln2_g.g,
                &mut layer.ln2_b.g,
            );
            // Attention backward.
            let d_attn_out = d_x_mid.clone();
            let mut d_ao = vec![0.0_f32; m * d];
            gemm_bt(m, d, d, &d_attn_out, &layer.wo.w, &mut d_ao);
            gemm_at(m, d, d, &layer_acts.ao, &d_attn_out, &mut layer.wo.g);
            let mut d_q = vec![0.0_f32; m * d];
            let mut d_k = vec![0.0_f32; m * d];
            let mut d_v = vec![0.0_f32; m * d];
            let scale = 1.0 / (dh as f32).sqrt();
            let chunks: Vec<(usize, usize)> = (0..b)
                .flat_map(|bi| (0..h).map(move |hi| (bi, hi)))
                .collect();
            let dq_ptr = SyncPtr(d_q.as_mut_ptr());
            let dk_ptr = SyncPtr(d_k.as_mut_ptr());
            let dv_ptr = SyncPtr(d_v.as_mut_ptr());
            chunks.par_iter().for_each(|&(bi, hi)| {
                let base = bi * t;
                let col0 = hi * dh;
                let pblock = &layer_acts.probs[(bi * h + hi) * t * t..];
                let dq_s = unsafe { std::slice::from_raw_parts_mut(dq_ptr.get(), m * d) };
                let dk_s = unsafe { std::slice::from_raw_parts_mut(dk_ptr.get(), m * d) };
                let dv_s = unsafe { std::slice::from_raw_parts_mut(dv_ptr.get(), m * d) };
                for i in 0..t {
                    let d_out =
                        &d_ao[(base + i) * d + col0..(base + i) * d + col0 + dh];
                    // dprobs and dscores for row i.
                    let mut dot_sum = 0.0_f32;
                    let mut dprob = vec![0.0_f32; i + 1];
                    for j in 0..=i {
                        let v_row =
                            &layer_acts.v[(base + j) * d + col0..(base + j) * d + col0 + dh];
                        let mut dot = 0.0_f32;
                        for c in 0..dh {
                            dot = d_out[c].mul_add(v_row[c], dot);
                        }
                        dprob[j] = dot;
                        dot_sum += dot * pblock[i * t + j];
                    }
                    let q_row =
                        &layer_acts.q[(base + i) * d + col0..(base + i) * d + col0 + dh];
                    for j in 0..=i {
                        let p = pblock[i * t + j];
                        let dscore = p * (dprob[j] - dot_sum) * scale;
                        let k_row = &layer_acts.k
                            [(base + j) * d + col0..(base + j) * d + col0 + dh];
                        let dq_row = &mut dq_s
                            [(base + i) * d + col0..(base + i) * d + col0 + dh];
                        for c in 0..dh {
                            dq_row[c] = dscore.mul_add(k_row[c], dq_row[c]);
                        }
                        let dk_row = &mut dk_s
                            [(base + j) * d + col0..(base + j) * d + col0 + dh];
                        for c in 0..dh {
                            dk_row[c] = dscore.mul_add(q_row[c], dk_row[c]);
                        }
                        let dv_row = &mut dv_s
                            [(base + j) * d + col0..(base + j) * d + col0 + dh];
                        for c in 0..dh {
                            dv_row[c] = p.mul_add(d_out[c], dv_row[c]);
                        }
                    }
                }
            });
            // Project q/k/v gradients back to y1 and weights.
            let mut d_y1 = vec![0.0_f32; m * d];
            gemm_bt(m, d, d, &d_q, &layer.wq.w, &mut d_y1);
            gemm_bt(m, d, d, &d_k, &layer.wk.w, &mut d_y1);
            gemm_bt(m, d, d, &d_v, &layer.wv.w, &mut d_y1);
            gemm_at(m, d, d, &layer_acts.y1, &d_q, &mut layer.wq.g);
            gemm_at(m, d, d, &layer_acts.y1, &d_k, &mut layer.wk.g);
            gemm_at(m, d, d, &layer_acts.y1, &d_v, &mut layer.wv.g);
            // LN1 backward: d_x_mid (residual) + LN1 path into x_in.
            let mut d_x_in = d_x_mid;
            layer_norm_backward(
                &layer_acts.x_in,
                &layer.ln1_g.w,
                &layer_acts.ln1_mean,
                &layer_acts.ln1_rstd,
                &d_y1,
                d,
                &mut d_x_in,
                &mut layer.ln1_g.g,
                &mut layer.ln1_b.g,
            );
            dx = d_x_in;
        }
        // Embedding backward (sequential scatter, deterministic).
        for (row, feat) in features.iter().enumerate().take(m) {
            let dx_row = &dx[row * d..(row + 1) * d];
            let lo = (feat.token & 0xff) as usize;
            let hi = (feat.token >> 8) as usize;
            let limb = (feat.limb as usize).min(3);
            let vidx = (feat.vidx as usize).min(VIDX_BUCKETS - 1);
            let role = (feat.role as usize).min(ROLE_COUNT - 1);
            for col in 0..d {
                let g = dx_row[col];
                self.e_lo.g[lo * d + col] += g;
                self.e_hi.g[hi * d + col] += g;
                self.e_limb.g[limb * d + col] += g;
                self.e_vidx.g[vidx * d + col] += g;
                self.e_role.g[role * d + col] += g;
                self.w_feat.g[col] += g * feat.feats[0];
                self.w_feat.g[d + col] += g * feat.feats[1];
            }
        }
    }
}

/// Send+Sync raw pointer wrapper for disjoint parallel writes.
struct SyncPtr(*mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

impl SyncPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}

/// Incremental decode state (KV ring buffer per layer).
pub struct DecodeState {
    /// Absolute position of the next token.
    pub pos: usize,
    window: usize,
    /// Per layer: (k ring, v ring, absolute positions).
    layers: Vec<(Vec<f32>, Vec<f32>, Vec<usize>)>,
}

impl DecodeState {
    /// Fresh decode state with the model's context window.
    pub fn new(cfg: &ModelConfig) -> Self {
        let window = cfg.context;
        Self {
            layers: (0..cfg.n_layers)
                .map(|_| {
                    (
                        vec![0.0_f32; window * cfg.d_model],
                        vec![0.0_f32; window * cfg.d_model],
                        vec![usize::MAX; window],
                    )
                })
                .collect(),
            pos: 0,
            window,
        }
    }
}

fn ln_row(x: &[f32], gain: &[f32], bias: &[f32], out: &mut [f32]) {
    let d = x.len();
    let mean = x.iter().sum::<f32>() / d as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    let rstd = 1.0 / (var + 1e-5).sqrt();
    for col in 0..d {
        out[col] = (x[col] - mean) * rstd * gain[col] + bias[col];
    }
}

fn matvec(w: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    for value in out.iter_mut().take(cols) {
        *value = 0.0;
    }
    for (row, x_val) in x.iter().enumerate().take(rows) {
        if *x_val == 0.0 {
            continue;
        }
        let w_row = &w[row * cols..(row + 1) * cols];
        for col in 0..cols {
            out[col] = x_val.mul_add(w_row[col], out[col]);
        }
    }
}

impl Backbone {
    /// Decodes one token incrementally, returning the final hidden state.
    pub fn decode_step(&self, state: &mut DecodeState, features: &TokenFeatures) -> Vec<f32> {
        let d = self.cfg.d_model;
        let h = self.cfg.n_heads;
        let dh = self.cfg.head_dim();
        let scale = 1.0 / (dh as f32).sqrt();
        let mut x = vec![0.0_f32; d];
        embed_into(self, features, &mut x);
        let slot = state.pos % state.window;
        for (layer, (k_ring, v_ring, pos_ring)) in
            self.layers.iter().zip(state.layers.iter_mut())
        {
            let mut y1 = vec![0.0_f32; d];
            ln_row(&x, &layer.ln1_g.w, &layer.ln1_b.w, &mut y1);
            let mut q = vec![0.0_f32; d];
            let mut k = vec![0.0_f32; d];
            let mut v = vec![0.0_f32; d];
            matvec(&layer.wq.w, &y1, d, d, &mut q);
            matvec(&layer.wk.w, &y1, d, d, &mut k);
            matvec(&layer.wv.w, &y1, d, d, &mut v);
            k_ring[slot * d..(slot + 1) * d].copy_from_slice(&k);
            v_ring[slot * d..(slot + 1) * d].copy_from_slice(&v);
            pos_ring[slot] = state.pos;
            let mut ao = vec![0.0_f32; d];
            for hi in 0..h {
                let slope = alibi_slope(hi, h);
                let col0 = hi * dh;
                let q_head = &q[col0..col0 + dh];
                let mut scores: Vec<(usize, f32)> = Vec::with_capacity(state.window);
                let mut max = f32::NEG_INFINITY;
                for cache_slot in 0..state.window {
                    let cache_pos = pos_ring[cache_slot];
                    if cache_pos == usize::MAX || cache_pos > state.pos {
                        continue;
                    }
                    let k_row = &k_ring[cache_slot * d + col0..cache_slot * d + col0 + dh];
                    let mut dot = 0.0_f32;
                    for c in 0..dh {
                        dot = q_head[c].mul_add(k_row[c], dot);
                    }
                    let score = dot * scale - slope * (state.pos - cache_pos) as f32;
                    scores.push((cache_slot, score));
                    if score > max {
                        max = score;
                    }
                }
                let mut total = 0.0_f32;
                for (_, score) in &mut scores {
                    *score = (*score - max).exp();
                    total += *score;
                }
                let inv = 1.0 / total;
                let out = &mut ao[col0..col0 + dh];
                for (cache_slot, p) in &scores {
                    let v_row = &v_ring[cache_slot * d + col0..cache_slot * d + col0 + dh];
                    let weight = *p * inv;
                    for c in 0..dh {
                        out[c] = weight.mul_add(v_row[c], out[c]);
                    }
                }
            }
            let mut attn_out = vec![0.0_f32; d];
            matvec(&layer.wo.w, &ao, d, d, &mut attn_out);
            for col in 0..d {
                x[col] += attn_out[col];
            }
            let mut y2 = vec![0.0_f32; d];
            ln_row(&x, &layer.ln2_g.w, &layer.ln2_b.w, &mut y2);
            let mut h_pre = vec![0.0_f32; self.cfg.d_ff];
            matvec(&layer.w1.w, &y2, d, self.cfg.d_ff, &mut h_pre);
            for (value, bias) in h_pre.iter_mut().zip(layer.b1.w.iter()) {
                *value = (*value + *bias).max(0.0);
            }
            let mut ffn_out = vec![0.0_f32; d];
            matvec(&layer.w2.w, &h_pre, self.cfg.d_ff, d, &mut ffn_out);
            for col in 0..d {
                x[col] += ffn_out[col] + layer.b2.w[col];
            }
        }
        state.pos += 1;
        let mut hidden = vec![0.0_f32; d];
        ln_row(&x, &self.lnf_g.w, &self.lnf_b.w, &mut hidden);
        hidden
    }

    /// Next-token byte-head prediction from a hidden state.
    pub fn predict_token(&self, hidden: &[f32]) -> u16 {
        let d = self.cfg.d_model;
        let mut lo_logits = vec![0.0_f32; 256];
        let mut hi_logits = vec![0.0_f32; 256];
        matvec(&self.head_lo.w, hidden, d, 256, &mut lo_logits);
        matvec(&self.head_hi.w, hidden, d, 256, &mut hi_logits);
        for (logit, bias) in lo_logits.iter_mut().zip(self.head_lo_b.w.iter()) {
            *logit += *bias;
        }
        for (logit, bias) in hi_logits.iter_mut().zip(self.head_hi_b.w.iter()) {
            *logit += *bias;
        }
        let lo = argmax(&lo_logits);
        let hi = argmax(&hi_logits);
        ((hi as u16) << 8) | lo as u16
    }
}

/// Index of the maximum value (first on ties).
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_val {
            best_val = *value;
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    fn tiny_cfg() -> ModelConfig {
        ModelConfig {
            context: 16,
            d_ff: 8,
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            use_aux: false,
            use_lookup: false,
        }
    }

    fn tiny_features(count: usize) -> Vec<TokenFeatures> {
        (0..count)
            .map(|index| TokenFeatures {
                feats: [0.1 * index as f32, -0.05 * index as f32],
                limb: (index % 4) as u8,
                role: (index % 3) as u8,
                token: (index * 37 % 65_536) as u16,
                vidx: (index % 7) as u8,
            })
            .collect()
    }

    /// Finite-difference gradient check on a scalar loss = sum of
    /// selected hidden entries. Validates the full backward path.
    #[test]
    fn backward_matches_finite_difference() {
        let cfg = tiny_cfg();
        let mut model = Backbone::init(&cfg, 42);
        // Push FFN pre-activations away from the ReLU kink so the
        // finite-difference probe stays on one side of it.
        for layer in &mut model.layers {
            for value in &mut layer.b1.w {
                *value = 0.3;
            }
        }
        let features = tiny_features(8);
        let (b, t) = (1, 8);
        let loss_of = |model: &Backbone| -> f64 {
            let acts = model.forward_train(&features, b, t);
            acts.hidden
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let weight = ((index % 5) as f64 - 2.0) * 3.0;
                    weight * f64::from(*value)
                })
                .sum()
        };
        let acts = model.forward_train(&features, b, t);
        let mut d_hidden = vec![0.0_f32; acts.hidden.len()];
        for (index, value) in d_hidden.iter_mut().enumerate() {
            *value = ((index % 5) as f32 - 2.0) * 3.0;
        }
        model.backward(&features, &acts, &d_hidden);
        // Check a few parameters across the stack.
        let eps = 3e-3_f32;
        let checks: Vec<(usize, usize)> = vec![(0, 3), (6, 0), (8, 5), (14, 2), (18, 7)];
        for (param_idx, weight_idx) in checks {
            let analytic = f64::from(model.params()[param_idx].g[weight_idx]);
            {
                let mut params = model.params_mut();
                params[param_idx].w[weight_idx] += eps;
            }
            let up = loss_of(&model);
            {
                let mut params = model.params_mut();
                params[param_idx].w[weight_idx] -= 2.0 * eps;
            }
            let down = loss_of(&model);
            {
                let mut params = model.params_mut();
                params[param_idx].w[weight_idx] += eps;
            }
            let numeric = (up - down) / (2.0 * f64::from(eps));
            let denom = analytic.abs().max(numeric.abs()).max(1e-4);
            // Relative agreement, with an absolute floor for gradients
            // near the f32 finite-difference noise floor.
            assert!(
                (analytic - numeric).abs() / denom < 0.08
                    || (analytic - numeric).abs() < 5e-4,
                "param {param_idx} weight {weight_idx}: analytic {analytic} vs numeric {numeric}"
            );
        }
    }

    /// Incremental decode must match the chunked training forward.
    #[test]
    fn decode_matches_forward() {
        let cfg = tiny_cfg();
        let model = Backbone::init(&cfg, 7);
        let features = tiny_features(12);
        let acts = model.forward_train(&features, 1, 12);
        let mut state = DecodeState::new(&cfg);
        for (index, feat) in features.iter().enumerate() {
            let hidden = model.decode_step(&mut state, feat);
            let d = cfg.d_model;
            for col in 0..d {
                let train_val = acts.hidden[index * d + col];
                assert!(
                    (hidden[col] - train_val).abs() < 1e-4,
                    "pos {index} col {col}: decode {} vs train {}",
                    hidden[col],
                    train_val
                );
            }
        }
    }
}
