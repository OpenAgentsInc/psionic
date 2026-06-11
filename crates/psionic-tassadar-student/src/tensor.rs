//! Minimal deterministic f32 tensor math for the W3 student baselines.
//!
//! Everything is row-major `Vec<f32>` with explicit shapes. Parallelism
//! is rayon over independent output rows, so results are bit-identical
//! across runs and thread counts (every output element is reduced
//! sequentially over `k`).

use rayon::prelude::*;

/// One trainable dense parameter with Adam moments.
#[derive(Clone, Debug)]
pub struct Param {
    /// Rows (input dim for linear weights).
    pub rows: usize,
    /// Columns (output dim for linear weights).
    pub cols: usize,
    /// Weights, row-major.
    pub w: Vec<f32>,
    /// Gradient accumulator.
    pub g: Vec<f32>,
    /// Adam first moment.
    pub m: Vec<f32>,
    /// Adam second moment.
    pub v: Vec<f32>,
}

impl Param {
    /// Zero-initialized parameter.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            cols,
            g: vec![0.0; rows * cols],
            m: vec![0.0; rows * cols],
            rows,
            v: vec![0.0; rows * cols],
            w: vec![0.0; rows * cols],
        }
    }

    /// Gaussian-ish init from a splitmix stream at the given std.
    pub fn randn(rows: usize, cols: usize, std: f32, rng: &mut SplitMix) -> Self {
        let mut param = Self::zeros(rows, cols);
        for value in &mut param.w {
            *value = rng.normal() * std;
        }
        param
    }
}

/// splitmix64 — deterministic stream for init and shuffling.
#[derive(Clone, Debug)]
pub struct SplitMix {
    state: u64,
}

impl SplitMix {
    /// New stream from a seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1_u64 << 24) as f32)
    }

    /// Approximate standard normal (sum of uniforms, CLT with 4 terms).
    pub fn normal(&mut self) -> f32 {
        let sum: f32 = (0..4).map(|_| self.uniform()).sum();
        (sum - 2.0) * (12.0_f32 / 4.0).sqrt()
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for index in (1..items.len()).rev() {
            let pick = (self.next_u64() % (index as u64 + 1)) as usize;
            items.swap(index, pick);
        }
    }
}

/// C(M,N) = A(M,K) x B(K,N), overwriting C.
pub fn gemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    let _ = m;
    c.par_chunks_mut(n).enumerate().for_each(|(row, c_row)| {
        for value in c_row.iter_mut() {
            *value = 0.0;
        }
        let a_row = &a[row * k..(row + 1) * k];
        for (kk, a_val) in a_row.iter().enumerate() {
            if *a_val == 0.0 {
                continue;
            }
            let b_row = &b[kk * n..(kk + 1) * n];
            for (c_val, b_val) in c_row.iter_mut().zip(b_row.iter()) {
                *c_val = a_val.mul_add(*b_val, *c_val);
            }
        }
    });
}

/// G_B(K,N) += A^T(M,K) x G_C(M,N): weight-gradient accumulation.
pub fn gemm_at(m: usize, k: usize, n: usize, a: &[f32], g_c: &[f32], g_b: &mut [f32]) {
    g_b.par_chunks_mut(n).enumerate().for_each(|(kk, g_row)| {
        for row in 0..m {
            let a_val = a[row * k + kk];
            if a_val == 0.0 {
                continue;
            }
            let c_row = &g_c[row * n..(row + 1) * n];
            for (g_val, c_val) in g_row.iter_mut().zip(c_row.iter()) {
                *g_val = a_val.mul_add(*c_val, *g_val);
            }
        }
    });
}

/// G_A(M,K) += G_C(M,N) x B^T(N,K) where B is stored (K,N).
pub fn gemm_bt(m: usize, k: usize, n: usize, g_c: &[f32], b: &[f32], g_a: &mut [f32]) {
    let _ = m;
    g_a.par_chunks_mut(k).enumerate().for_each(|(row, a_row)| {
        let c_row = &g_c[row * n..(row + 1) * n];
        for (kk, a_val) in a_row.iter_mut().enumerate() {
            let b_row = &b[kk * n..(kk + 1) * n];
            let mut total = 0.0_f32;
            for (c_val, b_val) in c_row.iter().zip(b_row.iter()) {
                total = c_val.mul_add(*b_val, total);
            }
            *a_val += total;
        }
    });
}

/// Numerically-stable softmax cross-entropy over one logit row.
/// Writes `probs` in place of `logits` and returns the loss.
pub fn softmax_ce_row(logits: &mut [f32], target: usize) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for value in logits.iter() {
        if *value > max {
            max = *value;
        }
    }
    let mut total = 0.0_f32;
    for value in logits.iter_mut() {
        *value = (*value - max).exp();
        total += *value;
    }
    let inv = 1.0 / total;
    for value in logits.iter_mut() {
        *value *= inv;
    }
    -(logits[target].max(1e-30)).ln()
}

/// AdamW step over one parameter.
#[allow(clippy::too_many_arguments)]
pub fn adamw(
    param: &mut Param,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u64,
    grad_scale: f32,
) {
    let bias1 = 1.0 - beta1.powi(step as i32);
    let bias2 = 1.0 - beta2.powi(step as i32);
    let w = &mut param.w;
    let g = &mut param.g;
    let m = &mut param.m;
    let v = &mut param.v;
    w.par_iter_mut()
        .zip(g.par_iter_mut())
        .zip(m.par_iter_mut().zip(v.par_iter_mut()))
        .for_each(|((w_val, g_val), (m_val, v_val))| {
            let grad = *g_val * grad_scale;
            *m_val = beta1 * *m_val + (1.0 - beta1) * grad;
            *v_val = beta2 * *v_val + (1.0 - beta2) * grad * grad;
            let m_hat = *m_val / bias1;
            let v_hat = *v_val / bias2;
            *w_val -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * *w_val);
            *g_val = 0.0;
        });
}

/// Global L2 norm over all gradients.
pub fn grad_norm(params: &[&Param]) -> f32 {
    let total: f64 = params
        .iter()
        .map(|param| {
            param
                .g
                .iter()
                .map(|g| f64::from(*g) * f64::from(*g))
                .sum::<f64>()
        })
        .sum();
    total.sqrt() as f32
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn gemm_matches_naive() {
        let mut rng = SplitMix::new(7);
        let (m, k, n) = (5, 4, 3);
        let a: Vec<f32> = (0..m * k).map(|_| rng.normal()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.normal()).collect();
        let mut c = vec![0.0_f32; m * n];
        gemm(m, k, n, &a, &b, &mut c);
        for row in 0..m {
            for col in 0..n {
                let mut want = 0.0_f32;
                for kk in 0..k {
                    want = a[row * k + kk].mul_add(b[kk * n + col], want);
                }
                assert!((c[row * n + col] - want).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn softmax_ce_is_stable() {
        let mut logits = vec![1000.0_f32, 1001.0, 999.0];
        let loss = softmax_ce_row(&mut logits, 1);
        assert!(loss < 1.0);
        let total: f32 = logits.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }
}
