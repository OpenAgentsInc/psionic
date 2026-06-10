use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::tassadar_alm_backend::{
    TassadarAlmAttentionRow, TassadarAlmCompiledBundle, TassadarAlmWiringRow,
};

/// Stable schema version for the numeric model artifact.
pub const TASSADAR_ALM_NUMERIC_MODEL_SCHEMA_VERSION: u16 = 1;
/// Stable executor identifier for the numeric leg.
pub const TASSADAR_ALM_NUMERIC_EXECUTOR_ID: &str = "tassadar.alm_numeric_executor.v1";
/// Claim boundary for the numeric materialization lane.
pub const TASSADAR_ALM_NUMERIC_CLAIM_BOUNDARY: &str = "the numeric model is a faithful f64 \
     re-encoding of one compiled ALM bundle - explicit coefficient arrays executed with \
     hard-max attention inside a checked exactness window of 2^53 - not a trained transformer; \
     it claims integer parity only while every intermediate stays inside the window, refuses \
     when one does not, and makes no softmax, learning, or served-route claim";

/// Exactness window: f64 represents every integer with |v| <= 2^53.
pub const TASSADAR_ALM_NUMERIC_EXACT_WINDOW: f64 = 9_007_199_254_740_992.0;

/// One numeric wiring row: a sparse linear map over the residual vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarAlmNumericWiringRow {
    /// Destination residual slot.
    pub out_slot: u32,
    /// Constant bias.
    pub bias: f64,
    /// `(coefficient, slot)` terms; empty terms make a constant or input.
    pub terms: Vec<(f64, u32)>,
    /// Input field landing on the slot before the linear map, if any.
    pub input_field: Option<u16>,
    /// Phase index of this row.
    pub phase: u32,
}

/// One numeric attention row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarAlmNumericAttentionRow {
    /// Hard-max parabolic-point read.
    KeyedRead {
        /// Source channel id.
        channel: u16,
        /// Residual slot carrying the query.
        query_slot: u32,
        /// Residual slot receiving the value.
        out_slot: u32,
        /// Phase index of this row.
        phase: u32,
    },
    /// Accumulator running sum.
    CumSum {
        /// Accumulator channel id.
        channel: u16,
        /// Residual slot carrying the contribution.
        value_slot: u32,
        /// Residual slot receiving the running sum.
        out_slot: u32,
        /// Phase index of this row.
        phase: u32,
    },
}

/// One numeric gated neuron: `out = value * max(gate, 0)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarAlmNumericFfnRow {
    /// Residual slot carrying the multiplicand.
    pub value_slot: u32,
    /// Residual slot carrying the gate operand.
    pub gate_slot: u32,
    /// Destination residual slot.
    pub out_slot: u32,
    /// Phase index of this row.
    pub phase: u32,
}

/// One numeric end-of-step keyed write emission.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarAlmNumericWriteRow {
    /// Target keyed channel.
    pub channel: u16,
    /// Residual slot carrying the key.
    pub key_slot: u32,
    /// Residual slot carrying the value.
    pub value_slot: u32,
}

/// One weights-shaped numeric model: a compiled bundle re-encoded as data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarAlmNumericModel {
    /// Stable schema version.
    pub schema_version: u16,
    /// Stable model identifier.
    pub model_id: String,
    /// Source graph digest.
    pub graph_digest: String,
    /// Source compiled-bundle digest.
    pub bundle_digest: String,
    /// Per-step input field count.
    pub input_field_count: u16,
    /// Residual width in slots.
    pub slot_count: u32,
    /// Scheduled layer count.
    pub layer_count: u32,
    /// Seed writes applied before step zero.
    pub seed_writes: Vec<(u16, f64, f64)>,
    /// Wiring rows in phase order.
    pub wiring: Vec<TassadarAlmNumericWiringRow>,
    /// Attention rows in phase order.
    pub attention: Vec<TassadarAlmNumericAttentionRow>,
    /// Gated-neuron rows in phase order.
    pub ffn: Vec<TassadarAlmNumericFfnRow>,
    /// End-of-step keyed write emissions in source-gate order.
    pub writes: Vec<TassadarAlmNumericWriteRow>,
    /// Residual slots exposed as per-step outputs.
    pub output_slots: Vec<u32>,
}

impl TassadarAlmNumericModel {
    /// Returns a stable digest over the full model encoding.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"tassadar_alm_numeric_model|");
        hasher.update(serde_json::to_vec(self).unwrap_or_default());
        hex::encode(hasher.finalize())
    }
}

/// Lowers one compiled bundle into its numeric model. No information is
/// lost; the model is the bundle re-expressed as coefficient data.
#[must_use]
pub fn materialize_tassadar_alm_numeric(
    bundle: &TassadarAlmCompiledBundle,
) -> TassadarAlmNumericModel {
    let wiring = bundle
        .wiring_rows
        .iter()
        .map(|row| match row {
            TassadarAlmWiringRow::Input {
                field,
                out_slot,
                phase,
            } => TassadarAlmNumericWiringRow {
                out_slot: *out_slot,
                bias: 0.0,
                terms: Vec::new(),
                input_field: Some(*field),
                phase: *phase,
            },
            TassadarAlmWiringRow::Const {
                value,
                out_slot,
                phase,
            } => TassadarAlmNumericWiringRow {
                out_slot: *out_slot,
                bias: *value as f64,
                terms: Vec::new(),
                input_field: None,
                phase: *phase,
            },
            TassadarAlmWiringRow::Linear {
                terms,
                bias,
                out_slot,
                phase,
            } => TassadarAlmNumericWiringRow {
                out_slot: *out_slot,
                bias: *bias as f64,
                terms: terms
                    .iter()
                    .map(|(coefficient, slot)| (*coefficient as f64, *slot))
                    .collect(),
                input_field: None,
                phase: *phase,
            },
        })
        .collect();
    let attention = bundle
        .attention_rows
        .iter()
        .map(|row| match row {
            TassadarAlmAttentionRow::KeyedRead {
                channel,
                query_slot,
                out_slot,
                phase,
            } => TassadarAlmNumericAttentionRow::KeyedRead {
                channel: *channel,
                query_slot: *query_slot,
                out_slot: *out_slot,
                phase: *phase,
            },
            TassadarAlmAttentionRow::CumSum {
                channel,
                value_slot,
                out_slot,
                phase,
            } => TassadarAlmNumericAttentionRow::CumSum {
                channel: *channel,
                value_slot: *value_slot,
                out_slot: *out_slot,
                phase: *phase,
            },
        })
        .collect();
    let ffn = bundle
        .ffn_rows
        .iter()
        .map(|row| TassadarAlmNumericFfnRow {
            value_slot: row.value_slot,
            gate_slot: row.gate_slot,
            out_slot: row.out_slot,
            phase: row.phase,
        })
        .collect();
    let writes = bundle
        .write_rows
        .iter()
        .map(|row| TassadarAlmNumericWriteRow {
            channel: row.channel,
            key_slot: row.key_slot,
            value_slot: row.value_slot,
        })
        .collect();
    TassadarAlmNumericModel {
        schema_version: TASSADAR_ALM_NUMERIC_MODEL_SCHEMA_VERSION,
        model_id: format!("alm.numeric.{}", bundle.graph_id),
        graph_digest: bundle.graph_digest.clone(),
        bundle_digest: bundle.stable_digest(),
        input_field_count: bundle.input_field_count,
        slot_count: bundle.slot_count,
        layer_count: bundle.layer_count,
        seed_writes: bundle
            .seed_writes
            .iter()
            .map(|(channel, key, value)| (*channel, *key as f64, *value as f64))
            .collect(),
        wiring,
        attention,
        ffn,
        writes,
        output_slots: bundle.output_slots.clone(),
    }
}

/// Execution failure for the numeric leg.
#[derive(Debug, Error, PartialEq)]
pub enum TassadarAlmNumericExecutionError {
    /// One step supplied the wrong number of input fields.
    #[error("step {step} supplies {found} input fields, expected {expected}")]
    InputArityMismatch {
        /// Failing step index.
        step: usize,
        /// Supplied field count.
        found: usize,
        /// Expected field count.
        expected: u16,
    },
    /// One keyed read found no point under its query.
    #[error("step {step} read missing key {key} on channel {channel}")]
    MissingKey {
        /// Failing step index.
        step: usize,
        /// Queried channel id.
        channel: u16,
        /// Missing key value.
        key: i64,
    },
    /// One intermediate value left the f64 exact-integer window.
    #[error("step {step} value {value:e} left the 2^53 exactness window")]
    ExactnessWindowExceeded {
        /// Failing step index.
        step: usize,
        /// Offending value.
        value: f64,
    },
}

/// One deterministic numeric execution trace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TassadarAlmNumericTrace {
    /// Executor identifier.
    pub executor_id: String,
    /// Digest of the executed model.
    pub model_digest: String,
    /// Source graph digest.
    pub graph_digest: String,
    /// Number of executed steps.
    pub step_count: usize,
    /// Per-step output rows, converted back to exact integers.
    pub step_outputs: Vec<Vec<i64>>,
    /// Stable digest over the output rows, comparable with the evaluator's.
    pub trace_digest: String,
}

#[derive(Clone, Copy, Debug)]
struct NumericPoint {
    key: f64,
    value: f64,
    write_order: u64,
}

fn check_window(value: f64, step: usize) -> Result<f64, TassadarAlmNumericExecutionError> {
    if value.abs() > TASSADAR_ALM_NUMERIC_EXACT_WINDOW {
        return Err(TassadarAlmNumericExecutionError::ExactnessWindowExceeded { step, value });
    }
    Ok(value)
}

/// Executes one numeric model in f64 inside the exactness window.
pub fn tassadar_alm_numeric_execute(
    model: &TassadarAlmNumericModel,
    steps: &[Vec<i64>],
) -> Result<TassadarAlmNumericTrace, TassadarAlmNumericExecutionError> {
    let mut points: BTreeMap<u16, Vec<NumericPoint>> = BTreeMap::new();
    let mut accumulators: BTreeMap<u16, f64> = BTreeMap::new();
    let mut write_order = 0_u64;
    for (channel, key, value) in &model.seed_writes {
        points.entry(*channel).or_default().push(NumericPoint {
            key: *key,
            value: *value,
            write_order,
        });
        write_order += 1;
    }
    // Phase-ordered plan.
    let mut plan: Vec<(u32, u8, usize)> = Vec::new();
    for (index, row) in model.attention.iter().enumerate() {
        let phase = match row {
            TassadarAlmNumericAttentionRow::KeyedRead { phase, .. }
            | TassadarAlmNumericAttentionRow::CumSum { phase, .. } => *phase,
        };
        plan.push((phase, 0, index));
    }
    for (index, row) in model.wiring.iter().enumerate() {
        plan.push((row.phase, 1, index));
    }
    for (index, row) in model.ffn.iter().enumerate() {
        plan.push((row.phase, 2, index));
    }
    plan.sort_unstable();
    let mut step_outputs: Vec<Vec<i64>> = Vec::with_capacity(steps.len());
    for (step_index, fields) in steps.iter().enumerate() {
        if fields.len() != model.input_field_count as usize {
            return Err(TassadarAlmNumericExecutionError::InputArityMismatch {
                step: step_index,
                found: fields.len(),
                expected: model.input_field_count,
            });
        }
        let mut residual: Vec<f64> = vec![0.0; model.slot_count as usize];
        for (_, kind, index) in &plan {
            match kind {
                1 => {
                    let row = &model.wiring[*index];
                    let mut total = row.bias;
                    if let Some(field) = row.input_field {
                        total += fields[field as usize] as f64;
                    }
                    for (coefficient, slot) in &row.terms {
                        total += coefficient * residual[*slot as usize];
                    }
                    residual[row.out_slot as usize] = check_window(total, step_index)?;
                }
                0 => match &model.attention[*index] {
                    TassadarAlmNumericAttentionRow::KeyedRead {
                        channel,
                        query_slot,
                        out_slot,
                        ..
                    } => {
                        let query = residual[*query_slot as usize];
                        let channel_points =
                            points.get(channel).map(Vec::as_slice).unwrap_or_default();
                        // Hard-max over parabolic scores 2qk - k^2 in f64;
                        // ties (duplicate keys) break to the latest write.
                        let mut best: Option<&NumericPoint> = None;
                        let mut best_score = f64::NEG_INFINITY;
                        for point in channel_points {
                            let score = 2.0 * query * point.key - point.key * point.key;
                            let better = match best {
                                None => true,
                                Some(current) => {
                                    score > best_score
                                        || (score == best_score
                                            && point.write_order > current.write_order)
                                }
                            };
                            if better {
                                best = Some(point);
                                best_score = score;
                            }
                        }
                        match best {
                            Some(point) if point.key == query => {
                                residual[*out_slot as usize] = point.value;
                            }
                            _ => {
                                return Err(TassadarAlmNumericExecutionError::MissingKey {
                                    step: step_index,
                                    channel: *channel,
                                    key: query as i64,
                                });
                            }
                        }
                    }
                    TassadarAlmNumericAttentionRow::CumSum {
                        channel,
                        value_slot,
                        out_slot,
                        ..
                    } => {
                        let contribution = residual[*value_slot as usize];
                        let total =
                            accumulators.get(channel).copied().unwrap_or(0.0) + contribution;
                        let total = check_window(total, step_index)?;
                        accumulators.insert(*channel, total);
                        residual[*out_slot as usize] = total;
                    }
                },
                _ => {
                    let row = &model.ffn[*index];
                    let gated = residual[row.gate_slot as usize].max(0.0);
                    let product = residual[row.value_slot as usize] * gated;
                    residual[row.out_slot as usize] = check_window(product, step_index)?;
                }
            }
        }
        for write in &model.writes {
            let key = residual[write.key_slot as usize];
            let value = residual[write.value_slot as usize];
            points.entry(write.channel).or_default().push(NumericPoint {
                key,
                value,
                write_order,
            });
            write_order += 1;
        }
        step_outputs.push(
            model
                .output_slots
                .iter()
                .map(|slot| residual[*slot as usize] as i64)
                .collect(),
        );
    }
    let mut hasher = Sha256::new();
    hasher.update(b"tassadar_alm_trace|");
    hasher.update(model.graph_digest.as_bytes());
    for row in &step_outputs {
        hasher.update(b"|row|");
        for value in row {
            hasher.update(value.to_le_bytes());
        }
    }
    let trace_digest = hex::encode(hasher.finalize());
    Ok(TassadarAlmNumericTrace {
        executor_id: TASSADAR_ALM_NUMERIC_EXECUTOR_ID.to_string(),
        model_digest: model.stable_digest(),
        graph_digest: model.graph_digest.clone(),
        step_count: steps.len(),
        step_outputs,
        trace_digest,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use psionic_ir::{
        tassadar_alm_running_sum_workload, tassadar_alm_stack_micro_workload,
        tassadar_alm_verb_parity_workload, TassadarAlmEvaluator,
    };
    use psionic_runtime::{
        TassadarCpuReferenceRunner, TassadarInstruction, TassadarProgram, TassadarWasmProfile,
    };

    use super::*;
    use crate::tassadar_alm_backend::compile_tassadar_alm_graph;
    use crate::tassadar_alm_wasm_interpreter::{
        tassadar_alm_wasm_collect, tassadar_alm_wasm_interpreter,
    };

    fn assert_numeric_parity(
        graph: &psionic_ir::TassadarAlmGraph,
        steps: &[Vec<i64>],
    ) -> TassadarAlmNumericTrace {
        let reference = TassadarAlmEvaluator::evaluate(graph, steps).expect("evaluates");
        let bundle = compile_tassadar_alm_graph(graph).expect("compiles");
        let model = materialize_tassadar_alm_numeric(&bundle);
        let numeric = tassadar_alm_numeric_execute(&model, steps).expect("numeric executes");
        assert_eq!(numeric.step_outputs, reference.step_outputs);
        assert_eq!(numeric.trace_digest, reference.trace_digest);
        numeric
    }

    #[test]
    fn committed_workloads_agree_through_the_numeric_model() {
        assert_numeric_parity(
            &tassadar_alm_running_sum_workload(),
            &[vec![3], vec![5], vec![-2], vec![10]],
        );
        let parity_steps: Vec<Vec<i64>> = [0_i64, 0, 1, 0, 0, 0, 1, 0, 0, 0]
            .iter()
            .map(|bit| vec![*bit])
            .collect();
        assert_numeric_parity(&tassadar_alm_verb_parity_workload(), &parity_steps);
        assert_numeric_parity(
            &tassadar_alm_stack_micro_workload(),
            &[vec![0, 3], vec![0, 5], vec![1, 0], vec![2, 0]],
        );
    }

    #[test]
    fn a_real_runtime_program_runs_as_a_numeric_model() {
        use TassadarInstruction as I;
        let program = TassadarProgram::new(
            "alm_numeric.loop_sum",
            &TassadarWasmProfile::article_i32_compute_v1(),
            2,
            1,
            vec![
                I::I32Const { value: 0 },
                I::LocalSet { local: 0 },
                I::I32Const { value: 1 },
                I::LocalSet { local: 1 },
                I::LocalGet { local: 0 },
                I::LocalGet { local: 1 },
                I::I32Add,
                I::LocalSet { local: 0 },
                I::LocalGet { local: 1 },
                I::I32Const { value: 1 },
                I::I32Add,
                I::LocalSet { local: 1 },
                I::LocalGet { local: 1 },
                I::I32Const { value: 6 },
                I::I32Lt,
                I::BrIf { target_pc: 4 },
                I::LocalGet { local: 0 },
                I::Output,
                I::Return,
            ],
        );
        let runner = TassadarCpuReferenceRunner::for_program(&program).expect("runner");
        let expected: Vec<i64> = runner
            .execute(&program)
            .expect("executes")
            .outputs
            .iter()
            .map(|v| i64::from(*v))
            .collect();
        let graph = tassadar_alm_wasm_interpreter(&program).expect("builds");
        let bundle = compile_tassadar_alm_graph(&graph).expect("compiles");
        let model = materialize_tassadar_alm_numeric(&bundle);
        let steps = vec![vec![0_i64]; 100];
        let trace = tassadar_alm_numeric_execute(&model, &steps).expect("numeric executes");
        let (outputs, halted) = tassadar_alm_wasm_collect(&trace.step_outputs);
        assert!(halted);
        assert_eq!(outputs, expected);
        assert_eq!(outputs, vec![15]);
    }

    #[test]
    fn the_model_is_portable_data() {
        let graph = tassadar_alm_stack_micro_workload();
        let bundle = compile_tassadar_alm_graph(&graph).expect("compiles");
        let model = materialize_tassadar_alm_numeric(&bundle);
        let encoded = serde_json::to_string(&model).expect("serializes");
        let decoded: TassadarAlmNumericModel =
            serde_json::from_str(&encoded).expect("deserializes");
        assert_eq!(decoded.stable_digest(), model.stable_digest());
        let steps = vec![vec![0, 3], vec![0, 5], vec![1, 0], vec![2, 0]];
        let original = tassadar_alm_numeric_execute(&model, &steps).expect("executes");
        let roundtrip = tassadar_alm_numeric_execute(&decoded, &steps).expect("executes");
        assert_eq!(original.step_outputs, roundtrip.step_outputs);
    }

    #[test]
    fn values_outside_the_exactness_window_refuse() {
        use psionic_ir::{
            TassadarAlmGate, TassadarAlmGraph, TassadarAlmValueId,
            TASSADAR_ALM_GRAPH_SCHEMA_VERSION,
        };
        let graph = TassadarAlmGraph {
            schema_version: TASSADAR_ALM_GRAPH_SCHEMA_VERSION,
            graph_id: "alm.numeric.window".to_string(),
            input_field_count: 1,
            channels: Vec::new(),
            seed_writes: Vec::new(),
            gates: vec![
                TassadarAlmGate::Input { field: 0 },
                TassadarAlmGate::Const { value: 1 << 30 },
                TassadarAlmGate::ReGlu {
                    value: TassadarAlmValueId(1),
                    gate: TassadarAlmValueId(1),
                },
                TassadarAlmGate::ReGlu {
                    value: TassadarAlmValueId(2),
                    gate: TassadarAlmValueId(1),
                },
            ],
            outputs: vec![TassadarAlmValueId(3)],
        };
        let bundle = compile_tassadar_alm_graph(&graph).expect("compiles");
        let model = materialize_tassadar_alm_numeric(&bundle);
        let error = tassadar_alm_numeric_execute(&model, &[vec![0]]).expect_err("refuses");
        assert!(matches!(
            error,
            TassadarAlmNumericExecutionError::ExactnessWindowExceeded { step: 0, .. }
        ));
    }
}

#[cfg(test)]
mod fixture_dump {
    #![allow(clippy::expect_used)]
    use psionic_runtime::{TassadarInstruction, TassadarProgram, TassadarWasmProfile};

    use super::*;
    use crate::tassadar_alm_backend::compile_tassadar_alm_graph;
    use crate::tassadar_alm_wasm_interpreter::tassadar_alm_wasm_interpreter;

    #[test]
    #[ignore]
    fn dump_poc_fixture() {
        use TassadarInstruction as I;
        let program = TassadarProgram::new(
            "tassadar_poc.loop_sum_v1",
            &TassadarWasmProfile::article_i32_compute_v1(),
            2,
            1,
            vec![
                I::I32Const { value: 0 },
                I::LocalSet { local: 0 },
                I::I32Const { value: 1 },
                I::LocalSet { local: 1 },
                I::LocalGet { local: 0 },
                I::LocalGet { local: 1 },
                I::I32Add,
                I::LocalSet { local: 0 },
                I::LocalGet { local: 1 },
                I::I32Const { value: 1 },
                I::I32Add,
                I::LocalSet { local: 1 },
                I::LocalGet { local: 1 },
                I::I32Const { value: 6 },
                I::I32Lt,
                I::BrIf { target_pc: 4 },
                I::LocalGet { local: 0 },
                I::Output,
                I::Return,
            ],
        );
        let graph = tassadar_alm_wasm_interpreter(&program).expect("builds");
        let bundle = compile_tassadar_alm_graph(&graph).expect("compiles");
        let model = materialize_tassadar_alm_numeric(&bundle);
        let steps: Vec<Vec<i64>> = vec![vec![0]; 80];
        let trace = tassadar_alm_numeric_execute(&model, &steps).expect("executes");
        let fixture = serde_json::json!({
            "fixtureId": "tassadar_poc.loop_sum_v1.numeric_fixture.v1",
            "generatedBy": "psionic crates/psionic-compiler tassadar_alm_numeric fixture_dump (psionic main)",
            "programId": "tassadar_poc.loop_sum_v1",
            "model": model,
            "steps": steps,
            "expectedTraceDigest": trace.trace_digest,
            "expectedModelDigest": model.stable_digest(),
            "expectedFinalRow": trace.step_outputs.last(),
            "expectedOutputs": [15],
            "claimBoundary": TASSADAR_ALM_NUMERIC_CLAIM_BOUNDARY,
        });
        std::fs::write(
            "/tmp/tassadar-poc-fixture.json",
            serde_json::to_vec_pretty(&fixture).expect("encodes"),
        )
        .expect("writes");
        eprintln!("trace_digest={}", trace.trace_digest);
    }
}
