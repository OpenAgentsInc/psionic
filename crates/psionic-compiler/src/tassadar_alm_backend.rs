use std::collections::BTreeMap;

use psionic_ir::{
    TassadarAlmEvaluationError, TassadarAlmGate, TassadarAlmGraph, TassadarAlmGraphError,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Stable schema version for the compiled ALM bundle.
pub const TASSADAR_ALM_COMPILED_BUNDLE_SCHEMA_VERSION: u16 = 1;
/// Stable compiler family identifier for ALM backend phase 1.
pub const TASSADAR_ALM_BACKEND_COMPILER_FAMILY: &str = "tassadar_alm_backend_list_schedule";
/// Stable compiler version identifier for ALM backend phase 1.
pub const TASSADAR_ALM_BACKEND_COMPILER_VERSION: &str = "v1";
/// Claim boundary for the compiled ALM bundle lane.
pub const TASSADAR_ALM_COMPILED_BUNDLE_CLAIM_BOUNDARY: &str = "the compiled ALM bundle executes \
     integer-exact analytical rows produced by a feasible-first list scheduler and an \
     interval-coloring slot allocator; it proves evaluator parity for committed workloads only \
     and does not claim optimal scheduling, tensor weight materialization, hull-cache decode, \
     Wasm intake, or any served-route capability";

/// Phase kinds inside the scheduled layer structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarAlmPhaseKind {
    /// Step-entry embedding phase carrying inputs and constants.
    Embedding,
    /// Attention phase carrying keyed reads and accumulator sums.
    Attention,
    /// Persist phase carrying linear wiring materialization.
    Persist,
    /// Feed-forward phase carrying ReGLU gates.
    Ffn,
}

/// Returns the phase kind for one global phase index.
#[must_use]
pub fn tassadar_alm_phase_kind(phase: u32) -> TassadarAlmPhaseKind {
    if phase == 0 {
        return TassadarAlmPhaseKind::Embedding;
    }
    match (phase - 1) % 4 {
        0 => TassadarAlmPhaseKind::Attention,
        1 | 3 => TassadarAlmPhaseKind::Persist,
        _ => TassadarAlmPhaseKind::Ffn,
    }
}

/// One scheduled gate placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarAlmGatePlacement {
    /// Gate index in the source graph.
    pub gate: u32,
    /// Global phase index (0 = embedding; layer L occupies 4L+1..=4L+4).
    pub phase: u32,
    /// Residual slot carrying the gate's value.
    pub slot: u32,
}

/// One stale-slot subtraction applied before a slot is reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarAlmSlotSubtraction {
    /// Phase at which the stale value is cleared.
    pub phase: u32,
    /// Reused residual slot.
    pub slot: u32,
    /// Gate whose stale value is subtracted.
    pub stale_gate: u32,
}

/// One compiled attention row.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarAlmAttentionRow {
    /// Keyed lookup head reading the latest value under the query key.
    KeyedRead {
        /// Source channel id.
        channel: u16,
        /// Residual slot carrying the query key.
        query_slot: u32,
        /// Residual slot receiving the read value.
        out_slot: u32,
        /// Phase index of this row.
        phase: u32,
    },
    /// Uniform-key accumulator head returning the running sum.
    CumSum {
        /// Accumulator channel id.
        channel: u16,
        /// Residual slot carrying the per-step contribution.
        value_slot: u32,
        /// Residual slot receiving the running sum.
        out_slot: u32,
        /// Phase index of this row.
        phase: u32,
    },
}

/// One compiled feed-forward ReGLU row computing `value * max(gate, 0)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarAlmFfnRow {
    /// Residual slot carrying the multiplicand.
    pub value_slot: u32,
    /// Residual slot carrying the gate operand.
    pub gate_slot: u32,
    /// Residual slot receiving the product.
    pub out_slot: u32,
    /// Phase index of this row.
    pub phase: u32,
}

/// One compiled wiring row materializing inputs, constants, or linear sums.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarAlmWiringRow {
    /// One per-step input field landing on a residual slot.
    Input {
        /// Input field index.
        field: u16,
        /// Destination residual slot.
        out_slot: u32,
        /// Phase index of this row.
        phase: u32,
    },
    /// One constant landing on a residual slot.
    Const {
        /// Constant value.
        value: i64,
        /// Destination residual slot.
        out_slot: u32,
        /// Phase index of this row.
        phase: u32,
    },
    /// One exact linear combination over residual slots.
    Linear {
        /// `(coefficient, slot)` terms.
        terms: Vec<(i64, u32)>,
        /// Constant bias.
        bias: i64,
        /// Destination residual slot.
        out_slot: u32,
        /// Phase index of this row.
        phase: u32,
    },
}

/// One end-of-step keyed-channel write emission.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarAlmWriteRow {
    /// Target keyed channel.
    pub channel: u16,
    /// Residual slot carrying the key.
    pub key_slot: u32,
    /// Residual slot carrying the written value.
    pub value_slot: u32,
}

/// One compiled, digest-pinned ALM bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarAlmCompiledBundle {
    /// Bundle schema version.
    pub schema_version: u16,
    /// Compiler family identifier.
    pub compiler_family: String,
    /// Compiler version identifier.
    pub compiler_version: String,
    /// Source graph identifier.
    pub graph_id: String,
    /// Source graph digest.
    pub graph_digest: String,
    /// Number of per-step input fields.
    pub input_field_count: u16,
    /// Number of scheduled layers (phase count excludes the embedding phase).
    pub layer_count: u32,
    /// Residual width in slots after interval-coloring reuse.
    pub slot_count: u32,
    /// Seed writes applied before step zero.
    pub seed_writes: Vec<(u16, i64, i64)>,
    /// Gate placements in the schedule.
    pub placements: Vec<TassadarAlmGatePlacement>,
    /// Stale-slot subtractions applied at reuse boundaries.
    pub subtractions: Vec<TassadarAlmSlotSubtraction>,
    /// Compiled wiring rows in phase order.
    pub wiring_rows: Vec<TassadarAlmWiringRow>,
    /// Compiled attention rows in phase order.
    pub attention_rows: Vec<TassadarAlmAttentionRow>,
    /// Compiled feed-forward rows in phase order.
    pub ffn_rows: Vec<TassadarAlmFfnRow>,
    /// End-of-step keyed-channel write emissions.
    pub write_rows: Vec<TassadarAlmWriteRow>,
    /// Residual slots exposed as per-step outputs.
    pub output_slots: Vec<u32>,
}

impl TassadarAlmCompiledBundle {
    /// Returns a stable digest over the full bundle encoding.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"tassadar_alm_compiled_bundle|");
        hasher.update(serde_json::to_vec(self).unwrap_or_default());
        hex::encode(hasher.finalize())
    }
}

/// Compilation failure for the ALM backend.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TassadarAlmBackendError {
    /// The source graph failed validation.
    #[error(transparent)]
    Graph(#[from] TassadarAlmGraphError),
    /// The scheduler produced an internally inconsistent placement.
    #[error("schedule invariant violated for gate {gate}: {reason}")]
    ScheduleInvariant {
        /// Offending gate index.
        gate: u32,
        /// Violated invariant description.
        reason: String,
    },
}

/// Execution failure for one compiled bundle run.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TassadarAlmCompiledExecutionError {
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
    /// One keyed read found no write under its query key.
    #[error("step {step} read missing key {key} on channel {channel}")]
    MissingKey {
        /// Failing step index.
        step: usize,
        /// Queried channel id.
        channel: u16,
        /// Missing key value.
        key: i64,
    },
    /// One exact integer operation overflowed 64 bits.
    #[error("step {step} overflowed exact 64-bit arithmetic")]
    Overflow {
        /// Failing step index.
        step: usize,
    },
}

/// One deterministic compiled-bundle execution trace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarAlmCompiledTrace {
    /// Digest of the executed bundle.
    pub bundle_digest: String,
    /// Source graph digest carried by the bundle.
    pub graph_digest: String,
    /// Number of executed steps.
    pub step_count: usize,
    /// Per-step output rows.
    pub step_outputs: Vec<Vec<i64>>,
    /// Stable digest over the output rows, comparable with the evaluator's.
    pub trace_digest: String,
}

struct ScheduledGate {
    phase: u32,
    slot: u32,
}

/// Compiles one validated ALM graph into a digest-pinned bundle.
pub fn compile_tassadar_alm_graph(
    graph: &TassadarAlmGraph,
) -> Result<TassadarAlmCompiledBundle, TassadarAlmBackendError> {
    graph.validate()?;
    let gate_count = graph.gates.len();
    // Phase scheduling: earliest feasible phase of the required kind, with
    // every dependency strictly earlier.
    let mut phases: Vec<u32> = vec![0; gate_count];
    // Alias map: a ChannelWrite gate aliases its value operand's slot.
    let mut alias: Vec<Option<u32>> = vec![None; gate_count];
    for (gate_index, gate) in graph.gates.iter().enumerate() {
        let deps: Vec<u32> = match gate {
            TassadarAlmGate::Input { .. } | TassadarAlmGate::Const { .. } => Vec::new(),
            TassadarAlmGate::Linear { terms, .. } => {
                terms.iter().map(|(_, value)| value.0).collect()
            }
            TassadarAlmGate::ReGlu { value, gate } => vec![value.0, gate.0],
            TassadarAlmGate::ChannelWrite { key, value, .. } => vec![key.0, value.0],
            TassadarAlmGate::ChannelRead { query, .. } => vec![query.0],
            TassadarAlmGate::CumSum { value, .. } => vec![value.0],
        };
        let earliest_after_deps = deps
            .iter()
            .map(|dep| phases[*dep as usize] + 1)
            .max()
            .unwrap_or(0);
        let phase = match gate {
            TassadarAlmGate::Input { .. } | TassadarAlmGate::Const { .. } => 0,
            TassadarAlmGate::Linear { .. } => {
                next_phase_of_kind(earliest_after_deps.max(1), TassadarAlmPhaseKind::Persist)
            }
            TassadarAlmGate::ReGlu { .. } => {
                next_phase_of_kind(earliest_after_deps.max(1), TassadarAlmPhaseKind::Ffn)
            }
            TassadarAlmGate::ChannelRead { .. } | TassadarAlmGate::CumSum { .. } => {
                next_phase_of_kind(earliest_after_deps.max(1), TassadarAlmPhaseKind::Attention)
            }
            TassadarAlmGate::ChannelWrite { value, .. } => {
                // Writes emit at end of step from materialized slots; the
                // placement phase records readiness, the value aliases its
                // operand slot.
                alias[gate_index] = Some(value.0);
                earliest_after_deps
            }
        };
        phases[gate_index] = phase;
    }
    let max_phase = phases.iter().copied().max().unwrap_or(0);
    let end_phase = max_phase + 1;
    // Lifetimes: birth at the gate's phase, death at the last consumer phase;
    // outputs and write operands live to end-of-step.
    let mut deaths: Vec<u32> = phases.clone();
    let mut record_use = |value: u32, phase: u32, deaths: &mut Vec<u32>| {
        if deaths[value as usize] < phase {
            deaths[value as usize] = phase;
        }
    };
    for (gate_index, gate) in graph.gates.iter().enumerate() {
        let phase = phases[gate_index];
        match gate {
            TassadarAlmGate::Input { .. } | TassadarAlmGate::Const { .. } => {}
            TassadarAlmGate::Linear { terms, .. } => {
                for (_, value) in terms {
                    record_use(value.0, phase, &mut deaths);
                }
            }
            TassadarAlmGate::ReGlu { value, gate } => {
                record_use(value.0, phase, &mut deaths);
                record_use(gate.0, phase, &mut deaths);
            }
            TassadarAlmGate::ChannelWrite { key, value, .. } => {
                record_use(key.0, end_phase, &mut deaths);
                record_use(value.0, end_phase, &mut deaths);
            }
            TassadarAlmGate::ChannelRead { query, .. } => {
                record_use(query.0, phase, &mut deaths);
            }
            TassadarAlmGate::CumSum { value, .. } => {
                record_use(value.0, phase, &mut deaths);
            }
        }
    }
    for output in &graph.outputs {
        record_use(output.0, end_phase, &mut deaths);
    }
    // Interval-coloring slot allocation in birth order with greedy reuse.
    let mut order: Vec<usize> = (0..gate_count).collect();
    order.sort_by_key(|gate| (phases[*gate], *gate));
    let mut slot_free_at: Vec<u32> = Vec::new();
    let mut slots: Vec<u32> = vec![0; gate_count];
    let mut subtractions: Vec<TassadarAlmSlotSubtraction> = Vec::new();
    let mut slot_last_gate: Vec<u32> = Vec::new();
    for gate_index in order {
        if let Some(target) = alias[gate_index] {
            slots[gate_index] = slots[target as usize];
            continue;
        }
        let birth = phases[gate_index];
        let death = deaths[gate_index];
        let mut chosen: Option<usize> = None;
        for (slot, free_at) in slot_free_at.iter().enumerate() {
            // A slot freed strictly before this birth can be reused.
            if *free_at < birth {
                chosen = Some(slot);
                break;
            }
        }
        match chosen {
            Some(slot) => {
                subtractions.push(TassadarAlmSlotSubtraction {
                    phase: birth,
                    slot: slot as u32,
                    stale_gate: slot_last_gate[slot],
                });
                slot_free_at[slot] = death;
                slot_last_gate[slot] = gate_index as u32;
                slots[gate_index] = slot as u32;
            }
            None => {
                slots[gate_index] = slot_free_at.len() as u32;
                slot_free_at.push(death);
                slot_last_gate.push(gate_index as u32);
            }
        }
    }
    let scheduled: Vec<ScheduledGate> = (0..gate_count)
        .map(|gate| ScheduledGate {
            phase: phases[gate],
            slot: slots[gate],
        })
        .collect();
    validate_schedule(graph, &scheduled)?;
    // Emit rows in phase order.
    let mut wiring_rows: Vec<TassadarAlmWiringRow> = Vec::new();
    let mut attention_rows: Vec<TassadarAlmAttentionRow> = Vec::new();
    let mut ffn_rows: Vec<TassadarAlmFfnRow> = Vec::new();
    let mut write_rows: Vec<TassadarAlmWriteRow> = Vec::new();
    let mut order_by_phase: Vec<usize> = (0..gate_count).collect();
    order_by_phase.sort_by_key(|gate| (phases[*gate], *gate));
    for gate_index in order_by_phase {
        let phase = phases[gate_index];
        let slot = slots[gate_index];
        match &graph.gates[gate_index] {
            TassadarAlmGate::Input { field } => {
                wiring_rows.push(TassadarAlmWiringRow::Input {
                    field: *field,
                    out_slot: slot,
                    phase,
                });
            }
            TassadarAlmGate::Const { value } => {
                wiring_rows.push(TassadarAlmWiringRow::Const {
                    value: *value,
                    out_slot: slot,
                    phase,
                });
            }
            TassadarAlmGate::Linear { terms, bias } => {
                wiring_rows.push(TassadarAlmWiringRow::Linear {
                    terms: terms
                        .iter()
                        .map(|(coefficient, value)| (*coefficient, slots[value.0 as usize]))
                        .collect(),
                    bias: *bias,
                    out_slot: slot,
                    phase,
                });
            }
            TassadarAlmGate::ReGlu { value, gate } => {
                ffn_rows.push(TassadarAlmFfnRow {
                    value_slot: slots[value.0 as usize],
                    gate_slot: slots[gate.0 as usize],
                    out_slot: slot,
                    phase,
                });
            }
            TassadarAlmGate::ChannelWrite {
                channel_id,
                key,
                value,
            } => {
                write_rows.push(TassadarAlmWriteRow {
                    channel: channel_id.0,
                    key_slot: slots[key.0 as usize],
                    value_slot: slots[value.0 as usize],
                });
            }
            TassadarAlmGate::ChannelRead { channel_id, query } => {
                attention_rows.push(TassadarAlmAttentionRow::KeyedRead {
                    channel: channel_id.0,
                    query_slot: slots[query.0 as usize],
                    out_slot: slot,
                    phase,
                });
            }
            TassadarAlmGate::CumSum { channel_id, value } => {
                attention_rows.push(TassadarAlmAttentionRow::CumSum {
                    channel: channel_id.0,
                    value_slot: slots[value.0 as usize],
                    out_slot: slot,
                    phase,
                });
            }
        }
    }
    let layer_count = max_phase.div_ceil(4);
    let placements = (0..gate_count)
        .map(|gate| TassadarAlmGatePlacement {
            gate: gate as u32,
            phase: phases[gate],
            slot: slots[gate],
        })
        .collect();
    Ok(TassadarAlmCompiledBundle {
        schema_version: TASSADAR_ALM_COMPILED_BUNDLE_SCHEMA_VERSION,
        compiler_family: TASSADAR_ALM_BACKEND_COMPILER_FAMILY.to_string(),
        compiler_version: TASSADAR_ALM_BACKEND_COMPILER_VERSION.to_string(),
        graph_id: graph.graph_id.clone(),
        graph_digest: graph.stable_digest(),
        input_field_count: graph.input_field_count,
        layer_count,
        slot_count: slot_free_at.len() as u32,
        seed_writes: graph
            .seed_writes
            .iter()
            .map(|seed| (seed.channel_id.0, seed.key, seed.value))
            .collect(),
        placements,
        subtractions,
        wiring_rows,
        attention_rows,
        ffn_rows,
        write_rows,
        output_slots: graph
            .outputs
            .iter()
            .map(|output| slots[output.0 as usize])
            .collect(),
    })
}

fn next_phase_of_kind(earliest: u32, kind: TassadarAlmPhaseKind) -> u32 {
    let mut phase = earliest.max(1);
    while tassadar_alm_phase_kind(phase) != kind {
        phase += 1;
    }
    phase
}

fn validate_schedule(
    graph: &TassadarAlmGraph,
    scheduled: &[ScheduledGate],
) -> Result<(), TassadarAlmBackendError> {
    for (gate_index, gate) in graph.gates.iter().enumerate() {
        let placement = &scheduled[gate_index];
        let expected_kind = match gate {
            TassadarAlmGate::Input { .. } | TassadarAlmGate::Const { .. } => {
                Some(TassadarAlmPhaseKind::Embedding)
            }
            TassadarAlmGate::Linear { .. } => Some(TassadarAlmPhaseKind::Persist),
            TassadarAlmGate::ReGlu { .. } => Some(TassadarAlmPhaseKind::Ffn),
            TassadarAlmGate::ChannelRead { .. } | TassadarAlmGate::CumSum { .. } => {
                Some(TassadarAlmPhaseKind::Attention)
            }
            TassadarAlmGate::ChannelWrite { .. } => None,
        };
        if let Some(kind) = expected_kind {
            if tassadar_alm_phase_kind(placement.phase) != kind {
                return Err(TassadarAlmBackendError::ScheduleInvariant {
                    gate: gate_index as u32,
                    reason: format!(
                        "phase {} has kind {:?}, expected {:?}",
                        placement.phase,
                        tassadar_alm_phase_kind(placement.phase),
                        kind
                    ),
                });
            }
        }
        let deps: Vec<u32> = match gate {
            TassadarAlmGate::Input { .. } | TassadarAlmGate::Const { .. } => Vec::new(),
            TassadarAlmGate::Linear { terms, .. } => {
                terms.iter().map(|(_, value)| value.0).collect()
            }
            TassadarAlmGate::ReGlu { value, gate } => vec![value.0, gate.0],
            TassadarAlmGate::ChannelWrite { key, value, .. } => vec![key.0, value.0],
            TassadarAlmGate::ChannelRead { query, .. } => vec![query.0],
            TassadarAlmGate::CumSum { value, .. } => vec![value.0],
        };
        for dep in deps {
            if scheduled[dep as usize].phase >= placement.phase
                && !matches!(gate, TassadarAlmGate::ChannelWrite { .. })
            {
                return Err(TassadarAlmBackendError::ScheduleInvariant {
                    gate: gate_index as u32,
                    reason: format!(
                        "dependency {dep} at phase {} is not strictly earlier than {}",
                        scheduled[dep as usize].phase, placement.phase
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Executes one compiled ALM bundle from its own rows only.
#[derive(Clone, Copy, Debug, Default)]
pub struct TassadarAlmCompiledExecutor;

impl TassadarAlmCompiledExecutor {
    /// Runs `bundle` over `steps`, one input-field row per step.
    pub fn execute(
        bundle: &TassadarAlmCompiledBundle,
        steps: &[Vec<i64>],
    ) -> Result<TassadarAlmCompiledTrace, TassadarAlmCompiledExecutionError> {
        let mut keyed: BTreeMap<u16, BTreeMap<i64, i64>> = BTreeMap::new();
        let mut accumulators: BTreeMap<u16, i64> = BTreeMap::new();
        for (channel, key, value) in &bundle.seed_writes {
            keyed.entry(*channel).or_default().insert(*key, *value);
        }
        // Phase-ordered row plan.
        let mut plan: Vec<(u32, RowRef)> = Vec::new();
        for (index, row) in bundle.wiring_rows.iter().enumerate() {
            let phase = match row {
                TassadarAlmWiringRow::Input { phase, .. }
                | TassadarAlmWiringRow::Const { phase, .. }
                | TassadarAlmWiringRow::Linear { phase, .. } => *phase,
            };
            plan.push((phase, RowRef::Wiring(index)));
        }
        for (index, row) in bundle.attention_rows.iter().enumerate() {
            let phase = match row {
                TassadarAlmAttentionRow::KeyedRead { phase, .. }
                | TassadarAlmAttentionRow::CumSum { phase, .. } => *phase,
            };
            plan.push((phase, RowRef::Attention(index)));
        }
        for (index, row) in bundle.ffn_rows.iter().enumerate() {
            plan.push((row.phase, RowRef::Ffn(index)));
        }
        plan.sort_by_key(|(phase, row)| (*phase, row.order_key()));
        let mut step_outputs: Vec<Vec<i64>> = Vec::with_capacity(steps.len());
        for (step_index, fields) in steps.iter().enumerate() {
            if fields.len() != bundle.input_field_count as usize {
                return Err(TassadarAlmCompiledExecutionError::InputArityMismatch {
                    step: step_index,
                    found: fields.len(),
                    expected: bundle.input_field_count,
                });
            }
            let mut residual: Vec<i64> = vec![0; bundle.slot_count as usize];
            for (_, row) in &plan {
                match row {
                    RowRef::Wiring(index) => match &bundle.wiring_rows[*index] {
                        TassadarAlmWiringRow::Input {
                            field, out_slot, ..
                        } => {
                            residual[*out_slot as usize] = fields[*field as usize];
                        }
                        TassadarAlmWiringRow::Const {
                            value, out_slot, ..
                        } => {
                            residual[*out_slot as usize] = *value;
                        }
                        TassadarAlmWiringRow::Linear {
                            terms,
                            bias,
                            out_slot,
                            ..
                        } => {
                            let mut total = *bias;
                            for (coefficient, slot) in terms {
                                let term = coefficient
                                    .checked_mul(residual[*slot as usize])
                                    .ok_or(TassadarAlmCompiledExecutionError::Overflow {
                                        step: step_index,
                                    })?;
                                total = total.checked_add(term).ok_or(
                                    TassadarAlmCompiledExecutionError::Overflow {
                                        step: step_index,
                                    },
                                )?;
                            }
                            residual[*out_slot as usize] = total;
                        }
                    },
                    RowRef::Attention(index) => match &bundle.attention_rows[*index] {
                        TassadarAlmAttentionRow::KeyedRead {
                            channel,
                            query_slot,
                            out_slot,
                            ..
                        } => {
                            let key = residual[*query_slot as usize];
                            let value = keyed
                                .get(channel)
                                .and_then(|cells| cells.get(&key))
                                .copied()
                                .ok_or(TassadarAlmCompiledExecutionError::MissingKey {
                                    step: step_index,
                                    channel: *channel,
                                    key,
                                })?;
                            residual[*out_slot as usize] = value;
                        }
                        TassadarAlmAttentionRow::CumSum {
                            channel,
                            value_slot,
                            out_slot,
                            ..
                        } => {
                            let contribution = residual[*value_slot as usize];
                            let current = accumulators.get(channel).copied().unwrap_or(0);
                            let total = current.checked_add(contribution).ok_or(
                                TassadarAlmCompiledExecutionError::Overflow { step: step_index },
                            )?;
                            accumulators.insert(*channel, total);
                            residual[*out_slot as usize] = total;
                        }
                    },
                    RowRef::Ffn(index) => {
                        let row = &bundle.ffn_rows[*index];
                        let gated = residual[row.gate_slot as usize].max(0);
                        let product = residual[row.value_slot as usize].checked_mul(gated).ok_or(
                            TassadarAlmCompiledExecutionError::Overflow { step: step_index },
                        )?;
                        residual[row.out_slot as usize] = product;
                    }
                }
            }
            for write in &bundle.write_rows {
                let key = residual[write.key_slot as usize];
                let value = residual[write.value_slot as usize];
                keyed.entry(write.channel).or_default().insert(key, value);
            }
            step_outputs.push(
                bundle
                    .output_slots
                    .iter()
                    .map(|slot| residual[*slot as usize])
                    .collect(),
            );
        }
        let mut hasher = Sha256::new();
        hasher.update(b"tassadar_alm_trace|");
        hasher.update(bundle.graph_digest.as_bytes());
        for row in &step_outputs {
            hasher.update(b"|row|");
            for value in row {
                hasher.update(value.to_le_bytes());
            }
        }
        let trace_digest = hex::encode(hasher.finalize());
        Ok(TassadarAlmCompiledTrace {
            bundle_digest: bundle.stable_digest(),
            graph_digest: bundle.graph_digest.clone(),
            step_count: steps.len(),
            step_outputs,
            trace_digest,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum RowRef {
    Wiring(usize),
    Attention(usize),
    Ffn(usize),
}

impl RowRef {
    fn order_key(self) -> (u8, usize) {
        match self {
            RowRef::Attention(index) => (0, index),
            RowRef::Wiring(index) => (1, index),
            RowRef::Ffn(index) => (2, index),
        }
    }
}

/// Maps one evaluator error family onto the compiled-execution family for
/// parity assertions in tests and conformance harnesses.
#[must_use]
pub fn tassadar_alm_errors_match(
    evaluator: &TassadarAlmEvaluationError,
    compiled: &TassadarAlmCompiledExecutionError,
) -> bool {
    matches!(
        (evaluator, compiled),
        (
            TassadarAlmEvaluationError::MissingKey { .. },
            TassadarAlmCompiledExecutionError::MissingKey { .. }
        ) | (
            TassadarAlmEvaluationError::Overflow { .. },
            TassadarAlmCompiledExecutionError::Overflow { .. }
        ) | (
            TassadarAlmEvaluationError::InputArityMismatch { .. },
            TassadarAlmCompiledExecutionError::InputArityMismatch { .. }
        )
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use psionic_ir::{
        tassadar_alm_running_sum_workload, tassadar_alm_stack_micro_workload,
        tassadar_alm_verb_parity_workload, TassadarAlmEvaluator,
    };

    use super::*;

    fn assert_parity(graph: &TassadarAlmGraph, steps: &[Vec<i64>]) -> TassadarAlmCompiledBundle {
        let bundle = compile_tassadar_alm_graph(graph).expect("compiles");
        let reference = TassadarAlmEvaluator::evaluate(graph, steps).expect("evaluates");
        let compiled = TassadarAlmCompiledExecutor::execute(&bundle, steps).expect("executes");
        assert_eq!(compiled.step_outputs, reference.step_outputs);
        assert_eq!(compiled.trace_digest, reference.trace_digest);
        bundle
    }

    #[test]
    fn running_sum_bundle_matches_evaluator_traces() {
        let graph = tassadar_alm_running_sum_workload();
        let bundle = assert_parity(&graph, &[vec![3], vec![5], vec![-2], vec![10]]);
        assert_eq!(bundle.layer_count, 1);
    }

    #[test]
    fn verb_parity_bundle_matches_evaluator_traces() {
        let graph = tassadar_alm_verb_parity_workload();
        let steps: Vec<Vec<i64>> = [0_i64, 0, 1, 0, 0, 0, 1, 0, 0, 0]
            .iter()
            .map(|bit| vec![*bit])
            .collect();
        assert_parity(&graph, &steps);
    }

    #[test]
    fn stack_micro_bundle_matches_evaluator_traces_and_reuses_slots() {
        let graph = tassadar_alm_stack_micro_workload();
        let steps = vec![vec![0, 3], vec![0, 5], vec![1, 0], vec![2, 0]];
        let bundle = assert_parity(&graph, &steps);
        assert!(
            (bundle.slot_count as usize) < graph.gates.len(),
            "expected slot reuse: {} slots for {} gates",
            bundle.slot_count,
            graph.gates.len()
        );
        assert!(!bundle.subtractions.is_empty());
    }

    #[test]
    fn bundle_digest_is_stable_and_graph_sensitive() {
        let graph = tassadar_alm_running_sum_workload();
        let a = compile_tassadar_alm_graph(&graph).expect("compiles");
        let b = compile_tassadar_alm_graph(&graph).expect("compiles");
        assert_eq!(a.stable_digest(), b.stable_digest());
        let mut renamed = graph.clone();
        renamed.graph_id = "alm.test.renamed".to_string();
        let c = compile_tassadar_alm_graph(&renamed).expect("compiles");
        assert_ne!(a.stable_digest(), c.stable_digest());
    }

    #[test]
    fn phase_kinds_follow_the_four_phase_layer_structure() {
        assert_eq!(tassadar_alm_phase_kind(0), TassadarAlmPhaseKind::Embedding);
        assert_eq!(tassadar_alm_phase_kind(1), TassadarAlmPhaseKind::Attention);
        assert_eq!(tassadar_alm_phase_kind(2), TassadarAlmPhaseKind::Persist);
        assert_eq!(tassadar_alm_phase_kind(3), TassadarAlmPhaseKind::Ffn);
        assert_eq!(tassadar_alm_phase_kind(4), TassadarAlmPhaseKind::Persist);
        assert_eq!(tassadar_alm_phase_kind(5), TassadarAlmPhaseKind::Attention);
    }

    #[test]
    fn refusals_match_between_evaluator_and_compiled_execution() {
        use psionic_ir::{
            TassadarAlmChannelDecl, TassadarAlmChannelId, TassadarAlmChannelKind, TassadarAlmGate,
            TassadarAlmValueId, TASSADAR_ALM_GRAPH_SCHEMA_VERSION,
        };
        let graph = TassadarAlmGraph {
            schema_version: TASSADAR_ALM_GRAPH_SCHEMA_VERSION,
            graph_id: "alm.test.compiled_missing_key".to_string(),
            input_field_count: 1,
            channels: vec![TassadarAlmChannelDecl {
                channel_id: TassadarAlmChannelId(0),
                name: "memory".to_string(),
                kind: TassadarAlmChannelKind::Keyed,
            }],
            seed_writes: Vec::new(),
            gates: vec![
                TassadarAlmGate::Input { field: 0 },
                TassadarAlmGate::ChannelRead {
                    channel_id: TassadarAlmChannelId(0),
                    query: TassadarAlmValueId(0),
                },
            ],
            outputs: vec![TassadarAlmValueId(1)],
        };
        let bundle = compile_tassadar_alm_graph(&graph).expect("compiles");
        let evaluator_error =
            TassadarAlmEvaluator::evaluate(&graph, &[vec![7]]).expect_err("refuses");
        let compiled_error =
            TassadarAlmCompiledExecutor::execute(&bundle, &[vec![7]]).expect_err("refuses");
        assert!(tassadar_alm_errors_match(&evaluator_error, &compiled_error));
    }
}
