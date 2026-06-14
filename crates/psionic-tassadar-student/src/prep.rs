//! `student_prep.v0.1` reader and the student serialization protocol
//! (`student_seq.v0.1`) for the W3 student program (openagents#4749).
//!
//! The prep files are emitted by the monorepo harness
//! (`apps/openagents.com/workers/api/scripts/tassadar-w3-student-prep.ts`)
//! from the verified `corpus.tassadar_trace.v0_2.w3_100m` snapshot; the
//! output rows in them are decoded from Tier-0/Tier-1-verified trace
//! records and the input rows from the deterministic workload builders
//! whose program hashes the Tier-1 replay already pinned.
//!
//! Student serialization protocol `student_seq.v0.1`:
//!   * preamble: every seed write as (key, value), 4 uint16 LE limbs per
//!     i64 — the program-state preamble that makes seeded keyed reads
//!     well-posed for a learner;
//!   * per step: the F input values (teacher-forced environment input),
//!     then the S output values (the verified corpus tokens), each as
//!     4 uint16 LE limbs;
//!   * the corpus's compact trace tokens are exactly the output-value
//!     limbs, in record order.

use std::fmt::Write as _;

use thiserror::Error;

/// Number of uint16 limbs per i64 value (trace_token.v0.1, uint16 width).
pub const LIMBS_PER_VALUE: usize = 4;

/// Typed prep-file decode failure.
#[derive(Debug, Error)]
pub enum PrepError {
    /// The file does not start with the TSPREP1 magic.
    #[error("bad prep magic")]
    BadMagic,
    /// The format version is unsupported.
    #[error("unsupported prep version {0}")]
    UnsupportedVersion(u32),
    /// The byte stream ended mid-field.
    #[error("truncated prep file at offset {0} reading {1}")]
    Truncated(usize, &'static str),
    /// A record carried an out-of-range family or split index.
    #[error("invalid enum value {value} for {what}")]
    InvalidEnum {
        /// Field name.
        what: &'static str,
        /// Observed value.
        value: u32,
    },
    /// A string field was not valid UTF-8.
    #[error("invalid utf8 in {0}")]
    InvalidUtf8(&'static str),
}

/// Program family of a record (frozen v0.1 family set).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// family.arithmetic_carry.v1
    ArithmeticCarry,
    /// family.memory_load_store.v1
    MemoryLoadStore,
    /// family.branch_gated_control.v1
    BranchGatedControl,
    /// family.application_state_machine.v1
    ApplicationStateMachine,
    /// family.near_miss_lookup.v1
    NearMissLookup,
    /// family.stack_loop_sum.compiled.v1
    StackLoopSum,
}

impl Family {
    /// Family from the prep-file index.
    pub fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::ArithmeticCarry),
            1 => Some(Self::MemoryLoadStore),
            2 => Some(Self::BranchGatedControl),
            3 => Some(Self::ApplicationStateMachine),
            4 => Some(Self::NearMissLookup),
            5 => Some(Self::StackLoopSum),
            _ => None,
        }
    }

    /// Canonical family id string.
    pub fn id(self) -> &'static str {
        match self {
            Self::ArithmeticCarry => "family.arithmetic_carry.v1",
            Self::MemoryLoadStore => "family.memory_load_store.v1",
            Self::BranchGatedControl => "family.branch_gated_control.v1",
            Self::ApplicationStateMachine => "family.application_state_machine.v1",
            Self::NearMissLookup => "family.near_miss_lookup.v1",
            Self::StackLoopSum => "family.stack_loop_sum.compiled.v1",
        }
    }
}

/// Training-split assignment of a record (W2 split policy v0.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// In-policy training record.
    Train,
    /// Held-out program family (economic family, compiled anchor).
    EvalHeldoutFamily,
    /// Train-family record beyond trainMaxSteps (2x/4x/8x).
    EvalLongHorizon,
    /// Near-miss lookup adversary family.
    EvalAdversarial,
}

impl Split {
    fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::Train),
            1 => Some(Self::EvalHeldoutFamily),
            2 => Some(Self::EvalLongHorizon),
            3 => Some(Self::EvalAdversarial),
            _ => None,
        }
    }
}

/// Divergence-cause class for one output slot, mapped from the family
/// builders' slot semantics (workload-families.ts).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlotCause {
    /// Echo of an input field — a fetch of environment input.
    WrongFetch,
    /// Hard-max keyed-read result.
    MemoryRead,
    /// Relu-gated branch selection.
    Branch,
    /// Cum-sum accumulator (carry-style running state).
    Carry,
    /// Affine combination output.
    Output,
    /// Compiled stack-machine anchor row.
    StackDepth,
}

impl SlotCause {
    /// Stable report label.
    pub fn label(self) -> &'static str {
        match self {
            Self::WrongFetch => "wrong_fetch",
            Self::MemoryRead => "memory_read",
            Self::Branch => "branch",
            Self::Carry => "carry",
            Self::Output => "output",
            Self::StackDepth => "stack_depth",
        }
    }
}

/// Where a keyed-write operand comes from within a step.
#[derive(Clone, Copy, Debug)]
pub enum OperandSrc {
    /// Input field index.
    Input(usize),
    /// Output slot index (within the record's output row).
    Output(usize),
}

/// Keyed-read structure of a family (channel 0 of the builders).
#[derive(Clone, Copy, Debug)]
pub struct ReadSpec {
    /// Input field carrying the read query key.
    pub query_input: usize,
    /// Output row index carrying the read result.
    pub result_out_idx: usize,
    /// Per-step write key operand.
    pub write_key: OperandSrc,
    /// Per-step write value operand.
    pub write_value: OperandSrc,
}

/// Static per-family semantics mirrored from the frozen v0.1 builders.
#[derive(Clone, Debug)]
pub struct FamilySpec {
    /// Input fields per step.
    pub f: usize,
    /// Output slots per step.
    pub s: usize,
    /// Divergence-cause class per output row index.
    pub causes: &'static [SlotCause],
    /// Keyed-read structure, if the family has one on channel 0.
    pub read: Option<ReadSpec>,
    /// Output row indexes used as predict-ahead auxiliary state targets.
    pub aux_out_idxs: &'static [usize],
    /// Branch family: input index of the branch bit and the gated rows.
    pub branch_selected_out_idxs: &'static [usize],
}

/// Family spec table (documented from workload-families.ts).
pub fn family_spec(family: Family) -> FamilySpec {
    use SlotCause::{Branch, Carry, MemoryRead, Output, StackDepth, WrongFetch};
    match family {
        Family::ArithmeticCarry => FamilySpec {
            aux_out_idxs: &[2, 3],
            branch_selected_out_idxs: &[],
            causes: &[Output, Output, Carry, Carry, Output, WrongFetch],
            f: 4,
            read: None,
            s: 6,
        },
        Family::MemoryLoadStore | Family::NearMissLookup => FamilySpec {
            aux_out_idxs: &[3],
            branch_selected_out_idxs: &[],
            causes: &[WrongFetch, WrongFetch, WrongFetch, MemoryRead, Output],
            f: 3,
            read: Some(ReadSpec {
                query_input: 2,
                result_out_idx: 3,
                write_key: OperandSrc::Input(0),
                write_value: OperandSrc::Input(1),
            }),
            s: 5,
        },
        Family::BranchGatedControl => FamilySpec {
            aux_out_idxs: &[1, 2],
            branch_selected_out_idxs: &[1, 4, 5],
            causes: &[WrongFetch, Branch, Carry, Carry, Branch, Branch],
            f: 3,
            read: None,
            s: 6,
        },
        Family::ApplicationStateMachine => FamilySpec {
            aux_out_idxs: &[1],
            branch_selected_out_idxs: &[],
            causes: &[WrongFetch, MemoryRead, Output, Carry, WrongFetch],
            f: 2,
            read: Some(ReadSpec {
                query_input: 0,
                result_out_idx: 1,
                write_key: OperandSrc::Input(0),
                write_value: OperandSrc::Output(2),
            }),
            s: 5,
        },
        Family::StackLoopSum => FamilySpec {
            aux_out_idxs: &[],
            branch_selected_out_idxs: &[],
            causes: &[StackDepth, StackDepth, StackDepth, StackDepth, StackDepth],
            f: 1,
            read: None,
            s: 5,
        },
    }
}

/// One prepared record.
#[derive(Clone, Debug)]
pub struct StudentRecord {
    /// Program family.
    pub family: Family,
    /// Split assignment.
    pub split: Split,
    /// Executed steps.
    pub step_count: usize,
    /// Input fields per step.
    pub f: usize,
    /// Output slots per step.
    pub s: usize,
    /// Seed writes as (channel, key, value).
    pub seed_writes: Vec<(u32, i64, i64)>,
    /// Corpus record id.
    pub record_id: String,
    /// Graph digest of the executed model.
    pub program_hash: String,
    /// Full verified trace digest.
    pub full_trace_digest: String,
    /// Final output row digest.
    pub final_output_digest: String,
    /// Row-major step inputs (`step_count * f`).
    pub inputs: Vec<i64>,
    /// Row-major verified step outputs (`step_count * s`).
    pub outputs: Vec<i64>,
    /// Numeric model JSON (eval records only).
    pub model_json: Option<String>,
}

/// Parsed prep file: header identity plus records.
#[derive(Debug)]
pub struct PrepFile {
    /// Corpus id from the manifest.
    pub corpus_id: String,
    /// Snapshot digest pinning the dataset.
    pub snapshot_digest: String,
    /// Executor hash of the trace factory.
    pub executor_hash: String,
    /// Records in file order.
    pub records: Vec<StudentRecord>,
}

struct Reader<'bytes> {
    bytes: &'bytes [u8],
    offset: usize,
}

impl<'bytes> Reader<'bytes> {
    fn take(&mut self, len: usize, what: &'static str) -> Result<&'bytes [u8], PrepError> {
        if self.offset + len > self.bytes.len() {
            return Err(PrepError::Truncated(self.offset, what));
        }
        let slice = &self.bytes[self.offset..self.offset + len];
        self.offset += len;
        Ok(slice)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, PrepError> {
        Ok(self.take(1, what)?[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, PrepError> {
        let raw = self.take(2, what)?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, PrepError> {
        let raw = self.take(4, what)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn i64(&mut self, what: &'static str) -> Result<i64, PrepError> {
        let raw = self.take(8, what)?;
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(raw);
        Ok(i64::from_le_bytes(bytes))
    }

    fn string(&mut self, what: &'static str) -> Result<String, PrepError> {
        let len = self.u32(what)? as usize;
        let raw = self.take(len, what)?;
        String::from_utf8(raw.to_vec()).map_err(|_| PrepError::InvalidUtf8(what))
    }

    fn done(&self) -> bool {
        self.offset >= self.bytes.len()
    }
}

/// Parses one `student_prep.v0.1` byte stream.
pub fn parse_prep(bytes: &[u8]) -> Result<PrepFile, PrepError> {
    let mut reader = Reader { bytes, offset: 0 };
    let magic = reader.take(8, "magic")?;
    if magic != b"TSPREP1\0" {
        return Err(PrepError::BadMagic);
    }
    let version = reader.u32("version")?;
    if version != 1 {
        return Err(PrepError::UnsupportedVersion(version));
    }
    let corpus_id = reader.string("corpus_id")?;
    let snapshot_digest = reader.string("snapshot_digest")?;
    let executor_hash = reader.string("executor_hash")?;
    let mut records = Vec::new();
    while !reader.done() {
        let family_idx = reader.u8("family")?;
        let family = Family::from_index(family_idx).ok_or(PrepError::InvalidEnum {
            value: u32::from(family_idx),
            what: "family",
        })?;
        let split_idx = reader.u8("split")?;
        let split = Split::from_index(split_idx).ok_or(PrepError::InvalidEnum {
            value: u32::from(split_idx),
            what: "split",
        })?;
        let _reserved = reader.u16("reserved")?;
        let step_count = reader.u32("step_count")? as usize;
        let f = reader.u8("f")? as usize;
        let s = reader.u8("s")? as usize;
        let seed_write_count = reader.u16("seed_write_count")? as usize;
        let record_id = reader.string("record_id")?;
        let program_hash = reader.string("program_hash")?;
        let full_trace_digest = reader.string("full_trace_digest")?;
        let final_output_digest = reader.string("final_output_digest")?;
        let mut seed_writes = Vec::with_capacity(seed_write_count);
        for _ in 0..seed_write_count {
            let channel = reader.u32("seed_channel")?;
            let key = reader.i64("seed_key")?;
            let value = reader.i64("seed_value")?;
            seed_writes.push((channel, key, value));
        }
        let mut inputs = Vec::with_capacity(step_count * f);
        for _ in 0..step_count * f {
            inputs.push(reader.i64("input")?);
        }
        let mut outputs = Vec::with_capacity(step_count * s);
        for _ in 0..step_count * s {
            outputs.push(reader.i64("output")?);
        }
        let model_json = {
            let text = reader.string("model_json")?;
            if text.is_empty() { None } else { Some(text) }
        };
        records.push(StudentRecord {
            family,
            split,
            step_count,
            f,
            s,
            seed_writes,
            record_id,
            program_hash,
            full_trace_digest,
            final_output_digest,
            inputs,
            outputs,
            model_json,
        });
    }
    Ok(PrepFile {
        corpus_id,
        snapshot_digest,
        executor_hash,
        records,
    })
}

/// Role of one token in the student sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenRole {
    /// Seed-write preamble (key or value limb).
    Seed,
    /// Step input limb (teacher-forced environment input).
    Input,
    /// Step output limb (the verified corpus token; the prediction
    /// target and the only positions scored by first divergence).
    Output,
}

/// One materialized student sequence.
#[derive(Clone, Debug)]
pub struct StudentSequence {
    /// uint16 limb tokens.
    pub tokens: Vec<u16>,
    /// Role per token.
    pub roles: Vec<TokenRole>,
    /// Limb index (0-3) per token.
    pub limb: Vec<u8>,
    /// Value index within its step block (0..f+s), 0/1 for seed limbs.
    pub value_idx: Vec<u8>,
    /// Two scalar features per token: tanh of the most recently
    /// completed i64 value at scales 2^-20 and 2^-40.
    pub feats: Vec<[f32; 2]>,
    /// Step index per token (seed preamble = usize::MAX).
    pub step_of: Vec<usize>,
    /// Output row index per token (only valid at Output roles).
    pub out_idx_of: Vec<u8>,
    /// Token index where each step's block starts.
    pub step_starts: Vec<usize>,
    /// Sequence index of the first output limb of each step.
    pub first_output_pos: Vec<usize>,
}

/// Encodes one i64 as 4 uint16 LE limbs (trace_token.v0.1).
pub fn limbs_of(value: i64) -> [u16; LIMBS_PER_VALUE] {
    let unsigned = value as u64;
    [
        (unsigned & 0xffff) as u16,
        ((unsigned >> 16) & 0xffff) as u16,
        ((unsigned >> 32) & 0xffff) as u16,
        ((unsigned >> 48) & 0xffff) as u16,
    ]
}

/// Reassembles one i64 from 4 uint16 LE limbs.
pub fn value_of_limbs(limbs: &[u16]) -> i64 {
    let mut unsigned = 0_u64;
    for (index, limb) in limbs.iter().enumerate().take(LIMBS_PER_VALUE) {
        unsigned |= u64::from(*limb) << (16 * index);
    }
    unsigned as i64
}

fn scalar_feats(value: i64) -> [f32; 2] {
    let v = value as f64;
    [
        ((v / f64::from(1 << 20)).tanh()) as f32,
        ((v / (2_f64).powi(40)).tanh()) as f32,
    ]
}

/// Materializes the `student_seq.v0.1` token sequence for one record.
pub fn build_sequence(record: &StudentRecord) -> StudentSequence {
    let f = record.f;
    let s = record.s;
    let seed_tokens = record.seed_writes.len() * 2 * LIMBS_PER_VALUE;
    let step_tokens = (f + s) * LIMBS_PER_VALUE;
    let total = seed_tokens + record.step_count * step_tokens;
    let mut seq = StudentSequence {
        feats: Vec::with_capacity(total),
        first_output_pos: Vec::with_capacity(record.step_count),
        limb: Vec::with_capacity(total),
        out_idx_of: Vec::with_capacity(total),
        roles: Vec::with_capacity(total),
        step_of: Vec::with_capacity(total),
        step_starts: Vec::with_capacity(record.step_count),
        tokens: Vec::with_capacity(total),
        value_idx: Vec::with_capacity(total),
    };
    let mut last_completed: i64 = 0;
    let push_value = |seq: &mut StudentSequence,
                      value: i64,
                      role: TokenRole,
                      value_idx: u8,
                      step: usize,
                      out_idx: u8,
                      last_completed: &mut i64| {
        let limbs = limbs_of(value);
        for (limb_idx, limb) in limbs.iter().enumerate() {
            seq.tokens.push(*limb);
            seq.roles.push(role);
            seq.limb.push(limb_idx as u8);
            seq.value_idx.push(value_idx);
            seq.feats.push(scalar_feats(*last_completed));
            seq.step_of.push(step);
            seq.out_idx_of.push(out_idx);
        }
        *last_completed = value;
    };
    for (write_idx, (_, key, value)) in record.seed_writes.iter().enumerate() {
        let _ = write_idx;
        push_value(
            &mut seq,
            *key,
            TokenRole::Seed,
            0,
            usize::MAX,
            u8::MAX,
            &mut last_completed,
        );
        push_value(
            &mut seq,
            *value,
            TokenRole::Seed,
            1,
            usize::MAX,
            u8::MAX,
            &mut last_completed,
        );
    }
    for step in 0..record.step_count {
        seq.step_starts.push(seq.tokens.len());
        for field in 0..f {
            push_value(
                &mut seq,
                record.inputs[step * f + field],
                TokenRole::Input,
                field as u8,
                step,
                u8::MAX,
                &mut last_completed,
            );
        }
        seq.first_output_pos.push(seq.tokens.len());
        for out in 0..s {
            push_value(
                &mut seq,
                record.outputs[step * s + out],
                TokenRole::Output,
                (f + out) as u8,
                step,
                out as u8,
                &mut last_completed,
            );
        }
    }
    seq
}

/// One keyed-read lookup instance (channel 0) for baseline (c).
#[derive(Clone, Debug)]
pub struct LookupInstance {
    /// Step index of the read.
    pub step: usize,
    /// Sequence position of the first limb of the read-result value.
    pub result_pos: usize,
    /// Query key.
    pub query: i64,
    /// Candidate keys at read time (seeds then per-step writes).
    pub candidate_keys: Vec<i64>,
    /// Candidate values at read time.
    pub candidate_values: Vec<i64>,
    /// Index of the latest candidate whose key equals the query.
    pub correct: usize,
}

/// Extracts the channel-0 lookup instances of one record, mirroring the
/// executor's write ordering (seeds, then end-of-step writes).
pub fn lookup_instances(record: &StudentRecord, seq: &StudentSequence) -> Vec<LookupInstance> {
    let spec = family_spec(record.family);
    let Some(read) = spec.read else {
        return Vec::new();
    };
    let mut keys: Vec<i64> = Vec::new();
    let mut values: Vec<i64> = Vec::new();
    for (channel, key, value) in &record.seed_writes {
        if *channel == 0 {
            keys.push(*key);
            values.push(*value);
        }
    }
    let mut instances = Vec::with_capacity(record.step_count);
    for step in 0..record.step_count {
        let query = record.inputs[step * record.f + read.query_input];
        let correct = keys
            .iter()
            .rposition(|key| *key == query)
            .unwrap_or(usize::MAX);
        if correct != usize::MAX {
            instances.push(LookupInstance {
                candidate_keys: keys.clone(),
                candidate_values: values.clone(),
                correct,
                query,
                result_pos: seq.first_output_pos[step] + read.result_out_idx * LIMBS_PER_VALUE,
                step,
            });
        }
        let write_key = match read.write_key {
            OperandSrc::Input(index) => record.inputs[step * record.f + index],
            OperandSrc::Output(index) => record.outputs[step * record.s + index],
        };
        let write_value = match read.write_value {
            OperandSrc::Input(index) => record.inputs[step * record.f + index],
            OperandSrc::Output(index) => record.outputs[step * record.s + index],
        };
        keys.push(write_key);
        values.push(write_value);
    }
    instances
}

/// Hex string to raw bytes; returns None on malformed hex.
pub fn hex_bytes(hex_text: &str) -> Option<Vec<u8>> {
    hex::decode(hex_text).ok()
}

/// Renders bytes as lowercase hex.
pub fn to_hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn limb_roundtrip_matches_trace_token_v01() {
        for value in [
            0_i64,
            1,
            -1,
            65_535,
            65_536,
            -65_536,
            i64::MAX,
            i64::MIN,
            123_456_789_012,
        ] {
            let limbs = limbs_of(value);
            assert_eq!(value_of_limbs(&limbs), value);
        }
        // -1 must be all-ones limbs (two's complement LE), matching the
        // TS traceTokensFromStepOutputs encoding.
        assert_eq!(limbs_of(-1), [0xffff, 0xffff, 0xffff, 0xffff]);
        // 2^16 + 3 → limb0 = 3, limb1 = 1.
        assert_eq!(limbs_of(65_539), [3, 1, 0, 0]);
    }

    #[test]
    fn sequence_layout_counts_tokens() {
        let record = StudentRecord {
            f: 2,
            family: Family::ApplicationStateMachine,
            final_output_digest: String::new(),
            full_trace_digest: String::new(),
            inputs: vec![7, 2, 7, 4],
            model_json: None,
            outputs: vec![10, 20, 30, 40, 50, 11, 21, 31, 41, 51],
            program_hash: String::new(),
            record_id: String::from("trace_test"),
            s: 5,
            seed_writes: vec![(0, 7, 70)],
            split: Split::EvalHeldoutFamily,
            step_count: 2,
        };
        let seq = build_sequence(&record);
        assert_eq!(seq.tokens.len(), (2 + 2 * (2 + 5)) * LIMBS_PER_VALUE);
        assert_eq!(seq.step_starts.len(), 2);
        assert_eq!(seq.roles[0], TokenRole::Seed);
        let first_out = seq.first_output_pos[0];
        assert_eq!(seq.roles[first_out], TokenRole::Output);
        assert_eq!(seq.tokens[first_out], 10);
        let instances = lookup_instances(&record, &seq);
        assert_eq!(instances.len(), 2);
        assert_eq!(instances[0].correct, 0);
        assert_eq!(instances[1].candidate_keys.len(), 2);
        // Latest write wins for duplicate keys, like the executor.
        assert_eq!(instances[1].correct, 1);
    }
}
