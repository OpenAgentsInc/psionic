use std::collections::BTreeMap;

use psionic_runtime::{
    TassadarArithmeticOp, TassadarExecution, TassadarHaltReason, TassadarInstruction,
    TassadarProgram, TassadarTraceEvent, TassadarTraceStep,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{TokenId, TokenSequence, TokenVocabulary, TokenizerBoundary};

const PAD_TOKEN: &str = "<pad>";
const BOS_TOKEN: &str = "<bos>";
const EOS_TOKEN: &str = "<eos>";
const UNKNOWN_TOKEN: &str = "<unk>";

const PROGRAM_TOKEN: &str = "<program>";
const TRACE_TOKEN: &str = "<trace>";
const HALT_TOKEN: &str = "<halt>";
const STEP_TOKEN: &str = "<step>";
const LIST_TOKEN: &str = "<list>";

const FIELD_LOCALS: &str = "<locals>";
const FIELD_MEMORY_SLOTS: &str = "<memory_slots>";
const FIELD_INITIAL_MEMORY: &str = "<initial_memory>";
const FIELD_STEP_INDEX: &str = "<step_index>";
const FIELD_PC: &str = "<pc>";
const FIELD_NEXT_PC: &str = "<next_pc>";
const FIELD_STACK_BEFORE: &str = "<stack_before>";
const FIELD_STACK_AFTER: &str = "<stack_after>";
const FIELD_LOCALS_AFTER: &str = "<locals_after>";
const FIELD_MEMORY_AFTER: &str = "<memory_after>";

const BOOL_FALSE: &str = "<bool_false>";
const BOOL_TRUE: &str = "<bool_true>";

const OP_I32_CONST: &str = "<op_i32_const>";
const OP_LOCAL_GET: &str = "<op_local_get>";
const OP_LOCAL_SET: &str = "<op_local_set>";
const OP_I32_ADD: &str = "<op_i32_add>";
const OP_I32_SUB: &str = "<op_i32_sub>";
const OP_I32_MUL: &str = "<op_i32_mul>";
const OP_I32_LT: &str = "<op_i32_lt>";
const OP_I32_LOAD: &str = "<op_i32_load>";
const OP_I32_STORE: &str = "<op_i32_store>";
const OP_BR_IF: &str = "<op_br_if>";
const OP_OUTPUT: &str = "<op_output>";
const OP_RETURN: &str = "<op_return>";

const EVENT_CONST_PUSH: &str = "<event_const_push>";
const EVENT_LOCAL_GET: &str = "<event_local_get>";
const EVENT_LOCAL_SET: &str = "<event_local_set>";
const EVENT_BINARY_ADD: &str = "<event_binary_add>";
const EVENT_BINARY_SUB: &str = "<event_binary_sub>";
const EVENT_BINARY_MUL: &str = "<event_binary_mul>";
const EVENT_BINARY_LT: &str = "<event_binary_lt>";
const EVENT_LOAD: &str = "<event_load>";
const EVENT_STORE: &str = "<event_store>";
const EVENT_BRANCH: &str = "<event_branch>";
const EVENT_OUTPUT: &str = "<event_output>";
const EVENT_RETURN: &str = "<event_return>";

const HALT_RETURNED: &str = "<halt_returned>";
const HALT_FELL_OFF_END: &str = "<halt_fell_off_end>";

/// Explicit symbolic target family for one tokenized Tassadar sequence dataset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarSequenceTraceFamily {
    /// Canonical CPU-style append-only instruction trace.
    #[default]
    SequentialCpuReference,
    /// Research-only anti-diagonal Sudoku assignment wavefront.
    SudokuDiagonalWavefront,
    /// Research-only parallel Hungarian assignment frontier.
    HungarianAssignmentFrontier,
}

impl TassadarSequenceTraceFamily {
    /// Returns the stable trace-family label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SequentialCpuReference => "sequential_cpu_reference",
            Self::SudokuDiagonalWavefront => "sudoku_diagonal_wavefront",
            Self::HungarianAssignmentFrontier => "hungarian_assignment_frontier",
        }
    }

    /// Returns the stable dataset-suffix label for non-canonical families.
    #[must_use]
    pub const fn dataset_suffix(self) -> Option<&'static str> {
        match self {
            Self::SequentialCpuReference => None,
            Self::SudokuDiagonalWavefront => Some("sudoku_diagonal_wavefront"),
            Self::HungarianAssignmentFrontier => Some("hungarian_assignment_frontier"),
        }
    }

    /// Returns the honest reconstruction scope for the trace family.
    #[must_use]
    pub const fn reconstruction_scope(self) -> &'static str {
        match self {
            Self::SequentialCpuReference => "full_cpu_trace",
            Self::SudokuDiagonalWavefront | Self::HungarianAssignmentFrontier => {
                "final_outputs_only"
            }
        }
    }
}

/// Structural supervision family derived from the canonical Tassadar trace ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarStructuralSupervisionFamily {
    /// Instruction-pointer state such as `pc` and `next_pc`.
    InstructionPointer,
    /// Conditional-branch state such as taken/not-taken outcomes.
    BranchOutcome,
    /// Stack-shape state that exposes stack-size deltas across steps.
    StackDelta,
    /// Memory-write state that exposes sparse memory diffs.
    MemoryDiff,
    /// Workload-specific structured state such as Hungarian dual variables.
    WorkloadSpecificState,
}

impl TassadarStructuralSupervisionFamily {
    /// Returns the stable family label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::InstructionPointer => "instruction_pointer",
            Self::BranchOutcome => "branch_outcome",
            Self::StackDelta => "stack_delta",
            Self::MemoryDiff => "memory_diff",
            Self::WorkloadSpecificState => "workload_specific_state",
        }
    }
}

/// Aggregate target-token coverage for one structural supervision family set.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarStructuralSupervisionCoverage {
    /// Total target tokens observed in the covered slice.
    pub total_target_token_count: u32,
    /// Target tokens tagged as instruction-pointer state.
    pub instruction_pointer_token_count: u32,
    /// Target tokens tagged as branch-outcome state.
    pub branch_outcome_token_count: u32,
    /// Target tokens tagged as stack-delta state.
    pub stack_delta_token_count: u32,
    /// Target tokens tagged as memory-diff state.
    pub memory_diff_token_count: u32,
    /// Target tokens tagged as workload-specific state.
    pub workload_specific_state_token_count: u32,
}

impl TassadarStructuralSupervisionCoverage {
    fn increment(&mut self, family: TassadarStructuralSupervisionFamily) {
        match family {
            TassadarStructuralSupervisionFamily::InstructionPointer => {
                self.instruction_pointer_token_count =
                    self.instruction_pointer_token_count.saturating_add(1);
            }
            TassadarStructuralSupervisionFamily::BranchOutcome => {
                self.branch_outcome_token_count = self.branch_outcome_token_count.saturating_add(1);
            }
            TassadarStructuralSupervisionFamily::StackDelta => {
                self.stack_delta_token_count = self.stack_delta_token_count.saturating_add(1);
            }
            TassadarStructuralSupervisionFamily::MemoryDiff => {
                self.memory_diff_token_count = self.memory_diff_token_count.saturating_add(1);
            }
            TassadarStructuralSupervisionFamily::WorkloadSpecificState => {
                self.workload_specific_state_token_count =
                    self.workload_specific_state_token_count.saturating_add(1);
            }
        }
    }

    /// Adds another coverage summary into `self`.
    pub fn accumulate(&mut self, other: &Self) {
        self.total_target_token_count = self
            .total_target_token_count
            .saturating_add(other.total_target_token_count);
        self.instruction_pointer_token_count = self
            .instruction_pointer_token_count
            .saturating_add(other.instruction_pointer_token_count);
        self.branch_outcome_token_count = self
            .branch_outcome_token_count
            .saturating_add(other.branch_outcome_token_count);
        self.stack_delta_token_count = self
            .stack_delta_token_count
            .saturating_add(other.stack_delta_token_count);
        self.memory_diff_token_count = self
            .memory_diff_token_count
            .saturating_add(other.memory_diff_token_count);
        self.workload_specific_state_token_count = self
            .workload_specific_state_token_count
            .saturating_add(other.workload_specific_state_token_count);
    }
}

/// Tokenized executor example with explicit prompt/target boundaries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarTokenizedExecutionSequence {
    /// Full ordered token sequence.
    pub sequence: TokenSequence,
    /// Prefix tokens that belong to the program prompt.
    pub prompt_token_count: usize,
    /// Tokens belonging to the predicted trace suffix.
    pub target_token_count: usize,
    /// Stable digest over the token ids.
    pub sequence_digest: String,
}

impl TassadarTokenizedExecutionSequence {
    fn new(sequence: TokenSequence, prompt_token_count: usize) -> Self {
        let target_token_count = sequence.len().saturating_sub(prompt_token_count);
        let sequence_digest = stable_digest(
            b"psionic_tassadar_tokenized_execution_sequence|",
            &sequence
                .as_slice()
                .iter()
                .map(|token| token.as_u32())
                .collect::<Vec<_>>(),
        );
        Self {
            sequence,
            prompt_token_count,
            target_token_count,
            sequence_digest,
        }
    }

    /// Returns token ids as raw `u32` values for data-package surfaces.
    #[must_use]
    pub fn token_ids_u32(&self) -> Vec<u32> {
        self.sequence
            .as_slice()
            .iter()
            .map(|token| token.as_u32())
            .collect()
    }
}

/// Symbolic decode of one tokenized execution sequence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarDecodedSymbolicSequence {
    /// Ordered symbolic token strings.
    pub tokens: Vec<String>,
    /// Prompt/target boundary.
    pub prompt_token_count: usize,
}

/// Deterministic byte-and-symbol tokenizer for Wasm program plus trace sequences.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TassadarTraceTokenizer {
    vocabulary: TokenVocabulary,
    lookup: BTreeMap<String, TokenId>,
    byte_token_start: u32,
}

impl Default for TassadarTraceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TassadarTraceTokenizer {
    /// Creates the canonical token vocabulary for the current Tassadar trace lane.
    #[must_use]
    pub fn new() -> Self {
        let mut tokens = vec![
            String::from(PAD_TOKEN),
            String::from(BOS_TOKEN),
            String::from(EOS_TOKEN),
            String::from(UNKNOWN_TOKEN),
            String::from(PROGRAM_TOKEN),
            String::from(TRACE_TOKEN),
            String::from(HALT_TOKEN),
            String::from(STEP_TOKEN),
            String::from(LIST_TOKEN),
            String::from(FIELD_LOCALS),
            String::from(FIELD_MEMORY_SLOTS),
            String::from(FIELD_INITIAL_MEMORY),
            String::from(FIELD_STEP_INDEX),
            String::from(FIELD_PC),
            String::from(FIELD_NEXT_PC),
            String::from(FIELD_STACK_BEFORE),
            String::from(FIELD_STACK_AFTER),
            String::from(FIELD_LOCALS_AFTER),
            String::from(FIELD_MEMORY_AFTER),
            String::from(BOOL_FALSE),
            String::from(BOOL_TRUE),
            String::from(OP_I32_CONST),
            String::from(OP_LOCAL_GET),
            String::from(OP_LOCAL_SET),
            String::from(OP_I32_ADD),
            String::from(OP_I32_SUB),
            String::from(OP_I32_MUL),
            String::from(OP_I32_LT),
            String::from(OP_I32_LOAD),
            String::from(OP_I32_STORE),
            String::from(OP_BR_IF),
            String::from(OP_OUTPUT),
            String::from(OP_RETURN),
            String::from(EVENT_CONST_PUSH),
            String::from(EVENT_LOCAL_GET),
            String::from(EVENT_LOCAL_SET),
            String::from(EVENT_BINARY_ADD),
            String::from(EVENT_BINARY_SUB),
            String::from(EVENT_BINARY_MUL),
            String::from(EVENT_BINARY_LT),
            String::from(EVENT_LOAD),
            String::from(EVENT_STORE),
            String::from(EVENT_BRANCH),
            String::from(EVENT_OUTPUT),
            String::from(EVENT_RETURN),
            String::from(HALT_RETURNED),
            String::from(HALT_FELL_OFF_END),
        ];
        let byte_token_start = tokens.len() as u32;
        for value in 0_u16..=255 {
            tokens.push(format!("<byte_{value:02x}>"));
        }

        let vocabulary = TokenVocabulary::new(
            tokens.clone(),
            TokenId(0),
            TokenId(1),
            TokenId(2),
            TokenId(3),
        );
        let lookup = tokens
            .into_iter()
            .enumerate()
            .map(|(index, token)| (token, TokenId(index as u32)))
            .collect();
        Self {
            vocabulary,
            lookup,
            byte_token_start,
        }
    }

    /// Returns a stable digest over the tokenizer contract.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(
            b"psionic_tassadar_trace_tokenizer|",
            &self.vocabulary.tokens(),
        )
    }

    /// Tokenizes one validated program and CPU-reference execution trace.
    #[must_use]
    pub fn tokenize_program_and_execution(
        &self,
        program: &TassadarProgram,
        execution: &TassadarExecution,
    ) -> TassadarTokenizedExecutionSequence {
        let mut tokens = Vec::new();
        let prompt_token_count = self.push_program_prompt(&mut tokens, program);
        for step in &execution.steps {
            self.push_step(&mut tokens, step);
        }
        tokens.push(self.token_id(HALT_TOKEN));
        self.push_halt_reason(&mut tokens, execution.halt_reason);
        tokens.push(self.vocabulary.eos_id());
        TassadarTokenizedExecutionSequence::new(TokenSequence::new(tokens), prompt_token_count)
    }

    /// Tokenizes one Sudoku solution as an anti-diagonal assignment wavefront.
    #[must_use]
    pub fn tokenize_program_and_sudoku_diagonal_wavefront(
        &self,
        program: &TassadarProgram,
        solved_outputs: &[i32],
        grid_width: usize,
    ) -> TassadarTokenizedExecutionSequence {
        let mut tokens = Vec::new();
        let prompt_token_count = self.push_program_prompt(&mut tokens, program);
        let mut emitted_count = 0_u32;
        for wavefront_index in 0..grid_width.saturating_mul(2).saturating_sub(1) {
            let mut payload = Vec::new();
            for row in 0..grid_width {
                let Some(col) = wavefront_index.checked_sub(row) else {
                    continue;
                };
                if col >= grid_width {
                    continue;
                }
                let cell_index = row.saturating_mul(grid_width).saturating_add(col);
                let Some(value) = solved_outputs.get(cell_index) else {
                    continue;
                };
                payload.push(cell_index as i32);
                payload.push(*value);
            }
            emitted_count = emitted_count.saturating_add((payload.len() / 2) as u32);
            self.push_wavefront_step(
                &mut tokens,
                wavefront_index as u32,
                (payload.len() / 2) as u32,
                emitted_count,
                payload.as_slice(),
                emitted_count as i32,
            );
        }
        tokens.push(self.token_id(HALT_TOKEN));
        self.push_halt_reason(&mut tokens, TassadarHaltReason::Returned);
        tokens.push(self.vocabulary.eos_id());
        TassadarTokenizedExecutionSequence::new(TokenSequence::new(tokens), prompt_token_count)
    }

    /// Tokenizes one Hungarian optimum as a parallel assignment frontier.
    #[must_use]
    pub fn tokenize_program_and_hungarian_assignment_frontier(
        &self,
        program: &TassadarProgram,
        optimal_assignment: &[i32],
        cost_matrix: &[i32],
        optimal_cost: i32,
    ) -> TassadarTokenizedExecutionSequence {
        let mut tokens = Vec::new();
        let prompt_token_count = self.push_program_prompt(&mut tokens, program);
        let dimension = optimal_assignment.len();
        let mut payload = Vec::with_capacity(dimension.saturating_mul(3));
        for (row, column) in optimal_assignment.iter().enumerate() {
            let column = *column as usize;
            let row_cost = cost_matrix
                .get(row.saturating_mul(dimension).saturating_add(column))
                .copied()
                .unwrap_or_default();
            payload.push(row as i32);
            payload.push(column as i32);
            payload.push(row_cost);
        }
        self.push_wavefront_step(
            &mut tokens,
            0,
            dimension as u32,
            optimal_cost.max(0) as u32,
            payload.as_slice(),
            optimal_cost,
        );
        tokens.push(self.token_id(HALT_TOKEN));
        self.push_halt_reason(&mut tokens, TassadarHaltReason::Returned);
        tokens.push(self.vocabulary.eos_id());
        TassadarTokenizedExecutionSequence::new(TokenSequence::new(tokens), prompt_token_count)
    }

    /// Decodes token ids back into symbolic token strings with the same prompt boundary.
    #[must_use]
    pub fn decode_symbolic(
        &self,
        tokenized: &TassadarTokenizedExecutionSequence,
    ) -> TassadarDecodedSymbolicSequence {
        TassadarDecodedSymbolicSequence {
            tokens: tokenized
                .sequence
                .as_slice()
                .iter()
                .map(|token| {
                    self.vocabulary
                        .token(*token)
                        .unwrap_or(UNKNOWN_TOKEN)
                        .to_string()
                })
                .collect(),
            prompt_token_count: tokenized.prompt_token_count,
        }
    }

    /// Extracts output values from one symbolic executor trace.
    #[must_use]
    pub fn extract_output_values(&self, tokens: &[TokenId]) -> Vec<i32> {
        let mut outputs = Vec::new();
        let output_token = self.token_id(EVENT_OUTPUT);
        let mut index = 0_usize;
        while index < tokens.len() {
            if tokens[index] == output_token {
                if let Some(value) = self.parse_i32(tokens, index + 1) {
                    outputs.push(value);
                }
                index = index.saturating_add(5);
                continue;
            }
            index = index.saturating_add(1);
        }
        outputs
    }

    /// Extracts the final halt marker from one symbolic executor trace.
    #[must_use]
    pub fn extract_halt_marker(&self, tokens: &[TokenId]) -> Option<String> {
        let halt_token = self.token_id(HALT_TOKEN);
        for window in tokens.windows(2) {
            if window[0] == halt_token {
                return self
                    .vocabulary
                    .token(window[1])
                    .map(std::string::ToString::to_string);
            }
        }
        None
    }

    /// Reconstructs final Sudoku outputs from one anti-diagonal wavefront trace.
    #[must_use]
    pub fn extract_sudoku_diagonal_wavefront_outputs(
        &self,
        tokens: &[TokenId],
        prompt_token_count: usize,
        cell_count: usize,
    ) -> Vec<i32> {
        let mut outputs = vec![0; cell_count];
        let mut index = prompt_token_count;
        while let Some(step) = self.parse_wavefront_step(tokens, index) {
            if step.stack_after.len() % 2 == 0 {
                for pair in step.stack_after.chunks_exact(2) {
                    let cell_index = pair[0].max(0) as usize;
                    if cell_index < outputs.len() {
                        outputs[cell_index] = pair[1];
                    }
                }
            }
            index = step.next_index;
        }
        outputs
    }

    /// Reconstructs final Hungarian outputs from one parallel assignment frontier trace.
    #[must_use]
    pub fn extract_hungarian_assignment_frontier_outputs(
        &self,
        tokens: &[TokenId],
        prompt_token_count: usize,
        dimension: usize,
    ) -> Vec<i32> {
        let mut assignments = vec![0; dimension];
        let mut total_cost = 0_i32;
        let mut index = prompt_token_count;
        while let Some(step) = self.parse_wavefront_step(tokens, index) {
            if let Some(summary_output) = step.summary_output {
                total_cost = summary_output;
            }
            if step.stack_after.len() % 3 == 0 {
                for triple in step.stack_after.chunks_exact(3) {
                    let row = triple[0].max(0) as usize;
                    if row < assignments.len() {
                        assignments[row] = triple[1];
                    }
                }
            }
            index = step.next_index;
        }
        assignments.push(total_cost);
        assignments
    }

    /// Classifies each target token into zero or more structural supervision families.
    #[must_use]
    pub fn classify_target_structural_supervision(
        &self,
        tokens: &[TokenId],
        prompt_token_count: usize,
    ) -> Vec<Vec<TassadarStructuralSupervisionFamily>> {
        let mut families = vec![Vec::new(); tokens.len().saturating_sub(prompt_token_count)];
        let mut index = prompt_token_count;
        let halt_token = self.token_id(HALT_TOKEN);
        let eos_token = self.vocabulary.eos_id();
        let step_token = self.token_id(STEP_TOKEN);
        while index < tokens.len() {
            let token = tokens[index];
            if token == halt_token || token == eos_token {
                break;
            }
            if token != step_token {
                index = index.saturating_add(1);
                continue;
            }
            index = index.saturating_add(1);
            index = self.consume_field_u32(
                tokens,
                prompt_token_count,
                index,
                FIELD_STEP_INDEX,
                None,
                &mut families,
            );
            index = self.consume_field_u32(
                tokens,
                prompt_token_count,
                index,
                FIELD_PC,
                Some(TassadarStructuralSupervisionFamily::InstructionPointer),
                &mut families,
            );
            index = self.consume_field_u32(
                tokens,
                prompt_token_count,
                index,
                FIELD_NEXT_PC,
                Some(TassadarStructuralSupervisionFamily::InstructionPointer),
                &mut families,
            );
            index = self.consume_instruction(tokens, index);
            let memory_write_slot =
                self.consume_event(tokens, prompt_token_count, index, &mut families);
            index = memory_write_slot.0;
            index = self.consume_stack_list(
                tokens,
                prompt_token_count,
                index,
                FIELD_STACK_BEFORE,
                &mut families,
            );
            index = self.consume_stack_list(
                tokens,
                prompt_token_count,
                index,
                FIELD_STACK_AFTER,
                &mut families,
            );
            index = self.consume_i32_list(tokens, index, FIELD_LOCALS_AFTER);
            index = self.consume_memory_after_list(
                tokens,
                prompt_token_count,
                index,
                memory_write_slot.1,
                &mut families,
            );
        }
        families
    }

    /// Summarizes structural-supervision coverage for one token sequence.
    #[must_use]
    pub fn summarize_target_structural_supervision(
        &self,
        tokens: &[TokenId],
        prompt_token_count: usize,
    ) -> TassadarStructuralSupervisionCoverage {
        let families = self.classify_target_structural_supervision(tokens, prompt_token_count);
        let mut coverage = TassadarStructuralSupervisionCoverage {
            total_target_token_count: families.len() as u32,
            ..TassadarStructuralSupervisionCoverage::default()
        };
        for token_families in families {
            for family in token_families {
                coverage.increment(family);
            }
        }
        coverage
    }

    fn push_step(&self, tokens: &mut Vec<TokenId>, step: &TassadarTraceStep) {
        tokens.push(self.token_id(STEP_TOKEN));
        tokens.push(self.token_id(FIELD_STEP_INDEX));
        self.push_u32(tokens, step.step_index as u32);
        tokens.push(self.token_id(FIELD_PC));
        self.push_u32(tokens, step.pc as u32);
        tokens.push(self.token_id(FIELD_NEXT_PC));
        self.push_u32(tokens, step.next_pc as u32);
        self.push_instruction(tokens, &step.instruction);
        self.push_event(tokens, &step.event);
        self.push_i32_list(tokens, FIELD_STACK_BEFORE, &step.stack_before);
        self.push_i32_list(tokens, FIELD_STACK_AFTER, &step.stack_after);
        self.push_i32_list(tokens, FIELD_LOCALS_AFTER, &step.locals_after);
        self.push_i32_list(tokens, FIELD_MEMORY_AFTER, &step.memory_after);
    }

    fn push_program_prompt(&self, tokens: &mut Vec<TokenId>, program: &TassadarProgram) -> usize {
        tokens.push(self.vocabulary.bos_id());
        tokens.push(self.token_id(PROGRAM_TOKEN));
        tokens.push(self.token_id(FIELD_LOCALS));
        self.push_u32(tokens, program.local_count as u32);
        tokens.push(self.token_id(FIELD_MEMORY_SLOTS));
        self.push_u32(tokens, program.memory_slots as u32);
        tokens.push(self.token_id(FIELD_INITIAL_MEMORY));
        self.push_u32(tokens, program.initial_memory.len() as u32);
        for value in &program.initial_memory {
            self.push_i32(tokens, *value);
        }
        for instruction in &program.instructions {
            self.push_instruction(tokens, instruction);
        }
        tokens.push(self.token_id(TRACE_TOKEN));
        tokens.len()
    }

    fn push_wavefront_step(
        &self,
        tokens: &mut Vec<TokenId>,
        step_index: u32,
        payload_item_count: u32,
        summary_index: u32,
        payload: &[i32],
        summary_output: i32,
    ) {
        tokens.push(self.token_id(STEP_TOKEN));
        tokens.push(self.token_id(FIELD_STEP_INDEX));
        self.push_u32(tokens, step_index);
        tokens.push(self.token_id(FIELD_PC));
        self.push_u32(tokens, payload_item_count);
        tokens.push(self.token_id(FIELD_NEXT_PC));
        self.push_u32(tokens, summary_index);
        tokens.push(self.token_id(OP_OUTPUT));
        tokens.push(self.token_id(EVENT_OUTPUT));
        self.push_i32(tokens, summary_output);
        self.push_i32_list(tokens, FIELD_STACK_BEFORE, &[]);
        self.push_i32_list(tokens, FIELD_STACK_AFTER, payload);
        self.push_i32_list(tokens, FIELD_LOCALS_AFTER, &[]);
        self.push_i32_list(tokens, FIELD_MEMORY_AFTER, &[]);
    }

    fn push_instruction(&self, tokens: &mut Vec<TokenId>, instruction: &TassadarInstruction) {
        match instruction {
            TassadarInstruction::I32Const { value } => {
                tokens.push(self.token_id(OP_I32_CONST));
                self.push_i32(tokens, *value);
            }
            TassadarInstruction::LocalGet { local } => {
                tokens.push(self.token_id(OP_LOCAL_GET));
                self.push_u32(tokens, u32::from(*local));
            }
            TassadarInstruction::LocalSet { local } => {
                tokens.push(self.token_id(OP_LOCAL_SET));
                self.push_u32(tokens, u32::from(*local));
            }
            TassadarInstruction::I32Add => tokens.push(self.token_id(OP_I32_ADD)),
            TassadarInstruction::I32Sub => tokens.push(self.token_id(OP_I32_SUB)),
            TassadarInstruction::I32Mul => tokens.push(self.token_id(OP_I32_MUL)),
            TassadarInstruction::I32Lt => tokens.push(self.token_id(OP_I32_LT)),
            TassadarInstruction::I32Load { slot } => {
                tokens.push(self.token_id(OP_I32_LOAD));
                self.push_u32(tokens, u32::from(*slot));
            }
            TassadarInstruction::I32Store { slot } => {
                tokens.push(self.token_id(OP_I32_STORE));
                self.push_u32(tokens, u32::from(*slot));
            }
            TassadarInstruction::BrIf { target_pc } => {
                tokens.push(self.token_id(OP_BR_IF));
                self.push_u32(tokens, u32::from(*target_pc));
            }
            TassadarInstruction::Output => tokens.push(self.token_id(OP_OUTPUT)),
            TassadarInstruction::Return => tokens.push(self.token_id(OP_RETURN)),
        }
    }

    fn push_event(&self, tokens: &mut Vec<TokenId>, event: &TassadarTraceEvent) {
        match event {
            TassadarTraceEvent::ConstPush { value } => {
                tokens.push(self.token_id(EVENT_CONST_PUSH));
                self.push_i32(tokens, *value);
            }
            TassadarTraceEvent::LocalGet { local, value } => {
                tokens.push(self.token_id(EVENT_LOCAL_GET));
                self.push_u32(tokens, u32::from(*local));
                self.push_i32(tokens, *value);
            }
            TassadarTraceEvent::LocalSet { local, value } => {
                tokens.push(self.token_id(EVENT_LOCAL_SET));
                self.push_u32(tokens, u32::from(*local));
                self.push_i32(tokens, *value);
            }
            TassadarTraceEvent::BinaryOp {
                op,
                left,
                right,
                result,
            } => {
                tokens.push(self.binary_event_token_id(*op));
                self.push_i32(tokens, *left);
                self.push_i32(tokens, *right);
                self.push_i32(tokens, *result);
            }
            TassadarTraceEvent::Load { slot, value } => {
                tokens.push(self.token_id(EVENT_LOAD));
                self.push_u32(tokens, u32::from(*slot));
                self.push_i32(tokens, *value);
            }
            TassadarTraceEvent::Store { slot, value } => {
                tokens.push(self.token_id(EVENT_STORE));
                self.push_u32(tokens, u32::from(*slot));
                self.push_i32(tokens, *value);
            }
            TassadarTraceEvent::Branch {
                condition,
                taken,
                target_pc,
            } => {
                tokens.push(self.token_id(EVENT_BRANCH));
                self.push_i32(tokens, *condition);
                tokens.push(if *taken {
                    self.token_id(BOOL_TRUE)
                } else {
                    self.token_id(BOOL_FALSE)
                });
                self.push_u32(tokens, *target_pc as u32);
            }
            TassadarTraceEvent::Output { value } => {
                tokens.push(self.token_id(EVENT_OUTPUT));
                self.push_i32(tokens, *value);
            }
            TassadarTraceEvent::Return => tokens.push(self.token_id(EVENT_RETURN)),
        }
    }

    fn push_halt_reason(&self, tokens: &mut Vec<TokenId>, halt_reason: TassadarHaltReason) {
        tokens.push(match halt_reason {
            TassadarHaltReason::Returned => self.token_id(HALT_RETURNED),
            TassadarHaltReason::FellOffEnd => self.token_id(HALT_FELL_OFF_END),
        });
    }

    fn push_i32_list(&self, tokens: &mut Vec<TokenId>, field_token: &str, values: &[i32]) {
        tokens.push(self.token_id(field_token));
        tokens.push(self.token_id(LIST_TOKEN));
        self.push_u32(tokens, values.len() as u32);
        for value in values {
            self.push_i32(tokens, *value);
        }
    }

    fn push_u32(&self, tokens: &mut Vec<TokenId>, value: u32) {
        for byte in value.to_le_bytes() {
            tokens.push(self.byte_token(byte));
        }
    }

    fn push_i32(&self, tokens: &mut Vec<TokenId>, value: i32) {
        for byte in value.to_le_bytes() {
            tokens.push(self.byte_token(byte));
        }
    }

    fn token_id(&self, token: &str) -> TokenId {
        *self
            .lookup
            .get(token)
            .expect("Tassadar trace token should exist")
    }

    fn byte_token(&self, value: u8) -> TokenId {
        TokenId(self.byte_token_start + u32::from(value))
    }

    fn binary_event_token_id(&self, op: TassadarArithmeticOp) -> TokenId {
        match op {
            TassadarArithmeticOp::Add => self.token_id(EVENT_BINARY_ADD),
            TassadarArithmeticOp::Sub => self.token_id(EVENT_BINARY_SUB),
            TassadarArithmeticOp::Mul => self.token_id(EVENT_BINARY_MUL),
            TassadarArithmeticOp::Lt => self.token_id(EVENT_BINARY_LT),
        }
    }

    fn parse_i32(&self, tokens: &[TokenId], start: usize) -> Option<i32> {
        let bytes = [
            self.byte_value(*tokens.get(start)?)?,
            self.byte_value(*tokens.get(start + 1)?)?,
            self.byte_value(*tokens.get(start + 2)?)?,
            self.byte_value(*tokens.get(start + 3)?)?,
        ];
        Some(i32::from_le_bytes(bytes))
    }

    fn byte_value(&self, token: TokenId) -> Option<u8> {
        let raw = token.as_u32();
        let end = self.byte_token_start + 256;
        if raw < self.byte_token_start || raw >= end {
            return None;
        }
        Some((raw - self.byte_token_start) as u8)
    }

    fn consume_field_u32(
        &self,
        tokens: &[TokenId],
        prompt_token_count: usize,
        index: usize,
        field_token: &str,
        family: Option<TassadarStructuralSupervisionFamily>,
        families: &mut [Vec<TassadarStructuralSupervisionFamily>],
    ) -> usize {
        if index >= tokens.len() || tokens[index] != self.token_id(field_token) {
            return index.saturating_add(1);
        }
        if let Some(family) = family {
            self.mark_target_range(prompt_token_count, index, 5, family, families);
        }
        index.saturating_add(5).min(tokens.len())
    }

    fn consume_instruction(&self, tokens: &[TokenId], index: usize) -> usize {
        if index >= tokens.len() {
            return index;
        }
        let token = tokens[index];
        let with_immediate = [
            self.token_id(OP_I32_CONST),
            self.token_id(OP_LOCAL_GET),
            self.token_id(OP_LOCAL_SET),
            self.token_id(OP_I32_LOAD),
            self.token_id(OP_I32_STORE),
            self.token_id(OP_BR_IF),
        ];
        if with_immediate.contains(&token) {
            index.saturating_add(5).min(tokens.len())
        } else {
            index.saturating_add(1).min(tokens.len())
        }
    }

    fn consume_event(
        &self,
        tokens: &[TokenId],
        prompt_token_count: usize,
        index: usize,
        families: &mut [Vec<TassadarStructuralSupervisionFamily>],
    ) -> (usize, Option<usize>) {
        if index >= tokens.len() {
            return (index, None);
        }
        let token = tokens[index];
        if token == self.token_id(EVENT_CONST_PUSH) {
            return (index.saturating_add(5).min(tokens.len()), None);
        }
        if token == self.token_id(EVENT_LOCAL_GET) || token == self.token_id(EVENT_LOCAL_SET) {
            return (index.saturating_add(9).min(tokens.len()), None);
        }
        if [
            self.token_id(EVENT_BINARY_ADD),
            self.token_id(EVENT_BINARY_SUB),
            self.token_id(EVENT_BINARY_MUL),
            self.token_id(EVENT_BINARY_LT),
        ]
        .contains(&token)
        {
            return (index.saturating_add(13).min(tokens.len()), None);
        }
        if token == self.token_id(EVENT_LOAD) {
            return (index.saturating_add(9).min(tokens.len()), None);
        }
        if token == self.token_id(EVENT_STORE) {
            self.mark_target_range(
                prompt_token_count,
                index,
                9,
                TassadarStructuralSupervisionFamily::MemoryDiff,
                families,
            );
            let slot = self
                .parse_u32(tokens, index.saturating_add(1))
                .map(|value| value as usize);
            return (index.saturating_add(9).min(tokens.len()), slot);
        }
        if token == self.token_id(EVENT_BRANCH) {
            self.mark_target_range(
                prompt_token_count,
                index,
                10,
                TassadarStructuralSupervisionFamily::BranchOutcome,
                families,
            );
            return (index.saturating_add(10).min(tokens.len()), None);
        }
        if token == self.token_id(EVENT_OUTPUT) {
            return (index.saturating_add(5).min(tokens.len()), None);
        }
        if token == self.token_id(EVENT_RETURN) {
            return (index.saturating_add(1).min(tokens.len()), None);
        }
        (index.saturating_add(1).min(tokens.len()), None)
    }

    fn consume_stack_list(
        &self,
        tokens: &[TokenId],
        prompt_token_count: usize,
        index: usize,
        field_token: &str,
        families: &mut [Vec<TassadarStructuralSupervisionFamily>],
    ) -> usize {
        if index.saturating_add(6) > tokens.len() || tokens[index] != self.token_id(field_token) {
            return index.saturating_add(1);
        }
        self.mark_target_range(
            prompt_token_count,
            index,
            6,
            TassadarStructuralSupervisionFamily::StackDelta,
            families,
        );
        self.consume_i32_list(tokens, index, field_token)
    }

    fn consume_i32_list(&self, tokens: &[TokenId], index: usize, field_token: &str) -> usize {
        if index.saturating_add(6) > tokens.len() || tokens[index] != self.token_id(field_token) {
            return index.saturating_add(1);
        }
        if tokens[index + 1] != self.token_id(LIST_TOKEN) {
            return index.saturating_add(1);
        }
        let value_count = self.parse_u32(tokens, index + 2).unwrap_or(0) as usize;
        index
            .saturating_add(6)
            .saturating_add(value_count.saturating_mul(4))
            .min(tokens.len())
    }

    fn consume_memory_after_list(
        &self,
        tokens: &[TokenId],
        prompt_token_count: usize,
        index: usize,
        memory_write_slot: Option<usize>,
        families: &mut [Vec<TassadarStructuralSupervisionFamily>],
    ) -> usize {
        if index.saturating_add(6) > tokens.len()
            || tokens[index] != self.token_id(FIELD_MEMORY_AFTER)
            || tokens[index + 1] != self.token_id(LIST_TOKEN)
        {
            return index.saturating_add(1);
        }
        let value_count = self.parse_u32(tokens, index + 2).unwrap_or(0) as usize;
        if let Some(slot) = memory_write_slot {
            if slot < value_count {
                let value_start = index
                    .saturating_add(6)
                    .saturating_add(slot.saturating_mul(4));
                self.mark_target_range(
                    prompt_token_count,
                    value_start,
                    4,
                    TassadarStructuralSupervisionFamily::MemoryDiff,
                    families,
                );
            }
        }
        index
            .saturating_add(6)
            .saturating_add(value_count.saturating_mul(4))
            .min(tokens.len())
    }

    fn parse_u32(&self, tokens: &[TokenId], start: usize) -> Option<u32> {
        let bytes = [
            self.byte_value(*tokens.get(start)?)?,
            self.byte_value(*tokens.get(start + 1)?)?,
            self.byte_value(*tokens.get(start + 2)?)?,
            self.byte_value(*tokens.get(start + 3)?)?,
        ];
        Some(u32::from_le_bytes(bytes))
    }

    fn parse_i32_list(
        &self,
        tokens: &[TokenId],
        index: usize,
        field_token: &str,
    ) -> Option<Vec<i32>> {
        if index.saturating_add(6) > tokens.len()
            || tokens.get(index).copied()? != self.token_id(field_token)
            || tokens.get(index + 1).copied()? != self.token_id(LIST_TOKEN)
        {
            return None;
        }
        let value_count = self.parse_u32(tokens, index + 2)? as usize;
        let values_start = index + 6;
        let mut values = Vec::with_capacity(value_count);
        for value_index in 0..value_count {
            values.push(self.parse_i32(tokens, values_start + value_index.saturating_mul(4))?);
        }
        Some(values)
    }

    fn parse_wavefront_step(
        &self,
        tokens: &[TokenId],
        index: usize,
    ) -> Option<ParsedWavefrontStep> {
        if tokens.get(index).copied()? != self.token_id(STEP_TOKEN) {
            return None;
        }
        if tokens.get(index + 1).copied()? != self.token_id(FIELD_STEP_INDEX) {
            return None;
        }
        let step_index = self.parse_u32(tokens, index + 2)?;
        if tokens.get(index + 6).copied()? != self.token_id(FIELD_PC) {
            return None;
        }
        let payload_item_count = self.parse_u32(tokens, index + 7)?;
        if tokens.get(index + 11).copied()? != self.token_id(FIELD_NEXT_PC) {
            return None;
        }
        let summary_index = self.parse_u32(tokens, index + 12)?;
        let op_index = index + 16;
        if tokens.get(op_index).copied()? != self.token_id(OP_OUTPUT) {
            return None;
        }
        if tokens.get(op_index + 1).copied()? != self.token_id(EVENT_OUTPUT) {
            return None;
        }
        let summary_output = self.parse_i32(tokens, op_index + 2)?;
        let stack_before_index = op_index + 6;
        let stack_before = self.parse_i32_list(tokens, stack_before_index, FIELD_STACK_BEFORE)?;
        let stack_after_index = stack_before_index + 6 + stack_before.len().saturating_mul(4);
        let stack_after = self.parse_i32_list(tokens, stack_after_index, FIELD_STACK_AFTER)?;
        let locals_after_index = stack_after_index + 6 + stack_after.len().saturating_mul(4);
        let locals_after = self.parse_i32_list(tokens, locals_after_index, FIELD_LOCALS_AFTER)?;
        let memory_after_index = locals_after_index + 6 + locals_after.len().saturating_mul(4);
        let memory_after = self.parse_i32_list(tokens, memory_after_index, FIELD_MEMORY_AFTER)?;
        let next_index = memory_after_index + 6 + memory_after.len().saturating_mul(4);
        Some(ParsedWavefrontStep {
            step_index,
            payload_item_count,
            summary_index,
            summary_output: Some(summary_output),
            stack_after,
            next_index,
        })
    }

    fn mark_target_range(
        &self,
        prompt_token_count: usize,
        start: usize,
        len: usize,
        family: TassadarStructuralSupervisionFamily,
        families: &mut [Vec<TassadarStructuralSupervisionFamily>],
    ) {
        let end = start.saturating_add(len);
        for absolute_index in start..end {
            if absolute_index < prompt_token_count {
                continue;
            }
            let target_index = absolute_index - prompt_token_count;
            let Some(entry) = families.get_mut(target_index) else {
                continue;
            };
            if !entry.contains(&family) {
                entry.push(family);
            }
        }
    }
}

impl TokenizerBoundary for TassadarTraceTokenizer {
    fn encode(&self, text: &str) -> TokenSequence {
        let tokens: Vec<_> = text
            .split_whitespace()
            .map(|piece| {
                self.lookup
                    .get(piece)
                    .copied()
                    .unwrap_or(self.vocabulary.unknown_id())
            })
            .collect();
        TokenSequence::new(tokens)
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .map(|token| self.vocabulary.token(*token).unwrap_or(UNKNOWN_TOKEN))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn vocabulary(&self) -> &TokenVocabulary {
        &self.vocabulary
    }
}

fn stable_digest<T>(prefix: &[u8], value: &T) -> String
where
    T: Serialize,
{
    let encoded = serde_json::to_vec(value).expect("Tassadar tokenizer value should serialize");
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(encoded);
    hex::encode(hasher.finalize())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ParsedWavefrontStep {
    step_index: u32,
    payload_item_count: u32,
    summary_index: u32,
    summary_output: Option<i32>,
    stack_after: Vec<i32>,
    next_index: usize,
}

#[cfg(test)]
mod tests {
    use crate::TokenizerBoundary;
    use psionic_runtime::{
        TassadarCpuReferenceRunner, tassadar_hungarian_10x10_corpus, tassadar_sudoku_9x9_corpus,
        tassadar_sudoku_v0_corpus,
    };

    use super::{
        TassadarSequenceTraceFamily, TassadarStructuralSupervisionCoverage, TassadarTraceTokenizer,
    };

    #[test]
    fn tokenizer_roundtrips_symbolic_tokens_for_sudoku_v0_reference_case()
    -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = TassadarTraceTokenizer::new();
        let case = tassadar_sudoku_v0_corpus()
            .into_iter()
            .next()
            .expect("sudoku corpus should not be empty");
        let execution = TassadarCpuReferenceRunner::for_program(&case.validation_case.program)?
            .execute(&case.validation_case.program)?;
        let tokenized =
            tokenizer.tokenize_program_and_execution(&case.validation_case.program, &execution);
        let decoded = tokenizer.decode_symbolic(&tokenized);
        let reencoded = tokenizer.encode(decoded.tokens.join(" ").as_str());

        assert_eq!(reencoded, tokenized.sequence);
        assert!(tokenized.prompt_token_count > 0);
        assert!(tokenized.target_token_count > 0);
        assert!(!tokenized.sequence_digest.is_empty());
        Ok(())
    }

    #[test]
    fn tokenizer_digest_is_stable() {
        let tokenizer = TassadarTraceTokenizer::new();

        assert_eq!(tokenizer.stable_digest(), tokenizer.stable_digest());
        assert!(tokenizer.vocabulary().len() > 256);
    }

    #[test]
    fn tokenizer_derives_structural_supervision_coverage_for_reference_trace()
    -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = TassadarTraceTokenizer::new();
        let case = tassadar_sudoku_v0_corpus()
            .into_iter()
            .next()
            .expect("sudoku corpus should not be empty");
        let execution = TassadarCpuReferenceRunner::for_program(&case.validation_case.program)?
            .execute(&case.validation_case.program)?;
        let tokenized =
            tokenizer.tokenize_program_and_execution(&case.validation_case.program, &execution);
        let coverage = tokenizer.summarize_target_structural_supervision(
            tokenized.sequence.as_slice(),
            tokenized.prompt_token_count,
        );

        assert_eq!(
            coverage.total_target_token_count,
            tokenized.target_token_count as u32
        );
        assert!(coverage.instruction_pointer_token_count > 0);
        assert!(coverage.stack_delta_token_count > 0);
        assert_eq!(
            coverage.workload_specific_state_token_count,
            TassadarStructuralSupervisionCoverage::default().workload_specific_state_token_count
        );
        Ok(())
    }

    #[test]
    fn tokenizer_roundtrips_sudoku_wavefront_outputs_exactly()
    -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = TassadarTraceTokenizer::new();
        let case = tassadar_sudoku_9x9_corpus()
            .into_iter()
            .next()
            .expect("sudoku corpus should not be empty");
        let execution = TassadarCpuReferenceRunner::for_program(&case.validation_case.program)?
            .execute(&case.validation_case.program)?;
        let tokenized = tokenizer.tokenize_program_and_sudoku_diagonal_wavefront(
            &case.validation_case.program,
            execution.outputs.as_slice(),
            9,
        );

        let reconstructed = tokenizer.extract_sudoku_diagonal_wavefront_outputs(
            tokenized.sequence.as_slice(),
            tokenized.prompt_token_count,
            execution.outputs.len(),
        );

        assert_eq!(reconstructed, execution.outputs);
        assert_eq!(
            TassadarSequenceTraceFamily::SudokuDiagonalWavefront.reconstruction_scope(),
            "final_outputs_only"
        );
        Ok(())
    }

    #[test]
    fn tokenizer_roundtrips_hungarian_assignment_frontier_outputs_exactly()
    -> Result<(), Box<dyn std::error::Error>> {
        let tokenizer = TassadarTraceTokenizer::new();
        let case = tassadar_hungarian_10x10_corpus()
            .into_iter()
            .next()
            .expect("hungarian corpus should not be empty");
        let execution = TassadarCpuReferenceRunner::for_program(&case.validation_case.program)?
            .execute(&case.validation_case.program)?;
        let tokenized = tokenizer.tokenize_program_and_hungarian_assignment_frontier(
            &case.validation_case.program,
            case.optimal_assignment.as_slice(),
            case.cost_matrix.as_slice(),
            case.optimal_cost,
        );

        let reconstructed = tokenizer.extract_hungarian_assignment_frontier_outputs(
            tokenized.sequence.as_slice(),
            tokenized.prompt_token_count,
            case.optimal_assignment.len(),
        );

        assert_eq!(reconstructed, execution.outputs);
        Ok(())
    }
}
