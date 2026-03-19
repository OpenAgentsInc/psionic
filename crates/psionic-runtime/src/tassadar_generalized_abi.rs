use std::collections::{BTreeMap, BTreeSet};

use psionic_ir::{
    TassadarGeneralizedAbiFixture, TassadarGeneralizedAbiMemoryRegion,
    TassadarGeneralizedAbiMemoryRegionRole, TassadarGeneralizedAbiParamKind,
    TassadarGeneralizedAbiResultKind,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::TassadarStructuredControlBinaryOp;

const TASSADAR_GENERALIZED_ABI_MAX_STEPS: usize = 32_768;

/// One instruction in the widened generalized ABI family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "opcode", rename_all = "snake_case")]
pub enum TassadarGeneralizedAbiInstruction {
    I32Const {
        value: i32,
    },
    LocalGet {
        local_index: u32,
    },
    LocalSet {
        local_index: u32,
    },
    BinaryOp {
        op: TassadarStructuredControlBinaryOp,
    },
    I32LoadRegionAtIndex {
        region_id: String,
        index_local_index: u32,
    },
    I32StoreRegionAtIndex {
        region_id: String,
        index_local_index: u32,
        value_local_index: u32,
    },
    BranchIfZero {
        target_pc: usize,
    },
    Jump {
        target_pc: usize,
    },
    Return,
}

/// One validated runtime program for the widened generalized ABI family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarGeneralizedAbiProgram {
    pub program_id: String,
    pub fixture_id: String,
    pub source_case_id: String,
    pub source_ref: String,
    pub export_name: String,
    pub program_shape_id: String,
    pub param_kinds: Vec<TassadarGeneralizedAbiParamKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_kind: Option<TassadarGeneralizedAbiResultKind>,
    pub local_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory_regions: Vec<TassadarGeneralizedAbiMemoryRegion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime_support_ids: Vec<String>,
    pub instructions: Vec<TassadarGeneralizedAbiInstruction>,
    pub claim_boundary: String,
}

impl TassadarGeneralizedAbiProgram {
    pub fn validate(&self) -> Result<(), TassadarGeneralizedAbiError> {
        if self.instructions.is_empty() {
            return Err(TassadarGeneralizedAbiError::NoInstructions);
        }
        if self.local_count < self.param_kinds.len() {
            return Err(TassadarGeneralizedAbiError::LocalCountTooSmall {
                local_count: self.local_count,
                param_count: self.param_kinds.len(),
            });
        }
        for (param_index, kind) in self.param_kinds.iter().copied().enumerate() {
            match kind {
                TassadarGeneralizedAbiParamKind::I32
                | TassadarGeneralizedAbiParamKind::PointerToI32
                | TassadarGeneralizedAbiParamKind::LengthI32 => {}
                unsupported => {
                    return Err(TassadarGeneralizedAbiError::UnsupportedParamKind {
                        param_index: param_index as u8,
                        kind: unsupported,
                    });
                }
            }
        }
        if let Some(result_kind) = self.result_kind
            && result_kind != TassadarGeneralizedAbiResultKind::I32
        {
            return Err(TassadarGeneralizedAbiError::UnsupportedResultKind { kind: result_kind });
        }
        let mut region_ids = BTreeSet::new();
        for region in &self.memory_regions {
            validate_memory_region(region, &self.param_kinds)?;
            if region.region_id.trim().is_empty() {
                return Err(TassadarGeneralizedAbiError::EmptyMemoryRegionId);
            }
            if !region_ids.insert(region.region_id.clone()) {
                return Err(TassadarGeneralizedAbiError::DuplicateMemoryRegionId {
                    region_id: region.region_id.clone(),
                });
            }
        }
        for (pc, instruction) in self.instructions.iter().enumerate() {
            match instruction {
                TassadarGeneralizedAbiInstruction::LocalGet { local_index }
                | TassadarGeneralizedAbiInstruction::LocalSet { local_index }
                    if *local_index as usize >= self.local_count =>
                {
                    return Err(TassadarGeneralizedAbiError::LocalOutOfRange {
                        local_index: *local_index,
                        local_count: self.local_count,
                    });
                }
                TassadarGeneralizedAbiInstruction::I32LoadRegionAtIndex {
                    region_id,
                    index_local_index,
                } => {
                    if *index_local_index as usize >= self.local_count {
                        return Err(TassadarGeneralizedAbiError::LocalOutOfRange {
                            local_index: *index_local_index,
                            local_count: self.local_count,
                        });
                    }
                    if !region_ids.contains(region_id) {
                        return Err(
                            TassadarGeneralizedAbiError::UnknownMemoryRegionInInstruction {
                                region_id: region_id.clone(),
                            },
                        );
                    }
                }
                TassadarGeneralizedAbiInstruction::I32StoreRegionAtIndex {
                    region_id,
                    index_local_index,
                    value_local_index,
                } => {
                    for local_index in [index_local_index, value_local_index] {
                        if *local_index as usize >= self.local_count {
                            return Err(TassadarGeneralizedAbiError::LocalOutOfRange {
                                local_index: *local_index,
                                local_count: self.local_count,
                            });
                        }
                    }
                    if !region_ids.contains(region_id) {
                        return Err(
                            TassadarGeneralizedAbiError::UnknownMemoryRegionInInstruction {
                                region_id: region_id.clone(),
                            },
                        );
                    }
                }
                TassadarGeneralizedAbiInstruction::BranchIfZero { target_pc }
                | TassadarGeneralizedAbiInstruction::Jump { target_pc }
                    if *target_pc >= self.instructions.len() =>
                {
                    return Err(TassadarGeneralizedAbiError::InvalidBranchTarget {
                        pc,
                        target_pc: *target_pc,
                        instruction_count: self.instructions.len(),
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// One invocation over the widened generalized ABI family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarGeneralizedAbiInvocation {
    pub args: Vec<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub heap_bytes: Vec<u8>,
}

impl TassadarGeneralizedAbiInvocation {
    #[must_use]
    pub fn new(args: Vec<i32>) -> Self {
        Self {
            args,
            heap_bytes: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_heap_bytes(mut self, heap_bytes: Vec<u8>) -> Self {
        self.heap_bytes = heap_bytes;
        self
    }
}

/// One observed output region after generalized ABI execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarGeneralizedAbiRegionObservation {
    pub region_id: String,
    pub role: TassadarGeneralizedAbiMemoryRegionRole,
    pub pointer: i32,
    pub length: i32,
    pub element_width_bytes: u8,
    pub words: Vec<i32>,
    pub region_digest: String,
}

/// One trace event in the widened generalized ABI family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TassadarGeneralizedAbiTraceEvent {
    ConstPush {
        value: i32,
    },
    LocalGet {
        local_index: u32,
        value: i32,
    },
    LocalSet {
        local_index: u32,
        value: i32,
    },
    BinaryOp {
        op: TassadarStructuredControlBinaryOp,
        left: i32,
        right: i32,
        result: i32,
    },
    RegionLoad {
        region_id: String,
        address: u32,
        value: i32,
    },
    RegionStore {
        region_id: String,
        address: u32,
        value: i32,
    },
    BranchIfZero {
        condition: i32,
        target_pc: usize,
        taken: bool,
    },
    Jump {
        target_pc: usize,
    },
    Return {
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<i32>,
    },
}

/// One append-only trace step in the widened generalized ABI family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarGeneralizedAbiTraceStep {
    pub step_index: usize,
    pub pc_before: usize,
    pub event: TassadarGeneralizedAbiTraceEvent,
    pub locals_after: Vec<i32>,
    pub operand_stack_after: Vec<i32>,
}

/// One complete execution result in the widened generalized ABI family.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarGeneralizedAbiExecution {
    pub program_id: String,
    pub fixture_id: String,
    pub export_name: String,
    pub invocation_arg_digest: String,
    pub heap_before_digest: String,
    pub heap_after_digest: String,
    pub steps: Vec<TassadarGeneralizedAbiTraceStep>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returned_value: Option<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_regions: Vec<TassadarGeneralizedAbiRegionObservation>,
}

impl TassadarGeneralizedAbiExecution {
    #[must_use]
    pub fn execution_digest(&self) -> String {
        stable_digest(b"tassadar_generalized_abi_execution|", self)
    }
}

/// Validation or execution failure for the widened generalized ABI family.
#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error_kind")]
pub enum TassadarGeneralizedAbiError {
    #[error("generalized ABI program declares no instructions")]
    NoInstructions,
    #[error(
        "generalized ABI program local_count {local_count} is smaller than param_count {param_count}"
    )]
    LocalCountTooSmall {
        local_count: usize,
        param_count: usize,
    },
    #[error("generalized ABI param {param_index} declares unsupported kind `{kind:?}`")]
    UnsupportedParamKind {
        param_index: u8,
        kind: TassadarGeneralizedAbiParamKind,
    },
    #[error("generalized ABI result declares unsupported kind `{kind:?}`")]
    UnsupportedResultKind {
        kind: TassadarGeneralizedAbiResultKind,
    },
    #[error("generalized ABI memory region id must not be empty")]
    EmptyMemoryRegionId,
    #[error("generalized ABI memory region `{region_id}` was declared more than once")]
    DuplicateMemoryRegionId { region_id: String },
    #[error(
        "generalized ABI memory region `{region_id}` pointer_param_index {pointer_param_index} is out of range for {param_count} params"
    )]
    MemoryRegionPointerParamOutOfRange {
        region_id: String,
        pointer_param_index: u8,
        param_count: usize,
    },
    #[error(
        "generalized ABI memory region `{region_id}` length_param_index {length_param_index} is out of range for {param_count} params"
    )]
    MemoryRegionLengthParamOutOfRange {
        region_id: String,
        length_param_index: u8,
        param_count: usize,
    },
    #[error(
        "generalized ABI memory region `{region_id}` pointer param {pointer_param_index} must be pointer_to_i32"
    )]
    MemoryRegionPointerParamKindMismatch {
        region_id: String,
        pointer_param_index: u8,
    },
    #[error(
        "generalized ABI memory region `{region_id}` length param {length_param_index} must be length_i32"
    )]
    MemoryRegionLengthParamKindMismatch {
        region_id: String,
        length_param_index: u8,
    },
    #[error(
        "generalized ABI memory region `{region_id}` element_width_bytes {element_width_bytes} is unsupported"
    )]
    UnsupportedMemoryRegionElementWidth {
        region_id: String,
        element_width_bytes: u8,
    },
    #[error("generalized ABI instruction referenced unknown memory region `{region_id}`")]
    UnknownMemoryRegionInInstruction { region_id: String },
    #[error("generalized ABI local {local_index} is out of range (local_count={local_count})")]
    LocalOutOfRange {
        local_index: u32,
        local_count: usize,
    },
    #[error(
        "generalized ABI instruction at pc {pc} branches to invalid target {target_pc} (instruction_count={instruction_count})"
    )]
    InvalidBranchTarget {
        pc: usize,
        target_pc: usize,
        instruction_count: usize,
    },
    #[error("generalized ABI invocation arg count mismatch: expected {expected}, actual {actual}")]
    InvocationArgCountMismatch { expected: usize, actual: usize },
    #[error("generalized ABI heap-backed invocation is missing heap bytes")]
    MissingHeapBytes,
    #[error("generalized ABI pointer arg {pointer} must be non-negative")]
    NegativePointer { pointer: i32 },
    #[error("generalized ABI length arg {length} must be non-negative")]
    NegativeLength { length: i32 },
    #[error("generalized ABI pointer {pointer} is not aligned to {required_alignment} bytes")]
    UnalignedPointer {
        pointer: i32,
        required_alignment: u8,
    },
    #[error(
        "generalized ABI memory region `{region_id}` overflows the invocation heap: pointer={pointer} length={length} element_width_bytes={element_width_bytes} heap_len={heap_len}"
    )]
    MemoryRegionOutOfRange {
        region_id: String,
        pointer: i32,
        length: i32,
        element_width_bytes: u8,
        heap_len: usize,
    },
    #[error(
        "generalized ABI memory region `{region_id}` length {actual_length} is shorter than the minimum {minimum_length}"
    )]
    MemoryRegionTooShort {
        region_id: String,
        actual_length: i32,
        minimum_length: u8,
    },
    #[error(
        "generalized ABI output region `{output_region_id}` aliases region `{other_region_id}`, which is forbidden under caller-owned output-buffer rules"
    )]
    AliasedMemoryRegions {
        output_region_id: String,
        other_region_id: String,
    },
    #[error(
        "generalized ABI region `{region_id}` index {index} is out of range for declared length {length}"
    )]
    MemoryRegionIndexOutOfRange {
        region_id: String,
        index: i32,
        length: i32,
    },
    #[error(
        "generalized ABI stack underflow at pc {pc} for {context}: needed {needed}, available {available}"
    )]
    StackUnderflow {
        pc: usize,
        context: String,
        needed: usize,
        available: usize,
    },
    #[error(
        "generalized ABI heap load at address {address} with width {width_bytes} exceeded heap_len {heap_len}"
    )]
    HeapLoadOutOfRange {
        address: u32,
        width_bytes: u32,
        heap_len: usize,
    },
    #[error(
        "generalized ABI heap store at address {address} with width {width_bytes} exceeded heap_len {heap_len}"
    )]
    HeapStoreOutOfRange {
        address: u32,
        width_bytes: u32,
        heap_len: usize,
    },
    #[error("generalized ABI execution exceeded the step limit of {max_steps}")]
    StepLimitExceeded { max_steps: usize },
}

/// Executes one validated generalized ABI program.
pub fn execute_tassadar_generalized_abi_program(
    program: &TassadarGeneralizedAbiProgram,
    invocation: &TassadarGeneralizedAbiInvocation,
) -> Result<TassadarGeneralizedAbiExecution, TassadarGeneralizedAbiError> {
    program.validate()?;
    validate_invocation(program, invocation)?;

    let mut locals = vec![0; program.local_count];
    for (index, value) in invocation.args.iter().copied().enumerate() {
        locals[index] = value;
    }
    let mut heap_bytes = invocation.heap_bytes.clone();
    let mut operand_stack = Vec::new();
    let mut steps = Vec::new();
    let mut pc = 0usize;
    let mut step_index = 0usize;
    let region_lookup = program
        .memory_regions
        .iter()
        .map(|region| (region.region_id.as_str(), region))
        .collect::<BTreeMap<_, _>>();

    loop {
        if step_index >= TASSADAR_GENERALIZED_ABI_MAX_STEPS {
            return Err(TassadarGeneralizedAbiError::StepLimitExceeded {
                max_steps: TASSADAR_GENERALIZED_ABI_MAX_STEPS,
            });
        }
        let instruction = program
            .instructions
            .get(pc)
            .ok_or(TassadarGeneralizedAbiError::InvalidBranchTarget {
                pc,
                target_pc: pc,
                instruction_count: program.instructions.len(),
            })?
            .clone();
        let pc_before = pc;
        let event = match instruction {
            TassadarGeneralizedAbiInstruction::I32Const { value } => {
                operand_stack.push(value);
                pc += 1;
                TassadarGeneralizedAbiTraceEvent::ConstPush { value }
            }
            TassadarGeneralizedAbiInstruction::LocalGet { local_index } => {
                let value = *locals.get(local_index as usize).ok_or(
                    TassadarGeneralizedAbiError::LocalOutOfRange {
                        local_index,
                        local_count: locals.len(),
                    },
                )?;
                operand_stack.push(value);
                pc += 1;
                TassadarGeneralizedAbiTraceEvent::LocalGet { local_index, value }
            }
            TassadarGeneralizedAbiInstruction::LocalSet { local_index } => {
                let value = pop_operand(&mut operand_stack, pc, "local.set")?;
                let local_count = locals.len();
                *locals.get_mut(local_index as usize).ok_or(
                    TassadarGeneralizedAbiError::LocalOutOfRange {
                        local_index,
                        local_count,
                    },
                )? = value;
                pc += 1;
                TassadarGeneralizedAbiTraceEvent::LocalSet { local_index, value }
            }
            TassadarGeneralizedAbiInstruction::BinaryOp { op } => {
                let right = pop_operand(&mut operand_stack, pc, "binary_op")?;
                let left = pop_operand(&mut operand_stack, pc, "binary_op")?;
                let result = execute_binary_op(op, left, right);
                operand_stack.push(result);
                pc += 1;
                TassadarGeneralizedAbiTraceEvent::BinaryOp {
                    op,
                    left,
                    right,
                    result,
                }
            }
            TassadarGeneralizedAbiInstruction::I32LoadRegionAtIndex {
                region_id,
                index_local_index,
            } => {
                let region = region_lookup.get(region_id.as_str()).copied().ok_or(
                    TassadarGeneralizedAbiError::UnknownMemoryRegionInInstruction {
                        region_id: region_id.clone(),
                    },
                )?;
                let address = memory_region_address(region, &locals, index_local_index)?;
                let value = load_i32(&heap_bytes, address)?;
                operand_stack.push(value);
                pc += 1;
                TassadarGeneralizedAbiTraceEvent::RegionLoad {
                    region_id,
                    address,
                    value,
                }
            }
            TassadarGeneralizedAbiInstruction::I32StoreRegionAtIndex {
                region_id,
                index_local_index,
                value_local_index,
            } => {
                let region = region_lookup.get(region_id.as_str()).copied().ok_or(
                    TassadarGeneralizedAbiError::UnknownMemoryRegionInInstruction {
                        region_id: region_id.clone(),
                    },
                )?;
                let address = memory_region_address(region, &locals, index_local_index)?;
                let value = *locals.get(value_local_index as usize).ok_or(
                    TassadarGeneralizedAbiError::LocalOutOfRange {
                        local_index: value_local_index,
                        local_count: locals.len(),
                    },
                )?;
                store_i32(&mut heap_bytes, address, value)?;
                pc += 1;
                TassadarGeneralizedAbiTraceEvent::RegionStore {
                    region_id,
                    address,
                    value,
                }
            }
            TassadarGeneralizedAbiInstruction::BranchIfZero { target_pc } => {
                let condition = pop_operand(&mut operand_stack, pc, "branch_if_zero")?;
                let taken = condition == 0;
                if taken {
                    pc = target_pc;
                } else {
                    pc += 1;
                }
                TassadarGeneralizedAbiTraceEvent::BranchIfZero {
                    condition,
                    target_pc,
                    taken,
                }
            }
            TassadarGeneralizedAbiInstruction::Jump { target_pc } => {
                pc = target_pc;
                TassadarGeneralizedAbiTraceEvent::Jump { target_pc }
            }
            TassadarGeneralizedAbiInstruction::Return => {
                let value = match program.result_kind {
                    Some(TassadarGeneralizedAbiResultKind::I32) => {
                        Some(pop_operand(&mut operand_stack, pc, "return_value")?)
                    }
                    Some(other) => {
                        return Err(TassadarGeneralizedAbiError::UnsupportedResultKind {
                            kind: other,
                        });
                    }
                    None => None,
                };
                steps.push(TassadarGeneralizedAbiTraceStep {
                    step_index,
                    pc_before,
                    event: TassadarGeneralizedAbiTraceEvent::Return { value },
                    locals_after: locals.clone(),
                    operand_stack_after: operand_stack.clone(),
                });
                let output_regions = program
                    .memory_regions
                    .iter()
                    .filter(|region| region.role == TassadarGeneralizedAbiMemoryRegionRole::Output)
                    .map(|region| capture_region_observation(region, &locals, &heap_bytes))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(TassadarGeneralizedAbiExecution {
                    program_id: program.program_id.clone(),
                    fixture_id: program.fixture_id.clone(),
                    export_name: program.export_name.clone(),
                    invocation_arg_digest: stable_digest(
                        b"tassadar_generalized_abi_invocation_args|",
                        &invocation.args,
                    ),
                    heap_before_digest: stable_digest(
                        b"tassadar_generalized_abi_heap_before|",
                        &invocation.heap_bytes,
                    ),
                    heap_after_digest: stable_digest(
                        b"tassadar_generalized_abi_heap_after|",
                        &heap_bytes,
                    ),
                    steps,
                    returned_value: value,
                    output_regions,
                });
            }
        };
        steps.push(TassadarGeneralizedAbiTraceStep {
            step_index,
            pc_before,
            event,
            locals_after: locals.clone(),
            operand_stack_after: operand_stack.clone(),
        });
        step_index = step_index.saturating_add(1);
    }
}

fn validate_memory_region(
    region: &TassadarGeneralizedAbiMemoryRegion,
    param_kinds: &[TassadarGeneralizedAbiParamKind],
) -> Result<(), TassadarGeneralizedAbiError> {
    if region.pointer_param_index as usize >= param_kinds.len() {
        return Err(
            TassadarGeneralizedAbiError::MemoryRegionPointerParamOutOfRange {
                region_id: region.region_id.clone(),
                pointer_param_index: region.pointer_param_index,
                param_count: param_kinds.len(),
            },
        );
    }
    if region.length_param_index as usize >= param_kinds.len() {
        return Err(
            TassadarGeneralizedAbiError::MemoryRegionLengthParamOutOfRange {
                region_id: region.region_id.clone(),
                length_param_index: region.length_param_index,
                param_count: param_kinds.len(),
            },
        );
    }
    if param_kinds[region.pointer_param_index as usize]
        != TassadarGeneralizedAbiParamKind::PointerToI32
    {
        return Err(
            TassadarGeneralizedAbiError::MemoryRegionPointerParamKindMismatch {
                region_id: region.region_id.clone(),
                pointer_param_index: region.pointer_param_index,
            },
        );
    }
    if param_kinds[region.length_param_index as usize] != TassadarGeneralizedAbiParamKind::LengthI32
    {
        return Err(
            TassadarGeneralizedAbiError::MemoryRegionLengthParamKindMismatch {
                region_id: region.region_id.clone(),
                length_param_index: region.length_param_index,
            },
        );
    }
    if region.element_width_bytes != 4 {
        return Err(
            TassadarGeneralizedAbiError::UnsupportedMemoryRegionElementWidth {
                region_id: region.region_id.clone(),
                element_width_bytes: region.element_width_bytes,
            },
        );
    }
    Ok(())
}

fn validate_invocation(
    program: &TassadarGeneralizedAbiProgram,
    invocation: &TassadarGeneralizedAbiInvocation,
) -> Result<(), TassadarGeneralizedAbiError> {
    if invocation.args.len() != program.param_kinds.len() {
        return Err(TassadarGeneralizedAbiError::InvocationArgCountMismatch {
            expected: program.param_kinds.len(),
            actual: invocation.args.len(),
        });
    }
    if !program.memory_regions.is_empty() && invocation.heap_bytes.is_empty() {
        return Err(TassadarGeneralizedAbiError::MissingHeapBytes);
    }

    let mut ranges = Vec::new();
    for region in &program.memory_regions {
        let pointer = invocation.args[region.pointer_param_index as usize];
        let length = invocation.args[region.length_param_index as usize];
        if pointer < 0 {
            return Err(TassadarGeneralizedAbiError::NegativePointer { pointer });
        }
        if length < 0 {
            return Err(TassadarGeneralizedAbiError::NegativeLength { length });
        }
        if pointer % i32::from(region.element_width_bytes) != 0 {
            return Err(TassadarGeneralizedAbiError::UnalignedPointer {
                pointer,
                required_alignment: region.element_width_bytes,
            });
        }
        if length < i32::from(region.minimum_length_elements) {
            return Err(TassadarGeneralizedAbiError::MemoryRegionTooShort {
                region_id: region.region_id.clone(),
                actual_length: length,
                minimum_length: region.minimum_length_elements,
            });
        }
        let needed_bytes = usize::try_from(pointer).unwrap_or_default()
            + usize::try_from(length).unwrap_or_default() * usize::from(region.element_width_bytes);
        if needed_bytes > invocation.heap_bytes.len() {
            return Err(TassadarGeneralizedAbiError::MemoryRegionOutOfRange {
                region_id: region.region_id.clone(),
                pointer,
                length,
                element_width_bytes: region.element_width_bytes,
                heap_len: invocation.heap_bytes.len(),
            });
        }
        ranges.push((
            region.region_id.clone(),
            region.role,
            usize::try_from(pointer).unwrap_or_default(),
            needed_bytes,
        ));
    }

    for (index, (region_id, role, start, end)) in ranges.iter().enumerate() {
        if *role != TassadarGeneralizedAbiMemoryRegionRole::Output || start == end {
            continue;
        }
        for (other_index, (other_region_id, _, other_start, other_end)) in ranges.iter().enumerate()
        {
            if other_index == index {
                continue;
            }
            if other_start == other_end {
                continue;
            }
            let overlaps = start < other_end && other_start < end;
            if overlaps {
                return Err(TassadarGeneralizedAbiError::AliasedMemoryRegions {
                    output_region_id: region_id.clone(),
                    other_region_id: other_region_id.clone(),
                });
            }
        }
    }

    Ok(())
}

fn memory_region_address(
    region: &TassadarGeneralizedAbiMemoryRegion,
    locals: &[i32],
    index_local_index: u32,
) -> Result<u32, TassadarGeneralizedAbiError> {
    let base_pointer = *locals.get(region.pointer_param_index as usize).ok_or(
        TassadarGeneralizedAbiError::LocalOutOfRange {
            local_index: u32::from(region.pointer_param_index),
            local_count: locals.len(),
        },
    )?;
    let declared_length = *locals.get(region.length_param_index as usize).ok_or(
        TassadarGeneralizedAbiError::LocalOutOfRange {
            local_index: u32::from(region.length_param_index),
            local_count: locals.len(),
        },
    )?;
    let index = *locals.get(index_local_index as usize).ok_or(
        TassadarGeneralizedAbiError::LocalOutOfRange {
            local_index: index_local_index,
            local_count: locals.len(),
        },
    )?;
    if index < 0 || index >= declared_length {
        return Err(TassadarGeneralizedAbiError::MemoryRegionIndexOutOfRange {
            region_id: region.region_id.clone(),
            index,
            length: declared_length,
        });
    }
    let base_pointer =
        u32::try_from(base_pointer).map_err(|_| TassadarGeneralizedAbiError::NegativePointer {
            pointer: base_pointer,
        })?;
    let index = u32::try_from(index)
        .map_err(|_| TassadarGeneralizedAbiError::NegativeLength { length: index })?;
    base_pointer
        .checked_add(index.saturating_mul(u32::from(region.element_width_bytes)))
        .ok_or(TassadarGeneralizedAbiError::HeapLoadOutOfRange {
            address: u32::MAX,
            width_bytes: u32::from(region.element_width_bytes),
            heap_len: 0,
        })
}

fn load_i32(heap_bytes: &[u8], address: u32) -> Result<i32, TassadarGeneralizedAbiError> {
    let address = usize::try_from(address).unwrap_or(usize::MAX);
    let end = address.saturating_add(4);
    if end > heap_bytes.len() {
        return Err(TassadarGeneralizedAbiError::HeapLoadOutOfRange {
            address: address as u32,
            width_bytes: 4,
            heap_len: heap_bytes.len(),
        });
    }
    Ok(i32::from_le_bytes(
        heap_bytes[address..end].try_into().expect("length checked"),
    ))
}

fn store_i32(
    heap_bytes: &mut [u8],
    address: u32,
    value: i32,
) -> Result<(), TassadarGeneralizedAbiError> {
    let address = usize::try_from(address).unwrap_or(usize::MAX);
    let end = address.saturating_add(4);
    if end > heap_bytes.len() {
        return Err(TassadarGeneralizedAbiError::HeapStoreOutOfRange {
            address: address as u32,
            width_bytes: 4,
            heap_len: heap_bytes.len(),
        });
    }
    heap_bytes[address..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn capture_region_observation(
    region: &TassadarGeneralizedAbiMemoryRegion,
    locals: &[i32],
    heap_bytes: &[u8],
) -> Result<TassadarGeneralizedAbiRegionObservation, TassadarGeneralizedAbiError> {
    let pointer = locals[region.pointer_param_index as usize];
    let length = locals[region.length_param_index as usize];
    let start = usize::try_from(pointer).unwrap_or_default();
    let byte_len =
        usize::try_from(length).unwrap_or_default() * usize::from(region.element_width_bytes);
    let end = start.saturating_add(byte_len);
    let mut words = Vec::new();
    let mut cursor = start;
    while cursor < end {
        words.push(load_i32(heap_bytes, cursor as u32)?);
        cursor = cursor.saturating_add(usize::from(region.element_width_bytes));
    }
    Ok(TassadarGeneralizedAbiRegionObservation {
        region_id: region.region_id.clone(),
        role: region.role,
        pointer,
        length,
        element_width_bytes: region.element_width_bytes,
        region_digest: stable_digest(
            b"tassadar_generalized_abi_region_observation|",
            &(region.region_id.as_str(), &words),
        ),
        words,
    })
}

fn pop_operand(
    operand_stack: &mut Vec<i32>,
    pc: usize,
    context: &str,
) -> Result<i32, TassadarGeneralizedAbiError> {
    let available = operand_stack.len();
    operand_stack
        .pop()
        .ok_or_else(|| TassadarGeneralizedAbiError::StackUnderflow {
            pc,
            context: String::from(context),
            needed: 1,
            available,
        })
}

fn execute_binary_op(op: TassadarStructuredControlBinaryOp, left: i32, right: i32) -> i32 {
    match op {
        TassadarStructuredControlBinaryOp::Add => left.saturating_add(right),
        TassadarStructuredControlBinaryOp::Sub => left.saturating_sub(right),
        TassadarStructuredControlBinaryOp::Mul => left.saturating_mul(right),
        TassadarStructuredControlBinaryOp::Eq => i32::from(left == right),
        TassadarStructuredControlBinaryOp::Ne => i32::from(left != right),
        TassadarStructuredControlBinaryOp::LtS => i32::from(left < right),
        TassadarStructuredControlBinaryOp::LtU => i32::from((left as u32) < (right as u32)),
        TassadarStructuredControlBinaryOp::GtS => i32::from(left > right),
        TassadarStructuredControlBinaryOp::GtU => i32::from((left as u32) > (right as u32)),
        TassadarStructuredControlBinaryOp::LeS => i32::from(left <= right),
        TassadarStructuredControlBinaryOp::LeU => i32::from((left as u32) <= (right as u32)),
        TassadarStructuredControlBinaryOp::GeS => i32::from(left >= right),
        TassadarStructuredControlBinaryOp::GeU => i32::from((left as u32) >= (right as u32)),
        TassadarStructuredControlBinaryOp::And => left & right,
        TassadarStructuredControlBinaryOp::Or => left | right,
        TassadarStructuredControlBinaryOp::Xor => left ^ right,
        TassadarStructuredControlBinaryOp::Shl => left.wrapping_shl(right as u32),
        TassadarStructuredControlBinaryOp::ShrS => left.wrapping_shr(right as u32),
        TassadarStructuredControlBinaryOp::ShrU => {
            ((left as u32).wrapping_shr(right as u32)) as i32
        }
    }
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    hex::encode(hasher.finalize())
}

#[must_use]
pub fn tassadar_generalized_abi_pair_add_invocation() -> TassadarGeneralizedAbiInvocation {
    TassadarGeneralizedAbiInvocation::new(vec![20, 22])
}

#[must_use]
pub fn tassadar_generalized_abi_dual_heap_dot_invocation() -> TassadarGeneralizedAbiInvocation {
    let mut bytes = Vec::new();
    for value in [1_i32, 2_i32, 3_i32, 4_i32, 5_i32, 6_i32] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    TassadarGeneralizedAbiInvocation::new(vec![0, 12, 3]).with_heap_bytes(bytes)
}

#[must_use]
pub fn tassadar_generalized_abi_dual_heap_dot_out_of_range_invocation()
-> TassadarGeneralizedAbiInvocation {
    let mut bytes = Vec::new();
    for value in [1_i32, 2_i32, 3_i32, 4_i32, 5_i32, 6_i32] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    TassadarGeneralizedAbiInvocation::new(vec![0, 16, 3]).with_heap_bytes(bytes)
}

#[must_use]
pub fn tassadar_generalized_abi_status_output_invocation() -> TassadarGeneralizedAbiInvocation {
    let mut bytes = Vec::new();
    for value in [5_i32, 2_i32, 9_i32, 3_i32, 0_i32, 0_i32] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    TassadarGeneralizedAbiInvocation::new(vec![0, 4, 16, 2]).with_heap_bytes(bytes)
}

#[must_use]
pub fn tassadar_generalized_abi_status_output_short_invocation() -> TassadarGeneralizedAbiInvocation
{
    let mut bytes = Vec::new();
    for value in [5_i32, 2_i32, 9_i32, 3_i32, 0_i32] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    TassadarGeneralizedAbiInvocation::new(vec![0, 4, 16, 1]).with_heap_bytes(bytes)
}

#[must_use]
pub fn tassadar_generalized_abi_status_output_aliasing_invocation()
-> TassadarGeneralizedAbiInvocation {
    let mut bytes = Vec::new();
    for value in [5_i32, 2_i32, 9_i32, 3_i32, 0_i32, 0_i32] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    TassadarGeneralizedAbiInvocation::new(vec![0, 4, 8, 2]).with_heap_bytes(bytes)
}

#[must_use]
pub fn tassadar_generalized_abi_program_id(fixture: &TassadarGeneralizedAbiFixture) -> String {
    format!(
        "tassadar.generalized_abi.{}.v1",
        fixture.fixture_id.as_str()
    )
}

#[cfg(test)]
mod tests {
    use super::{
        TassadarGeneralizedAbiError, TassadarGeneralizedAbiInstruction,
        TassadarGeneralizedAbiProgram, execute_tassadar_generalized_abi_program,
        tassadar_generalized_abi_dual_heap_dot_invocation,
        tassadar_generalized_abi_pair_add_invocation, tassadar_generalized_abi_program_id,
        tassadar_generalized_abi_status_output_aliasing_invocation,
        tassadar_generalized_abi_status_output_invocation,
        tassadar_generalized_abi_status_output_short_invocation,
    };
    use psionic_ir::TassadarGeneralizedAbiFixture;

    #[test]
    fn generalized_abi_scalar_and_output_shapes_are_exact() {
        let pair_fixture = TassadarGeneralizedAbiFixture::pair_add_i32();
        let pair_program = TassadarGeneralizedAbiProgram {
            program_id: tassadar_generalized_abi_program_id(&pair_fixture),
            fixture_id: String::from(pair_fixture.fixture_id.as_str()),
            source_case_id: pair_fixture.source_case_id.clone(),
            source_ref: pair_fixture.source_ref.clone(),
            export_name: pair_fixture.export_name.clone(),
            program_shape_id: pair_fixture.program_shape_id.clone(),
            param_kinds: pair_fixture.param_kinds.clone(),
            result_kind: pair_fixture.result_kinds.first().copied(),
            local_count: 2,
            memory_regions: Vec::new(),
            runtime_support_ids: pair_fixture.runtime_support_ids.clone(),
            instructions: vec![
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 0 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 1 },
                TassadarGeneralizedAbiInstruction::BinaryOp {
                    op: crate::TassadarStructuredControlBinaryOp::Add,
                },
                TassadarGeneralizedAbiInstruction::Return,
            ],
            claim_boundary: pair_fixture.claim_boundary.clone(),
        };

        let pair_execution = execute_tassadar_generalized_abi_program(
            &pair_program,
            &tassadar_generalized_abi_pair_add_invocation(),
        )
        .expect("pair add should execute");
        assert_eq!(pair_execution.returned_value, Some(42));

        let output_fixture = TassadarGeneralizedAbiFixture::sum_and_max_status_output();
        let output_program = TassadarGeneralizedAbiProgram {
            program_id: tassadar_generalized_abi_program_id(&output_fixture),
            fixture_id: String::from(output_fixture.fixture_id.as_str()),
            source_case_id: output_fixture.source_case_id.clone(),
            source_ref: output_fixture.source_ref.clone(),
            export_name: output_fixture.export_name.clone(),
            program_shape_id: output_fixture.program_shape_id.clone(),
            param_kinds: output_fixture.param_kinds.clone(),
            result_kind: output_fixture.result_kinds.first().copied(),
            local_count: 8,
            memory_regions: output_fixture.memory_regions.clone(),
            runtime_support_ids: output_fixture.runtime_support_ids.clone(),
            instructions: vec![
                TassadarGeneralizedAbiInstruction::I32Const { value: 0 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 4 },
                TassadarGeneralizedAbiInstruction::I32Const { value: 0 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 5 },
                TassadarGeneralizedAbiInstruction::I32Const { value: 0 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 6 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 6 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 1 },
                TassadarGeneralizedAbiInstruction::BinaryOp {
                    op: crate::TassadarStructuredControlBinaryOp::LtS,
                },
                TassadarGeneralizedAbiInstruction::BranchIfZero { target_pc: 27 },
                TassadarGeneralizedAbiInstruction::I32LoadRegionAtIndex {
                    region_id: String::from("input_values"),
                    index_local_index: 6,
                },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 7 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 4 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 7 },
                TassadarGeneralizedAbiInstruction::BinaryOp {
                    op: crate::TassadarStructuredControlBinaryOp::Add,
                },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 4 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 7 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 5 },
                TassadarGeneralizedAbiInstruction::BinaryOp {
                    op: crate::TassadarStructuredControlBinaryOp::GtS,
                },
                TassadarGeneralizedAbiInstruction::BranchIfZero { target_pc: 22 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 7 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 5 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 6 },
                TassadarGeneralizedAbiInstruction::I32Const { value: 1 },
                TassadarGeneralizedAbiInstruction::BinaryOp {
                    op: crate::TassadarStructuredControlBinaryOp::Add,
                },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 6 },
                TassadarGeneralizedAbiInstruction::Jump { target_pc: 6 },
                TassadarGeneralizedAbiInstruction::I32Const { value: 0 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 6 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 4 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 7 },
                TassadarGeneralizedAbiInstruction::I32StoreRegionAtIndex {
                    region_id: String::from("output_values"),
                    index_local_index: 6,
                    value_local_index: 7,
                },
                TassadarGeneralizedAbiInstruction::I32Const { value: 1 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 6 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 5 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 7 },
                TassadarGeneralizedAbiInstruction::I32StoreRegionAtIndex {
                    region_id: String::from("output_values"),
                    index_local_index: 6,
                    value_local_index: 7,
                },
                TassadarGeneralizedAbiInstruction::I32Const { value: 0 },
                TassadarGeneralizedAbiInstruction::Return,
            ],
            claim_boundary: output_fixture.claim_boundary,
        };

        let output_execution = execute_tassadar_generalized_abi_program(
            &output_program,
            &tassadar_generalized_abi_status_output_invocation(),
        )
        .expect("status output should execute");
        assert_eq!(output_execution.returned_value, Some(0));
        assert_eq!(output_execution.output_regions[0].words, vec![19, 9]);
    }

    #[test]
    fn generalized_abi_region_validation_refuses_aliasing_and_short_outputs() {
        let fixture = TassadarGeneralizedAbiFixture::sum_and_max_status_output();
        let program = TassadarGeneralizedAbiProgram {
            program_id: tassadar_generalized_abi_program_id(&fixture),
            fixture_id: String::from(fixture.fixture_id.as_str()),
            source_case_id: fixture.source_case_id.clone(),
            source_ref: fixture.source_ref.clone(),
            export_name: fixture.export_name.clone(),
            program_shape_id: fixture.program_shape_id.clone(),
            param_kinds: fixture.param_kinds.clone(),
            result_kind: fixture.result_kinds.first().copied(),
            local_count: 4,
            memory_regions: fixture.memory_regions.clone(),
            runtime_support_ids: fixture.runtime_support_ids.clone(),
            instructions: vec![
                TassadarGeneralizedAbiInstruction::I32Const { value: 0 },
                TassadarGeneralizedAbiInstruction::Return,
            ],
            claim_boundary: fixture.claim_boundary,
        };

        let short_error = execute_tassadar_generalized_abi_program(
            &program,
            &tassadar_generalized_abi_status_output_short_invocation(),
        )
        .expect_err("short output buffer should refuse");
        assert!(matches!(
            short_error,
            TassadarGeneralizedAbiError::MemoryRegionTooShort { .. }
        ));

        let alias_error = execute_tassadar_generalized_abi_program(
            &program,
            &tassadar_generalized_abi_status_output_aliasing_invocation(),
        )
        .expect_err("aliased output buffer should refuse");
        assert!(matches!(
            alias_error,
            TassadarGeneralizedAbiError::AliasedMemoryRegions { .. }
        ));
    }

    #[test]
    fn generalized_abi_multiple_pointer_inputs_are_exact() {
        let fixture = TassadarGeneralizedAbiFixture::dual_heap_dot_i32();
        let program = TassadarGeneralizedAbiProgram {
            program_id: tassadar_generalized_abi_program_id(&fixture),
            fixture_id: String::from(fixture.fixture_id.as_str()),
            source_case_id: fixture.source_case_id.clone(),
            source_ref: fixture.source_ref.clone(),
            export_name: fixture.export_name.clone(),
            program_shape_id: fixture.program_shape_id.clone(),
            param_kinds: fixture.param_kinds.clone(),
            result_kind: fixture.result_kinds.first().copied(),
            local_count: 7,
            memory_regions: fixture.memory_regions.clone(),
            runtime_support_ids: fixture.runtime_support_ids.clone(),
            instructions: vec![
                TassadarGeneralizedAbiInstruction::I32Const { value: 0 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 3 },
                TassadarGeneralizedAbiInstruction::I32Const { value: 0 },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 4 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 4 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 2 },
                TassadarGeneralizedAbiInstruction::BinaryOp {
                    op: crate::TassadarStructuredControlBinaryOp::LtS,
                },
                TassadarGeneralizedAbiInstruction::BranchIfZero { target_pc: 23 },
                TassadarGeneralizedAbiInstruction::I32LoadRegionAtIndex {
                    region_id: String::from("left_input"),
                    index_local_index: 4,
                },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 5 },
                TassadarGeneralizedAbiInstruction::I32LoadRegionAtIndex {
                    region_id: String::from("right_input"),
                    index_local_index: 4,
                },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 6 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 3 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 5 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 6 },
                TassadarGeneralizedAbiInstruction::BinaryOp {
                    op: crate::TassadarStructuredControlBinaryOp::Mul,
                },
                TassadarGeneralizedAbiInstruction::BinaryOp {
                    op: crate::TassadarStructuredControlBinaryOp::Add,
                },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 3 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 4 },
                TassadarGeneralizedAbiInstruction::I32Const { value: 1 },
                TassadarGeneralizedAbiInstruction::BinaryOp {
                    op: crate::TassadarStructuredControlBinaryOp::Add,
                },
                TassadarGeneralizedAbiInstruction::LocalSet { local_index: 4 },
                TassadarGeneralizedAbiInstruction::Jump { target_pc: 4 },
                TassadarGeneralizedAbiInstruction::LocalGet { local_index: 3 },
                TassadarGeneralizedAbiInstruction::Return,
            ],
            claim_boundary: fixture.claim_boundary,
        };

        let execution = execute_tassadar_generalized_abi_program(
            &program,
            &tassadar_generalized_abi_dual_heap_dot_invocation(),
        )
        .expect("dual heap dot should execute");
        assert_eq!(execution.returned_value, Some(32));
    }
}
