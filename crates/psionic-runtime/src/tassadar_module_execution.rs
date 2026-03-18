use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{TASSADAR_RUNTIME_BACKEND_ID, TassadarStructuredControlBinaryOp};

const TASSADAR_MODULE_EXECUTION_MAX_STEPS: usize = 4_096;

/// Value types admitted by the bounded module-execution lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarModuleValueType {
    /// 32-bit integer values.
    I32,
}

/// Mutability posture for one bounded module global.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarModuleGlobalMutability {
    /// Global cannot be written after initialization.
    Const,
    /// Global may be updated during execution.
    Mutable,
}

/// One global in the bounded module-execution lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarModuleGlobal {
    /// Stable global index.
    pub global_index: u32,
    /// Visible value type for the global.
    pub value_type: TassadarModuleValueType,
    /// Mutability posture.
    pub mutability: TassadarModuleGlobalMutability,
    /// Initial i32 value.
    pub initial_value: i32,
}

/// Table element kind admitted by the bounded module-execution lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarModuleTableElementKind {
    /// Function-reference table entries.
    Funcref,
}

/// One function table in the bounded module-execution lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarModuleTable {
    /// Stable table index.
    pub table_index: u32,
    /// Table element kind.
    pub element_kind: TassadarModuleTableElementKind,
    /// Minimum declared table size.
    pub min_entries: u32,
    /// Optional maximum declared table size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_entries: Option<u32>,
    /// Function indices stored in table order.
    pub elements: Vec<Option<u32>>,
}

/// Stub kinds admitted at the host-import boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarHostImportStubKind {
    /// Deterministic zero-side-effect import that returns one fixed i32.
    DeterministicI32Const,
    /// Explicit unsupported host-call boundary.
    UnsupportedHostCall,
}

/// Side-effect posture for deterministic import stubs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarDeterministicImportSideEffectPolicy {
    /// Deterministic stubs must stay side-effect free.
    NoSideEffects,
}

/// Typed refusal kind surfaced by the bounded module-execution lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarModuleExecutionRefusalKind {
    /// Host imports outside the deterministic stub set are refused.
    UnsupportedHostImport,
}

/// Hard capability boundary for host imports in the bounded module lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarHostImportCapabilityBoundary {
    /// Stub kinds admitted by the current boundary.
    pub supported_stub_kinds: Vec<TassadarHostImportStubKind>,
    /// Side-effect posture for admitted deterministic stubs.
    pub side_effect_policy: TassadarDeterministicImportSideEffectPolicy,
    /// Typed refusal emitted for unsupported host calls.
    pub unsupported_host_call_refusal: TassadarModuleExecutionRefusalKind,
    /// Plain-language boundary note.
    pub claim_boundary: String,
}

/// Runtime capability report for the bounded module-execution lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarModuleExecutionCapabilityReport {
    /// Runtime backend exposing the lane.
    pub runtime_backend: String,
    /// Whether globals are supported at all.
    pub supports_globals: bool,
    /// Whether mutable globals are supported.
    pub supports_mutable_globals: bool,
    /// Value types admitted for globals today.
    pub supported_global_value_types: Vec<TassadarModuleValueType>,
    /// Whether tables are supported at all.
    pub supports_tables: bool,
    /// Table element kind admitted today.
    pub supported_table_element_kind: TassadarModuleTableElementKind,
    /// Whether bounded indirect calls are supported.
    pub supports_call_indirect: bool,
    /// Maximum table entries admitted by the bounded lane.
    pub max_table_entries: u32,
    /// Hard capability boundary for host imports.
    pub host_import_boundary: TassadarHostImportCapabilityBoundary,
    /// Plain-language lane boundary.
    pub claim_boundary: String,
}

/// Returns the current runtime capability report for the bounded module lane.
#[must_use]
pub fn tassadar_module_execution_capability_report() -> TassadarModuleExecutionCapabilityReport {
    TassadarModuleExecutionCapabilityReport {
        runtime_backend: String::from(TASSADAR_RUNTIME_BACKEND_ID),
        supports_globals: true,
        supports_mutable_globals: true,
        supported_global_value_types: vec![TassadarModuleValueType::I32],
        supports_tables: true,
        supported_table_element_kind: TassadarModuleTableElementKind::Funcref,
        supports_call_indirect: true,
        max_table_entries: 64,
        host_import_boundary: TassadarHostImportCapabilityBoundary {
            supported_stub_kinds: vec![TassadarHostImportStubKind::DeterministicI32Const],
            side_effect_policy: TassadarDeterministicImportSideEffectPolicy::NoSideEffects,
            unsupported_host_call_refusal:
                TassadarModuleExecutionRefusalKind::UnsupportedHostImport,
            claim_boundary: String::from(
                "only deterministic zero-side-effect import stubs are admitted; arbitrary host calls remain explicitly refused",
            ),
        },
        claim_boundary: String::from(
            "bounded module execution covers i32 globals, funcref tables, zero-parameter indirect calls, and deterministic import stubs only; memories, arbitrary signatures, and arbitrary host calls remain out of scope",
        ),
    }
}

/// One import stub in the bounded module-execution lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "stub_kind", rename_all = "snake_case")]
pub enum TassadarHostImportStub {
    /// Deterministic import that returns one fixed i32 result.
    DeterministicI32Const {
        /// Stable import index.
        import_index: u32,
        /// Stable import reference.
        import_ref: String,
        /// Fixed return value.
        value: i32,
        /// Plain-language claim boundary.
        claim_boundary: String,
    },
    /// Unsupported host call that must refuse explicitly.
    UnsupportedHostCall {
        /// Stable import index.
        import_index: u32,
        /// Stable import reference.
        import_ref: String,
        /// Plain-language refusal detail.
        refusal_detail: String,
    },
}

impl TassadarHostImportStub {
    fn import_index(&self) -> u32 {
        match self {
            Self::DeterministicI32Const { import_index, .. }
            | Self::UnsupportedHostCall { import_index, .. } => *import_index,
        }
    }

    fn import_ref(&self) -> &str {
        match self {
            Self::DeterministicI32Const { import_ref, .. }
            | Self::UnsupportedHostCall { import_ref, .. } => import_ref.as_str(),
        }
    }
}

/// One instruction in the bounded module-execution lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "opcode", rename_all = "snake_case")]
pub enum TassadarModuleInstruction {
    /// Push one immediate i32.
    I32Const { value: i32 },
    /// Read one local.
    LocalGet { local_index: u32 },
    /// Pop one value into one local.
    LocalSet { local_index: u32 },
    /// Read one global.
    GlobalGet { global_index: u32 },
    /// Pop one value into one global.
    GlobalSet { global_index: u32 },
    /// Pop two i32 values and push one result.
    BinaryOp {
        op: TassadarStructuredControlBinaryOp,
    },
    /// Resolve one table slot and call the referenced function.
    CallIndirect { table_index: u32 },
    /// Invoke one host import stub.
    HostCall { import_index: u32 },
    /// Return from the current function.
    Return,
}

/// One function in the bounded module-execution lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarModuleFunction {
    /// Stable function index.
    pub function_index: u32,
    /// Stable function name.
    pub function_name: String,
    /// Number of parameters consumed by the function.
    pub param_count: u8,
    /// Number of locals including parameters.
    pub local_count: usize,
    /// Number of return values. Only 0 or 1 are supported.
    pub result_count: u8,
    /// Ordered instruction sequence.
    pub instructions: Vec<TassadarModuleInstruction>,
}

impl TassadarModuleFunction {
    /// Creates one bounded module function.
    #[must_use]
    pub fn new(
        function_index: u32,
        function_name: impl Into<String>,
        param_count: u8,
        local_count: usize,
        result_count: u8,
        instructions: Vec<TassadarModuleInstruction>,
    ) -> Self {
        Self {
            function_index,
            function_name: function_name.into(),
            param_count,
            local_count,
            result_count,
            instructions,
        }
    }
}

/// One bounded module-execution program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarModuleExecutionProgram {
    /// Stable program identifier.
    pub program_id: String,
    /// Entry function index.
    pub entry_function_index: u32,
    /// Maximum call depth before execution refuses.
    pub max_call_depth: u32,
    /// Declared globals in stable index order.
    pub globals: Vec<TassadarModuleGlobal>,
    /// Declared tables in stable index order.
    pub tables: Vec<TassadarModuleTable>,
    /// Declared host import stubs in stable index order.
    pub imports: Vec<TassadarHostImportStub>,
    /// Declared functions in stable index order.
    pub functions: Vec<TassadarModuleFunction>,
}

impl TassadarModuleExecutionProgram {
    /// Creates one bounded module-execution program.
    #[must_use]
    pub fn new(
        program_id: impl Into<String>,
        entry_function_index: u32,
        max_call_depth: u32,
        globals: Vec<TassadarModuleGlobal>,
        tables: Vec<TassadarModuleTable>,
        imports: Vec<TassadarHostImportStub>,
        functions: Vec<TassadarModuleFunction>,
    ) -> Self {
        Self {
            program_id: program_id.into(),
            entry_function_index,
            max_call_depth,
            globals,
            tables,
            imports,
            functions,
        }
    }

    /// Validates the public bounded module surface.
    pub fn validate(&self) -> Result<(), TassadarModuleExecutionError> {
        if self.functions.is_empty() {
            return Err(TassadarModuleExecutionError::NoFunctions);
        }
        let entry = self
            .functions
            .iter()
            .find(|function| function.function_index == self.entry_function_index)
            .ok_or(TassadarModuleExecutionError::MissingEntryFunction {
                entry_function_index: self.entry_function_index,
            })?;
        if entry.param_count != 0 {
            return Err(TassadarModuleExecutionError::EntryFunctionHasParameters {
                entry_function_index: self.entry_function_index,
                param_count: entry.param_count,
            });
        }

        for (expected_index, global) in self.globals.iter().enumerate() {
            if global.global_index != expected_index as u32 {
                return Err(TassadarModuleExecutionError::GlobalIndexDrift {
                    expected: expected_index as u32,
                    actual: global.global_index,
                });
            }
            if global.value_type != TassadarModuleValueType::I32 {
                return Err(TassadarModuleExecutionError::UnsupportedGlobalValueType {
                    global_index: global.global_index,
                    value_type: global.value_type,
                });
            }
        }

        for (expected_index, table) in self.tables.iter().enumerate() {
            if table.table_index != expected_index as u32 {
                return Err(TassadarModuleExecutionError::TableIndexDrift {
                    expected: expected_index as u32,
                    actual: table.table_index,
                });
            }
            if table.elements.len() < table.min_entries as usize {
                return Err(TassadarModuleExecutionError::TableBelowMinimum {
                    table_index: table.table_index,
                    min_entries: table.min_entries,
                    actual_entries: table.elements.len(),
                });
            }
            if let Some(max_entries) = table.max_entries
                && table.elements.len() > max_entries as usize
            {
                return Err(TassadarModuleExecutionError::TableAboveMaximum {
                    table_index: table.table_index,
                    max_entries,
                    actual_entries: table.elements.len(),
                });
            }
            for function_index in table.elements.iter().flatten() {
                if *function_index as usize >= self.functions.len() {
                    return Err(TassadarModuleExecutionError::TableFunctionOutOfRange {
                        table_index: table.table_index,
                        function_index: *function_index,
                        function_count: self.functions.len(),
                    });
                }
            }
        }

        for (expected_index, import) in self.imports.iter().enumerate() {
            if import.import_index() != expected_index as u32 {
                return Err(TassadarModuleExecutionError::ImportIndexDrift {
                    expected: expected_index as u32,
                    actual: import.import_index(),
                });
            }
            if import.import_ref().trim().is_empty() {
                return Err(TassadarModuleExecutionError::MissingImportRef {
                    import_index: import.import_index(),
                });
            }
        }

        for (expected_index, function) in self.functions.iter().enumerate() {
            if function.function_index != expected_index as u32 {
                return Err(TassadarModuleExecutionError::FunctionIndexDrift {
                    expected: expected_index as u32,
                    actual: function.function_index,
                });
            }
            if function.param_count != 0 {
                return Err(TassadarModuleExecutionError::UnsupportedParamCount {
                    function_index: function.function_index,
                    param_count: function.param_count,
                });
            }
            if function.local_count < usize::from(function.param_count) {
                return Err(TassadarModuleExecutionError::LocalCountTooSmall {
                    function_index: function.function_index,
                    param_count: function.param_count,
                    local_count: function.local_count,
                });
            }
            if function.result_count > 1 {
                return Err(TassadarModuleExecutionError::UnsupportedResultCount {
                    function_index: function.function_index,
                    result_count: function.result_count,
                });
            }
            for instruction in &function.instructions {
                match instruction {
                    TassadarModuleInstruction::LocalGet { local_index }
                    | TassadarModuleInstruction::LocalSet { local_index }
                        if *local_index as usize >= function.local_count =>
                    {
                        return Err(TassadarModuleExecutionError::LocalOutOfRange {
                            function_index: function.function_index,
                            local_index: *local_index,
                            local_count: function.local_count,
                        });
                    }
                    TassadarModuleInstruction::GlobalGet { global_index }
                    | TassadarModuleInstruction::GlobalSet { global_index }
                        if *global_index as usize >= self.globals.len() =>
                    {
                        return Err(TassadarModuleExecutionError::GlobalOutOfRange {
                            global_index: *global_index,
                            global_count: self.globals.len(),
                        });
                    }
                    TassadarModuleInstruction::CallIndirect { table_index }
                        if *table_index as usize >= self.tables.len() =>
                    {
                        return Err(TassadarModuleExecutionError::TableOutOfRange {
                            table_index: *table_index,
                            table_count: self.tables.len(),
                        });
                    }
                    TassadarModuleInstruction::HostCall { import_index }
                        if *import_index as usize >= self.imports.len() =>
                    {
                        return Err(TassadarModuleExecutionError::ImportOutOfRange {
                            import_index: *import_index,
                            import_count: self.imports.len(),
                        });
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

/// One visible frame snapshot in the module-execution trace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarModuleFrameSnapshot {
    /// Stable function index.
    pub function_index: u32,
    /// Stable function name.
    pub function_name: String,
    /// Program counter inside the function.
    pub pc: usize,
    /// Current locals.
    pub locals: Vec<i32>,
    /// Current operand stack.
    pub operand_stack: Vec<i32>,
}

/// One trace event in the bounded module-execution lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TassadarModuleTraceEvent {
    /// One constant was pushed.
    ConstPush { value: i32 },
    /// One local was read.
    LocalGet { local_index: u32, value: i32 },
    /// One local was written.
    LocalSet { local_index: u32, value: i32 },
    /// One global was read.
    GlobalGet { global_index: u32, value: i32 },
    /// One global was written.
    GlobalSet { global_index: u32, value: i32 },
    /// One binary operation completed.
    BinaryOp {
        op: TassadarStructuredControlBinaryOp,
        left: i32,
        right: i32,
        result: i32,
    },
    /// One indirect call pushed a new frame.
    CallIndirect {
        table_index: u32,
        selector: i32,
        function_index: u32,
    },
    /// One host import stub completed.
    HostCall {
        import_ref: String,
        stub_kind: TassadarHostImportStubKind,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<i32>,
    },
    /// One frame returned.
    Return {
        function_index: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<i32>,
        implicit: bool,
    },
}

/// One append-only trace step for the bounded module lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarModuleTraceStep {
    /// Step index in execution order.
    pub step_index: usize,
    /// Current frame depth after the step.
    pub frame_depth_after: usize,
    /// Event emitted by the step.
    pub event: TassadarModuleTraceEvent,
    /// Current globals after the step.
    pub globals_after: Vec<i32>,
    /// Current frame stack after the step.
    pub frame_stack_after: Vec<TassadarModuleFrameSnapshot>,
}

/// Terminal reason for one bounded module execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarModuleExecutionHaltReason {
    /// The entry function returned.
    Returned,
    /// The entry function fell off the end.
    FellOffEnd,
}

/// One complete execution result for the bounded module lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarModuleExecution {
    /// Stable program identifier.
    pub program_id: String,
    /// Ordered append-only trace steps.
    pub steps: Vec<TassadarModuleTraceStep>,
    /// Optional returned value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returned_value: Option<i32>,
    /// Final globals in stable index order.
    pub final_globals: Vec<i32>,
    /// Terminal halt reason.
    pub halt_reason: TassadarModuleExecutionHaltReason,
}

impl TassadarModuleExecution {
    /// Returns a stable digest over the visible execution truth.
    #[must_use]
    pub fn execution_digest(&self) -> String {
        stable_digest(b"tassadar_module_execution|", self)
    }
}

/// Validation or execution failure for the bounded module lane.
#[derive(Clone, Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TassadarModuleExecutionError {
    /// No functions were declared.
    #[error("module-execution program must contain at least one function")]
    NoFunctions,
    /// Entry function was missing.
    #[error("module-execution entry function {entry_function_index} is missing")]
    MissingEntryFunction { entry_function_index: u32 },
    /// Entry function declared parameters.
    #[error(
        "module-execution entry function {entry_function_index} declares unsupported param_count={param_count}"
    )]
    EntryFunctionHasParameters {
        entry_function_index: u32,
        param_count: u8,
    },
    /// Function indices drifted from stable order.
    #[error("module-execution function index drifted: expected {expected}, got {actual}")]
    FunctionIndexDrift { expected: u32, actual: u32 },
    /// One global index drifted from stable order.
    #[error("module-execution global index drifted: expected {expected}, got {actual}")]
    GlobalIndexDrift { expected: u32, actual: u32 },
    /// One table index drifted from stable order.
    #[error("module-execution table index drifted: expected {expected}, got {actual}")]
    TableIndexDrift { expected: u32, actual: u32 },
    /// One import index drifted from stable order.
    #[error("module-execution import index drifted: expected {expected}, got {actual}")]
    ImportIndexDrift { expected: u32, actual: u32 },
    /// One function declared unsupported parameters.
    #[error(
        "module-execution function {function_index} declares unsupported param_count={param_count}"
    )]
    UnsupportedParamCount {
        function_index: u32,
        param_count: u8,
    },
    /// One function declared too many results.
    #[error(
        "module-execution function {function_index} declares unsupported result_count={result_count}"
    )]
    UnsupportedResultCount {
        function_index: u32,
        result_count: u8,
    },
    /// One function declared fewer locals than parameters.
    #[error(
        "module-execution function {function_index} declared local_count={local_count} below param_count={param_count}"
    )]
    LocalCountTooSmall {
        function_index: u32,
        param_count: u8,
        local_count: usize,
    },
    /// One local index exceeded the declared local count.
    #[error(
        "module-execution function {function_index} referenced local {local_index} out of range (local_count={local_count})"
    )]
    LocalOutOfRange {
        function_index: u32,
        local_index: u32,
        local_count: usize,
    },
    /// One global value type is unsupported.
    #[error(
        "module-execution global {global_index} declared unsupported value type `{value_type:?}`"
    )]
    UnsupportedGlobalValueType {
        global_index: u32,
        value_type: TassadarModuleValueType,
    },
    /// One global index exceeded the declared global count.
    #[error("module-execution global {global_index} is out of range (global_count={global_count})")]
    GlobalOutOfRange {
        global_index: u32,
        global_count: usize,
    },
    /// One constant global was written.
    #[error("module-execution global {global_index} is const and cannot be written")]
    ImmutableGlobalWrite { global_index: u32 },
    /// One table index exceeded the declared table count.
    #[error("module-execution table {table_index} is out of range (table_count={table_count})")]
    TableOutOfRange {
        table_index: u32,
        table_count: usize,
    },
    /// One table had fewer elements than its minimum.
    #[error(
        "module-execution table {table_index} declared min_entries={min_entries} but carried {actual_entries} elements"
    )]
    TableBelowMinimum {
        table_index: u32,
        min_entries: u32,
        actual_entries: usize,
    },
    /// One table exceeded its maximum.
    #[error(
        "module-execution table {table_index} declared max_entries={max_entries} but carried {actual_entries} elements"
    )]
    TableAboveMaximum {
        table_index: u32,
        max_entries: u32,
        actual_entries: usize,
    },
    /// One function index in a table exceeded the declared function count.
    #[error(
        "module-execution table {table_index} referenced function {function_index} out of range (function_count={function_count})"
    )]
    TableFunctionOutOfRange {
        table_index: u32,
        function_index: u32,
        function_count: usize,
    },
    /// One call_indirect selector exceeded the table length.
    #[error(
        "module-execution table {table_index} selector {selector} is out of range (entry_count={entry_count})"
    )]
    TableSelectorOutOfRange {
        table_index: u32,
        selector: i32,
        entry_count: usize,
    },
    /// One indirect call hit an empty table slot.
    #[error("module-execution table {table_index} selector {selector} resolved to an empty slot")]
    EmptyTableEntry { table_index: u32, selector: i32 },
    /// One import index exceeded the declared import count.
    #[error("module-execution import {import_index} is out of range (import_count={import_count})")]
    ImportOutOfRange {
        import_index: u32,
        import_count: usize,
    },
    /// One import stub omitted its reference.
    #[error("module-execution import {import_index} is missing `import_ref`")]
    MissingImportRef { import_index: u32 },
    /// One host import stayed outside the admitted stub boundary.
    #[error("module-execution host import `{import_ref}` is unsupported: {detail}")]
    UnsupportedHostImport { import_ref: String, detail: String },
    /// One instruction needed more stack items than were available.
    #[error(
        "module-execution stack underflow in function {function_index} at pc {pc} for {context}: needed {needed}, available {available}"
    )]
    StackUnderflow {
        function_index: u32,
        pc: usize,
        context: String,
        needed: usize,
        available: usize,
    },
    /// Execution exceeded the bounded step limit.
    #[error("module-execution exceeded the step limit of {max_steps}")]
    StepLimitExceeded { max_steps: usize },
    /// Execution exceeded the bounded call-depth limit.
    #[error("module-execution exceeded max_call_depth={max_call_depth}")]
    CallDepthExceeded { max_call_depth: u32 },
}

/// Executes one validated bounded module-execution program.
pub fn execute_tassadar_module_execution_program(
    program: &TassadarModuleExecutionProgram,
) -> Result<TassadarModuleExecution, TassadarModuleExecutionError> {
    program.validate()?;
    let mut state = ModuleExecutionState::new(program)?;
    let mut halt_reason = TassadarModuleExecutionHaltReason::Returned;
    let returned_value;

    while !state.frames.is_empty() {
        if state.step_index >= TASSADAR_MODULE_EXECUTION_MAX_STEPS {
            return Err(TassadarModuleExecutionError::StepLimitExceeded {
                max_steps: TASSADAR_MODULE_EXECUTION_MAX_STEPS,
            });
        }
        let current_index = state.frames.len() - 1;
        let function_index = state.frames[current_index].function_index;
        let function = &program.functions[function_index as usize];
        if state.frames[current_index].pc >= function.instructions.len() {
            let value =
                finalize_function_return(&mut state.frames[current_index], function, false)?;
            state.frames.pop();
            if let Some(caller) = state.frames.last_mut() {
                if let Some(value) = value {
                    caller.operand_stack.push(value);
                }
                state.push_step(
                    program,
                    TassadarModuleTraceEvent::Return {
                        function_index,
                        value,
                        implicit: true,
                    },
                )?;
                continue;
            }
            halt_reason = TassadarModuleExecutionHaltReason::FellOffEnd;
            returned_value = value;
            state.push_step(
                program,
                TassadarModuleTraceEvent::Return {
                    function_index,
                    value,
                    implicit: true,
                },
            )?;
            return Ok(TassadarModuleExecution {
                program_id: program.program_id.clone(),
                steps: state.steps,
                returned_value,
                final_globals: state.globals,
                halt_reason,
            });
        }

        let instruction = function.instructions[state.frames[current_index].pc].clone();
        match instruction {
            TassadarModuleInstruction::I32Const { value } => {
                state.frames[current_index].operand_stack.push(value);
                state.frames[current_index].pc += 1;
                state.push_step(program, TassadarModuleTraceEvent::ConstPush { value })?;
            }
            TassadarModuleInstruction::LocalGet { local_index } => {
                let value = *state.frames[current_index]
                    .locals
                    .get(local_index as usize)
                    .ok_or(TassadarModuleExecutionError::LocalOutOfRange {
                        function_index,
                        local_index,
                        local_count: function.local_count,
                    })?;
                state.frames[current_index].operand_stack.push(value);
                state.frames[current_index].pc += 1;
                state.push_step(
                    program,
                    TassadarModuleTraceEvent::LocalGet { local_index, value },
                )?;
            }
            TassadarModuleInstruction::LocalSet { local_index } => {
                let value = pop_operand(
                    &mut state.frames[current_index],
                    function_index,
                    "local.set",
                )?;
                *state.frames[current_index]
                    .locals
                    .get_mut(local_index as usize)
                    .ok_or(TassadarModuleExecutionError::LocalOutOfRange {
                        function_index,
                        local_index,
                        local_count: function.local_count,
                    })? = value;
                state.frames[current_index].pc += 1;
                state.push_step(
                    program,
                    TassadarModuleTraceEvent::LocalSet { local_index, value },
                )?;
            }
            TassadarModuleInstruction::GlobalGet { global_index } => {
                let value = *state.globals.get(global_index as usize).ok_or(
                    TassadarModuleExecutionError::GlobalOutOfRange {
                        global_index,
                        global_count: state.globals.len(),
                    },
                )?;
                state.frames[current_index].operand_stack.push(value);
                state.frames[current_index].pc += 1;
                state.push_step(
                    program,
                    TassadarModuleTraceEvent::GlobalGet {
                        global_index,
                        value,
                    },
                )?;
            }
            TassadarModuleInstruction::GlobalSet { global_index } => {
                let value = pop_operand(
                    &mut state.frames[current_index],
                    function_index,
                    "global.set",
                )?;
                let global = program.globals.get(global_index as usize).ok_or(
                    TassadarModuleExecutionError::GlobalOutOfRange {
                        global_index,
                        global_count: program.globals.len(),
                    },
                )?;
                if global.mutability != TassadarModuleGlobalMutability::Mutable {
                    return Err(TassadarModuleExecutionError::ImmutableGlobalWrite {
                        global_index,
                    });
                }
                let global_count = state.globals.len();
                *state.globals.get_mut(global_index as usize).ok_or(
                    TassadarModuleExecutionError::GlobalOutOfRange {
                        global_index,
                        global_count,
                    },
                )? = value;
                state.frames[current_index].pc += 1;
                state.push_step(
                    program,
                    TassadarModuleTraceEvent::GlobalSet {
                        global_index,
                        value,
                    },
                )?;
            }
            TassadarModuleInstruction::BinaryOp { op } => {
                let right = pop_operand(
                    &mut state.frames[current_index],
                    function_index,
                    "binary_op",
                )?;
                let left = pop_operand(
                    &mut state.frames[current_index],
                    function_index,
                    "binary_op",
                )?;
                let result = execute_binary_op(op, left, right);
                state.frames[current_index].operand_stack.push(result);
                state.frames[current_index].pc += 1;
                state.push_step(
                    program,
                    TassadarModuleTraceEvent::BinaryOp {
                        op,
                        left,
                        right,
                        result,
                    },
                )?;
            }
            TassadarModuleInstruction::CallIndirect { table_index } => {
                let selector = pop_operand(
                    &mut state.frames[current_index],
                    function_index,
                    "call_indirect",
                )?;
                let table = program.tables.get(table_index as usize).ok_or(
                    TassadarModuleExecutionError::TableOutOfRange {
                        table_index,
                        table_count: program.tables.len(),
                    },
                )?;
                let function_index = table
                    .elements
                    .get(selector as usize)
                    .ok_or(TassadarModuleExecutionError::TableSelectorOutOfRange {
                        table_index,
                        selector,
                        entry_count: table.elements.len(),
                    })?
                    .ok_or(TassadarModuleExecutionError::EmptyTableEntry {
                        table_index,
                        selector,
                    })?;
                if state.frames.len() >= program.max_call_depth as usize {
                    return Err(TassadarModuleExecutionError::CallDepthExceeded {
                        max_call_depth: program.max_call_depth,
                    });
                }
                let target = &program.functions[function_index as usize];
                state.frames[current_index].pc += 1;
                state.frames.push(ModuleFrame::new(target));
                state.push_step(
                    program,
                    TassadarModuleTraceEvent::CallIndirect {
                        table_index,
                        selector,
                        function_index,
                    },
                )?;
            }
            TassadarModuleInstruction::HostCall { import_index } => {
                let import = program.imports.get(import_index as usize).ok_or(
                    TassadarModuleExecutionError::ImportOutOfRange {
                        import_index,
                        import_count: program.imports.len(),
                    },
                )?;
                state.frames[current_index].pc += 1;
                match import {
                    TassadarHostImportStub::DeterministicI32Const {
                        import_ref, value, ..
                    } => {
                        state.frames[current_index].operand_stack.push(*value);
                        state.push_step(
                            program,
                            TassadarModuleTraceEvent::HostCall {
                                import_ref: import_ref.clone(),
                                stub_kind: TassadarHostImportStubKind::DeterministicI32Const,
                                result: Some(*value),
                            },
                        )?;
                    }
                    TassadarHostImportStub::UnsupportedHostCall {
                        import_ref,
                        refusal_detail,
                        ..
                    } => {
                        return Err(TassadarModuleExecutionError::UnsupportedHostImport {
                            import_ref: import_ref.clone(),
                            detail: refusal_detail.clone(),
                        });
                    }
                }
            }
            TassadarModuleInstruction::Return => {
                let value =
                    finalize_function_return(&mut state.frames[current_index], function, true)?;
                state.frames.pop();
                if let Some(caller) = state.frames.last_mut() {
                    if let Some(value) = value {
                        caller.operand_stack.push(value);
                    }
                    state.push_step(
                        program,
                        TassadarModuleTraceEvent::Return {
                            function_index,
                            value,
                            implicit: false,
                        },
                    )?;
                    continue;
                }
                returned_value = value;
                state.push_step(
                    program,
                    TassadarModuleTraceEvent::Return {
                        function_index,
                        value,
                        implicit: false,
                    },
                )?;
                return Ok(TassadarModuleExecution {
                    program_id: program.program_id.clone(),
                    steps: state.steps,
                    returned_value,
                    final_globals: state.globals,
                    halt_reason,
                });
            }
        }
    }

    Ok(TassadarModuleExecution {
        program_id: program.program_id.clone(),
        steps: state.steps,
        returned_value: None,
        final_globals: state.globals,
        halt_reason: TassadarModuleExecutionHaltReason::Returned,
    })
}

/// Returns one seeded global-state parity program.
#[must_use]
pub fn tassadar_seeded_module_global_state_program() -> TassadarModuleExecutionProgram {
    TassadarModuleExecutionProgram::new(
        "tassadar.module_execution.global_state.v1",
        0,
        8,
        vec![TassadarModuleGlobal {
            global_index: 0,
            value_type: TassadarModuleValueType::I32,
            mutability: TassadarModuleGlobalMutability::Mutable,
            initial_value: 5,
        }],
        Vec::new(),
        Vec::new(),
        vec![TassadarModuleFunction::new(
            0,
            "entry",
            0,
            0,
            1,
            vec![
                TassadarModuleInstruction::GlobalGet { global_index: 0 },
                TassadarModuleInstruction::I32Const { value: 7 },
                TassadarModuleInstruction::BinaryOp {
                    op: TassadarStructuredControlBinaryOp::Add,
                },
                TassadarModuleInstruction::GlobalSet { global_index: 0 },
                TassadarModuleInstruction::GlobalGet { global_index: 0 },
                TassadarModuleInstruction::Return,
            ],
        )],
    )
}

/// Returns one seeded indirect-call program over one bounded table.
#[must_use]
pub fn tassadar_seeded_module_call_indirect_program() -> TassadarModuleExecutionProgram {
    TassadarModuleExecutionProgram::new(
        "tassadar.module_execution.call_indirect.v1",
        0,
        8,
        Vec::new(),
        vec![TassadarModuleTable {
            table_index: 0,
            element_kind: TassadarModuleTableElementKind::Funcref,
            min_entries: 2,
            max_entries: Some(2),
            elements: vec![Some(1), Some(2)],
        }],
        Vec::new(),
        vec![
            TassadarModuleFunction::new(
                0,
                "entry",
                0,
                0,
                1,
                vec![
                    TassadarModuleInstruction::I32Const { value: 1 },
                    TassadarModuleInstruction::CallIndirect { table_index: 0 },
                    TassadarModuleInstruction::Return,
                ],
            ),
            TassadarModuleFunction::new(
                1,
                "slot_zero",
                0,
                0,
                1,
                vec![
                    TassadarModuleInstruction::I32Const { value: 111 },
                    TassadarModuleInstruction::Return,
                ],
            ),
            TassadarModuleFunction::new(
                2,
                "slot_one",
                0,
                0,
                1,
                vec![
                    TassadarModuleInstruction::I32Const { value: 222 },
                    TassadarModuleInstruction::Return,
                ],
            ),
        ],
    )
}

/// Returns one seeded deterministic-import program.
#[must_use]
pub fn tassadar_seeded_module_deterministic_import_program() -> TassadarModuleExecutionProgram {
    TassadarModuleExecutionProgram::new(
        "tassadar.module_execution.import_stub.v1",
        0,
        8,
        Vec::new(),
        Vec::new(),
        vec![TassadarHostImportStub::DeterministicI32Const {
            import_index: 0,
            import_ref: String::from("env.clock_stub"),
            value: 41,
            claim_boundary: String::from(
                "deterministic zero-side-effect stub returning one fixed i32",
            ),
        }],
        vec![TassadarModuleFunction::new(
            0,
            "entry",
            0,
            0,
            1,
            vec![
                TassadarModuleInstruction::HostCall { import_index: 0 },
                TassadarModuleInstruction::I32Const { value: 1 },
                TassadarModuleInstruction::BinaryOp {
                    op: TassadarStructuredControlBinaryOp::Add,
                },
                TassadarModuleInstruction::Return,
            ],
        )],
    )
}

/// Returns one seeded unsupported-host-call refusal program.
#[must_use]
pub fn tassadar_seeded_module_unsupported_host_import_program() -> TassadarModuleExecutionProgram {
    TassadarModuleExecutionProgram::new(
        "tassadar.module_execution.unsupported_host_import.v1",
        0,
        8,
        Vec::new(),
        Vec::new(),
        vec![TassadarHostImportStub::UnsupportedHostCall {
            import_index: 0,
            import_ref: String::from("host.fs_write"),
            refusal_detail: String::from(
                "host calls outside deterministic stub admission remain unsupported",
            ),
        }],
        vec![TassadarModuleFunction::new(
            0,
            "entry",
            0,
            0,
            1,
            vec![
                TassadarModuleInstruction::HostCall { import_index: 0 },
                TassadarModuleInstruction::Return,
            ],
        )],
    )
}

struct ModuleFrame {
    function_index: u32,
    pc: usize,
    locals: Vec<i32>,
    operand_stack: Vec<i32>,
}

impl ModuleFrame {
    fn new(function: &TassadarModuleFunction) -> Self {
        Self {
            function_index: function.function_index,
            pc: 0,
            locals: vec![0; function.local_count],
            operand_stack: Vec::new(),
        }
    }
}

struct ModuleExecutionState {
    globals: Vec<i32>,
    frames: Vec<ModuleFrame>,
    steps: Vec<TassadarModuleTraceStep>,
    step_index: usize,
}

impl ModuleExecutionState {
    fn new(program: &TassadarModuleExecutionProgram) -> Result<Self, TassadarModuleExecutionError> {
        let entry = &program.functions[program.entry_function_index as usize];
        Ok(Self {
            globals: program
                .globals
                .iter()
                .map(|global| global.initial_value)
                .collect(),
            frames: vec![ModuleFrame::new(entry)],
            steps: Vec::new(),
            step_index: 0,
        })
    }

    fn push_step(
        &mut self,
        program: &TassadarModuleExecutionProgram,
        event: TassadarModuleTraceEvent,
    ) -> Result<(), TassadarModuleExecutionError> {
        if self.step_index >= TASSADAR_MODULE_EXECUTION_MAX_STEPS {
            return Err(TassadarModuleExecutionError::StepLimitExceeded {
                max_steps: TASSADAR_MODULE_EXECUTION_MAX_STEPS,
            });
        }
        self.steps.push(TassadarModuleTraceStep {
            step_index: self.step_index,
            frame_depth_after: self.frames.len(),
            event,
            globals_after: self.globals.clone(),
            frame_stack_after: self
                .frames
                .iter()
                .map(|frame| TassadarModuleFrameSnapshot {
                    function_index: frame.function_index,
                    function_name: program.functions[frame.function_index as usize]
                        .function_name
                        .clone(),
                    pc: frame.pc,
                    locals: frame.locals.clone(),
                    operand_stack: frame.operand_stack.clone(),
                })
                .collect(),
        });
        self.step_index = self.step_index.saturating_add(1);
        Ok(())
    }
}

fn pop_operand(
    frame: &mut ModuleFrame,
    function_index: u32,
    context: &str,
) -> Result<i32, TassadarModuleExecutionError> {
    let available = frame.operand_stack.len();
    frame
        .operand_stack
        .pop()
        .ok_or_else(|| TassadarModuleExecutionError::StackUnderflow {
            function_index,
            pc: frame.pc,
            context: String::from(context),
            needed: 1,
            available,
        })
}

fn finalize_function_return(
    frame: &mut ModuleFrame,
    function: &TassadarModuleFunction,
    explicit: bool,
) -> Result<Option<i32>, TassadarModuleExecutionError> {
    let value = match function.result_count {
        0 => None,
        1 => Some(pop_operand(
            frame,
            function.function_index,
            if explicit {
                "return"
            } else {
                "implicit_return"
            },
        )?),
        other => {
            return Err(TassadarModuleExecutionError::UnsupportedResultCount {
                function_index: function.function_index,
                result_count: other,
            });
        }
    };
    Ok(value)
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

#[cfg(test)]
mod tests {
    use super::{
        TassadarHostImportStubKind, TassadarModuleExecutionRefusalKind,
        TassadarModuleGlobalMutability, TassadarModuleValueType,
        execute_tassadar_module_execution_program, tassadar_module_execution_capability_report,
        tassadar_seeded_module_call_indirect_program,
        tassadar_seeded_module_deterministic_import_program,
        tassadar_seeded_module_global_state_program,
        tassadar_seeded_module_unsupported_host_import_program,
    };

    #[test]
    fn module_execution_capability_report_is_machine_legible() {
        let report = tassadar_module_execution_capability_report();
        assert!(report.supports_globals);
        assert!(report.supports_call_indirect);
        assert_eq!(
            report.supported_global_value_types,
            vec![TassadarModuleValueType::I32]
        );
        assert_eq!(
            report.host_import_boundary.supported_stub_kinds,
            vec![TassadarHostImportStubKind::DeterministicI32Const]
        );
        assert_eq!(
            report.host_import_boundary.unsupported_host_call_refusal,
            TassadarModuleExecutionRefusalKind::UnsupportedHostImport
        );
    }

    #[test]
    fn module_execution_global_state_parity_is_exact() {
        let program = tassadar_seeded_module_global_state_program();
        assert_eq!(
            program.globals[0].mutability,
            TassadarModuleGlobalMutability::Mutable
        );
        let execution = execute_tassadar_module_execution_program(&program).expect("execute");
        assert_eq!(execution.returned_value, Some(12));
        assert_eq!(execution.final_globals, vec![12]);
    }

    #[test]
    fn module_execution_call_indirect_dispatches_exactly() {
        let program = tassadar_seeded_module_call_indirect_program();
        let execution = execute_tassadar_module_execution_program(&program).expect("execute");
        assert_eq!(execution.returned_value, Some(222));
        assert!(execution.steps.iter().any(|step| matches!(
            step.event,
            super::TassadarModuleTraceEvent::CallIndirect { .. }
        )));
    }

    #[test]
    fn module_execution_deterministic_import_stub_is_exact() {
        let program = tassadar_seeded_module_deterministic_import_program();
        let execution = execute_tassadar_module_execution_program(&program).expect("execute");
        assert_eq!(execution.returned_value, Some(42));
    }

    #[test]
    fn module_execution_unsupported_host_call_refuses_explicitly() {
        let error = execute_tassadar_module_execution_program(
            &tassadar_seeded_module_unsupported_host_import_program(),
        )
        .expect_err("host call should refuse");
        assert!(matches!(
            error,
            super::TassadarModuleExecutionError::UnsupportedHostImport { .. }
        ));
    }
}
