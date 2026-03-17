//! Optional bounded MLX naming and module-layout facade above Psionic-native
//! surfaces.
//!
//! This crate is intentionally thin. It does not add a second execution path,
//! and it does not claim MLX-identical signatures or blanket upstream
//! compatibility. Instead, it groups supported Psionic-native surfaces under an
//! MLX-like module layout so adoption-facing code can stay separate from the
//! native crates.

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str =
    "optional bounded MLX naming and module-layout facade above Psionic-native surfaces";

/// Frozen upstream MLX window for this bounded compatibility shell.
pub const FROZEN_UPSTREAM_WINDOW: &str = "ml-explore/mlx:v0.31.0..v0.31.1";

/// Boundary note that keeps the naming shell honest.
pub const BOUNDARY_NOTE: &str =
    "This facade is a thin name and module shim over existing Psionic-native surfaces; it does not imply MLX-identical signatures, missing native semantics, or any Python, C, or Swift binding layer.";

/// MLX-like core surface above the native array facade.
pub mod core {
    pub use psionic_array::{
        Array, ArrayBackendCaptureArtifactRequest, ArrayBackendCaptureConfig,
        ArrayBackendCaptureFormat, ArrayBackendCaptureReceipt, ArrayCacheLimitControl,
        ArrayDevice, ArrayError, ArrayRuntimeResourceReport, ArrayStream, EvaluatedArray,
        PendingAsyncEval,
    };
    pub use psionic_core::{DType, Device, Shape, TensorData, TensorSpec};
    use psionic_array::{ArrayBackendDebugSnapshot, ArrayBackendDebugSupport, ArrayContext};

    /// Thin MLX-like context wrapper over the Psionic-native array context.
    #[derive(Clone, Debug)]
    pub struct Context {
        native: ArrayContext,
    }

    impl Context {
        /// Creates a CPU-backed context.
        #[must_use]
        pub fn cpu() -> Self {
            Self {
                native: ArrayContext::cpu(),
            }
        }

        /// Creates a seeded CPU-backed context.
        pub fn cpu_seeded(seed: u64) -> Result<Self, ArrayError> {
            Ok(Self {
                native: ArrayContext::cpu_seeded(seed)?,
            })
        }

        /// Creates a Metal-backed context.
        pub fn metal() -> Result<Self, ArrayError> {
            Ok(Self {
                native: ArrayContext::metal()?,
            })
        }

        /// Creates a seeded Metal-backed context.
        pub fn metal_seeded(seed: u64) -> Result<Self, ArrayError> {
            Ok(Self {
                native: ArrayContext::metal_seeded(seed)?,
            })
        }

        /// Creates a CUDA-backed context.
        pub fn cuda() -> Result<Self, ArrayError> {
            Ok(Self {
                native: ArrayContext::cuda()?,
            })
        }

        /// Creates a seeded CUDA-backed context.
        pub fn cuda_seeded(seed: u64) -> Result<Self, ArrayError> {
            Ok(Self {
                native: ArrayContext::cuda_seeded(seed)?,
            })
        }

        /// Returns the underlying Psionic-native context.
        #[must_use]
        pub fn native(&self) -> &ArrayContext {
            &self.native
        }

        /// Consumes the wrapper and returns the underlying Psionic-native context.
        #[must_use]
        pub fn into_native(self) -> ArrayContext {
            self.native
        }

        /// Builds one dense array from a shape plus `f32` values.
        pub fn array(&self, shape: Shape, values: Vec<f32>) -> Result<Array, ArrayError> {
            self.native.constant_f32(shape, values)
        }

        /// Builds one zero-filled dense `f32` array.
        pub fn zeros(&self, shape: Shape) -> Result<Array, ArrayError> {
            self.native.zeros_f32(shape)
        }

        /// Builds one one-filled dense `f32` array.
        pub fn ones(&self, shape: Shape) -> Result<Array, ArrayError> {
            self.native.ones_f32(shape)
        }

        /// Builds one dense `f32` array filled with the given scalar.
        pub fn full(&self, shape: Shape, value: f32) -> Result<Array, ArrayError> {
            self.native.full_f32(shape, value)
        }

        /// Builds one bounded random-uniform dense `f32` array.
        pub fn random_uniform(
            &self,
            shape: Shape,
            min: f32,
            max: f32,
        ) -> Result<Array, ArrayError> {
            self.native.random_uniform_f32(shape, min, max)
        }

        /// Builds one bounded random-normal dense `f32` array.
        pub fn random_normal(
            &self,
            shape: Shape,
            mean: f32,
            stddev: f32,
        ) -> Result<Array, ArrayError> {
            self.native.random_normal_f32(shape, mean, stddev)
        }

        /// Builds one bounded dense `f32` `arange`.
        pub fn arange(&self, start: f32, stop: f32, step: f32) -> Result<Array, ArrayError> {
            self.native.arange_f32(start, stop, step)
        }

        /// Builds one bounded dense `f32` `linspace`.
        pub fn linspace(&self, start: f32, stop: f32, count: usize) -> Result<Array, ArrayError> {
            self.native.linspace_f32(start, stop, count)
        }

        /// Builds one bounded dense identity-like `f32` array.
        pub fn eye(&self, rows: usize, cols: usize) -> Result<Array, ArrayError> {
            self.native.eye_f32(rows, cols)
        }

        /// Materializes one or more arrays through the bounded Psionic eval path.
        pub fn eval(&self, outputs: &[Array]) -> Result<Vec<EvaluatedArray>, ArrayError> {
            self.native.eval(outputs)
        }

        /// Starts one deferred bounded eval.
        pub fn async_eval(&self, outputs: &[Array]) -> Result<PendingAsyncEval, ArrayError> {
            self.native.async_eval(outputs)
        }

        /// Returns the selected device handle.
        #[must_use]
        pub fn device_handle(&self) -> &ArrayDevice {
            self.native.device_handle()
        }

        /// Returns the bounded runtime resource report.
        #[must_use]
        pub fn runtime_resource_report(&self) -> ArrayRuntimeResourceReport {
            self.native.runtime_resource_report()
        }

        /// Returns the bounded backend-debug support snapshot.
        #[must_use]
        pub fn backend_debug_support(&self) -> ArrayBackendDebugSupport {
            self.native.backend_debug_support()
        }

        /// Returns the bounded backend-debug snapshot.
        #[must_use]
        pub fn backend_debug_snapshot(&self) -> ArrayBackendDebugSnapshot {
            self.native.backend_debug_snapshot()
        }

        /// Applies one bounded cache limit change.
        #[must_use]
        pub fn configure_cache_limits(
            &self,
            limits: ArrayCacheLimitControl,
        ) -> ArrayRuntimeResourceReport {
            self.native.configure_cache_limits(limits)
        }
    }

    impl From<ArrayContext> for Context {
        fn from(native: ArrayContext) -> Self {
            Self { native }
        }
    }

    /// Returns one CPU-backed context.
    #[must_use]
    pub fn cpu() -> Context {
        Context::cpu()
    }

    /// Returns one seeded CPU-backed context.
    pub fn cpu_seeded(seed: u64) -> Result<Context, ArrayError> {
        Context::cpu_seeded(seed)
    }

    /// Returns one Metal-backed context.
    pub fn metal() -> Result<Context, ArrayError> {
        Context::metal()
    }

    /// Returns one seeded Metal-backed context.
    pub fn metal_seeded(seed: u64) -> Result<Context, ArrayError> {
        Context::metal_seeded(seed)
    }

    /// Returns one CUDA-backed context.
    pub fn cuda() -> Result<Context, ArrayError> {
        Context::cuda()
    }

    /// Returns one seeded CUDA-backed context.
    pub fn cuda_seeded(seed: u64) -> Result<Context, ArrayError> {
        Context::cuda_seeded(seed)
    }
}

/// MLX-like transform surface over the native autodiff and compile crates.
pub mod transforms {
    pub use psionic_compiler::{
        CompileTransform, CompileTransformConfig, CompileTransformDebugMode,
        CompileTransformError, CompileTransformTraceMode, compile_transform,
    };
    pub use psionic_ir::{
        AutodiffGraph, AutodiffGraphBuilder, CustomVjpTransform, CustomVjpTransformResult,
        VmapSupport, VmapTransformError, VmapUnsupportedReason, checkpoint, custom_vjp, grad, jvp,
        value_and_grad, vjp, vmap,
    };
}

/// MLX-like module layout for the native `nn` surface.
pub mod nn {
    pub use psionic_nn::*;
}

/// MLX-like optimizer layout split out for discoverability.
pub mod optimizers {
    pub use psionic_nn::{
        MultiOptimizer, MultiOptimizerStepReport, Optimizer, OptimizerConfig, OptimizerGroup,
        OptimizerKind, OptimizerModuleStepReport, OptimizerParameterState,
        OptimizerParameterStepReport, OptimizerStateSnapshot, SchedulerBinding, SchedulerConfig,
        SchedulerKind,
    };
}

/// MLX-like function/export compatibility layout.
pub mod io {
    pub use psionic_function_io::*;
}

/// MLX-like distributed layout above the native distributed surface.
pub mod distributed {
    pub use psionic_distributed::*;
}

/// Repo-owned compatibility reports that keep this facade bounded and reviewable.
pub mod reports {
    pub use psionic_compat::{
        MlxCompatibilityMatrixEntry, MlxCompatibilityMatrixReport,
        MlxCompatibilityMatrixStatus, MlxCompatibilityScopeReport,
        builtin_mlx_compatibility_matrix_report, builtin_mlx_compatibility_scope_report,
    };
}

/// Small prelude for the most common facade entrypoints.
pub mod prelude {
    pub use crate::core::{Array, Context};
    pub use crate::nn::Module;
}

#[cfg(test)]
mod tests {
    use super::{core, nn, optimizers, reports};
    use psionic_core::Shape;

    #[test]
    fn core_context_helpers_build_and_eval_bounded_cpu_arrays()
    -> Result<(), Box<dyn std::error::Error>> {
        let context = core::cpu_seeded(7)?;
        let ones = context.ones(Shape::new(vec![2, 2]))?;
        let filled = context.full(Shape::new(vec![2, 2]), 2.0)?;
        let summed = ones.add(&filled)?.sum_axis(1)?;
        let evaluated = summed.eval()?;
        assert_eq!(evaluated.spec().shape(), &Shape::new(vec![2]));
        Ok(())
    }

    #[test]
    fn module_layout_reexports_supported_public_types() {
        let _ = std::any::type_name::<core::Context>();
        let _ = std::any::type_name::<nn::Linear>();
        let _ = std::any::type_name::<optimizers::Optimizer>();
    }

    #[test]
    fn compatibility_reports_mark_the_naming_facade_as_bounded_and_late_surface() {
        let report = reports::builtin_mlx_compatibility_matrix_report();
        let naming = report
            .surfaces
            .iter()
            .find(|surface| surface.surface_id == "mlx_naming_facade_and_bindings")
            .expect("missing naming facade row");
        assert_eq!(
            naming.matrix_status,
            reports::MlxCompatibilityMatrixStatus::Convertible
        );
        assert!(
            naming
                .blocking_issue_refs
                .iter()
                .all(|issue| !issue.contains("PMLX-606"))
        );
        assert!(
            naming
                .summary
                .contains("psionic-mlx-compat")
        );
    }
}
