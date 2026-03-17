//! Optional bounded C ABI over `psionic-mlx-compat`.
//!
//! The binding stays intentionally small: it exposes compatibility reports and
//! one JSON-driven dense `f32` eval bridge over the bounded `psionic-mlx-compat`
//! core surface. That keeps the Rust-native substrate authoritative while still
//! giving C, Python-ctypes, or Swift callers a real interop lane.

use std::{
    collections::BTreeMap,
    ffi::{CStr, CString, c_char},
};

use psionic_core::{PsionicRefusal, Shape};
use psionic_mlx_compat::{core, reports};
use serde::{Deserialize, Serialize};

/// Human-readable crate ownership summary.
pub const CRATE_ROLE: &str = "optional bounded C ABI over psionic-mlx-compat";

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BindingBackend {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Clone, Debug, Deserialize)]
struct EvalRequest {
    backend: BindingBackend,
    #[serde(default)]
    seed: Option<u64>,
    steps: Vec<EvalStep>,
    output: String,
}

#[derive(Clone, Debug, Deserialize)]
struct EvalStep {
    id: String,
    #[serde(flatten)]
    op: EvalOp,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum EvalOp {
    Array { shape: Vec<usize>, values: Vec<f32> },
    Zeros { shape: Vec<usize> },
    Ones { shape: Vec<usize> },
    Full { shape: Vec<usize>, value: f32 },
    Arange { start: f32, stop: f32, step: f32 },
    Linspace { start: f32, stop: f32, count: usize },
    Eye { rows: usize, cols: usize },
    Add { lhs: String, rhs: String },
    Mul { lhs: String, rhs: String },
    Matmul { lhs: String, rhs: String },
    SumAxis { input: String, axis: usize },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum BindingValues {
    F32(Vec<f32>),
    I8(Vec<i8>),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
enum BindingResponse {
    Ok {
        shape: Vec<usize>,
        dtype: String,
        device: String,
        values: BindingValues,
    },
    Refusal {
        refusal: PsionicRefusal,
    },
    Error {
        message: String,
    },
}

fn build_context(backend: BindingBackend, seed: Option<u64>) -> Result<core::Context, String> {
    match (backend, seed) {
        (BindingBackend::Cpu, Some(seed)) => {
            core::cpu_seeded(seed).map_err(|error| error.to_string())
        }
        (BindingBackend::Cpu, None) => Ok(core::cpu()),
        (BindingBackend::Metal, Some(seed)) => {
            core::metal_seeded(seed).map_err(|error| error.to_string())
        }
        (BindingBackend::Metal, None) => core::metal().map_err(|error| error.to_string()),
        (BindingBackend::Cuda, Some(seed)) => {
            core::cuda_seeded(seed).map_err(|error| error.to_string())
        }
        (BindingBackend::Cuda, None) => core::cuda().map_err(|error| error.to_string()),
    }
}

fn get_array<'a>(
    arrays: &'a BTreeMap<String, core::Array>,
    id: &str,
) -> Result<&'a core::Array, BindingResponse> {
    arrays.get(id).ok_or_else(|| BindingResponse::Error {
        message: format!("eval request referenced unknown array id `{id}`"),
    })
}

fn response_from_array_error(error: psionic_mlx_compat::core::ArrayError) -> BindingResponse {
    match error {
        psionic_mlx_compat::core::ArrayError::Graph(graph_error) => {
            if let Some(refusal) = graph_error.refusal() {
                BindingResponse::Refusal { refusal }
            } else {
                BindingResponse::Error {
                    message: graph_error.to_string(),
                }
            }
        }
        other => BindingResponse::Error {
            message: other.to_string(),
        },
    }
}

fn evaluate_request(request: EvalRequest) -> BindingResponse {
    let context = match build_context(request.backend, request.seed) {
        Ok(context) => context,
        Err(message) => return BindingResponse::Error { message },
    };

    let mut arrays = BTreeMap::new();
    for step in request.steps {
        let array = match step.op {
            EvalOp::Array { shape, values } => context.array(Shape::new(shape), values),
            EvalOp::Zeros { shape } => context.zeros(Shape::new(shape)),
            EvalOp::Ones { shape } => context.ones(Shape::new(shape)),
            EvalOp::Full { shape, value } => context.full(Shape::new(shape), value),
            EvalOp::Arange { start, stop, step } => context.arange(start, stop, step),
            EvalOp::Linspace { start, stop, count } => context.linspace(start, stop, count),
            EvalOp::Eye { rows, cols } => context.eye(rows, cols),
            EvalOp::Add { lhs, rhs } => {
                let lhs = match get_array(&arrays, &lhs) {
                    Ok(lhs) => lhs,
                    Err(response) => return response,
                };
                let rhs = match get_array(&arrays, &rhs) {
                    Ok(rhs) => rhs,
                    Err(response) => return response,
                };
                lhs.add(rhs)
            }
            EvalOp::Mul { lhs, rhs } => {
                let lhs = match get_array(&arrays, &lhs) {
                    Ok(lhs) => lhs,
                    Err(response) => return response,
                };
                let rhs = match get_array(&arrays, &rhs) {
                    Ok(rhs) => rhs,
                    Err(response) => return response,
                };
                lhs.mul(rhs)
            }
            EvalOp::Matmul { lhs, rhs } => {
                let lhs = match get_array(&arrays, &lhs) {
                    Ok(lhs) => lhs,
                    Err(response) => return response,
                };
                let rhs = match get_array(&arrays, &rhs) {
                    Ok(rhs) => rhs,
                    Err(response) => return response,
                };
                lhs.matmul(rhs)
            }
            EvalOp::SumAxis { input, axis } => {
                let input = match get_array(&arrays, &input) {
                    Ok(input) => input,
                    Err(response) => return response,
                };
                input.sum_axis(axis)
            }
        };

        let array = match array {
            Ok(array) => array,
            Err(error) => return response_from_array_error(error),
        };
        arrays.insert(step.id, array);
    }

    let Some(output) = arrays.get(&request.output) else {
        return BindingResponse::Error {
            message: format!("eval request missing declared output `{}`", request.output),
        };
    };

    let evaluated = match output.eval() {
        Ok(evaluated) => evaluated,
        Err(error) => return response_from_array_error(error),
    };
    let host = match evaluated.to_host_data() {
        Ok(host) => host,
        Err(error) => return response_from_array_error(error),
    };
    let values = if let Some(values) = host.as_f32_slice() {
        BindingValues::F32(values.to_vec())
    } else if let Some(values) = host.as_i8_slice() {
        BindingValues::I8(values.to_vec())
    } else {
        return BindingResponse::Error {
            message: String::from("evaluated host payload exposed no supported scalar family"),
        };
    };

    BindingResponse::Ok {
        shape: evaluated.shape().dims().to_vec(),
        dtype: format!("{:?}", evaluated.dtype()).to_lowercase(),
        device: evaluated.device().to_string(),
        values,
    }
}

fn serialize_json<T: Serialize>(value: &T) -> *mut c_char {
    match serde_json::to_string(value) {
        Ok(json) => match CString::new(json) {
            Ok(json) => json.into_raw(),
            Err(error) => CString::new(
                serde_json::to_string(&BindingResponse::Error {
                    message: format!("failed to build C string: {error}"),
                })
                .unwrap_or_else(|_| {
                    String::from("{\"status\":\"error\",\"message\":\"failed to serialize error\"}")
                }),
            )
            .expect("static fallback JSON contains no NUL bytes")
            .into_raw(),
        },
        Err(error) => CString::new(
            serde_json::to_string(&BindingResponse::Error {
                message: format!("failed to serialize JSON payload: {error}"),
            })
            .unwrap_or_else(|_| {
                String::from("{\"status\":\"error\",\"message\":\"failed to serialize error\"}")
            }),
        )
        .expect("static fallback JSON contains no NUL bytes")
        .into_raw(),
    }
}

fn parse_request_json(request_json: *const c_char) -> Result<EvalRequest, BindingResponse> {
    if request_json.is_null() {
        return Err(BindingResponse::Error {
            message: String::from("eval request pointer was null"),
        });
    }
    let request = unsafe { CStr::from_ptr(request_json) };
    let request = request.to_str().map_err(|error| BindingResponse::Error {
        message: format!("eval request was not valid UTF-8: {error}"),
    })?;
    serde_json::from_str(request).map_err(|error| BindingResponse::Error {
        message: format!("eval request was not valid JSON: {error}"),
    })
}

fn to_owned_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|error| {
        serde_json::to_string(&BindingResponse::Error {
            message: format!("failed to serialize bounded JSON payload: {error}"),
        })
        .unwrap_or_else(|_| {
            String::from(
                "{\"status\":\"error\",\"message\":\"failed to serialize bounded JSON payload\"}",
            )
        })
    })
}

/// Returns one compatibility-scope report as owned JSON for Rust callers that
/// want the same bounded payload as the C ABI without raw-pointer handling.
#[must_use]
pub fn compatibility_scope_json_string() -> String {
    to_owned_json(&reports::builtin_mlx_compatibility_scope_report())
}

/// Returns one compatibility-matrix report as owned JSON for Rust callers that
/// want the same bounded payload as the C ABI without raw-pointer handling.
#[must_use]
pub fn compatibility_matrix_json_string() -> String {
    to_owned_json(&reports::builtin_mlx_compatibility_matrix_report())
}

/// Evaluates one bounded dense-array request encoded as JSON and returns the
/// same owned JSON payload as the C ABI.
#[must_use]
pub fn eval_json_string(request_json: &str) -> String {
    let response = match serde_json::from_str::<EvalRequest>(request_json) {
        Ok(request) => evaluate_request(request),
        Err(error) => BindingResponse::Error {
            message: format!("eval request was not valid JSON: {error}"),
        },
    };
    to_owned_json(&response)
}

/// Returns one compatibility-scope report as owned JSON.
#[unsafe(no_mangle)]
pub extern "C" fn psionic_mlx_capi_compatibility_scope_json() -> *mut c_char {
    serialize_json(&reports::builtin_mlx_compatibility_scope_report())
}

/// Returns one compatibility-matrix report as owned JSON.
#[unsafe(no_mangle)]
pub extern "C" fn psionic_mlx_capi_compatibility_matrix_json() -> *mut c_char {
    serialize_json(&reports::builtin_mlx_compatibility_matrix_report())
}

/// Evaluates one bounded dense `f32` request encoded as JSON and returns owned JSON.
#[unsafe(no_mangle)]
pub extern "C" fn psionic_mlx_capi_eval_json(request_json: *const c_char) -> *mut c_char {
    match parse_request_json(request_json) {
        Ok(request) => serialize_json(&evaluate_request(request)),
        Err(response) => serialize_json(&response),
    }
}

/// Frees one owned JSON string returned by this C ABI.
#[unsafe(no_mangle)]
pub extern "C" fn psionic_mlx_capi_string_free(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(value));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BindingResponse, BindingValues, compatibility_matrix_json_string, eval_json_string,
        evaluate_request,
    };
    use std::ffi::CString;

    #[test]
    fn eval_request_runs_bounded_cpu_graph() -> Result<(), Box<dyn std::error::Error>> {
        let request = serde_json::from_str(
            r#"{
                "backend": "cpu",
                "seed": 7,
                "steps": [
                    {"id": "lhs", "op": "ones", "shape": [2, 2]},
                    {"id": "rhs", "op": "full", "shape": [2, 2], "value": 2.0},
                    {"id": "sum", "op": "add", "lhs": "lhs", "rhs": "rhs"},
                    {"id": "reduced", "op": "sum_axis", "input": "sum", "axis": 1}
                ],
                "output": "reduced"
            }"#,
        )?;
        let response = evaluate_request(request);
        let BindingResponse::Ok {
            shape,
            dtype,
            device,
            values,
        } = response
        else {
            panic!("expected successful bounded CPU eval response");
        };
        assert_eq!(shape, vec![2]);
        assert_eq!(dtype, "f32");
        assert!(device.starts_with("cpu"));
        assert_eq!(values, BindingValues::F32(vec![6.0, 6.0]));
        Ok(())
    }

    #[test]
    fn eval_request_surfaces_typed_refusal_for_incompatible_matmul()
    -> Result<(), Box<dyn std::error::Error>> {
        let request = serde_json::from_str(
            r#"{
                "backend": "cpu",
                "steps": [
                    {"id": "lhs", "op": "ones", "shape": [2, 2]},
                    {"id": "rhs", "op": "ones", "shape": [3, 2]},
                    {"id": "bad", "op": "matmul", "lhs": "lhs", "rhs": "rhs"}
                ],
                "output": "bad"
            }"#,
        )?;
        let response = evaluate_request(request);
        assert!(matches!(response, BindingResponse::Refusal { .. }));
        Ok(())
    }

    #[test]
    fn ffi_exports_reports_and_eval_json() -> Result<(), Box<dyn std::error::Error>> {
        let request = CString::new(
            r#"{
                "backend": "cpu",
                "steps": [
                    {"id": "values", "op": "arange", "start": 0.0, "stop": 4.0, "step": 1.0}
                ],
                "output": "values"
            }"#,
        )?;

        let scope_ptr = super::psionic_mlx_capi_compatibility_scope_json();
        assert!(!scope_ptr.is_null());
        super::psionic_mlx_capi_string_free(scope_ptr);

        let response_ptr = super::psionic_mlx_capi_eval_json(request.as_ptr());
        assert!(!response_ptr.is_null());
        let response = unsafe { CString::from_raw(response_ptr) };
        let response = response.into_string()?;
        let response: BindingResponse = serde_json::from_str(&response)?;
        assert!(matches!(response, BindingResponse::Ok { .. }));
        Ok(())
    }

    #[test]
    fn safe_rust_wrappers_match_bounded_json_contract() -> Result<(), Box<dyn std::error::Error>> {
        let matrix = compatibility_matrix_json_string();
        let matrix: serde_json::Value = serde_json::from_str(&matrix)?;
        assert_eq!(matrix["schema_version"], 1);

        let response = eval_json_string(
            r#"{
                "backend": "cpu",
                "steps": [
                    {"id": "values", "op": "ones", "shape": [2, 2]}
                ],
                "output": "values"
            }"#,
        );
        let response: BindingResponse = serde_json::from_str(&response)?;
        assert!(matches!(response, BindingResponse::Ok { .. }));
        Ok(())
    }
}
