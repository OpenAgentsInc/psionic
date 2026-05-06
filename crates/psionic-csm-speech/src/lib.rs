//! CSM speech serving surface.

extern crate self as psionic_models;

#[path = "../../psionic-models/src/csm_parity.rs"]
mod csm_parity;

#[path = "../../psionic-models/src/csm.rs"]
mod csm;

#[path = "../../psionic-serve/src/tokio_runtime_telemetry_axum.rs"]
mod tokio_runtime_telemetry_axum;

#[path = "../../psionic-serve/src/csm_speech.rs"]
mod csm_speech;

pub use csm::*;
pub use csm_parity::*;
pub use csm_speech::*;
