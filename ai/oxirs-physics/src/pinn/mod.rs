//! Physics-Informed Neural Network (PINN) residual corrector.
//!
//! This module ships a tiny ResNet-style feed-forward network used as a
//! *correction term* on top of an existing physics solver. Concretely, given
//! a solver step
//!
//! ```text
//! state_{t+1} = solver_step(state_t)
//! ```
//!
//! the corrector applies a residual correction
//!
//! ```text
//! state_{t+1}^corr = state_{t+1} + nn(state_t)
//! ```
//!
//! The network is intentionally tiny (3 fully-connected layers with ReLU
//! plus a residual skip connection, ~10 K parameters when the feature width
//! is ≈ 32). Weights are loaded from a JSON file produced offline so this
//! crate does not depend on PyTorch or any GPU runtime.
//!
//! Everything in this module is gated behind the `pinn_correction` feature.
//! Without it, the public types are still visible (so users can construct
//! configurations) but [`PinnCorrector::apply`] returns
//! [`PinnError::FeatureDisabled`] so the existing CPU pipeline can take
//! over without conditional compilation at the call site.

pub mod corrector;
pub mod loader;
pub mod residual_model;

pub use corrector::PinnCorrector;
pub use loader::{load_residual_model_from_json, load_residual_model_from_path};
pub use residual_model::{Activation, PinnError, PinnResult, ResidualModel, ResidualModelConfig};
