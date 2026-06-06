//! Neuromorphic Stream Analytics — facade module
//!
//! Re-exports the full public API from the split sub-modules:
//! - `neuromorphic_analytics_types`    — neuron/synapse types, spike train types, network topology types
//! - `neuromorphic_analytics_network`  — LIF neuron model, spike propagation, synaptic dynamics
//! - `neuromorphic_analytics_learning` — STDP/plasticity learning rules
//! - `neuromorphic_analytics_engine`   — thin alias facade (re-exports from network)
//! - `neuromorphic_analytics_patterns` — thin alias facade (re-exports from learning)
//! - `neuromorphic_analytics_tests`    — integration tests (private)

pub use crate::neuromorphic_analytics_learning::*;
pub use crate::neuromorphic_analytics_network::*;
pub use crate::neuromorphic_analytics_types::*;
