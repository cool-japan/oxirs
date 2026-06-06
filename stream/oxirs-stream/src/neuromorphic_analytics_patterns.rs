//! Neuromorphic Analytics Patterns
//!
//! Temporal pattern detection from spike trains: coincidence detection,
//! polychronization group analysis, and sequence learning.
//!
//! This module re-exports from [`crate::neuromorphic_analytics_learning`], which
//! contains the STDP, Hebbian, BCM, Oja, and online-SGD learning rules used for
//! spike-pattern extraction and classification.

pub use crate::neuromorphic_analytics_learning::{
    apply_stdp_batch, bcm_update, compute_stdp_delta, dopamine_gated_gain, hebbian_update,
    homeostatic_scaling_factor, oja_update, online_firing_rate, online_sgd_update,
    update_plasticity_from_results,
};
