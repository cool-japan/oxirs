//! Machine Learning Predictor for Query Optimization (facade).
//!
//! This module provides ML-based cost prediction using SciRS2 for regression
//! analysis. It replaces heuristic-only cost estimation with learned models to
//! deliver up to 1.75x speedup through better query plan decisions.
//!
//! The implementation has been split into focused submodules to keep each
//! source file below the 2000-line refactor threshold:
//!
//! - [`ml_predictor_features`](super::ml_predictor_features) – value histograms,
//!   cardinality estimator, and feature extractor state.
//! - [`ml_predictor_model`](super::ml_predictor_model) – configuration, model
//!   parameter container, and prediction record types.
//! - [`ml_predictor_training`](super::ml_predictor_training) – the
//!   [`MLPredictor`] struct,
//!   training, prediction, and (de)serialization logic.
//!
//! All public items remain available from this path for backward compatibility
//! through the re-exports below.

pub use crate::advanced_optimizer::ml_predictor_features::*;
pub use crate::advanced_optimizer::ml_predictor_model::*;
pub use crate::advanced_optimizer::ml_predictor_training::*;
