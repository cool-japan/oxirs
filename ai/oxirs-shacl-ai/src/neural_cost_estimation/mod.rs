//! Neural Cost Estimation Engine Module
//!
//! This module provides advanced neural network-based cost estimation for query optimization
//! using multi-dimensional analysis, historical data, and real-time performance feedback.

pub mod config;
pub mod types;
pub mod deep_predictor;
pub mod feature_extractor;
pub mod historical_data;
pub mod feedback;
pub mod ensemble;
pub mod context;
pub mod uncertainty;
pub mod profiler;
pub mod core;

// Re-export main types and structs
pub use config::*;
pub use types::*;
pub use deep_predictor::*;
pub use feature_extractor::*;
pub use historical_data::*;
pub use feedback::*;
pub use ensemble::*;
pub use context::*;
pub use uncertainty::*;
pub use profiler::*;
pub use core::NeuralCostEstimationEngine;