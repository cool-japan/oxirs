//! Configuration Types for Real-time Adaptive Query Optimizer
//!
//! This module contains configuration structures for the adaptive query optimizer,
//! including settings for quantum optimization, neural transformers, and real-time adaptation.

use serde::{Deserialize, Serialize};

/// Configuration for adaptive query optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveOptimizerConfig {
    /// Enable quantum optimization
    pub enable_quantum_optimization: bool,

    /// Enable neural transformer optimization
    pub enable_neural_transformer: bool,

    /// Enable real-time adaptation
    pub enable_realtime_adaptation: bool,

    /// Enable online learning
    pub enable_online_learning: bool,

    /// Performance monitoring window (in queries)
    pub performance_window_size: usize,

    /// Plan cache size
    pub plan_cache_size: usize,

    /// Adaptation threshold (performance degradation %)
    pub adaptation_threshold: f64,

    /// Learning rate for online adaptation
    pub learning_rate: f64,

    /// Minimum queries before adaptation
    pub min_queries_for_adaptation: usize,

    /// Enable plan precomputation
    pub enable_plan_precomputation: bool,

    /// Maximum parallel optimizations
    pub max_parallel_optimizations: usize,

    /// Enable adaptive complexity analysis
    pub enable_adaptive_complexity: bool,

    /// Query timeout threshold (milliseconds)
    pub query_timeout_threshold: u64,

    /// Enable performance prediction
    pub enable_performance_prediction: bool,

    /// Prediction confidence threshold
    pub prediction_confidence_threshold: f64,
}

impl Default for AdaptiveOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_quantum_optimization: true,
            enable_neural_transformer: true,
            enable_realtime_adaptation: true,
            enable_online_learning: true,
            performance_window_size: 1000,
            plan_cache_size: 10000,
            adaptation_threshold: 0.15, // 15% degradation triggers adaptation
            learning_rate: 0.001,
            min_queries_for_adaptation: 50,
            enable_plan_precomputation: true,
            max_parallel_optimizations: 4,
            enable_adaptive_complexity: true,
            query_timeout_threshold: 30000, // 30 seconds
            enable_performance_prediction: true,
            prediction_confidence_threshold: 0.8,
        }
    }
}