//! Context Management Configuration

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub sliding_window_size: usize,
    pub max_context_length: usize,
    pub enable_summarization: bool,
    pub summarization_threshold: usize,
    pub enable_topic_tracking: bool,
    pub topic_drift_threshold: f32,
    pub enable_importance_scoring: bool,
    pub memory_optimization_enabled: bool,
    pub adaptive_window_size: bool,
    pub context_compression_ratio: f32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            sliding_window_size: 20,
            max_context_length: 4096,
            enable_summarization: true,
            summarization_threshold: 40,
            enable_topic_tracking: true,
            topic_drift_threshold: 0.7,
            enable_importance_scoring: true,
            memory_optimization_enabled: true,
            adaptive_window_size: true,
            context_compression_ratio: 0.6,
        }
    }
}
