//! Core types for AI orchestration
//!
//! This module contains the fundamental data structures and enums
//! used throughout the AI orchestration system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::ai_orchestrator::config::ModelSelectionStrategy;

/// Data characteristics for adaptive model selection
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub graph_size: usize,
    pub complexity_score: f64,
    pub sparsity_ratio: f64,
    pub hierarchy_depth: u32,
    pub pattern_diversity: f64,
    pub semantic_richness: f64,
}

impl Default for DataCharacteristics {
    fn default() -> Self {
        Self {
            graph_size: 1000,
            complexity_score: 0.5,
            sparsity_ratio: 0.1,
            hierarchy_depth: 3,
            pattern_diversity: 0.7,
            semantic_richness: 0.6,
        }
    }
}

/// Performance requirements for model selection
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    pub min_accuracy: f64,
    pub min_precision: f64,
    pub min_recall: f64,
    pub max_inference_time: Duration,
    pub max_memory_usage: f64,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            min_accuracy: 0.8,
            min_precision: 0.75,
            min_recall: 0.7,
            max_inference_time: Duration::from_millis(100),
            max_memory_usage: 500.0,
        }
    }
}

/// Result of model selection
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    pub models: Vec<SelectedModel>,
    pub selection_strategy: ModelSelectionStrategy,
    pub confidence: f64,
    pub selection_rationale: String,
}

/// Selected model with metadata
#[derive(Debug, Clone)]
pub struct SelectedModel {
    pub model_name: String,
    pub selection_score: f64,
    pub expected_performance: ModelPerformanceMetrics,
    pub selection_reason: String,
}

// ModelSelectionStrategy is defined in config.rs

/// Model performance metrics for selection
#[derive(Debug, Clone)]
pub struct ModelPerformanceMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub training_time: Duration,
    pub inference_time: Duration,
    pub memory_usage: f64,
    pub confidence_calibration: f64,
    pub robustness_score: f64,
}

impl Default for ModelPerformanceMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.8,
            precision: 0.8,
            recall: 0.8,
            f1_score: 0.8,
            training_time: Duration::from_secs(60),
            inference_time: Duration::from_millis(10),
            memory_usage: 100.0,
            confidence_calibration: 0.9,
            robustness_score: 0.85,
        }
    }
}

/// Adaptive learning insights from model orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningInsights {
    pub learning_rate_adaptation: f64,
    pub convergence_indicators: Vec<String>,
    pub adaptation_recommendations: Vec<String>,
    pub performance_trends: HashMap<String, f64>,
}

/// Confident shape with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidentShape {
    pub shape_id: String,
    pub confidence_score: f64,
    pub validation_accuracy: f64,
    pub learning_metadata: LearningMetadata,
}

/// Learning metadata for shape generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMetadata {
    pub iterations: usize,
    pub convergence_time: Duration,
    pub data_quality_score: f64,
    pub learning_algorithm: String,
    pub validation_method: String,
}

/// Orchestration metrics for AI coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationMetrics {
    pub coordination_efficiency: f64,
    pub resource_utilization: f64,
    pub latency_metrics: HashMap<String, Duration>,
    pub throughput_metrics: HashMap<String, f64>,
    pub error_rates: HashMap<String, f64>,
}

/// Quality analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAnalysis {
    pub overall_quality_score: f64,
    pub quality_dimensions: HashMap<String, f64>,
    pub quality_trends: Vec<f64>,
    pub recommendations: Vec<String>,
    pub quality_threshold_met: bool,
}
