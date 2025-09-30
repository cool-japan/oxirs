//! Dataset handling for neural architecture search

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training data provider trait
pub trait TrainingDataProvider: Send + Sync {
    fn get_batch(&self, batch_size: usize) -> Result<TrainingBatch>;
    fn get_full_dataset(&self) -> Result<TrainingDataset>;
    fn get_dataset_info(&self) -> DatasetInfo;
}

/// Validation data provider trait
pub trait ValidationDataProvider: Send + Sync {
    fn get_validation_set(&self) -> Result<ValidationDataset>;
    fn get_test_set(&self) -> Result<TestDataset>;
}

/// Downstream task evaluator trait
pub trait DownstreamTaskEvaluator: Send + Sync {
    fn evaluate(&self, embeddings: &[Vec<f32>], labels: &[usize]) -> Result<f64>;
    fn get_task_name(&self) -> &str;
    fn get_evaluation_config(&self) -> &TaskEvaluationConfig;
}

/// Training batch
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Input data
    pub inputs: Vec<Vec<f32>>,
    /// Target embeddings or labels
    pub targets: Vec<Vec<f32>>,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Training dataset
#[derive(Debug, Clone)]
pub struct TrainingDataset {
    /// All training samples
    pub samples: Vec<TrainingBatch>,
    /// Dataset statistics
    pub statistics: DatasetStatistics,
}

/// Validation dataset
#[derive(Debug, Clone)]
pub struct ValidationDataset {
    /// Validation samples
    pub samples: Vec<TrainingBatch>,
    /// Ground truth metrics
    pub ground_truth: HashMap<String, Vec<f64>>,
}

/// Test dataset
#[derive(Debug, Clone)]
pub struct TestDataset {
    /// Test samples
    pub samples: Vec<TrainingBatch>,
    /// Reference embeddings
    pub reference_embeddings: Vec<Vec<f32>>,
}

/// Dataset information
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Number of samples
    pub num_samples: usize,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Data type
    pub data_type: DataType,
    /// Domain information
    pub domain: String,
}

/// Data types
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Text,
    Image,
    Audio,
    Video,
    Tabular,
    Graph,
    TimeSeries,
    MultiModal,
}

/// Dataset statistics
#[derive(Debug, Clone)]
pub struct DatasetStatistics {
    /// Mean and standard deviation
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    /// Min and max values
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    /// Distribution properties
    pub skewness: Vec<f64>,
    pub kurtosis: Vec<f64>,
}

/// Task evaluation configuration
#[derive(Debug, Clone)]
pub struct TaskEvaluationConfig {
    /// Task type
    pub task_type: TaskType,
    /// Evaluation metrics
    pub metrics: Vec<String>,
    /// Cross-validation settings
    pub cv_folds: usize,
    /// Random seed
    pub random_seed: u64,
}

/// Task types for downstream evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    Classification,
    Regression,
    Clustering,
    SimilaritySearch,
    Retrieval,
    Recommendation,
}

/// Evaluation budget constraints
#[derive(Debug, Clone)]
pub struct EvaluationBudget {
    /// Maximum number of evaluations
    pub max_evaluations: usize,
    /// Maximum wall clock time
    pub max_time_minutes: usize,
    /// Maximum GPU hours
    pub max_gpu_hours: f64,
    /// Maximum CPU hours
    pub max_cpu_hours: f64,
    /// Maximum memory usage
    pub max_memory_gb: f64,
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Stratified sampling
    pub stratified: bool,
    /// Random seed
    pub random_seed: u64,
}