//! Experiment Tracking System for ML Operations
//!
//! This module provides comprehensive experiment tracking for machine learning
//! operations, including parameter logging, metric tracking, artifact management,
//! and experiment comparison capabilities.
//!
//! # Features
//!
//! - **Experiment Management**: Create, update, and organize experiments
//! - **Parameter Tracking**: Log hyperparameters and configuration
//! - **Metric Logging**: Track training and validation metrics over time
//! - **Artifact Storage**: Store and retrieve model artifacts
//! - **Experiment Comparison**: Compare multiple experiments side-by-side
//! - **Visualization**: Generate plots and reports
//! - **Search and Filter**: Query experiments by parameters and metrics
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_shacl_ai::experiment_tracking::{ExperimentTracker, Parameter, Metric};
//!
//! let tracker = ExperimentTracker::new();
//!
//! // Create an experiment
//! let experiment_id = tracker.create_experiment(
//!     "SHACL Shape Learning",
//!     "Learning shapes from RDF data with GNN"
//! ).unwrap();
//!
//! // Log parameters
//! tracker.log_parameter(&experiment_id, Parameter::Float {
//!     name: "learning_rate".to_string(),
//!     value: 0.001,
//! }).unwrap();
//!
//! // Log metrics
//! tracker.log_metric(&experiment_id, Metric {
//!     name: "accuracy".to_string(),
//!     value: 0.95,
//!     step: Some(100),
//!     timestamp: chrono::Utc::now(),
//! }).unwrap();
//! ```

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use thiserror::Error;
use uuid::Uuid;

use crate::{Result, ShaclAiError};

/// Experiment tracking error types
#[derive(Debug, Error)]
pub enum ExperimentError {
    #[error("Experiment not found: {0}")]
    ExperimentNotFound(String),

    #[error("Run not found: {0}")]
    RunNotFound(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Invalid metric: {0}")]
    InvalidMetric(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<ExperimentError> for ShaclAiError {
    fn from(err: ExperimentError) -> Self {
        ShaclAiError::DataProcessing(err.to_string())
    }
}

/// Experiment tracker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Base directory for artifact storage
    pub artifact_base_dir: PathBuf,

    /// Enable metric history tracking
    pub enable_metric_history: bool,

    /// Maximum metrics per experiment
    pub max_metrics_per_experiment: usize,

    /// Enable automatic checkpointing
    pub enable_checkpointing: bool,

    /// Checkpoint interval (number of steps)
    pub checkpoint_interval: usize,

    /// Enable experiment comparison
    pub enable_comparison: bool,

    /// Maximum experiments to compare at once
    pub max_comparison_size: usize,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            artifact_base_dir: PathBuf::from("/tmp/oxirs_experiments"),
            enable_metric_history: true,
            max_metrics_per_experiment: 100000,
            enable_checkpointing: true,
            checkpoint_interval: 1000,
            enable_comparison: true,
            max_comparison_size: 10,
        }
    }
}

/// Experiment status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    /// Experiment is being prepared
    Preparing,

    /// Experiment is currently running
    Running,

    /// Experiment completed successfully
    Completed,

    /// Experiment failed
    Failed,

    /// Experiment was canceled
    Canceled,
}

/// Experiment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Unique experiment ID
    pub id: String,

    /// Experiment name
    pub name: String,

    /// Experiment description
    pub description: String,

    /// Experiment status
    pub status: ExperimentStatus,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Start timestamp
    pub started_at: Option<DateTime<Utc>>,

    /// End timestamp
    pub ended_at: Option<DateTime<Utc>>,

    /// Tags for organization
    pub tags: Vec<String>,

    /// User/creator
    pub user: String,

    /// Parent experiment (for nested experiments)
    pub parent_id: Option<String>,

    /// Associated runs
    pub runs: Vec<String>,

    /// Experiment notes
    pub notes: String,
}

/// Experiment run (a single execution of an experiment)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRun {
    /// Unique run ID
    pub id: String,

    /// Parent experiment ID
    pub experiment_id: String,

    /// Run name
    pub name: String,

    /// Run status
    pub status: ExperimentStatus,

    /// Parameters used in this run
    pub parameters: HashMap<String, Parameter>,

    /// Metrics logged during this run
    pub metrics: HashMap<String, Vec<Metric>>,

    /// Artifacts generated by this run
    pub artifacts: Vec<Artifact>,

    /// Start timestamp
    pub started_at: DateTime<Utc>,

    /// End timestamp
    pub ended_at: Option<DateTime<Utc>>,

    /// Duration (seconds)
    pub duration_seconds: Option<f64>,

    /// Git commit hash (if applicable)
    pub git_commit: Option<String>,

    /// Run notes
    pub notes: String,
}

/// Parameter type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ParameterType {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
}

/// Parameter with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,

    /// Parameter value
    pub value: ParameterType,

    /// Parameter description
    pub description: Option<String>,
}

/// Metric type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum MetricType {
    /// Training metric
    #[default]
    Training,

    /// Validation metric
    Validation,

    /// Test metric
    Test,

    /// System metric (memory, CPU, etc.)
    System,

    /// Custom metric
    Custom,
}

/// Metric with timestamp and step information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// Metric name
    pub name: String,

    /// Metric value
    pub value: f64,

    /// Training step (optional)
    pub step: Option<usize>,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Metric type
    #[serde(default)]
    pub metric_type: MetricType,
}

/// Artifact type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Model file
    Model,

    /// Dataset file
    Dataset,

    /// Plot/visualization
    Plot,

    /// Log file
    Log,

    /// Configuration file
    Config,

    /// Other artifact
    Other,
}

/// Artifact metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact name
    pub name: String,

    /// Artifact type
    pub artifact_type: ArtifactType,

    /// File path
    pub path: PathBuf,

    /// File size (bytes)
    pub size_bytes: usize,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Description
    pub description: String,
}

/// Experiment metrics summary
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExperimentMetrics {
    /// Total experiments
    pub total_experiments: usize,

    /// Total runs
    pub total_runs: usize,

    /// Running experiments
    pub running_experiments: usize,

    /// Completed experiments
    pub completed_experiments: usize,

    /// Failed experiments
    pub failed_experiments: usize,

    /// Total parameters logged
    pub total_parameters_logged: usize,

    /// Total metrics logged
    pub total_metrics_logged: usize,

    /// Total artifacts stored
    pub total_artifacts_stored: usize,
}

/// Main experiment tracker
#[derive(Debug)]
pub struct ExperimentTracker {
    /// Configuration
    config: ExperimentConfig,

    /// Experiments storage (experiment_id -> experiment)
    experiments: Arc<DashMap<String, Experiment>>,

    /// Runs storage (run_id -> run)
    runs: Arc<DashMap<String, ExperimentRun>>,

    /// Active run ID (for convenience)
    active_run: Arc<RwLock<Option<String>>>,

    /// Tracker metrics
    metrics: Arc<RwLock<ExperimentMetrics>>,
}

impl ExperimentTracker {
    /// Create a new experiment tracker with default configuration
    pub fn new() -> Self {
        Self::with_config(ExperimentConfig::default())
    }

    /// Create a new experiment tracker with custom configuration
    pub fn with_config(config: ExperimentConfig) -> Self {
        Self {
            config,
            experiments: Arc::new(DashMap::new()),
            runs: Arc::new(DashMap::new()),
            active_run: Arc::new(RwLock::new(None)),
            metrics: Arc::new(RwLock::new(ExperimentMetrics::default())),
        }
    }

    /// Create a new experiment
    pub fn create_experiment(&self, name: &str, description: &str) -> Result<String> {
        let experiment = Experiment {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: description.to_string(),
            status: ExperimentStatus::Preparing,
            created_at: Utc::now(),
            started_at: None,
            ended_at: None,
            tags: Vec::new(),
            user: "oxirs-shacl-ai".to_string(),
            parent_id: None,
            runs: Vec::new(),
            notes: String::new(),
        };

        let id = experiment.id.clone();
        self.experiments.insert(id.clone(), experiment);

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_experiments += 1;
        }

        tracing::info!("Created experiment: {} ({})", name, id);
        Ok(id)
    }

    /// Start an experiment run
    pub fn start_run(&self, experiment_id: &str, run_name: &str) -> Result<String> {
        // Verify experiment exists
        if !self.experiments.contains_key(experiment_id) {
            return Err(ExperimentError::ExperimentNotFound(experiment_id.to_string()).into());
        }

        let run = ExperimentRun {
            id: Uuid::new_v4().to_string(),
            experiment_id: experiment_id.to_string(),
            name: run_name.to_string(),
            status: ExperimentStatus::Running,
            parameters: HashMap::new(),
            metrics: HashMap::new(),
            artifacts: Vec::new(),
            started_at: Utc::now(),
            ended_at: None,
            duration_seconds: None,
            git_commit: None,
            notes: String::new(),
        };

        let run_id = run.id.clone();

        // Add run to experiment
        if let Some(mut experiment) = self.experiments.get_mut(experiment_id) {
            experiment.runs.push(run_id.clone());
            experiment.status = ExperimentStatus::Running;
            if experiment.started_at.is_none() {
                experiment.started_at = Some(Utc::now());
            }
        }

        self.runs.insert(run_id.clone(), run);

        // Set as active run
        if let Ok(mut active) = self.active_run.write() {
            *active = Some(run_id.clone());
        }

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_runs += 1;
            metrics.running_experiments += 1;
        }

        tracing::info!("Started run: {} for experiment {}", run_name, experiment_id);
        Ok(run_id)
    }

    /// End an experiment run
    pub fn end_run(&self, run_id: &str, status: ExperimentStatus) -> Result<()> {
        if let Some(mut run) = self.runs.get_mut(run_id) {
            run.status = status.clone();
            run.ended_at = Some(Utc::now());
            run.duration_seconds =
                Some((run.ended_at.unwrap() - run.started_at).num_seconds() as f64);

            // Update experiment status
            if let Some(mut experiment) = self.experiments.get_mut(&run.experiment_id) {
                experiment.status = status.clone();
                experiment.ended_at = Some(Utc::now());
            }

            // Clear active run if it matches
            if let Ok(mut active) = self.active_run.write() {
                if active.as_ref() == Some(&run_id.to_string()) {
                    *active = None;
                }
            }

            // Update metrics
            if let Ok(mut metrics) = self.metrics.write() {
                metrics.running_experiments = metrics.running_experiments.saturating_sub(1);
                match status {
                    ExperimentStatus::Completed => metrics.completed_experiments += 1,
                    ExperimentStatus::Failed => metrics.failed_experiments += 1,
                    _ => {}
                }
            }

            tracing::info!("Ended run: {} with status {:?}", run_id, status);
            Ok(())
        } else {
            Err(ExperimentError::RunNotFound(run_id.to_string()).into())
        }
    }

    /// Log a parameter for a run
    pub fn log_parameter(&self, run_id: &str, parameter: Parameter) -> Result<()> {
        if let Some(mut run) = self.runs.get_mut(run_id) {
            run.parameters.insert(parameter.name.clone(), parameter);

            // Update metrics
            if let Ok(mut metrics) = self.metrics.write() {
                metrics.total_parameters_logged += 1;
            }

            Ok(())
        } else {
            Err(ExperimentError::RunNotFound(run_id.to_string()).into())
        }
    }

    /// Log a metric for a run
    pub fn log_metric(&self, run_id: &str, metric: Metric) -> Result<()> {
        if let Some(mut run) = self.runs.get_mut(run_id) {
            run.metrics
                .entry(metric.name.clone())
                .or_insert_with(Vec::new)
                .push(metric);

            // Update metrics
            if let Ok(mut metrics) = self.metrics.write() {
                metrics.total_metrics_logged += 1;
            }

            Ok(())
        } else {
            Err(ExperimentError::RunNotFound(run_id.to_string()).into())
        }
    }

    /// Log multiple metrics at once
    pub fn log_metrics(&self, run_id: &str, metrics: Vec<Metric>) -> Result<()> {
        for metric in metrics {
            self.log_metric(run_id, metric)?;
        }
        Ok(())
    }

    /// Add an artifact to a run
    pub fn log_artifact(&self, run_id: &str, artifact: Artifact) -> Result<()> {
        if let Some(mut run) = self.runs.get_mut(run_id) {
            run.artifacts.push(artifact);

            // Update metrics
            if let Ok(mut metrics) = self.metrics.write() {
                metrics.total_artifacts_stored += 1;
            }

            Ok(())
        } else {
            Err(ExperimentError::RunNotFound(run_id.to_string()).into())
        }
    }

    /// Get an experiment by ID
    pub fn get_experiment(&self, experiment_id: &str) -> Option<Experiment> {
        self.experiments.get(experiment_id).map(|e| e.clone())
    }

    /// Get a run by ID
    pub fn get_run(&self, run_id: &str) -> Option<ExperimentRun> {
        self.runs.get(run_id).map(|r| r.clone())
    }

    /// Get the active run ID
    pub fn get_active_run(&self) -> Option<String> {
        self.active_run.read().ok()?.clone()
    }

    /// List all experiments
    pub fn list_experiments(&self) -> Vec<Experiment> {
        self.experiments
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// List runs for an experiment
    pub fn list_runs(&self, experiment_id: &str) -> Vec<ExperimentRun> {
        self.runs
            .iter()
            .filter(|entry| entry.value().experiment_id == experiment_id)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Search experiments by name
    pub fn search_experiments(&self, query: &str) -> Vec<Experiment> {
        self.experiments
            .iter()
            .filter(|entry| {
                entry
                    .value()
                    .name
                    .to_lowercase()
                    .contains(&query.to_lowercase())
                    || entry
                        .value()
                        .description
                        .to_lowercase()
                        .contains(&query.to_lowercase())
            })
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Compare runs
    pub fn compare_runs(&self, run_ids: &[String]) -> Result<ComparisonResult> {
        if run_ids.len() > self.config.max_comparison_size {
            return Err(ExperimentError::InvalidParameter(format!(
                "Cannot compare more than {} runs",
                self.config.max_comparison_size
            ))
            .into());
        }

        let mut runs = Vec::new();
        for run_id in run_ids {
            if let Some(run) = self.get_run(run_id) {
                runs.push(run);
            } else {
                return Err(ExperimentError::RunNotFound(run_id.clone()).into());
            }
        }

        Ok(ComparisonResult {
            runs,
            parameter_comparison: HashMap::new(),
            metric_comparison: HashMap::new(),
            generated_at: Utc::now(),
        })
    }

    /// Get tracker metrics
    pub fn get_metrics(&self) -> Result<ExperimentMetrics> {
        Ok(self
            .metrics
            .read()
            .map_err(|e| ExperimentError::StorageError(format!("Failed to read metrics: {}", e)))?
            .clone())
    }

    /// Delete an experiment and all its runs
    pub fn delete_experiment(&self, experiment_id: &str) -> Result<()> {
        // Delete all runs for this experiment
        let runs_to_delete: Vec<String> = self
            .runs
            .iter()
            .filter(|entry| entry.value().experiment_id == experiment_id)
            .map(|entry| entry.key().clone())
            .collect();

        for run_id in runs_to_delete {
            self.runs.remove(&run_id);
        }

        // Delete experiment
        self.experiments.remove(experiment_id);

        tracing::info!("Deleted experiment: {}", experiment_id);
        Ok(())
    }
}

impl Default for ExperimentTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison result for multiple runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Runs being compared
    pub runs: Vec<ExperimentRun>,

    /// Parameter comparison
    pub parameter_comparison: HashMap<String, Vec<ParameterType>>,

    /// Metric comparison
    pub metric_comparison: HashMap<String, Vec<f64>>,

    /// Generated timestamp
    pub generated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = ExperimentTracker::new();
        let metrics = tracker.get_metrics().unwrap();
        assert_eq!(metrics.total_experiments, 0);
        assert_eq!(metrics.total_runs, 0);
    }

    #[test]
    fn test_create_experiment() {
        let tracker = ExperimentTracker::new();
        let experiment_id = tracker
            .create_experiment("Test Experiment", "Testing experiment tracking")
            .unwrap();

        let experiment = tracker.get_experiment(&experiment_id);
        assert!(experiment.is_some());
        assert_eq!(experiment.unwrap().name, "Test Experiment");

        let metrics = tracker.get_metrics().unwrap();
        assert_eq!(metrics.total_experiments, 1);
    }

    #[test]
    fn test_start_run() {
        let tracker = ExperimentTracker::new();
        let experiment_id = tracker
            .create_experiment("Test Experiment", "Testing")
            .unwrap();
        let run_id = tracker.start_run(&experiment_id, "Run 1").unwrap();

        let run = tracker.get_run(&run_id);
        assert!(run.is_some());
        assert_eq!(run.unwrap().name, "Run 1");

        let metrics = tracker.get_metrics().unwrap();
        assert_eq!(metrics.total_runs, 1);
        assert_eq!(metrics.running_experiments, 1);
    }

    #[test]
    fn test_log_parameter() {
        let tracker = ExperimentTracker::new();
        let experiment_id = tracker.create_experiment("Test", "Test").unwrap();
        let run_id = tracker.start_run(&experiment_id, "Run 1").unwrap();

        let parameter = Parameter {
            name: "learning_rate".to_string(),
            value: ParameterType::Float(0.001),
            description: Some("Learning rate for optimizer".to_string()),
        };

        tracker.log_parameter(&run_id, parameter).unwrap();

        let run = tracker.get_run(&run_id).unwrap();
        assert_eq!(run.parameters.len(), 1);
        assert!(run.parameters.contains_key("learning_rate"));
    }

    #[test]
    fn test_log_metric() {
        let tracker = ExperimentTracker::new();
        let experiment_id = tracker.create_experiment("Test", "Test").unwrap();
        let run_id = tracker.start_run(&experiment_id, "Run 1").unwrap();

        let metric = Metric {
            name: "accuracy".to_string(),
            value: 0.95,
            step: Some(100),
            timestamp: Utc::now(),
            metric_type: MetricType::Validation,
        };

        tracker.log_metric(&run_id, metric).unwrap();

        let run = tracker.get_run(&run_id).unwrap();
        assert_eq!(run.metrics.len(), 1);
        assert!(run.metrics.contains_key("accuracy"));
        assert_eq!(run.metrics.get("accuracy").unwrap().len(), 1);
    }

    #[test]
    fn test_end_run() {
        let tracker = ExperimentTracker::new();
        let experiment_id = tracker.create_experiment("Test", "Test").unwrap();
        let run_id = tracker.start_run(&experiment_id, "Run 1").unwrap();

        tracker
            .end_run(&run_id, ExperimentStatus::Completed)
            .unwrap();

        let run = tracker.get_run(&run_id).unwrap();
        assert_eq!(run.status, ExperimentStatus::Completed);
        assert!(run.ended_at.is_some());
        assert!(run.duration_seconds.is_some());

        let metrics = tracker.get_metrics().unwrap();
        assert_eq!(metrics.running_experiments, 0);
        assert_eq!(metrics.completed_experiments, 1);
    }

    #[test]
    fn test_search_experiments() {
        let tracker = ExperimentTracker::new();
        tracker
            .create_experiment("SHACL Shape Learning", "Learning shapes")
            .unwrap();
        tracker
            .create_experiment("RDF Validation", "Validating RDF")
            .unwrap();
        tracker
            .create_experiment("Graph Neural Network", "GNN training")
            .unwrap();

        let results = tracker.search_experiments("shape");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "SHACL Shape Learning");

        let results = tracker.search_experiments("validation");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_list_runs() {
        let tracker = ExperimentTracker::new();
        let experiment_id = tracker.create_experiment("Test", "Test").unwrap();

        tracker.start_run(&experiment_id, "Run 1").unwrap();
        tracker.start_run(&experiment_id, "Run 2").unwrap();
        tracker.start_run(&experiment_id, "Run 3").unwrap();

        let runs = tracker.list_runs(&experiment_id);
        assert_eq!(runs.len(), 3);
    }

    #[test]
    fn test_delete_experiment() {
        let tracker = ExperimentTracker::new();
        let experiment_id = tracker.create_experiment("Test", "Test").unwrap();
        tracker.start_run(&experiment_id, "Run 1").unwrap();

        tracker.delete_experiment(&experiment_id).unwrap();

        assert!(tracker.get_experiment(&experiment_id).is_none());
        assert_eq!(tracker.list_runs(&experiment_id).len(), 0);
    }

    #[test]
    fn test_active_run() {
        let tracker = ExperimentTracker::new();
        let experiment_id = tracker.create_experiment("Test", "Test").unwrap();
        let run_id = tracker.start_run(&experiment_id, "Run 1").unwrap();

        let active = tracker.get_active_run();
        assert!(active.is_some());
        assert_eq!(active.unwrap(), run_id);

        tracker
            .end_run(&run_id, ExperimentStatus::Completed)
            .unwrap();

        let active = tracker.get_active_run();
        assert!(active.is_none());
    }
}
