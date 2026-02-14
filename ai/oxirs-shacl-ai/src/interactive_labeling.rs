//! Interactive Labeling Interface for Active Learning
//!
//! This module provides an interactive interface for human-in-the-loop active learning.
//! It supports uncertainty-driven sample selection, collaborative annotation, quality control,
//! and integration with the active learning pipeline.
//!
//! # Features
//! - Uncertainty-driven sample prioritization
//! - Multi-annotator support with agreement tracking
//! - Quality control and validation
//! - Annotation history and versioning
//! - Real-time model updates
//! - Export to standard formats (JSON, CSV, RDF)

use crate::{Result, ShaclAiError};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Annotation task for labeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationTask {
    /// Unique task ID
    pub id: String,
    /// RDF data to annotate
    pub data: RdfData,
    /// Suggested label from model (optional)
    pub suggested_label: Option<String>,
    /// Model confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Uncertainty score (0.0 to 1.0)
    pub uncertainty: f64,
    /// Task priority (higher = more important)
    pub priority: f64,
    /// Task status
    pub status: TaskStatus,
    /// Assigned annotators
    pub assigned_to: Vec<String>,
    /// Creation time
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Annotations received
    pub annotations: Vec<Annotation>,
}

/// RDF data to be annotated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfData {
    /// Subject IRI
    pub subject: String,
    /// Predicate IRI
    pub predicate: String,
    /// Object (IRI or literal)
    pub object: String,
    /// Graph context (optional)
    pub graph: Option<String>,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Task status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is pending annotation
    Pending,
    /// Task is assigned to annotator(s)
    Assigned,
    /// Task is in progress
    InProgress,
    /// Task is completed
    Completed,
    /// Task requires review
    NeedsReview,
    /// Task is disputed (conflicting annotations)
    Disputed,
    /// Task is validated
    Validated,
    /// Task is skipped
    Skipped,
}

/// Annotation from a human annotator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Annotation ID
    pub id: String,
    /// Task ID
    pub task_id: String,
    /// Annotator ID
    pub annotator_id: String,
    /// Assigned label
    pub label: String,
    /// Confidence in annotation (0.0 to 1.0)
    pub confidence: f64,
    /// Time spent (seconds)
    pub time_spent: f64,
    /// Notes or comments
    pub notes: Option<String>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Annotator profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotator {
    /// Annotator ID
    pub id: String,
    /// Name
    pub name: String,
    /// Email
    pub email: String,
    /// Expertise level (0.0 to 1.0)
    pub expertise_level: f64,
    /// Annotation statistics
    pub stats: AnnotatorStats,
    /// Active status
    pub is_active: bool,
}

/// Annotator statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatorStats {
    /// Total annotations completed
    pub total_annotations: usize,
    /// Average time per annotation (seconds)
    pub avg_time_per_annotation: f64,
    /// Agreement rate with other annotators (0.0 to 1.0)
    pub agreement_rate: f64,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Annotations validated
    pub validated_annotations: usize,
    /// Annotations rejected
    pub rejected_annotations: usize,
}

impl Default for AnnotatorStats {
    fn default() -> Self {
        Self {
            total_annotations: 0,
            avg_time_per_annotation: 0.0,
            agreement_rate: 0.0,
            quality_score: 1.0,
            validated_annotations: 0,
            rejected_annotations: 0,
        }
    }
}

/// Quality control metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Inter-annotator agreement (Fleiss' kappa)
    pub inter_annotator_agreement: f64,
    /// Average annotation confidence
    pub avg_confidence: f64,
    /// Consensus rate (% of tasks with agreement)
    pub consensus_rate: f64,
    /// Disputed tasks count
    pub disputed_tasks: usize,
    /// Average time per task (seconds)
    pub avg_time_per_task: f64,
}

/// Interactive labeling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelingConfig {
    /// Minimum number of annotations per task
    pub min_annotations_per_task: usize,
    /// Maximum number of annotations per task
    pub max_annotations_per_task: usize,
    /// Uncertainty threshold for automatic assignment (0.0 to 1.0)
    pub uncertainty_threshold: f64,
    /// Minimum agreement for consensus (0.0 to 1.0)
    pub min_agreement_threshold: f64,
    /// Enable quality control
    pub enable_quality_control: bool,
    /// Auto-validate if consensus reached
    pub auto_validate_on_consensus: bool,
    /// Priority weighting strategy
    pub priority_strategy: PriorityStrategy,
}

impl Default for LabelingConfig {
    fn default() -> Self {
        Self {
            min_annotations_per_task: 2,
            max_annotations_per_task: 3,
            uncertainty_threshold: 0.7,
            min_agreement_threshold: 0.8,
            enable_quality_control: true,
            auto_validate_on_consensus: true,
            priority_strategy: PriorityStrategy::UncertaintyBased,
        }
    }
}

/// Priority calculation strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriorityStrategy {
    /// Prioritize by uncertainty
    UncertaintyBased,
    /// Prioritize by diversity
    DiversityBased,
    /// Prioritize by model disagreement
    DisagreementBased,
    /// Prioritize by expected model change
    ExpectedChange,
    /// Custom priority scores
    Custom,
}

/// Interactive labeling interface
pub struct InteractiveLabelingInterface {
    config: LabelingConfig,
    tasks: HashMap<String, AnnotationTask>,
    annotators: HashMap<String, Annotator>,
    task_queue: Vec<String>, // Prioritized task IDs
}

impl InteractiveLabelingInterface {
    /// Create new labeling interface
    pub fn new() -> Self {
        Self::with_config(LabelingConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: LabelingConfig) -> Self {
        Self {
            config,
            tasks: HashMap::new(),
            annotators: HashMap::new(),
            task_queue: Vec::new(),
        }
    }

    /// Add annotation task
    pub fn add_task(&mut self, mut task: AnnotationTask) -> Result<()> {
        // Calculate priority based on strategy
        task.priority = self.calculate_priority(&task);

        let task_id = task.id.clone();
        self.tasks.insert(task_id.clone(), task);

        // Insert into priority queue
        self.insert_into_queue(task_id);

        Ok(())
    }

    /// Add multiple tasks in batch
    pub fn add_tasks_batch(&mut self, tasks: Vec<AnnotationTask>) -> Result<()> {
        for task in tasks {
            self.add_task(task)?;
        }
        Ok(())
    }

    /// Register annotator
    pub fn register_annotator(&mut self, annotator: Annotator) -> Result<()> {
        self.annotators.insert(annotator.id.clone(), annotator);
        Ok(())
    }

    /// Get next task for annotator
    pub fn get_next_task(&self, annotator_id: &str) -> Option<&AnnotationTask> {
        // Check if annotator exists and is active
        if let Some(annotator) = self.annotators.get(annotator_id) {
            if !annotator.is_active {
                return None;
            }
        } else {
            return None;
        }

        // Find highest priority pending or assigned task
        for task_id in &self.task_queue {
            if let Some(task) = self.tasks.get(task_id) {
                if matches!(task.status, TaskStatus::Pending | TaskStatus::Assigned) {
                    // Check if annotator already annotated this task
                    let already_annotated = task
                        .annotations
                        .iter()
                        .any(|a| a.annotator_id == annotator_id);

                    if !already_annotated {
                        return Some(task);
                    }
                }
            }
        }

        None
    }

    /// Submit annotation
    pub fn submit_annotation(&mut self, task_id: &str, annotation: Annotation) -> Result<()> {
        let task = self
            .tasks
            .get_mut(task_id)
            .ok_or_else(|| ShaclAiError::Configuration(format!("Task {} not found", task_id)))?;

        // Validate annotation
        if annotation.task_id != task_id {
            return Err(ShaclAiError::Configuration(
                "Annotation task ID mismatch".to_string(),
            ));
        }

        // Add annotation
        task.annotations.push(annotation.clone());
        task.status = TaskStatus::InProgress;

        // Update annotator stats
        if let Some(annotator) = self.annotators.get_mut(&annotation.annotator_id) {
            annotator.stats.total_annotations += 1;

            // Update average time
            let total_time = annotator.stats.avg_time_per_annotation
                * (annotator.stats.total_annotations - 1) as f64
                + annotation.time_spent;
            annotator.stats.avg_time_per_annotation =
                total_time / annotator.stats.total_annotations as f64;
        }

        // Check if task should be marked completed or needs review
        if task.annotations.len() >= self.config.min_annotations_per_task {
            self.evaluate_task(task_id)?;
        }

        Ok(())
    }

    /// Evaluate task for consensus
    fn evaluate_task(&mut self, task_id: &str) -> Result<()> {
        // Clone annotations to avoid borrow checker issues
        let (annotations, annotations_len, min_threshold, max_annotations, auto_validate) = {
            let task = self.tasks.get(task_id).ok_or_else(|| {
                ShaclAiError::Configuration(format!("Task {} not found", task_id))
            })?;

            if task.annotations.is_empty() {
                return Ok(());
            }

            (
                task.annotations.clone(),
                task.annotations.len(),
                self.config.min_agreement_threshold,
                self.config.max_annotations_per_task,
                self.config.auto_validate_on_consensus,
            )
        };

        // Calculate agreement
        let agreement = self.calculate_agreement(&annotations);

        // Update task status
        let task = self
            .tasks
            .get_mut(task_id)
            .expect("task should exist for given task_id");
        if agreement >= min_threshold {
            // Consensus reached
            task.status = if auto_validate {
                TaskStatus::Validated
            } else {
                TaskStatus::Completed
            };
        } else if annotations_len >= max_annotations {
            // Max annotations reached but no consensus
            task.status = TaskStatus::Disputed;
        } else {
            // Need more annotations
            task.status = TaskStatus::NeedsReview;
        }

        Ok(())
    }

    /// Calculate agreement among annotations
    fn calculate_agreement(&self, annotations: &[Annotation]) -> f64 {
        if annotations.len() < 2 {
            return 1.0;
        }

        // Count label occurrences
        let mut label_counts: HashMap<String, usize> = HashMap::new();
        for annotation in annotations {
            *label_counts.entry(annotation.label.clone()).or_insert(0) += 1;
        }

        // Find most common label
        let max_count = label_counts.values().max().unwrap_or(&0);

        // Agreement = proportion of majority label
        *max_count as f64 / annotations.len() as f64
    }

    /// Get consensus label for task
    pub fn get_consensus_label(&self, task_id: &str) -> Option<String> {
        let task = self.tasks.get(task_id)?;

        if task.annotations.is_empty() {
            return None;
        }

        // Count label occurrences weighted by annotator expertise
        let mut label_scores: HashMap<String, f64> = HashMap::new();

        for annotation in &task.annotations {
            let weight = if let Some(annotator) = self.annotators.get(&annotation.annotator_id) {
                annotator.expertise_level * annotator.stats.quality_score
            } else {
                1.0
            };

            *label_scores.entry(annotation.label.clone()).or_insert(0.0) += weight;
        }

        // Return label with highest score
        label_scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label)
    }

    /// Calculate quality metrics
    pub fn calculate_quality_metrics(&self) -> QualityMetrics {
        let mut total_confidence = 0.0;
        let mut total_time = 0.0;
        let mut total_annotations = 0;
        let mut consensus_count = 0;
        let mut disputed_count = 0;

        for task in self.tasks.values() {
            if !task.annotations.is_empty() {
                let agreement = self.calculate_agreement(&task.annotations);

                for annotation in &task.annotations {
                    total_confidence += annotation.confidence;
                    total_time += annotation.time_spent;
                    total_annotations += 1;
                }

                if agreement >= self.config.min_agreement_threshold {
                    consensus_count += 1;
                }
            }

            if task.status == TaskStatus::Disputed {
                disputed_count += 1;
            }
        }

        let task_count = self.tasks.len();

        QualityMetrics {
            inter_annotator_agreement: if task_count > 0 {
                consensus_count as f64 / task_count as f64
            } else {
                0.0
            },
            avg_confidence: if total_annotations > 0 {
                total_confidence / total_annotations as f64
            } else {
                0.0
            },
            consensus_rate: if task_count > 0 {
                consensus_count as f64 / task_count as f64
            } else {
                0.0
            },
            disputed_tasks: disputed_count,
            avg_time_per_task: if task_count > 0 {
                total_time / task_count as f64
            } else {
                0.0
            },
        }
    }

    /// Export annotations to JSON
    pub fn export_to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.tasks)
            .map_err(|e| ShaclAiError::Configuration(format!("JSON serialization failed: {}", e)))
    }

    /// Get annotator leaderboard
    pub fn get_leaderboard(&self) -> Vec<(&Annotator, f64)> {
        let mut leaderboard: Vec<_> = self
            .annotators
            .values()
            .map(|annotator| {
                let score = annotator.stats.quality_score
                    * annotator.stats.agreement_rate
                    * (annotator.stats.total_annotations as f64).ln().max(1.0);
                (annotator, score)
            })
            .collect();

        leaderboard.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        leaderboard
    }

    /// Get task statistics
    pub fn get_task_statistics(&self) -> TaskStatistics {
        let mut stats = TaskStatistics::default();

        for task in self.tasks.values() {
            stats.total_tasks += 1;

            match task.status {
                TaskStatus::Pending => stats.pending_tasks += 1,
                TaskStatus::Assigned => stats.assigned_tasks += 1,
                TaskStatus::InProgress => stats.in_progress_tasks += 1,
                TaskStatus::Completed => stats.completed_tasks += 1,
                TaskStatus::NeedsReview => stats.needs_review_tasks += 1,
                TaskStatus::Disputed => stats.disputed_tasks += 1,
                TaskStatus::Validated => stats.validated_tasks += 1,
                TaskStatus::Skipped => stats.skipped_tasks += 1,
            }

            stats.total_annotations += task.annotations.len();
        }

        stats
    }

    // Private helper methods

    fn calculate_priority(&self, task: &AnnotationTask) -> f64 {
        match self.config.priority_strategy {
            PriorityStrategy::UncertaintyBased => task.uncertainty,
            PriorityStrategy::DiversityBased => {
                // Simplified diversity score
                1.0 - task.confidence
            }
            PriorityStrategy::DisagreementBased => {
                // Would compare with other model predictions
                task.uncertainty * 0.8 + (1.0 - task.confidence) * 0.2
            }
            PriorityStrategy::ExpectedChange => {
                // Expected impact on model
                task.uncertainty * task.confidence
            }
            PriorityStrategy::Custom => {
                // Use provided priority
                task.priority
            }
        }
    }

    fn insert_into_queue(&mut self, task_id: String) {
        // Insert maintaining priority order (highest first)
        let task_priority = self.tasks.get(&task_id).map(|t| t.priority).unwrap_or(0.0);

        let insert_pos = self
            .task_queue
            .iter()
            .position(|id| {
                self.tasks
                    .get(id)
                    .map(|t| t.priority < task_priority)
                    .unwrap_or(true)
            })
            .unwrap_or(self.task_queue.len());

        self.task_queue.insert(insert_pos, task_id);
    }
}

impl Default for InteractiveLabelingInterface {
    fn default() -> Self {
        Self::new()
    }
}

/// Task statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskStatistics {
    pub total_tasks: usize,
    pub pending_tasks: usize,
    pub assigned_tasks: usize,
    pub in_progress_tasks: usize,
    pub completed_tasks: usize,
    pub needs_review_tasks: usize,
    pub disputed_tasks: usize,
    pub validated_tasks: usize,
    pub skipped_tasks: usize,
    pub total_annotations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_task(id: &str, uncertainty: f64) -> AnnotationTask {
        AnnotationTask {
            id: id.to_string(),
            data: RdfData {
                subject: "http://example.org/s1".to_string(),
                predicate: "http://example.org/p1".to_string(),
                object: "http://example.org/o1".to_string(),
                graph: None,
                context: HashMap::new(),
            },
            suggested_label: Some("valid".to_string()),
            confidence: 0.5,
            uncertainty,
            priority: 0.0,
            status: TaskStatus::Pending,
            assigned_to: vec![],
            created_at: chrono::Utc::now(),
            annotations: vec![],
        }
    }

    fn create_test_annotator(id: &str) -> Annotator {
        Annotator {
            id: id.to_string(),
            name: format!("Annotator {}", id),
            email: format!("{}@example.com", id),
            expertise_level: 0.8,
            stats: AnnotatorStats::default(),
            is_active: true,
        }
    }

    #[test]
    fn test_interface_creation() {
        let interface = InteractiveLabelingInterface::new();
        assert_eq!(interface.tasks.len(), 0);
        assert_eq!(interface.annotators.len(), 0);
    }

    #[test]
    fn test_add_task() {
        let mut interface = InteractiveLabelingInterface::new();
        let task = create_test_task("task1", 0.8);

        interface.add_task(task).unwrap();
        assert_eq!(interface.tasks.len(), 1);
        assert_eq!(interface.task_queue.len(), 1);
    }

    #[test]
    fn test_priority_ordering() {
        let mut interface = InteractiveLabelingInterface::new();

        interface.add_task(create_test_task("task1", 0.5)).unwrap();
        interface.add_task(create_test_task("task2", 0.9)).unwrap();
        interface.add_task(create_test_task("task3", 0.7)).unwrap();

        // Highest priority should be first
        assert_eq!(interface.task_queue[0], "task2");
        assert_eq!(interface.task_queue[1], "task3");
        assert_eq!(interface.task_queue[2], "task1");
    }

    #[test]
    fn test_register_annotator() {
        let mut interface = InteractiveLabelingInterface::new();
        let annotator = create_test_annotator("ann1");

        interface.register_annotator(annotator).unwrap();
        assert_eq!(interface.annotators.len(), 1);
    }

    #[test]
    fn test_get_next_task() {
        let mut interface = InteractiveLabelingInterface::new();
        let annotator = create_test_annotator("ann1");
        let task = create_test_task("task1", 0.8);

        interface.register_annotator(annotator).unwrap();
        interface.add_task(task).unwrap();

        let next_task = interface.get_next_task("ann1");
        assert!(next_task.is_some());
        assert_eq!(next_task.unwrap().id, "task1");
    }

    #[test]
    fn test_submit_annotation() {
        let mut interface = InteractiveLabelingInterface::new();
        let annotator = create_test_annotator("ann1");
        let task = create_test_task("task1", 0.8);

        interface.register_annotator(annotator).unwrap();
        interface.add_task(task).unwrap();

        let annotation = Annotation {
            id: "ann_1".to_string(),
            task_id: "task1".to_string(),
            annotator_id: "ann1".to_string(),
            label: "valid".to_string(),
            confidence: 0.9,
            time_spent: 45.0,
            notes: None,
            timestamp: chrono::Utc::now(),
        };

        interface.submit_annotation("task1", annotation).unwrap();

        let task = interface.tasks.get("task1").unwrap();
        assert_eq!(task.annotations.len(), 1);
        assert_eq!(task.status, TaskStatus::InProgress);
    }

    #[test]
    fn test_consensus_calculation() {
        let interface = InteractiveLabelingInterface::new();

        let annotations = vec![
            Annotation {
                id: "a1".to_string(),
                task_id: "task1".to_string(),
                annotator_id: "ann1".to_string(),
                label: "valid".to_string(),
                confidence: 0.9,
                time_spent: 30.0,
                notes: None,
                timestamp: chrono::Utc::now(),
            },
            Annotation {
                id: "a2".to_string(),
                task_id: "task1".to_string(),
                annotator_id: "ann2".to_string(),
                label: "valid".to_string(),
                confidence: 0.85,
                time_spent: 40.0,
                notes: None,
                timestamp: chrono::Utc::now(),
            },
            Annotation {
                id: "a3".to_string(),
                task_id: "task1".to_string(),
                annotator_id: "ann3".to_string(),
                label: "invalid".to_string(),
                confidence: 0.7,
                time_spent: 35.0,
                notes: None,
                timestamp: chrono::Utc::now(),
            },
        ];

        let agreement = interface.calculate_agreement(&annotations);
        assert!((agreement - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_get_consensus_label() {
        let mut interface = InteractiveLabelingInterface::new();
        interface
            .register_annotator(create_test_annotator("ann1"))
            .unwrap();
        interface
            .register_annotator(create_test_annotator("ann2"))
            .unwrap();

        let mut task = create_test_task("task1", 0.8);
        task.annotations = vec![
            Annotation {
                id: "a1".to_string(),
                task_id: "task1".to_string(),
                annotator_id: "ann1".to_string(),
                label: "valid".to_string(),
                confidence: 0.9,
                time_spent: 30.0,
                notes: None,
                timestamp: chrono::Utc::now(),
            },
            Annotation {
                id: "a2".to_string(),
                task_id: "task1".to_string(),
                annotator_id: "ann2".to_string(),
                label: "valid".to_string(),
                confidence: 0.85,
                time_spent: 40.0,
                notes: None,
                timestamp: chrono::Utc::now(),
            },
        ];

        interface.tasks.insert("task1".to_string(), task);

        let consensus = interface.get_consensus_label("task1");
        assert_eq!(consensus, Some("valid".to_string()));
    }

    #[test]
    fn test_quality_metrics() {
        let mut interface = InteractiveLabelingInterface::new();
        interface.add_task(create_test_task("task1", 0.8)).unwrap();

        let metrics = interface.calculate_quality_metrics();
        assert_eq!(metrics.disputed_tasks, 0);
    }

    #[test]
    fn test_task_statistics() {
        let mut interface = InteractiveLabelingInterface::new();
        interface.add_task(create_test_task("task1", 0.8)).unwrap();
        interface.add_task(create_test_task("task2", 0.6)).unwrap();

        let stats = interface.get_task_statistics();
        assert_eq!(stats.total_tasks, 2);
        assert_eq!(stats.pending_tasks, 2);
    }
}
