//! Fine-tuning support for OxiRS Chat models
//!
//! This module provides infrastructure for model fine-tuning based on
//! conversation history, user feedback, and domain-specific datasets.
//!
//! # Features
//! - Training data collection from conversation history
//! - JSONL export format for OpenAI/HuggingFace fine-tuning
//! - Data quality filtering and deduplication
//! - Feedback collection and annotation
//! - Domain-specific dataset management
//! - Training job tracking and status management
//! - Model performance evaluation utilities

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Errors that can occur during fine-tuning operations
#[derive(Debug, Error)]
pub enum FinetuningError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Dataset not found: {0}")]
    DatasetNotFound(String),
    #[error("Insufficient training data: need {needed}, have {available}")]
    InsufficientData { needed: usize, available: usize },
    #[error("Training job not found: {0}")]
    JobNotFound(String),
    #[error("Invalid configuration: {0}")]
    ConfigurationError(String),
    #[error("Data quality error: {0}")]
    DataQualityError(String),
}

/// Result type for fine-tuning operations
pub type FinetuningResult<T> = Result<T, FinetuningError>;

/// Supported export formats for fine-tuning datasets
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// OpenAI Chat Completions JSONL format
    OpenAiChatJsonl,
    /// HuggingFace instruction format
    HuggingFaceInstruct,
    /// Alpaca instruction format
    AlpacaInstruct,
    /// Raw conversation pairs (input/output)
    ConversationPairs,
    /// Full conversation JSON
    FullConversationJson,
}

/// A single training example (conversation turn or exchange)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Unique identifier for this example
    pub id: String,
    /// The system prompt context
    pub system_prompt: Option<String>,
    /// The human/user message
    pub human_message: String,
    /// The expected assistant response
    pub assistant_response: String,
    /// Domain/topic category for this example
    pub domain: Option<String>,
    /// Quality score (0.0 - 1.0, higher is better)
    pub quality_score: f32,
    /// Whether this was reviewed and approved by a human
    pub human_reviewed: bool,
    /// User feedback rating (1-5)
    pub user_rating: Option<u8>,
    /// Tags for filtering and categorization
    pub tags: Vec<String>,
    /// Source session ID
    pub source_session_id: Option<String>,
    /// When this example was collected
    pub collected_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TrainingExample {
    /// Create a new training example
    pub fn new(human_message: String, assistant_response: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            system_prompt: None,
            human_message,
            assistant_response,
            domain: None,
            quality_score: 0.5,
            human_reviewed: false,
            user_rating: None,
            tags: Vec::new(),
            source_session_id: None,
            collected_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Check if this example meets the minimum quality threshold
    pub fn meets_quality_threshold(&self, threshold: f32) -> bool {
        self.quality_score >= threshold
    }

    /// Export as OpenAI chat completion format
    pub fn to_openai_format(&self) -> serde_json::Value {
        let mut messages = Vec::new();

        if let Some(ref system) = self.system_prompt {
            messages.push(serde_json::json!({
                "role": "system",
                "content": system
            }));
        }

        messages.push(serde_json::json!({
            "role": "user",
            "content": self.human_message
        }));

        messages.push(serde_json::json!({
            "role": "assistant",
            "content": self.assistant_response
        }));

        serde_json::json!({ "messages": messages })
    }

    /// Export as Alpaca instruction format
    pub fn to_alpaca_format(&self) -> serde_json::Value {
        serde_json::json!({
            "instruction": self.human_message,
            "input": "",
            "output": self.assistant_response
        })
    }

    /// Calculate a quality score based on various heuristics
    pub fn compute_quality_score(&mut self) {
        let mut score = 0.5f32;

        // Boost for longer, more detailed responses
        let response_len = self.assistant_response.len();
        if response_len > 100 {
            score += 0.1;
        }
        if response_len > 500 {
            score += 0.1;
        }

        // Boost for user rating
        if let Some(rating) = self.user_rating {
            score += (rating as f32 - 3.0) * 0.1;
        }

        // Boost for human review
        if self.human_reviewed {
            score += 0.2;
        }

        // Penalize very short responses (likely unhelpful)
        if response_len < 20 {
            score -= 0.3;
        }

        // Penalize very short user messages (may lack context)
        if self.human_message.len() < 10 {
            score -= 0.1;
        }

        self.quality_score = score.clamp(0.0, 1.0);
    }
}

/// A dataset of training examples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataset {
    /// Unique identifier for this dataset
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what this dataset covers
    pub description: String,
    /// Domain/topic focus (e.g., "sparql", "rdf", "general")
    pub domain: Option<String>,
    /// All training examples in this dataset
    pub examples: Vec<TrainingExample>,
    /// When this dataset was created
    pub created_at: DateTime<Utc>,
    /// When this dataset was last updated
    pub updated_at: DateTime<Utc>,
    /// Dataset version
    pub version: String,
    /// Tags for organization
    pub tags: Vec<String>,
    /// Dataset statistics
    pub stats: DatasetStats,
}

/// Statistics about a training dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_examples: usize,
    pub high_quality_examples: usize,
    pub human_reviewed_examples: usize,
    pub average_quality_score: f32,
    pub average_human_len: f32,
    pub average_assistant_len: f32,
    pub unique_domains: usize,
}

impl TrainingDataset {
    /// Create a new empty dataset
    pub fn new(id: String, name: String) -> Self {
        let now = Utc::now();
        Self {
            id,
            name,
            description: String::new(),
            domain: None,
            examples: Vec::new(),
            created_at: now,
            updated_at: now,
            version: "1.0.0".to_string(),
            tags: Vec::new(),
            stats: DatasetStats {
                total_examples: 0,
                high_quality_examples: 0,
                human_reviewed_examples: 0,
                average_quality_score: 0.0,
                average_human_len: 0.0,
                average_assistant_len: 0.0,
                unique_domains: 0,
            },
        }
    }

    /// Add a training example.
    ///
    /// If the example's `quality_score` is exactly 0.5 (the default),
    /// it will be recomputed from heuristics. Otherwise the caller's
    /// explicitly set score is preserved.
    pub fn add_example(&mut self, mut example: TrainingExample) {
        if (example.quality_score - 0.5).abs() < f32::EPSILON {
            example.compute_quality_score();
        }
        self.examples.push(example);
        self.updated_at = Utc::now();
        self.recompute_stats();
    }

    /// Get examples filtered by minimum quality score
    pub fn examples_above_quality(&self, threshold: f32) -> Vec<&TrainingExample> {
        self.examples
            .iter()
            .filter(|e| e.quality_score >= threshold)
            .collect()
    }

    /// Deduplicate examples by exact human_message match
    pub fn deduplicate(&mut self) -> usize {
        let before = self.examples.len();
        let mut seen_messages = std::collections::HashSet::new();
        self.examples.retain(|e| {
            let key = e.human_message.trim().to_lowercase();
            seen_messages.insert(key)
        });
        let removed = before - self.examples.len();
        if removed > 0 {
            info!("Deduplicated {} examples from dataset {}", removed, self.id);
            self.recompute_stats();
        }
        removed
    }

    /// Export examples in the specified format as JSONL string
    pub fn export_jsonl(
        &self,
        format: &ExportFormat,
        quality_threshold: f32,
    ) -> FinetuningResult<String> {
        let examples = self.examples_above_quality(quality_threshold);

        if examples.is_empty() {
            return Err(FinetuningError::InsufficientData {
                needed: 1,
                available: 0,
            });
        }

        let mut lines = Vec::with_capacity(examples.len());
        for example in &examples {
            let json = match format {
                ExportFormat::OpenAiChatJsonl => example.to_openai_format(),
                ExportFormat::AlpacaInstruct => example.to_alpaca_format(),
                ExportFormat::HuggingFaceInstruct => serde_json::json!({
                    "prompt": example.human_message,
                    "response": example.assistant_response,
                }),
                ExportFormat::ConversationPairs => serde_json::json!({
                    "input": example.human_message,
                    "output": example.assistant_response,
                }),
                ExportFormat::FullConversationJson => serde_json::to_value(example)?,
            };
            lines.push(serde_json::to_string(&json)?);
        }

        Ok(lines.join("\n"))
    }

    /// Recompute dataset statistics
    pub fn recompute_stats(&mut self) {
        let total = self.examples.len();
        if total == 0 {
            self.stats = DatasetStats {
                total_examples: 0,
                high_quality_examples: 0,
                human_reviewed_examples: 0,
                average_quality_score: 0.0,
                average_human_len: 0.0,
                average_assistant_len: 0.0,
                unique_domains: 0,
            };
            return;
        }

        let high_quality = self
            .examples
            .iter()
            .filter(|e| e.quality_score >= 0.7)
            .count();
        let reviewed = self.examples.iter().filter(|e| e.human_reviewed).count();
        let total_quality: f32 = self.examples.iter().map(|e| e.quality_score).sum();
        let total_human_len: usize = self.examples.iter().map(|e| e.human_message.len()).sum();
        let total_assistant_len: usize = self
            .examples
            .iter()
            .map(|e| e.assistant_response.len())
            .sum();
        let unique_domains: std::collections::HashSet<&str> = self
            .examples
            .iter()
            .filter_map(|e| e.domain.as_deref())
            .collect();

        self.stats = DatasetStats {
            total_examples: total,
            high_quality_examples: high_quality,
            human_reviewed_examples: reviewed,
            average_quality_score: total_quality / total as f32,
            average_human_len: total_human_len as f32 / total as f32,
            average_assistant_len: total_assistant_len as f32 / total as f32,
            unique_domains: unique_domains.len(),
        };
    }
}

/// Status of a fine-tuning job
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is waiting to be processed
    Pending,
    /// Training data is being prepared
    PreparingData,
    /// Job is actively training
    Training,
    /// Job has completed successfully
    Completed,
    /// Job has failed
    Failed,
    /// Job was cancelled
    Cancelled,
}

/// Configuration for a fine-tuning job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuningJobConfig {
    /// Base model to fine-tune (e.g., "gpt-3.5-turbo", "llama-2-7b")
    pub base_model: String,
    /// Target model name/suffix
    pub suffix: String,
    /// Number of training epochs
    pub epochs: u32,
    /// Batch size for training
    pub batch_size: u32,
    /// Learning rate multiplier
    pub learning_rate_multiplier: f32,
    /// Minimum quality threshold for training examples
    pub quality_threshold: f32,
    /// Export format for training data
    pub export_format: ExportFormat,
    /// Maximum number of training examples to use
    pub max_examples: Option<usize>,
    /// Whether to validate the model after training
    pub run_validation: bool,
    /// Validation split ratio (0.0 - 0.3)
    pub validation_split: f32,
}

impl Default for FinetuningJobConfig {
    fn default() -> Self {
        Self {
            base_model: "gpt-3.5-turbo".to_string(),
            suffix: "oxirs-chat".to_string(),
            epochs: 3,
            batch_size: 4,
            learning_rate_multiplier: 1.0,
            quality_threshold: 0.6,
            export_format: ExportFormat::OpenAiChatJsonl,
            max_examples: None,
            run_validation: true,
            validation_split: 0.1,
        }
    }
}

/// A fine-tuning job record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuningJob {
    /// Unique job identifier
    pub id: String,
    /// Dataset used for this job
    pub dataset_id: String,
    /// Job configuration
    pub config: FinetuningJobConfig,
    /// Current job status
    pub status: JobStatus,
    /// When the job was created
    pub created_at: DateTime<Utc>,
    /// When the job started training
    pub started_at: Option<DateTime<Utc>>,
    /// When the job completed or failed
    pub completed_at: Option<DateTime<Utc>>,
    /// External job ID from the fine-tuning provider (e.g., OpenAI job ID)
    pub external_job_id: Option<String>,
    /// Resulting fine-tuned model name
    pub result_model: Option<String>,
    /// Number of training examples used
    pub training_examples: usize,
    /// Number of validation examples
    pub validation_examples: usize,
    /// Current training step
    pub current_step: Option<u64>,
    /// Total training steps
    pub total_steps: Option<u64>,
    /// Training loss history
    pub training_loss: Vec<f32>,
    /// Validation loss history
    pub validation_loss: Vec<f32>,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Job metadata
    pub metadata: HashMap<String, String>,
}

impl FinetuningJob {
    /// Create a new fine-tuning job
    pub fn new(dataset_id: String, config: FinetuningJobConfig) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            dataset_id,
            config,
            status: JobStatus::Pending,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            external_job_id: None,
            result_model: None,
            training_examples: 0,
            validation_examples: 0,
            current_step: None,
            total_steps: None,
            training_loss: Vec::new(),
            validation_loss: Vec::new(),
            error_message: None,
            metadata: HashMap::new(),
        }
    }

    /// Mark the job as started
    pub fn start(&mut self) {
        self.status = JobStatus::Training;
        self.started_at = Some(Utc::now());
        info!("Fine-tuning job {} started", self.id);
    }

    /// Record a training step
    pub fn record_step(&mut self, step: u64, train_loss: f32, val_loss: Option<f32>) {
        self.current_step = Some(step);
        self.training_loss.push(train_loss);
        if let Some(vl) = val_loss {
            self.validation_loss.push(vl);
        }
        debug!(
            "Job {} step {}: train_loss={:.4}",
            self.id, step, train_loss
        );
    }

    /// Mark the job as completed
    pub fn complete(&mut self, result_model: String) {
        self.status = JobStatus::Completed;
        self.completed_at = Some(Utc::now());
        self.result_model = Some(result_model.clone());
        info!(
            "Fine-tuning job {} completed. Model: {}",
            self.id, result_model
        );
    }

    /// Mark the job as failed
    pub fn fail(&mut self, error: String) {
        self.status = JobStatus::Failed;
        self.completed_at = Some(Utc::now());
        self.error_message = Some(error.clone());
        warn!("Fine-tuning job {} failed: {}", self.id, error);
    }

    /// Get the duration of the job
    pub fn duration(&self) -> Option<chrono::Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }

    /// Get progress as a percentage (0.0 - 1.0)
    pub fn progress(&self) -> f32 {
        match (self.current_step, self.total_steps) {
            (Some(current), Some(total)) if total > 0 => current as f32 / total as f32,
            _ => match self.status {
                JobStatus::Completed => 1.0,
                JobStatus::Pending => 0.0,
                _ => 0.5,
            },
        }
    }

    /// Check if the job is currently active
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            JobStatus::Pending | JobStatus::PreparingData | JobStatus::Training
        )
    }
}

/// Feedback collected from users about conversation quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationFeedback {
    /// Unique feedback ID
    pub id: String,
    /// Session ID this feedback is for
    pub session_id: String,
    /// Message ID this feedback is for
    pub message_id: Option<String>,
    /// Rating (1-5, with 5 being best)
    pub rating: u8,
    /// Optional text feedback
    pub comment: Option<String>,
    /// Whether the user flagged this as incorrect
    pub flagged_incorrect: bool,
    /// Whether the user flagged this as harmful
    pub flagged_harmful: bool,
    /// Specific issue categories
    pub issue_categories: Vec<FeedbackIssue>,
    /// When this feedback was submitted
    pub submitted_at: DateTime<Utc>,
    /// User identifier (anonymized)
    pub user_id: Option<String>,
}

/// Categories of issues in feedback
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackIssue {
    Inaccurate,
    Unhelpful,
    TooLong,
    TooShort,
    OffTopic,
    Confusing,
    Biased,
    Outdated,
    MissingContext,
}

impl ConversationFeedback {
    /// Create new feedback for a session
    pub fn new(session_id: String, rating: u8) -> Self {
        let rating = rating.clamp(1, 5);
        Self {
            id: Uuid::new_v4().to_string(),
            session_id,
            message_id: None,
            rating,
            comment: None,
            flagged_incorrect: false,
            flagged_harmful: false,
            issue_categories: Vec::new(),
            submitted_at: Utc::now(),
            user_id: None,
        }
    }

    /// Convert rating to a quality score component
    pub fn rating_to_quality(&self) -> f32 {
        (self.rating as f32 - 1.0) / 4.0 // Maps 1-5 to 0.0-1.0
    }
}

/// Configuration for the fine-tuning data manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuningManagerConfig {
    /// Directory for storing datasets and jobs
    pub storage_dir: PathBuf,
    /// Minimum examples required before export
    pub min_examples_for_export: usize,
    /// Default quality threshold for training data
    pub default_quality_threshold: f32,
    /// Whether to auto-deduplicate on save
    pub auto_deduplicate: bool,
    /// Maximum dataset size (0 = unlimited)
    pub max_dataset_size: usize,
}

impl Default for FinetuningManagerConfig {
    fn default() -> Self {
        Self {
            storage_dir: PathBuf::from("data/finetuning"),
            min_examples_for_export: 10,
            default_quality_threshold: 0.5,
            auto_deduplicate: true,
            max_dataset_size: 100_000,
        }
    }
}

/// Manages training datasets, jobs, and feedback for fine-tuning
pub struct FinetuningManager {
    config: FinetuningManagerConfig,
    datasets: HashMap<String, TrainingDataset>,
    jobs: HashMap<String, FinetuningJob>,
    feedback: Vec<ConversationFeedback>,
}

impl FinetuningManager {
    /// Create a new fine-tuning manager
    pub fn new(config: FinetuningManagerConfig) -> FinetuningResult<Self> {
        fs::create_dir_all(&config.storage_dir)?;
        Ok(Self {
            config,
            datasets: HashMap::new(),
            jobs: HashMap::new(),
            feedback: Vec::new(),
        })
    }

    /// Create a new dataset
    pub fn create_dataset(
        &mut self,
        name: String,
        description: String,
    ) -> FinetuningResult<String> {
        let id = Uuid::new_v4().to_string();
        let mut dataset = TrainingDataset::new(id.clone(), name);
        dataset.description = description;
        info!("Created fine-tuning dataset: {}", dataset.id);
        self.datasets.insert(id.clone(), dataset);
        Ok(id)
    }

    /// Get a dataset by ID
    pub fn get_dataset(&self, id: &str) -> FinetuningResult<&TrainingDataset> {
        self.datasets
            .get(id)
            .ok_or_else(|| FinetuningError::DatasetNotFound(id.to_string()))
    }

    /// Get a mutable dataset by ID
    pub fn get_dataset_mut(&mut self, id: &str) -> FinetuningResult<&mut TrainingDataset> {
        self.datasets
            .get_mut(id)
            .ok_or_else(|| FinetuningError::DatasetNotFound(id.to_string()))
    }

    /// Add a training example to a dataset
    pub fn add_example(
        &mut self,
        dataset_id: &str,
        example: TrainingExample,
    ) -> FinetuningResult<()> {
        let max_size = self.config.max_dataset_size;
        let auto_dedup = self.config.auto_deduplicate;

        let dataset = self.get_dataset_mut(dataset_id)?;

        if max_size > 0 && dataset.examples.len() >= max_size {
            warn!(
                "Dataset {} is at maximum size ({}), dropping oldest example",
                dataset_id, max_size
            );
            dataset.examples.remove(0);
        }

        dataset.add_example(example);

        if auto_dedup && dataset.examples.len() % 1000 == 0 {
            dataset.deduplicate();
        }

        Ok(())
    }

    /// Submit user feedback
    pub fn submit_feedback(&mut self, feedback: ConversationFeedback) -> FinetuningResult<String> {
        let id = feedback.id.clone();
        debug!(
            "Received feedback for session {}: rating={}",
            feedback.session_id, feedback.rating
        );
        self.feedback.push(feedback);
        Ok(id)
    }

    /// Get all feedback for a session
    pub fn get_session_feedback(&self, session_id: &str) -> Vec<&ConversationFeedback> {
        self.feedback
            .iter()
            .filter(|f| f.session_id == session_id)
            .collect()
    }

    /// Create a fine-tuning job from a dataset
    pub fn create_job(
        &mut self,
        dataset_id: &str,
        config: FinetuningJobConfig,
    ) -> FinetuningResult<String> {
        // Validate dataset has enough examples
        let dataset = self.get_dataset(dataset_id)?;
        let available = dataset
            .examples_above_quality(config.quality_threshold)
            .len();

        if available < self.config.min_examples_for_export {
            return Err(FinetuningError::InsufficientData {
                needed: self.config.min_examples_for_export,
                available,
            });
        }

        let job = FinetuningJob::new(dataset_id.to_string(), config);
        let job_id = job.id.clone();
        info!(
            "Created fine-tuning job {} for dataset {}",
            job_id, dataset_id
        );
        self.jobs.insert(job_id.clone(), job);
        Ok(job_id)
    }

    /// Get a job by ID
    pub fn get_job(&self, job_id: &str) -> FinetuningResult<&FinetuningJob> {
        self.jobs
            .get(job_id)
            .ok_or_else(|| FinetuningError::JobNotFound(job_id.to_string()))
    }

    /// Get a mutable job by ID
    pub fn get_job_mut(&mut self, job_id: &str) -> FinetuningResult<&mut FinetuningJob> {
        self.jobs
            .get_mut(job_id)
            .ok_or_else(|| FinetuningError::JobNotFound(job_id.to_string()))
    }

    /// Export a dataset to a file in the specified format
    pub fn export_dataset(
        &self,
        dataset_id: &str,
        output_path: &Path,
        format: &ExportFormat,
        quality_threshold: Option<f32>,
    ) -> FinetuningResult<usize> {
        let dataset = self.get_dataset(dataset_id)?;
        let threshold = quality_threshold.unwrap_or(self.config.default_quality_threshold);
        let jsonl = dataset.export_jsonl(format, threshold)?;

        let line_count = jsonl.lines().count();
        fs::write(output_path, &jsonl)?;

        info!(
            "Exported {} examples from dataset {} to {:?}",
            line_count, dataset_id, output_path
        );
        Ok(line_count)
    }

    /// Get manager statistics
    pub fn statistics(&self) -> FinetuningManagerStats {
        let total_examples: usize = self.datasets.values().map(|d| d.examples.len()).sum();
        let active_jobs = self.jobs.values().filter(|j| j.is_active()).count();
        let completed_jobs = self
            .jobs
            .values()
            .filter(|j| j.status == JobStatus::Completed)
            .count();

        FinetuningManagerStats {
            total_datasets: self.datasets.len(),
            total_examples,
            total_jobs: self.jobs.len(),
            active_jobs,
            completed_jobs,
            failed_jobs: self
                .jobs
                .values()
                .filter(|j| j.status == JobStatus::Failed)
                .count(),
            total_feedback: self.feedback.len(),
            average_rating: if self.feedback.is_empty() {
                0.0
            } else {
                self.feedback.iter().map(|f| f.rating as f32).sum::<f32>()
                    / self.feedback.len() as f32
            },
        }
    }

    /// List all datasets
    pub fn list_datasets(&self) -> Vec<&TrainingDataset> {
        self.datasets.values().collect()
    }

    /// List all jobs, optionally filtered by status
    pub fn list_jobs(&self, status_filter: Option<&JobStatus>) -> Vec<&FinetuningJob> {
        match status_filter {
            Some(status) => self.jobs.values().filter(|j| &j.status == status).collect(),
            None => self.jobs.values().collect(),
        }
    }
}

/// Statistics about the fine-tuning manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinetuningManagerStats {
    pub total_datasets: usize,
    pub total_examples: usize,
    pub total_jobs: usize,
    pub active_jobs: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub total_feedback: usize,
    pub average_rating: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn make_example(human: &str, assistant: &str) -> TrainingExample {
        TrainingExample::new(human.to_string(), assistant.to_string())
    }

    #[test]
    fn test_training_example_quality_score() {
        let mut example = make_example(
            "How do I write a SPARQL query to find all classes?",
            "To find all classes in a SPARQL endpoint, you can use the following query:\n\nSELECT ?class WHERE { ?class a owl:Class }",
        );
        example.compute_quality_score();
        assert!(example.quality_score > 0.0);
        assert!(example.quality_score <= 1.0);
    }

    #[test]
    fn test_openai_format_export() {
        let example = make_example("What is RDF?", "RDF (Resource Description Framework) is a standard model for data interchange on the Web.");
        let json = example.to_openai_format();
        assert!(json.get("messages").is_some());
        let messages = json["messages"].as_array().expect("messages array");
        assert!(!messages.is_empty());
    }

    #[test]
    fn test_alpaca_format_export() {
        let example = make_example("Explain SPARQL", "SPARQL is a query language for RDF data.");
        let json = example.to_alpaca_format();
        assert!(json.get("instruction").is_some());
        assert!(json.get("output").is_some());
    }

    #[test]
    fn test_dataset_add_and_deduplicate() {
        let mut dataset = TrainingDataset::new("test-ds".to_string(), "Test Dataset".to_string());

        dataset.add_example(make_example("What is RDF?", "RDF is..."));
        dataset.add_example(make_example("What is SPARQL?", "SPARQL is..."));
        dataset.add_example(make_example("What is RDF?", "RDF is...")); // duplicate

        assert_eq!(dataset.examples.len(), 3);

        let removed = dataset.deduplicate();
        assert_eq!(removed, 1);
        assert_eq!(dataset.examples.len(), 2);
    }

    #[test]
    fn test_dataset_quality_filtering() {
        let mut dataset = TrainingDataset::new("filter-ds".to_string(), "Filter Test".to_string());

        let mut low_quality = make_example("Hi", "Hi!");
        low_quality.quality_score = 0.1;
        dataset.examples.push(low_quality);

        let mut high_quality = make_example(
            "How does SPARQL federation work?",
            "SPARQL federation allows querying multiple endpoints simultaneously using the SERVICE keyword.",
        );
        high_quality.quality_score = 0.9;
        dataset.examples.push(high_quality);

        let high = dataset.examples_above_quality(0.5);
        assert_eq!(high.len(), 1);
    }

    #[test]
    fn test_dataset_export_jsonl() {
        let mut dataset = TrainingDataset::new("export-ds".to_string(), "Export Test".to_string());

        let mut example = make_example(
            "What is OWL?",
            "OWL (Web Ontology Language) is a semantic web language for defining ontologies.",
        );
        example.quality_score = 0.8;
        dataset.examples.push(example);

        let jsonl = dataset
            .export_jsonl(&ExportFormat::OpenAiChatJsonl, 0.5)
            .expect("export");
        assert!(!jsonl.is_empty());
        // Verify it's valid JSON
        let line = jsonl.lines().next().expect("at least one line");
        serde_json::from_str::<serde_json::Value>(line).expect("valid JSON");
    }

    #[test]
    fn test_finetuning_manager_create_and_add() {
        let dir =
            env::temp_dir().join(format!("oxirs_finetuning_test_{}", Uuid::new_v4().simple()));
        let config = FinetuningManagerConfig {
            storage_dir: dir.clone(),
            min_examples_for_export: 1,
            ..Default::default()
        };
        let mut manager = FinetuningManager::new(config).expect("create manager");

        let dataset_id = manager
            .create_dataset("Test Dataset".to_string(), "A test dataset".to_string())
            .expect("create dataset");

        for i in 0..5 {
            let mut example = make_example(
                &format!("Question {}?", i),
                &format!("Answer {} with more detail about semantic web and RDF.", i),
            );
            example.quality_score = 0.8;
            manager
                .add_example(&dataset_id, example)
                .expect("add example");
        }

        let stats = manager.statistics();
        assert_eq!(stats.total_datasets, 1);
        assert_eq!(stats.total_examples, 5);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_create_finetuning_job() {
        let dir = env::temp_dir().join(format!("oxirs_job_test_{}", Uuid::new_v4().simple()));
        let config = FinetuningManagerConfig {
            storage_dir: dir.clone(),
            min_examples_for_export: 1,
            ..Default::default()
        };
        let mut manager = FinetuningManager::new(config).expect("create manager");

        let dataset_id = manager
            .create_dataset("Job Test".to_string(), "For job creation test".to_string())
            .expect("create dataset");

        for i in 0..3 {
            let mut example = make_example(
                &format!("SPARQL question {}?", i),
                &format!(
                    "Detailed SPARQL answer {} explaining the query pattern and expected results.",
                    i
                ),
            );
            example.quality_score = 0.8;
            manager
                .add_example(&dataset_id, example)
                .expect("add example");
        }

        let job_id = manager
            .create_job(&dataset_id, FinetuningJobConfig::default())
            .expect("create job");

        let job = manager.get_job(&job_id).expect("get job");
        assert_eq!(job.status, JobStatus::Pending);
        assert!(!job.id.is_empty());

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_job_lifecycle() {
        let job_config = FinetuningJobConfig::default();
        let mut job = FinetuningJob::new("dataset-1".to_string(), job_config);

        assert_eq!(job.status, JobStatus::Pending);
        assert!(job.is_active());
        assert_eq!(job.progress(), 0.0);

        job.start();
        assert_eq!(job.status, JobStatus::Training);

        job.record_step(10, 0.85, Some(0.90));
        assert_eq!(job.training_loss.len(), 1);

        job.complete("ft:gpt-3.5-turbo:oxirs".to_string());
        assert_eq!(job.status, JobStatus::Completed);
        assert!(!job.is_active());
        assert_eq!(job.progress(), 1.0);
        assert!(job.completed_at.is_some());
    }

    #[test]
    fn test_feedback_collection() {
        let dir = env::temp_dir().join(format!("oxirs_feedback_test_{}", Uuid::new_v4().simple()));
        let config = FinetuningManagerConfig {
            storage_dir: dir.clone(),
            ..Default::default()
        };
        let mut manager = FinetuningManager::new(config).expect("create manager");

        let feedback = ConversationFeedback::new("session-abc".to_string(), 5);
        manager.submit_feedback(feedback).expect("submit feedback");

        let session_feedback = manager.get_session_feedback("session-abc");
        assert_eq!(session_feedback.len(), 1);
        assert_eq!(session_feedback[0].rating, 5);

        let stats = manager.statistics();
        assert_eq!(stats.total_feedback, 1);
        assert_eq!(stats.average_rating, 5.0);

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_export_to_file() {
        let dir = env::temp_dir().join(format!("oxirs_export_test_{}", Uuid::new_v4().simple()));
        let config = FinetuningManagerConfig {
            storage_dir: dir.clone(),
            min_examples_for_export: 1,
            ..Default::default()
        };
        let mut manager = FinetuningManager::new(config).expect("create manager");

        let dataset_id = manager
            .create_dataset("Export Test".to_string(), "For export test".to_string())
            .expect("create dataset");

        let mut example = make_example(
            "What is triple store?",
            "A triple store is a purpose-built database for storing and querying RDF triples.",
        );
        example.quality_score = 0.9;
        manager.add_example(&dataset_id, example).expect("add");

        let output_path = dir.join("training_data.jsonl");
        let count = manager
            .export_dataset(
                &dataset_id,
                &output_path,
                &ExportFormat::OpenAiChatJsonl,
                Some(0.5),
            )
            .expect("export");

        assert_eq!(count, 1);
        assert!(output_path.exists());

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_dataset_stats_computation() {
        let mut dataset = TrainingDataset::new("stats-ds".to_string(), "Stats Test".to_string());

        for i in 0..10 {
            let mut example = make_example(
                &format!("Question {} about semantic web?", i),
                &format!("Detailed answer about semantic web topic {}.", i),
            );
            if i < 5 {
                example.quality_score = 0.8;
                example.human_reviewed = true;
            } else {
                example.quality_score = 0.4;
            }
            dataset.examples.push(example);
        }

        dataset.recompute_stats();

        assert_eq!(dataset.stats.total_examples, 10);
        assert_eq!(dataset.stats.high_quality_examples, 5);
        assert_eq!(dataset.stats.human_reviewed_examples, 5);
        assert!(dataset.stats.average_quality_score > 0.0);
    }
}
