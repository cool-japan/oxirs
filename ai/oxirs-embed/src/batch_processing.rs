//! Offline batch embedding generation with incremental updates
//!
//! This module provides comprehensive batch processing capabilities for generating
//! embeddings offline, with support for incremental updates, resumable jobs,
//! and efficient resource utilization.

use crate::{CacheManager, EmbeddingModel};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::fs;
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Batch processing manager for offline embedding generation
pub struct BatchProcessingManager {
    /// Active batch jobs
    active_jobs: Arc<RwLock<HashMap<Uuid, BatchJob>>>,
    /// Configuration
    config: BatchProcessingConfig,
    /// Cache manager for optimization
    cache_manager: Arc<CacheManager>,
    /// Concurrency semaphore
    semaphore: Arc<Semaphore>,
    /// Job persistence directory
    persistence_dir: PathBuf,
}

/// Configuration for batch processing
#[derive(Debug, Clone)]
pub struct BatchProcessingConfig {
    /// Maximum concurrent workers
    pub max_workers: usize,
    /// Chunk size for processing
    pub chunk_size: usize,
    /// Enable incremental updates
    pub enable_incremental: bool,
    /// Checkpoint frequency (number of chunks)
    pub checkpoint_frequency: usize,
    /// Enable resume from checkpoint
    pub enable_resume: bool,
    /// Maximum memory usage per worker (MB)
    pub max_memory_per_worker_mb: usize,
    /// Enable progress notifications
    pub enable_notifications: bool,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Output format configuration
    pub output_config: OutputConfig,
}

impl Default for BatchProcessingConfig {
    fn default() -> Self {
        Self {
            max_workers: num_cpus::get(),
            chunk_size: 1000,
            enable_incremental: true,
            checkpoint_frequency: 10,
            enable_resume: true,
            max_memory_per_worker_mb: 512,
            enable_notifications: true,
            retry_config: RetryConfig::default(),
            output_config: OutputConfig::default(),
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Initial backoff delay in milliseconds
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds
    pub max_backoff_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Output configuration
#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// Output format
    pub format: OutputFormat,
    /// Compression level (0-9)
    pub compression_level: u32,
    /// Include metadata
    pub include_metadata: bool,
    /// Batch output into files
    pub batch_output: bool,
    /// Maximum entities per output file
    pub max_entities_per_file: usize,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Parquet,
            compression_level: 6,
            include_metadata: true,
            batch_output: true,
            max_entities_per_file: 100_000,
        }
    }
}

/// Output formats
#[derive(Debug, Clone)]
pub enum OutputFormat {
    /// Apache Parquet format
    Parquet,
    /// Compressed JSON Lines
    JsonLines,
    /// Binary format (custom)
    Binary,
    /// HDF5 format
    HDF5,
}

/// Batch job definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    /// Unique job ID
    pub job_id: Uuid,
    /// Job name
    pub name: String,
    /// Job status
    pub status: JobStatus,
    /// Input specification
    pub input: BatchInput,
    /// Output specification
    pub output: BatchOutput,
    /// Processing configuration
    pub config: BatchJobConfig,
    /// Model information
    pub model_id: Uuid,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Started timestamp
    pub started_at: Option<DateTime<Utc>>,
    /// Completed timestamp
    pub completed_at: Option<DateTime<Utc>>,
    /// Progress information
    pub progress: JobProgress,
    /// Error information
    pub error: Option<String>,
    /// Checkpoint data
    pub checkpoint: Option<JobCheckpoint>,
}

/// Job status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Paused,
}

/// Batch input specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInput {
    /// Input type
    pub input_type: InputType,
    /// Input source
    pub source: String,
    /// Filter criteria
    pub filters: Option<HashMap<String, String>>,
    /// Incremental mode settings
    pub incremental: Option<IncrementalConfig>,
}

/// Input types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputType {
    /// List of entity IDs
    EntityList,
    /// File containing entity IDs
    EntityFile,
    /// SPARQL query result
    SparqlQuery,
    /// Database query
    DatabaseQuery,
    /// Stream source
    StreamSource,
}

/// Incremental processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Enable incremental processing
    pub enabled: bool,
    /// Last processed timestamp
    pub last_processed: Option<DateTime<Utc>>,
    /// Timestamp field name
    pub timestamp_field: String,
    /// Check for deletions
    pub check_deletions: bool,
    /// Existing embeddings source
    pub existing_embeddings_path: Option<String>,
}

/// Batch output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOutput {
    /// Output path
    pub path: String,
    /// Output format
    pub format: String,
    /// Compression settings
    pub compression: Option<String>,
    /// Partitioning strategy
    pub partitioning: Option<PartitioningStrategy>,
}

/// Partitioning strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitioningStrategy {
    /// No partitioning
    None,
    /// Partition by entity type
    ByEntityType,
    /// Partition by date
    ByDate,
    /// Partition by hash
    ByHash { num_partitions: usize },
    /// Custom partitioning
    Custom { field: String },
}

/// Job-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJobConfig {
    /// Chunk size for this job
    pub chunk_size: usize,
    /// Number of workers
    pub num_workers: usize,
    /// Retry configuration
    pub max_retries: usize,
    /// Enable caching
    pub use_cache: bool,
    /// Custom parameters
    pub custom_params: HashMap<String, String>,
}

/// Job progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobProgress {
    /// Total entities to process
    pub total_entities: usize,
    /// Entities processed
    pub processed_entities: usize,
    /// Entities failed
    pub failed_entities: usize,
    /// Current chunk being processed
    pub current_chunk: usize,
    /// Total chunks
    pub total_chunks: usize,
    /// Processing rate (entities/second)
    pub processing_rate: f64,
    /// Estimated time remaining
    pub eta_seconds: Option<u64>,
    /// Memory usage (MB)
    pub memory_usage_mb: f64,
}

impl Default for JobProgress {
    fn default() -> Self {
        Self {
            total_entities: 0,
            processed_entities: 0,
            failed_entities: 0,
            current_chunk: 0,
            total_chunks: 0,
            processing_rate: 0.0,
            eta_seconds: None,
            memory_usage_mb: 0.0,
        }
    }
}

/// Job checkpoint for resumability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobCheckpoint {
    /// Checkpoint timestamp
    pub timestamp: DateTime<Utc>,
    /// Last processed entity index
    pub last_processed_index: usize,
    /// Processed entity IDs
    pub processed_entities: HashSet<String>,
    /// Failed entity IDs with error messages
    pub failed_entities: HashMap<String, String>,
    /// Intermediate results path
    pub intermediate_results_path: String,
    /// Model state hash
    pub model_state_hash: String,
}

/// Batch processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingResult {
    /// Job ID
    pub job_id: Uuid,
    /// Processing statistics
    pub stats: BatchProcessingStats,
    /// Output information
    pub output_info: OutputInfo,
    /// Quality metrics
    pub quality_metrics: Option<QualityMetrics>,
}

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingStats {
    /// Total processing time
    pub total_time_seconds: f64,
    /// Total entities processed
    pub total_entities: usize,
    /// Successful embeddings
    pub successful_embeddings: usize,
    /// Failed embeddings
    pub failed_embeddings: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Average processing time per entity (ms)
    pub avg_time_per_entity_ms: f64,
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
}

/// Output information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputInfo {
    /// Output files created
    pub output_files: Vec<String>,
    /// Total output size (bytes)
    pub total_size_bytes: u64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Number of partitions
    pub num_partitions: usize,
}

/// Quality metrics for batch processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Average embedding norm
    pub avg_embedding_norm: f64,
    /// Embedding norm standard deviation
    pub embedding_norm_std: f64,
    /// Average cosine similarity to centroid
    pub avg_cosine_similarity: f64,
    /// Embedding dimension
    pub embedding_dimension: usize,
    /// Number of zero embeddings
    pub zero_embeddings: usize,
    /// Number of NaN embeddings
    pub nan_embeddings: usize,
}

impl BatchProcessingManager {
    /// Create a new batch processing manager
    pub fn new(
        config: BatchProcessingConfig,
        cache_manager: Arc<CacheManager>,
        persistence_dir: PathBuf,
    ) -> Self {
        Self {
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(config.max_workers)),
            config,
            cache_manager,
            persistence_dir,
        }
    }

    /// Submit a new batch job
    pub async fn submit_job(&self, job: BatchJob) -> Result<Uuid> {
        let job_id = job.job_id;

        // Validate job
        self.validate_job(&job).await?;

        // Store job
        {
            let mut jobs = self.active_jobs.write().await;
            jobs.insert(job_id, job.clone());
        }

        // Persist job configuration
        self.persist_job(&job).await?;

        info!("Submitted batch job: {} ({})", job.name, job_id);
        Ok(job_id)
    }

    /// Start processing a batch job
    pub async fn start_job(
        &self,
        job_id: Uuid,
        model: Arc<dyn EmbeddingModel + Send + Sync>,
    ) -> Result<JoinHandle<Result<BatchProcessingResult>>> {
        let job = {
            let mut jobs = self.active_jobs.write().await;
            let job = jobs
                .get_mut(&job_id)
                .ok_or_else(|| anyhow!("Job not found: {}", job_id))?;

            if !matches!(job.status, JobStatus::Pending | JobStatus::Paused) {
                return Err(anyhow!("Job {} is not in a startable state", job_id));
            }

            job.status = JobStatus::Running;
            job.started_at = Some(Utc::now());
            job.clone()
        };

        let manager = self.clone();
        let handle = tokio::spawn(async move { manager.process_job(job, model).await });

        Ok(handle)
    }

    /// Process a batch job
    async fn process_job(
        &self,
        job: BatchJob,
        model: Arc<dyn EmbeddingModel + Send + Sync>,
    ) -> Result<BatchProcessingResult> {
        let start_time = Instant::now();
        info!(
            "Starting batch job processing: {} ({})",
            job.name, job.job_id
        );

        // Load entities to process
        let entities = self.load_entities(&job).await?;

        // Filter entities for incremental processing
        let entities_to_process = if job
            .input
            .incremental
            .as_ref()
            .map(|inc| inc.enabled)
            .unwrap_or(false)
        {
            self.filter_incremental_entities(&job, entities).await?
        } else {
            entities
        };

        // Update job progress
        {
            let mut jobs = self.active_jobs.write().await;
            if let Some(active_job) = jobs.get_mut(&job.job_id) {
                active_job.progress.total_entities = entities_to_process.len();
                active_job.progress.total_chunks =
                    (entities_to_process.len() + job.config.chunk_size - 1) / job.config.chunk_size;
            }
        }

        // Process entities in chunks
        let chunks: Vec<_> = entities_to_process
            .chunks(job.config.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut successful_embeddings = 0;
        let mut failed_embeddings = 0;
        let mut cache_hits = 0;
        let mut cache_misses = 0;
        let mut processed_entities = HashSet::new();
        let mut failed_entities = HashMap::new();

        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            // Check if job was cancelled
            {
                let jobs = self.active_jobs.read().await;
                if let Some(active_job) = jobs.get(&job.job_id) {
                    if matches!(active_job.status, JobStatus::Cancelled) {
                        info!("Job {} was cancelled", job.job_id);
                        return Err(anyhow!("Job was cancelled"));
                    }
                }
            }

            // Process chunk
            let chunk_result = self
                .process_chunk(&job, chunk, chunk_idx, model.clone())
                .await?;

            // Update statistics
            successful_embeddings += chunk_result.successful;
            failed_embeddings += chunk_result.failed;
            cache_hits += chunk_result.cache_hits;
            cache_misses += chunk_result.cache_misses;

            // Track processed entities
            for entity in chunk {
                processed_entities.insert(entity.clone());
            }
            for (entity, error) in chunk_result.failures {
                failed_entities.insert(entity, error);
            }

            // Update progress
            self.update_job_progress(
                &job.job_id,
                chunk_idx + 1,
                successful_embeddings + failed_embeddings,
            )
            .await?;

            // Create checkpoint
            if chunk_idx % self.config.checkpoint_frequency == 0 {
                self.create_checkpoint(&job.job_id, &processed_entities, &failed_entities)
                    .await?;
            }

            info!(
                "Processed chunk {}/{} for job {}",
                chunk_idx + 1,
                chunks.len(),
                job.job_id
            );
        }

        // Finalize processing
        let processing_time = start_time.elapsed().as_secs_f64();
        let result = self
            .finalize_job_processing(
                &job,
                processing_time,
                successful_embeddings,
                failed_embeddings,
                cache_hits,
                cache_misses,
            )
            .await?;

        // Update job status
        {
            let mut jobs = self.active_jobs.write().await;
            if let Some(active_job) = jobs.get_mut(&job.job_id) {
                active_job.status = JobStatus::Completed;
                active_job.completed_at = Some(Utc::now());
            }
        }

        info!(
            "Completed batch job: {} in {:.2}s",
            job.job_id, processing_time
        );
        Ok(result)
    }

    /// Process a single chunk of entities
    async fn process_chunk(
        &self,
        job: &BatchJob,
        entities: &[String],
        chunk_idx: usize,
        model: Arc<dyn EmbeddingModel + Send + Sync>,
    ) -> Result<ChunkResult> {
        let _permit = self.semaphore.acquire().await?;

        let mut successful = 0;
        let mut failed = 0;
        let mut cache_hits = 0;
        let mut cache_misses = 0;
        let mut failures = HashMap::new();

        for entity in entities {
            match self
                .process_single_entity(entity, model.clone(), job.config.use_cache)
                .await
            {
                Ok(from_cache) => {
                    successful += 1;
                    if from_cache {
                        cache_hits += 1;
                    } else {
                        cache_misses += 1;
                    }
                }
                Err(e) => {
                    failed += 1;
                    failures.insert(entity.clone(), e.to_string());
                    warn!("Failed to process entity {}: {}", entity, e);
                }
            }
        }

        Ok(ChunkResult {
            chunk_idx,
            successful,
            failed,
            cache_hits,
            cache_misses,
            failures,
        })
    }

    /// Process a single entity
    async fn process_single_entity(
        &self,
        entity: &str,
        model: Arc<dyn EmbeddingModel + Send + Sync>,
        use_cache: bool,
    ) -> Result<bool> {
        if use_cache {
            // Check cache first
            if let Some(_embedding) = self.cache_manager.get_embedding(entity) {
                return Ok(true);
            }
        }

        // Generate embedding
        let embedding = model.get_entity_embedding(entity)?;

        // Cache the result
        if use_cache {
            self.cache_manager
                .put_embedding(entity.to_string(), embedding);
        }

        Ok(false)
    }

    /// Load entities to process based on input specification
    async fn load_entities(&self, job: &BatchJob) -> Result<Vec<String>> {
        match &job.input.input_type {
            InputType::EntityList => {
                // Parse entity list from source
                let entities: Vec<String> = serde_json::from_str(&job.input.source)?;
                Ok(entities)
            }
            InputType::EntityFile => {
                // Read entities from file
                let content = fs::read_to_string(&job.input.source).await?;
                let entities: Vec<String> = content
                    .lines()
                    .map(|line| line.trim().to_string())
                    .filter(|line| !line.is_empty())
                    .collect();
                Ok(entities)
            }
            InputType::SparqlQuery => {
                // Execute SPARQL query and extract entities
                // This would need to be implemented based on SPARQL engine
                warn!("SPARQL query input type not yet implemented");
                Ok(Vec::new())
            }
            InputType::DatabaseQuery => {
                // Execute database query and extract entities
                warn!("Database query input type not yet implemented");
                Ok(Vec::new())
            }
            InputType::StreamSource => {
                // Read from stream source
                warn!("Stream source input type not yet implemented");
                Ok(Vec::new())
            }
        }
    }

    /// Filter entities for incremental processing
    async fn filter_incremental_entities(
        &self,
        job: &BatchJob,
        entities: Vec<String>,
    ) -> Result<Vec<String>> {
        if let Some(incremental) = &job.input.incremental {
            if !incremental.enabled {
                return Ok(entities);
            }

            // Load existing embeddings if specified
            let existing_entities =
                if let Some(existing_path) = &incremental.existing_embeddings_path {
                    self.load_existing_entities(existing_path).await?
                } else {
                    HashSet::new()
                };

            // Filter out entities that already have embeddings
            let filtered: Vec<String> = entities
                .into_iter()
                .filter(|entity| !existing_entities.contains(entity))
                .collect();

            info!(
                "Incremental filtering: {} entities remaining after filtering",
                filtered.len()
            );
            Ok(filtered)
        } else {
            Ok(entities)
        }
    }

    /// Load existing entities from embeddings file
    async fn load_existing_entities(&self, path: &str) -> Result<HashSet<String>> {
        // This would depend on the output format
        // For now, assume a simple text file with entity IDs
        if Path::new(path).exists() {
            let content = fs::read_to_string(path).await?;
            let entities: HashSet<String> = content
                .lines()
                .map(|line| line.trim().to_string())
                .filter(|line| !line.is_empty())
                .collect();
            Ok(entities)
        } else {
            Ok(HashSet::new())
        }
    }

    /// Update job progress
    async fn update_job_progress(
        &self,
        job_id: &Uuid,
        current_chunk: usize,
        processed_entities: usize,
    ) -> Result<()> {
        let mut jobs = self.active_jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.progress.current_chunk = current_chunk;
            job.progress.processed_entities = processed_entities;

            // Calculate processing rate
            if let Some(started_at) = job.started_at {
                let elapsed = Utc::now().signed_duration_since(started_at);
                let elapsed_seconds = elapsed.num_seconds() as f64;
                if elapsed_seconds > 0.0 {
                    job.progress.processing_rate = processed_entities as f64 / elapsed_seconds;

                    // Estimate time remaining
                    let remaining_entities = job.progress.total_entities - processed_entities;
                    if job.progress.processing_rate > 0.0 {
                        let eta = remaining_entities as f64 / job.progress.processing_rate;
                        job.progress.eta_seconds = Some(eta as u64);
                    }
                }
            }
        }
        Ok(())
    }

    /// Create a checkpoint for job resumability
    async fn create_checkpoint(
        &self,
        job_id: &Uuid,
        processed_entities: &HashSet<String>,
        failed_entities: &HashMap<String, String>,
    ) -> Result<()> {
        let checkpoint = JobCheckpoint {
            timestamp: Utc::now(),
            last_processed_index: processed_entities.len(),
            processed_entities: processed_entities.clone(),
            failed_entities: failed_entities.clone(),
            intermediate_results_path: format!(
                "{}/checkpoint_{}.json",
                self.persistence_dir.display(),
                job_id
            ),
            model_state_hash: "placeholder".to_string(), // Would calculate actual hash
        };

        // Save checkpoint to disk
        let checkpoint_path = self
            .persistence_dir
            .join(format!("checkpoint_{job_id}.json"));
        let checkpoint_json = serde_json::to_string_pretty(&checkpoint)?;
        fs::write(checkpoint_path, checkpoint_json).await?;

        // Update job with checkpoint
        let mut jobs = self.active_jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.checkpoint = Some(checkpoint);
        }

        debug!("Created checkpoint for job {}", job_id);
        Ok(())
    }

    /// Finalize job processing and create result
    async fn finalize_job_processing(
        &self,
        job: &BatchJob,
        processing_time: f64,
        successful_embeddings: usize,
        failed_embeddings: usize,
        cache_hits: usize,
        cache_misses: usize,
    ) -> Result<BatchProcessingResult> {
        let total_entities = successful_embeddings + failed_embeddings;
        let avg_time_per_entity_ms = if total_entities > 0 {
            (processing_time * 1000.0) / total_entities as f64
        } else {
            0.0
        };

        let stats = BatchProcessingStats {
            total_time_seconds: processing_time,
            total_entities,
            successful_embeddings,
            failed_embeddings,
            cache_hits,
            cache_misses,
            avg_time_per_entity_ms,
            peak_memory_mb: 0.0,  // Would measure actual memory usage
            cpu_utilization: 0.0, // Would measure actual CPU usage
        };

        let output_info = OutputInfo {
            output_files: vec![job.output.path.clone()],
            total_size_bytes: 0, // Would calculate actual size
            compression_ratio: 1.0,
            num_partitions: 1,
        };

        Ok(BatchProcessingResult {
            job_id: job.job_id,
            stats,
            output_info,
            quality_metrics: None, // Would calculate if requested
        })
    }

    /// Validate a batch job before submission
    async fn validate_job(&self, job: &BatchJob) -> Result<()> {
        // Validate input source exists
        match &job.input.input_type {
            InputType::EntityFile => {
                if !Path::new(&job.input.source).exists() {
                    return Err(anyhow!("Input file does not exist: {}", job.input.source));
                }
            }
            _ => {} // Other validations would be implemented
        }

        // Validate output path is writable
        if let Some(parent) = Path::new(&job.output.path).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).await?;
            }
        }

        Ok(())
    }

    /// Persist job configuration to disk
    async fn persist_job(&self, job: &BatchJob) -> Result<()> {
        let job_path = self
            .persistence_dir
            .join(format!("job_{}.json", job.job_id));
        let job_json = serde_json::to_string_pretty(job)?;
        fs::write(job_path, job_json).await?;
        Ok(())
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: &Uuid) -> Option<JobStatus> {
        let jobs = self.active_jobs.read().await;
        jobs.get(job_id).map(|job| job.status.clone())
    }

    /// Get job progress
    pub async fn get_job_progress(&self, job_id: &Uuid) -> Option<JobProgress> {
        let jobs = self.active_jobs.read().await;
        jobs.get(job_id).map(|job| job.progress.clone())
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: &Uuid) -> Result<()> {
        let mut jobs = self.active_jobs.write().await;
        if let Some(job) = jobs.get_mut(job_id) {
            job.status = JobStatus::Cancelled;
            info!("Cancelled job: {}", job_id);
            Ok(())
        } else {
            Err(anyhow!("Job not found: {}", job_id))
        }
    }

    /// List all jobs
    pub async fn list_jobs(&self) -> Vec<BatchJob> {
        let jobs = self.active_jobs.read().await;
        jobs.values().cloned().collect()
    }
}

impl Clone for BatchProcessingManager {
    fn clone(&self) -> Self {
        Self {
            active_jobs: Arc::clone(&self.active_jobs),
            config: self.config.clone(),
            cache_manager: Arc::clone(&self.cache_manager),
            semaphore: Arc::clone(&self.semaphore),
            persistence_dir: self.persistence_dir.clone(),
        }
    }
}

/// Result of processing a single chunk
#[derive(Debug)]
#[allow(dead_code)]
struct ChunkResult {
    chunk_idx: usize,
    successful: usize,
    failed: usize,
    cache_hits: usize,
    cache_misses: usize,
    failures: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_batch_job_creation() {
        let job = BatchJob {
            job_id: Uuid::new_v4(),
            name: "test_job".to_string(),
            status: JobStatus::Pending,
            input: BatchInput {
                input_type: InputType::EntityList,
                source: r#"["entity1", "entity2", "entity3"]"#.to_string(),
                filters: None,
                incremental: None,
            },
            output: BatchOutput {
                path: "/tmp/output".to_string(),
                format: "parquet".to_string(),
                compression: Some("gzip".to_string()),
                partitioning: Some(PartitioningStrategy::None),
            },
            config: BatchJobConfig {
                chunk_size: 100,
                num_workers: 4,
                max_retries: 3,
                use_cache: true,
                custom_params: HashMap::new(),
            },
            model_id: Uuid::new_v4(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            progress: JobProgress::default(),
            error: None,
            checkpoint: None,
        };

        assert_eq!(job.status, JobStatus::Pending);
        assert_eq!(job.name, "test_job");
    }

    #[tokio::test]
    async fn test_batch_processing_manager_creation() {
        let config = BatchProcessingConfig::default();
        let cache_config = crate::CacheConfig::default();
        let cache_manager = Arc::new(CacheManager::new(cache_config));
        let temp_dir = tempdir().unwrap();

        let manager =
            BatchProcessingManager::new(config, cache_manager, temp_dir.path().to_path_buf());

        assert_eq!(manager.config.max_workers, num_cpus::get());
        assert_eq!(manager.config.chunk_size, 1000);
    }

    #[test]
    fn test_incremental_config() {
        let incremental = IncrementalConfig {
            enabled: true,
            last_processed: Some(Utc::now()),
            timestamp_field: "updated_at".to_string(),
            check_deletions: true,
            existing_embeddings_path: Some("/path/to/existing".to_string()),
        };

        assert!(incremental.enabled);
        assert!(incremental.last_processed.is_some());
        assert_eq!(incremental.timestamp_field, "updated_at");
    }
}
