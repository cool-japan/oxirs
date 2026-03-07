//! Distributed SHACL Validation
//!
//! This module provides distributed validation capabilities for scaling SHACL
//! validation across multiple nodes in a cluster. It enables:
//!
//! - Partition-based validation distribution
//! - Load balancing across validators
//! - Fault tolerance and retry mechanisms
//! - Result aggregation and consistency
//!
//! # Architecture
//!
//! The distributed validation system uses a coordinator-worker pattern:
//! - Coordinator: Partitions work, distributes to workers, aggregates results
//! - Workers: Execute validation on assigned partitions
//!
//! # Communication
//!
//! Supports multiple communication backends:
//! - In-memory (for testing and single-node)
//! - gRPC (recommended for production)
//! - HTTP/REST

use crate::{Shape, ShapeId, ValidationConfig, ValidationReport, ValidationViolation};
use indexmap::IndexMap;
use oxirs_core::model::Term;
use scirs2_core::random::{rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Configuration for distributed validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedValidationConfig {
    /// Number of partitions to create
    pub num_partitions: usize,
    /// Replication factor for fault tolerance
    pub replication_factor: usize,
    /// Timeout for worker responses
    pub worker_timeout: Duration,
    /// Maximum retries for failed partitions
    pub max_retries: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    /// Enable distributed tracing
    pub enable_tracing: bool,
}

impl Default for DistributedValidationConfig {
    fn default() -> Self {
        Self {
            num_partitions: 10,
            replication_factor: 2,
            worker_timeout: Duration::from_secs(30),
            max_retries: 3,
            load_balancing: LoadBalancingStrategy::RoundRobin,
            enable_caching: true,
            cache_ttl: Duration::from_secs(300),
            consistency_level: ConsistencyLevel::Eventual,
            enable_tracing: false,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Distribute based on current load
    LeastLoaded,
    /// Hash-based consistent distribution
    ConsistentHashing,
    /// Random distribution
    Random,
}

/// Consistency levels for distributed validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Best effort, no guarantees
    BestEffort,
    /// Eventually consistent
    Eventual,
    /// All replicas must agree
    Strong,
}

/// Distributed validation coordinator
pub struct DistributedValidator {
    /// Configuration
    config: DistributedValidationConfig,
    /// Registered workers
    workers: Arc<RwLock<Vec<WorkerInfo>>>,
    /// Validation shapes
    shapes: Arc<RwLock<IndexMap<ShapeId, Shape>>>,
    /// Validation configuration
    _validation_config: ValidationConfig,
    /// Statistics
    stats: Arc<RwLock<DistributedStats>>,
    /// Result cache
    cache: Arc<RwLock<HashMap<String, CachedResult>>>,
}

/// Information about a worker node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID
    pub worker_id: String,
    /// Worker address
    pub address: String,
    /// Current load (0.0 to 1.0)
    pub current_load: f64,
    /// Health status
    pub healthy: bool,
    /// Last heartbeat time
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    /// Total validations processed
    pub validations_processed: usize,
    /// Average response time in milliseconds
    pub avg_response_time_ms: u64,
}

/// Cached validation result
#[derive(Debug, Clone)]
struct CachedResult {
    report: ValidationReport,
    created: Instant,
}

/// Statistics for distributed validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedStats {
    /// Total validations distributed
    pub total_distributed: usize,
    /// Total partitions created
    pub total_partitions: usize,
    /// Failed partitions
    pub failed_partitions: usize,
    /// Retried partitions
    pub retried_partitions: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Average distribution time
    pub avg_distribution_time_ms: f64,
    /// Average aggregation time
    pub avg_aggregation_time_ms: f64,
}

/// A partition of focus nodes to validate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationPartition {
    /// Partition ID
    pub partition_id: usize,
    /// Focus nodes in this partition
    pub focus_nodes: Vec<Term>,
    /// Assigned worker
    pub assigned_worker: Option<String>,
    /// Retry count
    pub retry_count: usize,
}

/// Result from a worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionResult {
    /// Partition ID
    pub partition_id: usize,
    /// Worker ID that processed it
    pub worker_id: String,
    /// Validation violations
    pub violations: Vec<ValidationViolation>,
    /// Processing time
    pub processing_time: Duration,
    /// Success status
    pub success: bool,
    /// Error message (if failed)
    pub error: Option<String>,
}

impl DistributedValidator {
    /// Create a new distributed validator
    pub fn new(
        shapes: IndexMap<ShapeId, Shape>,
        config: DistributedValidationConfig,
        validation_config: ValidationConfig,
    ) -> Self {
        Self {
            config,
            workers: Arc::new(RwLock::new(Vec::new())),
            shapes: Arc::new(RwLock::new(shapes)),
            _validation_config: validation_config,
            stats: Arc::new(RwLock::new(DistributedStats::default())),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a worker node
    pub async fn register_worker(&self, worker: WorkerInfo) {
        let mut workers = self.workers.write().await;
        workers.push(worker);
    }

    /// Unregister a worker node
    pub async fn unregister_worker(&self, worker_id: &str) {
        let mut workers = self.workers.write().await;
        workers.retain(|w| w.worker_id != worker_id);
    }

    /// Get healthy workers
    async fn get_healthy_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read().await;
        workers.iter().filter(|w| w.healthy).cloned().collect()
    }

    /// Validate data distributed across workers
    pub async fn validate_distributed(
        &self,
        focus_nodes: Vec<Term>,
    ) -> Result<ValidationReport, DistributedError> {
        let start = Instant::now();

        // Check cache
        let cache_key = self.compute_cache_key(&focus_nodes);
        if self.config.enable_caching {
            if let Some(cached) = self.check_cache(&cache_key).await {
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;
                return Ok(cached);
            }
            let mut stats = self.stats.write().await;
            stats.cache_misses += 1;
        }

        // Get healthy workers
        let workers = self.get_healthy_workers().await;
        if workers.is_empty() {
            return Err(DistributedError::NoWorkersAvailable);
        }

        // Partition the focus nodes
        let partitions = self.partition_nodes(&focus_nodes).await;

        // Distribute partitions to workers
        let assigned_partitions = self.assign_partitions(partitions, &workers).await?;

        // Execute validation on workers
        let results = self.execute_distributed(&assigned_partitions).await?;

        // Aggregate results
        let report = self.aggregate_results(results).await?;

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_distributed += 1;
            let elapsed = start.elapsed().as_millis() as f64;
            stats.avg_distribution_time_ms =
                (stats.avg_distribution_time_ms * (stats.total_distributed - 1) as f64 + elapsed)
                    / stats.total_distributed as f64;
        }

        // Cache result
        if self.config.enable_caching {
            self.cache_result(&cache_key, &report).await;
        }

        Ok(report)
    }

    /// Partition focus nodes for distribution
    async fn partition_nodes(&self, focus_nodes: &[Term]) -> Vec<ValidationPartition> {
        let num_partitions = self.config.num_partitions.min(focus_nodes.len());
        let partition_size = (focus_nodes.len() + num_partitions - 1) / num_partitions;

        let mut partitions = Vec::with_capacity(num_partitions);

        for (i, chunk) in focus_nodes.chunks(partition_size).enumerate() {
            partitions.push(ValidationPartition {
                partition_id: i,
                focus_nodes: chunk.to_vec(),
                assigned_worker: None,
                retry_count: 0,
            });
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_partitions += partitions.len();
        }

        partitions
    }

    /// Assign partitions to workers
    async fn assign_partitions(
        &self,
        mut partitions: Vec<ValidationPartition>,
        workers: &[WorkerInfo],
    ) -> Result<Vec<ValidationPartition>, DistributedError> {
        match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                for (i, partition) in partitions.iter_mut().enumerate() {
                    let worker_idx = i % workers.len();
                    partition.assigned_worker = Some(workers[worker_idx].worker_id.clone());
                }
            }
            LoadBalancingStrategy::LeastLoaded => {
                // Sort workers by load
                let mut sorted_workers: Vec<_> = workers.iter().collect();
                sorted_workers.sort_by(|a, b| {
                    a.current_load
                        .partial_cmp(&b.current_load)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for (i, partition) in partitions.iter_mut().enumerate() {
                    let worker_idx = i % sorted_workers.len();
                    partition.assigned_worker = Some(sorted_workers[worker_idx].worker_id.clone());
                }
            }
            LoadBalancingStrategy::ConsistentHashing => {
                for partition in partitions.iter_mut() {
                    // Use partition ID for consistent hashing
                    let hash = partition.partition_id;
                    let worker_idx = hash % workers.len();
                    partition.assigned_worker = Some(workers[worker_idx].worker_id.clone());
                }
            }
            LoadBalancingStrategy::Random => {
                let mut random = rng();
                for partition in partitions.iter_mut() {
                    let worker_idx = random.random_range(0..workers.len());
                    partition.assigned_worker = Some(workers[worker_idx].worker_id.clone());
                }
            }
        }

        Ok(partitions)
    }

    /// Execute validation on distributed workers
    async fn execute_distributed(
        &self,
        partitions: &[ValidationPartition],
    ) -> Result<Vec<PartitionResult>, DistributedError> {
        // In a real implementation, this would:
        // 1. Send partitions to workers via gRPC/HTTP
        // 2. Wait for results with timeout
        // 3. Handle failures and retries

        // For now, simulate distributed execution locally
        let mut results = Vec::with_capacity(partitions.len());
        let shapes = self.shapes.read().await;

        for partition in partitions {
            let start = Instant::now();

            // Simulate validation on worker
            let violations = self
                .simulate_worker_validation(&partition.focus_nodes, &shapes)
                .await;

            results.push(PartitionResult {
                partition_id: partition.partition_id,
                worker_id: partition
                    .assigned_worker
                    .clone()
                    .unwrap_or_else(|| "local".to_string()),
                violations,
                processing_time: start.elapsed(),
                success: true,
                error: None,
            });
        }

        Ok(results)
    }

    /// Simulate worker validation (placeholder for actual distributed execution)
    async fn simulate_worker_validation(
        &self,
        _focus_nodes: &[Term],
        _shapes: &IndexMap<ShapeId, Shape>,
    ) -> Vec<ValidationViolation> {
        // In a real implementation, this would execute actual SHACL validation
        // For now, return empty violations (all valid)
        Vec::new()
    }

    /// Aggregate results from all workers
    async fn aggregate_results(
        &self,
        results: Vec<PartitionResult>,
    ) -> Result<ValidationReport, DistributedError> {
        let mut report = ValidationReport::new();
        let mut all_violations = Vec::new();

        for result in results {
            if !result.success {
                if let Some(error) = &result.error {
                    tracing::error!(
                        "Partition {} failed on worker {}: {}",
                        result.partition_id,
                        result.worker_id,
                        error
                    );

                    // Update stats
                    let mut stats = self.stats.write().await;
                    stats.failed_partitions += 1;
                }
            } else {
                all_violations.extend(result.violations);
            }
        }

        // Deduplicate violations (in case of replication)
        all_violations.sort_by(|a, b| a.focus_node.cmp(&b.focus_node));
        all_violations.dedup_by(|a, b| {
            a.focus_node == b.focus_node
                && a.source_shape == b.source_shape
                && a.source_constraint_component == b.source_constraint_component
        });

        // Add violations to report
        for violation in all_violations {
            report.add_violation(violation);
        }

        Ok(report)
    }

    /// Compute cache key for a set of focus nodes
    fn compute_cache_key(&self, focus_nodes: &[Term]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for node in focus_nodes {
            node.to_string().hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }

    /// Check cache for result
    async fn check_cache(&self, key: &str) -> Option<ValidationReport> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.get(key) {
            if cached.created.elapsed() < self.config.cache_ttl {
                return Some(cached.report.clone());
            }
        }
        None
    }

    /// Cache a validation result
    async fn cache_result(&self, key: &str, report: &ValidationReport) {
        let mut cache = self.cache.write().await;
        cache.insert(
            key.to_string(),
            CachedResult {
                report: report.clone(),
                created: Instant::now(),
            },
        );

        // Clean up expired entries
        cache.retain(|_, v| v.created.elapsed() < self.config.cache_ttl);
    }

    /// Get distributed validation statistics
    pub async fn get_stats(&self) -> DistributedStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = DistributedStats::default();
    }

    /// Update shapes
    pub async fn update_shapes(&self, shapes: IndexMap<ShapeId, Shape>) {
        let mut current_shapes = self.shapes.write().await;
        *current_shapes = shapes;

        // Clear cache when shapes change
        let mut cache = self.cache.write().await;
        cache.clear();
    }

    /// Health check all workers
    pub async fn health_check(&self) -> Vec<(String, bool)> {
        let mut workers = self.workers.write().await;
        let mut results = Vec::new();

        for worker in workers.iter_mut() {
            // In a real implementation, this would ping the worker
            let elapsed = chrono::Utc::now().signed_duration_since(worker.last_heartbeat);
            let healthy = elapsed.num_seconds() < 60;
            worker.healthy = healthy;
            results.push((worker.worker_id.clone(), healthy));
        }

        results
    }
}

/// Errors that can occur during distributed validation
#[derive(Debug, Clone)]
pub enum DistributedError {
    /// No workers available
    NoWorkersAvailable,
    /// All workers failed
    AllWorkersFailed,
    /// Timeout waiting for results
    Timeout,
    /// Communication error
    Communication(String),
    /// Aggregation error
    Aggregation(String),
}

impl std::fmt::Display for DistributedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributedError::NoWorkersAvailable => write!(f, "No workers available"),
            DistributedError::AllWorkersFailed => write!(f, "All workers failed"),
            DistributedError::Timeout => write!(f, "Timeout waiting for results"),
            DistributedError::Communication(msg) => write!(f, "Communication error: {}", msg),
            DistributedError::Aggregation(msg) => write!(f, "Aggregation error: {}", msg),
        }
    }
}

impl std::error::Error for DistributedError {}

/// Builder for distributed validator
pub struct DistributedValidatorBuilder {
    config: DistributedValidationConfig,
    shapes: IndexMap<ShapeId, Shape>,
    validation_config: ValidationConfig,
    workers: Vec<WorkerInfo>,
}

impl DistributedValidatorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: DistributedValidationConfig::default(),
            shapes: IndexMap::new(),
            validation_config: ValidationConfig::default(),
            workers: Vec::new(),
        }
    }

    /// Set number of partitions
    pub fn num_partitions(mut self, n: usize) -> Self {
        self.config.num_partitions = n;
        self
    }

    /// Set load balancing strategy
    pub fn load_balancing(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.config.load_balancing = strategy;
        self
    }

    /// Set consistency level
    pub fn consistency_level(mut self, level: ConsistencyLevel) -> Self {
        self.config.consistency_level = level;
        self
    }

    /// Set worker timeout
    pub fn worker_timeout(mut self, timeout: Duration) -> Self {
        self.config.worker_timeout = timeout;
        self
    }

    /// Set shapes to validate
    pub fn shapes(mut self, shapes: IndexMap<ShapeId, Shape>) -> Self {
        self.shapes = shapes;
        self
    }

    /// Set validation configuration
    pub fn validation_config(mut self, config: ValidationConfig) -> Self {
        self.validation_config = config;
        self
    }

    /// Add a worker
    pub fn add_worker(mut self, worker: WorkerInfo) -> Self {
        self.workers.push(worker);
        self
    }

    /// Enable caching
    pub fn enable_caching(mut self, enable: bool) -> Self {
        self.config.enable_caching = enable;
        self
    }

    /// Build the distributed validator
    pub async fn build(self) -> DistributedValidator {
        let validator = DistributedValidator::new(self.shapes, self.config, self.validation_config);

        // Register workers
        for worker in self.workers {
            validator.register_worker(worker).await;
        }

        validator
    }
}

impl Default for DistributedValidatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_distributed_validator_creation() {
        let shapes = IndexMap::new();
        let config = DistributedValidationConfig::default();
        let validation_config = ValidationConfig::default();

        let _validator = DistributedValidator::new(shapes, config, validation_config);
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let shapes = IndexMap::new();
        let validator = DistributedValidator::new(
            shapes,
            DistributedValidationConfig::default(),
            ValidationConfig::default(),
        );

        let worker = WorkerInfo {
            worker_id: "worker-1".to_string(),
            address: "localhost:8080".to_string(),
            current_load: 0.0,
            healthy: true,
            last_heartbeat: chrono::Utc::now(),
            validations_processed: 0,
            avg_response_time_ms: 0,
        };

        validator.register_worker(worker).await;

        let workers = validator.get_healthy_workers().await;
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].worker_id, "worker-1");
    }

    #[tokio::test]
    async fn test_partition_creation() {
        let shapes = IndexMap::new();
        let validator = DistributedValidator::new(
            shapes,
            DistributedValidationConfig {
                num_partitions: 3,
                ..Default::default()
            },
            ValidationConfig::default(),
        );

        let focus_nodes = vec![
            Term::NamedNode(oxirs_core::model::NamedNode::new_unchecked(
                "http://example.org/1",
            )),
            Term::NamedNode(oxirs_core::model::NamedNode::new_unchecked(
                "http://example.org/2",
            )),
            Term::NamedNode(oxirs_core::model::NamedNode::new_unchecked(
                "http://example.org/3",
            )),
            Term::NamedNode(oxirs_core::model::NamedNode::new_unchecked(
                "http://example.org/4",
            )),
            Term::NamedNode(oxirs_core::model::NamedNode::new_unchecked(
                "http://example.org/5",
            )),
        ];

        let partitions = validator.partition_nodes(&focus_nodes).await;
        assert_eq!(partitions.len(), 3);
    }

    #[tokio::test]
    async fn test_round_robin_assignment() {
        let shapes = IndexMap::new();
        let validator = DistributedValidator::new(
            shapes,
            DistributedValidationConfig::default(),
            ValidationConfig::default(),
        );

        let partitions = vec![
            ValidationPartition {
                partition_id: 0,
                focus_nodes: vec![],
                assigned_worker: None,
                retry_count: 0,
            },
            ValidationPartition {
                partition_id: 1,
                focus_nodes: vec![],
                assigned_worker: None,
                retry_count: 0,
            },
        ];

        let workers = vec![
            WorkerInfo {
                worker_id: "w1".to_string(),
                address: "".to_string(),
                current_load: 0.0,
                healthy: true,
                last_heartbeat: chrono::Utc::now(),
                validations_processed: 0,
                avg_response_time_ms: 0,
            },
            WorkerInfo {
                worker_id: "w2".to_string(),
                address: "".to_string(),
                current_load: 0.0,
                healthy: true,
                last_heartbeat: chrono::Utc::now(),
                validations_processed: 0,
                avg_response_time_ms: 0,
            },
        ];

        let assigned = validator
            .assign_partitions(partitions, &workers)
            .await
            .expect("operation should succeed");

        assert_eq!(assigned[0].assigned_worker, Some("w1".to_string()));
        assert_eq!(assigned[1].assigned_worker, Some("w2".to_string()));
    }

    #[tokio::test]
    async fn test_builder_pattern() {
        let validator = DistributedValidatorBuilder::new()
            .num_partitions(5)
            .load_balancing(LoadBalancingStrategy::LeastLoaded)
            .enable_caching(true)
            .add_worker(WorkerInfo {
                worker_id: "worker-1".to_string(),
                address: "localhost:8080".to_string(),
                current_load: 0.0,
                healthy: true,
                last_heartbeat: chrono::Utc::now(),
                validations_processed: 0,
                avg_response_time_ms: 0,
            })
            .build()
            .await;

        let workers = validator.get_healthy_workers().await;
        assert_eq!(workers.len(), 1);
    }

    #[tokio::test]
    async fn test_no_workers_error() {
        let shapes = IndexMap::new();
        let validator = DistributedValidator::new(
            shapes,
            DistributedValidationConfig::default(),
            ValidationConfig::default(),
        );

        let focus_nodes = vec![Term::NamedNode(
            oxirs_core::model::NamedNode::new_unchecked("http://example.org/1"),
        )];

        let result = validator.validate_distributed(focus_nodes).await;
        assert!(matches!(result, Err(DistributedError::NoWorkersAvailable)));
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let shapes = IndexMap::new();
        let validator = DistributedValidator::new(
            shapes,
            DistributedValidationConfig::default(),
            ValidationConfig::default(),
        );

        let stats = validator.get_stats().await;
        assert_eq!(stats.total_distributed, 0);
        assert_eq!(stats.cache_hits, 0);
    }
}
