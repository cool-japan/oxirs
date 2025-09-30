//! # Request Batching Optimization
//!
//! This module implements sophisticated request batching for federated queries,
//! including adaptive batching strategies, request pipelining, and latency optimization.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{interval, sleep};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{executor::GraphQLResponse, service_registry::ServiceRegistry};

/// Request batching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    /// Maximum number of requests per batch
    pub max_batch_size: usize,
    /// Maximum time to wait before sending a batch
    pub max_batch_delay: Duration,
    /// Minimum batch size to optimize latency
    pub min_batch_size: usize,
    /// Enable adaptive batching based on load
    pub enable_adaptive_batching: bool,
    /// Target latency for adaptive batching
    pub target_latency_ms: u64,
    /// Maximum number of concurrent batches per service
    pub max_concurrent_batches: usize,
    /// Enable request pipelining
    pub enable_pipelining: bool,
    /// Pipeline depth for concurrent requests
    pub pipeline_depth: usize,
    /// Latency vs throughput optimization preference
    pub optimization_preference: OptimizationPreference,
    /// Enable smart request grouping
    pub enable_smart_grouping: bool,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 50,
            max_batch_delay: Duration::from_millis(10),
            min_batch_size: 1,
            enable_adaptive_batching: true,
            target_latency_ms: 50,
            max_concurrent_batches: 10,
            enable_pipelining: true,
            pipeline_depth: 5,
            optimization_preference: OptimizationPreference::Balanced,
            enable_smart_grouping: true,
        }
    }
}

/// Optimization preference for batching strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationPreference {
    /// Optimize for lowest latency
    Latency,
    /// Optimize for highest throughput
    Throughput,
    /// Balance between latency and throughput
    Balanced,
    /// Adaptive optimization based on load
    Adaptive,
}

/// Batching strategy based on current conditions
#[derive(Debug, Clone)]
pub enum BatchingStrategy {
    /// Immediate execution for latency-critical requests
    Immediate,
    /// Small batches for balanced performance
    SmallBatch { size: usize },
    /// Large batches for throughput optimization
    LargeBatch { size: usize },
    /// Time-based batching with fixed delay
    TimeBased { delay: Duration },
    /// Adaptive batching based on current load
    Adaptive { size: usize, delay: Duration },
}

/// Request for batching
#[derive(Debug)]
pub struct BatchableRequest {
    pub id: String,
    pub service_id: String,
    pub query: String,
    pub variables: Option<serde_json::Value>,
    pub priority: RequestPriority,
    pub timestamp: Instant,
    pub timeout: Option<Duration>,
    pub response_sender: Option<tokio::sync::oneshot::Sender<Result<GraphQLResponse>>>,
}

impl Clone for BatchableRequest {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            service_id: self.service_id.clone(),
            query: self.query.clone(),
            variables: self.variables.clone(),
            priority: self.priority.clone(),
            timestamp: self.timestamp,
            timeout: self.timeout,
            response_sender: None, // Cannot clone oneshot::Sender, so set to None
        }
    }
}

/// Request priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Batch of requests for a specific service
#[derive(Debug)]
pub struct RequestBatch {
    pub id: String,
    pub service_id: String,
    pub requests: Vec<BatchableRequest>,
    pub created_at: Instant,
    pub batch_strategy: BatchingStrategy,
    pub estimated_processing_time: Duration,
}

/// Batching statistics for monitoring
#[derive(Debug, Clone)]
pub struct BatchingStatistics {
    pub total_requests: u64,
    pub batched_requests: u64,
    pub total_batches: u64,
    pub average_batch_size: f64,
    pub average_batch_delay: Duration,
    pub average_processing_time: Duration,
    pub throughput_requests_per_second: f64,
    pub latency_p50: Duration,
    pub latency_p95: Duration,
    pub latency_p99: Duration,
    pub pipeline_utilization: f64,
    pub batch_efficiency: f64,
}

impl Default for BatchingStatistics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            batched_requests: 0,
            total_batches: 0,
            average_batch_size: 0.0,
            average_batch_delay: Duration::from_millis(0),
            average_processing_time: Duration::from_millis(0),
            throughput_requests_per_second: 0.0,
            latency_p50: Duration::from_millis(0),
            latency_p95: Duration::from_millis(0),
            latency_p99: Duration::from_millis(0),
            pipeline_utilization: 0.0,
            batch_efficiency: 0.0,
        }
    }
}

/// Performance metrics for adaptive batching
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub current_load: f64,
    pub average_latency: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub queue_depth: usize,
    pub pipeline_backlog: usize,
}

/// Request batcher for federated queries
pub struct RequestBatcher {
    config: BatchingConfig,
    pending_requests: Arc<RwLock<HashMap<String, VecDeque<BatchableRequest>>>>,
    active_batches: Arc<RwLock<HashMap<String, Vec<RequestBatch>>>>,
    statistics: Arc<RwLock<BatchingStatistics>>,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    service_semaphores: Arc<RwLock<HashMap<String, Arc<Semaphore>>>>,
    batch_scheduler_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Default for RequestBatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestBatcher {
    /// Create a new request batcher
    pub fn new() -> Self {
        Self::with_config(BatchingConfig::default())
    }

    /// Create a new request batcher with configuration
    pub fn with_config(config: BatchingConfig) -> Self {
        Self {
            config,
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            active_batches: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(BatchingStatistics::default())),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics {
                current_load: 0.0,
                average_latency: Duration::from_millis(0),
                throughput: 0.0,
                error_rate: 0.0,
                queue_depth: 0,
                pipeline_backlog: 0,
            })),
            service_semaphores: Arc::new(RwLock::new(HashMap::new())),
            batch_scheduler_handle: None,
        }
    }

    /// Start the batch scheduler
    pub async fn start(&mut self) -> Result<()> {
        if self.batch_scheduler_handle.is_some() {
            return Err(anyhow!("Batch scheduler already started"));
        }

        let config = self.config.clone();
        let pending_requests = Arc::clone(&self.pending_requests);
        let active_batches = Arc::clone(&self.active_batches);
        let statistics = Arc::clone(&self.statistics);
        let performance_metrics = Arc::clone(&self.performance_metrics);

        let handle = tokio::spawn(async move {
            Self::batch_scheduler_loop(
                config,
                pending_requests,
                active_batches,
                statistics,
                performance_metrics,
            )
            .await;
        });

        self.batch_scheduler_handle = Some(handle);
        info!("Request batcher started");
        Ok(())
    }

    /// Stop the batch scheduler
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(handle) = self.batch_scheduler_handle.take() {
            handle.abort();
            info!("Request batcher stopped");
        }
        Ok(())
    }

    /// Submit a request for batching
    pub async fn submit_request(&self, request: BatchableRequest) -> Result<()> {
        let service_id = request.service_id.clone();

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_requests += 1;
        }

        // Add to pending requests
        {
            let mut pending = self.pending_requests.write().await;
            pending
                .entry(service_id.clone())
                .or_insert_with(VecDeque::new)
                .push_back(request);
        }

        // Update queue depth metrics
        self.update_queue_metrics().await;

        debug!("Request submitted for batching to service: {}", service_id);
        Ok(())
    }

    /// Get current batching statistics
    pub async fn get_statistics(&self) -> BatchingStatistics {
        self.statistics.read().await.clone()
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }

    /// Execute a batch of requests
    pub async fn execute_batch(
        &self,
        batch: RequestBatch,
        service_registry: &ServiceRegistry,
    ) -> Result<Vec<GraphQLResponse>> {
        let start_time = Instant::now();

        debug!(
            "Executing batch {} with {} requests for service {}",
            batch.id,
            batch.requests.len(),
            batch.service_id
        );

        // Get service semaphore for concurrency control
        let semaphore = self.get_service_semaphore(&batch.service_id).await;
        let _permit = semaphore
            .acquire()
            .await
            .map_err(|e| anyhow!("Failed to acquire semaphore: {}", e))?;

        // Choose execution strategy based on batch characteristics
        let responses = match &batch.batch_strategy {
            BatchingStrategy::Immediate => {
                self.execute_immediate_batch(&batch, service_registry)
                    .await?
            }
            BatchingStrategy::SmallBatch { .. } | BatchingStrategy::LargeBatch { .. } => {
                self.execute_grouped_batch(&batch, service_registry).await?
            }
            BatchingStrategy::TimeBased { .. } => {
                self.execute_time_based_batch(&batch, service_registry)
                    .await?
            }
            BatchingStrategy::Adaptive { .. } => {
                self.execute_adaptive_batch(&batch, service_registry)
                    .await?
            }
        };

        let execution_time = start_time.elapsed();

        // Update statistics
        self.update_batch_statistics(&batch, execution_time).await;

        Ok(responses)
    }

    /// Choose optimal batching strategy based on current conditions
    pub async fn choose_batching_strategy(
        &self,
        _service_id: &str,
        queue_depth: usize,
    ) -> BatchingStrategy {
        let metrics = self.performance_metrics.read().await;

        match self.config.optimization_preference {
            OptimizationPreference::Latency => {
                if queue_depth == 1
                    || metrics.average_latency
                        > Duration::from_millis(self.config.target_latency_ms)
                {
                    BatchingStrategy::Immediate
                } else {
                    BatchingStrategy::SmallBatch {
                        size: self.config.min_batch_size.max(3),
                    }
                }
            }
            OptimizationPreference::Throughput => BatchingStrategy::LargeBatch {
                size: self.config.max_batch_size,
            },
            OptimizationPreference::Balanced => {
                let optimal_size = (queue_depth / 2)
                    .clamp(self.config.min_batch_size, self.config.max_batch_size / 2);
                BatchingStrategy::SmallBatch { size: optimal_size }
            }
            OptimizationPreference::Adaptive => {
                self.choose_adaptive_strategy(&metrics, queue_depth).await
            }
        }
    }

    // Private helper methods

    async fn batch_scheduler_loop(
        config: BatchingConfig,
        pending_requests: Arc<RwLock<HashMap<String, VecDeque<BatchableRequest>>>>,
        active_batches: Arc<RwLock<HashMap<String, Vec<RequestBatch>>>>,
        statistics: Arc<RwLock<BatchingStatistics>>,
        performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    ) {
        let mut interval = interval(Duration::from_millis(1)); // Check every 1ms for responsiveness

        loop {
            interval.tick().await;

            // Process each service's pending requests
            let service_ids: Vec<String> = {
                let pending = pending_requests.read().await;
                pending.keys().cloned().collect()
            };

            for service_id in service_ids {
                let should_create_batch = {
                    let pending = pending_requests.read().await;
                    if let Some(queue) = pending.get(&service_id) {
                        if queue.is_empty() {
                            false
                        } else {
                            // Check if we should create a batch based on size or time
                            let oldest_request_age = queue
                                .front()
                                .map(|r| r.timestamp.elapsed())
                                .unwrap_or(Duration::from_secs(0));

                            queue.len() >= config.max_batch_size
                                || oldest_request_age >= config.max_batch_delay
                                || (queue.len() >= config.min_batch_size
                                    && oldest_request_age >= config.max_batch_delay / 2)
                        }
                    } else {
                        false
                    }
                };

                if should_create_batch {
                    if let Err(e) = Self::create_and_schedule_batch(
                        &service_id,
                        &config,
                        &pending_requests,
                        &active_batches,
                        &statistics,
                    )
                    .await
                    {
                        warn!("Failed to create batch for service {}: {}", service_id, e);
                    }
                }
            }

            // Update performance metrics periodically
            if statistics.read().await.total_requests % 100 == 0 {
                Self::update_performance_metrics(
                    &performance_metrics,
                    &pending_requests,
                    &active_batches,
                )
                .await;
            }
        }
    }

    async fn create_and_schedule_batch(
        service_id: &str,
        config: &BatchingConfig,
        pending_requests: &Arc<RwLock<HashMap<String, VecDeque<BatchableRequest>>>>,
        active_batches: &Arc<RwLock<HashMap<String, Vec<RequestBatch>>>>,
        statistics: &Arc<RwLock<BatchingStatistics>>,
    ) -> Result<()> {
        let mut requests_to_batch = Vec::new();

        // Extract requests from pending queue
        {
            let mut pending = pending_requests.write().await;
            if let Some(queue) = pending.get_mut(service_id) {
                let batch_size = queue.len().min(config.max_batch_size);

                // Prioritize high-priority requests
                let mut temp_queue: Vec<_> = queue.drain(..).collect();
                temp_queue.sort_by_key(|r| std::cmp::Reverse(r.priority.clone()));

                requests_to_batch = temp_queue.into_iter().take(batch_size).collect();

                // Put remaining requests back
                for remaining in requests_to_batch.split_off(batch_size) {
                    queue.push_back(remaining);
                }
            }
        }

        if requests_to_batch.is_empty() {
            return Ok(());
        }

        // Create batch
        let batch_id = Uuid::new_v4().to_string();
        let queue_depth = requests_to_batch.len();

        // Determine batching strategy (simplified for now)
        let strategy = BatchingStrategy::SmallBatch { size: queue_depth };

        let batch = RequestBatch {
            id: batch_id.clone(),
            service_id: service_id.to_string(),
            requests: requests_to_batch,
            created_at: Instant::now(),
            batch_strategy: strategy,
            estimated_processing_time: Duration::from_millis(50), // Estimate based on batch size
        };

        // Add to active batches
        {
            let mut active = active_batches.write().await;
            active
                .entry(service_id.to_string())
                .or_insert_with(Vec::new)
                .push(batch);
        }

        // Update statistics
        {
            let mut stats = statistics.write().await;
            stats.total_batches += 1;
            stats.batched_requests += queue_depth as u64;
            stats.average_batch_size = stats.batched_requests as f64 / stats.total_batches as f64;
        }

        debug!(
            "Created batch {} with {} requests for service {}",
            batch_id, queue_depth, service_id
        );
        Ok(())
    }

    async fn update_performance_metrics(
        performance_metrics: &Arc<RwLock<PerformanceMetrics>>,
        pending_requests: &Arc<RwLock<HashMap<String, VecDeque<BatchableRequest>>>>,
        active_batches: &Arc<RwLock<HashMap<String, Vec<RequestBatch>>>>,
    ) {
        let queue_depth = {
            let pending = pending_requests.read().await;
            pending.values().map(|q| q.len()).sum()
        };

        let pipeline_backlog = {
            let active = active_batches.read().await;
            active.values().map(|batches| batches.len()).sum()
        };

        let mut metrics = performance_metrics.write().await;
        metrics.queue_depth = queue_depth;
        metrics.pipeline_backlog = pipeline_backlog;
        metrics.current_load = (queue_depth + pipeline_backlog) as f64 / 100.0; // Normalize to 0-1 scale
    }

    async fn update_queue_metrics(&self) {
        let queue_depth = {
            let pending = self.pending_requests.read().await;
            pending.values().map(|q| q.len()).sum()
        };

        let mut metrics = self.performance_metrics.write().await;
        metrics.queue_depth = queue_depth;
    }

    async fn get_service_semaphore(&self, service_id: &str) -> Arc<Semaphore> {
        let mut semaphores = self.service_semaphores.write().await;
        semaphores
            .entry(service_id.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(self.config.max_concurrent_batches)))
            .clone()
    }

    async fn execute_immediate_batch(
        &self,
        batch: &RequestBatch,
        service_registry: &ServiceRegistry,
    ) -> Result<Vec<GraphQLResponse>> {
        // Execute requests immediately without further batching
        let mut responses = Vec::new();

        for request in &batch.requests {
            let response = self
                .execute_single_request(request, service_registry)
                .await?;
            responses.push(response);
        }

        Ok(responses)
    }

    async fn execute_grouped_batch(
        &self,
        batch: &RequestBatch,
        service_registry: &ServiceRegistry,
    ) -> Result<Vec<GraphQLResponse>> {
        if self.config.enable_smart_grouping {
            self.execute_smart_grouped_batch(batch, service_registry)
                .await
        } else {
            self.execute_simple_grouped_batch(batch, service_registry)
                .await
        }
    }

    async fn execute_smart_grouped_batch(
        &self,
        batch: &RequestBatch,
        service_registry: &ServiceRegistry,
    ) -> Result<Vec<GraphQLResponse>> {
        // Group similar requests for more efficient batching
        let mut query_groups: HashMap<String, Vec<&BatchableRequest>> = HashMap::new();

        for request in &batch.requests {
            // Simple grouping by query structure (could be more sophisticated)
            let query_key = self.extract_query_signature(&request.query);
            query_groups.entry(query_key).or_default().push(request);
        }

        let mut all_responses = Vec::new();

        for (_, group) in query_groups {
            if group.len() == 1 {
                // Single request - execute directly
                let response = self
                    .execute_single_request(group[0], service_registry)
                    .await?;
                all_responses.push(response);
            } else {
                // Multiple similar requests - batch them
                let batched_response = self.execute_request_group(group, service_registry).await?;
                all_responses.extend(batched_response);
            }
        }

        Ok(all_responses)
    }

    async fn execute_simple_grouped_batch(
        &self,
        batch: &RequestBatch,
        _service_registry: &ServiceRegistry,
    ) -> Result<Vec<GraphQLResponse>> {
        // Simple parallel execution
        let mut tasks = Vec::new();

        let requests: Vec<_> = batch.requests.clone();
        for request in requests {
            let task: tokio::task::JoinHandle<Result<GraphQLResponse, anyhow::Error>> =
                tokio::spawn(async move {
                    // Simulate request execution
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Ok(GraphQLResponse {
                        data: serde_json::json!({
                            "batchId": request.id,
                            "serviceId": request.service_id,
                            "result": "batched execution"
                        }),
                        errors: Vec::new(),
                        extensions: None,
                    })
                });

            tasks.push(task);
        }

        let mut responses = Vec::new();
        for task in tasks {
            let response = task.await??;
            responses.push(response);
        }

        Ok(responses)
    }

    async fn execute_time_based_batch(
        &self,
        batch: &RequestBatch,
        service_registry: &ServiceRegistry,
    ) -> Result<Vec<GraphQLResponse>> {
        // Wait for the specified delay then execute
        if let BatchingStrategy::TimeBased { delay } = &batch.batch_strategy {
            sleep(*delay).await;
        }

        self.execute_simple_grouped_batch(batch, service_registry)
            .await
    }

    async fn execute_adaptive_batch(
        &self,
        batch: &RequestBatch,
        service_registry: &ServiceRegistry,
    ) -> Result<Vec<GraphQLResponse>> {
        // Use adaptive strategy based on current metrics
        let metrics = self.performance_metrics.read().await;

        if metrics.current_load > 0.8 {
            // High load - use aggressive batching
            self.execute_simple_grouped_batch(batch, service_registry)
                .await
        } else if metrics.average_latency > Duration::from_millis(100) {
            // High latency - prefer immediate execution
            self.execute_immediate_batch(batch, service_registry).await
        } else {
            // Normal conditions - use smart grouping
            self.execute_smart_grouped_batch(batch, service_registry)
                .await
        }
    }

    async fn execute_single_request(
        &self,
        request: &BatchableRequest,
        _service_registry: &ServiceRegistry,
    ) -> Result<GraphQLResponse> {
        // Mock implementation - would make actual service call
        Ok(GraphQLResponse {
            data: serde_json::json!({
                "requestId": request.id,
                "serviceId": request.service_id,
                "query": request.query,
                "result": "executed"
            }),
            errors: Vec::new(),
            extensions: None,
        })
    }

    async fn execute_request_group(
        &self,
        group: Vec<&BatchableRequest>,
        _service_registry: &ServiceRegistry,
    ) -> Result<Vec<GraphQLResponse>> {
        // Mock implementation for grouped execution
        let mut responses = Vec::new();

        for request in group {
            responses.push(GraphQLResponse {
                data: serde_json::json!({
                    "requestId": request.id,
                    "serviceId": request.service_id,
                    "result": "grouped execution"
                }),
                errors: Vec::new(),
                extensions: None,
            });
        }

        Ok(responses)
    }

    fn extract_query_signature(&self, query: &str) -> String {
        // Extract a signature from the query for grouping
        // This is a simplified implementation
        let words: Vec<&str> = query.split_whitespace().take(3).collect();
        words.join(" ")
    }

    async fn choose_adaptive_strategy(
        &self,
        metrics: &PerformanceMetrics,
        queue_depth: usize,
    ) -> BatchingStrategy {
        if metrics.current_load < 0.3 && metrics.average_latency < Duration::from_millis(20) {
            // Low load, low latency - prefer immediate execution
            BatchingStrategy::Immediate
        } else if metrics.current_load > 0.7 {
            // High load - use large batches for throughput
            BatchingStrategy::LargeBatch {
                size: self.config.max_batch_size,
            }
        } else {
            // Medium load - adaptive batching
            let optimal_size = (queue_depth as f64 * 0.7) as usize;
            let optimal_delay =
                Duration::from_millis((self.config.target_latency_ms as f64 * 0.3) as u64);

            BatchingStrategy::Adaptive {
                size: optimal_size.clamp(self.config.min_batch_size, self.config.max_batch_size),
                delay: optimal_delay,
            }
        }
    }

    async fn update_batch_statistics(&self, batch: &RequestBatch, execution_time: Duration) {
        let mut stats = self.statistics.write().await;

        // Update processing time
        let total_batches = stats.total_batches;
        if total_batches > 0 {
            let current_avg = stats.average_processing_time.as_millis() as f64;
            let new_avg = (current_avg * (total_batches - 1) as f64
                + execution_time.as_millis() as f64)
                / total_batches as f64;
            stats.average_processing_time = Duration::from_millis(new_avg as u64);
        } else {
            stats.average_processing_time = execution_time;
        }

        // Update batch efficiency
        let ideal_time = Duration::from_millis(10 * batch.requests.len() as u64); // Assume 10ms per request ideally
        stats.batch_efficiency =
            (ideal_time.as_millis() as f64 / execution_time.as_millis() as f64).min(1.0);

        // Update throughput
        if execution_time.as_secs_f64() > 0.0 {
            stats.throughput_requests_per_second =
                batch.requests.len() as f64 / execution_time.as_secs_f64();
        }
    }
}

impl Drop for RequestBatcher {
    fn drop(&mut self) {
        if let Some(handle) = self.batch_scheduler_handle.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_request_batcher_creation() {
        let batcher = RequestBatcher::new();
        assert_eq!(batcher.config.max_batch_size, 50);
    }

    #[tokio::test]
    async fn test_request_submission() {
        let batcher = RequestBatcher::new();

        let request = BatchableRequest {
            id: "test-1".to_string(),
            service_id: "test-service".to_string(),
            query: "{ test }".to_string(),
            variables: None,
            priority: RequestPriority::Normal,
            timestamp: Instant::now(),
            timeout: None,
            response_sender: None,
        };

        let result = batcher.submit_request(request).await;
        assert!(result.is_ok());

        let stats = batcher.get_statistics().await;
        assert_eq!(stats.total_requests, 1);
    }

    #[tokio::test]
    async fn test_batching_strategy_selection() {
        let batcher = RequestBatcher::with_config(BatchingConfig {
            optimization_preference: OptimizationPreference::Latency,
            ..Default::default()
        });

        let strategy = batcher.choose_batching_strategy("test-service", 1).await;
        assert!(matches!(strategy, BatchingStrategy::Immediate));

        let strategy = batcher.choose_batching_strategy("test-service", 10).await;
        assert!(matches!(strategy, BatchingStrategy::SmallBatch { .. }));
    }

    #[tokio::test]
    async fn test_performance_metrics() {
        let batcher = RequestBatcher::new();
        let metrics = batcher.get_performance_metrics().await;

        assert_eq!(metrics.queue_depth, 0);
        assert_eq!(metrics.pipeline_backlog, 0);
    }
}
