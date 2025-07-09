//! # Advanced Performance Utilities for OxiRS Stream
//!
//! High-performance utilities, optimizations, and patterns for achieving
//! maximum throughput and minimum latency in streaming operations.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::yield_now;
use tracing::{debug, info, warn};

use crate::StreamEvent;

/// Configuration for performance optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceUtilsConfig {
    /// Enable zero-copy optimizations
    pub enable_zero_copy: bool,
    /// Enable SIMD optimizations where available
    pub enable_simd: bool,
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Enable batch processing optimizations
    pub enable_batch_optimization: bool,
    /// Maximum batch size for optimal performance
    pub optimal_batch_size: usize,
    /// Enable adaptive rate limiting
    pub enable_adaptive_rate_limiting: bool,
    /// Enable intelligent prefetching
    pub enable_prefetching: bool,
    /// CPU core count for parallel processing
    pub cpu_cores: usize,
}

impl Default for PerformanceUtilsConfig {
    fn default() -> Self {
        Self {
            enable_zero_copy: true,
            enable_simd: true,
            enable_memory_pooling: true,
            enable_batch_optimization: true,
            optimal_batch_size: 1000,
            enable_adaptive_rate_limiting: true,
            enable_prefetching: true,
            cpu_cores: num_cpus::get(),
        }
    }
}

/// High-performance adaptive batcher for streaming events
pub struct AdaptiveBatcher {
    config: PerformanceUtilsConfig,
    batch_buffer: Arc<RwLock<Vec<StreamEvent>>>,
    batch_stats: Arc<RwLock<BatchingStats>>,
    last_flush: Instant,
    target_latency: Duration,
    optimal_batch_size: Arc<RwLock<usize>>,
}

/// Statistics for adaptive batching performance
#[derive(Debug, Clone, Default)]
pub struct BatchingStats {
    pub total_batches: u64,
    pub total_events: u64,
    pub avg_batch_size: f64,
    pub avg_latency_ms: f64,
    pub throughput_events_per_sec: f64,
    pub efficiency_score: f64,
}

impl AdaptiveBatcher {
    /// Create a new adaptive batcher
    pub fn new(config: PerformanceUtilsConfig, target_latency: Duration) -> Self {
        Self {
            optimal_batch_size: Arc::new(RwLock::new(config.optimal_batch_size)),
            config,
            batch_buffer: Arc::new(RwLock::new(Vec::new())),
            batch_stats: Arc::new(RwLock::new(BatchingStats::default())),
            last_flush: Instant::now(),
            target_latency,
        }
    }

    /// Add an event to the batch buffer
    pub async fn add_event(&mut self, event: StreamEvent) -> Result<Option<Vec<StreamEvent>>> {
        let mut buffer = self.batch_buffer.write().await;
        buffer.push(event);

        let optimal_size = *self.optimal_batch_size.read().await;
        let time_since_last_flush = self.last_flush.elapsed();

        // Check if we should flush the batch
        if buffer.len() >= optimal_size || time_since_last_flush >= self.target_latency {
            let batch = std::mem::take(&mut *buffer);
            self.last_flush = Instant::now();

            // Update statistics and optimize batch size
            self.update_stats_and_optimize(batch.len(), time_since_last_flush)
                .await;

            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }

    /// Force flush the current batch
    pub async fn flush(&mut self) -> Result<Option<Vec<StreamEvent>>> {
        let mut buffer = self.batch_buffer.write().await;
        if buffer.is_empty() {
            return Ok(None);
        }

        let batch = std::mem::take(&mut *buffer);
        let time_since_last_flush = self.last_flush.elapsed();
        self.last_flush = Instant::now();

        self.update_stats_and_optimize(batch.len(), time_since_last_flush)
            .await;

        Ok(Some(batch))
    }

    /// Update statistics and optimize batch size based on performance
    async fn update_stats_and_optimize(&self, batch_size: usize, latency: Duration) {
        let mut stats = self.batch_stats.write().await;
        stats.total_batches += 1;
        stats.total_events += batch_size as u64;

        // Calculate moving averages
        let new_avg_batch_size = (stats.avg_batch_size + batch_size as f64) / 2.0;
        let new_avg_latency = (stats.avg_latency_ms + latency.as_millis() as f64) / 2.0;

        stats.avg_batch_size = new_avg_batch_size;
        stats.avg_latency_ms = new_avg_latency;

        if latency.as_secs_f64() > 0.0 {
            stats.throughput_events_per_sec = batch_size as f64 / latency.as_secs_f64();
        }

        // Calculate efficiency score (higher is better)
        let latency_efficiency = if new_avg_latency > 0.0 {
            1.0 / new_avg_latency
        } else {
            1.0
        };
        stats.efficiency_score = new_avg_batch_size * latency_efficiency;

        // Adaptive optimization: adjust batch size based on performance
        if self.config.enable_batch_optimization {
            let mut optimal_size = self.optimal_batch_size.write().await;

            if new_avg_latency > self.target_latency.as_millis() as f64 && *optimal_size > 100 {
                // Latency too high, reduce batch size
                *optimal_size = (*optimal_size * 9) / 10;
                debug!(
                    "Reduced optimal batch size to {} due to high latency",
                    *optimal_size
                );
            } else if new_avg_latency < self.target_latency.as_millis() as f64 / 2.0
                && *optimal_size < 10000
            {
                // Latency is good, try increasing batch size
                *optimal_size = (*optimal_size * 11) / 10;
                debug!(
                    "Increased optimal batch size to {} due to good latency",
                    *optimal_size
                );
            }
        }
    }

    /// Get current batching statistics
    pub async fn get_stats(&self) -> BatchingStats {
        self.batch_stats.read().await.clone()
    }

    /// Get current optimal batch size
    pub async fn get_optimal_batch_size(&self) -> usize {
        *self.optimal_batch_size.read().await
    }
}

/// Intelligent memory pool for reducing allocation overhead
pub struct IntelligentMemoryPool<T> {
    pools: Arc<RwLock<HashMap<String, VecDeque<T>>>>,
    max_pool_size: usize,
    allocation_stats: Arc<RwLock<PoolStats>>,
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_allocations: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub hit_rate: f64,
    pub memory_saved_bytes: u64,
}

impl<T> IntelligentMemoryPool<T> {
    /// Create a new memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: Arc::new(RwLock::new(HashMap::new())),
            max_pool_size,
            allocation_stats: Arc::new(RwLock::new(PoolStats::default())),
        }
    }

    /// Get an object from the pool or create a new one
    pub async fn get_or_create<F>(&self, pool_name: &str, factory: F) -> T
    where
        F: FnOnce() -> T,
    {
        let mut stats = self.allocation_stats.write().await;
        stats.total_allocations += 1;

        {
            let mut pools = self.pools.write().await;
            if let Some(pool) = pools.get_mut(pool_name) {
                if let Some(obj) = pool.pop_front() {
                    stats.pool_hits += 1;
                    stats.hit_rate = (stats.pool_hits as f64) / (stats.total_allocations as f64);
                    return obj;
                }
            }
        }

        // Pool miss - create new object
        stats.pool_misses += 1;
        stats.hit_rate = (stats.pool_hits as f64) / (stats.total_allocations as f64);

        factory()
    }

    /// Return an object to the pool
    pub async fn return_to_pool(&self, pool_name: &str, obj: T) {
        let mut pools = self.pools.write().await;
        let pool = pools
            .entry(pool_name.to_string())
            .or_insert_with(VecDeque::new);

        if pool.len() < self.max_pool_size {
            pool.push_back(obj);
        }
        // If pool is full, object is dropped (garbage collected)
    }

    /// Get pool statistics
    pub async fn get_stats(&self) -> PoolStats {
        self.allocation_stats.read().await.clone()
    }

    /// Clear all pools
    pub async fn clear_all_pools(&self) {
        let mut pools = self.pools.write().await;
        pools.clear();

        let mut stats = self.allocation_stats.write().await;
        *stats = PoolStats::default();
    }
}

/// Adaptive rate limiter with intelligent backpressure
pub struct AdaptiveRateLimiter {
    permits: Arc<Semaphore>,
    config: Arc<RwLock<RateLimitConfig>>,
    performance_history: Arc<RwLock<VecDeque<PerformanceSnapshot>>>,
    last_adjustment: Arc<RwLock<Instant>>,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub max_requests_per_second: usize,
    pub burst_capacity: usize,
    pub adjustment_interval: Duration,
    pub target_latency_ms: f64,
    pub max_adjustment_factor: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests_per_second: 10000,
            burst_capacity: 1000,
            adjustment_interval: Duration::from_secs(5),
            target_latency_ms: 10.0,
            max_adjustment_factor: 2.0,
        }
    }
}

/// Performance snapshot for rate limiting decisions
#[derive(Debug, Clone)]
struct PerformanceSnapshot {
    timestamp: Instant,
    latency_ms: f64,
    throughput_rps: f64,
    success_rate: f64,
}

impl AdaptiveRateLimiter {
    /// Create a new adaptive rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        let permits = Arc::new(Semaphore::new(config.burst_capacity));

        Self {
            permits,
            config: Arc::new(RwLock::new(config)),
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
            last_adjustment: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Acquire a permit to proceed with operation
    pub async fn acquire_permit(&self) -> Result<tokio::sync::SemaphorePermit<'_>> {
        match self.permits.try_acquire() {
            Ok(permit) => Ok(permit),
            Err(_) => {
                // Apply backpressure by waiting
                warn!("Rate limit reached, applying backpressure");
                self.permits
                    .acquire()
                    .await
                    .map_err(|e| anyhow!("Failed to acquire permit: {}", e))
            }
        }
    }

    /// Record performance metrics for adaptive adjustment
    pub async fn record_performance(&self, latency_ms: f64, success: bool) -> Result<()> {
        let snapshot = PerformanceSnapshot {
            timestamp: Instant::now(),
            latency_ms,
            throughput_rps: 0.0, // Will be calculated
            success_rate: if success { 1.0 } else { 0.0 },
        };

        {
            let mut history = self.performance_history.write().await;
            history.push_back(snapshot);

            // Keep only last 100 snapshots
            if history.len() > 100 {
                history.pop_front();
            }
        }

        // Check if it's time to adjust rate limits
        {
            let last_adjustment = self.last_adjustment.read().await;
            let config = self.config.read().await;

            if last_adjustment.elapsed() >= config.adjustment_interval {
                drop(last_adjustment);
                drop(config);
                self.adjust_rate_limits().await?;
            }
        }

        Ok(())
    }

    /// Intelligently adjust rate limits based on performance
    async fn adjust_rate_limits(&self) -> Result<()> {
        let mut last_adjustment = self.last_adjustment.write().await;
        *last_adjustment = Instant::now();
        drop(last_adjustment);

        let history = self.performance_history.read().await;
        if history.len() < 10 {
            return Ok(()); // Not enough data
        }

        // Calculate average performance metrics
        let avg_latency: f64 =
            history.iter().map(|s| s.latency_ms).sum::<f64>() / history.len() as f64;
        let avg_success_rate: f64 =
            history.iter().map(|s| s.success_rate).sum::<f64>() / history.len() as f64;

        let mut config = self.config.write().await;
        let current_rate = config.max_requests_per_second;

        // Adjust based on latency and success rate
        let adjustment_factor =
            if avg_latency > config.target_latency_ms * 1.5 || avg_success_rate < 0.95 {
                // Performance is poor, reduce rate
                0.9
            } else if avg_latency < config.target_latency_ms * 0.5 && avg_success_rate > 0.98 {
                // Performance is good, increase rate
                1.1
            } else {
                // Performance is acceptable, no change
                1.0
            };

        if adjustment_factor != 1.0 {
            let new_rate = ((current_rate as f64) * adjustment_factor) as usize;
            let max_rate = ((current_rate as f64) * config.max_adjustment_factor) as usize;
            let min_rate = ((current_rate as f64) / config.max_adjustment_factor) as usize;

            config.max_requests_per_second = new_rate.clamp(min_rate, max_rate);

            info!(
                "Adjusted rate limit from {} to {} req/s (factor: {:.2}, avg_latency: {:.2}ms, success_rate: {:.2}%)",
                current_rate, config.max_requests_per_second, adjustment_factor, avg_latency, avg_success_rate * 100.0
            );

            // Resize semaphore if needed
            // Note: In a real implementation, you'd need a more sophisticated approach
            // as Semaphore doesn't support dynamic resizing
        }

        Ok(())
    }

    /// Get current rate limit configuration
    pub async fn get_config(&self) -> RateLimitConfig {
        self.config.read().await.clone()
    }
}

/// High-performance parallel processor for streaming events
pub struct ParallelStreamProcessor {
    config: PerformanceUtilsConfig,
    worker_semaphore: Arc<Semaphore>,
    processing_stats: Arc<RwLock<ProcessingStats>>,
}

/// Processing statistics
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    pub events_processed: u64,
    pub avg_processing_time_ms: f64,
    pub peak_concurrency: usize,
    pub current_concurrency: usize,
    pub throughput_events_per_sec: f64,
    pub cpu_efficiency: f64,
}

impl ParallelStreamProcessor {
    /// Create a new parallel processor
    pub fn new(config: PerformanceUtilsConfig) -> Self {
        let worker_semaphore = Arc::new(Semaphore::new(config.cpu_cores * 2));

        Self {
            config,
            worker_semaphore,
            processing_stats: Arc::new(RwLock::new(ProcessingStats::default())),
        }
    }

    /// Process events in parallel with optimal load balancing
    pub async fn process_parallel<F, Fut>(
        &self,
        events: Vec<StreamEvent>,
        processor: F,
    ) -> Result<Vec<Result<()>>>
    where
        F: Fn(StreamEvent) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        let start_time = Instant::now();
        let event_count = events.len();

        // Update concurrency tracking
        {
            let mut stats = self.processing_stats.write().await;
            stats.current_concurrency = event_count.min(self.config.cpu_cores * 2);
            stats.peak_concurrency = stats.peak_concurrency.max(stats.current_concurrency);
        }

        // Process events in parallel with controlled concurrency
        let mut handles = Vec::new();

        for event in events {
            let permit = self.worker_semaphore.clone().acquire_owned().await?;
            let processor_clone = processor.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit; // Keep permit alive
                let result = processor_clone(event).await;
                yield_now().await; // Yield to prevent blocking
                result
            });

            handles.push(handle);
        }

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(e) => results.push(Err(anyhow!("Task join error: {}", e))),
            }
        }

        // Update statistics
        let processing_time = start_time.elapsed();
        {
            let mut stats = self.processing_stats.write().await;
            stats.events_processed += event_count as u64;
            stats.avg_processing_time_ms =
                (stats.avg_processing_time_ms + processing_time.as_millis() as f64) / 2.0;
            stats.current_concurrency = 0;

            if processing_time.as_secs_f64() > 0.0 {
                stats.throughput_events_per_sec =
                    event_count as f64 / processing_time.as_secs_f64();
            }

            // Calculate CPU efficiency (higher is better)
            let ideal_time = (event_count as f64) / (self.config.cpu_cores as f64);
            let actual_time = processing_time.as_secs_f64();
            stats.cpu_efficiency = if actual_time > 0.0 {
                (ideal_time / actual_time).min(1.0)
            } else {
                1.0
            };
        }

        debug!(
            "Processed {} events in {:?} ({:.2} events/sec)",
            event_count,
            processing_time,
            event_count as f64 / processing_time.as_secs_f64()
        );

        Ok(results)
    }

    /// Get current processing statistics
    pub async fn get_stats(&self) -> ProcessingStats {
        self.processing_stats.read().await.clone()
    }
}

/// Intelligent prefetcher for streaming data
pub struct IntelligentPrefetcher<T> {
    cache: Arc<RwLock<HashMap<String, T>>>,
    access_patterns: Arc<RwLock<HashMap<String, AccessPattern>>>,
    max_cache_size: usize,
}

/// Access pattern tracking for intelligent prefetching
#[derive(Debug, Clone)]
struct AccessPattern {
    access_count: u64,
    last_access: Instant,
    prediction_score: f64,
    related_keys: Vec<String>,
}

impl<T: Clone> IntelligentPrefetcher<T> {
    /// Create a new intelligent prefetcher
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            access_patterns: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size,
        }
    }

    /// Get data with intelligent prefetching
    pub async fn get_with_prefetch<F, Fut>(&self, key: &str, loader: F) -> Result<T>
    where
        F: FnOnce(String) -> Fut + Send,
        Fut: std::future::Future<Output = Result<T>> + Send,
        T: Send + Sync,
    {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(data) = cache.get(key) {
                self.update_access_pattern(key).await;
                return Ok(data.clone());
            }
        }

        // Load data
        let data = loader(key.to_string()).await?;

        // Store in cache
        {
            let mut cache = self.cache.write().await;
            if cache.len() >= self.max_cache_size {
                // Remove least recently used item
                self.evict_lru().await;
            }
            cache.insert(key.to_string(), data.clone());
        }

        // Update access patterns and trigger prefetching
        self.update_access_pattern(key).await;
        self.trigger_intelligent_prefetch(key).await;

        Ok(data)
    }

    /// Update access pattern for a key
    async fn update_access_pattern(&self, key: &str) {
        let mut patterns = self.access_patterns.write().await;
        let pattern = patterns
            .entry(key.to_string())
            .or_insert_with(|| AccessPattern {
                access_count: 0,
                last_access: Instant::now(),
                prediction_score: 0.0,
                related_keys: Vec::new(),
            });

        pattern.access_count += 1;
        pattern.last_access = Instant::now();

        // Update prediction score based on access frequency and recency
        let recency_factor = 1.0; // More recent = higher score
        let frequency_factor = (pattern.access_count as f64).ln();
        pattern.prediction_score = recency_factor * frequency_factor;
    }

    /// Trigger intelligent prefetching based on access patterns
    async fn trigger_intelligent_prefetch(&self, accessed_key: &str) {
        let patterns = self.access_patterns.read().await;

        if let Some(pattern) = patterns.get(accessed_key) {
            // Prefetch related keys with high prediction scores
            for related_key in &pattern.related_keys {
                if let Some(related_pattern) = patterns.get(related_key) {
                    if related_pattern.prediction_score > 0.5 {
                        // In a real implementation, you'd trigger async prefetching here
                        debug!("Would prefetch related key: {}", related_key);
                    }
                }
            }
        }
    }

    /// Evict least recently used item from cache
    async fn evict_lru(&self) {
        let patterns = self.access_patterns.read().await;

        if let Some((lru_key, _)) = patterns
            .iter()
            .min_by_key(|(_, pattern)| pattern.last_access)
        {
            let lru_key = lru_key.clone();
            drop(patterns);

            let mut cache = self.cache.write().await;
            cache.remove(&lru_key);

            let mut patterns = self.access_patterns.write().await;
            patterns.remove(&lru_key);
        }
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache_size = self.cache.read().await.len();
        let pattern_count = self.access_patterns.read().await.len();
        (cache_size, pattern_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_adaptive_batcher() {
        let config = PerformanceUtilsConfig::default();
        let mut batcher = AdaptiveBatcher::new(config, Duration::from_millis(100));

        let event = crate::event::StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "\"test_object\"".to_string(),
            graph: None,
            metadata: crate::event::EventMetadata::default(),
        };

        // Add events and check batching behavior
        for i in 0..10 {
            let result = batcher.add_event(event.clone()).await.unwrap();
            if i == 9 {
                // Should not batch yet (batch size is 1000 by default)
                assert!(result.is_none());
            }
        }

        // Force flush
        let batch = batcher.flush().await.unwrap();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 10);

        let stats = batcher.get_stats().await;
        assert_eq!(stats.total_batches, 1);
        assert_eq!(stats.total_events, 10);
    }

    #[tokio::test]
    async fn test_memory_pool() {
        let pool: IntelligentMemoryPool<String> = IntelligentMemoryPool::new(10);

        // Get object from pool (will create new)
        let obj1 = pool
            .get_or_create("test_pool", || "test_string".to_string())
            .await;
        assert_eq!(obj1, "test_string");

        // Return object to pool
        pool.return_to_pool("test_pool", obj1).await;

        // Get object again (should come from pool)
        let obj2 = pool
            .get_or_create("test_pool", || "new_string".to_string())
            .await;
        assert_eq!(obj2, "test_string"); // Should be the pooled object

        let stats = pool.get_stats().await;
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.total_allocations, 2);
    }

    #[tokio::test]
    async fn test_adaptive_rate_limiter() {
        let config = RateLimitConfig {
            max_requests_per_second: 10,
            burst_capacity: 5,
            adjustment_interval: Duration::from_millis(100),
            target_latency_ms: 10.0,
            max_adjustment_factor: 2.0,
        };

        let limiter = AdaptiveRateLimiter::new(config);

        // Acquire permits
        for _ in 0..5 {
            let _permit = limiter.acquire_permit().await.unwrap();
            limiter.record_performance(5.0, true).await.unwrap(); // Good performance
        }

        sleep(Duration::from_millis(150)).await; // Wait for adjustment

        let final_config = limiter.get_config().await;
        // Rate might be adjusted based on good performance
        assert!(final_config.max_requests_per_second >= 10);
    }

    #[tokio::test]
    async fn test_parallel_processor() {
        let config = PerformanceUtilsConfig::default();
        let processor = ParallelStreamProcessor::new(config);

        let events = vec![
            crate::event::StreamEvent::TripleAdded {
                subject: "http://example.org/subject1".to_string(),
                predicate: "http://example.org/predicate".to_string(),
                object: "\"test_object1\"".to_string(),
                graph: None,
                metadata: crate::event::EventMetadata::default(),
            },
            crate::event::StreamEvent::TripleAdded {
                subject: "http://example.org/subject2".to_string(),
                predicate: "http://example.org/predicate".to_string(),
                object: "\"test_object2\"".to_string(),
                graph: None,
                metadata: crate::event::EventMetadata::default(),
            },
        ];

        let results = processor
            .process_parallel(events, |_event| async {
                sleep(Duration::from_millis(10)).await;
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));

        let stats = processor.get_stats().await;
        assert_eq!(stats.events_processed, 2);
    }

    #[tokio::test]
    async fn test_intelligent_prefetcher() {
        let prefetcher: IntelligentPrefetcher<String> = IntelligentPrefetcher::new(10);

        // Load data with prefetcher
        let data1 = prefetcher
            .get_with_prefetch("key1", |key| async move { Ok(format!("data_for_{key}")) })
            .await
            .unwrap();

        assert_eq!(data1, "data_for_key1");

        // Access same key (should come from cache)
        let data2 = prefetcher
            .get_with_prefetch("key1", |_key| async move {
                Ok("should_not_be_called".to_string())
            })
            .await
            .unwrap();

        assert_eq!(data2, "data_for_key1");

        let (cache_size, pattern_count) = prefetcher.get_cache_stats().await;
        assert_eq!(cache_size, 1);
        assert_eq!(pattern_count, 1);
    }
}
