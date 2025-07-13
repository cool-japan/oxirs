//! # Streaming Protocols Optimization
//!
//! This module implements advanced streaming protocols for federated query results,
//! including cursor-based pagination, adaptive page sizing, prefetching strategies,
//! and memory-bounded streaming for optimal performance.

use anyhow::{anyhow, Result};
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::service_registry::ServiceRegistry;

/// Streaming optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Default page size for pagination
    pub default_page_size: usize,
    /// Maximum page size allowed
    pub max_page_size: usize,
    /// Minimum page size for efficiency
    pub min_page_size: usize,
    /// Enable adaptive page sizing
    pub enable_adaptive_paging: bool,
    /// Memory limit for streaming buffers (bytes)
    pub memory_limit_bytes: usize,
    /// Enable prefetching
    pub enable_prefetching: bool,
    /// Prefetch buffer size
    pub prefetch_buffer_size: usize,
    /// Maximum concurrent streams per service
    pub max_concurrent_streams: usize,
    /// Stream timeout duration
    pub stream_timeout: Duration,
    /// Enable compression for large results
    pub enable_compression: bool,
    /// Compression threshold (bytes)
    pub compression_threshold: usize,
    /// Streaming protocol type
    pub protocol_type: StreamingProtocol,
    /// Back-pressure threshold
    pub backpressure_threshold: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            default_page_size: 100,
            max_page_size: 10000,
            min_page_size: 10,
            enable_adaptive_paging: true,
            memory_limit_bytes: 100 * 1024 * 1024, // 100MB
            enable_prefetching: true,
            prefetch_buffer_size: 3,
            max_concurrent_streams: 50,
            stream_timeout: Duration::from_secs(60),
            enable_compression: true,
            compression_threshold: 1024, // 1KB
            protocol_type: StreamingProtocol::Http2Stream,
            backpressure_threshold: 1000,
        }
    }
}

/// Supported streaming protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingProtocol {
    /// Standard HTTP/1.1 chunked transfer
    HttpChunked,
    /// HTTP/2 streaming
    Http2Stream,
    /// Server-Sent Events (SSE)
    ServerSentEvents,
    /// WebSocket streaming
    WebSocket,
    /// gRPC streaming
    GrpcStream,
    /// Custom binary protocol
    CustomBinary,
}

/// Cursor for pagination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationCursor {
    /// Unique cursor identifier
    pub cursor_id: String,
    /// Service-specific cursor data
    pub cursor_data: serde_json::Value,
    /// Current position in result set
    pub position: u64,
    /// Total estimated count (if known)
    pub total_count: Option<u64>,
    /// Expiration time for cursor
    pub expires_at: chrono::DateTime<chrono::Utc>,
    /// Sort order information
    pub sort_info: Option<SortInfo>,
}

/// Sort information for cursor-based pagination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortInfo {
    pub field: String,
    pub direction: SortDirection,
    pub data_type: SortDataType,
}

/// Sort direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Data types for sorting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortDataType {
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
}

/// Streaming result chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Chunk identifier
    pub chunk_id: String,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Chunk data
    pub data: serde_json::Value,
    /// Size in bytes
    pub size_bytes: usize,
    /// Indicates if this is the last chunk
    pub is_final: bool,
    /// Pagination cursor for next chunk
    pub next_cursor: Option<PaginationCursor>,
    /// Compression used (if any)
    pub compression: Option<String>,
    /// Metadata for the chunk
    pub metadata: HashMap<String, String>,
}

/// Adaptive paging strategy
#[derive(Debug, Clone)]
pub struct AdaptivePagingStrategy {
    /// Current page size
    pub current_page_size: usize,
    /// Performance history for page sizes
    pub performance_history: VecDeque<PagePerformance>,
    /// Last adjustment time
    pub last_adjustment: Instant,
    /// Adjustment cooldown period
    pub cooldown_period: Duration,
}

/// Performance metrics for a specific page size
#[derive(Debug, Clone)]
pub struct PagePerformance {
    pub page_size: usize,
    pub latency: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub memory_usage: usize,
    pub timestamp: Instant,
}

/// Prefetch strategy configuration
#[derive(Debug, Clone)]
pub struct PrefetchStrategy {
    /// Enable intelligent prefetching
    pub enabled: bool,
    /// Number of pages to prefetch
    pub pages_ahead: usize,
    /// Prefetch based on access patterns
    pub pattern_based: bool,
    /// Maximum memory for prefetch buffer
    pub max_memory_bytes: usize,
}

/// Stream statistics for monitoring
#[derive(Debug, Clone)]
pub struct StreamingStatistics {
    pub total_streams: u64,
    pub active_streams: u64,
    pub bytes_streamed: u64,
    pub chunks_sent: u64,
    pub average_chunk_size: f64,
    pub average_latency: Duration,
    pub throughput_mbps: f64,
    pub compression_ratio: f64,
    pub prefetch_hit_rate: f64,
    pub memory_usage_bytes: u64,
    pub error_count: u64,
}

impl Default for StreamingStatistics {
    fn default() -> Self {
        Self {
            total_streams: 0,
            active_streams: 0,
            bytes_streamed: 0,
            chunks_sent: 0,
            average_chunk_size: 0.0,
            average_latency: Duration::from_millis(0),
            throughput_mbps: 0.0,
            compression_ratio: 1.0,
            prefetch_hit_rate: 0.0,
            memory_usage_bytes: 0,
            error_count: 0,
        }
    }
}

/// Memory-bounded streaming controller
#[derive(Debug, Clone)]
pub struct MemoryBoundedController {
    /// Current memory usage
    memory_usage: Arc<RwLock<usize>>,
    /// Memory limit
    memory_limit: usize,
    /// Semaphore for memory allocation
    memory_semaphore: Arc<Semaphore>,
    /// Pending requests waiting for memory
    pending_allocations: Arc<RwLock<VecDeque<PendingAllocation>>>,
}

/// Pending memory allocation request
#[derive(Debug)]
struct PendingAllocation {
    request_id: String,
    size_bytes: usize,
    priority: AllocationPriority,
    timestamp: Instant,
    response_sender: tokio::sync::oneshot::Sender<Result<MemoryAllocation>>,
}

/// Memory allocation priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AllocationPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Memory allocation handle
#[derive(Debug)]
pub struct MemoryAllocation {
    allocation_id: String,
    size_bytes: usize,
    controller: Arc<MemoryBoundedController>,
}

impl Drop for MemoryAllocation {
    fn drop(&mut self) {
        let controller = Arc::clone(&self.controller);
        let size = self.size_bytes;

        tokio::spawn(async move {
            controller.deallocate(size).await;
        });
    }
}

/// Streaming optimizer for federated query results
#[derive(Clone)]
pub struct StreamingOptimizer {
    config: StreamingConfig,
    statistics: Arc<RwLock<StreamingStatistics>>,
    memory_controller: Arc<MemoryBoundedController>,
    adaptive_strategies: Arc<RwLock<HashMap<String, AdaptivePagingStrategy>>>,
    prefetch_cache: Arc<RwLock<HashMap<String, PrefetchedData>>>,
    active_streams: Arc<RwLock<HashMap<String, ActiveStream>>>,
    /// Error tracking for rate calculation
    error_counter: Arc<AtomicU64>,
    total_requests: Arc<AtomicU64>,
}

/// Prefetched data cache entry
#[derive(Debug, Clone)]
struct PrefetchedData {
    chunks: Vec<StreamChunk>,
    cursor: Option<PaginationCursor>,
    cached_at: Instant,
    access_count: u32,
    total_size_bytes: usize,
}

/// Active stream information
#[derive(Debug)]
struct ActiveStream {
    stream_id: String,
    service_id: String,
    created_at: Instant,
    last_activity: Instant,
    chunks_sent: u64,
    bytes_sent: u64,
    cursor: Option<PaginationCursor>,
    page_strategy: AdaptivePagingStrategy,
}

impl Default for StreamingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingOptimizer {
    /// Create a new streaming optimizer
    pub fn new() -> Self {
        Self::with_config(StreamingConfig::default())
    }

    /// Create a new streaming optimizer with configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        let memory_controller = Arc::new(MemoryBoundedController::new(config.memory_limit_bytes));

        Self {
            config,
            statistics: Arc::new(RwLock::new(StreamingStatistics::default())),
            memory_controller,
            adaptive_strategies: Arc::new(RwLock::new(HashMap::new())),
            prefetch_cache: Arc::new(RwLock::new(HashMap::new())),
            active_streams: Arc::new(RwLock::new(HashMap::new())),
            error_counter: Arc::new(AtomicU64::new(0)),
            total_requests: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a streaming cursor for paginated results
    pub async fn create_cursor(
        &self,
        service_id: &str,
        query: &str,
        sort_info: Option<SortInfo>,
    ) -> Result<PaginationCursor> {
        let cursor_id = Uuid::new_v4().to_string();
        let cursor_data = self.generate_cursor_data(service_id, query).await?;

        let cursor = PaginationCursor {
            cursor_id: cursor_id.clone(),
            cursor_data,
            position: 0,
            total_count: None,
            expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
            sort_info,
        };

        debug!(
            "Created pagination cursor {} for service {}",
            cursor_id, service_id
        );
        Ok(cursor)
    }

    /// Stream results with cursor-based pagination
    pub async fn stream_paginated_results(
        self: Arc<Self>,
        service_id: String,
        cursor: PaginationCursor,
        service_registry: Arc<ServiceRegistry>,
    ) -> Result<impl Stream<Item = Result<StreamChunk>>> {
        let stream_id = Uuid::new_v4().to_string();
        let (sender, receiver) = mpsc::channel(self.config.prefetch_buffer_size);

        // Initialize adaptive paging strategy
        let strategy = self.get_or_create_adaptive_strategy(&service_id).await;

        // Create active stream record
        let active_stream = ActiveStream {
            stream_id: stream_id.clone(),
            service_id: service_id.clone(),
            created_at: Instant::now(),
            last_activity: Instant::now(),
            chunks_sent: 0,
            bytes_sent: 0,
            cursor: Some(cursor.clone()),
            page_strategy: strategy,
        };

        self.active_streams
            .write()
            .await
            .insert(stream_id.clone(), active_stream);

        // Start streaming task
        let optimizer = Arc::clone(&self);
        let stream_id_clone = stream_id.clone();
        let service_id_clone = service_id.clone();
        let service_registry_clone = Arc::clone(&service_registry);
        let mut current_cursor = cursor;
        let sender_clone = sender;

        tokio::spawn(async move {
            let mut chunk_sequence = 0u64;

            loop {
                // Check if stream is still active
                if !optimizer.is_stream_active(&stream_id_clone).await {
                    break;
                }

                // Allocate memory for this chunk
                let estimated_chunk_size = optimizer.estimate_chunk_size(&current_cursor).await;
                let memory_allocation = match optimizer
                    .memory_controller
                    .allocate(estimated_chunk_size, AllocationPriority::Normal)
                    .await
                {
                    Ok(allocation) => allocation,
                    Err(e) => {
                        warn!("Failed to allocate memory for stream chunk: {}", e);
                        let _ = sender_clone.send(Err(e)).await;
                        break;
                    }
                };

                // Fetch next chunk with adaptive page sizing
                let chunk_result = optimizer
                    .fetch_next_chunk(
                        &service_id_clone,
                        &current_cursor,
                        chunk_sequence,
                        Arc::clone(&service_registry_clone),
                    )
                    .await;

                match chunk_result {
                    Ok(chunk) => {
                        let is_final = chunk.is_final;

                        // Update cursor for next iteration
                        if let Some(next_cursor) = &chunk.next_cursor {
                            current_cursor = next_cursor.clone();
                        }

                        // Update stream statistics
                        optimizer
                            .update_stream_statistics(&stream_id_clone, &chunk)
                            .await;

                        // Send chunk
                        if sender_clone.send(Ok(chunk)).await.is_err() {
                            debug!("Stream receiver dropped for stream {}", stream_id_clone);
                            break;
                        }

                        chunk_sequence += 1;

                        // Check if this was the final chunk
                        if is_final {
                            break;
                        }

                        // Apply backpressure if needed
                        if chunk_sequence % 10 == 0 {
                            optimizer.apply_backpressure(&stream_id_clone).await;
                        }
                    }
                    Err(e) => {
                        warn!("Error fetching stream chunk: {}", e);
                        let _ = sender_clone.send(Err(e)).await;
                        break;
                    }
                }

                // Drop memory allocation (will be freed automatically)
                drop(memory_allocation);
            }

            // Clean up stream
            optimizer.cleanup_stream(&stream_id_clone).await;
        });

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_streams += 1;
            stats.active_streams += 1;
        }

        Ok(ReceiverStream::new(receiver))
    }

    /// Stream results with adaptive page sizing
    pub async fn stream_adaptive_results(
        &self,
        service_id: &str,
        query: &str,
        initial_page_size: Option<usize>,
        service_registry: &ServiceRegistry,
    ) -> Result<impl Stream<Item = Result<StreamChunk>> + use<>> {
        // Create initial cursor
        let cursor = self.create_cursor(service_id, query, None).await?;

        // Override page size if specified
        if let Some(page_size) = initial_page_size {
            self.set_adaptive_page_size(service_id, page_size).await;
        }

        Arc::new((*self).clone())
            .stream_paginated_results(
                service_id.to_string(),
                cursor,
                Arc::new(service_registry.clone()),
            )
            .await
    }

    /// Prefetch next chunks based on access patterns
    pub async fn prefetch_ahead(
        &self,
        service_id: &str,
        cursor: &PaginationCursor,
        service_registry: Arc<ServiceRegistry>,
    ) -> Result<()> {
        if !self.config.enable_prefetching {
            return Ok(());
        }

        let prefetch_key = format!("{}:{}", service_id, cursor.cursor_id);

        // Check if already prefetching or cached
        {
            let cache = self.prefetch_cache.read().await;
            if cache.contains_key(&prefetch_key) {
                return Ok(());
            }
        }

        // Start prefetch task
        let optimizer = Arc::new(self.clone());
        let service_id_clone = service_id.to_string();
        let cursor_clone = cursor.clone();
        let service_registry_clone = Arc::clone(&service_registry);
        let prefetch_key_clone = prefetch_key.clone();

        tokio::spawn(async move {
            let mut current_cursor = cursor_clone;
            let mut prefetched_chunks = Vec::new();
            let mut total_size = 0usize;

            for i in 0..optimizer.config.prefetch_buffer_size {
                // Check memory limit
                if total_size > optimizer.config.memory_limit_bytes / 4 {
                    break;
                }

                match optimizer
                    .fetch_next_chunk(
                        &service_id_clone,
                        &current_cursor,
                        i as u64,
                        Arc::clone(&service_registry_clone),
                    )
                    .await
                {
                    Ok(chunk) => {
                        total_size += chunk.size_bytes;

                        if let Some(next_cursor) = &chunk.next_cursor {
                            current_cursor = next_cursor.clone();
                        }

                        let is_final = chunk.is_final;
                        prefetched_chunks.push(chunk);

                        if is_final {
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Prefetch failed for {}: {}", prefetch_key_clone, e);
                        break;
                    }
                }
            }

            if !prefetched_chunks.is_empty() {
                let chunk_count = prefetched_chunks.len();
                let prefetched_data = PrefetchedData {
                    chunks: prefetched_chunks,
                    cursor: Some(current_cursor),
                    cached_at: Instant::now(),
                    access_count: 0,
                    total_size_bytes: total_size,
                };

                optimizer
                    .prefetch_cache
                    .write()
                    .await
                    .insert(prefetch_key_clone, prefetched_data);

                debug!("Prefetched {} chunks for {}", chunk_count, service_id_clone);
            }
        });

        Ok(())
    }

    /// Get streaming statistics
    pub async fn get_statistics(&self) -> StreamingStatistics {
        self.statistics.read().await.clone()
    }

    /// Reset streaming statistics
    pub async fn reset_statistics(&self) {
        *self.statistics.write().await = StreamingStatistics::default();
    }

    // Private helper methods

    async fn generate_cursor_data(
        &self,
        service_id: &str,
        query: &str,
    ) -> Result<serde_json::Value> {
        // Generate service-specific cursor data
        Ok(serde_json::json!({
            "service_id": service_id,
            "query_hash": format!("{:x}", md5::compute(query)),
            "timestamp": chrono::Utc::now().timestamp(),
            "page_size": self.config.default_page_size
        }))
    }

    async fn get_or_create_adaptive_strategy(&self, service_id: &str) -> AdaptivePagingStrategy {
        let mut strategies = self.adaptive_strategies.write().await;

        strategies
            .entry(service_id.to_string())
            .or_insert_with(|| AdaptivePagingStrategy {
                current_page_size: self.config.default_page_size,
                performance_history: VecDeque::with_capacity(50),
                last_adjustment: Instant::now(),
                cooldown_period: Duration::from_secs(30),
            })
            .clone()
    }

    async fn set_adaptive_page_size(&self, service_id: &str, page_size: usize) {
        let mut strategies = self.adaptive_strategies.write().await;
        if let Some(strategy) = strategies.get_mut(service_id) {
            strategy.current_page_size =
                page_size.clamp(self.config.min_page_size, self.config.max_page_size);
        }
    }

    async fn fetch_next_chunk(
        &self,
        service_id: &str,
        cursor: &PaginationCursor,
        sequence: u64,
        service_registry: Arc<ServiceRegistry>,
    ) -> Result<StreamChunk> {
        let strategy = self.get_or_create_adaptive_strategy(service_id).await;

        // Check prefetch cache first
        let prefetch_key = format!("{}:{}", service_id, cursor.cursor_id);

        if let Some(prefetched) = self.get_from_prefetch_cache(&prefetch_key, sequence).await {
            return Ok(prefetched);
        }

        // Fetch from service
        let start_time = Instant::now();
        let chunk_data = self
            .fetch_chunk_from_service(
                service_id,
                cursor,
                strategy.current_page_size,
                &service_registry,
            )
            .await?;

        let fetch_time = start_time.elapsed();

        // Create chunk
        let chunk_id = Uuid::new_v4().to_string();
        let data_size = self.estimate_data_size(&chunk_data);

        let chunk = StreamChunk {
            chunk_id,
            sequence,
            data: chunk_data.clone(),
            size_bytes: data_size,
            is_final: false, // Will be determined by service response
            next_cursor: self.generate_next_cursor(cursor, sequence).await?,
            compression: self.compress_chunk_data(&chunk_data).await?,
            metadata: self.generate_chunk_metadata(service_id, cursor).await,
        };

        // Update adaptive strategy performance
        self.update_adaptive_strategy_performance(
            service_id,
            strategy.current_page_size,
            fetch_time,
            data_size,
        )
        .await;

        Ok(chunk)
    }

    async fn get_from_prefetch_cache(&self, cache_key: &str, sequence: u64) -> Option<StreamChunk> {
        let mut cache = self.prefetch_cache.write().await;

        if let Some(prefetched) = cache.get_mut(cache_key) {
            if let Some(chunk) = prefetched.chunks.get(sequence as usize) {
                prefetched.access_count += 1;

                // Update prefetch hit rate
                tokio::spawn({
                    let stats = Arc::clone(&self.statistics);
                    async move {
                        let mut stats = stats.write().await;
                        let total_requests = stats.chunks_sent + 1;
                        let hit_rate = (stats.prefetch_hit_rate * (total_requests - 1) as f64
                            + 1.0)
                            / total_requests as f64;
                        stats.prefetch_hit_rate = hit_rate;
                    }
                });

                return Some(chunk.clone());
            }
        }

        None
    }

    async fn fetch_chunk_from_service(
        &self,
        service_id: &str,
        cursor: &PaginationCursor,
        page_size: usize,
        _service_registry: &ServiceRegistry,
    ) -> Result<serde_json::Value> {
        // Mock implementation - would make actual service call
        let results: Vec<serde_json::Value> = (0..page_size)
            .map(|i| {
                serde_json::json!({
                    "id": cursor.position + i as u64,
                    "data": format!("Record {} from {}", cursor.position + i as u64, service_id),
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })
            })
            .collect();

        let mut map = serde_json::Map::new();
        map.insert("results".to_string(), serde_json::Value::Array(results));
        map.insert(
            "hasMore".to_string(),
            serde_json::Value::Bool((cursor.position + page_size as u64) < 1000),
        );
        map.insert(
            "totalCount".to_string(),
            serde_json::Value::Number(serde_json::Number::from(1000)),
        );
        let mock_data = serde_json::Value::Object(map);

        Ok(mock_data)
    }

    async fn generate_next_cursor(
        &self,
        current: &PaginationCursor,
        _sequence: u64,
    ) -> Result<Option<PaginationCursor>> {
        // Generate next cursor based on current position
        if current.position + self.config.default_page_size as u64 >= 1000 {
            // Mock end condition
            return Ok(None);
        }

        let next_cursor = PaginationCursor {
            cursor_id: current.cursor_id.clone(),
            cursor_data: current.cursor_data.clone(),
            position: current.position + self.config.default_page_size as u64,
            total_count: current.total_count,
            expires_at: current.expires_at,
            sort_info: current.sort_info.clone(),
        };

        Ok(Some(next_cursor))
    }

    async fn generate_chunk_metadata(
        &self,
        service_id: &str,
        cursor: &PaginationCursor,
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("service_id".to_string(), service_id.to_string());
        metadata.insert("cursor_id".to_string(), cursor.cursor_id.clone());
        metadata.insert("position".to_string(), cursor.position.to_string());
        metadata.insert(
            "protocol".to_string(),
            format!("{:?}", self.config.protocol_type),
        );
        metadata
    }

    fn estimate_data_size(&self, data: &serde_json::Value) -> usize {
        // Rough estimation of JSON data size
        serde_json::to_string(data).map(|s| s.len()).unwrap_or(0)
    }

    async fn estimate_chunk_size(&self, _cursor: &PaginationCursor) -> usize {
        // Estimate chunk size based on page size and historical data
        let strategy = self.get_or_create_adaptive_strategy("default").await;
        strategy.current_page_size * 200 // Rough estimate: 200 bytes per record
    }

    async fn update_adaptive_strategy_performance(
        &self,
        service_id: &str,
        page_size: usize,
        latency: Duration,
        data_size: usize,
    ) {
        let mut strategies = self.adaptive_strategies.write().await;

        if let Some(strategy) = strategies.get_mut(service_id) {
            let performance = PagePerformance {
                page_size,
                latency,
                throughput: data_size as f64 / latency.as_secs_f64(),
                error_rate: self.calculate_error_rate(),
                memory_usage: data_size,
                timestamp: Instant::now(),
            };

            strategy.performance_history.push_back(performance);
            if strategy.performance_history.len() > 50 {
                strategy.performance_history.pop_front();
            }

            // Adjust page size based on performance
            if self.config.enable_adaptive_paging {
                self.adjust_page_size(strategy).await;
            }
        }
    }

    async fn adjust_page_size(&self, strategy: &mut AdaptivePagingStrategy) {
        if strategy.last_adjustment.elapsed() < strategy.cooldown_period {
            return;
        }

        if strategy.performance_history.len() < 5 {
            return;
        }

        // Analyze recent performance
        let recent_performances: Vec<&PagePerformance> =
            strategy.performance_history.iter().rev().take(5).collect();

        let avg_latency: Duration = recent_performances
            .iter()
            .map(|p| p.latency)
            .sum::<Duration>()
            / recent_performances.len() as u32;

        let _avg_throughput: f64 = recent_performances
            .iter()
            .map(|p| p.throughput)
            .sum::<f64>()
            / recent_performances.len() as f64;

        // Adjust based on performance
        if avg_latency > Duration::from_millis(1000)
            && strategy.current_page_size > self.config.min_page_size
        {
            // High latency - reduce page size
            strategy.current_page_size =
                (strategy.current_page_size * 8 / 10).max(self.config.min_page_size);
            strategy.last_adjustment = Instant::now();
        } else if avg_latency < Duration::from_millis(100)
            && strategy.current_page_size < self.config.max_page_size
        {
            // Low latency - increase page size
            strategy.current_page_size =
                (strategy.current_page_size * 12 / 10).min(self.config.max_page_size);
            strategy.last_adjustment = Instant::now();
        }
    }

    async fn is_stream_active(&self, stream_id: &str) -> bool {
        self.active_streams.read().await.contains_key(stream_id)
    }

    async fn update_stream_statistics(&self, stream_id: &str, chunk: &StreamChunk) {
        // Update global statistics
        {
            let mut stats = self.statistics.write().await;
            stats.chunks_sent += 1;
            stats.bytes_streamed += chunk.size_bytes as u64;
            stats.average_chunk_size = stats.bytes_streamed as f64 / stats.chunks_sent as f64;
        }

        // Update stream-specific statistics
        if let Some(stream) = self.active_streams.write().await.get_mut(stream_id) {
            stream.chunks_sent += 1;
            stream.bytes_sent += chunk.size_bytes as u64;
            stream.last_activity = Instant::now();
        }
    }

    async fn apply_backpressure(&self, _stream_id: &str) {
        // Check if backpressure is needed
        let stats = self.statistics.read().await;
        if stats.active_streams > self.config.backpressure_threshold as u64 {
            // Apply backpressure by sleeping
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn cleanup_stream(&self, stream_id: &str) {
        self.active_streams.write().await.remove(stream_id);

        let mut stats = self.statistics.write().await;
        if stats.active_streams > 0 {
            stats.active_streams -= 1;
        }
    }

    /// Compress chunk data if compression is enabled
    async fn compress_chunk_data(&self, data: &serde_json::Value) -> Result<Option<String>> {
        if !self.config.enable_compression {
            return Ok(None);
        }

        // Simple compression check based on data size
        match serde_json::to_string(data) {
            Ok(json_str) => {
                if json_str.len() > self.config.compression_threshold {
                    // In a real implementation, you would actually compress the data here
                    // For now, just return the compression type
                    Ok(Some("gzip".to_string()))
                } else {
                    Ok(None) // Don't compress small data
                }
            }
            Err(e) => Err(anyhow!("Failed to serialize data for compression: {}", e)),
        }
    }

    /// Calculate current error rate
    fn calculate_error_rate(&self) -> f64 {
        let errors = self.error_counter.load(Ordering::Relaxed);
        let total = self.total_requests.load(Ordering::Relaxed);

        if total == 0 {
            0.0
        } else {
            (errors as f64) / (total as f64)
        }
    }

    /// Record a request result for error rate tracking
    pub fn record_request_result(&self, is_error: bool) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        if is_error {
            self.error_counter.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl MemoryBoundedController {
    fn new(memory_limit: usize) -> Self {
        Self {
            memory_usage: Arc::new(RwLock::new(0)),
            memory_limit,
            memory_semaphore: Arc::new(Semaphore::new(memory_limit)),
            pending_allocations: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    async fn allocate(
        &self,
        size_bytes: usize,
        _priority: AllocationPriority,
    ) -> Result<MemoryAllocation> {
        // Check if we can allocate immediately
        let current_usage = *self.memory_usage.read().await;
        if current_usage + size_bytes <= self.memory_limit {
            if let Ok(_permit) = self
                .memory_semaphore
                .clone()
                .try_acquire_many_owned(size_bytes as u32)
            {
                *self.memory_usage.write().await += size_bytes;

                return Ok(MemoryAllocation {
                    allocation_id: Uuid::new_v4().to_string(),
                    size_bytes,
                    controller: Arc::new(self.clone()),
                });
            }
        }

        // If immediate allocation failed, return error for now
        // In a full implementation, we could queue the request
        Err(anyhow!(
            "Memory allocation failed: {} bytes requested, {} available",
            size_bytes,
            self.memory_limit - current_usage
        ))
    }

    async fn deallocate(&self, size_bytes: usize) {
        let mut usage = self.memory_usage.write().await;
        if *usage >= size_bytes {
            *usage -= size_bytes;
        } else {
            *usage = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_streaming_optimizer_creation() {
        let optimizer = StreamingOptimizer::new();
        assert_eq!(optimizer.config.default_page_size, 100);
    }

    #[tokio::test]
    async fn test_cursor_creation() {
        let optimizer = StreamingOptimizer::new();
        let cursor = optimizer
            .create_cursor("test-service", "SELECT * FROM test", None)
            .await
            .unwrap();

        assert!(!cursor.cursor_id.is_empty());
        assert_eq!(cursor.position, 0);
        assert!(cursor.expires_at > chrono::Utc::now());
    }

    #[tokio::test]
    async fn test_adaptive_paging_strategy() {
        let optimizer = StreamingOptimizer::new();
        let strategy = optimizer
            .get_or_create_adaptive_strategy("test-service")
            .await;

        assert_eq!(strategy.current_page_size, 100);
        assert!(strategy.performance_history.is_empty());
    }

    #[tokio::test]
    async fn test_memory_bounded_controller() {
        let controller = MemoryBoundedController::new(1000);

        // Should succeed for small allocation
        let allocation = controller.allocate(500, AllocationPriority::Normal).await;
        assert!(allocation.is_ok());

        let _alloc = allocation.unwrap();

        // Should fail for allocation that exceeds limit
        let allocation2 = controller.allocate(600, AllocationPriority::Normal).await;
        assert!(allocation2.is_err());
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let optimizer = StreamingOptimizer::new();
        let stats = optimizer.get_statistics().await;

        assert_eq!(stats.total_streams, 0);
        assert_eq!(stats.active_streams, 0);
        assert_eq!(stats.bytes_streamed, 0);
    }
}
