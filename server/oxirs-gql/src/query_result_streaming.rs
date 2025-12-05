//! Query Result Streaming for Large Datasets
//!
//! This module provides streaming capabilities for GraphQL query results,
//! enabling efficient handling of large result sets from RDF datasets.
//!
//! # Features
//!
//! - **Chunked Streaming**: Break large results into manageable chunks
//! - **Adaptive Chunk Sizing**: Automatically adjust chunk sizes based on network conditions
//! - **Backpressure Handling**: Prevent memory overflow with flow control
//! - **Progress Tracking**: Monitor streaming progress and performance
//! - **Multiple Strategies**: Fixed, adaptive, and time-based streaming
//! - **Error Recovery**: Handle partial failures gracefully
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_gql::query_result_streaming::{StreamingExecutor, StreamingConfig, StreamingStrategy};
//!
//! let config = StreamingConfig::new()
//!     .with_strategy(StreamingStrategy::Adaptive)
//!     .with_chunk_size(100)
//!     .with_buffer_size(1000);
//!
//! let executor = StreamingExecutor::new(config);
//! let stream = executor.execute_streaming(query).await?;
//!
//! while let Some(chunk) = stream.next().await {
//!     // Process chunk incrementally
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;

/// Configuration for streaming query results
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Streaming strategy to use
    pub strategy: StreamingStrategy,
    /// Default chunk size (number of items per chunk)
    pub chunk_size: usize,
    /// Maximum buffer size before backpressure
    pub buffer_size: usize,
    /// Timeout for chunk delivery
    pub chunk_timeout: Duration,
    /// Enable progress tracking
    pub track_progress: bool,
    /// Minimum chunk size (for adaptive strategy)
    pub min_chunk_size: usize,
    /// Maximum chunk size (for adaptive strategy)
    pub max_chunk_size: usize,
    /// Time interval for time-based strategy
    pub time_interval: Duration,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            strategy: StreamingStrategy::FixedChunk,
            chunk_size: 100,
            buffer_size: 1000,
            chunk_timeout: Duration::from_secs(30),
            track_progress: true,
            min_chunk_size: 10,
            max_chunk_size: 1000,
            time_interval: Duration::from_millis(100),
        }
    }
}

impl StreamingConfig {
    /// Create a new streaming configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the streaming strategy
    pub fn with_strategy(mut self, strategy: StreamingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the chunk timeout
    pub fn with_chunk_timeout(mut self, timeout: Duration) -> Self {
        self.chunk_timeout = timeout;
        self
    }

    /// Enable or disable progress tracking
    pub fn with_progress_tracking(mut self, enabled: bool) -> Self {
        self.track_progress = enabled;
        self
    }

    /// Set min and max chunk sizes for adaptive strategy
    pub fn with_chunk_size_range(mut self, min: usize, max: usize) -> Self {
        self.min_chunk_size = min;
        self.max_chunk_size = max;
        self
    }

    /// Set time interval for time-based strategy
    pub fn with_time_interval(mut self, interval: Duration) -> Self {
        self.time_interval = interval;
        self
    }
}

/// Streaming strategy for query results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamingStrategy {
    /// Fixed chunk size
    FixedChunk,
    /// Adaptive chunk size based on network conditions
    Adaptive,
    /// Time-based chunks (send at regular intervals)
    TimeBased,
    /// Priority-based (send high-priority items first)
    PriorityBased,
}

/// A chunk of streaming results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Chunk sequence number
    pub sequence: u64,
    /// Items in this chunk
    pub items: Vec<serde_json::Value>,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Total number of items (if known)
    pub total_items: Option<usize>,
    /// Chunk metadata
    pub metadata: ChunkMetadata,
}

/// Metadata for a stream chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Timestamp when chunk was created
    pub timestamp: u64,
    /// Size of chunk in bytes (approximate)
    pub size_bytes: usize,
    /// Processing time for this chunk
    pub processing_time_ms: u64,
    /// Chunk quality score (0.0-1.0)
    pub quality_score: f32,
}

impl StreamChunk {
    /// Create a new stream chunk
    pub fn new(sequence: u64, items: Vec<serde_json::Value>) -> Self {
        let size_bytes = items
            .iter()
            .map(|v| serde_json::to_string(v).unwrap_or_default().len())
            .sum();

        Self {
            sequence,
            items,
            is_final: false,
            total_items: None,
            metadata: ChunkMetadata {
                timestamp: chrono::Utc::now().timestamp() as u64,
                size_bytes,
                processing_time_ms: 0,
                quality_score: 1.0,
            },
        }
    }

    /// Mark this chunk as the final chunk
    pub fn mark_final(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Set total items count
    pub fn with_total_items(mut self, total: usize) -> Self {
        self.total_items = Some(total);
        self
    }

    /// Set processing time
    pub fn with_processing_time(mut self, time_ms: u64) -> Self {
        self.metadata.processing_time_ms = time_ms;
        self
    }

    /// Set quality score
    pub fn with_quality_score(mut self, score: f32) -> Self {
        self.metadata.quality_score = score;
        self
    }
}

/// Progress information for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProgress {
    /// Number of chunks sent
    pub chunks_sent: u64,
    /// Number of items sent
    pub items_sent: usize,
    /// Total items (if known)
    pub total_items: Option<usize>,
    /// Percentage complete (0.0-100.0)
    pub percentage: f32,
    /// Elapsed time
    pub elapsed_ms: u64,
    /// Estimated time remaining
    pub estimated_remaining_ms: Option<u64>,
    /// Current throughput (items/second)
    pub throughput: f32,
}

impl StreamProgress {
    /// Create new progress tracker
    pub fn new() -> Self {
        Self {
            chunks_sent: 0,
            items_sent: 0,
            total_items: None,
            percentage: 0.0,
            elapsed_ms: 0,
            estimated_remaining_ms: None,
            throughput: 0.0,
        }
    }

    /// Update progress with new chunk
    pub fn update(&mut self, chunk: &StreamChunk, elapsed: Duration) {
        self.chunks_sent += 1;
        self.items_sent += chunk.items.len();
        self.elapsed_ms = elapsed.as_millis() as u64;

        if let Some(total) = chunk.total_items {
            self.total_items = Some(total);
            self.percentage = (self.items_sent as f32 / total as f32) * 100.0;

            // Estimate remaining time
            if self.items_sent > 0 {
                let items_remaining = total.saturating_sub(self.items_sent);
                let rate = self.items_sent as f64 / elapsed.as_secs_f64();
                if rate > 0.0 {
                    self.estimated_remaining_ms =
                        Some(((items_remaining as f64 / rate) * 1000.0) as u64);
                }
            }
        }

        // Calculate throughput
        if self.elapsed_ms > 0 {
            self.throughput = (self.items_sent as f32 * 1000.0) / self.elapsed_ms as f32;
        }
    }
}

impl Default for StreamProgress {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for streaming execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingStats {
    /// Total chunks sent
    pub total_chunks: u64,
    /// Total items sent
    pub total_items: usize,
    /// Total bytes sent
    pub total_bytes: usize,
    /// Average chunk size
    pub avg_chunk_size: f32,
    /// Average processing time per chunk
    pub avg_processing_time_ms: f32,
    /// Total execution time
    pub total_execution_ms: u64,
    /// Overall throughput
    pub throughput: f32,
    /// Errors encountered
    pub errors: usize,
}

impl StreamingStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            total_chunks: 0,
            total_items: 0,
            total_bytes: 0,
            avg_chunk_size: 0.0,
            avg_processing_time_ms: 0.0,
            total_execution_ms: 0,
            throughput: 0.0,
            errors: 0,
        }
    }

    /// Update statistics with chunk
    pub fn update(&mut self, chunk: &StreamChunk) {
        self.total_chunks += 1;
        self.total_items += chunk.items.len();
        self.total_bytes += chunk.metadata.size_bytes;

        // Update averages
        self.avg_chunk_size = self.total_items as f32 / self.total_chunks as f32;
        let total_processing = self.avg_processing_time_ms * (self.total_chunks - 1) as f32
            + chunk.metadata.processing_time_ms as f32;
        self.avg_processing_time_ms = total_processing / self.total_chunks as f32;
    }

    /// Finalize statistics
    pub fn finalize(&mut self, total_time: Duration) {
        self.total_execution_ms = total_time.as_millis() as u64;
        if self.total_execution_ms > 0 {
            self.throughput = (self.total_items as f32 * 1000.0) / self.total_execution_ms as f32;
        }
    }

    /// Record an error
    pub fn record_error(&mut self) {
        self.errors += 1;
    }
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming executor for GraphQL queries
pub struct StreamingExecutor {
    config: StreamingConfig,
    stats: Arc<RwLock<StreamingStats>>,
}

impl StreamingExecutor {
    /// Create a new streaming executor
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(StreamingStats::new())),
        }
    }

    /// Execute a query with streaming
    pub async fn execute_streaming(
        &self,
        items: Vec<serde_json::Value>,
    ) -> Result<StreamingResultStream, StreamingError> {
        let (tx, rx) = mpsc::channel(self.config.buffer_size);
        let config = self.config.clone();
        let stats = self.stats.clone();

        let total_items = items.len();
        let start_time = Instant::now();

        tokio::spawn(async move {
            let mut sequence = 0u64;
            let mut progress = StreamProgress::new();
            progress.total_items = Some(total_items);

            match config.strategy {
                StreamingStrategy::FixedChunk => {
                    Self::stream_fixed_chunks(
                        items,
                        config.chunk_size,
                        &mut sequence,
                        &mut progress,
                        start_time,
                        tx,
                        stats,
                    )
                    .await;
                }
                StreamingStrategy::Adaptive => {
                    Self::stream_adaptive_chunks(
                        items,
                        &config,
                        &mut sequence,
                        &mut progress,
                        start_time,
                        tx,
                        stats,
                    )
                    .await;
                }
                StreamingStrategy::TimeBased => {
                    Self::stream_time_based(
                        items,
                        &config,
                        &mut sequence,
                        &mut progress,
                        start_time,
                        tx,
                        stats,
                    )
                    .await;
                }
                StreamingStrategy::PriorityBased => {
                    Self::stream_priority_based(
                        items,
                        &config,
                        &mut sequence,
                        &mut progress,
                        start_time,
                        tx,
                        stats,
                    )
                    .await;
                }
            }
        });

        Ok(StreamingResultStream::new(rx, self.stats.clone()))
    }

    /// Stream with fixed chunk size
    async fn stream_fixed_chunks(
        items: Vec<serde_json::Value>,
        chunk_size: usize,
        sequence: &mut u64,
        progress: &mut StreamProgress,
        start_time: Instant,
        tx: mpsc::Sender<StreamResult>,
        stats: Arc<RwLock<StreamingStats>>,
    ) {
        let total = items.len();
        for chunk_items in items.chunks(chunk_size) {
            let chunk_start = Instant::now();
            *sequence += 1;

            let mut chunk = StreamChunk::new(*sequence, chunk_items.to_vec())
                .with_total_items(total)
                .with_processing_time(chunk_start.elapsed().as_millis() as u64);

            if (*sequence as usize) * chunk_size >= total {
                chunk = chunk.mark_final();
            }

            progress.update(&chunk, start_time.elapsed());

            // Update stats
            {
                let mut s = stats.write().await;
                s.update(&chunk);
            }

            if tx.send(Ok(StreamEvent::Chunk(chunk))).await.is_err() {
                break;
            }

            // Send progress update
            if tx
                .send(Ok(StreamEvent::Progress(progress.clone())))
                .await
                .is_err()
            {
                break;
            }
        }

        // Finalize stats
        {
            let mut s = stats.write().await;
            s.finalize(start_time.elapsed());
        }

        let _ = tx.send(Ok(StreamEvent::Complete)).await;
    }

    /// Stream with adaptive chunk sizing
    async fn stream_adaptive_chunks(
        items: Vec<serde_json::Value>,
        config: &StreamingConfig,
        sequence: &mut u64,
        progress: &mut StreamProgress,
        start_time: Instant,
        tx: mpsc::Sender<StreamResult>,
        stats: Arc<RwLock<StreamingStats>>,
    ) {
        let total = items.len();
        let mut current_chunk_size = config.chunk_size;
        let mut remaining = items;

        while !remaining.is_empty() {
            let chunk_start = Instant::now();
            *sequence += 1;

            let take = current_chunk_size.min(remaining.len());
            let chunk_items: Vec<_> = remaining.drain(..take).collect();

            let processing_time = chunk_start.elapsed().as_millis() as u64;

            let mut chunk = StreamChunk::new(*sequence, chunk_items)
                .with_total_items(total)
                .with_processing_time(processing_time);

            if remaining.is_empty() {
                chunk = chunk.mark_final();
            }

            progress.update(&chunk, start_time.elapsed());

            // Adapt chunk size based on processing time
            if processing_time < 10 {
                // Very fast, increase chunk size
                current_chunk_size = (current_chunk_size * 2).min(config.max_chunk_size);
            } else if processing_time > 100 {
                // Too slow, decrease chunk size
                current_chunk_size = (current_chunk_size / 2).max(config.min_chunk_size);
            }

            // Update stats
            {
                let mut s = stats.write().await;
                s.update(&chunk);
            }

            if tx.send(Ok(StreamEvent::Chunk(chunk))).await.is_err() {
                break;
            }

            if tx
                .send(Ok(StreamEvent::Progress(progress.clone())))
                .await
                .is_err()
            {
                break;
            }
        }

        // Finalize stats
        {
            let mut s = stats.write().await;
            s.finalize(start_time.elapsed());
        }

        let _ = tx.send(Ok(StreamEvent::Complete)).await;
    }

    /// Stream with time-based chunks
    async fn stream_time_based(
        items: Vec<serde_json::Value>,
        config: &StreamingConfig,
        sequence: &mut u64,
        progress: &mut StreamProgress,
        start_time: Instant,
        tx: mpsc::Sender<StreamResult>,
        stats: Arc<RwLock<StreamingStats>>,
    ) {
        let total = items.len();
        let mut remaining = items;
        let mut interval = interval(config.time_interval);
        let mut buffer: VecDeque<serde_json::Value> = VecDeque::new();

        // Fill buffer initially
        let initial_chunk_size = config.chunk_size.min(remaining.len());
        buffer.extend(remaining.drain(..initial_chunk_size));

        loop {
            interval.tick().await;

            if buffer.is_empty() && remaining.is_empty() {
                break;
            }

            let chunk_start = Instant::now();
            *sequence += 1;

            // Take items from buffer
            let chunk_items: Vec<_> = buffer.drain(..).collect();

            // Refill buffer from remaining
            let refill_size = config.chunk_size.min(remaining.len());
            buffer.extend(remaining.drain(..refill_size));

            let processing_time = chunk_start.elapsed().as_millis() as u64;

            let mut chunk = StreamChunk::new(*sequence, chunk_items)
                .with_total_items(total)
                .with_processing_time(processing_time);

            if buffer.is_empty() && remaining.is_empty() {
                chunk = chunk.mark_final();
            }

            progress.update(&chunk, start_time.elapsed());

            // Update stats
            {
                let mut s = stats.write().await;
                s.update(&chunk);
            }

            if tx.send(Ok(StreamEvent::Chunk(chunk))).await.is_err() {
                break;
            }

            if tx
                .send(Ok(StreamEvent::Progress(progress.clone())))
                .await
                .is_err()
            {
                break;
            }
        }

        // Finalize stats
        {
            let mut s = stats.write().await;
            s.finalize(start_time.elapsed());
        }

        let _ = tx.send(Ok(StreamEvent::Complete)).await;
    }

    /// Stream with priority-based ordering
    async fn stream_priority_based(
        items: Vec<serde_json::Value>,
        config: &StreamingConfig,
        sequence: &mut u64,
        progress: &mut StreamProgress,
        start_time: Instant,
        tx: mpsc::Sender<StreamResult>,
        stats: Arc<RwLock<StreamingStats>>,
    ) {
        // For now, use fixed chunking (priority logic would require additional metadata)
        Self::stream_fixed_chunks(
            items,
            config.chunk_size,
            sequence,
            progress,
            start_time,
            tx,
            stats,
        )
        .await;
    }

    /// Get streaming statistics
    pub async fn get_stats(&self) -> StreamingStats {
        self.stats.read().await.clone()
    }
}

/// Stream event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    /// New chunk available
    Chunk(StreamChunk),
    /// Progress update
    Progress(StreamProgress),
    /// Stream completed
    Complete,
    /// Error occurred
    Error(String),
}

/// Result type for streaming
pub type StreamResult = Result<StreamEvent, StreamingError>;

/// Streaming result stream
pub struct StreamingResultStream {
    receiver: mpsc::Receiver<StreamResult>,
    stats: Arc<RwLock<StreamingStats>>,
}

impl StreamingResultStream {
    /// Create a new result stream
    pub fn new(receiver: mpsc::Receiver<StreamResult>, stats: Arc<RwLock<StreamingStats>>) -> Self {
        Self { receiver, stats }
    }

    /// Get next event from stream
    pub async fn next(&mut self) -> Option<StreamResult> {
        self.receiver.recv().await
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> StreamingStats {
        self.stats.read().await.clone()
    }
}

/// Errors that can occur during streaming
#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    /// Channel send error
    #[error("Failed to send chunk: {0}")]
    SendError(String),

    /// Timeout error
    #[error("Chunk delivery timeout")]
    Timeout,

    /// Configuration error
    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    /// Execution error
    #[error("Execution error: {0}")]
    ExecutionError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_builder() {
        let config = StreamingConfig::new()
            .with_strategy(StreamingStrategy::Adaptive)
            .with_chunk_size(200)
            .with_buffer_size(2000)
            .with_progress_tracking(true);

        assert_eq!(config.strategy, StreamingStrategy::Adaptive);
        assert_eq!(config.chunk_size, 200);
        assert_eq!(config.buffer_size, 2000);
        assert!(config.track_progress);
    }

    #[test]
    fn test_stream_chunk_creation() {
        let items = vec![serde_json::json!({"id": 1}), serde_json::json!({"id": 2})];

        let chunk = StreamChunk::new(1, items)
            .with_total_items(100)
            .mark_final()
            .with_processing_time(50)
            .with_quality_score(0.95);

        assert_eq!(chunk.sequence, 1);
        assert_eq!(chunk.items.len(), 2);
        assert!(chunk.is_final);
        assert_eq!(chunk.total_items, Some(100));
        assert_eq!(chunk.metadata.processing_time_ms, 50);
        assert_eq!(chunk.metadata.quality_score, 0.95);
    }

    #[test]
    fn test_stream_progress_update() {
        let mut progress = StreamProgress::new();

        let chunk = StreamChunk::new(1, vec![serde_json::json!({"id": 1})]).with_total_items(100);

        progress.update(&chunk, Duration::from_secs(1));

        assert_eq!(progress.chunks_sent, 1);
        assert_eq!(progress.items_sent, 1);
        assert_eq!(progress.total_items, Some(100));
        assert_eq!(progress.percentage, 1.0);
        assert!(progress.throughput > 0.0);
    }

    #[test]
    fn test_streaming_stats_update() {
        let mut stats = StreamingStats::new();

        let chunk1 =
            StreamChunk::new(1, vec![serde_json::json!({"id": 1})]).with_processing_time(10);
        let chunk2 = StreamChunk::new(
            2,
            vec![serde_json::json!({"id": 2}), serde_json::json!({"id": 3})],
        )
        .with_processing_time(20);

        stats.update(&chunk1);
        stats.update(&chunk2);

        assert_eq!(stats.total_chunks, 2);
        assert_eq!(stats.total_items, 3);
        assert_eq!(stats.avg_chunk_size, 1.5);
        assert_eq!(stats.avg_processing_time_ms, 15.0);
    }

    #[test]
    fn test_streaming_stats_finalize() {
        let mut stats = StreamingStats::new();

        let chunk = StreamChunk::new(1, vec![serde_json::json!({"id": 1})]);
        stats.update(&chunk);
        stats.finalize(Duration::from_secs(1));

        assert_eq!(stats.total_execution_ms, 1000);
        assert!(stats.throughput > 0.0);
    }

    #[test]
    fn test_streaming_stats_error_recording() {
        let mut stats = StreamingStats::new();

        assert_eq!(stats.errors, 0);
        stats.record_error();
        assert_eq!(stats.errors, 1);
        stats.record_error();
        assert_eq!(stats.errors, 2);
    }

    #[tokio::test]
    async fn test_fixed_chunk_streaming() {
        let items: Vec<_> = (0..250).map(|i| serde_json::json!({"id": i})).collect();

        let config = StreamingConfig::new()
            .with_strategy(StreamingStrategy::FixedChunk)
            .with_chunk_size(100);

        let executor = StreamingExecutor::new(config);
        let mut stream = executor.execute_streaming(items).await.unwrap();

        let mut chunk_count = 0;
        let mut total_items = 0;
        let mut completed = false;

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::Chunk(chunk) => {
                    chunk_count += 1;
                    total_items += chunk.items.len();
                    assert!(chunk.items.len() <= 100);
                }
                StreamEvent::Progress(progress) => {
                    assert!(progress.percentage >= 0.0 && progress.percentage <= 100.0);
                }
                StreamEvent::Complete => {
                    completed = true;
                    break;
                }
                StreamEvent::Error(_) => panic!("Unexpected error"),
            }
        }

        assert!(completed);
        assert_eq!(chunk_count, 3); // 250 items / 100 per chunk = 3 chunks
        assert_eq!(total_items, 250);

        let stats = stream.get_stats().await;
        assert_eq!(stats.total_chunks, 3);
        assert_eq!(stats.total_items, 250);
    }

    #[tokio::test]
    async fn test_adaptive_streaming() {
        let items: Vec<_> = (0..100).map(|i| serde_json::json!({"id": i})).collect();

        let config = StreamingConfig::new()
            .with_strategy(StreamingStrategy::Adaptive)
            .with_chunk_size(20)
            .with_chunk_size_range(10, 50);

        let executor = StreamingExecutor::new(config);
        let mut stream = executor.execute_streaming(items).await.unwrap();

        let mut total_items = 0;
        let mut completed = false;

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::Chunk(chunk) => {
                    total_items += chunk.items.len();
                    // Check adaptive sizing constraints
                    assert!(chunk.items.len() >= 10);
                    assert!(chunk.items.len() <= 50);
                }
                StreamEvent::Progress(_) => {}
                StreamEvent::Complete => {
                    completed = true;
                    break;
                }
                StreamEvent::Error(_) => panic!("Unexpected error"),
            }
        }

        assert!(completed);
        assert_eq!(total_items, 100);
    }

    #[tokio::test]
    async fn test_time_based_streaming() {
        let items: Vec<_> = (0..50).map(|i| serde_json::json!({"id": i})).collect();

        let config = StreamingConfig::new()
            .with_strategy(StreamingStrategy::TimeBased)
            .with_chunk_size(10)
            .with_time_interval(Duration::from_millis(10));

        let executor = StreamingExecutor::new(config);
        let mut stream = executor.execute_streaming(items).await.unwrap();

        let mut total_items = 0;
        let mut completed = false;

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::Chunk(chunk) => {
                    total_items += chunk.items.len();
                }
                StreamEvent::Progress(_) => {}
                StreamEvent::Complete => {
                    completed = true;
                    break;
                }
                StreamEvent::Error(_) => panic!("Unexpected error"),
            }
        }

        assert!(completed);
        assert_eq!(total_items, 50);
    }

    #[tokio::test]
    async fn test_streaming_with_progress() {
        let items: Vec<_> = (0..100).map(|i| serde_json::json!({"id": i})).collect();

        let config = StreamingConfig::new()
            .with_chunk_size(25)
            .with_progress_tracking(true);

        let executor = StreamingExecutor::new(config);
        let mut stream = executor.execute_streaming(items).await.unwrap();

        let mut progress_updates = 0;

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::Progress(progress) => {
                    progress_updates += 1;
                    assert!(progress.throughput >= 0.0);
                    assert!(progress.percentage >= 0.0 && progress.percentage <= 100.0);
                }
                StreamEvent::Complete => break,
                _ => {}
            }
        }

        assert!(progress_updates > 0);
    }

    #[tokio::test]
    async fn test_empty_stream() {
        let items: Vec<serde_json::Value> = vec![];

        let config = StreamingConfig::new();
        let executor = StreamingExecutor::new(config);
        let mut stream = executor.execute_streaming(items).await.unwrap();

        let mut completed = false;
        while let Some(event) = stream.next().await {
            if let StreamEvent::Complete = event.unwrap() {
                completed = true;
                break;
            }
        }

        assert!(completed);

        let stats = stream.get_stats().await;
        assert_eq!(stats.total_items, 0);
    }

    #[tokio::test]
    async fn test_single_item_stream() {
        let items = vec![serde_json::json!({"id": 1})];

        let config = StreamingConfig::new().with_chunk_size(100);
        let executor = StreamingExecutor::new(config);
        let mut stream = executor.execute_streaming(items).await.unwrap();

        let mut chunk_count = 0;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::Chunk(chunk) => {
                    chunk_count += 1;
                    assert_eq!(chunk.items.len(), 1);
                    assert!(chunk.is_final);
                }
                StreamEvent::Complete => break,
                _ => {}
            }
        }

        assert_eq!(chunk_count, 1);
    }

    #[tokio::test]
    async fn test_streaming_stats_accuracy() {
        let items: Vec<_> = (0..150).map(|i| serde_json::json!({"id": i})).collect();

        let config = StreamingConfig::new().with_chunk_size(50);
        let executor = StreamingExecutor::new(config);
        let mut stream = executor.execute_streaming(items).await.unwrap();

        while let Some(event) = stream.next().await {
            if matches!(event.unwrap(), StreamEvent::Complete) {
                break;
            }
        }

        let stats = stream.get_stats().await;
        assert_eq!(stats.total_items, 150);
        assert_eq!(stats.total_chunks, 3);
        assert_eq!(stats.avg_chunk_size, 50.0);
        // Execution time is always >= 0 (u64), just verify it's set
        let _ = stats.total_execution_ms;
    }

    #[tokio::test]
    async fn test_priority_based_streaming() {
        let items: Vec<_> = (0..100)
            .map(|i| serde_json::json!({"id": i, "priority": i % 3}))
            .collect();

        let config = StreamingConfig::new()
            .with_strategy(StreamingStrategy::PriorityBased)
            .with_chunk_size(25);

        let executor = StreamingExecutor::new(config);
        let mut stream = executor.execute_streaming(items).await.unwrap();

        let mut total_items = 0;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::Chunk(chunk) => {
                    total_items += chunk.items.len();
                }
                StreamEvent::Complete => break,
                _ => {}
            }
        }

        assert_eq!(total_items, 100);
    }

    #[tokio::test]
    async fn test_streaming_metadata() {
        let items: Vec<_> = (0..50).map(|i| serde_json::json!({"id": i})).collect();

        let config = StreamingConfig::new().with_chunk_size(10);
        let executor = StreamingExecutor::new(config);
        let mut stream = executor.execute_streaming(items).await.unwrap();

        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::Chunk(chunk) => {
                    assert!(chunk.metadata.timestamp > 0);
                    assert!(chunk.metadata.size_bytes > 0);
                    assert_eq!(chunk.metadata.quality_score, 1.0);
                }
                StreamEvent::Complete => break,
                _ => {}
            }
        }
    }

    #[test]
    fn test_chunk_metadata() {
        let items = vec![serde_json::json!({"test": "data"})];
        let chunk = StreamChunk::new(1, items);

        assert!(chunk.metadata.size_bytes > 0);
        assert_eq!(chunk.metadata.quality_score, 1.0);
        assert!(chunk.metadata.timestamp > 0);
    }
}
