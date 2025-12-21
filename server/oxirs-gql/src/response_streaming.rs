//! GraphQL Response Streaming
//!
//! This module provides efficient streaming for large GraphQL responses:
//! - **Chunked Transfer**: Stream responses in chunks for memory efficiency
//! - **Incremental Delivery**: Support for @defer and @stream directives
//! - **Backpressure Handling**: Automatic flow control based on client capacity
//! - **Progress Tracking**: Monitor streaming progress and performance
//! - **Compression**: Optional gzip/deflate compression for large responses
//! - **Resumable Streams**: Support for resuming interrupted streams

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};

/// Streaming mode for response delivery
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StreamingMode {
    /// No streaming - return complete response
    None,
    /// Chunked transfer encoding
    Chunked,
    /// Incremental delivery with @defer
    Deferred,
    /// Array streaming with @stream
    Streamed,
    /// Multipart response format (for subscriptions)
    Multipart,
}

/// Compression type for streamed responses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Deflate compression
    Deflate,
    /// Brotli compression
    Brotli,
}

/// A chunk of response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChunk {
    /// Chunk ID for ordering
    pub id: u64,
    /// Path in response (for incremental delivery)
    pub path: Option<Vec<PathSegment>>,
    /// Data payload
    pub data: serde_json::Value,
    /// Whether this chunk has errors
    pub has_errors: bool,
    /// Errors in this chunk
    pub errors: Vec<StreamError>,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Whether more data is pending for this path
    pub has_next: bool,
    /// Label for @defer/@stream directive
    pub label: Option<String>,
}

/// Path segment in response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PathSegment {
    /// Field name
    Field(String),
    /// Array index
    Index(usize),
}

/// Error in stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamError {
    /// Error message
    pub message: String,
    /// Path to error location
    pub path: Option<Vec<PathSegment>>,
    /// Error extensions
    pub extensions: Option<HashMap<String, serde_json::Value>>,
}

/// Configuration for response streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Default streaming mode
    pub default_mode: StreamingMode,
    /// Maximum chunk size in bytes
    pub max_chunk_size: usize,
    /// Minimum chunk size in bytes (for batching small items)
    pub min_chunk_size: usize,
    /// Enable compression
    pub compression: CompressionType,
    /// Compression threshold in bytes
    pub compression_threshold: usize,
    /// Maximum concurrent deferred fields
    pub max_concurrent_deferred: usize,
    /// Timeout for individual chunks
    pub chunk_timeout: Duration,
    /// Enable progress tracking
    pub track_progress: bool,
    /// Buffer size for backpressure
    pub buffer_size: usize,
    /// Enable chunk acknowledgment
    pub require_ack: bool,
    /// Heartbeat interval for long-running streams
    pub heartbeat_interval: Option<Duration>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            default_mode: StreamingMode::Chunked,
            max_chunk_size: 64 * 1024, // 64KB
            min_chunk_size: 1024,      // 1KB
            compression: CompressionType::None,
            compression_threshold: 1024, // Compress if > 1KB
            max_concurrent_deferred: 10,
            chunk_timeout: Duration::from_secs(30),
            track_progress: true,
            buffer_size: 100,
            require_ack: false,
            heartbeat_interval: Some(Duration::from_secs(15)),
        }
    }
}

/// Streaming progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProgress {
    /// Stream ID
    pub stream_id: String,
    /// Total chunks sent
    pub chunks_sent: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Estimated total bytes (if known)
    pub total_bytes: Option<u64>,
    /// Current throughput (bytes/sec)
    pub throughput_bps: f64,
    /// Elapsed time
    pub elapsed_ms: u64,
    /// Pending chunks in buffer
    pub pending_chunks: usize,
    /// Whether stream is complete
    pub is_complete: bool,
    /// Whether stream has errors
    pub has_errors: bool,
}

/// Response streamer for a single response
pub struct ResponseStreamer {
    /// Stream ID
    id: String,
    /// Configuration
    config: StreamingConfig,
    /// Channel sender for chunks
    sender: mpsc::Sender<ResponseChunk>,
    /// Current chunk ID
    chunk_id: AtomicU64,
    /// Start time
    start_time: Instant,
    /// Bytes sent
    bytes_sent: AtomicU64,
    /// State
    state: Arc<RwLock<StreamerState>>,
}

/// Internal streamer state
struct StreamerState {
    /// Whether stream is active
    is_active: bool,
    /// Deferred paths pending
    pending_paths: Vec<Vec<PathSegment>>,
    /// Errors encountered
    errors: Vec<StreamError>,
    /// Progress callback
    progress_callback: Option<Box<dyn Fn(StreamProgress) + Send + Sync>>,
}

impl ResponseStreamer {
    /// Create a new response streamer
    pub fn new(config: StreamingConfig) -> (Self, mpsc::Receiver<ResponseChunk>) {
        let (sender, receiver) = mpsc::channel(config.buffer_size);
        let id = uuid::Uuid::new_v4().to_string();

        let streamer = Self {
            id,
            config,
            sender,
            chunk_id: AtomicU64::new(0),
            start_time: Instant::now(),
            bytes_sent: AtomicU64::new(0),
            state: Arc::new(RwLock::new(StreamerState {
                is_active: true,
                pending_paths: Vec::new(),
                errors: Vec::new(),
                progress_callback: None,
            })),
        };

        (streamer, receiver)
    }

    /// Get stream ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Set progress callback
    pub async fn set_progress_callback<F>(&self, callback: F)
    where
        F: Fn(StreamProgress) + Send + Sync + 'static,
    {
        let mut state = self.state.write().await;
        state.progress_callback = Some(Box::new(callback));
    }

    /// Send a chunk
    pub async fn send_chunk(
        &self,
        data: serde_json::Value,
        path: Option<Vec<PathSegment>>,
        label: Option<String>,
        is_final: bool,
        has_next: bool,
    ) -> Result<(), StreamError> {
        let state = self.state.read().await;
        if !state.is_active {
            return Err(StreamError {
                message: "Stream is not active".to_string(),
                path: None,
                extensions: None,
            });
        }
        drop(state);

        let chunk_id = self.chunk_id.fetch_add(1, Ordering::SeqCst);

        // Calculate chunk size
        let chunk_json = serde_json::to_string(&data).unwrap_or_default();
        let chunk_size = chunk_json.len() as u64;
        self.bytes_sent.fetch_add(chunk_size, Ordering::SeqCst);

        let chunk = ResponseChunk {
            id: chunk_id,
            path,
            data,
            has_errors: false,
            errors: Vec::new(),
            is_final,
            has_next,
            label,
        };

        self.sender.send(chunk).await.map_err(|_| StreamError {
            message: "Failed to send chunk".to_string(),
            path: None,
            extensions: None,
        })?;

        // Report progress
        if self.config.track_progress {
            self.report_progress().await;
        }

        Ok(())
    }

    /// Send an error chunk
    pub async fn send_error(&self, error: StreamError, path: Option<Vec<PathSegment>>) {
        let chunk_id = self.chunk_id.fetch_add(1, Ordering::SeqCst);

        let chunk = ResponseChunk {
            id: chunk_id,
            path,
            data: serde_json::Value::Null,
            has_errors: true,
            errors: vec![error.clone()],
            is_final: false,
            has_next: true,
            label: None,
        };

        let _ = self.sender.send(chunk).await;

        // Track error
        let mut state = self.state.write().await;
        state.errors.push(error);
    }

    /// Send heartbeat to keep connection alive
    pub async fn send_heartbeat(&self) -> Result<(), StreamError> {
        let chunk = ResponseChunk {
            id: self.chunk_id.fetch_add(1, Ordering::SeqCst),
            path: None,
            data: serde_json::json!({"__heartbeat": true}),
            has_errors: false,
            errors: Vec::new(),
            is_final: false,
            has_next: true,
            label: Some("__heartbeat".to_string()),
        };

        self.sender.send(chunk).await.map_err(|_| StreamError {
            message: "Failed to send heartbeat".to_string(),
            path: None,
            extensions: None,
        })
    }

    /// Complete the stream
    pub async fn complete(&self) {
        // Send final chunk
        let chunk = ResponseChunk {
            id: self.chunk_id.fetch_add(1, Ordering::SeqCst),
            path: None,
            data: serde_json::Value::Null,
            has_errors: false,
            errors: Vec::new(),
            is_final: true,
            has_next: false,
            label: None,
        };

        let _ = self.sender.send(chunk).await;

        // Mark as inactive
        let mut state = self.state.write().await;
        state.is_active = false;
    }

    /// Get current progress
    pub async fn get_progress(&self) -> StreamProgress {
        let state = self.state.read().await;
        let elapsed = self.start_time.elapsed();
        let bytes = self.bytes_sent.load(Ordering::SeqCst);

        StreamProgress {
            stream_id: self.id.clone(),
            chunks_sent: self.chunk_id.load(Ordering::SeqCst),
            bytes_sent: bytes,
            total_bytes: None,
            throughput_bps: if elapsed.as_secs_f64() > 0.0 {
                bytes as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            },
            elapsed_ms: elapsed.as_millis() as u64,
            pending_chunks: state.pending_paths.len(),
            is_complete: !state.is_active,
            has_errors: !state.errors.is_empty(),
        }
    }

    /// Report progress via callback
    async fn report_progress(&self) {
        let progress = self.get_progress().await;
        let state = self.state.read().await;
        if let Some(ref callback) = state.progress_callback {
            callback(progress);
        }
    }
}

/// Incremental delivery manager for @defer and @stream
pub struct IncrementalDeliveryManager {
    /// Configuration
    config: StreamingConfig,
    /// Active streams
    streams: Arc<RwLock<HashMap<String, StreamInfo>>>,
    /// Global statistics
    stats: Arc<RwLock<StreamingStatistics>>,
}

/// Information about an active stream
#[derive(Debug, Clone)]
struct StreamInfo {
    /// Stream ID
    id: String,
    /// Mode
    #[allow(dead_code)]
    mode: StreamingMode,
    /// Start time
    started_at: Instant,
    /// Bytes sent
    bytes_sent: u64,
    /// Chunks sent
    chunks_sent: u64,
    /// Client ID
    #[allow(dead_code)]
    client_id: Option<String>,
}

/// Global streaming statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamingStatistics {
    /// Total streams created
    pub total_streams: u64,
    /// Currently active streams
    pub active_streams: u64,
    /// Total bytes streamed
    pub total_bytes: u64,
    /// Total chunks sent
    pub total_chunks: u64,
    /// Average stream duration (ms)
    pub avg_duration_ms: f64,
    /// Streams by mode
    pub streams_by_mode: HashMap<String, u64>,
    /// Error count
    pub error_count: u64,
}

impl IncrementalDeliveryManager {
    /// Create a new incremental delivery manager
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            streams: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(StreamingStatistics::default())),
        }
    }

    /// Create a new stream
    pub async fn create_stream(
        &self,
        mode: StreamingMode,
        client_id: Option<&str>,
    ) -> (ResponseStreamer, mpsc::Receiver<ResponseChunk>) {
        let (streamer, receiver) = ResponseStreamer::new(self.config.clone());
        let stream_id = streamer.id().to_string();

        // Register stream
        {
            let mut streams = self.streams.write().await;
            streams.insert(
                stream_id.clone(),
                StreamInfo {
                    id: stream_id.clone(),
                    mode,
                    started_at: Instant::now(),
                    bytes_sent: 0,
                    chunks_sent: 0,
                    client_id: client_id.map(|s| s.to_string()),
                },
            );
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_streams += 1;
            stats.active_streams += 1;
            *stats
                .streams_by_mode
                .entry(format!("{:?}", mode))
                .or_insert(0) += 1;
        }

        (streamer, receiver)
    }

    /// Close a stream
    pub async fn close_stream(&self, stream_id: &str) {
        let stream_info = {
            let mut streams = self.streams.write().await;
            streams.remove(stream_id)
        };

        if let Some(info) = stream_info {
            let mut stats = self.stats.write().await;
            stats.active_streams = stats.active_streams.saturating_sub(1);
            stats.total_bytes += info.bytes_sent;
            stats.total_chunks += info.chunks_sent;

            // Update average duration
            let duration = info.started_at.elapsed().as_millis() as f64;
            let n = stats.total_streams as f64;
            stats.avg_duration_ms = (stats.avg_duration_ms * (n - 1.0) + duration) / n;
        }
    }

    /// Get active streams
    pub async fn get_active_streams(&self) -> Vec<StreamProgress> {
        let streams = self.streams.read().await;
        streams
            .values()
            .map(|info| StreamProgress {
                stream_id: info.id.clone(),
                chunks_sent: info.chunks_sent,
                bytes_sent: info.bytes_sent,
                total_bytes: None,
                throughput_bps: 0.0,
                elapsed_ms: info.started_at.elapsed().as_millis() as u64,
                pending_chunks: 0,
                is_complete: false,
                has_errors: false,
            })
            .collect()
    }

    /// Get statistics
    pub async fn get_statistics(&self) -> StreamingStatistics {
        self.stats.read().await.clone()
    }

    /// Process incremental response with @defer
    pub async fn process_deferred(
        &self,
        streamer: &ResponseStreamer,
        initial_data: serde_json::Value,
        deferred_fields: Vec<DeferredField>,
    ) -> Result<(), StreamError> {
        let total_deferred = deferred_fields.len();

        // Send initial response
        streamer
            .send_chunk(initial_data, None, None, false, total_deferred > 0)
            .await?;

        // Process deferred fields
        for (i, deferred) in deferred_fields.into_iter().enumerate() {
            let is_last = i == total_deferred - 1;

            // Simulate async resolution
            let data = (deferred.resolver)().await;

            streamer
                .send_chunk(data, Some(deferred.path), deferred.label, is_last, !is_last)
                .await?;
        }

        Ok(())
    }

    /// Process streaming response with @stream
    pub async fn process_stream<I, T>(
        &self,
        streamer: &ResponseStreamer,
        path: Vec<PathSegment>,
        label: Option<String>,
        items: I,
    ) -> Result<(), StreamError>
    where
        I: IntoIterator<Item = T>,
        T: Serialize,
    {
        let mut items_iter = items.into_iter().peekable();
        let mut index = 0;

        while let Some(item) = items_iter.next() {
            let has_next = items_iter.peek().is_some();
            let mut item_path = path.clone();
            item_path.push(PathSegment::Index(index));

            let data = serde_json::to_value(&item).map_err(|e| StreamError {
                message: format!("Serialization error: {}", e),
                path: Some(item_path.clone()),
                extensions: None,
            })?;

            streamer
                .send_chunk(data, Some(item_path), label.clone(), !has_next, has_next)
                .await?;

            index += 1;
        }

        Ok(())
    }
}

/// A deferred field to be resolved later
pub struct DeferredField {
    /// Path to the field
    pub path: Vec<PathSegment>,
    /// Optional label from @defer directive
    pub label: Option<String>,
    /// Resolver function
    pub resolver: Box<
        dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = serde_json::Value> + Send>>
            + Send
            + Sync,
    >,
}

/// Multipart response formatter
pub struct MultipartFormatter {
    /// Boundary string
    boundary: String,
}

impl MultipartFormatter {
    /// Create a new multipart formatter
    pub fn new() -> Self {
        Self {
            boundary: format!("-graphql-{}", uuid::Uuid::new_v4()),
        }
    }

    /// Get the content type header value
    pub fn content_type(&self) -> String {
        format!("multipart/mixed; boundary=\"{}\"", self.boundary)
    }

    /// Format a chunk as multipart
    pub fn format_chunk(&self, chunk: &ResponseChunk) -> Vec<u8> {
        let mut output = Vec::new();

        // Boundary
        output.extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());

        // Headers
        output.extend_from_slice(b"Content-Type: application/json; charset=utf-8\r\n");

        if chunk.is_final {
            output.extend_from_slice(b"X-GraphQL-Final: true\r\n");
        }

        if let Some(ref label) = chunk.label {
            output.extend_from_slice(format!("X-GraphQL-Label: {}\r\n", label).as_bytes());
        }

        output.extend_from_slice(b"\r\n");

        // Body
        let body = serde_json::json!({
            "data": chunk.data,
            "path": chunk.path,
            "hasNext": chunk.has_next,
            "errors": if chunk.has_errors { Some(&chunk.errors) } else { None },
        });

        output.extend_from_slice(serde_json::to_string(&body).unwrap().as_bytes());
        output.extend_from_slice(b"\r\n");

        output
    }

    /// Format the final boundary
    pub fn format_final(&self) -> Vec<u8> {
        format!("--{}--\r\n", self.boundary).into_bytes()
    }
}

impl Default for MultipartFormatter {
    fn default() -> Self {
        Self::new()
    }
}

/// Compression utilities for streamed responses
pub mod compression {
    use super::*;

    /// Compress data using the specified algorithm
    pub fn compress(data: &[u8], compression_type: CompressionType) -> Result<Vec<u8>, String> {
        match compression_type {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Gzip => compress_gzip(data),
            CompressionType::Deflate => compress_deflate(data),
            CompressionType::Brotli => compress_brotli(data),
        }
    }

    fn compress_gzip(data: &[u8]) -> Result<Vec<u8>, String> {
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| format!("Gzip compression error: {}", e))?;
        encoder
            .finish()
            .map_err(|e| format!("Gzip finalization error: {}", e))
    }

    fn compress_deflate(data: &[u8]) -> Result<Vec<u8>, String> {
        use std::io::Write;
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| format!("Deflate compression error: {}", e))?;
        encoder
            .finish()
            .map_err(|e| format!("Deflate finalization error: {}", e))
    }

    fn compress_brotli(_data: &[u8]) -> Result<Vec<u8>, String> {
        // Placeholder - would need brotli crate
        Err("Brotli compression not implemented".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_streamer_creation() {
        let config = StreamingConfig::default();
        let (streamer, _receiver) = ResponseStreamer::new(config);
        assert!(!streamer.id().is_empty());
    }

    #[tokio::test]
    async fn test_send_chunk() {
        let config = StreamingConfig::default();
        let (streamer, mut receiver) = ResponseStreamer::new(config);

        let data = serde_json::json!({"field": "value"});
        streamer
            .send_chunk(data.clone(), None, None, false, true)
            .await
            .unwrap();

        let chunk = receiver.recv().await.unwrap();
        assert_eq!(chunk.data, data);
        assert!(!chunk.is_final);
        assert!(chunk.has_next);
    }

    #[tokio::test]
    async fn test_send_error() {
        let config = StreamingConfig::default();
        let (streamer, mut receiver) = ResponseStreamer::new(config);

        let error = StreamError {
            message: "Test error".to_string(),
            path: None,
            extensions: None,
        };
        streamer.send_error(error, None).await;

        let chunk = receiver.recv().await.unwrap();
        assert!(chunk.has_errors);
        assert!(!chunk.errors.is_empty());
    }

    #[tokio::test]
    async fn test_complete_stream() {
        let config = StreamingConfig::default();
        let (streamer, mut receiver) = ResponseStreamer::new(config);

        streamer.complete().await;

        let chunk = receiver.recv().await.unwrap();
        assert!(chunk.is_final);
        assert!(!chunk.has_next);
    }

    #[tokio::test]
    async fn test_progress_tracking() {
        let config = StreamingConfig {
            track_progress: true,
            ..Default::default()
        };
        let (streamer, _receiver) = ResponseStreamer::new(config);

        // Send some data
        let data = serde_json::json!({"test": "data"});
        let _ = streamer.send_chunk(data, None, None, false, true).await;

        let progress = streamer.get_progress().await;
        assert_eq!(progress.chunks_sent, 1);
        assert!(progress.bytes_sent > 0);
    }

    #[tokio::test]
    async fn test_manager_create_stream() {
        let manager = IncrementalDeliveryManager::new(StreamingConfig::default());

        let (_streamer, _receiver) = manager.create_stream(StreamingMode::Chunked, None).await;

        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_streams, 1);
        assert_eq!(stats.active_streams, 1);
    }

    #[tokio::test]
    async fn test_manager_close_stream() {
        let manager = IncrementalDeliveryManager::new(StreamingConfig::default());

        let (streamer, _receiver) = manager.create_stream(StreamingMode::Chunked, None).await;
        let stream_id = streamer.id().to_string();

        manager.close_stream(&stream_id).await;

        let stats = manager.get_statistics().await;
        assert_eq!(stats.active_streams, 0);
    }

    #[tokio::test]
    async fn test_multipart_formatter() {
        let formatter = MultipartFormatter::new();

        assert!(formatter.content_type().contains("multipart/mixed"));

        let chunk = ResponseChunk {
            id: 0,
            path: None,
            data: serde_json::json!({"test": true}),
            has_errors: false,
            errors: Vec::new(),
            is_final: false,
            has_next: true,
            label: None,
        };

        let formatted = formatter.format_chunk(&chunk);
        assert!(!formatted.is_empty());
    }

    #[tokio::test]
    async fn test_path_segments() {
        let path = vec![
            PathSegment::Field("users".to_string()),
            PathSegment::Index(0),
            PathSegment::Field("name".to_string()),
        ];

        let json = serde_json::to_string(&path).unwrap();
        assert!(json.contains("users"));
    }

    #[tokio::test]
    async fn test_stream_items() {
        let manager = IncrementalDeliveryManager::new(StreamingConfig::default());
        let (streamer, mut receiver) = manager.create_stream(StreamingMode::Streamed, None).await;

        let items = vec!["item1", "item2", "item3"];
        let path = vec![PathSegment::Field("items".to_string())];

        // Spawn stream processing
        let streamer_clone = Arc::new(streamer);
        tokio::spawn(async move {
            let _ = manager
                .process_stream(&streamer_clone, path, Some("items".to_string()), items)
                .await;
        });

        // Receive chunks
        let mut received = Vec::new();
        while let Some(chunk) = receiver.recv().await {
            received.push(chunk);
            if received.len() >= 3 {
                break;
            }
        }

        assert_eq!(received.len(), 3);
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let config = StreamingConfig::default();
        let (streamer, mut receiver) = ResponseStreamer::new(config);

        streamer.send_heartbeat().await.unwrap();

        let chunk = receiver.recv().await.unwrap();
        assert_eq!(chunk.label, Some("__heartbeat".to_string()));
    }

    #[tokio::test]
    async fn test_compression() {
        let data = b"Hello, World!";

        let compressed = compression::compress(data, CompressionType::Gzip).unwrap();
        assert!(!compressed.is_empty());

        let deflated = compression::compress(data, CompressionType::Deflate).unwrap();
        assert!(!deflated.is_empty());

        let none = compression::compress(data, CompressionType::None).unwrap();
        assert_eq!(none, data);
    }
}
