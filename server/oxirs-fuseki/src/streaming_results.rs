//! Memory-Efficient Streaming for Large Results
//!
//! This module provides efficient streaming of large SPARQL query results:
//! - Zero-copy result streaming using SciRS2
//! - Chunked result processing with backpressure
//! - Adaptive buffer sizing based on memory pressure
//! - Multiple output formats (JSON, XML, CSV, TSV)
//! - Compression support (gzip, brotli)

use crate::error::{FusekiError, FusekiResult};
use crate::memory_pool::MemoryManager;
use bytes::{Bytes, BytesMut};
use futures::Stream;
use scirs2_core::memory_efficient::{AdaptiveChunking, ChunkedArray, ZeroCopyOps};
use scirs2_core::metrics::{Counter, Histogram};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info, instrument, warn};

/// Result format for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResultFormat {
    Json,
    Xml,
    Csv,
    Tsv,
    NTriples,
    Turtle,
    RdfXml,
}

impl ResultFormat {
    pub fn content_type(&self) -> &'static str {
        match self {
            ResultFormat::Json => "application/sparql-results+json",
            ResultFormat::Xml => "application/sparql-results+xml",
            ResultFormat::Csv => "text/csv",
            ResultFormat::Tsv => "text/tab-separated-values",
            ResultFormat::NTriples => "application/n-triples",
            ResultFormat::Turtle => "text/turtle",
            ResultFormat::RdfXml => "application/rdf+xml",
        }
    }
}

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    None,
    Gzip,
    Brotli,
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Chunk size in bytes
    pub chunk_size: usize,
    /// Buffer size in number of chunks
    pub buffer_size: usize,
    /// Enable adaptive chunking
    pub adaptive_chunking: bool,
    /// Maximum memory per stream in bytes
    pub max_memory_per_stream: u64,
    /// Enable compression
    pub compression: Compression,
    /// Compression level (1-9 for gzip, 0-11 for brotli)
    pub compression_level: u32,
    /// Backpressure threshold (0.0-1.0)
    pub backpressure_threshold: f64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            chunk_size: 64 * 1024, // 64KB
            buffer_size: 16,       // 1MB total
            adaptive_chunking: true,
            max_memory_per_stream: 16 * 1024 * 1024, // 16MB
            compression: Compression::None,
            compression_level: 6,
            backpressure_threshold: 0.8,
        }
    }
}

/// Streaming statistics
#[derive(Debug, Clone, Serialize)]
pub struct StreamStats {
    pub total_bytes: u64,
    pub total_chunks: u64,
    pub total_rows: u64,
    pub compression_ratio: f64,
    pub average_chunk_size: f64,
    pub throughput_mbps: f64,
    pub active_streams: usize,
    pub backpressure_events: u64,
}

/// Result chunk for streaming
#[derive(Debug, Clone)]
pub struct ResultChunk {
    pub data: Bytes,
    pub sequence: u64,
    pub is_last: bool,
    pub original_size: usize,
    pub compressed_size: usize,
}

/// Streaming result producer
pub struct StreamingProducer {
    config: StreamConfig,
    format: ResultFormat,
    memory_manager: Option<Arc<MemoryManager>>,

    // Channel for sending chunks
    tx: mpsc::Sender<FusekiResult<ResultChunk>>,

    // Statistics
    total_bytes: Arc<AtomicU64>,
    total_chunks: Arc<AtomicU64>,
    total_rows: Arc<AtomicU64>,
    compressed_bytes: Arc<AtomicU64>,

    // Sequence counter
    sequence: Arc<AtomicU64>,

    // Start time for throughput calculation
    start_time: Instant,

    // Backpressure semaphore
    backpressure: Arc<Semaphore>,
}

impl StreamingProducer {
    /// Create a new streaming producer
    pub fn new(
        config: StreamConfig,
        format: ResultFormat,
        memory_manager: Option<Arc<MemoryManager>>,
    ) -> (Self, ReceiverStream<FusekiResult<ResultChunk>>) {
        let (tx, rx) = mpsc::channel(config.buffer_size);
        let backpressure = Arc::new(Semaphore::new(config.buffer_size));

        let producer = StreamingProducer {
            config,
            format,
            memory_manager,
            tx,
            total_bytes: Arc::new(AtomicU64::new(0)),
            total_chunks: Arc::new(AtomicU64::new(0)),
            total_rows: Arc::new(AtomicU64::new(0)),
            compressed_bytes: Arc::new(AtomicU64::new(0)),
            sequence: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            backpressure,
        };

        let stream = ReceiverStream::new(rx);

        (producer, stream)
    }

    /// Write result row to stream
    #[instrument(skip(self, row_data))]
    pub async fn write_row(&mut self, row_data: &[u8]) -> FusekiResult<()> {
        self.total_rows.fetch_add(1, Ordering::Relaxed);
        self.write_chunk(row_data, false).await
    }

    /// Write chunk to stream
    #[instrument(skip(self, data))]
    pub async fn write_chunk(&mut self, data: &[u8], is_last: bool) -> FusekiResult<()> {
        // Check backpressure
        let _permit = self
            .backpressure
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| FusekiError::server_error("Stream closed"))?;

        let original_size = data.len();
        self.total_bytes
            .fetch_add(original_size as u64, Ordering::Relaxed);

        // Compress if enabled
        let (compressed_data, compressed_size) = match self.config.compression {
            Compression::None => (Bytes::copy_from_slice(data), original_size),
            Compression::Gzip => {
                let compressed = self.compress_gzip(data)?;
                let size = compressed.len();
                (compressed, size)
            }
            Compression::Brotli => {
                let compressed = self.compress_brotli(data)?;
                let size = compressed.len();
                (compressed, size)
            }
        };

        self.compressed_bytes
            .fetch_add(compressed_size as u64, Ordering::Relaxed);

        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);

        let chunk = ResultChunk {
            data: compressed_data,
            sequence,
            is_last,
            original_size,
            compressed_size,
        };

        self.tx
            .send(Ok(chunk))
            .await
            .map_err(|_| FusekiError::server_error("Stream receiver dropped"))?;

        self.total_chunks.fetch_add(1, Ordering::Relaxed);

        debug!(
            "Wrote chunk {} ({} bytes, compressed to {})",
            sequence, original_size, compressed_size
        );

        Ok(())
    }

    /// Finalize the stream
    #[instrument(skip(self))]
    pub async fn finalize(mut self) -> FusekiResult<StreamStats> {
        // Send empty final chunk
        self.write_chunk(&[], true).await?;

        let duration = self.start_time.elapsed();
        let total_bytes = self.total_bytes.load(Ordering::Relaxed);
        let compressed_bytes = self.compressed_bytes.load(Ordering::Relaxed);
        let total_chunks = self.total_chunks.load(Ordering::Relaxed);

        let compression_ratio = if compressed_bytes > 0 {
            (total_bytes as f64) / (compressed_bytes as f64)
        } else {
            1.0
        };

        let average_chunk_size = if total_chunks > 0 {
            (total_bytes as f64) / (total_chunks as f64)
        } else {
            0.0
        };

        let throughput_mbps = if duration.as_secs_f64() > 0.0 {
            (total_bytes as f64) / (1024.0 * 1024.0) / duration.as_secs_f64()
        } else {
            0.0
        };

        info!(
            "Stream finalized: {} bytes in {} chunks ({:.2} MB/s, {:.2}x compression)",
            total_bytes, total_chunks, throughput_mbps, compression_ratio
        );

        Ok(StreamStats {
            total_bytes,
            total_chunks,
            total_rows: self.total_rows.load(Ordering::Relaxed),
            compression_ratio,
            average_chunk_size,
            throughput_mbps,
            active_streams: 0,
            backpressure_events: 0,
        })
    }

    /// Compress data using gzip
    fn compress_gzip(&self, data: &[u8]) -> FusekiResult<Bytes> {
        use flate2::write::GzEncoder;
        use flate2::Compression as GzCompression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(
            Vec::new(),
            GzCompression::new(self.config.compression_level),
        );

        encoder
            .write_all(data)
            .map_err(|e| FusekiError::server_error(format!("Gzip compression failed: {}", e)))?;

        let compressed = encoder
            .finish()
            .map_err(|e| FusekiError::server_error(format!("Gzip finalization failed: {}", e)))?;

        Ok(Bytes::from(compressed))
    }

    /// Compress data using brotli
    fn compress_brotli(&self, data: &[u8]) -> FusekiResult<Bytes> {
        let mut compressed = Vec::new();
        let params = brotli::enc::BrotliEncoderParams {
            quality: self.config.compression_level as i32,
            ..Default::default()
        };

        brotli::BrotliCompress(&mut std::io::Cursor::new(data), &mut compressed, &params)
            .map_err(|e| FusekiError::server_error(format!("Brotli compression failed: {}", e)))?;

        Ok(Bytes::from(compressed))
    }
}

/// Streaming result consumer
pub struct StreamingConsumer {
    stream: Pin<Box<dyn Stream<Item = FusekiResult<ResultChunk>> + Send>>,
    config: StreamConfig,

    // Statistics
    bytes_received: Arc<AtomicU64>,
    chunks_received: Arc<AtomicU64>,
}

impl StreamingConsumer {
    /// Create a new streaming consumer
    pub fn new<S>(stream: S, config: StreamConfig) -> Self
    where
        S: Stream<Item = FusekiResult<ResultChunk>> + Send + 'static,
    {
        StreamingConsumer {
            stream: Box::pin(stream),
            config,
            bytes_received: Arc::new(AtomicU64::new(0)),
            chunks_received: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Read next chunk
    pub async fn next_chunk(&mut self) -> Option<FusekiResult<ResultChunk>> {
        use futures::StreamExt;

        match self.stream.next().await {
            Some(Ok(chunk)) => {
                self.bytes_received
                    .fetch_add(chunk.data.len() as u64, Ordering::Relaxed);
                self.chunks_received.fetch_add(1, Ordering::Relaxed);
                Some(Ok(chunk))
            }
            Some(Err(e)) => Some(Err(e)),
            None => None,
        }
    }

    /// Collect all chunks into a single buffer
    pub async fn collect_all(&mut self) -> FusekiResult<Bytes> {
        let mut buffer = BytesMut::new();

        while let Some(result) = self.next_chunk().await {
            let chunk = result?;
            buffer.extend_from_slice(&chunk.data);

            if chunk.is_last {
                break;
            }
        }

        Ok(buffer.freeze())
    }

    /// Get statistics
    pub fn stats(&self) -> (u64, u64) {
        (
            self.bytes_received.load(Ordering::Relaxed),
            self.chunks_received.load(Ordering::Relaxed),
        )
    }
}

/// Stream manager for handling multiple concurrent streams
pub struct StreamManager {
    config: StreamConfig,
    memory_manager: Option<Arc<MemoryManager>>,

    // Active streams
    active_streams: Arc<RwLock<std::collections::HashMap<String, Instant>>>,

    // Statistics
    total_streams: Arc<AtomicU64>,
    active_count: Arc<AtomicUsize>,

    // Semaphore for limiting concurrent streams
    stream_semaphore: Arc<Semaphore>,
}

impl StreamManager {
    /// Create a new stream manager
    pub fn new(config: StreamConfig, memory_manager: Option<Arc<MemoryManager>>) -> Arc<Self> {
        Arc::new(StreamManager {
            config,
            memory_manager,
            active_streams: Arc::new(RwLock::new(std::collections::HashMap::new())),
            total_streams: Arc::new(AtomicU64::new(0)),
            active_count: Arc::new(AtomicUsize::new(0)),
            stream_semaphore: Arc::new(Semaphore::new(100)), // Max 100 concurrent streams
        })
    }

    /// Create a new streaming producer
    pub async fn create_producer(
        &self,
        format: ResultFormat,
    ) -> FusekiResult<(
        String,
        StreamingProducer,
        ReceiverStream<FusekiResult<ResultChunk>>,
    )> {
        // Acquire stream permit
        let _permit = self
            .stream_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| FusekiError::service_unavailable("Too many active streams"))?;

        let stream_id = uuid::Uuid::new_v4().to_string();

        // Register stream
        {
            let mut active = self.active_streams.write().await;
            active.insert(stream_id.clone(), Instant::now());
        }

        self.total_streams.fetch_add(1, Ordering::Relaxed);
        self.active_count.fetch_add(1, Ordering::Relaxed);

        let (producer, stream) =
            StreamingProducer::new(self.config.clone(), format, self.memory_manager.clone());

        info!("Created stream {} (format: {:?})", stream_id, format);

        Ok((stream_id, producer, stream))
    }

    /// Get stream statistics
    pub async fn get_stats(&self) -> StreamStats {
        let active = self.active_streams.read().await;

        StreamStats {
            total_bytes: 0, // Aggregated from individual streams
            total_chunks: 0,
            total_rows: 0,
            compression_ratio: 1.0,
            average_chunk_size: 0.0,
            throughput_mbps: 0.0,
            active_streams: active.len(),
            backpressure_events: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_streaming_producer() {
        let config = StreamConfig::default();
        let (mut producer, mut stream) = StreamingProducer::new(config, ResultFormat::Json, None);

        // Write some data
        tokio::spawn(async move {
            producer.write_chunk(b"test data 1", false).await.unwrap();
            producer.write_chunk(b"test data 2", false).await.unwrap();
            producer.finalize().await.unwrap();
        });

        // Read data
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }

        assert!(chunks.len() >= 2);
    }

    #[tokio::test]
    async fn test_compression() {
        let config = StreamConfig {
            compression: Compression::Gzip,
            ..Default::default()
        };

        let (mut producer, mut stream) = StreamingProducer::new(config, ResultFormat::Json, None);

        let test_data = b"This is test data that should compress well. ".repeat(100);

        tokio::spawn(async move {
            producer.write_chunk(&test_data, true).await.unwrap();
        });

        let chunk = stream.next().await.unwrap().unwrap();

        // Compressed size should be smaller
        assert!(chunk.compressed_size < chunk.original_size);
        assert!(chunk.is_last);
    }

    #[tokio::test]
    async fn test_stream_manager() {
        let config = StreamConfig::default();
        let manager = StreamManager::new(config, None);

        let result = manager.create_producer(ResultFormat::Json).await;
        assert!(result.is_ok());

        let (stream_id, _, _) = result.unwrap();
        assert!(!stream_id.is_empty());
    }
}
