//! Query Result Compression and Streaming
//!
//! This module provides advanced result processing capabilities including compression,
//! streaming, and adaptive data transfer optimization for federated query results.

use anyhow::{anyhow, Result};
use bytes::{Bytes, BytesMut};
use flate2::{write::GzEncoder, Compression as GzCompression};
use futures::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::io::Write;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info, warn};

use crate::executor::{GraphQLResponse, SparqlResults};
use crate::QueryResult;

/// Result compression and streaming manager
#[derive(Debug)]
pub struct ResultStreamingManager {
    config: StreamingConfig,
    compression_stats: Arc<RwLock<CompressionStatistics>>,
    active_streams: Arc<RwLock<HashMap<String, StreamMetrics>>>,
}

/// Configuration for result streaming and compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Enable result compression
    pub enable_compression: bool,
    /// Compression algorithm to use
    pub compression_algorithm: CompressionAlgorithm,
    /// Compression level (1-9 for gzip, 1-21 for brotli)
    pub compression_level: u32,
    /// Minimum result size to trigger compression (bytes)
    pub compression_threshold: usize,
    /// Streaming chunk size
    pub chunk_size: usize,
    /// Buffer size for streaming
    pub buffer_size: usize,
    /// Enable adaptive streaming
    pub enable_adaptive_streaming: bool,
    /// Streaming timeout per chunk
    pub chunk_timeout: Duration,
    /// Enable result caching
    pub enable_result_caching: bool,
    /// Maximum cached result size
    pub max_cache_size: usize,
    /// Compression quality vs speed preference (0.0-1.0, higher = better compression)
    pub compression_preference: f64,
}

/// Supported compression algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Deflate,
    Brotli,
    Lz4,
    Zstd,
}

/// Result streaming format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResultFormat {
    Json,
    MessagePack,
    Avro,
    Protobuf,
    Csv,
    Xml,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Gzip,
            compression_level: 6,
            compression_threshold: 1024, // 1KB
            chunk_size: 8192,            // 8KB chunks
            buffer_size: 65536,          // 64KB buffer
            enable_adaptive_streaming: true,
            chunk_timeout: Duration::from_secs(30),
            enable_result_caching: true,
            max_cache_size: 10 * 1024 * 1024, // 10MB
            compression_preference: 0.7,      // Favor compression over speed
        }
    }
}

/// Compressed query result
#[derive(Debug, Clone)]
pub struct CompressedResult {
    /// Compressed data
    pub data: Bytes,
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,
    /// Compression level used
    pub level: u32,
    /// Compression time taken
    pub compression_time: Duration,
    /// Compression ratio (compressed / original)
    pub compression_ratio: f64,
}

/// Streaming result chunk
#[derive(Debug, Clone)]
pub struct ResultChunk {
    /// Chunk sequence number
    pub sequence: u64,
    /// Chunk data
    pub data: Bytes,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Chunk metadata
    pub metadata: ChunkMetadata,
}

/// Metadata for result chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Total expected chunks
    pub total_chunks: Option<u64>,
    /// Chunk compression info
    pub compression: Option<CompressionInfo>,
    /// Result format
    pub format: ResultFormat,
    /// Estimated total result size
    pub estimated_total_size: Option<usize>,
}

/// Compression information for a chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionInfo {
    pub algorithm: CompressionAlgorithm,
    pub level: u32,
    pub original_size: usize,
    pub compressed_size: usize,
}

/// Statistics for compression operations
#[derive(Debug, Default, Clone)]
pub struct CompressionStatistics {
    pub total_compressions: u64,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
    pub total_compression_time: Duration,
    pub average_compression_ratio: f64,
    pub algorithm_stats: HashMap<CompressionAlgorithm, AlgorithmStats>,
}

/// Per-algorithm compression statistics
#[derive(Debug, Default, Clone)]
pub struct AlgorithmStats {
    pub uses: u64,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
    pub total_compression_time: Duration,
    pub average_compression_ratio: f64,
}

/// Metrics for active streams
#[derive(Debug, Clone)]
pub struct StreamMetrics {
    pub stream_id: String,
    pub start_time: Instant,
    pub chunks_sent: u64,
    pub bytes_sent: u64,
    pub last_chunk_time: Instant,
    pub is_completed: bool,
    pub error_count: u64,
}

impl ResultStreamingManager {
    /// Create a new result streaming manager
    pub fn new() -> Self {
        Self::with_config(StreamingConfig::default())
    }

    /// Create a new result streaming manager with custom configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        Self {
            config,
            compression_stats: Arc::new(RwLock::new(CompressionStatistics::default())),
            active_streams: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Compress query results
    pub async fn compress_result(&self, data: &[u8]) -> Result<CompressedResult> {
        let start_time = Instant::now();

        if !self.config.enable_compression || data.len() < self.config.compression_threshold {
            return Ok(CompressedResult {
                data: Bytes::copy_from_slice(data),
                original_size: data.len(),
                compressed_size: data.len(),
                algorithm: CompressionAlgorithm::None,
                level: 0,
                compression_time: Duration::from_nanos(0),
                compression_ratio: 1.0,
            });
        }

        let compressed_data = self.compress_data(data).await?;
        let compression_time = start_time.elapsed();
        let compression_ratio = compressed_data.len() as f64 / data.len() as f64;

        // Update statistics
        self.update_compression_stats(
            self.config.compression_algorithm,
            data.len(),
            compressed_data.len(),
            compression_time,
        )
        .await;

        info!(
            "Compressed {} bytes to {} bytes (ratio: {:.2}, time: {:?})",
            data.len(),
            compressed_data.len(),
            compression_ratio,
            compression_time
        );

        let compressed_size = compressed_data.len();
        Ok(CompressedResult {
            data: Bytes::from(compressed_data),
            original_size: data.len(),
            compressed_size,
            algorithm: self.config.compression_algorithm,
            level: self.config.compression_level,
            compression_time,
            compression_ratio,
        })
    }

    /// Decompress query results
    pub async fn decompress_result(&self, compressed: &CompressedResult) -> Result<Bytes> {
        if compressed.algorithm == CompressionAlgorithm::None {
            return Ok(compressed.data.clone());
        }

        let decompressed = self
            .decompress_data(&compressed.data, compressed.algorithm)
            .await?;

        if decompressed.len() != compressed.original_size {
            warn!(
                "Decompressed size mismatch: expected {}, got {}",
                compressed.original_size,
                decompressed.len()
            );
        }

        Ok(Bytes::from(decompressed))
    }

    /// Create a streaming result from query results
    pub async fn create_stream(
        &self,
        result: QueryResult,
        format: ResultFormat,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResultChunk>> + Send>>> {
        let stream_id = uuid::Uuid::new_v4().to_string();

        // Initialize stream metrics
        let metrics = StreamMetrics {
            stream_id: stream_id.clone(),
            start_time: Instant::now(),
            chunks_sent: 0,
            bytes_sent: 0,
            last_chunk_time: Instant::now(),
            is_completed: false,
            error_count: 0,
        };

        self.active_streams
            .write()
            .await
            .insert(stream_id.clone(), metrics);

        // Serialize the result
        let serialized_data = self.serialize_result(&result, &format).await?;

        // Create chunk stream
        let chunk_stream = self
            .create_chunk_stream(stream_id, serialized_data, format)
            .await?;

        Ok(chunk_stream)
    }

    /// Create an adaptive streaming result that adjusts chunk size based on network conditions
    pub async fn create_adaptive_stream(
        &self,
        result: QueryResult,
        format: ResultFormat,
        network_conditions: NetworkConditions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResultChunk>> + Send>>> {
        if !self.config.enable_adaptive_streaming {
            return self.create_stream(result, format).await;
        }

        let stream_id = uuid::Uuid::new_v4().to_string();

        // Adjust configuration based on network conditions
        let adaptive_config = self.adapt_config_for_network(&network_conditions);

        // Initialize stream metrics
        let metrics = StreamMetrics {
            stream_id: stream_id.clone(),
            start_time: Instant::now(),
            chunks_sent: 0,
            bytes_sent: 0,
            last_chunk_time: Instant::now(),
            is_completed: false,
            error_count: 0,
        };

        self.active_streams
            .write()
            .await
            .insert(stream_id.clone(), metrics);

        // Serialize with adaptive configuration
        let serialized_data = self.serialize_result(&result, &format).await?;

        // Create adaptive chunk stream
        let chunk_stream = self
            .create_adaptive_chunk_stream(stream_id, serialized_data, format, adaptive_config)
            .await?;

        Ok(chunk_stream)
    }

    /// Get compression statistics
    pub async fn get_compression_stats(&self) -> CompressionStatistics {
        self.compression_stats.read().await.clone()
    }

    /// Get active stream metrics
    pub async fn get_stream_metrics(&self) -> HashMap<String, StreamMetrics> {
        self.active_streams.read().await.clone()
    }

    /// Cleanup completed streams
    pub async fn cleanup_completed_streams(&self) -> usize {
        let mut streams = self.active_streams.write().await;
        let initial_count = streams.len();

        streams.retain(|_, metrics| {
            !metrics.is_completed && metrics.start_time.elapsed() < Duration::from_secs(3600)
        });

        initial_count - streams.len()
    }

    // Private helper methods

    async fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.config.compression_algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => {
                let mut encoder = GzEncoder::new(
                    Vec::new(),
                    GzCompression::new(self.config.compression_level),
                );
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
            CompressionAlgorithm::Deflate => {
                // Simplified deflate implementation using flate2
                let mut encoder = flate2::write::DeflateEncoder::new(
                    Vec::new(),
                    flate2::Compression::new(self.config.compression_level),
                );
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
            CompressionAlgorithm::Brotli => {
                // Placeholder for brotli compression
                // In a real implementation, you'd use the brotli crate
                warn!("Brotli compression not implemented, falling back to gzip");
                let mut encoder = GzEncoder::new(
                    Vec::new(),
                    GzCompression::new(self.config.compression_level),
                );
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
            CompressionAlgorithm::Lz4 => {
                // Placeholder for LZ4 compression
                // In a real implementation, you'd use the lz4 crate
                warn!("LZ4 compression not implemented, falling back to gzip");
                let mut encoder = GzEncoder::new(
                    Vec::new(),
                    GzCompression::new(self.config.compression_level),
                );
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
            CompressionAlgorithm::Zstd => {
                // Placeholder for Zstandard compression
                // In a real implementation, you'd use the zstd crate
                warn!("Zstd compression not implemented, falling back to gzip");
                let mut encoder = GzEncoder::new(
                    Vec::new(),
                    GzCompression::new(self.config.compression_level),
                );
                encoder.write_all(data)?;
                Ok(encoder.finish()?)
            }
        }
    }

    async fn decompress_data(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
    ) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::None => Ok(data.to_vec()),
            CompressionAlgorithm::Gzip => {
                use flate2::read::GzDecoder;
                use std::io::Read;

                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
            CompressionAlgorithm::Deflate => {
                use flate2::read::DeflateDecoder;
                use std::io::Read;

                let mut decoder = DeflateDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
            _ => {
                // For unsupported algorithms, try gzip as fallback
                warn!(
                    "Decompression algorithm {:?} not fully implemented, trying gzip",
                    algorithm
                );
                use flate2::read::GzDecoder;
                use std::io::Read;

                let mut decoder = GzDecoder::new(data);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                Ok(decompressed)
            }
        }
    }

    async fn serialize_result(
        &self,
        result: &QueryResult,
        format: &ResultFormat,
    ) -> Result<Vec<u8>> {
        match format {
            ResultFormat::Json => {
                serde_json::to_vec(result).map_err(|e| anyhow!("JSON serialization failed: {}", e))
            }
            ResultFormat::MessagePack => {
                // Placeholder for MessagePack serialization
                warn!("MessagePack serialization not implemented, falling back to JSON");
                serde_json::to_vec(result).map_err(|e| anyhow!("JSON serialization failed: {}", e))
            }
            ResultFormat::Csv => {
                // Simple CSV conversion for SPARQL results
                match result {
                    QueryResult::Sparql(sparql_results) => {
                        self.sparql_bindings_to_csv(sparql_results).await
                    }
                    QueryResult::GraphQL(_) => {
                        warn!("CSV format not suitable for GraphQL results, falling back to JSON");
                        serde_json::to_vec(result)
                            .map_err(|e| anyhow!("JSON serialization failed: {}", e))
                    }
                }
            }
            _ => {
                warn!(
                    "Serialization format {:?} not implemented, falling back to JSON",
                    format
                );
                serde_json::to_vec(result).map_err(|e| anyhow!("JSON serialization failed: {}", e))
            }
        }
    }

    async fn sparql_bindings_to_csv(
        &self,
        bindings: &Vec<HashMap<String, oxirs_core::Term>>,
    ) -> Result<Vec<u8>> {
        let mut csv_data = Vec::new();

        if bindings.is_empty() {
            return Ok(csv_data);
        }

        // Extract variable names from the first binding
        let vars: Vec<String> = bindings[0].keys().cloned().collect();

        // Write header
        let header = vars.join(",");
        csv_data.extend_from_slice(header.as_bytes());
        csv_data.push(b'\n');

        // Write data rows
        for binding in bindings {
            let row: Vec<String> = vars
                .iter()
                .map(|var| {
                    binding
                        .get(var)
                        .map(|term| format!("\"{}\"", term.to_string().replace('"', "\"\"")))
                        .unwrap_or_default()
                })
                .collect();
            csv_data.extend_from_slice(row.join(",").as_bytes());
            csv_data.push(b'\n');
        }

        Ok(csv_data)
    }

    async fn sparql_to_csv(&self, results: &SparqlResults) -> Result<Vec<u8>> {
        let mut csv_data = Vec::new();

        // Write header
        let header = results.head.vars.join(",");
        csv_data.extend_from_slice(header.as_bytes());
        csv_data.push(b'\n');

        // Write data rows
        for binding in &results.results.bindings {
            let row: Vec<String> = results
                .head
                .vars
                .iter()
                .map(|var| {
                    binding
                        .get(var)
                        .map(|sparql_value| {
                            format!("\"{}\"", sparql_value.value.replace('"', "\"\""))
                        })
                        .unwrap_or_default()
                })
                .collect();
            csv_data.extend_from_slice(row.join(",").as_bytes());
            csv_data.push(b'\n');
        }

        Ok(csv_data)
    }

    async fn create_chunk_stream(
        &self,
        stream_id: String,
        data: Vec<u8>,
        format: ResultFormat,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResultChunk>> + Send>>> {
        let chunk_size = self.config.chunk_size;
        let total_chunks = (data.len() + chunk_size - 1) / chunk_size;

        let (tx, rx) = mpsc::channel(32);
        let compression_enabled = self.config.enable_compression;
        let compression_algorithm = self.config.compression_algorithm;
        let compression_level = self.config.compression_level;
        let active_streams = self.active_streams.clone();

        tokio::spawn(async move {
            for (chunk_idx, chunk_data) in data.chunks(chunk_size).enumerate() {
                let sequence = chunk_idx as u64;
                let is_final = chunk_idx == total_chunks - 1;

                // Optionally compress chunk
                let (final_data, compression_info) = if compression_enabled
                    && chunk_data.len() > 512
                {
                    // Only compress chunks larger than 512 bytes
                    match Self::compress_chunk(chunk_data, compression_algorithm, compression_level)
                        .await
                    {
                        Ok((compressed, info)) => (compressed, Some(info)),
                        Err(_) => (Bytes::copy_from_slice(chunk_data), None),
                    }
                } else {
                    (Bytes::copy_from_slice(chunk_data), None)
                };

                let chunk = ResultChunk {
                    sequence,
                    data: final_data.clone(),
                    is_final,
                    metadata: ChunkMetadata {
                        total_chunks: Some(total_chunks as u64),
                        compression: compression_info,
                        format: format.clone(),
                        estimated_total_size: Some(data.len()),
                    },
                };

                // Update stream metrics
                if let Ok(mut streams) = active_streams.try_write() {
                    if let Some(metrics) = streams.get_mut(&stream_id) {
                        metrics.chunks_sent += 1;
                        metrics.bytes_sent += final_data.len() as u64;
                        metrics.last_chunk_time = Instant::now();
                        metrics.is_completed = is_final;
                    }
                }

                if tx.send(Ok(chunk)).await.is_err() {
                    break; // Receiver dropped
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    async fn create_adaptive_chunk_stream(
        &self,
        stream_id: String,
        data: Vec<u8>,
        format: ResultFormat,
        adaptive_config: AdaptiveConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResultChunk>> + Send>>> {
        let (tx, rx) = mpsc::channel(32);
        let active_streams = self.active_streams.clone();

        tokio::spawn(async move {
            let mut current_chunk_size = adaptive_config.initial_chunk_size;
            let mut chunk_idx = 0;
            let mut data_offset = 0;

            while data_offset < data.len() {
                let chunk_end = (data_offset + current_chunk_size).min(data.len());
                let chunk_data = &data[data_offset..chunk_end];
                let is_final = chunk_end == data.len();

                let chunk = ResultChunk {
                    sequence: chunk_idx,
                    data: Bytes::copy_from_slice(chunk_data),
                    is_final,
                    metadata: ChunkMetadata {
                        total_chunks: None, // Unknown for adaptive streaming
                        compression: None,
                        format: format.clone(),
                        estimated_total_size: Some(data.len()),
                    },
                };

                // Update stream metrics
                if let Ok(mut streams) = active_streams.try_write() {
                    if let Some(metrics) = streams.get_mut(&stream_id) {
                        metrics.chunks_sent += 1;
                        metrics.bytes_sent += chunk_data.len() as u64;
                        metrics.last_chunk_time = Instant::now();
                        metrics.is_completed = is_final;
                    }
                }

                if tx.send(Ok(chunk)).await.is_err() {
                    break; // Receiver dropped
                }

                data_offset = chunk_end;
                chunk_idx += 1;

                // Adapt chunk size based on performance
                current_chunk_size =
                    Self::adapt_chunk_size(current_chunk_size, &adaptive_config, chunk_idx);
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    async fn compress_chunk(
        data: &[u8],
        algorithm: CompressionAlgorithm,
        level: u32,
    ) -> Result<(Bytes, CompressionInfo)> {
        let start_time = Instant::now();

        let compressed = match algorithm {
            CompressionAlgorithm::Gzip => {
                let mut encoder = GzEncoder::new(Vec::new(), GzCompression::new(level));
                encoder.write_all(data)?;
                encoder.finish()?
            }
            _ => {
                // Fallback to gzip for unsupported algorithms
                let mut encoder = GzEncoder::new(Vec::new(), GzCompression::new(level));
                encoder.write_all(data)?;
                encoder.finish()?
            }
        };

        let compression_info = CompressionInfo {
            algorithm,
            level,
            original_size: data.len(),
            compressed_size: compressed.len(),
        };

        Ok((Bytes::from(compressed), compression_info))
    }

    fn adapt_chunk_size(current_size: usize, config: &AdaptiveConfig, chunk_count: u64) -> usize {
        // Simple adaptation strategy - increase chunk size over time for better throughput
        let growth_factor = 1.0 + (chunk_count as f64 * config.growth_rate).min(config.max_growth);
        let new_size = (current_size as f64 * growth_factor) as usize;
        new_size
            .min(config.max_chunk_size)
            .max(config.min_chunk_size)
    }

    fn adapt_config_for_network(&self, conditions: &NetworkConditions) -> AdaptiveConfig {
        let base_chunk_size = if conditions.bandwidth_mbps > 100.0 {
            32768 // 32KB for high bandwidth
        } else if conditions.bandwidth_mbps > 10.0 {
            16384 // 16KB for medium bandwidth
        } else {
            4096 // 4KB for low bandwidth
        };

        AdaptiveConfig {
            initial_chunk_size: base_chunk_size,
            min_chunk_size: 1024,
            max_chunk_size: base_chunk_size * 4,
            growth_rate: if conditions.latency_ms < 50.0 {
                0.1
            } else {
                0.05
            },
            max_growth: 2.0,
        }
    }

    async fn update_compression_stats(
        &self,
        algorithm: CompressionAlgorithm,
        original_size: usize,
        compressed_size: usize,
        compression_time: Duration,
    ) {
        let mut stats = self.compression_stats.write().await;

        stats.total_compressions += 1;
        stats.total_original_bytes += original_size as u64;
        stats.total_compressed_bytes += compressed_size as u64;
        stats.total_compression_time += compression_time;

        let ratio = compressed_size as f64 / original_size as f64;
        stats.average_compression_ratio =
            (stats.average_compression_ratio * (stats.total_compressions - 1) as f64 + ratio)
                / stats.total_compressions as f64;

        // Update algorithm-specific stats
        let algo_stats = stats.algorithm_stats.entry(algorithm).or_default();
        algo_stats.uses += 1;
        algo_stats.total_original_bytes += original_size as u64;
        algo_stats.total_compressed_bytes += compressed_size as u64;
        algo_stats.total_compression_time += compression_time;
        algo_stats.average_compression_ratio =
            (algo_stats.average_compression_ratio * (algo_stats.uses - 1) as f64 + ratio)
                / algo_stats.uses as f64;
    }
}

impl Default for ResultStreamingManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Network conditions for adaptive streaming
#[derive(Debug, Clone)]
pub struct NetworkConditions {
    pub bandwidth_mbps: f64,
    pub latency_ms: f64,
    pub packet_loss_percent: f64,
    pub jitter_ms: f64,
}

/// Cursor-based pagination for large result sets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationCursor {
    /// Unique cursor identifier
    pub cursor_id: String,
    /// Current offset in the result set
    pub offset: usize,
    /// Page size
    pub page_size: usize,
    /// Total estimated result count (if known)
    pub total_count: Option<usize>,
    /// Encoded position state for resuming
    pub position_state: String,
    /// Cursor creation timestamp
    pub created_at: u64,
    /// Cursor expiration timestamp
    pub expires_at: u64,
}

/// Paginated query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResult {
    /// Current page data
    pub data: QueryResult,
    /// Pagination information
    pub pagination: PaginationInfo,
}

/// Pagination metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationInfo {
    /// Current cursor
    pub current_cursor: Option<PaginationCursor>,
    /// Next page cursor (if available)
    pub next_cursor: Option<PaginationCursor>,
    /// Previous page cursor (if available)
    pub previous_cursor: Option<PaginationCursor>,
    /// Whether there are more pages
    pub has_next_page: bool,
    /// Whether there are previous pages
    pub has_previous_page: bool,
    /// Current page number (1-based)
    pub page_number: usize,
    /// Total pages (if known)
    pub total_pages: Option<usize>,
    /// Items per page
    pub page_size: usize,
    /// Total items (if known)
    pub total_items: Option<usize>,
}

/// Pagination configuration
#[derive(Debug, Clone)]
pub struct PaginationConfig {
    /// Default page size
    pub default_page_size: usize,
    /// Maximum page size allowed
    pub max_page_size: usize,
    /// Minimum page size allowed
    pub min_page_size: usize,
    /// Cursor expiration time
    pub cursor_ttl: Duration,
    /// Enable cursor caching
    pub enable_cursor_caching: bool,
    /// Maximum number of cached cursors
    pub max_cached_cursors: usize,
}

impl Default for PaginationConfig {
    fn default() -> Self {
        Self {
            default_page_size: 100,
            max_page_size: 10000,
            min_page_size: 1,
            cursor_ttl: Duration::from_secs(3600), // 1 hour
            enable_cursor_caching: true,
            max_cached_cursors: 1000,
        }
    }
}

/// Cursor-based pagination manager
#[derive(Debug)]
pub struct PaginationManager {
    config: PaginationConfig,
    active_cursors: Arc<RwLock<HashMap<String, PaginationCursor>>>,
    cached_results: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl PaginationManager {
    /// Create a new pagination manager
    pub fn new() -> Self {
        Self::with_config(PaginationConfig::default())
    }

    /// Create a new pagination manager with configuration
    pub fn with_config(config: PaginationConfig) -> Self {
        Self {
            config,
            active_cursors: Arc::new(RwLock::new(HashMap::new())),
            cached_results: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a paginated result from a large dataset
    pub async fn paginate_result(
        &self,
        result: QueryResult,
        page_size: Option<usize>,
        cursor: Option<PaginationCursor>,
    ) -> Result<PaginatedResult> {
        let effective_page_size = page_size
            .unwrap_or(self.config.default_page_size)
            .clamp(self.config.min_page_size, self.config.max_page_size);

        let starting_offset = cursor.as_ref().map(|c| c.offset).unwrap_or(0);

        let (paginated_data, total_count) = self
            .extract_page(&result, starting_offset, effective_page_size)
            .await?;

        let current_page = (starting_offset / effective_page_size) + 1;
        let total_pages =
            total_count.map(|count| (count + effective_page_size - 1) / effective_page_size);

        let has_next_page = starting_offset + effective_page_size
            < total_count.unwrap_or(starting_offset + effective_page_size + 1);
        let has_previous_page = starting_offset > 0;

        // Generate cursors
        let current_cursor = self
            .create_cursor(starting_offset, effective_page_size, total_count)
            .await?;

        let next_cursor = if has_next_page {
            Some(
                self.create_cursor(
                    starting_offset + effective_page_size,
                    effective_page_size,
                    total_count,
                )
                .await?,
            )
        } else {
            None
        };

        let previous_cursor = if has_previous_page && starting_offset >= effective_page_size {
            Some(
                self.create_cursor(
                    starting_offset - effective_page_size,
                    effective_page_size,
                    total_count,
                )
                .await?,
            )
        } else {
            None
        };

        let pagination_info = PaginationInfo {
            current_cursor: Some(current_cursor),
            next_cursor,
            previous_cursor,
            has_next_page,
            has_previous_page,
            page_number: current_page,
            total_pages,
            page_size: effective_page_size,
            total_items: total_count,
        };

        Ok(PaginatedResult {
            data: paginated_data,
            pagination: pagination_info,
        })
    }

    /// Create a cursor for streaming large result sets
    pub async fn create_streaming_cursor(
        &self,
        result_id: String,
        page_size: usize,
        total_count: Option<usize>,
    ) -> Result<PaginationCursor> {
        let cursor = self.create_cursor(0, page_size, total_count).await?;

        // Store cursor for later retrieval
        self.store_cursor(cursor.clone()).await?;

        Ok(cursor)
    }

    /// Get the next page using a cursor
    pub async fn get_next_page(
        &self,
        cursor: &PaginationCursor,
        result_provider: impl Fn(usize, usize) -> Result<QueryResult>,
    ) -> Result<PaginatedResult> {
        self.validate_cursor(cursor).await?;

        let next_offset = cursor.offset + cursor.page_size;
        let result = result_provider(next_offset, cursor.page_size)?;

        self.paginate_result(result, Some(cursor.page_size), Some(cursor.clone()))
            .await
    }

    /// Clean up expired cursors
    pub async fn cleanup_expired_cursors(&self) -> usize {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut cursors = self.active_cursors.write().await;
        let initial_count = cursors.len();

        cursors.retain(|_, cursor| cursor.expires_at > current_time);

        let removed_count = initial_count - cursors.len();

        // Also cleanup cached results for removed cursors
        if removed_count > 0 {
            let mut cache = self.cached_results.write().await;
            cache.retain(|cursor_id, _| cursors.contains_key(cursor_id));
        }

        removed_count
    }

    /// Get cursor statistics
    pub async fn get_cursor_statistics(&self) -> CursorStatistics {
        let cursors = self.active_cursors.read().await;
        let cache = self.cached_results.read().await;

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let expired_count = cursors
            .values()
            .filter(|cursor| cursor.expires_at <= current_time)
            .count();

        CursorStatistics {
            active_cursors: cursors.len(),
            expired_cursors: expired_count,
            cached_results: cache.len(),
            total_cache_size_bytes: cache.values().map(|data| data.len()).sum(),
        }
    }

    // Private helper methods

    async fn extract_page(
        &self,
        result: &QueryResult,
        offset: usize,
        page_size: usize,
    ) -> Result<(QueryResult, Option<usize>)> {
        match result {
            QueryResult::Sparql(sparql_results) => {
                let total_count = sparql_results.results.bindings.len();
                let end_idx = (offset + page_size).min(total_count);

                if offset >= total_count {
                    // Return empty result
                    let empty_result = QueryResult::Sparql(SparqlResults {
                        head: sparql_results.head.clone(),
                        results: crate::executor::SparqlResultsData { bindings: vec![] },
                    });
                    return Ok((empty_result, Some(total_count)));
                }

                let page_bindings = sparql_results.results.bindings[offset..end_idx].to_vec();

                let paginated_result = QueryResult::Sparql(SparqlResults {
                    head: sparql_results.head.clone(),
                    results: crate::executor::SparqlResultsData {
                        bindings: page_bindings,
                    },
                });

                Ok((paginated_result, Some(total_count)))
            }
            QueryResult::GraphQL(graphql_response) => {
                // For GraphQL, we'll treat the entire response as one page
                // In a real implementation, you'd need to parse the GraphQL response
                // and implement field-level pagination
                Ok((result.clone(), None))
            }
        }
    }

    async fn create_cursor(
        &self,
        offset: usize,
        page_size: usize,
        total_count: Option<usize>,
    ) -> Result<PaginationCursor> {
        let cursor_id = uuid::Uuid::new_v4().to_string();
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let position_state = base64::encode(format!("{}:{}", offset, page_size));

        let cursor = PaginationCursor {
            cursor_id,
            offset,
            page_size,
            total_count,
            position_state,
            created_at: current_time,
            expires_at: current_time + self.config.cursor_ttl.as_secs(),
        };

        Ok(cursor)
    }

    async fn store_cursor(&self, cursor: PaginationCursor) -> Result<()> {
        let mut cursors = self.active_cursors.write().await;

        // Clean up old cursors if we're at capacity
        if cursors.len() >= self.config.max_cached_cursors {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Remove expired cursors first
            cursors.retain(|_, c| c.expires_at > current_time);

            // If still at capacity, remove oldest cursors
            if cursors.len() >= self.config.max_cached_cursors {
                let mut cursor_vec: Vec<_> = cursors.drain().collect();
                cursor_vec.sort_by_key(|(_, c)| c.created_at);
                cursor_vec.truncate(self.config.max_cached_cursors - 1);

                for (id, c) in cursor_vec {
                    cursors.insert(id, c);
                }
            }
        }

        cursors.insert(cursor.cursor_id.clone(), cursor);
        Ok(())
    }

    async fn validate_cursor(&self, cursor: &PaginationCursor) -> Result<()> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if cursor.expires_at <= current_time {
            return Err(anyhow!("Cursor has expired"));
        }

        // Validate position state
        let decoded_state = base64::decode(&cursor.position_state)
            .map_err(|_| anyhow!("Invalid cursor position state"))?;

        let state_str = String::from_utf8(decoded_state)
            .map_err(|_| anyhow!("Invalid cursor position state encoding"))?;

        let parts: Vec<&str> = state_str.split(':').collect();
        if parts.len() != 2 {
            return Err(anyhow!("Invalid cursor position state format"));
        }

        let offset: usize = parts[0]
            .parse()
            .map_err(|_| anyhow!("Invalid cursor offset"))?;
        let page_size: usize = parts[1]
            .parse()
            .map_err(|_| anyhow!("Invalid cursor page size"))?;

        if offset != cursor.offset || page_size != cursor.page_size {
            return Err(anyhow!("Cursor state mismatch"));
        }

        Ok(())
    }
}

impl Default for PaginationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for cursor management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorStatistics {
    pub active_cursors: usize,
    pub expired_cursors: usize,
    pub cached_results: usize,
    pub total_cache_size_bytes: usize,
}

/// Adaptive streaming configuration
#[derive(Debug, Clone)]
struct AdaptiveConfig {
    initial_chunk_size: usize,
    min_chunk_size: usize,
    max_chunk_size: usize,
    growth_rate: f64,
    max_growth: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compression() {
        let manager = ResultStreamingManager::new();
        let test_data = b"Hello, World! This is a test string for compression.".repeat(100);

        let compressed = manager.compress_result(&test_data).await.unwrap();
        assert!(compressed.compressed_size < compressed.original_size);
        assert!(compressed.compression_ratio < 1.0);

        let decompressed = manager.decompress_result(&compressed).await.unwrap();
        assert_eq!(decompressed.as_ref(), test_data.as_slice());
    }

    #[tokio::test]
    async fn test_streaming() {
        let manager = ResultStreamingManager::new();

        use crate::executor::{SparqlHead, SparqlResultsData};

        let test_result = QueryResult::Sparql(SparqlResults {
            head: SparqlHead { vars: vec![] },
            results: SparqlResultsData { bindings: vec![] },
        });

        let mut stream = manager
            .create_stream(test_result, ResultFormat::Json)
            .await
            .unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk_result) = stream.next().await {
            chunks.push(chunk_result.unwrap());
        }

        assert!(!chunks.is_empty());
        assert!(chunks.last().unwrap().is_final);
    }

    #[tokio::test]
    async fn test_csv_serialization() {
        let manager = ResultStreamingManager::new();

        use crate::executor::{SparqlHead, SparqlResultsData, SparqlValue};

        let sparql_result = SparqlResults {
            head: SparqlHead {
                vars: vec!["name".to_string(), "age".to_string()],
            },
            results: SparqlResultsData {
                bindings: vec![{
                    let mut binding = HashMap::new();
                    binding.insert(
                        "name".to_string(),
                        SparqlValue {
                            value_type: "uri".to_string(),
                            value: "http://example.org/john".to_string(),
                            datatype: None,
                            lang: None,
                        },
                    );
                    binding.insert(
                        "age".to_string(),
                        SparqlValue {
                            value_type: "literal".to_string(),
                            value: "25".to_string(),
                            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
                            lang: None,
                        },
                    );
                    binding
                }],
            },
        };

        let csv_data = manager.sparql_to_csv(&sparql_result).await.unwrap();
        let csv_string = String::from_utf8(csv_data).unwrap();

        assert!(csv_string.contains("name,age"));
        assert!(csv_string.contains("john"));
        assert!(csv_string.contains("25"));
    }

    #[tokio::test]
    async fn test_adaptive_streaming() {
        let manager = ResultStreamingManager::new();

        use crate::executor::{SparqlHead, SparqlResultsData};

        let test_result = QueryResult::Sparql(SparqlResults {
            head: SparqlHead { vars: vec![] },
            results: SparqlResultsData { bindings: vec![] },
        });

        let network_conditions = NetworkConditions {
            bandwidth_mbps: 50.0,
            latency_ms: 20.0,
            packet_loss_percent: 0.1,
            jitter_ms: 5.0,
        };

        let mut stream = manager
            .create_adaptive_stream(test_result, ResultFormat::Json, network_conditions)
            .await
            .unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk_result) = stream.next().await {
            chunks.push(chunk_result.unwrap());
        }

        assert!(!chunks.is_empty());
        assert!(chunks.last().unwrap().is_final);
    }

    #[test]
    fn test_compression_config() {
        let config = StreamingConfig::default();
        assert!(config.enable_compression);
        assert_eq!(config.compression_algorithm, CompressionAlgorithm::Gzip);
        assert_eq!(config.compression_level, 6);
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        let manager = ResultStreamingManager::new();
        let test_data = b"Test data for statistics".repeat(50);

        let _compressed = manager.compress_result(&test_data).await.unwrap();

        let stats = manager.get_compression_stats().await;
        assert_eq!(stats.total_compressions, 1);
        assert!(stats.total_original_bytes > 0);
        assert!(stats.total_compressed_bytes > 0);
    }
}
