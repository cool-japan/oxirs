//! # Network Optimization for Federated Queries
//!
//! This module implements advanced network optimization techniques including
//! compression, encoding, and bandwidth optimization for federated query processing.

use anyhow::{anyhow, Result};
use brotli::{enc::BrotliEncoderParams, CompressorWriter, Decompressor};
use bytes::Bytes;
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use rmp_serde::to_vec;
use serde::{Deserialize, Serialize};
use serde_cbor;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::debug;
use zstd::{decode_all, encode_all};

use crate::FederatedService;

/// Network optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimizerConfig {
    /// Enable compression for large results
    pub enable_compression: bool,
    /// Minimum result size for compression (bytes)
    pub compression_threshold: usize,
    /// Preferred compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Enable binary encoding for structured data
    pub enable_binary_encoding: bool,
    /// Enable selective compression based on content type
    pub enable_selective_compression: bool,
    /// Bandwidth usage optimization level
    pub bandwidth_optimization_level: BandwidthOptimizationLevel,
    /// Enable adaptive compression based on network conditions
    pub enable_adaptive_compression: bool,
    /// Network performance monitoring interval
    pub monitoring_interval: Duration,
}

impl Default for NetworkOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_threshold: 1024, // 1KB
            compression_algorithm: CompressionAlgorithm::Gzip,
            enable_binary_encoding: true,
            enable_selective_compression: true,
            bandwidth_optimization_level: BandwidthOptimizationLevel::High,
            enable_adaptive_compression: true,
            monitoring_interval: Duration::from_secs(10),
        }
    }
}

/// Supported compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Gzip compression (good balance of speed and compression)
    Gzip,
    /// Brotli compression (better compression ratio, slower)
    Brotli,
    /// LZ4 compression (very fast, lower compression ratio)
    Lz4,
    /// Zstd compression (good balance, newer algorithm)
    Zstd,
    /// No compression
    None,
}

/// Bandwidth optimization levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BandwidthOptimizationLevel {
    /// Minimal optimization for low-latency networks
    Low,
    /// Moderate optimization for typical networks
    Medium,
    /// Aggressive optimization for high-latency networks
    High,
    /// Maximum optimization for constrained networks
    Maximum,
}

/// Network optimization statistics
#[derive(Debug, Clone)]
pub struct NetworkOptimizationStats {
    pub total_requests: u64,
    pub compressed_requests: u64,
    pub total_bytes_original: u64,
    pub total_bytes_compressed: u64,
    pub average_compression_ratio: f64,
    pub compression_time_ms: f64,
    pub decompression_time_ms: f64,
    pub bandwidth_saved_bytes: u64,
}

impl Default for NetworkOptimizationStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            compressed_requests: 0,
            total_bytes_original: 0,
            total_bytes_compressed: 0,
            average_compression_ratio: 1.0,
            compression_time_ms: 0.0,
            decompression_time_ms: 0.0,
            bandwidth_saved_bytes: 0,
        }
    }
}

/// Network performance metrics
#[derive(Debug, Clone)]
pub struct NetworkPerformanceMetrics {
    pub latency_ms: f64,
    pub bandwidth_mbps: f64,
    pub packet_loss_rate: f64,
    pub jitter_ms: f64,
    pub connection_quality: ConnectionQuality,
}

/// Connection quality assessment
#[derive(Debug, Clone)]
pub enum ConnectionQuality {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Result encoding format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodingFormat {
    /// JSON text encoding
    Json,
    /// MessagePack binary encoding
    MessagePack,
    /// Protocol Buffers binary encoding
    ProtocolBuffers,
    /// Apache Avro binary encoding
    Avro,
    /// CBOR binary encoding
    Cbor,
}

/// Compressed data container
#[derive(Debug, Clone)]
pub struct CompressedData {
    pub data: Bytes,
    pub algorithm: CompressionAlgorithm,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub encoding_format: EncodingFormat,
}

/// Network optimizer for federated queries
pub struct NetworkOptimizer {
    config: NetworkOptimizerConfig,
    stats: Arc<RwLock<NetworkOptimizationStats>>,
    performance_metrics: Arc<RwLock<HashMap<String, NetworkPerformanceMetrics>>>,
    compression_cache: Arc<RwLock<HashMap<u64, CompressedData>>>,
}

impl Default for NetworkOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkOptimizer {
    /// Create a new network optimizer
    pub fn new() -> Self {
        Self::with_config(NetworkOptimizerConfig::default())
    }

    /// Create a new network optimizer with configuration
    pub fn with_config(config: NetworkOptimizerConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(NetworkOptimizationStats::default())),
            performance_metrics: Arc::new(RwLock::new(HashMap::new())),
            compression_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Compress data using the configured algorithm
    pub async fn compress_data(
        &self,
        data: &[u8],
        encoding_format: EncodingFormat,
    ) -> Result<CompressedData> {
        let start_time = Instant::now();
        let original_size = data.len();

        // Check if compression is beneficial
        if !self.config.enable_compression || original_size < self.config.compression_threshold {
            return Ok(CompressedData {
                data: Bytes::copy_from_slice(data),
                algorithm: CompressionAlgorithm::None,
                original_size,
                compressed_size: original_size,
                compression_ratio: 1.0,
                encoding_format,
            });
        }

        let compressed_data = match self.config.compression_algorithm {
            CompressionAlgorithm::Gzip => self.compress_gzip(data).await?,
            CompressionAlgorithm::Brotli => self.compress_brotli(data).await?,
            CompressionAlgorithm::Lz4 => self.compress_lz4(data).await?,
            CompressionAlgorithm::Zstd => self.compress_zstd(data).await?,
            CompressionAlgorithm::None => Bytes::copy_from_slice(data),
        };

        let compressed_size = compressed_data.len();
        let compression_ratio = original_size as f64 / compressed_size as f64;
        let compression_time = start_time.elapsed().as_millis() as f64;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        if compressed_size < original_size {
            stats.compressed_requests += 1;
            stats.bandwidth_saved_bytes += (original_size - compressed_size) as u64;
        }
        stats.total_bytes_original += original_size as u64;
        stats.total_bytes_compressed += compressed_size as u64;
        stats.compression_time_ms = (stats.compression_time_ms * (stats.total_requests - 1) as f64
            + compression_time)
            / stats.total_requests as f64;
        stats.average_compression_ratio =
            stats.total_bytes_original as f64 / stats.total_bytes_compressed as f64;

        Ok(CompressedData {
            data: compressed_data,
            algorithm: self.config.compression_algorithm.clone(),
            original_size,
            compressed_size,
            compression_ratio,
            encoding_format,
        })
    }

    /// Decompress data
    pub async fn decompress_data(&self, compressed: &CompressedData) -> Result<Bytes> {
        if matches!(compressed.algorithm, CompressionAlgorithm::None) {
            return Ok(compressed.data.clone());
        }

        let start_time = Instant::now();
        let decompressed = match compressed.algorithm {
            CompressionAlgorithm::Gzip => self.decompress_gzip(&compressed.data).await?,
            CompressionAlgorithm::Brotli => self.decompress_brotli(&compressed.data).await?,
            CompressionAlgorithm::Lz4 => self.decompress_lz4(&compressed.data).await?,
            CompressionAlgorithm::Zstd => self.decompress_zstd(&compressed.data).await?,
            CompressionAlgorithm::None => compressed.data.clone(),
        };

        let decompression_time = start_time.elapsed().as_millis() as f64;

        // Update decompression statistics
        let mut stats = self.stats.write().await;
        stats.decompression_time_ms =
            (stats.decompression_time_ms * (stats.total_requests - 1) as f64 + decompression_time)
                / stats.total_requests as f64;

        Ok(decompressed)
    }

    /// Choose optimal encoding format based on data characteristics
    pub async fn choose_optimal_encoding(
        &self,
        data_type: &str,
        data_size: usize,
        network_metrics: &NetworkPerformanceMetrics,
    ) -> EncodingFormat {
        match self.config.bandwidth_optimization_level {
            BandwidthOptimizationLevel::Low => EncodingFormat::Json,
            BandwidthOptimizationLevel::Medium => {
                if data_size > 10000 {
                    EncodingFormat::MessagePack
                } else {
                    EncodingFormat::Json
                }
            }
            BandwidthOptimizationLevel::High => match data_type {
                "sparql_results" => EncodingFormat::MessagePack,
                "graphql_response" => EncodingFormat::Cbor,
                "schema_data" => EncodingFormat::ProtocolBuffers,
                _ => EncodingFormat::MessagePack,
            },
            BandwidthOptimizationLevel::Maximum => {
                if matches!(
                    network_metrics.connection_quality,
                    ConnectionQuality::Poor | ConnectionQuality::Critical
                ) {
                    EncodingFormat::ProtocolBuffers
                } else {
                    EncodingFormat::MessagePack
                }
            }
        }
    }

    /// Choose optimal compression algorithm based on network conditions
    pub async fn choose_optimal_compression(
        &self,
        data_size: usize,
        network_metrics: &NetworkPerformanceMetrics,
    ) -> CompressionAlgorithm {
        if !self.config.enable_adaptive_compression {
            return self.config.compression_algorithm.clone();
        }

        match network_metrics.connection_quality {
            ConnectionQuality::Excellent => {
                if data_size > 1_000_000 {
                    CompressionAlgorithm::Brotli // Best compression for large data
                } else {
                    CompressionAlgorithm::Gzip // Good balance
                }
            }
            ConnectionQuality::Good => CompressionAlgorithm::Gzip,
            ConnectionQuality::Fair => {
                if network_metrics.latency_ms > 100.0 {
                    CompressionAlgorithm::Gzip // Worth the compression time
                } else {
                    CompressionAlgorithm::Lz4 // Favor speed
                }
            }
            ConnectionQuality::Poor => CompressionAlgorithm::Brotli, // Maximize compression
            ConnectionQuality::Critical => CompressionAlgorithm::Brotli, // Every byte counts
        }
    }

    /// Monitor network performance for a service
    pub async fn monitor_network_performance(
        &self,
        service_id: &str,
        service: &FederatedService,
    ) -> Result<NetworkPerformanceMetrics> {
        let start_time = Instant::now();

        // Perform lightweight network test (ping-like)
        let latency = self.measure_latency(service).await?;
        let bandwidth = self.estimate_bandwidth(service).await?;
        let packet_loss = self.estimate_packet_loss(service).await?;
        let jitter = self.measure_jitter(service).await?;

        let quality = self.assess_connection_quality(latency, bandwidth, packet_loss, jitter);

        let metrics = NetworkPerformanceMetrics {
            latency_ms: latency,
            bandwidth_mbps: bandwidth,
            packet_loss_rate: packet_loss,
            jitter_ms: jitter,
            connection_quality: quality,
        };

        // Cache metrics
        let mut performance_metrics = self.performance_metrics.write().await;
        performance_metrics.insert(service_id.to_string(), metrics.clone());

        Ok(metrics)
    }

    /// Apply selective compression based on content type and size
    pub async fn apply_selective_compression(
        &self,
        data: &[u8],
        content_type: &str,
        service_metrics: &NetworkPerformanceMetrics,
    ) -> Result<CompressedData> {
        if !self.config.enable_selective_compression {
            return self.compress_data(data, EncodingFormat::Json).await;
        }

        let encoding_format = self
            .choose_optimal_encoding(content_type, data.len(), service_metrics)
            .await;

        // Encode data in optimal format first
        let encoded_data = self.encode_data(data, &encoding_format).await?;

        // Then apply compression
        self.compress_data(&encoded_data, encoding_format).await
    }

    /// Get current optimization statistics
    pub async fn get_statistics(&self) -> NetworkOptimizationStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_statistics(&self) {
        let mut stats = self.stats.write().await;
        *stats = NetworkOptimizationStats::default();
    }

    // Private helper methods

    async fn compress_gzip(&self, data: &[u8]) -> Result<Bytes> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        Ok(Bytes::from(compressed))
    }

    async fn decompress_gzip(&self, data: &[u8]) -> Result<Bytes> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(Bytes::from(decompressed))
    }

    async fn compress_brotli(&self, data: &[u8]) -> Result<Bytes> {
        let mut output = Vec::new();
        let params = BrotliEncoderParams::default();
        let mut writer = CompressorWriter::with_params(&mut output, 4096, &params);
        writer.write_all(data)?;
        drop(writer);
        Ok(Bytes::from(output))
    }

    async fn decompress_brotli(&self, data: &[u8]) -> Result<Bytes> {
        let mut decompressor = Decompressor::new(data, 4096);
        let mut output = Vec::new();
        decompressor.read_to_end(&mut output)?;
        Ok(Bytes::from(output))
    }

    async fn compress_lz4(&self, data: &[u8]) -> Result<Bytes> {
        let compressed = compress_prepend_size(data);
        debug!(
            "LZ4 compressed {} bytes to {} bytes",
            data.len(),
            compressed.len()
        );
        Ok(Bytes::from(compressed))
    }

    async fn decompress_lz4(&self, data: &[u8]) -> Result<Bytes> {
        let decompressed = decompress_size_prepended(data)
            .map_err(|e| anyhow!("LZ4 decompression failed: {}", e))?;
        debug!(
            "LZ4 decompressed {} bytes to {} bytes",
            data.len(),
            decompressed.len()
        );
        Ok(Bytes::from(decompressed))
    }

    async fn compress_zstd(&self, data: &[u8]) -> Result<Bytes> {
        let compressed = encode_all(data, 6) // Level 6 for good balance
            .map_err(|e| anyhow!("Zstd compression failed: {}", e))?;
        debug!(
            "Zstd compressed {} bytes to {} bytes",
            data.len(),
            compressed.len()
        );
        Ok(Bytes::from(compressed))
    }

    async fn decompress_zstd(&self, data: &[u8]) -> Result<Bytes> {
        let decompressed =
            decode_all(data).map_err(|e| anyhow!("Zstd decompression failed: {}", e))?;
        debug!(
            "Zstd decompressed {} bytes to {} bytes",
            data.len(),
            decompressed.len()
        );
        Ok(Bytes::from(decompressed))
    }

    async fn encode_data(&self, data: &[u8], format: &EncodingFormat) -> Result<Vec<u8>> {
        match format {
            EncodingFormat::Json => Ok(data.to_vec()),
            EncodingFormat::MessagePack => {
                // Parse JSON and encode as MessagePack
                match serde_json::from_slice::<serde_json::Value>(data) {
                    Ok(value) => {
                        let encoded = to_vec(&value)
                            .map_err(|e| anyhow!("MessagePack encoding failed: {}", e))?;
                        debug!(
                            "MessagePack encoded {} bytes to {} bytes",
                            data.len(),
                            encoded.len()
                        );
                        Ok(encoded)
                    }
                    Err(_) => {
                        // If not valid JSON, return as binary MessagePack
                        let encoded = to_vec(&data)
                            .map_err(|e| anyhow!("MessagePack encoding failed: {}", e))?;
                        Ok(encoded)
                    }
                }
            }
            EncodingFormat::ProtocolBuffers => {
                // For generic data, we'll use a simple wrapper
                // In a real implementation, you'd define proper protobuf schemas
                Ok(data.to_vec()) // Simplified for now
            }
            EncodingFormat::Avro => {
                // Avro encoding would require schema definition
                // For now, return as-is
                Ok(data.to_vec())
            }
            EncodingFormat::Cbor => {
                // Parse JSON and encode as CBOR
                match serde_json::from_slice::<serde_json::Value>(data) {
                    Ok(value) => {
                        let encoded = serde_cbor::to_vec(&value)
                            .map_err(|e| anyhow!("CBOR encoding failed: {}", e))?;
                        debug!(
                            "CBOR encoded {} bytes to {} bytes",
                            data.len(),
                            encoded.len()
                        );
                        Ok(encoded)
                    }
                    Err(_) => {
                        // If not valid JSON, encode raw data as CBOR bytes
                        let encoded = serde_cbor::to_vec(&data)
                            .map_err(|e| anyhow!("CBOR encoding failed: {}", e))?;
                        Ok(encoded)
                    }
                }
            }
        }
    }

    async fn measure_latency(&self, service: &FederatedService) -> Result<f64> {
        // Simplified latency measurement
        // In a real implementation, this would perform actual network tests
        Ok(50.0) // Placeholder: 50ms
    }

    async fn estimate_bandwidth(&self, service: &FederatedService) -> Result<f64> {
        // Simplified bandwidth estimation
        Ok(100.0) // Placeholder: 100 Mbps
    }

    async fn estimate_packet_loss(&self, service: &FederatedService) -> Result<f64> {
        // Simplified packet loss estimation
        Ok(0.001) // Placeholder: 0.1% packet loss
    }

    async fn measure_jitter(&self, service: &FederatedService) -> Result<f64> {
        // Simplified jitter measurement
        Ok(5.0) // Placeholder: 5ms jitter
    }

    fn assess_connection_quality(
        &self,
        latency: f64,
        bandwidth: f64,
        packet_loss: f64,
        jitter: f64,
    ) -> ConnectionQuality {
        if latency < 20.0 && bandwidth > 100.0 && packet_loss < 0.001 && jitter < 5.0 {
            ConnectionQuality::Excellent
        } else if latency < 50.0 && bandwidth > 50.0 && packet_loss < 0.01 && jitter < 10.0 {
            ConnectionQuality::Good
        } else if latency < 100.0 && bandwidth > 10.0 && packet_loss < 0.05 && jitter < 20.0 {
            ConnectionQuality::Fair
        } else if latency < 200.0 && bandwidth > 1.0 && packet_loss < 0.1 && jitter < 50.0 {
            ConnectionQuality::Poor
        } else {
            ConnectionQuality::Critical
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_compression_basic() {
        let optimizer = NetworkOptimizer::new();
        let test_data = b"Hello, World! This is a test string for compression.";

        let compressed = optimizer
            .compress_data(test_data, EncodingFormat::Json)
            .await
            .unwrap();
        assert!(compressed.compressed_size <= compressed.original_size);

        let decompressed = optimizer.decompress_data(&compressed).await.unwrap();
        assert_eq!(decompressed.as_ref(), test_data);
    }

    #[tokio::test]
    async fn test_encoding_format_selection() {
        let optimizer = NetworkOptimizer::new();
        let metrics = NetworkPerformanceMetrics {
            latency_ms: 100.0,
            bandwidth_mbps: 50.0,
            packet_loss_rate: 0.01,
            jitter_ms: 10.0,
            connection_quality: ConnectionQuality::Good,
        };

        let format = optimizer
            .choose_optimal_encoding("sparql_results", 5000, &metrics)
            .await;
        // Should choose an appropriate format based on configuration
        assert!(matches!(
            format,
            EncodingFormat::Json | EncodingFormat::MessagePack
        ));
    }

    #[tokio::test]
    async fn test_adaptive_compression() {
        let optimizer = NetworkOptimizer::with_config(NetworkOptimizerConfig {
            enable_adaptive_compression: true,
            ..Default::default()
        });

        let metrics = NetworkPerformanceMetrics {
            latency_ms: 200.0,
            bandwidth_mbps: 5.0,
            packet_loss_rate: 0.05,
            jitter_ms: 30.0,
            connection_quality: ConnectionQuality::Poor,
        };

        let algorithm = optimizer.choose_optimal_compression(50000, &metrics).await;
        // Should choose Brotli for poor connections to maximize compression
        assert!(matches!(algorithm, CompressionAlgorithm::Brotli));
    }

    #[tokio::test]
    async fn test_statistics_tracking() {
        // Create optimizer with lower compression threshold
        let config = NetworkOptimizerConfig {
            compression_threshold: 10, // Lower threshold so test data gets compressed
            ..Default::default()
        };
        let optimizer = NetworkOptimizer::with_config(config);
        let test_data = b"Test data for statistics tracking.";

        // Perform some compressions
        for _ in 0..5 {
            let _ = optimizer
                .compress_data(test_data, EncodingFormat::Json)
                .await
                .unwrap();
        }

        let stats = optimizer.get_statistics().await;
        assert_eq!(stats.total_requests, 5);
        assert!(stats.total_bytes_original > 0);
    }
}
