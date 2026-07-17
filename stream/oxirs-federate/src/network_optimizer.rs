//! # Network Optimization for Federated Queries
//!
//! This module implements advanced network optimization techniques including
//! compression, encoding, and bandwidth optimization for federated query processing.

use anyhow::{anyhow, Result};
use bytes::Bytes;
use oxiarc_zstd::{decode_all, encode_all};
use rmp_serde::to_vec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::debug;

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

/// Number of HEAD probes issued per [`NetworkOptimizer::probe_service`] call.
/// Multiple probes are needed to derive packet loss (failure ratio) and
/// jitter (variance across samples) -- a single request can only ever give
/// a latency point estimate.
const NETWORK_PROBE_COUNT: usize = 3;

/// Per-probe timeout: a probe that doesn't complete within this window counts
/// as a lost packet for [`NetworkOptimizer::estimate_packet_loss`].
const NETWORK_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Raw results of a [`NetworkOptimizer::probe_service`] round: the
/// round-trip time of every probe that completed, plus any `Content-Length`
/// observed (used to derive a real, if rough, bandwidth estimate).
#[derive(Debug, Clone, Default)]
struct NetworkProbeSamples {
    /// Round-trip time of each successful probe, in milliseconds.
    successful_rtts_ms: Vec<f64>,
    /// Total probes attempted (successes + failures/timeouts).
    attempted: usize,
    /// `Content-Length` reported on the first successful probe that had one.
    content_length: Option<u64>,
}

impl NetworkProbeSamples {
    fn mean_latency_ms(&self) -> Option<f64> {
        if self.successful_rtts_ms.is_empty() {
            return None;
        }
        Some(self.successful_rtts_ms.iter().sum::<f64>() / self.successful_rtts_ms.len() as f64)
    }

    fn packet_loss_rate(&self) -> f64 {
        if self.attempted == 0 {
            return 0.0;
        }
        let lost = self.attempted.saturating_sub(self.successful_rtts_ms.len());
        lost as f64 / self.attempted as f64
    }

    /// Sample standard deviation of successful RTTs, in milliseconds. `0.0`
    /// with fewer than two samples (jitter is undefined for a single point).
    fn jitter_ms(&self) -> f64 {
        let n = self.successful_rtts_ms.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean_latency_ms().unwrap_or(0.0);
        let variance = self
            .successful_rtts_ms
            .iter()
            .map(|rtt| (rtt - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        variance.sqrt()
    }
}

/// Network optimizer for federated queries
pub struct NetworkOptimizer {
    config: NetworkOptimizerConfig,
    stats: Arc<RwLock<NetworkOptimizationStats>>,
    performance_metrics: Arc<RwLock<HashMap<String, NetworkPerformanceMetrics>>>,
    #[allow(dead_code)]
    compression_cache: Arc<RwLock<HashMap<u64, CompressedData>>>,
    /// HTTP client used for the real (non-fabricated) network condition
    /// probes in [`Self::monitor_network_performance`].
    http_client: reqwest::Client,
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
            http_client: reqwest::Client::builder()
                .timeout(NETWORK_PROBE_TIMEOUT)
                .user_agent("oxirs-federate-network-optimizer/1.0")
                .build()
                .unwrap_or_default(),
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
        let _start_time = Instant::now();

        // Perform a real lightweight network test (timed HEAD probes) once,
        // then derive all four metrics from the same probe round instead of
        // issuing a separate round of requests per metric.
        let samples = self.probe_service(service).await;

        let latency = self.measure_latency(&samples, &service.endpoint)?;
        let bandwidth = self.estimate_bandwidth(&samples, &service.endpoint)?;
        let packet_loss = self.estimate_packet_loss(&samples);
        let jitter = self.measure_jitter(&samples);

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
        // flate2 `Compression::default()` mapped to level 6 (Pure Rust oxiarc-deflate).
        let compressed = oxiarc_deflate::gzip_compress(data, 6)
            .map_err(|e| anyhow!("Gzip compression failed: {}", e))?;
        Ok(Bytes::from(compressed))
    }

    async fn decompress_gzip(&self, data: &[u8]) -> Result<Bytes> {
        let decompressed = oxiarc_deflate::gzip_decompress(data)
            .map_err(|e| anyhow!("Gzip decompression failed: {}", e))?;
        Ok(Bytes::from(decompressed))
    }

    async fn compress_brotli(&self, data: &[u8]) -> Result<Bytes> {
        // brotli `BrotliEncoderParams::default()` quality is 11 (max); preserve it.
        let output = oxiarc_brotli::compress(data, 11)
            .map_err(|e| anyhow!("Brotli compression failed: {}", e))?;
        Ok(Bytes::from(output))
    }

    async fn decompress_brotli(&self, data: &[u8]) -> Result<Bytes> {
        let output = oxiarc_brotli::decompress(data)
            .map_err(|e| anyhow!("Brotli decompression failed: {}", e))?;
        Ok(Bytes::from(output))
    }

    async fn compress_lz4(&self, data: &[u8]) -> Result<Bytes> {
        let compressed =
            oxiarc_lz4::compress(data).map_err(|e| anyhow!("LZ4 compression failed: {}", e))?;
        debug!(
            "LZ4 compressed {} bytes to {} bytes",
            data.len(),
            compressed.len()
        );
        Ok(Bytes::from(compressed))
    }

    async fn decompress_lz4(&self, data: &[u8]) -> Result<Bytes> {
        let decompressed = oxiarc_lz4::decompress(data, 100 * 1024 * 1024)
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
                        let mut encoded = Vec::new();
                        ciborium::ser::into_writer(&value, &mut encoded)
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
                        let mut encoded = Vec::new();
                        ciborium::ser::into_writer(&data, &mut encoded)
                            .map_err(|e| anyhow!("CBOR encoding failed: {}", e))?;
                        Ok(encoded)
                    }
                }
            }
        }
    }

    /// Issue [`NETWORK_PROBE_COUNT`] real HTTP HEAD requests against
    /// `service.endpoint` and record their round-trip times (and any
    /// `Content-Length` header seen). This is the single source of ground
    /// truth for `measure_latency`/`estimate_bandwidth`/
    /// `estimate_packet_loss`/`measure_jitter` below -- none of them
    /// fabricate a constant any more.
    async fn probe_service(&self, service: &FederatedService) -> NetworkProbeSamples {
        let mut samples = NetworkProbeSamples {
            attempted: NETWORK_PROBE_COUNT,
            ..Default::default()
        };

        for _ in 0..NETWORK_PROBE_COUNT {
            let start = Instant::now();
            match self
                .http_client
                .head(&service.endpoint)
                .timeout(NETWORK_PROBE_TIMEOUT)
                .send()
                .await
            {
                Ok(response) => {
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    samples.successful_rtts_ms.push(elapsed_ms);
                    if samples.content_length.is_none() {
                        // `Response::content_length()` reflects the actual
                        // body frame size hint, which hyper reports as `0`
                        // for a HEAD response regardless of the declared
                        // `Content-Length` header (HEAD responses never
                        // carry a body). Read the header value directly so
                        // a real declared payload size is captured.
                        samples.content_length = response
                            .headers()
                            .get(reqwest::header::CONTENT_LENGTH)
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| v.parse::<u64>().ok());
                    }
                }
                Err(e) => {
                    debug!(
                        endpoint = %service.endpoint,
                        error = %e,
                        "network probe failed (counted as a lost packet)"
                    );
                }
            }
        }

        samples
    }

    /// Mean round-trip time across the probe round, in milliseconds.
    /// Errors out (rather than fabricating a number) when every probe in
    /// the round failed -- there is no real latency to report for an
    /// endpoint that could not be reached at all.
    fn measure_latency(&self, samples: &NetworkProbeSamples, endpoint: &str) -> Result<f64> {
        samples.mean_latency_ms().ok_or_else(|| {
            anyhow!(
                "unable to measure latency to '{endpoint}': all {} probe(s) failed \
                 (endpoint unreachable within {:?})",
                samples.attempted,
                NETWORK_PROBE_TIMEOUT
            )
        })
    }

    /// Rough bandwidth estimate derived from a real probe's `Content-Length`
    /// header and its round-trip time (bytes transferred / time taken).
    /// This is necessarily an approximation (a HEAD response is typically
    /// small and RTT-dominated rather than throughput-dominated), but it is
    /// computed from an actual request/response, not a hardcoded constant.
    ///
    /// Errors out only when every probe failed outright (nothing to derive
    /// anything from -- mirrors [`Self::measure_latency`]). When probes
    /// succeeded but none reported a `Content-Length` (common for dynamic
    /// SPARQL/GraphQL endpoints, which often omit it on HEAD), this returns
    /// an honest `Ok(0.0)` -- "no throughput evidence" -- rather than
    /// failing the whole monitoring round or guessing a positive number;
    /// `0.0` correctly steers [`Self::assess_connection_quality`] toward
    /// the conservative `Critical` tier instead of overstating confidence.
    fn estimate_bandwidth(&self, samples: &NetworkProbeSamples, endpoint: &str) -> Result<f64> {
        let Some(latency_ms) = samples.mean_latency_ms() else {
            return Err(anyhow!(
                "unable to estimate bandwidth to '{endpoint}': all {} probe(s) failed",
                samples.attempted
            ));
        };

        let Some(content_length) = samples.content_length else {
            debug!(
                endpoint,
                "no successful probe reported a Content-Length header; bandwidth \
                 cannot be estimated, reporting 0.0 (unknown) rather than a guess"
            );
            return Ok(0.0);
        };

        if latency_ms <= 0.0 {
            return Ok(0.0);
        }
        let seconds = latency_ms / 1000.0;
        let mbps = (content_length as f64 * 8.0) / seconds / 1_000_000.0;
        Ok(mbps)
    }

    /// Fraction of probes in the round that failed/timed out.
    fn estimate_packet_loss(&self, samples: &NetworkProbeSamples) -> f64 {
        samples.packet_loss_rate()
    }

    /// Sample standard deviation of successful probe RTTs, in milliseconds.
    fn measure_jitter(&self, samples: &NetworkProbeSamples) -> f64 {
        samples.jitter_ms()
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

    /// Spawn a minimal raw-TCP HTTP/1.1 server on an OS-assigned loopback
    /// port that serves `response` verbatim to up to `connections_to_serve`
    /// connections, then stops accepting. Used to give the network probe
    /// regression tests below a real socket to talk to instead of mocking
    /// out `reqwest` internals.
    fn spawn_mock_http_server(
        response: &'static str,
        connections_to_serve: usize,
    ) -> std::net::SocketAddr {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind should succeed");
        let addr = listener.local_addr().expect("local_addr should succeed");
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for _ in 0..connections_to_serve {
                if let Ok((mut stream, _)) = listener.accept() {
                    let mut buf = [0u8; 1024];
                    // Drain (some of) the request; we don't need to parse it.
                    let _ = stream.read(&mut buf);
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                }
            }
        });
        addr
    }

    /// Regression test: `probe_service`/`measure_latency`/`estimate_bandwidth`/
    /// `estimate_packet_loss` used to be `measure_latency->Ok(50.0)`,
    /// `estimate_bandwidth->Ok(100.0)`, `estimate_packet_loss->Ok(0.001)`
    /// hardcoded constants regardless of `_service`. They must now reflect
    /// real timed HTTP probes against the service endpoint.
    #[tokio::test]
    async fn test_probe_service_measures_real_http_responses() {
        let response = "HTTP/1.1 200 OK\r\nContent-Length: 4096\r\nConnection: close\r\n\r\n";
        let addr = spawn_mock_http_server(response, NETWORK_PROBE_COUNT);

        let optimizer = NetworkOptimizer::new();
        let service = FederatedService {
            endpoint: format!("http://{addr}/"),
            ..Default::default()
        };

        let samples = optimizer.probe_service(&service).await;
        assert_eq!(samples.attempted, NETWORK_PROBE_COUNT);
        assert_eq!(
            samples.successful_rtts_ms.len(),
            NETWORK_PROBE_COUNT,
            "all probes against a healthy mock server should succeed"
        );
        assert_eq!(samples.content_length, Some(4096));

        let latency = optimizer
            .measure_latency(&samples, &service.endpoint)
            .expect("latency should be measurable from real probes");
        assert!(latency >= 0.0);

        let bandwidth = optimizer
            .estimate_bandwidth(&samples, &service.endpoint)
            .expect("bandwidth should be derivable from the Content-Length header");
        assert!(
            bandwidth > 0.0,
            "bandwidth must be a real positive estimate, not the old fabricated 100.0 constant \
             regardless of endpoint"
        );

        assert_eq!(
            optimizer.estimate_packet_loss(&samples),
            0.0,
            "no probes were lost against a healthy server"
        );
    }

    /// Regression test: an endpoint nothing is listening on used to still
    /// report `Ok(50.0)` latency / `Ok(0.001)` packet loss as if it were
    /// perfectly healthy. It must now surface real failure information.
    #[tokio::test]
    async fn test_measure_latency_errors_on_unreachable_endpoint() {
        // Bind then immediately drop: reserves a port nothing listens on.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind should succeed");
        let addr = listener.local_addr().expect("local_addr should succeed");
        drop(listener);

        let optimizer = NetworkOptimizer::new();
        let service = FederatedService {
            endpoint: format!("http://{addr}/"),
            ..Default::default()
        };

        let samples = optimizer.probe_service(&service).await;
        assert_eq!(
            samples.successful_rtts_ms.len(),
            0,
            "every probe against an unreachable endpoint must fail"
        );
        assert_eq!(samples.packet_loss_rate(), 1.0);

        let result = optimizer.measure_latency(&samples, &service.endpoint);
        assert!(
            result.is_err(),
            "measure_latency must error on total probe failure instead of fabricating 50.0ms"
        );

        let bandwidth_result = optimizer.estimate_bandwidth(&samples, &service.endpoint);
        assert!(
            bandwidth_result.is_err(),
            "estimate_bandwidth must error on total probe failure instead of fabricating 100.0 Mbps"
        );
    }

    /// Regression test: bandwidth used to always claim 100.0 Mbps even when
    /// the endpoint responded but gave no size information to estimate
    /// throughput from. `0.0` ("no evidence"), not a guessed positive
    /// number, is the honest answer.
    #[tokio::test]
    async fn test_estimate_bandwidth_is_zero_without_content_length() {
        let response = "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n";
        let addr = spawn_mock_http_server(response, NETWORK_PROBE_COUNT);

        let optimizer = NetworkOptimizer::new();
        let service = FederatedService {
            endpoint: format!("http://{addr}/"),
            ..Default::default()
        };

        let samples = optimizer.probe_service(&service).await;
        assert_eq!(samples.content_length, None);

        let bandwidth = optimizer
            .estimate_bandwidth(&samples, &service.endpoint)
            .expect("succeeded probes without Content-Length report 0.0, not an error");
        assert_eq!(bandwidth, 0.0);
    }

    #[tokio::test]
    async fn test_compression_basic() {
        let optimizer = NetworkOptimizer::new();
        let test_data = b"Hello, World! This is a test string for compression.";

        let compressed = optimizer
            .compress_data(test_data, EncodingFormat::Json)
            .await
            .expect("operation should succeed");
        assert!(compressed.compressed_size <= compressed.original_size);

        let decompressed = optimizer
            .decompress_data(&compressed)
            .await
            .expect("async operation should succeed");
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
                .expect("operation should succeed");
        }

        let stats = optimizer.get_statistics().await;
        assert_eq!(stats.total_requests, 5);
        assert!(stats.total_bytes_original > 0);
    }

    /// Round-trip the migrated Pure Rust oxiarc paths (gzip + brotli) through
    /// the public compress/decompress API, including an incompressible buffer,
    /// to guarantee wire-format fidelity after the brotli/flate2 migration.
    #[tokio::test]
    async fn test_gzip_brotli_roundtrip() {
        let compressible = b"Federated query result payload. ".repeat(40);
        // xorshift64 PRNG: incompressible byte stream, no external crates.
        let mut random: Vec<u8> = Vec::with_capacity(2048);
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        for _ in 0..2048 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            random.push((state & 0xFF) as u8);
        }

        for algorithm in [CompressionAlgorithm::Gzip, CompressionAlgorithm::Brotli] {
            let label = format!("{algorithm:?}");
            let config = NetworkOptimizerConfig {
                compression_algorithm: algorithm,
                compression_threshold: 1, // force compression for both buffers
                ..Default::default()
            };
            let optimizer = NetworkOptimizer::with_config(config);

            for payload in [compressible.as_slice(), random.as_slice()] {
                let compressed = optimizer
                    .compress_data(payload, EncodingFormat::Json)
                    .await
                    .expect("compression should succeed");
                assert!(
                    !matches!(compressed.algorithm, CompressionAlgorithm::None),
                    "{label} should actually compress"
                );

                let decompressed = optimizer
                    .decompress_data(&compressed)
                    .await
                    .expect("decompression should succeed");
                assert_eq!(
                    decompressed.as_ref(),
                    payload,
                    "round-trip mismatch for {label} (len {})",
                    payload.len()
                );
            }
        }
    }
}
