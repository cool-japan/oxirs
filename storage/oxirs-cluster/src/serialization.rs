//! # Advanced Serialization Module
//!
//! Provides high-performance serialization with compression, binary encoding,
//! and schema evolution support for distributed consensus messages.

use anyhow::{anyhow, Result};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Cursor, Read, Write};
use std::time::{Duration, Instant};

/// Compression algorithms supported by the serialization layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 fast compression
    Lz4,
    /// Zstd high-efficiency compression
    Zstd,
    /// Deflate compression (compatibility)
    Deflate,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::Lz4
    }
}

/// Serialization format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// MessagePack binary format (fast)
    MessagePack,
    /// Protocol Buffers (compatibility)
    ProtocolBuffers,
    /// Bincode (Rust-native)
    Bincode,
    /// JSON (debugging/compatibility)
    Json,
}

impl Default for SerializationFormat {
    fn default() -> Self {
        Self::MessagePack
    }
}

/// Schema version for message compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Major version (breaking changes)
    pub major: u32,
    /// Minor version (compatible additions)
    pub minor: u32,
    /// Patch version (bug fixes)
    pub patch: u32,
}

impl Default for SchemaVersion {
    fn default() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
        }
    }
}

impl SchemaVersion {
    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &SchemaVersion) -> bool {
        // Major version must match exactly
        if self.major != other.major {
            return false;
        }
        // Minor version can be forward compatible
        self.minor >= other.minor
    }
}

/// Serialization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationConfig {
    /// Compression algorithm to use
    pub compression: CompressionAlgorithm,
    /// Serialization format
    pub format: SerializationFormat,
    /// Schema version
    pub schema_version: SchemaVersion,
    /// Enable checksumming for corruption detection
    pub enable_checksums: bool,
    /// Compression threshold (don't compress smaller messages)
    pub compression_threshold: usize,
    /// Enable performance metrics collection
    pub enable_metrics: bool,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            compression: CompressionAlgorithm::Lz4,
            format: SerializationFormat::MessagePack,
            schema_version: SchemaVersion::default(),
            enable_checksums: true,
            compression_threshold: 1024, // 1KB
            enable_metrics: true,
        }
    }
}

/// Serialized message with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedMessage {
    /// Schema version used for serialization
    pub schema_version: SchemaVersion,
    /// Compression algorithm used
    pub compression: CompressionAlgorithm,
    /// Serialization format used
    pub format: SerializationFormat,
    /// Message payload
    pub payload: Vec<u8>,
    /// CRC32 checksum (if enabled)
    pub checksum: Option<u32>,
    /// Original size before compression
    pub original_size: usize,
    /// Compression ratio (compressed_size / original_size)
    pub compression_ratio: f32,
}

/// Performance metrics for serialization operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SerializationMetrics {
    /// Total messages serialized
    pub messages_serialized: u64,
    /// Total messages deserialized
    pub messages_deserialized: u64,
    /// Total bytes serialized (before compression)
    pub bytes_serialized: u64,
    /// Total bytes deserialized (after decompression)
    pub bytes_deserialized: u64,
    /// Total serialization time
    pub serialization_time: Duration,
    /// Total deserialization time
    pub deserialization_time: Duration,
    /// Total compression time
    pub compression_time: Duration,
    /// Total decompression time
    pub decompression_time: Duration,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Checksum verification failures
    pub checksum_failures: u64,
    /// Schema version mismatches
    pub schema_mismatches: u64,
}

impl SerializationMetrics {
    /// Calculate overall throughput (messages per second)
    pub fn throughput(&self) -> f64 {
        let total_time = self.serialization_time + self.deserialization_time;
        if total_time.is_zero() {
            0.0
        } else {
            (self.messages_serialized + self.messages_deserialized) as f64
                / total_time.as_secs_f64()
        }
    }

    /// Calculate average message size
    pub fn avg_message_size(&self) -> f64 {
        if self.messages_serialized == 0 {
            0.0
        } else {
            self.bytes_serialized as f64 / self.messages_serialized as f64
        }
    }
}

/// High-performance message serializer with compression and schema versioning
pub struct MessageSerializer {
    config: SerializationConfig,
    metrics: SerializationMetrics,
}

impl MessageSerializer {
    /// Create a new message serializer with default configuration
    pub fn new() -> Self {
        Self::with_config(SerializationConfig::default())
    }

    /// Create a new message serializer with custom configuration
    pub fn with_config(config: SerializationConfig) -> Self {
        Self {
            config,
            metrics: SerializationMetrics::default(),
        }
    }

    /// Serialize a message with compression and metadata
    pub fn serialize<T: Serialize>(&mut self, message: &T) -> Result<SerializedMessage> {
        let start_time = Instant::now();

        // Serialize to binary format
        let serialized_data = match self.config.format {
            SerializationFormat::MessagePack => rmp_serde::to_vec(message)
                .map_err(|e| anyhow!("MessagePack serialization failed: {}", e))?,
            SerializationFormat::Bincode => bincode::serialize(message)
                .map_err(|e| anyhow!("Bincode serialization failed: {}", e))?,
            SerializationFormat::Json => serde_json::to_vec(message)
                .map_err(|e| anyhow!("JSON serialization failed: {}", e))?,
            SerializationFormat::ProtocolBuffers => {
                // For now, fall back to bincode for protobuf
                bincode::serialize(message)
                    .map_err(|e| anyhow!("ProtocolBuffers serialization failed: {}", e))?
            }
        };

        let original_size = serialized_data.len();

        // Apply compression if message is large enough
        let (compressed_data, compression_used) = if original_size
            > self.config.compression_threshold
        {
            let compression_start = Instant::now();
            let compressed = match self.config.compression {
                CompressionAlgorithm::None => (serialized_data.clone(), CompressionAlgorithm::None),
                CompressionAlgorithm::Lz4 => {
                    let compressed = compress_prepend_size(&serialized_data);
                    (compressed, CompressionAlgorithm::Lz4)
                }
                CompressionAlgorithm::Zstd => {
                    let compressed = zstd::bulk::compress(&serialized_data, 3)
                        .map_err(|e| anyhow!("Zstd compression failed: {}", e))?;
                    (compressed, CompressionAlgorithm::Zstd)
                }
                CompressionAlgorithm::Deflate => {
                    use flate2::write::DeflateEncoder;
                    use flate2::Compression;
                    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
                    encoder
                        .write_all(&serialized_data)
                        .map_err(|e| anyhow!("Deflate compression failed: {}", e))?;
                    let compressed = encoder
                        .finish()
                        .map_err(|e| anyhow!("Deflate compression failed: {}", e))?;
                    (compressed, CompressionAlgorithm::Deflate)
                }
            };
            if self.config.enable_metrics {
                self.metrics.compression_time += compression_start.elapsed();
            }
            compressed
        } else {
            (serialized_data, CompressionAlgorithm::None)
        };

        // Calculate compression ratio
        let compression_ratio = if original_size > 0 {
            compressed_data.len() as f32 / original_size as f32
        } else {
            1.0
        };

        // Calculate checksum if enabled
        let checksum = if self.config.enable_checksums {
            Some(crc32fast::hash(&compressed_data))
        } else {
            None
        };

        let serialized_message = SerializedMessage {
            schema_version: self.config.schema_version,
            compression: compression_used,
            format: self.config.format,
            payload: compressed_data,
            checksum,
            original_size,
            compression_ratio,
        };

        // Update metrics
        if self.config.enable_metrics {
            self.metrics.messages_serialized += 1;
            self.metrics.bytes_serialized += original_size as u64;
            self.metrics.serialization_time += start_time.elapsed();

            // Update rolling average compression ratio
            let total_messages = self.metrics.messages_serialized as f32;
            self.metrics.avg_compression_ratio =
                (self.metrics.avg_compression_ratio * (total_messages - 1.0) + compression_ratio)
                    / total_messages;
        }

        Ok(serialized_message)
    }

    /// Deserialize a message with decompression and validation
    pub fn deserialize<T: for<'de> Deserialize<'de>>(
        &mut self,
        message: &SerializedMessage,
    ) -> Result<T> {
        let start_time = Instant::now();

        // Check schema compatibility
        if !self
            .config
            .schema_version
            .is_compatible_with(&message.schema_version)
        {
            if self.config.enable_metrics {
                self.metrics.schema_mismatches += 1;
            }
            return Err(anyhow!(
                "Schema version mismatch: expected {:?}, got {:?}",
                self.config.schema_version,
                message.schema_version
            ));
        }

        // Verify checksum if enabled
        if self.config.enable_checksums {
            if let Some(expected_checksum) = message.checksum {
                let actual_checksum = crc32fast::hash(&message.payload);
                if actual_checksum != expected_checksum {
                    if self.config.enable_metrics {
                        self.metrics.checksum_failures += 1;
                    }
                    return Err(anyhow!(
                        "Checksum verification failed: expected {}, got {}",
                        expected_checksum,
                        actual_checksum
                    ));
                }
            }
        }

        // Decompress data
        let decompression_start = Instant::now();
        let decompressed_data = match message.compression {
            CompressionAlgorithm::None => message.payload.clone(),
            CompressionAlgorithm::Lz4 => decompress_size_prepended(&message.payload)
                .map_err(|e| anyhow!("LZ4 decompression failed: {}", e))?,
            CompressionAlgorithm::Zstd => {
                zstd::bulk::decompress(&message.payload, message.original_size)
                    .map_err(|e| anyhow!("Zstd decompression failed: {}", e))?
            }
            CompressionAlgorithm::Deflate => {
                use flate2::read::DeflateDecoder;
                let mut decoder = DeflateDecoder::new(Cursor::new(&message.payload));
                let mut decompressed = Vec::new();
                decoder
                    .read_to_end(&mut decompressed)
                    .map_err(|e| anyhow!("Deflate decompression failed: {}", e))?;
                decompressed
            }
        };

        if self.config.enable_metrics {
            self.metrics.decompression_time += decompression_start.elapsed();
        }

        // Deserialize from binary format
        let deserialized: T = match message.format {
            SerializationFormat::MessagePack => rmp_serde::from_slice(&decompressed_data)
                .map_err(|e| anyhow!("MessagePack deserialization failed: {}", e))?,
            SerializationFormat::Bincode => bincode::deserialize(&decompressed_data)
                .map_err(|e| anyhow!("Bincode deserialization failed: {}", e))?,
            SerializationFormat::Json => serde_json::from_slice(&decompressed_data)
                .map_err(|e| anyhow!("JSON deserialization failed: {}", e))?,
            SerializationFormat::ProtocolBuffers => {
                // For now, fall back to bincode for protobuf
                bincode::deserialize(&decompressed_data)
                    .map_err(|e| anyhow!("ProtocolBuffers deserialization failed: {}", e))?
            }
        };

        // Update metrics
        if self.config.enable_metrics {
            self.metrics.messages_deserialized += 1;
            self.metrics.bytes_deserialized += decompressed_data.len() as u64;
            self.metrics.deserialization_time += start_time.elapsed();
        }

        Ok(deserialized)
    }

    /// Get current performance metrics
    pub fn metrics(&self) -> &SerializationMetrics {
        &self.metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.metrics = SerializationMetrics::default();
    }

    /// Update serialization configuration
    pub fn update_config(&mut self, config: SerializationConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &SerializationConfig {
        &self.config
    }

    /// Benchmark serialization performance for a given message type
    pub fn benchmark<T: Serialize + for<'de> Deserialize<'de> + Clone>(
        &mut self,
        message: &T,
        iterations: usize,
    ) -> Result<BenchmarkResults> {
        let mut results = BenchmarkResults {
            iterations,
            total_serialization_time: Duration::ZERO,
            total_deserialization_time: Duration::ZERO,
            min_serialization_time: Duration::MAX,
            max_serialization_time: Duration::ZERO,
            min_deserialization_time: Duration::MAX,
            max_deserialization_time: Duration::ZERO,
            total_compressed_size: 0,
            total_uncompressed_size: 0,
            compression_ratios: Vec::new(),
        };

        for _ in 0..iterations {
            // Benchmark serialization
            let serialize_start = Instant::now();
            let serialized = self.serialize(message)?;
            let serialize_time = serialize_start.elapsed();

            results.total_serialization_time += serialize_time;
            results.min_serialization_time = results.min_serialization_time.min(serialize_time);
            results.max_serialization_time = results.max_serialization_time.max(serialize_time);
            results.total_compressed_size += serialized.payload.len();
            results.total_uncompressed_size += serialized.original_size;
            results
                .compression_ratios
                .push(serialized.compression_ratio);

            // Benchmark deserialization
            let deserialize_start = Instant::now();
            let _deserialized: T = self.deserialize(&serialized)?;
            let deserialize_time = deserialize_start.elapsed();

            results.total_deserialization_time += deserialize_time;
            results.min_deserialization_time =
                results.min_deserialization_time.min(deserialize_time);
            results.max_deserialization_time =
                results.max_deserialization_time.max(deserialize_time);
        }

        Ok(results)
    }
}

impl Default for MessageSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark results for serialization performance testing
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub iterations: usize,
    pub total_serialization_time: Duration,
    pub total_deserialization_time: Duration,
    pub min_serialization_time: Duration,
    pub max_serialization_time: Duration,
    pub min_deserialization_time: Duration,
    pub max_deserialization_time: Duration,
    pub total_compressed_size: usize,
    pub total_uncompressed_size: usize,
    pub compression_ratios: Vec<f32>,
}

impl BenchmarkResults {
    /// Calculate average serialization time
    pub fn avg_serialization_time(&self) -> Duration {
        self.total_serialization_time / self.iterations as u32
    }

    /// Calculate average deserialization time
    pub fn avg_deserialization_time(&self) -> Duration {
        self.total_deserialization_time / self.iterations as u32
    }

    /// Calculate average compression ratio
    pub fn avg_compression_ratio(&self) -> f32 {
        if self.compression_ratios.is_empty() {
            1.0
        } else {
            self.compression_ratios.iter().sum::<f32>() / self.compression_ratios.len() as f32
        }
    }

    /// Calculate throughput (messages per second)
    pub fn throughput(&self) -> f64 {
        let total_time = self.total_serialization_time + self.total_deserialization_time;
        if total_time.is_zero() {
            0.0
        } else {
            (self.iterations * 2) as f64 / total_time.as_secs_f64() // *2 for serialize + deserialize
        }
    }
}

/// Adaptive compression selector that chooses optimal compression based on data characteristics
pub struct AdaptiveCompression {
    /// Sample size for algorithm evaluation
    sample_size: usize,
    /// Performance history for different algorithms
    performance_history: HashMap<CompressionAlgorithm, Vec<f32>>,
    /// Current best algorithm
    current_best: CompressionAlgorithm,
}

impl AdaptiveCompression {
    /// Create a new adaptive compression selector
    pub fn new() -> Self {
        Self {
            sample_size: 100,
            performance_history: HashMap::new(),
            current_best: CompressionAlgorithm::Lz4,
        }
    }

    /// Evaluate and potentially update the best compression algorithm
    pub fn evaluate_and_select(&mut self, data: &[u8]) -> CompressionAlgorithm {
        // If we don't have enough samples, stick with current best
        if data.len() < 1024 {
            return self.current_best;
        }

        // Periodically re-evaluate algorithms
        let total_samples: usize = self.performance_history.values().map(|v| v.len()).sum();
        if total_samples % self.sample_size == 0 {
            self.benchmark_algorithms(data);
        }

        self.current_best
    }

    /// Benchmark all compression algorithms on sample data
    fn benchmark_algorithms(&mut self, sample_data: &[u8]) {
        let algorithms = [
            CompressionAlgorithm::None,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Deflate,
        ];

        let mut best_score = f32::MIN;
        let mut best_algorithm = self.current_best;

        for &algorithm in &algorithms {
            if let Ok(score) = self.benchmark_algorithm(algorithm, sample_data) {
                self.performance_history
                    .entry(algorithm)
                    .or_default()
                    .push(score);

                // Keep only recent samples
                let history = self.performance_history.get_mut(&algorithm).unwrap();
                if history.len() > self.sample_size {
                    history.remove(0);
                }

                // Calculate average score
                let avg_score = history.iter().sum::<f32>() / history.len() as f32;
                if avg_score > best_score {
                    best_score = avg_score;
                    best_algorithm = algorithm;
                }
            }
        }

        self.current_best = best_algorithm;
    }

    /// Benchmark a single compression algorithm
    fn benchmark_algorithm(&self, algorithm: CompressionAlgorithm, data: &[u8]) -> Result<f32> {
        let start_time = Instant::now();

        let compressed_size = match algorithm {
            CompressionAlgorithm::None => data.len(),
            CompressionAlgorithm::Lz4 => compress_prepend_size(data).len(),
            CompressionAlgorithm::Zstd => zstd::bulk::compress(data, 3)
                .map_err(|e| anyhow!("Zstd benchmark failed: {}", e))?
                .len(),
            CompressionAlgorithm::Deflate => {
                use flate2::write::DeflateEncoder;
                use flate2::Compression;
                let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
                encoder
                    .write_all(data)
                    .map_err(|e| anyhow!("Deflate benchmark failed: {}", e))?;
                encoder
                    .finish()
                    .map_err(|e| anyhow!("Deflate benchmark failed: {}", e))?
                    .len()
            }
        };

        let compression_time = start_time.elapsed();
        let compression_ratio = data.len() as f32 / compressed_size as f32;
        let speed = data.len() as f32 / compression_time.as_secs_f32(); // bytes per second

        // Score combines compression ratio and speed (weighted towards speed for real-time use)
        let score = (compression_ratio * 0.3) + (speed / 1_000_000.0 * 0.7); // normalize speed to MB/s

        Ok(score)
    }

    /// Get current best algorithm
    pub fn current_best(&self) -> CompressionAlgorithm {
        self.current_best
    }

    /// Get performance history for debugging
    pub fn performance_history(&self) -> &HashMap<CompressionAlgorithm, Vec<f32>> {
        &self.performance_history
    }
}

impl Default for AdaptiveCompression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestMessage {
        id: u64,
        data: String,
        values: Vec<i32>,
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut serializer = MessageSerializer::new();
        let message = TestMessage {
            id: 12345,
            data: "Hello, distributed world!".to_string(),
            values: vec![1, 2, 3, 4, 5],
        };

        let serialized = serializer.serialize(&message).unwrap();
        let deserialized: TestMessage = serializer.deserialize(&serialized).unwrap();

        assert_eq!(message, deserialized);
    }

    #[test]
    fn test_compression_algorithms() {
        let algorithms = [
            CompressionAlgorithm::None,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Deflate,
        ];

        for algorithm in algorithms {
            let config = SerializationConfig {
                compression: algorithm,
                compression_threshold: 0, // Always compress
                ..Default::default()
            };

            let mut serializer = MessageSerializer::with_config(config);
            let message = TestMessage {
                id: 12345,
                data: "A".repeat(1000), // Large string for compression
                values: (0..100).collect(),
            };

            let serialized = serializer.serialize(&message).unwrap();
            let deserialized: TestMessage = serializer.deserialize(&serialized).unwrap();

            assert_eq!(message, deserialized);
            assert_eq!(serialized.compression, algorithm);
        }
    }

    #[test]
    fn test_schema_version_compatibility() {
        let v1_0_0 = SchemaVersion {
            major: 1,
            minor: 0,
            patch: 0,
        };
        let v1_1_0 = SchemaVersion {
            major: 1,
            minor: 1,
            patch: 0,
        };
        let v2_0_0 = SchemaVersion {
            major: 2,
            minor: 0,
            patch: 0,
        };

        assert!(v1_1_0.is_compatible_with(&v1_0_0));
        assert!(!v1_0_0.is_compatible_with(&v1_1_0));
        assert!(!v1_0_0.is_compatible_with(&v2_0_0));
        assert!(!v2_0_0.is_compatible_with(&v1_0_0));
    }

    #[test]
    fn test_checksum_verification() {
        let config = SerializationConfig {
            enable_checksums: true,
            ..Default::default()
        };

        let mut serializer = MessageSerializer::with_config(config);
        let message = TestMessage {
            id: 12345,
            data: "Test message".to_string(),
            values: vec![1, 2, 3],
        };

        let mut serialized = serializer.serialize(&message).unwrap();
        assert!(serialized.checksum.is_some());

        // Corrupt the payload
        serialized.payload[0] ^= 0xFF;

        let result: Result<TestMessage> = serializer.deserialize(&serialized);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Checksum verification failed"));
    }

    #[test]
    fn test_adaptive_compression() {
        let mut adaptive = AdaptiveCompression::new();
        let data = b"Hello, world!".repeat(100);

        let algorithm1 = adaptive.evaluate_and_select(&data);
        let algorithm2 = adaptive.evaluate_and_select(&data);

        // Should return a valid algorithm
        assert!(matches!(
            algorithm1,
            CompressionAlgorithm::None
                | CompressionAlgorithm::Lz4
                | CompressionAlgorithm::Zstd
                | CompressionAlgorithm::Deflate
        ));
        assert_eq!(algorithm1, algorithm2); // Should be stable for same data
    }

    #[test]
    fn test_metrics_collection() {
        let config = SerializationConfig {
            enable_metrics: true,
            ..Default::default()
        };

        let mut serializer = MessageSerializer::with_config(config);
        let message = TestMessage {
            id: 12345,
            data: "Test message".to_string(),
            values: vec![1, 2, 3],
        };

        let serialized = serializer.serialize(&message).unwrap();
        let _deserialized: TestMessage = serializer.deserialize(&serialized).unwrap();

        let metrics = serializer.metrics();
        assert_eq!(metrics.messages_serialized, 1);
        assert_eq!(metrics.messages_deserialized, 1);
        assert!(metrics.bytes_serialized > 0);
        assert!(metrics.bytes_deserialized > 0);
        assert!(metrics.serialization_time > Duration::ZERO);
        assert!(metrics.deserialization_time > Duration::ZERO);
    }
}
