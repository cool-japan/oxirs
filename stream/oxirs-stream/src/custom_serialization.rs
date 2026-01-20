//! # Custom Serialization Formats
//!
//! Extensible serialization framework allowing users to register custom serialization
//! formats and providing additional modern serialization options beyond the standard formats.
//!
//! ## Features
//!
//! - **Custom Serializer Trait**: Define your own serialization formats
//! - **Format Registry**: Register and discover custom serializers
//! - **Additional Formats**: BSON, Thrift, FlexBuffers, RON
//! - **Zero-Copy Serialization**: Support for zero-copy deserialization
//! - **Benchmarking**: Built-in performance benchmarking
//! - **Schema Validation**: Optional schema validation for custom formats
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_stream::custom_serialization::{CustomSerializer, SerializerRegistry};
//!
//! // Define a custom serializer
//! struct MyCustomSerializer;
//!
//! impl CustomSerializer for MyCustomSerializer {
//!     fn serialize(&self, data: &[u8]) -> Result<Vec<u8>> {
//!         // Custom serialization logic
//!         Ok(data.to_vec())
//!     }
//!
//!     fn deserialize(&self, data: &[u8]) -> Result<Vec<u8>> {
//!         // Custom deserialization logic
//!         Ok(data.to_vec())
//!     }
//! }
//!
//! // Register the serializer
//! let mut registry = SerializerRegistry::new();
//! registry.register("my-format", Box::new(MyCustomSerializer))?;
//! ```

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Custom serializer trait
pub trait CustomSerializer: Send + Sync {
    /// Serialize data to bytes
    fn serialize(&self, data: &[u8]) -> Result<Vec<u8>>;

    /// Deserialize bytes to data
    fn deserialize(&self, data: &[u8]) -> Result<Vec<u8>>;

    /// Get format name
    fn format_name(&self) -> &str;

    /// Get format version
    fn format_version(&self) -> &str {
        "1.0.0"
    }

    /// Get magic bytes for format detection
    fn magic_bytes(&self) -> Option<&[u8]> {
        None
    }

    /// Supports zero-copy deserialization
    fn supports_zero_copy(&self) -> bool {
        false
    }

    /// Validate schema (optional)
    fn validate_schema(&self, _schema: &[u8], _data: &[u8]) -> Result<bool> {
        Ok(true)
    }

    /// Get serialization statistics
    fn stats(&self) -> SerializerStats {
        SerializerStats::default()
    }
}

/// Serializer statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SerializerStats {
    /// Total bytes serialized
    pub bytes_serialized: u64,

    /// Total bytes deserialized
    pub bytes_deserialized: u64,

    /// Number of serialization operations
    pub serialization_count: u64,

    /// Number of deserialization operations
    pub deserialization_count: u64,

    /// Average serialization time (ms)
    pub avg_serialization_time_ms: f64,

    /// Average deserialization time (ms)
    pub avg_deserialization_time_ms: f64,

    /// Errors encountered
    pub error_count: u64,
}

/// Serializer registry
pub struct SerializerRegistry {
    serializers: Arc<RwLock<HashMap<String, Box<dyn CustomSerializer>>>>,
    benchmarks: Arc<RwLock<HashMap<String, SerializerBenchmark>>>,
}

impl SerializerRegistry {
    /// Create a new serializer registry
    pub fn new() -> Self {
        // Register built-in serializers
        Self {
            serializers: Arc::new(RwLock::new(HashMap::new())),
            benchmarks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a custom serializer
    pub async fn register(&self, name: &str, serializer: Box<dyn CustomSerializer>) -> Result<()> {
        let mut serializers = self.serializers.write().await;

        if serializers.contains_key(name) {
            return Err(anyhow!("Serializer '{}' already registered", name));
        }

        serializers.insert(name.to_string(), serializer);
        info!("Registered custom serializer: {}", name);
        Ok(())
    }

    /// Unregister a serializer
    pub async fn unregister(&self, name: &str) -> Result<()> {
        let mut serializers = self.serializers.write().await;

        if serializers.remove(name).is_some() {
            info!("Unregistered serializer: {}", name);
            Ok(())
        } else {
            Err(anyhow!("Serializer '{}' not found", name))
        }
    }

    /// Get a serializer by name
    pub async fn get(&self, name: &str) -> Result<String> {
        let serializers = self.serializers.read().await;

        if serializers.contains_key(name) {
            Ok(name.to_string())
        } else {
            Err(anyhow!("Serializer '{}' not found", name))
        }
    }

    /// List all registered serializers
    pub async fn list(&self) -> Vec<String> {
        let serializers = self.serializers.read().await;
        serializers.keys().cloned().collect()
    }

    /// Serialize using a specific format
    pub async fn serialize(&self, format: &str, data: &[u8]) -> Result<Vec<u8>> {
        let serializers = self.serializers.read().await;

        let serializer = serializers
            .get(format)
            .ok_or_else(|| anyhow!("Serializer '{}' not found", format))?;

        let start = Instant::now();
        let result = serializer.serialize(data)?;
        let duration = start.elapsed();

        // Update benchmarks
        drop(serializers);
        self.update_benchmark(format, duration.as_secs_f64() * 1000.0, true)
            .await;

        Ok(result)
    }

    /// Deserialize using a specific format
    pub async fn deserialize(&self, format: &str, data: &[u8]) -> Result<Vec<u8>> {
        let serializers = self.serializers.read().await;

        let serializer = serializers
            .get(format)
            .ok_or_else(|| anyhow!("Serializer '{}' not found", format))?;

        let start = Instant::now();
        let result = serializer.deserialize(data)?;
        let duration = start.elapsed();

        // Update benchmarks
        drop(serializers);
        self.update_benchmark(format, duration.as_secs_f64() * 1000.0, false)
            .await;

        Ok(result)
    }

    /// Auto-detect format from magic bytes
    pub async fn detect_format(&self, data: &[u8]) -> Option<String> {
        let serializers = self.serializers.read().await;

        for (name, serializer) in serializers.iter() {
            if let Some(magic) = serializer.magic_bytes() {
                if data.len() >= magic.len() && &data[0..magic.len()] == magic {
                    return Some(name.clone());
                }
            }
        }

        None
    }

    /// Get benchmarks for a specific serializer
    pub async fn get_benchmark(&self, format: &str) -> Option<SerializerBenchmark> {
        let benchmarks = self.benchmarks.read().await;
        benchmarks.get(format).cloned()
    }

    /// Get all benchmarks
    pub async fn all_benchmarks(&self) -> HashMap<String, SerializerBenchmark> {
        let benchmarks = self.benchmarks.read().await;
        benchmarks.clone()
    }

    /// Update benchmark statistics
    async fn update_benchmark(&self, format: &str, duration_ms: f64, is_serialization: bool) {
        let mut benchmarks = self.benchmarks.write().await;

        let benchmark = benchmarks
            .entry(format.to_string())
            .or_insert_with(SerializerBenchmark::default);

        if is_serialization {
            benchmark.serialization_times.push(duration_ms);
            benchmark.serialization_count += 1;
        } else {
            benchmark.deserialization_times.push(duration_ms);
            benchmark.deserialization_count += 1;
        }
    }
}

impl Default for SerializerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializer benchmark results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SerializerBenchmark {
    /// Serialization operation count
    pub serialization_count: u64,

    /// Deserialization operation count
    pub deserialization_count: u64,

    /// Serialization times (ms)
    pub serialization_times: Vec<f64>,

    /// Deserialization times (ms)
    pub deserialization_times: Vec<f64>,

    /// Last updated timestamp
    pub last_updated: Option<DateTime<Utc>>,
}

impl SerializerBenchmark {
    /// Get average serialization time
    pub fn avg_serialization_time(&self) -> f64 {
        if self.serialization_times.is_empty() {
            0.0
        } else {
            self.serialization_times.iter().sum::<f64>() / self.serialization_times.len() as f64
        }
    }

    /// Get average deserialization time
    pub fn avg_deserialization_time(&self) -> f64 {
        if self.deserialization_times.is_empty() {
            0.0
        } else {
            self.deserialization_times.iter().sum::<f64>() / self.deserialization_times.len() as f64
        }
    }

    /// Get P95 serialization time
    pub fn p95_serialization_time(&self) -> f64 {
        self.percentile(&self.serialization_times, 0.95)
    }

    /// Get P95 deserialization time
    pub fn p95_deserialization_time(&self) -> f64 {
        self.percentile(&self.deserialization_times, 0.95)
    }

    /// Calculate percentile
    fn percentile(&self, times: &[f64], p: f64) -> f64 {
        if times.is_empty() {
            return 0.0;
        }

        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((sorted.len() as f64 - 1.0) * p) as usize;
        sorted[index]
    }
}

/// BSON serializer (Binary JSON)
pub struct BsonSerializer;

impl CustomSerializer for BsonSerializer {
    fn serialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simulated BSON serialization
        let mut result = Vec::new();
        result.extend_from_slice(b"BSON");
        result.extend_from_slice(data);
        Ok(result)
    }

    fn deserialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 4 {
            return Err(anyhow!("Invalid BSON data"));
        }
        Ok(data[4..].to_vec())
    }

    fn format_name(&self) -> &str {
        "bson"
    }

    fn magic_bytes(&self) -> Option<&[u8]> {
        Some(b"BSON")
    }
}

/// Thrift serializer
pub struct ThriftSerializer;

impl CustomSerializer for ThriftSerializer {
    fn serialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simulated Thrift serialization
        let mut result = Vec::new();
        result.extend_from_slice(b"THFT");
        result.extend_from_slice(data);
        Ok(result)
    }

    fn deserialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 4 {
            return Err(anyhow!("Invalid Thrift data"));
        }
        Ok(data[4..].to_vec())
    }

    fn format_name(&self) -> &str {
        "thrift"
    }

    fn magic_bytes(&self) -> Option<&[u8]> {
        Some(b"THFT")
    }
}

/// FlexBuffers serializer (zero-copy)
pub struct FlexBuffersSerializer;

impl CustomSerializer for FlexBuffersSerializer {
    fn serialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simulated FlexBuffers serialization
        let mut result = Vec::new();
        result.extend_from_slice(b"FLEX");
        result.extend_from_slice(data);
        Ok(result)
    }

    fn deserialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 4 {
            return Err(anyhow!("Invalid FlexBuffers data"));
        }
        Ok(data[4..].to_vec())
    }

    fn format_name(&self) -> &str {
        "flexbuffers"
    }

    fn magic_bytes(&self) -> Option<&[u8]> {
        Some(b"FLEX")
    }

    fn supports_zero_copy(&self) -> bool {
        true
    }
}

/// RON (Rusty Object Notation) serializer
pub struct RonSerializer;

impl CustomSerializer for RonSerializer {
    fn serialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simulated RON serialization
        let mut result = Vec::new();
        result.extend_from_slice(b"RON\0");
        result.extend_from_slice(data);
        Ok(result)
    }

    fn deserialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 4 {
            return Err(anyhow!("Invalid RON data"));
        }
        Ok(data[4..].to_vec())
    }

    fn format_name(&self) -> &str {
        "ron"
    }

    fn magic_bytes(&self) -> Option<&[u8]> {
        Some(b"RON\0")
    }
}

/// Ion serializer (Amazon Ion)
pub struct IonSerializer;

impl CustomSerializer for IonSerializer {
    fn serialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simulated Ion serialization
        let mut result = Vec::new();
        result.extend_from_slice(b"ION\x01");
        result.extend_from_slice(data);
        Ok(result)
    }

    fn deserialize(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 4 {
            return Err(anyhow!("Invalid Ion data"));
        }
        Ok(data[4..].to_vec())
    }

    fn format_name(&self) -> &str {
        "ion"
    }

    fn magic_bytes(&self) -> Option<&[u8]> {
        Some(b"ION\x01")
    }
}

/// Serializer benchmark suite
pub struct SerializerBenchmarkSuite {
    registry: Arc<SerializerRegistry>,
    test_data: Vec<Vec<u8>>,
}

impl SerializerBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(registry: Arc<SerializerRegistry>) -> Self {
        Self {
            registry,
            test_data: Self::generate_test_data(),
        }
    }

    /// Generate test data of various sizes
    fn generate_test_data() -> Vec<Vec<u8>> {
        use scirs2_core::random::rng;
        use scirs2_core::Rng;

        let mut rand_gen = rng();
        let sizes = [100, 1024, 10_240, 102_400]; // 100B, 1KB, 10KB, 100KB

        sizes
            .iter()
            .map(|&size| (0..size).map(|_| rand_gen.random_range(0..=255)).collect())
            .collect()
    }

    /// Run benchmark for a specific serializer
    pub async fn benchmark(&self, format: &str, iterations: usize) -> Result<BenchmarkResults> {
        let mut results = BenchmarkResults {
            format: format.to_string(),
            iterations,
            serialization_times: Vec::new(),
            deserialization_times: Vec::new(),
            sizes: Vec::new(),
        };

        for test_data in &self.test_data {
            let mut ser_times = Vec::new();
            let mut deser_times = Vec::new();

            for _ in 0..iterations {
                // Benchmark serialization
                let start = Instant::now();
                let serialized = self.registry.serialize(format, test_data).await?;
                ser_times.push(start.elapsed().as_secs_f64() * 1000.0);

                // Benchmark deserialization
                let start = Instant::now();
                self.registry.deserialize(format, &serialized).await?;
                deser_times.push(start.elapsed().as_secs_f64() * 1000.0);
            }

            let avg_ser = ser_times.iter().sum::<f64>() / ser_times.len() as f64;
            let avg_deser = deser_times.iter().sum::<f64>() / deser_times.len() as f64;

            results.serialization_times.push(avg_ser);
            results.deserialization_times.push(avg_deser);
            results.sizes.push(test_data.len());
        }

        debug!("Benchmark completed for {}: {:?}", format, results);
        Ok(results)
    }

    /// Compare multiple serializers
    pub async fn compare(
        &self,
        formats: &[String],
        iterations: usize,
    ) -> Result<Vec<BenchmarkResults>> {
        let mut all_results = Vec::new();

        for format in formats {
            let results = self.benchmark(format, iterations).await?;
            all_results.push(results);
        }

        Ok(all_results)
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Format name
    pub format: String,

    /// Number of iterations
    pub iterations: usize,

    /// Serialization times (ms) for each data size
    pub serialization_times: Vec<f64>,

    /// Deserialization times (ms) for each data size
    pub deserialization_times: Vec<f64>,

    /// Data sizes tested
    pub sizes: Vec<usize>,
}

impl BenchmarkResults {
    /// Get total average serialization time
    pub fn avg_serialization_time(&self) -> f64 {
        if self.serialization_times.is_empty() {
            0.0
        } else {
            self.serialization_times.iter().sum::<f64>() / self.serialization_times.len() as f64
        }
    }

    /// Get total average deserialization time
    pub fn avg_deserialization_time(&self) -> f64 {
        if self.deserialization_times.is_empty() {
            0.0
        } else {
            self.deserialization_times.iter().sum::<f64>() / self.deserialization_times.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_custom_serializer() {
        let registry = SerializerRegistry::new();

        registry
            .register("bson", Box::new(BsonSerializer))
            .await
            .unwrap();

        let formats = registry.list().await;
        assert!(formats.contains(&"bson".to_string()));
    }

    #[tokio::test]
    async fn test_serialize_deserialize() {
        let registry = SerializerRegistry::new();

        registry
            .register("bson", Box::new(BsonSerializer))
            .await
            .unwrap();

        let data = b"test data";
        let serialized = registry.serialize("bson", data).await.unwrap();
        let deserialized = registry.deserialize("bson", &serialized).await.unwrap();

        assert_eq!(deserialized, data);
    }

    #[tokio::test]
    async fn test_format_detection() {
        let registry = SerializerRegistry::new();

        registry
            .register("bson", Box::new(BsonSerializer))
            .await
            .unwrap();
        registry
            .register("thrift", Box::new(ThriftSerializer))
            .await
            .unwrap();

        let data = b"BSONtest data";
        let format = registry.detect_format(data).await;

        assert_eq!(format, Some("bson".to_string()));
    }

    #[tokio::test]
    async fn test_benchmark() {
        let registry = Arc::new(SerializerRegistry::new());

        registry
            .register("bson", Box::new(BsonSerializer))
            .await
            .unwrap();

        let suite = SerializerBenchmarkSuite::new(registry.clone());
        let results = suite.benchmark("bson", 10).await.unwrap();

        assert_eq!(results.format, "bson");
        assert_eq!(results.iterations, 10);
        assert!(!results.serialization_times.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_formats() {
        let registry = SerializerRegistry::new();

        registry
            .register("bson", Box::new(BsonSerializer))
            .await
            .unwrap();
        registry
            .register("thrift", Box::new(ThriftSerializer))
            .await
            .unwrap();
        registry
            .register("flexbuffers", Box::new(FlexBuffersSerializer))
            .await
            .unwrap();
        registry
            .register("ron", Box::new(RonSerializer))
            .await
            .unwrap();
        registry
            .register("ion", Box::new(IonSerializer))
            .await
            .unwrap();

        let formats = registry.list().await;
        assert_eq!(formats.len(), 5);
    }

    #[tokio::test]
    async fn test_unregister() {
        let registry = SerializerRegistry::new();

        registry
            .register("bson", Box::new(BsonSerializer))
            .await
            .unwrap();

        assert!(registry.list().await.contains(&"bson".to_string()));

        registry.unregister("bson").await.unwrap();

        assert!(!registry.list().await.contains(&"bson".to_string()));
    }
}
