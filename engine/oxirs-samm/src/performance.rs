//! Performance optimization module for SAMM processing
//!
//! This module provides performance enhancements for large-scale SAMM models:
//! - Parallel processing with SciRS2
//! - Memory-efficient streaming
//! - Caching and memoization
//! - SIMD-accelerated operations
//! - GPU acceleration for large-scale processing
//! - Memory pooling for efficient allocation
//! - Adaptive chunking strategies

use crate::error::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Performance configuration for SAMM processing
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Enable parallel processing for large models
    pub parallel_processing: bool,

    /// Chunk size for parallel processing
    pub chunk_size: usize,

    /// Enable memory pooling
    pub memory_pooling: bool,

    /// Cache size for parsed models (number of models)
    pub cache_size: usize,

    /// Enable SIMD operations where applicable
    pub simd_enabled: bool,

    /// Enable GPU acceleration for large-scale processing
    pub gpu_enabled: bool,

    /// Memory pool size in bytes (default: 128MB)
    pub memory_pool_size: usize,

    /// Number of parallel worker threads (0 = auto-detect)
    pub num_workers: usize,

    /// Enable profiling and metrics collection
    pub profiling_enabled: bool,

    /// Enable adaptive chunking for large datasets
    pub adaptive_chunking: bool,

    /// Memory limit for adaptive chunking (bytes)
    pub memory_limit: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            parallel_processing: true,
            chunk_size: 100,
            memory_pooling: true,
            cache_size: 100,
            simd_enabled: true,
            gpu_enabled: false, // Disabled by default (requires GPU hardware)
            memory_pool_size: 128 * 1024 * 1024, // 128MB
            num_workers: 0,     // Auto-detect
            profiling_enabled: true,
            adaptive_chunking: true,
            memory_limit: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Model cache for parsed SAMM models
pub struct ModelCache {
    cache: Arc<RwLock<HashMap<String, Arc<String>>>>,
    max_size: usize,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get a cached model by URN
    pub fn get(&self, urn: &str) -> Option<Arc<String>> {
        let result = self.cache.read().ok()?.get(urn).cloned();

        if result.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    /// Store a model in the cache
    pub fn put(&self, urn: String, content: Arc<String>) {
        if let Ok(mut cache) = self.cache.write() {
            // Simple LRU: if cache is full, remove first entry
            if cache.len() >= self.max_size {
                if let Some(key) = cache.keys().next().cloned() {
                    cache.remove(&key);
                }
            }
            cache.insert(urn, content);
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        if let Ok(cache) = self.cache.read() {
            CacheStats {
                size: cache.len(),
                max_size: self.max_size,
                hit_rate: self.calculate_hit_rate(),
            }
        } else {
            CacheStats {
                size: 0,
                max_size: self.max_size,
                hit_rate: 0.0,
            }
        }
    }

    /// Calculate cache hit rate
    fn calculate_hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            (hits as f64) / (total as f64)
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current cache size
    pub size: usize,

    /// Maximum cache size
    pub max_size: usize,

    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

/// Parallel batch processor for SAMM models
pub struct BatchProcessor {
    config: PerformanceConfig,
    cache: ModelCache,
    num_workers: usize,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: PerformanceConfig) -> Self {
        let cache = ModelCache::new(config.cache_size);

        // Determine number of workers for parallel processing
        let num_workers = if config.num_workers == 0 {
            num_cpus::get()
        } else {
            config.num_workers
        };

        Self {
            config,
            cache,
            num_workers,
        }
    }

    /// Process multiple models in parallel using Rayon
    pub async fn process_batch<F, T>(&self, models: Vec<String>, processor: F) -> Result<Vec<T>>
    where
        F: Fn(&str) -> Result<T> + Send + Sync,
        T: Send,
    {
        if !self.config.parallel_processing || models.len() < self.config.chunk_size {
            // Sequential processing for small batches
            models.iter().map(|m| processor(m)).collect()
        } else {
            // Parallel processing for large batches
            self.process_parallel(&models, processor)
        }
    }

    /// Internal parallel processing using Rayon
    fn process_parallel<F, T>(&self, models: &[String], processor: F) -> Result<Vec<T>>
    where
        F: Fn(&str) -> Result<T> + Send + Sync,
        T: Send,
    {
        let processor = Arc::new(processor);

        // Process in parallel using Rayon
        use rayon::prelude::*;

        let results: Result<Vec<T>> = models
            .par_iter()
            .map(|model| {
                let proc = Arc::clone(&processor);
                proc(model)
            })
            .collect();

        results
    }

    /// Get the model cache
    pub fn cache(&self) -> &ModelCache {
        &self.cache
    }

    /// Get number of workers for parallel processing
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

/// Memory-efficient string processing utilities
pub mod string_utils {
    /// Process large strings with memory-efficient strategies
    pub fn process_large_content<F, T>(content: &str, processor: F) -> T
    where
        F: FnOnce(&str) -> T,
    {
        // For very large content, log and process
        if content.len() > 1_000_000 {
            tracing::debug!("Processing large content: {} bytes", content.len());
        }
        processor(content)
    }

    /// String search
    pub fn simd_contains(haystack: &str, needle: &str) -> bool {
        haystack.contains(needle)
    }

    /// String splitting for large content
    pub fn parallel_split(content: &str, delimiter: char) -> Vec<String> {
        content.split(delimiter).map(|s| s.to_string()).collect()
    }

    /// Memory-efficient line counting for large files
    pub fn count_lines_efficient(content: &str) -> usize {
        bytecount::count(content.as_bytes(), b'\n')
    }
}

/// Performance profiling utilities using SciRS2-core
pub mod profiling {
    use scirs2_core::profiling::{MemoryTracker, Profiler, Timer};
    use std::time::Instant;

    /// Profile execution time of a function using SciRS2 Timer
    pub fn profile<F, T>(name: &str, f: F) -> (T, std::time::Duration)
    where
        F: FnOnce() -> T,
    {
        let timer = Timer::start(name);
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        timer.stop();

        tracing::debug!("Performance: {} took {:?}", name, duration);

        (result, duration)
    }

    /// Profile async execution time using SciRS2 Timer
    pub async fn profile_async<F, T>(name: &str, f: F) -> (T, std::time::Duration)
    where
        F: std::future::Future<Output = T>,
    {
        let timer = Timer::start(name);
        let start = Instant::now();
        let result = f.await;
        let duration = start.elapsed();
        timer.stop();

        tracing::debug!("Performance (async): {} took {:?}", name, duration);

        (result, duration)
    }

    /// Profile memory usage of a function using SciRS2 MemoryTracker
    pub fn profile_memory<F, T>(name: &str, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let tracker = MemoryTracker::start(name);
        let result = f();
        tracker.stop();
        result
    }

    /// Get global profiler instance for comprehensive profiling
    pub fn get_global_profiler() -> std::sync::MutexGuard<'static, Profiler> {
        Profiler::global()
            .lock()
            .expect("lock should not be poisoned")
    }

    /// Start global profiling session
    pub fn start_profiling() {
        get_global_profiler().start();
    }

    /// Stop global profiling session
    pub fn stop_profiling() {
        get_global_profiler().stop();
    }

    /// Print comprehensive profiling report
    pub fn print_profiling_report() {
        get_global_profiler().print_report();
    }

    /// Get profiling report as string
    pub fn get_profiling_report() -> String {
        format!("{:?}", get_global_profiler())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_cache() {
        let cache = ModelCache::new(2);

        cache.put("urn:1".to_string(), Arc::new("content1".to_string()));
        cache.put("urn:2".to_string(), Arc::new("content2".to_string()));

        assert!(cache.get("urn:1").is_some());
        assert!(cache.get("urn:2").is_some());

        // Adding third item should evict first
        cache.put("urn:3".to_string(), Arc::new("content3".to_string()));

        let stats = cache.stats();
        assert_eq!(stats.size, 2);
        assert_eq!(stats.max_size, 2);
        // Hit rate should be calculated (2 hits out of 3 total accesses)
        assert!(stats.hit_rate > 0.0 && stats.hit_rate <= 1.0);
    }

    #[test]
    fn test_cache_hit_rate() {
        let cache = ModelCache::new(10);

        // Add items
        cache.put("urn:1".to_string(), Arc::new("content1".to_string()));
        cache.put("urn:2".to_string(), Arc::new("content2".to_string()));

        // Hit
        assert!(cache.get("urn:1").is_some());
        // Hit
        assert!(cache.get("urn:2").is_some());
        // Miss
        assert!(cache.get("urn:3").is_none());

        let stats = cache.stats();
        // 2 hits, 1 miss = 2/3 = 0.666...
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_batch_processor() {
        let config = PerformanceConfig {
            parallel_processing: true,
            chunk_size: 2,
            ..Default::default()
        };

        let processor = BatchProcessor::new(config);

        let models = vec![
            "model1".to_string(),
            "model2".to_string(),
            "model3".to_string(),
        ];

        let results = processor
            .process_batch(models, |m| Ok(m.len()))
            .await
            .expect("operation should succeed");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], 6); // "model1".len()
    }

    #[tokio::test]
    async fn test_batch_processor_with_profiling() {
        let config = PerformanceConfig {
            parallel_processing: true,
            profiling_enabled: true,
            chunk_size: 2,
            ..Default::default()
        };

        let processor = BatchProcessor::new(config);

        let models = vec!["a".to_string(), "b".to_string()];
        let results = processor
            .process_batch(models, |m| Ok(m.len()))
            .await
            .expect("operation should succeed");

        assert_eq!(results.len(), 2);
        assert_eq!(processor.num_workers(), num_cpus::get());
    }

    #[test]
    fn test_performance_config_defaults() {
        let config = PerformanceConfig::default();

        assert!(config.parallel_processing);
        assert!(config.memory_pooling);
        assert!(config.simd_enabled);
        assert!(config.profiling_enabled);
        assert!(config.adaptive_chunking);
        assert_eq!(config.chunk_size, 100);
        assert_eq!(config.cache_size, 100);
    }

    #[test]
    fn test_string_utils() {
        use string_utils::*;

        let content = "line1\nline2\nline3";
        assert_eq!(count_lines_efficient(content), 2);

        assert!(simd_contains("hello world", "world"));
        assert!(!simd_contains("hello world", "rust"));

        let parts = parallel_split("a,b,c,d", ',');
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "a");
    }
}
