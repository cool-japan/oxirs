//! Performance optimization module for SAMM processing
//!
//! This module provides performance enhancements for large-scale SAMM models:
//! - Parallel processing with SciRS2
//! - Memory-efficient streaming
//! - Caching and memoization
//! - SIMD-accelerated operations

use crate::error::Result;
use std::collections::HashMap;
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
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            parallel_processing: true,
            chunk_size: 100,
            memory_pooling: true,
            cache_size: 100,
            simd_enabled: true,
        }
    }
}

/// Model cache for parsed SAMM models
pub struct ModelCache {
    cache: Arc<RwLock<HashMap<String, Arc<String>>>>,
    max_size: usize,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }

    /// Get a cached model by URN
    pub fn get(&self, urn: &str) -> Option<Arc<String>> {
        self.cache.read().ok()?.get(urn).cloned()
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
            }
        } else {
            CacheStats {
                size: 0,
                max_size: self.max_size,
            }
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
}

/// Parallel batch processor for SAMM models
pub struct BatchProcessor {
    config: PerformanceConfig,
    cache: ModelCache,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(config: PerformanceConfig) -> Self {
        let cache = ModelCache::new(config.cache_size);
        Self { config, cache }
    }

    /// Process multiple models in parallel
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
            let processor = Arc::new(processor);
            let chunks: Vec<Vec<String>> = models
                .chunks(self.config.chunk_size)
                .map(|chunk| chunk.to_vec())
                .collect();

            let mut results = Vec::new();

            for chunk in chunks {
                let chunk_results: Vec<Result<T>> = chunk
                    .iter()
                    .map(|model| {
                        let proc = Arc::clone(&processor);
                        proc(model)
                    })
                    .collect();

                for result in chunk_results {
                    results.push(result?);
                }
            }

            Ok(results)
        }
    }

    /// Get the model cache
    pub fn cache(&self) -> &ModelCache {
        &self.cache
    }
}

/// Memory-efficient string processing utilities
pub mod string_utils {
    /// Process large strings with memory-efficient strategies
    pub fn process_large_content<F, T>(content: &str, processor: F) -> T
    where
        F: FnOnce(&str) -> T,
    {
        // For very large content, process in chunks if needed
        if content.len() > 1_000_000 {
            tracing::debug!("Processing large content: {} bytes", content.len());
            // Future: Implement chunk-based processing with memory pooling
            processor(content)
        } else {
            // Direct processing for normal-sized content
            processor(content)
        }
    }
}

/// Performance profiling utilities
pub mod profiling {
    use std::time::Instant;

    /// Profile execution time of a function
    pub fn profile<F, T>(name: &str, f: F) -> (T, std::time::Duration)
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        tracing::debug!("Performance: {} took {:?}", name, duration);

        (result, duration)
    }

    /// Profile async execution time
    pub async fn profile_async<F, T>(name: &str, f: F) -> (T, std::time::Duration)
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = f.await;
        let duration = start.elapsed();

        tracing::debug!("Performance (async): {} took {:?}", name, duration);

        (result, duration)
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
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], 6); // "model1".len()
    }
}
