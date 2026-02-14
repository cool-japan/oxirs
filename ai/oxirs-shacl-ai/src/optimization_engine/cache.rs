//! Cache manager for constraint validation results

use crate::{shape::AiShape, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

use super::config::OptimizationConfig;
use super::types::CacheConfiguration;

/// Cache manager for constraint validation results
#[derive(Debug)]
pub struct CacheManager {
    constraint_cache: Arc<RwLock<HashMap<String, CachedConstraintResult>>>,
    shape_cache: Arc<RwLock<HashMap<String, CachedShapeResult>>>,
    cache_stats: Arc<Mutex<CacheStatistics>>,
    config: OptimizationConfig,
}

/// Cached constraint validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedConstraintResult {
    pub constraint_key: String,
    pub validation_result: bool,
    pub error_message: Option<String>,
    pub execution_time_ms: f64,
    pub cached_at: chrono::DateTime<chrono::Utc>,
    pub access_count: usize,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
}

/// Cached shape validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedShapeResult {
    pub shape_key: String,
    pub validation_success: bool,
    pub constraint_results: Vec<CachedConstraintResult>,
    pub total_execution_time_ms: f64,
    pub cached_at: chrono::DateTime<chrono::Utc>,
    pub access_count: usize,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_evictions: usize,
    pub average_lookup_time_ms: f64,
    pub memory_usage_bytes: usize,
}

impl CacheStatistics {
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_requests as f64
        }
    }
}

impl CacheManager {
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            constraint_cache: Arc::new(RwLock::new(HashMap::new())),
            shape_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(Mutex::new(CacheStatistics::default())),
            config,
        }
    }

    pub async fn configure_for_shape(
        &mut self,
        shape: &AiShape,
        config: &CacheConfiguration,
    ) -> Result<()> {
        tracing::info!(
            "Configuring cache for shape {} with {} cacheable constraints",
            shape.id(),
            config.cacheable_constraints.len()
        );
        Ok(())
    }

    pub async fn get_statistics(&self) -> Result<CacheStatistics> {
        let stats = self
            .cache_stats
            .lock()
            .expect("lock should not be poisoned")
            .clone();
        Ok(stats)
    }
}
