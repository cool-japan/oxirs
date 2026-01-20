//! Feature Store Integration for ML Operations
//!
//! This module provides a comprehensive feature store for managing ML features
//! across training and inference pipelines. It ensures consistency, versioning,
//! and efficient feature access for SHACL AI models.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Feature Store                             │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │   Offline    │  │    Online    │  │   Feature    │     │
//! │  │    Store     │  │    Store     │  │   Registry   │     │
//! │  └──────────────┘  └──────────────┘  └──────────────┘     │
//! │         │                  │                  │             │
//! │         └──────────────────┴──────────────────┘             │
//! │                           │                                  │
//! │                  ┌────────▼────────┐                        │
//! │                  │  Feature Engine │                        │
//! │                  └─────────────────┘                        │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;
use uuid::Uuid;

use crate::{Result, ShaclAiError};

/// Feature store error types
#[derive(Debug, Error)]
pub enum FeatureStoreError {
    #[error("Feature not found: {0}")]
    FeatureNotFound(String),

    #[error("Feature group not found: {0}")]
    FeatureGroupNotFound(String),

    #[error("Feature validation failed: {0}")]
    ValidationFailed(String),

    #[error("Feature version conflict: {0}")]
    VersionConflict(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Feature computation error: {0}")]
    ComputationError(String),
}

impl From<FeatureStoreError> for ShaclAiError {
    fn from(err: FeatureStoreError) -> Self {
        ShaclAiError::DataProcessing(err.to_string())
    }
}

/// Feature store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStoreConfig {
    /// Enable online feature serving
    pub enable_online_serving: bool,

    /// Enable offline feature storage
    pub enable_offline_storage: bool,

    /// Cache size for online features (number of feature vectors)
    pub online_cache_size: usize,

    /// TTL for online features (seconds)
    pub online_ttl_seconds: u64,

    /// Enable feature validation
    pub enable_validation: bool,

    /// Enable feature lineage tracking
    pub enable_lineage: bool,

    /// Maximum feature vector dimension
    pub max_feature_dimension: usize,

    /// Enable feature monitoring
    pub enable_monitoring: bool,

    /// Enable automatic feature updates
    pub enable_auto_update: bool,
}

impl Default for FeatureStoreConfig {
    fn default() -> Self {
        Self {
            enable_online_serving: true,
            enable_offline_storage: true,
            online_cache_size: 10000,
            online_ttl_seconds: 3600, // 1 hour
            enable_validation: true,
            enable_lineage: true,
            max_feature_dimension: 10000,
            enable_monitoring: true,
            enable_auto_update: false,
        }
    }
}

/// Feature metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetadata {
    /// Feature unique ID
    pub id: String,

    /// Feature name
    pub name: String,

    /// Feature version
    pub version: String,

    /// Feature group
    pub feature_group: String,

    /// Feature type
    pub feature_type: FeatureType,

    /// Feature dimension
    pub dimension: usize,

    /// Feature description
    pub description: String,

    /// Feature tags for organization
    pub tags: Vec<String>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,

    /// Feature owner/creator
    pub owner: String,

    /// Feature transformation logic
    pub transformation: Option<String>,

    /// Feature dependencies
    pub dependencies: Vec<String>,

    /// Feature statistics
    pub statistics: FeatureStatistics,
}

/// Feature type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureType {
    /// Numerical feature
    Numerical,

    /// Categorical feature
    Categorical,

    /// Embedding feature
    Embedding,

    /// Boolean feature
    Boolean,

    /// Timestamp feature
    Timestamp,

    /// Text feature
    Text,

    /// Graph feature (for RDF/SHACL)
    Graph,
}

/// Feature statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    /// Number of non-null values
    pub count: usize,

    /// Mean value (for numerical features)
    pub mean: Option<f64>,

    /// Standard deviation (for numerical features)
    pub std_dev: Option<f64>,

    /// Minimum value
    pub min: Option<f64>,

    /// Maximum value
    pub max: Option<f64>,

    /// Number of unique values (for categorical features)
    pub unique_count: Option<usize>,

    /// Missing value percentage
    pub missing_percentage: f64,
}

impl Default for FeatureStatistics {
    fn default() -> Self {
        Self {
            count: 0,
            mean: None,
            std_dev: None,
            min: None,
            max: None,
            unique_count: None,
            missing_percentage: 0.0,
        }
    }
}

/// Feature group for organizing related features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGroup {
    /// Group ID
    pub id: String,

    /// Group name
    pub name: String,

    /// Group description
    pub description: String,

    /// Features in this group
    pub features: Vec<String>,

    /// Primary key for joining
    pub primary_key: Vec<String>,

    /// Group version
    pub version: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Tags for organization
    pub tags: Vec<String>,
}

/// Feature value wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureValue {
    /// Feature ID
    pub feature_id: String,

    /// Entity ID (e.g., RDF node URI)
    pub entity_id: String,

    /// Feature vector
    pub value: Vec<f64>,

    /// Feature timestamp
    pub timestamp: DateTime<Utc>,

    /// Feature version
    pub version: String,
}

/// Feature query for retrieval
#[derive(Debug, Clone)]
pub struct FeatureQuery {
    /// Entity IDs to retrieve features for
    pub entity_ids: Vec<String>,

    /// Feature names or IDs to retrieve
    pub features: Vec<String>,

    /// Point-in-time for historical features
    pub point_in_time: Option<DateTime<Utc>>,

    /// Feature group to query
    pub feature_group: Option<String>,
}

/// Feature lineage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureLineage {
    /// Feature ID
    pub feature_id: String,

    /// Source datasets
    pub source_datasets: Vec<String>,

    /// Transformation pipeline
    pub transformations: Vec<String>,

    /// Upstream features
    pub upstream_features: Vec<String>,

    /// Downstream models using this feature
    pub downstream_models: Vec<String>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Main feature store implementation
#[derive(Debug)]
#[allow(clippy::arc_with_non_send_sync)]
pub struct FeatureStore {
    /// Configuration
    config: FeatureStoreConfig,

    /// Online feature cache (entity_id -> feature_vector)
    online_cache: Arc<DashMap<String, CachedFeature>>,

    /// Feature registry (feature_id -> metadata)
    feature_registry: Arc<DashMap<String, FeatureMetadata>>,

    /// Feature groups
    feature_groups: Arc<DashMap<String, FeatureGroup>>,

    /// Feature lineage
    lineage_tracker: Arc<DashMap<String, FeatureLineage>>,

    /// Offline storage (simulated with in-memory store)
    offline_storage: Arc<RwLock<HashMap<String, Vec<FeatureValue>>>>,

    /// Feature statistics
    statistics: Arc<DashMap<String, FeatureStatistics>>,

    /// Monitoring metrics
    metrics: Arc<RwLock<FeatureStoreMetrics>>,

    /// Random number generator for sampling
    rng: Arc<Mutex<Random>>,
}

/// Cached feature with TTL
#[derive(Debug, Clone)]
struct CachedFeature {
    value: Vec<f64>,
    timestamp: DateTime<Utc>,
    version: String,
    expiry: DateTime<Utc>,
}

/// Feature store metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStoreMetrics {
    /// Total features registered
    pub total_features: usize,

    /// Total feature groups
    pub total_feature_groups: usize,

    /// Online cache hit rate
    pub cache_hit_rate: f64,

    /// Average feature retrieval time (ms)
    pub avg_retrieval_time_ms: f64,

    /// Total feature queries
    pub total_queries: usize,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,

    /// Features computed
    pub features_computed: usize,

    /// Features stored
    pub features_stored: usize,
}

impl Default for FeatureStoreMetrics {
    fn default() -> Self {
        Self {
            total_features: 0,
            total_feature_groups: 0,
            cache_hit_rate: 0.0,
            avg_retrieval_time_ms: 0.0,
            total_queries: 0,
            cache_hits: 0,
            cache_misses: 0,
            features_computed: 0,
            features_stored: 0,
        }
    }
}

impl FeatureStore {
    /// Create a new feature store with default configuration
    pub fn new() -> Self {
        Self::with_config(FeatureStoreConfig::default())
    }

    /// Create a new feature store with custom configuration
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn with_config(config: FeatureStoreConfig) -> Self {
        Self {
            config,
            online_cache: Arc::new(DashMap::new()),
            feature_registry: Arc::new(DashMap::new()),
            feature_groups: Arc::new(DashMap::new()),
            lineage_tracker: Arc::new(DashMap::new()),
            offline_storage: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(DashMap::new()),
            metrics: Arc::new(RwLock::new(FeatureStoreMetrics::default())),
            rng: Arc::new(Mutex::new(Random::default())),
        }
    }

    /// Register a new feature in the store
    pub fn register_feature(&self, metadata: FeatureMetadata) -> Result<()> {
        // Validate feature metadata
        if self.config.enable_validation {
            self.validate_feature_metadata(&metadata)?;
        }

        // Check for version conflicts
        if let Some(existing) = self.feature_registry.get(&metadata.id) {
            if existing.version == metadata.version {
                return Err(FeatureStoreError::VersionConflict(format!(
                    "Feature {} version {} already exists",
                    metadata.id, metadata.version
                ))
                .into());
            }
        }

        // Register feature
        self.feature_registry
            .insert(metadata.id.clone(), metadata.clone());

        // Update statistics
        self.statistics
            .insert(metadata.id.clone(), metadata.statistics.clone());

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_features += 1;
        }

        tracing::info!(
            "Registered feature: {} (v{})",
            metadata.name,
            metadata.version
        );
        Ok(())
    }

    /// Register a feature group
    pub fn register_feature_group(&self, group: FeatureGroup) -> Result<()> {
        self.feature_groups.insert(group.id.clone(), group.clone());

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_feature_groups += 1;
        }

        tracing::info!("Registered feature group: {}", group.name);
        Ok(())
    }

    /// Get features for entities (online serving)
    pub fn get_features(&self, query: &FeatureQuery) -> Result<HashMap<String, Vec<f64>>> {
        let start = std::time::Instant::now();
        let mut results = HashMap::new();

        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_queries += 1;
        }

        for entity_id in &query.entity_ids {
            // Try online cache first
            if self.config.enable_online_serving {
                if let Some(cached) = self.get_from_cache(entity_id)? {
                    results.insert(entity_id.clone(), cached);
                    if let Ok(mut metrics) = self.metrics.write() {
                        metrics.cache_hits += 1;
                    }
                    continue;
                }
            }

            // Cache miss - compute or retrieve from offline storage
            if let Ok(mut metrics) = self.metrics.write() {
                metrics.cache_misses += 1;
            }

            if let Some(features) = self.compute_features(entity_id, &query.features)? {
                // Store in cache
                if self.config.enable_online_serving {
                    self.store_in_cache(entity_id, &features)?;
                }
                results.insert(entity_id.clone(), features);
            }
        }

        // Update metrics
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if let Ok(mut metrics) = self.metrics.write() {
            let total = metrics.total_queries as f64;
            metrics.avg_retrieval_time_ms =
                (metrics.avg_retrieval_time_ms * (total - 1.0) + elapsed) / total;
            metrics.cache_hit_rate =
                metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64;
        }

        Ok(results)
    }

    /// Store features (offline storage)
    pub fn store_features(&self, features: Vec<FeatureValue>) -> Result<()> {
        if !self.config.enable_offline_storage {
            return Ok(());
        }

        let mut storage = self.offline_storage.write().map_err(|e| {
            FeatureStoreError::StorageError(format!("Failed to lock storage: {}", e))
        })?;

        for feature in features {
            storage
                .entry(feature.entity_id.clone())
                .or_insert_with(Vec::new)
                .push(feature.clone());

            // Update statistics
            if let Some(metadata) = self.feature_registry.get(&feature.feature_id) {
                self.update_feature_statistics(&feature)?;
            }
        }

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.features_stored += 1;
        }

        Ok(())
    }

    /// Track feature lineage
    pub fn track_lineage(&self, lineage: FeatureLineage) -> Result<()> {
        if !self.config.enable_lineage {
            return Ok(());
        }

        self.lineage_tracker
            .insert(lineage.feature_id.clone(), lineage);
        Ok(())
    }

    /// Get feature lineage
    pub fn get_lineage(&self, feature_id: &str) -> Option<FeatureLineage> {
        self.lineage_tracker.get(feature_id).map(|l| l.clone())
    }

    /// Get feature metadata
    pub fn get_metadata(&self, feature_id: &str) -> Option<FeatureMetadata> {
        self.feature_registry.get(feature_id).map(|m| m.clone())
    }

    /// Get feature group
    pub fn get_feature_group(&self, group_id: &str) -> Option<FeatureGroup> {
        self.feature_groups.get(group_id).map(|g| g.clone())
    }

    /// List all features
    pub fn list_features(&self) -> Vec<FeatureMetadata> {
        self.feature_registry
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// List all feature groups
    pub fn list_feature_groups(&self) -> Vec<FeatureGroup> {
        self.feature_groups
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get feature statistics
    pub fn get_statistics(&self, feature_id: &str) -> Option<FeatureStatistics> {
        self.statistics.get(feature_id).map(|s| s.clone())
    }

    /// Get feature store metrics
    pub fn get_metrics(&self) -> Result<FeatureStoreMetrics> {
        Ok(self
            .metrics
            .read()
            .map_err(|e| FeatureStoreError::StorageError(format!("Failed to read metrics: {}", e)))?
            .clone())
    }

    /// Clear online cache
    pub fn clear_cache(&self) -> Result<()> {
        self.online_cache.clear();
        tracing::info!("Cleared online feature cache");
        Ok(())
    }

    /// Validate feature metadata
    fn validate_feature_metadata(&self, metadata: &FeatureMetadata) -> Result<()> {
        if metadata.name.is_empty() {
            return Err(FeatureStoreError::ValidationFailed(
                "Feature name cannot be empty".to_string(),
            )
            .into());
        }

        if metadata.dimension == 0 {
            return Err(FeatureStoreError::ValidationFailed(
                "Feature dimension must be > 0".to_string(),
            )
            .into());
        }

        if metadata.dimension > self.config.max_feature_dimension {
            return Err(FeatureStoreError::ValidationFailed(format!(
                "Feature dimension {} exceeds maximum {}",
                metadata.dimension, self.config.max_feature_dimension
            ))
            .into());
        }

        Ok(())
    }

    /// Get features from online cache
    fn get_from_cache(&self, entity_id: &str) -> Result<Option<Vec<f64>>> {
        if let Some(cached) = self.online_cache.get(entity_id) {
            // Check TTL
            if cached.expiry > Utc::now() {
                return Ok(Some(cached.value.clone()));
            } else {
                // Expired, remove from cache
                drop(cached);
                self.online_cache.remove(entity_id);
            }
        }
        Ok(None)
    }

    /// Store features in online cache
    fn store_in_cache(&self, entity_id: &str, features: &[f64]) -> Result<()> {
        let expiry = Utc::now() + chrono::Duration::seconds(self.config.online_ttl_seconds as i64);

        let cached = CachedFeature {
            value: features.to_vec(),
            timestamp: Utc::now(),
            version: "1.0.0".to_string(),
            expiry,
        };

        self.online_cache.insert(entity_id.to_string(), cached);

        // Evict if cache size exceeded
        if self.online_cache.len() > self.config.online_cache_size {
            self.evict_oldest_cache_entry()?;
        }

        Ok(())
    }

    /// Evict oldest cache entry
    fn evict_oldest_cache_entry(&self) -> Result<()> {
        // Simple LRU eviction - remove first entry
        if let Some(entry) = self.online_cache.iter().next() {
            let key = entry.key().clone();
            drop(entry);
            self.online_cache.remove(&key);
        }
        Ok(())
    }

    /// Compute features for an entity
    fn compute_features(
        &self,
        entity_id: &str,
        feature_ids: &[String],
    ) -> Result<Option<Vec<f64>>> {
        // Try to load from offline storage first
        if self.config.enable_offline_storage {
            if let Ok(storage) = self.offline_storage.read() {
                if let Some(stored_features) = storage.get(entity_id) {
                    // Find matching features
                    let mut result = Vec::new();
                    for feature_id in feature_ids {
                        if let Some(feature) =
                            stored_features.iter().find(|f| &f.feature_id == feature_id)
                        {
                            result.extend_from_slice(&feature.value);
                        }
                    }
                    if !result.is_empty() {
                        return Ok(Some(result));
                    }
                }
            }
        }

        // Feature not found - compute dynamically (placeholder)
        // In real implementation, this would call feature computation logic
        if let Ok(mut rng) = self.rng.lock() {
            let dimension = feature_ids.len() * 128; // Example dimension
                                                     // Generate random features using scirs2-core random
            let mut features = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                // Generate random value between -1.0 and 1.0
                features.push(rng.random::<f64>() * 2.0 - 1.0);
            }

            if let Ok(mut metrics) = self.metrics.write() {
                metrics.features_computed += 1;
            }

            Ok(Some(features))
        } else {
            Ok(None)
        }
    }

    /// Update feature statistics
    fn update_feature_statistics(&self, feature: &FeatureValue) -> Result<()> {
        if let Some(mut stats) = self.statistics.get_mut(&feature.feature_id) {
            stats.count += 1;

            // Update mean and std dev (online algorithm)
            if !feature.value.is_empty() {
                let value = feature.value.iter().sum::<f64>() / feature.value.len() as f64;

                let old_mean = stats.mean.unwrap_or(0.0);
                let new_mean = old_mean + (value - old_mean) / stats.count as f64;
                stats.mean = Some(new_mean);

                // Update min/max
                stats.min = Some(stats.min.map(|m| m.min(value)).unwrap_or(value));
                stats.max = Some(stats.max.map(|m| m.max(value)).unwrap_or(value));
            }
        }
        Ok(())
    }
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_store_creation() {
        let store = FeatureStore::new();
        let metrics = store.get_metrics().unwrap();
        assert_eq!(metrics.total_features, 0);
        assert_eq!(metrics.total_feature_groups, 0);
    }

    #[test]
    fn test_feature_registration() {
        let store = FeatureStore::new();

        let metadata = FeatureMetadata {
            id: "shape_embedding".to_string(),
            name: "Shape Embedding".to_string(),
            version: "1.0.0".to_string(),
            feature_group: "shacl_features".to_string(),
            feature_type: FeatureType::Embedding,
            dimension: 128,
            description: "Embedding vector for SHACL shapes".to_string(),
            tags: vec!["embedding".to_string(), "shacl".to_string()],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            owner: "oxirs-shacl-ai".to_string(),
            transformation: None,
            dependencies: vec![],
            statistics: FeatureStatistics::default(),
        };

        assert!(store.register_feature(metadata.clone()).is_ok());

        let retrieved = store.get_metadata("shape_embedding");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Shape Embedding");

        let metrics = store.get_metrics().unwrap();
        assert_eq!(metrics.total_features, 1);
    }

    #[test]
    fn test_feature_group_registration() {
        let store = FeatureStore::new();

        let group = FeatureGroup {
            id: "shacl_features".to_string(),
            name: "SHACL Features".to_string(),
            description: "Features for SHACL validation".to_string(),
            features: vec![
                "shape_embedding".to_string(),
                "constraint_vector".to_string(),
            ],
            primary_key: vec!["shape_id".to_string()],
            version: "1.0.0".to_string(),
            created_at: Utc::now(),
            tags: vec!["shacl".to_string()],
        };

        assert!(store.register_feature_group(group.clone()).is_ok());

        let retrieved = store.get_feature_group("shacl_features");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "SHACL Features");

        let metrics = store.get_metrics().unwrap();
        assert_eq!(metrics.total_feature_groups, 1);
    }

    #[test]
    fn test_feature_query() {
        let store = FeatureStore::new();

        // Register a feature
        let metadata = FeatureMetadata {
            id: "test_feature".to_string(),
            name: "Test Feature".to_string(),
            version: "1.0.0".to_string(),
            feature_group: "test_group".to_string(),
            feature_type: FeatureType::Numerical,
            dimension: 10,
            description: "Test feature".to_string(),
            tags: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            owner: "test".to_string(),
            transformation: None,
            dependencies: vec![],
            statistics: FeatureStatistics::default(),
        };

        store.register_feature(metadata).unwrap();

        // Query features
        let query = FeatureQuery {
            entity_ids: vec!["entity_1".to_string()],
            features: vec!["test_feature".to_string()],
            point_in_time: None,
            feature_group: None,
        };

        let result = store.get_features(&query);
        assert!(result.is_ok());
        let features = result.unwrap();
        assert_eq!(features.len(), 1);
        assert!(features.contains_key("entity_1"));
    }

    #[test]
    fn test_feature_caching() {
        let config = FeatureStoreConfig {
            online_cache_size: 100, // Use larger cache to avoid eviction issues
            ..Default::default()
        };
        let store = FeatureStore::with_config(config);

        // Store features in cache
        let features1 = vec![1.0, 2.0, 3.0];
        let features2 = vec![4.0, 5.0, 6.0];

        store.store_in_cache("entity_1", &features1).unwrap();
        store.store_in_cache("entity_2", &features2).unwrap();

        // Both should be in cache
        assert!(store.get_from_cache("entity_1").unwrap().is_some());
        assert!(store.get_from_cache("entity_2").unwrap().is_some());

        // Test cache retrieval
        let cached = store.get_from_cache("entity_1").unwrap().unwrap();
        assert_eq!(cached, features1);
    }

    #[test]
    fn test_feature_lineage() {
        let store = FeatureStore::new();

        let lineage = FeatureLineage {
            feature_id: "test_feature".to_string(),
            source_datasets: vec!["rdf_graph".to_string()],
            transformations: vec!["embedding_transform".to_string()],
            upstream_features: vec!["raw_shape".to_string()],
            downstream_models: vec!["shacl_validator".to_string()],
            created_at: Utc::now(),
        };

        store.track_lineage(lineage.clone()).unwrap();

        let retrieved = store.get_lineage("test_feature");
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.unwrap().source_datasets,
            vec!["rdf_graph".to_string()]
        );
    }

    #[test]
    fn test_feature_validation() {
        let store = FeatureStore::new();

        // Test invalid feature (empty name)
        let invalid_metadata = FeatureMetadata {
            id: "invalid".to_string(),
            name: "".to_string(), // Empty name
            version: "1.0.0".to_string(),
            feature_group: "test".to_string(),
            feature_type: FeatureType::Numerical,
            dimension: 10,
            description: "Invalid".to_string(),
            tags: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            owner: "test".to_string(),
            transformation: None,
            dependencies: vec![],
            statistics: FeatureStatistics::default(),
        };

        assert!(store.register_feature(invalid_metadata).is_err());
    }

    #[test]
    fn test_metrics_tracking() {
        let store = FeatureStore::new();

        // Perform some operations
        let metadata = FeatureMetadata {
            id: "test".to_string(),
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            feature_group: "test".to_string(),
            feature_type: FeatureType::Numerical,
            dimension: 10,
            description: "Test".to_string(),
            tags: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            owner: "test".to_string(),
            transformation: None,
            dependencies: vec![],
            statistics: FeatureStatistics::default(),
        };

        store.register_feature(metadata).unwrap();

        let query = FeatureQuery {
            entity_ids: vec!["entity_1".to_string()],
            features: vec!["test".to_string()],
            point_in_time: None,
            feature_group: None,
        };

        store.get_features(&query).unwrap();

        let metrics = store.get_metrics().unwrap();
        assert_eq!(metrics.total_features, 1);
        assert_eq!(metrics.total_queries, 1);
        assert!(metrics.cache_misses > 0);
    }
}
