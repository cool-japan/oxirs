//! Main tiering manager that coordinates tier operations

use super::access_tracker::AccessTracker;
use super::config::TieringConfig;
use super::metrics::TierMetrics;
use super::policies::TierTransitionReason;
use super::storage_backends::{ColdTierStorage, HotTierStorage, StorageBackend, WarmTierStorage};
use super::tier_optimizer::{TierOptimizationRecommendation, TierOptimizer};
use super::types::{IndexMetadata, StorageTier, TierStatistics, TierTransition};
use anyhow::Result;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Main tiering manager
pub struct TieringManager {
    /// Configuration
    config: TieringConfig,
    /// Index metadata registry
    indices: Arc<RwLock<HashMap<String, IndexMetadata>>>,
    /// Hot tier storage
    hot_storage: Arc<RwLock<Box<dyn StorageBackend>>>,
    /// Warm tier storage
    warm_storage: Arc<RwLock<Box<dyn StorageBackend>>>,
    /// Cold tier storage
    cold_storage: Arc<RwLock<Box<dyn StorageBackend>>>,
    /// Access tracker
    access_tracker: Arc<RwLock<AccessTracker>>,
    /// Metrics collector
    metrics: Arc<TierMetrics>,
    /// Tier optimizer
    optimizer: Arc<RwLock<TierOptimizer>>,
    /// Background task handle (for cleanup and optimization)
    _background_task: Option<std::thread::JoinHandle<()>>,
}

impl TieringManager {
    /// Create a new tiering manager
    pub fn new(config: TieringConfig) -> Result<Self> {
        config.validate()?;

        // Create storage backends
        let hot_storage: Box<dyn StorageBackend> = Box::new(HotTierStorage::new());

        let warm_storage: Box<dyn StorageBackend> = Box::new(WarmTierStorage::new(
            config.storage_base_path.join("warm"),
            config.warm_tier_compression,
            config.warm_tier_compression_level,
        )?);

        let cold_storage: Box<dyn StorageBackend> = Box::new(ColdTierStorage::new(
            config.storage_base_path.join("cold"),
            config.cold_tier_compression_level,
        )?);

        // Create access tracker
        let access_tracker = AccessTracker::new(10000, Duration::from_secs(86400));

        // Create tier optimizer
        let optimizer = TierOptimizer::new(config.clone());

        // Initialize metrics
        let metrics = Arc::new(TierMetrics::new());

        // Set initial tier capacities
        metrics.update_tier_usage(StorageTier::Hot, 0, config.hot_tier_capacity_bytes());
        metrics.update_tier_usage(StorageTier::Warm, 0, config.warm_tier_capacity_bytes());
        metrics.update_tier_usage(StorageTier::Cold, 0, config.cold_tier_capacity_bytes());

        Ok(Self {
            config,
            indices: Arc::new(RwLock::new(HashMap::new())),
            hot_storage: Arc::new(RwLock::new(hot_storage)),
            warm_storage: Arc::new(RwLock::new(warm_storage)),
            cold_storage: Arc::new(RwLock::new(cold_storage)),
            access_tracker: Arc::new(RwLock::new(access_tracker)),
            metrics,
            optimizer: Arc::new(RwLock::new(optimizer)),
            _background_task: None,
        })
    }

    /// Register a new index with metadata
    pub fn register_index(&self, index_id: String, metadata: IndexMetadata) -> Result<()> {
        let mut indices = self.indices.write();
        indices.insert(index_id, metadata);
        Ok(())
    }

    /// Store index data in appropriate tier
    pub fn store_index(&self, index_id: &str, data: &[u8], tier: StorageTier) -> Result<()> {
        let start = SystemTime::now();

        // Store in appropriate backend
        let storage = match tier {
            StorageTier::Hot => self.hot_storage.clone(),
            StorageTier::Warm => self.warm_storage.clone(),
            StorageTier::Cold => self.cold_storage.clone(),
        };

        storage.write().save_index(index_id, data)?;

        // Update metadata
        let mut indices = self.indices.write();
        if let Some(metadata) = indices.get_mut(index_id) {
            metadata.current_tier = tier;
            metadata.size_bytes = data.len() as u64;
            metadata.last_modified = SystemTime::now();
        }

        // Update metrics
        let duration = start.elapsed().unwrap_or(Duration::ZERO);
        self.metrics.record_bytes_written(tier, data.len() as u64);

        tracing::info!(
            "Stored index {} in {:?} tier ({} bytes, {:?})",
            index_id,
            tier,
            data.len(),
            duration
        );

        Ok(())
    }

    /// Load index data from its current tier
    pub fn load_index(&self, index_id: &str) -> Result<Vec<u8>> {
        let start = SystemTime::now();

        // Get current tier
        let tier = {
            let indices = self.indices.read();
            indices
                .get(index_id)
                .map(|m| m.current_tier)
                .ok_or_else(|| anyhow::anyhow!("Index {} not found", index_id))?
        };

        // Load from appropriate backend
        let storage = match tier {
            StorageTier::Hot => self.hot_storage.clone(),
            StorageTier::Warm => self.warm_storage.clone(),
            StorageTier::Cold => self.cold_storage.clone(),
        };

        let data = storage.read().load_index(index_id)?;

        // Record access
        let latency_us = start.elapsed().unwrap_or(Duration::ZERO).as_micros() as u64;
        self.access_tracker
            .write()
            .record_access(index_id, latency_us);

        // Update metrics
        self.metrics.record_query(tier, latency_us, true);
        self.metrics.record_bytes_read(tier, data.len() as u64);

        // Update metadata
        {
            let mut indices = self.indices.write();
            if let Some(metadata) = indices.get_mut(index_id) {
                self.access_tracker.read().update_metadata(metadata);
            }
        }

        Ok(data)
    }

    /// Transition an index between tiers
    pub fn transition_index(
        &self,
        index_id: &str,
        target_tier: StorageTier,
        reason: TierTransitionReason,
    ) -> Result<()> {
        let start = SystemTime::now();

        // Get current state
        let (current_tier, size_bytes) = {
            let indices = self.indices.read();
            let metadata = indices
                .get(index_id)
                .ok_or_else(|| anyhow::anyhow!("Index {} not found", index_id))?;
            (metadata.current_tier, metadata.size_bytes)
        };

        if current_tier == target_tier {
            return Ok(()); // Already in target tier
        }

        // Load from current tier
        let source_storage = match current_tier {
            StorageTier::Hot => self.hot_storage.clone(),
            StorageTier::Warm => self.warm_storage.clone(),
            StorageTier::Cold => self.cold_storage.clone(),
        };

        let data = source_storage.read().load_index(index_id)?;

        // Save to target tier
        let target_storage = match target_tier {
            StorageTier::Hot => self.hot_storage.clone(),
            StorageTier::Warm => self.warm_storage.clone(),
            StorageTier::Cold => self.cold_storage.clone(),
        };

        target_storage.write().save_index(index_id, &data)?;

        // Delete from source tier (unless it's a promotion and we want gradual transition)
        if !self.config.gradual_transition.enabled {
            source_storage.write().delete_index(index_id)?;
        }

        // Update metadata
        {
            let mut indices = self.indices.write();
            if let Some(metadata) = indices.get_mut(index_id) {
                metadata.current_tier = target_tier;
                metadata.last_modified = SystemTime::now();
            }
        }

        // Record transition
        let duration = start.elapsed().unwrap_or(Duration::ZERO);
        let transition = TierTransition {
            index_id: index_id.to_string(),
            from_tier: current_tier,
            to_tier: target_tier,
            reason: format!("{:?}", reason),
            timestamp: SystemTime::now(),
            duration,
            success: true,
            error: None,
        };

        self.metrics.record_transition(transition);

        // Update tier usage
        self.update_tier_usage_metrics();

        tracing::info!(
            "Transitioned index {} from {:?} to {:?} ({} bytes, {:?})",
            index_id,
            current_tier,
            target_tier,
            size_bytes,
            duration
        );

        Ok(())
    }

    /// Run tier optimization and return recommendations
    pub fn optimize_tiers(&self) -> Result<Vec<TierOptimizationRecommendation>> {
        let indices: Vec<IndexMetadata> = {
            let indices = self.indices.read();
            indices.values().cloned().collect()
        };

        let tier_stats = self.get_tier_statistics_array();

        let mut optimizer = self.optimizer.write();
        let recommendations = optimizer.optimize_tier_placements(&indices, &tier_stats);

        Ok(recommendations)
    }

    /// Apply optimization recommendations automatically
    pub fn apply_optimizations(&self, limit: Option<usize>) -> Result<Vec<String>> {
        let recommendations = self.optimize_tiers()?;

        let mut applied = Vec::new();
        let limit = limit.unwrap_or(usize::MAX);

        for (i, rec) in recommendations.iter().enumerate() {
            if i >= limit {
                break;
            }

            // Check if transition is worthwhile
            if rec.priority < 0.5 {
                continue; // Skip low-priority transitions
            }

            match self.transition_index(&rec.index_id, rec.recommended_tier, rec.reason.clone()) {
                Ok(_) => {
                    applied.push(rec.index_id.clone());
                }
                Err(e) => {
                    tracing::warn!("Failed to transition index {}: {}", rec.index_id, e);
                }
            }
        }

        Ok(applied)
    }

    /// Get statistics for all tiers
    pub fn get_tier_statistics(&self) -> HashMap<StorageTier, TierStatistics> {
        self.metrics.get_all_tier_statistics()
    }

    /// Get tier statistics as array [Hot, Warm, Cold]
    fn get_tier_statistics_array(&self) -> [TierStatistics; 3] {
        [
            self.metrics.get_tier_statistics(StorageTier::Hot),
            self.metrics.get_tier_statistics(StorageTier::Warm),
            self.metrics.get_tier_statistics(StorageTier::Cold),
        ]
    }

    /// Update tier usage metrics
    fn update_tier_usage_metrics(&self) {
        // Calculate usage for each tier
        let indices = self.indices.read();

        let mut hot_usage = 0u64;
        let mut warm_usage = 0u64;
        let mut cold_usage = 0u64;

        let mut hot_count = 0;
        let mut warm_count = 0;
        let mut cold_count = 0;

        for metadata in indices.values() {
            match metadata.current_tier {
                StorageTier::Hot => {
                    hot_usage += metadata.size_bytes;
                    hot_count += 1;
                }
                StorageTier::Warm => {
                    warm_usage += metadata.size_bytes;
                    warm_count += 1;
                }
                StorageTier::Cold => {
                    cold_usage += metadata.size_bytes;
                    cold_count += 1;
                }
            }
        }

        self.metrics.update_tier_usage(
            StorageTier::Hot,
            hot_usage,
            self.config.hot_tier_capacity_bytes(),
        );
        self.metrics.update_tier_usage(
            StorageTier::Warm,
            warm_usage,
            self.config.warm_tier_capacity_bytes(),
        );
        self.metrics.update_tier_usage(
            StorageTier::Cold,
            cold_usage,
            self.config.cold_tier_capacity_bytes(),
        );

        self.metrics.update_index_count(StorageTier::Hot, hot_count);
        self.metrics
            .update_index_count(StorageTier::Warm, warm_count);
        self.metrics
            .update_index_count(StorageTier::Cold, cold_count);
    }

    /// Get index metadata
    pub fn get_index_metadata(&self, index_id: &str) -> Option<IndexMetadata> {
        let indices = self.indices.read();
        indices.get(index_id).cloned()
    }

    /// List all registered indices
    pub fn list_indices(&self) -> Vec<String> {
        let indices = self.indices.read();
        indices.keys().cloned().collect()
    }

    /// Get metrics
    pub fn get_metrics(&self) -> Arc<TierMetrics> {
        self.metrics.clone()
    }

    /// Cleanup old access history
    pub fn cleanup_history(&self) {
        let mut tracker = self.access_tracker.write();
        tracker.cleanup_old_entries(self.config.metrics_retention);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiering::types::{AccessStatistics, IndexType, PerformanceMetrics};
    use std::collections::HashMap;

    fn create_test_config() -> TieringConfig {
        TieringConfig::development()
    }

    fn create_test_metadata(id: &str, tier: StorageTier) -> IndexMetadata {
        IndexMetadata {
            index_id: id.to_string(),
            current_tier: tier,
            size_bytes: 1024 * 1024, // 1 MB
            compressed_size_bytes: 512 * 1024,
            vector_count: 10_000,
            dimension: 128,
            index_type: IndexType::Hnsw,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            last_modified: SystemTime::now(),
            access_stats: AccessStatistics::default(),
            performance_metrics: PerformanceMetrics::default(),
            storage_path: None,
            custom_metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_tiering_manager_creation() {
        let config = create_test_config();
        let manager = TieringManager::new(config).unwrap();

        let stats = manager.get_tier_statistics();
        assert!(stats.contains_key(&StorageTier::Hot));
        assert!(stats.contains_key(&StorageTier::Warm));
        assert!(stats.contains_key(&StorageTier::Cold));
    }

    #[test]
    fn test_register_and_store_index() {
        let config = create_test_config();
        let manager = TieringManager::new(config).unwrap();

        let metadata = create_test_metadata("test_index", StorageTier::Hot);
        manager
            .register_index("test_index".to_string(), metadata)
            .unwrap();

        let data = vec![1, 2, 3, 4, 5];
        manager
            .store_index("test_index", &data, StorageTier::Hot)
            .unwrap();

        let loaded = manager.load_index("test_index").unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_tier_transition() {
        let config = create_test_config();
        let manager = TieringManager::new(config).unwrap();

        let metadata = create_test_metadata("test_index", StorageTier::Hot);
        manager
            .register_index("test_index".to_string(), metadata)
            .unwrap();

        let data = vec![1, 2, 3, 4, 5];
        manager
            .store_index("test_index", &data, StorageTier::Hot)
            .unwrap();

        // Transition to warm tier
        manager
            .transition_index(
                "test_index",
                StorageTier::Warm,
                TierTransitionReason::CostOptimization,
            )
            .unwrap();

        let metadata = manager.get_index_metadata("test_index").unwrap();
        assert_eq!(metadata.current_tier, StorageTier::Warm);
    }

    #[test]
    fn test_list_indices() {
        let config = create_test_config();
        let manager = TieringManager::new(config).unwrap();

        let metadata1 = create_test_metadata("index1", StorageTier::Hot);
        let metadata2 = create_test_metadata("index2", StorageTier::Warm);

        manager
            .register_index("index1".to_string(), metadata1)
            .unwrap();
        manager
            .register_index("index2".to_string(), metadata2)
            .unwrap();

        let indices = manager.list_indices();
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&"index1".to_string()));
        assert!(indices.contains(&"index2".to_string()));
    }
}
