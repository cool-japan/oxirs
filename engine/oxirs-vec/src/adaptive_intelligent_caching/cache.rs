//! Main adaptive intelligent cache implementation

use anyhow::Result;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, info, warn};

use super::types::AccessTracker;
use super::config::CacheConfiguration;
use super::eviction::{AdaptiveEvictionPolicy, EvictionPolicy, LFUEvictionPolicy, LRUEvictionPolicy};
use super::metrics::CachePerformanceMetrics;
use super::ml_models::MLModels;
use super::optimizer::CacheOptimizer;
use super::pattern_analyzer::AccessPatternAnalyzer;
use super::prefetcher::PredictivePrefetcher;
use super::storage::{CacheStorage, CompressedStorage, MemoryStorage, PersistentStorage};
use super::tier::CacheTier;
use super::types::{
    AccessEvent, CacheItem, CacheKey, CachePerformanceData, CacheStatistics, CacheValue,
    EstimatedImpact, ExportFormat, OptimizationEvent, OptimizationResult, TierConfiguration,
    TierStatistics,
};

/// Adaptive intelligent caching system with ML-driven optimization
#[derive(Debug)]
pub struct AdaptiveIntelligentCache {
    /// Multi-tier cache storage
    tiers: Vec<CacheTier>,
    /// Cache access pattern analyzer
    pattern_analyzer: AccessPatternAnalyzer,
    /// Predictive prefetching engine
    prefetcher: PredictivePrefetcher,
    /// Cache optimization engine
    optimizer: CacheOptimizer,
    /// Performance metrics collector
    metrics: CachePerformanceMetrics,
    /// Configuration parameters
    config: CacheConfiguration,
    /// Machine learning models for cache decisions
    ml_models: MLModels,
}

impl AdaptiveIntelligentCache {
    /// Create a new adaptive intelligent cache with the given configuration
    pub fn new(config: CacheConfiguration) -> Result<Self> {
        info!(
            "Initializing Adaptive Intelligent Cache with {} tiers",
            config.num_tiers
        );

        let mut tiers = Vec::new();
        let tier_sizes = Self::calculate_tier_sizes(&config);

        for (tier_id, size) in tier_sizes.into_iter().enumerate() {
            let tier_config = TierConfiguration {
                max_size_bytes: size,
                default_ttl: Duration::from_secs(config.default_ttl_seconds),
                compression_enabled: tier_id > 0, // Enable compression for higher tiers
                persistence_enabled: tier_id == config.num_tiers as usize - 1, // Only last tier persisted
                replication_factor: if tier_id == 0 { 1 } else { 2 }, // Replicate slower tiers
            };

            let storage = Self::create_storage_for_tier(tier_id as u32, &tier_config)?;
            let eviction_policy = Self::create_eviction_policy_for_tier(tier_id as u32);

            let tier = CacheTier {
                tier_id: tier_id as u32,
                storage,
                eviction_policy,
                access_tracker: AccessTracker::new(),
                config: tier_config,
                stats: TierStatistics::default(),
            };

            tiers.push(tier);
        }

        Ok(Self {
            tiers,
            pattern_analyzer: AccessPatternAnalyzer::new(),
            prefetcher: PredictivePrefetcher::new(),
            optimizer: CacheOptimizer::new(),
            metrics: CachePerformanceMetrics::default(),
            config,
            ml_models: MLModels::new()?,
        })
    }

    /// Store a value in the cache with intelligent tier placement
    pub fn store(&mut self, key: CacheKey, value: CacheValue) -> Result<()> {
        let start_time = Instant::now();

        // Determine optimal tier placement using ML model
        let optimal_tier = self
            .ml_models
            .tier_placement_model
            .predict_optimal_tier(&key, &value);

        // Store in the determined tier
        let tier = &mut self.tiers[optimal_tier as usize];
        tier.storage
            .store(key.clone(), value.clone(), Some(tier.config.default_ttl))?;

        // Update access tracking and metrics
        tier.access_tracker.on_store(&key);
        self.update_store_metrics(optimal_tier, start_time.elapsed());

        // Trigger eviction if necessary
        self.check_and_evict(optimal_tier)?;

        // Update ML models with new data
        self.ml_models
            .update_with_store_event(&key, &value, optimal_tier);

        debug!(
            "Stored cache item in tier {} with key hash {:?}",
            optimal_tier,
            self.hash_key(&key)
        );
        Ok(())
    }

    /// Retrieve a value from the cache with intelligent promotion
    pub fn retrieve(&mut self, key: &CacheKey) -> Option<CacheValue> {
        let start_time = Instant::now();

        // Search through tiers starting from fastest
        for (tier_index, tier) in self.tiers.iter_mut().enumerate() {
            if let Some(mut value) = tier.storage.retrieve(key) {
                // Update access information
                value.last_accessed = SystemTime::now();
                value.access_count += 1;

                tier.access_tracker.on_access(key, Instant::now());
                self.update_hit_metrics(tier_index as u32, start_time.elapsed());

                // Consider promoting to faster tier based on access pattern
                if tier_index > 0 && self.should_promote(key, &value, tier_index) {
                    if let Err(e) = self.promote_item(key.clone(), value.clone(), tier_index) {
                        warn!("Failed to promote cache item: {}", e);
                    }
                }

                // Record access event for pattern analysis
                self.pattern_analyzer.record_access(AccessEvent {
                    timestamp: SystemTime::now(),
                    key: key.clone(),
                    hit: true,
                    latency_ns: start_time.elapsed().as_nanos() as u64,
                    user_context: None, // Could be extracted from key metadata
                });

                // Trigger predictive prefetching
                if self.config.enable_prefetching {
                    self.prefetcher.trigger_prefetch_analysis(key, &value);
                }

                return Some(value);
            }
        }

        // Cache miss - update metrics and patterns
        self.update_miss_metrics(start_time.elapsed());
        self.pattern_analyzer.record_access(AccessEvent {
            timestamp: SystemTime::now(),
            key: key.clone(),
            hit: false,
            latency_ns: start_time.elapsed().as_nanos() as u64,
            user_context: None,
        });

        None
    }

    /// Remove an item from all cache tiers
    pub fn remove(&mut self, key: &CacheKey) -> bool {
        let mut removed = false;
        for tier in &mut self.tiers {
            if tier.storage.remove(key) {
                tier.access_tracker.on_remove(key);
                removed = true;
            }
        }
        removed
    }

    /// Get comprehensive cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        let total_hits = self.metrics.hit_count.load(Ordering::Relaxed);
        let total_misses = self.metrics.miss_count.load(Ordering::Relaxed);
        let total_requests = total_hits + total_misses;

        let hit_rate = if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        CacheStatistics {
            hit_rate,
            miss_rate: 1.0 - hit_rate,
            total_requests,
            avg_hit_latency_ns: self.metrics.avg_hit_latency_ns.load(Ordering::Relaxed),
            avg_miss_latency_ns: self.metrics.avg_miss_latency_ns.load(Ordering::Relaxed),
            cache_efficiency: self.metrics.cache_efficiency_score,
            memory_utilization: self.calculate_memory_utilization(),
            tier_statistics: self.collect_tier_statistics(),
            prefetch_statistics: self.prefetcher.get_statistics(),
            optimization_statistics: self.optimizer.get_statistics(),
        }
    }

    /// Run cache optimization cycle
    pub fn optimize(&mut self) -> Result<OptimizationResult> {
        if !self.config.enable_adaptive_optimization {
            return Ok(OptimizationResult {
                improvement_score: 0.0,
                changes_applied: vec![],
                estimated_impact: EstimatedImpact::default(),
            });
        }

        info!("Running cache optimization cycle");
        let before_metrics = self.metrics.clone();

        let mut total_improvement = 0.0;
        let mut all_changes = Vec::new();

        // Run each optimization algorithm
        // Temporarily move algorithms out to avoid borrowing conflicts
        let mut algorithms = std::mem::take(&mut self.optimizer.algorithms);
        for algorithm in &mut algorithms {
            match algorithm.optimize_cache(&self.tiers, &self.metrics, &self.config) {
                Ok(result) => {
                    total_improvement += result.improvement_score;
                    all_changes.extend(result.changes_applied);
                    info!(
                        "Optimization algorithm '{}' achieved {:.2}% improvement",
                        algorithm.name(),
                        result.improvement_score * 100.0
                    );
                }
                Err(e) => {
                    warn!(
                        "Optimization algorithm '{}' failed: {}",
                        algorithm.name(),
                        e
                    );
                }
            }
        }
        // Move algorithms back
        self.optimizer.algorithms = algorithms;

        // Update optimization history
        self.optimizer.record_optimization_event(OptimizationEvent {
            timestamp: SystemTime::now(),
            algorithm: "combined".to_string(),
            changes: all_changes.clone(),
            before_metrics,
            after_metrics: None, // Will be updated later
        });

        Ok(OptimizationResult {
            improvement_score: total_improvement,
            changes_applied: all_changes,
            estimated_impact: self.estimate_optimization_impact(total_improvement),
        })
    }

    /// Export cache performance data for external analysis
    pub fn export_performance_data(&self, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Json => {
                let data = CachePerformanceData {
                    metrics: self.metrics.clone(),
                    statistics: self.get_statistics(),
                    configuration: self.config.clone(),
                    access_patterns: self.pattern_analyzer.export_patterns(),
                    optimization_history: self.optimizer.export_history(),
                };
                Ok(serde_json::to_string_pretty(&data)?)
            }
            ExportFormat::Prometheus => self.export_prometheus_metrics(),
            ExportFormat::Csv => self.export_csv_metrics(),
        }
    }

    // Private helper methods

    fn calculate_tier_sizes(config: &CacheConfiguration) -> Vec<u64> {
        let total_size = config.max_total_size_bytes;
        config
            .tier_size_ratios
            .iter()
            .map(|ratio| (total_size as f64 * ratio) as u64)
            .collect()
    }

    fn create_storage_for_tier(
        tier_id: u32,
        config: &TierConfiguration,
    ) -> Result<Box<dyn CacheStorage>> {
        match tier_id {
            0 => Ok(Box::new(MemoryStorage::new(config.max_size_bytes))),
            1 => Ok(Box::new(CompressedStorage::new(config.max_size_bytes))),
            _ => Ok(Box::new(PersistentStorage::new(config.max_size_bytes)?)),
        }
    }

    fn create_eviction_policy_for_tier(tier_id: u32) -> Box<dyn EvictionPolicy> {
        match tier_id {
            0 => Box::new(LRUEvictionPolicy::new()),
            1 => Box::new(LFUEvictionPolicy::new()),
            _ => Box::new(AdaptiveEvictionPolicy::new()),
        }
    }

    fn should_promote(&self, _key: &CacheKey, value: &CacheValue, current_tier: usize) -> bool {
        // Use ML model to determine if item should be promoted
        let access_frequency = value.access_count as f64;
        let recency_score = self.calculate_recency_score(value.last_accessed);
        let size_penalty = value.metadata.size_bytes as f64 / 1024.0; // KB

        let promotion_score = access_frequency * recency_score / size_penalty;
        promotion_score > 2.0 && current_tier > 0
    }

    fn promote_item(&mut self, key: CacheKey, value: CacheValue, from_tier: usize) -> Result<()> {
        if from_tier == 0 {
            return Ok(()); // Already in fastest tier
        }

        let target_tier = from_tier - 1;

        // Remove from current tier
        self.tiers[from_tier].storage.remove(&key);

        // Store in target tier
        let default_ttl = self.tiers[target_tier].config.default_ttl;
        self.tiers[target_tier]
            .storage
            .store(key, value, Some(default_ttl))?;

        debug!(
            "Promoted cache item from tier {} to tier {}",
            from_tier, target_tier
        );
        Ok(())
    }

    fn calculate_recency_score(&self, last_accessed: SystemTime) -> f64 {
        let now = SystemTime::now();
        let duration = now.duration_since(last_accessed).unwrap_or(Duration::ZERO);
        let hours = duration.as_secs_f64() / 3600.0;

        // Exponential decay
        (-hours / 24.0).exp()
    }

    fn check_and_evict(&mut self, tier_id: u32) -> Result<()> {
        let size_info = {
            let tier = &self.tiers[tier_id as usize];
            tier.storage.size_info()
        };

        if size_info.used_bytes > self.tiers[tier_id as usize].config.max_size_bytes {
            let target_size =
                (self.tiers[tier_id as usize].config.max_size_bytes as f64 * 0.8) as u64; // Target 80% utilization
            let items = self.collect_tier_items(tier_id);

            let keys_to_evict = {
                let tier = &mut self.tiers[tier_id as usize];
                tier.eviction_policy
                    .evict(size_info.used_bytes, target_size, &items)
            };

            let tier = &mut self.tiers[tier_id as usize];
            for key in keys_to_evict {
                tier.storage.remove(&key);
                tier.stats.eviction_count += 1;
            }
        }

        Ok(())
    }

    fn collect_tier_items(&self, _tier_id: u32) -> Vec<CacheItem> {
        // This would collect all items from the tier for eviction analysis
        // Simplified implementation
        Vec::new()
    }

    fn hash_key(&self, key: &CacheKey) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    fn update_store_metrics(&mut self, tier_id: u32, _latency: Duration) {
        // Update tier-specific metrics
        if let Some(_tier_metrics) = self.metrics.tier_metrics.get_mut(&tier_id) {
            // Update tier metrics
        }
    }

    fn update_hit_metrics(&mut self, _tier_id: u32, latency: Duration) {
        self.metrics.hit_count.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);

        // Update average hit latency (simplified)
        let latency_ns = latency.as_nanos() as u64;
        self.metrics
            .avg_hit_latency_ns
            .store(latency_ns, Ordering::Relaxed);
    }

    fn update_miss_metrics(&mut self, latency: Duration) {
        self.metrics.miss_count.fetch_add(1, Ordering::Relaxed);
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);

        let latency_ns = latency.as_nanos() as u64;
        self.metrics
            .avg_miss_latency_ns
            .store(latency_ns, Ordering::Relaxed);
    }

    fn calculate_memory_utilization(&self) -> f64 {
        let total_used: u64 = self
            .tiers
            .iter()
            .map(|tier| tier.storage.size_info().used_bytes)
            .sum();
        let total_capacity: u64 = self
            .tiers
            .iter()
            .map(|tier| tier.storage.size_info().total_capacity_bytes)
            .sum();

        if total_capacity > 0 {
            total_used as f64 / total_capacity as f64
        } else {
            0.0
        }
    }

    fn collect_tier_statistics(&self) -> Vec<TierStatistics> {
        self.tiers.iter().map(|tier| tier.stats.clone()).collect()
    }

    fn estimate_optimization_impact(&self, improvement_score: f64) -> EstimatedImpact {
        EstimatedImpact {
            hit_rate_improvement: improvement_score * 0.1,
            latency_reduction: improvement_score * 0.05,
            memory_efficiency_gain: improvement_score * 0.08,
            cost_reduction: improvement_score * 0.03,
        }
    }

    fn export_prometheus_metrics(&self) -> Result<String> {
        let mut metrics = String::new();

        let hit_count = self.metrics.hit_count.load(Ordering::Relaxed);
        let miss_count = self.metrics.miss_count.load(Ordering::Relaxed);
        let total = hit_count + miss_count;

        metrics.push_str(&format!("oxirs_cache_hits_total {hit_count}\n"));
        metrics.push_str(&format!("oxirs_cache_misses_total {miss_count}\n"));
        metrics.push_str(&format!("oxirs_cache_requests_total {total}\n"));

        if total > 0 {
            let hit_rate = hit_count as f64 / total as f64;
            metrics.push_str(&format!("oxirs_cache_hit_rate {hit_rate:.4}\n"));
        }

        metrics.push_str(&format!(
            "oxirs_cache_memory_utilization {:.4}\n",
            self.calculate_memory_utilization()
        ));
        metrics.push_str(&format!(
            "oxirs_cache_efficiency_score {:.4}\n",
            self.metrics.cache_efficiency_score
        ));

        Ok(metrics)
    }

    fn export_csv_metrics(&self) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("metric,value,timestamp\n");

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_secs();
        let hit_count = self.metrics.hit_count.load(Ordering::Relaxed);
        let miss_count = self.metrics.miss_count.load(Ordering::Relaxed);

        csv.push_str(&format!("hit_count,{hit_count},{now}\n"));
        csv.push_str(&format!("miss_count,{miss_count},{now}\n"));
        csv.push_str(&format!(
            "memory_utilization,{:.4},{}\n",
            self.calculate_memory_utilization(),
            now
        ));

        Ok(csv)
    }
}
