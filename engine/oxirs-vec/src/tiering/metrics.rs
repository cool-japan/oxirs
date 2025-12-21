//! Metrics collection for tiering system

use super::types::{StorageTier, TierStatistics, TierTransition};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Tier metrics collector
#[derive(Debug, Clone)]
pub struct TierMetrics {
    /// Statistics for each tier
    tier_stats: Arc<Mutex<HashMap<StorageTier, TierStatistics>>>,
    /// Transition history
    transition_history: Arc<Mutex<Vec<TierTransition>>>,
    /// Performance metrics
    performance_metrics: Arc<Mutex<PerformanceMetrics>>,
    /// Cost metrics
    cost_metrics: Arc<Mutex<CostMetrics>>,
}

impl Default for TierMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TierMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let mut tier_stats = HashMap::new();
        tier_stats.insert(StorageTier::Hot, TierStatistics::default());
        tier_stats.insert(StorageTier::Warm, TierStatistics::default());
        tier_stats.insert(StorageTier::Cold, TierStatistics::default());

        Self {
            tier_stats: Arc::new(Mutex::new(tier_stats)),
            transition_history: Arc::new(Mutex::new(Vec::new())),
            performance_metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            cost_metrics: Arc::new(Mutex::new(CostMetrics::default())),
        }
    }

    /// Record a query to a tier
    pub fn record_query(&self, tier: StorageTier, latency_us: u64, hit: bool) {
        let mut stats = self.tier_stats.lock().unwrap();
        if let Some(tier_stat) = stats.get_mut(&tier) {
            tier_stat.total_queries += 1;

            // Update average latency (exponential moving average)
            let alpha = 0.1;
            tier_stat.avg_query_latency_us = ((1.0 - alpha) * tier_stat.avg_query_latency_us as f64
                + alpha * latency_us as f64) as u64;

            // Update hit rate
            if hit {
                tier_stat.hit_rate = (tier_stat.hit_rate * (tier_stat.total_queries - 1) as f64
                    + 1.0)
                    / tier_stat.total_queries as f64;
            } else {
                tier_stat.hit_rate = (tier_stat.hit_rate * (tier_stat.total_queries - 1) as f64)
                    / tier_stat.total_queries as f64;
            }

            tier_stat.last_updated = SystemTime::now();
        }
    }

    /// Record a tier transition
    pub fn record_transition(&self, transition: TierTransition) {
        // Extract tier information before moving transition
        let from_tier = transition.from_tier;
        let to_tier = transition.to_tier;

        let mut history = self.transition_history.lock().unwrap();
        history.push(transition);

        // Update tier statistics
        let mut stats = self.tier_stats.lock().unwrap();
        if let Some(from_tier_stats) = stats.get_mut(&from_tier) {
            from_tier_stats.promotions += 1;
        }
        if let Some(to_tier_stats) = stats.get_mut(&to_tier) {
            to_tier_stats.demotions += 1;
        }
    }

    /// Update tier capacity and usage
    pub fn update_tier_usage(&self, tier: StorageTier, used_bytes: u64, capacity_bytes: u64) {
        let mut stats = self.tier_stats.lock().unwrap();
        if let Some(tier_stat) = stats.get_mut(&tier) {
            tier_stat.used_bytes = used_bytes;
            tier_stat.capacity_bytes = capacity_bytes;
            tier_stat.last_updated = SystemTime::now();
        }
    }

    /// Update index count for a tier
    pub fn update_index_count(&self, tier: StorageTier, count: usize) {
        let mut stats = self.tier_stats.lock().unwrap();
        if let Some(tier_stat) = stats.get_mut(&tier) {
            tier_stat.index_count = count;
        }
    }

    /// Record bytes read from a tier
    pub fn record_bytes_read(&self, tier: StorageTier, bytes: u64) {
        let mut stats = self.tier_stats.lock().unwrap();
        if let Some(tier_stat) = stats.get_mut(&tier) {
            tier_stat.bytes_read += bytes;
        }
    }

    /// Record bytes written to a tier
    pub fn record_bytes_written(&self, tier: StorageTier, bytes: u64) {
        let mut stats = self.tier_stats.lock().unwrap();
        if let Some(tier_stat) = stats.get_mut(&tier) {
            tier_stat.bytes_written += bytes;
        }
    }

    /// Get statistics for a tier
    pub fn get_tier_statistics(&self, tier: StorageTier) -> TierStatistics {
        let stats = self.tier_stats.lock().unwrap();
        stats.get(&tier).cloned().unwrap_or_default()
    }

    /// Get all tier statistics
    pub fn get_all_tier_statistics(&self) -> HashMap<StorageTier, TierStatistics> {
        self.tier_stats.lock().unwrap().clone()
    }

    /// Get transition history
    pub fn get_transition_history(&self, limit: Option<usize>) -> Vec<TierTransition> {
        let history = self.transition_history.lock().unwrap();
        if let Some(lim) = limit {
            history.iter().rev().take(lim).cloned().collect()
        } else {
            history.clone()
        }
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.lock().unwrap().clone()
    }

    /// Update performance metrics
    pub fn update_performance_metrics<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut PerformanceMetrics),
    {
        let mut metrics = self.performance_metrics.lock().unwrap();
        update_fn(&mut metrics);
    }

    /// Get cost metrics
    pub fn get_cost_metrics(&self) -> CostMetrics {
        self.cost_metrics.lock().unwrap().clone()
    }

    /// Update cost metrics
    pub fn update_cost_metrics<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut CostMetrics),
    {
        let mut metrics = self.cost_metrics.lock().unwrap();
        update_fn(&mut metrics);
    }

    /// Calculate overall system efficiency
    pub fn calculate_efficiency(&self) -> TieringEfficiency {
        let stats = self.tier_stats.lock().unwrap();
        let perf = self.performance_metrics.lock().unwrap();
        let cost = self.cost_metrics.lock().unwrap();

        // Overall hit rate (weighted by tier)
        let hot_stat = stats.get(&StorageTier::Hot).unwrap();
        let warm_stat = stats.get(&StorageTier::Warm).unwrap();
        let cold_stat = stats.get(&StorageTier::Cold).unwrap();

        let total_queries =
            hot_stat.total_queries + warm_stat.total_queries + cold_stat.total_queries;
        let overall_hit_rate = if total_queries > 0 {
            (hot_stat.hit_rate * hot_stat.total_queries as f64
                + warm_stat.hit_rate * warm_stat.total_queries as f64
                + cold_stat.hit_rate * cold_stat.total_queries as f64)
                / total_queries as f64
        } else {
            0.0
        };

        // Average latency (weighted by tier)
        let avg_latency = if total_queries > 0 {
            (hot_stat.avg_query_latency_us * hot_stat.total_queries
                + warm_stat.avg_query_latency_us * warm_stat.total_queries
                + cold_stat.avg_query_latency_us * cold_stat.total_queries)
                / total_queries
        } else {
            0
        };

        // Utilization efficiency
        let total_capacity =
            hot_stat.capacity_bytes + warm_stat.capacity_bytes + cold_stat.capacity_bytes;
        let total_used = hot_stat.used_bytes + warm_stat.used_bytes + cold_stat.used_bytes;
        let utilization_efficiency = if total_capacity > 0 {
            total_used as f64 / total_capacity as f64
        } else {
            0.0
        };

        TieringEfficiency {
            overall_hit_rate,
            avg_query_latency_us: avg_latency,
            utilization_efficiency,
            cost_per_query: cost.total_query_cost / total_queries.max(1) as f64,
            cost_per_gb_hour: cost.total_storage_cost
                / ((total_used as f64 / (1024.0 * 1024.0 * 1024.0)) * perf.uptime_hours),
            transitions_per_hour: perf.total_transitions as f64 / perf.uptime_hours,
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        let mut stats = self.tier_stats.lock().unwrap();
        for tier_stat in stats.values_mut() {
            *tier_stat = TierStatistics::default();
        }

        let mut history = self.transition_history.lock().unwrap();
        history.clear();

        let mut perf = self.performance_metrics.lock().unwrap();
        *perf = PerformanceMetrics::default();

        let mut cost = self.cost_metrics.lock().unwrap();
        *cost = CostMetrics::default();
    }
}

/// Performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of tier transitions
    pub total_transitions: u64,
    /// Successful transitions
    pub successful_transitions: u64,
    /// Failed transitions
    pub failed_transitions: u64,
    /// Average transition duration
    pub avg_transition_duration: Duration,
    /// Total system uptime in hours
    pub uptime_hours: f64,
    /// Peak memory usage in bytes
    pub peak_memory_usage_bytes: u64,
    /// Average CPU utilization (0.0 - 1.0)
    pub avg_cpu_utilization: f64,
}

/// Cost metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Total storage cost
    pub total_storage_cost: f64,
    /// Total query cost
    pub total_query_cost: f64,
    /// Total transition cost
    pub total_transition_cost: f64,
    /// Cost by tier
    pub cost_by_tier: HashMap<String, f64>,
}

/// Overall tiering efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringEfficiency {
    /// Overall hit rate across all tiers
    pub overall_hit_rate: f64,
    /// Average query latency in microseconds
    pub avg_query_latency_us: u64,
    /// Storage utilization efficiency (0.0 - 1.0)
    pub utilization_efficiency: f64,
    /// Cost per query
    pub cost_per_query: f64,
    /// Cost per GB-hour
    pub cost_per_gb_hour: f64,
    /// Transitions per hour
    pub transitions_per_hour: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_metrics_basic() {
        let metrics = TierMetrics::new();

        // Record some queries
        metrics.record_query(StorageTier::Hot, 100, true);
        metrics.record_query(StorageTier::Hot, 200, true);
        metrics.record_query(StorageTier::Hot, 150, false);

        let stats = metrics.get_tier_statistics(StorageTier::Hot);
        assert_eq!(stats.total_queries, 3);
        assert!(stats.avg_query_latency_us > 0);
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_transition_recording() {
        let metrics = TierMetrics::new();

        let transition = TierTransition {
            index_id: "test_index".to_string(),
            from_tier: StorageTier::Warm,
            to_tier: StorageTier::Hot,
            reason: "High access frequency".to_string(),
            timestamp: SystemTime::now(),
            duration: Duration::from_secs(5),
            success: true,
            error: None,
        };

        metrics.record_transition(transition);

        let history = metrics.get_transition_history(Some(10));
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_efficiency_calculation() {
        let metrics = TierMetrics::new();

        // Setup some statistics
        metrics.update_tier_usage(
            StorageTier::Hot,
            8 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
        );
        metrics.record_query(StorageTier::Hot, 100, true);
        metrics.record_query(StorageTier::Hot, 150, true);

        let efficiency = metrics.calculate_efficiency();
        assert!(efficiency.overall_hit_rate > 0.0);
        assert!(efficiency.avg_query_latency_us > 0);
    }
}
