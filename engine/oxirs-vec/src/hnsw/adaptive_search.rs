//! Adaptive ef_search parameter tuning for HNSW
//!
//! This module provides intelligent, data-driven optimization of the ef_search parameter
//! to automatically balance search accuracy and latency based on query characteristics.

use anyhow::Result;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

/// Configuration for adaptive ef_search tuning
#[derive(Debug, Clone)]
pub struct AdaptiveSearchConfig {
    /// Initial ef_search value
    pub initial_ef_search: usize,
    /// Minimum allowed ef_search
    pub min_ef_search: usize,
    /// Maximum allowed ef_search
    pub max_ef_search: usize,
    /// Target recall (0.0 to 1.0)
    pub target_recall: f32,
    /// Target latency in milliseconds
    pub target_latency_ms: f64,
    /// Tolerance for recall deviation
    pub recall_tolerance: f32,
    /// Tolerance for latency deviation (as fraction)
    pub latency_tolerance: f64,
    /// Number of queries to consider for adaptation
    pub adaptation_window: usize,
    /// Minimum queries before adaptation starts
    pub min_queries_for_adaptation: usize,
    /// Adaptation step size (multiplier for ef_search adjustment)
    pub adaptation_rate: f32,
    /// Enable aggressive optimization
    pub aggressive_mode: bool,
}

impl Default for AdaptiveSearchConfig {
    fn default() -> Self {
        Self {
            initial_ef_search: 64,
            min_ef_search: 16,
            max_ef_search: 512,
            target_recall: 0.95,
            target_latency_ms: 10.0,
            recall_tolerance: 0.02,
            latency_tolerance: 0.2,
            adaptation_window: 100,
            min_queries_for_adaptation: 20,
            adaptation_rate: 0.1,
            aggressive_mode: false,
        }
    }
}

/// Query performance metrics
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    /// Query execution time
    pub latency_ms: f64,
    /// Achieved recall (if ground truth available)
    pub recall: Option<f32>,
    /// Number of distance calculations performed
    pub distance_computations: usize,
    /// ef_search value used
    pub ef_search_used: usize,
    /// Query timestamp
    pub timestamp: Instant,
}

/// Adaptive search statistics
#[derive(Debug, Clone)]
pub struct AdaptiveSearchStats {
    /// Total queries processed
    pub total_queries: usize,
    /// Current ef_search value
    pub current_ef_search: usize,
    /// Average latency over window
    pub avg_latency_ms: f64,
    /// Average recall over window (if available)
    pub avg_recall: Option<f32>,
    /// Number of adaptations performed
    pub adaptation_count: usize,
    /// Last adaptation timestamp
    pub last_adaptation: Option<Instant>,
    /// Current performance score (0.0 to 1.0)
    pub performance_score: f32,
}

/// Adaptive ef_search tuner
pub struct AdaptiveSearchTuner {
    config: AdaptiveSearchConfig,
    current_ef_search: Arc<RwLock<usize>>,
    query_history: Arc<RwLock<VecDeque<QueryMetrics>>>,
    stats: Arc<RwLock<AdaptiveSearchStats>>,
}

impl AdaptiveSearchTuner {
    /// Create a new adaptive search tuner
    pub fn new(config: AdaptiveSearchConfig) -> Self {
        let initial_ef = config.initial_ef_search;

        Self {
            config,
            current_ef_search: Arc::new(RwLock::new(initial_ef)),
            query_history: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(AdaptiveSearchStats {
                total_queries: 0,
                current_ef_search: initial_ef,
                avg_latency_ms: 0.0,
                avg_recall: None,
                adaptation_count: 0,
                last_adaptation: None,
                performance_score: 0.5,
            })),
        }
    }

    /// Get the current ef_search value to use
    pub fn get_ef_search(&self) -> usize {
        *self.current_ef_search.read()
    }

    /// Record query metrics and potentially adapt ef_search
    pub fn record_query(&self, metrics: QueryMetrics) -> Result<()> {
        let mut history = self.query_history.write();
        let mut stats = self.stats.write();

        // Add to history
        history.push_back(metrics.clone());
        stats.total_queries += 1;

        // Trim history to window size
        while history.len() > self.config.adaptation_window {
            history.pop_front();
        }

        // Update statistics
        self.update_statistics(&mut stats, &history);

        // Attempt adaptation if enough data
        if history.len() >= self.config.min_queries_for_adaptation {
            // Pass mutable reference to avoid re-acquiring the lock
            self.adapt_ef_search_internal(&mut stats, &history)?;
        }

        Ok(())
    }

    /// Update running statistics
    fn update_statistics(&self, stats: &mut AdaptiveSearchStats, history: &VecDeque<QueryMetrics>) {
        if history.is_empty() {
            return;
        }

        // Calculate average latency
        let sum_latency: f64 = history.iter().map(|m| m.latency_ms).sum();
        stats.avg_latency_ms = sum_latency / history.len() as f64;

        // Calculate average recall if available
        let recalls: Vec<f32> = history.iter().filter_map(|m| m.recall).collect();

        if !recalls.is_empty() {
            let sum_recall: f32 = recalls.iter().sum();
            stats.avg_recall = Some(sum_recall / recalls.len() as f32);
        }

        // Calculate performance score (weighted combination of recall and latency)
        let recall_score = stats.avg_recall.unwrap_or(0.8); // Default to 0.8 if unknown
        let latency_ratio = self.config.target_latency_ms / stats.avg_latency_ms.max(0.001);
        let latency_score = latency_ratio.min(1.0);

        // Weight recall more heavily (70/30 split)
        stats.performance_score = (0.7 * recall_score + 0.3 * latency_score as f32).min(1.0);

        stats.current_ef_search = *self.current_ef_search.read();
    }

    /// Adapt ef_search based on performance (internal method with mutable refs)
    fn adapt_ef_search_internal(
        &self,
        stats: &mut AdaptiveSearchStats,
        _history: &VecDeque<QueryMetrics>,
    ) -> Result<()> {
        let mut current_ef = self.current_ef_search.write();

        let avg_latency = stats.avg_latency_ms;
        let avg_recall = stats.avg_recall;

        // Determine if we need to adjust
        let recall_too_low = avg_recall
            .is_some_and(|r| r < self.config.target_recall - self.config.recall_tolerance);

        let recall_sufficient = match avg_recall {
            Some(r) => r >= self.config.target_recall,
            None => true,
        };

        let latency_too_high =
            avg_latency > self.config.target_latency_ms * (1.0 + self.config.latency_tolerance);
        let latency_acceptable = avg_latency <= self.config.target_latency_ms;

        // Decision logic
        let new_ef = if recall_too_low {
            // Recall is too low, increase ef_search
            let increase = if self.config.aggressive_mode {
                (*current_ef as f32 * (1.0 + 2.0 * self.config.adaptation_rate)) as usize
            } else {
                (*current_ef as f32 * (1.0 + self.config.adaptation_rate)) as usize
            };
            increase.min(self.config.max_ef_search)
        } else if recall_sufficient && latency_too_high {
            // Recall is good but latency is too high, decrease ef_search
            let decrease = if self.config.aggressive_mode {
                (*current_ef as f32 * (1.0 - 2.0 * self.config.adaptation_rate)) as usize
            } else {
                (*current_ef as f32 * (1.0 - self.config.adaptation_rate)) as usize
            };
            decrease.max(self.config.min_ef_search)
        } else if recall_sufficient && latency_acceptable {
            // Both metrics are good, try to reduce ef_search slightly for better performance
            let decrease =
                (*current_ef as f32 * (1.0 - 0.5 * self.config.adaptation_rate)) as usize;
            decrease.max(self.config.min_ef_search)
        } else {
            // No change needed
            *current_ef
        };

        // Apply change if significant
        if new_ef != *current_ef {
            tracing::debug!(
                "Adapting ef_search: {} -> {} (recall: {:?}, latency: {:.2}ms)",
                *current_ef,
                new_ef,
                avg_recall,
                avg_latency
            );

            *current_ef = new_ef;
            stats.current_ef_search = new_ef;
            stats.adaptation_count += 1;
            stats.last_adaptation = Some(Instant::now());
        }

        Ok(())
    }

    /// Get current statistics
    pub fn stats(&self) -> AdaptiveSearchStats {
        self.stats.read().clone()
    }

    /// Reset adaptation (useful for testing or mode changes)
    pub fn reset(&self) {
        let mut ef_search = self.current_ef_search.write();
        *ef_search = self.config.initial_ef_search;

        let mut history = self.query_history.write();
        history.clear();

        let mut stats = self.stats.write();
        *stats = AdaptiveSearchStats {
            total_queries: 0,
            current_ef_search: self.config.initial_ef_search,
            avg_latency_ms: 0.0,
            avg_recall: None,
            adaptation_count: 0,
            last_adaptation: None,
            performance_score: 0.5,
        };
    }

    /// Manually set ef_search (overrides adaptation)
    pub fn set_ef_search(&self, ef_search: usize) {
        let mut current_ef = self.current_ef_search.write();
        *current_ef = ef_search.clamp(self.config.min_ef_search, self.config.max_ef_search);

        let mut stats = self.stats.write();
        stats.current_ef_search = *current_ef;
    }

    /// Get configuration
    pub fn config(&self) -> &AdaptiveSearchConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_search_creation() {
        let config = AdaptiveSearchConfig::default();
        let tuner = AdaptiveSearchTuner::new(config);
        assert_eq!(tuner.get_ef_search(), 64);
    }

    #[test]
    fn test_record_query() {
        let tuner = AdaptiveSearchTuner::new(AdaptiveSearchConfig::default());

        let metrics = QueryMetrics {
            latency_ms: 5.0,
            recall: Some(0.95),
            distance_computations: 100,
            ef_search_used: 64,
            timestamp: Instant::now(),
        };

        assert!(tuner.record_query(metrics).is_ok());

        let stats = tuner.stats();
        assert_eq!(stats.total_queries, 1);
    }

    #[test]
    fn test_adaptation_on_low_recall() {
        let config = AdaptiveSearchConfig {
            min_queries_for_adaptation: 5,
            target_recall: 0.95,
            initial_ef_search: 32,
            ..Default::default()
        };

        let tuner = AdaptiveSearchTuner::new(config);

        // Record queries with low recall
        for _ in 0..10 {
            let metrics = QueryMetrics {
                latency_ms: 2.0,
                recall: Some(0.80), // Below target
                distance_computations: 50,
                ef_search_used: 32,
                timestamp: Instant::now(),
            };
            tuner.record_query(metrics).unwrap();
        }

        // ef_search should have increased
        assert!(tuner.get_ef_search() > 32);
    }

    #[test]
    fn test_adaptation_on_high_latency() {
        let config = AdaptiveSearchConfig {
            min_queries_for_adaptation: 5,
            target_latency_ms: 5.0,
            target_recall: 0.90,
            initial_ef_search: 128,
            ..Default::default()
        };

        let tuner = AdaptiveSearchTuner::new(config);

        // Record queries with high latency but good recall
        for _ in 0..10 {
            let metrics = QueryMetrics {
                latency_ms: 15.0,   // Well above target
                recall: Some(0.98), // Good recall
                distance_computations: 200,
                ef_search_used: 128,
                timestamp: Instant::now(),
            };
            tuner.record_query(metrics).unwrap();
        }

        // ef_search should have decreased
        assert!(tuner.get_ef_search() < 128);
    }

    #[test]
    fn test_manual_override() {
        let tuner = AdaptiveSearchTuner::new(AdaptiveSearchConfig::default());
        tuner.set_ef_search(200);
        assert_eq!(tuner.get_ef_search(), 200);
    }

    #[test]
    fn test_reset() {
        let config = AdaptiveSearchConfig::default();
        let initial_ef = config.initial_ef_search;
        let tuner = AdaptiveSearchTuner::new(config);

        tuner.set_ef_search(200);
        tuner.reset();

        assert_eq!(tuner.get_ef_search(), initial_ef);
        assert_eq!(tuner.stats().total_queries, 0);
    }
}
