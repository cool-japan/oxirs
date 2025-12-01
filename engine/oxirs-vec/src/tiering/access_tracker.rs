//! Access pattern tracking for tiering decisions

use super::types::{AccessPattern, AccessStatistics, IndexMetadata};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

/// Access pattern tracker
pub struct AccessTracker {
    /// Access history (index_id -> timestamps)
    access_history: HashMap<String, VecDeque<SystemTime>>,
    /// Query latencies (index_id -> latencies in microseconds)
    latency_history: HashMap<String, VecDeque<u64>>,
    /// Maximum history size
    max_history_size: usize,
    /// Time window for access pattern analysis
    analysis_window: Duration,
}

impl AccessTracker {
    /// Create a new access tracker
    pub fn new(max_history_size: usize, analysis_window: Duration) -> Self {
        Self {
            access_history: HashMap::new(),
            latency_history: HashMap::new(),
            max_history_size,
            analysis_window,
        }
    }

    /// Record an access to an index
    pub fn record_access(&mut self, index_id: &str, latency_us: u64) {
        let now = SystemTime::now();

        // Record access timestamp
        let history = self.access_history.entry(index_id.to_string()).or_default();
        history.push_back(now);

        // Maintain history size
        while history.len() > self.max_history_size {
            history.pop_front();
        }

        // Record latency
        let latencies = self
            .latency_history
            .entry(index_id.to_string())
            .or_default();
        latencies.push_back(latency_us);

        // Maintain latency history size
        while latencies.len() > self.max_history_size {
            latencies.pop_front();
        }
    }

    /// Get access statistics for an index
    pub fn get_statistics(&self, index_id: &str) -> AccessStatistics {
        let now = SystemTime::now();

        let history = self.access_history.get(index_id);
        let latencies = self.latency_history.get(index_id);

        if let Some(hist) = history {
            let total_queries = hist.len() as u64;

            // Count queries in different time windows
            let queries_last_hour =
                self.count_queries_in_window(hist, now, Duration::from_secs(3600));
            let queries_last_day =
                self.count_queries_in_window(hist, now, Duration::from_secs(86400));
            let queries_last_week =
                self.count_queries_in_window(hist, now, Duration::from_secs(604800));

            // Calculate QPS
            let avg_qps = if !hist.is_empty() {
                let time_span = now
                    .duration_since(*hist.front().unwrap())
                    .unwrap_or(Duration::from_secs(1))
                    .as_secs_f64();
                total_queries as f64 / time_span.max(1.0)
            } else {
                0.0
            };

            // Calculate peak QPS (in 1-minute windows)
            let peak_qps = self.calculate_peak_qps(hist, now);

            // Calculate latency percentiles
            let query_latencies = if let Some(lats) = latencies {
                self.calculate_latency_percentiles(lats)
            } else {
                Default::default()
            };

            // Determine access pattern
            let access_pattern = self.classify_access_pattern(
                avg_qps,
                queries_last_hour,
                queries_last_day,
                queries_last_week,
            );

            AccessStatistics {
                total_queries,
                queries_last_hour,
                queries_last_day,
                queries_last_week,
                avg_qps,
                peak_qps,
                last_access_time: hist.back().copied(),
                access_pattern,
                query_latencies,
            }
        } else {
            Default::default()
        }
    }

    /// Update metadata with current access statistics
    pub fn update_metadata(&self, metadata: &mut IndexMetadata) {
        metadata.access_stats = self.get_statistics(&metadata.index_id);
        metadata.last_accessed = metadata
            .access_stats
            .last_access_time
            .unwrap_or_else(SystemTime::now);
    }

    /// Count queries in a time window
    fn count_queries_in_window(
        &self,
        history: &VecDeque<SystemTime>,
        now: SystemTime,
        window: Duration,
    ) -> u64 {
        history
            .iter()
            .filter(|&&t| now.duration_since(t).unwrap_or(Duration::MAX) <= window)
            .count() as u64
    }

    /// Calculate peak QPS in 1-minute windows
    fn calculate_peak_qps(&self, history: &VecDeque<SystemTime>, _now: SystemTime) -> f64 {
        if history.is_empty() {
            return 0.0;
        }

        let window = Duration::from_secs(60);
        let mut max_qps: f64 = 0.0;

        // Sample windows to avoid quadratic complexity
        let sample_size = history.len().min(100);
        for i in (0..history.len()).step_by(history.len() / sample_size.max(1)) {
            if let Some(&time) = history.get(i) {
                let count = self.count_queries_in_window(history, time + window, window);
                let qps = count as f64 / 60.0;
                max_qps = max_qps.max(qps);
            }
        }

        max_qps
    }

    /// Calculate latency percentiles
    fn calculate_latency_percentiles(
        &self,
        latencies: &VecDeque<u64>,
    ) -> super::types::LatencyPercentiles {
        if latencies.is_empty() {
            return Default::default();
        }

        let mut sorted: Vec<u64> = latencies.iter().copied().collect();
        sorted.sort_unstable();

        let p50 = sorted[sorted.len() * 50 / 100];
        let p95 = sorted[sorted.len() * 95 / 100];
        let p99 = sorted[sorted.len() * 99 / 100];
        let max = *sorted.last().unwrap();

        super::types::LatencyPercentiles { p50, p95, p99, max }
    }

    /// Classify access pattern
    fn classify_access_pattern(
        &self,
        avg_qps: f64,
        queries_last_hour: u64,
        queries_last_day: u64,
        queries_last_week: u64,
    ) -> AccessPattern {
        // Hot: > 10 QPS sustained
        if avg_qps > 10.0 {
            return AccessPattern::Hot;
        }

        // Warm: 1-10 QPS
        if avg_qps > 1.0 {
            return AccessPattern::Warm;
        }

        // Cold: < 1 QPS
        if avg_qps < 1.0 && queries_last_day < 100 {
            return AccessPattern::Cold;
        }

        // Bursty: High variance in access rates
        let hour_rate = queries_last_hour as f64 / 1.0;
        let day_rate = queries_last_day as f64 / 24.0;
        if hour_rate > day_rate * 3.0 {
            return AccessPattern::Bursty;
        }

        // Seasonal: Access patterns vary by time
        // (Simplified heuristic - could use ML for better detection)
        let week_rate = queries_last_week as f64 / (7.0 * 24.0);
        if day_rate > week_rate * 2.0 || day_rate < week_rate * 0.5 {
            return AccessPattern::Seasonal;
        }

        AccessPattern::Unknown
    }

    /// Predict future access based on historical patterns
    pub fn predict_future_access(&self, index_id: &str, horizon: Duration) -> f64 {
        let stats = self.get_statistics(index_id);

        // Simple prediction: exponentially weighted moving average
        let recent_qps = stats.avg_qps;
        let historical_qps = if stats.total_queries > 0 {
            stats.total_queries as f64
                / stats
                    .last_access_time
                    .and_then(|t| SystemTime::now().duration_since(t).ok())
                    .unwrap_or(Duration::from_secs(1))
                    .as_secs_f64()
        } else {
            0.0
        };

        // Weight recent activity more heavily
        let alpha = 0.7;
        let predicted_qps = alpha * recent_qps + (1.0 - alpha) * historical_qps;

        // Add some uncertainty (simplified, without using distribution)
        let uncertainty = predicted_qps * 0.05; // 5% uncertainty
        (predicted_qps + uncertainty) * horizon.as_secs_f64()
    }

    /// Clear old history entries
    pub fn cleanup_old_entries(&mut self, retention_period: Duration) {
        let now = SystemTime::now();

        for history in self.access_history.values_mut() {
            while let Some(&front) = history.front() {
                if now.duration_since(front).unwrap_or(Duration::ZERO) > retention_period {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }

        for latencies in self.latency_history.values_mut() {
            // Keep same size as access history
            while latencies.len() > self.max_history_size {
                latencies.pop_front();
            }
        }

        // Remove empty entries
        self.access_history.retain(|_, v| !v.is_empty());
        self.latency_history.retain(|_, v| !v.is_empty());
    }

    /// Get indices sorted by access score (descending)
    pub fn get_hot_indices(&self, limit: usize) -> Vec<(String, f64)> {
        let mut indices: Vec<(String, f64)> = self
            .access_history
            .keys()
            .map(|id| {
                let stats = self.get_statistics(id);
                let score = super::policies::calculate_access_score(&stats);
                (id.clone(), score)
            })
            .collect();

        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indices.truncate(limit);
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_tracker_basic() {
        let mut tracker = AccessTracker::new(1000, Duration::from_secs(3600));

        // Record some accesses
        for _ in 0..10 {
            tracker.record_access("index1", 1000);
        }

        let stats = tracker.get_statistics("index1");
        assert_eq!(stats.total_queries, 10);
        assert!(stats.avg_qps > 0.0);
    }

    #[test]
    fn test_access_pattern_classification() {
        let mut tracker = AccessTracker::new(1000, Duration::from_secs(3600));

        // Simulate hot access pattern (high QPS)
        for _ in 0..1000 {
            tracker.record_access("hot_index", 500);
        }

        let stats = tracker.get_statistics("hot_index");
        assert!(matches!(
            stats.access_pattern,
            AccessPattern::Hot | AccessPattern::Warm
        ));
    }

    #[test]
    fn test_latency_percentiles() {
        let mut tracker = AccessTracker::new(1000, Duration::from_secs(3600));

        // Record varying latencies
        let latencies = vec![100, 200, 300, 500, 1000, 2000, 5000];
        for &lat in &latencies {
            tracker.record_access("index1", lat);
        }

        let stats = tracker.get_statistics("index1");
        assert!(stats.query_latencies.p50 > 0);
        assert!(stats.query_latencies.p99 > stats.query_latencies.p50);
    }

    #[test]
    fn test_hot_indices() {
        let mut tracker = AccessTracker::new(1000, Duration::from_secs(3600));

        // Create indices with different access patterns
        for _ in 0..100 {
            tracker.record_access("hot_index", 100);
        }
        for _ in 0..10 {
            tracker.record_access("warm_index", 200);
        }
        tracker.record_access("cold_index", 300);

        let hot = tracker.get_hot_indices(3);
        assert_eq!(hot.len(), 3);
        assert_eq!(hot[0].0, "hot_index");
    }
}
