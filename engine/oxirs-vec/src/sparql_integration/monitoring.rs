//! Performance monitoring and statistics for SPARQL vector operations

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Performance monitoring for vector operations
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    pub query_stats: Arc<RwLock<QueryStats>>,
    pub operation_timings: Arc<RwLock<HashMap<String, Vec<Duration>>>>,
    pub cache_stats: Arc<RwLock<CacheStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct QueryStats {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub avg_response_time: Duration,
    pub max_response_time: Duration,
    pub min_response_time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_size: usize,
    pub cache_capacity: usize,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            query_stats: Arc::new(RwLock::new(QueryStats::default())),
            operation_timings: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    pub fn record_query(&self, duration: Duration, success: bool) {
        let mut stats = self.query_stats.write();
        stats.total_queries += 1;

        if success {
            stats.successful_queries += 1;
        } else {
            stats.failed_queries += 1;
        }

        if stats.total_queries == 1 {
            stats.avg_response_time = duration;
            stats.max_response_time = duration;
            stats.min_response_time = duration;
        } else {
            // Update running average
            let total_time = stats
                .avg_response_time
                .mul_f64(stats.total_queries as f64 - 1.0)
                + duration;
            stats.avg_response_time = total_time.div_f64(stats.total_queries as f64);

            if duration > stats.max_response_time {
                stats.max_response_time = duration;
            }
            if duration < stats.min_response_time {
                stats.min_response_time = duration;
            }
        }
    }

    pub fn record_operation(&self, operation: &str, duration: Duration) {
        let mut timings = self.operation_timings.write();
        timings
            .entry(operation.to_string())
            .or_default()
            .push(duration);
    }

    pub fn record_cache_hit(&self) {
        let mut stats = self.cache_stats.write();
        stats.cache_hits += 1;
    }

    pub fn record_cache_miss(&self) {
        let mut stats = self.cache_stats.write();
        stats.cache_misses += 1;
    }

    pub fn update_cache_size(&self, size: usize, capacity: usize) {
        let mut stats = self.cache_stats.write();
        stats.cache_size = size;
        stats.cache_capacity = capacity;
    }

    pub fn get_stats(&self) -> (QueryStats, CacheStats) {
        let query_stats = self.query_stats.read().clone();
        let cache_stats = self.cache_stats.read().clone();
        (query_stats, cache_stats)
    }

    pub fn get_operation_timings(&self) -> HashMap<String, Vec<Duration>> {
        self.operation_timings.read().clone()
    }

    pub fn reset_stats(&self) {
        *self.query_stats.write() = QueryStats::default();
        *self.operation_timings.write() = HashMap::new();
        *self.cache_stats.write() = CacheStats::default();
    }

    /// Generate a comprehensive performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let (query_stats, cache_stats) = self.get_stats();
        let operation_timings = self.get_operation_timings();

        let mut operation_summaries = HashMap::new();
        for (op, timings) in operation_timings {
            if !timings.is_empty() {
                let total_time: Duration = timings.iter().sum();
                let avg_time = total_time / timings.len() as u32;
                let max_time = timings.iter().max().copied().unwrap_or_default();
                let min_time = timings.iter().min().copied().unwrap_or_default();

                operation_summaries.insert(
                    op,
                    OperationSummary {
                        count: timings.len(),
                        total_time,
                        avg_time,
                        max_time,
                        min_time,
                    },
                );
            }
        }

        PerformanceReport {
            query_stats,
            cache_stats,
            operation_summaries,
            report_time: std::time::SystemTime::now(),
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance report containing comprehensive statistics
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub query_stats: QueryStats,
    pub cache_stats: CacheStats,
    pub operation_summaries: HashMap<String, OperationSummary>,
    pub report_time: std::time::SystemTime,
}

#[derive(Debug, Clone)]
pub struct OperationSummary {
    pub count: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub max_time: Duration,
    pub min_time: Duration,
}

impl PerformanceReport {
    /// Get cache hit ratio as a percentage
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_requests = self.cache_stats.cache_hits + self.cache_stats.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            (self.cache_stats.cache_hits as f64 / total_requests as f64) * 100.0
        }
    }

    /// Get query success ratio as a percentage
    pub fn query_success_ratio(&self) -> f64 {
        if self.query_stats.total_queries == 0 {
            0.0
        } else {
            (self.query_stats.successful_queries as f64 / self.query_stats.total_queries as f64)
                * 100.0
        }
    }

    /// Format the report as a human-readable string
    pub fn format_summary(&self) -> String {
        format!(
            "Performance Report (generated at {:?})\n\
            Query Statistics:\n\
            - Total queries: {}\n\
            - Successful: {} ({:.1}%)\n\
            - Failed: {}\n\
            - Avg response time: {:?}\n\
            - Min/Max response time: {:?}/{:?}\n\
            Cache Statistics:\n\
            - Cache hits: {}\n\
            - Cache misses: {}\n\
            - Hit ratio: {:.1}%\n\
            - Cache utilization: {}/{} ({:.1}%)\n\
            Operation Summaries: {} operations tracked",
            self.report_time,
            self.query_stats.total_queries,
            self.query_stats.successful_queries,
            self.query_success_ratio(),
            self.query_stats.failed_queries,
            self.query_stats.avg_response_time,
            self.query_stats.min_response_time,
            self.query_stats.max_response_time,
            self.cache_stats.cache_hits,
            self.cache_stats.cache_misses,
            self.cache_hit_ratio(),
            self.cache_stats.cache_size,
            self.cache_stats.cache_capacity,
            if self.cache_stats.cache_capacity > 0 {
                (self.cache_stats.cache_size as f64 / self.cache_stats.cache_capacity as f64)
                    * 100.0
            } else {
                0.0
            },
            self.operation_summaries.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_performance_monitor_basic() {
        let monitor = PerformanceMonitor::new();

        // Record some queries
        monitor.record_query(Duration::from_millis(100), true);
        monitor.record_query(Duration::from_millis(200), true);
        monitor.record_query(Duration::from_millis(50), false);

        let (query_stats, _) = monitor.get_stats();
        assert_eq!(query_stats.total_queries, 3);
        assert_eq!(query_stats.successful_queries, 2);
        assert_eq!(query_stats.failed_queries, 1);
    }

    #[test]
    fn test_cache_stats() {
        let monitor = PerformanceMonitor::new();

        monitor.record_cache_hit();
        monitor.record_cache_hit();
        monitor.record_cache_miss();

        let report = monitor.generate_report();
        assert_eq!(report.cache_stats.cache_hits, 2);
        assert_eq!(report.cache_stats.cache_misses, 1);
        assert!((report.cache_hit_ratio() - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_operation_timings() {
        let monitor = PerformanceMonitor::new();

        monitor.record_operation("search", Duration::from_millis(100));
        monitor.record_operation("search", Duration::from_millis(200));
        monitor.record_operation("embed", Duration::from_millis(50));

        let report = monitor.generate_report();
        assert_eq!(report.operation_summaries.len(), 2);

        let search_summary = report.operation_summaries.get("search").unwrap();
        assert_eq!(search_summary.count, 2);
        assert_eq!(search_summary.avg_time, Duration::from_millis(150));
    }
}
