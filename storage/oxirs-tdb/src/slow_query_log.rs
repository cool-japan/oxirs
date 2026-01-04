//! # Slow Query Logging and Analysis
//!
//! Provides comprehensive slow query logging for production monitoring and
//! performance analysis. Captures detailed query execution metrics and
//! provides analysis tools for identifying bottlenecks.

use crate::dictionary::NodeId;
use crate::query_hints::IndexType;
use scirs2_core::metrics::{Counter, Histogram, Timer};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Configuration for slow query logging
#[derive(Debug, Clone)]
pub struct SlowQueryConfig {
    /// Threshold for logging slow queries (milliseconds)
    pub threshold_ms: u64,
    /// Maximum number of slow queries to keep in memory
    pub max_entries: usize,
    /// Enable detailed execution plan logging
    pub log_execution_plan: bool,
    /// Enable stack trace capture for very slow queries
    pub capture_stack_trace: bool,
    /// Very slow query threshold for stack trace (milliseconds)
    pub very_slow_threshold_ms: u64,
    /// Enable automatic analysis
    pub enable_analysis: bool,
}

impl Default for SlowQueryConfig {
    fn default() -> Self {
        Self {
            threshold_ms: 100,            // Log queries > 100ms
            max_entries: 1000,            // Keep last 1000 slow queries
            log_execution_plan: true,     // Log execution plans
            capture_stack_trace: false,   // Disabled by default (performance cost)
            very_slow_threshold_ms: 1000, // Stack trace for queries > 1s
            enable_analysis: true,        // Enable analysis
        }
    }
}

/// Slow query log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQueryEntry {
    /// Unique query ID
    pub query_id: u64,
    /// Timestamp when query started
    pub timestamp: SystemTime,
    /// Query execution time
    pub duration_ms: u64,
    /// Query pattern (subject, predicate, object)
    pub pattern: QueryPattern,
    /// Index used for execution
    pub index_used: Option<IndexType>,
    /// Number of results returned
    pub result_count: usize,
    /// Number of triples scanned
    pub triples_scanned: usize,
    /// Execution plan explanation
    pub execution_plan: Option<String>,
    /// Error message if query failed
    pub error: Option<String>,
    /// Transaction ID
    pub transaction_id: Option<u64>,
    /// Stack trace (if captured)
    pub stack_trace: Option<String>,
}

/// Query pattern for logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Subject (Some for bound, None for unbound)
    pub subject: Option<NodeId>,
    /// Predicate (Some for bound, None for unbound)
    pub predicate: Option<NodeId>,
    /// Object (Some for bound, None for unbound)
    pub object: Option<NodeId>,
}

/// Query finish information for logging
#[derive(Debug, Clone)]
pub struct QueryFinishInfo {
    /// Query pattern (subject, predicate, object)
    pub pattern: QueryPattern,
    /// Index used for execution
    pub index_used: Option<IndexType>,
    /// Number of results returned
    pub result_count: usize,
    /// Number of triples scanned
    pub triples_scanned: usize,
    /// Execution plan explanation
    pub execution_plan: Option<String>,
    /// Error message if query failed
    pub error: Option<String>,
    /// Transaction ID
    pub transaction_id: Option<u64>,
}

impl QueryPattern {
    /// Create new query pattern
    pub fn new(s: Option<NodeId>, p: Option<NodeId>, o: Option<NodeId>) -> Self {
        Self {
            subject: s,
            predicate: p,
            object: o,
        }
    }

    /// Get pattern signature for grouping
    pub fn signature(&self) -> String {
        format!(
            "{}-{}-{}",
            if self.subject.is_some() { "B" } else { "U" },
            if self.predicate.is_some() { "B" } else { "U" },
            if self.object.is_some() { "B" } else { "U" }
        )
    }

    /// Count bound positions
    pub fn bound_count(&self) -> usize {
        let mut count = 0;
        if self.subject.is_some() {
            count += 1;
        }
        if self.predicate.is_some() {
            count += 1;
        }
        if self.object.is_some() {
            count += 1;
        }
        count
    }
}

/// Analysis summary for slow queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQueryAnalysis {
    /// Total slow queries
    pub total_slow_queries: usize,
    /// Average duration of slow queries (ms)
    pub avg_duration_ms: f64,
    /// Maximum duration (ms)
    pub max_duration_ms: u64,
    /// Most common query patterns
    pub common_patterns: Vec<(String, usize)>,
    /// Most expensive query patterns (by total time)
    pub expensive_patterns: Vec<(String, f64)>,
    /// Index usage statistics
    pub index_usage: std::collections::HashMap<IndexType, usize>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Slow query logger
pub struct SlowQueryLogger {
    /// Configuration
    config: SlowQueryConfig,
    /// Slow query log entries
    entries: parking_lot::RwLock<VecDeque<SlowQueryEntry>>,
    /// Next query ID
    next_query_id: AtomicU64,
    /// Metrics
    slow_query_counter: Counter,
    query_duration_histogram: Histogram,
    very_slow_query_counter: Counter,
}

impl SlowQueryLogger {
    /// Create a new slow query logger
    pub fn new(config: SlowQueryConfig) -> Self {
        Self {
            config,
            entries: parking_lot::RwLock::new(VecDeque::new()),
            next_query_id: AtomicU64::new(1),
            slow_query_counter: Counter::new("slow_queries_total".to_string()),
            query_duration_histogram: Histogram::new("slow_query_duration_ms".to_string()),
            very_slow_query_counter: Counter::new("very_slow_queries_total".to_string()),
        }
    }

    /// Start tracking a query
    pub fn start_query(&self) -> QueryTracker<'_> {
        QueryTracker {
            logger: self,
            query_id: self.next_query_id.fetch_add(1, Ordering::Relaxed),
            start_time: Instant::now(),
            timestamp: SystemTime::now(),
        }
    }

    /// Log a slow query
    pub fn log_slow_query(&self, entry: SlowQueryEntry) {
        // Update metrics
        self.slow_query_counter.inc();
        self.query_duration_histogram
            .observe(entry.duration_ms as f64);

        if entry.duration_ms >= self.config.very_slow_threshold_ms {
            self.very_slow_query_counter.inc();
        }

        // Add to log
        let mut entries = self.entries.write();
        entries.push_back(entry);

        // Enforce maximum size
        while entries.len() > self.config.max_entries {
            entries.pop_front();
        }
    }

    /// Get slow query entries
    pub fn get_entries(&self) -> Vec<SlowQueryEntry> {
        self.entries.read().iter().cloned().collect()
    }

    /// Get entries matching a pattern signature
    pub fn get_entries_by_pattern(&self, signature: &str) -> Vec<SlowQueryEntry> {
        self.entries
            .read()
            .iter()
            .filter(|e| e.pattern.signature() == signature)
            .cloned()
            .collect()
    }

    /// Get slowest queries
    pub fn get_slowest(&self, limit: usize) -> Vec<SlowQueryEntry> {
        let mut entries = self.get_entries();
        entries.sort_by(|a, b| b.duration_ms.cmp(&a.duration_ms));
        entries.truncate(limit);
        entries
    }

    /// Analyze slow queries
    pub fn analyze(&self) -> SlowQueryAnalysis {
        let entries = self.get_entries();

        if entries.is_empty() {
            return SlowQueryAnalysis {
                total_slow_queries: 0,
                avg_duration_ms: 0.0,
                max_duration_ms: 0,
                common_patterns: Vec::new(),
                expensive_patterns: Vec::new(),
                index_usage: std::collections::HashMap::new(),
                recommendations: Vec::new(),
            };
        }

        // Calculate statistics
        let total_duration: u64 = entries.iter().map(|e| e.duration_ms).sum();
        let avg_duration = total_duration as f64 / entries.len() as f64;
        let max_duration = entries.iter().map(|e| e.duration_ms).max().unwrap_or(0);

        // Pattern frequency analysis
        let mut pattern_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut pattern_total_time: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();

        for entry in &entries {
            let signature = entry.pattern.signature();
            *pattern_counts.entry(signature.clone()).or_insert(0) += 1;
            *pattern_total_time.entry(signature).or_insert(0) += entry.duration_ms;
        }

        // Most common patterns
        let mut common_patterns: Vec<(String, usize)> = pattern_counts.into_iter().collect();
        common_patterns.sort_by(|a, b| b.1.cmp(&a.1));
        common_patterns.truncate(10);

        // Most expensive patterns (by total time)
        let mut expensive_patterns: Vec<(String, f64)> = pattern_total_time
            .into_iter()
            .map(|(k, v)| (k, v as f64))
            .collect();
        expensive_patterns
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        expensive_patterns.truncate(10);

        // Index usage statistics
        let mut index_usage: std::collections::HashMap<IndexType, usize> =
            std::collections::HashMap::new();
        for entry in &entries {
            if let Some(index) = entry.index_used {
                *index_usage.entry(index).or_insert(0) += 1;
            }
        }

        // Generate recommendations
        let recommendations = self.generate_recommendations(&entries);

        SlowQueryAnalysis {
            total_slow_queries: entries.len(),
            avg_duration_ms: avg_duration,
            max_duration_ms: max_duration,
            common_patterns,
            expensive_patterns,
            index_usage,
            recommendations,
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self, entries: &[SlowQueryEntry]) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for full scans
        let full_scans = entries
            .iter()
            .filter(|e| e.pattern.bound_count() == 0)
            .count();

        if full_scans > entries.len() / 10 {
            recommendations.push(format!(
                "High number of full scans detected ({} out of {}). Consider adding filters or indexes.",
                full_scans,
                entries.len()
            ));
        }

        // Check for very slow queries
        let very_slow = entries
            .iter()
            .filter(|e| e.duration_ms >= self.config.very_slow_threshold_ms)
            .count();

        if very_slow > 0 {
            recommendations.push(format!(
                "{} queries exceeded {} ms threshold. Investigate execution plans and consider query optimization.",
                very_slow, self.config.very_slow_threshold_ms
            ));
        }

        // Check for low selectivity queries
        let low_selectivity = entries
            .iter()
            .filter(|e| e.result_count > 0 && e.triples_scanned > e.result_count * 10)
            .count();

        if low_selectivity > entries.len() / 5 {
            recommendations.push(format!(
                "{} queries have low selectivity (scanning many triples for few results). Consider adding more specific filters.",
                low_selectivity
            ));
        }

        // Check average duration
        let avg_duration: f64 =
            entries.iter().map(|e| e.duration_ms as f64).sum::<f64>() / entries.len() as f64;

        if avg_duration > (self.config.threshold_ms as f64 * 5.0) {
            recommendations.push(format!(
                "Average slow query duration ({:.2} ms) is significantly above threshold. Consider system-wide optimization.",
                avg_duration
            ));
        }

        if recommendations.is_empty() {
            recommendations.push("No major issues detected. Monitor regularly.".to_string());
        }

        recommendations
    }

    /// Clear all slow query entries
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Get configuration
    pub fn config(&self) -> &SlowQueryConfig {
        &self.config
    }

    /// Get metrics
    pub fn get_metrics(&self) -> SlowQueryMetrics {
        let histogram_stats = self.query_duration_histogram.get_stats();

        SlowQueryMetrics {
            total_slow_queries: self.slow_query_counter.get(),
            very_slow_queries: self.very_slow_query_counter.get(),
            avg_duration_ms: histogram_stats.mean,
            entries_in_memory: self.entries.read().len(),
        }
    }
}

/// Metrics for slow query logger
#[derive(Debug, Clone)]
pub struct SlowQueryMetrics {
    /// Total slow queries logged
    pub total_slow_queries: u64,
    /// Total very slow queries
    pub very_slow_queries: u64,
    /// Average duration
    pub avg_duration_ms: f64,
    /// Current entries in memory
    pub entries_in_memory: usize,
}

/// Query tracker for automatic logging
pub struct QueryTracker<'a> {
    logger: &'a SlowQueryLogger,
    query_id: u64,
    start_time: Instant,
    timestamp: SystemTime,
}

impl<'a> QueryTracker<'a> {
    /// Finish tracking and log if slow
    pub fn finish(self, info: QueryFinishInfo) {
        let duration = self.start_time.elapsed();
        let duration_ms = duration.as_millis() as u64;

        // Only log if exceeds threshold
        if duration_ms >= self.logger.config.threshold_ms {
            let stack_trace = if self.logger.config.capture_stack_trace
                && duration_ms >= self.logger.config.very_slow_threshold_ms
            {
                // In production, this would capture actual stack trace
                // For now, we just note that it would be captured
                Some("Stack trace would be captured here".to_string())
            } else {
                None
            };

            let entry = SlowQueryEntry {
                query_id: self.query_id,
                timestamp: self.timestamp,
                duration_ms,
                pattern: info.pattern,
                index_used: info.index_used,
                result_count: info.result_count,
                triples_scanned: info.triples_scanned,
                execution_plan: info.execution_plan,
                error: info.error,
                transaction_id: info.transaction_id,
                stack_trace,
            };

            self.logger.log_slow_query(entry);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slow_query_logger_creation() {
        let config = SlowQueryConfig::default();
        let logger = SlowQueryLogger::new(config);

        let metrics = logger.get_metrics();
        assert_eq!(metrics.total_slow_queries, 0);
        assert_eq!(metrics.entries_in_memory, 0);
    }

    #[test]
    fn test_query_pattern_signature() {
        let pattern1 = QueryPattern::new(Some(NodeId::new(1)), None, None);
        assert_eq!(pattern1.signature(), "B-U-U");

        let pattern2 = QueryPattern::new(None, Some(NodeId::new(2)), None);
        assert_eq!(pattern2.signature(), "U-B-U");

        let pattern3 = QueryPattern::new(
            Some(NodeId::new(1)),
            Some(NodeId::new(2)),
            Some(NodeId::new(3)),
        );
        assert_eq!(pattern3.signature(), "B-B-B");
    }

    #[test]
    fn test_log_slow_query() {
        let config = SlowQueryConfig {
            threshold_ms: 50,
            ..Default::default()
        };
        let logger = SlowQueryLogger::new(config);

        let entry = SlowQueryEntry {
            query_id: 1,
            timestamp: SystemTime::now(),
            duration_ms: 150,
            pattern: QueryPattern::new(None, None, None),
            index_used: Some(IndexType::SPO),
            result_count: 100,
            triples_scanned: 1000,
            execution_plan: Some("Full scan using SPO index".to_string()),
            error: None,
            transaction_id: Some(42),
            stack_trace: None,
        };

        logger.log_slow_query(entry);

        let entries = logger.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].duration_ms, 150);
    }

    #[test]
    fn test_max_entries_limit() {
        let config = SlowQueryConfig {
            max_entries: 5,
            ..Default::default()
        };
        let logger = SlowQueryLogger::new(config);

        // Log 10 entries
        for i in 0..10 {
            let entry = SlowQueryEntry {
                query_id: i,
                timestamp: SystemTime::now(),
                duration_ms: 100 + i,
                pattern: QueryPattern::new(None, None, None),
                index_used: Some(IndexType::SPO),
                result_count: 10,
                triples_scanned: 100,
                execution_plan: None,
                error: None,
                transaction_id: None,
                stack_trace: None,
            };
            logger.log_slow_query(entry);
        }

        let entries = logger.get_entries();
        assert_eq!(entries.len(), 5);
        // Should keep the latest 5 (IDs 5-9)
        assert_eq!(entries[0].query_id, 5);
        assert_eq!(entries[4].query_id, 9);
    }

    #[test]
    fn test_get_slowest_queries() {
        let logger = SlowQueryLogger::new(SlowQueryConfig::default());

        // Log queries with different durations
        for (id, duration) in [(1, 100), (2, 500), (3, 200), (4, 800), (5, 150)] {
            let entry = SlowQueryEntry {
                query_id: id,
                timestamp: SystemTime::now(),
                duration_ms: duration,
                pattern: QueryPattern::new(None, None, None),
                index_used: Some(IndexType::SPO),
                result_count: 10,
                triples_scanned: 100,
                execution_plan: None,
                error: None,
                transaction_id: None,
                stack_trace: None,
            };
            logger.log_slow_query(entry);
        }

        let slowest = logger.get_slowest(3);
        assert_eq!(slowest.len(), 3);
        assert_eq!(slowest[0].duration_ms, 800); // ID 4
        assert_eq!(slowest[1].duration_ms, 500); // ID 2
        assert_eq!(slowest[2].duration_ms, 200); // ID 3
    }

    #[test]
    fn test_analyze_slow_queries() {
        let logger = SlowQueryLogger::new(SlowQueryConfig::default());

        // Log various query patterns
        for i in 0..10 {
            let pattern = if i % 3 == 0 {
                QueryPattern::new(None, None, None) // Full scan
            } else if i % 3 == 1 {
                QueryPattern::new(Some(NodeId::new(1)), None, None) // Subject bound
            } else {
                QueryPattern::new(
                    Some(NodeId::new(1)),
                    Some(NodeId::new(2)),
                    Some(NodeId::new(3)),
                ) // All bound
            };

            let entry = SlowQueryEntry {
                query_id: i,
                timestamp: SystemTime::now(),
                duration_ms: 100 + i * 10,
                pattern,
                index_used: Some(IndexType::SPO),
                result_count: 10,
                triples_scanned: 100,
                execution_plan: None,
                error: None,
                transaction_id: None,
                stack_trace: None,
            };
            logger.log_slow_query(entry);
        }

        let analysis = logger.analyze();
        assert_eq!(analysis.total_slow_queries, 10);
        assert!(analysis.avg_duration_ms > 0.0);
        assert!(!analysis.common_patterns.is_empty());
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_pattern_grouping() {
        let logger = SlowQueryLogger::new(SlowQueryConfig::default());

        // Log same pattern multiple times
        for i in 0..5 {
            let entry = SlowQueryEntry {
                query_id: i,
                timestamp: SystemTime::now(),
                duration_ms: 100,
                pattern: QueryPattern::new(Some(NodeId::new(1)), None, None),
                index_used: Some(IndexType::SPO),
                result_count: 10,
                triples_scanned: 100,
                execution_plan: None,
                error: None,
                transaction_id: None,
                stack_trace: None,
            };
            logger.log_slow_query(entry);
        }

        let entries = logger.get_entries_by_pattern("B-U-U");
        assert_eq!(entries.len(), 5);
    }

    #[test]
    fn test_query_tracker() {
        let logger = SlowQueryLogger::new(SlowQueryConfig {
            threshold_ms: 50,
            ..Default::default()
        });

        let tracker = logger.start_query();
        std::thread::sleep(Duration::from_millis(60));

        tracker.finish(QueryFinishInfo {
            pattern: QueryPattern::new(None, None, None),
            index_used: Some(IndexType::SPO),
            result_count: 100,
            triples_scanned: 1000,
            execution_plan: Some("Test plan".to_string()),
            error: None,
            transaction_id: Some(42),
        });

        let entries = logger.get_entries();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].duration_ms >= 50);
    }

    #[test]
    fn test_below_threshold_not_logged() {
        let logger = SlowQueryLogger::new(SlowQueryConfig {
            threshold_ms: 1000,
            ..Default::default()
        });

        let tracker = logger.start_query();
        std::thread::sleep(Duration::from_millis(10));

        tracker.finish(QueryFinishInfo {
            pattern: QueryPattern::new(None, None, None),
            index_used: Some(IndexType::SPO),
            result_count: 10,
            triples_scanned: 100,
            execution_plan: None,
            error: None,
            transaction_id: None,
        });

        let entries = logger.get_entries();
        assert_eq!(entries.len(), 0); // Should not be logged
    }

    #[test]
    fn test_clear_entries() {
        let logger = SlowQueryLogger::new(SlowQueryConfig::default());

        for i in 0..5 {
            let entry = SlowQueryEntry {
                query_id: i,
                timestamp: SystemTime::now(),
                duration_ms: 100,
                pattern: QueryPattern::new(None, None, None),
                index_used: Some(IndexType::SPO),
                result_count: 10,
                triples_scanned: 100,
                execution_plan: None,
                error: None,
                transaction_id: None,
                stack_trace: None,
            };
            logger.log_slow_query(entry);
        }

        assert_eq!(logger.get_entries().len(), 5);

        logger.clear();
        assert_eq!(logger.get_entries().len(), 0);
    }

    #[test]
    fn test_recommendations_full_scans() {
        let logger = SlowQueryLogger::new(SlowQueryConfig::default());

        // Log mostly full scans
        for i in 0..20 {
            let pattern = if i < 15 {
                QueryPattern::new(None, None, None) // Full scan
            } else {
                QueryPattern::new(Some(NodeId::new(1)), None, None)
            };

            let entry = SlowQueryEntry {
                query_id: i,
                timestamp: SystemTime::now(),
                duration_ms: 100,
                pattern,
                index_used: Some(IndexType::SPO),
                result_count: 100,
                triples_scanned: 1000,
                execution_plan: None,
                error: None,
                transaction_id: None,
                stack_trace: None,
            };
            logger.log_slow_query(entry);
        }

        let analysis = logger.analyze();
        assert!(!analysis.recommendations.is_empty());
        // Should recommend addressing full scans
        assert!(analysis
            .recommendations
            .iter()
            .any(|r| r.contains("full scan")));
    }

    #[test]
    fn test_metrics_tracking() {
        let logger = SlowQueryLogger::new(SlowQueryConfig {
            threshold_ms: 50,
            very_slow_threshold_ms: 200,
            ..Default::default()
        });

        // Log a slow query
        let entry1 = SlowQueryEntry {
            query_id: 1,
            timestamp: SystemTime::now(),
            duration_ms: 150,
            pattern: QueryPattern::new(None, None, None),
            index_used: Some(IndexType::SPO),
            result_count: 10,
            triples_scanned: 100,
            execution_plan: None,
            error: None,
            transaction_id: None,
            stack_trace: None,
        };
        logger.log_slow_query(entry1);

        // Log a very slow query
        let entry2 = SlowQueryEntry {
            query_id: 2,
            timestamp: SystemTime::now(),
            duration_ms: 500,
            pattern: QueryPattern::new(None, None, None),
            index_used: Some(IndexType::SPO),
            result_count: 10,
            triples_scanned: 100,
            execution_plan: None,
            error: None,
            transaction_id: None,
            stack_trace: None,
        };
        logger.log_slow_query(entry2);

        let metrics = logger.get_metrics();
        assert_eq!(metrics.total_slow_queries, 2);
        assert_eq!(metrics.very_slow_queries, 1);
        assert!(metrics.avg_duration_ms > 0.0);
    }
}
