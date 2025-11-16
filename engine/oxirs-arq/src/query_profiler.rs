//! SPARQL query profiling for performance analysis
//!
//! This module provides comprehensive profiling capabilities for SPARQL query execution,
//! tracking parse time, planning time, execution time, and resource usage.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Query execution phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryPhase {
    /// Query parsing phase
    Parsing,
    /// Query planning/optimization phase
    Planning,
    /// Query execution phase
    Execution,
    /// Result materialization phase
    Materialization,
}

impl QueryPhase {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            QueryPhase::Parsing => "Parsing",
            QueryPhase::Planning => "Planning",
            QueryPhase::Execution => "Execution",
            QueryPhase::Materialization => "Materialization",
        }
    }
}

/// Statistics for a single query execution
#[derive(Debug, Clone)]
pub struct QueryStats {
    /// Query text or ID
    pub query_id: String,
    /// Total execution time
    pub total_duration: Duration,
    /// Time spent in each phase
    pub phase_durations: HashMap<QueryPhase, Duration>,
    /// Number of triples matched
    pub triples_matched: usize,
    /// Number of results returned
    pub results_count: usize,
    /// Peak memory usage (bytes)
    pub peak_memory: usize,
    /// Number of joins performed
    pub joins_performed: usize,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,
    /// Start timestamp
    pub start_time: Option<Instant>,
    /// End timestamp
    pub end_time: Option<Instant>,
}

impl QueryStats {
    /// Create new query statistics
    pub fn new(query_id: String) -> Self {
        Self {
            query_id,
            total_duration: Duration::default(),
            phase_durations: HashMap::new(),
            triples_matched: 0,
            results_count: 0,
            peak_memory: 0,
            joins_performed: 0,
            cache_hit_rate: 0.0,
            start_time: None,
            end_time: None,
        }
    }

    /// Get duration for a specific phase
    pub fn phase_duration(&self, phase: QueryPhase) -> Duration {
        self.phase_durations
            .get(&phase)
            .copied()
            .unwrap_or_default()
    }

    /// Get throughput (results per second)
    pub fn throughput(&self) -> f64 {
        if self.total_duration.as_secs_f64() > 0.0 {
            self.results_count as f64 / self.total_duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get MB processed per second
    pub fn mb_per_second(&self) -> f64 {
        let mb = self.peak_memory as f64 / (1024.0 * 1024.0);
        if self.total_duration.as_secs_f64() > 0.0 {
            mb / self.total_duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Generate human-readable report
    pub fn report(&self) -> String {
        let mut lines = vec![
            format!("Query Execution Statistics: {}", self.query_id),
            format!(
                "  Total Duration: {:.3}s",
                self.total_duration.as_secs_f64()
            ),
        ];

        // Phase breakdowns
        for phase in &[
            QueryPhase::Parsing,
            QueryPhase::Planning,
            QueryPhase::Execution,
            QueryPhase::Materialization,
        ] {
            if let Some(duration) = self.phase_durations.get(phase) {
                let percentage = if self.total_duration.as_secs_f64() > 0.0 {
                    (duration.as_secs_f64() / self.total_duration.as_secs_f64()) * 100.0
                } else {
                    0.0
                };
                lines.push(format!(
                    "    {}: {:.3}s ({:.1}%)",
                    phase.name(),
                    duration.as_secs_f64(),
                    percentage
                ));
            }
        }

        lines.extend(vec![
            format!("  Triples Matched: {}", self.triples_matched),
            format!("  Results: {}", self.results_count),
            format!("  Joins: {}", self.joins_performed),
            format!("  Cache Hit Rate: {:.1}%", self.cache_hit_rate * 100.0),
            format!(
                "  Peak Memory: {:.2} MB",
                self.peak_memory as f64 / (1024.0 * 1024.0)
            ),
            format!("  Throughput: {:.0} results/s", self.throughput()),
        ]);

        lines.join("\n")
    }
}

/// SPARQL query profiler
pub struct QueryProfiler {
    /// Current query being profiled
    current_query: Option<QueryStats>,
    /// Phase start times
    phase_start: HashMap<QueryPhase, Instant>,
    /// Whether profiling is enabled
    enabled: bool,
    /// History of profiled queries (limited to last N)
    history: Vec<QueryStats>,
    /// Maximum history size
    max_history: usize,
}

impl QueryProfiler {
    /// Create a new query profiler
    pub fn new() -> Self {
        Self {
            current_query: None,
            phase_start: HashMap::new(),
            enabled: true,
            history: Vec::new(),
            max_history: 100,
        }
    }

    /// Create a disabled profiler (no overhead)
    pub fn disabled() -> Self {
        Self {
            current_query: None,
            phase_start: HashMap::new(),
            enabled: false,
            history: Vec::new(),
            max_history: 0,
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start profiling a query
    pub fn start_query(&mut self, query_id: String) {
        if !self.enabled {
            return;
        }

        let mut stats = QueryStats::new(query_id);
        stats.start_time = Some(Instant::now());
        self.current_query = Some(stats);
    }

    /// Start a query phase
    pub fn start_phase(&mut self, phase: QueryPhase) {
        if !self.enabled || self.current_query.is_none() {
            return;
        }

        self.phase_start.insert(phase, Instant::now());
    }

    /// End a query phase
    pub fn end_phase(&mut self, phase: QueryPhase) {
        if !self.enabled || self.current_query.is_none() {
            return;
        }

        if let Some(start) = self.phase_start.remove(&phase) {
            let duration = start.elapsed();
            if let Some(ref mut stats) = self.current_query {
                stats.phase_durations.insert(phase, duration);
            }
        }
    }

    /// Record triples matched
    pub fn record_triples(&mut self, count: usize) {
        if let Some(ref mut stats) = self.current_query {
            stats.triples_matched += count;
        }
    }

    /// Record results count
    pub fn record_results(&mut self, count: usize) {
        if let Some(ref mut stats) = self.current_query {
            stats.results_count = count;
        }
    }

    /// Record memory usage
    pub fn record_memory(&mut self, bytes: usize) {
        if let Some(ref mut stats) = self.current_query {
            stats.peak_memory = stats.peak_memory.max(bytes);
        }
    }

    /// Record join operation
    pub fn record_join(&mut self) {
        if let Some(ref mut stats) = self.current_query {
            stats.joins_performed += 1;
        }
    }

    /// Record cache hit rate
    pub fn record_cache_hit_rate(&mut self, rate: f64) {
        if let Some(ref mut stats) = self.current_query {
            stats.cache_hit_rate = rate.clamp(0.0, 1.0);
        }
    }

    /// End profiling current query
    pub fn end_query(&mut self) -> Option<QueryStats> {
        if !self.enabled {
            return None;
        }

        if let Some(mut stats) = self.current_query.take() {
            stats.end_time = Some(Instant::now());
            if let (Some(start), Some(end)) = (stats.start_time, stats.end_time) {
                stats.total_duration = end.duration_since(start);
            }

            // Add to history
            if self.history.len() >= self.max_history {
                self.history.remove(0);
            }
            self.history.push(stats.clone());

            Some(stats)
        } else {
            None
        }
    }

    /// Get current query statistics
    pub fn current_stats(&self) -> Option<&QueryStats> {
        self.current_query.as_ref()
    }

    /// Get query history
    pub fn history(&self) -> &[QueryStats] {
        &self.history
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get average statistics across all queries in history
    pub fn average_stats(&self) -> Option<AverageStats> {
        if self.history.is_empty() {
            return None;
        }

        let count = self.history.len() as f64;
        let total_duration: Duration = self.history.iter().map(|s| s.total_duration).sum();
        let avg_triples = self
            .history
            .iter()
            .map(|s| s.triples_matched)
            .sum::<usize>() as f64
            / count;
        let avg_results =
            self.history.iter().map(|s| s.results_count).sum::<usize>() as f64 / count;
        let avg_joins = self
            .history
            .iter()
            .map(|s| s.joins_performed)
            .sum::<usize>() as f64
            / count;
        let avg_cache_hit = self.history.iter().map(|s| s.cache_hit_rate).sum::<f64>() / count;
        let avg_memory = self.history.iter().map(|s| s.peak_memory).sum::<usize>() as f64 / count;

        Some(AverageStats {
            query_count: self.history.len(),
            avg_duration: Duration::from_secs_f64(total_duration.as_secs_f64() / count),
            avg_triples_matched: avg_triples,
            avg_results_count: avg_results,
            avg_joins_performed: avg_joins,
            avg_cache_hit_rate: avg_cache_hit,
            avg_peak_memory: avg_memory,
        })
    }

    /// Generate summary report
    pub fn summary_report(&self) -> String {
        if self.history.is_empty() {
            return "No query history available".to_string();
        }

        let mut lines = vec![format!(
            "Query Profiler Summary ({} queries)",
            self.history.len()
        )];

        if let Some(avg) = self.average_stats() {
            lines.push(avg.report());
        }

        lines.join("\n")
    }
}

impl Default for QueryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Average statistics across multiple queries
#[derive(Debug, Clone)]
pub struct AverageStats {
    /// Number of queries
    pub query_count: usize,
    /// Average duration
    pub avg_duration: Duration,
    /// Average triples matched
    pub avg_triples_matched: f64,
    /// Average results count
    pub avg_results_count: f64,
    /// Average joins performed
    pub avg_joins_performed: f64,
    /// Average cache hit rate
    pub avg_cache_hit_rate: f64,
    /// Average peak memory
    pub avg_peak_memory: f64,
}

impl AverageStats {
    /// Generate report
    pub fn report(&self) -> String {
        format!(
            "Average Statistics:\n\
             - Queries: {}\n\
             - Duration: {:.3}s\n\
             - Triples Matched: {:.0}\n\
             - Results: {:.0}\n\
             - Joins: {:.1}\n\
             - Cache Hit Rate: {:.1}%\n\
             - Peak Memory: {:.2} MB",
            self.query_count,
            self.avg_duration.as_secs_f64(),
            self.avg_triples_matched,
            self.avg_results_count,
            self.avg_joins_performed,
            self.avg_cache_hit_rate * 100.0,
            self.avg_peak_memory / (1024.0 * 1024.0)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_query_stats_creation() {
        let stats = QueryStats::new("test_query".to_string());
        assert_eq!(stats.query_id, "test_query");
        assert_eq!(stats.triples_matched, 0);
        assert_eq!(stats.results_count, 0);
    }

    #[test]
    fn test_query_phase_duration() {
        let mut stats = QueryStats::new("test".to_string());
        stats
            .phase_durations
            .insert(QueryPhase::Parsing, Duration::from_millis(100));
        stats
            .phase_durations
            .insert(QueryPhase::Execution, Duration::from_millis(500));

        assert_eq!(stats.phase_duration(QueryPhase::Parsing).as_millis(), 100);
        assert_eq!(stats.phase_duration(QueryPhase::Execution).as_millis(), 500);
        assert_eq!(stats.phase_duration(QueryPhase::Planning).as_millis(), 0);
    }

    #[test]
    fn test_profiler_basic() {
        let mut profiler = QueryProfiler::new();
        assert!(profiler.is_enabled());

        profiler.start_query("SELECT * WHERE { ?s ?p ?o }".to_string());
        thread::sleep(Duration::from_millis(10));

        profiler.record_triples(100);
        profiler.record_results(50);
        profiler.record_join();

        let stats = profiler.end_query().unwrap();
        assert_eq!(stats.triples_matched, 100);
        assert_eq!(stats.results_count, 50);
        assert_eq!(stats.joins_performed, 1);
        assert!(stats.total_duration.as_millis() >= 10);
    }

    #[test]
    fn test_profiler_phases() {
        let mut profiler = QueryProfiler::new();

        profiler.start_query("test".to_string());

        profiler.start_phase(QueryPhase::Parsing);
        thread::sleep(Duration::from_millis(10));
        profiler.end_phase(QueryPhase::Parsing);

        profiler.start_phase(QueryPhase::Execution);
        thread::sleep(Duration::from_millis(20));
        profiler.end_phase(QueryPhase::Execution);

        let stats = profiler.end_query().unwrap();
        assert!(stats.phase_duration(QueryPhase::Parsing).as_millis() >= 10);
        assert!(stats.phase_duration(QueryPhase::Execution).as_millis() >= 20);
    }

    #[test]
    fn test_disabled_profiler() {
        let mut profiler = QueryProfiler::disabled();
        assert!(!profiler.is_enabled());

        profiler.start_query("test".to_string());
        profiler.record_triples(100);

        let stats = profiler.end_query();
        assert!(stats.is_none());
    }

    #[test]
    fn test_profiler_history() {
        let mut profiler = QueryProfiler::new();

        for i in 0..5 {
            profiler.start_query(format!("query_{}", i));
            profiler.record_results(i * 10);
            profiler.end_query();
        }

        assert_eq!(profiler.history().len(), 5);
        assert_eq!(profiler.history()[0].results_count, 0);
        assert_eq!(profiler.history()[4].results_count, 40);
    }

    #[test]
    fn test_average_stats() {
        let mut profiler = QueryProfiler::new();

        for i in 1..=3 {
            profiler.start_query(format!("query_{}", i));
            profiler.record_triples(i * 100);
            profiler.record_results(i * 10);
            profiler.end_query();
        }

        let avg = profiler.average_stats().unwrap();
        assert_eq!(avg.query_count, 3);
        assert_eq!(avg.avg_triples_matched, 200.0); // (100 + 200 + 300) / 3
        assert_eq!(avg.avg_results_count, 20.0); // (10 + 20 + 30) / 3
    }

    #[test]
    fn test_throughput() {
        let mut stats = QueryStats::new("test".to_string());
        stats.results_count = 1000;
        stats.total_duration = Duration::from_secs(2);

        assert_eq!(stats.throughput(), 500.0);
    }

    #[test]
    fn test_cache_hit_rate_clamping() {
        let mut profiler = QueryProfiler::new();
        profiler.start_query("test".to_string());

        profiler.record_cache_hit_rate(1.5); // Should clamp to 1.0
        let stats = profiler.end_query().unwrap();
        assert_eq!(stats.cache_hit_rate, 1.0);
    }
}
