//! Comprehensive Query Profiling for OxiRS
//!
//! This module provides detailed profiling capabilities for SPARQL queries,
//! leveraging scirs2_core's profiling and metrics infrastructure.
//!
//! # Features
//! - Query execution time tracking
//! - Memory allocation profiling
//! - Cardinality estimation validation
//! - Index usage statistics
//! - Pattern matching performance
//! - Join order analysis
//! - Cache hit rates
//!
//! # Example
//! ```rust,no_run
//! use oxirs_core::query::query_profiler::{QueryProfiler, ProfilerConfig};
//!
//! let config = ProfilerConfig::default();
//! let profiler = QueryProfiler::new(config);
//!
//! let session = profiler.start_session("SELECT ?s ?p ?o WHERE { ?s ?p ?o }");
//! // Execute query...
//! let stats = session.finish();
//!
//! println!("Query took {}ms", stats.total_time_ms);
//! println!("Triples matched: {}", stats.triples_matched);
//! ```

use crate::OxirsError;
use parking_lot::RwLock;
use scirs2_core::metrics::{Counter, Histogram, MetricsRegistry, Timer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Query profiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable detailed profiling (may impact performance)
    pub enable_detailed: bool,

    /// Track memory allocations
    pub track_memory: bool,

    /// Profile pattern matching
    pub profile_patterns: bool,

    /// Profile join operations
    pub profile_joins: bool,

    /// Profile index usage
    pub profile_indexes: bool,

    /// Maximum number of profiled queries to keep in memory
    pub max_history: usize,

    /// Enable slow query logging
    pub slow_query_threshold_ms: u64,

    /// Sample rate (1.0 = profile all queries, 0.1 = profile 10%)
    pub sample_rate: f32,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_detailed: true,
            track_memory: true,
            profile_patterns: true,
            profile_joins: true,
            profile_indexes: true,
            max_history: 1000,
            slow_query_threshold_ms: 1000,
            sample_rate: 1.0,
        }
    }
}

/// Comprehensive query statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatistics {
    /// Total execution time in milliseconds
    pub total_time_ms: u64,

    /// Parsing time in milliseconds
    pub parse_time_ms: u64,

    /// Planning time in milliseconds
    pub planning_time_ms: u64,

    /// Execution time in milliseconds
    pub execution_time_ms: u64,

    /// Number of triples matched
    pub triples_matched: u64,

    /// Number of results produced
    pub results_count: u64,

    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,

    /// Number of pattern matches
    pub pattern_matches: HashMap<String, u64>,

    /// Index access counts
    pub index_accesses: HashMap<String, u64>,

    /// Join operation counts
    pub join_operations: u64,

    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f32,

    /// Query plan hash for comparison
    pub plan_hash: u64,

    /// Timestamp of query execution
    pub timestamp: u64,
}

impl Default for QueryStatistics {
    fn default() -> Self {
        Self {
            total_time_ms: 0,
            parse_time_ms: 0,
            planning_time_ms: 0,
            execution_time_ms: 0,
            triples_matched: 0,
            results_count: 0,
            peak_memory_bytes: 0,
            pattern_matches: HashMap::new(),
            index_accesses: HashMap::new(),
            join_operations: 0,
            cache_hit_rate: 0.0,
            plan_hash: 0,
            timestamp: 0,
        }
    }
}

/// Profiling statistics aggregated across multiple queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingStatistics {
    /// Total number of queries profiled
    pub total_queries: u64,

    /// Average execution time
    pub avg_execution_time_ms: f64,

    /// Median execution time
    pub median_execution_time_ms: f64,

    /// 95th percentile execution time
    pub p95_execution_time_ms: f64,

    /// 99th percentile execution time
    pub p99_execution_time_ms: f64,

    /// Slowest query time
    pub max_execution_time_ms: u64,

    /// Fastest query time
    pub min_execution_time_ms: u64,

    /// Total triples matched
    pub total_triples_matched: u64,

    /// Average triples per query
    pub avg_triples_per_query: f64,

    /// Most common patterns
    pub top_patterns: Vec<(String, u64)>,

    /// Most accessed indexes
    pub top_indexes: Vec<(String, u64)>,

    /// Overall cache hit rate
    pub overall_cache_hit_rate: f32,

    /// Number of slow queries
    pub slow_query_count: u64,
}

impl Default for ProfilingStatistics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            avg_execution_time_ms: 0.0,
            median_execution_time_ms: 0.0,
            p95_execution_time_ms: 0.0,
            p99_execution_time_ms: 0.0,
            max_execution_time_ms: 0,
            min_execution_time_ms: u64::MAX,
            total_triples_matched: 0,
            avg_triples_per_query: 0.0,
            top_patterns: Vec::new(),
            top_indexes: Vec::new(),
            overall_cache_hit_rate: 0.0,
            slow_query_count: 0,
        }
    }
}

/// A profiled query with its associated metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfiledQuery {
    /// Original query text
    pub query_text: String,

    /// Query statistics
    pub statistics: QueryStatistics,

    /// Query type (SELECT, CONSTRUCT, ASK, DESCRIBE, UPDATE)
    pub query_type: String,

    /// Whether this was a slow query
    pub is_slow: bool,

    /// Optimization opportunities identified
    pub optimization_hints: Vec<String>,
}

/// Active profiling session for a single query
pub struct QueryProfilingSession {
    /// Query text
    #[allow(dead_code)]
    query_text: String,

    /// Start time
    start_time: Instant,

    /// Statistics being collected
    statistics: QueryStatistics,

    /// Metric timers
    timers: HashMap<String, Instant>,

    /// Configuration
    config: ProfilerConfig,

    /// Session ID for tracking
    #[allow(dead_code)]
    session_id: String,
}

impl QueryProfilingSession {
    /// Mark the start of a phase (parsing, planning, execution)
    pub fn start_phase(&mut self, phase: &str) {
        self.timers.insert(phase.to_string(), Instant::now());
    }

    /// Mark the end of a phase and record its duration
    pub fn end_phase(&mut self, phase: &str) {
        if let Some(start) = self.timers.remove(phase) {
            let duration = start.elapsed();
            let duration_ms = duration.as_millis() as u64;

            match phase {
                "parse" => self.statistics.parse_time_ms = duration_ms,
                "planning" => self.statistics.planning_time_ms = duration_ms,
                "execution" => self.statistics.execution_time_ms = duration_ms,
                _ => {}
            }
        }
    }

    /// Record a pattern match
    pub fn record_pattern(&mut self, pattern: String) {
        if self.config.profile_patterns {
            *self.statistics.pattern_matches.entry(pattern).or_insert(0) += 1;
        }
    }

    /// Record an index access
    pub fn record_index_access(&mut self, index_name: String) {
        if self.config.profile_indexes {
            *self
                .statistics
                .index_accesses
                .entry(index_name)
                .or_insert(0) += 1;
        }
    }

    /// Record a join operation
    pub fn record_join(&mut self) {
        if self.config.profile_joins {
            self.statistics.join_operations += 1;
        }
    }

    /// Record triples matched
    pub fn record_triples_matched(&mut self, count: u64) {
        self.statistics.triples_matched += count;
    }

    /// Record results produced
    pub fn record_results(&mut self, count: u64) {
        self.statistics.results_count = count;
    }

    /// Record cache hit/miss
    pub fn record_cache_access(&mut self, hit: bool) {
        // Update running average
        let total = self.statistics.triples_matched as f32;
        if total > 0.0 {
            let hits = if hit { 1.0 } else { 0.0 };
            self.statistics.cache_hit_rate =
                (self.statistics.cache_hit_rate * (total - 1.0) + hits) / total;
        }
    }

    /// Set plan hash for deduplication
    pub fn set_plan_hash(&mut self, hash: u64) {
        self.statistics.plan_hash = hash;
    }

    /// Finish profiling and return statistics
    pub fn finish(mut self) -> QueryStatistics {
        let total_duration = self.start_time.elapsed();
        self.statistics.total_time_ms = total_duration.as_millis() as u64;
        self.statistics.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Record peak memory if tracking is enabled
        if self.config.track_memory {
            // Use system memory tracking as scirs2_core memory tracking requires profiling feature
            // Get current process memory usage
            #[cfg(target_os = "linux")]
            {
                if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                    for line in status.lines() {
                        if line.starts_with("VmRSS:") {
                            if let Some(kb_str) = line.split_whitespace().nth(1) {
                                if let Ok(kb) = kb_str.parse::<u64>() {
                                    self.statistics.peak_memory_bytes = kb * 1024;
                                }
                            }
                            break;
                        }
                    }
                }
            }

            #[cfg(target_os = "macos")]
            {
                // Use mach API for macOS
                use std::mem;
                extern "C" {
                    fn mach_task_self() -> u32;
                    fn task_info(
                        task: u32,
                        flavor: u32,
                        task_info: *mut u8,
                        count: *mut u32,
                    ) -> i32;
                }

                const MACH_TASK_BASIC_INFO: u32 = 20;
                const MACH_TASK_BASIC_INFO_COUNT: u32 = 10;

                #[repr(C)]
                struct MachTaskBasicInfo {
                    virtual_size: u64,
                    resident_size: u64,
                    // ... other fields we don't need
                }

                unsafe {
                    let mut info: MachTaskBasicInfo = mem::zeroed();
                    let mut count = MACH_TASK_BASIC_INFO_COUNT;
                    let result = task_info(
                        mach_task_self(),
                        MACH_TASK_BASIC_INFO,
                        &mut info as *mut _ as *mut u8,
                        &mut count,
                    );
                    if result == 0 {
                        self.statistics.peak_memory_bytes = info.resident_size;
                    }
                }
            }

            #[cfg(target_os = "windows")]
            {
                // Use Windows API for memory tracking
                use std::mem;
                #[repr(C)]
                struct ProcessMemoryCounters {
                    cb: u32,
                    page_fault_count: u32,
                    peak_working_set_size: usize,
                    working_set_size: usize,
                    quota_peak_paged_pool_usage: usize,
                    quota_paged_pool_usage: usize,
                    quota_peak_non_paged_pool_usage: usize,
                    quota_non_paged_pool_usage: usize,
                    pagefile_usage: usize,
                    peak_pagefile_usage: usize,
                }

                extern "system" {
                    fn GetCurrentProcess() -> *mut std::ffi::c_void;
                    fn K32GetProcessMemoryInfo(
                        process: *mut std::ffi::c_void,
                        counters: *mut ProcessMemoryCounters,
                        cb: u32,
                    ) -> i32;
                }

                unsafe {
                    let mut counters: ProcessMemoryCounters = mem::zeroed();
                    counters.cb = mem::size_of::<ProcessMemoryCounters>() as u32;
                    let result =
                        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb);
                    if result != 0 {
                        self.statistics.peak_memory_bytes = counters.working_set_size as u64;
                    }
                }
            }
        }

        self.statistics
    }
}

/// Main query profiler
pub struct QueryProfiler {
    /// Configuration
    config: ProfilerConfig,

    /// Profiling history
    history: Arc<RwLock<Vec<ProfiledQuery>>>,

    /// Metric registry
    #[allow(dead_code)]
    metrics: Arc<MetricsRegistry>,

    /// Query execution timer
    query_timer: Arc<Timer>,

    /// Query counter
    query_counter: Arc<Counter>,

    /// Triples matched histogram
    triples_histogram: Arc<Histogram>,
}

impl QueryProfiler {
    /// Create a new query profiler with the given configuration
    pub fn new(config: ProfilerConfig) -> Self {
        let metrics = Arc::new(MetricsRegistry::new());

        let query_timer = Arc::new(Timer::new("query_execution_time".to_string()));
        let query_counter = Arc::new(Counter::new("total_queries".to_string()));
        let triples_histogram = Arc::new(Histogram::new("triples_matched".to_string()));

        Self {
            config,
            history: Arc::new(RwLock::new(Vec::new())),
            metrics,
            query_timer,
            query_counter,
            triples_histogram,
        }
    }

    /// Start a new profiling session for a query
    pub fn start_session(&self, query_text: &str) -> QueryProfilingSession {
        // Increment query counter
        self.query_counter.inc();

        // Generate session ID
        let session_id = format!("query_{}", fastrand::u64(..));

        QueryProfilingSession {
            query_text: query_text.to_string(),
            start_time: Instant::now(),
            statistics: QueryStatistics::default(),
            timers: HashMap::new(),
            config: self.config.clone(),
            session_id,
        }
    }

    /// Record a completed query
    pub fn record_query(
        &self,
        query_text: String,
        statistics: QueryStatistics,
        query_type: String,
    ) {
        // Record metrics
        self.query_timer
            .observe(std::time::Duration::from_millis(statistics.total_time_ms));
        self.triples_histogram
            .observe(statistics.triples_matched as f64);

        // Check if slow query
        let is_slow = statistics.total_time_ms >= self.config.slow_query_threshold_ms;

        // Identify optimization opportunities
        let optimization_hints = self.identify_optimization_hints(&statistics);

        let profiled = ProfiledQuery {
            query_text,
            statistics,
            query_type,
            is_slow,
            optimization_hints,
        };

        // Add to history
        let mut history = self.history.write();
        history.push(profiled);

        // Trim history if needed
        if history.len() > self.config.max_history {
            history.remove(0);
        }
    }

    /// Get aggregated profiling statistics
    pub fn get_statistics(&self) -> ProfilingStatistics {
        let history = self.history.read();

        if history.is_empty() {
            return ProfilingStatistics::default();
        }

        let mut times: Vec<u64> = history.iter().map(|q| q.statistics.total_time_ms).collect();
        times.sort_unstable();

        let total_queries = history.len() as u64;
        let sum_time: u64 = times.iter().sum();
        let avg_time = sum_time as f64 / total_queries as f64;

        let median = times[times.len() / 2];
        let p95_idx = (times.len() as f64 * 0.95) as usize;
        let p99_idx = (times.len() as f64 * 0.99) as usize;
        let p95 = times.get(p95_idx).copied().unwrap_or(0);
        let p99 = times.get(p99_idx).copied().unwrap_or(0);

        let total_triples: u64 = history.iter().map(|q| q.statistics.triples_matched).sum();
        let avg_triples = total_triples as f64 / total_queries as f64;

        let slow_count = history.iter().filter(|q| q.is_slow).count() as u64;

        // Aggregate pattern usage
        let mut pattern_counts: HashMap<String, u64> = HashMap::new();
        for query in history.iter() {
            for (pattern, count) in &query.statistics.pattern_matches {
                *pattern_counts.entry(pattern.clone()).or_insert(0) += count;
            }
        }
        let mut top_patterns: Vec<_> = pattern_counts.into_iter().collect();
        top_patterns.sort_by(|a, b| b.1.cmp(&a.1));
        top_patterns.truncate(10);

        // Aggregate index usage
        let mut index_counts: HashMap<String, u64> = HashMap::new();
        for query in history.iter() {
            for (index, count) in &query.statistics.index_accesses {
                *index_counts.entry(index.clone()).or_insert(0) += count;
            }
        }
        let mut top_indexes: Vec<_> = index_counts.into_iter().collect();
        top_indexes.sort_by(|a, b| b.1.cmp(&a.1));
        top_indexes.truncate(10);

        // Calculate overall cache hit rate
        let total_cache_hits: f32 = history.iter().map(|q| q.statistics.cache_hit_rate).sum();
        let overall_cache_hit_rate = total_cache_hits / total_queries as f32;

        ProfilingStatistics {
            total_queries,
            avg_execution_time_ms: avg_time,
            median_execution_time_ms: median as f64,
            p95_execution_time_ms: p95 as f64,
            p99_execution_time_ms: p99 as f64,
            max_execution_time_ms: *times.last().unwrap_or(&0),
            min_execution_time_ms: *times.first().unwrap_or(&0),
            total_triples_matched: total_triples,
            avg_triples_per_query: avg_triples,
            top_patterns,
            top_indexes,
            overall_cache_hit_rate,
            slow_query_count: slow_count,
        }
    }

    /// Get recent slow queries
    pub fn get_slow_queries(&self, limit: usize) -> Vec<ProfiledQuery> {
        let history = self.history.read();
        history
            .iter()
            .filter(|q| q.is_slow)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Clear profiling history
    pub fn clear_history(&self) {
        self.history.write().clear();
    }

    /// Export profiling data as JSON
    pub fn export_json(&self) -> Result<String, OxirsError> {
        let stats = self.get_statistics();
        serde_json::to_string_pretty(&stats).map_err(|e| {
            OxirsError::Serialize(format!("Failed to serialize profiling data: {}", e))
        })
    }

    /// Identify optimization hints based on statistics
    fn identify_optimization_hints(&self, stats: &QueryStatistics) -> Vec<String> {
        let mut hints = Vec::new();

        // 1. Execution time analysis
        if stats.execution_time_ms > self.config.slow_query_threshold_ms {
            hints.push(format!(
                "⚠️  Slow execution ({}ms) - consider adding indexes or optimizing patterns",
                stats.execution_time_ms
            ));

            // Additional context if parsing is slow
            if stats.parse_time_ms > stats.execution_time_ms / 4 {
                hints.push(format!(
                    "💡 High parse time ({}ms, {:.1}% of total) - consider caching parsed queries",
                    stats.parse_time_ms,
                    (stats.parse_time_ms as f64 / stats.total_time_ms as f64) * 100.0
                ));
            }

            // Planning overhead
            if stats.planning_time_ms > stats.execution_time_ms / 4 {
                hints.push(format!(
                    "💡 High planning time ({}ms, {:.1}% of total) - enable query plan caching",
                    stats.planning_time_ms,
                    (stats.planning_time_ms as f64 / stats.total_time_ms as f64) * 100.0
                ));
            }
        }

        // 2. Cache effectiveness
        if stats.cache_hit_rate < 0.5 {
            hints.push(format!(
                "💾 Low cache hit rate ({:.1}%) - query may benefit from result caching",
                stats.cache_hit_rate * 100.0
            ));
        } else if stats.cache_hit_rate > 0.9 {
            hints.push(format!(
                "✅ Excellent cache hit rate ({:.1}%) - caching is working well",
                stats.cache_hit_rate * 100.0
            ));
        }

        // 3. Join optimization
        if stats.join_operations > 10 {
            hints.push(format!(
                "🔗 Many join operations ({}) - consider reordering patterns for better selectivity",
                stats.join_operations
            ));

            // Specific advice for join-heavy queries
            if stats.join_operations > 20 {
                hints.push(
                    "💡 Excessive joins - break query into smaller subqueries or use UNION instead"
                        .to_string(),
                );
            }
        }

        // 4. Selectivity analysis
        if stats.triples_matched > 10000 && stats.results_count < 100 {
            let selectivity = stats.results_count as f64 / stats.triples_matched as f64;
            hints.push(format!(
                "🎯 High selectivity gap (matched {} triples, returned {} results, {:.3}% selectivity) - add more selective patterns early",
                stats.triples_matched, stats.results_count, selectivity * 100.0
            ));
        }

        // 5. Pattern-specific hints
        if !stats.pattern_matches.is_empty() {
            // Find most expensive pattern
            if let Some((pattern, count)) = stats.pattern_matches.iter().max_by_key(|(_, c)| *c) {
                if *count > stats.pattern_matches.len() as u64 * 2 {
                    hints.push(format!(
                        "📊 Pattern '{}' heavily used ({} times) - ensure it has appropriate index",
                        pattern, count
                    ));
                }
            }
        }

        // 6. Index usage analysis
        if !stats.index_accesses.is_empty() {
            let total_accesses: u64 = stats.index_accesses.values().sum();
            if total_accesses > 1000 {
                hints.push(format!(
                    "🗂️  High index access count ({}) - consider index consolidation or query simplification",
                    total_accesses
                ));
            }

            // Check for missing index hints
            if stats.pattern_matches.len() > stats.index_accesses.len() {
                hints.push(
                    "💡 Some patterns may not be using indexes - review query structure"
                        .to_string(),
                );
            }
        }

        // 7. Memory usage hints
        if stats.peak_memory_bytes > 100 * 1024 * 1024 {
            // >100MB
            hints.push(format!(
                "💾 High memory usage ({:.1}MB) - consider streaming results or pagination",
                stats.peak_memory_bytes as f64 / (1024.0 * 1024.0)
            ));
        }

        // 8. Results size hints
        if stats.results_count == 0 {
            hints.push(
                "ℹ️  Query returned no results - verify query logic and data availability"
                    .to_string(),
            );
        } else if stats.results_count > 10000 {
            hints.push(format!(
                "📈 Large result set ({} results) - consider adding LIMIT clause or pagination",
                stats.results_count
            ));
        }

        // 9. Overall performance assessment
        let efficiency_score = if stats.triples_matched > 0 {
            (stats.results_count as f64 / stats.triples_matched as f64) * 1000.0
                / stats.total_time_ms as f64
        } else {
            0.0
        };

        if efficiency_score < 0.1 && stats.results_count > 0 {
            hints.push(
                "⚡ Low query efficiency - review overall query structure and indexing strategy"
                    .to_string(),
            );
        }

        hints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = QueryProfiler::new(config);

        let stats = profiler.get_statistics();
        assert_eq!(stats.total_queries, 0);
    }

    #[test]
    fn test_session_lifecycle() {
        let config = ProfilerConfig::default();
        let profiler = QueryProfiler::new(config);

        let mut session = profiler.start_session("SELECT * WHERE { ?s ?p ?o }");

        session.start_phase("parse");
        std::thread::sleep(std::time::Duration::from_millis(10));
        session.end_phase("parse");

        session.start_phase("planning");
        std::thread::sleep(std::time::Duration::from_millis(10));
        session.end_phase("planning");

        session.start_phase("execution");
        session.record_triples_matched(100);
        session.record_results(10);
        std::thread::sleep(std::time::Duration::from_millis(10));
        session.end_phase("execution");

        let stats = session.finish();

        assert!(stats.total_time_ms >= 30);
        assert_eq!(stats.triples_matched, 100);
        assert_eq!(stats.results_count, 10);
    }

    #[test]
    fn test_pattern_recording() {
        let config = ProfilerConfig::default();
        let profiler = QueryProfiler::new(config);

        let mut session = profiler.start_session("SELECT * WHERE { ?s ?p ?o }");

        session.record_pattern("SPO".to_string());
        session.record_pattern("SPO".to_string());
        session.record_pattern("POS".to_string());

        let stats = session.finish();

        assert_eq!(stats.pattern_matches.get("SPO"), Some(&2));
        assert_eq!(stats.pattern_matches.get("POS"), Some(&1));
    }

    #[test]
    fn test_optimization_hints() {
        let config = ProfilerConfig {
            slow_query_threshold_ms: 100,
            ..Default::default()
        };
        let profiler = QueryProfiler::new(config);

        let stats = QueryStatistics {
            total_time_ms: 200,
            execution_time_ms: 200,
            cache_hit_rate: 0.3,
            join_operations: 15,
            triples_matched: 50000,
            results_count: 50,
            ..Default::default()
        };

        let hints = profiler.identify_optimization_hints(&stats);

        assert!(!hints.is_empty());
        assert!(hints.iter().any(|h| h.contains("Slow execution")));
        assert!(hints.iter().any(|h| h.contains("Low cache hit rate")));
        assert!(hints.iter().any(|h| h.contains("Many join operations")));
        assert!(hints.iter().any(|h| h.contains("High selectivity gap")));
    }

    #[test]
    fn test_statistics_aggregation() {
        let config = ProfilerConfig::default();
        let profiler = QueryProfiler::new(config);

        // Record multiple queries
        for i in 0..10 {
            let stats = QueryStatistics {
                total_time_ms: 100 + i * 10,
                triples_matched: 1000 + i * 100,
                ..Default::default()
            };

            profiler.record_query(format!("Query {}", i), stats, "SELECT".to_string());
        }

        let agg_stats = profiler.get_statistics();

        assert_eq!(agg_stats.total_queries, 10);
        assert!(agg_stats.avg_execution_time_ms > 0.0);
        assert!(agg_stats.avg_triples_per_query > 0.0);
    }
}
