//! Query hint support for optimization
//!
//! Query hints allow applications to provide guidance to the query optimizer
//! for better performance. Hints can suggest index usage, join strategies,
//! and other optimization techniques.

use serde::{Deserialize, Serialize};

/// Query hints for optimization
#[derive(Debug, Clone, Default)]
pub struct QueryHints {
    /// Suggested index to use (SPO, POS, or OSP)
    pub preferred_index: Option<IndexType>,
    /// Maximum number of results to return
    pub limit: Option<usize>,
    /// Number of results to skip (for pagination)
    pub offset: Option<usize>,
    /// Whether to use bloom filter for existence checks
    pub use_bloom_filter: Option<bool>,
    /// Whether to enable result caching
    pub enable_caching: Option<bool>,
    /// Estimated result size (helps with memory allocation)
    pub estimated_result_size: Option<usize>,
    /// Query timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Whether to collect detailed execution statistics
    pub collect_stats: bool,
}

/// Type of index to use for a query
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// Subject-Predicate-Object index (best for S-first patterns)
    SPO,
    /// Predicate-Object-Subject index (best for P-first patterns)
    POS,
    /// Object-Subject-Predicate index (best for O-first patterns)
    OSP,
}

impl QueryHints {
    /// Create new query hints with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set preferred index
    pub fn with_index(mut self, index: IndexType) -> Self {
        self.preferred_index = Some(index);
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set result offset
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Enable/disable bloom filter
    pub fn with_bloom_filter(mut self, enable: bool) -> Self {
        self.use_bloom_filter = Some(enable);
        self
    }

    /// Enable/disable caching
    pub fn with_caching(mut self, enable: bool) -> Self {
        self.enable_caching = Some(enable);
        self
    }

    /// Set estimated result size
    pub fn with_estimated_size(mut self, size: usize) -> Self {
        self.estimated_result_size = Some(size);
        self
    }

    /// Set query timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Enable statistics collection
    pub fn with_stats(mut self, collect: bool) -> Self {
        self.collect_stats = collect;
        self
    }

    /// Automatically select the best index based on query pattern
    ///
    /// Returns the optimal index type based on which components are bound:
    /// - If S is bound: use SPO
    /// - If P is bound but S is not: use POS
    /// - If O is bound but S and P are not: use OSP
    pub fn auto_select_index(s_bound: bool, p_bound: bool, o_bound: bool) -> IndexType {
        match (s_bound, p_bound, o_bound) {
            (true, _, _) => IndexType::SPO,          // S bound -> SPO
            (false, true, _) => IndexType::POS,      // P bound, S not -> POS
            (false, false, true) => IndexType::OSP,  // O bound, S,P not -> OSP
            (false, false, false) => IndexType::SPO, // All unbound -> SPO (default)
        }
    }

    /// Merge with another set of hints (other takes precedence)
    pub fn merge(&self, other: &QueryHints) -> QueryHints {
        QueryHints {
            preferred_index: other.preferred_index.or(self.preferred_index),
            limit: other.limit.or(self.limit),
            offset: other.offset.or(self.offset),
            use_bloom_filter: other.use_bloom_filter.or(self.use_bloom_filter),
            enable_caching: other.enable_caching.or(self.enable_caching),
            estimated_result_size: other.estimated_result_size.or(self.estimated_result_size),
            timeout_ms: other.timeout_ms.or(self.timeout_ms),
            collect_stats: other.collect_stats || self.collect_stats,
        }
    }

    /// Apply pagination to results
    pub fn apply_pagination<T>(&self, results: Vec<T>) -> Vec<T> {
        let offset = self.offset.unwrap_or(0);
        let limit = self.limit.unwrap_or(usize::MAX);

        results.into_iter().skip(offset).take(limit).collect()
    }
}

/// Query execution statistics
#[derive(Debug, Clone, Default)]
pub struct QueryStats {
    /// Number of pages accessed
    pub pages_accessed: usize,
    /// Number of index entries scanned
    pub index_entries_scanned: usize,
    /// Number of results found
    pub results_found: usize,
    /// Query execution time in microseconds
    pub execution_time_us: u64,
    /// Index used for the query
    pub index_used: Option<IndexType>,
    /// Whether bloom filter was used
    pub bloom_filter_used: bool,
    /// Number of bloom filter false positives
    pub bloom_false_positives: usize,
    /// Whether results were cached
    pub cached: bool,
}

impl QueryStats {
    /// Create new query statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the index used
    pub fn with_index(mut self, index: IndexType) -> Self {
        self.index_used = Some(index);
        self
    }

    /// Record bloom filter usage
    pub fn with_bloom_filter(mut self, used: bool, false_positives: usize) -> Self {
        self.bloom_filter_used = used;
        self.bloom_false_positives = false_positives;
        self
    }

    /// Record execution time
    pub fn with_execution_time(mut self, time_us: u64) -> Self {
        self.execution_time_us = time_us;
        self
    }

    /// Calculate average time per result (in microseconds)
    pub fn avg_time_per_result(&self) -> f64 {
        if self.results_found == 0 {
            0.0
        } else {
            self.execution_time_us as f64 / self.results_found as f64
        }
    }

    /// Calculate bloom filter effectiveness (percentage of false positives)
    pub fn bloom_filter_effectiveness(&self) -> f64 {
        if !self.bloom_filter_used || self.index_entries_scanned == 0 {
            0.0
        } else {
            1.0 - (self.bloom_false_positives as f64 / self.index_entries_scanned as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_hints_builder() {
        let hints = QueryHints::new()
            .with_index(IndexType::SPO)
            .with_limit(100)
            .with_offset(10)
            .with_bloom_filter(true)
            .with_caching(false)
            .with_timeout(5000);

        assert_eq!(hints.preferred_index, Some(IndexType::SPO));
        assert_eq!(hints.limit, Some(100));
        assert_eq!(hints.offset, Some(10));
        assert_eq!(hints.use_bloom_filter, Some(true));
        assert_eq!(hints.enable_caching, Some(false));
        assert_eq!(hints.timeout_ms, Some(5000));
    }

    #[test]
    fn test_auto_select_index() {
        // S bound -> SPO
        assert_eq!(
            QueryHints::auto_select_index(true, false, false),
            IndexType::SPO
        );
        assert_eq!(
            QueryHints::auto_select_index(true, true, false),
            IndexType::SPO
        );
        assert_eq!(
            QueryHints::auto_select_index(true, false, true),
            IndexType::SPO
        );

        // P bound, S not -> POS
        assert_eq!(
            QueryHints::auto_select_index(false, true, false),
            IndexType::POS
        );
        assert_eq!(
            QueryHints::auto_select_index(false, true, true),
            IndexType::POS
        );

        // O bound, S,P not -> OSP
        assert_eq!(
            QueryHints::auto_select_index(false, false, true),
            IndexType::OSP
        );

        // All unbound -> SPO (default)
        assert_eq!(
            QueryHints::auto_select_index(false, false, false),
            IndexType::SPO
        );
    }

    #[test]
    fn test_hints_merge() {
        let hints1 = QueryHints::new().with_limit(100).with_offset(10);

        let hints2 = QueryHints::new().with_limit(50).with_bloom_filter(true);

        let merged = hints1.merge(&hints2);

        assert_eq!(merged.limit, Some(50)); // hints2 takes precedence
        assert_eq!(merged.offset, Some(10)); // from hints1
        assert_eq!(merged.use_bloom_filter, Some(true)); // from hints2
    }

    #[test]
    fn test_apply_pagination() {
        let hints = QueryHints::new().with_limit(3).with_offset(2);

        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let paginated = hints.apply_pagination(data);

        assert_eq!(paginated, vec![3, 4, 5]); // Skip 2, take 3
    }

    #[test]
    fn test_apply_pagination_no_offset() {
        let hints = QueryHints::new().with_limit(3);

        let data = vec![1, 2, 3, 4, 5];
        let paginated = hints.apply_pagination(data);

        assert_eq!(paginated, vec![1, 2, 3]);
    }

    #[test]
    fn test_apply_pagination_no_limit() {
        let hints = QueryHints::new().with_offset(2);

        let data = vec![1, 2, 3, 4, 5];
        let paginated = hints.apply_pagination(data);

        assert_eq!(paginated, vec![3, 4, 5]);
    }

    #[test]
    fn test_query_stats() {
        let stats = QueryStats::new()
            .with_index(IndexType::POS)
            .with_bloom_filter(true, 5)
            .with_execution_time(1000);

        assert_eq!(stats.index_used, Some(IndexType::POS));
        assert!(stats.bloom_filter_used);
        assert_eq!(stats.bloom_false_positives, 5);
        assert_eq!(stats.execution_time_us, 1000);
    }

    #[test]
    fn test_query_stats_avg_time() {
        let mut stats = QueryStats::new().with_execution_time(1000);
        stats.results_found = 10;

        assert_eq!(stats.avg_time_per_result(), 100.0);
    }

    #[test]
    fn test_bloom_filter_effectiveness() {
        let mut stats = QueryStats::new().with_bloom_filter(true, 10);
        stats.index_entries_scanned = 100;

        // 10 false positives out of 100 scans = 90% effectiveness
        assert_eq!(stats.bloom_filter_effectiveness(), 0.9);
    }
}
