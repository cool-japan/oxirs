//! High-performance triple pattern filtering with auto-vectorization
//!
//! This module provides optimized triple filtering using cache-efficient operations.
//! The Rust compiler auto-vectorizes the comparison loops when SIMD instructions
//! are available on the target CPU (SSE, AVX, NEON, etc.).

use crate::dictionary::NodeId;
use crate::error::Result;
use crate::index::Triple;

/// High-performance triple filter with auto-vectorization
///
/// Uses cache-efficient batch processing for triple pattern matching.
/// The compiler auto-vectorizes comparison operations when possible.
pub struct SimdTripleFilter {
    /// Chunk size for batch processing (optimized for cache lines)
    chunk_size: usize,
    /// Statistics
    stats: FilterStats,
}

/// Filter statistics
#[derive(Debug, Clone, Default)]
pub struct FilterStats {
    /// Total triples processed
    pub triples_processed: u64,
    /// Total triples matched
    pub triples_matched: u64,
    /// Batch operations performed
    pub simd_ops: u64,
    /// Scalar operations performed (for remainder)
    pub scalar_ops: u64,
}

/// Triple pattern for filtering
#[derive(Debug, Clone)]
pub struct SimdTriplePattern {
    /// Subject (None = wildcard)
    pub subject: Option<NodeId>,
    /// Predicate (None = wildcard)
    pub predicate: Option<NodeId>,
    /// Object (None = wildcard)
    pub object: Option<NodeId>,
}

impl SimdTriplePattern {
    /// Create a new pattern matching all triples
    pub fn any() -> Self {
        Self {
            subject: None,
            predicate: None,
            object: None,
        }
    }

    /// Create a pattern with specific subject
    pub fn with_subject(subject: NodeId) -> Self {
        Self {
            subject: Some(subject),
            predicate: None,
            object: None,
        }
    }

    /// Create a pattern with specific predicate
    pub fn with_predicate(predicate: NodeId) -> Self {
        Self {
            subject: None,
            predicate: Some(predicate),
            object: None,
        }
    }

    /// Create a pattern with specific object
    pub fn with_object(object: NodeId) -> Self {
        Self {
            subject: None,
            predicate: None,
            object: Some(object),
        }
    }

    /// Create a pattern matching specific triple
    pub fn exact(subject: NodeId, predicate: NodeId, object: NodeId) -> Self {
        Self {
            subject: Some(subject),
            predicate: Some(predicate),
            object: Some(object),
        }
    }

    /// Check if pattern matches a triple
    #[inline(always)]
    pub fn matches(&self, triple: &Triple) -> bool {
        if let Some(s) = self.subject {
            if s != triple.subject {
                return false;
            }
        }
        if let Some(p) = self.predicate {
            if p != triple.predicate {
                return false;
            }
        }
        if let Some(o) = self.object {
            if o != triple.object {
                return false;
            }
        }
        true
    }
}

impl SimdTripleFilter {
    /// Create a new triple filter
    pub fn new() -> Self {
        Self {
            chunk_size: 64, // Optimized for cache line efficiency
            stats: FilterStats::default(),
        }
    }

    /// Filter triples and return indices of matches
    ///
    /// Returns indices for memory efficiency. Use `filter_triples_owned`
    /// to get a `Vec<Triple>` instead.
    pub fn filter_indices(
        &mut self,
        triples: &[Triple],
        pattern: &SimdTriplePattern,
    ) -> Result<Vec<usize>> {
        let mut matches = Vec::new();
        let n = triples.len();

        self.stats.triples_processed += n as u64;

        // Fast path: no filtering needed
        if pattern.subject.is_none() && pattern.predicate.is_none() && pattern.object.is_none() {
            self.stats.triples_matched += n as u64;
            return Ok((0..n).collect());
        }

        // Process in cache-efficient chunks
        let chunk_size = self.chunk_size;
        for start in (0..n).step_by(chunk_size) {
            let end = (start + chunk_size).min(n);
            self.filter_chunk(triples, pattern, start, end, &mut matches);
            self.stats.simd_ops += 1;
        }

        Ok(matches)
    }

    /// Filter triples and return owned vector
    pub fn filter_triples_owned(
        &mut self,
        triples: &[Triple],
        pattern: &SimdTriplePattern,
    ) -> Result<Vec<Triple>> {
        let indices = self.filter_indices(triples, pattern)?;
        Ok(indices.iter().map(|&i| triples[i]).collect())
    }

    /// Filter chunk of triples (auto-vectorized by compiler)
    #[inline]
    fn filter_chunk(
        &mut self,
        triples: &[Triple],
        pattern: &SimdTriplePattern,
        start: usize,
        end: usize,
        matches: &mut Vec<usize>,
    ) {
        for (offset, triple) in triples[start..end].iter().enumerate() {
            if pattern.matches(triple) {
                matches.push(start + offset);
                self.stats.triples_matched += 1;
            }
        }
    }

    /// Batch filter multiple patterns efficiently
    pub fn filter_batch(
        &mut self,
        triples: &[Triple],
        patterns: &[SimdTriplePattern],
    ) -> Result<Vec<Vec<usize>>> {
        let mut all_matches = vec![Vec::new(); patterns.len()];

        if patterns.is_empty() {
            return Ok(all_matches);
        }

        // Single-pass batch filtering
        for (i, triple) in triples.iter().enumerate() {
            for (pattern_idx, pattern) in patterns.iter().enumerate() {
                if pattern.matches(triple) {
                    all_matches[pattern_idx].push(i);
                }
            }
        }

        self.stats.triples_processed += (triples.len() * patterns.len()) as u64;

        Ok(all_matches)
    }

    /// Count triples matching a pattern (no allocation)
    pub fn count_matches(&mut self, triples: &[Triple], pattern: &SimdTriplePattern) -> u64 {
        let count = triples.iter().filter(|t| pattern.matches(t)).count() as u64;
        self.stats.triples_processed += triples.len() as u64;
        self.stats.triples_matched += count;
        count
    }

    /// Get filter statistics
    pub fn stats(&self) -> &FilterStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = FilterStats::default();
    }

    /// Get chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get batch processing efficiency
    pub fn batch_efficiency(&self) -> f64 {
        let total_ops = self.stats.simd_ops + self.stats.scalar_ops;
        if total_ops == 0 {
            return 0.0;
        }
        (self.stats.simd_ops as f64 / total_ops as f64) * 100.0
    }
}

impl Default for SimdTripleFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_triples(count: usize) -> Vec<Triple> {
        (0..count)
            .map(|i| Triple {
                subject: NodeId::new((i / 100) as u64 + 1),
                predicate: NodeId::new((i / 10) as u64 + 1),
                object: NodeId::new(i as u64 + 1),
            })
            .collect()
    }

    #[test]
    fn test_filter_creation() {
        let filter = SimdTripleFilter::new();
        assert!(filter.chunk_size() > 0);
    }

    #[test]
    fn test_filter_all_pattern() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(1000);
        let pattern = SimdTriplePattern::any();

        let indices = filter.filter_indices(&triples, &pattern).unwrap();
        assert_eq!(indices.len(), 1000);
        assert_eq!(filter.stats().triples_matched, 1000);
    }

    #[test]
    fn test_filter_by_subject() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(1000);
        let pattern = SimdTriplePattern::with_subject(NodeId::new(5));

        let matches = filter.filter_triples_owned(&triples, &pattern).unwrap();

        for triple in &matches {
            assert_eq!(triple.subject, NodeId::new(5));
        }
    }

    #[test]
    fn test_filter_by_predicate() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(1000);
        let pattern = SimdTriplePattern::with_predicate(NodeId::new(10));

        let matches = filter.filter_triples_owned(&triples, &pattern).unwrap();

        for triple in &matches {
            assert_eq!(triple.predicate, NodeId::new(10));
        }
    }

    #[test]
    fn test_filter_by_object() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(1000);
        let pattern = SimdTriplePattern::with_object(NodeId::new(500));

        let matches = filter.filter_triples_owned(&triples, &pattern).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].object, NodeId::new(500));
    }

    #[test]
    fn test_filter_exact_triple() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(1000);
        let pattern = SimdTriplePattern::exact(NodeId::new(5), NodeId::new(10), NodeId::new(500));

        let matches = filter.filter_triples_owned(&triples, &pattern).unwrap();

        for triple in &matches {
            assert_eq!(triple.subject, NodeId::new(5));
            assert_eq!(triple.predicate, NodeId::new(10));
            assert_eq!(triple.object, NodeId::new(500));
        }
    }

    #[test]
    fn test_batch_filtering() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(1000);

        let patterns = vec![
            SimdTriplePattern::with_subject(NodeId::new(1)),
            SimdTriplePattern::with_predicate(NodeId::new(10)),
            SimdTriplePattern::with_object(NodeId::new(100)),
        ];

        let results = filter.filter_batch(&triples, &patterns).unwrap();
        assert_eq!(results.len(), 3);

        for (i, indices) in results.iter().enumerate() {
            for &idx in indices {
                assert!(patterns[i].matches(&triples[idx]));
            }
        }
    }

    #[test]
    fn test_count_matches() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(1000);
        let pattern = SimdTriplePattern::with_subject(NodeId::new(3));

        let count = filter.count_matches(&triples, &pattern);
        let matches = filter.filter_triples_owned(&triples, &pattern).unwrap();

        assert_eq!(count, matches.len() as u64);
    }

    #[test]
    fn test_filter_statistics() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(1000);
        let pattern = SimdTriplePattern::with_subject(NodeId::new(5));

        filter.filter_indices(&triples, &pattern).unwrap();

        let stats = filter.stats();
        assert_eq!(stats.triples_processed, 1000);
        assert!(stats.triples_matched > 0);
    }

    #[test]
    fn test_batch_efficiency() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(1000);
        let pattern = SimdTriplePattern::with_subject(NodeId::new(5));

        filter.filter_indices(&triples, &pattern).unwrap();

        let efficiency = filter.batch_efficiency();
        assert!((0.0..=100.0).contains(&efficiency));
    }

    #[test]
    fn test_reset_stats() {
        let mut filter = SimdTripleFilter::new();
        let triples = create_test_triples(100);
        let pattern = SimdTriplePattern::any();

        filter.filter_indices(&triples, &pattern).unwrap();
        assert!(filter.stats().triples_processed > 0);

        filter.reset_stats();
        assert_eq!(filter.stats().triples_processed, 0);
        assert_eq!(filter.stats().triples_matched, 0);
    }

    #[test]
    fn test_empty_triples() {
        let mut filter = SimdTripleFilter::new();
        let triples = Vec::new();
        let pattern = SimdTriplePattern::any();

        let matches = filter.filter_indices(&triples, &pattern).unwrap();
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_pattern_matching() {
        let pattern = SimdTriplePattern::exact(NodeId::new(1), NodeId::new(2), NodeId::new(3));

        let matching = Triple {
            subject: NodeId::new(1),
            predicate: NodeId::new(2),
            object: NodeId::new(3),
        };

        let non_matching = Triple {
            subject: NodeId::new(1),
            predicate: NodeId::new(2),
            object: NodeId::new(999),
        };

        assert!(pattern.matches(&matching));
        assert!(!pattern.matches(&non_matching));
    }
}
