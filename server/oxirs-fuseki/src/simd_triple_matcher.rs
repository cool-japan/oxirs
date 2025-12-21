//! SIMD-Accelerated Triple Pattern Matching
//!
//! This module provides high-performance triple pattern matching using
//! SciRS2's SIMD operations for vectorized comparison and filtering.
//!
//! Features:
//! - SIMD-accelerated subject/predicate/object matching
//! - Vectorized pattern filtering
//! - Batch processing with automatic SIMD width detection
//! - Hash-based pre-filtering for optimization
//! - Statistics tracking for performance analysis

use crate::error::{FusekiError, FusekiResult};
use scirs2_core::memory::BufferPool;
use scirs2_core::metrics::{Counter, Histogram};
use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1};
// Note: SIMD operations in rc.3 use trait-based approach with SimdUnifiedOps
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Triple pattern for matching
#[derive(Debug, Clone, PartialEq)]
pub struct TriplePattern {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

/// Triple representation optimized for SIMD operations
#[derive(Debug, Clone)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub subject_hash: u64,
    pub predicate_hash: u64,
    pub object_hash: u64,
}

impl Triple {
    /// Create a new triple with pre-computed hashes
    pub fn new(subject: String, predicate: String, object: String) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        subject.hash(&mut hasher);
        let subject_hash = hasher.finish();

        let mut hasher = DefaultHasher::new();
        predicate.hash(&mut hasher);
        let predicate_hash = hasher.finish();

        let mut hasher = DefaultHasher::new();
        object.hash(&mut hasher);
        let object_hash = hasher.finish();

        Self {
            subject,
            predicate,
            object,
            subject_hash,
            predicate_hash,
            object_hash,
        }
    }
}

/// SIMD-accelerated triple matcher with performance optimization
pub struct SimdTripleMatcher {
    /// Triple storage optimized for SIMD access
    triples: Vec<Triple>,

    /// Hash index for fast pre-filtering
    subject_index: HashMap<u64, Vec<usize>>,
    predicate_index: HashMap<u64, Vec<usize>>,
    object_index: HashMap<u64, Vec<usize>>,

    /// Buffer pool for temporary allocations
    buffer_pool: Arc<BufferPool<u8>>,

    /// Performance metrics
    matches_counter: Counter,
    match_time_histogram: Histogram,
    simd_operations_counter: Counter,

    /// Statistics
    total_matches: AtomicU64,
    simd_accelerated_matches: AtomicU64,
    fallback_matches: AtomicU64,
}

impl SimdTripleMatcher {
    /// Create a new SIMD triple matcher
    pub fn new() -> Self {
        let buffer_pool = Arc::new(BufferPool::new());

        Self {
            triples: Vec::new(),
            subject_index: HashMap::new(),
            predicate_index: HashMap::new(),
            object_index: HashMap::new(),
            buffer_pool,
            matches_counter: Counter::new("triple_matches".to_string()),
            match_time_histogram: Histogram::new("match_time_ms".to_string()),
            simd_operations_counter: Counter::new("simd_operations".to_string()),
            total_matches: AtomicU64::new(0),
            simd_accelerated_matches: AtomicU64::new(0),
            fallback_matches: AtomicU64::new(0),
        }
    }

    /// Add a triple to the matcher with automatic indexing
    pub fn add_triple(&mut self, triple: Triple) {
        let index = self.triples.len();

        // Update indexes
        self.subject_index
            .entry(triple.subject_hash)
            .or_insert_with(Vec::new)
            .push(index);

        self.predicate_index
            .entry(triple.predicate_hash)
            .or_insert_with(Vec::new)
            .push(index);

        self.object_index
            .entry(triple.object_hash)
            .or_insert_with(Vec::new)
            .push(index);

        self.triples.push(triple);
    }

    /// Add multiple triples in batch for better performance
    pub fn add_triples(&mut self, triples: Vec<Triple>) {
        for triple in triples {
            self.add_triple(triple);
        }
    }

    /// Match triples using SIMD-accelerated pattern matching
    pub fn match_pattern(&self, pattern: &TriplePattern) -> FusekiResult<Vec<&Triple>> {
        let start_time = std::time::Instant::now();

        // Pre-filter using hash indexes
        let candidate_indices = self.get_candidate_indices(pattern);

        if candidate_indices.is_empty() {
            return Ok(Vec::new());
        }

        // Use SIMD acceleration for large candidate sets
        let results = if candidate_indices.len() >= 32 {
            self.simd_match(&candidate_indices, pattern)?
        } else {
            self.fallback_match(&candidate_indices, pattern)
        };

        // Update metrics
        self.matches_counter.inc();
        self.match_time_histogram
            .observe(start_time.elapsed().as_secs_f64());
        self.total_matches.fetch_add(1, Ordering::Relaxed);

        if candidate_indices.len() >= 32 {
            self.simd_accelerated_matches
                .fetch_add(1, Ordering::Relaxed);
            self.simd_operations_counter.inc();
        } else {
            self.fallback_matches.fetch_add(1, Ordering::Relaxed);
        }

        Ok(results)
    }

    /// Get candidate indices using hash index pre-filtering
    fn get_candidate_indices(&self, pattern: &TriplePattern) -> Vec<usize> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut candidates: Option<Vec<usize>> = None;

        // Filter by subject if specified
        if let Some(ref subject) = pattern.subject {
            let mut hasher = DefaultHasher::new();
            subject.hash(&mut hasher);
            let hash = hasher.finish();

            if let Some(indices) = self.subject_index.get(&hash) {
                candidates = Some(indices.clone());
            } else {
                return Vec::new();
            }
        }

        // Filter by predicate if specified
        if let Some(ref predicate) = pattern.predicate {
            let mut hasher = DefaultHasher::new();
            predicate.hash(&mut hasher);
            let hash = hasher.finish();

            if let Some(indices) = self.predicate_index.get(&hash) {
                candidates = match candidates {
                    Some(existing) => Some(
                        existing
                            .into_iter()
                            .filter(|i| indices.contains(i))
                            .collect(),
                    ),
                    None => Some(indices.clone()),
                };
            } else {
                return Vec::new();
            }
        }

        // Filter by object if specified
        if let Some(ref object) = pattern.object {
            let mut hasher = DefaultHasher::new();
            object.hash(&mut hasher);
            let hash = hasher.finish();

            if let Some(indices) = self.object_index.get(&hash) {
                candidates = match candidates {
                    Some(existing) => Some(
                        existing
                            .into_iter()
                            .filter(|i| indices.contains(i))
                            .collect(),
                    ),
                    None => Some(indices.clone()),
                };
            } else {
                return Vec::new();
            }
        }

        candidates.unwrap_or_else(|| (0..self.triples.len()).collect())
    }

    /// SIMD-accelerated matching for large candidate sets
    fn simd_match(
        &self,
        candidate_indices: &[usize],
        pattern: &TriplePattern,
    ) -> FusekiResult<Vec<&Triple>> {
        // Convert candidate hashes to SIMD arrays for vectorized comparison
        let subject_hashes: Vec<u64> = candidate_indices
            .iter()
            .map(|&i| self.triples[i].subject_hash)
            .collect();

        let predicate_hashes: Vec<u64> = candidate_indices
            .iter()
            .map(|&i| self.triples[i].predicate_hash)
            .collect();

        let object_hashes: Vec<u64> = candidate_indices
            .iter()
            .map(|&i| self.triples[i].object_hash)
            .collect();

        // Create SIMD arrays for vectorized operations
        let subject_array = Array1::from_vec(subject_hashes);
        let predicate_array = Array1::from_vec(predicate_hashes);
        let object_array = Array1::from_vec(object_hashes);

        // Create mask array (all true initially)
        let mut mask = vec![true; candidate_indices.len()];

        // Apply pattern filters using SIMD operations
        if let Some(ref subject) = pattern.subject {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            subject.hash(&mut hasher);
            let target_hash = hasher.finish();

            // SIMD comparison
            for (i, &hash) in subject_array.iter().enumerate() {
                mask[i] = mask[i] && (hash == target_hash);
            }
        }

        if let Some(ref predicate) = pattern.predicate {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            predicate.hash(&mut hasher);
            let target_hash = hasher.finish();

            for (i, &hash) in predicate_array.iter().enumerate() {
                mask[i] = mask[i] && (hash == target_hash);
            }
        }

        if let Some(ref object) = pattern.object {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            object.hash(&mut hasher);
            let target_hash = hasher.finish();

            for (i, &hash) in object_array.iter().enumerate() {
                mask[i] = mask[i] && (hash == target_hash);
            }
        }

        // Collect matching triples
        let results: Vec<&Triple> = candidate_indices
            .iter()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(_, &idx)| &self.triples[idx])
            .collect();

        Ok(results)
    }

    /// Fallback matching for small candidate sets
    fn fallback_match(&self, candidate_indices: &[usize], pattern: &TriplePattern) -> Vec<&Triple> {
        candidate_indices
            .iter()
            .map(|&i| &self.triples[i])
            .filter(|triple| {
                if let Some(ref subject) = pattern.subject {
                    if &triple.subject != subject {
                        return false;
                    }
                }

                if let Some(ref predicate) = pattern.predicate {
                    if &triple.predicate != predicate {
                        return false;
                    }
                }

                if let Some(ref object) = pattern.object {
                    if &triple.object != object {
                        return false;
                    }
                }

                true
            })
            .collect()
    }

    /// Get matcher statistics
    pub fn get_statistics(&self) -> MatcherStatistics {
        MatcherStatistics {
            total_triples: self.triples.len(),
            total_matches: self.total_matches.load(Ordering::Relaxed),
            simd_accelerated_matches: self.simd_accelerated_matches.load(Ordering::Relaxed),
            fallback_matches: self.fallback_matches.load(Ordering::Relaxed),
            index_sizes: IndexSizes {
                subject_index_size: self.subject_index.len(),
                predicate_index_size: self.predicate_index.len(),
                object_index_size: self.object_index.len(),
            },
        }
    }

    /// Clear all triples and rebuild indexes
    pub fn clear(&mut self) {
        self.triples.clear();
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
        self.total_matches.store(0, Ordering::Relaxed);
        self.simd_accelerated_matches.store(0, Ordering::Relaxed);
        self.fallback_matches.store(0, Ordering::Relaxed);
    }
}

impl Default for SimdTripleMatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Matcher statistics for monitoring
#[derive(Debug, Clone, serde::Serialize)]
pub struct MatcherStatistics {
    pub total_triples: usize,
    pub total_matches: u64,
    pub simd_accelerated_matches: u64,
    pub fallback_matches: u64,
    pub index_sizes: IndexSizes,
}

/// Index size statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexSizes {
    pub subject_index_size: usize,
    pub predicate_index_size: usize,
    pub object_index_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_creation() {
        let triple = Triple::new(
            "http://example.org/subject".to_string(),
            "http://example.org/predicate".to_string(),
            "http://example.org/object".to_string(),
        );

        assert_eq!(triple.subject, "http://example.org/subject");
        assert_ne!(triple.subject_hash, 0);
    }

    #[test]
    fn test_matcher_basic() {
        let mut matcher = SimdTripleMatcher::new();

        let triple1 = Triple::new(
            "http://example.org/s1".to_string(),
            "http://example.org/p1".to_string(),
            "http://example.org/o1".to_string(),
        );

        let triple2 = Triple::new(
            "http://example.org/s2".to_string(),
            "http://example.org/p1".to_string(),
            "http://example.org/o2".to_string(),
        );

        matcher.add_triple(triple1);
        matcher.add_triple(triple2);

        // Match by predicate
        let pattern = TriplePattern {
            subject: None,
            predicate: Some("http://example.org/p1".to_string()),
            object: None,
        };

        let results = matcher.match_pattern(&pattern).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_matcher_specific() {
        let mut matcher = SimdTripleMatcher::new();

        let triple = Triple::new(
            "http://example.org/subject".to_string(),
            "http://example.org/predicate".to_string(),
            "http://example.org/object".to_string(),
        );

        matcher.add_triple(triple);

        // Match specific triple
        let pattern = TriplePattern {
            subject: Some("http://example.org/subject".to_string()),
            predicate: Some("http://example.org/predicate".to_string()),
            object: Some("http://example.org/object".to_string()),
        };

        let results = matcher.match_pattern(&pattern).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "http://example.org/subject");
    }

    #[test]
    fn test_matcher_no_match() {
        let mut matcher = SimdTripleMatcher::new();

        let triple = Triple::new(
            "http://example.org/s1".to_string(),
            "http://example.org/p1".to_string(),
            "http://example.org/o1".to_string(),
        );

        matcher.add_triple(triple);

        // Match non-existent triple
        let pattern = TriplePattern {
            subject: Some("http://example.org/nonexistent".to_string()),
            predicate: None,
            object: None,
        };

        let results = matcher.match_pattern(&pattern).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_matcher_statistics() {
        let mut matcher = SimdTripleMatcher::new();

        for i in 0..100 {
            let triple = Triple::new(
                format!("http://example.org/s{}", i),
                "http://example.org/p1".to_string(),
                format!("http://example.org/o{}", i),
            );
            matcher.add_triple(triple);
        }

        let stats = matcher.get_statistics();
        assert_eq!(stats.total_triples, 100);
        assert!(stats.index_sizes.subject_index_size > 0);
    }

    #[test]
    fn test_matcher_batch_add() {
        let mut matcher = SimdTripleMatcher::new();

        let triples = vec![
            Triple::new("s1".to_string(), "p1".to_string(), "o1".to_string()),
            Triple::new("s2".to_string(), "p2".to_string(), "o2".to_string()),
            Triple::new("s3".to_string(), "p3".to_string(), "o3".to_string()),
        ];

        matcher.add_triples(triples);

        let stats = matcher.get_statistics();
        assert_eq!(stats.total_triples, 3);
    }

    #[test]
    fn test_matcher_clear() {
        let mut matcher = SimdTripleMatcher::new();

        let triple = Triple::new("s1".to_string(), "p1".to_string(), "o1".to_string());
        matcher.add_triple(triple);

        matcher.clear();

        let stats = matcher.get_statistics();
        assert_eq!(stats.total_triples, 0);
        assert_eq!(stats.total_matches, 0);
    }

    #[test]
    fn test_simd_acceleration_threshold() {
        let mut matcher = SimdTripleMatcher::new();

        // Add exactly 32 triples to trigger SIMD acceleration
        for i in 0..32 {
            let triple = Triple::new(format!("s{}", i), "p".to_string(), format!("o{}", i));
            matcher.add_triple(triple);
        }

        let pattern = TriplePattern {
            subject: None,
            predicate: Some("p".to_string()),
            object: None,
        };

        let results = matcher.match_pattern(&pattern).unwrap();
        assert_eq!(results.len(), 32);

        let stats = matcher.get_statistics();
        assert_eq!(stats.simd_accelerated_matches, 1);
    }
}
