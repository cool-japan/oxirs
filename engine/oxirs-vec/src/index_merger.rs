//! ANN vector index merging — combines multiple flat indices into one (v1.1.0 round 14).
//!
//! Provides utilities to:
//! - Build flat vector indices from individual entries
//! - Merge multiple flat indices with last-write-wins deduplication
//! - Filter entries during merge
//! - Split large indices into even partitions
//! - Collect merge statistics

use std::collections::HashMap;

/// A single vector entry stored in a flat index.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorEntry {
    /// Unique identifier for this vector.
    pub id: u64,
    /// The raw vector data.
    pub vector: Vec<f32>,
    /// Arbitrary string metadata attached to the entry.
    pub metadata: HashMap<String, String>,
}

impl VectorEntry {
    /// Create a new entry with the given `id` and `vector`.
    pub fn new(id: u64, vector: Vec<f32>) -> Self {
        Self {
            id,
            vector,
            metadata: HashMap::new(),
        }
    }

    /// Create a new entry with `id`, `vector`, and metadata.
    pub fn with_metadata(id: u64, vector: Vec<f32>, metadata: HashMap<String, String>) -> Self {
        Self {
            id,
            vector,
            metadata,
        }
    }
}

/// A flat in-memory ANN index holding a collection of [`VectorEntry`] values.
///
/// All entries in the index must have the same dimensionality.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatIndex {
    /// All stored entries.
    pub entries: Vec<VectorEntry>,
    /// Dimensionality of the vectors stored in this index.
    pub dims: usize,
}

impl FlatIndex {
    /// Create an empty flat index for vectors of the given dimensionality.
    pub fn new(dims: usize) -> Self {
        Self {
            entries: Vec::new(),
            dims,
        }
    }

    /// Insert an entry.  Returns an error if the entry's vector length does
    /// not match the index dimensionality.
    pub fn insert(&mut self, entry: VectorEntry) -> Result<(), MergeError> {
        if entry.vector.len() != self.dims {
            return Err(MergeError::DimensionMismatch {
                expected: self.dims,
                got: entry.vector.len(),
            });
        }
        self.entries.push(entry);
        Ok(())
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Statistics collected during a merge operation.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeStats {
    /// Number of input indices that were merged.
    pub input_count: usize,
    /// Total number of entries across all input indices before deduplication.
    pub total_before: usize,
    /// Number of entries removed by deduplication.
    pub deduplicated: usize,
    /// Number of entries in the merged output index.
    pub total_after: usize,
}

/// Errors that can occur during index merge / split operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MergeError {
    /// Two indices have different dimensionalities.
    DimensionMismatch { expected: usize, got: usize },
    /// No input indices were provided.
    EmptyInput,
    /// The requested number of parts is invalid (0).
    InvalidParts,
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MergeError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            }
            MergeError::EmptyInput => write!(f, "No input indices provided"),
            MergeError::InvalidParts => {
                write!(f, "Number of parts must be greater than zero")
            }
        }
    }
}

impl std::error::Error for MergeError {}

/// Combines multiple [`FlatIndex`] instances into a single merged index.
///
/// # Deduplication
/// When two entries share the same `id`, the **last one wins** (insertion
/// order across indices, then within each index).
#[derive(Debug, Default)]
pub struct IndexMerger {
    indices: Vec<FlatIndex>,
}

impl IndexMerger {
    /// Create a new, empty merger.
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
        }
    }

    /// Add an index to the merge set.
    pub fn add_index(&mut self, idx: FlatIndex) {
        self.indices.push(idx);
    }

    /// Merge all added indices into a single [`FlatIndex`].
    ///
    /// Deduplication is performed on `id`: if multiple entries share the same
    /// ID, the one that appears **latest** (last index, last position within
    /// that index) wins.
    ///
    /// Returns [`MergeError::EmptyInput`] if no indices have been added.
    pub fn merge(&mut self) -> Result<FlatIndex, MergeError> {
        if self.indices.is_empty() {
            return Err(MergeError::EmptyInput);
        }

        let dims = self.indices[0].dims;

        // Validate that all indices share the same dimensionality
        for idx in &self.indices {
            if idx.dims != dims {
                return Err(MergeError::DimensionMismatch {
                    expected: dims,
                    got: idx.dims,
                });
            }
        }

        // Last-write-wins deduplication using an ordered map
        // (we use a Vec to preserve insertion order for iteration, and a
        // HashMap for O(1) lookup / update)
        let mut order: Vec<u64> = Vec::new();
        let mut map: HashMap<u64, VectorEntry> = HashMap::new();

        for idx in &self.indices {
            for entry in &idx.entries {
                if !map.contains_key(&entry.id) {
                    order.push(entry.id);
                }
                map.insert(entry.id, entry.clone());
            }
        }

        let mut out = FlatIndex::new(dims);
        for id in &order {
            if let Some(entry) = map.remove(id) {
                out.entries.push(entry);
            }
        }

        Ok(out)
    }

    /// Merge all indices, retaining only entries for which `filter` returns
    /// `true`.  Deduplication happens **before** filtering.
    pub fn merge_with_filter<F>(&mut self, filter: F) -> Result<FlatIndex, MergeError>
    where
        F: Fn(&VectorEntry) -> bool,
    {
        let merged = self.merge()?;
        let dims = merged.dims;
        let mut out = FlatIndex::new(dims);
        for entry in merged.entries {
            if filter(&entry) {
                out.entries.push(entry);
            }
        }
        Ok(out)
    }

    /// Merge all indices and return both the merged index and statistics.
    pub fn merge_with_stats(&mut self) -> Result<(FlatIndex, MergeStats), MergeError> {
        if self.indices.is_empty() {
            return Err(MergeError::EmptyInput);
        }

        let input_count = self.indices.len();
        let total_before: usize = self.indices.iter().map(|i| i.len()).sum();

        let merged = self.merge()?;
        let total_after = merged.len();
        let deduplicated = total_before.saturating_sub(total_after);

        let stats = MergeStats {
            input_count,
            total_before,
            deduplicated,
            total_after,
        };
        Ok((merged, stats))
    }

    /// Split a [`FlatIndex`] into `parts` evenly-sized sub-indices.
    ///
    /// If the number of entries does not divide evenly, the first
    /// `entries.len() % parts` partitions will each receive one extra entry.
    ///
    /// Returns [`MergeError::InvalidParts`] if `parts == 0`.
    pub fn split(idx: &FlatIndex, parts: usize) -> Vec<FlatIndex> {
        if parts == 0 {
            return vec![];
        }
        if idx.is_empty() {
            return (0..parts).map(|_| FlatIndex::new(idx.dims)).collect();
        }

        let n = idx.entries.len();
        let base = n / parts;
        let remainder = n % parts;

        let mut result = Vec::with_capacity(parts);
        let mut offset = 0usize;

        for i in 0..parts {
            let chunk_size = base + if i < remainder { 1 } else { 0 };
            let mut sub = FlatIndex::new(idx.dims);
            sub.entries
                .extend_from_slice(&idx.entries[offset..offset + chunk_size]);
            offset += chunk_size;
            result.push(sub);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: u64, dims: usize, val: f32) -> VectorEntry {
        VectorEntry::new(id, vec![val; dims])
    }

    fn make_index(dims: usize, ids: &[(u64, f32)]) -> FlatIndex {
        let mut idx = FlatIndex::new(dims);
        for (id, val) in ids {
            idx.insert(make_entry(*id, dims, *val)).expect("insert ok");
        }
        idx
    }

    // -- FlatIndex -----------------------------------------------------------

    #[test]
    fn test_flat_index_new_is_empty() {
        let idx = FlatIndex::new(4);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.dims, 4);
    }

    #[test]
    fn test_flat_index_insert_valid() {
        let mut idx = FlatIndex::new(3);
        let entry = make_entry(1, 3, 0.5);
        assert!(idx.insert(entry).is_ok());
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_flat_index_insert_dimension_mismatch() {
        let mut idx = FlatIndex::new(3);
        let entry = make_entry(1, 4, 0.5);
        assert_eq!(
            idx.insert(entry),
            Err(MergeError::DimensionMismatch {
                expected: 3,
                got: 4
            })
        );
    }

    #[test]
    fn test_flat_index_is_not_empty_after_insert() {
        let mut idx = FlatIndex::new(2);
        idx.insert(make_entry(1, 2, 1.0)).expect("ok");
        assert!(!idx.is_empty());
    }

    // -- IndexMerger::merge -------------------------------------------------

    #[test]
    fn test_merge_empty_returns_error() {
        let mut merger = IndexMerger::new();
        assert_eq!(merger.merge(), Err(MergeError::EmptyInput));
    }

    #[test]
    fn test_merge_single_index() {
        let idx = make_index(2, &[(1, 1.0), (2, 2.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(idx);
        let out = merger.merge().expect("merge ok");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_merge_two_disjoint_indices() {
        let a = make_index(2, &[(1, 1.0), (2, 2.0)]);
        let b = make_index(2, &[(3, 3.0), (4, 4.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(a);
        merger.add_index(b);
        let out = merger.merge().expect("merge ok");
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_merge_deduplication_last_write_wins() {
        // Both indices contain ID 1; the one in `b` should survive.
        let a = make_index(2, &[(1, 1.0)]);
        let b = make_index(2, &[(1, 9.9)]);
        let mut merger = IndexMerger::new();
        merger.add_index(a);
        merger.add_index(b);
        let out = merger.merge().expect("merge ok");
        assert_eq!(out.len(), 1);
        assert!((out.entries[0].vector[0] - 9.9).abs() < 1e-6);
    }

    #[test]
    fn test_merge_deduplication_count() {
        let a = make_index(2, &[(1, 1.0), (2, 2.0)]);
        let b = make_index(2, &[(2, 2.5), (3, 3.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(a);
        merger.add_index(b);
        let out = merger.merge().expect("merge ok");
        // IDs: 1, 2 (from b), 3
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_merge_dimension_mismatch_error() {
        let a = make_index(2, &[(1, 1.0)]);
        let b = make_index(3, &[(2, 2.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(a);
        merger.add_index(b);
        assert!(merger.merge().is_err());
    }

    #[test]
    fn test_merge_preserves_metadata() {
        let mut meta = HashMap::new();
        meta.insert("key".to_string(), "val".to_string());
        let entry = VectorEntry::with_metadata(42, vec![1.0, 2.0], meta.clone());
        let mut idx = FlatIndex::new(2);
        idx.insert(entry).expect("ok");
        let mut merger = IndexMerger::new();
        merger.add_index(idx);
        let out = merger.merge().expect("ok");
        assert_eq!(out.entries[0].metadata.get("key"), Some(&"val".to_string()));
    }

    // -- merge_with_filter ---------------------------------------------------

    #[test]
    fn test_merge_with_filter_keeps_matching() {
        let idx = make_index(2, &[(1, 1.0), (2, 2.0), (3, 3.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(idx);
        let out = merger.merge_with_filter(|e| e.id % 2 == 1).expect("ok");
        assert_eq!(out.len(), 2);
        assert!(out.entries.iter().all(|e| e.id % 2 == 1));
    }

    #[test]
    fn test_merge_with_filter_all_excluded() {
        let idx = make_index(2, &[(1, 1.0), (2, 2.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(idx);
        let out = merger.merge_with_filter(|_| false).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn test_merge_with_filter_all_included() {
        let idx = make_index(2, &[(1, 1.0), (2, 2.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(idx);
        let out = merger.merge_with_filter(|_| true).expect("ok");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_merge_with_filter_empty_input() {
        let mut merger = IndexMerger::new();
        assert_eq!(
            merger.merge_with_filter(|_| true),
            Err(MergeError::EmptyInput)
        );
    }

    // -- merge_with_stats ----------------------------------------------------

    #[test]
    fn test_merge_stats_no_dedup() {
        let a = make_index(2, &[(1, 1.0), (2, 2.0)]);
        let b = make_index(2, &[(3, 3.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(a);
        merger.add_index(b);
        let (out, stats) = merger.merge_with_stats().expect("ok");
        assert_eq!(stats.input_count, 2);
        assert_eq!(stats.total_before, 3);
        assert_eq!(stats.deduplicated, 0);
        assert_eq!(stats.total_after, 3);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_merge_stats_with_dedup() {
        let a = make_index(2, &[(1, 1.0), (2, 2.0)]);
        let b = make_index(2, &[(2, 9.0), (3, 3.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(a);
        merger.add_index(b);
        let (_out, stats) = merger.merge_with_stats().expect("ok");
        assert_eq!(stats.total_before, 4);
        assert_eq!(stats.deduplicated, 1);
        assert_eq!(stats.total_after, 3);
    }

    #[test]
    fn test_merge_stats_empty_input() {
        let mut merger = IndexMerger::new();
        assert_eq!(merger.merge_with_stats(), Err(MergeError::EmptyInput));
    }

    // -- split ---------------------------------------------------------------

    #[test]
    fn test_split_even() {
        let idx = make_index(2, &[(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)]);
        let parts = IndexMerger::split(&idx, 2);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].len(), 2);
        assert_eq!(parts[1].len(), 2);
    }

    #[test]
    fn test_split_uneven() {
        let idx = make_index(2, &[(1, 1.0), (2, 2.0), (3, 3.0)]);
        let parts = IndexMerger::split(&idx, 2);
        assert_eq!(parts.len(), 2);
        // 3 entries / 2 parts → first gets 2, second gets 1
        assert_eq!(parts[0].len(), 2);
        assert_eq!(parts[1].len(), 1);
    }

    #[test]
    fn test_split_into_one() {
        let idx = make_index(2, &[(1, 1.0), (2, 2.0)]);
        let parts = IndexMerger::split(&idx, 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 2);
    }

    #[test]
    fn test_split_zero_parts() {
        let idx = make_index(2, &[(1, 1.0)]);
        let parts = IndexMerger::split(&idx, 0);
        assert!(parts.is_empty());
    }

    #[test]
    fn test_split_empty_index() {
        let idx = FlatIndex::new(3);
        let parts = IndexMerger::split(&idx, 3);
        assert_eq!(parts.len(), 3);
        assert!(parts.iter().all(|p| p.is_empty()));
    }

    #[test]
    fn test_split_more_parts_than_entries() {
        let idx = make_index(2, &[(1, 1.0), (2, 2.0)]);
        let parts = IndexMerger::split(&idx, 5);
        assert_eq!(parts.len(), 5);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_split_preserves_dims() {
        let idx = make_index(7, &[(1, 1.0), (2, 2.0), (3, 3.0)]);
        let parts = IndexMerger::split(&idx, 2);
        for p in &parts {
            assert_eq!(p.dims, 7);
        }
    }

    #[test]
    fn test_split_total_count_preserved() {
        let ids: Vec<(u64, f32)> = (1u64..=10).map(|i| (i, i as f32)).collect();
        let idx = make_index(4, &ids);
        let parts = IndexMerger::split(&idx, 3);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 10);
    }

    // -- Error display -------------------------------------------------------

    #[test]
    fn test_error_display_empty_input() {
        let e = MergeError::EmptyInput;
        assert!(e.to_string().contains("No input"));
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = MergeError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        let s = e.to_string();
        assert!(s.contains("4"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_error_display_invalid_parts() {
        let e = MergeError::InvalidParts;
        assert!(e.to_string().contains("zero"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(MergeError::EmptyInput);
        assert!(e.to_string().contains("No input"));
    }

    // -- VectorEntry ---------------------------------------------------------

    #[test]
    fn test_vector_entry_new() {
        let e = VectorEntry::new(7, vec![1.0, 2.0, 3.0]);
        assert_eq!(e.id, 7);
        assert_eq!(e.vector.len(), 3);
        assert!(e.metadata.is_empty());
    }

    #[test]
    fn test_vector_entry_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("source".into(), "test".into());
        let e = VectorEntry::with_metadata(1, vec![0.0], meta);
        assert_eq!(e.metadata.get("source"), Some(&"test".to_string()));
    }

    #[test]
    fn test_index_merger_default() {
        let _m: IndexMerger = IndexMerger::default();
    }

    #[test]
    fn test_merge_three_indices() {
        let a = make_index(2, &[(1, 1.0)]);
        let b = make_index(2, &[(2, 2.0)]);
        let c = make_index(2, &[(3, 3.0)]);
        let mut merger = IndexMerger::new();
        merger.add_index(a);
        merger.add_index(b);
        merger.add_index(c);
        let out = merger.merge().expect("ok");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_merge_large_index() {
        let pairs: Vec<(u64, f32)> = (1u64..=100).map(|i| (i, i as f32)).collect();
        let idx = make_index(4, &pairs);
        let mut merger = IndexMerger::new();
        merger.add_index(idx);
        let out = merger.merge().expect("ok");
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_split_four_parts() {
        let pairs: Vec<(u64, f32)> = (1u64..=8).map(|i| (i, i as f32)).collect();
        let idx = make_index(2, &pairs);
        let parts = IndexMerger::split(&idx, 4);
        assert_eq!(parts.len(), 4);
        assert!(parts.iter().all(|p| p.len() == 2));
    }

    #[test]
    fn test_merge_filter_by_vector_value() {
        let pairs: Vec<(u64, f32)> = (1u64..=10).map(|i| (i, i as f32)).collect();
        let idx = make_index(2, &pairs);
        let mut merger = IndexMerger::new();
        merger.add_index(idx);
        // Only keep entries where first vector element >= 5.0
        let out = merger
            .merge_with_filter(|e| e.vector[0] >= 5.0)
            .expect("ok");
        assert_eq!(out.len(), 6); // 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
    }

    #[test]
    fn test_flat_index_dims_preserved_through_merge() {
        let idx = make_index(128, &[(1, 0.1), (2, 0.2)]);
        let mut merger = IndexMerger::new();
        merger.add_index(idx);
        let out = merger.merge().expect("ok");
        assert_eq!(out.dims, 128);
    }

    #[test]
    fn test_stats_input_count_three() {
        let mut merger = IndexMerger::new();
        merger.add_index(make_index(2, &[(1, 1.0)]));
        merger.add_index(make_index(2, &[(2, 2.0)]));
        merger.add_index(make_index(2, &[(3, 3.0)]));
        let (_, stats) = merger.merge_with_stats().expect("ok");
        assert_eq!(stats.input_count, 3);
    }

    #[test]
    fn test_split_single_entry_many_parts() {
        let idx = make_index(2, &[(42, 1.0)]);
        let parts = IndexMerger::split(&idx, 4);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 1);
        assert_eq!(parts.len(), 4);
    }
}
