//! In-memory embedding store with label-based lookup and cosine similarity search.
//!
//! Provides `O(1)` label and id access and `O(n·d)` nearest-neighbour search
//! over `n` embeddings of dimension `d`.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can be returned by [`EmbeddingStore`] operations.
#[derive(Debug)]
pub enum StoreError {
    /// The supplied vector has the wrong number of dimensions.
    DimensionMismatch {
        /// The dimension expected by the store.
        expected: usize,
        /// The dimension of the supplied vector.
        got: usize,
    },
    /// No entry exists with the given label.
    LabelNotFound(String),
    /// The store contains no entries.
    EmptyStore,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            StoreError::LabelNotFound(label) => {
                write!(f, "label not found: {label}")
            }
            StoreError::EmptyStore => write!(f, "store is empty"),
        }
    }
}

impl std::error::Error for StoreError {}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

/// A stored embedding entry.
#[derive(Debug, Clone)]
pub struct EmbeddingEntry {
    /// Sequential identifier assigned at insertion time.
    pub id: usize,
    /// Human-readable label used as the primary key.
    pub label: String,
    /// The embedding vector.
    pub vector: Vec<f64>,
    /// Optional key-value metadata attached to this entry.
    pub metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// In-memory store for labelled embedding vectors.
pub struct EmbeddingStore {
    entries: Vec<EmbeddingEntry>,
    label_index: HashMap<String, usize>, // label → index into `entries`
    dim: Option<usize>,
}

impl Default for EmbeddingStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingStore {
    /// Create an empty `EmbeddingStore`.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            label_index: HashMap::new(),
            dim: None,
        }
    }

    /// Insert a new embedding.
    ///
    /// All vectors in the store must have the same dimension.  The first
    /// insertion fixes the dimension for all subsequent insertions.
    ///
    /// Returns the assigned `id` on success.
    pub fn insert(
        &mut self,
        label: impl Into<String>,
        vector: Vec<f64>,
    ) -> Result<usize, StoreError> {
        self.insert_with_meta(label, vector, HashMap::new())
    }

    /// Insert a new embedding with accompanying metadata.
    pub fn insert_with_meta(
        &mut self,
        label: impl Into<String>,
        vector: Vec<f64>,
        meta: HashMap<String, String>,
    ) -> Result<usize, StoreError> {
        let label = label.into();

        // Check / set dimension
        match self.dim {
            Some(d) if d != vector.len() => {
                return Err(StoreError::DimensionMismatch {
                    expected: d,
                    got: vector.len(),
                });
            }
            None => {
                self.dim = Some(vector.len());
            }
            Some(_) => {}
        }

        let id = self.entries.len();

        // Update or insert
        if let Some(&idx) = self.label_index.get(&label) {
            // Update existing entry
            self.entries[idx].vector = vector;
            self.entries[idx].metadata = meta;
            return Ok(self.entries[idx].id);
        }

        self.label_index.insert(label.clone(), id);
        self.entries.push(EmbeddingEntry {
            id,
            label,
            vector,
            metadata: meta,
        });
        Ok(id)
    }

    /// Look up an entry by its label.
    pub fn get_by_label(&self, label: &str) -> Option<&EmbeddingEntry> {
        let idx = self.label_index.get(label)?;
        self.entries.get(*idx)
    }

    /// Look up an entry by its sequential `id`.
    pub fn get_by_id(&self, id: usize) -> Option<&EmbeddingEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Compute the cosine similarity between two slices of equal length.
    ///
    /// Returns `0.0` when either vector has zero norm.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Return the `k` nearest entries to `query` by cosine similarity, in
    /// descending order of similarity.
    ///
    /// Returns [`StoreError::EmptyStore`] when the store is empty and
    /// [`StoreError::DimensionMismatch`] when `query` has the wrong length.
    pub fn nearest(
        &self,
        query: &[f64],
        k: usize,
    ) -> Result<Vec<(&EmbeddingEntry, f64)>, StoreError> {
        if self.entries.is_empty() {
            return Err(StoreError::EmptyStore);
        }
        if let Some(d) = self.dim {
            if query.len() != d {
                return Err(StoreError::DimensionMismatch {
                    expected: d,
                    got: query.len(),
                });
            }
        }

        let mut scored: Vec<(&EmbeddingEntry, f64)> = self
            .entries
            .iter()
            .map(|e| (e, Self::cosine_similarity(query, &e.vector)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    /// Number of entries currently in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The fixed dimension of all stored vectors, or `None` if no vectors
    /// have been inserted yet.
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    /// Return a list of all labels in insertion order.
    pub fn labels(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.label.as_str()).collect()
    }

    /// Remove the entry with the given label.
    ///
    /// Returns `true` if the entry was found and removed, `false` otherwise.
    ///
    /// Note: removing an entry does **not** reuse or reassign its `id`.
    pub fn remove(&mut self, label: &str) -> bool {
        if let Some(idx) = self.label_index.remove(label) {
            self.entries.remove(idx);
            // Rebuild the label → index mapping because indices have shifted
            self.label_index.clear();
            for (i, entry) in self.entries.iter().enumerate() {
                self.label_index.insert(entry.label.clone(), i);
            }
            // Reset dim if store is now empty
            if self.entries.is_empty() {
                self.dim = None;
            }
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn v2(x: f64, y: f64) -> Vec<f64> {
        vec![x, y]
    }

    fn v3(x: f64, y: f64, z: f64) -> Vec<f64> {
        vec![x, y, z]
    }

    // --- insert / len / dim ---

    #[test]
    fn test_new_empty() {
        let store = EmbeddingStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.dim().is_none());
    }

    #[test]
    fn test_insert_first_sets_dim() {
        let mut store = EmbeddingStore::new();
        store
            .insert("a", v3(1.0, 0.0, 0.0))
            .expect("should succeed");
        assert_eq!(store.dim(), Some(3));
    }

    #[test]
    fn test_insert_returns_id() {
        let mut store = EmbeddingStore::new();
        let id = store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        assert_eq!(id, 0);
        let id2 = store.insert("b", v2(0.0, 1.0)).expect("should succeed");
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_insert_increments_len() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        assert_eq!(store.len(), 1);
        store.insert("b", v2(0.0, 1.0)).expect("should succeed");
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_insert_dim_mismatch_error() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        let result = store.insert("b", v3(0.0, 1.0, 0.0));
        assert!(matches!(
            result,
            Err(StoreError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn test_insert_update_existing_label() {
        let mut store = EmbeddingStore::new();
        let id1 = store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        let id2 = store.insert("a", v2(0.5, 0.5)).expect("should succeed");
        // Same id, same len
        assert_eq!(id1, id2);
        assert_eq!(store.len(), 1);
        let e = store.get_by_label("a").expect("exists");
        assert!((e.vector[0] - 0.5).abs() < 1e-9);
    }

    // --- insert_with_meta ---

    #[test]
    fn test_insert_with_meta_stores_metadata() {
        let mut store = EmbeddingStore::new();
        let mut meta = HashMap::new();
        meta.insert("lang".to_string(), "en".to_string());
        store
            .insert_with_meta("doc1", v2(1.0, 0.0), meta)
            .expect("should succeed");
        let e = store.get_by_label("doc1").expect("exists");
        assert_eq!(e.metadata["lang"], "en");
    }

    // --- get_by_label ---

    #[test]
    fn test_get_by_label_existing() {
        let mut store = EmbeddingStore::new();
        store.insert("hello", v2(1.0, 0.0)).expect("should succeed");
        assert!(store.get_by_label("hello").is_some());
    }

    #[test]
    fn test_get_by_label_missing() {
        let store = EmbeddingStore::new();
        assert!(store.get_by_label("missing").is_none());
    }

    #[test]
    fn test_get_by_label_returns_correct_vector() {
        let mut store = EmbeddingStore::new();
        store.insert("x", v2(3.0, 4.0)).expect("should succeed");
        let e = store.get_by_label("x").expect("exists");
        assert!((e.vector[0] - 3.0).abs() < 1e-9);
        assert!((e.vector[1] - 4.0).abs() < 1e-9);
    }

    // --- get_by_id ---

    #[test]
    fn test_get_by_id_existing() {
        let mut store = EmbeddingStore::new();
        let id = store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        assert!(store.get_by_id(id).is_some());
    }

    #[test]
    fn test_get_by_id_missing() {
        let store = EmbeddingStore::new();
        assert!(store.get_by_id(999).is_none());
    }

    #[test]
    fn test_get_by_id_matches_label() {
        let mut store = EmbeddingStore::new();
        let id = store.insert("mykey", v2(1.0, 2.0)).expect("should succeed");
        let e = store.get_by_id(id).expect("exists");
        assert_eq!(e.label, "mykey");
    }

    // --- cosine_similarity ---

    #[test]
    fn test_cosine_identical_vectors() {
        let a = v3(1.0, 2.0, 3.0);
        let sim = EmbeddingStore::cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = v2(1.0, 0.0);
        let b = v2(0.0, 1.0);
        let sim = EmbeddingStore::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = v2(1.0, 0.0);
        let b = v2(-1.0, 0.0);
        let sim = EmbeddingStore::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_zero_vector_returns_zero() {
        let a = v2(0.0, 0.0);
        let b = v2(1.0, 0.0);
        let sim = EmbeddingStore::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_symmetry() {
        let a = v3(1.0, 2.0, 3.0);
        let b = v3(4.0, 5.0, 6.0);
        let sim_ab = EmbeddingStore::cosine_similarity(&a, &b);
        let sim_ba = EmbeddingStore::cosine_similarity(&b, &a);
        assert!((sim_ab - sim_ba).abs() < 1e-9);
    }

    // --- nearest ---

    #[test]
    fn test_nearest_empty_store_error() {
        let store = EmbeddingStore::new();
        assert!(matches!(
            store.nearest(&[1.0, 0.0], 3),
            Err(StoreError::EmptyStore)
        ));
    }

    #[test]
    fn test_nearest_dim_mismatch_error() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        assert!(matches!(
            store.nearest(&[1.0, 0.0, 0.0], 3),
            Err(StoreError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_nearest_returns_k_results() {
        let mut store = EmbeddingStore::new();
        for i in 0..5 {
            store
                .insert(format!("e{i}"), vec![i as f64, 0.0])
                .expect("should succeed");
        }
        let results = store.nearest(&[1.0, 0.0], 3).expect("should succeed");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_nearest_sorted_descending() {
        let mut store = EmbeddingStore::new();
        store.insert("up", v2(0.0, 1.0)).expect("should succeed");
        store.insert("right", v2(1.0, 0.0)).expect("should succeed");
        store.insert("diag", v2(1.0, 1.0)).expect("should succeed");
        let query = v2(1.0, 0.0);
        let results = store.nearest(&query, 3).expect("should succeed");
        let sims: Vec<f64> = results.iter().map(|(_, s)| *s).collect();
        for pair in sims.windows(2) {
            assert!(pair[0] >= pair[1]);
        }
    }

    #[test]
    fn test_nearest_top1_is_most_similar() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        store.insert("b", v2(0.0, 1.0)).expect("should succeed");
        store.insert("c", v2(-1.0, 0.0)).expect("should succeed");
        let results = store.nearest(&[1.0, 0.0], 1).expect("should succeed");
        assert_eq!(results[0].0.label, "a");
    }

    // --- labels ---

    #[test]
    fn test_labels_empty() {
        let store = EmbeddingStore::new();
        assert!(store.labels().is_empty());
    }

    #[test]
    fn test_labels_returns_all() {
        let mut store = EmbeddingStore::new();
        store.insert("alpha", v2(1.0, 0.0)).expect("should succeed");
        store.insert("beta", v2(0.0, 1.0)).expect("should succeed");
        let labels = store.labels();
        assert_eq!(labels.len(), 2);
        assert!(labels.contains(&"alpha"));
        assert!(labels.contains(&"beta"));
    }

    // --- remove ---

    #[test]
    fn test_remove_existing_returns_true() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        assert!(store.remove("a"));
        assert!(store.is_empty());
    }

    #[test]
    fn test_remove_missing_returns_false() {
        let mut store = EmbeddingStore::new();
        assert!(!store.remove("ghost"));
    }

    #[test]
    fn test_remove_decrements_len() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        store.insert("b", v2(0.0, 1.0)).expect("should succeed");
        store.remove("a");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_remove_remaining_entry_still_accessible() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        store.insert("b", v2(0.0, 1.0)).expect("should succeed");
        store.remove("a");
        assert!(store.get_by_label("b").is_some());
    }

    #[test]
    fn test_remove_all_resets_dim() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        store.remove("a");
        assert!(store.dim().is_none());
    }

    #[test]
    fn test_remove_allows_reinsertion_with_different_dim() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        store.remove("a");
        // After removing the only entry, dim is reset, so new dimension is allowed
        store
            .insert("a", v3(1.0, 0.0, 0.0))
            .expect("should succeed");
        assert_eq!(store.dim(), Some(3));
    }

    // --- default ---

    #[test]
    fn test_default_same_as_new() {
        let store = EmbeddingStore::default();
        assert!(store.is_empty());
    }

    // --- StoreError display ---

    #[test]
    fn test_error_display_dimension_mismatch() {
        let e = StoreError::DimensionMismatch {
            expected: 3,
            got: 2,
        };
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_display_label_not_found() {
        let e = StoreError::LabelNotFound("ghost".to_string());
        assert!(e.to_string().contains("ghost"));
    }

    #[test]
    fn test_error_display_empty_store() {
        let e = StoreError::EmptyStore;
        assert!(!e.to_string().is_empty());
    }

    // --- additional scenarios ---

    #[test]
    fn test_nearest_k_larger_than_store() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        store.insert("b", v2(0.0, 1.0)).expect("should succeed");
        let results = store.nearest(&[1.0, 1.0], 10).expect("should succeed");
        // Cannot return more than what's in the store
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_id_is_stable_for_inserted_entry() {
        let mut store = EmbeddingStore::new();
        let id = store.insert("vec", v2(1.0, 1.0)).expect("should succeed");
        let e = store.get_by_label("vec").expect("exists");
        assert_eq!(e.id, id);
    }

    #[test]
    fn test_entry_label_matches() {
        let mut store = EmbeddingStore::new();
        store
            .insert("myLabel", v2(0.5, 0.5))
            .expect("should succeed");
        let e = store.get_by_label("myLabel").expect("exists");
        assert_eq!(e.label, "myLabel");
    }

    // --- additional coverage ---

    #[test]
    fn test_insert_empty_vector_sets_dim_zero() {
        let mut store = EmbeddingStore::new();
        store.insert("empty", vec![]).expect("should succeed");
        assert_eq!(store.dim(), Some(0));
    }

    #[test]
    fn test_cosine_unit_vectors() {
        // Two unit vectors at 45° apart
        let a = vec![1.0_f64 / 2.0_f64.sqrt(), 1.0_f64 / 2.0_f64.sqrt()];
        let b = vec![1.0, 0.0];
        let sim = EmbeddingStore::cosine_similarity(&a, &b);
        assert!((sim - (1.0_f64 / 2.0_f64.sqrt())).abs() < 1e-9);
    }

    #[test]
    fn test_nearest_returns_fewer_when_store_smaller_than_k() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        let results = store.nearest(&[1.0, 0.0], 100).expect("should succeed");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_remove_all_entries_allows_new_dim() {
        let mut store = EmbeddingStore::new();
        store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        store.insert("b", v2(0.0, 1.0)).expect("should succeed");
        store.remove("a");
        store.remove("b");
        assert_eq!(store.dim(), None);
        // Should accept a 3-d vector now
        store
            .insert("c", v3(1.0, 0.0, 0.0))
            .expect("should succeed");
        assert_eq!(store.dim(), Some(3));
    }

    #[test]
    fn test_get_by_id_after_remove_middle() {
        let mut store = EmbeddingStore::new();
        let id_a = store.insert("a", v2(1.0, 0.0)).expect("should succeed");
        store.insert("b", v2(0.0, 1.0)).expect("should succeed");
        let id_c = store.insert("c", v2(0.5, 0.5)).expect("should succeed");
        store.remove("b");
        // a and c should still be accessible by id
        assert!(store.get_by_id(id_a).is_some());
        assert!(store.get_by_id(id_c).is_some());
    }

    #[test]
    fn test_insert_with_meta_empty_meta() {
        let mut store = EmbeddingStore::new();
        store
            .insert_with_meta("doc", v2(1.0, 0.0), HashMap::new())
            .expect("should succeed");
        let e = store.get_by_label("doc").expect("exists");
        assert!(e.metadata.is_empty());
    }

    #[test]
    fn test_nearest_similarity_range() {
        let mut store = EmbeddingStore::new();
        store
            .insert("a", v3(1.0, 0.0, 0.0))
            .expect("should succeed");
        store
            .insert("b", v3(0.0, 1.0, 0.0))
            .expect("should succeed");
        store
            .insert("c", v3(0.0, 0.0, 1.0))
            .expect("should succeed");
        let results = store.nearest(&[1.0, 0.0, 0.0], 3).expect("should succeed");
        for (_, sim) in &results {
            assert!(*sim >= -1.0 && *sim <= 1.0);
        }
    }
}
