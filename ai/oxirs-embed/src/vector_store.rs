//! In-memory vector store with namespace support.
//!
//! Provides `VectorStore`: a namespace-partitioned collection of embedding
//! vectors with cosine-similarity search, upsert/delete, and dimension
//! enforcement.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single vector stored in the vector store.
#[derive(Debug, Clone)]
pub struct VectorEntry {
    /// Unique identifier within its namespace.
    pub id: String,
    /// Namespace this entry belongs to.
    pub namespace: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
    /// Creation timestamp (arbitrary unit — caller-supplied, e.g. Unix ms).
    pub created_at: u64,
}

/// A single result returned by a similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Entry identifier.
    pub id: String,
    /// Namespace the entry lives in.
    pub namespace: String,
    /// Cosine similarity score in `[-1, 1]` (1 = identical direction).
    pub score: f32,
    /// Entry metadata.
    pub metadata: HashMap<String, String>,
}

/// Aggregate statistics about the store.
#[derive(Debug, Clone)]
pub struct VectorStoreStats {
    /// Total number of stored vectors across all namespaces.
    pub total_vectors: usize,
    /// Number of distinct namespaces that contain at least one vector.
    pub namespace_count: usize,
    /// Dimensionality of stored vectors, or `None` when the store is empty or
    /// no dimension was enforced.
    pub dimension: Option<usize>,
}

/// Errors returned by store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// The supplied vector does not match the store's configured dimensionality.
    DimensionMismatch { expected: usize, got: usize },
    /// An empty (zero-length) vector was supplied.
    EmptyVector,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            StoreError::EmptyVector => write!(f, "vector must not be empty"),
        }
    }
}

impl std::error::Error for StoreError {}

// ─────────────────────────────────────────────────────────────────────────────
// VectorStore
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory vector store partitioned by namespace.
pub struct VectorStore {
    /// namespace → list of entries.
    namespaces: HashMap<String, Vec<VectorEntry>>,
    /// Optional fixed dimensionality; enforced on upsert.
    dimension: Option<usize>,
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Compute cosine similarity between two equal-length slices.
///
/// Returns `0.0` if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStore {
    /// Create an empty store without a fixed dimension.
    pub fn new() -> Self {
        Self {
            namespaces: HashMap::new(),
            dimension: None,
        }
    }

    /// Create an empty store that enforces a fixed dimension on upsert.
    pub fn with_dimension(dim: usize) -> Self {
        Self {
            namespaces: HashMap::new(),
            dimension: Some(dim),
        }
    }

    // ── validation helper ─────────────────────────────────────────────────────

    fn validate_vector(&self, vector: &[f32]) -> Result<(), StoreError> {
        if vector.is_empty() {
            return Err(StoreError::EmptyVector);
        }
        if let Some(dim) = self.dimension {
            if vector.len() != dim {
                return Err(StoreError::DimensionMismatch {
                    expected: dim,
                    got: vector.len(),
                });
            }
        }
        Ok(())
    }

    // ── CRUD ──────────────────────────────────────────────────────────────────

    /// Insert or update a vector entry.
    ///
    /// Returns `Ok(true)` if the entry was newly inserted, `Ok(false)` if an
    /// existing entry with the same `(namespace, id)` was replaced.
    ///
    /// When the store is dimension-aware (created with [`VectorStore::with_dimension`]),
    /// vectors of a different length are rejected.
    pub fn upsert(&mut self, entry: VectorEntry) -> Result<bool, StoreError> {
        self.validate_vector(&entry.vector)?;
        let entries = self.namespaces.entry(entry.namespace.clone()).or_default();
        if let Some(existing) = entries.iter_mut().find(|e| e.id == entry.id) {
            *existing = entry;
            return Ok(false);
        }
        entries.push(entry);
        Ok(true)
    }

    /// Delete the entry with the given `(namespace, id)`.
    ///
    /// Returns `true` if the entry existed and was removed.
    pub fn delete(&mut self, namespace: &str, id: &str) -> bool {
        match self.namespaces.get_mut(namespace) {
            Some(entries) => {
                let before = entries.len();
                entries.retain(|e| e.id != id);
                entries.len() < before
            }
            None => false,
        }
    }

    /// Look up a specific entry by `(namespace, id)`.
    pub fn get(&self, namespace: &str, id: &str) -> Option<&VectorEntry> {
        self.namespaces
            .get(namespace)
            .and_then(|entries| entries.iter().find(|e| e.id == id))
    }

    /// Check whether a specific `(namespace, id)` exists.
    pub fn contains(&self, namespace: &str, id: &str) -> bool {
        self.get(namespace, id).is_some()
    }

    // ── search ────────────────────────────────────────────────────────────────

    /// Return the `top_k` most-similar entries in `namespace` to `query`,
    /// sorted by descending cosine similarity.
    pub fn search(&self, namespace: &str, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let entries = match self.namespaces.get(namespace) {
            Some(e) => e,
            None => return Vec::new(),
        };
        let mut scored: Vec<(f32, &VectorEntry)> = entries
            .iter()
            .filter(|e| e.vector.len() == query.len())
            .map(|e| (cosine_similarity(&e.vector, query), e))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(top_k)
            .map(|(score, entry)| SearchResult {
                id: entry.id.clone(),
                namespace: entry.namespace.clone(),
                score,
                metadata: entry.metadata.clone(),
            })
            .collect()
    }

    /// Return the `top_k` most-similar entries across **all** namespaces.
    pub fn search_all_namespaces(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<(f32, &VectorEntry)> = self
            .namespaces
            .values()
            .flat_map(|entries| entries.iter())
            .filter(|e| e.vector.len() == query.len())
            .map(|e| (cosine_similarity(&e.vector, query), e))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(top_k)
            .map(|(score, entry)| SearchResult {
                id: entry.id.clone(),
                namespace: entry.namespace.clone(),
                score,
                metadata: entry.metadata.clone(),
            })
            .collect()
    }

    // ── listing ───────────────────────────────────────────────────────────────

    /// List all entries in `namespace`.  Returns an empty slice if the
    /// namespace does not exist.
    pub fn list(&self, namespace: &str) -> Vec<&VectorEntry> {
        match self.namespaces.get(namespace) {
            Some(entries) => entries.iter().collect(),
            None => Vec::new(),
        }
    }

    // ── namespace management ──────────────────────────────────────────────────

    /// Remove all entries belonging to `namespace`.
    ///
    /// Returns the number of entries that were deleted.
    pub fn delete_namespace(&mut self, namespace: &str) -> usize {
        match self.namespaces.remove(namespace) {
            Some(entries) => entries.len(),
            None => 0,
        }
    }

    // ── statistics ────────────────────────────────────────────────────────────

    /// Return aggregate statistics about the store.
    pub fn stats(&self) -> VectorStoreStats {
        let total_vectors: usize = self.namespaces.values().map(|v| v.len()).sum();
        let namespace_count = self.namespaces.values().filter(|v| !v.is_empty()).count();
        // Derive dimension from first stored vector if not explicitly configured.
        let dimension = self.dimension.or_else(|| {
            self.namespaces
                .values()
                .flat_map(|v| v.iter())
                .next()
                .map(|e| e.vector.len())
        });
        VectorStoreStats {
            total_vectors,
            namespace_count,
            dimension,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, ns: &str, v: Vec<f32>) -> VectorEntry {
        VectorEntry {
            id: id.to_string(),
            namespace: ns.to_string(),
            vector: v,
            metadata: HashMap::new(),
            created_at: 0,
        }
    }

    fn unit_vec(dim: usize, one_at: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[one_at] = 1.0;
        v
    }

    // ── upsert new / update ───────────────────────────────────────────────────

    #[test]
    fn test_upsert_new_returns_true() {
        let mut store = VectorStore::new();
        let is_new = store
            .upsert(entry("e1", "ns", vec![1.0, 0.0]))
            .expect("should succeed");
        assert!(is_new);
    }

    #[test]
    fn test_upsert_update_returns_false() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("e1", "ns", vec![1.0, 0.0]))
            .expect("should succeed");
        let is_new = store
            .upsert(entry("e1", "ns", vec![0.0, 1.0]))
            .expect("should succeed");
        assert!(!is_new);
    }

    #[test]
    fn test_upsert_update_replaces_vector() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("e1", "ns", vec![1.0, 0.0]))
            .expect("should succeed");
        store
            .upsert(entry("e1", "ns", vec![0.0, 1.0]))
            .expect("should succeed");
        let got = store.get("ns", "e1").expect("should succeed");
        assert_eq!(got.vector, vec![0.0, 1.0]);
    }

    #[test]
    fn test_upsert_empty_vector_errors() {
        let mut store = VectorStore::new();
        let res = store.upsert(entry("e1", "ns", vec![]));
        assert_eq!(res, Err(StoreError::EmptyVector));
    }

    #[test]
    fn test_upsert_dimension_mismatch_errors() {
        let mut store = VectorStore::with_dimension(3);
        let res = store.upsert(entry("e1", "ns", vec![1.0, 2.0]));
        assert_eq!(
            res,
            Err(StoreError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
    }

    #[test]
    fn test_upsert_correct_dimension_ok() {
        let mut store = VectorStore::with_dimension(2);
        assert!(store.upsert(entry("e1", "ns", vec![1.0, 0.0])).is_ok());
    }

    // ── delete ────────────────────────────────────────────────────────────────

    #[test]
    fn test_delete_existing() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("e1", "ns", vec![1.0]))
            .expect("should succeed");
        assert!(store.delete("ns", "e1"));
        assert!(store.get("ns", "e1").is_none());
    }

    #[test]
    fn test_delete_nonexistent_returns_false() {
        let mut store = VectorStore::new();
        assert!(!store.delete("ns", "ghost"));
    }

    // ── get ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_get_existing() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("e1", "ns", vec![1.0, 2.0]))
            .expect("should succeed");
        assert!(store.get("ns", "e1").is_some());
    }

    #[test]
    fn test_get_nonexistent_none() {
        let store = VectorStore::new();
        assert!(store.get("ns", "missing").is_none());
    }

    // ── contains ─────────────────────────────────────────────────────────────

    #[test]
    fn test_contains_true() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("x", "ns", vec![1.0]))
            .expect("should succeed");
        assert!(store.contains("ns", "x"));
    }

    #[test]
    fn test_contains_false() {
        let store = VectorStore::new();
        assert!(!store.contains("ns", "x"));
    }

    // ── search sorted by score ────────────────────────────────────────────────

    #[test]
    fn test_search_sorted_by_score_descending() {
        let mut store = VectorStore::new();
        // e1 aligned with query [1,0,0]
        store
            .upsert(entry("e1", "ns", unit_vec(3, 0)))
            .expect("should succeed");
        // e2 aligned with [0,1,0] — low similarity to query
        store
            .upsert(entry("e2", "ns", unit_vec(3, 1)))
            .expect("should succeed");
        let query = unit_vec(3, 0);
        let results = store.search("ns", &query, 2);
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
        assert_eq!(results[0].id, "e1");
    }

    #[test]
    fn test_search_top_k_limit() {
        let mut store = VectorStore::new();
        for i in 0..10 {
            store
                .upsert(entry(&i.to_string(), "ns", vec![i as f32]))
                .expect("should succeed");
        }
        let results = store.search("ns", &[5.0_f32], 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_same_vector_max_score() {
        let mut store = VectorStore::new();
        let v = vec![1.0_f32, 1.0, 1.0];
        store
            .upsert(entry("e", "ns", v.clone()))
            .expect("should succeed");
        let results = store.search("ns", &v, 1);
        assert_eq!(results.len(), 1);
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_empty_namespace_returns_empty() {
        let store = VectorStore::new();
        let results = store.search("missing-ns", &[1.0_f32], 5);
        assert!(results.is_empty());
    }

    // ── search_all_namespaces ─────────────────────────────────────────────────

    #[test]
    fn test_search_all_namespaces_cross_namespace() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns1", vec![1.0_f32, 0.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns2", vec![0.0_f32, 1.0]))
            .expect("should succeed");
        let results = store.search_all_namespaces(&[1.0_f32, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a"); // aligned with query
    }

    #[test]
    fn test_search_all_namespaces_top_k() {
        let mut store = VectorStore::new();
        for i in 0..5 {
            store
                .upsert(entry(&format!("e{i}"), &format!("ns{i}"), vec![i as f32]))
                .expect("should succeed");
        }
        let results = store.search_all_namespaces(&[2.0_f32], 2);
        assert_eq!(results.len(), 2);
    }

    // ── list ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_list_all_in_namespace() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns", vec![1.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns", vec![2.0]))
            .expect("should succeed");
        let listed = store.list("ns");
        assert_eq!(listed.len(), 2);
    }

    #[test]
    fn test_list_nonexistent_namespace_empty() {
        let store = VectorStore::new();
        assert!(store.list("ghost").is_empty());
    }

    // ── delete_namespace ──────────────────────────────────────────────────────

    #[test]
    fn test_delete_namespace_returns_count() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns", vec![1.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns", vec![2.0]))
            .expect("should succeed");
        assert_eq!(store.delete_namespace("ns"), 2);
    }

    #[test]
    fn test_delete_namespace_nonexistent_returns_zero() {
        let mut store = VectorStore::new();
        assert_eq!(store.delete_namespace("ghost"), 0);
    }

    #[test]
    fn test_delete_namespace_removes_entries() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns", vec![1.0]))
            .expect("should succeed");
        store.delete_namespace("ns");
        assert!(store.list("ns").is_empty());
    }

    // ── stats ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty_store() {
        let store = VectorStore::new();
        let s = store.stats();
        assert_eq!(s.total_vectors, 0);
        assert_eq!(s.namespace_count, 0);
        assert!(s.dimension.is_none());
    }

    #[test]
    fn test_stats_counts_correctly() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns1", vec![1.0, 2.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns1", vec![3.0, 4.0]))
            .expect("should succeed");
        store
            .upsert(entry("c", "ns2", vec![5.0, 6.0]))
            .expect("should succeed");
        let s = store.stats();
        assert_eq!(s.total_vectors, 3);
        assert_eq!(s.namespace_count, 2);
        assert_eq!(s.dimension, Some(2));
    }

    #[test]
    fn test_stats_with_configured_dimension() {
        let store = VectorStore::with_dimension(128);
        let s = store.stats();
        assert_eq!(s.dimension, Some(128));
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_upsert_multiple_namespaces() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns1", vec![1.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns2", vec![2.0]))
            .expect("should succeed");
        assert!(store.contains("ns1", "a"));
        assert!(store.contains("ns2", "b"));
    }

    #[test]
    fn test_delete_from_wrong_namespace() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns1", vec![1.0]))
            .expect("should succeed");
        assert!(!store.delete("ns2", "a")); // wrong namespace
        assert!(store.contains("ns1", "a")); // still exists
    }

    #[test]
    fn test_search_returns_correct_namespace() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns1", vec![1.0_f32, 0.0]))
            .expect("should succeed");
        let results = store.search("ns1", &[1.0_f32, 0.0], 1);
        assert_eq!(results[0].namespace, "ns1");
    }

    #[test]
    fn test_search_all_namespaces_empty_store() {
        let store = VectorStore::new();
        assert!(store.search_all_namespaces(&[1.0_f32], 5).is_empty());
    }

    #[test]
    fn test_metadata_stored_and_retrieved() {
        let mut store = VectorStore::new();
        let mut meta = HashMap::new();
        meta.insert("source".into(), "test".into());
        let mut e = entry("e1", "ns", vec![1.0]);
        e.metadata = meta;
        store.upsert(e).expect("should succeed");
        let got = store.get("ns", "e1").expect("should succeed");
        assert_eq!(got.metadata.get("source").map(|s| s.as_str()), Some("test"));
    }

    #[test]
    fn test_created_at_stored() {
        let mut store = VectorStore::new();
        let mut e = entry("e1", "ns", vec![1.0]);
        e.created_at = 12345678;
        store.upsert(e).expect("should succeed");
        assert_eq!(
            store.get("ns", "e1").expect("should succeed").created_at,
            12345678
        );
    }

    #[test]
    fn test_store_error_display_empty_vector() {
        let e = StoreError::EmptyVector;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_store_error_display_dimension_mismatch() {
        let e = StoreError::DimensionMismatch {
            expected: 3,
            got: 5,
        };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("5"));
    }

    #[test]
    fn test_stats_namespace_count_after_delete() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns1", vec![1.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns2", vec![1.0]))
            .expect("should succeed");
        store.delete_namespace("ns1");
        let s = store.stats();
        assert_eq!(s.namespace_count, 1);
    }

    #[test]
    fn test_search_scores_in_range() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns", vec![1.0_f32, 0.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns", vec![0.0_f32, 1.0]))
            .expect("should succeed");
        let results = store.search("ns", &[0.7_f32, 0.7], 2);
        for r in &results {
            assert!(r.score >= -1.0 && r.score <= 1.0);
        }
    }

    #[test]
    fn test_search_returns_metadata() {
        let mut store = VectorStore::new();
        let mut e = entry("e1", "ns", vec![1.0_f32, 0.0]);
        e.metadata.insert("key".into(), "val".into());
        store.upsert(e).expect("should succeed");
        let results = store.search("ns", &[1.0_f32, 0.0], 1);
        assert_eq!(
            results[0].metadata.get("key").map(|s| s.as_str()),
            Some("val")
        );
    }

    #[test]
    fn test_upsert_different_ids_same_namespace() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns", vec![1.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns", vec![2.0]))
            .expect("should succeed");
        store
            .upsert(entry("c", "ns", vec![3.0]))
            .expect("should succeed");
        assert_eq!(store.list("ns").len(), 3);
    }

    #[test]
    fn test_delete_reduces_list_count() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns", vec![1.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns", vec![2.0]))
            .expect("should succeed");
        store.delete("ns", "a");
        assert_eq!(store.list("ns").len(), 1);
    }

    #[test]
    fn test_search_all_namespaces_result_ids_correct() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("best", "ns1", vec![1.0_f32, 0.0]))
            .expect("should succeed");
        store
            .upsert(entry("other", "ns2", vec![0.0_f32, 1.0]))
            .expect("should succeed");
        let results = store.search_all_namespaces(&[1.0_f32, 0.0], 1);
        assert_eq!(results[0].id, "best");
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns", vec![1.0_f32, 0.0]))
            .expect("should succeed");
        store
            .upsert(entry("b", "ns", vec![-1.0_f32, 0.0]))
            .expect("should succeed");
        let results = store.search("ns", &[1.0_f32, 0.0], 2);
        // "a" should score higher than "b"
        assert_eq!(results[0].id, "a");
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_with_dimension_rejects_extra_dims() {
        let mut store = VectorStore::with_dimension(2);
        let res = store.upsert(entry("e", "ns", vec![1.0, 2.0, 3.0]));
        assert!(matches!(
            res,
            Err(StoreError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn test_upsert_returns_new_flag_consistently() {
        let mut store = VectorStore::new();
        let r1 = store
            .upsert(entry("e", "ns", vec![1.0]))
            .expect("should succeed");
        let r2 = store
            .upsert(entry("e", "ns", vec![2.0]))
            .expect("should succeed");
        assert!(r1); // new
        assert!(!r2); // update
    }

    #[test]
    fn test_stats_total_includes_all_namespaces() {
        let mut store = VectorStore::new();
        for i in 0..5 {
            store
                .upsert(entry(&i.to_string(), &format!("ns{i}"), vec![i as f32]))
                .expect("should succeed");
        }
        assert_eq!(store.stats().total_vectors, 5);
    }

    #[test]
    fn test_get_after_update_returns_new_vector() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("e", "ns", vec![1.0, 0.0]))
            .expect("should succeed");
        store
            .upsert(entry("e", "ns", vec![0.0, 1.0]))
            .expect("should succeed");
        let got = store.get("ns", "e").expect("should succeed");
        assert_eq!(got.vector, vec![0.0_f32, 1.0]);
    }

    #[test]
    fn test_search_zero_top_k() {
        let mut store = VectorStore::new();
        store
            .upsert(entry("a", "ns", vec![1.0_f32]))
            .expect("should succeed");
        let results = store.search("ns", &[1.0_f32], 0);
        assert!(results.is_empty());
    }
}
