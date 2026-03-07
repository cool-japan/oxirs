//! Materialized Query Result Views
//!
//! This module manages cached subquery result sets that can be reused when the same
//! (or structurally equivalent) subquery is repeated in different queries.
//! Views are automatically invalidated when their underlying RDF patterns change.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// An RDF term appearing in a query result binding
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfTerm {
    /// Named node (IRI)
    Iri(String),
    /// Blank node
    BlankNode(String),
    /// Literal with optional datatype and language tag
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
}

impl RdfTerm {
    /// Convenience constructor for a plain string literal
    pub fn plain_literal(value: impl Into<String>) -> Self {
        Self::Literal {
            value: value.into(),
            datatype: None,
            lang: None,
        }
    }

    /// Convenience constructor for an IRI term
    pub fn iri(value: impl Into<String>) -> Self {
        Self::Iri(value.into())
    }

    /// Convenience constructor for a blank node
    pub fn blank_node(value: impl Into<String>) -> Self {
        Self::BlankNode(value.into())
    }

    /// Returns true if this term is an IRI
    pub fn is_iri(&self) -> bool {
        matches!(self, RdfTerm::Iri(_))
    }

    /// Returns true if this term is a literal
    pub fn is_literal(&self) -> bool {
        matches!(self, RdfTerm::Literal { .. })
    }
}

impl std::fmt::Display for RdfTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfTerm::Iri(iri) => write!(f, "<{iri}>"),
            RdfTerm::BlankNode(id) => write!(f, "_:{id}"),
            RdfTerm::Literal {
                value,
                datatype,
                lang,
            } => {
                write!(f, "\"{value}\"")?;
                if let Some(dt) = datatype {
                    write!(f, "^^<{dt}>")?;
                } else if let Some(lang_tag) = lang {
                    write!(f, "@{lang_tag}")?;
                }
                Ok(())
            }
        }
    }
}

/// A single result row from a SPARQL query - maps variable names to RDF terms
pub type BindingRow = HashMap<String, RdfTerm>;

/// Threshold in rows: views with fewer rows are kept in memory
const MEMORY_ROW_THRESHOLD: usize = 10_000;

/// Where the view data is stored
pub enum ViewData {
    /// Result set small enough to hold in memory
    InMemory(Vec<BindingRow>),
    /// Large result set persisted to a temporary file
    OnDisk { path: PathBuf, row_count: usize },
}

impl std::fmt::Debug for ViewData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViewData::InMemory(rows) => write!(f, "InMemory({} rows)", rows.len()),
            ViewData::OnDisk { path, row_count } => {
                write!(f, "OnDisk({row_count} rows @ {})", path.display())
            }
        }
    }
}

/// A single materialized view entry
#[derive(Debug)]
pub struct MaterializedView {
    /// Hash of the query pattern that produced this view
    pub query_hash: String,
    /// Textual representation of the source query pattern
    pub query_pattern: String,
    /// Number of result rows
    pub result_size: usize,
    /// When the view was first created
    pub created_at: Instant,
    /// When the view was last accessed (for LRU eviction)
    pub last_accessed: Instant,
    /// Time-to-live for this view
    pub ttl: Duration,
    /// How many times this view has been accessed (hit count)
    pub access_count: u64,
    /// The actual data
    pub data: ViewData,
    /// Set of predicate IRIs that this view depends on (for targeted invalidation)
    pub dependent_predicates: Vec<String>,
}

impl MaterializedView {
    /// Returns true if the view's TTL has elapsed
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() >= self.ttl
    }

    /// Returns the in-memory rows if available
    pub fn in_memory_rows(&self) -> Option<&[BindingRow]> {
        match &self.data {
            ViewData::InMemory(rows) => Some(rows),
            ViewData::OnDisk { .. } => None,
        }
    }
}

/// Global hit/miss counters shared across the manager
#[derive(Debug, Default)]
struct HitCounters {
    hits: u64,
    misses: u64,
}

/// Summary statistics for the view manager
#[derive(Debug, Clone)]
pub struct ViewManagerStats {
    pub total_views: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub hit_ratio: f64,
    pub total_rows_cached: usize,
    pub on_disk_views: usize,
    pub in_memory_views: usize,
}

/// Configuration for the materialized view manager
#[derive(Debug, Clone)]
pub struct ViewManagerConfig {
    /// Maximum number of views to keep in the cache
    pub max_views: usize,
    /// Default TTL for new views
    pub default_ttl: Duration,
    /// Directory for on-disk spill files (uses std::env::temp_dir() by default)
    pub spill_dir: PathBuf,
    /// Row threshold below which views are kept in memory
    pub memory_row_threshold: usize,
}

impl Default for ViewManagerConfig {
    fn default() -> Self {
        Self {
            max_views: 256,
            default_ttl: Duration::from_secs(300),
            spill_dir: std::env::temp_dir().join("oxirs_view_cache"),
            memory_row_threshold: MEMORY_ROW_THRESHOLD,
        }
    }
}

/// Manager for materialized query result views
///
/// Thread-safe: all public methods acquire only short-lived locks.
pub struct MaterializedViewManager {
    views: Arc<RwLock<HashMap<String, MaterializedView>>>,
    counters: Arc<RwLock<HitCounters>>,
    config: ViewManagerConfig,
}

impl MaterializedViewManager {
    /// Create a new view manager with explicit configuration
    pub fn with_config(config: ViewManagerConfig) -> Self {
        // Pre-create spill directory (ignore errors if it already exists)
        let _ = std::fs::create_dir_all(&config.spill_dir);
        Self {
            views: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HitCounters::default())),
            config,
        }
    }

    /// Create a new view manager with default settings
    pub fn new(max_views: usize, default_ttl: Duration) -> Self {
        let config = ViewManagerConfig {
            max_views,
            default_ttl,
            ..ViewManagerConfig::default()
        };
        Self::with_config(config)
    }

    /// Store a new view.
    ///
    /// If the cache is full, the least-recently-used view is evicted first.
    /// Large result sets (exceeding the memory row threshold) are spilled to disk.
    pub fn store_view(
        &self,
        query_hash: &str,
        pattern: &str,
        results: Vec<BindingRow>,
        dependent_predicates: Vec<String>,
    ) -> Result<()> {
        let result_size = results.len();
        let ttl = self.config.default_ttl;

        let data = if result_size > self.config.memory_row_threshold {
            self.spill_to_disk(query_hash, &results)?
        } else {
            ViewData::InMemory(results)
        };

        let view = MaterializedView {
            query_hash: query_hash.to_string(),
            query_pattern: pattern.to_string(),
            result_size,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            ttl,
            access_count: 0,
            data,
            dependent_predicates,
        };

        let mut views = self
            .views
            .write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {e}"))?;

        // Evict if at capacity
        if views.len() >= self.config.max_views && !views.contains_key(query_hash) {
            self.evict_lru_locked(&mut views);
        }

        views.insert(query_hash.to_string(), view);
        Ok(())
    }

    /// Retrieve a view by query hash.  Returns None on cache miss or if expired.
    pub fn get_view(&self, query_hash: &str) -> Option<Vec<BindingRow>> {
        let mut views = match self.views.write().ok() {
            Some(v) => v,
            None => {
                self.record_miss();
                return None;
            }
        };

        // Key not found -> cache miss
        if !views.contains_key(query_hash) {
            self.record_miss();
            return None;
        }

        // Check expiry before borrowing mutably
        let is_expired = views.get(query_hash).is_some_and(|v| v.is_expired());
        if is_expired {
            views.remove(query_hash);
            self.record_miss();
            return None;
        }

        let view = views.get_mut(query_hash)?;

        // Update access metadata
        view.last_accessed = Instant::now();
        view.access_count += 1;

        let result = match &view.data {
            ViewData::InMemory(rows) => Some(rows.clone()),
            ViewData::OnDisk { path, .. } => self.load_from_disk(path).ok(),
        };

        if result.is_some() {
            self.record_hit();
        } else {
            self.record_miss();
        }
        result
    }

    /// Invalidate all views that depend on a given predicate IRI.
    ///
    /// Returns the number of views removed.
    pub fn invalidate_by_predicate(&self, predicate_iri: &str) -> usize {
        let Ok(mut views) = self.views.write() else {
            return 0;
        };
        let before = views.len();
        views.retain(|_, view| {
            !view
                .dependent_predicates
                .contains(&predicate_iri.to_string())
        });
        before - views.len()
    }

    /// Invalidate views whose query pattern contains the given substring.
    /// Useful for invalidating based on graph pattern structure changes.
    pub fn invalidate_pattern(&self, affected_pattern: &str) -> usize {
        let Ok(mut views) = self.views.write() else {
            return 0;
        };
        let before = views.len();
        views.retain(|_, view| !view.query_pattern.contains(affected_pattern));
        before - views.len()
    }

    /// Remove all views whose TTL has elapsed.
    ///
    /// Returns the number of expired views removed.
    pub fn evict_expired(&self) -> usize {
        let Ok(mut views) = self.views.write() else {
            return 0;
        };
        let before = views.len();
        // Collect paths to delete on disk
        let to_delete: Vec<PathBuf> = views
            .values()
            .filter(|v| v.is_expired())
            .filter_map(|v| {
                if let ViewData::OnDisk { path, .. } = &v.data {
                    Some(path.clone())
                } else {
                    None
                }
            })
            .collect();
        views.retain(|_, v| !v.is_expired());
        let removed = before - views.len();

        // Best-effort deletion of spill files
        for path in to_delete {
            let _ = std::fs::remove_file(&path);
        }
        removed
    }

    /// Explicitly remove a single view by hash
    pub fn invalidate_view(&self, query_hash: &str) -> bool {
        let Ok(mut views) = self.views.write() else {
            return false;
        };
        if let Some(view) = views.remove(query_hash) {
            if let ViewData::OnDisk { path, .. } = view.data {
                let _ = std::fs::remove_file(&path);
            }
            true
        } else {
            false
        }
    }

    /// Retrieve current statistics snapshot
    pub fn get_stats(&self) -> ViewManagerStats {
        let views = self.views.read().unwrap_or_else(|e| e.into_inner());
        let counters = self.counters.read().unwrap_or_else(|e| e.into_inner());

        let total_rows_cached: usize = views.values().map(|v| v.result_size).sum();
        let on_disk_views = views
            .values()
            .filter(|v| matches!(&v.data, ViewData::OnDisk { .. }))
            .count();
        let in_memory_views = views.len() - on_disk_views;

        let total = counters.hits + counters.misses;
        let hit_ratio = if total > 0 {
            counters.hits as f64 / total as f64
        } else {
            0.0
        };

        ViewManagerStats {
            total_views: views.len(),
            hit_count: counters.hits,
            miss_count: counters.misses,
            hit_ratio,
            total_rows_cached,
            on_disk_views,
            in_memory_views,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn evict_lru_locked(&self, views: &mut HashMap<String, MaterializedView>) {
        // Find the key with the oldest last_accessed time
        let oldest_key = views
            .iter()
            .min_by_key(|(_, v)| v.last_accessed)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest_key {
            if let Some(view) = views.remove(&key) {
                if let ViewData::OnDisk { path, .. } = view.data {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }

    fn spill_to_disk(&self, query_hash: &str, results: &[BindingRow]) -> Result<ViewData> {
        let safe_hash = query_hash
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .take(32)
            .collect::<String>();
        let file_name = format!("view_{safe_hash}.json");
        let path = self.config.spill_dir.join(file_name);

        let json = serde_json::to_vec(
            &results
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|(k, v)| (k.clone(), format!("{v}")))
                        .collect::<HashMap<String, String>>()
                })
                .collect::<Vec<_>>(),
        )
        .map_err(|e| anyhow!("Serialization error: {e}"))?;

        std::fs::write(&path, &json)
            .map_err(|e| anyhow!("Failed to write spill file {}: {e}", path.display()))?;

        Ok(ViewData::OnDisk {
            path,
            row_count: results.len(),
        })
    }

    fn load_from_disk(&self, path: &PathBuf) -> Result<Vec<BindingRow>> {
        let bytes = std::fs::read(path).map_err(|e| anyhow!("Failed to read spill file: {e}"))?;
        let raw: Vec<HashMap<String, String>> =
            serde_json::from_slice(&bytes).map_err(|e| anyhow!("Deserialization error: {e}"))?;

        let rows = raw
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|(k, v)| (k, RdfTerm::plain_literal(v)))
                    .collect::<BindingRow>()
            })
            .collect();
        Ok(rows)
    }

    fn record_hit(&self) {
        if let Ok(mut c) = self.counters.write() {
            c.hits += 1;
        }
    }

    fn record_miss(&self) {
        if let Ok(mut c) = self.counters.write() {
            c.misses += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_row(key: &str, val: &str) -> BindingRow {
        let mut row = BindingRow::new();
        row.insert(key.to_string(), RdfTerm::plain_literal(val));
        row
    }

    fn temp_manager() -> MaterializedViewManager {
        let config = ViewManagerConfig {
            max_views: 10,
            default_ttl: Duration::from_secs(60),
            spill_dir: std::env::temp_dir().join("oxirs_view_cache_test"),
            ..Default::default()
        };
        MaterializedViewManager::with_config(config)
    }

    #[test]
    fn test_store_and_retrieve_in_memory_view() {
        let manager = temp_manager();
        let rows = vec![make_row("s", "http://example.org/a")];
        manager
            .store_view("hash1", "SELECT * WHERE { ?s ?p ?o }", rows.clone(), vec![])
            .unwrap();

        let retrieved = manager.get_view("hash1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 1);
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let manager = temp_manager();
        assert!(manager.get_view("nonexistent_hash").is_none());
    }

    #[test]
    fn test_expired_view_returns_none() {
        // Extremely short TTL
        let config = ViewManagerConfig {
            max_views: 10,
            default_ttl: Duration::from_nanos(1),
            spill_dir: std::env::temp_dir().join("oxirs_view_cache_test_ttl"),
            ..Default::default()
        };
        let manager = MaterializedViewManager::with_config(config);

        let rows = vec![make_row("s", "http://example.org/a")];
        manager
            .store_view("hash_ttl", "pattern", rows, vec![])
            .unwrap();

        // Sleep to ensure TTL expires
        std::thread::sleep(Duration::from_millis(5));

        assert!(
            manager.get_view("hash_ttl").is_none(),
            "Expired view should return None"
        );
    }

    #[test]
    fn test_invalidate_pattern() {
        let manager = temp_manager();
        let rows = vec![make_row("s", "http://example.org/a")];
        manager
            .store_view(
                "hash2",
                "SELECT * WHERE { ?s foaf:name ?name }",
                rows.clone(),
                vec![],
            )
            .unwrap();
        manager
            .store_view(
                "hash3",
                "SELECT * WHERE { ?s rdf:type ?type }",
                rows,
                vec![],
            )
            .unwrap();

        let removed = manager.invalidate_pattern("foaf:name");
        assert_eq!(removed, 1, "Should remove exactly one view");
        assert!(
            manager.get_view("hash2").is_none(),
            "Invalidated view should be gone"
        );
        assert!(
            manager.get_view("hash3").is_some(),
            "Other view should remain"
        );
    }

    #[test]
    fn test_invalidate_by_predicate() {
        let manager = temp_manager();
        let rows = vec![make_row("s", "http://example.org/a")];
        manager
            .store_view(
                "hash_pred1",
                "pattern_a",
                rows.clone(),
                vec!["http://xmlns.com/foaf/0.1/name".to_string()],
            )
            .unwrap();
        manager
            .store_view(
                "hash_pred2",
                "pattern_b",
                rows,
                vec!["http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()],
            )
            .unwrap();

        let removed = manager.invalidate_by_predicate("http://xmlns.com/foaf/0.1/name");
        assert_eq!(removed, 1);
        assert!(manager.get_view("hash_pred1").is_none());
        assert!(manager.get_view("hash_pred2").is_some());
    }

    #[test]
    fn test_evict_expired() {
        let config = ViewManagerConfig {
            max_views: 10,
            default_ttl: Duration::from_nanos(1),
            spill_dir: std::env::temp_dir().join("oxirs_view_cache_test_evict"),
            ..Default::default()
        };
        let manager = MaterializedViewManager::with_config(config);

        let rows = vec![make_row("x", "val")];
        manager
            .store_view("exp1", "pat", rows.clone(), vec![])
            .unwrap();
        manager.store_view("exp2", "pat2", rows, vec![]).unwrap();

        std::thread::sleep(Duration::from_millis(5));
        let removed = manager.evict_expired();
        assert_eq!(removed, 2, "Both expired views should be evicted");
    }

    #[test]
    fn test_lru_eviction_when_full() {
        let config = ViewManagerConfig {
            max_views: 2,
            default_ttl: Duration::from_secs(300),
            spill_dir: std::env::temp_dir().join("oxirs_view_cache_test_lru"),
            ..Default::default()
        };
        let manager = MaterializedViewManager::with_config(config);

        let rows = vec![make_row("a", "val")];
        manager
            .store_view("lru1", "pat1", rows.clone(), vec![])
            .unwrap();
        // Touch lru1 to make it more recently used
        let _ = manager.get_view("lru1");
        manager
            .store_view("lru2", "pat2", rows.clone(), vec![])
            .unwrap();
        // Now add a third - lru1 was accessed most recently, lru2 second
        // lru1 is most recent (was accessed after creation)
        manager.store_view("lru3", "pat3", rows, vec![]).unwrap();

        let stats = manager.get_stats();
        assert_eq!(
            stats.total_views, 2,
            "Manager should enforce max_views capacity"
        );
    }

    #[test]
    fn test_hit_miss_stats() {
        let manager = temp_manager();
        let rows = vec![make_row("s", "http://example.org/a")];
        manager
            .store_view("stat_hash", "pattern", rows, vec![])
            .unwrap();

        let _ = manager.get_view("stat_hash"); // hit
        let _ = manager.get_view("missing"); // miss

        let stats = manager.get_stats();
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
        assert!((stats.hit_ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_explicit_invalidate_single_view() {
        let manager = temp_manager();
        let rows = vec![make_row("s", "val")];
        manager.store_view("del_hash", "pat", rows, vec![]).unwrap();
        assert!(manager.get_view("del_hash").is_some());

        let removed = manager.invalidate_view("del_hash");
        assert!(removed, "Should report successful removal");
        assert!(manager.get_view("del_hash").is_none());
    }

    #[test]
    fn test_rdf_term_display() {
        let iri = RdfTerm::iri("http://example.org/s");
        assert_eq!(format!("{iri}"), "<http://example.org/s>");

        let blank = RdfTerm::blank_node("b1");
        assert_eq!(format!("{blank}"), "_:b1");

        let lit = RdfTerm::plain_literal("hello");
        assert_eq!(format!("{lit}"), "\"hello\"");

        let typed = RdfTerm::Literal {
            value: "42".to_string(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
            lang: None,
        };
        assert!(format!("{typed}").contains("^^"));
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;
    use std::time::Duration;

    fn make_row(key: &str, val: &str) -> BindingRow {
        let mut row = BindingRow::new();
        row.insert(key.to_string(), RdfTerm::plain_literal(val));
        row
    }

    fn make_iri_row(key: &str, iri: &str) -> BindingRow {
        let mut row = BindingRow::new();
        row.insert(key.to_string(), RdfTerm::iri(iri));
        row
    }

    fn long_ttl_manager() -> MaterializedViewManager {
        let config = ViewManagerConfig {
            max_views: 20,
            default_ttl: Duration::from_secs(3600),
            spill_dir: std::env::temp_dir().join("oxirs_view_ext_test_long"),
            ..Default::default()
        };
        MaterializedViewManager::with_config(config)
    }

    // --- RdfTerm tests ---

    #[test]
    fn test_rdf_term_iri_is_iri() {
        let term = RdfTerm::iri("http://example.org/s");
        assert!(term.is_iri());
        assert!(!term.is_literal());
    }

    #[test]
    fn test_rdf_term_literal_is_literal() {
        let term = RdfTerm::plain_literal("hello");
        assert!(term.is_literal());
        assert!(!term.is_iri());
    }

    #[test]
    fn test_rdf_term_blank_node_is_neither() {
        let term = RdfTerm::blank_node("b0");
        assert!(!term.is_iri());
        assert!(!term.is_literal());
    }

    #[test]
    fn test_rdf_term_literal_with_lang_display() {
        let term = RdfTerm::Literal {
            value: "hello".to_string(),
            datatype: None,
            lang: Some("en".to_string()),
        };
        let s = format!("{term}");
        assert!(
            s.contains("@en"),
            "Lang-tagged literal should include @lang"
        );
    }

    #[test]
    fn test_rdf_term_equality() {
        let a = RdfTerm::iri("http://example.org/x");
        let b = RdfTerm::iri("http://example.org/x");
        let c = RdfTerm::iri("http://example.org/y");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // --- MaterializedViewManager store/retrieve ---

    #[test]
    fn test_store_multiple_views_and_retrieve_all() {
        let manager = long_ttl_manager();
        for i in 0..5 {
            let rows = vec![make_row("x", &format!("val{i}"))];
            manager
                .store_view(&format!("hash_{i}"), &format!("pattern_{i}"), rows, vec![])
                .unwrap();
        }
        for i in 0..5 {
            assert!(
                manager.get_view(&format!("hash_{i}")).is_some(),
                "View {i} should be retrievable"
            );
        }
    }

    #[test]
    fn test_get_view_increments_hit_count() {
        let manager = long_ttl_manager();
        let rows = vec![make_row("k", "v")];
        manager.store_view("h_hit", "pat", rows, vec![]).unwrap();

        let _ = manager.get_view("h_hit");
        let _ = manager.get_view("h_hit");

        let stats = manager.get_stats();
        assert_eq!(
            stats.hit_count, 2,
            "Two successful gets should count as two hits"
        );
    }

    #[test]
    fn test_get_missing_view_increments_miss_count() {
        let manager = long_ttl_manager();
        let _ = manager.get_view("does_not_exist_1");
        let _ = manager.get_view("does_not_exist_2");

        let stats = manager.get_stats();
        assert_eq!(stats.miss_count, 2, "Two misses should be recorded");
    }

    #[test]
    fn test_stats_total_rows_cached() {
        let manager = long_ttl_manager();
        let rows: Vec<BindingRow> = (0..5).map(|i| make_row("k", &i.to_string())).collect();
        manager.store_view("rows5", "pat", rows, vec![]).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_rows_cached, 5, "Should track total cached rows");
    }

    #[test]
    fn test_stats_in_memory_vs_on_disk_count() {
        let manager = long_ttl_manager();
        let rows = vec![make_iri_row("s", "http://example.org/a")];
        manager.store_view("in_mem", "pat", rows, vec![]).unwrap();

        let stats = manager.get_stats();
        assert!(
            stats.in_memory_views >= 1 || stats.on_disk_views >= 1,
            "View should be tracked in stats"
        );
    }

    // --- Invalidation tests ---

    #[test]
    fn test_invalidate_nonexistent_view_returns_false() {
        let manager = long_ttl_manager();
        let removed = manager.invalidate_view("no_such_hash");
        assert!(!removed, "Removing non-existent view should return false");
    }

    #[test]
    fn test_invalidate_pattern_with_no_matching_views() {
        let manager = long_ttl_manager();
        let removed = manager.invalidate_pattern("some_predicate");
        assert_eq!(removed, 0, "No views should be removed when none match");
    }

    #[test]
    fn test_invalidate_by_predicate_removes_only_matching() {
        let manager = long_ttl_manager();
        let rows = vec![make_row("s", "val")];
        // Pass the target predicate in dependent_predicates for the matching view
        manager
            .store_view(
                "pred_match",
                "pat_with_target",
                rows.clone(),
                vec!["http://example.org/target_pred".to_string()],
            )
            .unwrap();
        // No dependent predicates for the non-matching view
        manager
            .store_view("no_match", "pat_without_target", rows, vec![])
            .unwrap();

        let removed = manager.invalidate_by_predicate("http://example.org/target_pred");
        assert_eq!(removed, 1, "Only the matching view should be invalidated");
        assert!(
            manager.get_view("no_match").is_some(),
            "Non-matching view should remain"
        );
    }

    // --- Expiry tests ---

    #[test]
    fn test_evict_expired_on_empty_manager() {
        let manager = long_ttl_manager();
        let removed = manager.evict_expired();
        assert_eq!(removed, 0, "Evicting empty manager should remove 0 views");
    }

    #[test]
    fn test_non_expired_view_not_evicted() {
        let manager = long_ttl_manager();
        let rows = vec![make_row("k", "v")];
        manager.store_view("live", "pat", rows, vec![]).unwrap();

        let removed = manager.evict_expired();
        assert_eq!(removed, 0, "Live view should not be evicted");
        assert!(manager.get_view("live").is_some());
    }

    // --- ViewManagerConfig ---

    #[test]
    fn test_view_manager_config_default_values() {
        let config = ViewManagerConfig::default();
        assert!(config.max_views > 0, "Default max_views should be positive");
        assert!(
            config.default_ttl.as_secs() > 0,
            "Default TTL should be positive"
        );
    }

    #[test]
    fn test_new_constructor_equivalent_to_with_config() {
        let mgr1 = MaterializedViewManager::new(50, Duration::from_secs(120));
        let mgr2 = MaterializedViewManager::with_config(ViewManagerConfig {
            max_views: 50,
            default_ttl: Duration::from_secs(120),
            ..Default::default()
        });

        // Both should start empty
        assert!(mgr1.get_view("x").is_none());
        assert!(mgr2.get_view("x").is_none());
    }
}
