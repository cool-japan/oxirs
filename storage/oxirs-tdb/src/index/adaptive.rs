//! Adaptive index strategies for OxiRS TDB.
//!
//! Provides smart index selection based on observed query patterns, and
//! a tiered-storage abstraction with hot/warm/cold layers.

use std::collections::HashMap;

// ─── Index Types ──────────────────────────────────────────────────────────────

/// Category of index structure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    /// Hash-based index — O(1) equality lookups
    HashIndex,
    /// B-Tree index — O(log n), supports range scans
    BTreeIndex,
    /// Bitmap index — efficient for low-cardinality columns
    BitmapIndex,
    /// Compressed bitmap index — roaring-bitmap style, very low cardinality
    CompressedBitmapIndex,
}

// ─── IndexStats ──────────────────────────────────────────────────────────────

/// Runtime statistics for a single index instance
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Type of this index
    pub index_type: IndexType,
    /// Number of distinct indexed entries
    pub entry_count: usize,
    /// Approximate memory/disk usage in bytes
    pub size_bytes: usize,
    /// Average lookup time in milliseconds
    pub avg_lookup_ms: f64,
    /// Cache / bloom-filter hit rate in [0, 1]
    pub hit_rate: f64,
}

// ─── QueryPattern ────────────────────────────────────────────────────────────

/// Observed access pattern for a particular predicate
#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// RDF predicate URI
    pub predicate: String,
    /// Whether equality lookups are used (e.g. `FILTER(?x = <iri>)`)
    pub is_equality: bool,
    /// Whether range queries are used (e.g. FILTER(?x > 5))
    pub is_range: bool,
    /// Number of times this predicate has been queried
    pub frequency: u64,
    /// Fraction of triples selected per query (lower = more selective)
    pub selectivity: f64,
}

// ─── Thresholds ──────────────────────────────────────────────────────────────

/// Thresholds that govern which index type is selected
#[derive(Debug, Clone)]
pub struct IndexSelectionThresholds {
    /// Maximum cardinality to use bitmap index (e.g. boolean columns)
    pub bitmap_max_cardinality: usize,
    /// Minimum query frequency to prefer hash index for equality
    pub hash_min_frequency: u64,
    /// Whether to use B-Tree for range queries
    pub btree_for_ranges: bool,
}

impl Default for IndexSelectionThresholds {
    fn default() -> Self {
        Self {
            bitmap_max_cardinality: 2,
            hash_min_frequency: 10,
            btree_for_ranges: true,
        }
    }
}

// ─── AdaptiveIndexSelector ───────────────────────────────────────────────────

/// Tracks query patterns and recommends the optimal index type per predicate.
pub struct AdaptiveIndexSelector {
    patterns: HashMap<String, QueryPattern>,
    thresholds: IndexSelectionThresholds,
}

impl AdaptiveIndexSelector {
    /// Create a new selector with default thresholds.
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            thresholds: IndexSelectionThresholds::default(),
        }
    }

    /// Create a selector with custom thresholds.
    pub fn with_thresholds(thresholds: IndexSelectionThresholds) -> Self {
        Self {
            patterns: HashMap::new(),
            thresholds,
        }
    }

    /// Record a query observation for `predicate`.
    pub fn observe_query(&mut self, predicate: &str, is_equality: bool, is_range: bool) {
        let entry = self
            .patterns
            .entry(predicate.to_string())
            .or_insert_with(|| QueryPattern {
                predicate: predicate.to_string(),
                is_equality: false,
                is_range: false,
                frequency: 0,
                selectivity: 1.0,
            });
        entry.frequency += 1;
        if is_equality {
            entry.is_equality = true;
        }
        if is_range {
            entry.is_range = true;
        }
    }

    /// Recommend an index type for `predicate` based on observed patterns.
    ///
    /// Decision rules (evaluated in order):
    /// 1. If range queries observed and `btree_for_ranges` → [`IndexType::BTreeIndex`]
    /// 2. If equality queries and frequency ≥ `hash_min_frequency` → [`IndexType::HashIndex`]
    /// 3. If cardinality (approximated by frequency) ≤ `bitmap_max_cardinality` → [`IndexType::CompressedBitmapIndex`]
    /// 4. Default → [`IndexType::BTreeIndex`]
    pub fn recommend(&self, predicate: &str) -> IndexType {
        match self.patterns.get(predicate) {
            None => IndexType::BTreeIndex,
            Some(p) => {
                if p.is_range && self.thresholds.btree_for_ranges {
                    return IndexType::BTreeIndex;
                }
                if p.is_equality && p.frequency >= self.thresholds.hash_min_frequency {
                    return IndexType::HashIndex;
                }
                if p.frequency <= self.thresholds.bitmap_max_cardinality as u64 {
                    return IndexType::CompressedBitmapIndex;
                }
                IndexType::BTreeIndex
            }
        }
    }

    /// Return recommendations for every tracked predicate.
    pub fn all_recommendations(&self) -> Vec<(String, IndexType)> {
        let mut recs: Vec<_> = self
            .patterns
            .keys()
            .map(|p| (p.clone(), self.recommend(p)))
            .collect();
        recs.sort_by(|a, b| a.0.cmp(&b.0));
        recs
    }

    /// Return the top-`n` predicates by query frequency, highest first.
    pub fn top_predicates_by_frequency(&self, n: usize) -> Vec<(&str, u64)> {
        let mut entries: Vec<(&str, u64)> = self
            .patterns
            .iter()
            .map(|(k, v)| (k.as_str(), v.frequency))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }
}

impl Default for AdaptiveIndexSelector {
    fn default() -> Self {
        Self::new()
    }
}

// ─── TieredStorage ───────────────────────────────────────────────────────────

/// A simple three-tier storage abstraction: hot (in-memory), warm (in-memory,
/// compressed), and cold (disk-backed, path only tracked here).
///
/// Eviction policy: when a layer exceeds its threshold, the oldest-inserted
/// entries (by insertion order) are evicted to the next layer.
pub struct TieredStorage {
    hot_layer: HashMap<String, Vec<u8>>,
    hot_order: Vec<String>,
    warm_layer: HashMap<String, Vec<u8>>,
    warm_order: Vec<String>,
    cold_layer_path: Option<std::path::PathBuf>,
    hot_threshold: usize,
    warm_threshold: usize,
}

impl TieredStorage {
    /// Create a new tiered storage without a cold-layer path.
    pub fn new(hot_threshold: usize, warm_threshold: usize) -> Self {
        Self {
            hot_layer: HashMap::new(),
            hot_order: Vec::new(),
            warm_layer: HashMap::new(),
            warm_order: Vec::new(),
            cold_layer_path: None,
            hot_threshold,
            warm_threshold,
        }
    }

    /// Create a new tiered storage with a cold-layer path.
    pub fn with_cold_path(
        hot_threshold: usize,
        warm_threshold: usize,
        cold_path: std::path::PathBuf,
    ) -> Self {
        let mut s = Self::new(hot_threshold, warm_threshold);
        s.cold_layer_path = Some(cold_path);
        s
    }

    /// Retrieve a value, checking hot then warm layers.
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        if let Some(v) = self.hot_layer.get(key) {
            return Some(v.clone());
        }
        self.warm_layer.get(key).cloned()
    }

    /// Insert or update a value. The entry always lands in the hot layer.
    pub fn put(&mut self, key: &str, value: Vec<u8>) {
        if !self.hot_layer.contains_key(key) {
            self.hot_order.push(key.to_string());
        }
        self.hot_layer.insert(key.to_string(), value);
    }

    /// Evict entries from the hot layer to the warm layer when hot exceeds
    /// its threshold.  Returns the number of entries evicted.
    pub fn evict_hot_to_warm(&mut self) -> usize {
        let mut evicted = 0;
        while self.hot_layer.len() > self.hot_threshold {
            if self.hot_order.is_empty() {
                break;
            }
            let key = self.hot_order.remove(0);
            if let Some(val) = self.hot_layer.remove(&key) {
                if !self.warm_layer.contains_key(&key) {
                    self.warm_order.push(key.clone());
                }
                // In a real implementation we would compress here; for the
                // purposes of this abstraction we store as-is.
                self.warm_layer.insert(key, val);
                evicted += 1;
            }
        }
        evicted
    }

    /// Evict entries from the warm layer to the cold layer (disk) when warm
    /// exceeds its threshold.  Returns the number of entries evicted.
    pub fn evict_warm_to_cold(&mut self) -> usize {
        let mut evicted = 0;
        while self.warm_layer.len() > self.warm_threshold {
            if self.warm_order.is_empty() {
                break;
            }
            let key = self.warm_order.remove(0);
            if let Some(val) = self.warm_layer.remove(&key) {
                // Write to disk if a cold-layer path is configured.
                if let Some(ref cold_path) = self.cold_layer_path {
                    let file_path = cold_path.join(sanitize_key(&key));
                    let _ = std::fs::create_dir_all(cold_path);
                    let _ = std::fs::write(file_path, &val);
                }
                evicted += 1;
            }
        }
        evicted
    }

    /// Number of entries currently in the hot layer.
    pub fn hot_count(&self) -> usize {
        self.hot_layer.len()
    }

    /// Number of entries currently in the warm layer.
    pub fn warm_count(&self) -> usize {
        self.warm_layer.len()
    }
}

/// Convert a key string to a safe filename component.
fn sanitize_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AdaptiveIndexSelector ────────────────────────────────────────────────

    #[test]
    fn test_selector_new() {
        let sel = AdaptiveIndexSelector::new();
        assert!(sel.patterns.is_empty());
    }

    #[test]
    fn test_recommend_unknown_predicate_is_btree() {
        let sel = AdaptiveIndexSelector::new();
        assert_eq!(sel.recommend("http://unknown"), IndexType::BTreeIndex);
    }

    #[test]
    fn test_observe_query_increments_frequency() {
        let mut sel = AdaptiveIndexSelector::new();
        sel.observe_query("http://p", true, false);
        sel.observe_query("http://p", true, false);
        assert_eq!(sel.patterns["http://p"].frequency, 2);
    }

    #[test]
    fn test_observe_query_sets_equality_flag() {
        let mut sel = AdaptiveIndexSelector::new();
        sel.observe_query("http://p", true, false);
        assert!(sel.patterns["http://p"].is_equality);
        assert!(!sel.patterns["http://p"].is_range);
    }

    #[test]
    fn test_observe_query_sets_range_flag() {
        let mut sel = AdaptiveIndexSelector::new();
        sel.observe_query("http://p", false, true);
        assert!(sel.patterns["http://p"].is_range);
    }

    #[test]
    fn test_recommend_hash_for_frequent_equality() {
        let mut sel = AdaptiveIndexSelector::new();
        for _ in 0..15 {
            sel.observe_query("http://p", true, false);
        }
        // frequency=15 >= hash_min_frequency=10, equality=true, no range
        assert_eq!(sel.recommend("http://p"), IndexType::HashIndex);
    }

    #[test]
    fn test_recommend_btree_for_range_overrides_hash() {
        let mut sel = AdaptiveIndexSelector::new();
        for _ in 0..20 {
            sel.observe_query("http://p", true, true); // both equality and range
        }
        // Range takes priority when btree_for_ranges = true
        assert_eq!(sel.recommend("http://p"), IndexType::BTreeIndex);
    }

    #[test]
    fn test_recommend_compressed_bitmap_low_frequency() {
        let mut sel = AdaptiveIndexSelector::new();
        sel.observe_query("http://p", false, false); // frequency=1, no range, no equality
        assert_eq!(sel.recommend("http://p"), IndexType::CompressedBitmapIndex);
    }

    #[test]
    fn test_recommend_btree_default_medium_frequency_no_flags() {
        let mut sel = AdaptiveIndexSelector::new();
        for _ in 0..5 {
            sel.observe_query("http://p", false, false);
        }
        // frequency=5 > bitmap_max_cardinality=2, not equality-frequent, no range
        assert_eq!(sel.recommend("http://p"), IndexType::BTreeIndex);
    }

    #[test]
    fn test_all_recommendations_returns_all() {
        let mut sel = AdaptiveIndexSelector::new();
        sel.observe_query("http://a", true, false);
        sel.observe_query("http://b", false, true);
        let recs = sel.all_recommendations();
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn test_all_recommendations_sorted() {
        let mut sel = AdaptiveIndexSelector::new();
        sel.observe_query("http://z", true, false);
        sel.observe_query("http://a", true, false);
        sel.observe_query("http://m", true, false);
        let recs = sel.all_recommendations();
        let keys: Vec<_> = recs.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_top_predicates_by_frequency() {
        let mut sel = AdaptiveIndexSelector::new();
        for _ in 0..10 {
            sel.observe_query("http://frequent", true, false);
        }
        for _ in 0..2 {
            sel.observe_query("http://rare", false, false);
        }
        for _ in 0..5 {
            sel.observe_query("http://medium", false, false);
        }
        let top = sel.top_predicates_by_frequency(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "http://frequent");
        assert_eq!(top[0].1, 10);
    }

    #[test]
    fn test_top_predicates_by_frequency_empty() {
        let sel = AdaptiveIndexSelector::new();
        assert!(sel.top_predicates_by_frequency(5).is_empty());
    }

    #[test]
    fn test_with_thresholds() {
        let thresh = IndexSelectionThresholds {
            bitmap_max_cardinality: 100,
            hash_min_frequency: 1,
            btree_for_ranges: false,
        };
        let mut sel = AdaptiveIndexSelector::with_thresholds(thresh);
        sel.observe_query("http://p", true, true); // range, but btree_for_ranges=false
                                                   // Now: not range-preferred, equality=true, frequency=1 >= 1 → HashIndex
        assert_eq!(sel.recommend("http://p"), IndexType::HashIndex);
    }

    // ── TieredStorage ────────────────────────────────────────────────────────

    #[test]
    fn test_tiered_storage_new() {
        let ts = TieredStorage::new(10, 5);
        assert_eq!(ts.hot_count(), 0);
        assert_eq!(ts.warm_count(), 0);
    }

    #[test]
    fn test_put_and_get_from_hot() {
        let mut ts = TieredStorage::new(10, 5);
        ts.put("key1", b"value1".to_vec());
        let got = ts.get("key1").unwrap();
        assert_eq!(got, b"value1");
    }

    #[test]
    fn test_get_missing_returns_none() {
        let ts = TieredStorage::new(10, 5);
        assert!(ts.get("missing").is_none());
    }

    #[test]
    fn test_hot_count_after_puts() {
        let mut ts = TieredStorage::new(10, 5);
        ts.put("k1", vec![1]);
        ts.put("k2", vec![2]);
        assert_eq!(ts.hot_count(), 2);
    }

    #[test]
    fn test_evict_hot_to_warm() {
        let mut ts = TieredStorage::new(2, 10);
        ts.put("k1", vec![1]);
        ts.put("k2", vec![2]);
        ts.put("k3", vec![3]); // hot_count = 3 > threshold 2
        let evicted = ts.evict_hot_to_warm();
        assert!(evicted > 0);
        assert!(ts.hot_count() <= 2);
    }

    #[test]
    fn test_evict_hot_to_warm_no_eviction_needed() {
        let mut ts = TieredStorage::new(10, 5);
        ts.put("k1", vec![1]);
        let evicted = ts.evict_hot_to_warm();
        assert_eq!(evicted, 0);
        assert_eq!(ts.hot_count(), 1);
    }

    #[test]
    fn test_evicted_entries_accessible_from_warm() {
        let mut ts = TieredStorage::new(1, 10);
        ts.put("k1", b"first".to_vec());
        ts.put("k2", b"second".to_vec()); // hot_count = 2 > 1
        ts.evict_hot_to_warm();
        // k1 should have been evicted to warm
        assert!(ts.warm_count() > 0);
        assert!(ts.get("k1").is_some() || ts.get("k2").is_some());
    }

    #[test]
    fn test_evict_warm_to_cold_no_path() {
        let mut ts = TieredStorage::new(1, 2);
        ts.put("k1", vec![1]);
        ts.put("k2", vec![2]);
        ts.evict_hot_to_warm();
        ts.put("k3", vec![3]);
        ts.put("k4", vec![4]);
        ts.evict_hot_to_warm(); // warm now has entries
                                // Manually push warm over limit
        ts.warm_layer.insert("extra1".to_string(), vec![9]);
        ts.warm_layer.insert("extra2".to_string(), vec![9]);
        ts.warm_order.push("extra1".to_string());
        ts.warm_order.push("extra2".to_string());
        let evicted = ts.evict_warm_to_cold();
        // Should have evicted some warm entries
        assert!(evicted > 0 || ts.warm_count() <= 2);
    }

    #[test]
    fn test_evict_warm_to_cold_with_path() {
        let cold_dir = std::env::temp_dir().join("oxirs_tdb_tiered_cold_test");
        std::fs::remove_dir_all(&cold_dir).ok();
        let mut ts = TieredStorage::with_cold_path(1, 1, cold_dir.clone());
        ts.put("k1", b"v1".to_vec());
        ts.put("k2", b"v2".to_vec());
        ts.evict_hot_to_warm();
        ts.warm_layer.insert("w1".to_string(), b"warm1".to_vec());
        ts.warm_layer.insert("w2".to_string(), b"warm2".to_vec());
        ts.warm_order.push("w1".to_string());
        ts.warm_order.push("w2".to_string());
        ts.evict_warm_to_cold();
        std::fs::remove_dir_all(&cold_dir).ok();
    }
}
