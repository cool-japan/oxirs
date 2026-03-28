//! Tag-based time series indexing.
//!
//! Provides an inverted index on tag key-value pairs for efficient
//! series lookup, cardinality tracking, composite tag queries (AND/OR),
//! prefix/regex/NOT filters, auto-completion, statistics, index
//! maintenance (add/remove/rebuild), and wildcard matching.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

// ── TagFilter ────────────────────────────────────────────────────────────────

/// A single filter predicate on one tag key.
#[derive(Debug, Clone, PartialEq)]
pub enum TagFilter {
    /// Exact match on a tag value.
    Equals {
        /// Tag key.
        key: String,
        /// Expected value.
        value: String,
    },
    /// Prefix match on a tag value.
    Prefix {
        /// Tag key.
        key: String,
        /// Required prefix.
        prefix: String,
    },
    /// Regex match on a tag value (simplified: substring containment).
    Regex {
        /// Tag key.
        key: String,
        /// Pattern that must appear as a substring.
        pattern: String,
    },
    /// Negation: series must NOT have this key-value pair.
    Not {
        /// Tag key.
        key: String,
        /// Excluded value.
        value: String,
    },
    /// Wildcard match on a tag value (`*` matches any sequence of characters,
    /// `?` matches exactly one character).
    Wildcard {
        /// Tag key.
        key: String,
        /// Wildcard pattern.
        pattern: String,
    },
}

// ── CompositeFilter ──────────────────────────────────────────────────────────

/// A composite filter combining multiple tag filters.
#[derive(Debug, Clone, PartialEq)]
pub enum CompositeFilter {
    /// A single filter.
    Single(TagFilter),
    /// All sub-filters must match (intersection).
    And(Vec<CompositeFilter>),
    /// At least one sub-filter must match (union).
    Or(Vec<CompositeFilter>),
}

// ── TagStats ─────────────────────────────────────────────────────────────────

/// Statistics about tag usage across all indexed series.
#[derive(Debug, Clone, Default)]
pub struct TagStats {
    /// Total number of distinct tag keys.
    pub total_keys: usize,
    /// Total number of distinct key-value pairs.
    pub total_pairs: usize,
    /// Total number of indexed series.
    pub total_series: usize,
    /// Number of distinct values per key.
    pub cardinality: HashMap<String, usize>,
    /// Most common tag keys, ordered by number of series using them (desc).
    pub most_common_keys: Vec<(String, usize)>,
}

// ── IndexStats ───────────────────────────────────────────────────────────────

/// Internal index maintenance statistics.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Number of add operations.
    pub adds: u64,
    /// Number of remove operations.
    pub removes: u64,
    /// Number of full rebuilds.
    pub rebuilds: u64,
    /// Number of queries evaluated.
    pub queries: u64,
}

// ── TagIndex ─────────────────────────────────────────────────────────────────

/// An inverted index mapping `(tag_key, tag_value)` pairs to sets of series IDs.
///
/// This is the primary structure for tag-based filtering in the TSDB.
pub struct TagIndex {
    /// Inverted index: `(key, value) -> set of series_ids`.
    index: HashMap<String, HashMap<String, BTreeSet<String>>>,
    /// Forward index: `series_id -> set of (key, value)` pairs.
    forward: HashMap<String, BTreeSet<(String, String)>>,
    /// All known series ids.
    series_ids: BTreeSet<String>,
    /// Maintenance statistics.
    stats: IndexStats,
}

impl Default for TagIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl TagIndex {
    /// Create an empty tag index.
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            forward: HashMap::new(),
            series_ids: BTreeSet::new(),
            stats: IndexStats::default(),
        }
    }

    /// Return internal maintenance statistics.
    pub fn index_stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Add a series with the given tags.
    ///
    /// `tags` is a slice of `(key, value)` pairs.
    pub fn add_series(&mut self, series_id: impl Into<String>, tags: &[(&str, &str)]) {
        let id = series_id.into();
        self.series_ids.insert(id.clone());
        self.stats.adds += 1;

        let entry = self.forward.entry(id.clone()).or_default();

        for &(key, value) in tags {
            let key_map = self.index.entry(key.to_string()).or_default();
            let val_set = key_map.entry(value.to_string()).or_default();
            val_set.insert(id.clone());
            entry.insert((key.to_string(), value.to_string()));
        }
    }

    /// Remove a series from the index.
    pub fn remove_series(&mut self, series_id: &str) -> bool {
        if !self.series_ids.remove(series_id) {
            return false;
        }
        self.stats.removes += 1;

        if let Some(tags) = self.forward.remove(series_id) {
            for (key, value) in tags {
                if let Some(key_map) = self.index.get_mut(&key) {
                    if let Some(val_set) = key_map.get_mut(&value) {
                        val_set.remove(series_id);
                        if val_set.is_empty() {
                            key_map.remove(&value);
                        }
                    }
                    if key_map.is_empty() {
                        self.index.remove(&key);
                    }
                }
            }
        }
        true
    }

    /// Check whether a series exists in the index.
    pub fn contains_series(&self, series_id: &str) -> bool {
        self.series_ids.contains(series_id)
    }

    /// Return the total number of indexed series.
    pub fn series_count(&self) -> usize {
        self.series_ids.len()
    }

    /// Return all indexed series IDs.
    pub fn all_series(&self) -> Vec<String> {
        self.series_ids.iter().cloned().collect()
    }

    /// Return the tags associated with a series.
    pub fn tags_for_series(&self, series_id: &str) -> Vec<(String, String)> {
        self.forward
            .get(series_id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Return the number of distinct values for a given tag key.
    pub fn cardinality(&self, key: &str) -> usize {
        self.index.get(key).map(|m| m.len()).unwrap_or(0)
    }

    /// Return all distinct tag keys.
    pub fn tag_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.index.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Return all distinct values for a given key.
    pub fn tag_values(&self, key: &str) -> Vec<String> {
        self.index
            .get(key)
            .map(|m| {
                let mut vals: Vec<String> = m.keys().cloned().collect();
                vals.sort();
                vals
            })
            .unwrap_or_default()
    }

    /// Auto-complete tag values for a key based on a prefix.
    pub fn autocomplete(&self, key: &str, prefix: &str) -> Vec<String> {
        self.index
            .get(key)
            .map(|m| {
                let mut matches: Vec<String> = m
                    .keys()
                    .filter(|v| v.starts_with(prefix))
                    .cloned()
                    .collect();
                matches.sort();
                matches
            })
            .unwrap_or_default()
    }

    /// Evaluate a single tag filter and return matching series IDs.
    pub fn evaluate_filter(&mut self, filter: &TagFilter) -> BTreeSet<String> {
        self.stats.queries += 1;
        match filter {
            TagFilter::Equals { key, value } => self.lookup_exact(key, value),
            TagFilter::Prefix { key, prefix } => self.lookup_prefix(key, prefix),
            TagFilter::Regex { key, pattern } => self.lookup_regex(key, pattern),
            TagFilter::Not { key, value } => self.lookup_not(key, value),
            TagFilter::Wildcard { key, pattern } => self.lookup_wildcard(key, pattern),
        }
    }

    /// Evaluate a composite filter.
    pub fn evaluate(&mut self, filter: &CompositeFilter) -> BTreeSet<String> {
        match filter {
            CompositeFilter::Single(f) => self.evaluate_filter(f),
            CompositeFilter::And(filters) => {
                let mut result: Option<BTreeSet<String>> = None;
                for f in filters {
                    let set = self.evaluate(f);
                    result = Some(match result {
                        None => set,
                        Some(prev) => prev.intersection(&set).cloned().collect(),
                    });
                }
                result.unwrap_or_default()
            }
            CompositeFilter::Or(filters) => {
                let mut result = BTreeSet::new();
                for f in filters {
                    let set = self.evaluate(f);
                    result = result.union(&set).cloned().collect();
                }
                result
            }
        }
    }

    /// Enumerate series that have a specific tag key (any value).
    pub fn series_with_key(&self, key: &str) -> BTreeSet<String> {
        self.index
            .get(key)
            .map(|m| {
                let mut result = BTreeSet::new();
                for ids in m.values() {
                    result = result.union(ids).cloned().collect();
                }
                result
            })
            .unwrap_or_default()
    }

    /// Compute comprehensive tag statistics.
    pub fn tag_stats(&self) -> TagStats {
        let mut total_pairs = 0usize;
        let mut cardinality = HashMap::new();
        let mut key_usage: BTreeMap<String, usize> = BTreeMap::new();

        for (key, val_map) in &self.index {
            let card = val_map.len();
            total_pairs += card;
            cardinality.insert(key.clone(), card);

            // Unique series per key
            let mut unique: HashSet<&String> = HashSet::new();
            for ids in val_map.values() {
                for id in ids {
                    unique.insert(id);
                }
            }
            key_usage.insert(key.clone(), unique.len());
        }

        let mut most_common: Vec<(String, usize)> = key_usage.into_iter().collect();
        most_common.sort_by_key(|m| Reverse(m.1));

        TagStats {
            total_keys: self.index.len(),
            total_pairs,
            total_series: self.series_ids.len(),
            cardinality,
            most_common_keys: most_common,
        }
    }

    /// Rebuild the entire index from the forward map.
    ///
    /// Useful after bulk modifications that may leave the inverted index
    /// inconsistent.
    pub fn rebuild(&mut self) {
        self.stats.rebuilds += 1;
        self.index.clear();

        for (series_id, tags) in &self.forward {
            for (key, value) in tags {
                let key_map = self.index.entry(key.clone()).or_default();
                let val_set = key_map.entry(value.clone()).or_default();
                val_set.insert(series_id.clone());
            }
        }
    }

    // ── Private filter implementations ───────────────────────────────────

    fn lookup_exact(&self, key: &str, value: &str) -> BTreeSet<String> {
        self.index
            .get(key)
            .and_then(|m| m.get(value))
            .cloned()
            .unwrap_or_default()
    }

    fn lookup_prefix(&self, key: &str, prefix: &str) -> BTreeSet<String> {
        let mut result = BTreeSet::new();
        if let Some(val_map) = self.index.get(key) {
            for (val, ids) in val_map {
                if val.starts_with(prefix) {
                    result = result.union(ids).cloned().collect();
                }
            }
        }
        result
    }

    fn lookup_regex(&self, key: &str, pattern: &str) -> BTreeSet<String> {
        let mut result = BTreeSet::new();
        if let Some(val_map) = self.index.get(key) {
            for (val, ids) in val_map {
                if val.contains(pattern) {
                    result = result.union(ids).cloned().collect();
                }
            }
        }
        result
    }

    fn lookup_not(&self, key: &str, value: &str) -> BTreeSet<String> {
        let excluded = self.lookup_exact(key, value);
        self.series_ids.difference(&excluded).cloned().collect()
    }

    fn lookup_wildcard(&self, key: &str, pattern: &str) -> BTreeSet<String> {
        let mut result = BTreeSet::new();
        if let Some(val_map) = self.index.get(key) {
            for (val, ids) in val_map {
                if wildcard_matches(pattern, val) {
                    result = result.union(ids).cloned().collect();
                }
            }
        }
        result
    }
}

// ── Wildcard matching ────────────────────────────────────────────────────────

/// Simple wildcard matching: `*` matches zero or more characters,
/// `?` matches exactly one character.
fn wildcard_matches(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    wildcard_dp(&p, &t)
}

/// Dynamic-programming wildcard match.
fn wildcard_dp(pattern: &[char], text: &[char]) -> bool {
    let m = pattern.len();
    let n = text.len();
    // dp[i][j] = true if pattern[0..i] matches text[0..j]
    let mut dp = vec![vec![false; n + 1]; m + 1];
    dp[0][0] = true;

    // Leading '*' can match empty string
    for i in 1..=m {
        if pattern[i - 1] == '*' {
            dp[i][0] = dp[i - 1][0];
        }
    }

    for i in 1..=m {
        for j in 1..=n {
            if pattern[i - 1] == '*' {
                // '*' matches zero chars (dp[i-1][j]) or one more char (dp[i][j-1])
                dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            } else if pattern[i - 1] == '?' || pattern[i - 1] == text[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }
    dp[m][n]
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index() -> TagIndex {
        let mut idx = TagIndex::new();
        idx.add_series(
            "s1",
            &[("host", "web-1"), ("region", "us-east"), ("env", "prod")],
        );
        idx.add_series(
            "s2",
            &[("host", "web-2"), ("region", "us-east"), ("env", "staging")],
        );
        idx.add_series(
            "s3",
            &[("host", "db-1"), ("region", "eu-west"), ("env", "prod")],
        );
        idx.add_series(
            "s4",
            &[("host", "db-2"), ("region", "eu-west"), ("env", "dev")],
        );
        idx
    }

    // ── Basic add/remove ─────────────────────────────────────────────────

    #[test]
    fn test_new_index_empty() {
        let idx = TagIndex::new();
        assert_eq!(idx.series_count(), 0);
        assert!(idx.all_series().is_empty());
    }

    #[test]
    fn test_add_series() {
        let mut idx = TagIndex::new();
        idx.add_series("s1", &[("host", "web-1")]);
        assert_eq!(idx.series_count(), 1);
        assert!(idx.contains_series("s1"));
    }

    #[test]
    fn test_add_multiple_series() {
        let idx = build_index();
        assert_eq!(idx.series_count(), 4);
    }

    #[test]
    fn test_remove_series() {
        let mut idx = build_index();
        assert!(idx.remove_series("s1"));
        assert!(!idx.contains_series("s1"));
        assert_eq!(idx.series_count(), 3);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut idx = build_index();
        assert!(!idx.remove_series("nope"));
    }

    #[test]
    fn test_remove_cleans_inverted_index() {
        let mut idx = TagIndex::new();
        idx.add_series("s1", &[("host", "unique-host")]);
        idx.remove_series("s1");
        assert_eq!(idx.cardinality("host"), 0);
    }

    #[test]
    fn test_tags_for_series() {
        let idx = build_index();
        let tags = idx.tags_for_series("s1");
        assert_eq!(tags.len(), 3);
        assert!(tags.contains(&("host".to_string(), "web-1".to_string())));
    }

    #[test]
    fn test_tags_for_nonexistent_series() {
        let idx = build_index();
        assert!(idx.tags_for_series("nope").is_empty());
    }

    // ── Cardinality and keys ─────────────────────────────────────────────

    #[test]
    fn test_cardinality() {
        let idx = build_index();
        assert_eq!(idx.cardinality("host"), 4); // web-1, web-2, db-1, db-2
        assert_eq!(idx.cardinality("region"), 2); // us-east, eu-west
        assert_eq!(idx.cardinality("env"), 3); // prod, staging, dev
    }

    #[test]
    fn test_cardinality_unknown_key() {
        let idx = build_index();
        assert_eq!(idx.cardinality("nonexistent"), 0);
    }

    #[test]
    fn test_tag_keys() {
        let idx = build_index();
        let keys = idx.tag_keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"host".to_string()));
        assert!(keys.contains(&"region".to_string()));
        assert!(keys.contains(&"env".to_string()));
    }

    #[test]
    fn test_tag_values() {
        let idx = build_index();
        let vals = idx.tag_values("env");
        assert_eq!(vals, vec!["dev", "prod", "staging"]);
    }

    #[test]
    fn test_tag_values_unknown_key() {
        let idx = build_index();
        assert!(idx.tag_values("nope").is_empty());
    }

    // ── Autocomplete ─────────────────────────────────────────────────────

    #[test]
    fn test_autocomplete() {
        let idx = build_index();
        let results = idx.autocomplete("host", "web");
        assert_eq!(results, vec!["web-1", "web-2"]);
    }

    #[test]
    fn test_autocomplete_no_match() {
        let idx = build_index();
        let results = idx.autocomplete("host", "xyz");
        assert!(results.is_empty());
    }

    #[test]
    fn test_autocomplete_empty_prefix() {
        let idx = build_index();
        let results = idx.autocomplete("region", "");
        assert_eq!(results.len(), 2); // all values
    }

    // ── Equality filter ──────────────────────────────────────────────────

    #[test]
    fn test_filter_equals() {
        let mut idx = build_index();
        let filter = TagFilter::Equals {
            key: "env".to_string(),
            value: "prod".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert_eq!(result.len(), 2);
        assert!(result.contains("s1"));
        assert!(result.contains("s3"));
    }

    #[test]
    fn test_filter_equals_no_match() {
        let mut idx = build_index();
        let filter = TagFilter::Equals {
            key: "env".to_string(),
            value: "test".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert!(result.is_empty());
    }

    // ── Prefix filter ────────────────────────────────────────────────────

    #[test]
    fn test_filter_prefix() {
        let mut idx = build_index();
        let filter = TagFilter::Prefix {
            key: "host".to_string(),
            prefix: "db".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert_eq!(result.len(), 2);
        assert!(result.contains("s3"));
        assert!(result.contains("s4"));
    }

    #[test]
    fn test_filter_prefix_full_match() {
        let mut idx = build_index();
        let filter = TagFilter::Prefix {
            key: "host".to_string(),
            prefix: "web-1".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert_eq!(result.len(), 1);
        assert!(result.contains("s1"));
    }

    // ── Regex filter ─────────────────────────────────────────────────────

    #[test]
    fn test_filter_regex_substring() {
        let mut idx = build_index();
        let filter = TagFilter::Regex {
            key: "region".to_string(),
            pattern: "east".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert_eq!(result.len(), 2);
        assert!(result.contains("s1"));
        assert!(result.contains("s2"));
    }

    // ── NOT filter ───────────────────────────────────────────────────────

    #[test]
    fn test_filter_not() {
        let mut idx = build_index();
        let filter = TagFilter::Not {
            key: "env".to_string(),
            value: "prod".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert_eq!(result.len(), 2); // s2(staging), s4(dev)
        assert!(result.contains("s2"));
        assert!(result.contains("s4"));
    }

    // ── Wildcard filter ──────────────────────────────────────────────────

    #[test]
    fn test_filter_wildcard_star() {
        let mut idx = build_index();
        let filter = TagFilter::Wildcard {
            key: "host".to_string(),
            pattern: "web-*".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert_eq!(result.len(), 2);
        assert!(result.contains("s1"));
        assert!(result.contains("s2"));
    }

    #[test]
    fn test_filter_wildcard_question() {
        let mut idx = build_index();
        let filter = TagFilter::Wildcard {
            key: "host".to_string(),
            pattern: "db-?".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert_eq!(result.len(), 2);
        assert!(result.contains("s3"));
        assert!(result.contains("s4"));
    }

    #[test]
    fn test_filter_wildcard_mixed() {
        let mut idx = build_index();
        let filter = TagFilter::Wildcard {
            key: "region".to_string(),
            pattern: "*-*".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert_eq!(result.len(), 4); // all regions match x-y
    }

    // ── Composite filters ────────────────────────────────────────────────

    #[test]
    fn test_composite_and() {
        let mut idx = build_index();
        let filter = CompositeFilter::And(vec![
            CompositeFilter::Single(TagFilter::Equals {
                key: "region".to_string(),
                value: "us-east".to_string(),
            }),
            CompositeFilter::Single(TagFilter::Equals {
                key: "env".to_string(),
                value: "prod".to_string(),
            }),
        ]);
        let result = idx.evaluate(&filter);
        assert_eq!(result.len(), 1);
        assert!(result.contains("s1"));
    }

    #[test]
    fn test_composite_or() {
        let mut idx = build_index();
        let filter = CompositeFilter::Or(vec![
            CompositeFilter::Single(TagFilter::Equals {
                key: "host".to_string(),
                value: "web-1".to_string(),
            }),
            CompositeFilter::Single(TagFilter::Equals {
                key: "host".to_string(),
                value: "db-1".to_string(),
            }),
        ]);
        let result = idx.evaluate(&filter);
        assert_eq!(result.len(), 2);
        assert!(result.contains("s1"));
        assert!(result.contains("s3"));
    }

    #[test]
    fn test_composite_nested() {
        let mut idx = build_index();
        // (region=us-east OR region=eu-west) AND env=prod
        let filter = CompositeFilter::And(vec![
            CompositeFilter::Or(vec![
                CompositeFilter::Single(TagFilter::Equals {
                    key: "region".to_string(),
                    value: "us-east".to_string(),
                }),
                CompositeFilter::Single(TagFilter::Equals {
                    key: "region".to_string(),
                    value: "eu-west".to_string(),
                }),
            ]),
            CompositeFilter::Single(TagFilter::Equals {
                key: "env".to_string(),
                value: "prod".to_string(),
            }),
        ]);
        let result = idx.evaluate(&filter);
        assert_eq!(result.len(), 2); // s1 and s3
    }

    #[test]
    fn test_composite_single_wrapping() {
        let mut idx = build_index();
        let filter = CompositeFilter::Single(TagFilter::Equals {
            key: "env".to_string(),
            value: "dev".to_string(),
        });
        let result = idx.evaluate(&filter);
        assert_eq!(result.len(), 1);
        assert!(result.contains("s4"));
    }

    // ── Series with key ──────────────────────────────────────────────────

    #[test]
    fn test_series_with_key() {
        let idx = build_index();
        let result = idx.series_with_key("region");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_series_with_unknown_key() {
        let idx = build_index();
        assert!(idx.series_with_key("nonexistent").is_empty());
    }

    // ── Tag statistics ───────────────────────────────────────────────────

    #[test]
    fn test_tag_stats() {
        let idx = build_index();
        let stats = idx.tag_stats();
        assert_eq!(stats.total_keys, 3);
        assert_eq!(stats.total_series, 4);
        assert!(stats.total_pairs > 0);
    }

    #[test]
    fn test_tag_stats_cardinality() {
        let idx = build_index();
        let stats = idx.tag_stats();
        assert_eq!(stats.cardinality.get("host").copied().unwrap_or(0), 4);
    }

    #[test]
    fn test_tag_stats_most_common() {
        let idx = build_index();
        let stats = idx.tag_stats();
        // All 3 keys are used by all 4 series
        assert!(!stats.most_common_keys.is_empty());
    }

    // ── Index rebuild ────────────────────────────────────────────────────

    #[test]
    fn test_rebuild_index() {
        let mut idx = build_index();
        idx.rebuild();
        // Should still return same results
        let filter = TagFilter::Equals {
            key: "env".to_string(),
            value: "prod".to_string(),
        };
        let result = idx.evaluate_filter(&filter);
        assert_eq!(result.len(), 2);
        assert_eq!(idx.index_stats().rebuilds, 1);
    }

    // ── Index stats ──────────────────────────────────────────────────────

    #[test]
    fn test_index_stats_adds() {
        let idx = build_index();
        assert_eq!(idx.index_stats().adds, 4);
    }

    #[test]
    fn test_index_stats_removes() {
        let mut idx = build_index();
        idx.remove_series("s1");
        assert_eq!(idx.index_stats().removes, 1);
    }

    #[test]
    fn test_index_stats_queries() {
        let mut idx = build_index();
        let filter = TagFilter::Equals {
            key: "env".to_string(),
            value: "prod".to_string(),
        };
        idx.evaluate_filter(&filter);
        idx.evaluate_filter(&filter);
        assert_eq!(idx.index_stats().queries, 2);
    }

    // ── Wildcard matching helper ─────────────────────────────────────────

    #[test]
    fn test_wildcard_exact() {
        assert!(wildcard_matches("hello", "hello"));
        assert!(!wildcard_matches("hello", "world"));
    }

    #[test]
    fn test_wildcard_star_only() {
        assert!(wildcard_matches("*", "anything"));
        assert!(wildcard_matches("*", ""));
    }

    #[test]
    fn test_wildcard_question_mark() {
        assert!(wildcard_matches("h?llo", "hello"));
        assert!(!wildcard_matches("h?llo", "heello"));
    }

    #[test]
    fn test_wildcard_complex() {
        assert!(wildcard_matches("*.example.*", "www.example.com"));
        assert!(!wildcard_matches("*.example.*", "www.other.com"));
    }

    #[test]
    fn test_wildcard_empty_pattern() {
        assert!(wildcard_matches("", ""));
        assert!(!wildcard_matches("", "x"));
    }

    // ── Default ──────────────────────────────────────────────────────────

    #[test]
    fn test_default_index() {
        let idx = TagIndex::default();
        assert_eq!(idx.series_count(), 0);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_add_series_no_tags() {
        let mut idx = TagIndex::new();
        idx.add_series("empty", &[]);
        assert!(idx.contains_series("empty"));
        assert!(idx.tags_for_series("empty").is_empty());
    }

    #[test]
    fn test_duplicate_add_same_tags() {
        let mut idx = TagIndex::new();
        idx.add_series("s1", &[("host", "web-1")]);
        idx.add_series("s1", &[("host", "web-1")]);
        assert_eq!(idx.series_count(), 1);
        // Should have one tag pair
        let tags = idx.tags_for_series("s1");
        assert_eq!(tags.len(), 1);
    }

    #[test]
    fn test_add_series_with_same_key_different_values() {
        let mut idx = TagIndex::new();
        idx.add_series("s1", &[("tag", "val1")]);
        idx.add_series("s2", &[("tag", "val2")]);
        assert_eq!(idx.cardinality("tag"), 2);
    }
}
