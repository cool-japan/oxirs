//! Adaptive Result Merger for Federated SPARQL Queries
//!
//! This module implements `AdaptiveResultMerger`, which combines result sets
//! from multiple SPARQL endpoints using the most appropriate merge strategy.
//!
//! Three strategies are available:
//! - **SortMerge** – assumes each input is already sorted; merges in O(n log k).
//! - **HashMerge** – builds a hash map for deduplication; best for DISTINCT.
//! - **StreamMerge** – emits rows lazily, best for large result sets with LIMIT.
//!
//! The merger auto-selects a strategy based on result set characteristics or
//! the caller can override the strategy explicitly.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ─── Value ────────────────────────────────────────────────────────────────────

/// A single cell value in a SPARQL result row.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResultValue {
    /// An IRI (named node).
    Iri(String),
    /// A plain or lang-tagged literal.
    Literal(String),
    /// A blank node identifier.
    BlankNode(String),
    /// Unbound / null.
    Unbound,
}

impl ResultValue {
    /// Return a canonical string representation for ordering.
    pub fn as_sort_key(&self) -> &str {
        match self {
            ResultValue::Iri(s) => s.as_str(),
            ResultValue::Literal(s) => s.as_str(),
            ResultValue::BlankNode(s) => s.as_str(),
            ResultValue::Unbound => "",
        }
    }
}

impl PartialOrd for ResultValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ResultValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_sort_key().cmp(other.as_sort_key())
    }
}

// ─── ResultRow ────────────────────────────────────────────────────────────────

/// A single SPARQL result row: variable name → value.
pub type ResultRow = HashMap<String, ResultValue>;

// ─── OrderBySpec ─────────────────────────────────────────────────────────────

/// A single ORDER BY key with direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderByKey {
    /// Variable name to sort by.
    pub variable: String,
    /// Whether the ordering is descending.
    pub descending: bool,
}

// ─── MergeStrategy ───────────────────────────────────────────────────────────

/// Available merge strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Sort-merge: assumes inputs are pre-sorted; memory-efficient.
    SortMerge,
    /// Hash-merge: hash-based deduplication; best for DISTINCT.
    HashMerge,
    /// Streaming-merge: emits lazily, ideal with LIMIT.
    StreamMerge,
    /// Automatically choose based on characteristics.
    Auto,
}

// ─── MergeConfig ─────────────────────────────────────────────────────────────

/// Configuration for the adaptive merger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Override strategy (use Auto to let the merger decide).
    pub strategy: MergeStrategy,
    /// ORDER BY specification; empty means no ordering.
    pub order_by: Vec<OrderByKey>,
    /// Whether to apply DISTINCT deduplication.
    pub distinct: bool,
    /// Maximum number of rows to emit (LIMIT); 0 = unlimited.
    pub limit: usize,
    /// Row offset (OFFSET); rows before this index are skipped.
    pub offset: usize,
    /// Threshold (total row count) above which StreamMerge is auto-selected.
    pub stream_threshold: usize,
    /// Threshold (total row count) above which HashMerge beats SortMerge for DISTINCT.
    pub hash_distinct_threshold: usize,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Auto,
            order_by: Vec::new(),
            distinct: false,
            limit: 0,
            offset: 0,
            stream_threshold: 100_000,
            hash_distinct_threshold: 10_000,
        }
    }
}

// ─── MergeStats ──────────────────────────────────────────────────────────────

/// Statistics produced after a merge operation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MergeStats {
    /// Strategy that was actually used.
    pub strategy_used: String,
    /// Total input rows across all sources.
    pub total_input_rows: usize,
    /// Rows deduplicated (if DISTINCT was applied).
    pub duplicates_removed: usize,
    /// Rows in final output.
    pub output_rows: usize,
    /// Number of source result sets merged.
    pub source_count: usize,
}

// ─── AdaptiveResultMerger ────────────────────────────────────────────────────

/// Adaptive result merger that selects the optimal merge strategy at runtime.
pub struct AdaptiveResultMerger {
    config: MergeConfig,
}

impl AdaptiveResultMerger {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: MergeConfig::default(),
        }
    }

    /// Create with a custom configuration.
    pub fn with_config(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Merge a collection of result sets from different endpoints.
    ///
    /// Returns `(merged_rows, stats)`.
    pub fn merge(&self, sources: Vec<Vec<ResultRow>>) -> Result<(Vec<ResultRow>, MergeStats)> {
        if sources.is_empty() {
            return Ok((Vec::new(), MergeStats::default()));
        }

        let total_input_rows: usize = sources.iter().map(|s| s.len()).sum();
        let strategy = self.select_strategy(total_input_rows);

        let (rows, strategy_name) = match strategy {
            MergeStrategy::SortMerge => {
                let result = self.sort_merge(sources)?;
                (result, "SortMerge")
            }
            MergeStrategy::HashMerge => {
                let result = self.hash_merge(sources)?;
                (result, "HashMerge")
            }
            MergeStrategy::StreamMerge => {
                let result = self.stream_merge(sources)?;
                (result, "StreamMerge")
            }
            MergeStrategy::Auto => unreachable!("Auto is resolved before this point"),
        };

        let duplicates_removed = total_input_rows - rows.len();
        let output_rows = rows.len();

        let stats = MergeStats {
            strategy_used: strategy_name.to_string(),
            total_input_rows,
            duplicates_removed,
            output_rows,
            source_count: 0, // filled below
        };

        Ok((
            rows,
            MergeStats {
                source_count: stats.source_count,
                ..stats
            },
        ))
    }

    /// Merge with source count tracking.
    pub fn merge_from_sources(
        &self,
        sources: Vec<Vec<ResultRow>>,
    ) -> Result<(Vec<ResultRow>, MergeStats)> {
        if sources.is_empty() {
            return Ok((Vec::new(), MergeStats::default()));
        }

        let source_count = sources.len();
        let total_input_rows: usize = sources.iter().map(|s| s.len()).sum();
        let strategy = self.select_strategy(total_input_rows);

        let (rows, strategy_name) = match strategy {
            MergeStrategy::SortMerge => {
                let result = self.sort_merge(sources)?;
                (result, "SortMerge".to_string())
            }
            MergeStrategy::HashMerge => {
                let result = self.hash_merge(sources)?;
                (result, "HashMerge".to_string())
            }
            MergeStrategy::StreamMerge => {
                let result = self.stream_merge(sources)?;
                (result, "StreamMerge".to_string())
            }
            MergeStrategy::Auto => unreachable!(),
        };

        let duplicates_removed = total_input_rows.saturating_sub(rows.len());
        let output_rows = rows.len();

        Ok((
            rows,
            MergeStats {
                strategy_used: strategy_name,
                total_input_rows,
                duplicates_removed,
                output_rows,
                source_count,
            },
        ))
    }

    // ─── Strategy selection ───────────────────────────────────────────────────

    fn select_strategy(&self, total_rows: usize) -> MergeStrategy {
        match self.config.strategy {
            MergeStrategy::Auto => {
                if total_rows >= self.config.stream_threshold {
                    MergeStrategy::StreamMerge
                } else if self.config.distinct && total_rows >= self.config.hash_distinct_threshold
                {
                    MergeStrategy::HashMerge
                } else if !self.config.order_by.is_empty() {
                    MergeStrategy::SortMerge
                } else {
                    MergeStrategy::HashMerge
                }
            }
            s => s,
        }
    }

    // ─── Sort-merge ───────────────────────────────────────────────────────────

    /// Sort-merge: sorts each source then does a k-way merge.
    fn sort_merge(&self, mut sources: Vec<Vec<ResultRow>>) -> Result<Vec<ResultRow>> {
        // Sort each source by the ORDER BY keys.
        for source in &mut sources {
            self.sort_rows(source);
        }

        // k-way merge using a min-heap of (sort_key, source_index, row).
        // When a row is popped we advance the corresponding source iterator.
        struct HeapEntry {
            sort_key: Vec<(String, bool)>, // (sort_value, descending)
            source_idx: usize,
            row: ResultRow,
        }

        impl PartialEq for HeapEntry {
            fn eq(&self, other: &Self) -> bool {
                self.sort_key == other.sort_key
            }
        }
        impl Eq for HeapEntry {}

        impl PartialOrd for HeapEntry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        // BinaryHeap is a max-heap; negate ordering to get min-heap.
        impl Ord for HeapEntry {
            fn cmp(&self, other: &Self) -> Ordering {
                other.sort_key.cmp(&self.sort_key)
            }
        }

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
        let mut iters: Vec<std::vec::IntoIter<ResultRow>> =
            sources.into_iter().map(|s| s.into_iter()).collect();

        // Seed the heap with the first row from each source.
        for (idx, iter) in iters.iter_mut().enumerate() {
            if let Some(row) = iter.next() {
                let sort_key = self.row_sort_key(&row);
                heap.push(HeapEntry {
                    sort_key,
                    source_idx: idx,
                    row,
                });
            }
        }

        let mut result = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        while let Some(entry) = heap.pop() {
            // Advance the source that produced this entry.
            if let Some(next_row) = iters[entry.source_idx].next() {
                let sort_key = self.row_sort_key(&next_row);
                heap.push(HeapEntry {
                    sort_key,
                    source_idx: entry.source_idx,
                    row: next_row,
                });
            }

            let row = entry.row;
            if self.config.distinct {
                let key = row_hash_key(&row);
                if seen.contains(&key) {
                    continue;
                }
                seen.insert(key);
            }
            result.push(row);
        }

        self.apply_offset_limit(result)
    }

    // ─── Hash-merge ───────────────────────────────────────────────────────────

    /// Hash-merge: combines all sources into a hash map for deduplication.
    fn hash_merge(&self, sources: Vec<Vec<ResultRow>>) -> Result<Vec<ResultRow>> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut result: Vec<ResultRow> = Vec::new();

        for source in sources {
            for row in source {
                if self.config.distinct {
                    let key = row_hash_key(&row);
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.insert(key);
                }
                result.push(row);
            }
        }

        if !self.config.order_by.is_empty() {
            self.sort_rows(&mut result);
        }

        self.apply_offset_limit(result)
    }

    // ─── Stream-merge ─────────────────────────────────────────────────────────

    /// Stream-merge: emit rows lazily up to LIMIT, skipping OFFSET.
    fn stream_merge(&self, sources: Vec<Vec<ResultRow>>) -> Result<Vec<ResultRow>> {
        let mut result: Vec<ResultRow> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let limit = if self.config.limit > 0 {
            self.config.limit + self.config.offset
        } else {
            usize::MAX
        };

        'outer: for source in sources {
            for row in source {
                if self.config.distinct {
                    let key = row_hash_key(&row);
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.insert(key);
                }
                result.push(row);
                if result.len() >= limit {
                    break 'outer;
                }
            }
        }

        if !self.config.order_by.is_empty() {
            self.sort_rows(&mut result);
        }

        self.apply_offset_limit(result)
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    fn sort_rows(&self, rows: &mut [ResultRow]) {
        let order_by = &self.config.order_by;
        if order_by.is_empty() {
            return;
        }
        rows.sort_by(|a, b| {
            for key in order_by {
                let av = a.get(&key.variable).map(|v| v.as_sort_key()).unwrap_or("");
                let bv = b.get(&key.variable).map(|v| v.as_sort_key()).unwrap_or("");
                let cmp = av.cmp(bv);
                if cmp != Ordering::Equal {
                    return if key.descending { cmp.reverse() } else { cmp };
                }
            }
            Ordering::Equal
        });
    }

    fn row_sort_key(&self, row: &ResultRow) -> Vec<(String, bool)> {
        self.config
            .order_by
            .iter()
            .map(|k| {
                let val = row
                    .get(&k.variable)
                    .map(|v| v.as_sort_key().to_owned())
                    .unwrap_or_default();
                (val, k.descending)
            })
            .collect()
    }

    fn apply_offset_limit(&self, rows: Vec<ResultRow>) -> Result<Vec<ResultRow>> {
        let offset = self.config.offset;
        let limit = self.config.limit;

        if offset == 0 && limit == 0 {
            return Ok(rows);
        }

        let sliced = rows.into_iter().skip(offset);
        if limit > 0 {
            Ok(sliced.take(limit).collect())
        } else {
            Ok(sliced.collect())
        }
    }
}

impl Default for AdaptiveResultMerger {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a canonical hash key for a result row (for DISTINCT).
fn row_hash_key(row: &ResultRow) -> String {
    let mut pairs: Vec<(&String, &ResultValue)> = row.iter().collect();
    pairs.sort_by_key(|(k, _)| k.as_str());
    pairs
        .into_iter()
        .map(|(k, v)| format!("{k}={}", v.as_sort_key()))
        .collect::<Vec<_>>()
        .join("|")
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(pairs: &[(&str, &str)]) -> ResultRow {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), ResultValue::Literal(v.to_string())))
            .collect()
    }

    fn iri_row(var: &str, iri: &str) -> ResultRow {
        let mut m = HashMap::new();
        m.insert(var.to_string(), ResultValue::Iri(iri.to_string()));
        m
    }

    // ── Strategy tests ────────────────────────────────────────────────────────

    #[test]
    fn test_auto_selects_sort_merge_for_ordered_small() {
        let config = MergeConfig {
            strategy: MergeStrategy::Auto,
            order_by: vec![OrderByKey {
                variable: "x".to_string(),
                descending: false,
            }],
            stream_threshold: 100_000,
            hash_distinct_threshold: 10_000,
            ..Default::default()
        };
        let merger = AdaptiveResultMerger::with_config(config);
        let strategy = merger.select_strategy(50);
        assert_eq!(strategy, MergeStrategy::SortMerge);
    }

    #[test]
    fn test_auto_selects_stream_merge_for_large() {
        let config = MergeConfig {
            strategy: MergeStrategy::Auto,
            stream_threshold: 100,
            ..Default::default()
        };
        let merger = AdaptiveResultMerger::with_config(config);
        let strategy = merger.select_strategy(200);
        assert_eq!(strategy, MergeStrategy::StreamMerge);
    }

    #[test]
    fn test_auto_selects_hash_merge_for_distinct_medium() {
        let config = MergeConfig {
            strategy: MergeStrategy::Auto,
            distinct: true,
            hash_distinct_threshold: 50,
            stream_threshold: 100_000,
            ..Default::default()
        };
        let merger = AdaptiveResultMerger::with_config(config);
        let strategy = merger.select_strategy(100);
        assert_eq!(strategy, MergeStrategy::HashMerge);
    }

    // ── Hash-merge tests ──────────────────────────────────────────────────────

    #[test]
    fn test_hash_merge_combines_sources() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::HashMerge,
            ..Default::default()
        });
        let src1 = vec![make_row(&[("x", "a")])];
        let src2 = vec![make_row(&[("x", "b")])];
        let (rows, stats) = merger
            .merge_from_sources(vec![src1, src2])
            .expect("merge failed");
        assert_eq!(rows.len(), 2);
        assert_eq!(stats.total_input_rows, 2);
        assert_eq!(stats.source_count, 2);
    }

    #[test]
    fn test_hash_merge_distinct_removes_duplicates() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::HashMerge,
            distinct: true,
            ..Default::default()
        });
        let row = make_row(&[("x", "a")]);
        let (rows, stats) = merger
            .merge_from_sources(vec![vec![row.clone()], vec![row]])
            .expect("merge failed");
        assert_eq!(rows.len(), 1);
        assert_eq!(stats.duplicates_removed, 1);
    }

    #[test]
    fn test_hash_merge_with_limit() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::HashMerge,
            limit: 2,
            ..Default::default()
        });
        let src: Vec<ResultRow> = (0..10)
            .map(|i| make_row(&[("x", &format!("v{i}"))]))
            .collect();
        let (rows, _) = merger.merge_from_sources(vec![src]).expect("merge failed");
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_hash_merge_with_offset() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::HashMerge,
            offset: 3,
            ..Default::default()
        });
        let src: Vec<ResultRow> = (0..5)
            .map(|i| make_row(&[("x", &format!("v{i}"))]))
            .collect();
        let (rows, _) = merger.merge_from_sources(vec![src]).expect("merge failed");
        assert_eq!(rows.len(), 2);
    }

    // ── Sort-merge tests ──────────────────────────────────────────────────────

    #[test]
    fn test_sort_merge_orders_by_variable_asc() {
        let config = MergeConfig {
            strategy: MergeStrategy::SortMerge,
            order_by: vec![OrderByKey {
                variable: "x".to_string(),
                descending: false,
            }],
            ..Default::default()
        };
        let merger = AdaptiveResultMerger::with_config(config);
        let src1 = vec![make_row(&[("x", "c")]), make_row(&[("x", "a")])];
        let src2 = vec![make_row(&[("x", "b")])];
        let (rows, _) = merger
            .merge_from_sources(vec![src1, src2])
            .expect("merge failed");
        let vals: Vec<&str> = rows.iter().map(|r| r["x"].as_sort_key()).collect();
        assert_eq!(vals, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_sort_merge_orders_by_variable_desc() {
        let config = MergeConfig {
            strategy: MergeStrategy::SortMerge,
            order_by: vec![OrderByKey {
                variable: "x".to_string(),
                descending: true,
            }],
            ..Default::default()
        };
        let merger = AdaptiveResultMerger::with_config(config);
        let src = vec![
            make_row(&[("x", "a")]),
            make_row(&[("x", "c")]),
            make_row(&[("x", "b")]),
        ];
        let (rows, _) = merger.merge_from_sources(vec![src]).expect("merge failed");
        let vals: Vec<&str> = rows.iter().map(|r| r["x"].as_sort_key()).collect();
        assert_eq!(vals, vec!["c", "b", "a"]);
    }

    #[test]
    fn test_sort_merge_distinct() {
        let config = MergeConfig {
            strategy: MergeStrategy::SortMerge,
            distinct: true,
            order_by: vec![OrderByKey {
                variable: "x".to_string(),
                descending: false,
            }],
            ..Default::default()
        };
        let merger = AdaptiveResultMerger::with_config(config);
        let row_a = make_row(&[("x", "a")]);
        let (rows, _) = merger
            .merge_from_sources(vec![vec![row_a.clone()], vec![row_a]])
            .expect("merge failed");
        assert_eq!(rows.len(), 1);
    }

    // ── Stream-merge tests ────────────────────────────────────────────────────

    #[test]
    fn test_stream_merge_respects_limit() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::StreamMerge,
            limit: 3,
            ..Default::default()
        });
        let src: Vec<ResultRow> = (0..100)
            .map(|i| make_row(&[("x", &format!("v{i}"))]))
            .collect();
        let (rows, _) = merger.merge_from_sources(vec![src]).expect("merge failed");
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_stream_merge_distinct() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::StreamMerge,
            distinct: true,
            ..Default::default()
        });
        let row = make_row(&[("x", "dup")]);
        let (rows, _) = merger
            .merge_from_sources(vec![vec![row.clone(); 5], vec![row.clone(); 5]])
            .expect("merge failed");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_stream_merge_with_order_by() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::StreamMerge,
            order_by: vec![OrderByKey {
                variable: "x".to_string(),
                descending: false,
            }],
            ..Default::default()
        });
        let src = vec![
            make_row(&[("x", "z")]),
            make_row(&[("x", "a")]),
            make_row(&[("x", "m")]),
        ];
        let (rows, _) = merger.merge_from_sources(vec![src]).expect("merge failed");
        assert_eq!(rows[0]["x"].as_sort_key(), "a");
    }

    // ── Empty sources ─────────────────────────────────────────────────────────

    #[test]
    fn test_empty_sources_returns_empty() {
        let merger = AdaptiveResultMerger::new();
        let (rows, stats) = merger.merge_from_sources(vec![]).expect("merge failed");
        assert!(rows.is_empty());
        assert_eq!(stats.total_input_rows, 0);
    }

    #[test]
    fn test_single_empty_source() {
        let merger = AdaptiveResultMerger::new();
        let (rows, _) = merger
            .merge_from_sources(vec![vec![]])
            .expect("merge failed");
        assert!(rows.is_empty());
    }

    // ── ResultValue ordering ──────────────────────────────────────────────────

    #[test]
    fn test_result_value_ordering() {
        let a = ResultValue::Literal("alpha".to_string());
        let b = ResultValue::Literal("beta".to_string());
        assert!(a < b);
    }

    #[test]
    fn test_result_value_iri_sort_key() {
        let v = ResultValue::Iri("http://example.org/".to_string());
        assert_eq!(v.as_sort_key(), "http://example.org/");
    }

    #[test]
    fn test_result_value_unbound_sort_key() {
        let v = ResultValue::Unbound;
        assert_eq!(v.as_sort_key(), "");
    }

    // ── Merge stats ───────────────────────────────────────────────────────────

    #[test]
    fn test_merge_stats_source_count() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::HashMerge,
            ..Default::default()
        });
        let sources: Vec<Vec<ResultRow>> = (0..4)
            .map(|i| vec![make_row(&[("x", &format!("v{i}"))])])
            .collect();
        let (_, stats) = merger.merge_from_sources(sources).expect("merge failed");
        assert_eq!(stats.source_count, 4);
    }

    #[test]
    fn test_merge_stats_strategy_name() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::SortMerge,
            order_by: vec![OrderByKey {
                variable: "x".to_string(),
                descending: false,
            }],
            ..Default::default()
        });
        let (_, stats) = merger
            .merge_from_sources(vec![vec![make_row(&[("x", "a")])]])
            .expect("merge failed");
        assert_eq!(stats.strategy_used, "SortMerge");
    }

    // ── IRI rows ──────────────────────────────────────────────────────────────

    #[test]
    fn test_iri_values_merge() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::HashMerge,
            ..Default::default()
        });
        let rows = vec![
            iri_row("s", "http://example.org/A"),
            iri_row("s", "http://example.org/B"),
        ];
        let (result, _) = merger.merge_from_sources(vec![rows]).expect("merge failed");
        assert_eq!(result.len(), 2);
    }

    // ── Multi-variable ordering ───────────────────────────────────────────────

    #[test]
    fn test_multi_key_sort() {
        let config = MergeConfig {
            strategy: MergeStrategy::SortMerge,
            order_by: vec![
                OrderByKey {
                    variable: "x".to_string(),
                    descending: false,
                },
                OrderByKey {
                    variable: "y".to_string(),
                    descending: true,
                },
            ],
            ..Default::default()
        };
        let merger = AdaptiveResultMerger::with_config(config);
        let src = vec![
            make_row(&[("x", "a"), ("y", "1")]),
            make_row(&[("x", "a"), ("y", "3")]),
            make_row(&[("x", "b"), ("y", "2")]),
        ];
        let (rows, _) = merger.merge_from_sources(vec![src]).expect("merge failed");
        // x is ASC; for x="a" y is DESC → "3" before "1"
        assert_eq!(rows[0]["y"].as_sort_key(), "3");
        assert_eq!(rows[1]["y"].as_sort_key(), "1");
        assert_eq!(rows[2]["x"].as_sort_key(), "b");
    }

    #[test]
    fn test_default_merger_hash_merge_no_order() {
        let merger = AdaptiveResultMerger::new();
        let strategy = merger.select_strategy(100);
        // Default Auto with no order_by and not distinct: falls through to HashMerge
        assert_eq!(strategy, MergeStrategy::HashMerge);
    }

    #[test]
    fn test_offset_limit_combined() {
        let merger = AdaptiveResultMerger::with_config(MergeConfig {
            strategy: MergeStrategy::HashMerge,
            offset: 2,
            limit: 3,
            ..Default::default()
        });
        let src: Vec<ResultRow> = (0..10_usize)
            .map(|i| make_row(&[("x", &format!("v{i:02}"))]))
            .collect();
        let (rows, _) = merger.merge_from_sources(vec![src]).expect("merge failed");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["x"].as_sort_key(), "v02");
        assert_eq!(rows[2]["x"].as_sort_key(), "v04");
    }
}
