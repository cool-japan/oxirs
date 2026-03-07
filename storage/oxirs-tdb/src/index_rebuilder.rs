//! B-tree index rebuild from raw triple data (v1.1.0 round 14).
//!
//! Provides utilities to reconstruct sorted triple-store indexes from raw
//! `TripleRecord` data.  The six supported index orders (SPO, POS, OSP,
//! GSPO, GPOS, GOSP) mirror those used by Apache Jena TDB2.

use std::collections::HashMap;
use std::hash::Hash;
use std::time::Instant;

/// A single RDF triple, optionally bound to a named graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TripleRecord {
    /// Subject IRI or blank-node identifier.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object IRI, blank-node identifier, or literal value.
    pub object: String,
    /// Named graph IRI, or `None` for the default graph.
    pub graph: Option<String>,
}

impl TripleRecord {
    /// Create a triple in the default graph.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph: None,
        }
    }

    /// Create a triple in a named graph.
    pub fn in_graph(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph: Some(graph.into()),
        }
    }
}

/// The ordering of fields within a triple-store index.
///
/// Triple-only orders apply regardless of graph.
/// Graph-qualified orders incorporate the named graph as the first key field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexOrder {
    /// Subject → Predicate → Object
    SPO,
    /// Predicate → Object → Subject
    POS,
    /// Object → Subject → Predicate
    OSP,
    /// Graph → Subject → Predicate → Object
    GSPO,
    /// Graph → Predicate → Object → Subject
    GPOS,
    /// Graph → Object → Subject → Predicate
    GOSP,
}

impl IndexOrder {
    /// All defined index orders.
    pub fn all() -> &'static [IndexOrder] {
        &[
            IndexOrder::SPO,
            IndexOrder::POS,
            IndexOrder::OSP,
            IndexOrder::GSPO,
            IndexOrder::GPOS,
            IndexOrder::GOSP,
        ]
    }

    /// Build the sort key for a given [`TripleRecord`] under this order.
    ///
    /// Graph-qualified orders prepend the graph IRI (or `""` for the default
    /// graph) as the first key component.
    pub fn key_for(&self, rec: &TripleRecord) -> Vec<String> {
        let g = rec.graph.as_deref().unwrap_or("").to_string();
        match self {
            IndexOrder::SPO => vec![
                rec.subject.clone(),
                rec.predicate.clone(),
                rec.object.clone(),
            ],
            IndexOrder::POS => vec![
                rec.predicate.clone(),
                rec.object.clone(),
                rec.subject.clone(),
            ],
            IndexOrder::OSP => vec![
                rec.object.clone(),
                rec.subject.clone(),
                rec.predicate.clone(),
            ],
            IndexOrder::GSPO => vec![
                g,
                rec.subject.clone(),
                rec.predicate.clone(),
                rec.object.clone(),
            ],
            IndexOrder::GPOS => vec![
                g,
                rec.predicate.clone(),
                rec.object.clone(),
                rec.subject.clone(),
            ],
            IndexOrder::GOSP => vec![
                g,
                rec.object.clone(),
                rec.subject.clone(),
                rec.predicate.clone(),
            ],
        }
    }
}

/// A single entry in a sorted index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    /// The sort key for this entry under the chosen [`IndexOrder`].
    pub key: Vec<String>,
    /// Byte offset of the corresponding triple in the raw data file.
    pub offset: u64,
}

/// Statistics collected during an index rebuild.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RebuildStats {
    /// Total number of [`TripleRecord`] instances processed.
    pub records_processed: usize,
    /// Number of [`IndexEntry`] values written to the output.
    pub entries_written: usize,
    /// Wall-clock duration of the rebuild in milliseconds.
    pub duration_ms: u64,
    /// Number of records that produced errors (e.g. missing graph for
    /// graph-qualified orders — currently always 0, reserved for future use).
    pub errors: usize,
}

/// Rebuilds one or more B-tree indexes from raw triple data.
///
/// # Example
/// ```rust
/// use oxirs_tdb::index_rebuilder::{IndexRebuilder, IndexOrder, TripleRecord};
///
/// let mut rebuilder = IndexRebuilder::new(IndexOrder::SPO);
/// rebuilder.add_triple(TripleRecord::new("s", "p", "o"));
/// let entries = rebuilder.build();
/// assert_eq!(entries.len(), 1);
/// ```
#[derive(Debug)]
pub struct IndexRebuilder {
    order: IndexOrder,
    records: Vec<TripleRecord>,
}

impl IndexRebuilder {
    /// Create a new rebuilder for the specified index order.
    pub fn new(order: IndexOrder) -> Self {
        Self {
            order,
            records: Vec::new(),
        }
    }

    /// Append a single triple record.
    pub fn add_triple(&mut self, rec: TripleRecord) {
        self.records.push(rec);
    }

    /// Sort all accumulated records according to [`IndexOrder`] and produce
    /// the corresponding [`IndexEntry`] list.
    ///
    /// Entries are assigned monotonically increasing offsets (0, 1, 2, …),
    /// which can be replaced with real byte offsets by the caller.
    pub fn build(&self) -> Vec<IndexEntry> {
        let mut keyed: Vec<(Vec<String>, u64)> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, rec)| (self.order.key_for(rec), i as u64))
            .collect();

        keyed.sort_by(|(a, _), (b, _)| a.cmp(b));

        keyed
            .into_iter()
            .map(|(key, offset)| IndexEntry { key, offset })
            .collect()
    }

    /// Build sorted index entries for **all six** index orders from a slice of
    /// records.
    pub fn build_all_orders(records: &[TripleRecord]) -> HashMap<IndexOrder, Vec<IndexEntry>> {
        let mut result = HashMap::new();
        for &order in IndexOrder::all() {
            let mut rb = IndexRebuilder::new(order);
            for rec in records {
                rb.add_triple(rec.clone());
            }
            result.insert(order, rb.build());
        }
        result
    }

    /// Consume a `Vec<TripleRecord>`, rebuild the index, and return timing /
    /// count statistics alongside the entries.
    pub fn rebuild_with_stats(records: Vec<TripleRecord>) -> (Vec<IndexEntry>, RebuildStats) {
        let start = Instant::now();
        let records_processed = records.len();
        let mut rb = IndexRebuilder::new(IndexOrder::SPO);
        for rec in records {
            rb.add_triple(rec);
        }
        let entries = rb.build();
        let entries_written = entries.len();
        let duration_ms = start.elapsed().as_millis() as u64;

        let stats = RebuildStats {
            records_processed,
            entries_written,
            duration_ms,
            errors: 0,
        };
        (entries, stats)
    }

    /// Verify that a slice of [`IndexEntry`] values is strictly sorted by key.
    ///
    /// Returns `true` if the entries are in non-descending order (duplicates
    /// are allowed).
    pub fn verify(entries: &[IndexEntry]) -> bool {
        entries.windows(2).all(|w| w[0].key <= w[1].key)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(s: &str, p: &str, o: &str) -> TripleRecord {
        TripleRecord::new(s, p, o)
    }

    fn triple_g(s: &str, p: &str, o: &str, g: &str) -> TripleRecord {
        TripleRecord::in_graph(s, p, o, g)
    }

    // -- TripleRecord --------------------------------------------------------

    #[test]
    fn test_triple_record_default_graph() {
        let rec = triple("s", "p", "o");
        assert!(rec.graph.is_none());
    }

    #[test]
    fn test_triple_record_named_graph() {
        let rec = triple_g("s", "p", "o", "g");
        assert_eq!(rec.graph.as_deref(), Some("g"));
    }

    // -- IndexOrder::key_for ------------------------------------------------

    #[test]
    fn test_key_spo() {
        let rec = triple("s", "p", "o");
        let key = IndexOrder::SPO.key_for(&rec);
        assert_eq!(key, vec!["s", "p", "o"]);
    }

    #[test]
    fn test_key_pos() {
        let rec = triple("s", "p", "o");
        let key = IndexOrder::POS.key_for(&rec);
        assert_eq!(key, vec!["p", "o", "s"]);
    }

    #[test]
    fn test_key_osp() {
        let rec = triple("s", "p", "o");
        let key = IndexOrder::OSP.key_for(&rec);
        assert_eq!(key, vec!["o", "s", "p"]);
    }

    #[test]
    fn test_key_gspo() {
        let rec = triple_g("s", "p", "o", "g");
        let key = IndexOrder::GSPO.key_for(&rec);
        assert_eq!(key, vec!["g", "s", "p", "o"]);
    }

    #[test]
    fn test_key_gpos() {
        let rec = triple_g("s", "p", "o", "g");
        let key = IndexOrder::GPOS.key_for(&rec);
        assert_eq!(key, vec!["g", "p", "o", "s"]);
    }

    #[test]
    fn test_key_gosp() {
        let rec = triple_g("s", "p", "o", "g");
        let key = IndexOrder::GOSP.key_for(&rec);
        assert_eq!(key, vec!["g", "o", "s", "p"]);
    }

    #[test]
    fn test_key_gspo_default_graph_empty_string() {
        let rec = triple("s", "p", "o");
        let key = IndexOrder::GSPO.key_for(&rec);
        assert_eq!(key[0], "");
    }

    // -- IndexRebuilder::build ----------------------------------------------

    #[test]
    fn test_build_empty() {
        let rb = IndexRebuilder::new(IndexOrder::SPO);
        let entries = rb.build();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_build_single_triple() {
        let mut rb = IndexRebuilder::new(IndexOrder::SPO);
        rb.add_triple(triple("s", "p", "o"));
        let entries = rb.build();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, vec!["s", "p", "o"]);
    }

    #[test]
    fn test_build_spo_sorted() {
        let mut rb = IndexRebuilder::new(IndexOrder::SPO);
        rb.add_triple(triple("z", "p", "o"));
        rb.add_triple(triple("a", "p", "o"));
        rb.add_triple(triple("m", "p", "o"));
        let entries = rb.build();
        assert!(IndexRebuilder::verify(&entries));
        assert_eq!(entries[0].key[0], "a");
        assert_eq!(entries[2].key[0], "z");
    }

    #[test]
    fn test_build_pos_sorted() {
        let mut rb = IndexRebuilder::new(IndexOrder::POS);
        rb.add_triple(triple("s1", "p2", "o"));
        rb.add_triple(triple("s2", "p1", "o"));
        let entries = rb.build();
        assert!(IndexRebuilder::verify(&entries));
        assert_eq!(entries[0].key[0], "p1");
    }

    #[test]
    fn test_build_osp_sorted() {
        let mut rb = IndexRebuilder::new(IndexOrder::OSP);
        rb.add_triple(triple("s", "p", "z_obj"));
        rb.add_triple(triple("s", "p", "a_obj"));
        let entries = rb.build();
        assert!(IndexRebuilder::verify(&entries));
        assert_eq!(entries[0].key[0], "a_obj");
    }

    #[test]
    fn test_build_assigns_offsets() {
        let mut rb = IndexRebuilder::new(IndexOrder::SPO);
        for i in 0..5u64 {
            rb.add_triple(triple(&format!("s{i}"), "p", "o"));
        }
        let entries = rb.build();
        // Offsets are original record indices, reassigned after sorting
        // The set of offsets should contain values from 0..5
        let mut offsets: Vec<u64> = entries.iter().map(|e| e.offset).collect();
        offsets.sort_unstable();
        assert_eq!(offsets, vec![0, 1, 2, 3, 4]);
    }

    // -- build_all_orders ---------------------------------------------------

    #[test]
    fn test_build_all_orders_returns_six_keys() {
        let recs = vec![triple("s", "p", "o")];
        let all = IndexRebuilder::build_all_orders(&recs);
        assert_eq!(all.len(), 6);
        for order in IndexOrder::all() {
            assert!(all.contains_key(order));
        }
    }

    #[test]
    fn test_build_all_orders_same_count() {
        let recs = vec![
            triple("s1", "p", "o"),
            triple("s2", "p", "o"),
            triple("s3", "p", "o"),
        ];
        let all = IndexRebuilder::build_all_orders(&recs);
        for entries in all.values() {
            assert_eq!(entries.len(), 3);
        }
    }

    #[test]
    fn test_build_all_orders_each_is_sorted() {
        let recs = vec![
            triple("z", "p", "o"),
            triple("a", "p", "o"),
            triple("m", "p", "o"),
        ];
        let all = IndexRebuilder::build_all_orders(&recs);
        for entries in all.values() {
            assert!(IndexRebuilder::verify(entries));
        }
    }

    // -- rebuild_with_stats -------------------------------------------------

    #[test]
    fn test_rebuild_with_stats_count() {
        let recs = vec![triple("s1", "p", "o"), triple("s2", "p", "o")];
        let (_entries, stats) = IndexRebuilder::rebuild_with_stats(recs);
        assert_eq!(stats.records_processed, 2);
        assert_eq!(stats.entries_written, 2);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_rebuild_with_stats_empty() {
        let (_entries, stats) = IndexRebuilder::rebuild_with_stats(vec![]);
        assert_eq!(stats.records_processed, 0);
        assert_eq!(stats.entries_written, 0);
    }

    #[test]
    fn test_rebuild_with_stats_duration_non_negative() {
        let recs: Vec<TripleRecord> = (0..100)
            .map(|i| triple(&format!("s{i}"), "p", "o"))
            .collect();
        let (_entries, stats) = IndexRebuilder::rebuild_with_stats(recs);
        // duration_ms is u64, always >= 0
        let _ = stats.duration_ms;
        assert_eq!(stats.errors, 0);
    }

    // -- verify -------------------------------------------------------------

    #[test]
    fn test_verify_sorted_true() {
        let entries = vec![
            IndexEntry {
                key: vec!["a".into()],
                offset: 0,
            },
            IndexEntry {
                key: vec!["b".into()],
                offset: 1,
            },
            IndexEntry {
                key: vec!["c".into()],
                offset: 2,
            },
        ];
        assert!(IndexRebuilder::verify(&entries));
    }

    #[test]
    fn test_verify_unsorted_false() {
        let entries = vec![
            IndexEntry {
                key: vec!["b".into()],
                offset: 0,
            },
            IndexEntry {
                key: vec!["a".into()],
                offset: 1,
            },
        ];
        assert!(!IndexRebuilder::verify(&entries));
    }

    #[test]
    fn test_verify_empty_true() {
        assert!(IndexRebuilder::verify(&[]));
    }

    #[test]
    fn test_verify_single_true() {
        let entries = vec![IndexEntry {
            key: vec!["x".into()],
            offset: 0,
        }];
        assert!(IndexRebuilder::verify(&entries));
    }

    #[test]
    fn test_verify_duplicates_allowed() {
        let entries = vec![
            IndexEntry {
                key: vec!["a".into()],
                offset: 0,
            },
            IndexEntry {
                key: vec!["a".into()],
                offset: 1,
            },
        ];
        assert!(IndexRebuilder::verify(&entries));
    }

    // -- IndexOrder::all ----------------------------------------------------

    #[test]
    fn test_all_orders_count() {
        assert_eq!(IndexOrder::all().len(), 6);
    }

    // -- Integration --------------------------------------------------------

    #[test]
    fn test_spo_order_multi_component() {
        let mut rb = IndexRebuilder::new(IndexOrder::SPO);
        rb.add_triple(triple("b_s", "a_p", "o"));
        rb.add_triple(triple("a_s", "z_p", "o"));
        rb.add_triple(triple("a_s", "a_p", "o"));
        let entries = rb.build();
        assert!(IndexRebuilder::verify(&entries));
        // First: a_s / a_p / o
        assert_eq!(entries[0].key, vec!["a_s", "a_p", "o"]);
        // Second: a_s / z_p / o
        assert_eq!(entries[1].key, vec!["a_s", "z_p", "o"]);
        // Third: b_s / a_p / o
        assert_eq!(entries[2].key, vec!["b_s", "a_p", "o"]);
    }

    #[test]
    fn test_build_pos_key_order() {
        let mut rb = IndexRebuilder::new(IndexOrder::POS);
        rb.add_triple(triple("s", "alpha", "o1"));
        rb.add_triple(triple("s", "beta", "o2"));
        let entries = rb.build();
        assert_eq!(entries[0].key[0], "alpha");
        assert_eq!(entries[1].key[0], "beta");
    }

    #[test]
    fn test_build_osp_key_order() {
        let mut rb = IndexRebuilder::new(IndexOrder::OSP);
        rb.add_triple(triple("s", "p", "b_obj"));
        rb.add_triple(triple("s", "p", "a_obj"));
        let entries = rb.build();
        assert_eq!(entries[0].key[0], "a_obj");
        assert_eq!(entries[1].key[0], "b_obj");
    }

    #[test]
    fn test_triple_record_fields() {
        let rec = triple_g("my_s", "my_p", "my_o", "my_g");
        assert_eq!(rec.subject, "my_s");
        assert_eq!(rec.predicate, "my_p");
        assert_eq!(rec.object, "my_o");
        assert_eq!(rec.graph.as_deref(), Some("my_g"));
    }

    #[test]
    fn test_rebuild_stats_fields() {
        let recs = vec![triple("s", "p", "o")];
        let (entries, stats) = IndexRebuilder::rebuild_with_stats(recs);
        assert_eq!(stats.records_processed, 1);
        assert_eq!(stats.entries_written, entries.len());
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_index_order_all_unique() {
        let all = IndexOrder::all();
        let mut seen = std::collections::HashSet::new();
        for order in all {
            assert!(seen.insert(format!("{order:?}")), "Duplicate order");
        }
    }

    #[test]
    fn test_build_gspo_named_graph_sorted() {
        let mut rb = IndexRebuilder::new(IndexOrder::GSPO);
        rb.add_triple(triple_g("s2", "p", "o", "graph_b"));
        rb.add_triple(triple_g("s1", "p", "o", "graph_a"));
        let entries = rb.build();
        assert_eq!(entries[0].key[0], "graph_a");
        assert_eq!(entries[1].key[0], "graph_b");
    }

    #[test]
    fn test_large_build_is_sorted() {
        let mut rb = IndexRebuilder::new(IndexOrder::SPO);
        for i in (0u32..100).rev() {
            rb.add_triple(triple(&format!("s{i:03}"), "p", "o"));
        }
        let entries = rb.build();
        assert!(IndexRebuilder::verify(&entries));
        assert_eq!(entries[0].key[0], "s000");
        assert_eq!(entries[99].key[0], "s099");
    }
}
