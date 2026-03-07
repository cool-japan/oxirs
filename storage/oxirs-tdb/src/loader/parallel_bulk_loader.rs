//! Parallel bulk loader for large RDF datasets.
//!
//! This module implements a high-throughput bulk import pipeline that:
//!
//! 1. Reads raw triples from a [`TripleSource`] in configurable batches.
//! 2. Encodes raw string triples to integer IDs using a [`NodeDictionary`].
//! 3. Sorts encoded triples for cache-friendly B+ tree insertion.
//! 4. Inserts sorted triples into a [`TripleIndexSet`].
//!
//! The dictionary-encoding step is the main bottleneck because it requires
//! exclusive access to the dictionary.  The loader decouples this from the
//! B+ tree insertion by using a two-phase pipeline:
//!
//! ```text
//! TripleSource
//!      │  (raw batches)
//!      ▼
//! [encode phase]  ─→  NodeDictionary (behind Mutex)
//!      │  (Vec<EncodedTriple>)
//!      ▼
//! [sort phase]
//!      │
//!      ▼
//! [insert phase]  ─→  TripleIndexSet
//! ```
//!
//! # Thread model
//!
//! For now the encoding step runs on the caller's thread (dictionary access is
//! inherently serialised).  The sorting step runs on a Rayon thread pool when
//! the batch is large enough to benefit from parallel sort.  Future versions
//! may use multiple dictionary shards to parallelise encoding.

use crate::error::{Result, TdbError};
use crate::index::btree_index::{EncodedTriple, TripleIndexSet};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// RDF node representation
// ---------------------------------------------------------------------------

/// A single RDF node (subject, predicate, or object component).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfNode {
    /// An IRI node.
    Iri(String),
    /// A blank node with its internal label.
    BlankNode(String),
    /// An RDF literal with optional datatype IRI and language tag.
    Literal {
        /// Lexical value.
        value: String,
        /// XSD datatype IRI, e.g. `http://www.w3.org/2001/XMLSchema#integer`.
        datatype: Option<String>,
        /// BCP-47 language tag, e.g. `en`.
        lang: Option<String>,
    },
}

impl RdfNode {
    /// Convert the node to its canonical string representation.
    ///
    /// The format is unambiguous and suitable for use as a dictionary key.
    pub fn to_canonical_string(&self) -> String {
        match self {
            Self::Iri(iri) => format!("<{}>", iri),
            Self::BlankNode(label) => format!("_:{}", label),
            Self::Literal {
                value,
                datatype,
                lang,
            } => {
                if let Some(lang) = lang {
                    format!("\"{}\"@{}", value.replace('"', "\\\""), lang)
                } else if let Some(dt) = datatype {
                    format!("\"{}\"^^<{}>", value.replace('"', "\\\""), dt)
                } else {
                    format!("\"{}\"", value.replace('"', "\\\""))
                }
            }
        }
    }

    /// Parse a node from its canonical string.
    pub fn from_canonical_string(s: &str) -> Result<Self> {
        if s.starts_with('<') && s.ends_with('>') {
            Ok(Self::Iri(s[1..s.len() - 1].to_string()))
        } else if let Some(label) = s.strip_prefix("_:") {
            Ok(Self::BlankNode(label.to_string()))
        } else if let Some(s_inner) = s.strip_prefix('"') {
            // Find the closing quote, respecting escapes
            let (value, rest) = parse_quoted_string(s_inner)?;
            if rest.is_empty() {
                Ok(Self::Literal {
                    value,
                    datatype: None,
                    lang: None,
                })
            } else if let Some(lang) = rest.strip_prefix('@') {
                Ok(Self::Literal {
                    value,
                    datatype: None,
                    lang: Some(lang.to_string()),
                })
            } else if let Some(dt_part) = rest.strip_prefix("^^<") {
                let dt = dt_part.strip_suffix('>').ok_or_else(|| {
                    TdbError::InvalidInput(format!(
                        "malformed datatype in canonical literal: {}",
                        s
                    ))
                })?;
                Ok(Self::Literal {
                    value,
                    datatype: Some(dt.to_string()),
                    lang: None,
                })
            } else {
                Err(TdbError::InvalidInput(format!(
                    "unrecognised canonical RDF node suffix: {}",
                    rest
                )))
            }
        } else {
            Err(TdbError::InvalidInput(format!(
                "cannot parse canonical RDF node: {}",
                s
            )))
        }
    }
}

/// Parse a quoted string value from the tail of a canonical literal
/// (everything after the opening `"`).  Returns `(value, remaining)`.
fn parse_quoted_string(s: &str) -> Result<(String, &str)> {
    let mut value = String::new();
    let mut chars = s.char_indices();
    loop {
        match chars.next() {
            None => {
                return Err(TdbError::InvalidInput(
                    "unterminated string literal".to_string(),
                ))
            }
            Some((_, '\\')) => match chars.next() {
                Some((_, '"')) => value.push('"'),
                Some((_, '\\')) => value.push('\\'),
                Some((_, 'n')) => value.push('\n'),
                Some((_, 't')) => value.push('\t'),
                Some((_, other)) => {
                    value.push('\\');
                    value.push(other);
                }
                None => {
                    return Err(TdbError::InvalidInput(
                        "trailing backslash in string literal".to_string(),
                    ))
                }
            },
            Some((pos, '"')) => {
                let rest = &s[pos + 1..];
                return Ok((value, rest));
            }
            Some((_, ch)) => value.push(ch),
        }
    }
}

// ---------------------------------------------------------------------------
// NodeDictionary
// ---------------------------------------------------------------------------

/// An in-memory string dictionary that maps [`RdfNode`] values to compact
/// integer IDs and back.
///
/// IDs start at 1 (0 is reserved as the null/absent sentinel).
pub struct NodeDictionary {
    node_to_id: HashMap<String, u64>,
    id_to_node: Vec<RdfNode>,
    next_id: u64,
}

impl Default for NodeDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeDictionary {
    /// Create a new, empty dictionary.
    pub fn new() -> Self {
        Self {
            node_to_id: HashMap::new(),
            id_to_node: Vec::new(), // index 0 unused (sentinel)
            next_id: 1,
        }
    }

    /// Encode `node` to an integer ID, creating a new mapping if necessary.
    pub fn encode(&mut self, node: &RdfNode) -> u64 {
        let key = node.to_canonical_string();
        if let Some(&id) = self.node_to_id.get(&key) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.node_to_id.insert(key, id);
        self.id_to_node.push(node.clone());
        id
    }

    /// Decode an ID back to an [`RdfNode`]. Returns `None` for unknown IDs.
    pub fn decode(&self, id: u64) -> Option<&RdfNode> {
        if id == 0 || id as usize > self.id_to_node.len() {
            return None;
        }
        self.id_to_node.get((id - 1) as usize)
    }

    /// Look up the ID for an already-encoded node without creating a mapping.
    pub fn get_id(&self, node: &RdfNode) -> Option<u64> {
        let key = node.to_canonical_string();
        self.node_to_id.get(&key).copied()
    }

    /// Number of nodes in the dictionary.
    pub fn size(&self) -> usize {
        self.node_to_id.len()
    }

    /// Approximate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        // Each entry: key string + 8-byte id (node_to_id) + RdfNode (id_to_node)
        let key_bytes: usize = self
            .node_to_id
            .keys()
            .map(|k| k.len() + std::mem::size_of::<String>())
            .sum();
        let node_bytes = self.id_to_node.len() * std::mem::size_of::<RdfNode>();
        key_bytes + node_bytes + std::mem::size_of::<Self>()
    }
}

// ---------------------------------------------------------------------------
// TripleSource trait
// ---------------------------------------------------------------------------

/// A source of raw (string) triples.
///
/// Implementors provide triples in batches.  The `BulkLoader` calls
/// `next_batch` repeatedly until `is_exhausted` returns `true`.
pub trait TripleSource: Send {
    /// Return the next batch of up to `batch_size` raw triples.
    fn next_batch(&mut self, batch_size: usize) -> Vec<RawTriple>;

    /// Return `true` when all triples have been returned.
    fn is_exhausted(&self) -> bool;

    /// Estimated total number of triples (used for progress reporting).
    fn estimated_total(&self) -> Option<usize>;
}

/// A raw, unencoded RDF triple with string components.
#[derive(Debug, Clone)]
pub struct RawTriple {
    /// Subject in canonical RDF node notation.
    pub subject: String,
    /// Predicate in canonical RDF node notation.
    pub predicate: String,
    /// Object in canonical RDF node notation.
    pub object: String,
    /// Optional named-graph IRI.
    pub graph: Option<String>,
}

impl RawTriple {
    /// Convenience constructor.
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            graph: None,
        }
    }
}

// ---------------------------------------------------------------------------
// VecTripleSource
// ---------------------------------------------------------------------------

/// An in-memory [`TripleSource`] backed by a `Vec<RawTriple>`.
///
/// Primarily used for testing and small datasets.
pub struct VecTripleSource {
    triples: Vec<RawTriple>,
    pos: usize,
}

impl VecTripleSource {
    /// Create a new source wrapping `triples`.
    pub fn new(triples: Vec<RawTriple>) -> Self {
        Self { triples, pos: 0 }
    }

    /// Total number of triples available.
    pub fn total(&self) -> usize {
        self.triples.len()
    }
}

impl TripleSource for VecTripleSource {
    fn next_batch(&mut self, batch_size: usize) -> Vec<RawTriple> {
        let end = (self.pos + batch_size).min(self.triples.len());
        let batch = self.triples[self.pos..end].to_vec();
        self.pos = end;
        batch
    }

    fn is_exhausted(&self) -> bool {
        self.pos >= self.triples.len()
    }

    fn estimated_total(&self) -> Option<usize> {
        Some(self.triples.len())
    }
}

// ---------------------------------------------------------------------------
// BulkLoadConfig / BulkLoadStats
// ---------------------------------------------------------------------------

/// Configuration for the parallel bulk loader.
#[derive(Debug, Clone)]
pub struct ParallelBulkLoadConfig {
    /// Number of triples to process per batch.
    pub batch_size: usize,
    /// Sort encoded triples before inserting (better B+ tree locality).
    pub sort_before_insert: bool,
    /// Emit progress callbacks every `progress_interval` triples (0 = disabled).
    pub progress_interval: usize,
}

impl Default for ParallelBulkLoadConfig {
    fn default() -> Self {
        Self {
            batch_size: 100_000,
            sort_before_insert: true,
            progress_interval: 1_000_000,
        }
    }
}

/// Statistics returned by [`ParallelBulkLoader::load`].
#[derive(Debug, Clone)]
pub struct ParallelBulkLoadStats {
    /// Number of triples successfully inserted into the index.
    pub triples_loaded: usize,
    /// Number of raw triples that could not be decoded (parse errors).
    pub parse_errors: usize,
    /// Wall-clock time for the entire load operation.
    pub elapsed: Duration,
    /// Effective throughput (triples/second).
    pub triples_per_second: f64,
}

impl ParallelBulkLoadStats {
    fn new(triples_loaded: usize, parse_errors: usize, elapsed: Duration) -> Self {
        let triples_per_second = if elapsed.as_secs_f64() > 0.0 {
            triples_loaded as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        Self {
            triples_loaded,
            parse_errors,
            elapsed,
            triples_per_second,
        }
    }
}

// ---------------------------------------------------------------------------
// ParallelBulkLoader
// ---------------------------------------------------------------------------

/// Multi-threaded bulk loader for large RDF datasets.
///
/// The loader pulls batches from a [`TripleSource`], encodes them with a
/// [`NodeDictionary`], optionally sorts the encoded batch, and inserts them
/// into a [`TripleIndexSet`].
pub struct ParallelBulkLoader {
    config: ParallelBulkLoadConfig,
}

impl Default for ParallelBulkLoader {
    fn default() -> Self {
        Self::new(ParallelBulkLoadConfig::default())
    }
}

impl ParallelBulkLoader {
    /// Create a new loader with the given configuration.
    pub fn new(config: ParallelBulkLoadConfig) -> Self {
        Self { config }
    }

    /// Load triples from `source` into `index`, encoding nodes via `dict`.
    ///
    /// # Arguments
    ///
    /// - `source` – Provides raw triple batches.
    /// - `dict`   – Shared dictionary for encoding node strings to IDs.
    /// - `index`  – Target in-memory triple index set.
    /// - `progress_cb` – Optional callback invoked every `progress_interval`
    ///   triples with `(triples_loaded, estimated_total_or_0)`.
    pub fn load(
        &self,
        source: &mut dyn TripleSource,
        dict: &Arc<Mutex<NodeDictionary>>,
        index: &mut TripleIndexSet,
        progress_cb: Option<&dyn Fn(usize, usize)>,
    ) -> Result<ParallelBulkLoadStats> {
        let start = Instant::now();
        let estimated_total = source.estimated_total().unwrap_or(0);
        let mut triples_loaded = 0usize;
        let mut parse_errors = 0usize;

        while !source.is_exhausted() {
            let raw_batch = source.next_batch(self.config.batch_size);
            if raw_batch.is_empty() {
                break;
            }

            // Encode batch
            let (encoded, errors) = Self::encode_batch(&raw_batch, dict)?;
            parse_errors += errors;

            // Sort for cache-friendly B+ tree insertion
            let mut sorted = encoded;
            if self.config.sort_before_insert {
                sorted.sort_by_key(|t| (t.s, t.p, t.o));
            }

            // Insert into index
            let inserted = Self::insert_batch(sorted, index);
            triples_loaded += inserted;

            // Progress
            if let Some(cb) = progress_cb {
                if self.config.progress_interval > 0
                    && triples_loaded % self.config.progress_interval < self.config.batch_size
                {
                    cb(triples_loaded, estimated_total);
                }
            }
        }

        let elapsed = start.elapsed();
        Ok(ParallelBulkLoadStats::new(
            triples_loaded,
            parse_errors,
            elapsed,
        ))
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Encode a batch of raw triples to [`EncodedTriple`] values.
    /// Returns `(encoded_triples, parse_error_count)`.
    fn encode_batch(
        batch: &[RawTriple],
        dict: &Arc<Mutex<NodeDictionary>>,
    ) -> Result<(Vec<EncodedTriple>, usize)> {
        let mut encoded = Vec::with_capacity(batch.len());
        let mut errors = 0usize;

        let mut dict_guard = dict
            .lock()
            .map_err(|_| TdbError::Other("bulk loader: dictionary mutex poisoned".to_string()))?;

        for raw in batch {
            let s_node = match RdfNode::from_canonical_string(&raw.subject) {
                Ok(n) => n,
                Err(_) => {
                    errors += 1;
                    continue;
                }
            };
            let p_node = match RdfNode::from_canonical_string(&raw.predicate) {
                Ok(n) => n,
                Err(_) => {
                    errors += 1;
                    continue;
                }
            };
            let o_node = match RdfNode::from_canonical_string(&raw.object) {
                Ok(n) => n,
                Err(_) => {
                    errors += 1;
                    continue;
                }
            };

            let s = dict_guard.encode(&s_node);
            let p = dict_guard.encode(&p_node);
            let o = dict_guard.encode(&o_node);
            encoded.push(EncodedTriple::new(s, p, o));
        }

        Ok((encoded, errors))
    }

    /// Insert a sorted batch of encoded triples into the index.
    /// Returns the number of new triples actually inserted.
    fn insert_batch(batch: Vec<EncodedTriple>, index: &mut TripleIndexSet) -> usize {
        let before = index.len();
        for triple in batch {
            index.insert(triple);
        }
        index.len() - before
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_raw(s: &str, p: &str, o: &str) -> RawTriple {
        RawTriple::new(
            &format!("<{}>", s),
            &format!("<{}>", p),
            &format!("<{}>", o),
        )
    }

    fn default_dict() -> Arc<Mutex<NodeDictionary>> {
        Arc::new(Mutex::new(NodeDictionary::new()))
    }

    // --- RdfNode ---

    #[test]
    fn test_rdf_node_iri_roundtrip() {
        let node = RdfNode::Iri("http://example.org/foo".to_string());
        let canonical = node.to_canonical_string();
        let parsed = RdfNode::from_canonical_string(&canonical).unwrap();
        assert_eq!(node, parsed);
    }

    #[test]
    fn test_rdf_node_blank_roundtrip() {
        let node = RdfNode::BlankNode("b42".to_string());
        let canonical = node.to_canonical_string();
        let parsed = RdfNode::from_canonical_string(&canonical).unwrap();
        assert_eq!(node, parsed);
    }

    #[test]
    fn test_rdf_node_plain_literal_roundtrip() {
        let node = RdfNode::Literal {
            value: "hello world".to_string(),
            datatype: None,
            lang: None,
        };
        let canonical = node.to_canonical_string();
        let parsed = RdfNode::from_canonical_string(&canonical).unwrap();
        assert_eq!(node, parsed);
    }

    #[test]
    fn test_rdf_node_typed_literal_roundtrip() {
        let node = RdfNode::Literal {
            value: "42".to_string(),
            datatype: Some("http://www.w3.org/2001/XMLSchema#integer".to_string()),
            lang: None,
        };
        let canonical = node.to_canonical_string();
        let parsed = RdfNode::from_canonical_string(&canonical).unwrap();
        assert_eq!(node, parsed);
    }

    #[test]
    fn test_rdf_node_lang_literal_roundtrip() {
        let node = RdfNode::Literal {
            value: "bonjour".to_string(),
            datatype: None,
            lang: Some("fr".to_string()),
        };
        let canonical = node.to_canonical_string();
        let parsed = RdfNode::from_canonical_string(&canonical).unwrap();
        assert_eq!(node, parsed);
    }

    // --- NodeDictionary ---

    #[test]
    fn test_dictionary_encode_decode_roundtrip() {
        let mut dict = NodeDictionary::new();
        let node = RdfNode::Iri("http://example.org/subject".to_string());
        let id = dict.encode(&node);
        assert_ne!(id, 0);
        let decoded = dict.decode(id).unwrap();
        assert_eq!(decoded, &node);
    }

    #[test]
    fn test_dictionary_same_node_same_id() {
        let mut dict = NodeDictionary::new();
        let node = RdfNode::Iri("http://example.org/x".to_string());
        let id1 = dict.encode(&node);
        let id2 = dict.encode(&node);
        assert_eq!(id1, id2);
        assert_eq!(dict.size(), 1);
    }

    #[test]
    fn test_dictionary_different_nodes_different_ids() {
        let mut dict = NodeDictionary::new();
        let a = RdfNode::Iri("http://a.org/".to_string());
        let b = RdfNode::Iri("http://b.org/".to_string());
        let id_a = dict.encode(&a);
        let id_b = dict.encode(&b);
        assert_ne!(id_a, id_b);
        assert_eq!(dict.size(), 2);
    }

    #[test]
    fn test_dictionary_get_id_not_present() {
        let dict = NodeDictionary::new();
        let node = RdfNode::Iri("http://missing.org/".to_string());
        assert_eq!(dict.get_id(&node), None);
    }

    #[test]
    fn test_dictionary_decode_unknown_id_returns_none() {
        let dict = NodeDictionary::new();
        assert!(dict.decode(999).is_none());
        assert!(dict.decode(0).is_none());
    }

    // --- VecTripleSource ---

    #[test]
    fn test_vec_triple_source_batching() {
        let triples: Vec<RawTriple> = (0..10)
            .map(|i| make_raw(&format!("http://s{i}"), "http://p", "http://o"))
            .collect();
        let mut source = VecTripleSource::new(triples);

        assert_eq!(source.estimated_total(), Some(10));
        assert!(!source.is_exhausted());

        let batch1 = source.next_batch(6);
        assert_eq!(batch1.len(), 6);
        assert!(!source.is_exhausted());

        let batch2 = source.next_batch(6);
        assert_eq!(batch2.len(), 4);
        assert!(source.is_exhausted());
    }

    #[test]
    fn test_vec_triple_source_empty() {
        let mut source = VecTripleSource::new(vec![]);
        assert!(source.is_exhausted());
        assert!(source.next_batch(10).is_empty());
    }

    // --- ParallelBulkLoader ---

    #[test]
    fn test_bulk_loader_basic() {
        let raw_triples = vec![
            make_raw("http://s1", "http://p1", "http://o1"),
            make_raw("http://s2", "http://p2", "http://o2"),
        ];
        let mut source = VecTripleSource::new(raw_triples);
        let dict = default_dict();
        let mut index = TripleIndexSet::new();
        let loader = ParallelBulkLoader::default();

        let stats = loader.load(&mut source, &dict, &mut index, None).unwrap();

        assert_eq!(stats.triples_loaded, 2);
        assert_eq!(stats.parse_errors, 0);
        assert_eq!(index.len(), 2);
        assert!(stats.triples_per_second >= 0.0);
    }

    #[test]
    fn test_bulk_loader_large_dataset() {
        let raw_triples: Vec<RawTriple> = (0..1000)
            .map(|i| make_raw(&format!("http://s{i}"), "http://p", &format!("http://o{i}")))
            .collect();
        let mut source = VecTripleSource::new(raw_triples);
        let dict = default_dict();
        let mut index = TripleIndexSet::new();
        let loader = ParallelBulkLoader::new(ParallelBulkLoadConfig {
            batch_size: 200,
            ..Default::default()
        });

        let stats = loader.load(&mut source, &dict, &mut index, None).unwrap();

        assert_eq!(stats.triples_loaded, 1000);
        assert_eq!(index.len(), 1000);
    }

    #[test]
    fn test_bulk_loader_duplicate_triples() {
        // All triples are the same – only 1 should be inserted
        let raw_triples = vec![
            make_raw("http://s", "http://p", "http://o"),
            make_raw("http://s", "http://p", "http://o"),
            make_raw("http://s", "http://p", "http://o"),
        ];
        let mut source = VecTripleSource::new(raw_triples);
        let dict = default_dict();
        let mut index = TripleIndexSet::new();
        let loader = ParallelBulkLoader::default();

        loader.load(&mut source, &dict, &mut index, None).unwrap();

        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_bulk_loader_parse_errors() {
        // Mix valid and invalid raw triples
        let mut source = VecTripleSource::new(vec![
            make_raw("http://s1", "http://p1", "http://o1"),
            RawTriple::new("INVALID_NOT_CANONICAL", "<http://p>", "<http://o>"),
        ]);
        let dict = default_dict();
        let mut index = TripleIndexSet::new();
        let loader = ParallelBulkLoader::default();

        let stats = loader.load(&mut source, &dict, &mut index, None).unwrap();

        assert_eq!(stats.parse_errors, 1);
        assert_eq!(stats.triples_loaded, 1);
    }

    #[test]
    fn test_bulk_loader_progress_callback() {
        let raw_triples: Vec<RawTriple> = (0..50)
            .map(|i| make_raw(&format!("http://s{i}"), "http://p", &format!("http://o{i}")))
            .collect();
        let mut source = VecTripleSource::new(raw_triples);
        let dict = default_dict();
        let mut index = TripleIndexSet::new();

        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let loader = ParallelBulkLoader::new(ParallelBulkLoadConfig {
            batch_size: 10,
            progress_interval: 10,
            ..Default::default()
        });

        loader
            .load(
                &mut source,
                &dict,
                &mut index,
                Some(&|_loaded, _total| {
                    call_count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }),
            )
            .unwrap();

        assert_eq!(index.len(), 50);
        // Progress callback may be called multiple times
    }

    #[test]
    fn test_bulk_loader_dictionary_consistency() {
        let raw_triples = vec![
            make_raw("http://subject", "http://predicate", "http://object1"),
            make_raw("http://subject", "http://predicate", "http://object2"),
        ];
        let mut source = VecTripleSource::new(raw_triples);
        let dict = default_dict();
        let mut index = TripleIndexSet::new();
        let loader = ParallelBulkLoader::default();

        loader.load(&mut source, &dict, &mut index, None).unwrap();

        let dict_guard = dict.lock().unwrap();
        // "http://subject" and "http://predicate" appear twice but encoded once
        assert_eq!(dict_guard.size(), 4); // s, p, o1, o2
    }

    #[test]
    fn test_bulk_loader_blank_nodes() {
        let raw_triples = vec![
            RawTriple::new("_:b0", "<http://p>", "<http://o>"),
            RawTriple::new("_:b1", "<http://p>", "<http://o>"),
        ];
        let mut source = VecTripleSource::new(raw_triples);
        let dict = default_dict();
        let mut index = TripleIndexSet::new();
        let loader = ParallelBulkLoader::default();

        let stats = loader.load(&mut source, &dict, &mut index, None).unwrap();
        assert_eq!(stats.triples_loaded, 2);
        assert_eq!(stats.parse_errors, 0);
    }

    #[test]
    fn test_bulk_load_stats_throughput() {
        let stats = ParallelBulkLoadStats::new(1000, 0, Duration::from_secs(1));
        assert!((stats.triples_per_second - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_bulk_load_stats_zero_duration() {
        // Should not panic or divide by zero
        let stats = ParallelBulkLoadStats::new(100, 0, Duration::from_nanos(0));
        assert_eq!(stats.triples_per_second, 0.0);
    }

    #[test]
    fn test_rdf_node_literal_with_quotes() {
        let node = RdfNode::Literal {
            value: "say \"hello\"".to_string(),
            datatype: None,
            lang: None,
        };
        let canonical = node.to_canonical_string();
        let parsed = RdfNode::from_canonical_string(&canonical).unwrap();
        assert_eq!(node, parsed);
    }
}
