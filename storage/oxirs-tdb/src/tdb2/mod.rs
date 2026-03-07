//! # TDB2-Compatible Data Format
//!
//! Implements the TDB2 on-disk format as used in Apache Jena TDB2.
//! TDB2 uses DADB (Direct-Access Database) architecture with a NodeTable
//! for term interning and a TripleIndex (SPO/POS/OSP) for efficient queries.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │               Tdb2Database                       │
//! ├──────────────────────┬───────────────────────────┤
//! │     NodeTable        │       TripleIndex          │
//! │  RdfTerm <-> u64 ID  │  BTreeMap<(s,p,o), ()>    │
//! │  forward: term->id   │  SPO, POS, OSP indexes     │
//! │  reverse: id->term   │                            │
//! └──────────────────────┴───────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use oxirs_tdb::tdb2::{Tdb2Database, RdfTerm};
//!
//! let mut db = Tdb2Database::new();
//! let s = RdfTerm::Iri("http://example.org/subject".to_string());
//! let p = RdfTerm::Iri("http://example.org/predicate".to_string());
//! let o = RdfTerm::Literal {
//!     value: "hello".to_string(),
//!     lang: Some("en".to_string()),
//!     datatype: None,
//! };
//! db.insert_triple(&s, &p, &o).unwrap();
//! ```

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Result, TdbError};

// ─────────────────────────────────────────────────────────────────────────────
// RdfTerm
// ─────────────────────────────────────────────────────────────────────────────

/// An RDF term: IRI, Literal, or Blank Node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfTerm {
    /// An IRI reference, e.g. `http://example.org/name`.
    Iri(String),
    /// An RDF Literal with optional language tag or datatype IRI.
    Literal {
        /// The lexical form of the literal.
        value: String,
        /// Optional BCP 47 language tag (e.g. `"en"`, `"fr-CA"`).
        lang: Option<String>,
        /// Optional datatype IRI (e.g. `"http://www.w3.org/2001/XMLSchema#integer"`).
        datatype: Option<String>,
    },
    /// A blank node with a local identifier.
    BlankNode(String),
}

impl RdfTerm {
    /// Produce a canonical string key used for interning comparisons.
    ///
    /// - IRI: `<{iri}>`
    /// - Literal: `"{value}"@{lang}^^{datatype}` (absent components omitted)
    /// - Blank node: `_:{id}`
    pub fn term_key(&self) -> String {
        match self {
            RdfTerm::Iri(iri) => format!("<{}>", iri),
            RdfTerm::Literal {
                value,
                lang,
                datatype,
            } => {
                let mut key = format!("\"{}\"", value);
                if let Some(l) = lang {
                    key.push('@');
                    key.push_str(l);
                }
                if let Some(dt) = datatype {
                    key.push_str("^^");
                    key.push_str(dt);
                }
                key
            }
            RdfTerm::BlankNode(id) => format!("_:{}", id),
        }
    }

    /// Returns `true` if this term is an IRI.
    pub fn is_iri(&self) -> bool {
        matches!(self, RdfTerm::Iri(_))
    }

    /// Returns `true` if this term is a Literal.
    pub fn is_literal(&self) -> bool {
        matches!(self, RdfTerm::Literal { .. })
    }

    /// Returns `true` if this term is a Blank Node.
    pub fn is_blank_node(&self) -> bool {
        matches!(self, RdfTerm::BlankNode(_))
    }
}

impl std::fmt::Display for RdfTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.term_key())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeTable
// ─────────────────────────────────────────────────────────────────────────────

/// Bidirectional mapping between RDF terms and compact integer node IDs.
///
/// Closely mirrors the TDB2 node table (DADB). IDs start at 1 (0 is reserved
/// as a sentinel "null" value in triple indexes).
///
/// Thread safety: `NodeTable` itself is not thread-safe; wrap in `Mutex`/`RwLock`
/// if shared access is needed.
pub struct NodeTable {
    forward: HashMap<String, u64>,
    reverse: HashMap<u64, RdfTerm>,
    next_id: AtomicU64,
}

impl std::fmt::Debug for NodeTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeTable")
            .field("count", &self.forward.len())
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

impl NodeTable {
    /// Create a new, empty node table.
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            reverse: HashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    /// Intern an RDF term: return its existing ID or assign a new one.
    ///
    /// IDs are assigned sequentially starting from 1.
    pub fn intern(&mut self, term: &RdfTerm) -> u64 {
        let key = term.term_key();
        if let Some(&id) = self.forward.get(&key) {
            return id;
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.forward.insert(key, id);
        self.reverse.insert(id, term.clone());
        id
    }

    /// Look up the RDF term for a given node ID.
    ///
    /// Returns `None` if the ID has not been assigned.
    pub fn lookup(&self, id: u64) -> Option<&RdfTerm> {
        self.reverse.get(&id)
    }

    /// Look up the ID for a given RDF term.
    ///
    /// Returns `None` if the term has not been interned.
    pub fn lookup_term(&self, term: &RdfTerm) -> Option<u64> {
        let key = term.term_key();
        self.forward.get(&key).copied()
    }

    /// Return the number of interned terms.
    pub fn len(&self) -> u64 {
        self.forward.len() as u64
    }

    /// Return `true` if no terms have been interned yet.
    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Clear all interned terms and reset the ID counter.
    pub fn clear(&mut self) {
        self.forward.clear();
        self.reverse.clear();
        self.next_id.store(1, Ordering::Relaxed);
    }

    /// Serialize the node table to a byte vector (simple custom binary format).
    ///
    /// Format: `[n: u64 LE][id: u64 LE][key_len: u32 LE][key: UTF-8 bytes] * n`
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let count = self.forward.len() as u64;
        buf.extend_from_slice(&count.to_le_bytes());
        for (key, &id) in &self.forward {
            buf.extend_from_slice(&id.to_le_bytes());
            let key_bytes = key.as_bytes();
            buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(key_bytes);
        }
        buf
    }

    /// Deserialize a node table from bytes previously written by `serialize`.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(TdbError::Deserialization(
                "NodeTable data too short for count".to_string(),
            ));
        }
        let count = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| TdbError::Deserialization("count slice error".to_string()))?,
        );

        let mut table = NodeTable::new();
        let mut pos = 8usize;
        let mut max_id: u64 = 0;

        for _ in 0..count {
            if pos + 12 > data.len() {
                return Err(TdbError::Deserialization(
                    "NodeTable truncated at id/key_len".to_string(),
                ));
            }
            let id = u64::from_le_bytes(
                data[pos..pos + 8]
                    .try_into()
                    .map_err(|_| TdbError::Deserialization("id slice error".to_string()))?,
            );
            pos += 8;
            let key_len = u32::from_le_bytes(
                data[pos..pos + 4]
                    .try_into()
                    .map_err(|_| TdbError::Deserialization("key_len slice error".to_string()))?,
            ) as usize;
            pos += 4;

            if pos + key_len > data.len() {
                return Err(TdbError::Deserialization(
                    "NodeTable truncated at key bytes".to_string(),
                ));
            }
            let key = std::str::from_utf8(&data[pos..pos + key_len])
                .map_err(|_| TdbError::Deserialization("NodeTable key not UTF-8".to_string()))?
                .to_string();
            pos += key_len;

            let term = parse_term_key(&key)?;
            if id > max_id {
                max_id = id;
            }
            table.forward.insert(key, id);
            table.reverse.insert(id, term);
        }

        if max_id > 0 {
            table.next_id.store(max_id + 1, Ordering::Relaxed);
        }

        Ok(table)
    }
}

impl Default for NodeTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse an RDF term from its canonical key representation (inverse of `term_key`).
fn parse_term_key(key: &str) -> Result<RdfTerm> {
    if key.starts_with('<') && key.ends_with('>') {
        let iri = &key[1..key.len() - 1];
        return Ok(RdfTerm::Iri(iri.to_string()));
    }
    if let Some(rest) = key.strip_prefix("_:") {
        return Ok(RdfTerm::BlankNode(rest.to_string()));
    }
    if let Some(after_open_quote) = key.strip_prefix('"') {
        // Find closing quote in the remainder
        let closing = after_open_quote.find('"');
        if let Some(ci) = closing {
            let value = after_open_quote[..ci].to_string();
            let suffix = &after_open_quote[ci + 1..];
            let lang = if suffix.starts_with('@') {
                let lang_end = suffix.find("^^").unwrap_or(suffix.len());
                Some(suffix[1..lang_end].to_string())
            } else {
                None
            };
            let datatype = suffix
                .find("^^")
                .map(|dt_start| suffix[dt_start + 2..].to_string());
            return Ok(RdfTerm::Literal {
                value,
                lang,
                datatype,
            });
        }
    }
    Err(TdbError::Deserialization(format!(
        "Cannot parse term key: {}",
        key
    )))
}

// ─────────────────────────────────────────────────────────────────────────────
// TripleIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Triple index with three orderings: SPO, POS, and OSP.
///
/// Mirrors TDB2's triple index structure. Each ordering enables efficient
/// pattern matching when different components are bound.
///
/// - **SPO**: efficient lookup by subject, then subject+predicate
/// - **POS**: efficient lookup by predicate, then predicate+object
/// - **OSP**: efficient lookup by object, then object+subject
#[derive(Debug)]
pub struct TripleIndex {
    /// Subject-Predicate-Object ordering
    spo: BTreeMap<(u64, u64, u64), ()>,
    /// Predicate-Object-Subject ordering
    pos: BTreeMap<(u64, u64, u64), ()>,
    /// Object-Subject-Predicate ordering
    osp: BTreeMap<(u64, u64, u64), ()>,
}

impl TripleIndex {
    /// Create a new, empty triple index.
    pub fn new() -> Self {
        Self {
            spo: BTreeMap::new(),
            pos: BTreeMap::new(),
            osp: BTreeMap::new(),
        }
    }

    /// Insert a triple `(s_id, p_id, o_id)` into all three indexes.
    ///
    /// Duplicate inserts are silently ignored (idempotent).
    pub fn add(&mut self, s_id: u64, p_id: u64, o_id: u64) -> Result<()> {
        self.spo.insert((s_id, p_id, o_id), ());
        self.pos.insert((p_id, o_id, s_id), ());
        self.osp.insert((o_id, s_id, p_id), ());
        Ok(())
    }

    /// Remove a triple from all three indexes.
    ///
    /// Returns `true` if the triple existed.
    pub fn remove(&mut self, s_id: u64, p_id: u64, o_id: u64) -> bool {
        let existed = self.spo.remove(&(s_id, p_id, o_id)).is_some();
        self.pos.remove(&(p_id, o_id, s_id));
        self.osp.remove(&(o_id, s_id, p_id));
        existed
    }

    /// Query triples matching the given optional pattern components.
    ///
    /// Each component is `None` for wildcard. Uses the most efficient
    /// index ordering based on which components are bound.
    ///
    /// Returns a `Vec<(s_id, p_id, o_id)>`.
    pub fn query_spo(
        &self,
        s: Option<u64>,
        p: Option<u64>,
        o: Option<u64>,
    ) -> Vec<(u64, u64, u64)> {
        match (s, p, o) {
            // All bound — point lookup in SPO
            (Some(sv), Some(pv), Some(ov)) => {
                if self.spo.contains_key(&(sv, pv, ov)) {
                    vec![(sv, pv, ov)]
                } else {
                    vec![]
                }
            }

            // Subject + Predicate — range in SPO
            (Some(sv), Some(pv), None) => {
                let lo = (sv, pv, 0);
                let hi = (sv, pv, u64::MAX);
                self.spo
                    .range(lo..=hi)
                    .map(|(&(s2, p2, o2), _)| (s2, p2, o2))
                    .collect()
            }

            // Subject only — range in SPO
            (Some(sv), None, None) => {
                let lo = (sv, 0, 0);
                let hi = (sv, u64::MAX, u64::MAX);
                self.spo
                    .range(lo..=hi)
                    .map(|(&(s2, p2, o2), _)| (s2, p2, o2))
                    .collect()
            }

            // Predicate + Object — range in POS
            (None, Some(pv), Some(ov)) => {
                let lo = (pv, ov, 0);
                let hi = (pv, ov, u64::MAX);
                self.pos
                    .range(lo..=hi)
                    .map(|(&(p2, o2, s2), _)| (s2, p2, o2))
                    .collect()
            }

            // Predicate only — range in POS
            (None, Some(pv), None) => {
                let lo = (pv, 0, 0);
                let hi = (pv, u64::MAX, u64::MAX);
                self.pos
                    .range(lo..=hi)
                    .map(|(&(p2, o2, s2), _)| (s2, p2, o2))
                    .collect()
            }

            // Object only — range in OSP
            (None, None, Some(ov)) => {
                let lo = (ov, 0, 0);
                let hi = (ov, u64::MAX, u64::MAX);
                self.osp
                    .range(lo..=hi)
                    .map(|(&(o2, s2, p2), _)| (s2, p2, o2))
                    .collect()
            }

            // Subject + Object — scan OSP for object, filter by subject
            (Some(sv), None, Some(ov)) => {
                let lo = (ov, sv, 0);
                let hi = (ov, sv, u64::MAX);
                self.osp
                    .range(lo..=hi)
                    .map(|(&(o2, s2, p2), _)| (s2, p2, o2))
                    .collect()
            }

            // Wildcard — full scan (use SPO order)
            (None, None, None) => self.spo.keys().map(|&(s2, p2, o2)| (s2, p2, o2)).collect(),
        }
    }

    /// Return the total number of unique triples in the index.
    pub fn triple_count(&self) -> u64 {
        self.spo.len() as u64
    }

    /// Check whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.spo.is_empty()
    }

    /// Clear all triples from all indexes.
    pub fn clear(&mut self) {
        self.spo.clear();
        self.pos.clear();
        self.osp.clear();
    }
}

impl Default for TripleIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tdb2Database
// ─────────────────────────────────────────────────────────────────────────────

/// TDB2-compatible database combining a [`NodeTable`] and a [`TripleIndex`].
///
/// This is the primary entry point for TDB2 format operations.
#[derive(Debug)]
pub struct Tdb2Database {
    node_table: NodeTable,
    triple_index: TripleIndex,
}

impl Tdb2Database {
    /// Create a new, empty TDB2 database.
    pub fn new() -> Self {
        Self {
            node_table: NodeTable::new(),
            triple_index: TripleIndex::new(),
        }
    }

    /// Open a TDB2 database from a directory path.
    ///
    /// If the directory does not contain serialised data, returns an empty
    /// database (equivalent to `new`).  Serialised data is expected as two
    /// files: `nodes.tdb2` (NodeTable) and `triples.tdb2` (TripleIndex).
    pub fn open(path: &Path) -> Result<Self> {
        let nodes_path = path.join("nodes.tdb2");
        let triples_path = path.join("triples.tdb2");

        let node_table = if nodes_path.exists() {
            let data = std::fs::read(&nodes_path).map_err(TdbError::Io)?;
            NodeTable::deserialize(&data)?
        } else {
            NodeTable::new()
        };

        let triple_index = if triples_path.exists() {
            let data = std::fs::read(&triples_path).map_err(TdbError::Io)?;
            TripleIndex::deserialize(&data)?
        } else {
            TripleIndex::new()
        };

        Ok(Self {
            node_table,
            triple_index,
        })
    }

    /// Persist the database to a directory.
    ///
    /// Creates `nodes.tdb2` and `triples.tdb2` in the given directory.
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path).map_err(TdbError::Io)?;

        let nodes_data = self.node_table.serialize();
        std::fs::write(path.join("nodes.tdb2"), &nodes_data).map_err(TdbError::Io)?;

        let triples_data = self.triple_index.serialize();
        std::fs::write(path.join("triples.tdb2"), &triples_data).map_err(TdbError::Io)?;

        Ok(())
    }

    /// Insert a triple `(s, p, o)` into the database.
    ///
    /// Terms are interned into the node table automatically.
    pub fn insert_triple(&mut self, s: &RdfTerm, p: &RdfTerm, o: &RdfTerm) -> Result<()> {
        let s_id = self.node_table.intern(s);
        let p_id = self.node_table.intern(p);
        let o_id = self.node_table.intern(o);
        self.triple_index.add(s_id, p_id, o_id)
    }

    /// Query triples matching the given pattern.
    ///
    /// Each component may be `None` (wildcard) or `Some(&RdfTerm)`.
    pub fn query(
        &self,
        s: Option<&RdfTerm>,
        p: Option<&RdfTerm>,
        o: Option<&RdfTerm>,
    ) -> Vec<(RdfTerm, RdfTerm, RdfTerm)> {
        let s_id = s.and_then(|t| self.node_table.lookup_term(t));
        let p_id = p.and_then(|t| self.node_table.lookup_term(t));
        let o_id = o.and_then(|t| self.node_table.lookup_term(t));

        // If any bound component is unknown, no triples can match
        if s.is_some() && s_id.is_none() {
            return vec![];
        }
        if p.is_some() && p_id.is_none() {
            return vec![];
        }
        if o.is_some() && o_id.is_none() {
            return vec![];
        }

        self.triple_index
            .query_spo(s_id, p_id, o_id)
            .into_iter()
            .filter_map(|(sv, pv, ov)| {
                let st = self.node_table.lookup(sv)?.clone();
                let pt = self.node_table.lookup(pv)?.clone();
                let ot = self.node_table.lookup(ov)?.clone();
                Some((st, pt, ot))
            })
            .collect()
    }

    /// Return the total number of triples in the database.
    pub fn triple_count(&self) -> u64 {
        self.triple_index.triple_count()
    }

    /// Return the total number of interned terms in the node table.
    pub fn node_count(&self) -> u64 {
        self.node_table.len()
    }

    /// Remove a triple from the database.
    ///
    /// Returns `true` if the triple existed and was removed.
    pub fn remove_triple(&mut self, s: &RdfTerm, p: &RdfTerm, o: &RdfTerm) -> Result<bool> {
        let s_id = match self.node_table.lookup_term(s) {
            Some(id) => id,
            None => return Ok(false),
        };
        let p_id = match self.node_table.lookup_term(p) {
            Some(id) => id,
            None => return Ok(false),
        };
        let o_id = match self.node_table.lookup_term(o) {
            Some(id) => id,
            None => return Ok(false),
        };
        Ok(self.triple_index.remove(s_id, p_id, o_id))
    }

    /// Iterate over all triples in SPO order.
    pub fn iter_triples(&self) -> impl Iterator<Item = (RdfTerm, RdfTerm, RdfTerm)> + '_ {
        self.triple_index
            .spo
            .keys()
            .filter_map(move |&(sv, pv, ov)| {
                let st = self.node_table.lookup(sv)?.clone();
                let pt = self.node_table.lookup(pv)?.clone();
                let ot = self.node_table.lookup(ov)?.clone();
                Some((st, pt, ot))
            })
    }

    /// Return a reference to the internal node table.
    pub fn node_table(&self) -> &NodeTable {
        &self.node_table
    }

    /// Return a reference to the internal triple index.
    pub fn triple_index(&self) -> &TripleIndex {
        &self.triple_index
    }

    /// Clear all data from the database.
    pub fn clear(&mut self) {
        self.node_table.clear();
        self.triple_index.clear();
    }
}

impl Default for Tdb2Database {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TripleIndex serialization (for Tdb2Database::save / open)
// ─────────────────────────────────────────────────────────────────────────────

impl TripleIndex {
    /// Serialize the triple index to bytes.
    ///
    /// Format: `[n: u64 LE][s: u64 LE][p: u64 LE][o: u64 LE] * n` (SPO order)
    pub fn serialize(&self) -> Vec<u8> {
        let count = self.spo.len() as u64;
        let mut buf = Vec::with_capacity(8 + count as usize * 24);
        buf.extend_from_slice(&count.to_le_bytes());
        for &(s, p, o) in self.spo.keys() {
            buf.extend_from_slice(&s.to_le_bytes());
            buf.extend_from_slice(&p.to_le_bytes());
            buf.extend_from_slice(&o.to_le_bytes());
        }
        buf
    }

    /// Deserialize a triple index from bytes.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(TdbError::Deserialization(
                "TripleIndex data too short".to_string(),
            ));
        }
        let count = u64::from_le_bytes(
            data[..8]
                .try_into()
                .map_err(|_| TdbError::Deserialization("count slice error".to_string()))?,
        ) as usize;

        let expected_len = 8 + count * 24;
        if data.len() < expected_len {
            return Err(TdbError::Deserialization(format!(
                "TripleIndex data truncated: need {} bytes, got {}",
                expected_len,
                data.len()
            )));
        }

        let mut index = TripleIndex::new();
        let mut pos = 8;
        for _ in 0..count {
            let s = u64::from_le_bytes(
                data[pos..pos + 8]
                    .try_into()
                    .map_err(|_| TdbError::Deserialization("s slice error".to_string()))?,
            );
            let p = u64::from_le_bytes(
                data[pos + 8..pos + 16]
                    .try_into()
                    .map_err(|_| TdbError::Deserialization("p slice error".to_string()))?,
            );
            let o = u64::from_le_bytes(
                data[pos + 16..pos + 24]
                    .try_into()
                    .map_err(|_| TdbError::Deserialization("o slice error".to_string()))?,
            );
            index.add(s, p, o)?;
            pos += 24;
        }

        Ok(index)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tdb2Format
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level TDB2 format wrapper, analogous to TDB2's `DatasetGraphTDB`.
///
/// Provides a high-level entry point with open/new semantics that delegates
/// to [`Tdb2Database`].
#[derive(Debug)]
pub struct Tdb2Format {
    db: Tdb2Database,
    /// Optional directory path (set when opened from disk).
    path: Option<std::path::PathBuf>,
}

impl Tdb2Format {
    /// Create a new, empty in-memory TDB2 format instance.
    pub fn new() -> Self {
        Self {
            db: Tdb2Database::new(),
            path: None,
        }
    }

    /// Open a TDB2 database from a directory path.
    ///
    /// If the directory does not exist or lacks TDB2 files, an empty
    /// database is returned.
    pub fn open(path: &Path) -> Result<Self> {
        let db = Tdb2Database::open(path)?;
        Ok(Self {
            db,
            path: Some(path.to_path_buf()),
        })
    }

    /// Flush changes to disk (if a path was provided at open time).
    pub fn flush(&self) -> Result<()> {
        if let Some(ref p) = self.path {
            self.db.save(p)?;
        }
        Ok(())
    }

    /// Access the underlying database mutably.
    pub fn db_mut(&mut self) -> &mut Tdb2Database {
        &mut self.db
    }

    /// Access the underlying database.
    pub fn db(&self) -> &Tdb2Database {
        &self.db
    }

    /// Convenience: insert a triple.
    pub fn insert_triple(&mut self, s: &RdfTerm, p: &RdfTerm, o: &RdfTerm) -> Result<()> {
        self.db.insert_triple(s, p, o)
    }

    /// Convenience: query triples.
    pub fn query(
        &self,
        s: Option<&RdfTerm>,
        p: Option<&RdfTerm>,
        o: Option<&RdfTerm>,
    ) -> Vec<(RdfTerm, RdfTerm, RdfTerm)> {
        self.db.query(s, p, o)
    }

    /// Convenience: triple count.
    pub fn triple_count(&self) -> u64 {
        self.db.triple_count()
    }

    /// Convenience: node count.
    pub fn node_count(&self) -> u64 {
        self.db.node_count()
    }
}

impl Default for Tdb2Format {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn iri(s: &str) -> RdfTerm {
        RdfTerm::Iri(s.to_string())
    }

    fn lit(v: &str) -> RdfTerm {
        RdfTerm::Literal {
            value: v.to_string(),
            lang: None,
            datatype: None,
        }
    }

    fn lit_lang(v: &str, lang: &str) -> RdfTerm {
        RdfTerm::Literal {
            value: v.to_string(),
            lang: Some(lang.to_string()),
            datatype: None,
        }
    }

    fn lit_dt(v: &str, dt: &str) -> RdfTerm {
        RdfTerm::Literal {
            value: v.to_string(),
            lang: None,
            datatype: Some(dt.to_string()),
        }
    }

    fn blank(id: &str) -> RdfTerm {
        RdfTerm::BlankNode(id.to_string())
    }

    // ── 1. RdfTerm ───────────────────────────────────────────────────────────

    #[test]
    fn test_rdf_term_iri_key() {
        let term = iri("http://example.org/foo");
        assert_eq!(term.term_key(), "<http://example.org/foo>");
        assert!(term.is_iri());
        assert!(!term.is_literal());
        assert!(!term.is_blank_node());
    }

    #[test]
    fn test_rdf_term_literal_key() {
        let term = lit("hello");
        assert_eq!(term.term_key(), "\"hello\"");
        assert!(!term.is_iri());
        assert!(term.is_literal());
    }

    #[test]
    fn test_rdf_term_literal_with_lang_key() {
        let term = lit_lang("hello", "en");
        assert_eq!(term.term_key(), "\"hello\"@en");
    }

    #[test]
    fn test_rdf_term_literal_with_datatype_key() {
        let term = lit_dt("42", "http://www.w3.org/2001/XMLSchema#integer");
        assert_eq!(
            term.term_key(),
            "\"42\"^^http://www.w3.org/2001/XMLSchema#integer"
        );
    }

    #[test]
    fn test_rdf_term_blank_node_key() {
        let term = blank("b1");
        assert_eq!(term.term_key(), "_:b1");
        assert!(term.is_blank_node());
    }

    #[test]
    fn test_rdf_term_equality() {
        assert_eq!(iri("http://a"), iri("http://a"));
        assert_ne!(iri("http://a"), iri("http://b"));
        assert_ne!(iri("http://a"), lit("http://a"));
    }

    // ── 2. NodeTable ─────────────────────────────────────────────────────────

    #[test]
    fn test_node_table_intern_iri() {
        let mut nt = NodeTable::new();
        let t = iri("http://example.org/s");
        let id = nt.intern(&t);
        assert!(id >= 1);
        assert_eq!(nt.lookup(id), Some(&t));
    }

    #[test]
    fn test_node_table_intern_literal() {
        let mut nt = NodeTable::new();
        let t = lit("hello world");
        let id = nt.intern(&t);
        assert_eq!(nt.lookup(id), Some(&t));
    }

    #[test]
    fn test_node_table_intern_blank_node() {
        let mut nt = NodeTable::new();
        let t = blank("b42");
        let id = nt.intern(&t);
        assert_eq!(nt.lookup(id), Some(&t));
    }

    #[test]
    fn test_node_table_deduplication() {
        let mut nt = NodeTable::new();
        let t = iri("http://example.org/same");
        let id1 = nt.intern(&t);
        let id2 = nt.intern(&t);
        assert_eq!(id1, id2, "Same term must get same ID");
    }

    #[test]
    fn test_node_table_ids_start_at_one() {
        let mut nt = NodeTable::new();
        let id = nt.intern(&iri("http://first"));
        assert_eq!(id, 1, "First ID must be 1");
    }

    #[test]
    fn test_node_table_sequential_ids() {
        let mut nt = NodeTable::new();
        let id1 = nt.intern(&iri("http://a"));
        let id2 = nt.intern(&iri("http://b"));
        let id3 = nt.intern(&iri("http://c"));
        assert_eq!(id2, id1 + 1);
        assert_eq!(id3, id2 + 1);
    }

    #[test]
    fn test_node_table_lookup_unknown() {
        let nt = NodeTable::new();
        assert_eq!(nt.lookup(999), None);
    }

    #[test]
    fn test_node_table_lookup_term() {
        let mut nt = NodeTable::new();
        let t = iri("http://example.org/find");
        let id = nt.intern(&t);
        assert_eq!(nt.lookup_term(&t), Some(id));
    }

    #[test]
    fn test_node_table_len() {
        let mut nt = NodeTable::new();
        assert_eq!(nt.len(), 0);
        nt.intern(&iri("http://a"));
        nt.intern(&iri("http://b"));
        nt.intern(&iri("http://a")); // duplicate
        assert_eq!(nt.len(), 2);
    }

    #[test]
    fn test_node_table_serialize_deserialize() {
        let mut nt = NodeTable::new();
        nt.intern(&iri("http://example.org/s"));
        nt.intern(&lit_lang("hello", "en"));
        nt.intern(&blank("b1"));

        let bytes = nt.serialize();
        let nt2 = NodeTable::deserialize(&bytes).expect("deserialize failed");
        assert_eq!(nt2.len(), 3);
        assert!(nt2.lookup_term(&iri("http://example.org/s")).is_some());
    }

    // ── 3. TripleIndex ───────────────────────────────────────────────────────

    #[test]
    fn test_triple_index_add_query_all_bound() {
        let mut idx = TripleIndex::new();
        idx.add(1, 2, 3).unwrap();
        let res = idx.query_spo(Some(1), Some(2), Some(3));
        assert_eq!(res, vec![(1, 2, 3)]);
    }

    #[test]
    fn test_triple_index_query_by_subject() {
        let mut idx = TripleIndex::new();
        idx.add(1, 2, 3).unwrap();
        idx.add(1, 4, 5).unwrap();
        idx.add(9, 2, 3).unwrap();
        let res = idx.query_spo(Some(1), None, None);
        assert_eq!(res.len(), 2);
        assert!(res.contains(&(1, 2, 3)));
        assert!(res.contains(&(1, 4, 5)));
    }

    #[test]
    fn test_triple_index_query_by_predicate() {
        let mut idx = TripleIndex::new();
        idx.add(1, 10, 3).unwrap();
        idx.add(2, 10, 4).unwrap();
        idx.add(3, 20, 5).unwrap();
        let res = idx.query_spo(None, Some(10), None);
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_triple_index_query_by_object() {
        let mut idx = TripleIndex::new();
        idx.add(1, 2, 99).unwrap();
        idx.add(3, 4, 99).unwrap();
        idx.add(5, 6, 77).unwrap();
        let res = idx.query_spo(None, None, Some(99));
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_triple_index_full_scan() {
        let mut idx = TripleIndex::new();
        idx.add(1, 2, 3).unwrap();
        idx.add(4, 5, 6).unwrap();
        let res = idx.query_spo(None, None, None);
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_triple_index_deduplication() {
        let mut idx = TripleIndex::new();
        idx.add(1, 2, 3).unwrap();
        idx.add(1, 2, 3).unwrap(); // duplicate
        assert_eq!(idx.triple_count(), 1);
    }

    #[test]
    fn test_triple_index_remove() {
        let mut idx = TripleIndex::new();
        idx.add(1, 2, 3).unwrap();
        assert!(idx.remove(1, 2, 3));
        assert!(!idx.remove(1, 2, 3)); // already removed
        assert_eq!(idx.triple_count(), 0);
    }

    #[test]
    fn test_triple_index_predicate_object() {
        let mut idx = TripleIndex::new();
        idx.add(1, 5, 10).unwrap();
        idx.add(2, 5, 10).unwrap();
        idx.add(3, 5, 20).unwrap();
        let res = idx.query_spo(None, Some(5), Some(10));
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_triple_index_subject_object() {
        let mut idx = TripleIndex::new();
        idx.add(1, 5, 10).unwrap();
        idx.add(1, 6, 10).unwrap();
        idx.add(2, 5, 10).unwrap();
        let res = idx.query_spo(Some(1), None, Some(10));
        assert_eq!(res.len(), 2);
        assert!(res.contains(&(1, 5, 10)));
        assert!(res.contains(&(1, 6, 10)));
    }

    #[test]
    fn test_triple_index_serialize_deserialize() {
        let mut idx = TripleIndex::new();
        idx.add(1, 2, 3).unwrap();
        idx.add(4, 5, 6).unwrap();

        let bytes = idx.serialize();
        let idx2 = TripleIndex::deserialize(&bytes).expect("deserialize failed");
        assert_eq!(idx2.triple_count(), 2);
    }

    // ── 4. Tdb2Database ──────────────────────────────────────────────────────

    #[test]
    fn test_tdb2_database_insert_query() {
        let mut db = Tdb2Database::new();
        let s = iri("http://s");
        let p = iri("http://p");
        let o = lit("value");
        db.insert_triple(&s, &p, &o).unwrap();

        let res = db.query(Some(&s), None, None);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, s);
        assert_eq!(res[0].1, p);
        assert_eq!(res[0].2, o);
    }

    #[test]
    fn test_tdb2_database_triple_count() {
        let mut db = Tdb2Database::new();
        assert_eq!(db.triple_count(), 0);
        db.insert_triple(&iri("http://s1"), &iri("http://p"), &lit("a"))
            .unwrap();
        db.insert_triple(&iri("http://s2"), &iri("http://p"), &lit("b"))
            .unwrap();
        assert_eq!(db.triple_count(), 2);
    }

    #[test]
    fn test_tdb2_database_node_count() {
        let mut db = Tdb2Database::new();
        db.insert_triple(&iri("http://s"), &iri("http://p"), &lit("o"))
            .unwrap();
        // s, p, o → 3 unique terms
        assert_eq!(db.node_count(), 3);
    }

    #[test]
    fn test_tdb2_database_remove_triple() {
        let mut db = Tdb2Database::new();
        let s = iri("http://s");
        let p = iri("http://p");
        let o = lit("o");
        db.insert_triple(&s, &p, &o).unwrap();
        let removed = db.remove_triple(&s, &p, &o).unwrap();
        assert!(removed);
        assert_eq!(db.triple_count(), 0);
    }

    #[test]
    fn test_tdb2_database_remove_nonexistent() {
        let mut db = Tdb2Database::new();
        let removed = db
            .remove_triple(&iri("http://s"), &iri("http://p"), &lit("o"))
            .unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_tdb2_database_query_by_predicate() {
        let mut db = Tdb2Database::new();
        let p = iri("http://type");
        db.insert_triple(&iri("http://s1"), &p, &iri("http://Class"))
            .unwrap();
        db.insert_triple(&iri("http://s2"), &p, &iri("http://Class"))
            .unwrap();
        db.insert_triple(&iri("http://s3"), &iri("http://other"), &lit("x"))
            .unwrap();
        let res = db.query(None, Some(&p), None);
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_tdb2_database_query_empty_returns_empty() {
        let db = Tdb2Database::new();
        let res = db.query(None, None, None);
        assert!(res.is_empty());
    }

    #[test]
    fn test_tdb2_database_query_unknown_term() {
        let mut db = Tdb2Database::new();
        db.insert_triple(&iri("http://s"), &iri("http://p"), &lit("o"))
            .unwrap();
        let unknown = iri("http://unknown");
        let res = db.query(Some(&unknown), None, None);
        assert!(res.is_empty());
    }

    #[test]
    fn test_tdb2_database_literal_with_lang() {
        let mut db = Tdb2Database::new();
        let s = iri("http://s");
        let p = iri("http://label");
        let o = lit_lang("Hello", "en");
        db.insert_triple(&s, &p, &o).unwrap();
        let res = db.query(None, None, Some(&o));
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn test_tdb2_database_literal_with_datatype() {
        let mut db = Tdb2Database::new();
        let s = iri("http://s");
        let p = iri("http://age");
        let o = lit_dt("42", "http://www.w3.org/2001/XMLSchema#integer");
        db.insert_triple(&s, &p, &o).unwrap();
        let res = db.query(None, None, Some(&o));
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn test_tdb2_database_blank_node() {
        let mut db = Tdb2Database::new();
        let s = blank("b1");
        let p = iri("http://p");
        let o = lit("value");
        db.insert_triple(&s, &p, &o).unwrap();
        let res = db.query(Some(&s), None, None);
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn test_tdb2_database_large_dataset() {
        let mut db = Tdb2Database::new();
        let p = iri("http://hasValue");
        for i in 0..500u64 {
            let s = iri(&format!("http://subject/{}", i));
            let o = lit(&format!("value_{}", i));
            db.insert_triple(&s, &p, &o).unwrap();
        }
        assert_eq!(db.triple_count(), 500);
        // predicate is shared across all triples
        let res = db.query(None, Some(&p), None);
        assert_eq!(res.len(), 500);
    }

    #[test]
    fn test_tdb2_database_iter_triples() {
        let mut db = Tdb2Database::new();
        db.insert_triple(&iri("http://s1"), &iri("http://p"), &lit("a"))
            .unwrap();
        db.insert_triple(&iri("http://s2"), &iri("http://p"), &lit("b"))
            .unwrap();
        let triples: Vec<_> = db.iter_triples().collect();
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_tdb2_database_save_and_open() {
        let tmp_dir = env::temp_dir().join("tdb2_test_save_open");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let mut db = Tdb2Database::new();
        db.insert_triple(&iri("http://s"), &iri("http://p"), &lit("hello"))
            .unwrap();
        db.save(&tmp_dir).unwrap();

        let db2 = Tdb2Database::open(&tmp_dir).unwrap();
        assert_eq!(db2.triple_count(), 1);
        let res = db2.query(None, None, None);
        assert_eq!(res.len(), 1);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // ── 5. Tdb2Format ────────────────────────────────────────────────────────

    #[test]
    fn test_tdb2_format_new() {
        let fmt = Tdb2Format::new();
        assert_eq!(fmt.triple_count(), 0);
        assert_eq!(fmt.node_count(), 0);
    }

    #[test]
    fn test_tdb2_format_insert_query() {
        let mut fmt = Tdb2Format::new();
        fmt.insert_triple(&iri("http://s"), &iri("http://p"), &lit("o"))
            .unwrap();
        assert_eq!(fmt.triple_count(), 1);
        let res = fmt.query(None, None, None);
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn test_tdb2_format_open_empty_dir() {
        let tmp_dir = env::temp_dir().join("tdb2_format_empty");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let fmt = Tdb2Format::open(&tmp_dir).unwrap();
        assert_eq!(fmt.triple_count(), 0);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_tdb2_format_flush() {
        let tmp_dir = env::temp_dir().join("tdb2_format_flush");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let mut fmt = Tdb2Format::open(&tmp_dir).unwrap();
        fmt.insert_triple(&iri("http://a"), &iri("http://b"), &lit("c"))
            .unwrap();
        fmt.flush().unwrap();

        let fmt2 = Tdb2Format::open(&tmp_dir).unwrap();
        assert_eq!(fmt2.triple_count(), 1);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_tdb2_database_deduplication() {
        let mut db = Tdb2Database::new();
        let s = iri("http://s");
        let p = iri("http://p");
        let o = lit("o");
        db.insert_triple(&s, &p, &o).unwrap();
        db.insert_triple(&s, &p, &o).unwrap(); // duplicate
        assert_eq!(db.triple_count(), 1);
        assert_eq!(db.node_count(), 3); // still 3 unique terms
    }
}
