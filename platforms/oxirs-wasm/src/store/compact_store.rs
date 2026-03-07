//! Memory-efficient triple store for WebAssembly
//!
//! Uses integer dictionaries (string interning) and sorted Vec-based indexes
//! to minimize memory consumption within WASM's 4 GB address space.
//!
//! Design choices:
//! - `NodeId` is `u32` (not `u64`) to halve index entry size in 32-bit WASM
//! - Triples stored as `(NodeId, NodeId, NodeId)` tuples: 12 bytes per triple
//! - Two sorted indexes: SPO (primary) and PSO (predicate-oriented lookups)
//! - Dictionary stores term strings once; IDs are used everywhere else

use std::collections::HashMap;

/// Compact node identifier – 32-bit for WASM compactness
pub type NodeId = u32;

/// The null/sentinel node ID (used internally only)
const NULL_ID: NodeId = u32::MAX;

/// An RDF term: IRI, blank node, or literal
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfTerm {
    /// An IRI reference, e.g. `http://example.org/alice`
    Iri(String),
    /// A blank node, e.g. `b1`
    BlankNode(String),
    /// A plain (untyped) literal
    PlainLiteral(String),
    /// A language-tagged literal
    LangLiteral { value: String, lang: String },
    /// A datatype-tagged literal
    TypedLiteral { value: String, datatype: String },
}

impl RdfTerm {
    /// Serialize to a canonical string key for dictionary lookup
    pub fn canonical_key(&self) -> String {
        match self {
            RdfTerm::Iri(iri) => format!("I:{}", iri),
            RdfTerm::BlankNode(id) => format!("B:{}", id),
            RdfTerm::PlainLiteral(v) => format!("L:{}", v),
            RdfTerm::LangLiteral { value, lang } => format!("LL:{}@{}", value, lang),
            RdfTerm::TypedLiteral { value, datatype } => {
                format!("LT:{}^^{}", value, datatype)
            }
        }
    }

    /// Create an IRI term
    pub fn iri(s: impl Into<String>) -> Self {
        RdfTerm::Iri(s.into())
    }

    /// Create a blank node term
    pub fn blank(id: impl Into<String>) -> Self {
        RdfTerm::BlankNode(id.into())
    }

    /// Create a plain literal term
    pub fn literal(v: impl Into<String>) -> Self {
        RdfTerm::PlainLiteral(v.into())
    }

    /// Create a language-tagged literal
    pub fn lang_literal(v: impl Into<String>, lang: impl Into<String>) -> Self {
        RdfTerm::LangLiteral {
            value: v.into(),
            lang: lang.into(),
        }
    }

    /// Create a typed literal
    pub fn typed_literal(v: impl Into<String>, datatype: impl Into<String>) -> Self {
        RdfTerm::TypedLiteral {
            value: v.into(),
            datatype: datatype.into(),
        }
    }

    /// Return the raw value string (IRI, blank node ID, or literal value)
    pub fn value(&self) -> &str {
        match self {
            RdfTerm::Iri(s) => s,
            RdfTerm::BlankNode(s) => s,
            RdfTerm::PlainLiteral(s) => s,
            RdfTerm::LangLiteral { value, .. } => value,
            RdfTerm::TypedLiteral { value, .. } => value,
        }
    }

    /// Return the datatype IRI, if any
    pub fn datatype(&self) -> Option<&str> {
        match self {
            RdfTerm::TypedLiteral { datatype, .. } => Some(datatype),
            _ => None,
        }
    }

    /// Return the language tag, if any
    pub fn lang(&self) -> Option<&str> {
        match self {
            RdfTerm::LangLiteral { lang, .. } => Some(lang),
            _ => None,
        }
    }

    /// Return true if this is an IRI
    pub fn is_iri(&self) -> bool {
        matches!(self, RdfTerm::Iri(_))
    }

    /// Return true if this is a blank node
    pub fn is_blank_node(&self) -> bool {
        matches!(self, RdfTerm::BlankNode(_))
    }

    /// Return true if this is any kind of literal
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            RdfTerm::PlainLiteral(_) | RdfTerm::LangLiteral { .. } | RdfTerm::TypedLiteral { .. }
        )
    }
}

impl std::fmt::Display for RdfTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfTerm::Iri(iri) => write!(f, "<{}>", iri),
            RdfTerm::BlankNode(id) => write!(f, "_:{}", id),
            RdfTerm::PlainLiteral(v) => write!(f, "\"{}\"", v),
            RdfTerm::LangLiteral { value, lang } => write!(f, "\"{}\"@{}", value, lang),
            RdfTerm::TypedLiteral { value, datatype } => {
                write!(f, "\"{}\"^^<{}>", value, datatype)
            }
        }
    }
}

// -----------------------------------------------------------------------
// CompactDictionary
// -----------------------------------------------------------------------

/// String-to-integer dictionary for RDF term interning.
///
/// Guarantees:
/// - Each unique term gets a stable, monotonically assigned `NodeId`
/// - Lookup is O(1) via HashMap
/// - All IDs start at 0; the maximum is `u32::MAX - 1`
#[derive(Debug, Clone)]
pub struct CompactDictionary {
    /// Forward map: canonical key string → ID
    str_to_id: HashMap<String, NodeId>,
    /// Backward map: ID → owned RdfTerm
    id_to_term: Vec<RdfTerm>,
    /// Next free ID
    next_id: NodeId,
}

impl CompactDictionary {
    /// Create an empty dictionary
    pub fn new() -> Self {
        Self {
            str_to_id: HashMap::new(),
            id_to_term: Vec::new(),
            next_id: 0,
        }
    }

    /// Intern a term, returning its ID (inserting if not present)
    pub fn intern(&mut self, term: &RdfTerm) -> NodeId {
        let key = term.canonical_key();
        if let Some(&id) = self.str_to_id.get(&key) {
            return id;
        }
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("NodeId overflow (> 4 billion unique terms)");
        self.str_to_id.insert(key, id);
        self.id_to_term.push(term.clone());
        id
    }

    /// Look up a term by its ID, returning None if the ID is out of range
    pub fn lookup(&self, id: NodeId) -> Option<&RdfTerm> {
        self.id_to_term.get(id as usize)
    }

    /// Get the ID of a term without inserting, returning None if not found
    pub fn get_id(&self, term: &RdfTerm) -> Option<NodeId> {
        let key = term.canonical_key();
        self.str_to_id.get(&key).copied()
    }

    /// Number of unique terms stored
    pub fn size(&self) -> usize {
        self.id_to_term.len()
    }

    /// Approximate memory usage in bytes
    pub fn memory_estimate_bytes(&self) -> usize {
        // Each entry: ~50 bytes avg for string key + 8 bytes for hash entry + term storage
        self.id_to_term
            .iter()
            .map(|t| {
                let key_len = t.canonical_key().len();
                let term_len = match t {
                    RdfTerm::Iri(s) => s.len() + 8,
                    RdfTerm::BlankNode(s) => s.len() + 8,
                    RdfTerm::PlainLiteral(s) => s.len() + 8,
                    RdfTerm::LangLiteral { value, lang } => value.len() + lang.len() + 16,
                    RdfTerm::TypedLiteral { value, datatype } => value.len() + datatype.len() + 16,
                };
                key_len + term_len + 16 // HashMap overhead estimate
            })
            .sum()
    }
}

impl Default for CompactDictionary {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------
// CompactTripleStore
// -----------------------------------------------------------------------

/// Memory-efficient RDF triple store for WebAssembly.
///
/// All strings are interned via [`CompactDictionary`], and triples are stored
/// as `(NodeId, NodeId, NodeId)` tuples (12 bytes each). Two sorted indexes
/// enable efficient subject-based (SPO) and predicate-based (PSO) access.
///
/// # Complexity
/// | Operation            | Complexity        |
/// |----------------------|-------------------|
/// | Insert               | O(log n) amortized|
/// | Contains             | O(log n)          |
/// | Delete               | O(log n)          |
/// | Find by subject      | O(log n + k)      |
/// | Find by predicate    | O(log n + k)      |
/// | Find p+o             | O(log n + k)      |
///
/// `n` = number of triples, `k` = number of matching results.
#[derive(Debug, Clone)]
pub struct CompactTripleStore {
    /// Term dictionary (shared for all three positions)
    dict: CompactDictionary,
    /// SPO-sorted triples (primary index)
    spo: Vec<(NodeId, NodeId, NodeId)>,
    /// PSO-sorted triples (predicate-oriented index)
    pso: Vec<(NodeId, NodeId, NodeId)>,
    /// Whether the indexes need re-sorting after bulk inserts
    dirty: bool,
}

impl CompactTripleStore {
    /// Create a new empty store
    pub fn new() -> Self {
        Self {
            dict: CompactDictionary::new(),
            spo: Vec::new(),
            pso: Vec::new(),
            dirty: false,
        }
    }

    /// Insert a triple into the store.
    /// Duplicates are silently ignored.
    pub fn insert(&mut self, s: &RdfTerm, p: &RdfTerm, o: &RdfTerm) {
        let s_id = self.dict.intern(s);
        let p_id = self.dict.intern(p);
        let o_id = self.dict.intern(o);
        let triple = (s_id, p_id, o_id);

        self.ensure_sorted();

        // Check for duplicate in SPO index
        if self.spo.binary_search(&triple).is_ok() {
            return; // Already present
        }

        self.spo.push(triple);
        self.pso.push((p_id, s_id, o_id));
        self.dirty = true;
    }

    /// Delete a triple from the store.
    /// Returns `true` if the triple was found and removed.
    pub fn delete(&mut self, s: &RdfTerm, p: &RdfTerm, o: &RdfTerm) -> bool {
        let s_id = match self.dict.get_id(s) {
            Some(id) => id,
            None => return false,
        };
        let p_id = match self.dict.get_id(p) {
            Some(id) => id,
            None => return false,
        };
        let o_id = match self.dict.get_id(o) {
            Some(id) => id,
            None => return false,
        };

        self.ensure_sorted();

        let triple_spo = (s_id, p_id, o_id);
        match self.spo.binary_search(&triple_spo) {
            Ok(pos) => {
                self.spo.remove(pos);
                // Remove from PSO as well (linear scan – acceptable since deletion is rare)
                let triple_pso = (p_id, s_id, o_id);
                if let Ok(ppos) = self.pso.binary_search(&triple_pso) {
                    self.pso.remove(ppos);
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Check whether a triple exists in the store
    pub fn contains(&self, s: &RdfTerm, p: &RdfTerm, o: &RdfTerm) -> bool {
        let s_id = match self.dict.get_id(s) {
            Some(id) => id,
            None => return false,
        };
        let p_id = match self.dict.get_id(p) {
            Some(id) => id,
            None => return false,
        };
        let o_id = match self.dict.get_id(o) {
            Some(id) => id,
            None => return false,
        };

        // Use binary search; but we might be dirty so try linear if needed
        if self.dirty {
            return self.spo.contains(&(s_id, p_id, o_id));
        }

        self.spo.binary_search(&(s_id, p_id, o_id)).is_ok()
    }

    /// Return the number of triples stored
    pub fn triple_count(&self) -> usize {
        self.spo.len()
    }

    /// Return the number of unique terms (subjects + predicates + objects)
    pub fn term_count(&self) -> usize {
        self.dict.size()
    }

    /// Find all triples matching the given subject
    pub fn find_by_subject(&mut self, s: &RdfTerm) -> Vec<(RdfTerm, RdfTerm, RdfTerm)> {
        let s_id = match self.dict.get_id(s) {
            Some(id) => id,
            None => return vec![],
        };

        self.ensure_sorted();

        // Binary search for the first entry with this subject
        let first = match self.spo.binary_search_by(|&(si, _, _)| si.cmp(&s_id)) {
            Ok(pos) => {
                // Back up to the first matching
                let mut start = pos;
                while start > 0 && self.spo[start - 1].0 == s_id {
                    start -= 1;
                }
                start
            }
            Err(_) => return vec![],
        };

        let mut result = Vec::new();
        for &(si, pi, oi) in &self.spo[first..] {
            if si != s_id {
                break;
            }
            if let (Some(subj), Some(pred), Some(obj)) = (
                self.dict.lookup(si),
                self.dict.lookup(pi),
                self.dict.lookup(oi),
            ) {
                result.push((subj.clone(), pred.clone(), obj.clone()));
            }
        }
        result
    }

    /// Find all triples matching the given predicate
    pub fn find_by_predicate(&mut self, p: &RdfTerm) -> Vec<(RdfTerm, RdfTerm, RdfTerm)> {
        let p_id = match self.dict.get_id(p) {
            Some(id) => id,
            None => return vec![],
        };

        self.ensure_sorted();

        let first = match self.pso.binary_search_by(|&(pi, _, _)| pi.cmp(&p_id)) {
            Ok(pos) => {
                let mut start = pos;
                while start > 0 && self.pso[start - 1].0 == p_id {
                    start -= 1;
                }
                start
            }
            Err(_) => return vec![],
        };

        let mut result = Vec::new();
        for &(pi, si, oi) in &self.pso[first..] {
            if pi != p_id {
                break;
            }
            if let (Some(subj), Some(pred), Some(obj)) = (
                self.dict.lookup(si),
                self.dict.lookup(pi),
                self.dict.lookup(oi),
            ) {
                result.push((subj.clone(), pred.clone(), obj.clone()));
            }
        }
        result
    }

    /// Find all subjects that have a given predicate-object pair
    pub fn find_by_predicate_object(&mut self, p: &RdfTerm, o: &RdfTerm) -> Vec<RdfTerm> {
        let p_id = match self.dict.get_id(p) {
            Some(id) => id,
            None => return vec![],
        };
        let o_id = match self.dict.get_id(o) {
            Some(id) => id,
            None => return vec![],
        };

        self.ensure_sorted();

        // Scan PSO index for matching (p, *, o)
        let first = match self.pso.binary_search_by(|&(pi, _, _)| pi.cmp(&p_id)) {
            Ok(pos) => {
                let mut start = pos;
                while start > 0 && self.pso[start - 1].0 == p_id {
                    start -= 1;
                }
                start
            }
            Err(_) => return vec![],
        };

        let mut result = Vec::new();
        for &(pi, si, oi) in &self.pso[first..] {
            if pi != p_id {
                break;
            }
            if oi == o_id {
                if let Some(subj) = self.dict.lookup(si) {
                    result.push(subj.clone());
                }
            }
        }
        result
    }

    /// Iterate over all triples in SPO order
    pub fn iter_all(&mut self) -> impl Iterator<Item = (RdfTerm, RdfTerm, RdfTerm)> + '_ {
        self.ensure_sorted();
        let dict = &self.dict;
        self.spo.iter().filter_map(move |&(si, pi, oi)| {
            let s = dict.lookup(si)?.clone();
            let p = dict.lookup(pi)?.clone();
            let o = dict.lookup(oi)?.clone();
            Some((s, p, o))
        })
    }

    /// Approximate memory usage in bytes
    pub fn memory_estimate_bytes(&self) -> usize {
        let triple_bytes = (self.spo.len() + self.pso.len()) * 12; // 3 × u32 per triple
        let dict_bytes = self.dict.memory_estimate_bytes();
        triple_bytes + dict_bytes
    }

    /// Get a reference to the dictionary (for advanced use)
    pub fn dictionary(&self) -> &CompactDictionary {
        &self.dict
    }

    /// Sort both indexes if they have been modified (lazy sort)
    fn ensure_sorted(&mut self) {
        if self.dirty {
            self.spo.sort_unstable();
            self.pso.sort_unstable();
            self.dirty = false;
        }
    }

    /// Bulk-insert from an iterator of (subject, predicate, object) triples.
    /// More efficient than individual inserts because sorting happens once.
    pub fn bulk_insert<I>(&mut self, triples: I)
    where
        I: IntoIterator<Item = (RdfTerm, RdfTerm, RdfTerm)>,
    {
        for (s, p, o) in triples {
            let s_id = self.dict.intern(&s);
            let p_id = self.dict.intern(&p);
            let o_id = self.dict.intern(&o);
            self.spo.push((s_id, p_id, o_id));
            self.pso.push((p_id, s_id, o_id));
        }
        self.dirty = true;
        // Deduplicate after sort
        self.ensure_sorted();
        self.spo.dedup();
        self.pso.dedup();
    }
}

impl Default for CompactTripleStore {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(local: &str) -> RdfTerm {
        RdfTerm::iri(format!("http://example.org/{}", local))
    }

    fn lit(v: &str) -> RdfTerm {
        RdfTerm::literal(v)
    }

    #[test]
    fn test_insert_and_count() {
        let mut store = CompactTripleStore::new();
        store.insert(&ex("alice"), &ex("knows"), &ex("bob"));
        store.insert(&ex("alice"), &ex("name"), &lit("Alice"));
        assert_eq!(store.triple_count(), 2);
    }

    #[test]
    fn test_insert_duplicate() {
        let mut store = CompactTripleStore::new();
        store.insert(&ex("s"), &ex("p"), &ex("o"));
        store.insert(&ex("s"), &ex("p"), &ex("o")); // Duplicate
        assert_eq!(store.triple_count(), 1);
    }

    #[test]
    fn test_contains() {
        let mut store = CompactTripleStore::new();
        store.insert(&ex("s"), &ex("p"), &ex("o"));
        assert!(store.contains(&ex("s"), &ex("p"), &ex("o")));
        assert!(!store.contains(&ex("s"), &ex("p"), &ex("other")));
    }

    #[test]
    fn test_delete() {
        let mut store = CompactTripleStore::new();
        store.insert(&ex("s"), &ex("p"), &ex("o"));
        assert_eq!(store.triple_count(), 1);

        let deleted = store.delete(&ex("s"), &ex("p"), &ex("o"));
        assert!(deleted);
        assert_eq!(store.triple_count(), 0);

        let again = store.delete(&ex("s"), &ex("p"), &ex("o"));
        assert!(!again);
    }

    #[test]
    fn test_find_by_subject() {
        let mut store = CompactTripleStore::new();
        store.insert(&ex("alice"), &ex("knows"), &ex("bob"));
        store.insert(&ex("alice"), &ex("name"), &lit("Alice"));
        store.insert(&ex("bob"), &ex("name"), &lit("Bob"));

        let results = store.find_by_subject(&ex("alice"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_by_predicate() {
        let mut store = CompactTripleStore::new();
        store.insert(&ex("alice"), &ex("knows"), &ex("bob"));
        store.insert(&ex("alice"), &ex("knows"), &ex("carol"));
        store.insert(&ex("alice"), &ex("name"), &lit("Alice"));

        let results = store.find_by_predicate(&ex("knows"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_by_predicate_object() {
        let mut store = CompactTripleStore::new();
        store.insert(&ex("alice"), &ex("knows"), &ex("bob"));
        store.insert(&ex("carol"), &ex("knows"), &ex("bob"));
        store.insert(&ex("alice"), &ex("knows"), &ex("dave"));

        let subjects = store.find_by_predicate_object(&ex("knows"), &ex("bob"));
        assert_eq!(subjects.len(), 2);
    }

    #[test]
    fn test_bulk_insert() {
        let mut store = CompactTripleStore::new();
        let triples: Vec<(RdfTerm, RdfTerm, RdfTerm)> = (0..100)
            .map(|i| (ex(&format!("s{}", i)), ex("p"), ex(&format!("o{}", i))))
            .collect();

        store.bulk_insert(triples);
        assert_eq!(store.triple_count(), 100);
    }

    #[test]
    fn test_memory_estimate() {
        let mut store = CompactTripleStore::new();
        store.insert(&ex("s"), &ex("p"), &ex("o"));

        let mem = store.memory_estimate_bytes();
        assert!(mem > 0);
    }

    #[test]
    fn test_dictionary_intern() {
        let mut dict = CompactDictionary::new();
        let term = RdfTerm::iri("http://example.org/");
        let id1 = dict.intern(&term);
        let id2 = dict.intern(&term); // Same term again
        assert_eq!(id1, id2);
        assert_eq!(dict.size(), 1);
    }

    #[test]
    fn test_dictionary_lookup() {
        let mut dict = CompactDictionary::new();
        let term = RdfTerm::LangLiteral {
            value: "hello".to_string(),
            lang: "en".to_string(),
        };
        let id = dict.intern(&term);
        let retrieved = dict.lookup(id).expect("term should be found");
        assert_eq!(retrieved, &term);
    }

    #[test]
    fn test_rdf_term_kinds() {
        let iri = RdfTerm::iri("http://example.org/");
        assert!(iri.is_iri());
        assert!(!iri.is_blank_node());
        assert!(!iri.is_literal());

        let bnode = RdfTerm::blank("b0");
        assert!(bnode.is_blank_node());

        let plain_lit = RdfTerm::literal("hello");
        assert!(plain_lit.is_literal());

        let lang_lit = RdfTerm::lang_literal("hello", "en");
        assert!(lang_lit.is_literal());
        assert_eq!(lang_lit.lang(), Some("en"));

        let typed_lit = RdfTerm::typed_literal("42", "http://www.w3.org/2001/XMLSchema#integer");
        assert!(typed_lit.is_literal());
        assert_eq!(
            typed_lit.datatype(),
            Some("http://www.w3.org/2001/XMLSchema#integer")
        );
    }

    #[test]
    fn test_rdf_term_display() {
        let iri = RdfTerm::iri("http://example.org/");
        assert_eq!(format!("{}", iri), "<http://example.org/>");

        let bnode = RdfTerm::blank("b0");
        assert_eq!(format!("{}", bnode), "_:b0");

        let plain_lit = RdfTerm::literal("hello");
        assert_eq!(format!("{}", plain_lit), "\"hello\"");
    }
}
