//! Compressed dictionary for RDF-star terms, including quoted triples.
//!
//! The `StarDictionary` compresses the string space of RDF-star data by
//! assigning monotonically increasing integer IDs to each distinct term string.
//! Quoted triples receive their own ID namespace and are stored by their
//! component IDs, avoiding redundant string storage.
//!
//! # Memory layout
//!
//! | Component | Storage |
//! |-----------|---------|
//! | Simple terms (IRI, literal, blank node) | `str_to_id` + `id_to_str` |
//! | Quoted triples | `quoted_to_id` + `id_to_quoted` |
//!
//! The `EncodedTerm` enum tags each ID with its kind so that decoding is
//! unambiguous.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_star::storage::star_dict::{StarDictionary, EncodedTerm};
//! use oxirs_star::{StarTerm, StarTriple};
//!
//! let mut dict = StarDictionary::new();
//!
//! let alice = StarTerm::iri("http://ex.org/alice").unwrap();
//! let encoded = dict.encode_term(&alice).unwrap();
//!
//! if let EncodedTerm::Iri(id) = encoded {
//!     println!("Alice IRI has ID: {}", id);
//! }
//!
//! let decoded = dict.decode_term(&encoded).unwrap();
//! assert_eq!(decoded, alice);
//! ```

use std::collections::HashMap;

use crate::{StarError, StarResult, StarTerm, StarTriple};

// ---- Type aliases ----------------------------------------------------------

/// Opaque identifier for a simple term (IRI, blank node, or literal).
pub type TermId = u64;

/// Opaque identifier for a quoted triple.
pub type QuotedId = u64;

// ---- EncodedTerm -----------------------------------------------------------

/// A compressed representation of an RDF-star term.
///
/// Each variant carries the appropriate numeric ID rather than the raw string,
/// reducing memory usage for datasets with repeated terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EncodedTerm {
    /// Compressed IRI: the ID maps to the IRI string in the dictionary.
    Iri(TermId),
    /// Compressed blank node: the ID maps to the blank-node label.
    BlankNode(TermId),
    /// Compressed literal: the ID maps to the full literal representation.
    Literal(TermId),
    /// Compressed quoted triple: the ID maps to a `(s, p, o)` triple of
    /// `EncodedTerm`s in the quoted-triple table.
    QuotedTriple(QuotedId),
}

impl EncodedTerm {
    /// Return `true` if this is a quoted triple.
    pub fn is_quoted(&self) -> bool {
        matches!(self, Self::QuotedTriple(_))
    }

    /// Return the numeric ID regardless of kind.
    pub fn raw_id(&self) -> u64 {
        match self {
            Self::Iri(id) | Self::BlankNode(id) | Self::Literal(id) => *id,
            Self::QuotedTriple(id) => *id,
        }
    }
}

// ---- TermKey ----------------------------------------------------------------

/// Internal string key used for term deduplication.
///
/// Includes a kind discriminant so that an IRI `"foo"` and a literal `"foo"`
/// receive different IDs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TermKey {
    Iri(String),
    BlankNode(String),
    /// Stores the canonical form: `"lexical"^^<datatype>@lang`.
    Literal(String),
}

impl TermKey {
    fn from_term(term: &StarTerm) -> Option<Self> {
        match term {
            StarTerm::NamedNode(n) => Some(Self::Iri(n.iri.clone())),
            StarTerm::BlankNode(b) => Some(Self::BlankNode(b.id.clone())),
            StarTerm::Literal(l) => {
                // Canonical form includes datatype and language tag.
                let mut repr = l.value.clone();
                if let Some(ref dt) = l.datatype {
                    repr.push_str("^^");
                    repr.push_str(&dt.iri);
                }
                if let Some(ref lang) = l.language {
                    repr.push('@');
                    repr.push_str(lang);
                }
                Some(Self::Literal(repr))
            }
            StarTerm::Variable(_) | StarTerm::QuotedTriple(_) => None,
        }
    }
}

// ---- StarDictionary --------------------------------------------------------

/// Bidirectional dictionary for RDF-star terms.
///
/// Provides O(1) amortised encoding and O(1) decoding.  The dictionary grows
/// monotonically; there is no removal operation, matching typical triple-store
/// bulk-load usage patterns.
pub struct StarDictionary {
    /// String-key → ID for simple terms.
    str_to_id: HashMap<TermKey, TermId>,
    /// ID → canonical string for simple terms.  Index 0 is unused (IDs start
    /// at 1) so that 0 can act as a sentinel.
    id_to_str: Vec<(TermKey, String)>,
    next_term_id: TermId,

    /// (s_encoded, p_encoded, o_encoded) → quoted ID.
    quoted_to_id: HashMap<(EncodedTerm, EncodedTerm, EncodedTerm), QuotedId>,
    /// ID → component triple of `EncodedTerm`s.  Index 0 is unused.
    id_to_quoted: Vec<(EncodedTerm, EncodedTerm, EncodedTerm)>,
    next_quoted_id: QuotedId,
}

impl StarDictionary {
    /// Create an empty dictionary.  IDs for both namespaces start at 1.
    pub fn new() -> Self {
        // Pre-allocate slot 0 as a dummy so IDs start at 1.
        Self {
            str_to_id: HashMap::new(),
            id_to_str: vec![(TermKey::Iri(String::new()), String::new())],
            next_term_id: 1,

            quoted_to_id: HashMap::new(),
            id_to_quoted: vec![(
                EncodedTerm::Iri(0),
                EncodedTerm::Iri(0),
                EncodedTerm::Iri(0),
            )],
            next_quoted_id: 1,
        }
    }

    // ---- Encoding -----------------------------------------------------------

    /// Encode an RDF-star term, assigning a new ID if this term has not been
    /// seen before.
    ///
    /// Variables are not encodable and return an error.
    pub fn encode_term(&mut self, term: &StarTerm) -> StarResult<EncodedTerm> {
        match term {
            StarTerm::Variable(v) => Err(StarError::invalid_term_type(format!(
                "Variables cannot be encoded in the dictionary: ?{}",
                v.name
            ))),

            StarTerm::QuotedTriple(inner) => {
                // Recursively encode components, then intern the triple.
                let s = self.encode_term(&inner.subject)?;
                let p = self.encode_term(&inner.predicate)?;
                let o = self.encode_term(&inner.object)?;
                Ok(self.intern_quoted(s, p, o))
            }

            StarTerm::NamedNode(n) => {
                let key = TermKey::Iri(n.iri.clone());
                Ok(EncodedTerm::Iri(self.intern_str(key, &n.iri)))
            }
            StarTerm::BlankNode(b) => {
                let key = TermKey::BlankNode(b.id.clone());
                Ok(EncodedTerm::BlankNode(self.intern_str(key, &b.id)))
            }
            StarTerm::Literal(_l) => {
                let tk = TermKey::from_term(term).expect("literal always has a term key");
                let repr = match tk.clone() {
                    TermKey::Literal(s) => s,
                    _ => unreachable!(),
                };
                Ok(EncodedTerm::Literal(self.intern_str(tk, &repr)))
            }
        }
    }

    /// Encode all three terms of a triple.
    pub fn encode_triple(
        &mut self,
        triple: &StarTriple,
    ) -> StarResult<(EncodedTerm, EncodedTerm, EncodedTerm)> {
        let s = self.encode_term(&triple.subject)?;
        let p = self.encode_term(&triple.predicate)?;
        let o = self.encode_term(&triple.object)?;
        Ok((s, p, o))
    }

    // ---- Decoding -----------------------------------------------------------

    /// Decode an `EncodedTerm` back to a `StarTerm`.
    ///
    /// Returns `None` if the ID does not exist in the dictionary.
    pub fn decode_term(&self, encoded: &EncodedTerm) -> Option<StarTerm> {
        match encoded {
            EncodedTerm::Iri(id) => {
                let (key, _) = self.id_to_str.get(*id as usize)?;
                if let TermKey::Iri(iri) = key {
                    StarTerm::iri(iri).ok()
                } else {
                    None
                }
            }
            EncodedTerm::BlankNode(id) => {
                let (key, _) = self.id_to_str.get(*id as usize)?;
                if let TermKey::BlankNode(bn) = key {
                    StarTerm::blank_node(bn).ok()
                } else {
                    None
                }
            }
            EncodedTerm::Literal(id) => {
                let (key, repr) = self.id_to_str.get(*id as usize)?;
                if let TermKey::Literal(_) = key {
                    // Parse the canonical form back to a Literal.
                    Self::decode_literal_repr(repr)
                } else {
                    None
                }
            }
            EncodedTerm::QuotedTriple(id) => {
                let (s_enc, p_enc, o_enc) = self.id_to_quoted.get(*id as usize)?;
                let s = self.decode_term(s_enc)?;
                let p = self.decode_term(p_enc)?;
                let o = self.decode_term(o_enc)?;
                Some(StarTerm::quoted_triple(StarTriple::new(s, p, o)))
            }
        }
    }

    // ---- Statistics ---------------------------------------------------------

    /// Number of simple terms (IRIs + blank nodes + literals) in the
    /// dictionary.
    pub fn term_count(&self) -> usize {
        // Subtract 1 for the dummy slot 0.
        self.id_to_str.len().saturating_sub(1)
    }

    /// Number of quoted triples in the dictionary.
    pub fn quoted_count(&self) -> usize {
        self.id_to_quoted.len().saturating_sub(1)
    }

    /// Approximate heap memory usage in bytes.
    ///
    /// This is an estimate based on the sizes of the stored strings and
    /// hash-map overhead.
    pub fn memory_bytes(&self) -> usize {
        // String storage: sum of all stored string bytes.
        let str_bytes: usize = self
            .id_to_str
            .iter()
            .map(|(_, s)| s.len() + std::mem::size_of::<String>())
            .sum();

        // HashMap overhead: ~64 bytes per entry (key + value + metadata).
        let str_map_bytes = self.str_to_id.len() * 64;

        // Quoted triple storage: fixed-size tuples.
        let quoted_bytes = self.id_to_quoted.len() * 3 * std::mem::size_of::<EncodedTerm>();
        let quoted_map_bytes = self.quoted_to_id.len() * 64;

        str_bytes + str_map_bytes + quoted_bytes + quoted_map_bytes
    }

    /// Return the next term ID that would be assigned (useful for testing).
    pub fn next_term_id(&self) -> TermId {
        self.next_term_id
    }

    /// Return the next quoted ID that would be assigned.
    pub fn next_quoted_id(&self) -> QuotedId {
        self.next_quoted_id
    }

    // ---- Private helpers ----------------------------------------------------

    /// Intern a simple-term string and return its ID.  If the key already
    /// exists the existing ID is returned without inserting a duplicate.
    fn intern_str(&mut self, key: TermKey, repr: &str) -> TermId {
        if let Some(&existing) = self.str_to_id.get(&key) {
            return existing;
        }
        let id = self.next_term_id;
        self.str_to_id.insert(key.clone(), id);
        self.id_to_str.push((key, repr.to_owned()));
        self.next_term_id += 1;
        id
    }

    /// Intern a quoted triple (s, p, o) encoded tuple and return its ID.
    fn intern_quoted(&mut self, s: EncodedTerm, p: EncodedTerm, o: EncodedTerm) -> EncodedTerm {
        let key = (s.clone(), p.clone(), o.clone());
        if let Some(&existing) = self.quoted_to_id.get(&key) {
            return EncodedTerm::QuotedTriple(existing);
        }
        let id = self.next_quoted_id;
        self.quoted_to_id.insert(key, id);
        self.id_to_quoted.push((s, p, o));
        self.next_quoted_id += 1;
        EncodedTerm::QuotedTriple(id)
    }

    /// Parse a literal canonical representation of the form
    /// `lexical^^<datatype>@lang` back to a `StarTerm`.
    fn decode_literal_repr(repr: &str) -> Option<StarTerm> {
        // Split on `^^` first (datatype), then `@` (language tag).
        if let Some(dt_pos) = repr.find("^^") {
            let value = &repr[..dt_pos];
            let datatype = &repr[dt_pos + 2..];
            StarTerm::literal_with_datatype(value, datatype).ok()
        } else if let Some(lang_pos) = repr.rfind('@') {
            let value = &repr[..lang_pos];
            let lang = &repr[lang_pos + 1..];
            StarTerm::literal_with_language(value, lang).ok()
        } else {
            StarTerm::literal(repr).ok()
        }
    }
}

impl Default for StarDictionary {
    fn default() -> Self {
        Self::new()
    }
}

// ---- CompressedTripleSet ---------------------------------------------------

/// A compact storage structure for encoded triples.
///
/// Triples are stored as `(EncodedTerm, EncodedTerm, EncodedTerm)` tuples,
/// dramatically reducing memory compared to storing full string terms.
pub struct CompressedTripleSet {
    dict: StarDictionary,
    triples: Vec<(EncodedTerm, EncodedTerm, EncodedTerm)>,
}

impl CompressedTripleSet {
    /// Create an empty compressed triple set.
    pub fn new() -> Self {
        Self {
            dict: StarDictionary::new(),
            triples: Vec::new(),
        }
    }

    /// Insert a triple, encoding it via the dictionary.
    pub fn insert(&mut self, triple: &StarTriple) -> StarResult<()> {
        let encoded = self.dict.encode_triple(triple)?;
        self.triples.push(encoded);
        Ok(())
    }

    /// Decode and return all triples.
    pub fn iter_triples(&self) -> impl Iterator<Item = Option<StarTriple>> + '_ {
        self.triples.iter().map(move |(s, p, o)| {
            let s_term = self.dict.decode_term(s)?;
            let p_term = self.dict.decode_term(p)?;
            let o_term = self.dict.decode_term(o)?;
            Some(StarTriple::new(s_term, p_term, o_term))
        })
    }

    /// Number of triples stored.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Return `true` if no triples have been inserted.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Access the underlying dictionary (for statistics / inspection).
    pub fn dictionary(&self) -> &StarDictionary {
        &self.dict
    }

    /// Approximate heap memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.dict.memory_bytes() + self.triples.len() * 3 * std::mem::size_of::<EncodedTerm>()
    }
}

impl Default for CompressedTripleSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BlankNode, Literal, NamedNode};

    fn iri(s: &str) -> StarTerm {
        StarTerm::NamedNode(NamedNode { iri: s.to_owned() })
    }

    fn blank(id: &str) -> StarTerm {
        StarTerm::BlankNode(BlankNode { id: id.to_owned() })
    }

    fn lit(v: &str) -> StarTerm {
        StarTerm::Literal(Literal {
            value: v.to_owned(),
            language: None,
            datatype: None,
        })
    }

    fn lit_lang(v: &str, lang: &str) -> StarTerm {
        StarTerm::Literal(Literal {
            value: v.to_owned(),
            language: Some(lang.to_owned()),
            datatype: None,
        })
    }

    fn lit_dt(v: &str, dt: &str) -> StarTerm {
        StarTerm::Literal(Literal {
            value: v.to_owned(),
            language: None,
            datatype: Some(NamedNode { iri: dt.to_owned() }),
        })
    }

    #[test]
    fn test_iri_round_trip() {
        let mut dict = StarDictionary::new();
        let term = iri("http://example.org/subject");
        let encoded = dict.encode_term(&term).expect("encode ok");
        assert!(matches!(encoded, EncodedTerm::Iri(_)));
        let decoded = dict.decode_term(&encoded).expect("decode ok");
        assert_eq!(decoded, term);
    }

    #[test]
    fn test_blank_node_round_trip() {
        let mut dict = StarDictionary::new();
        let term = blank("b1");
        let encoded = dict.encode_term(&term).expect("encode ok");
        assert!(matches!(encoded, EncodedTerm::BlankNode(_)));
        let decoded = dict.decode_term(&encoded).expect("decode ok");
        assert_eq!(decoded, term);
    }

    #[test]
    fn test_literal_round_trip() {
        let mut dict = StarDictionary::new();
        let term = lit("hello world");
        let encoded = dict.encode_term(&term).expect("encode ok");
        assert!(matches!(encoded, EncodedTerm::Literal(_)));
        let decoded = dict.decode_term(&encoded).expect("decode ok");
        assert_eq!(decoded, term);
    }

    #[test]
    fn test_literal_with_language_round_trip() {
        let mut dict = StarDictionary::new();
        let term = lit_lang("Bonjour", "fr");
        let encoded = dict.encode_term(&term).expect("encode ok");
        let decoded = dict.decode_term(&encoded).expect("decode ok");
        assert_eq!(decoded, term);
    }

    #[test]
    fn test_literal_with_datatype_round_trip() {
        let mut dict = StarDictionary::new();
        let term = lit_dt("42", "http://www.w3.org/2001/XMLSchema#integer");
        let encoded = dict.encode_term(&term).expect("encode ok");
        let decoded = dict.decode_term(&encoded).expect("decode ok");
        assert_eq!(decoded, term);
    }

    #[test]
    fn test_deduplication() {
        let mut dict = StarDictionary::new();
        let alice = iri("http://example.org/alice");
        let id1 = dict.encode_term(&alice).expect("encode ok");
        let id2 = dict.encode_term(&alice).expect("encode ok");
        assert_eq!(id1, id2);
        assert_eq!(dict.term_count(), 1);
    }

    #[test]
    fn test_quoted_triple_round_trip() {
        let mut dict = StarDictionary::new();

        let inner = StarTriple::new(
            iri("http://example.org/alice"),
            iri("http://example.org/age"),
            lit("30"),
        );
        let quoted = StarTerm::quoted_triple(inner.clone());

        let encoded = dict.encode_term(&quoted).expect("encode ok");
        assert!(encoded.is_quoted());

        let decoded = dict.decode_term(&encoded).expect("decode ok");
        assert_eq!(decoded, quoted);
    }

    #[test]
    fn test_nested_quoted_triple_round_trip() {
        let mut dict = StarDictionary::new();

        let inner = StarTriple::new(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            iri("http://ex.org/o"),
        );
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://ex.org/meta"),
            lit("evidence"),
        );
        let doubly_quoted = StarTerm::quoted_triple(outer);

        let encoded = dict.encode_term(&doubly_quoted).expect("encode ok");
        let decoded = dict.decode_term(&encoded).expect("decode ok");
        assert_eq!(decoded, doubly_quoted);
    }

    #[test]
    fn test_quoted_deduplication() {
        let mut dict = StarDictionary::new();

        let triple = StarTriple::new(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            iri("http://ex.org/o"),
        );
        let qt = StarTerm::quoted_triple(triple);

        let id1 = dict.encode_term(&qt).expect("encode ok");
        let id2 = dict.encode_term(&qt).expect("encode ok");
        assert_eq!(id1, id2);
        assert_eq!(dict.quoted_count(), 1);
    }

    #[test]
    fn test_variable_encoding_fails() {
        let mut dict = StarDictionary::new();
        let var = StarTerm::variable("x").expect("valid variable");
        assert!(dict.encode_term(&var).is_err());
    }

    #[test]
    fn test_term_count_is_accurate() {
        let mut dict = StarDictionary::new();
        dict.encode_term(&iri("http://ex.org/a")).expect("ok");
        dict.encode_term(&iri("http://ex.org/b")).expect("ok");
        dict.encode_term(&blank("b1")).expect("ok");
        dict.encode_term(&lit("hello")).expect("ok");
        // Re-encode existing – should not increase count.
        dict.encode_term(&iri("http://ex.org/a")).expect("ok");
        assert_eq!(dict.term_count(), 4);
    }

    #[test]
    fn test_compressed_triple_set() {
        let mut set = CompressedTripleSet::new();

        for i in 0..100_usize {
            let triple = StarTriple::new(
                iri(&format!("http://ex.org/s{i}")),
                iri("http://ex.org/p"),
                lit(&format!("{i}")),
            );
            set.insert(&triple).expect("insert ok");
        }

        assert_eq!(set.len(), 100);
        // The predicate "http://ex.org/p" should be deduplicated to 1 entry.
        // Subjects + literals + predicate = 100 + 100 + 1 = 201
        assert_eq!(set.dictionary().term_count(), 201);

        // Verify round-trip for all stored triples.
        let decoded: Vec<StarTriple> = set
            .iter_triples()
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(decoded.len(), 100);
    }

    #[test]
    fn test_memory_bytes_grows_with_insertions() {
        let mut dict = StarDictionary::new();
        let before = dict.memory_bytes();
        for i in 0..50_usize {
            dict.encode_term(&iri(&format!("http://ex.org/r{i}")))
                .expect("ok");
        }
        let after = dict.memory_bytes();
        assert!(after > before, "memory should grow after insertions");
    }

    #[test]
    fn test_encode_triple() {
        let mut dict = StarDictionary::new();
        let triple = StarTriple::new(
            iri("http://ex.org/s"),
            iri("http://ex.org/p"),
            iri("http://ex.org/o"),
        );
        let (s, p, o) = dict.encode_triple(&triple).expect("encode ok");
        assert!(matches!(s, EncodedTerm::Iri(_)));
        assert!(matches!(p, EncodedTerm::Iri(_)));
        assert!(matches!(o, EncodedTerm::Iri(_)));
    }
}
