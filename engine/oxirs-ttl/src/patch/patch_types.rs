//! Data types for the RDF Patch protocol.
//!
//! Contains: [`PatchError`], [`PatchResult`], [`PatchHeader`], [`PatchTerm`],
//! [`PatchTriple`], [`PatchQuad`], [`PatchChange`], [`RdfPatch`], [`PatchStats`], [`Graph`].

use crate::writer::{RdfTerm, TermType};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

// ─── Error ───────────────────────────────────────────────────────────────────

/// Error produced when parsing an RDF Patch document
#[derive(Debug, Clone)]
pub struct PatchError {
    /// 1-based line number where the error occurred
    pub line: usize,
    /// Human-readable description
    pub message: String,
}

impl PatchError {
    pub(crate) fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }

    pub(crate) fn at(line: usize, msg: impl fmt::Display) -> Self {
        Self::new(line, msg.to_string())
    }
}

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "patch error at line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for PatchError {}

/// Convenience result alias
pub type PatchResult<T> = Result<T, PatchError>;

// ─── Data model ──────────────────────────────────────────────────────────────

/// A header entry in an RDF Patch document
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchHeader {
    /// `H version <value>` — patch format version
    Version(String),
    /// `H prev <uuid>` — IRI/UUID of the previous patch in the chain
    Previous(String),
    /// `H id <uuid>` — IRI/UUID identifying this patch
    Id(String),
    /// Any other `H key <value>` header not defined in the spec
    Unknown {
        /// The header key
        key: String,
        /// The header value
        value: String,
    },
}

impl PatchHeader {
    /// Return the header key string as it appears in the serialised format
    pub fn key(&self) -> &str {
        match self {
            PatchHeader::Version(_) => "version",
            PatchHeader::Previous(_) => "prev",
            PatchHeader::Id(_) => "id",
            PatchHeader::Unknown { key, .. } => key.as_str(),
        }
    }

    /// Return the header value string
    pub fn value(&self) -> &str {
        match self {
            PatchHeader::Version(v) | PatchHeader::Previous(v) | PatchHeader::Id(v) => v.as_str(),
            PatchHeader::Unknown { value, .. } => value.as_str(),
        }
    }
}

impl fmt::Display for PatchHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H {} {}", self.key(), self.value())
    }
}

/// A subject/object position that accepts either a named node, blank node, or literal.
/// Used internally for parsed term positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchTerm(pub RdfTerm);

impl PatchTerm {
    /// Create from an IRI (angle-bracket or prefixed form, already resolved)
    pub fn iri(iri: impl Into<String>) -> Self {
        Self(RdfTerm::iri(iri))
    }

    /// Create from a blank node identifier
    pub fn blank_node(id: impl Into<String>) -> Self {
        Self(RdfTerm::blank_node(id))
    }

    /// Create from a plain literal value
    pub fn literal(value: impl Into<String>) -> Self {
        Self(RdfTerm::simple_literal(value))
    }

    /// Create from a language-tagged literal
    pub fn lang_literal(value: impl Into<String>, lang: impl Into<String>) -> Self {
        Self(RdfTerm::lang_literal(value, lang))
    }

    /// Create from a typed literal
    pub fn typed_literal(value: impl Into<String>, datatype: impl Into<String>) -> Self {
        Self(RdfTerm::typed_literal(value, datatype))
    }

    /// Access the underlying [`RdfTerm`]
    pub fn term(&self) -> &RdfTerm {
        &self.0
    }

    /// Return `true` if this term is an IRI
    pub fn is_iri(&self) -> bool {
        self.0.term_type == TermType::Iri
    }

    /// Return `true` if this term is a blank node
    pub fn is_blank_node(&self) -> bool {
        self.0.term_type == TermType::BlankNode
    }

    /// Return the lexical value
    pub fn value(&self) -> &str {
        &self.0.value
    }
}

impl fmt::Display for PatchTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0.term_type {
            TermType::Iri => write!(f, "<{}>", self.0.value),
            TermType::BlankNode => write!(f, "_:{}", self.0.value),
            TermType::Literal { datatype, lang } => {
                // Escape internal quotes
                let escaped = self.0.value.replace('\\', "\\\\").replace('"', "\\\"");
                write!(f, "\"{escaped}\"")?;
                if let Some(l) = lang {
                    write!(f, "@{l}")?;
                } else if let Some(dt) = datatype {
                    write!(f, "^^<{dt}>")?;
                }
                Ok(())
            }
        }
    }
}

/// A triple (subject, predicate, object) using [`PatchTerm`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchTriple {
    /// The subject term
    pub subject: PatchTerm,
    /// The predicate term
    pub predicate: PatchTerm,
    /// The object term
    pub object: PatchTerm,
}

impl PatchTriple {
    /// Construct a new triple from three terms
    pub fn new(subject: PatchTerm, predicate: PatchTerm, object: PatchTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

impl fmt::Display for PatchTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// A quad (subject, predicate, object, graph) using [`PatchTerm`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchQuad {
    /// The subject term
    pub subject: PatchTerm,
    /// The predicate term
    pub predicate: PatchTerm,
    /// The object term
    pub object: PatchTerm,
    /// The named graph term
    pub graph: PatchTerm,
}

impl PatchQuad {
    /// Construct a new quad
    pub fn new(
        subject: PatchTerm,
        predicate: PatchTerm,
        object: PatchTerm,
        graph: PatchTerm,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            graph,
        }
    }
}

impl fmt::Display for PatchQuad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {} {} .",
            self.subject, self.predicate, self.object, self.graph
        )
    }
}

/// A single change line in an RDF Patch document
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchChange {
    /// `PA prefix <iri>` — add a prefix declaration
    AddPrefix {
        /// Namespace prefix label
        prefix: String,
        /// IRI bound to the prefix
        iri: String,
    },
    /// `PD prefix <iri>` — delete a prefix declaration
    DeletePrefix {
        /// Namespace prefix label
        prefix: String,
        /// IRI bound to the prefix
        iri: String,
    },
    /// `A <s> <p> <o> .` — add a triple
    AddTriple(PatchTriple),
    /// `D <s> <p> <o> .` — delete a triple
    DeleteTriple(PatchTriple),
    /// `A <s> <p> <o> <g> .` — add a quad
    AddQuad(PatchQuad),
    /// `D <s> <p> <o> <g> .` — delete a quad
    DeleteQuad(PatchQuad),
    /// `TX` — begin a transaction block
    TransactionBegin,
    /// `TC` — commit a transaction block
    TransactionCommit,
    /// `TA` — abort a transaction block
    TransactionAbort,
}

impl PatchChange {
    /// Return the line prefix that represents this change in the patch format
    pub fn line_prefix(&self) -> &'static str {
        match self {
            PatchChange::AddPrefix { .. } => "PA",
            PatchChange::DeletePrefix { .. } => "PD",
            PatchChange::AddTriple(_) => "A",
            PatchChange::DeleteTriple(_) => "D",
            PatchChange::AddQuad(_) => "A",
            PatchChange::DeleteQuad(_) => "D",
            PatchChange::TransactionBegin => "TX",
            PatchChange::TransactionCommit => "TC",
            PatchChange::TransactionAbort => "TA",
        }
    }

    /// Return `true` if this change adds data
    pub fn is_add(&self) -> bool {
        matches!(
            self,
            PatchChange::AddTriple(_) | PatchChange::AddQuad(_) | PatchChange::AddPrefix { .. }
        )
    }

    /// Return `true` if this change deletes data
    pub fn is_delete(&self) -> bool {
        matches!(
            self,
            PatchChange::DeleteTriple(_)
                | PatchChange::DeleteQuad(_)
                | PatchChange::DeletePrefix { .. }
        )
    }

    /// Return `true` if this is a transaction control statement
    pub fn is_transaction_control(&self) -> bool {
        matches!(
            self,
            PatchChange::TransactionBegin
                | PatchChange::TransactionCommit
                | PatchChange::TransactionAbort
        )
    }
}

impl fmt::Display for PatchChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatchChange::AddPrefix { prefix, iri } => {
                write!(f, "PA {prefix} <{iri}>")
            }
            PatchChange::DeletePrefix { prefix, iri } => {
                write!(f, "PD {prefix} <{iri}>")
            }
            PatchChange::AddTriple(t) => write!(f, "A {t}"),
            PatchChange::DeleteTriple(t) => write!(f, "D {t}"),
            PatchChange::AddQuad(q) => write!(f, "A {q}"),
            PatchChange::DeleteQuad(q) => write!(f, "D {q}"),
            PatchChange::TransactionBegin => write!(f, "TX"),
            PatchChange::TransactionCommit => write!(f, "TC"),
            PatchChange::TransactionAbort => write!(f, "TA"),
        }
    }
}

// ─── RdfPatch ────────────────────────────────────────────────────────────────

/// A complete RDF Patch document: a list of headers followed by change lines
#[derive(Debug, Clone, Default)]
pub struct RdfPatch {
    /// Header entries (`H key value`)
    pub headers: Vec<PatchHeader>,
    /// Change entries (`TX / TC / TA / PA / PD / A / D`)
    pub changes: Vec<PatchChange>,
}

impl RdfPatch {
    /// Construct an empty patch
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a patch with headers and changes
    pub fn with_changes(headers: Vec<PatchHeader>, changes: Vec<PatchChange>) -> Self {
        Self { headers, changes }
    }

    /// Return the `id` header value if present
    pub fn id(&self) -> Option<&str> {
        self.headers.iter().find_map(|h| {
            if let PatchHeader::Id(v) = h {
                Some(v.as_str())
            } else {
                None
            }
        })
    }

    /// Return the `prev` header value if present
    pub fn previous(&self) -> Option<&str> {
        self.headers.iter().find_map(|h| {
            if let PatchHeader::Previous(v) = h {
                Some(v.as_str())
            } else {
                None
            }
        })
    }

    /// Count how many triple/quad additions are in the patch
    pub fn add_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, PatchChange::AddTriple(_) | PatchChange::AddQuad(_)))
            .count()
    }

    /// Count how many triple/quad deletions are in the patch
    pub fn delete_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, PatchChange::DeleteTriple(_) | PatchChange::DeleteQuad(_)))
            .count()
    }

    /// Return `true` if the patch contains no headers and no changes
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty() && self.changes.is_empty()
    }
}

// ─── Statistics ──────────────────────────────────────────────────────────────

/// Statistics collected when applying a patch to a graph
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PatchStats {
    /// Number of triples that were actually inserted
    pub triples_added: usize,
    /// Number of triples that were actually removed
    pub triples_deleted: usize,
    /// Number of prefix declarations that were added
    pub prefixes_added: usize,
    /// Number of prefix declarations that were removed
    pub prefixes_deleted: usize,
    /// Number of transaction blocks encountered
    pub transactions: usize,
    /// Number of transaction aborts encountered
    pub aborts: usize,
}

// ─── In-memory graph ─────────────────────────────────────────────────────────

/// A minimal in-memory RDF graph used for patch application and diff generation.
///
/// Triples are stored as `(subject, predicate, object)` tuples of [`PatchTerm`].
/// Prefix mappings are stored separately.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    /// Set of (subject, predicate, object) triples
    pub triples: HashSet<String>,
    /// Prefix → IRI mappings
    pub prefixes: BTreeMap<String, String>,
    /// Raw triple objects for retrieval (mirrors `triples`)
    triple_objects: Vec<PatchTriple>,
}

impl Graph {
    /// Construct an empty graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the graph; returns `true` if newly inserted
    pub fn add_triple(&mut self, triple: PatchTriple) -> bool {
        let key = Self::triple_key(&triple);
        if self.triples.insert(key) {
            self.triple_objects.push(triple);
            true
        } else {
            false
        }
    }

    /// Remove a triple from the graph; returns `true` if it was present
    pub fn remove_triple(&mut self, triple: &PatchTriple) -> bool {
        let key = Self::triple_key(triple);
        if self.triples.remove(&key) {
            self.triple_objects.retain(|t| Self::triple_key(t) != key);
            true
        } else {
            false
        }
    }

    /// Return `true` if the triple is present in the graph
    pub fn contains(&self, triple: &PatchTriple) -> bool {
        self.triples.contains(&Self::triple_key(triple))
    }

    /// Number of triples in the graph
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Return `true` if the graph has no triples
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Iterate over all triples in the graph
    pub fn iter(&self) -> impl Iterator<Item = &PatchTriple> {
        self.triple_objects.iter()
    }

    pub(crate) fn triple_key(t: &PatchTriple) -> String {
        format!("{}\x00{}\x00{}", t.subject, t.predicate, t.object)
    }
}
