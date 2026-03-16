//! RDF Patch Protocol implementation
//!
//! RDF Patch is a format for expressing changes to RDF datasets.
//! Each patch consists of optional header lines followed by change lines.
//!
//! # Format Overview
//!
//! ```text
//! H id <uuid>
//! H prev <uuid>
//! TX
//! PA ex <http://example.org/>
//! A <http://example.org/s> <http://example.org/p> <http://example.org/o>
//! D <http://example.org/s> <http://example.org/p> <http://example.org/old>
//! TC
//! ```
//!
//! Line prefixes:
//! - `H`  — header (version, id, prev)
//! - `TX` — transaction begin
//! - `TC` — transaction commit
//! - `TA` — transaction abort
//! - `PA` — add prefix
//! - `PD` — delete prefix
//! - `A`  — add triple or quad
//! - `D`  — delete triple or quad
//!
//! # References
//!
//! <https://afs.github.io/rdf-patch/>

use crate::writer::{RdfTerm, TermType};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::io::{BufRead, BufReader, Read};

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
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }

    fn at(line: usize, msg: impl fmt::Display) -> Self {
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

    fn triple_key(t: &PatchTriple) -> String {
        format!("{}\x00{}\x00{}", t.subject, t.predicate, t.object)
    }
}

// ─── PatchParser ─────────────────────────────────────────────────────────────

/// Parser for the RDF Patch text format
pub struct PatchParser;

impl PatchParser {
    /// Parse an entire RDF Patch document from a string
    pub fn parse(input: &str) -> PatchResult<RdfPatch> {
        let mut headers = Vec::new();
        let mut changes = Vec::new();
        let mut prefixes: BTreeMap<String, String> = BTreeMap::new();

        for (idx, raw_line) in input.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw_line.trim();

            // Skip blank lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(rest) = line.strip_prefix("H ") {
                let header = Self::parse_header(rest.trim(), line_no)?;
                headers.push(header);
            } else if line == "TX" {
                changes.push(PatchChange::TransactionBegin);
            } else if line == "TC" {
                changes.push(PatchChange::TransactionCommit);
            } else if line == "TA" {
                changes.push(PatchChange::TransactionAbort);
            } else if let Some(rest) = line.strip_prefix("PA ") {
                let (prefix, iri) = Self::parse_prefix_decl(rest.trim(), line_no)?;
                prefixes.insert(prefix.clone(), iri.clone());
                changes.push(PatchChange::AddPrefix { prefix, iri });
            } else if let Some(rest) = line.strip_prefix("PD ") {
                let (prefix, iri) = Self::parse_prefix_decl(rest.trim(), line_no)?;
                changes.push(PatchChange::DeletePrefix { prefix, iri });
            } else if let Some(rest) = line.strip_prefix("A ") {
                let change = Self::parse_triple_or_quad("A", rest.trim(), &prefixes, line_no)?;
                changes.push(change);
            } else if let Some(rest) = line.strip_prefix("D ") {
                let change = Self::parse_triple_or_quad("D", rest.trim(), &prefixes, line_no)?;
                changes.push(change);
            } else {
                return Err(PatchError::at(
                    line_no,
                    format!("unrecognised line: {line:?}"),
                ));
            }
        }

        Ok(RdfPatch { headers, changes })
    }

    /// Create a streaming iterator that parses one [`PatchChange`] at a time.
    /// Headers are skipped in streaming mode (only change lines are yielded).
    pub fn parse_streaming(reader: impl Read) -> impl Iterator<Item = PatchResult<PatchChange>> {
        StreamingPatchParser::new(reader)
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn parse_header(rest: &str, line_no: usize) -> PatchResult<PatchHeader> {
        // rest is `key <value>` or `key value`
        let mut parts = rest.splitn(2, ' ');
        let key = parts
            .next()
            .ok_or_else(|| PatchError::at(line_no, "missing header key"))?
            .trim();
        let value_raw = parts.next().unwrap_or("").trim();
        let value = strip_angle_brackets(value_raw);
        match key {
            "version" => Ok(PatchHeader::Version(value.to_string())),
            "prev" => Ok(PatchHeader::Previous(value.to_string())),
            "id" => Ok(PatchHeader::Id(value.to_string())),
            other => Ok(PatchHeader::Unknown {
                key: other.to_string(),
                value: value.to_string(),
            }),
        }
    }

    fn parse_prefix_decl(rest: &str, line_no: usize) -> PatchResult<(String, String)> {
        // rest is `prefix <iri>` or `prefix: <iri>`
        let mut parts = rest.splitn(2, ' ');
        let prefix_raw = parts
            .next()
            .ok_or_else(|| PatchError::at(line_no, "missing prefix name"))?
            .trim_end_matches(':');
        let iri_raw = parts
            .next()
            .ok_or_else(|| PatchError::at(line_no, "missing prefix IRI"))?
            .trim();
        let iri = strip_angle_brackets(iri_raw);
        Ok((prefix_raw.to_string(), iri.to_string()))
    }

    fn parse_triple_or_quad(
        op: &str,
        rest: &str,
        prefixes: &BTreeMap<String, String>,
        line_no: usize,
    ) -> PatchResult<PatchChange> {
        // Strip trailing ' .' if present
        let rest = rest.trim_end_matches('.').trim();
        let terms = tokenise_terms(rest, prefixes, line_no)?;
        match terms.len() {
            3 => {
                let triple = PatchTriple::new(terms[0].clone(), terms[1].clone(), terms[2].clone());
                if op == "A" {
                    Ok(PatchChange::AddTriple(triple))
                } else {
                    Ok(PatchChange::DeleteTriple(triple))
                }
            }
            4 => {
                let quad = PatchQuad::new(
                    terms[0].clone(),
                    terms[1].clone(),
                    terms[2].clone(),
                    terms[3].clone(),
                );
                if op == "A" {
                    Ok(PatchChange::AddQuad(quad))
                } else {
                    Ok(PatchChange::DeleteQuad(quad))
                }
            }
            n => Err(PatchError::at(
                line_no,
                format!("expected 3 or 4 terms, got {n}"),
            )),
        }
    }
}

// ─── Streaming parser ────────────────────────────────────────────────────────

struct StreamingPatchParser<R: Read> {
    reader: BufReader<R>,
    line_no: usize,
    prefixes: BTreeMap<String, String>,
    done: bool,
}

impl<R: Read> StreamingPatchParser<R> {
    fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            line_no: 0,
            prefixes: BTreeMap::new(),
            done: false,
        }
    }
}

impl<R: Read> Iterator for StreamingPatchParser<R> {
    type Item = PatchResult<PatchChange>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            let mut raw = String::new();
            match self.reader.read_line(&mut raw) {
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(PatchError::at(self.line_no, e.to_string())));
                }
                Ok(_) => {}
            }
            self.line_no += 1;
            let line = raw.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Headers — skip silently in streaming mode
            if line.starts_with("H ") {
                continue;
            }

            let result = if line == "TX" {
                Ok(PatchChange::TransactionBegin)
            } else if line == "TC" {
                Ok(PatchChange::TransactionCommit)
            } else if line == "TA" {
                Ok(PatchChange::TransactionAbort)
            } else if let Some(rest) = line.strip_prefix("PA ") {
                match parse_prefix_decl_inline(rest.trim(), self.line_no) {
                    Ok((prefix, iri)) => {
                        self.prefixes.insert(prefix.clone(), iri.clone());
                        Ok(PatchChange::AddPrefix { prefix, iri })
                    }
                    Err(e) => Err(e),
                }
            } else if let Some(rest) = line.strip_prefix("PD ") {
                match parse_prefix_decl_inline(rest.trim(), self.line_no) {
                    Ok((prefix, iri)) => Ok(PatchChange::DeletePrefix { prefix, iri }),
                    Err(e) => Err(e),
                }
            } else if let Some(rest) = line.strip_prefix("A ") {
                PatchParser::parse_triple_or_quad("A", rest.trim(), &self.prefixes, self.line_no)
            } else if let Some(rest) = line.strip_prefix("D ") {
                PatchParser::parse_triple_or_quad("D", rest.trim(), &self.prefixes, self.line_no)
            } else {
                Err(PatchError::at(
                    self.line_no,
                    format!("unrecognised line: {line:?}"),
                ))
            };

            return Some(result);
        }
    }
}

// ─── PatchSerializer ─────────────────────────────────────────────────────────

/// Serializes [`RdfPatch`] documents to the RDF Patch text format
pub struct PatchSerializer;

impl PatchSerializer {
    /// Serialize an [`RdfPatch`] to a string
    pub fn serialize(patch: &RdfPatch) -> String {
        let mut out = String::new();
        for header in &patch.headers {
            out.push_str(&header.to_string());
            out.push('\n');
        }
        if !patch.headers.is_empty() && !patch.changes.is_empty() {
            out.push('\n');
        }
        for change in &patch.changes {
            out.push_str(&change.to_string());
            out.push('\n');
        }
        out
    }

    /// Serialize a single [`PatchChange`] line (no newline appended)
    pub fn serialize_change(change: &PatchChange) -> String {
        change.to_string()
    }
}

// ─── apply_patch ─────────────────────────────────────────────────────────────

/// Apply an [`RdfPatch`] to an in-memory [`Graph`], updating it in place.
///
/// Transactions are honoured: changes between `TX`/`TA` are rolled back on abort.
/// Returns [`PatchStats`] summarising what was modified.
pub fn apply_patch(graph: &mut Graph, patch: &RdfPatch) -> PatchResult<PatchStats> {
    let mut stats = PatchStats::default();
    let mut in_tx = false;
    // Staged changes for the current transaction block
    let mut tx_adds: Vec<PatchTriple> = Vec::new();
    let mut tx_deletes: Vec<PatchTriple> = Vec::new();
    let mut tx_prefix_adds: Vec<(String, String)> = Vec::new();

    for change in &patch.changes {
        match change {
            PatchChange::TransactionBegin => {
                in_tx = true;
                tx_adds.clear();
                tx_deletes.clear();
                tx_prefix_adds.clear();
                stats.transactions += 1;
            }
            PatchChange::TransactionCommit => {
                // Commit staged changes
                for t in tx_adds.drain(..) {
                    if graph.add_triple(t) {
                        stats.triples_added += 1;
                    }
                }
                for t in &tx_deletes {
                    if graph.remove_triple(t) {
                        stats.triples_deleted += 1;
                    }
                }
                tx_deletes.clear();
                for (p, i) in tx_prefix_adds.drain(..) {
                    graph.prefixes.insert(p, i);
                    stats.prefixes_added += 1;
                }
                in_tx = false;
            }
            PatchChange::TransactionAbort => {
                // Discard staged changes
                tx_adds.clear();
                tx_deletes.clear();
                tx_prefix_adds.clear();
                in_tx = false;
                stats.aborts += 1;
            }
            PatchChange::AddPrefix { prefix, iri } => {
                if in_tx {
                    tx_prefix_adds.push((prefix.clone(), iri.clone()));
                } else {
                    graph.prefixes.insert(prefix.clone(), iri.clone());
                    stats.prefixes_added += 1;
                }
            }
            PatchChange::DeletePrefix { prefix, .. } => {
                graph.prefixes.remove(prefix.as_str());
                stats.prefixes_deleted += 1;
            }
            PatchChange::AddTriple(t) => {
                if in_tx {
                    tx_adds.push(t.clone());
                } else if graph.add_triple(t.clone()) {
                    stats.triples_added += 1;
                }
            }
            PatchChange::DeleteTriple(t) => {
                if in_tx {
                    tx_deletes.push(t.clone());
                } else if graph.remove_triple(t) {
                    stats.triples_deleted += 1;
                }
            }
            // Quads are not supported on simple Graph; treat as triple
            PatchChange::AddQuad(q) => {
                let t = PatchTriple::new(q.subject.clone(), q.predicate.clone(), q.object.clone());
                if in_tx {
                    tx_adds.push(t);
                } else if graph.add_triple(t) {
                    stats.triples_added += 1;
                }
            }
            PatchChange::DeleteQuad(q) => {
                let t = PatchTriple::new(q.subject.clone(), q.predicate.clone(), q.object.clone());
                if in_tx {
                    tx_deletes.push(t.clone());
                } else if graph.remove_triple(&t) {
                    stats.triples_deleted += 1;
                }
            }
        }
    }

    Ok(stats)
}

// ─── diff_to_patch ───────────────────────────────────────────────────────────

/// Generate a minimal [`RdfPatch`] that transforms `old` into `new`.
///
/// All deletions come before additions in the generated patch, matching
/// the convention used by most RDF Patch tools.
pub fn diff_to_patch(old: &Graph, new: &Graph) -> RdfPatch {
    let mut changes = Vec::new();

    // Deletes: triples in old but not new
    for triple in old.iter() {
        if !new.contains(triple) {
            changes.push(PatchChange::DeleteTriple(triple.clone()));
        }
    }

    // Adds: triples in new but not old
    for triple in new.iter() {
        if !old.contains(triple) {
            changes.push(PatchChange::AddTriple(triple.clone()));
        }
    }

    // Prefix adds: in new but not old
    for (prefix, iri) in &new.prefixes {
        if old.prefixes.get(prefix) != Some(iri) {
            changes.push(PatchChange::AddPrefix {
                prefix: prefix.clone(),
                iri: iri.clone(),
            });
        }
    }

    // Prefix deletes: in old but not new
    for (prefix, iri) in &old.prefixes {
        if !new.prefixes.contains_key(prefix.as_str()) {
            changes.push(PatchChange::DeletePrefix {
                prefix: prefix.clone(),
                iri: iri.clone(),
            });
        }
    }

    RdfPatch {
        headers: Vec::new(),
        changes,
    }
}

// ─── Term tokeniser ──────────────────────────────────────────────────────────

/// Tokenise a whitespace-separated sequence of RDF terms.
/// Handles IRIs (`<...>`), blank nodes (`_:id`), literals (`"..."`), and
/// prefixed names (`prefix:local`).
fn tokenise_terms(
    input: &str,
    prefixes: &BTreeMap<String, String>,
    line_no: usize,
) -> PatchResult<Vec<PatchTerm>> {
    let mut terms = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;

    while pos < chars.len() {
        // Skip whitespace
        while pos < chars.len() && chars[pos].is_whitespace() {
            pos += 1;
        }
        if pos >= chars.len() {
            break;
        }

        if chars[pos] == '<' {
            // IRI
            pos += 1;
            let start = pos;
            while pos < chars.len() && chars[pos] != '>' {
                pos += 1;
            }
            if pos >= chars.len() {
                return Err(PatchError::at(line_no, "unterminated IRI"));
            }
            let iri: String = chars[start..pos].iter().collect();
            pos += 1; // consume '>'
            terms.push(PatchTerm::iri(iri));
        } else if chars[pos] == '"' {
            // Literal
            pos += 1;
            let mut value = String::new();
            while pos < chars.len() {
                if chars[pos] == '\\' && pos + 1 < chars.len() {
                    pos += 1;
                    match chars[pos] {
                        '"' => value.push('"'),
                        '\\' => value.push('\\'),
                        'n' => value.push('\n'),
                        'r' => value.push('\r'),
                        't' => value.push('\t'),
                        c => {
                            value.push('\\');
                            value.push(c);
                        }
                    }
                    pos += 1;
                } else if chars[pos] == '"' {
                    break;
                } else {
                    value.push(chars[pos]);
                    pos += 1;
                }
            }
            if pos >= chars.len() {
                return Err(PatchError::at(line_no, "unterminated literal"));
            }
            pos += 1; // consume closing '"'

            // Check for language tag or datatype
            if pos < chars.len() && chars[pos] == '@' {
                pos += 1;
                let start = pos;
                while pos < chars.len() && !chars[pos].is_whitespace() {
                    pos += 1;
                }
                let lang: String = chars[start..pos].iter().collect();
                terms.push(PatchTerm::lang_literal(value, lang));
            } else if pos + 1 < chars.len() && chars[pos] == '^' && chars[pos + 1] == '^' {
                pos += 2;
                if pos >= chars.len() || chars[pos] != '<' {
                    return Err(PatchError::at(line_no, "expected '<' after '^^'"));
                }
                pos += 1;
                let start = pos;
                while pos < chars.len() && chars[pos] != '>' {
                    pos += 1;
                }
                if pos >= chars.len() {
                    return Err(PatchError::at(line_no, "unterminated datatype IRI"));
                }
                let dt: String = chars[start..pos].iter().collect();
                pos += 1;
                terms.push(PatchTerm::typed_literal(value, dt));
            } else {
                terms.push(PatchTerm::literal(value));
            }
        } else if pos + 1 < chars.len() && chars[pos] == '_' && chars[pos + 1] == ':' {
            // Blank node
            pos += 2;
            let start = pos;
            while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != '.' {
                pos += 1;
            }
            let id: String = chars[start..pos].iter().collect();
            terms.push(PatchTerm::blank_node(id));
        } else if chars[pos] == '.' {
            // Trailing dot — stop
            pos += 1;
        } else {
            // Possibly a prefixed name `prefix:local`
            let start = pos;
            while pos < chars.len() && !chars[pos].is_whitespace() && chars[pos] != '.' {
                pos += 1;
            }
            let token: String = chars[start..pos].iter().collect();
            if let Some(colon_pos) = token.find(':') {
                let ns = &token[..colon_pos];
                let local = &token[colon_pos + 1..];
                match prefixes.get(ns) {
                    Some(base) => {
                        let full = format!("{base}{local}");
                        terms.push(PatchTerm::iri(full));
                    }
                    None => {
                        return Err(PatchError::at(
                            line_no,
                            format!("unknown prefix '{ns}' in '{token}'"),
                        ))
                    }
                }
            } else if token.is_empty() || token == "." {
                // skip
            } else {
                return Err(PatchError::at(
                    line_no,
                    format!("unexpected token '{token}'"),
                ));
            }
        }
    }

    Ok(terms)
}

/// Strip surrounding `<...>` from an IRI token, if present
fn strip_angle_brackets(s: &str) -> &str {
    if s.starts_with('<') && s.ends_with('>') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

/// Inline prefix-decl parser used in the streaming parser
fn parse_prefix_decl_inline(rest: &str, line_no: usize) -> PatchResult<(String, String)> {
    let mut parts = rest.splitn(2, ' ');
    let prefix_raw = parts
        .next()
        .ok_or_else(|| PatchError::at(line_no, "missing prefix name"))?
        .trim_end_matches(':');
    let iri_raw = parts
        .next()
        .ok_or_else(|| PatchError::at(line_no, "missing prefix IRI"))?
        .trim();
    let iri = strip_angle_brackets(iri_raw);
    Ok((prefix_raw.to_string(), iri.to_string()))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a simple triple
    fn triple(s: &str, p: &str, o: &str) -> PatchTriple {
        PatchTriple::new(PatchTerm::iri(s), PatchTerm::iri(p), PatchTerm::iri(o))
    }

    fn triple_lit(s: &str, p: &str, o: &str) -> PatchTriple {
        PatchTriple::new(PatchTerm::iri(s), PatchTerm::iri(p), PatchTerm::literal(o))
    }

    // ── Header parsing ────────────────────────────────────────────────────

    #[test]
    fn test_parse_header_id() {
        let patch = PatchParser::parse("H id <urn:uuid:1234>\n").expect("should succeed");
        assert_eq!(patch.headers.len(), 1);
        assert_eq!(patch.id(), Some("urn:uuid:1234"));
    }

    #[test]
    fn test_parse_header_prev() {
        let patch = PatchParser::parse("H prev <urn:uuid:abcd>\n").expect("should succeed");
        assert_eq!(patch.previous(), Some("urn:uuid:abcd"));
    }

    #[test]
    fn test_parse_header_version() {
        let patch = PatchParser::parse("H version 1\n").expect("should succeed");
        matches!(&patch.headers[0], PatchHeader::Version(v) if v == "1");
    }

    #[test]
    fn test_parse_header_unknown() {
        let patch = PatchParser::parse("H custom myval\n").expect("should succeed");
        assert!(matches!(&patch.headers[0], PatchHeader::Unknown { key, .. } if key == "custom"));
    }

    #[test]
    fn test_parse_multiple_headers() {
        let input = "H id <urn:1>\nH prev <urn:0>\nH version 2\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        assert_eq!(patch.headers.len(), 3);
    }

    // ── Transaction control ───────────────────────────────────────────────

    #[test]
    fn test_parse_tx_tc() {
        let patch = PatchParser::parse("TX\nTC\n").expect("should succeed");
        assert_eq!(patch.changes.len(), 2);
        assert!(matches!(patch.changes[0], PatchChange::TransactionBegin));
        assert!(matches!(patch.changes[1], PatchChange::TransactionCommit));
    }

    #[test]
    fn test_parse_ta() {
        let patch = PatchParser::parse("TX\nTA\n").expect("should succeed");
        assert!(matches!(patch.changes[1], PatchChange::TransactionAbort));
    }

    #[test]
    fn test_transaction_control_predicates() {
        assert!(PatchChange::TransactionBegin.is_transaction_control());
        assert!(PatchChange::TransactionCommit.is_transaction_control());
        assert!(PatchChange::TransactionAbort.is_transaction_control());
    }

    // ── Prefix parsing ────────────────────────────────────────────────────

    #[test]
    fn test_parse_prefix_add() {
        let patch = PatchParser::parse("PA ex <http://example.org/>\n").expect("should succeed");
        assert_eq!(patch.changes.len(), 1);
        match &patch.changes[0] {
            PatchChange::AddPrefix { prefix, iri } => {
                assert_eq!(prefix, "ex");
                assert_eq!(iri, "http://example.org/");
            }
            _ => panic!("unexpected change type"),
        }
    }

    #[test]
    fn test_parse_prefix_delete() {
        let patch = PatchParser::parse("PD ex <http://example.org/>\n").expect("should succeed");
        assert!(
            matches!(&patch.changes[0], PatchChange::DeletePrefix { prefix, .. } if prefix == "ex")
        );
    }

    #[test]
    fn test_prefix_resolution_in_triple() {
        let input = "PA ex <http://example.org/>\nA ex:s ex:p ex:o .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        assert_eq!(patch.changes.len(), 2);
        if let PatchChange::AddTriple(t) = &patch.changes[1] {
            assert_eq!(t.subject.value(), "http://example.org/s");
        } else {
            panic!("expected AddTriple");
        }
    }

    // ── Triple operations ─────────────────────────────────────────────────

    #[test]
    fn test_parse_add_triple() {
        let input = "A <http://s> <http://p> <http://o> .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        assert!(matches!(&patch.changes[0], PatchChange::AddTriple(_)));
    }

    #[test]
    fn test_parse_delete_triple() {
        let input = "D <http://s> <http://p> <http://o> .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        assert!(matches!(&patch.changes[0], PatchChange::DeleteTriple(_)));
    }

    #[test]
    fn test_parse_triple_with_literal() {
        let input = "A <http://s> <http://p> \"hello\" .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        if let PatchChange::AddTriple(t) = &patch.changes[0] {
            assert!(
                t.object.0.term_type
                    == TermType::Literal {
                        datatype: None,
                        lang: None
                    }
            );
            assert_eq!(t.object.value(), "hello");
        } else {
            panic!("expected AddTriple");
        }
    }

    #[test]
    fn test_parse_literal_with_language() {
        let input = "A <http://s> <http://p> \"hello\"@en .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        if let PatchChange::AddTriple(t) = &patch.changes[0] {
            assert!(matches!(
                &t.object.0.term_type,
                TermType::Literal { lang: Some(l), .. } if l == "en"
            ));
        } else {
            panic!("expected AddTriple");
        }
    }

    #[test]
    fn test_parse_literal_with_datatype() {
        let input =
            "A <http://s> <http://p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        if let PatchChange::AddTriple(t) = &patch.changes[0] {
            assert!(matches!(
                &t.object.0.term_type,
                TermType::Literal { datatype: Some(dt), .. }
                if dt == "http://www.w3.org/2001/XMLSchema#integer"
            ));
        } else {
            panic!("expected AddTriple");
        }
    }

    #[test]
    fn test_parse_triple_blank_node() {
        let input = "A _:b0 <http://p> <http://o> .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        if let PatchChange::AddTriple(t) = &patch.changes[0] {
            assert!(t.subject.is_blank_node());
            assert_eq!(t.subject.value(), "b0");
        } else {
            panic!("expected AddTriple");
        }
    }

    // ── Quad operations ───────────────────────────────────────────────────

    #[test]
    fn test_parse_add_quad() {
        let input = "A <http://s> <http://p> <http://o> <http://g> .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        assert!(matches!(&patch.changes[0], PatchChange::AddQuad(_)));
    }

    #[test]
    fn test_parse_delete_quad() {
        let input = "D <http://s> <http://p> <http://o> <http://g> .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        assert!(matches!(&patch.changes[0], PatchChange::DeleteQuad(_)));
    }

    #[test]
    fn test_quad_graph_term() {
        let input = "A <http://s> <http://p> <http://o> <http://graph1> .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        if let PatchChange::AddQuad(q) = &patch.changes[0] {
            assert_eq!(q.graph.value(), "http://graph1");
        } else {
            panic!("expected AddQuad");
        }
    }

    // ── Serialization ─────────────────────────────────────────────────────

    #[test]
    fn test_serialize_header() {
        let patch = RdfPatch {
            headers: vec![PatchHeader::Id("urn:1".to_string())],
            changes: vec![],
        };
        let s = PatchSerializer::serialize(&patch);
        assert!(s.contains("H id urn:1"));
    }

    #[test]
    fn test_serialize_add_triple() {
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![PatchChange::AddTriple(triple(
                "http://s", "http://p", "http://o",
            ))],
        };
        let s = PatchSerializer::serialize(&patch);
        assert!(s.contains("A <http://s> <http://p> <http://o>"));
    }

    #[test]
    fn test_serialize_delete_triple() {
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![PatchChange::DeleteTriple(triple(
                "http://s", "http://p", "http://o",
            ))],
        };
        let s = PatchSerializer::serialize(&patch);
        assert!(s.starts_with("D "));
    }

    #[test]
    fn test_serialize_prefix_add() {
        let change = PatchChange::AddPrefix {
            prefix: "ex".to_string(),
            iri: "http://example.org/".to_string(),
        };
        let s = PatchSerializer::serialize_change(&change);
        assert_eq!(s, "PA ex <http://example.org/>");
    }

    #[test]
    fn test_serialize_transaction_control() {
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![
                PatchChange::TransactionBegin,
                PatchChange::TransactionCommit,
            ],
        };
        let s = PatchSerializer::serialize(&patch);
        assert!(s.contains("TX"));
        assert!(s.contains("TC"));
    }

    #[test]
    fn test_serialize_literal() {
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![PatchChange::AddTriple(triple_lit(
                "http://s", "http://p", "hello",
            ))],
        };
        let s = PatchSerializer::serialize(&patch);
        assert!(s.contains("\"hello\""));
    }

    // ── Apply patch ───────────────────────────────────────────────────────

    #[test]
    fn test_apply_add_triple() {
        let mut graph = Graph::new();
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![PatchChange::AddTriple(triple(
                "http://s", "http://p", "http://o",
            ))],
        };
        let stats = apply_patch(&mut graph, &patch).expect("should succeed");
        assert_eq!(stats.triples_added, 1);
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_apply_delete_triple() {
        let mut graph = Graph::new();
        let t = triple("http://s", "http://p", "http://o");
        graph.add_triple(t.clone());
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![PatchChange::DeleteTriple(t)],
        };
        let stats = apply_patch(&mut graph, &patch).expect("should succeed");
        assert_eq!(stats.triples_deleted, 1);
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn test_apply_idempotent_add() {
        let mut graph = Graph::new();
        let t = triple("http://s", "http://p", "http://o");
        graph.add_triple(t.clone());
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![PatchChange::AddTriple(t)],
        };
        let stats = apply_patch(&mut graph, &patch).expect("should succeed");
        // Should not double-count
        assert_eq!(stats.triples_added, 0);
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_apply_prefix_add() {
        let mut graph = Graph::new();
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![PatchChange::AddPrefix {
                prefix: "ex".to_string(),
                iri: "http://example.org/".to_string(),
            }],
        };
        let stats = apply_patch(&mut graph, &patch).expect("should succeed");
        assert_eq!(stats.prefixes_added, 1);
        assert_eq!(
            graph.prefixes.get("ex").map(String::as_str),
            Some("http://example.org/")
        );
    }

    #[test]
    fn test_apply_transaction_commit() {
        let mut graph = Graph::new();
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![
                PatchChange::TransactionBegin,
                PatchChange::AddTriple(triple("http://s", "http://p", "http://o")),
                PatchChange::TransactionCommit,
            ],
        };
        let stats = apply_patch(&mut graph, &patch).expect("should succeed");
        assert_eq!(stats.triples_added, 1);
        assert_eq!(stats.transactions, 1);
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_apply_transaction_abort() {
        let mut graph = Graph::new();
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![
                PatchChange::TransactionBegin,
                PatchChange::AddTriple(triple("http://s", "http://p", "http://o")),
                PatchChange::TransactionAbort,
            ],
        };
        let stats = apply_patch(&mut graph, &patch).expect("should succeed");
        assert_eq!(stats.aborts, 1);
        // Graph must remain empty — abort rolls back staged changes
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn test_apply_multiple_changes() {
        let mut graph = Graph::new();
        let t1 = triple("http://a", "http://p", "http://x");
        let t2 = triple("http://b", "http://p", "http://y");
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![
                PatchChange::AddTriple(t1.clone()),
                PatchChange::AddTriple(t2.clone()),
                PatchChange::DeleteTriple(t1),
            ],
        };
        let stats = apply_patch(&mut graph, &patch).expect("should succeed");
        assert_eq!(stats.triples_added, 2);
        assert_eq!(stats.triples_deleted, 1);
        assert_eq!(graph.len(), 1);
    }

    // ── Graph diff ────────────────────────────────────────────────────────

    #[test]
    fn test_diff_to_patch_add() {
        let old = Graph::new();
        let mut new_graph = Graph::new();
        new_graph.add_triple(triple("http://s", "http://p", "http://o"));
        let patch = diff_to_patch(&old, &new_graph);
        assert_eq!(patch.add_count(), 1);
        assert_eq!(patch.delete_count(), 0);
    }

    #[test]
    fn test_diff_to_patch_delete() {
        let mut old = Graph::new();
        old.add_triple(triple("http://s", "http://p", "http://o"));
        let new_graph = Graph::new();
        let patch = diff_to_patch(&old, &new_graph);
        assert_eq!(patch.add_count(), 0);
        assert_eq!(patch.delete_count(), 1);
    }

    #[test]
    fn test_diff_to_patch_no_change() {
        let mut old = Graph::new();
        old.add_triple(triple("http://s", "http://p", "http://o"));
        let new_graph = old.clone();
        let patch = diff_to_patch(&old, &new_graph);
        assert!(patch.changes.is_empty());
    }

    #[test]
    fn test_diff_to_patch_prefix_added() {
        let old = Graph::new();
        let mut new_graph = Graph::new();
        new_graph
            .prefixes
            .insert("ex".to_string(), "http://example.org/".to_string());
        let patch = diff_to_patch(&old, &new_graph);
        assert!(patch
            .changes
            .iter()
            .any(|c| matches!(c, PatchChange::AddPrefix { .. })));
    }

    #[test]
    fn test_diff_to_patch_prefix_removed() {
        let mut old = Graph::new();
        old.prefixes
            .insert("ex".to_string(), "http://example.org/".to_string());
        let new_graph = Graph::new();
        let patch = diff_to_patch(&old, &new_graph);
        assert!(patch
            .changes
            .iter()
            .any(|c| matches!(c, PatchChange::DeletePrefix { .. })));
    }

    // ── Round-trip ────────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_simple() {
        let input = "H id <urn:1>\nA <http://s> <http://p> <http://o> .\nD <http://s> <http://p> <http://old> .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        let serialized = PatchSerializer::serialize(&patch);
        let reparsed = PatchParser::parse(&serialized).expect("should succeed");
        assert_eq!(reparsed.headers.len(), patch.headers.len());
        assert_eq!(reparsed.changes.len(), patch.changes.len());
    }

    #[test]
    fn test_round_trip_with_prefixes() {
        let input = "PA ex <http://example.org/>\nA ex:s ex:p ex:o .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        let serialized = PatchSerializer::serialize(&patch);
        // After serialisation ex:s becomes <http://example.org/s>
        assert!(serialized.contains("<http://example.org/s>"));
        // Re-parse the serialised form
        let reparsed = PatchParser::parse(&serialized).expect("should succeed");
        assert_eq!(reparsed.changes.len(), 2);
    }

    #[test]
    fn test_round_trip_transaction() {
        let input = "TX\nA <http://s> <http://p> <http://o> .\nTC\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        let serialized = PatchSerializer::serialize(&patch);
        let reparsed = PatchParser::parse(&serialized).expect("should succeed");
        assert_eq!(reparsed.changes.len(), 3);
        assert!(matches!(reparsed.changes[0], PatchChange::TransactionBegin));
        assert!(matches!(
            reparsed.changes[2],
            PatchChange::TransactionCommit
        ));
    }

    #[test]
    fn test_round_trip_with_blank_nodes() {
        let input = "A _:b0 <http://p> <http://o> .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        let s = PatchSerializer::serialize(&patch);
        let reparsed = PatchParser::parse(&s).expect("should succeed");
        if let PatchChange::AddTriple(t) = &reparsed.changes[0] {
            assert!(t.subject.is_blank_node());
        } else {
            panic!("expected AddTriple");
        }
    }

    #[test]
    fn test_round_trip_literal_with_lang() {
        let input = "A <http://s> <http://p> \"bonjour\"@fr .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        let s = PatchSerializer::serialize(&patch);
        let reparsed = PatchParser::parse(&s).expect("should succeed");
        if let PatchChange::AddTriple(t) = &reparsed.changes[0] {
            assert!(matches!(
                &t.object.0.term_type,
                TermType::Literal { lang: Some(l), .. } if l == "fr"
            ));
        } else {
            panic!("expected AddTriple");
        }
    }

    #[test]
    fn test_round_trip_literal_with_datatype() {
        let dt = "http://www.w3.org/2001/XMLSchema#integer";
        let input = format!("A <http://s> <http://p> \"42\"^^<{dt}> .\n");
        let patch = PatchParser::parse(&input).expect("should succeed");
        let s = PatchSerializer::serialize(&patch);
        let reparsed = PatchParser::parse(&s).expect("should succeed");
        if let PatchChange::AddTriple(t) = &reparsed.changes[0] {
            assert!(matches!(
                &t.object.0.term_type,
                TermType::Literal { datatype: Some(d), .. } if d == dt
            ));
        } else {
            panic!("expected AddTriple");
        }
    }

    // ── Streaming parser ──────────────────────────────────────────────────

    #[test]
    fn test_streaming_parser_basic() {
        let input = "TX\nA <http://s> <http://p> <http://o> .\nTC\n";
        let changes: Vec<_> = PatchParser::parse_streaming(input.as_bytes()).collect();
        assert_eq!(changes.len(), 3);
        assert!(changes[0]
            .as_ref()
            .map(|c| matches!(c, PatchChange::TransactionBegin))
            .unwrap_or(false));
    }

    #[test]
    fn test_streaming_skips_headers() {
        let input = "H id <urn:1>\nA <http://s> <http://p> <http://o> .\n";
        let changes: Vec<_> = PatchParser::parse_streaming(input.as_bytes()).collect();
        // Header is skipped in streaming mode
        assert_eq!(changes.len(), 1);
    }

    #[test]
    fn test_streaming_parser_prefixes() {
        let input = "PA ex <http://example.org/>\nA ex:s ex:p ex:o .\n";
        let changes: Vec<_> = PatchParser::parse_streaming(input.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .expect("should succeed");
        assert_eq!(changes.len(), 2);
    }

    #[test]
    fn test_streaming_parser_multiple_batches() {
        let input = "A <http://s1> <http://p> <http://o1> .\nA <http://s2> <http://p> <http://o2> .\nD <http://s1> <http://p> <http://o1> .\n";
        let changes: Vec<_> = PatchParser::parse_streaming(input.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .expect("should succeed");
        assert_eq!(changes.len(), 3);
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    #[test]
    fn test_empty_patch() {
        let patch = PatchParser::parse("").expect("should succeed");
        assert!(patch.is_empty());
    }

    #[test]
    fn test_comments_ignored() {
        let input =
            "# This is a comment\nA <http://s> <http://p> <http://o> .\n# Another comment\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        assert_eq!(patch.changes.len(), 1);
    }

    #[test]
    fn test_blank_lines_ignored() {
        let input = "\n\nA <http://s> <http://p> <http://o> .\n\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        assert_eq!(patch.changes.len(), 1);
    }

    #[test]
    fn test_error_unknown_line() {
        let result = PatchParser::parse("UNKNOWN_CMD <http://x>\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unterminated_iri() {
        let result = PatchParser::parse("A <http://s <http://p> <http://o> .\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_patch_change_predicates() {
        assert!(PatchChange::AddTriple(triple("h://s", "h://p", "h://o")).is_add());
        assert!(PatchChange::DeleteTriple(triple("h://s", "h://p", "h://o")).is_delete());
        assert!(!PatchChange::AddTriple(triple("h://s", "h://p", "h://o")).is_delete());
        assert!(!PatchChange::DeleteTriple(triple("h://s", "h://p", "h://o")).is_add());
    }

    #[test]
    fn test_graph_contains() {
        let mut g = Graph::new();
        let t = triple("http://s", "http://p", "http://o");
        assert!(!g.contains(&t));
        g.add_triple(t.clone());
        assert!(g.contains(&t));
        g.remove_triple(&t);
        assert!(!g.contains(&t));
    }

    #[test]
    fn test_graph_len() {
        let mut g = Graph::new();
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
        g.add_triple(triple("http://s", "http://p", "http://o"));
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_apply_patch_from_parsed_text() {
        let input =
            "PA ex <http://example.org/>\nTX\nA ex:alice <http://type> <http://Person> .\nTC\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        let mut graph = Graph::new();
        let stats = apply_patch(&mut graph, &patch).expect("should succeed");
        assert_eq!(stats.triples_added, 1);
        assert_eq!(stats.transactions, 1);
    }

    #[test]
    fn test_patch_stats_default() {
        let stats = PatchStats::default();
        assert_eq!(stats.triples_added, 0);
        assert_eq!(stats.triples_deleted, 0);
        assert_eq!(stats.prefixes_added, 0);
        assert_eq!(stats.prefixes_deleted, 0);
        assert_eq!(stats.transactions, 0);
        assert_eq!(stats.aborts, 0);
    }

    #[test]
    fn test_patch_header_key_value() {
        let h = PatchHeader::Id("urn:test".to_string());
        assert_eq!(h.key(), "id");
        assert_eq!(h.value(), "urn:test");
    }

    #[test]
    fn test_diff_then_apply_round_trip() {
        let mut old = Graph::new();
        old.add_triple(triple("http://s", "http://p", "http://o1"));
        old.add_triple(triple("http://s", "http://p", "http://o2"));

        let mut new_graph = Graph::new();
        new_graph.add_triple(triple("http://s", "http://p", "http://o2"));
        new_graph.add_triple(triple("http://s", "http://p", "http://o3"));

        let patch = diff_to_patch(&old, &new_graph);
        // Apply patch to old to get new
        let mut result = old.clone();
        apply_patch(&mut result, &patch).expect("should succeed");

        assert_eq!(result.len(), new_graph.len());
        for t in new_graph.iter() {
            assert!(result.contains(t), "missing triple: {t}");
        }
    }

    #[test]
    fn test_serialize_then_parse_complete_patch() {
        let mut patch = RdfPatch::new();
        patch
            .headers
            .push(PatchHeader::Id("urn:test:42".to_string()));
        patch
            .headers
            .push(PatchHeader::Previous("urn:test:41".to_string()));
        patch.changes.push(PatchChange::AddPrefix {
            prefix: "foaf".to_string(),
            iri: "http://xmlns.com/foaf/0.1/".to_string(),
        });
        patch.changes.push(PatchChange::TransactionBegin);
        patch.changes.push(PatchChange::AddTriple(triple(
            "http://example.org/alice",
            "http://xmlns.com/foaf/0.1/name",
            "http://example.org/literal_placeholder",
        )));
        patch.changes.push(PatchChange::TransactionCommit);

        let serialized = PatchSerializer::serialize(&patch);
        let reparsed = PatchParser::parse(&serialized).expect("should succeed");

        assert_eq!(reparsed.id(), Some("urn:test:42"));
        assert_eq!(reparsed.previous(), Some("urn:test:41"));
        assert_eq!(reparsed.changes.len(), 4);
    }

    #[test]
    fn test_quad_apply_to_simple_graph() {
        let mut graph = Graph::new();
        let q = PatchQuad::new(
            PatchTerm::iri("http://s"),
            PatchTerm::iri("http://p"),
            PatchTerm::iri("http://o"),
            PatchTerm::iri("http://graph1"),
        );
        let patch = RdfPatch {
            headers: vec![],
            changes: vec![PatchChange::AddQuad(q)],
        };
        let stats = apply_patch(&mut graph, &patch).expect("should succeed");
        assert_eq!(stats.triples_added, 1);
    }

    #[test]
    fn test_escaped_literal() {
        let input = "A <http://s> <http://p> \"say \\\"hello\\\"\" .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        if let PatchChange::AddTriple(t) = &patch.changes[0] {
            assert_eq!(t.object.value(), "say \"hello\"");
        } else {
            panic!("expected AddTriple");
        }
    }

    #[test]
    fn test_newline_in_literal_escape() {
        let input = "A <http://s> <http://p> \"line1\\nline2\" .\n";
        let patch = PatchParser::parse(input).expect("should succeed");
        if let PatchChange::AddTriple(t) = &patch.changes[0] {
            assert!(t.object.value().contains('\n'));
        } else {
            panic!("expected AddTriple");
        }
    }

    #[test]
    fn test_patch_change_line_prefix() {
        assert_eq!(PatchChange::TransactionBegin.line_prefix(), "TX");
        assert_eq!(PatchChange::TransactionCommit.line_prefix(), "TC");
        assert_eq!(PatchChange::TransactionAbort.line_prefix(), "TA");
        assert_eq!(
            PatchChange::AddTriple(triple("http://s", "http://p", "http://o")).line_prefix(),
            "A"
        );
        assert_eq!(
            PatchChange::DeleteTriple(triple("http://s", "http://p", "http://o")).line_prefix(),
            "D"
        );
    }
}
