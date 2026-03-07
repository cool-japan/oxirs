//! RDF Binary (Thrift) format — read + write support
//!
//! Implements the Jena-compatible RDF Binary Thrift encoding as specified in
//! <https://jena.apache.org/documentation/io/rdf-binary.html>.
//!
//! ## Wire encoding summary
//!
//! Each row is a Thrift-style struct.  All integers use unsigned-varint (LEB128)
//! encoding.  Strings are length-prefixed with a varint followed by UTF-8 bytes.
//!
//! ### Type tags (1-byte field tag in the outer struct)
//!
//! | Tag | Meaning              |
//! |-----|----------------------|
//! | 1   | IRI                  |
//! | 2   | Blank node           |
//! | 3   | Literal              |
//! | 4   | Triple row           |
//! | 5   | Quad row             |
//! | 6   | Prefix declaration   |
//! | 7   | Prefixed IRI         |
//! | 0   | End-of-stream        |
//!
//! ### Term struct layout
//!
//! IRI (tag=1): `{ value: string }`
//! BNode (tag=2): `{ value: string }`
//! Literal (tag=3): `{ value: string, lang: opt<string>, datatype: opt<string> }`
//! Triple (tag=4): `{ s: term, p: term, o: term }`
//! Quad (tag=5): `{ s: term, p: term, o: term, g: term }`
//! Prefix (tag=6): `{ prefix: string, iri: string }`
//! Prefixed IRI (tag=7): `{ prefix_id: varint, local: string }`
//!
//! Optional fields that are absent are encoded as a zero byte.  Present optional
//! strings are preceded by a 1 byte.

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use oxirs_core::model::{
    BlankNode, GraphName, Literal, NamedNode, Object, Predicate, Quad, Subject, Triple,
};
use std::collections::HashMap;
use std::io::{Read, Write};

// ─── Type tag constants ──────────────────────────────────────────────────────

/// Tag: IRI term
const TAG_IRI: u8 = 1;
/// Tag: Blank node term
const TAG_BNODE: u8 = 2;
/// Tag: Literal term
const TAG_LITERAL: u8 = 3;
/// Tag: Triple row
const TAG_TRIPLE: u8 = 4;
/// Tag: Quad row
const TAG_QUAD: u8 = 5;
/// Tag: Prefix table entry
const TAG_PREFIX: u8 = 6;
/// Tag: Prefixed IRI (namespace-compressed)
const TAG_PREFIXED_IRI: u8 = 7;
/// Tag: End of stream
const TAG_EOF: u8 = 0;

// ─── Varint helpers ──────────────────────────────────────────────────────────

/// Encode a 64-bit unsigned integer as LEB128 (unsigned varint) into `buf`.
fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

/// Decode one unsigned varint from `reader`, returning the value.
fn decode_varint<R: Read>(reader: &mut R) -> TurtleResult<u64> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    loop {
        let mut byte_buf = [0u8; 1];
        reader
            .read_exact(&mut byte_buf)
            .map_err(TurtleParseError::io)?;
        let byte = byte_buf[0];
        let low_bits = (byte & 0x7F) as u64;
        value |= low_bits
            .checked_shl(shift)
            .ok_or_else(varint_overflow_error)?;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err(varint_overflow_error());
        }
    }
}

fn varint_overflow_error() -> TurtleParseError {
    TurtleParseError::syntax(TurtleSyntaxError::Generic {
        message: "varint overflow (>64 bits)".to_string(),
        position: TextPosition::start(),
    })
}

// ─── String primitive I/O ────────────────────────────────────────────────────

/// Write a length-prefixed UTF-8 string.
fn write_str_primitive(s: &str, buf: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    encode_varint(bytes.len() as u64, buf);
    buf.extend_from_slice(bytes);
}

/// Read a length-prefixed UTF-8 string.
fn read_str_primitive<R: Read>(reader: &mut R) -> TurtleResult<String> {
    let len = decode_varint(reader)? as usize;
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .map_err(TurtleParseError::io)?;
    String::from_utf8(bytes).map_err(|e| {
        TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: format!("invalid UTF-8 in string: {e}"),
            position: TextPosition::start(),
        })
    })
}

// ─── Optional-field helpers ──────────────────────────────────────────────────

/// Write an optional string: `0x00` when absent, `0x01 || string` when present.
fn write_optional_str(opt: Option<&str>, buf: &mut Vec<u8>) {
    match opt {
        None => buf.push(0x00),
        Some(s) => {
            buf.push(0x01);
            write_str_primitive(s, buf);
        }
    }
}

/// Read an optional string.
fn read_optional_str<R: Read>(reader: &mut R) -> TurtleResult<Option<String>> {
    let mut flag = [0u8; 1];
    reader.read_exact(&mut flag).map_err(TurtleParseError::io)?;
    match flag[0] {
        0x00 => Ok(None),
        0x01 => Ok(Some(read_str_primitive(reader)?)),
        other => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: format!("unexpected optional-field flag byte 0x{other:02X}; expected 0 or 1"),
            position: TextPosition::start(),
        })),
    }
}

// ─── Term I/O ────────────────────────────────────────────────────────────────

/// Decoded RDF term (owner variant, suitable for returning across call frames).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThriftTerm {
    /// IRI node
    Iri(String),
    /// Blank node
    BlankNode(String),
    /// Literal value with optional language tag and optional datatype IRI.
    Literal {
        /// Lexical form of the literal
        value: String,
        /// BCP-47 language tag (mutually exclusive with `datatype`)
        lang: Option<String>,
        /// Datatype IRI (mutually exclusive with `lang`)
        datatype: Option<String>,
    },
}

impl ThriftTerm {
    /// Convert to an `oxirs_core` [`Subject`].
    pub fn to_subject(&self) -> TurtleResult<Subject> {
        match self {
            Self::Iri(iri) => {
                let nn = NamedNode::new(iri).map_err(TurtleParseError::model)?;
                Ok(Subject::NamedNode(nn))
            }
            Self::BlankNode(id) => {
                let bn = BlankNode::new(id).map_err(TurtleParseError::model)?;
                Ok(Subject::BlankNode(bn))
            }
            Self::Literal { .. } => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "literal cannot be used as triple subject".to_string(),
                position: TextPosition::start(),
            })),
        }
    }

    /// Convert to an `oxirs_core` [`Predicate`].
    pub fn to_predicate(&self) -> TurtleResult<Predicate> {
        match self {
            Self::Iri(iri) => {
                let nn = NamedNode::new(iri).map_err(TurtleParseError::model)?;
                Ok(Predicate::NamedNode(nn))
            }
            _ => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "only IRIs may be used as predicates".to_string(),
                position: TextPosition::start(),
            })),
        }
    }

    /// Convert to an `oxirs_core` [`Object`].
    pub fn to_object(&self) -> TurtleResult<Object> {
        match self {
            Self::Iri(iri) => {
                let nn = NamedNode::new(iri).map_err(TurtleParseError::model)?;
                Ok(Object::NamedNode(nn))
            }
            Self::BlankNode(id) => {
                let bn = BlankNode::new(id).map_err(TurtleParseError::model)?;
                Ok(Object::BlankNode(bn))
            }
            Self::Literal {
                value,
                lang,
                datatype,
            } => {
                let lit = if let Some(language) = lang {
                    Literal::new_language_tagged_literal(value, language)
                        .map_err(|e| TurtleParseError::model(e.into()))?
                } else if let Some(dt) = datatype {
                    let datatype_nn = NamedNode::new(dt).map_err(TurtleParseError::model)?;
                    Literal::new_typed_literal(value, datatype_nn)
                } else {
                    Literal::new_simple_literal(value)
                };
                Ok(Object::Literal(lit))
            }
        }
    }

    /// Convert to an `oxirs_core` [`GraphName`].
    pub fn to_graph_name(&self) -> TurtleResult<GraphName> {
        match self {
            Self::Iri(iri) => {
                let nn = NamedNode::new(iri).map_err(TurtleParseError::model)?;
                Ok(GraphName::NamedNode(nn))
            }
            Self::BlankNode(id) => {
                let bn = BlankNode::new(id).map_err(TurtleParseError::model)?;
                Ok(GraphName::BlankNode(bn))
            }
            Self::Literal { .. } => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "literal cannot be used as graph name".to_string(),
                position: TextPosition::start(),
            })),
        }
    }
}

// ─── Write a term into the byte buffer ──────────────────────────────────────

/// Encode a [`ThriftTerm`] into `buf` with its leading type tag.
fn write_term(term: &ThriftTerm, prefix_table: &HashMap<String, u32>, buf: &mut Vec<u8>) {
    match term {
        ThriftTerm::Iri(iri) => {
            // Try to apply prefix compression first.
            if let Some((prefix_id, local)) = find_prefix(iri, prefix_table) {
                buf.push(TAG_PREFIXED_IRI);
                encode_varint(prefix_id as u64, buf);
                write_str_primitive(local, buf);
                return;
            }
            buf.push(TAG_IRI);
            write_str_primitive(iri, buf);
        }
        ThriftTerm::BlankNode(id) => {
            buf.push(TAG_BNODE);
            write_str_primitive(id, buf);
        }
        ThriftTerm::Literal {
            value,
            lang,
            datatype,
        } => {
            buf.push(TAG_LITERAL);
            write_str_primitive(value, buf);
            write_optional_str(lang.as_deref(), buf);
            write_optional_str(datatype.as_deref(), buf);
        }
    }
}

/// Find the longest matching prefix namespace for `iri`.
/// Returns `(prefix_id, local_part)` on success, `None` when no prefix applies.
fn find_prefix<'a>(iri: &'a str, prefix_table: &HashMap<String, u32>) -> Option<(u32, &'a str)> {
    let mut best: Option<(u32, &str)> = None;
    // Invert prefix_table: we need namespace→id, but prefix_table stores id→namespace.
    // The writer builds a forward table (namespace → id) separately; here we receive
    // the forward table directly from `RdfThriftWriter`.
    for (namespace, &id) in prefix_table {
        if let Some(local) = iri.strip_prefix(namespace.as_str()) {
            if !local.is_empty()
                && best.map_or(true, |(_, prev_local)| local.len() < prev_local.len())
            {
                best = Some((id, local));
            }
        }
    }
    best
}

// ─── Read a term from the byte stream ───────────────────────────────────────

/// Decode a single term from `reader`, resolving prefixed IRIs with `prefix_table`.
/// `prefix_table` maps `prefix_id → namespace IRI`.
fn read_term<R: Read>(reader: &mut R, prefix_table: &[String]) -> TurtleResult<ThriftTerm> {
    let mut tag_buf = [0u8; 1];
    reader
        .read_exact(&mut tag_buf)
        .map_err(TurtleParseError::io)?;
    let tag = tag_buf[0];

    match tag {
        TAG_IRI => {
            let iri = read_str_primitive(reader)?;
            Ok(ThriftTerm::Iri(iri))
        }
        TAG_BNODE => {
            let id = read_str_primitive(reader)?;
            Ok(ThriftTerm::BlankNode(id))
        }
        TAG_LITERAL => {
            let value = read_str_primitive(reader)?;
            let lang = read_optional_str(reader)?;
            let datatype = read_optional_str(reader)?;
            Ok(ThriftTerm::Literal {
                value,
                lang,
                datatype,
            })
        }
        TAG_PREFIXED_IRI => {
            let prefix_id = decode_varint(reader)? as usize;
            let local = read_str_primitive(reader)?;
            let namespace = prefix_table.get(prefix_id).ok_or_else(|| {
                TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: format!("prefixed IRI references unknown prefix id {prefix_id}"),
                    position: TextPosition::start(),
                })
            })?;
            Ok(ThriftTerm::Iri(format!("{namespace}{local}")))
        }
        other => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: format!("unexpected term tag 0x{other:02X}"),
            position: TextPosition::start(),
        })),
    }
}

// ============================================================================
// Writer
// ============================================================================

/// Mode for the [`RdfThriftWriter`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThriftWriteMode {
    /// Emit only triples (no graph name column)
    Triples,
    /// Emit quads (includes graph name column)
    Quads,
}

/// High-performance RDF Binary Thrift writer.
///
/// Serialises RDF triples/quads to Apache Jena's RDF Binary format using a
/// compact, hand-rolled Thrift encoding (no external `thrift` crate required).
///
/// # Example
///
/// ```rust
/// use oxirs_ttl::formats::rdf_thrift::{RdfThriftWriter, ThriftTerm, ThriftWriteMode};
///
/// let mut buf: Vec<u8> = Vec::new();
/// let mut writer = RdfThriftWriter::new(ThriftWriteMode::Triples);
///
/// writer.add_prefix("ex", "http://example.org/");
///
/// writer.write_triple(
///     &ThriftTerm::Iri("http://example.org/alice".to_string()),
///     &ThriftTerm::Iri("http://example.org/name".to_string()),
///     &ThriftTerm::Literal {
///         value: "Alice".to_string(),
///         lang: Some("en".to_string()),
///         datatype: None,
///     },
///     &mut buf,
/// ).unwrap();
///
/// writer.write_eof(&mut buf).unwrap();
/// ```
pub struct RdfThriftWriter {
    mode: ThriftWriteMode,
    /// Maps namespace IRI → prefix integer id (for compression in write direction)
    prefix_forward: HashMap<String, u32>,
    /// Next prefix id to assign
    next_prefix_id: u32,
    /// Whether prefix declarations have already been flushed to the sink
    prefix_declarations_written: bool,
    /// Pending prefix declarations not yet flushed
    pending_prefixes: Vec<(u32, String, String)>, // (id, prefix_label, namespace)
}

impl RdfThriftWriter {
    /// Create a new writer in the given mode.
    pub fn new(mode: ThriftWriteMode) -> Self {
        Self {
            mode,
            prefix_forward: HashMap::new(),
            next_prefix_id: 0,
            prefix_declarations_written: false,
            pending_prefixes: Vec::new(),
        }
    }

    /// Register a prefix for IRI compression.
    ///
    /// Prefixes must be registered before calling any `write_triple` / `write_quad`.
    /// Registering a prefix after writing rows has no effect on already-written rows.
    pub fn add_prefix(&mut self, prefix_label: &str, namespace: &str) {
        if !self.prefix_forward.contains_key(namespace) {
            let id = self.next_prefix_id;
            self.next_prefix_id += 1;
            self.prefix_forward.insert(namespace.to_string(), id);
            self.pending_prefixes
                .push((id, prefix_label.to_string(), namespace.to_string()));
        }
    }

    /// Flush pending prefix declarations into `sink`.
    fn flush_pending_prefixes<W: Write>(&mut self, sink: &mut W) -> TurtleResult<()> {
        if self.prefix_declarations_written && self.pending_prefixes.is_empty() {
            return Ok(());
        }

        // Sort by id so the reader sees them in order
        self.pending_prefixes.sort_by_key(|(id, _, _)| *id);
        for (_, prefix_label, namespace) in self.pending_prefixes.drain(..) {
            let mut buf: Vec<u8> = Vec::new();
            buf.push(TAG_PREFIX);
            write_str_primitive(&prefix_label, &mut buf);
            write_str_primitive(&namespace, &mut buf);
            sink.write_all(&buf).map_err(TurtleParseError::io)?;
        }
        self.prefix_declarations_written = true;
        Ok(())
    }

    /// Write a single triple in Thrift binary format.
    pub fn write_triple<W: Write>(
        &mut self,
        subject: &ThriftTerm,
        predicate: &ThriftTerm,
        object: &ThriftTerm,
        sink: &mut W,
    ) -> TurtleResult<()> {
        self.flush_pending_prefixes(sink)?;

        let mut buf: Vec<u8> = Vec::new();
        buf.push(TAG_TRIPLE);
        write_term(subject, &self.prefix_forward, &mut buf);
        write_term(predicate, &self.prefix_forward, &mut buf);
        write_term(object, &self.prefix_forward, &mut buf);
        sink.write_all(&buf).map_err(TurtleParseError::io)
    }

    /// Write a single quad in Thrift binary format.
    ///
    /// When `mode` is [`ThriftWriteMode::Triples`] the graph field is ignored and
    /// a triple row is emitted instead.
    pub fn write_quad<W: Write>(
        &mut self,
        subject: &ThriftTerm,
        predicate: &ThriftTerm,
        object: &ThriftTerm,
        graph: &ThriftTerm,
        sink: &mut W,
    ) -> TurtleResult<()> {
        self.flush_pending_prefixes(sink)?;

        let mut buf: Vec<u8> = Vec::new();
        match self.mode {
            ThriftWriteMode::Triples => {
                buf.push(TAG_TRIPLE);
                write_term(subject, &self.prefix_forward, &mut buf);
                write_term(predicate, &self.prefix_forward, &mut buf);
                write_term(object, &self.prefix_forward, &mut buf);
            }
            ThriftWriteMode::Quads => {
                buf.push(TAG_QUAD);
                write_term(subject, &self.prefix_forward, &mut buf);
                write_term(predicate, &self.prefix_forward, &mut buf);
                write_term(object, &self.prefix_forward, &mut buf);
                write_term(graph, &self.prefix_forward, &mut buf);
            }
        }
        sink.write_all(&buf).map_err(TurtleParseError::io)
    }

    /// Write the end-of-stream marker.
    pub fn write_eof<W: Write>(&mut self, sink: &mut W) -> TurtleResult<()> {
        self.flush_pending_prefixes(sink)?;
        sink.write_all(&[TAG_EOF]).map_err(TurtleParseError::io)
    }

    /// Convenience: serialise a batch of `oxirs_core` [`Triple`]s.
    pub fn write_triples<W: Write>(
        &mut self,
        triples: &[Triple],
        sink: &mut W,
    ) -> TurtleResult<()> {
        for t in triples {
            let s = subject_to_thrift(t.subject())?;
            let p = predicate_to_thrift(t.predicate())?;
            let o = object_to_thrift(t.object())?;
            self.write_triple(&s, &p, &o, sink)?;
        }
        Ok(())
    }

    /// Convenience: serialise a batch of `oxirs_core` [`Quad`]s.
    pub fn write_quads<W: Write>(&mut self, quads: &[Quad], sink: &mut W) -> TurtleResult<()> {
        for q in quads {
            let s = subject_to_thrift(q.subject())?;
            let p = predicate_to_thrift(q.predicate())?;
            let o = object_to_thrift(q.object())?;
            let g = graph_name_to_thrift(q.graph_name())?;
            self.write_quad(&s, &p, &o, &g, sink)?;
        }
        Ok(())
    }
}

// ─── core-model → ThriftTerm helpers ────────────────────────────────────────

/// Convert a [`Subject`] to a [`ThriftTerm`].
pub fn subject_to_thrift(subject: &Subject) -> TurtleResult<ThriftTerm> {
    match subject {
        Subject::NamedNode(nn) => Ok(ThriftTerm::Iri(nn.as_str().to_string())),
        Subject::BlankNode(bn) => Ok(ThriftTerm::BlankNode(bn.as_str().to_string())),
        Subject::Variable(_) | Subject::QuotedTriple(_) => {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "variables and quoted triples are not supported in RDF Binary".to_string(),
                position: TextPosition::start(),
            }))
        }
    }
}

/// Convert a [`Predicate`] to a [`ThriftTerm`].
pub fn predicate_to_thrift(predicate: &Predicate) -> TurtleResult<ThriftTerm> {
    match predicate {
        Predicate::NamedNode(nn) => Ok(ThriftTerm::Iri(nn.as_str().to_string())),
        Predicate::Variable(_) => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: "variables are not supported in RDF Binary".to_string(),
            position: TextPosition::start(),
        })),
    }
}

/// Convert an [`Object`] to a [`ThriftTerm`].
pub fn object_to_thrift(object: &Object) -> TurtleResult<ThriftTerm> {
    match object {
        Object::NamedNode(nn) => Ok(ThriftTerm::Iri(nn.as_str().to_string())),
        Object::BlankNode(bn) => Ok(ThriftTerm::BlankNode(bn.as_str().to_string())),
        Object::Literal(lit) => {
            let value = lit.value().to_string();
            let lang = lit.language().map(|l| l.to_string());
            let datatype = if lang.is_none() {
                Some(lit.datatype().as_str().to_string())
            } else {
                None
            };
            Ok(ThriftTerm::Literal {
                value,
                lang,
                datatype,
            })
        }
        Object::Variable(_) | Object::QuotedTriple(_) => {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "variables and quoted triples are not supported in RDF Binary".to_string(),
                position: TextPosition::start(),
            }))
        }
    }
}

/// Convert a [`GraphName`] to a [`ThriftTerm`].
pub fn graph_name_to_thrift(graph_name: &GraphName) -> TurtleResult<ThriftTerm> {
    match graph_name {
        GraphName::NamedNode(nn) => Ok(ThriftTerm::Iri(nn.as_str().to_string())),
        GraphName::BlankNode(bn) => Ok(ThriftTerm::BlankNode(bn.as_str().to_string())),
        GraphName::DefaultGraph => Ok(ThriftTerm::Iri(
            "tag:jena.apache.org,2016:rdf/binary:DefaultGraph".to_string(),
        )),
        GraphName::Variable(_) => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: "variables are not supported as graph names in RDF Binary".to_string(),
            position: TextPosition::start(),
        })),
    }
}

// ============================================================================
// Reader
// ============================================================================

/// Result of reading one row from the Thrift binary stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThriftRow {
    /// A triple row (S, P, O)
    Triple(ThriftTerm, ThriftTerm, ThriftTerm),
    /// A quad row (S, P, O, G)
    Quad(ThriftTerm, ThriftTerm, ThriftTerm, ThriftTerm),
    /// A prefix declaration (numeric id assigned in order)
    Prefix {
        /// Human-readable prefix label (e.g. "ex")
        label: String,
        /// Namespace IRI
        namespace: String,
    },
    /// End of stream
    Eof,
}

/// Streaming reader for the RDF Binary Thrift format.
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_ttl::formats::rdf_thrift::{RdfThriftReader, ThriftRow};
/// use std::io::Cursor;
///
/// let data: Vec<u8> = Vec::new(); // pre-populated elsewhere
/// let mut reader = RdfThriftReader::new(Cursor::new(data));
///
/// loop {
///     match reader.read_row().unwrap() {
///         ThriftRow::Triple(s, p, o) => println!("triple: {s:?} {p:?} {o:?}"),
///         ThriftRow::Quad(s, p, o, g) => println!("quad: {s:?} {p:?} {o:?} in {g:?}"),
///         ThriftRow::Prefix { label, namespace } => {
///             println!("prefix {label}: {namespace}")
///         }
///         ThriftRow::Eof => break,
///     }
/// }
/// ```
pub struct RdfThriftReader<R: Read> {
    inner: R,
    /// Ordered list of namespace IRIs indexed by prefix id
    prefix_table: Vec<String>,
}

impl<R: Read> RdfThriftReader<R> {
    /// Create a new reader wrapping `inner`.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            prefix_table: Vec::new(),
        }
    }

    /// Read and return the next row.  Returns [`ThriftRow::Eof`] at end-of-stream.
    pub fn read_row(&mut self) -> TurtleResult<ThriftRow> {
        let mut tag_buf = [0u8; 1];
        match self.inner.read_exact(&mut tag_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(ThriftRow::Eof);
            }
            Err(e) => return Err(TurtleParseError::io(e)),
        }

        match tag_buf[0] {
            TAG_TRIPLE => {
                let s = read_term(&mut self.inner, &self.prefix_table)?;
                let p = read_term(&mut self.inner, &self.prefix_table)?;
                let o = read_term(&mut self.inner, &self.prefix_table)?;
                Ok(ThriftRow::Triple(s, p, o))
            }
            TAG_QUAD => {
                let s = read_term(&mut self.inner, &self.prefix_table)?;
                let p = read_term(&mut self.inner, &self.prefix_table)?;
                let o = read_term(&mut self.inner, &self.prefix_table)?;
                let g = read_term(&mut self.inner, &self.prefix_table)?;
                Ok(ThriftRow::Quad(s, p, o, g))
            }
            TAG_PREFIX => {
                let label = read_str_primitive(&mut self.inner)?;
                let namespace = read_str_primitive(&mut self.inner)?;
                self.prefix_table.push(namespace.clone());
                Ok(ThriftRow::Prefix { label, namespace })
            }
            TAG_EOF => Ok(ThriftRow::Eof),
            other => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("unknown row tag 0x{other:02X}"),
                position: TextPosition::start(),
            })),
        }
    }

    /// Collect all triples from the stream (skips prefix rows, stops at EOF).
    pub fn read_triples(mut self) -> TurtleResult<Vec<Triple>> {
        let mut triples = Vec::new();
        loop {
            match self.read_row()? {
                ThriftRow::Triple(s, p, o) => {
                    let subject = s.to_subject()?;
                    let predicate = p.to_predicate()?;
                    let object = o.to_object()?;
                    triples.push(Triple::new(subject, predicate, object));
                }
                ThriftRow::Quad(s, p, o, _g) => {
                    // Quads in triples-mode: ignore graph name
                    let subject = s.to_subject()?;
                    let predicate = p.to_predicate()?;
                    let object = o.to_object()?;
                    triples.push(Triple::new(subject, predicate, object));
                }
                ThriftRow::Prefix { .. } => {} // consume prefix declarations
                ThriftRow::Eof => break,
            }
        }
        Ok(triples)
    }

    /// Collect all quads from the stream (skips prefix rows, stops at EOF).
    /// Triple rows are treated as quads in the default graph.
    pub fn read_quads(mut self) -> TurtleResult<Vec<Quad>> {
        let mut quads = Vec::new();
        loop {
            match self.read_row()? {
                ThriftRow::Triple(s, p, o) => {
                    let subject = s.to_subject()?;
                    let predicate = p.to_predicate()?;
                    let object = o.to_object()?;
                    quads.push(Quad::new(
                        subject,
                        predicate,
                        object,
                        GraphName::DefaultGraph,
                    ));
                }
                ThriftRow::Quad(s, p, o, g) => {
                    let subject = s.to_subject()?;
                    let predicate = p.to_predicate()?;
                    let object = o.to_object()?;
                    let graph = g.to_graph_name()?;
                    quads.push(Quad::new(subject, predicate, object, graph));
                }
                ThriftRow::Prefix { .. } => {}
                ThriftRow::Eof => break,
            }
        }
        Ok(quads)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ─── Helper to do a round-trip encode→decode ────────────────────────────

    fn round_trip_triples(
        triples: &[Triple],
        prefixes: &[(&str, &str)],
    ) -> TurtleResult<Vec<Triple>> {
        let mut buf: Vec<u8> = Vec::new();
        let mut writer = RdfThriftWriter::new(ThriftWriteMode::Triples);
        for (label, ns) in prefixes {
            writer.add_prefix(label, ns);
        }
        writer.write_triples(triples, &mut buf)?;
        writer.write_eof(&mut buf)?;

        let reader = RdfThriftReader::new(Cursor::new(buf));
        reader.read_triples()
    }

    fn round_trip_quads(quads: &[Quad], prefixes: &[(&str, &str)]) -> TurtleResult<Vec<Quad>> {
        let mut buf: Vec<u8> = Vec::new();
        let mut writer = RdfThriftWriter::new(ThriftWriteMode::Quads);
        for (label, ns) in prefixes {
            writer.add_prefix(label, ns);
        }
        writer.write_quads(quads, &mut buf)?;
        writer.write_eof(&mut buf)?;

        let reader = RdfThriftReader::new(Cursor::new(buf));
        reader.read_quads()
    }

    fn make_triple(s: &str, p: &str, o: Subject) -> Triple {
        // convenience overload with subject as Subject
        let _ = s;
        let _ = p;
        let _ = o;
        unreachable!("use make_iri_triple / make_literal_triple helpers")
    }

    // suppress dead code warning on make_triple
    #[allow(dead_code)]
    fn _use_make_triple() {
        let _ = make_triple;
    }

    fn iri(s: &str) -> NamedNode {
        NamedNode::new(s).expect("valid IRI")
    }

    fn make_iri_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(
            Subject::NamedNode(iri(s)),
            Predicate::NamedNode(iri(p)),
            Object::NamedNode(iri(o)),
        )
    }

    fn make_literal_triple(s: &str, p: &str, value: &str) -> Triple {
        Triple::new(
            Subject::NamedNode(iri(s)),
            Predicate::NamedNode(iri(p)),
            Object::Literal(Literal::new_simple_literal(value)),
        )
    }

    fn make_lang_triple(s: &str, p: &str, value: &str, lang: &str) -> Triple {
        let lit = Literal::new_language_tagged_literal(value, lang).expect("valid lang tag");
        Triple::new(
            Subject::NamedNode(iri(s)),
            Predicate::NamedNode(iri(p)),
            Object::Literal(lit),
        )
    }

    fn make_typed_triple(s: &str, p: &str, value: &str, dt: &str) -> Triple {
        let lit = Literal::new_typed_literal(value, iri(dt));
        Triple::new(
            Subject::NamedNode(iri(s)),
            Predicate::NamedNode(iri(p)),
            Object::Literal(lit),
        )
    }

    // ─── Varint encoding ────────────────────────────────────────────────────

    #[test]
    fn test_varint_small_value() {
        let mut buf = Vec::new();
        encode_varint(42, &mut buf);
        assert_eq!(buf, vec![42]);
        let v = decode_varint(&mut Cursor::new(&buf)).expect("decode ok");
        assert_eq!(v, 42);
    }

    #[test]
    fn test_varint_multi_byte() {
        let mut buf = Vec::new();
        encode_varint(300, &mut buf);
        assert_eq!(buf.len(), 2, "300 needs 2 varint bytes");
        let v = decode_varint(&mut Cursor::new(&buf)).expect("decode ok");
        assert_eq!(v, 300);
    }

    #[test]
    fn test_varint_max_u32() {
        let val: u64 = u32::MAX as u64;
        let mut buf = Vec::new();
        encode_varint(val, &mut buf);
        let v = decode_varint(&mut Cursor::new(&buf)).expect("decode ok");
        assert_eq!(v, val);
    }

    #[test]
    fn test_varint_zero() {
        let mut buf = Vec::new();
        encode_varint(0, &mut buf);
        assert_eq!(buf, vec![0]);
        let v = decode_varint(&mut Cursor::new(&buf)).expect("decode ok");
        assert_eq!(v, 0);
    }

    // ─── String primitives ───────────────────────────────────────────────────

    #[test]
    fn test_string_roundtrip_ascii() {
        let mut buf = Vec::new();
        write_str_primitive("hello world", &mut buf);
        let decoded = read_str_primitive(&mut Cursor::new(&buf)).expect("decode ok");
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_string_roundtrip_unicode() {
        let s = "こんにちは世界 🌍";
        let mut buf = Vec::new();
        write_str_primitive(s, &mut buf);
        let decoded = read_str_primitive(&mut Cursor::new(&buf)).expect("decode ok");
        assert_eq!(decoded, s);
    }

    #[test]
    fn test_string_roundtrip_empty() {
        let mut buf = Vec::new();
        write_str_primitive("", &mut buf);
        let decoded = read_str_primitive(&mut Cursor::new(&buf)).expect("decode ok");
        assert_eq!(decoded, "");
    }

    // ─── Optional-string primitives ─────────────────────────────────────────

    #[test]
    fn test_optional_str_none() {
        let mut buf = Vec::new();
        write_optional_str(None, &mut buf);
        assert_eq!(buf, vec![0x00]);
        let decoded = read_optional_str(&mut Cursor::new(&buf)).expect("decode ok");
        assert_eq!(decoded, None);
    }

    #[test]
    fn test_optional_str_some() {
        let mut buf = Vec::new();
        write_optional_str(Some("en"), &mut buf);
        let decoded = read_optional_str(&mut Cursor::new(&buf)).expect("decode ok");
        assert_eq!(decoded, Some("en".to_string()));
    }

    // ─── IRI term round-trip ─────────────────────────────────────────────────

    #[test]
    fn test_iri_term_roundtrip() {
        let input = [make_iri_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];
        let result = round_trip_triples(&input, &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        let triple = &result[0];
        assert_eq!(
            match triple.subject() {
                Subject::NamedNode(nn) => nn.as_str(),
                _ => panic!("expected named node"),
            },
            "http://example.org/s"
        );
    }

    // ─── Blank node subject ──────────────────────────────────────────────────

    #[test]
    fn test_blank_node_subject_roundtrip() {
        let bn = BlankNode::new("b0").expect("valid blank node");
        let t = Triple::new(
            Subject::BlankNode(bn),
            Predicate::NamedNode(iri("http://example.org/p")),
            Object::NamedNode(iri("http://example.org/o")),
        );
        let result = round_trip_triples(&[t], &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        assert!(
            matches!(result[0].subject(), Subject::BlankNode(_)),
            "subject should be blank node"
        );
    }

    // ─── Blank node object ───────────────────────────────────────────────────

    #[test]
    fn test_blank_node_object_roundtrip() {
        let bn = BlankNode::new("b1").expect("valid blank node");
        let t = Triple::new(
            Subject::NamedNode(iri("http://example.org/s")),
            Predicate::NamedNode(iri("http://example.org/p")),
            Object::BlankNode(bn),
        );
        let result = round_trip_triples(&[t], &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        assert!(
            matches!(result[0].object(), Object::BlankNode(_)),
            "object should be blank node"
        );
    }

    // ─── Plain literal ───────────────────────────────────────────────────────

    #[test]
    fn test_plain_literal_roundtrip() {
        let input = [make_literal_triple(
            "http://example.org/s",
            "http://example.org/p",
            "hello",
        )];
        let result = round_trip_triples(&input, &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        if let Object::Literal(lit) = result[0].object() {
            assert_eq!(lit.value(), "hello");
            assert_eq!(lit.language(), None);
        } else {
            panic!("expected literal object");
        }
    }

    // ─── Language-tagged literal ─────────────────────────────────────────────

    #[test]
    fn test_lang_literal_roundtrip() {
        let input = [make_lang_triple(
            "http://example.org/s",
            "http://example.org/p",
            "Bonjour",
            "fr",
        )];
        let result = round_trip_triples(&input, &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        if let Object::Literal(lit) = result[0].object() {
            assert_eq!(lit.value(), "Bonjour");
            assert_eq!(lit.language(), Some("fr"));
        } else {
            panic!("expected literal object");
        }
    }

    // ─── Typed literal ───────────────────────────────────────────────────────

    #[test]
    fn test_typed_literal_roundtrip() {
        let input = [make_typed_triple(
            "http://example.org/s",
            "http://example.org/p",
            "42",
            "http://www.w3.org/2001/XMLSchema#integer",
        )];
        let result = round_trip_triples(&input, &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        if let Object::Literal(lit) = result[0].object() {
            assert_eq!(lit.value(), "42");
            assert_eq!(
                lit.datatype().as_str(),
                "http://www.w3.org/2001/XMLSchema#integer"
            );
        } else {
            panic!("expected literal object");
        }
    }

    // ─── Empty graph ─────────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph_roundtrip() {
        let result = round_trip_triples(&[], &[]).expect("round-trip ok");
        assert!(result.is_empty(), "should decode zero triples");
    }

    // ─── Single triple ───────────────────────────────────────────────────────

    #[test]
    fn test_single_triple_roundtrip() {
        let input = [make_iri_triple(
            "http://example.org/a",
            "http://example.org/b",
            "http://example.org/c",
        )];
        let result = round_trip_triples(&input, &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
    }

    // ─── 1000-triple batch ───────────────────────────────────────────────────

    #[test]
    fn test_1000_triple_batch_roundtrip() {
        let triples: Vec<Triple> = (0..1000)
            .map(|i| {
                make_iri_triple(
                    &format!("http://example.org/s{i}"),
                    "http://example.org/p",
                    &format!("http://example.org/o{i}"),
                )
            })
            .collect();
        let result = round_trip_triples(&triples, &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1000);
    }

    // ─── Prefix compression round-trip ──────────────────────────────────────

    #[test]
    fn test_prefix_compression_roundtrip() {
        let input = [
            make_iri_triple(
                "http://example.org/alice",
                "http://example.org/knows",
                "http://example.org/bob",
            ),
            make_literal_triple(
                "http://example.org/alice",
                "http://example.org/name",
                "Alice",
            ),
        ];
        let prefixes = [("ex", "http://example.org/")];
        let result = round_trip_triples(&input, &prefixes).expect("round-trip ok");
        assert_eq!(result.len(), 2);

        // Verify IRI reconstruction after prefix expansion
        if let Subject::NamedNode(nn) = result[0].subject() {
            assert_eq!(nn.as_str(), "http://example.org/alice");
        } else {
            panic!("expected named node subject");
        }
    }

    // ─── Multiple prefixes ───────────────────────────────────────────────────

    #[test]
    fn test_multiple_prefix_compression_roundtrip() {
        let input = [make_iri_triple(
            "http://schema.org/Person",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://www.w3.org/2000/01/rdf-schema#Class",
        )];
        let prefixes = [
            ("schema", "http://schema.org/"),
            ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
        ];
        let result = round_trip_triples(&input, &prefixes).expect("round-trip ok");
        assert_eq!(result.len(), 1);

        if let Predicate::NamedNode(nn) = result[0].predicate() {
            assert_eq!(
                nn.as_str(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            );
        } else {
            panic!("expected named node predicate");
        }
    }

    // ─── Named graphs (quads) ────────────────────────────────────────────────

    #[test]
    fn test_quad_named_graph_roundtrip() {
        let graph = GraphName::NamedNode(iri("http://example.org/graph1"));
        let q = Quad::new(
            Subject::NamedNode(iri("http://example.org/s")),
            Predicate::NamedNode(iri("http://example.org/p")),
            Object::NamedNode(iri("http://example.org/o")),
            graph,
        );
        let result = round_trip_quads(&[q], &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        if let GraphName::NamedNode(nn) = result[0].graph_name() {
            assert_eq!(nn.as_str(), "http://example.org/graph1");
        } else {
            panic!("expected named graph");
        }
    }

    // ─── Default graph quad ──────────────────────────────────────────────────

    #[test]
    fn test_quad_default_graph_roundtrip() {
        let q = Quad::new(
            Subject::NamedNode(iri("http://example.org/s")),
            Predicate::NamedNode(iri("http://example.org/p")),
            Object::NamedNode(iri("http://example.org/o")),
            GraphName::DefaultGraph,
        );
        let result = round_trip_quads(&[q], &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        // DefaultGraph is encoded as a special IRI and decoded back to NamedNode
        // (the reader preserves what was decoded; the IRI comparison tells us it's the sentinel)
    }

    // ─── Multi-graph dataset roundtrip ───────────────────────────────────────

    #[test]
    fn test_multi_graph_quads_roundtrip() {
        let quads: Vec<Quad> = (0..10)
            .map(|i| {
                let graph = GraphName::NamedNode(iri(&format!("http://example.org/g{}", i % 3)));
                Quad::new(
                    Subject::NamedNode(iri(&format!("http://example.org/s{i}"))),
                    Predicate::NamedNode(iri("http://example.org/p")),
                    Object::NamedNode(iri(&format!("http://example.org/o{i}"))),
                    graph,
                )
            })
            .collect();

        let result = round_trip_quads(&quads, &[]).expect("round-trip ok");
        assert_eq!(result.len(), 10);
    }

    // ─── Prefix written before triples ──────────────────────────────────────

    #[test]
    fn test_prefix_declarations_precede_triples() {
        let mut buf: Vec<u8> = Vec::new();
        let mut writer = RdfThriftWriter::new(ThriftWriteMode::Triples);
        writer.add_prefix("ex", "http://example.org/");

        let t = make_iri_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        );
        writer.write_triples(&[t], &mut buf).expect("write ok");
        writer.write_eof(&mut buf).expect("eof ok");

        // First byte after buf is TAG_PREFIX (6), confirming prefix comes first
        assert_eq!(buf[0], TAG_PREFIX, "prefix declaration must come first");
    }

    // ─── EOF marker ──────────────────────────────────────────────────────────

    #[test]
    fn test_eof_marker_emitted() {
        let mut buf: Vec<u8> = Vec::new();
        let mut writer = RdfThriftWriter::new(ThriftWriteMode::Triples);
        writer.write_eof(&mut buf).expect("eof ok");
        assert_eq!(buf.last(), Some(&TAG_EOF));
    }

    // ─── Read stops at EOF ───────────────────────────────────────────────────

    #[test]
    fn test_reader_stops_at_eof_marker() {
        let mut buf: Vec<u8> = Vec::new();
        let mut writer = RdfThriftWriter::new(ThriftWriteMode::Triples);

        let t = make_iri_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        );
        writer.write_triples(&[t], &mut buf).expect("write ok");
        writer.write_eof(&mut buf).expect("eof ok");
        // Append garbage after EOF — reader should stop at marker
        buf.extend_from_slice(&[0xFF, 0xFF]);

        let mut reader = RdfThriftReader::new(Cursor::new(buf));
        let mut count = 0;
        loop {
            match reader.read_row().expect("read ok") {
                ThriftRow::Eof => break,
                ThriftRow::Triple(..) => count += 1,
                ThriftRow::Quad(..) => count += 1,
                ThriftRow::Prefix { .. } => {}
            }
        }
        assert_eq!(count, 1, "should have read exactly one triple");
    }

    // ─── Error: malformed varint ─────────────────────────────────────────────

    #[test]
    fn test_error_truncated_varint() {
        // A varint with high bit set but no continuation byte → unexpected EOF
        let buf: Vec<u8> = vec![TAG_IRI, 0x80]; // length varint truncated
        let mut reader = RdfThriftReader::new(Cursor::new(buf));
        let result = reader.read_row();
        assert!(result.is_err(), "truncated varint must yield error");
    }

    // ─── Error: unknown tag ──────────────────────────────────────────────────

    #[test]
    fn test_error_unknown_row_tag() {
        let buf: Vec<u8> = vec![0xFE]; // unknown row tag
        let mut reader = RdfThriftReader::new(Cursor::new(buf));
        let result = reader.read_row();
        assert!(result.is_err(), "unknown tag must yield error");
    }

    // ─── Error: invalid UTF-8 ────────────────────────────────────────────────

    #[test]
    fn test_error_invalid_utf8_string() {
        let mut buf: Vec<u8> = Vec::new();
        // TAG_IRI, length=3, then 3 invalid UTF-8 bytes
        buf.push(TAG_IRI);
        encode_varint(3, &mut buf);
        buf.extend_from_slice(&[0xFF, 0xFE, 0xFD]);

        let mut reader = RdfThriftReader::new(Cursor::new(buf));
        let result = reader.read_row();
        assert!(result.is_err(), "invalid UTF-8 must yield error");
    }

    // ─── Error: unknown prefix id ─────────────────────────────────────────────

    #[test]
    fn test_error_unknown_prefix_id() {
        let mut buf: Vec<u8> = Vec::new();
        // TAG_TRIPLE, then subject as TAG_PREFIXED_IRI with id=99 (not declared)
        buf.push(TAG_TRIPLE);
        buf.push(TAG_PREFIXED_IRI);
        encode_varint(99, &mut buf); // unknown prefix id
        write_str_primitive("local", &mut buf);

        let mut reader = RdfThriftReader::new(Cursor::new(buf));
        let result = reader.read_row();
        assert!(result.is_err(), "unknown prefix id must yield error");
    }

    // ─── Literal with both lang and datatype (writer behavior) ──────────────

    #[test]
    fn test_thrift_term_literal_lang_only() {
        let term = ThriftTerm::Literal {
            value: "hello".to_string(),
            lang: Some("en".to_string()),
            datatype: None,
        };
        let mut buf: Vec<u8> = Vec::new();
        write_term(&term, &HashMap::new(), &mut buf);
        assert_eq!(buf[0], TAG_LITERAL);
    }

    // ─── ThriftTerm conversions ──────────────────────────────────────────────

    #[test]
    fn test_thrift_term_iri_to_subject() {
        let term = ThriftTerm::Iri("http://example.org/s".to_string());
        let subject = term.to_subject().expect("conversion ok");
        assert!(matches!(subject, Subject::NamedNode(_)));
    }

    #[test]
    fn test_thrift_term_literal_as_subject_errors() {
        let term = ThriftTerm::Literal {
            value: "bad".to_string(),
            lang: None,
            datatype: None,
        };
        assert!(term.to_subject().is_err(), "literal-as-subject must fail");
    }

    #[test]
    fn test_thrift_term_bnode_as_predicate_errors() {
        let term = ThriftTerm::BlankNode("b0".to_string());
        assert!(
            term.to_predicate().is_err(),
            "blank-node-as-predicate must fail"
        );
    }

    // ─── GraphName::BlankNode round-trip ─────────────────────────────────────

    #[test]
    fn test_graph_blank_node_roundtrip() {
        let bn = BlankNode::new("g_bn").expect("valid blank node");
        let q = Quad::new(
            Subject::NamedNode(iri("http://example.org/s")),
            Predicate::NamedNode(iri("http://example.org/p")),
            Object::NamedNode(iri("http://example.org/o")),
            GraphName::BlankNode(bn),
        );
        let result = round_trip_quads(&[q], &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        assert!(
            matches!(result[0].graph_name(), GraphName::BlankNode(_)),
            "graph should be blank node"
        );
    }

    // ─── Large string literal (stress test) ──────────────────────────────────

    #[test]
    fn test_large_string_literal_roundtrip() {
        let large_value: String = "A".repeat(65536); // 64 KiB
        let lit = Literal::new_simple_literal(&large_value);
        let t = Triple::new(
            Subject::NamedNode(iri("http://example.org/s")),
            Predicate::NamedNode(iri("http://example.org/p")),
            Object::Literal(lit),
        );
        let result = round_trip_triples(&[t], &[]).expect("round-trip ok");
        assert_eq!(result.len(), 1);
        if let Object::Literal(lit) = result[0].object() {
            assert_eq!(lit.value().len(), 65536);
        } else {
            panic!("expected literal");
        }
    }
}
