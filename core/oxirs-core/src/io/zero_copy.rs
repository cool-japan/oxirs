//! Zero-copy serialization for RDF data
//!
//! This module provides efficient serialization/deserialization with minimal memory copies.

use crate::model::{Term, Triple};
use crate::OxirsError;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use memmap2::{Mmap, MmapMut};
use std::borrow::Cow;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::str;

/// Trait for types that can be serialized with zero-copy
pub trait ZeroCopySerialize {
    /// Serialize to a writer
    fn serialize_to<W: Write>(&self, writer: &mut W) -> io::Result<()>;

    /// Get the serialized size in bytes
    fn serialized_size(&self) -> usize;

    /// Serialize to a byte buffer
    fn serialize_to_bytes(&self, buf: &mut BytesMut);
}

/// Trait for types that can be deserialized with zero-copy
pub trait ZeroCopyDeserialize<'a>: Sized {
    /// Deserialize from a byte slice
    fn deserialize_from(data: &'a [u8]) -> Result<(Self, &'a [u8]), OxirsError>;

    /// Deserialize from a Bytes buffer
    fn deserialize_from_bytes(buf: &mut Bytes) -> Result<Self, OxirsError>;
}

/// Zero-copy string that can be borrowed or owned
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ZeroCopyStr<'a>(pub Cow<'a, str>);

impl<'a> ZeroCopyStr<'a> {
    pub fn new_borrowed(s: &'a str) -> Self {
        ZeroCopyStr(Cow::Borrowed(s))
    }

    pub fn new_owned(s: String) -> Self {
        ZeroCopyStr(Cow::Owned(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_owned(self) -> String {
        self.0.into_owned()
    }
}

/// Zero-copy IRI representation
#[derive(Debug, Clone)]
pub struct ZeroCopyIri<'a> {
    value: ZeroCopyStr<'a>,
}

impl<'a> ZeroCopyIri<'a> {
    pub fn new(value: ZeroCopyStr<'a>) -> Self {
        Self { value }
    }

    pub fn as_str(&self) -> &str {
        self.value.as_str()
    }
}

/// Zero-copy blank node
#[derive(Debug, Clone)]
pub struct ZeroCopyBlankNode<'a> {
    id: ZeroCopyStr<'a>,
}

impl<'a> ZeroCopyBlankNode<'a> {
    pub fn new(id: ZeroCopyStr<'a>) -> Self {
        Self { id }
    }

    pub fn id(&self) -> &str {
        self.id.as_str()
    }
}

/// Zero-copy literal
#[derive(Debug, Clone)]
pub struct ZeroCopyLiteral<'a> {
    value: ZeroCopyStr<'a>,
    language: Option<ZeroCopyStr<'a>>,
    datatype: Option<ZeroCopyIri<'a>>,
}

impl<'a> ZeroCopyLiteral<'a> {
    pub fn new_simple(value: ZeroCopyStr<'a>) -> Self {
        Self {
            value,
            language: None,
            datatype: None,
        }
    }

    pub fn new_language_tagged(value: ZeroCopyStr<'a>, language: ZeroCopyStr<'a>) -> Self {
        Self {
            value,
            language: Some(language),
            datatype: None,
        }
    }

    pub fn new_typed(value: ZeroCopyStr<'a>, datatype: ZeroCopyIri<'a>) -> Self {
        Self {
            value,
            language: None,
            datatype: Some(datatype),
        }
    }

    pub fn value(&self) -> &str {
        self.value.as_str()
    }

    pub fn language(&self) -> Option<&str> {
        self.language.as_ref().map(|l| l.as_str())
    }

    pub fn datatype(&self) -> Option<&ZeroCopyIri<'a>> {
        self.datatype.as_ref()
    }
}

/// Zero-copy term
#[derive(Debug, Clone)]
pub enum ZeroCopyTerm<'a> {
    NamedNode(ZeroCopyIri<'a>),
    BlankNode(ZeroCopyBlankNode<'a>),
    Literal(ZeroCopyLiteral<'a>),
    Variable(ZeroCopyStr<'a>),
    QuotedTriple(Box<ZeroCopyTriple<'a>>),
}

/// Zero-copy triple
#[derive(Debug, Clone)]
pub struct ZeroCopyTriple<'a> {
    pub subject: ZeroCopyTerm<'a>,
    pub predicate: ZeroCopyIri<'a>,
    pub object: ZeroCopyTerm<'a>,
}

/// Zero-copy quad
#[derive(Debug, Clone)]
pub struct ZeroCopyQuad<'a> {
    pub subject: ZeroCopyTerm<'a>,
    pub predicate: ZeroCopyIri<'a>,
    pub object: ZeroCopyTerm<'a>,
    pub graph: Option<ZeroCopyTerm<'a>>,
}

// Binary format constants
#[allow(dead_code)]
const FORMAT_VERSION: u8 = 1;
const TERM_NAMED_NODE: u8 = 0;
const TERM_BLANK_NODE: u8 = 1;
const TERM_LITERAL_SIMPLE: u8 = 2;
const TERM_LITERAL_LANG: u8 = 3;
const TERM_LITERAL_TYPED: u8 = 4;
const TERM_VARIABLE: u8 = 5;
const TERM_QUOTED_TRIPLE: u8 = 6;

/// Write a length-prefixed string
fn write_string<W: Write>(writer: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    let len = bytes.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(bytes)?;
    Ok(())
}

/// Read a length-prefixed string (zero-copy)
fn read_string(data: &[u8]) -> Result<(&str, &[u8]), OxirsError> {
    if data.len() < 4 {
        return Err(OxirsError::Parse(
            "Insufficient data for string length".into(),
        ));
    }

    let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let data = &data[4..];

    if data.len() < len {
        return Err(OxirsError::Parse("Insufficient data for string".into()));
    }

    let s = str::from_utf8(&data[..len])
        .map_err(|e| OxirsError::Parse(format!("Invalid UTF-8: {e}")))?;

    Ok((s, &data[len..]))
}

impl ZeroCopySerialize for Term {
    fn serialize_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            Term::NamedNode(n) => {
                writer.write_all(&[TERM_NAMED_NODE])?;
                write_string(writer, n.as_str())?;
            }
            Term::BlankNode(b) => {
                writer.write_all(&[TERM_BLANK_NODE])?;
                write_string(writer, b.as_str())?;
            }
            Term::Literal(l) => {
                if let Some(lang) = l.language() {
                    writer.write_all(&[TERM_LITERAL_LANG])?;
                    write_string(writer, l.value())?;
                    write_string(writer, lang)?;
                } else if l.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    writer.write_all(&[TERM_LITERAL_TYPED])?;
                    write_string(writer, l.value())?;
                    write_string(writer, l.datatype().as_str())?;
                } else {
                    writer.write_all(&[TERM_LITERAL_SIMPLE])?;
                    write_string(writer, l.value())?;
                }
            }
            Term::Variable(v) => {
                writer.write_all(&[TERM_VARIABLE])?;
                write_string(writer, v.as_str())?;
            }
            Term::QuotedTriple(qt) => {
                writer.write_all(&[TERM_QUOTED_TRIPLE])?;
                // Convert and serialize the inner triple components
                let subject_term = match qt.subject() {
                    crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
                    crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
                    crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
                    crate::model::Subject::QuotedTriple(nested_qt) => {
                        Term::QuotedTriple(nested_qt.clone())
                    }
                };
                subject_term.serialize_to(writer)?;

                let predicate_term = match qt.predicate() {
                    crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
                    crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
                };
                predicate_term.serialize_to(writer)?;

                let object_term = match qt.object() {
                    crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
                    crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
                    crate::model::Object::Literal(l) => Term::Literal(l.clone()),
                    crate::model::Object::Variable(v) => Term::Variable(v.clone()),
                    crate::model::Object::QuotedTriple(nested_qt) => {
                        Term::QuotedTriple(nested_qt.clone())
                    }
                };
                object_term.serialize_to(writer)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self) -> usize {
        1 + match self {
            Term::NamedNode(n) => 4 + n.as_str().len(),
            Term::BlankNode(b) => 4 + b.as_str().len(),
            Term::Literal(l) => {
                if let Some(lang) = l.language() {
                    4 + l.value().len() + 4 + lang.len()
                } else if l.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    4 + l.value().len() + 4 + l.datatype().as_str().len()
                } else {
                    4 + l.value().len()
                }
            }
            Term::Variable(v) => 4 + v.as_str().len(),
            Term::QuotedTriple(qt) => {
                // Convert and calculate sizes for the inner triple components
                let subject_term = match qt.subject() {
                    crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
                    crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
                    crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
                    crate::model::Subject::QuotedTriple(nested_qt) => {
                        Term::QuotedTriple(nested_qt.clone())
                    }
                };

                let predicate_term = match qt.predicate() {
                    crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
                    crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
                };

                let object_term = match qt.object() {
                    crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
                    crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
                    crate::model::Object::Literal(l) => Term::Literal(l.clone()),
                    crate::model::Object::Variable(v) => Term::Variable(v.clone()),
                    crate::model::Object::QuotedTriple(nested_qt) => {
                        Term::QuotedTriple(nested_qt.clone())
                    }
                };

                subject_term.serialized_size()
                    + predicate_term.serialized_size()
                    + object_term.serialized_size()
            }
        }
    }

    fn serialize_to_bytes(&self, buf: &mut BytesMut) {
        match *self {
            Term::NamedNode(ref n) => {
                buf.put_u8(TERM_NAMED_NODE);
                buf.put_u32_le(n.as_str().len() as u32);
                buf.put_slice(n.as_str().as_bytes());
            }
            Term::BlankNode(ref b) => {
                buf.put_u8(TERM_BLANK_NODE);
                buf.put_u32_le(b.as_str().len() as u32);
                buf.put_slice(b.as_str().as_bytes());
            }
            Term::Literal(ref l) => {
                if let Some(lang) = l.language() {
                    buf.put_u8(TERM_LITERAL_LANG);
                    buf.put_u32_le(l.value().len() as u32);
                    buf.put_slice(l.value().as_bytes());
                    buf.put_u32_le(lang.len() as u32);
                    buf.put_slice(lang.as_bytes());
                } else if l.datatype().as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                    buf.put_u8(TERM_LITERAL_TYPED);
                    buf.put_u32_le(l.value().len() as u32);
                    buf.put_slice(l.value().as_bytes());
                    buf.put_u32_le(l.datatype().as_str().len() as u32);
                    buf.put_slice(l.datatype().as_str().as_bytes());
                } else {
                    buf.put_u8(TERM_LITERAL_SIMPLE);
                    buf.put_u32_le(l.value().len() as u32);
                    buf.put_slice(l.value().as_bytes());
                }
            }
            Term::Variable(ref v) => {
                buf.put_u8(TERM_VARIABLE);
                buf.put_u32_le(v.as_str().len() as u32);
                buf.put_slice(v.as_str().as_bytes());
            }
            Term::QuotedTriple(ref qt) => {
                buf.put_u8(TERM_QUOTED_TRIPLE);
                // Convert and serialize the inner triple components
                let subject_term = match qt.subject() {
                    crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
                    crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
                    crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
                    crate::model::Subject::QuotedTriple(nested_qt) => {
                        Term::QuotedTriple(nested_qt.clone())
                    }
                };
                subject_term.serialize_to_bytes(buf);

                let predicate_term = match qt.predicate() {
                    crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
                    crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
                };
                predicate_term.serialize_to_bytes(buf);

                let object_term = match qt.object() {
                    crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
                    crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
                    crate::model::Object::Literal(l) => Term::Literal(l.clone()),
                    crate::model::Object::Variable(v) => Term::Variable(v.clone()),
                    crate::model::Object::QuotedTriple(nested_qt) => {
                        Term::QuotedTriple(nested_qt.clone())
                    }
                };
                object_term.serialize_to_bytes(buf);
            }
        }
    }
}

impl<'a> ZeroCopyDeserialize<'a> for ZeroCopyTerm<'a> {
    fn deserialize_from(data: &'a [u8]) -> Result<(Self, &'a [u8]), OxirsError> {
        if data.is_empty() {
            return Err(OxirsError::Parse("No data for term type".into()));
        }

        let term_type = data[0];
        let data = &data[1..];

        match term_type {
            TERM_NAMED_NODE => {
                let (iri, rest) = read_string(data)?;
                Ok((
                    ZeroCopyTerm::NamedNode(ZeroCopyIri::new(ZeroCopyStr::new_borrowed(iri))),
                    rest,
                ))
            }
            TERM_BLANK_NODE => {
                let (id, rest) = read_string(data)?;
                Ok((
                    ZeroCopyTerm::BlankNode(ZeroCopyBlankNode::new(ZeroCopyStr::new_borrowed(id))),
                    rest,
                ))
            }
            TERM_LITERAL_SIMPLE => {
                let (value, rest) = read_string(data)?;
                Ok((
                    ZeroCopyTerm::Literal(ZeroCopyLiteral::new_simple(ZeroCopyStr::new_borrowed(
                        value,
                    ))),
                    rest,
                ))
            }
            TERM_LITERAL_LANG => {
                let (value, data) = read_string(data)?;
                let (lang, rest) = read_string(data)?;
                Ok((
                    ZeroCopyTerm::Literal(ZeroCopyLiteral::new_language_tagged(
                        ZeroCopyStr::new_borrowed(value),
                        ZeroCopyStr::new_borrowed(lang),
                    )),
                    rest,
                ))
            }
            TERM_LITERAL_TYPED => {
                let (value, data) = read_string(data)?;
                let (datatype, rest) = read_string(data)?;
                Ok((
                    ZeroCopyTerm::Literal(ZeroCopyLiteral::new_typed(
                        ZeroCopyStr::new_borrowed(value),
                        ZeroCopyIri::new(ZeroCopyStr::new_borrowed(datatype)),
                    )),
                    rest,
                ))
            }
            TERM_VARIABLE => {
                let (name, rest) = read_string(data)?;
                Ok((
                    ZeroCopyTerm::Variable(ZeroCopyStr::new_borrowed(name)),
                    rest,
                ))
            }
            TERM_QUOTED_TRIPLE => {
                // Deserialize the inner triple components
                let (subject, data) = ZeroCopyTerm::deserialize_from(data)?;
                let (predicate_term, data) = ZeroCopyTerm::deserialize_from(data)?;
                let (object, rest) = ZeroCopyTerm::deserialize_from(data)?;

                // Ensure predicate is a named node
                let predicate = match predicate_term {
                    ZeroCopyTerm::NamedNode(iri) => iri,
                    _ => return Err(OxirsError::Parse("Predicate must be a named node".into())),
                };

                let triple = ZeroCopyTriple {
                    subject,
                    predicate,
                    object,
                };

                Ok((ZeroCopyTerm::QuotedTriple(Box::new(triple)), rest))
            }
            _ => Err(OxirsError::Parse(format!("Unknown term type: {term_type}"))),
        }
    }

    fn deserialize_from_bytes(buf: &mut Bytes) -> Result<Self, OxirsError> {
        if buf.remaining() == 0 {
            return Err(OxirsError::Parse("No data for term type".into()));
        }

        let term_type = buf.get_u8();

        match term_type {
            TERM_NAMED_NODE => {
                let len = buf.get_u32_le() as usize;
                let bytes = buf.split_to(len);
                let iri = str::from_utf8(&bytes)
                    .map_err(|e| OxirsError::Parse(format!("Invalid UTF-8: {e}")))?;
                Ok(ZeroCopyTerm::NamedNode(ZeroCopyIri::new(
                    ZeroCopyStr::new_owned(iri.to_string()),
                )))
            }
            TERM_BLANK_NODE => {
                let len = buf.get_u32_le() as usize;
                let bytes = buf.split_to(len);
                let id = str::from_utf8(&bytes)
                    .map_err(|e| OxirsError::Parse(format!("Invalid UTF-8: {e}")))?;
                Ok(ZeroCopyTerm::BlankNode(ZeroCopyBlankNode::new(
                    ZeroCopyStr::new_owned(id.to_string()),
                )))
            }
            TERM_LITERAL_SIMPLE => {
                let len = buf.get_u32_le() as usize;
                let bytes = buf.split_to(len);
                let value = str::from_utf8(&bytes)
                    .map_err(|e| OxirsError::Parse(format!("Invalid UTF-8: {e}")))?;
                Ok(ZeroCopyTerm::Literal(ZeroCopyLiteral::new_simple(
                    ZeroCopyStr::new_owned(value.to_string()),
                )))
            }
            TERM_QUOTED_TRIPLE => {
                // Deserialize the inner triple components
                let subject = ZeroCopyTerm::deserialize_from_bytes(buf)?;
                let predicate_term = ZeroCopyTerm::deserialize_from_bytes(buf)?;
                let object = ZeroCopyTerm::deserialize_from_bytes(buf)?;

                // Ensure predicate is a named node
                let predicate = match predicate_term {
                    ZeroCopyTerm::NamedNode(iri) => iri,
                    _ => return Err(OxirsError::Parse("Predicate must be a named node".into())),
                };

                let triple = ZeroCopyTriple {
                    subject,
                    predicate,
                    object,
                };

                Ok(ZeroCopyTerm::QuotedTriple(Box::new(triple)))
            }
            _ => Err(OxirsError::Parse(format!("Unknown term type: {term_type}"))),
        }
    }
}

impl ZeroCopySerialize for Triple {
    fn serialize_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Convert Subject to Term
        let subject_term = match self.subject() {
            crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
            crate::model::Subject::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        subject_term.serialize_to(writer)?;

        // Convert Predicate to Term
        let predicate_term = match self.predicate() {
            crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
        };
        predicate_term.serialize_to(writer)?;

        // Convert Object to Term
        let object_term = match self.object() {
            crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Object::Literal(l) => Term::Literal(l.clone()),
            crate::model::Object::Variable(v) => Term::Variable(v.clone()),
            crate::model::Object::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        object_term.serialize_to(writer)?;

        Ok(())
    }

    fn serialized_size(&self) -> usize {
        // Convert to Terms and use their serialized_size
        let subject_term: Term = match self.subject() {
            crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
            crate::model::Subject::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };

        let predicate_term: Term = match self.predicate() {
            crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
        };

        let object_term: Term = match self.object() {
            crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Object::Literal(l) => Term::Literal(l.clone()),
            crate::model::Object::Variable(v) => Term::Variable(v.clone()),
            crate::model::Object::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };

        subject_term.serialized_size()
            + predicate_term.serialized_size()
            + object_term.serialized_size()
    }

    fn serialize_to_bytes(&self, buf: &mut BytesMut) {
        // Convert and serialize subject
        let subject_term: Term = match self.subject() {
            crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
            crate::model::Subject::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        subject_term.serialize_to_bytes(buf);

        // Convert and serialize predicate
        let predicate_term: Term = match self.predicate() {
            crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
        };
        predicate_term.serialize_to_bytes(buf);

        // Convert and serialize object
        let object_term: Term = match self.object() {
            crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Object::Literal(l) => Term::Literal(l.clone()),
            crate::model::Object::Variable(v) => Term::Variable(v.clone()),
            crate::model::Object::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        object_term.serialize_to_bytes(buf);
    }
}

impl<'a> ZeroCopyDeserialize<'a> for ZeroCopyTriple<'a> {
    fn deserialize_from(data: &'a [u8]) -> Result<(Self, &'a [u8]), OxirsError> {
        let (subject, data) = ZeroCopyTerm::deserialize_from(data)?;
        let (predicate, data) = ZeroCopyTerm::deserialize_from(data)?;
        let (object, data) = ZeroCopyTerm::deserialize_from(data)?;

        let predicate_iri = match predicate {
            ZeroCopyTerm::NamedNode(iri) => iri,
            _ => return Err(OxirsError::Parse("Predicate must be IRI".into())),
        };

        Ok((
            ZeroCopyTriple {
                subject,
                predicate: predicate_iri,
                object,
            },
            data,
        ))
    }

    fn deserialize_from_bytes(buf: &mut Bytes) -> Result<Self, OxirsError> {
        let subject = ZeroCopyTerm::deserialize_from_bytes(buf)?;
        let predicate = ZeroCopyTerm::deserialize_from_bytes(buf)?;
        let object = ZeroCopyTerm::deserialize_from_bytes(buf)?;

        let predicate_iri = match predicate {
            ZeroCopyTerm::NamedNode(iri) => iri,
            _ => return Err(OxirsError::Parse("Predicate must be IRI".into())),
        };

        Ok(ZeroCopyTriple {
            subject,
            predicate: predicate_iri,
            object,
        })
    }
}

/// Memory-mapped file for zero-copy reading
pub struct MmapReader {
    _file: File,
    mmap: Mmap,
}

impl MmapReader {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { _file: file, mmap })
    }

    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Iterate over triples in the file with zero-copy
    pub fn iter_triples(&self) -> ZeroCopyTripleIterator<'_> {
        ZeroCopyTripleIterator {
            data: self.data(),
            offset: 0,
        }
    }
}

/// Iterator over zero-copy triples
pub struct ZeroCopyTripleIterator<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Iterator for ZeroCopyTripleIterator<'a> {
    type Item = Result<ZeroCopyTriple<'a>, OxirsError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }

        match ZeroCopyTriple::deserialize_from(&self.data[self.offset..]) {
            Ok((triple, rest)) => {
                self.offset = self.data.len() - rest.len();
                Some(Ok(triple))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Memory-mapped file for zero-copy writing
pub struct MmapWriter {
    file: File,
    mmap: MmapMut,
    position: usize,
}

impl MmapWriter {
    pub fn new<P: AsRef<Path>>(path: P, capacity: usize) -> io::Result<Self> {
        let file = File::create(path)?;
        file.set_len(capacity as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self {
            file,
            mmap,
            position: 0,
        })
    }

    pub fn write_triple(&mut self, triple: &Triple) -> io::Result<()> {
        let size = triple.serialized_size();
        if self.position + size > self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "MmapWriter capacity exceeded",
            ));
        }

        let mut cursor = io::Cursor::new(&mut self.mmap[self.position..]);
        triple.serialize_to(&mut cursor)?;
        self.position += size;
        Ok(())
    }

    pub fn finalize(self) -> io::Result<()> {
        // Truncate file to actual size
        self.file.set_len(self.position as u64)?;
        self.mmap.flush()?;
        Ok(())
    }
}

// Implement ZeroCopySerialize for Subject, Predicate, and Object
impl ZeroCopySerialize for crate::model::Subject {
    fn serialize_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let term: Term = match self {
            crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
            crate::model::Subject::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        term.serialize_to(writer)
    }

    fn serialized_size(&self) -> usize {
        let term: Term = match self {
            crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
            crate::model::Subject::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        term.serialized_size()
    }

    fn serialize_to_bytes(&self, buf: &mut BytesMut) {
        let term: Term = match self {
            crate::model::Subject::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Subject::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Subject::Variable(v) => Term::Variable(v.clone()),
            crate::model::Subject::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        term.serialize_to_bytes(buf)
    }
}

impl ZeroCopySerialize for crate::model::Predicate {
    fn serialize_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let term: Term = match self {
            crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
        };
        term.serialize_to(writer)
    }

    fn serialized_size(&self) -> usize {
        let term: Term = match self {
            crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
        };
        term.serialized_size()
    }

    fn serialize_to_bytes(&self, buf: &mut BytesMut) {
        let term: Term = match self {
            crate::model::Predicate::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Predicate::Variable(v) => Term::Variable(v.clone()),
        };
        term.serialize_to_bytes(buf)
    }
}

impl ZeroCopySerialize for crate::model::Object {
    fn serialize_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let term: Term = match self {
            crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Object::Literal(l) => Term::Literal(l.clone()),
            crate::model::Object::Variable(v) => Term::Variable(v.clone()),
            crate::model::Object::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        term.serialize_to(writer)
    }

    fn serialized_size(&self) -> usize {
        let term: Term = match self {
            crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Object::Literal(l) => Term::Literal(l.clone()),
            crate::model::Object::Variable(v) => Term::Variable(v.clone()),
            crate::model::Object::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        term.serialized_size()
    }

    fn serialize_to_bytes(&self, buf: &mut BytesMut) {
        let term: Term = match self {
            crate::model::Object::NamedNode(n) => Term::NamedNode(n.clone()),
            crate::model::Object::BlankNode(b) => Term::BlankNode(b.clone()),
            crate::model::Object::Literal(l) => Term::Literal(l.clone()),
            crate::model::Object::Variable(v) => Term::Variable(v.clone()),
            crate::model::Object::QuotedTriple(qt) => Term::QuotedTriple(qt.clone()),
        };
        term.serialize_to_bytes(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Literal, NamedNode};

    #[test]
    fn test_term_serialization() {
        let term = Term::NamedNode(NamedNode::new("http://example.org/test").unwrap());

        let mut buf = Vec::new();
        term.serialize_to(&mut buf).unwrap();

        let (deserialized, rest) = ZeroCopyTerm::deserialize_from(&buf).unwrap();
        assert!(rest.is_empty());

        match deserialized {
            ZeroCopyTerm::NamedNode(iri) => {
                assert_eq!(iri.as_str(), "http://example.org/test");
            }
            _ => panic!("Wrong term type"),
        }
    }

    #[test]
    fn test_triple_serialization() {
        let triple = Triple::new(
            NamedNode::new("http://example.org/subject").unwrap(),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Literal::new("Object"),
        );

        let mut buf = Vec::new();
        triple.serialize_to(&mut buf).unwrap();

        let (deserialized, rest) = ZeroCopyTriple::deserialize_from(&buf).unwrap();
        assert!(rest.is_empty());

        match &deserialized.subject {
            ZeroCopyTerm::NamedNode(iri) => {
                assert_eq!(iri.as_str(), "http://example.org/subject");
            }
            _ => panic!("Wrong subject type"),
        }

        assert_eq!(
            deserialized.predicate.as_str(),
            "http://example.org/predicate"
        );

        match &deserialized.object {
            ZeroCopyTerm::Literal(lit) => {
                assert_eq!(lit.value(), "Object");
            }
            _ => panic!("Wrong object type"),
        }
    }

    #[test]
    fn test_bytes_serialization() {
        let term = Term::Literal(Literal::new("Hello, World!"));

        let mut buf = BytesMut::with_capacity(term.serialized_size());
        term.serialize_to_bytes(&mut buf);

        let mut bytes = buf.freeze();
        let deserialized = ZeroCopyTerm::deserialize_from_bytes(&mut bytes).unwrap();

        match deserialized {
            ZeroCopyTerm::Literal(lit) => {
                assert_eq!(lit.value(), "Hello, World!");
            }
            _ => panic!("Wrong term type"),
        }
    }
}
