//! Binary encoding and decoding for RDF terms
//!
//! This implementation is extracted and adapted from Oxigraph's binary_encoder.rs
//! to provide zero-dependency binary serialization with optimal storage efficiency.

use crate::store::encoding::{EncodedQuad, EncodedTerm, SmallString, StrHash};
use crate::OxirsError;
use std::io::{Cursor, Read};
use std::mem::size_of;

/// Maximum size of an encoded term in bytes
pub const WRITTEN_TERM_MAX_SIZE: usize = size_of::<u8>() + 2 * size_of::<StrHash>();

// Encoded term type constants
const TYPE_DEFAULT_GRAPH: u8 = 0;
const TYPE_NAMED_NODE_ID: u8 = 1;
const TYPE_NUMERICAL_BLANK_NODE_ID: u8 = 8;
const TYPE_SMALL_BLANK_NODE_ID: u8 = 9;
const TYPE_BIG_BLANK_NODE_ID: u8 = 10;
const TYPE_SMALL_STRING_LITERAL: u8 = 16;
const TYPE_BIG_STRING_LITERAL: u8 = 17;
const TYPE_SMALL_SMALL_LANG_STRING_LITERAL: u8 = 20;
const TYPE_SMALL_BIG_LANG_STRING_LITERAL: u8 = 21;
const TYPE_BIG_SMALL_LANG_STRING_LITERAL: u8 = 22;
const TYPE_BIG_BIG_LANG_STRING_LITERAL: u8 = 23;
const TYPE_SMALL_SMALL_TYPED_LITERAL: u8 = 24;
const TYPE_SMALL_BIG_TYPED_LITERAL: u8 = 25;
const TYPE_BIG_SMALL_TYPED_LITERAL: u8 = 26;
const TYPE_BIG_BIG_TYPED_LITERAL: u8 = 27;

/// Quad encoding variations for different sort orders
#[derive(Clone, Copy, Debug)]
pub enum QuadEncoding {
    /// Subject, Predicate, Object, Graph
    Spog,
    /// Predicate, Object, Subject, Graph
    Posg,
    /// Object, Subject, Predicate, Graph
    Ospg,
    /// Graph, Subject, Predicate, Object
    Gspo,
    /// Graph, Predicate, Object, Subject
    Gpos,
    /// Graph, Object, Subject, Predicate
    Gosp,
}

impl QuadEncoding {
    /// Decodes a quad from a buffer according to this encoding
    pub fn decode(self, buffer: &[u8]) -> Result<EncodedQuad, OxirsError> {
        let mut cursor = Cursor::new(buffer);
        match self {
            Self::Spog => decode_spog_quad(&mut cursor),
            Self::Posg => decode_posg_quad(&mut cursor),
            Self::Ospg => decode_ospg_quad(&mut cursor),
            Self::Gspo => decode_gspo_quad(&mut cursor),
            Self::Gpos => decode_gpos_quad(&mut cursor),
            Self::Gosp => decode_gosp_quad(&mut cursor),
        }
    }

    /// Encodes a quad to a buffer according to this encoding
    pub fn encode(self, quad: &EncodedQuad, buffer: &mut Vec<u8>) -> Result<(), OxirsError> {
        match self {
            Self::Spog => encode_spog_quad(quad, buffer),
            Self::Posg => encode_posg_quad(quad, buffer),
            Self::Ospg => encode_ospg_quad(quad, buffer),
            Self::Gspo => encode_gspo_quad(quad, buffer),
            Self::Gpos => encode_gpos_quad(quad, buffer),
            Self::Gosp => encode_gosp_quad(quad, buffer),
        }
    }
}

/// Encodes a term to a binary representation
pub fn encode_term(term: &EncodedTerm, buffer: &mut Vec<u8>) -> Result<(), OxirsError> {
    match term {
        EncodedTerm::DefaultGraph => {
            buffer.push(TYPE_DEFAULT_GRAPH);
        }
        EncodedTerm::NamedNode { iri_id } => {
            buffer.push(TYPE_NAMED_NODE_ID);
            buffer.extend_from_slice(&iri_id.to_be_bytes());
        }
        EncodedTerm::NumericalBlankNode { id } => {
            buffer.push(TYPE_NUMERICAL_BLANK_NODE_ID);
            buffer.extend_from_slice(id);
        }
        EncodedTerm::SmallBlankNode(id) => {
            buffer.push(TYPE_SMALL_BLANK_NODE_ID);
            encode_small_string(id, buffer);
        }
        EncodedTerm::BigBlankNode { id_id } => {
            buffer.push(TYPE_BIG_BLANK_NODE_ID);
            buffer.extend_from_slice(&id_id.to_be_bytes());
        }
        EncodedTerm::SmallStringLiteral(value) => {
            buffer.push(TYPE_SMALL_STRING_LITERAL);
            encode_small_string(value, buffer);
        }
        EncodedTerm::BigStringLiteral { value_id } => {
            buffer.push(TYPE_BIG_STRING_LITERAL);
            buffer.extend_from_slice(&value_id.to_be_bytes());
        }
        EncodedTerm::SmallSmallLangStringLiteral { value, language } => {
            buffer.push(TYPE_SMALL_SMALL_LANG_STRING_LITERAL);
            encode_small_string(value, buffer);
            encode_small_string(language, buffer);
        }
        EncodedTerm::SmallBigLangStringLiteral { value, language_id } => {
            buffer.push(TYPE_SMALL_BIG_LANG_STRING_LITERAL);
            encode_small_string(value, buffer);
            buffer.extend_from_slice(&language_id.to_be_bytes());
        }
        EncodedTerm::BigSmallLangStringLiteral { value_id, language } => {
            buffer.push(TYPE_BIG_SMALL_LANG_STRING_LITERAL);
            buffer.extend_from_slice(&value_id.to_be_bytes());
            encode_small_string(language, buffer);
        }
        EncodedTerm::BigBigLangStringLiteral {
            value_id,
            language_id,
        } => {
            buffer.push(TYPE_BIG_BIG_LANG_STRING_LITERAL);
            buffer.extend_from_slice(&value_id.to_be_bytes());
            buffer.extend_from_slice(&language_id.to_be_bytes());
        }
        EncodedTerm::SmallSmallTypedLiteral { value, datatype } => {
            buffer.push(TYPE_SMALL_SMALL_TYPED_LITERAL);
            encode_small_string(value, buffer);
            encode_small_string(datatype, buffer);
        }
        EncodedTerm::SmallBigTypedLiteral { value, datatype_id } => {
            buffer.push(TYPE_SMALL_BIG_TYPED_LITERAL);
            encode_small_string(value, buffer);
            buffer.extend_from_slice(&datatype_id.to_be_bytes());
        }
        EncodedTerm::BigSmallTypedLiteral { value_id, datatype } => {
            buffer.push(TYPE_BIG_SMALL_TYPED_LITERAL);
            buffer.extend_from_slice(&value_id.to_be_bytes());
            encode_small_string(datatype, buffer);
        }
        EncodedTerm::BigBigTypedLiteral {
            value_id,
            datatype_id,
        } => {
            buffer.push(TYPE_BIG_BIG_TYPED_LITERAL);
            buffer.extend_from_slice(&value_id.to_be_bytes());
            buffer.extend_from_slice(&datatype_id.to_be_bytes());
        }
    }
    Ok(())
}

/// Decodes a term from a binary representation
pub fn decode_term(buffer: &mut Cursor<&[u8]>) -> Result<EncodedTerm, OxirsError> {
    let mut type_byte = [0u8; 1];
    buffer
        .read_exact(&mut type_byte)
        .map_err(|e| OxirsError::Store(format!("Failed to read type byte: {e}")))?;

    match type_byte[0] {
        TYPE_DEFAULT_GRAPH => Ok(EncodedTerm::DefaultGraph),
        TYPE_NAMED_NODE_ID => {
            let iri_id = read_str_hash(buffer)?;
            Ok(EncodedTerm::NamedNode { iri_id })
        }
        TYPE_NUMERICAL_BLANK_NODE_ID => {
            let mut id = [0u8; 16];
            buffer
                .read_exact(&mut id)
                .map_err(|e| OxirsError::Store(format!("Failed to read blank node ID: {e}")))?;
            Ok(EncodedTerm::NumericalBlankNode { id })
        }
        TYPE_SMALL_BLANK_NODE_ID => {
            let id = decode_small_string(buffer)?;
            Ok(EncodedTerm::SmallBlankNode(id))
        }
        TYPE_BIG_BLANK_NODE_ID => {
            let id_id = read_str_hash(buffer)?;
            Ok(EncodedTerm::BigBlankNode { id_id })
        }
        TYPE_SMALL_STRING_LITERAL => {
            let value = decode_small_string(buffer)?;
            Ok(EncodedTerm::SmallStringLiteral(value))
        }
        TYPE_BIG_STRING_LITERAL => {
            let value_id = read_str_hash(buffer)?;
            Ok(EncodedTerm::BigStringLiteral { value_id })
        }
        TYPE_SMALL_SMALL_LANG_STRING_LITERAL => {
            let value = decode_small_string(buffer)?;
            let language = decode_small_string(buffer)?;
            Ok(EncodedTerm::SmallSmallLangStringLiteral { value, language })
        }
        TYPE_SMALL_BIG_LANG_STRING_LITERAL => {
            let value = decode_small_string(buffer)?;
            let language_id = read_str_hash(buffer)?;
            Ok(EncodedTerm::SmallBigLangStringLiteral { value, language_id })
        }
        TYPE_BIG_SMALL_LANG_STRING_LITERAL => {
            let value_id = read_str_hash(buffer)?;
            let language = decode_small_string(buffer)?;
            Ok(EncodedTerm::BigSmallLangStringLiteral { value_id, language })
        }
        TYPE_BIG_BIG_LANG_STRING_LITERAL => {
            let value_id = read_str_hash(buffer)?;
            let language_id = read_str_hash(buffer)?;
            Ok(EncodedTerm::BigBigLangStringLiteral {
                value_id,
                language_id,
            })
        }
        TYPE_SMALL_SMALL_TYPED_LITERAL => {
            let value = decode_small_string(buffer)?;
            let datatype = decode_small_string(buffer)?;
            Ok(EncodedTerm::SmallSmallTypedLiteral { value, datatype })
        }
        TYPE_SMALL_BIG_TYPED_LITERAL => {
            let value = decode_small_string(buffer)?;
            let datatype_id = read_str_hash(buffer)?;
            Ok(EncodedTerm::SmallBigTypedLiteral { value, datatype_id })
        }
        TYPE_BIG_SMALL_TYPED_LITERAL => {
            let value_id = read_str_hash(buffer)?;
            let datatype = decode_small_string(buffer)?;
            Ok(EncodedTerm::BigSmallTypedLiteral { value_id, datatype })
        }
        TYPE_BIG_BIG_TYPED_LITERAL => {
            let value_id = read_str_hash(buffer)?;
            let datatype_id = read_str_hash(buffer)?;
            Ok(EncodedTerm::BigBigTypedLiteral {
                value_id,
                datatype_id,
            })
        }
        type_byte => Err(OxirsError::Store(format!(
            "Unknown encoded term type: {}",
            type_byte
        ))),
    }
}

/// Encodes a small string
fn encode_small_string(small_string: &SmallString, buffer: &mut Vec<u8>) {
    buffer.push(small_string.len() as u8);
    buffer.extend_from_slice(small_string.as_str().as_bytes());
}

/// Decodes a small string
fn decode_small_string(buffer: &mut Cursor<&[u8]>) -> Result<SmallString, OxirsError> {
    let mut len_byte = [0u8; 1];
    buffer
        .read_exact(&mut len_byte)
        .map_err(|e| OxirsError::Store(format!("Failed to read string length: {e}")))?;

    let len = len_byte[0] as usize;
    if len > 15 {
        return Err(OxirsError::Store(format!(
            "SmallString length {} exceeds maximum of 15",
            len
        )));
    }

    let mut data = [0u8; 16];
    if len > 0 {
        buffer
            .read_exact(&mut data[..len])
            .map_err(|e| OxirsError::Store(format!("Failed to read string data: {e}")))?;
    }

    let s = std::str::from_utf8(&data[..len])
        .map_err(|e| OxirsError::Store(format!("Invalid UTF-8 in small string: {e}")))?;

    SmallString::new(s)
        .ok_or_else(|| OxirsError::Store("String too long for SmallString".to_string()))
}

/// Reads a StrHash from the buffer
fn read_str_hash(buffer: &mut Cursor<&[u8]>) -> Result<StrHash, OxirsError> {
    let mut hash_bytes = [0u8; 16];
    buffer
        .read_exact(&mut hash_bytes)
        .map_err(|e| OxirsError::Store(format!("Failed to read StrHash: {e}")))?;
    Ok(StrHash::from_be_bytes(hash_bytes))
}

// Quad encoding functions

fn encode_spog_quad(quad: &EncodedQuad, buffer: &mut Vec<u8>) -> Result<(), OxirsError> {
    encode_term(&quad.subject, buffer)?;
    encode_term(&quad.predicate, buffer)?;
    encode_term(&quad.object, buffer)?;
    encode_term(&quad.graph_name, buffer)
}

fn decode_spog_quad(cursor: &mut Cursor<&[u8]>) -> Result<EncodedQuad, OxirsError> {
    let subject = decode_term(cursor)?;
    let predicate = decode_term(cursor)?;
    let object = decode_term(cursor)?;
    let graph_name = decode_term(cursor)?;
    Ok(EncodedQuad::new(subject, predicate, object, graph_name))
}

fn encode_posg_quad(quad: &EncodedQuad, buffer: &mut Vec<u8>) -> Result<(), OxirsError> {
    encode_term(&quad.predicate, buffer)?;
    encode_term(&quad.object, buffer)?;
    encode_term(&quad.subject, buffer)?;
    encode_term(&quad.graph_name, buffer)
}

fn decode_posg_quad(cursor: &mut Cursor<&[u8]>) -> Result<EncodedQuad, OxirsError> {
    let predicate = decode_term(cursor)?;
    let object = decode_term(cursor)?;
    let subject = decode_term(cursor)?;
    let graph_name = decode_term(cursor)?;
    Ok(EncodedQuad::new(subject, predicate, object, graph_name))
}

fn encode_ospg_quad(quad: &EncodedQuad, buffer: &mut Vec<u8>) -> Result<(), OxirsError> {
    encode_term(&quad.object, buffer)?;
    encode_term(&quad.subject, buffer)?;
    encode_term(&quad.predicate, buffer)?;
    encode_term(&quad.graph_name, buffer)
}

fn decode_ospg_quad(cursor: &mut Cursor<&[u8]>) -> Result<EncodedQuad, OxirsError> {
    let object = decode_term(cursor)?;
    let subject = decode_term(cursor)?;
    let predicate = decode_term(cursor)?;
    let graph_name = decode_term(cursor)?;
    Ok(EncodedQuad::new(subject, predicate, object, graph_name))
}

fn encode_gspo_quad(quad: &EncodedQuad, buffer: &mut Vec<u8>) -> Result<(), OxirsError> {
    encode_term(&quad.graph_name, buffer)?;
    encode_term(&quad.subject, buffer)?;
    encode_term(&quad.predicate, buffer)?;
    encode_term(&quad.object, buffer)
}

fn decode_gspo_quad(cursor: &mut Cursor<&[u8]>) -> Result<EncodedQuad, OxirsError> {
    let graph_name = decode_term(cursor)?;
    let subject = decode_term(cursor)?;
    let predicate = decode_term(cursor)?;
    let object = decode_term(cursor)?;
    Ok(EncodedQuad::new(subject, predicate, object, graph_name))
}

fn encode_gpos_quad(quad: &EncodedQuad, buffer: &mut Vec<u8>) -> Result<(), OxirsError> {
    encode_term(&quad.graph_name, buffer)?;
    encode_term(&quad.predicate, buffer)?;
    encode_term(&quad.object, buffer)?;
    encode_term(&quad.subject, buffer)
}

fn decode_gpos_quad(cursor: &mut Cursor<&[u8]>) -> Result<EncodedQuad, OxirsError> {
    let graph_name = decode_term(cursor)?;
    let predicate = decode_term(cursor)?;
    let object = decode_term(cursor)?;
    let subject = decode_term(cursor)?;
    Ok(EncodedQuad::new(subject, predicate, object, graph_name))
}

fn encode_gosp_quad(quad: &EncodedQuad, buffer: &mut Vec<u8>) -> Result<(), OxirsError> {
    encode_term(&quad.graph_name, buffer)?;
    encode_term(&quad.object, buffer)?;
    encode_term(&quad.subject, buffer)?;
    encode_term(&quad.predicate, buffer)
}

fn decode_gosp_quad(cursor: &mut Cursor<&[u8]>) -> Result<EncodedQuad, OxirsError> {
    let graph_name = decode_term(cursor)?;
    let object = decode_term(cursor)?;
    let subject = decode_term(cursor)?;
    let predicate = decode_term(cursor)?;
    Ok(EncodedQuad::new(subject, predicate, object, graph_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::*;

    #[test]
    fn test_term_encoding_roundtrip() {
        let terms = vec![
            EncodedTerm::DefaultGraph,
            EncodedTerm::NamedNode {
                iri_id: StrHash::new("http://example.org/test"),
            },
            EncodedTerm::SmallBlankNode(SmallString::new("test").unwrap()),
            EncodedTerm::SmallStringLiteral(SmallString::new("hello").unwrap()),
            EncodedTerm::SmallSmallLangStringLiteral {
                value: SmallString::new("hello").unwrap(),
                language: SmallString::new("en").unwrap(),
            },
        ];

        for term in terms {
            let mut buffer = Vec::new();
            encode_term(&term, &mut buffer).unwrap();

            let mut cursor = Cursor::new(buffer.as_slice());
            let decoded = decode_term(&mut cursor).unwrap();

            assert_eq!(term, decoded);
        }
    }

    #[test]
    fn test_quad_encoding_roundtrip() {
        let quad = EncodedQuad::new(
            EncodedTerm::NamedNode {
                iri_id: StrHash::new("http://example.org/s"),
            },
            EncodedTerm::NamedNode {
                iri_id: StrHash::new("http://example.org/p"),
            },
            EncodedTerm::SmallStringLiteral(SmallString::new("object").unwrap()),
            EncodedTerm::DefaultGraph,
        );

        let encodings = [
            QuadEncoding::Spog,
            QuadEncoding::Posg,
            QuadEncoding::Ospg,
            QuadEncoding::Gspo,
            QuadEncoding::Gpos,
            QuadEncoding::Gosp,
        ];

        for encoding in &encodings {
            let mut buffer = Vec::new();
            encoding.encode(&quad, &mut buffer).unwrap();

            let decoded = encoding.decode(&buffer).unwrap();
            assert_eq!(quad, decoded);
        }
    }

    #[test]
    fn test_small_string_encoding() {
        let strings = ["", "test", "hello world", "emoji🚀"];

        for s in &strings {
            if let Some(small_string) = SmallString::new(s) {
                let mut buffer = Vec::new();
                encode_small_string(&small_string, &mut buffer);

                let mut cursor = Cursor::new(buffer.as_slice());
                let decoded = decode_small_string(&mut cursor).unwrap();

                assert_eq!(small_string.as_str(), decoded.as_str());
            }
        }
    }

    #[test]
    fn test_str_hash_encoding() {
        let hash = StrHash::new("http://example.org/test");
        let bytes = hash.to_be_bytes();
        let reconstructed = StrHash::from_be_bytes(bytes);
        assert_eq!(hash, reconstructed);
    }
}
