//! Internal types for the RDF/XML parser.
//!
//! Contains RDF vocabulary constants, internal state machine types,
//! and the parser's core data structures.

use crate::model::{BlankNode, NamedNode, NamedOrBlankNode};
use oxiri::Iri;
use quick_xml::{NsReader, Writer};
use std::collections::{HashMap, HashSet};

// ──────────────────────────────────────────────────────────────────────────────
// RDF vocabulary constants
// ──────────────────────────────────────────────────────────────────────────────

pub(super) const RDF_ABOUT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#about";
pub(super) const RDF_ABOUT_EACH: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#aboutEach";
pub(super) const RDF_ABOUT_EACH_PREFIX: &str =
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#aboutEachPrefix";
pub(super) const RDF_BAG_ID: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#bagID";
pub(super) const RDF_DATATYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#datatype";
pub(super) const RDF_DESCRIPTION: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Description";
pub(super) const RDF_ID: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#ID";
pub(super) const RDF_LI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#li";
pub(super) const RDF_NODE_ID: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nodeID";
pub(super) const RDF_PARSE_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#parseType";
pub(super) const RDF_RDF: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#RDF";
pub(super) const RDF_RESOURCE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#resource";

pub(super) const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
pub(super) const RDF_NIL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";
pub(super) const RDF_FIRST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
pub(super) const RDF_REST: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
pub(super) const RDF_STATEMENT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement";
pub(super) const RDF_SUBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject";
pub(super) const RDF_PREDICATE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate";
pub(super) const RDF_OBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object";
pub(super) const RDF_XML_LITERAL: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral";

pub(super) const RESERVED_RDF_ELEMENTS: [&str; 11] = [
    RDF_ABOUT,
    RDF_ABOUT_EACH,
    RDF_ABOUT_EACH_PREFIX,
    RDF_BAG_ID,
    RDF_DATATYPE,
    RDF_ID,
    RDF_LI,
    RDF_NODE_ID,
    RDF_PARSE_TYPE,
    RDF_RDF,
    RDF_RESOURCE,
];
pub(super) const RESERVED_RDF_ATTRIBUTES: [&str; 5] = [
    RDF_ABOUT_EACH,
    RDF_ABOUT_EACH_PREFIX,
    RDF_LI,
    RDF_RDF,
    RDF_RESOURCE,
];

// ──────────────────────────────────────────────────────────────────────────────
// NodeOrText
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub(super) enum NodeOrText {
    Node(NamedOrBlankNode),
    Text(String),
}

// ──────────────────────────────────────────────────────────────────────────────
// RdfXmlState — parser state machine
// ──────────────────────────────────────────────────────────────────────────────

pub(super) enum RdfXmlState {
    Doc {
        base_iri: Option<Iri<String>>,
    },
    Rdf {
        base_iri: Option<Iri<String>>,
        language: Option<String>,
    },
    NodeElt {
        base_iri: Option<Iri<String>>,
        language: Option<String>,
        subject: NamedOrBlankNode,
        li_counter: u64,
    },
    PropertyElt {
        // Resource, Literal or Empty property element
        iri: NamedNode,
        base_iri: Option<Iri<String>>,
        language: Option<String>,
        subject: NamedOrBlankNode,
        object: Option<NodeOrText>,
        id_attr: Option<NamedNode>,
        datatype_attr: Option<NamedNode>,
    },
    ParseTypeCollectionPropertyElt {
        iri: NamedNode,
        base_iri: Option<Iri<String>>,
        language: Option<String>,
        subject: NamedOrBlankNode,
        objects: Vec<NamedOrBlankNode>,
        id_attr: Option<NamedNode>,
    },
    ParseTypeLiteralPropertyElt {
        iri: NamedNode,
        base_iri: Option<Iri<String>>,
        language: Option<String>,
        subject: NamedOrBlankNode,
        writer: Writer<Vec<u8>>,
        id_attr: Option<NamedNode>,
        emit: bool, // false for parseTypeOtherPropertyElt support
    },
}

// ──────────────────────────────────────────────────────────────────────────────
// NodeElementAttributes
// ──────────────────────────────────────────────────────────────────────────────

/// Attributes for a node element
pub(super) struct NodeElementAttributes {
    pub(super) id_attr: Option<NamedNode>,
    pub(super) node_id_attr: Option<BlankNode>,
    pub(super) about_attr: Option<NamedNode>,
    pub(super) type_attr: Option<NamedNode>,
    pub(super) property_attrs: Vec<(NamedNode, String)>,
}

// ──────────────────────────────────────────────────────────────────────────────
// InternalRdfXmlParser
// ──────────────────────────────────────────────────────────────────────────────

pub(super) struct InternalRdfXmlParser<R> {
    pub(super) reader: NsReader<R>,
    pub(super) state: Vec<RdfXmlState>,
    pub(super) custom_entities: HashMap<String, String>,
    pub(super) in_literal_depth: usize,
    pub(super) known_rdf_id: HashSet<String>,
    pub(super) is_end: bool,
    pub(super) lenient: bool,
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper functions
// ──────────────────────────────────────────────────────────────────────────────

pub(super) fn is_object_defined(object: &Option<NodeOrText>) -> bool {
    match object {
        Some(NodeOrText::Node(_)) => true,
        Some(NodeOrText::Text(t)) => !t.bytes().all(is_whitespace),
        None => false,
    }
}

pub(super) fn is_whitespace(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

pub(super) fn is_utf8(encoding: &[u8]) -> bool {
    matches!(
        encoding.to_ascii_lowercase().as_slice(),
        b"unicode-1-1-utf-8"
            | b"unicode11utf8"
            | b"unicode20utf8"
            | b"utf-8"
            | b"utf8"
            | b"x-unicode20utf8"
    )
}
