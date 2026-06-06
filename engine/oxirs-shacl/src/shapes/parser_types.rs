use oxirs_core::model::{BlankNode, NamedNode, Object, Subject};

use crate::{Result, ShaclError};

/// Resolve a named-node object from a graph triple query result.
pub(crate) fn extract_named_node(obj: &Object) -> Option<NamedNode> {
    if let Object::NamedNode(n) = obj {
        Some(n.clone())
    } else {
        None
    }
}

/// Resolve a string literal value from a graph triple query result.
pub(crate) fn extract_string_literal(obj: &Object) -> Option<String> {
    if let Object::Literal(lit) = obj {
        Some(lit.value().to_string())
    } else {
        None
    }
}

/// Resolve an integer literal value from a graph triple query result.
pub(crate) fn extract_integer_literal(obj: &Object) -> Option<i64> {
    if let Object::Literal(lit) = obj {
        lit.value().parse::<i64>().ok()
    } else {
        None
    }
}

/// Resolve a boolean literal value from a graph triple query result.
pub(crate) fn extract_bool_literal(obj: &Object) -> Option<bool> {
    if let Object::Literal(lit) = obj {
        match lit.value() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    } else {
        None
    }
}

/// Build an `rdf:type` named node.
pub(crate) fn rdf_type_node() -> Result<NamedNode> {
    NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid RDF type IRI: {e}")))
}

/// Build an `rdf:first` named node.
pub(crate) fn rdf_first_node() -> Result<NamedNode> {
    NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#first")
        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid first property IRI: {e}")))
}

/// Build an `rdf:rest` named node.
pub(crate) fn rdf_rest_node() -> Result<NamedNode> {
    NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest")
        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid rest property IRI: {e}")))
}

/// Build an `rdf:nil` named node.
pub(crate) fn rdf_nil_node() -> Result<NamedNode> {
    NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil")
        .map_err(|e| ShaclError::ShapeParsing(format!("Invalid nil IRI: {e}")))
}

/// Convert a blank-node `Subject` to a `Subject::BlankNode`.
pub(crate) fn blank_as_subject(blank_node: &BlankNode) -> Subject {
    Subject::BlankNode(blank_node.clone())
}
