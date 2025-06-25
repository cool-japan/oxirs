//! Common node types used across RDF/XML and JSON-LD
//!
//! This module provides node types that are shared between different
//! RDF serialization formats.

use crate::model::{NamedNode, BlankNode};

/// A node that can be either a named node (IRI) or a blank node
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NamedOrBlankNode {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
}

impl From<NamedNode> for NamedOrBlankNode {
    fn from(node: NamedNode) -> Self {
        NamedOrBlankNode::NamedNode(node)
    }
}

impl From<BlankNode> for NamedOrBlankNode {
    fn from(node: BlankNode) -> Self {
        NamedOrBlankNode::BlankNode(node)
    }
}

impl From<NamedOrBlankNode> for crate::model::term::Subject {
    fn from(node: NamedOrBlankNode) -> Self {
        match node {
            NamedOrBlankNode::NamedNode(n) => crate::model::term::Subject::NamedNode(n),
            NamedOrBlankNode::BlankNode(n) => crate::model::term::Subject::BlankNode(n),
        }
    }
}

impl NamedOrBlankNode {
    pub fn as_ref(&self) -> NamedOrBlankNodeRef<'_> {
        match self {
            NamedOrBlankNode::NamedNode(n) => NamedOrBlankNodeRef::NamedNode(n),
            NamedOrBlankNode::BlankNode(n) => NamedOrBlankNodeRef::BlankNode(n),
        }
    }
}

/// Reference variant of NamedOrBlankNode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamedOrBlankNodeRef<'a> {
    NamedNode(&'a NamedNode),
    BlankNode(&'a BlankNode),
}

impl<'a> From<&'a NamedOrBlankNode> for NamedOrBlankNodeRef<'a> {
    fn from(node: &'a NamedOrBlankNode) -> Self {
        node.as_ref()
    }
}

impl From<NamedOrBlankNode> for crate::model::term::Object {
    fn from(node: NamedOrBlankNode) -> Self {
        match node {
            NamedOrBlankNode::NamedNode(n) => crate::model::term::Object::NamedNode(n),
            NamedOrBlankNode::BlankNode(n) => crate::model::term::Object::BlankNode(n),
        }
    }
}