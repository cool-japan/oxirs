//! RDF-star data model providing type-safe handling of quoted triples.
//!
//! This module implements the core RDF-star data model extending standard RDF
//! with support for quoted triples as first-class citizens.

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::Arc;

use oxirs_core::model::{NamedNode as CoreNamedNode, Literal as CoreLiteral, BlankNode as CoreBlankNode};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{StarError, StarResult};

/// RDF-star term that can be an IRI, blank node, literal, or quoted triple
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StarTerm {
    /// Named node (IRI reference)
    NamedNode(NamedNode),
    /// Blank node with identifier
    BlankNode(BlankNode),
    /// Literal value with optional language tag and datatype
    Literal(Literal),
    /// Quoted triple (RDF-star extension)
    QuotedTriple(Box<StarTriple>),
    /// Variable (used in SPARQL-star queries)
    Variable(Variable),
}

/// Named node (IRI) implementation for RDF-star
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NamedNode {
    pub iri: String,
}

/// Blank node implementation for RDF-star
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlankNode {
    pub id: String,
}

/// Literal implementation for RDF-star
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    pub value: String,
    pub language: Option<String>,
    pub datatype: Option<NamedNode>,
}

/// Variable implementation for SPARQL-star
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
}

/// RDF-star triple that may contain quoted triples in any position
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StarTriple {
    pub subject: StarTerm,
    pub predicate: StarTerm,
    pub object: StarTerm,
}

/// RDF-star quad extending triples with optional graph context
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StarQuad {
    pub subject: StarTerm,
    pub predicate: StarTerm,
    pub object: StarTerm,
    pub graph: Option<StarTerm>,
}

impl StarTerm {
    /// Create a new IRI term
    pub fn iri(iri: &str) -> StarResult<Self> {
        if iri.is_empty() {
            return Err(StarError::InvalidTermType("IRI cannot be empty".to_string()));
        }
        Ok(StarTerm::NamedNode(NamedNode {
            iri: iri.to_string(),
        }))
    }

    /// Create a new blank node term
    pub fn blank_node(id: &str) -> StarResult<Self> {
        if id.is_empty() {
            return Err(StarError::InvalidTermType("Blank node ID cannot be empty".to_string()));
        }
        Ok(StarTerm::BlankNode(BlankNode {
            id: id.to_string(),
        }))
    }

    /// Create a new literal term
    pub fn literal(value: &str) -> StarResult<Self> {
        Ok(StarTerm::Literal(Literal {
            value: value.to_string(),
            language: None,
            datatype: None,
        }))
    }

    /// Create a new literal term with language tag
    pub fn literal_with_language(value: &str, language: &str) -> StarResult<Self> {
        Ok(StarTerm::Literal(Literal {
            value: value.to_string(),
            language: Some(language.to_string()),
            datatype: None,
        }))
    }

    /// Create a new literal term with datatype
    pub fn literal_with_datatype(value: &str, datatype: &str) -> StarResult<Self> {
        Ok(StarTerm::Literal(Literal {
            value: value.to_string(),
            language: None,
            datatype: Some(NamedNode {
                iri: datatype.to_string(),
            }),
        }))
    }

    /// Create a new quoted triple term
    pub fn quoted_triple(triple: StarTriple) -> Self {
        StarTerm::QuotedTriple(Box::new(triple))
    }

    /// Create a new variable term
    pub fn variable(name: &str) -> StarResult<Self> {
        if name.is_empty() {
            return Err(StarError::InvalidTermType("Variable name cannot be empty".to_string()));
        }
        Ok(StarTerm::Variable(Variable {
            name: name.to_string(),
        }))
    }

    /// Check if this term is a named node
    pub fn is_named_node(&self) -> bool {
        matches!(self, StarTerm::NamedNode(_))
    }

    /// Check if this term is a blank node
    pub fn is_blank_node(&self) -> bool {
        matches!(self, StarTerm::BlankNode(_))
    }

    /// Check if this term is a literal
    pub fn is_literal(&self) -> bool {
        matches!(self, StarTerm::Literal(_))
    }

    /// Check if this term is a quoted triple
    pub fn is_quoted_triple(&self) -> bool {
        matches!(self, StarTerm::QuotedTriple(_))
    }

    /// Check if this term is a variable
    pub fn is_variable(&self) -> bool {
        matches!(self, StarTerm::Variable(_))
    }

    /// Get the IRI if this is a named node
    pub fn as_named_node(&self) -> Option<&NamedNode> {
        match self {
            StarTerm::NamedNode(node) => Some(node),
            _ => None,
        }
    }

    /// Get the blank node if this is a blank node
    pub fn as_blank_node(&self) -> Option<&BlankNode> {
        match self {
            StarTerm::BlankNode(node) => Some(node),
            _ => None,
        }
    }

    /// Get the literal if this is a literal
    pub fn as_literal(&self) -> Option<&Literal> {
        match self {
            StarTerm::Literal(literal) => Some(literal),
            _ => None,
        }
    }

    /// Get the quoted triple if this is a quoted triple
    pub fn as_quoted_triple(&self) -> Option<&StarTriple> {
        match self {
            StarTerm::QuotedTriple(triple) => Some(triple),
            _ => None,
        }
    }

    /// Get the variable if this is a variable
    pub fn as_variable(&self) -> Option<&Variable> {
        match self {
            StarTerm::Variable(var) => Some(var),
            _ => None,
        }
    }

    /// Check if this term can be used as a subject
    pub fn can_be_subject(&self) -> bool {
        matches!(self, StarTerm::NamedNode(_) | StarTerm::BlankNode(_) | StarTerm::QuotedTriple(_))
    }

    /// Check if this term can be used as a predicate
    pub fn can_be_predicate(&self) -> bool {
        matches!(self, StarTerm::NamedNode(_))
    }

    /// Check if this term can be used as an object
    pub fn can_be_object(&self) -> bool {
        true // All terms can be objects in RDF-star
    }

    /// Calculate the nesting depth of quoted triples
    pub fn nesting_depth(&self) -> usize {
        match self {
            StarTerm::QuotedTriple(triple) => {
                1 + triple.subject.nesting_depth()
                    .max(triple.predicate.nesting_depth())
                    .max(triple.object.nesting_depth())
            }
            _ => 0,
        }
    }
}

impl StarTriple {
    /// Create a new RDF-star triple
    pub fn new(subject: StarTerm, predicate: StarTerm, object: StarTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Validate that the triple is well-formed according to RDF-star rules
    pub fn validate(&self) -> StarResult<()> {
        if !self.subject.can_be_subject() {
            return Err(StarError::InvalidQuotedTriple(
                format!("Invalid subject term: {:?}", self.subject)
            ));
        }

        if !self.predicate.can_be_predicate() {
            return Err(StarError::InvalidQuotedTriple(
                format!("Invalid predicate term: {:?}", self.predicate)
            ));
        }

        if !self.object.can_be_object() {
            return Err(StarError::InvalidQuotedTriple(
                format!("Invalid object term: {:?}", self.object)
            ));
        }

        Ok(())
    }

    /// Get the maximum nesting depth in this triple
    pub fn nesting_depth(&self) -> usize {
        self.subject
            .nesting_depth()
            .max(self.predicate.nesting_depth())
            .max(self.object.nesting_depth())
    }

    /// Check if this triple contains any quoted triples
    pub fn contains_quoted_triples(&self) -> bool {
        self.subject.is_quoted_triple()
            || self.predicate.is_quoted_triple()
            || self.object.is_quoted_triple()
    }

    /// Convert to a quad with optional graph
    pub fn to_quad(self, graph: Option<StarTerm>) -> StarQuad {
        StarQuad {
            subject: self.subject,
            predicate: self.predicate,
            object: self.object,
            graph,
        }
    }
}

impl StarQuad {
    /// Create a new RDF-star quad
    pub fn new(subject: StarTerm, predicate: StarTerm, object: StarTerm, graph: Option<StarTerm>) -> Self {
        Self {
            subject,
            predicate,
            object,
            graph,
        }
    }

    /// Convert to a triple (losing graph information)
    pub fn to_triple(self) -> StarTriple {
        StarTriple {
            subject: self.subject,
            predicate: self.predicate,
            object: self.object,
        }
    }

    /// Validate that the quad is well-formed
    pub fn validate(&self) -> StarResult<()> {
        // Validate the triple part
        let triple = StarTriple {
            subject: self.subject.clone(),
            predicate: self.predicate.clone(),
            object: self.object.clone(),
        };
        triple.validate()?;

        // Validate graph name if present
        if let Some(ref graph) = self.graph {
            if !matches!(graph, StarTerm::NamedNode(_) | StarTerm::BlankNode(_)) {
                return Err(StarError::InvalidQuotedTriple(
                    "Graph name must be a named node or blank node".to_string()
                ));
            }
        }

        Ok(())
    }
}

impl fmt::Display for StarTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StarTerm::NamedNode(node) => write!(f, "<{}>", node.iri),
            StarTerm::BlankNode(node) => write!(f, "_:{}", node.id),
            StarTerm::Literal(literal) => {
                write!(f, "\"{}\"", literal.value)?;
                if let Some(ref lang) = literal.language {
                    write!(f, "@{}", lang)?;
                }
                if let Some(ref datatype) = literal.datatype {
                    write!(f, "^^<{}>", datatype.iri)?;
                }
                Ok(())
            }
            StarTerm::QuotedTriple(triple) => write!(f, "<<{} {} {}>>", triple.subject, triple.predicate, triple.object),
            StarTerm::Variable(var) => write!(f, "?{}", var.name),
        }
    }
}

impl fmt::Display for StarTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
    }
}

impl fmt::Display for StarQuad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.subject, self.predicate, self.object)?;
        if let Some(ref graph) = self.graph {
            write!(f, " {}", graph)?;
        }
        write!(f, " .")
    }
}

/// Trait for collecting RDF-star terms from data structures
pub trait StarTermVisitor {
    /// Visit a term during traversal
    fn visit_term(&mut self, term: &StarTerm);
}

impl StarTriple {
    /// Apply a visitor to all terms in this triple
    pub fn visit_terms<V: StarTermVisitor>(&self, visitor: &mut V) {
        visitor.visit_term(&self.subject);
        visitor.visit_term(&self.predicate);
        visitor.visit_term(&self.object);
        
        // Recursively visit quoted triples
        if let StarTerm::QuotedTriple(triple) = &self.subject {
            triple.visit_terms(visitor);
        }
        if let StarTerm::QuotedTriple(triple) = &self.predicate {
            triple.visit_terms(visitor);
        }
        if let StarTerm::QuotedTriple(triple) = &self.object {
            triple.visit_terms(visitor);
        }
    }
}

/// Graph container for RDF-star triples
#[derive(Debug, Clone, Default)]
pub struct StarGraph {
    triples: Vec<StarTriple>,
    statistics: HashMap<String, usize>,
}

impl StarGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
            statistics: HashMap::new(),
        }
    }

    /// Add a triple to the graph
    pub fn insert(&mut self, triple: StarTriple) -> StarResult<()> {
        triple.validate()?;
        self.triples.push(triple);
        *self.statistics.entry("triples".to_string()).or_insert(0) += 1;
        Ok(())
    }

    /// Get all triples in the graph
    pub fn triples(&self) -> &[StarTriple] {
        &self.triples
    }

    /// Check if the graph contains a specific triple
    pub fn contains(&self, triple: &StarTriple) -> bool {
        self.triples.contains(triple)
    }

    /// Remove a triple from the graph
    pub fn remove(&mut self, triple: &StarTriple) -> bool {
        if let Some(pos) = self.triples.iter().position(|t| t == triple) {
            self.triples.remove(pos);
            if let Some(count) = self.statistics.get_mut("triples") {
                *count = count.saturating_sub(1);
            }
            true
        } else {
            false
        }
    }

    /// Get the number of triples in the graph
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Clear all triples from the graph
    pub fn clear(&mut self) {
        self.triples.clear();
        self.statistics.clear();
    }

    /// Get statistics about the graph
    pub fn statistics(&self) -> &HashMap<String, usize> {
        &self.statistics
    }

    /// Count quoted triples in the graph
    pub fn count_quoted_triples(&self) -> usize {
        let mut count = 0;
        for triple in &self.triples {
            if triple.contains_quoted_triples() {
                count += 1;
            }
        }
        count
    }

    /// Get maximum nesting depth in the graph
    pub fn max_nesting_depth(&self) -> usize {
        self.triples
            .iter()
            .map(|t| t.nesting_depth())
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_term_creation() {
        let iri = StarTerm::iri("http://example.org/test").unwrap();
        assert!(iri.is_named_node());
        assert!(!iri.is_literal());

        let literal = StarTerm::literal("test value").unwrap();
        assert!(literal.is_literal());
        assert!(!literal.is_named_node());

        let blank = StarTerm::blank_node("b1").unwrap();
        assert!(blank.is_blank_node());
    }

    #[test]
    fn test_quoted_triple_creation() {
        let subject = StarTerm::iri("http://example.org/alice").unwrap();
        let predicate = StarTerm::iri("http://example.org/age").unwrap();
        let object = StarTerm::literal("25").unwrap();

        let triple = StarTriple::new(subject, predicate, object);
        assert!(triple.validate().is_ok());

        let quoted = StarTerm::quoted_triple(triple);
        assert!(quoted.is_quoted_triple());
        assert_eq!(quoted.nesting_depth(), 1);
    }

    #[test]
    fn test_nested_quoted_triples() {
        // Create inner triple
        let inner = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/knows").unwrap(),
            StarTerm::iri("http://example.org/bob").unwrap(),
        );

        // Create outer triple with quoted triple as subject
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        );

        assert_eq!(outer.nesting_depth(), 1);
        assert!(outer.contains_quoted_triples());
    }

    #[test]
    fn test_star_graph_operations() {
        let mut graph = StarGraph::new();
        assert!(graph.is_empty());

        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );

        graph.insert(triple.clone()).unwrap();
        assert_eq!(graph.len(), 1);
        assert!(graph.contains(&triple));

        graph.remove(&triple);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_validation() {
        // Valid triple
        let valid = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::literal("object").unwrap(),
        );
        assert!(valid.validate().is_ok());

        // Invalid predicate (literal cannot be predicate)
        let invalid = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::literal("invalid_predicate").unwrap(),
            StarTerm::literal("object").unwrap(),
        );
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_display_formatting() {
        let triple = StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("25").unwrap(),
        );

        let display_str = format!("{}", triple);
        assert!(display_str.contains("<http://example.org/alice>"));
        assert!(display_str.contains("\"25\""));

        let quoted = StarTerm::quoted_triple(triple.clone());
        let quoted_str = format!("{}", quoted);
        assert!(quoted_str.starts_with("<<"));
        assert!(quoted_str.ends_with(">>"));
    }
}