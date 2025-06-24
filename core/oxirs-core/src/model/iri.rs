//! IRI and Named Node implementations for RDF

use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use crate::model::{RdfTerm, SubjectTerm, PredicateTerm, ObjectTerm, GraphNameTerm};
use crate::OxirsError;

/// An RDF Named Node (IRI)
/// 
/// Represents an IRI (Internationalized Resource Identifier) reference.
/// This is one of the core RDF term types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct NamedNode {
    iri: String,
}

impl NamedNode {
    /// Creates a new named node from an IRI string
    /// 
    /// # Arguments
    /// * `iri` - The IRI string
    /// 
    /// # Errors
    /// Returns an error if the IRI is invalid
    pub fn new(iri: impl Into<String>) -> Result<Self, OxirsError> {
        let iri = iri.into();
        
        // Basic IRI validation
        if iri.is_empty() {
            return Err(OxirsError::Parse("IRI cannot be empty".to_string()));
        }
        
        // TODO: Add more comprehensive IRI validation
        // For now, accept any non-empty string
        
        Ok(NamedNode { iri })
    }
    
    /// Creates a new named node from an IRI string without validation
    /// 
    /// # Safety
    /// The caller must ensure the IRI is valid
    pub fn new_unchecked(iri: impl Into<String>) -> Self {
        NamedNode { iri: iri.into() }
    }
    
    /// Returns the IRI as a string slice
    pub fn as_str(&self) -> &str {
        &self.iri
    }
    
    /// Returns the IRI as a string
    pub fn into_string(self) -> String {
        self.iri
    }
}

impl FromStr for NamedNode {
    type Err = OxirsError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl fmt::Display for NamedNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.iri)
    }
}

impl RdfTerm for NamedNode {
    fn as_str(&self) -> &str {
        &self.iri
    }
    
    fn is_named_node(&self) -> bool {
        true
    }
}

impl SubjectTerm for NamedNode {}
impl PredicateTerm for NamedNode {}
impl ObjectTerm for NamedNode {}
impl GraphNameTerm for NamedNode {}

/// A borrowed named node
/// 
/// This is an optimized version for temporary references
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NamedNodeRef<'a> {
    iri: &'a str,
}

impl<'a> NamedNodeRef<'a> {
    /// Creates a new named node reference
    pub fn new(iri: &'a str) -> Result<Self, OxirsError> {
        if iri.is_empty() {
            return Err(OxirsError::Parse("IRI cannot be empty".to_string()));
        }
        
        Ok(NamedNodeRef { iri })
    }
    
    /// Creates a new named node reference without validation
    pub fn new_unchecked(iri: &'a str) -> Self {
        NamedNodeRef { iri }
    }
    
    /// Returns the IRI as a string slice
    pub fn as_str(&self) -> &str {
        self.iri
    }
    
    /// Converts to an owned NamedNode
    pub fn to_owned(&self) -> NamedNode {
        NamedNode::new_unchecked(self.iri.to_string())
    }
}

impl<'a> fmt::Display for NamedNodeRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.iri)
    }
}

impl<'a> RdfTerm for NamedNodeRef<'a> {
    fn as_str(&self) -> &str {
        self.iri
    }
    
    fn is_named_node(&self) -> bool {
        true
    }
}

impl<'a> From<NamedNodeRef<'a>> for NamedNode {
    fn from(node_ref: NamedNodeRef<'a>) -> Self {
        node_ref.to_owned()
    }
}

impl<'a> From<&'a NamedNode> for NamedNodeRef<'a> {
    fn from(node: &'a NamedNode) -> Self {
        NamedNodeRef::new_unchecked(node.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_named_node_creation() {
        let node = NamedNode::new("http://example.org/test").unwrap();
        assert_eq!(node.as_str(), "http://example.org/test");
        assert!(node.is_named_node());
    }
    
    #[test]
    fn test_named_node_display() {
        let node = NamedNode::new("http://example.org/test").unwrap();
        assert_eq!(format!("{}", node), "<http://example.org/test>");
    }
    
    #[test]
    fn test_empty_iri_error() {
        assert!(NamedNode::new("").is_err());
    }
    
    #[test]
    fn test_named_node_ref() {
        let node_ref = NamedNodeRef::new("http://example.org/test").unwrap();
        assert_eq!(node_ref.as_str(), "http://example.org/test");
        
        let owned = node_ref.to_owned();
        assert_eq!(owned.as_str(), "http://example.org/test");
    }
}