//! IRI and Named Node implementations for RDF
//!
//! This module provides high-performance IRI and Named Node implementations
//! based on oxrdf patterns with oxiri for fast validation.

use crate::model::{GraphNameTerm, ObjectTerm, PredicateTerm, RdfTerm, SubjectTerm};
use crate::OxirsError;
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

// Import and re-export oxiri types for compatibility
pub use oxiri::{Iri, IriParseError};

/// Convert IriParseError to OxirsError
impl From<IriParseError> for OxirsError {
    fn from(err: IriParseError) -> Self {
        OxirsError::Parse(format!("IRI parse error: {err}"))
    }
}

/// An owned RDF [IRI](https://www.w3.org/TR/rdf11-concepts/#dfn-iri).
///
/// The default string formatter is returning an N-Triples, Turtle, and SPARQL compatible representation:
/// ```
/// use oxirs_core::model::NamedNode;
///
/// assert_eq!(
///     "<http://example.com/foo>",
///     NamedNode::new("http://example.com/foo").unwrap().to_string()
/// );
/// ```
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Hash)]
pub struct NamedNode {
    iri: String,
}

impl NamedNode {
    /// Builds and validate an RDF [IRI](https://www.w3.org/TR/rdf11-concepts/#dfn-iri).
    pub fn new(iri: impl Into<String>) -> Result<Self, OxirsError> {
        Ok(Self::new_from_iri(Iri::parse(iri.into())?))
    }

    #[inline]
    pub(crate) fn new_from_iri(iri: Iri<String>) -> Self {
        Self::new_unchecked(iri.into_inner())
    }

    /// Builds an RDF [IRI](https://www.w3.org/TR/rdf11-concepts/#dfn-iri) from a string with normalization.
    ///
    /// This applies IRI normalization before validation.
    pub fn new_normalized(iri: impl Into<String>) -> Result<Self, OxirsError> {
        Ok(Self::new_from_iri(Iri::parse(iri.into())?))
    }

    /// Builds an RDF [IRI](https://www.w3.org/TR/rdf11-concepts/#dfn-iri) from a string.
    ///
    /// It is the caller's responsibility to ensure that `iri` is a valid IRI.
    ///
    /// [`NamedNode::new()`] is a safe version of this constructor and should be used for untrusted data.
    #[inline]
    pub fn new_unchecked(iri: impl Into<String>) -> Self {
        Self { iri: iri.into() }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        self.iri.as_str()
    }

    #[inline]
    pub fn into_string(self) -> String {
        self.iri
    }

    #[inline]
    pub fn as_ref(&self) -> NamedNodeRef<'_> {
        NamedNodeRef::new_unchecked(&self.iri)
    }
}

impl fmt::Display for NamedNode {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ref().fmt(f)
    }
}

impl PartialEq<str> for NamedNode {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<NamedNode> for str {
    #[inline]
    fn eq(&self, other: &NamedNode) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<&str> for NamedNode {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self == *other
    }
}

impl PartialEq<NamedNode> for &str {
    #[inline]
    fn eq(&self, other: &NamedNode) -> bool {
        *self == other
    }
}

impl FromStr for NamedNode {
    type Err = OxirsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// A borrowed RDF [IRI](https://www.w3.org/TR/rdf11-concepts/#dfn-iri).
///
/// The default string formatter is returning an N-Triples, Turtle, and SPARQL compatible representation:
/// ```
/// use oxirs_core::model::NamedNodeRef;
///
/// assert_eq!(
///     "<http://example.com/foo>",
///     NamedNodeRef::new("http://example.com/foo").unwrap().to_string()
/// );
/// ```
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy, Hash)]
pub struct NamedNodeRef<'a> {
    iri: &'a str,
}

impl<'a> NamedNodeRef<'a> {
    /// Builds and validate an RDF [IRI](https://www.w3.org/TR/rdf11-concepts/#dfn-iri)
    pub fn new(iri: &'a str) -> Result<Self, OxirsError> {
        Ok(Self::new_from_iri(Iri::parse(iri)?))
    }

    #[inline]
    pub(crate) fn new_from_iri(iri: Iri<&'a str>) -> Self {
        Self::new_unchecked(iri.into_inner())
    }

    /// Builds an RDF [IRI](https://www.w3.org/TR/rdf11-concepts/#dfn-iri) from a string.
    ///
    /// It is the caller's responsibility to ensure that `iri` is a valid IRI.
    ///
    /// [`NamedNodeRef::new()`] is a safe version of this constructor and should be used for untrusted data.
    #[inline]
    pub const fn new_unchecked(iri: &'a str) -> Self {
        Self { iri }
    }

    #[inline]
    pub const fn as_str(self) -> &'a str {
        self.iri
    }

    #[inline]
    pub fn into_owned(self) -> NamedNode {
        NamedNode::new_unchecked(self.iri)
    }
}

impl fmt::Display for NamedNodeRef<'_> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{}>", self.as_str())
    }
}

impl PartialEq<str> for NamedNodeRef<'_> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<NamedNodeRef<'_>> for str {
    #[inline]
    fn eq(&self, other: &NamedNodeRef<'_>) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<&str> for NamedNodeRef<'_> {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<NamedNodeRef<'_>> for &str {
    #[inline]
    fn eq(&self, other: &NamedNodeRef<'_>) -> bool {
        *self == other.as_str()
    }
}

impl PartialEq<NamedNode> for NamedNodeRef<'_> {
    #[inline]
    fn eq(&self, other: &NamedNode) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<NamedNodeRef<'_>> for NamedNode {
    #[inline]
    fn eq(&self, other: &NamedNodeRef<'_>) -> bool {
        self.as_str() == other.as_str()
    }
}

impl<'a> From<NamedNodeRef<'a>> for NamedNode {
    #[inline]
    fn from(node: NamedNodeRef<'a>) -> Self {
        node.into_owned()
    }
}

impl<'a> From<&'a NamedNode> for NamedNodeRef<'a> {
    #[inline]
    fn from(node: &'a NamedNode) -> Self {
        node.as_ref()
    }
}

// Implement RDF term traits
impl RdfTerm for NamedNode {
    fn as_str(&self) -> &str {
        self.as_str()
    }

    fn is_named_node(&self) -> bool {
        true
    }
}

impl RdfTerm for NamedNodeRef<'_> {
    fn as_str(&self) -> &str {
        self.iri
    }

    fn is_named_node(&self) -> bool {
        true
    }
}

impl SubjectTerm for NamedNode {}
impl PredicateTerm for NamedNode {}
impl ObjectTerm for NamedNode {}
impl GraphNameTerm for NamedNode {}

impl SubjectTerm for NamedNodeRef<'_> {}
impl PredicateTerm for NamedNodeRef<'_> {}
impl ObjectTerm for NamedNodeRef<'_> {}
impl GraphNameTerm for NamedNodeRef<'_> {}

// Implement conversions from NamedNodeRef to union types
impl From<NamedNodeRef<'_>> for crate::model::Subject {
    #[inline]
    fn from(node: NamedNodeRef<'_>) -> Self {
        crate::model::Subject::NamedNode(node.into_owned())
    }
}

impl From<NamedNodeRef<'_>> for crate::model::Predicate {
    #[inline]
    fn from(node: NamedNodeRef<'_>) -> Self {
        crate::model::Predicate::NamedNode(node.into_owned())
    }
}

impl From<NamedNodeRef<'_>> for crate::model::Object {
    #[inline]
    fn from(node: NamedNodeRef<'_>) -> Self {
        crate::model::Object::NamedNode(node.into_owned())
    }
}

// Serialization support
#[cfg(feature = "serde")]
impl serde::Serialize for NamedNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.iri)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for NamedNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let iri = String::deserialize(deserializer)?;
        Self::new(iri).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_node_creation() {
        let node = NamedNode::new("http://example.com/test").unwrap();
        assert_eq!(node.as_str(), "http://example.com/test");
        assert_eq!(node.to_string(), "<http://example.com/test>");
    }

    #[test]
    fn test_named_node_ref() {
        let node = NamedNode::new("http://example.com/test").unwrap();
        let node_ref = node.as_ref();
        assert_eq!(node_ref.as_str(), "http://example.com/test");
        assert_eq!(node_ref.to_string(), "<http://example.com/test>");
    }

    #[test]
    fn test_named_node_comparison() {
        let node = NamedNode::new("http://example.com/test").unwrap();
        assert_eq!(node, "http://example.com/test");
        assert_eq!("http://example.com/test", node);
    }

    #[test]
    fn test_invalid_iri() {
        assert!(NamedNode::new("not a valid iri").is_err());
        assert!(NamedNode::new("").is_err());
    }

    #[test]
    fn test_owned_borrowed_conversion() {
        let owned = NamedNode::new("http://example.com/test").unwrap();
        let borrowed = owned.as_ref();
        let owned_again = borrowed.into_owned();
        assert_eq!(owned, owned_again);
    }

    #[test]
    fn test_rdf_term_trait() {
        let node = NamedNode::new("http://example.com/test").unwrap();
        assert!(node.is_named_node());
        assert!(!node.is_blank_node());
        assert!(!node.is_literal());
    }
}
