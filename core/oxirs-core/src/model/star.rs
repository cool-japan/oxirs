//! RDF-star (RDF*) support for statement annotations
//!
//! This module implements RDF-star extensions for SPARQL 1.2 compliance,
//! allowing triples to be used as subjects or objects in other triples.

use crate::model::{Triple, Subject, Predicate, Object, NamedNode, RdfTerm, SubjectTerm, ObjectTerm};
use crate::OxirsError;
use std::fmt;
use std::sync::Arc;

/// A quoted triple that can be used as a subject or object in RDF-star
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuotedTriple {
    /// The inner triple being quoted
    inner: Arc<Triple>,
}

impl QuotedTriple {
    /// Create a new quoted triple
    pub fn new(triple: Triple) -> Self {
        QuotedTriple {
            inner: Arc::new(triple),
        }
    }
    
    /// Create from an existing Arc<Triple>
    pub fn from_arc(triple: Arc<Triple>) -> Self {
        QuotedTriple { inner: triple }
    }
    
    /// Get the inner triple
    pub fn inner(&self) -> &Triple {
        &self.inner
    }
    
    /// Get the subject of the quoted triple
    pub fn subject(&self) -> &Subject {
        self.inner.subject()
    }
    
    /// Get the predicate of the quoted triple
    pub fn predicate(&self) -> &Predicate {
        self.inner.predicate()
    }
    
    /// Get the object of the quoted triple
    pub fn object(&self) -> &Object {
        self.inner.object()
    }
    
    /// Convert to a triple reference
    pub fn as_ref(&self) -> QuotedTripleRef<'_> {
        QuotedTripleRef { inner: &self.inner }
    }
}

impl fmt::Display for QuotedTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<< {} >>", self.inner)
    }
}

impl RdfTerm for QuotedTriple {
    fn as_str(&self) -> &str {
        // For quoted triples, we return a synthetic string representation
        "<<quoted-triple>>"
    }
    
    fn is_quoted_triple(&self) -> bool {
        true
    }
}

impl SubjectTerm for QuotedTriple {}
impl ObjectTerm for QuotedTriple {}

#[cfg(feature = "serde")]
impl serde::Serialize for QuotedTriple {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.inner.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for QuotedTriple {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let triple = Triple::deserialize(deserializer)?;
        Ok(QuotedTriple::new(triple))
    }
}

/// A borrowed quoted triple reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuotedTripleRef<'a> {
    inner: &'a Triple,
}

impl<'a> QuotedTripleRef<'a> {
    /// Create a new quoted triple reference
    pub fn new(triple: &'a Triple) -> Self {
        QuotedTripleRef { inner: triple }
    }
    
    /// Get the inner triple
    pub fn inner(&self) -> &'a Triple {
        self.inner
    }
    
    /// Convert to owned quoted triple
    pub fn to_owned(&self) -> QuotedTriple {
        QuotedTriple::new(self.inner.clone())
    }
}

impl<'a> fmt::Display for QuotedTripleRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<< {} >>", self.inner)
    }
}

impl<'a> RdfTerm for QuotedTripleRef<'a> {
    fn as_str(&self) -> &str {
        "<<quoted-triple>>"
    }
    
    fn is_quoted_triple(&self) -> bool {
        true
    }
}

/// RDF-star annotation syntax support
pub struct Annotation {
    /// The statement being annotated (as a quoted triple)
    pub statement: QuotedTriple,
    /// The annotation property
    pub property: NamedNode,
    /// The annotation value
    pub value: Object,
}

impl Annotation {
    /// Create a new annotation
    pub fn new(statement: Triple, property: NamedNode, value: Object) -> Self {
        Annotation {
            statement: QuotedTriple::new(statement),
            property,
            value,
        }
    }
    
    /// Convert annotation to a regular triple with quoted triple as subject
    pub fn to_triple(&self) -> Triple {
        Triple::new(
            Subject::QuotedTriple(Box::new(self.statement.clone())),
            self.property.clone(),
            self.value.clone(),
        )
    }
}

/// RDF-star pattern for SPARQL queries
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StarPattern {
    /// A regular triple pattern
    Triple(crate::query::algebra::TriplePattern),
    /// A quoted triple pattern
    QuotedTriple {
        /// Subject pattern (can be nested quoted triple)
        subject: Box<StarPattern>,
        /// Predicate pattern
        predicate: crate::query::algebra::TermPattern,
        /// Object pattern (can be nested quoted triple)
        object: Box<StarPattern>,
    },
    /// An annotation pattern
    Annotation {
        /// The annotated statement pattern
        statement: Box<StarPattern>,
        /// Annotation property pattern
        property: crate::query::algebra::TermPattern,
        /// Annotation value pattern
        value: crate::query::algebra::TermPattern,
    },
}

impl StarPattern {
    /// Check if pattern contains variables
    pub fn has_variables(&self) -> bool {
        match self {
            StarPattern::Triple(_) => false, // TODO: implement properly when TriplePattern is available
            StarPattern::QuotedTriple { subject, predicate: _, object } => {
                subject.has_variables() || object.has_variables()
            }
            StarPattern::Annotation { statement, property: _, value: _ } => {
                statement.has_variables()
            }
        }
    }
    
    /// Get all variables in the pattern
    pub fn variables(&self) -> Vec<crate::model::Variable> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars
    }
    
    fn collect_variables(&self, vars: &mut Vec<crate::model::Variable>) {
        match self {
            StarPattern::Triple(_) => {
                // TODO: implement properly when query algebra is available
            }
            StarPattern::QuotedTriple { subject, predicate: _, object } => {
                subject.collect_variables(vars);
                object.collect_variables(vars);
            }
            StarPattern::Annotation { statement, property: _, value: _ } => {
                statement.collect_variables(vars);
            }
        }
    }
}

/// RDF-star serialization format extensions
pub mod serialization {
    use super::*;
    
    /// Turtle-star syntax extensions
    pub mod turtle_star {
        use super::*;
        
        /// Serialize a quoted triple in Turtle-star syntax
        pub fn serialize_quoted_triple(qt: &QuotedTriple) -> String {
            format!("<< {} {} {} >>", 
                qt.subject(), 
                qt.predicate(), 
                qt.object()
            )
        }
        
        /// Parse a quoted triple from Turtle-star syntax
        pub fn parse_quoted_triple(input: &str) -> Result<QuotedTriple, OxirsError> {
            // Simplified parser - in production would use proper tokenization
            let trimmed = input.trim();
            if !trimmed.starts_with("<<") || !trimmed.ends_with(">>") {
                return Err(OxirsError::Parse("Invalid quoted triple syntax".to_string()));
            }
            
            // Extract inner content
            let inner = &trimmed[2..trimmed.len()-2].trim();
            
            // Parse inner triple (simplified)
            // In production, would integrate with full Turtle parser
            Err(OxirsError::Parse("Quoted triple parsing not yet implemented".to_string()))
        }
    }
    
    /// SPARQL-star syntax extensions
    pub mod sparql_star {
        use super::*;
        
        /// Format a star pattern for SPARQL
        pub fn format_star_pattern(pattern: &StarPattern) -> String {
            match pattern {
                StarPattern::Triple(_) => {
                    // TODO: implement properly when query algebra is available
                    "TRIPLE_PATTERN".to_string()
                }
                StarPattern::QuotedTriple { subject, predicate: _, object } => {
                    format!("<< {} {} {} >>", 
                        format_star_pattern(subject),
                        "PREDICATE",
                        format_star_pattern(object)
                    )
                }
                StarPattern::Annotation { statement, property: _, value: _ } => {
                    format!("{} {} {}", 
                        format_star_pattern(statement),
                        "PROPERTY",
                        "VALUE"
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NamedNode, Literal};
    
    #[test]
    fn test_quoted_triple() {
        let subject = NamedNode::new("http://example.org/alice").unwrap();
        let predicate = NamedNode::new("http://example.org/says").unwrap();
        let object = Object::Literal(Literal::new("Hello"));
        
        let triple = Triple::new(subject, predicate, object);
        let quoted = QuotedTriple::new(triple.clone());
        
        assert_eq!(quoted.inner(), &triple);
        assert_eq!(format!("{}", quoted), "<< <http://example.org/alice> <http://example.org/says> \"Hello\" . >>");
    }
    
    #[test]
    fn test_annotation() {
        let subject = NamedNode::new("http://example.org/alice").unwrap();
        let predicate = NamedNode::new("http://example.org/age").unwrap();
        let object = Object::Literal(Literal::new_typed("30", NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap()));
        
        let statement = Triple::new(subject, predicate, object);
        let ann_property = NamedNode::new("http://example.org/confidence").unwrap();
        let ann_value = Object::Literal(Literal::new_typed("0.9", NamedNode::new("http://www.w3.org/2001/XMLSchema#double").unwrap()));
        
        let annotation = Annotation::new(statement, ann_property, ann_value);
        let ann_triple = annotation.to_triple();
        
        assert!(matches!(ann_triple.subject(), Subject::QuotedTriple(_)));
    }
}