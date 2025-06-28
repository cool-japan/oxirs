//! RDF Triple implementation

use crate::model::RdfTerm;
use crate::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Variable};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};

/// An RDF Triple
///
/// Represents an RDF statement with subject, predicate, and object.
/// This is the fundamental unit of RDF data.
///
/// Implements ordering for use in BTree indexes for efficient storage and retrieval.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Triple {
    subject: Subject,
    predicate: Predicate,
    object: Object,
}

impl Triple {
    /// Creates a new triple
    pub fn new(
        subject: impl Into<Subject>,
        predicate: impl Into<Predicate>,
        object: impl Into<Object>,
    ) -> Self {
        Triple {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
    
    /// Returns a reference to this triple
    pub fn as_ref(&self) -> TripleRef<'_> {
        TripleRef::from(self)
    }

    /// Returns the subject of this triple
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// Returns the predicate of this triple
    pub fn predicate(&self) -> &Predicate {
        &self.predicate
    }

    /// Returns the object of this triple
    pub fn object(&self) -> &Object {
        &self.object
    }

    /// Decomposes the triple into its components
    pub fn into_parts(self) -> (Subject, Predicate, Object) {
        (self.subject, self.predicate, self.object)
    }

    /// Returns true if this triple contains any variables
    pub fn has_variables(&self) -> bool {
        matches!(self.subject, Subject::Variable(_))
            || matches!(self.predicate, Predicate::Variable(_))
            || matches!(self.object, Object::Variable(_))
    }

    /// Returns true if this triple is ground (contains no variables)
    pub fn is_ground(&self) -> bool {
        !self.has_variables()
    }

    /// Returns true if this triple matches the given pattern
    ///
    /// None values in the pattern act as wildcards matching any term.
    pub fn matches_pattern(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> bool {
        if let Some(s) = subject {
            if &self.subject != s {
                return false;
            }
        }

        if let Some(p) = predicate {
            if &self.predicate != p {
                return false;
            }
        }

        if let Some(o) = object {
            if &self.object != o {
                return false;
            }
        }

        true
    }

    /// Returns the canonical order of this triple for sorting
    ///
    /// This enables efficient storage in BTree-based indexes.
    /// Order: Subject -> Predicate -> Object
    fn canonical_ordering(&self) -> (u8, &str, u8, &str, u8, &str) {
        let subject_ord = match &self.subject {
            Subject::NamedNode(_) => 0,
            Subject::BlankNode(_) => 1,
            Subject::Variable(_) => 2,
            Subject::QuotedTriple(_) => 3,
        };

        let predicate_ord = match &self.predicate {
            Predicate::NamedNode(_) => 0,
            Predicate::Variable(_) => 1,
        };

        let object_ord = match &self.object {
            Object::NamedNode(_) => 0,
            Object::BlankNode(_) => 1,
            Object::Literal(_) => 2,
            Object::Variable(_) => 3,
            Object::QuotedTriple(_) => 4,
        };

        (
            subject_ord,
            self.subject_str(),
            predicate_ord,
            self.predicate_str(),
            object_ord,
            self.object_str(),
        )
    }

    /// Returns the subject as a string for ordering
    fn subject_str(&self) -> &str {
        match &self.subject {
            Subject::NamedNode(n) => n.as_str(),
            Subject::BlankNode(b) => b.as_str(),
            Subject::Variable(v) => v.as_str(),
            Subject::QuotedTriple(_) => "<<quoted-triple>>",
        }
    }

    /// Returns the predicate as a string for ordering
    fn predicate_str(&self) -> &str {
        match &self.predicate {
            Predicate::NamedNode(n) => n.as_str(),
            Predicate::Variable(v) => v.as_str(),
        }
    }

    /// Returns the object as a string for ordering
    fn object_str(&self) -> &str {
        match &self.object {
            Object::NamedNode(n) => n.as_str(),
            Object::BlankNode(b) => b.as_str(),
            Object::Literal(l) => l.as_str(),
            Object::Variable(v) => v.as_str(),
            Object::QuotedTriple(_) => "<<quoted-triple>>",
        }
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
    }
}

// Display implementations for term unions
impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Subject::NamedNode(n) => write!(f, "{}", n),
            Subject::BlankNode(b) => write!(f, "{}", b),
            Subject::Variable(v) => write!(f, "{}", v),
            Subject::QuotedTriple(qt) => write!(f, "{}", qt),
        }
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Predicate::NamedNode(n) => write!(f, "{}", n),
            Predicate::Variable(v) => write!(f, "{}", v),
        }
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Object::NamedNode(n) => write!(f, "{}", n),
            Object::BlankNode(b) => write!(f, "{}", b),
            Object::Literal(l) => write!(f, "{}", l),
            Object::Variable(v) => write!(f, "{}", v),
            Object::QuotedTriple(qt) => write!(f, "{}", qt),
        }
    }
}

/// A borrowed triple reference for zero-copy operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TripleRef<'a> {
    subject: SubjectRef<'a>,
    predicate: PredicateRef<'a>,
    object: ObjectRef<'a>,
}

impl<'a> TripleRef<'a> {
    /// Creates a new triple reference
    pub fn new(
        subject: SubjectRef<'a>,
        predicate: PredicateRef<'a>,
        object: ObjectRef<'a>,
    ) -> Self {
        TripleRef {
            subject,
            predicate,
            object,
        }
    }

    /// Returns the subject
    pub fn subject(&self) -> SubjectRef<'a> {
        self.subject
    }

    /// Returns the predicate
    pub fn predicate(&self) -> PredicateRef<'a> {
        self.predicate
    }

    /// Returns the object
    pub fn object(&self) -> ObjectRef<'a> {
        self.object
    }

    /// Converts to an owned triple
    pub fn to_owned(&self) -> Triple {
        Triple {
            subject: self.subject.to_owned(),
            predicate: self.predicate.to_owned(),
            object: self.object.to_owned(),
        }
    }
    
    /// Converts to an owned triple (alias for to_owned)
    pub fn into_owned(self) -> Triple {
        self.to_owned()
    }
    
    /// Creates a QuadRef from this triple with the specified graph
    pub fn in_graph(self, graph_name: Option<&'a crate::model::NamedNode>) -> crate::model::QuadRef<'a> {
        let graph_ref = match graph_name {
            Some(node) => crate::model::GraphNameRef::NamedNode(node),
            None => crate::model::GraphNameRef::DefaultGraph,
        };
        crate::model::QuadRef::new(self.subject, self.predicate, self.object, graph_ref)
    }
}

impl<'a> fmt::Display for TripleRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {} .", self.subject, self.predicate, self.object)
    }
}

/// Borrowed subject reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubjectRef<'a> {
    NamedNode(&'a NamedNode),
    BlankNode(&'a BlankNode),
    Variable(&'a Variable),
}

impl<'a> SubjectRef<'a> {
    /// Converts to an owned subject
    pub fn to_owned(&self) -> Subject {
        match self {
            SubjectRef::NamedNode(n) => Subject::NamedNode((*n).clone()),
            SubjectRef::BlankNode(b) => Subject::BlankNode((*b).clone()),
            SubjectRef::Variable(v) => Subject::Variable((*v).clone()),
        }
    }
}

impl<'a> fmt::Display for SubjectRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubjectRef::NamedNode(n) => write!(f, "{}", n),
            SubjectRef::BlankNode(b) => write!(f, "{}", b),
            SubjectRef::Variable(v) => write!(f, "{}", v),
        }
    }
}

/// Borrowed predicate reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredicateRef<'a> {
    NamedNode(&'a NamedNode),
    Variable(&'a Variable),
}

impl<'a> PredicateRef<'a> {
    /// Converts to an owned predicate
    pub fn to_owned(&self) -> Predicate {
        match self {
            PredicateRef::NamedNode(n) => Predicate::NamedNode((*n).clone()),
            PredicateRef::Variable(v) => Predicate::Variable((*v).clone()),
        }
    }
}

impl<'a> fmt::Display for PredicateRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredicateRef::NamedNode(n) => write!(f, "{}", n),
            PredicateRef::Variable(v) => write!(f, "{}", v),
        }
    }
}

/// Borrowed object reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectRef<'a> {
    NamedNode(&'a NamedNode),
    BlankNode(&'a BlankNode),
    Literal(&'a Literal),
    Variable(&'a Variable),
}

impl<'a> ObjectRef<'a> {
    /// Converts to an owned object
    pub fn to_owned(&self) -> Object {
        match self {
            ObjectRef::NamedNode(n) => Object::NamedNode((*n).clone()),
            ObjectRef::BlankNode(b) => Object::BlankNode((*b).clone()),
            ObjectRef::Literal(l) => Object::Literal((*l).clone()),
            ObjectRef::Variable(v) => Object::Variable((*v).clone()),
        }
    }
}

impl<'a> fmt::Display for ObjectRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectRef::NamedNode(n) => write!(f, "{}", n),
            ObjectRef::BlankNode(b) => write!(f, "{}", b),
            ObjectRef::Literal(l) => write!(f, "{}", l),
            ObjectRef::Variable(v) => write!(f, "{}", v),
        }
    }
}

// Conversion implementations
impl<'a> From<&'a Subject> for SubjectRef<'a> {
    fn from(subject: &'a Subject) -> Self {
        match subject {
            Subject::NamedNode(n) => SubjectRef::NamedNode(n),
            Subject::BlankNode(b) => SubjectRef::BlankNode(b),
            Subject::Variable(v) => SubjectRef::Variable(v),
            Subject::QuotedTriple(_) => panic!("QuotedTriple not supported in SubjectRef"),
        }
    }
}

impl<'a> From<&'a Predicate> for PredicateRef<'a> {
    fn from(predicate: &'a Predicate) -> Self {
        match predicate {
            Predicate::NamedNode(n) => PredicateRef::NamedNode(n),
            Predicate::Variable(v) => PredicateRef::Variable(v),
        }
    }
}

impl<'a> From<&'a Object> for ObjectRef<'a> {
    fn from(object: &'a Object) -> Self {
        match object {
            Object::NamedNode(n) => ObjectRef::NamedNode(n),
            Object::BlankNode(b) => ObjectRef::BlankNode(b),
            Object::Literal(l) => ObjectRef::Literal(l),
            Object::Variable(v) => ObjectRef::Variable(v),
            Object::QuotedTriple(_) => panic!("QuotedTriple not supported in ObjectRef"),
        }
    }
}

impl<'a> From<&'a Triple> for TripleRef<'a> {
    fn from(triple: &'a Triple) -> Self {
        TripleRef {
            subject: triple.subject().into(),
            predicate: triple.predicate().into(),
            object: triple.object().into(),
        }
    }
}

impl<'a> From<TripleRef<'a>> for Triple {
    fn from(triple_ref: TripleRef<'a>) -> Self {
        triple_ref.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BlankNode, Literal, NamedNode};

    #[test]
    fn test_triple_creation() {
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");

        let triple = Triple::new(subject.clone(), predicate.clone(), object.clone());

        assert!(triple.is_ground());
        assert!(!triple.has_variables());
    }

    #[test]
    fn test_triple_with_variable() {
        let subject = Variable::new("x").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");

        let triple = Triple::new(subject, predicate, object);

        assert!(!triple.is_ground());
        assert!(triple.has_variables());
    }

    #[test]
    fn test_triple_display() {
        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("o");

        let triple = Triple::new(subject, predicate, object);
        let display_str = format!("{}", triple);

        assert!(display_str.contains("http://example.org/s"));
        assert!(display_str.contains("http://example.org/p"));
        assert!(display_str.contains("\"o\""));
        assert!(display_str.ends_with(" ."));
    }

    #[test]
    fn test_triple_ref() {
        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("o");

        let triple = Triple::new(subject, predicate, object);
        let triple_ref = TripleRef::from(&triple);
        let triple_owned = triple_ref.to_owned();

        assert_eq!(triple, triple_owned);
    }

    #[test]
    fn test_pattern_matching() {
        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("o");

        let triple = Triple::new(subject.clone(), predicate.clone(), object.clone());

        // Test exact match
        assert!(triple.matches_pattern(
            Some(&Subject::NamedNode(subject.clone())),
            Some(&Predicate::NamedNode(predicate.clone())),
            Some(&Object::Literal(object.clone()))
        ));

        // Test wildcard matches
        assert!(triple.matches_pattern(None, None, None));
        assert!(triple.matches_pattern(Some(&Subject::NamedNode(subject.clone())), None, None));
        assert!(triple.matches_pattern(None, Some(&Predicate::NamedNode(predicate.clone())), None));
        assert!(triple.matches_pattern(None, None, Some(&Object::Literal(object.clone()))));

        // Test non-matches
        let different_subject = NamedNode::new("http://example.org/different").unwrap();
        assert!(!triple.matches_pattern(Some(&Subject::NamedNode(different_subject)), None, None));
    }

    #[test]
    fn test_triple_ordering() {
        let subject1 = NamedNode::new("http://example.org/a").unwrap();
        let subject2 = NamedNode::new("http://example.org/b").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("o");

        let triple1 = Triple::new(subject1, predicate.clone(), object.clone());
        let triple2 = Triple::new(subject2, predicate, object);

        assert!(triple1 < triple2);

        let mut triples = vec![triple2.clone(), triple1.clone()];
        triples.sort();
        assert_eq!(triples, vec![triple1, triple2]);
    }

    #[test]
    fn test_triple_serialization() {
        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("o");

        let triple = Triple::new(subject, predicate, object);
        let json = serde_json::to_string(&triple).unwrap();
        let deserialized: Triple = serde_json::from_str(&json).unwrap();

        assert_eq!(triple, deserialized);
    }
}
