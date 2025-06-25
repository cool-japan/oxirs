//! Pattern matching for RDF triples
//!
//! This module provides pattern matching functionality for querying RDF triples.

use crate::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple, Variable, RdfTerm};
use serde::{Deserialize, Serialize};

/// A pattern for matching triples
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: Option<SubjectPattern>,
    pub predicate: Option<PredicatePattern>,
    pub object: Option<ObjectPattern>,
}

impl TriplePattern {
    /// Create a new triple pattern
    pub fn new(
        subject: Option<SubjectPattern>,
        predicate: Option<PredicatePattern>,
        object: Option<ObjectPattern>,
    ) -> Self {
        TriplePattern {
            subject,
            predicate,
            object,
        }
    }

    /// Get the subject pattern
    pub fn subject(&self) -> Option<&SubjectPattern> {
        self.subject.as_ref()
    }

    /// Get the predicate pattern
    pub fn predicate(&self) -> Option<&PredicatePattern> {
        self.predicate.as_ref()
    }

    /// Get the object pattern
    pub fn object(&self) -> Option<&ObjectPattern> {
        self.object.as_ref()
    }

    /// Check if a triple matches this pattern
    pub fn matches(&self, triple: &Triple) -> bool {
        // Check subject
        if let Some(ref subject_pattern) = self.subject {
            if !subject_pattern.matches(triple.subject()) {
                return false;
            }
        }

        // Check predicate
        if let Some(ref predicate_pattern) = self.predicate {
            if !predicate_pattern.matches(triple.predicate()) {
                return false;
            }
        }

        // Check object
        if let Some(ref object_pattern) = self.object {
            if !object_pattern.matches(triple.object()) {
                return false;
            }
        }

        true
    }
}

/// Pattern for matching subjects
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubjectPattern {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Variable(Variable),
}

impl SubjectPattern {
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            SubjectPattern::NamedNode(nn) => nn.as_str(),
            SubjectPattern::BlankNode(bn) => bn.as_str(),
            SubjectPattern::Variable(v) => v.as_str(),
        }
    }

    fn matches(&self, subject: &Subject) -> bool {
        match (self, subject) {
            (SubjectPattern::NamedNode(pn), Subject::NamedNode(sn)) => pn == sn,
            (SubjectPattern::BlankNode(pb), Subject::BlankNode(sb)) => pb == sb,
            (SubjectPattern::Variable(_), _) => true,
            _ => false,
        }
    }
}

/// Pattern for matching predicates
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredicatePattern {
    NamedNode(NamedNode),
    Variable(Variable),
}

impl PredicatePattern {
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            PredicatePattern::NamedNode(nn) => nn.as_str(),
            PredicatePattern::Variable(v) => v.as_str(),
        }
    }

    fn matches(&self, predicate: &Predicate) -> bool {
        match (self, predicate) {
            (PredicatePattern::NamedNode(pn), Predicate::NamedNode(sn)) => pn == sn,
            (PredicatePattern::Variable(_), _) => true,
            _ => false,
        }
    }
}

/// Pattern for matching objects
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectPattern {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Literal(Literal),
    Variable(Variable),
}

impl ObjectPattern {
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            ObjectPattern::NamedNode(nn) => nn.as_str(),
            ObjectPattern::BlankNode(bn) => bn.as_str(),
            ObjectPattern::Literal(l) => l.value(),
            ObjectPattern::Variable(v) => v.as_str(),
        }
    }

    fn matches(&self, object: &Object) -> bool {
        match (self, object) {
            (ObjectPattern::NamedNode(pn), Object::NamedNode(on)) => pn == on,
            (ObjectPattern::BlankNode(pb), Object::BlankNode(ob)) => pb == ob,
            (ObjectPattern::Literal(pl), Object::Literal(ol)) => pl == ol,
            (ObjectPattern::Variable(_), _) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matching() {
        let subject = NamedNode::new("http://example.org/s").unwrap();
        let predicate = NamedNode::new("http://example.org/p").unwrap();
        let object = Literal::new("o");

        let triple = Triple::new(subject.clone(), predicate.clone(), object.clone());

        // Test exact match
        let pattern = TriplePattern::new(
            Some(SubjectPattern::NamedNode(subject.clone())),
            Some(PredicatePattern::NamedNode(predicate.clone())),
            Some(ObjectPattern::Literal(object.clone())),
        );
        assert!(pattern.matches(&triple));

        // Test wildcard match
        let pattern = TriplePattern::new(None, None, None);
        assert!(pattern.matches(&triple));

        // Test variable match
        let pattern = TriplePattern::new(
            Some(SubjectPattern::Variable(Variable::new("s").unwrap())),
            Some(PredicatePattern::Variable(Variable::new("p").unwrap())),
            Some(ObjectPattern::Variable(Variable::new("o").unwrap())),
        );
        assert!(pattern.matches(&triple));

        // Test non-match
        let different_subject = NamedNode::new("http://example.org/different").unwrap();
        let pattern = TriplePattern::new(
            Some(SubjectPattern::NamedNode(different_subject)),
            None,
            None,
        );
        assert!(!pattern.matches(&triple));
    }
}
