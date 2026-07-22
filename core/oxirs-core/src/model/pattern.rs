//! Pattern matching for RDF triples
//!
//! This module provides pattern matching functionality for querying RDF triples.

use crate::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple, Variable};

/// A pattern for matching triples
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SubjectPattern {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Variable(Variable),
    /// RDF-star: a quoted triple used as a subject
    QuotedTriple(Box<crate::query::algebra::AlgebraTriplePattern>),
}

impl SubjectPattern {
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            SubjectPattern::NamedNode(nn) => nn.as_str(),
            SubjectPattern::BlankNode(bn) => bn.as_str(),
            SubjectPattern::Variable(v) => v.as_str(),
            SubjectPattern::QuotedTriple(_) => "<<quoted-triple>>",
        }
    }

    fn matches(&self, subject: &Subject) -> bool {
        match (self, subject) {
            (SubjectPattern::NamedNode(pn), Subject::NamedNode(sn)) => pn == sn,
            (SubjectPattern::BlankNode(pb), Subject::BlankNode(sb)) => pb == sb,
            (SubjectPattern::Variable(_), _) => true,
            // A quoted-triple pattern matches a quoted-triple subject only when the
            // inner pattern structurally matches the inner triple (variables act as
            // wildcards). A fully-ground inner pattern requires exact equality.
            (SubjectPattern::QuotedTriple(pat), Subject::QuotedTriple(sqt)) => {
                algebra_pattern_matches_triple(pat, sqt.inner())
            }
            _ => false,
        }
    }
}

/// Structurally match a SPARQL algebra triple pattern against a concrete triple.
///
/// Variables in the pattern act as wildcards; concrete terms (including nested
/// quoted triples) must match exactly. This is used for RDF-star quoted-triple
/// pattern matching so that a ground quoted-triple pattern such as
/// `<< <a> <b> <c> >> ?p ?o` does NOT spuriously match a triple whose subject is
/// `<< <x> <y> <z> >>`.
pub(crate) fn algebra_pattern_matches_triple(
    pattern: &crate::query::algebra::AlgebraTriplePattern,
    triple: &Triple,
) -> bool {
    term_pattern_matches_subject(&pattern.subject, triple.subject())
        && term_pattern_matches_predicate(&pattern.predicate, triple.predicate())
        && term_pattern_matches_object(&pattern.object, triple.object())
}

fn term_pattern_matches_subject(
    pattern: &crate::query::algebra::TermPattern,
    subject: &Subject,
) -> bool {
    use crate::query::algebra::TermPattern;
    match (pattern, subject) {
        (TermPattern::Variable(_), _) => true,
        (TermPattern::NamedNode(pn), Subject::NamedNode(sn)) => pn == sn,
        (TermPattern::BlankNode(pb), Subject::BlankNode(sb)) => pb == sb,
        (TermPattern::QuotedTriple(pat), Subject::QuotedTriple(sqt)) => {
            algebra_pattern_matches_triple(pat, sqt.inner())
        }
        _ => false,
    }
}

fn term_pattern_matches_predicate(
    pattern: &crate::query::algebra::TermPattern,
    predicate: &Predicate,
) -> bool {
    use crate::query::algebra::TermPattern;
    match (pattern, predicate) {
        (TermPattern::Variable(_), _) => true,
        (TermPattern::NamedNode(pn), Predicate::NamedNode(sn)) => pn == sn,
        _ => false,
    }
}

fn term_pattern_matches_object(
    pattern: &crate::query::algebra::TermPattern,
    object: &Object,
) -> bool {
    use crate::query::algebra::TermPattern;
    match (pattern, object) {
        (TermPattern::Variable(_), _) => true,
        (TermPattern::NamedNode(pn), Object::NamedNode(on)) => pn == on,
        (TermPattern::BlankNode(pb), Object::BlankNode(ob)) => pb == ob,
        (TermPattern::Literal(pl), Object::Literal(ol)) => pl == ol,
        (TermPattern::QuotedTriple(pat), Object::QuotedTriple(oqt)) => {
            algebra_pattern_matches_triple(pat, oqt.inner())
        }
        _ => false,
    }
}

/// Pattern for matching predicates
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ObjectPattern {
    NamedNode(NamedNode),
    BlankNode(BlankNode),
    Literal(Literal),
    Variable(Variable),
    /// RDF-star: a quoted triple used as an object
    QuotedTriple(Box<crate::query::algebra::AlgebraTriplePattern>),
}

impl ObjectPattern {
    /// Get the string representation
    pub fn as_str(&self) -> &str {
        match self {
            ObjectPattern::NamedNode(nn) => nn.as_str(),
            ObjectPattern::BlankNode(bn) => bn.as_str(),
            ObjectPattern::Literal(l) => l.value(),
            ObjectPattern::Variable(v) => v.as_str(),
            ObjectPattern::QuotedTriple(_) => "<<quoted-triple>>",
        }
    }

    fn matches(&self, object: &Object) -> bool {
        match (self, object) {
            (ObjectPattern::NamedNode(pn), Object::NamedNode(on)) => pn == on,
            (ObjectPattern::BlankNode(pb), Object::BlankNode(ob)) => pb == ob,
            (ObjectPattern::Literal(pl), Object::Literal(ol)) => pl == ol,
            (ObjectPattern::Variable(_), _) => true,
            // A quoted-triple pattern matches a quoted-triple object only when the
            // inner pattern structurally matches the inner triple (variables act as
            // wildcards). A fully-ground inner pattern requires exact equality.
            (ObjectPattern::QuotedTriple(pat), Object::QuotedTriple(oqt)) => {
                algebra_pattern_matches_triple(pat, oqt.inner())
            }
            _ => false,
        }
    }
}

// TryFrom implementations for converting TermPattern to positional patterns.
// Using TryFrom rather than From because certain TermPattern variants are semantically
// invalid for particular positions (e.g., Literal cannot be a subject in RDF).
use crate::query::algebra::TermPattern;

/// Error type for invalid TermPattern-to-positional-pattern conversions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidPatternConversion {
    pub reason: &'static str,
}

impl std::fmt::Display for InvalidPatternConversion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Invalid pattern conversion: {}", self.reason)
    }
}

impl std::error::Error for InvalidPatternConversion {}

impl TryFrom<TermPattern> for SubjectPattern {
    type Error = InvalidPatternConversion;

    fn try_from(term: TermPattern) -> Result<Self, Self::Error> {
        match term {
            TermPattern::NamedNode(n) => Ok(SubjectPattern::NamedNode(n)),
            TermPattern::BlankNode(b) => Ok(SubjectPattern::BlankNode(b)),
            TermPattern::Variable(v) => Ok(SubjectPattern::Variable(v)),
            TermPattern::QuotedTriple(qt) => Ok(SubjectPattern::QuotedTriple(qt)),
            TermPattern::Literal(_) => Err(InvalidPatternConversion {
                reason: "Literals cannot be used as subjects in RDF",
            }),
        }
    }
}

impl TryFrom<TermPattern> for PredicatePattern {
    type Error = InvalidPatternConversion;

    fn try_from(term: TermPattern) -> Result<Self, Self::Error> {
        match term {
            TermPattern::NamedNode(n) => Ok(PredicatePattern::NamedNode(n)),
            TermPattern::Variable(v) => Ok(PredicatePattern::Variable(v)),
            TermPattern::BlankNode(_) => Err(InvalidPatternConversion {
                reason: "Blank nodes cannot be used as predicates in RDF",
            }),
            TermPattern::Literal(_) => Err(InvalidPatternConversion {
                reason: "Literals cannot be used as predicates in RDF",
            }),
            TermPattern::QuotedTriple(_) => Err(InvalidPatternConversion {
                reason: "Quoted triples cannot be used as predicates in RDF",
            }),
        }
    }
}

impl TryFrom<TermPattern> for ObjectPattern {
    type Error = InvalidPatternConversion;

    fn try_from(term: TermPattern) -> Result<Self, Self::Error> {
        match term {
            TermPattern::NamedNode(n) => Ok(ObjectPattern::NamedNode(n)),
            TermPattern::BlankNode(b) => Ok(ObjectPattern::BlankNode(b)),
            TermPattern::Literal(l) => Ok(ObjectPattern::Literal(l)),
            TermPattern::Variable(v) => Ok(ObjectPattern::Variable(v)),
            TermPattern::QuotedTriple(qt) => Ok(ObjectPattern::QuotedTriple(qt)),
        }
    }
}

// TryFrom implementations for converting patterns to concrete terms
impl TryFrom<&SubjectPattern> for Subject {
    type Error = ();

    fn try_from(pattern: &SubjectPattern) -> Result<Self, Self::Error> {
        match pattern {
            SubjectPattern::NamedNode(n) => Ok(Subject::NamedNode(n.clone())),
            SubjectPattern::BlankNode(b) => Ok(Subject::BlankNode(b.clone())),
            SubjectPattern::Variable(_) => Err(()),
            SubjectPattern::QuotedTriple(qt) => {
                // Convert AlgebraTriplePattern to a concrete QuotedTriple if possible.
                // This requires all sub-terms to be ground (non-variable).
                use crate::model::star::QuotedTriple;
                let inner_subj = match &qt.subject {
                    TermPattern::NamedNode(n) => Subject::NamedNode(n.clone()),
                    TermPattern::BlankNode(b) => Subject::BlankNode(b.clone()),
                    _ => return Err(()),
                };
                let inner_pred = match &qt.predicate {
                    TermPattern::NamedNode(n) => Predicate::NamedNode(n.clone()),
                    _ => return Err(()),
                };
                let inner_obj = match &qt.object {
                    TermPattern::NamedNode(n) => Object::NamedNode(n.clone()),
                    TermPattern::BlankNode(b) => Object::BlankNode(b.clone()),
                    TermPattern::Literal(l) => Object::Literal(l.clone()),
                    _ => return Err(()),
                };
                Ok(Subject::QuotedTriple(Box::new(QuotedTriple::new(
                    Triple::new(inner_subj, inner_pred, inner_obj),
                ))))
            }
        }
    }
}

impl TryFrom<&PredicatePattern> for Predicate {
    type Error = ();

    fn try_from(pattern: &PredicatePattern) -> Result<Self, Self::Error> {
        match pattern {
            PredicatePattern::NamedNode(n) => Ok(Predicate::NamedNode(n.clone())),
            PredicatePattern::Variable(_) => Err(()),
        }
    }
}

impl TryFrom<&ObjectPattern> for Object {
    type Error = ();

    fn try_from(pattern: &ObjectPattern) -> Result<Self, Self::Error> {
        match pattern {
            ObjectPattern::NamedNode(n) => Ok(Object::NamedNode(n.clone())),
            ObjectPattern::BlankNode(b) => Ok(Object::BlankNode(b.clone())),
            ObjectPattern::Literal(l) => Ok(Object::Literal(l.clone())),
            ObjectPattern::Variable(_) => Err(()),
            ObjectPattern::QuotedTriple(qt) => {
                // Convert AlgebraTriplePattern to a concrete QuotedTriple if all terms are ground.
                use crate::model::star::QuotedTriple;
                let inner_subj = match &qt.subject {
                    TermPattern::NamedNode(n) => Subject::NamedNode(n.clone()),
                    TermPattern::BlankNode(b) => Subject::BlankNode(b.clone()),
                    _ => return Err(()),
                };
                let inner_pred = match &qt.predicate {
                    TermPattern::NamedNode(n) => Predicate::NamedNode(n.clone()),
                    _ => return Err(()),
                };
                let inner_obj = match &qt.object {
                    TermPattern::NamedNode(n) => Object::NamedNode(n.clone()),
                    TermPattern::BlankNode(b) => Object::BlankNode(b.clone()),
                    TermPattern::Literal(l) => Object::Literal(l.clone()),
                    _ => return Err(()),
                };
                Ok(Object::QuotedTriple(Box::new(QuotedTriple::new(
                    Triple::new(inner_subj, inner_pred, inner_obj),
                ))))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matching() {
        let subject = NamedNode::new("http://example.org/s").expect("valid IRI");
        let predicate = NamedNode::new("http://example.org/p").expect("valid IRI");
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
            Some(SubjectPattern::Variable(
                Variable::new("s").expect("valid variable name"),
            )),
            Some(PredicatePattern::Variable(
                Variable::new("p").expect("valid variable name"),
            )),
            Some(ObjectPattern::Variable(
                Variable::new("o").expect("valid variable name"),
            )),
        );
        assert!(pattern.matches(&triple));

        // Test non-match
        let different_subject = NamedNode::new("http://example.org/different").expect("valid IRI");
        let pattern = TriplePattern::new(
            Some(SubjectPattern::NamedNode(different_subject)),
            None,
            None,
        );
        assert!(!pattern.matches(&triple));
    }

    #[test]
    fn regression_ground_quoted_triple_pattern_is_exact() {
        use crate::model::star::QuotedTriple;
        use crate::query::algebra::{AlgebraTriplePattern, TermPattern};

        let nn = |s: &str| NamedNode::new(s).expect("valid IRI");

        // Inner triple stored in the data: << <a> <b> <c> >>
        let stored_inner = Triple::new(nn("http://ex/a"), nn("http://ex/b"), nn("http://ex/c"));
        let subject = Subject::QuotedTriple(Box::new(QuotedTriple::new(stored_inner.clone())));

        // Ground pattern << <a> <b> <c> >> must match the identical inner triple.
        let ground_match = AlgebraTriplePattern::new(
            TermPattern::NamedNode(nn("http://ex/a")),
            TermPattern::NamedNode(nn("http://ex/b")),
            TermPattern::NamedNode(nn("http://ex/c")),
        );
        let sp_match = SubjectPattern::QuotedTriple(Box::new(ground_match));
        assert!(sp_match.matches(&subject));

        // Ground pattern << <x> <y> <z> >> must NOT match (previously returned true).
        let ground_diff = AlgebraTriplePattern::new(
            TermPattern::NamedNode(nn("http://ex/x")),
            TermPattern::NamedNode(nn("http://ex/y")),
            TermPattern::NamedNode(nn("http://ex/z")),
        );
        let sp_diff = SubjectPattern::QuotedTriple(Box::new(ground_diff));
        assert!(!sp_diff.matches(&subject));

        // A pattern with variable positions acts as a wildcard and matches.
        let var_pat = AlgebraTriplePattern::new(
            TermPattern::NamedNode(nn("http://ex/a")),
            TermPattern::Variable(Variable::new("p").expect("valid var")),
            TermPattern::Variable(Variable::new("o").expect("valid var")),
        );
        let sp_var = SubjectPattern::QuotedTriple(Box::new(var_pat));
        assert!(sp_var.matches(&subject));

        // Same semantics for the object position.
        let object = Object::QuotedTriple(Box::new(QuotedTriple::new(stored_inner)));
        let op_diff = ObjectPattern::QuotedTriple(Box::new(AlgebraTriplePattern::new(
            TermPattern::NamedNode(nn("http://ex/x")),
            TermPattern::NamedNode(nn("http://ex/y")),
            TermPattern::NamedNode(nn("http://ex/z")),
        )));
        assert!(!op_diff.matches(&object));
    }
}
