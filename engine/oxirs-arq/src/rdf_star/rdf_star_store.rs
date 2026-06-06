//! In-memory RDF-star store and pattern-matching helpers.
//!
//! Defines [`AnnotatedTriple`] (a quoted triple plus its annotations) and
//! [`RdfStarStore`] — a self-contained store for quoted triples and their
//! annotations that does not depend on the optional `oxirs-star` crate.
//! It underpins the SPARQL-star completeness tests.

use crate::rdf_star::rdf_star_operator::{StarOperator, StarPattern};
use crate::rdf_star::rdf_star_terms::{QuotedTriple, StarObject, StarPredicate, StarSubject};
use oxirs_core::model::NamedNode;
use std::collections::HashMap;

// ─── In-memory RDF-star store ─────────────────────────────────────────────────

/// A triple stored together with its annotations
#[derive(Debug, Clone)]
pub struct AnnotatedTriple {
    /// The quoted triple
    pub triple: QuotedTriple,
    /// Annotation (predicate → object) pairs
    pub annotations: HashMap<String, StarObject>,
}

impl AnnotatedTriple {
    /// Construct a new annotated triple without any annotations
    pub fn new(triple: QuotedTriple) -> Self {
        Self {
            triple,
            annotations: HashMap::new(),
        }
    }

    /// Add or overwrite an annotation
    pub fn annotate(&mut self, predicate: &NamedNode, object: StarObject) {
        self.annotations.insert(predicate.to_string(), object);
    }

    /// Remove an annotation; returns `true` if it was present
    pub fn remove_annotation(&mut self, predicate: &NamedNode) -> bool {
        self.annotations.remove(&predicate.to_string()).is_some()
    }

    /// Return the annotation for the given predicate, if any
    pub fn annotation(&self, predicate: &NamedNode) -> Option<&StarObject> {
        self.annotations.get(&predicate.to_string())
    }
}

/// In-memory store for RDF-star quoted triples and their annotations.
///
/// This is a self-contained store that does not depend on the optional `oxirs-star` crate.
/// It underpins the completeness tests below.
#[derive(Debug, Clone, Default)]
pub struct RdfStarStore {
    /// Annotated triples, keyed by the serialised form of the quoted triple
    triples: HashMap<String, AnnotatedTriple>,
}

impl RdfStarStore {
    /// Construct an empty store
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the number of quoted triples (with or without annotations) in the store
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Return `true` if the store is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Insert or retrieve the entry for a quoted triple
    fn key(triple: &QuotedTriple) -> String {
        triple.to_string()
    }

    /// Assert that a quoted triple exists in the store (idempotent)
    pub fn assert_triple(&mut self, triple: QuotedTriple) {
        self.triples
            .entry(Self::key(&triple))
            .or_insert_with(|| AnnotatedTriple::new(triple));
    }

    /// Retract a quoted triple and all its annotations
    pub fn retract_triple(&mut self, triple: &QuotedTriple) -> bool {
        self.triples.remove(&Self::key(triple)).is_some()
    }

    /// Add an annotation to an existing quoted triple.
    /// If the triple has not been asserted yet, it is created automatically.
    pub fn add_annotation(
        &mut self,
        triple: &QuotedTriple,
        predicate: &NamedNode,
        object: StarObject,
    ) {
        let entry = self
            .triples
            .entry(Self::key(triple))
            .or_insert_with(|| AnnotatedTriple::new(triple.clone()));
        entry.annotate(predicate, object);
    }

    /// Remove a single annotation from a quoted triple.
    /// Returns `true` if the annotation was present.
    pub fn remove_annotation(&mut self, triple: &QuotedTriple, predicate: &NamedNode) -> bool {
        if let Some(entry) = self.triples.get_mut(&Self::key(triple)) {
            entry.remove_annotation(predicate)
        } else {
            false
        }
    }

    /// Look up all annotations for a quoted triple
    pub fn annotations(&self, triple: &QuotedTriple) -> Option<&AnnotatedTriple> {
        self.triples.get(&Self::key(triple))
    }

    /// Return `true` if the quoted triple exists in the store
    pub fn contains(&self, triple: &QuotedTriple) -> bool {
        self.triples.contains_key(&Self::key(triple))
    }

    /// Iterate over all annotated triples
    pub fn iter(&self) -> impl Iterator<Item = &AnnotatedTriple> {
        self.triples.values()
    }

    /// Apply a [`StarOperator`] to this store.
    ///
    /// Returns the matching [`AnnotatedTriple`] entries for `FindAnnotations`,
    /// or an empty vec for mutating operators.
    pub fn apply_operator(&mut self, op: StarOperator) -> Vec<AnnotatedTriple> {
        match op {
            StarOperator::AssertQuoted { triple } => {
                self.assert_triple(triple);
                Vec::new()
            }
            StarOperator::RetractQuoted { triple } => {
                self.retract_triple(&triple);
                Vec::new()
            }
            StarOperator::AddAnnotation {
                triple,
                annotations,
            } => {
                for ann in &annotations {
                    self.add_annotation(&triple, &ann.predicate, ann.object.clone());
                }
                Vec::new()
            }
            StarOperator::RemoveAnnotation { triple, predicate } => {
                self.remove_annotation(&triple, &predicate);
                Vec::new()
            }
            StarOperator::FindAnnotations { pattern } => self.find_annotations(&pattern),
        }
    }

    /// Match all annotation triples against the given [`StarPattern`]
    pub fn find_annotations(&self, pattern: &StarPattern) -> Vec<AnnotatedTriple> {
        self.triples
            .values()
            .filter(|entry| triple_matches_pattern(&entry.triple, &pattern.quoted))
            .filter(|entry| {
                // Filter annotation (pred, obj) pairs
                entry.annotations.iter().any(|(pred_key, obj)| {
                    predicate_matches(&pattern.predicate, pred_key)
                        && object_matches(&pattern.object, obj)
                })
            })
            .cloned()
            .collect()
    }
}

// ─── Pattern matching helpers ─────────────────────────────────────────────────

/// Return `true` if a quoted triple matches a pattern (variables bind to anything)
pub(crate) fn triple_matches_pattern(triple: &QuotedTriple, pattern: &QuotedTriple) -> bool {
    subject_matches(&pattern.subject, &triple.subject)
        && predicate_matches_terms(&pattern.predicate, &triple.predicate)
        && object_matches_object(&pattern.object, &triple.object)
}

fn subject_matches(pattern: &StarSubject, value: &StarSubject) -> bool {
    match pattern {
        StarSubject::Variable(_) => true,
        StarSubject::NamedNode(pn) => {
            matches!(value, StarSubject::NamedNode(vn) if vn == pn)
        }
        StarSubject::BlankNode(pb) => {
            matches!(value, StarSubject::BlankNode(vb) if vb == pb)
        }
        StarSubject::Quoted(pq) => {
            matches!(value, StarSubject::Quoted(vq) if triple_matches_pattern(vq, pq))
        }
    }
}

fn predicate_matches_terms(pattern: &StarPredicate, value: &StarPredicate) -> bool {
    match pattern {
        StarPredicate::Variable(_) => true,
        StarPredicate::NamedNode(pn) => {
            matches!(value, StarPredicate::NamedNode(vn) if vn == pn)
        }
    }
}

pub(crate) fn object_matches_object(pattern: &StarObject, value: &StarObject) -> bool {
    match pattern {
        StarObject::Variable(_) => true,
        StarObject::NamedNode(pn) => {
            matches!(value, StarObject::NamedNode(vn) if vn == pn)
        }
        StarObject::BlankNode(pb) => {
            matches!(value, StarObject::BlankNode(vb) if vb == pb)
        }
        StarObject::Literal(pl) => {
            matches!(value, StarObject::Literal(vl) if vl == pl)
        }
        StarObject::Quoted(pq) => {
            matches!(value, StarObject::Quoted(vq) if triple_matches_pattern(vq, pq))
        }
    }
}

/// Match a predicate pattern (may be variable) against a serialised predicate key
// The key is stored as `NamedNode::to_string()` (angle-bracket IRI), so the
// comparison must use the same Display formatting — suppressing cmp_owned.
#[allow(clippy::cmp_owned)]
fn predicate_matches(pattern: &StarPredicate, key: &str) -> bool {
    match pattern {
        StarPredicate::Variable(_) => true,
        StarPredicate::NamedNode(n) => n.to_string() == key,
    }
}

/// Match an object pattern against a concrete object value
fn object_matches(pattern: &StarObject, value: &StarObject) -> bool {
    object_matches_object(pattern, value)
}
