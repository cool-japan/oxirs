//! Dataset Implementation
//!
//! This module provides dataset abstractions and implementations for query execution.

use crate::algebra::{PropertyPath, Term as AlgebraTerm, TriplePattern};
use crate::path::{PathDataset, PropertyPath as PathPropertyPath};
use anyhow::{anyhow, Result};
use oxirs_core::RdfTerm;
use std::collections::HashSet;

/// Dataset trait for data access during query execution
pub trait Dataset: Send + Sync {
    /// Find all triples matching the given pattern
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>>;

    /// Check if a triple exists in the dataset
    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool>;

    /// Get all subjects in the dataset
    fn subjects(&self) -> Result<Vec<AlgebraTerm>>;

    /// Get all predicates in the dataset
    fn predicates(&self) -> Result<Vec<AlgebraTerm>>;

    /// Get all objects in the dataset
    fn objects(&self) -> Result<Vec<AlgebraTerm>>;
}

/// In-memory dataset implementation for testing
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    triples: Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>,
}

impl InMemoryDataset {
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    pub fn add_triple(
        &mut self,
        subject: AlgebraTerm,
        predicate: AlgebraTerm,
        object: AlgebraTerm,
    ) {
        self.triples.push((subject, predicate, object));
    }

    pub fn from_triples(triples: Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>) -> Self {
        Self { triples }
    }
}

impl Dataset for InMemoryDataset {
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        let results = self
            .triples
            .iter()
            .filter(|(s, p, o)| {
                matches_term(&pattern.subject, s)
                    && matches_term(&pattern.predicate, p)
                    && matches_term(&pattern.object, o)
            })
            .cloned()
            .collect();
        Ok(results)
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(s, p, o)| s == subject && p == predicate && o == object))
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        let subjects: HashSet<_> = self.triples.iter().map(|(s, _, _)| s.clone()).collect();
        Ok(subjects.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        let predicates: HashSet<_> = self.triples.iter().map(|(_, p, _)| p.clone()).collect();
        Ok(predicates.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        let objects: HashSet<_> = self.triples.iter().map(|(_, _, o)| o.clone()).collect();
        Ok(objects.into_iter().collect())
    }
}

impl Default for InMemoryDataset {
    fn default() -> Self {
        Self::new()
    }
}

fn matches_term(pattern: &AlgebraTerm, term: &AlgebraTerm) -> bool {
    match pattern {
        AlgebraTerm::Variable(_) => true, // Variables match any term
        _ => pattern == term,
    }
}

/// Adapter to make Dataset implement PathDataset
pub struct DatasetPathAdapter<'a> {
    dataset: &'a dyn Dataset,
}

impl<'a> DatasetPathAdapter<'a> {
    pub fn new(dataset: &'a dyn Dataset) -> Self {
        Self { dataset }
    }
}

impl<'a> PathDataset for DatasetPathAdapter<'a> {
    fn find_outgoing(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
    ) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            subject.clone(),
            predicate.clone(),
            AlgebraTerm::Variable(crate::algebra::Variable::new("?o")?),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(_, _, o)| o).collect())
    }

    fn find_incoming(
        &self,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            AlgebraTerm::Variable(crate::algebra::Variable::new("?s")?),
            predicate.clone(),
            object.clone(),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(s, _, _)| s).collect())
    }

    fn find_predicates(
        &self,
        subject: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            subject.clone(),
            AlgebraTerm::Variable(crate::algebra::Variable::new("?p")?),
            object.clone(),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(_, p, _)| p).collect())
    }

    fn get_predicates(&self) -> Result<Vec<AlgebraTerm>> {
        self.dataset.predicates()
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        self.dataset.contains_triple(subject, predicate, object)
    }
}

/// Convert algebra PropertyPath to path module PropertyPath
pub fn convert_property_path(path: &PropertyPath) -> Result<PathPropertyPath> {
    match path {
        PropertyPath::Iri(iri) => Ok(PathPropertyPath::Direct(AlgebraTerm::Iri(iri.clone()))),
        PropertyPath::Variable(var) => {
            Ok(PathPropertyPath::Direct(AlgebraTerm::Variable(var.clone())))
        }
        PropertyPath::Inverse(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::Inverse(Box::new(inner_path)))
        }
        PropertyPath::Sequence(left, right) => {
            let left_path = convert_property_path(left)?;
            let right_path = convert_property_path(right)?;
            Ok(PathPropertyPath::Sequence(
                Box::new(left_path),
                Box::new(right_path),
            ))
        }
        PropertyPath::Alternative(left, right) => {
            let left_path = convert_property_path(left)?;
            let right_path = convert_property_path(right)?;
            Ok(PathPropertyPath::Alternative(
                Box::new(left_path),
                Box::new(right_path),
            ))
        }
        PropertyPath::ZeroOrMore(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::ZeroOrMore(Box::new(inner_path)))
        }
        PropertyPath::OneOrMore(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::OneOrMore(Box::new(inner_path)))
        }
        PropertyPath::ZeroOrOne(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::ZeroOrOne(Box::new(inner_path)))
        }
        PropertyPath::NegatedPropertySet(paths) => {
            let mut terms = Vec::new();
            for p in paths {
                match p {
                    PropertyPath::Iri(iri) => terms.push(AlgebraTerm::Iri(iri.clone())),
                    PropertyPath::Variable(var) => terms.push(AlgebraTerm::Variable(var.clone())),
                    _ => {
                        return Err(anyhow!(
                            "Negated property set can only contain IRIs or variables"
                        ))
                    }
                }
            }
            Ok(PathPropertyPath::NegatedPropertySet(terms))
        }
    }
}

/// Adapter to make ConcreteStore implement Dataset trait
/// This is primarily for benchmarking and testing purposes
pub struct ConcreteStoreDataset {
    store: std::sync::Arc<oxirs_core::rdf_store::ConcreteStore>,
}

impl ConcreteStoreDataset {
    pub fn new(store: oxirs_core::rdf_store::ConcreteStore) -> Self {
        Self {
            store: std::sync::Arc::new(store),
        }
    }

    pub fn from_arc(store: std::sync::Arc<oxirs_core::rdf_store::ConcreteStore>) -> Self {
        Self { store }
    }
}

impl Clone for ConcreteStoreDataset {
    fn clone(&self) -> Self {
        Self {
            store: std::sync::Arc::clone(&self.store),
        }
    }
}

impl Dataset for ConcreteStoreDataset {
    fn find_triples(
        &self,
        pattern: &TriplePattern,
    ) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        use oxirs_core::rdf_store::Store;

        // Convert pattern to ConcreteStore query
        let subject = match &pattern.subject {
            AlgebraTerm::Iri(iri) => Some(oxirs_core::model::Subject::NamedNode(iri.clone())),
            AlgebraTerm::Variable(_) => None,
            AlgebraTerm::BlankNode(id) => Some(oxirs_core::model::Subject::BlankNode(
                oxirs_core::model::BlankNode::new(id)
                    .map_err(|e| anyhow!("Invalid blank node: {}", e))?,
            )),
            _ => return Err(anyhow!("Invalid subject in pattern")),
        };

        let predicate = match &pattern.predicate {
            AlgebraTerm::Iri(iri) => Some(oxirs_core::model::Predicate::NamedNode(iri.clone())),
            AlgebraTerm::Variable(_) => None, // Wildcard - match any predicate
            AlgebraTerm::PropertyPath(path) => {
                // For simple property paths that are just IRIs, extract the IRI
                match path {
                    crate::algebra::PropertyPath::Iri(iri) => {
                        Some(oxirs_core::model::Predicate::NamedNode(iri.clone()))
                    }
                    crate::algebra::PropertyPath::Variable(_) => None, // Wildcard
                    _ => {
                        return Err(anyhow!(
                            "Complex property paths not yet supported in find_triples"
                        ))
                    }
                }
            }
            _ => {
                return Err(anyhow!(
                    "Predicate must be IRI, variable, or property path, got: {:?}",
                    pattern.predicate
                ))
            }
        };

        let object = match &pattern.object {
            AlgebraTerm::Iri(iri) => Some(oxirs_core::model::Object::NamedNode(iri.clone())),
            AlgebraTerm::Literal(lit) => Some(oxirs_core::model::Object::Literal(
                oxirs_core::model::Literal::new(&lit.value),
            )),
            AlgebraTerm::BlankNode(id) => Some(oxirs_core::model::Object::BlankNode(
                oxirs_core::model::BlankNode::new(id)
                    .map_err(|e| anyhow!("Invalid blank node: {}", e))?,
            )),
            AlgebraTerm::Variable(_) => None,
            _ => return Err(anyhow!("Invalid object in pattern")),
        };

        // Query the store
        let quads = self.store.find_quads(
            subject.as_ref(),
            predicate.as_ref(),
            object.as_ref(),
            None, // default graph
        )?;

        // Convert quads to triples
        let triples: Vec<_> = quads
            .into_iter()
            .filter_map(|quad| {
                let s = match quad.subject() {
                    oxirs_core::model::Subject::NamedNode(n) => {
                        AlgebraTerm::Iri(oxirs_core::model::NamedNode::new(n.as_str()).ok()?)
                    }
                    oxirs_core::model::Subject::BlankNode(b) => {
                        AlgebraTerm::BlankNode(b.as_str().to_string())
                    }
                    oxirs_core::model::Subject::Variable(v) => AlgebraTerm::Variable(v.clone()),
                    oxirs_core::model::Subject::QuotedTriple(_) => {
                        // Skip RDF-star quoted triples for now
                        return None;
                    }
                };

                let p = AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(quad.predicate().as_str()).ok()?,
                );

                let o = match quad.object() {
                    oxirs_core::model::Object::NamedNode(n) => {
                        AlgebraTerm::Iri(oxirs_core::model::NamedNode::new(n.as_str()).ok()?)
                    }
                    oxirs_core::model::Object::Literal(l) => {
                        AlgebraTerm::Literal(crate::algebra::Literal {
                            value: l.value().to_string(),
                            datatype: None,
                            language: l.language().map(|s| s.to_string()),
                        })
                    }
                    oxirs_core::model::Object::BlankNode(b) => {
                        AlgebraTerm::BlankNode(b.as_str().to_string())
                    }
                    oxirs_core::model::Object::Variable(v) => AlgebraTerm::Variable(v.clone()),
                    oxirs_core::model::Object::QuotedTriple(_) => {
                        // Skip RDF-star quoted triples for now
                        return None;
                    }
                };

                Some((s, p, o))
            })
            .collect();

        Ok(triples)
    }

    fn contains_triple(
        &self,
        subject: &AlgebraTerm,
        predicate: &AlgebraTerm,
        object: &AlgebraTerm,
    ) -> Result<bool> {
        let pattern = TriplePattern::new(subject.clone(), predicate.clone(), object.clone());
        let triples = self.find_triples(&pattern)?;
        Ok(!triples.is_empty())
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        use oxirs_core::rdf_store::Store;
        let quads = self.store.find_quads(None, None, None, None)?;
        let subjects: HashSet<_> = quads
            .into_iter()
            .filter_map(|quad| match quad.subject() {
                oxirs_core::model::Subject::NamedNode(n) => Some(AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(n.as_str()).ok()?,
                )),
                oxirs_core::model::Subject::BlankNode(b) => {
                    Some(AlgebraTerm::BlankNode(b.as_str().to_string()))
                }
                oxirs_core::model::Subject::Variable(v) => Some(AlgebraTerm::Variable(v.clone())),
                oxirs_core::model::Subject::QuotedTriple(_) => None,
            })
            .collect();
        Ok(subjects.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        use oxirs_core::rdf_store::Store;
        let quads = self.store.find_quads(None, None, None, None)?;
        let predicates: HashSet<_> = quads
            .into_iter()
            .filter_map(|quad| {
                Some(AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(quad.predicate().as_str()).ok()?,
                ))
            })
            .collect();
        Ok(predicates.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        use oxirs_core::rdf_store::Store;
        let quads = self.store.find_quads(None, None, None, None)?;
        let objects: HashSet<_> = quads
            .into_iter()
            .filter_map(|quad| match quad.object() {
                oxirs_core::model::Object::NamedNode(n) => Some(AlgebraTerm::Iri(
                    oxirs_core::model::NamedNode::new(n.as_str()).ok()?,
                )),
                oxirs_core::model::Object::Literal(l) => {
                    Some(AlgebraTerm::Literal(crate::algebra::Literal {
                        value: l.value().to_string(),
                        datatype: None,
                        language: l.language().map(|s| s.to_string()),
                    }))
                }
                oxirs_core::model::Object::BlankNode(b) => {
                    Some(AlgebraTerm::BlankNode(b.as_str().to_string()))
                }
                oxirs_core::model::Object::Variable(v) => Some(AlgebraTerm::Variable(v.clone())),
                oxirs_core::model::Object::QuotedTriple(_) => None,
            })
            .collect();
        Ok(objects.into_iter().collect())
    }
}
