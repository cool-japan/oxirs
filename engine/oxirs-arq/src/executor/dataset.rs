//! Dataset Implementation
//!
//! This module provides dataset abstractions and implementations for query execution.

use crate::algebra::{Term as AlgebraTerm, TriplePattern, PropertyPath};
use crate::path::{PropertyPath as PathPropertyPath, PathDataset};
use anyhow::{anyhow, Result};
use std::collections::HashSet;

/// Dataset trait for data access during query execution
pub trait Dataset: Send + Sync {
    /// Find all triples matching the given pattern
    fn find_triples(&self, pattern: &TriplePattern) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>>;

    /// Check if a triple exists in the dataset
    fn contains_triple(&self, subject: &AlgebraTerm, predicate: &AlgebraTerm, object: &AlgebraTerm) -> Result<bool>;

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

    pub fn add_triple(&mut self, subject: AlgebraTerm, predicate: AlgebraTerm, object: AlgebraTerm) {
        self.triples.push((subject, predicate, object));
    }

    pub fn from_triples(triples: Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>) -> Self {
        Self { triples }
    }
}

impl Dataset for InMemoryDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
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

    fn contains_triple(&self, subject: &AlgebraTerm, predicate: &AlgebraTerm, object: &AlgebraTerm) -> Result<bool> {
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
    fn find_outgoing(&self, subject: &AlgebraTerm, predicate: &AlgebraTerm) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            subject.clone(),
            predicate.clone(),
            AlgebraTerm::Variable("?o".to_string()),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(_, _, o)| o).collect())
    }

    fn find_incoming(&self, predicate: &AlgebraTerm, object: &AlgebraTerm) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            AlgebraTerm::Variable("?s".to_string()),
            predicate.clone(),
            object.clone(),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(s, _, _)| s).collect())
    }

    fn find_predicates(&self, subject: &AlgebraTerm, object: &AlgebraTerm) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            subject.clone(),
            AlgebraTerm::Variable("?p".to_string()),
            object.clone(),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(_, p, _)| p).collect())
    }

    fn get_predicates(&self) -> Result<Vec<AlgebraTerm>> {
        self.dataset.predicates()
    }

    fn contains_triple(&self, subject: &AlgebraTerm, predicate: &AlgebraTerm, object: &AlgebraTerm) -> Result<bool> {
        self.dataset.contains_triple(subject, predicate, object)
    }
}

/// Convert algebra PropertyPath to path module PropertyPath
pub fn convert_property_path(path: &PropertyPath) -> Result<PathPropertyPath> {
    match path {
        PropertyPath::Iri(iri) => Ok(PathPropertyPath::Direct(AlgebraTerm::Iri(iri.clone()))),
        PropertyPath::Variable(var) => Ok(PathPropertyPath::Direct(AlgebraTerm::Variable(var.clone()))),
        PropertyPath::Inverse(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::Inverse(Box::new(inner_path)))
        }
        PropertyPath::Sequence(left, right) => {
            let left_path = convert_property_path(left)?;
            let right_path = convert_property_path(right)?;
            Ok(PathPropertyPath::Sequence(Box::new(left_path), Box::new(right_path)))
        }
        PropertyPath::Alternative(left, right) => {
            let left_path = convert_property_path(left)?;
            let right_path = convert_property_path(right)?;
            Ok(PathPropertyPath::Alternative(Box::new(left_path), Box::new(right_path)))
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
                    _ => return Err(anyhow!("Negated property set can only contain IRIs or variables")),
                }
            }
            Ok(PathPropertyPath::NegatedPropertySet(terms))
        }
    }
}