//! RDF graph abstraction and operations

use crate::model::*;
use crate::{OxirsError, Result};
use std::collections::BTreeSet;

/// RDF graph representation
///
/// A graph is a collection of RDF triples. This implementation uses a BTreeSet
/// for efficient storage and retrieval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    triples: BTreeSet<Triple>,
}

impl Graph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Graph {
            triples: BTreeSet::new(),
        }
    }

    /// Create a graph from an iterator of triples
    pub fn from_triples<I>(triples: I) -> Self
    where
        I: IntoIterator<Item = Triple>,
    {
        Graph {
            triples: triples.into_iter().collect(),
        }
    }

    /// Add a triple to the graph
    pub fn add_triple(&mut self, triple: Triple) -> bool {
        self.triples.insert(triple)
    }

    /// Add a triple to the graph using string components
    pub fn add_triple_str(&mut self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        let subject_node = NamedNode::new(subject)?;
        let predicate_node = NamedNode::new(predicate)?;
        let object_literal = Literal::new(object);

        let triple = Triple::new(subject_node, predicate_node, object_literal);
        Ok(self.add_triple(triple))
    }

    /// Remove a triple from the graph
    pub fn remove_triple(&mut self, triple: &Triple) -> bool {
        self.triples.remove(triple)
    }

    /// Check if a triple exists in the graph
    pub fn contains_triple(&self, triple: &Triple) -> bool {
        self.triples.contains(triple)
    }

    /// Query triples matching the given pattern
    ///
    /// None values act as wildcards matching any term.
    pub fn query_triples(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Vec<Triple> {
        self.triples
            .iter()
            .filter(|triple| triple.matches_pattern(subject, predicate, object))
            .cloned()
            .collect()
    }

    /// Get all triples as a vector
    pub fn triples(&self) -> Vec<Triple> {
        self.triples.iter().cloned().collect()
    }

    /// Iterate over all triples
    pub fn iter_triples(&self) -> impl Iterator<Item = &Triple> {
        self.triples.iter()
    }

    /// Get all subjects in the graph
    pub fn subjects(&self) -> BTreeSet<Subject> {
        self.triples.iter().map(|t| t.subject().clone()).collect()
    }

    /// Get all predicates in the graph
    pub fn predicates(&self) -> BTreeSet<Predicate> {
        self.triples.iter().map(|t| t.predicate().clone()).collect()
    }

    /// Get all objects in the graph
    pub fn objects(&self) -> BTreeSet<Object> {
        self.triples.iter().map(|t| t.object().clone()).collect()
    }

    /// Merge another graph into this one
    pub fn merge(&mut self, other: &Graph) {
        for triple in &other.triples {
            self.triples.insert(triple.clone());
        }
    }

    /// Create a new graph containing the union of this graph and another
    pub fn union(&self, other: &Graph) -> Graph {
        let mut result = self.clone();
        result.merge(other);
        result
    }

    /// Create a new graph containing the intersection of this graph and another
    pub fn intersection(&self, other: &Graph) -> Graph {
        let intersection_triples: BTreeSet<Triple> =
            self.triples.intersection(&other.triples).cloned().collect();

        Graph {
            triples: intersection_triples,
        }
    }

    /// Create a new graph containing triples in this graph but not in the other
    pub fn difference(&self, other: &Graph) -> Graph {
        let difference_triples: BTreeSet<Triple> =
            self.triples.difference(&other.triples).cloned().collect();

        Graph {
            triples: difference_triples,
        }
    }

    /// Clear all triples from the graph
    pub fn clear(&mut self) {
        self.triples.clear();
    }

    /// Get the number of triples in the graph
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Check if the graph is isomorphic to another graph
    ///
    /// This is a simplified check that doesn't handle blank node isomorphism properly.
    /// For proper blank node isomorphism, a more sophisticated algorithm would be needed.
    pub fn is_isomorphic_to(&self, other: &Graph) -> bool {
        // Simple implementation: check if both graphs have the same triples
        // This doesn't handle blank node renaming properly
        self.triples == other.triples
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over triples in a graph
pub struct GraphIter<'a> {
    inner: std::collections::btree_set::Iter<'a, Triple>,
}

impl<'a> Iterator for GraphIter<'a> {
    type Item = &'a Triple;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a> IntoIterator for &'a Graph {
    type Item = &'a Triple;
    type IntoIter = GraphIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        GraphIter {
            inner: self.triples.iter(),
        }
    }
}

impl IntoIterator for Graph {
    type Item = Triple;
    type IntoIter = std::collections::btree_set::IntoIter<Triple>;

    fn into_iter(self) -> Self::IntoIter {
        self.triples.into_iter()
    }
}

impl FromIterator<Triple> for Graph {
    fn from_iter<I: IntoIterator<Item = Triple>>(iter: I) -> Self {
        Graph {
            triples: iter.into_iter().collect(),
        }
    }
}

impl Extend<Triple> for Graph {
    fn extend<I: IntoIterator<Item = Triple>>(&mut self, iter: I) {
        self.triples.extend(iter);
    }
}
