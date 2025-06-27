//! RDF Graph implementation

use crate::model::{Object, Predicate, Subject, Triple, TripleRef};
use std::collections::HashSet;
use std::iter::FromIterator;

/// An in-memory RDF Graph
///
/// A graph is a set of RDF triples. This implementation uses a HashSet
/// for efficient insertion, removal, and lookup operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    triples: HashSet<Triple>,
}

impl Graph {
    /// Creates a new empty graph
    pub fn new() -> Self {
        Graph {
            triples: HashSet::new(),
        }
    }

    /// Creates a new graph with the specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Graph {
            triples: HashSet::with_capacity(capacity),
        }
    }

    /// Creates a graph from a vector of triples
    pub fn from_triples(triples: Vec<Triple>) -> Self {
        Graph {
            triples: triples.into_iter().collect(),
        }
    }

    /// Inserts a triple into the graph
    ///
    /// Returns `true` if the triple was not already present, `false` otherwise.
    pub fn insert(&mut self, triple: Triple) -> bool {
        self.triples.insert(triple)
    }

    /// Removes a triple from the graph
    ///
    /// Returns `true` if the triple was present, `false` otherwise.
    pub fn remove(&mut self, triple: &Triple) -> bool {
        self.triples.remove(triple)
    }

    /// Returns `true` if the graph contains the specified triple
    pub fn contains(&self, triple: &Triple) -> bool {
        self.triples.contains(triple)
    }

    /// Returns the number of triples in the graph
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Returns `true` if the graph contains no triples
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Clears the graph, removing all triples
    pub fn clear(&mut self) {
        self.triples.clear();
    }

    /// Returns an iterator over all triples in the graph
    pub fn iter(&self) -> impl Iterator<Item = &Triple> {
        self.triples.iter()
    }

    /// Returns an iterator over all triples in the graph as references
    pub fn iter_ref(&self) -> impl Iterator<Item = TripleRef<'_>> {
        self.triples.iter().map(|t| t.into())
    }

    /// Finds all triples matching the given pattern
    ///
    /// `None` values in the pattern act as wildcards.
    pub fn triples_for_pattern<'a>(
        &'a self,
        subject: Option<&'a Subject>,
        predicate: Option<&'a Predicate>,
        object: Option<&'a Object>,
    ) -> impl Iterator<Item = &'a Triple> {
        self.triples.iter().filter(move |triple| {
            if let Some(s) = subject {
                if triple.subject() != s {
                    return false;
                }
            }
            if let Some(p) = predicate {
                if triple.predicate() != p {
                    return false;
                }
            }
            if let Some(o) = object {
                if triple.object() != o {
                    return false;
                }
            }
            true
        })
    }

    /// Finds all triples with the given subject
    pub fn triples_for_subject<'a>(
        &'a self,
        subject: &'a Subject,
    ) -> impl Iterator<Item = &'a Triple> {
        self.triples_for_pattern(Some(subject), None, None)
    }

    /// Finds all triples with the given predicate
    pub fn triples_for_predicate<'a>(
        &'a self,
        predicate: &'a Predicate,
    ) -> impl Iterator<Item = &'a Triple> {
        self.triples_for_pattern(None, Some(predicate), None)
    }

    /// Finds all triples with the given object
    pub fn triples_for_object<'a>(
        &'a self,
        object: &'a Object,
    ) -> impl Iterator<Item = &'a Triple> {
        self.triples_for_pattern(None, None, Some(object))
    }

    /// Finds all triples with the given subject and predicate
    pub fn triples_for_subject_predicate<'a>(
        &'a self,
        subject: &'a Subject,
        predicate: &'a Predicate,
    ) -> impl Iterator<Item = &'a Triple> {
        self.triples_for_pattern(Some(subject), Some(predicate), None)
    }

    /// Extends the graph with triples from an iterator
    pub fn extend<I>(&mut self, triples: I)
    where
        I: IntoIterator<Item = Triple>,
    {
        self.triples.extend(triples);
    }

    /// Retains only the triples specified by the predicate
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&Triple) -> bool,
    {
        self.triples.retain(f);
    }

    /// Creates the union of this graph with another graph
    pub fn union(&self, other: &Graph) -> Graph {
        let mut result = self.clone();
        result.triples.extend(other.triples.iter().cloned());
        result
    }

    /// Creates the intersection of this graph with another graph
    pub fn intersection(&self, other: &Graph) -> Graph {
        Graph {
            triples: self.triples.intersection(&other.triples).cloned().collect(),
        }
    }

    /// Creates the difference of this graph with another graph
    pub fn difference(&self, other: &Graph) -> Graph {
        Graph {
            triples: self.triples.difference(&other.triples).cloned().collect(),
        }
    }

    /// Returns `true` if this graph is a subset of another graph
    pub fn is_subset(&self, other: &Graph) -> bool {
        self.triples.is_subset(&other.triples)
    }

    /// Returns `true` if this graph is a superset of another graph
    pub fn is_superset(&self, other: &Graph) -> bool {
        self.triples.is_superset(&other.triples)
    }

    /// Returns `true` if this graph is disjoint from another graph
    pub fn is_disjoint(&self, other: &Graph) -> bool {
        self.triples.is_disjoint(&other.triples)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<Triple> for Graph {
    fn from_iter<T: IntoIterator<Item = Triple>>(iter: T) -> Self {
        Graph {
            triples: HashSet::from_iter(iter),
        }
    }
}

impl Extend<Triple> for Graph {
    fn extend<T: IntoIterator<Item = Triple>>(&mut self, iter: T) {
        self.triples.extend(iter);
    }
}

impl IntoIterator for Graph {
    type Item = Triple;
    type IntoIter = std::collections::hash_set::IntoIter<Triple>;

    fn into_iter(self) -> Self::IntoIter {
        self.triples.into_iter()
    }
}

impl<'a> IntoIterator for &'a Graph {
    type Item = &'a Triple;
    type IntoIter = std::collections::hash_set::Iter<'a, Triple>;

    fn into_iter(self) -> Self::IntoIter {
        self.triples.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    fn create_test_triple() -> Triple {
        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate = NamedNode::new("http://example.org/predicate").unwrap();
        let object = Literal::new("object");
        Triple::new(subject, predicate, object)
    }

    #[test]
    fn test_graph_basic_operations() {
        let mut graph = Graph::new();
        let triple = create_test_triple();

        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);

        assert!(graph.insert(triple.clone()));
        assert!(!graph.is_empty());
        assert_eq!(graph.len(), 1);
        assert!(graph.contains(&triple));

        assert!(!graph.insert(triple.clone())); // Already exists
        assert_eq!(graph.len(), 1);

        assert!(graph.remove(&triple));
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert!(!graph.contains(&triple));
    }

    #[test]
    fn test_graph_iteration() {
        let mut graph = Graph::new();
        let triple1 = create_test_triple();

        let subject2 = NamedNode::new("http://example.org/subject2").unwrap();
        let predicate2 = NamedNode::new("http://example.org/predicate2").unwrap();
        let object2 = Literal::new("object2");
        let triple2 = Triple::new(subject2, predicate2, object2);

        graph.insert(triple1.clone());
        graph.insert(triple2.clone());

        let mut collected: Vec<_> = graph.iter().cloned().collect();
        collected.sort_by_key(|t| format!("{}", t));

        assert_eq!(collected.len(), 2);
        assert!(collected.contains(&triple1));
        assert!(collected.contains(&triple2));
    }

    #[test]
    fn test_graph_pattern_matching() {
        let mut graph = Graph::new();

        let subject = NamedNode::new("http://example.org/subject").unwrap();
        let predicate1 = NamedNode::new("http://example.org/predicate1").unwrap();
        let predicate2 = NamedNode::new("http://example.org/predicate2").unwrap();
        let object1 = Literal::new("object1");
        let object2 = Literal::new("object2");

        let triple1 = Triple::new(subject.clone(), predicate1.clone(), object1);
        let triple2 = Triple::new(subject.clone(), predicate2, object2);

        graph.insert(triple1.clone());
        graph.insert(triple2.clone());

        // Find by subject
        let by_subject: Vec<_> = graph
            .triples_for_subject(&Subject::NamedNode(subject.clone()))
            .cloned()
            .collect();
        assert_eq!(by_subject.len(), 2);

        // Find by predicate
        let by_predicate: Vec<_> = graph
            .triples_for_predicate(&Predicate::NamedNode(predicate1))
            .cloned()
            .collect();
        assert_eq!(by_predicate.len(), 1);
        assert_eq!(by_predicate[0], triple1);
    }

    #[test]
    fn test_graph_set_operations() {
        let mut graph1 = Graph::new();
        let mut graph2 = Graph::new();

        let triple1 = create_test_triple();
        let subject2 = NamedNode::new("http://example.org/subject2").unwrap();
        let predicate2 = NamedNode::new("http://example.org/predicate2").unwrap();
        let object2 = Literal::new("object2");
        let triple2 = Triple::new(subject2, predicate2, object2);

        graph1.insert(triple1.clone());
        graph2.insert(triple1.clone());
        graph2.insert(triple2.clone());

        let union = graph1.union(&graph2);
        assert_eq!(union.len(), 2);

        let intersection = graph1.intersection(&graph2);
        assert_eq!(intersection.len(), 1);
        assert!(intersection.contains(&triple1));

        assert!(graph1.is_subset(&graph2));
        assert!(!graph1.is_superset(&graph2));
        assert!(graph2.is_superset(&graph1));
    }
}
