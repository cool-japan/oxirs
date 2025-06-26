//! RDF graph abstraction and operations

use crate::model::*;
use crate::concurrent::{ParallelBatchProcessor, BatchOperation, BatchConfig};
use crate::{OxirsError, Result};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::sync::Arc;

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

    // Parallel batch processing methods

    /// Insert triples in parallel batches
    ///
    /// This method uses parallel processing to insert a large collection of triples
    /// efficiently. It automatically batches the triples and processes them across
    /// multiple CPU cores.
    pub fn par_insert_batch(&mut self, triples: Vec<Triple>) -> Result<usize> {
        if triples.is_empty() {
            return Ok(0);
        }

        let config = BatchConfig::auto();
        let batch_size = config.batch_size;
        let processor = ParallelBatchProcessor::new(config);
        
        // Split triples into batches
        let operations: Vec<_> = triples
            .par_chunks(batch_size)
            .map(|chunk| BatchOperation::insert(chunk.to_vec()))
            .collect();
        
        // Submit all operations
        processor.submit_batch(operations)?;
        
        // Process operations and collect results
        let all_triples = Arc::new(parking_lot::Mutex::new(Vec::new()));
        
        let all_triples_clone = all_triples.clone();
        processor.process(move |op| -> Result<()> {
            match op {
                BatchOperation::Insert(batch_triples) => {
                    all_triples_clone.lock().extend(batch_triples);
                    Ok(())
                }
                _ => Ok(()),
            }
        })?;
        
        // Now insert all triples into the graph
        let mut inserted = 0;
        for triple in all_triples.lock().drain(..) {
            if self.triples.insert(triple) {
                inserted += 1;
            }
        }
        
        Ok(inserted)
    }

    /// Remove triples in parallel batches
    ///
    /// This method uses parallel processing to remove a large collection of triples
    /// efficiently. It automatically batches the triples and processes them across
    /// multiple CPU cores.
    pub fn par_remove_batch(&mut self, triples: Vec<Triple>) -> Result<usize> {
        if triples.is_empty() {
            return Ok(0);
        }

        let config = BatchConfig::auto();
        let batch_size = config.batch_size;
        let processor = ParallelBatchProcessor::new(config);
        
        // Split triples into batches
        let operations: Vec<_> = triples
            .par_chunks(batch_size)
            .map(|chunk| BatchOperation::remove(chunk.to_vec()))
            .collect();
        
        // Submit all operations
        processor.submit_batch(operations)?;
        
        // Process operations and collect results
        let triples_to_remove = Arc::new(parking_lot::Mutex::new(Vec::new()));
        
        let triples_clone = triples_to_remove.clone();
        processor.process(move |op| -> Result<()> {
            match op {
                BatchOperation::Remove(batch_triples) => {
                    triples_clone.lock().extend(batch_triples);
                    Ok(())
                }
                _ => Ok(()),
            }
        })?;
        
        // Now remove all triples from the graph
        let mut removed = 0;
        for triple in triples_to_remove.lock().drain(..) {
            if self.triples.remove(&triple) {
                removed += 1;
            }
        }
        
        Ok(removed)
    }

    /// Query triples in parallel batches
    ///
    /// This method performs multiple queries in parallel, returning all matching triples.
    /// Each query pattern is processed concurrently for improved performance.
    pub fn par_query_batch(
        &self,
        queries: Vec<(Option<Subject>, Option<Predicate>, Option<Object>)>,
    ) -> Result<Vec<Vec<Triple>>> {
        if queries.is_empty() {
            return Ok(vec![]);
        }

        let config = BatchConfig::auto();
        let processor = ParallelBatchProcessor::new(config);
        
        // Convert queries to operations
        let operations: Vec<_> = queries
            .into_iter()
            .map(|(s, p, o)| BatchOperation::query(s, p, o))
            .collect();
        
        // Submit all operations
        processor.submit_batch(operations)?;
        
        // Clone the triples for processing
        let triples = self.triples.clone();
        
        let results = processor.process(move |op| -> Result<Vec<Triple>> {
            match op {
                BatchOperation::Query { subject, predicate, object } => {
                    let matching: Vec<Triple> = triples
                        .iter()
                        .filter(|triple| {
                            triple.matches_pattern(
                                subject.as_ref(),
                                predicate.as_ref(),
                                object.as_ref(),
                            )
                        })
                        .cloned()
                        .collect();
                    Ok(matching)
                }
                _ => Ok(vec![]),
            }
        })?;
        
        Ok(results)
    }

    /// Apply a transformation function to all triples in parallel
    ///
    /// This method applies a transformation function to each triple in the graph
    /// in parallel. The function can return None to remove a triple or Some(triple)
    /// to replace it.
    pub fn par_transform<F>(&mut self, transform_fn: F) -> Result<(usize, usize)>
    where
        F: Fn(&Triple) -> Option<Triple> + Send + Sync + 'static,
    {
        let triples: Vec<Triple> = self.triples.iter().cloned().collect();
        if triples.is_empty() {
            return Ok((0, 0));
        }

        let transform_fn = Arc::new(transform_fn);
        
        // Process triples in parallel
        let results: Vec<(Option<Triple>, Triple)> = triples
            .par_iter()
            .map(|triple| {
                let result = transform_fn(triple);
                (result, triple.clone())
            })
            .collect();
        
        // Apply transformations
        let mut transformed = 0;
        let mut removed = 0;
        
        for (new_triple, old_triple) in results {
            match new_triple {
                Some(new) if new != old_triple => {
                    self.triples.remove(&old_triple);
                    self.triples.insert(new);
                    transformed += 1;
                }
                None => {
                    self.triples.remove(&old_triple);
                    removed += 1;
                }
                _ => {} // No change
            }
        }
        
        Ok((transformed, removed))
    }

    /// Create a parallel iterator over the graph's triples
    ///
    /// This allows for parallel processing of triples using rayon's parallel iterator traits.
    pub fn par_iter(&self) -> impl ParallelIterator<Item = &Triple> {
        self.triples.par_iter()
    }

    /// Count triples matching patterns in parallel
    ///
    /// This method counts the number of triples matching each pattern in parallel.
    pub fn par_count_patterns(
        &self,
        patterns: Vec<(Option<Subject>, Option<Predicate>, Option<Object>)>,
    ) -> Vec<usize> {
        patterns
            .par_iter()
            .map(|(subject, predicate, object)| {
                self.triples
                    .iter()
                    .filter(|triple| {
                        triple.matches_pattern(
                            subject.as_ref(),
                            predicate.as_ref(),
                            object.as_ref(),
                        )
                    })
                    .count()
            })
            .collect()
    }

    /// Find unique values for a given position in parallel
    ///
    /// This method finds all unique subjects, predicates, or objects in parallel.
    pub fn par_unique_terms(&self) -> (BTreeSet<Subject>, BTreeSet<Predicate>, BTreeSet<Object>) {
        let terms: Vec<(Subject, Predicate, Object)> = self
            .triples
            .par_iter()
            .map(|triple| {
                (
                    triple.subject().clone(),
                    triple.predicate().clone(),
                    triple.object().clone(),
                )
            })
            .collect();
        
        let mut subjects = BTreeSet::new();
        let mut predicates = BTreeSet::new();
        let mut objects = BTreeSet::new();
        
        for (s, p, o) in terms {
            subjects.insert(s);
            predicates.insert(p);
            objects.insert(o);
        }
        
        (subjects, predicates, objects)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn create_test_triple(id: usize) -> Triple {
        Triple::new(
            Subject::NamedNode(NamedNode::new(&format!("http://subject/{}", id)).unwrap()),
            Predicate::NamedNode(NamedNode::new(&format!("http://predicate/{}", id)).unwrap()),
            Object::NamedNode(NamedNode::new(&format!("http://object/{}", id)).unwrap()),
        )
    }

    fn create_test_triples(count: usize) -> Vec<Triple> {
        (0..count).map(create_test_triple).collect()
    }

    #[test]
    fn test_par_insert_batch() {
        let mut graph = Graph::new();
        let triples = create_test_triples(10000);
        
        let start = Instant::now();
        let inserted = graph.par_insert_batch(triples.clone()).unwrap();
        let duration = start.elapsed();
        
        println!("Parallel insert of 10000 triples took: {:?}", duration);
        assert_eq!(inserted, 10000);
        assert_eq!(graph.len(), 10000);
        
        // Verify all triples are present
        for triple in &triples {
            assert!(graph.contains_triple(triple));
        }
    }

    #[test]
    fn test_par_insert_batch_with_duplicates() {
        let mut graph = Graph::new();
        let mut triples = create_test_triples(5000);
        // Add duplicates
        triples.extend(create_test_triples(2500));
        
        let inserted = graph.par_insert_batch(triples).unwrap();
        
        // Should only insert unique triples
        assert_eq!(inserted, 5000);
        assert_eq!(graph.len(), 5000);
    }

    #[test]
    fn test_par_remove_batch() {
        let mut graph = Graph::new();
        let triples = create_test_triples(10000);
        graph.extend(triples.clone());
        
        // Remove half of them
        let to_remove: Vec<Triple> = triples.iter().step_by(2).cloned().collect();
        
        let start = Instant::now();
        let removed = graph.par_remove_batch(to_remove.clone()).unwrap();
        let duration = start.elapsed();
        
        println!("Parallel remove of 5000 triples took: {:?}", duration);
        assert_eq!(removed, 5000);
        assert_eq!(graph.len(), 5000);
        
        // Verify correct triples were removed
        for (i, triple) in triples.iter().enumerate() {
            if i % 2 == 0 {
                assert!(!graph.contains_triple(triple));
            } else {
                assert!(graph.contains_triple(triple));
            }
        }
    }

    #[test]
    fn test_par_query_batch() {
        let mut graph = Graph::new();
        let triples = create_test_triples(1000);
        graph.extend(triples);
        
        // Create multiple query patterns
        let queries: Vec<_> = (0..100)
            .map(|i| {
                (
                    Some(Subject::NamedNode(NamedNode::new(&format!("http://subject/{}", i)).unwrap())),
                    None,
                    None,
                )
            })
            .collect();
        
        let start = Instant::now();
        let results = graph.par_query_batch(queries).unwrap();
        let duration = start.elapsed();
        
        println!("Parallel query of 100 patterns took: {:?}", duration);
        assert_eq!(results.len(), 100);
        
        // Each query should match exactly one triple
        for (i, result) in results.iter().enumerate() {
            if i < 1000 {
                assert_eq!(result.len(), 1);
            } else {
                assert_eq!(result.len(), 0);
            }
        }
    }

    #[test]
    fn test_par_transform() {
        let mut graph = Graph::new();
        let triples = create_test_triples(1000);
        graph.extend(triples);
        
        // Transform function: change predicate for even subjects
        let transform_fn = |triple: &Triple| -> Option<Triple> {
            if let Subject::NamedNode(node) = triple.subject() {
                let uri = node.as_str();
                if let Some(id_str) = uri.strip_prefix("http://subject/") {
                    if let Ok(id) = id_str.parse::<usize>() {
                        if id % 2 == 0 {
                            // Transform: change predicate
                            return Some(Triple::new(
                                triple.subject().clone(),
                                Predicate::NamedNode(NamedNode::new("http://predicate/transformed").unwrap()),
                                triple.object().clone(),
                            ));
                        } else if id % 3 == 0 {
                            // Remove
                            return None;
                        }
                    }
                }
            }
            Some(triple.clone())
        };
        
        let start = Instant::now();
        let (transformed, removed) = graph.par_transform(transform_fn).unwrap();
        let duration = start.elapsed();
        
        println!("Parallel transform took: {:?}", duration);
        println!("Transformed: {}, Removed: {}", transformed, removed);
        
        // Verify transformations
        let transformed_predicate = Predicate::NamedNode(NamedNode::new("http://predicate/transformed").unwrap());
        let transformed_count = graph.query_triples(None, Some(&transformed_predicate), None).len();
        assert!(transformed_count > 0);
    }

    #[test]
    fn test_par_count_patterns() {
        let mut graph = Graph::new();
        
        // Create triples with different patterns
        for i in 0..100 {
            for j in 0..10 {
                let triple = Triple::new(
                    Subject::NamedNode(NamedNode::new(&format!("http://subject/{}", i)).unwrap()),
                    Predicate::NamedNode(NamedNode::new(&format!("http://predicate/{}", j)).unwrap()),
                    Object::NamedNode(NamedNode::new(&format!("http://object/{}", i * 10 + j)).unwrap()),
                );
                graph.add_triple(triple);
            }
        }
        
        // Count patterns
        let patterns: Vec<_> = (0..10)
            .map(|i| {
                (
                    None,
                    Some(Predicate::NamedNode(NamedNode::new(&format!("http://predicate/{}", i)).unwrap())),
                    None,
                )
            })
            .collect();
        
        let counts = graph.par_count_patterns(patterns);
        
        // Each predicate should appear 100 times
        for count in counts {
            assert_eq!(count, 100);
        }
    }

    #[test]
    fn test_par_unique_terms() {
        let mut graph = Graph::new();
        let triples = create_test_triples(1000);
        graph.extend(triples);
        
        let start = Instant::now();
        let (subjects, predicates, objects) = graph.par_unique_terms();
        let duration = start.elapsed();
        
        println!("Parallel unique terms extraction took: {:?}", duration);
        
        assert_eq!(subjects.len(), 1000);
        assert_eq!(predicates.len(), 1000);
        assert_eq!(objects.len(), 1000);
    }

    #[test]
    fn test_par_iter() {
        let mut graph = Graph::new();
        let triples = create_test_triples(1000);
        graph.extend(triples);
        
        // Count triples using parallel iterator
        let count = graph.par_iter().count();
        assert_eq!(count, 1000);
        
        // Filter using parallel iterator
        let filtered: Vec<_> = graph
            .par_iter()
            .filter(|triple| {
                if let Subject::NamedNode(node) = triple.subject() {
                    node.as_str().ends_with("0")
                } else {
                    false
                }
            })
            .cloned()
            .collect();
        
        assert_eq!(filtered.len(), 100);
    }

    #[test]
    fn test_parallel_performance_comparison() {
        let triple_count = 50000;
        let triples = create_test_triples(triple_count);
        
        // Sequential insert
        let mut graph1 = Graph::new();
        let start = Instant::now();
        for triple in &triples {
            graph1.add_triple(triple.clone());
        }
        let seq_duration = start.elapsed();
        
        // Parallel insert
        let mut graph2 = Graph::new();
        let start = Instant::now();
        graph2.par_insert_batch(triples.clone()).unwrap();
        let par_duration = start.elapsed();
        
        println!("Performance comparison for {} triples:", triple_count);
        println!("  Sequential insert: {:?}", seq_duration);
        println!("  Parallel insert: {:?}", par_duration);
        println!("  Speedup: {:.2}x", seq_duration.as_secs_f64() / par_duration.as_secs_f64());
        
        assert_eq!(graph1.len(), graph2.len());
    }

    #[test]
    fn test_empty_operations() {
        let mut graph = Graph::new();
        
        // Test empty insert
        let inserted = graph.par_insert_batch(vec![]).unwrap();
        assert_eq!(inserted, 0);
        
        // Test empty remove
        let removed = graph.par_remove_batch(vec![]).unwrap();
        assert_eq!(removed, 0);
        
        // Test empty query
        let results = graph.par_query_batch(vec![]).unwrap();
        assert!(results.is_empty());
        
        // Test empty transform
        let (transformed, removed) = graph.par_transform(|t| Some(t.clone())).unwrap();
        assert_eq!(transformed, 0);
        assert_eq!(removed, 0);
    }
}
