//! RDF graph abstraction and operations

use crate::concurrent::{BatchConfig, BatchOperation, ParallelBatchProcessor};
use crate::model::*;
use crate::Result;
#[cfg(feature = "parallel")]
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

    /// Alias for add_triple for compatibility
    pub fn insert(&mut self, triple: Triple) -> bool {
        self.add_triple(triple)
    }

    /// Iterate over all triples (alias for iter_triples)
    pub fn iter(&self) -> impl Iterator<Item = &Triple> {
        self.triples.iter()
    }

    /// Check if a triple exists in the graph (alias for contains_triple)
    pub fn contains(&self, triple: &Triple) -> bool {
        self.contains_triple(triple)
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
    #[cfg(feature = "parallel")]
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
    #[cfg(feature = "parallel")]
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
    #[cfg(feature = "parallel")]
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
                BatchOperation::Query {
                    subject,
                    predicate,
                    object,
                } => {
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
    #[cfg(feature = "parallel")]
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
    #[cfg(feature = "parallel")]
    pub fn par_iter(&self) -> impl ParallelIterator<Item = &Triple> {
        self.triples.par_iter()
    }

    /// Count triples matching patterns in parallel
    ///
    /// This method counts the number of triples matching each pattern in parallel.
    #[cfg(feature = "parallel")]
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
    #[cfg(feature = "parallel")]
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
            Subject::NamedNode(NamedNode::new(format!("http://subject/{id}")).unwrap()),
            Predicate::NamedNode(NamedNode::new(format!("http://predicate/{id}")).unwrap()),
            Object::NamedNode(NamedNode::new(format!("http://object/{id}")).unwrap()),
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

        println!("Parallel insert of 10000 triples took: {duration:?}");
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

        println!("Parallel remove of 5000 triples took: {duration:?}");
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
                    Some(Subject::NamedNode(
                        NamedNode::new(format!("http://subject/{i}")).unwrap(),
                    )),
                    None,
                    None,
                )
            })
            .collect();

        let start = Instant::now();
        let results = graph.par_query_batch(queries).unwrap();
        let duration = start.elapsed();

        println!("Parallel query of 100 patterns took: {duration:?}");
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
                                Predicate::NamedNode(
                                    NamedNode::new("http://predicate/transformed").unwrap(),
                                ),
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

        println!("Parallel transform took: {duration:?}");
        println!("Transformed: {transformed}, Removed: {removed}");

        // Verify transformations
        let transformed_predicate =
            Predicate::NamedNode(NamedNode::new("http://predicate/transformed").unwrap());
        let transformed_count = graph
            .query_triples(None, Some(&transformed_predicate), None)
            .len();
        assert!(transformed_count > 0);
    }

    #[test]
    fn test_par_count_patterns() {
        let mut graph = Graph::new();

        // Create triples with different patterns
        for i in 0..100 {
            for j in 0..10 {
                let triple = Triple::new(
                    Subject::NamedNode(NamedNode::new(format!("http://subject/{i}")).unwrap()),
                    Predicate::NamedNode(NamedNode::new(format!("http://predicate/{j}")).unwrap()),
                    Object::NamedNode(
                        NamedNode::new(format!("http://object/{}", i * 10 + j)).unwrap(),
                    ),
                );
                graph.add_triple(triple);
            }
        }

        // Count patterns
        let patterns: Vec<_> = (0..10)
            .map(|i| {
                (
                    None,
                    Some(Predicate::NamedNode(
                        NamedNode::new(format!("http://predicate/{i}")).unwrap(),
                    )),
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

        println!("Parallel unique terms extraction took: {duration:?}");

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

        println!("Performance comparison for {triple_count} triples:");
        println!("  Sequential insert: {seq_duration:?}");
        println!("  Parallel insert: {par_duration:?}");
        println!(
            "  Speedup: {:.2}x",
            seq_duration.as_secs_f64() / par_duration.as_secs_f64()
        );

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

/// Thread-safe concurrent graph for multi-threaded access
///
/// This struct wraps the Graph in an Arc<RwLock<_>> to provide safe concurrent
/// access across multiple threads. It implements reader-writer semantics where
/// multiple readers can access the graph simultaneously, but only one writer
/// can modify it at a time.
#[derive(Debug, Clone)]
pub struct ConcurrentGraph {
    inner: Arc<parking_lot::RwLock<Graph>>,
}

impl ConcurrentGraph {
    /// Create a new empty concurrent graph
    pub fn new() -> Self {
        Self {
            inner: Arc::new(parking_lot::RwLock::new(Graph::new())),
        }
    }

    /// Create a concurrent graph from an existing graph
    pub fn from_graph(graph: Graph) -> Self {
        Self {
            inner: Arc::new(parking_lot::RwLock::new(graph)),
        }
    }

    /// Add a triple to the graph (thread-safe)
    pub fn add_triple(&self, triple: Triple) -> bool {
        self.inner.write().add_triple(triple)
    }

    /// Add multiple triples atomically
    pub fn add_triples(&self, triples: Vec<Triple>) -> usize {
        let mut graph = self.inner.write();
        let mut added = 0;
        for triple in triples {
            if graph.add_triple(triple) {
                added += 1;
            }
        }
        added
    }

    /// Remove a triple from the graph (thread-safe)
    pub fn remove_triple(&self, triple: &Triple) -> bool {
        self.inner.write().remove_triple(triple)
    }

    /// Check if a triple exists in the graph (thread-safe read)
    pub fn contains_triple(&self, triple: &Triple) -> bool {
        self.inner.read().contains_triple(triple)
    }

    /// Query triples matching the given pattern (thread-safe read)
    pub fn query_triples(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Vec<Triple> {
        self.inner.read().query_triples(subject, predicate, object)
    }

    /// Get the number of triples (thread-safe read)
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Check if the graph is empty (thread-safe read)
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Get all triples as a vector (thread-safe read)
    pub fn triples(&self) -> Vec<Triple> {
        self.inner.read().triples()
    }

    /// Merge another graph into this one (thread-safe)
    pub fn merge(&self, other: &Graph) {
        self.inner.write().merge(other)
    }

    /// Merge another concurrent graph into this one (thread-safe)
    pub fn merge_concurrent(&self, other: &ConcurrentGraph) {
        let other_triples = other.triples();
        let mut graph = self.inner.write();
        for triple in other_triples {
            graph.add_triple(triple);
        }
    }

    /// Create a union with another graph (thread-safe read)
    pub fn union(&self, other: &Graph) -> Graph {
        self.inner.read().union(other)
    }

    /// Create an intersection with another graph (thread-safe read)
    pub fn intersection(&self, other: &Graph) -> Graph {
        self.inner.read().intersection(other)
    }

    /// Clear all triples from the graph (thread-safe)
    pub fn clear(&self) {
        self.inner.write().clear()
    }

    /// Execute a read operation with access to the underlying graph
    pub fn with_read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Graph) -> R,
    {
        let graph = self.inner.read();
        f(&graph)
    }

    /// Execute a write operation with access to the underlying graph
    pub fn with_write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Graph) -> R,
    {
        let mut graph = self.inner.write();
        f(&mut graph)
    }

    /// Parallel batch insert (thread-safe)
    #[cfg(feature = "parallel")]
    pub fn par_insert_batch(&self, triples: Vec<Triple>) -> Result<usize> {
        self.inner.write().par_insert_batch(triples)
    }

    /// Parallel batch remove (thread-safe)
    #[cfg(feature = "parallel")]
    pub fn par_remove_batch(&self, triples: Vec<Triple>) -> Result<usize> {
        self.inner.write().par_remove_batch(triples)
    }

    /// Parallel batch query (thread-safe read)
    #[cfg(feature = "parallel")]
    pub fn par_query_batch(
        &self,
        queries: Vec<(Option<Subject>, Option<Predicate>, Option<Object>)>,
    ) -> Result<Vec<Vec<Triple>>> {
        self.inner.read().par_query_batch(queries)
    }

    /// Get subjects concurrently
    pub fn subjects(&self) -> BTreeSet<Subject> {
        self.inner.read().subjects()
    }

    /// Get predicates concurrently
    pub fn predicates(&self) -> BTreeSet<Predicate> {
        self.inner.read().predicates()
    }

    /// Get objects concurrently
    pub fn objects(&self) -> BTreeSet<Object> {
        self.inner.read().objects()
    }
}

impl Default for ConcurrentGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread pool for concurrent graph operations
#[allow(dead_code)]
pub struct GraphThreadPool {
    #[cfg(feature = "parallel")]
    pool: rayon::ThreadPool,
    max_batch_size: usize,
}

impl GraphThreadPool {
    /// Create a new graph thread pool
    pub fn new() -> Result<Self> {
        #[cfg(feature = "parallel")]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get())
                .thread_name(|index| format!("oxirs-graph-{index}"))
                .build()
                .map_err(|e| crate::OxirsError::ConcurrencyError(e.to_string()))?;

            Ok(Self {
                pool,
                max_batch_size: 10_000,
            })
        }
        #[cfg(not(feature = "parallel"))]
        {
            Ok(Self {
                max_batch_size: 10_000,
            })
        }
    }

    /// Create a thread pool with custom configuration
    pub fn with_config(num_threads: usize, max_batch_size: usize) -> Result<Self> {
        #[cfg(feature = "parallel")]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .thread_name(|index| format!("oxirs-graph-{index}"))
                .build()
                .map_err(|e| crate::OxirsError::ConcurrencyError(e.to_string()))?;

            Ok(Self {
                pool,
                max_batch_size,
            })
        }
        #[cfg(not(feature = "parallel"))]
        {
            Ok(Self { max_batch_size })
        }
    }

    /// Process triples concurrently
    pub fn process_triples<F, R>(&self, triples: Vec<Triple>, processor: F) -> Vec<R>
    where
        F: Fn(Triple) -> R + Sync + Send,
        R: Send,
    {
        #[cfg(feature = "parallel")]
        {
            self.pool
                .install(|| triples.into_par_iter().map(processor).collect())
        }
        #[cfg(not(feature = "parallel"))]
        {
            triples.into_iter().map(processor).collect()
        }
    }

    /// Process graph operations concurrently
    pub fn process_graphs<F, R>(&self, graphs: Vec<Graph>, processor: F) -> Vec<R>
    where
        F: Fn(Graph) -> R + Sync + Send,
        R: Send,
    {
        #[cfg(feature = "parallel")]
        {
            self.pool
                .install(|| graphs.into_par_iter().map(processor).collect())
        }
        #[cfg(not(feature = "parallel"))]
        {
            graphs.into_iter().map(processor).collect()
        }
    }

    /// Parallel merge multiple graphs
    pub fn merge_graphs(&self, graphs: Vec<Graph>) -> Graph {
        if graphs.is_empty() {
            return Graph::new();
        }

        #[cfg(feature = "parallel")]
        {
            self.pool.install(|| {
                graphs.into_par_iter().reduce(Graph::new, |mut acc, graph| {
                    acc.merge(&graph);
                    acc
                })
            })
        }
        #[cfg(not(feature = "parallel"))]
        {
            graphs.into_iter().fold(Graph::new(), |mut acc, graph| {
                acc.merge(&graph);
                acc
            })
        }
    }

    /// Get the underlying thread pool (only available with parallel feature)
    #[cfg(feature = "parallel")]
    pub fn inner(&self) -> &rayon::ThreadPool {
        &self.pool
    }
}

impl Default for GraphThreadPool {
    fn default() -> Self {
        Self::new().expect("Failed to create default thread pool")
    }
}

#[cfg(test)]
mod concurrent_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_concurrent_graph_basic_operations() {
        let graph = ConcurrentGraph::new();

        let triple = Triple::new(
            NamedNode::new("http://example.org/s").unwrap(),
            NamedNode::new("http://example.org/p").unwrap(),
            Literal::new("test"),
        );

        // Test basic operations
        assert!(graph.add_triple(triple.clone()));
        assert!(graph.contains_triple(&triple));
        assert_eq!(graph.len(), 1);
        assert!(!graph.is_empty());

        // Test removal
        assert!(graph.remove_triple(&triple));
        assert!(!graph.contains_triple(&triple));
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_concurrent_access() {
        let graph = ConcurrentGraph::new();

        let counter = Arc::new(AtomicUsize::new(0));

        // Spawn multiple reader threads
        let mut handles = vec![];

        for i in 0..10 {
            let g = graph.clone();
            let c = counter.clone();

            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let triple = Triple::new(
                        NamedNode::new(format!("http://example.org/s{}", i * 100 + j)).unwrap(),
                        NamedNode::new("http://example.org/p").unwrap(),
                        Literal::new(format!("value{j}")),
                    );

                    if g.add_triple(triple) {
                        c.fetch_add(1, Ordering::Relaxed);
                    }

                    // Small delay to encourage interleaving
                    thread::sleep(Duration::from_nanos(1));
                }
            }));
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify results
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
        assert_eq!(graph.len(), 1000);
    }

    #[test]
    fn test_concurrent_graph_merge() {
        let graph1 = ConcurrentGraph::new();
        let graph2 = ConcurrentGraph::new();

        // Add different triples to each graph
        for i in 0..100 {
            let triple1 = Triple::new(
                NamedNode::new(format!("http://example.org/s1_{i}")).unwrap(),
                NamedNode::new("http://example.org/p").unwrap(),
                Literal::new(format!("value{i}")),
            );
            graph1.add_triple(triple1);

            let triple2 = Triple::new(
                NamedNode::new(format!("http://example.org/s2_{i}")).unwrap(),
                NamedNode::new("http://example.org/p").unwrap(),
                Literal::new(format!("value{i}")),
            );
            graph2.add_triple(triple2);
        }

        // Merge graphs
        graph1.merge_concurrent(&graph2);

        assert_eq!(graph1.len(), 200);
        assert_eq!(graph2.len(), 100);
    }

    #[test]
    fn test_graph_thread_pool() {
        let pool = GraphThreadPool::new().unwrap();

        // Create test triples
        let triples: Vec<Triple> = (0..1000)
            .map(|i| {
                Triple::new(
                    NamedNode::new(format!("http://example.org/s{i}")).unwrap(),
                    NamedNode::new("http://example.org/p").unwrap(),
                    Literal::new(format!("value{i}")),
                )
            })
            .collect();

        // Process triples concurrently
        let results = pool.process_triples(triples.clone(), |triple| {
            // Simulate some processing
            triple.to_string().len()
        });

        assert_eq!(results.len(), 1000);
        assert!(results.iter().all(|&len| len > 0));
    }

    #[test]
    fn test_concurrent_with_operations() {
        let graph = ConcurrentGraph::new();

        // Test with_read
        let initial_len = graph.with_read(|g| g.len());
        assert_eq!(initial_len, 0);

        // Test with_write
        graph.with_write(|g| {
            for i in 0..10 {
                let triple = Triple::new(
                    NamedNode::new(format!("http://example.org/s{i}")).unwrap(),
                    NamedNode::new("http://example.org/p").unwrap(),
                    Literal::new(format!("value{i}")),
                );
                g.add_triple(triple);
            }
        });

        let final_len = graph.with_read(|g| g.len());
        assert_eq!(final_len, 10);
    }
}
