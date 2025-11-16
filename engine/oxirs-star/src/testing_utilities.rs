//! Testing utilities for RDF-star applications
//!
//! This module provides comprehensive testing utilities including graph generators,
//! test data builders, assertion helpers, mocking, and property-based testing support.

use crate::annotations::TripleAnnotation;
use crate::model::{StarGraph, StarTerm, StarTriple};
use scirs2_core::random::{Rng, SeedableRng, StdRng};
use thiserror::Error;

/// Testing errors
#[derive(Error, Debug)]
pub enum TestingError {
    #[error("Test data generation failed: {0}")]
    GenerationFailed(String),

    #[error("Assertion failed: {0}")]
    AssertionFailed(String),

    #[error("Invalid test configuration: {0}")]
    InvalidConfig(String),
}

/// Test graph builder for creating test data
pub struct TestGraphBuilder {
    graph: StarGraph,
    rng: StdRng,
    counter: usize,
}

impl TestGraphBuilder {
    /// Create a new test graph builder
    pub fn new() -> Self {
        // Use a non-deterministic seed based on system time
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self {
            graph: StarGraph::new(),
            rng: StdRng::seed_from_u64(seed),
            counter: 0,
        }
    }

    /// Add a simple triple
    pub fn add_triple(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<&mut Self, TestingError> {
        let triple = StarTriple::new(
            StarTerm::iri(subject).map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::iri(predicate).map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::literal(object).map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
        );

        let _ = self.graph.insert(triple);
        Ok(self)
    }

    /// Add a quoted triple
    pub fn add_quoted_triple(
        &mut self,
        inner_subject: &str,
        inner_predicate: &str,
        inner_object: &str,
        meta_predicate: &str,
        meta_object: &str,
    ) -> Result<&mut Self, TestingError> {
        let inner = StarTriple::new(
            StarTerm::iri(inner_subject)
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::iri(inner_predicate)
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::literal(inner_object)
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
        );

        let meta = StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri(meta_predicate)
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::literal(meta_object)
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
        );

        let _ = self.graph.insert(meta);
        Ok(self)
    }

    /// Generate random triples
    pub fn generate_random_triples(&mut self, count: usize) -> &mut Self {
        for _ in 0..count {
            let subject = format!("http://example.org/s{}", self.counter);
            let predicate = format!("http://example.org/p{}", self.rng.gen_range(0..10));
            let object = format!("object_{}", self.rng.random::<u32>());

            if let (Ok(s), Ok(p), Ok(o)) = (
                StarTerm::iri(&subject),
                StarTerm::iri(&predicate),
                StarTerm::literal(&object),
            ) {
                let _ = self.graph.insert(StarTriple::new(s, p, o));
            }

            self.counter += 1;
        }

        self
    }

    /// Generate nested quoted triples
    pub fn generate_nested_triples(&mut self, depth: usize) -> Result<&mut Self, TestingError> {
        let base = StarTriple::new(
            StarTerm::iri(&format!("http://example.org/s{}", self.counter))
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::iri("http://example.org/base_pred")
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::literal("base_object")
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
        );

        let mut current = base;

        for level in 0..depth {
            current = StarTriple::new(
                StarTerm::quoted_triple(current),
                StarTerm::iri(&format!("http://example.org/meta{}", level))
                    .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
                StarTerm::literal(&format!("meta_object_{}", level))
                    .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            );
        }

        let _ = self.graph.insert(current);
        self.counter += 1;

        Ok(self)
    }

    /// Build the graph
    pub fn build(self) -> StarGraph {
        self.graph
    }

    /// Get mutable reference to graph
    pub fn graph_mut(&mut self) -> &mut StarGraph {
        &mut self.graph
    }
}

impl Default for TestGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Assertion helpers for RDF-star testing
pub struct RdfStarAssertions;

impl RdfStarAssertions {
    /// Assert graph contains triple
    pub fn assert_contains_triple(
        graph: &StarGraph,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<(), TestingError> {
        for triple in graph.iter() {
            if let (Some(s_nn), Some(p_nn)) = (
                triple.subject.as_named_node(),
                triple.predicate.as_named_node(),
            ) {
                if s_nn.iri == subject && p_nn.iri == predicate {
                    if let Some(o_lit) = triple.object.as_literal() {
                        if o_lit.value == object {
                            return Ok(());
                        }
                    } else if let Some(o_nn) = triple.object.as_named_node() {
                        if o_nn.iri == object {
                            return Ok(());
                        }
                    }
                }
            }
        }

        Err(TestingError::AssertionFailed(format!(
            "Triple not found: <{}> <{}> \"{}\"",
            subject, predicate, object
        )))
    }

    /// Assert graph size
    pub fn assert_size(graph: &StarGraph, expected: usize) -> Result<(), TestingError> {
        if graph.len() == expected {
            Ok(())
        } else {
            Err(TestingError::AssertionFailed(format!(
                "Expected {} triples, found {}",
                expected,
                graph.len()
            )))
        }
    }

    /// Assert graph is empty
    pub fn assert_empty(graph: &StarGraph) -> Result<(), TestingError> {
        Self::assert_size(graph, 0)
    }

    /// Assert graph is not empty
    pub fn assert_not_empty(graph: &StarGraph) -> Result<(), TestingError> {
        if graph.is_empty() {
            Err(TestingError::AssertionFailed("Graph is empty".to_string()))
        } else {
            Ok(())
        }
    }

    /// Assert graph contains quoted triple
    pub fn assert_has_quoted_triples(graph: &StarGraph) -> Result<(), TestingError> {
        for triple in graph.iter() {
            if matches!(triple.subject, StarTerm::QuotedTriple(_))
                || matches!(triple.object, StarTerm::QuotedTriple(_))
            {
                return Ok(());
            }
        }

        Err(TestingError::AssertionFailed(
            "No quoted triples found in graph".to_string(),
        ))
    }

    /// Assert maximum nesting depth
    pub fn assert_max_depth(graph: &StarGraph, max_depth: usize) -> Result<(), TestingError> {
        for triple in graph.iter() {
            let depth = Self::get_triple_depth(triple);
            if depth > max_depth {
                return Err(TestingError::AssertionFailed(format!(
                    "Triple exceeds maximum depth {}: found depth {}",
                    max_depth, depth
                )));
            }
        }

        Ok(())
    }

    /// Get nesting depth of a triple
    fn get_triple_depth(triple: &StarTriple) -> usize {
        let subject_depth = Self::get_term_depth(&triple.subject);
        let object_depth = Self::get_term_depth(&triple.object);
        subject_depth.max(object_depth)
    }

    /// Get nesting depth of a term
    fn get_term_depth(term: &StarTerm) -> usize {
        match term {
            StarTerm::QuotedTriple(qt) => 1 + Self::get_triple_depth(qt),
            _ => 0,
        }
    }

    /// Assert graphs are equal
    pub fn assert_graphs_equal(graph1: &StarGraph, graph2: &StarGraph) -> Result<(), TestingError> {
        if graph1.len() != graph2.len() {
            return Err(TestingError::AssertionFailed(format!(
                "Graphs have different sizes: {} vs {}",
                graph1.len(),
                graph2.len()
            )));
        }

        // Compare triples (simple comparison)
        for triple in graph1.iter() {
            if !graph2.contains(triple) {
                return Err(TestingError::AssertionFailed(format!(
                    "Triple not found in second graph: {:?}",
                    triple
                )));
            }
        }

        Ok(())
    }
}

/// Test data generator for property-based testing
pub struct PropertyTestGenerator {
    rng: StdRng,
}

impl PropertyTestGenerator {
    /// Create a new generator
    pub fn new() -> Self {
        // Use a non-deterministic seed based on system time
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generate random IRI
    pub fn gen_iri(&mut self) -> String {
        format!("http://example.org/resource{}", self.rng.random::<u32>())
    }

    /// Generate random literal
    pub fn gen_literal(&mut self) -> String {
        format!("literal_{}", self.rng.random::<u32>())
    }

    /// Generate random triple
    pub fn gen_triple(&mut self) -> Result<StarTriple, TestingError> {
        Ok(StarTriple::new(
            StarTerm::iri(&self.gen_iri())
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::iri(&self.gen_iri())
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::literal(&self.gen_literal())
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
        ))
    }

    /// Generate random graph
    pub fn gen_graph(
        &mut self,
        min_size: usize,
        max_size: usize,
    ) -> Result<StarGraph, TestingError> {
        let size = self.rng.gen_range(min_size..=max_size);
        let mut graph = StarGraph::new();

        for _ in 0..size {
            let _ = graph.insert(self.gen_triple()?);
        }

        Ok(graph)
    }

    /// Generate random quoted triple
    pub fn gen_quoted_triple(&mut self) -> Result<StarTriple, TestingError> {
        let inner = self.gen_triple()?;

        Ok(StarTriple::new(
            StarTerm::quoted_triple(inner),
            StarTerm::iri(&self.gen_iri())
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
            StarTerm::literal(&self.gen_literal())
                .map_err(|e| TestingError::GenerationFailed(e.to_string()))?,
        ))
    }
}

impl Default for PropertyTestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock annotation builder for testing
pub struct MockAnnotationBuilder {
    annotation: TripleAnnotation,
}

impl MockAnnotationBuilder {
    /// Create a new mock annotation
    pub fn new() -> Self {
        Self {
            annotation: TripleAnnotation::new(),
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.annotation.confidence = Some(confidence);
        self
    }

    /// Set source
    pub fn with_source(mut self, source: &str) -> Self {
        self.annotation.source = Some(source.to_string());
        self
    }

    /// Set quality score
    pub fn with_quality_score(mut self, score: f64) -> Self {
        self.annotation.quality_score = Some(score);
        self
    }

    /// Build the annotation
    pub fn build(self) -> TripleAnnotation {
        self.annotation
    }
}

impl Default for MockAnnotationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Test fixture manager
pub struct TestFixture {
    /// Name of the fixture
    pub name: String,

    /// Test graph
    pub graph: StarGraph,

    /// Test annotations
    pub annotations: Vec<TripleAnnotation>,

    /// Expected results
    pub expected_results: Vec<String>,
}

impl TestFixture {
    /// Create a new test fixture
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            graph: StarGraph::new(),
            annotations: Vec::new(),
            expected_results: Vec::new(),
        }
    }

    /// Load fixture from builder
    pub fn from_builder(name: &str, builder: TestGraphBuilder) -> Self {
        Self {
            name: name.to_string(),
            graph: builder.build(),
            annotations: Vec::new(),
            expected_results: Vec::new(),
        }
    }

    /// Add expected result
    pub fn expect(&mut self, result: String) {
        self.expected_results.push(result);
    }

    /// Verify expected results
    pub fn verify(&self, actual_results: &[String]) -> Result<(), TestingError> {
        if self.expected_results.len() != actual_results.len() {
            return Err(TestingError::AssertionFailed(format!(
                "Expected {} results, got {}",
                self.expected_results.len(),
                actual_results.len()
            )));
        }

        for (expected, actual) in self.expected_results.iter().zip(actual_results.iter()) {
            if expected != actual {
                return Err(TestingError::AssertionFailed(format!(
                    "Result mismatch: expected '{}', got '{}'",
                    expected, actual
                )));
            }
        }

        Ok(())
    }
}

/// Benchmark helper for performance testing
pub struct BenchmarkHelper {
    operations: Vec<BenchmarkOperation>,
}

#[derive(Clone)]
pub struct BenchmarkOperation {
    pub name: String,
    pub duration_ns: u128,
    pub operations_count: usize,
}

impl BenchmarkHelper {
    /// Create a new benchmark helper
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Time an operation
    pub fn time_operation<F, R>(&mut self, name: &str, operations_count: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed();

        self.operations.push(BenchmarkOperation {
            name: name.to_string(),
            duration_ns: duration.as_nanos(),
            operations_count,
        });

        result
    }

    /// Get operations per second
    pub fn ops_per_second(&self, operation_name: &str) -> Option<f64> {
        for op in &self.operations {
            if op.name == operation_name {
                let duration_s = op.duration_ns as f64 / 1_000_000_000.0;
                return Some(op.operations_count as f64 / duration_s);
            }
        }
        None
    }

    /// Print benchmark results
    pub fn print_results(&self) {
        println!("\n=== Benchmark Results ===");
        for op in &self.operations {
            let duration_ms = op.duration_ns as f64 / 1_000_000.0;
            let ops_per_sec =
                (op.operations_count as f64 / (op.duration_ns as f64 / 1_000_000_000.0)) as u64;

            println!(
                "{}: {:.2}ms ({} ops, {} ops/sec)",
                op.name, duration_ms, op.operations_count, ops_per_sec
            );
        }
    }
}

impl Default for BenchmarkHelper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_builder() -> Result<(), TestingError> {
        let mut builder = TestGraphBuilder::new();
        builder.add_triple("http://example.org/s", "http://example.org/p", "object")?;
        builder.generate_random_triples(5);
        let graph = builder.build();

        assert!(graph.len() >= 6);
        Ok(())
    }

    #[test]
    fn test_assertions() -> Result<(), TestingError> {
        let mut builder = TestGraphBuilder::new();
        builder.add_triple("http://example.org/alice", "http://example.org/age", "30")?;

        let graph = builder.build();

        RdfStarAssertions::assert_size(&graph, 1)?;
        RdfStarAssertions::assert_not_empty(&graph)?;
        RdfStarAssertions::assert_contains_triple(
            &graph,
            "http://example.org/alice",
            "http://example.org/age",
            "30",
        )?;

        Ok(())
    }

    #[test]
    fn test_quoted_triple_assertion() -> Result<(), TestingError> {
        let mut builder = TestGraphBuilder::new();
        builder.add_quoted_triple(
            "http://example.org/alice",
            "http://example.org/age",
            "30",
            "http://example.org/certainty",
            "0.9",
        )?;

        let graph = builder.build();

        RdfStarAssertions::assert_has_quoted_triples(&graph)?;

        Ok(())
    }

    #[test]
    fn test_property_generator() -> Result<(), TestingError> {
        let mut gen = PropertyTestGenerator::new();

        let iri = gen.gen_iri();
        assert!(iri.starts_with("http://"));

        let triple = gen.gen_triple()?;
        assert!(triple.subject.as_named_node().is_some());

        let graph = gen.gen_graph(5, 10)?;
        assert!(graph.len() >= 5 && graph.len() <= 10);

        Ok(())
    }

    #[test]
    fn test_mock_annotation() {
        let annotation = MockAnnotationBuilder::new()
            .with_confidence(0.9)
            .with_source("test source")
            .with_quality_score(0.8)
            .build();

        assert_eq!(annotation.confidence, Some(0.9));
        assert_eq!(annotation.source, Some("test source".to_string()));
        assert_eq!(annotation.quality_score, Some(0.8));
    }

    #[test]
    fn test_benchmark_helper() {
        let mut bench = BenchmarkHelper::new();

        bench.time_operation("test_op", 1000, || {
            // Simulate work
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(i);
            }
            sum
        });

        assert!(bench.ops_per_second("test_op").is_some());
    }

    #[test]
    fn test_nested_triple_generation() -> Result<(), TestingError> {
        let mut builder = TestGraphBuilder::new();
        builder.generate_nested_triples(3)?;

        let graph = builder.build();

        RdfStarAssertions::assert_not_empty(&graph)?;
        RdfStarAssertions::assert_has_quoted_triples(&graph)?;

        Ok(())
    }
}
