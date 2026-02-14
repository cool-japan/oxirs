//! RDF Graph Utilities
//!
//! This module provides utilities for working with RDF graphs, including:
//! - Graph merging and combining
//! - Graph comparison and diff generation
//! - Graph transformation operations
//! - Graph statistics and analysis
//!
//! # Examples
//!
//! ## Merging Graphs
//!
//! ```rust
//! use oxirs_ttl::toolkit::graph_utils::GraphMerger;
//! use oxirs_core::model::{Triple, NamedNode};
//!
//! let graph1 = vec![
//!     Triple::new(
//!         NamedNode::new("http://example.org/s1")?,
//!         NamedNode::new("http://example.org/p")?,
//!         NamedNode::new("http://example.org/o1")?,
//!     ),
//! ];
//!
//! let graph2 = vec![
//!     Triple::new(
//!         NamedNode::new("http://example.org/s2")?,
//!         NamedNode::new("http://example.org/p")?,
//!         NamedNode::new("http://example.org/o2")?,
//!     ),
//! ];
//!
//! let merger = GraphMerger::new();
//! let merged = merger.merge(&[graph1, graph2]);
//! assert_eq!(merged.len(), 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Finding Graph Differences
//!
//! ```rust
//! use oxirs_ttl::toolkit::graph_utils::GraphDiff;
//! use oxirs_core::model::{Triple, NamedNode};
//!
//! let graph1 = vec![
//!     Triple::new(
//!         NamedNode::new("http://example.org/s")?,
//!         NamedNode::new("http://example.org/p")?,
//!         NamedNode::new("http://example.org/o1")?,
//!     ),
//! ];
//!
//! let graph2 = vec![
//!     Triple::new(
//!         NamedNode::new("http://example.org/s")?,
//!         NamedNode::new("http://example.org/p")?,
//!         NamedNode::new("http://example.org/o2")?,
//!     ),
//! ];
//!
//! let diff = GraphDiff::compute(&graph1, &graph2);
//! assert_eq!(diff.added().len(), 1);
//! assert_eq!(diff.removed().len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use oxirs_core::model::{Quad, Triple};
use oxirs_core::RdfTerm;
use std::collections::{HashMap, HashSet};

/// Graph merger for combining multiple RDF graphs
///
/// Provides efficient merging of RDF graphs with deduplication.
#[derive(Debug, Default)]
pub struct GraphMerger {
    deduplicate: bool,
}

impl GraphMerger {
    /// Create a new graph merger with default settings
    pub fn new() -> Self {
        Self { deduplicate: true }
    }

    /// Create a merger that allows duplicate triples
    pub fn with_duplicates() -> Self {
        Self { deduplicate: false }
    }

    /// Merge multiple graphs into a single graph
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_ttl::toolkit::graph_utils::GraphMerger;
    ///
    /// let graphs: Vec<Vec<oxirs_core::model::Triple>> = vec![];
    /// let merger = GraphMerger::new();
    /// let merged = merger.merge(&graphs);
    /// ```
    pub fn merge(&self, graphs: &[Vec<Triple>]) -> Vec<Triple> {
        if !self.deduplicate {
            return graphs.iter().flat_map(|g| g.iter().cloned()).collect();
        }

        let mut seen = HashSet::new();
        let mut result = Vec::new();

        for graph in graphs {
            for triple in graph {
                if seen.insert(triple.clone()) {
                    result.push(triple.clone());
                }
            }
        }

        result
    }

    /// Merge multiple quad graphs
    pub fn merge_quads(&self, graphs: &[Vec<Quad>]) -> Vec<Quad> {
        if !self.deduplicate {
            return graphs.iter().flat_map(|g| g.iter().cloned()).collect();
        }

        let mut seen = HashSet::new();
        let mut result = Vec::new();

        for graph in graphs {
            for quad in graph {
                if seen.insert(quad.clone()) {
                    result.push(quad.clone());
                }
            }
        }

        result
    }

    /// Merge two graphs in-place (modifying the first graph)
    pub fn merge_into(&self, target: &mut Vec<Triple>, source: &[Triple]) {
        if !self.deduplicate {
            target.extend_from_slice(source);
            return;
        }

        let existing: HashSet<_> = target.iter().cloned().collect();
        for triple in source {
            if !existing.contains(triple) {
                target.push(triple.clone());
            }
        }
    }
}

/// Represents differences between two RDF graphs
#[derive(Debug, Clone)]
pub struct GraphDiff {
    added: Vec<Triple>,
    removed: Vec<Triple>,
    common: Vec<Triple>,
}

impl GraphDiff {
    /// Compute the difference between two graphs
    ///
    /// Returns a diff showing triples added, removed, and common to both graphs.
    pub fn compute(graph1: &[Triple], graph2: &[Triple]) -> Self {
        let set1: HashSet<_> = graph1.iter().cloned().collect();
        let set2: HashSet<_> = graph2.iter().cloned().collect();

        let added: Vec<_> = set2.difference(&set1).cloned().collect();
        let removed: Vec<_> = set1.difference(&set2).cloned().collect();
        let common: Vec<_> = set1.intersection(&set2).cloned().collect();

        Self {
            added,
            removed,
            common,
        }
    }

    /// Get triples that were added in the second graph
    pub fn added(&self) -> &[Triple] {
        &self.added
    }

    /// Get triples that were removed from the first graph
    pub fn removed(&self) -> &[Triple] {
        &self.removed
    }

    /// Get triples common to both graphs
    pub fn common(&self) -> &[Triple] {
        &self.common
    }

    /// Check if the graphs are identical
    pub fn is_identical(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }

    /// Get the total number of changes (additions + removals)
    pub fn change_count(&self) -> usize {
        self.added.len() + self.removed.len()
    }

    /// Generate a summary of the diff
    pub fn summary(&self) -> DiffSummary {
        DiffSummary {
            added_count: self.added.len(),
            removed_count: self.removed.len(),
            common_count: self.common.len(),
            total_changes: self.change_count(),
        }
    }
}

/// Summary statistics for a graph diff
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffSummary {
    /// Number of triples added
    pub added_count: usize,
    /// Number of triples removed
    pub removed_count: usize,
    /// Number of triples common to both
    pub common_count: usize,
    /// Total number of changes
    pub total_changes: usize,
}

impl std::fmt::Display for DiffSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Graph Diff Summary: +{} -{} ={} (total changes: {})",
            self.added_count, self.removed_count, self.common_count, self.total_changes
        )
    }
}

/// Graph transformation utilities
#[derive(Debug)]
pub struct GraphTransformer;

impl GraphTransformer {
    /// Filter triples by predicate
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_ttl::toolkit::graph_utils::GraphTransformer;
    /// use oxirs_core::model::{Triple, NamedNode};
    ///
    /// let triples = vec![
    ///     Triple::new(
    ///         NamedNode::new("http://example.org/s")?,
    ///         NamedNode::new("http://example.org/p1")?,
    ///         NamedNode::new("http://example.org/o")?,
    ///     ),
    ///     Triple::new(
    ///         NamedNode::new("http://example.org/s")?,
    ///         NamedNode::new("http://example.org/p2")?,
    ///         NamedNode::new("http://example.org/o")?,
    ///     ),
    /// ];
    ///
    /// let filtered = GraphTransformer::filter_by_predicate(
    ///     &triples,
    ///     |p| p.ends_with("p1")
    /// );
    /// assert_eq!(filtered.len(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn filter_by_predicate<F>(triples: &[Triple], predicate: F) -> Vec<Triple>
    where
        F: Fn(&str) -> bool,
    {
        triples
            .iter()
            .filter(|t| predicate(t.predicate().as_str()))
            .cloned()
            .collect()
    }

    /// Filter triples by subject
    pub fn filter_by_subject<F>(triples: &[Triple], predicate: F) -> Vec<Triple>
    where
        F: Fn(&oxirs_core::model::Subject) -> bool,
    {
        triples
            .iter()
            .filter(|t| predicate(t.subject()))
            .cloned()
            .collect()
    }

    /// Group triples by subject
    ///
    /// Returns a map from subject to all triples with that subject.
    pub fn group_by_subject(triples: &[Triple]) -> HashMap<String, Vec<Triple>> {
        let mut groups: HashMap<String, Vec<Triple>> = HashMap::new();

        for triple in triples {
            let subject_str = triple.subject().to_string();
            groups.entry(subject_str).or_default().push(triple.clone());
        }

        groups
    }

    /// Group triples by predicate
    pub fn group_by_predicate(triples: &[Triple]) -> HashMap<String, Vec<Triple>> {
        let mut groups: HashMap<String, Vec<Triple>> = HashMap::new();

        for triple in triples {
            let predicate_str = triple.predicate().to_string();
            groups
                .entry(predicate_str)
                .or_default()
                .push(triple.clone());
        }

        groups
    }

    /// Get all unique subjects in the graph
    pub fn unique_subjects(triples: &[Triple]) -> Vec<String> {
        let subjects: HashSet<_> = triples.iter().map(|t| t.subject().to_string()).collect();
        subjects.into_iter().collect()
    }

    /// Get all unique predicates in the graph
    pub fn unique_predicates(triples: &[Triple]) -> Vec<String> {
        let predicates: HashSet<_> = triples.iter().map(|t| t.predicate().to_string()).collect();
        predicates.into_iter().collect()
    }

    /// Get all unique objects in the graph
    pub fn unique_objects(triples: &[Triple]) -> Vec<String> {
        let objects: HashSet<_> = triples.iter().map(|t| t.object().to_string()).collect();
        objects.into_iter().collect()
    }
}

/// Advanced graph statistics
#[derive(Debug, Clone, PartialEq)]
pub struct AdvancedGraphStats {
    /// Total number of triples
    pub triple_count: usize,
    /// Number of unique subjects
    pub unique_subjects: usize,
    /// Number of unique predicates
    pub unique_predicates: usize,
    /// Number of unique objects
    pub unique_objects: usize,
    /// Average triples per subject
    pub avg_triples_per_subject: f64,
    /// Maximum triples for any subject
    pub max_triples_per_subject: usize,
    /// Number of subjects with only one triple
    pub singleton_subjects: usize,
    /// Most common predicates (top 10)
    pub top_predicates: Vec<(String, usize)>,
}

impl AdvancedGraphStats {
    /// Compute advanced statistics for a graph
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_ttl::toolkit::graph_utils::AdvancedGraphStats;
    /// use oxirs_core::model::{Triple, NamedNode};
    ///
    /// let triples = vec![
    ///     Triple::new(
    ///         NamedNode::new("http://example.org/s1")?,
    ///         NamedNode::new("http://example.org/p")?,
    ///         NamedNode::new("http://example.org/o1")?,
    ///     ),
    ///     Triple::new(
    ///         NamedNode::new("http://example.org/s1")?,
    ///         NamedNode::new("http://example.org/p")?,
    ///         NamedNode::new("http://example.org/o2")?,
    ///     ),
    /// ];
    ///
    /// let stats = AdvancedGraphStats::compute(&triples);
    /// assert_eq!(stats.triple_count, 2);
    /// assert_eq!(stats.unique_subjects, 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compute(triples: &[Triple]) -> Self {
        let triple_count = triples.len();

        // Count unique subjects, predicates, objects
        let subjects: HashSet<_> = triples.iter().map(|t| t.subject().to_string()).collect();
        let predicates: HashSet<_> = triples.iter().map(|t| t.predicate().to_string()).collect();
        let objects: HashSet<_> = triples.iter().map(|t| t.object().to_string()).collect();

        let unique_subjects = subjects.len();
        let unique_predicates = predicates.len();
        let unique_objects = objects.len();

        // Group by subject to compute per-subject stats
        let mut subject_counts: HashMap<String, usize> = HashMap::new();
        for triple in triples {
            *subject_counts
                .entry(triple.subject().to_string())
                .or_insert(0) += 1;
        }

        let max_triples_per_subject = subject_counts.values().max().copied().unwrap_or(0);
        let singleton_subjects = subject_counts.values().filter(|&&c| c == 1).count();
        let avg_triples_per_subject = if unique_subjects > 0 {
            triple_count as f64 / unique_subjects as f64
        } else {
            0.0
        };

        // Count predicate frequencies
        let mut predicate_counts: HashMap<String, usize> = HashMap::new();
        for triple in triples {
            *predicate_counts
                .entry(triple.predicate().to_string())
                .or_insert(0) += 1;
        }

        // Get top 10 most common predicates
        let mut predicate_vec: Vec<_> = predicate_counts.into_iter().collect();
        predicate_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let top_predicates = predicate_vec.into_iter().take(10).collect();

        Self {
            triple_count,
            unique_subjects,
            unique_predicates,
            unique_objects,
            avg_triples_per_subject,
            max_triples_per_subject,
            singleton_subjects,
            top_predicates,
        }
    }

    /// Generate a formatted report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Advanced Graph Statistics ===\n");
        report.push_str(&format!("Total triples: {}\n", self.triple_count));
        report.push_str(&format!("Unique subjects: {}\n", self.unique_subjects));
        report.push_str(&format!("Unique predicates: {}\n", self.unique_predicates));
        report.push_str(&format!("Unique objects: {}\n", self.unique_objects));
        report.push_str(&format!(
            "Average triples per subject: {:.2}\n",
            self.avg_triples_per_subject
        ));
        report.push_str(&format!(
            "Max triples per subject: {}\n",
            self.max_triples_per_subject
        ));
        report.push_str(&format!(
            "Singleton subjects: {}\n",
            self.singleton_subjects
        ));
        report.push_str("\nTop predicates:\n");
        for (i, (pred, count)) in self.top_predicates.iter().enumerate() {
            report.push_str(&format!("  {}. {} ({})\n", i + 1, pred, count));
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::NamedNode;

    fn create_test_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(
            NamedNode::new(s).expect("valid IRI"),
            NamedNode::new(p).expect("valid IRI"),
            NamedNode::new(o).expect("valid IRI"),
        )
    }

    #[test]
    fn test_graph_merger() {
        let graph1 = vec![create_test_triple(
            "http://example.org/s1",
            "http://example.org/p",
            "http://example.org/o",
        )];

        let graph2 = vec![
            create_test_triple(
                "http://example.org/s1",
                "http://example.org/p",
                "http://example.org/o",
            ),
            create_test_triple(
                "http://example.org/s2",
                "http://example.org/p",
                "http://example.org/o",
            ),
        ];

        let merger = GraphMerger::new();
        let merged = merger.merge(&[graph1, graph2]);

        // Should deduplicate
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_graph_merger_with_duplicates() {
        let graph1 = vec![create_test_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];

        let graph2 = vec![create_test_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];

        let merger = GraphMerger::with_duplicates();
        let merged = merger.merge(&[graph1, graph2]);

        // Should NOT deduplicate
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_graph_diff() {
        let graph1 = vec![
            create_test_triple(
                "http://example.org/s",
                "http://example.org/p1",
                "http://example.org/o",
            ),
            create_test_triple(
                "http://example.org/s",
                "http://example.org/p2",
                "http://example.org/o",
            ),
        ];

        let graph2 = vec![
            create_test_triple(
                "http://example.org/s",
                "http://example.org/p2",
                "http://example.org/o",
            ),
            create_test_triple(
                "http://example.org/s",
                "http://example.org/p3",
                "http://example.org/o",
            ),
        ];

        let diff = GraphDiff::compute(&graph1, &graph2);

        assert_eq!(diff.added().len(), 1); // p3
        assert_eq!(diff.removed().len(), 1); // p1
        assert_eq!(diff.common().len(), 1); // p2
        assert_eq!(diff.change_count(), 2);
        assert!(!diff.is_identical());
    }

    #[test]
    fn test_graph_transformer_filter() {
        let triples = vec![
            create_test_triple(
                "http://example.org/s",
                "http://example.org/p1",
                "http://example.org/o",
            ),
            create_test_triple(
                "http://example.org/s",
                "http://example.org/p2",
                "http://example.org/o",
            ),
        ];

        let filtered = GraphTransformer::filter_by_predicate(&triples, |p| p.ends_with("p1"));

        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_graph_transformer_grouping() {
        let triples = vec![
            create_test_triple(
                "http://example.org/s1",
                "http://example.org/p",
                "http://example.org/o1",
            ),
            create_test_triple(
                "http://example.org/s1",
                "http://example.org/p",
                "http://example.org/o2",
            ),
            create_test_triple(
                "http://example.org/s2",
                "http://example.org/p",
                "http://example.org/o3",
            ),
        ];

        let groups = GraphTransformer::group_by_subject(&triples);
        assert_eq!(groups.len(), 2);
        assert_eq!(
            groups
                .get("<http://example.org/s1>")
                .expect("key should exist")
                .len(),
            2
        );
        assert_eq!(
            groups
                .get("<http://example.org/s2>")
                .expect("key should exist")
                .len(),
            1
        );
    }

    #[test]
    fn test_advanced_stats() {
        let triples = vec![
            create_test_triple(
                "http://example.org/s1",
                "http://example.org/p",
                "http://example.org/o1",
            ),
            create_test_triple(
                "http://example.org/s1",
                "http://example.org/p",
                "http://example.org/o2",
            ),
            create_test_triple(
                "http://example.org/s2",
                "http://example.org/p",
                "http://example.org/o3",
            ),
        ];

        let stats = AdvancedGraphStats::compute(&triples);

        assert_eq!(stats.triple_count, 3);
        assert_eq!(stats.unique_subjects, 2);
        assert_eq!(stats.unique_predicates, 1);
        assert_eq!(stats.unique_objects, 3);
        assert_eq!(stats.max_triples_per_subject, 2);
        assert_eq!(stats.singleton_subjects, 1);
        assert!((stats.avg_triples_per_subject - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_unique_extraction() {
        let triples = vec![
            create_test_triple(
                "http://example.org/s1",
                "http://example.org/p1",
                "http://example.org/o1",
            ),
            create_test_triple(
                "http://example.org/s1",
                "http://example.org/p2",
                "http://example.org/o2",
            ),
            create_test_triple(
                "http://example.org/s2",
                "http://example.org/p1",
                "http://example.org/o1",
            ),
        ];

        let subjects = GraphTransformer::unique_subjects(&triples);
        assert_eq!(subjects.len(), 2);

        let predicates = GraphTransformer::unique_predicates(&triples);
        assert_eq!(predicates.len(), 2);

        let objects = GraphTransformer::unique_objects(&triples);
        assert_eq!(objects.len(), 2);
    }
}
