//! Graph diff tool for comparing annotated RDF-star graphs
//!
//! This module provides comprehensive diff capabilities for RDF-star graphs,
//! including triple comparison, annotation changes, provenance tracking,
//! and multiple output formats.

use crate::annotations::{AnnotationStore, ProvenanceRecord, TripleAnnotation};
use crate::model::{StarGraph, StarTriple};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use thiserror::Error;
use tracing::info;

/// Errors related to graph diff operations
#[derive(Error, Debug)]
pub enum DiffError {
    #[error("Comparison failed: {0}")]
    ComparisonFailed(String),

    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Type of change in a diff
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeType {
    /// Triple added in new version
    Added,
    /// Triple removed from old version
    Removed,
    /// Triple exists in both but annotations changed
    Modified,
    /// Triple unchanged
    Unchanged,
}

impl fmt::Display for ChangeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChangeType::Added => write!(f, "ADDED"),
            ChangeType::Removed => write!(f, "REMOVED"),
            ChangeType::Modified => write!(f, "MODIFIED"),
            ChangeType::Unchanged => write!(f, "UNCHANGED"),
        }
    }
}

/// Annotation change details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationChange {
    /// Type of change
    pub change_type: ChangeType,

    /// Old annotation (if exists)
    pub old_annotation: Option<TripleAnnotation>,

    /// New annotation (if exists)
    pub new_annotation: Option<TripleAnnotation>,

    /// Specific fields that changed
    pub changed_fields: Vec<String>,

    /// Provenance changes
    pub provenance_changes: Vec<ProvenanceChange>,
}

/// Provenance change details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceChange {
    /// Change type
    pub change_type: ChangeType,

    /// Old provenance
    pub old_provenance: Option<ProvenanceRecord>,

    /// New provenance
    pub new_provenance: Option<ProvenanceRecord>,
}

/// Triple difference entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleDiff {
    /// The triple being compared
    pub triple: StarTriple,

    /// Type of change
    pub change_type: ChangeType,

    /// Annotation changes
    pub annotation_changes: Option<AnnotationChange>,

    /// Context information
    pub context: HashMap<String, String>,
}

/// Graph diff result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDiff {
    /// Comparison timestamp
    pub timestamp: DateTime<Utc>,

    /// Old graph identifier
    pub old_id: String,

    /// New graph identifier
    pub new_id: String,

    /// All differences
    pub differences: Vec<TripleDiff>,

    /// Summary statistics
    pub summary: DiffSummary,

    /// Comparison options used
    pub options: DiffOptions,
}

/// Diff summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    /// Total triples in old graph
    pub old_triple_count: usize,

    /// Total triples in new graph
    pub new_triple_count: usize,

    /// Number of added triples
    pub added_count: usize,

    /// Number of removed triples
    pub removed_count: usize,

    /// Number of modified triples
    pub modified_count: usize,

    /// Number of unchanged triples
    pub unchanged_count: usize,

    /// Total annotation changes
    pub annotation_changes_count: usize,

    /// Total provenance changes
    pub provenance_changes_count: usize,

    /// Similarity percentage
    pub similarity_percentage: f64,
}

impl DiffSummary {
    /// Calculate similarity percentage
    fn calculate_similarity(old_count: usize, new_count: usize, unchanged: usize) -> f64 {
        let total = old_count.max(new_count);
        if total == 0 {
            100.0
        } else {
            (unchanged as f64 / total as f64) * 100.0
        }
    }
}

/// Options for diff comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffOptions {
    /// Compare annotations
    pub compare_annotations: bool,

    /// Compare provenance chains
    pub compare_provenance: bool,

    /// Compare trust scores
    pub compare_trust_scores: bool,

    /// Compare timestamps
    pub compare_timestamps: bool,

    /// Include unchanged triples in output
    pub include_unchanged: bool,

    /// Ignore metadata changes
    pub ignore_metadata: bool,

    /// Context depth for nested triples
    pub context_depth: usize,
}

impl Default for DiffOptions {
    fn default() -> Self {
        Self {
            compare_annotations: true,
            compare_provenance: true,
            compare_trust_scores: true,
            compare_timestamps: false,
            include_unchanged: false,
            ignore_metadata: false,
            context_depth: 10,
        }
    }
}

/// Graph diff tool
pub struct GraphDiffTool {
    /// Comparison options
    options: DiffOptions,
}

impl GraphDiffTool {
    /// Create a new diff tool with default options
    pub fn new() -> Self {
        Self {
            options: DiffOptions::default(),
        }
    }

    /// Create with custom options
    pub fn with_options(options: DiffOptions) -> Self {
        Self { options }
    }

    /// Compare two graphs
    pub fn compare(
        &self,
        old_graph: &StarGraph,
        new_graph: &StarGraph,
        old_annotations: Option<&AnnotationStore>,
        new_annotations: Option<&AnnotationStore>,
    ) -> Result<GraphDiff, DiffError> {
        info!("Starting graph comparison");

        let old_triples: HashSet<StarTriple> = old_graph.triples().iter().cloned().collect();
        let new_triples: HashSet<StarTriple> = new_graph.triples().iter().cloned().collect();

        let mut differences = Vec::new();

        // Find added triples
        for triple in new_triples.difference(&old_triples) {
            let annotation_changes = if self.options.compare_annotations {
                self.get_annotation_changes(triple, None, new_annotations)?
            } else {
                None
            };

            differences.push(TripleDiff {
                triple: triple.clone(),
                change_type: ChangeType::Added,
                annotation_changes,
                context: HashMap::new(),
            });
        }

        // Find removed triples
        for triple in old_triples.difference(&new_triples) {
            let annotation_changes = if self.options.compare_annotations {
                self.get_annotation_changes(triple, old_annotations, None)?
            } else {
                None
            };

            differences.push(TripleDiff {
                triple: triple.clone(),
                change_type: ChangeType::Removed,
                annotation_changes,
                context: HashMap::new(),
            });
        }

        // Find unchanged/modified triples
        for triple in old_triples.intersection(&new_triples) {
            let annotation_changes = if self.options.compare_annotations {
                self.compare_annotations(triple, old_annotations, new_annotations)?
            } else {
                None
            };

            let change_type = if annotation_changes.is_some() {
                ChangeType::Modified
            } else {
                ChangeType::Unchanged
            };

            if self.options.include_unchanged || change_type == ChangeType::Modified {
                differences.push(TripleDiff {
                    triple: triple.clone(),
                    change_type,
                    annotation_changes,
                    context: HashMap::new(),
                });
            }
        }

        // Calculate summary
        let added_count = differences
            .iter()
            .filter(|d| d.change_type == ChangeType::Added)
            .count();
        let removed_count = differences
            .iter()
            .filter(|d| d.change_type == ChangeType::Removed)
            .count();
        let modified_count = differences
            .iter()
            .filter(|d| d.change_type == ChangeType::Modified)
            .count();
        let unchanged_count = old_triples
            .intersection(&new_triples)
            .count()
            .saturating_sub(modified_count);

        let annotation_changes_count = differences
            .iter()
            .filter(|d| d.annotation_changes.is_some())
            .count();

        let provenance_changes_count: usize = differences
            .iter()
            .filter_map(|d| d.annotation_changes.as_ref())
            .map(|ac| ac.provenance_changes.len())
            .sum();

        let similarity_percentage = DiffSummary::calculate_similarity(
            old_triples.len(),
            new_triples.len(),
            unchanged_count,
        );

        let summary = DiffSummary {
            old_triple_count: old_triples.len(),
            new_triple_count: new_triples.len(),
            added_count,
            removed_count,
            modified_count,
            unchanged_count,
            annotation_changes_count,
            provenance_changes_count,
            similarity_percentage,
        };

        Ok(GraphDiff {
            timestamp: Utc::now(),
            old_id: format!("old_{}", old_graph.len()),
            new_id: format!("new_{}", new_graph.len()),
            differences,
            summary,
            options: self.options.clone(),
        })
    }

    /// Get annotation changes for a triple (added/removed)
    fn get_annotation_changes(
        &self,
        triple: &StarTriple,
        old_store: Option<&AnnotationStore>,
        new_store: Option<&AnnotationStore>,
    ) -> Result<Option<AnnotationChange>, DiffError> {
        let old_annotation = old_store.and_then(|store| store.get_annotation(triple));
        let new_annotation = new_store.and_then(|store| store.get_annotation(triple));

        if old_annotation.is_none() && new_annotation.is_none() {
            return Ok(None);
        }

        let change_type = match (&old_annotation, &new_annotation) {
            (None, Some(_)) => ChangeType::Added,
            (Some(_), None) => ChangeType::Removed,
            _ => ChangeType::Modified,
        };

        Ok(Some(AnnotationChange {
            change_type,
            old_annotation: old_annotation.cloned(),
            new_annotation: new_annotation.cloned(),
            changed_fields: vec![],
            provenance_changes: vec![],
        }))
    }

    /// Compare annotations for a triple that exists in both graphs
    fn compare_annotations(
        &self,
        triple: &StarTriple,
        old_store: Option<&AnnotationStore>,
        new_store: Option<&AnnotationStore>,
    ) -> Result<Option<AnnotationChange>, DiffError> {
        let old_annotation = old_store.and_then(|store| store.get_annotation(triple));
        let new_annotation = new_store.and_then(|store| store.get_annotation(triple));

        match (old_annotation, new_annotation) {
            (Some(old), Some(new)) => {
                let mut changed_fields = Vec::new();
                let provenance_changes = Vec::new();

                // Compare confidence
                if self.options.compare_trust_scores && old.confidence != new.confidence {
                    changed_fields.push("confidence".to_string());
                }

                // Compare source
                if old.source != new.source {
                    changed_fields.push("source".to_string());
                }

                // Compare timestamps
                if self.options.compare_timestamps && old.timestamp != new.timestamp {
                    changed_fields.push("timestamp".to_string());
                }

                // Compare provenance counts (simplified)
                if self.options.compare_provenance {
                    let old_prov_count = old.provenance.len();
                    let new_prov_count = new.provenance.len();

                    if old_prov_count != new_prov_count {
                        changed_fields.push("provenance_count".to_string());
                    }
                }

                if changed_fields.is_empty() && provenance_changes.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(AnnotationChange {
                        change_type: ChangeType::Modified,
                        old_annotation: Some(old.clone()),
                        new_annotation: Some(new.clone()),
                        changed_fields,
                        provenance_changes,
                    }))
                }
            }
            (Some(_), None) => self.get_annotation_changes(triple, old_store, new_store),
            (None, Some(_)) => self.get_annotation_changes(triple, old_store, new_store),
            (None, None) => Ok(None),
        }
    }

    /// Format diff as human-readable text
    pub fn format_text(&self, diff: &GraphDiff) -> String {
        let mut output = String::new();

        output.push_str("=== Graph Diff Report ===\n\n");
        output.push_str(&format!("Timestamp: {}\n", diff.timestamp));
        output.push_str(&format!("Old Graph: {}\n", diff.old_id));
        output.push_str(&format!("New Graph: {}\n\n", diff.new_id));

        output.push_str("--- Summary ---\n");
        output.push_str(&format!("Old triples: {}\n", diff.summary.old_triple_count));
        output.push_str(&format!("New triples: {}\n", diff.summary.new_triple_count));
        output.push_str(&format!("Added: {}\n", diff.summary.added_count));
        output.push_str(&format!("Removed: {}\n", diff.summary.removed_count));
        output.push_str(&format!("Modified: {}\n", diff.summary.modified_count));
        output.push_str(&format!("Unchanged: {}\n", diff.summary.unchanged_count));
        output.push_str(&format!(
            "Similarity: {:.2}%\n\n",
            diff.summary.similarity_percentage
        ));

        output.push_str("--- Changes ---\n");
        for (i, change) in diff.differences.iter().enumerate() {
            output.push_str(&format!(
                "\n{}. {} {}\n",
                i + 1,
                change.change_type,
                self.format_triple(&change.triple)
            ));

            if let Some(ref ann_change) = change.annotation_changes {
                output.push_str("   Annotation changes:\n");
                for field in &ann_change.changed_fields {
                    output.push_str(&format!("   - {}\n", field));
                }
                if !ann_change.provenance_changes.is_empty() {
                    output.push_str(&format!(
                        "   - {} provenance changes\n",
                        ann_change.provenance_changes.len()
                    ));
                }
            }
        }

        output
    }

    /// Format triple as string
    fn format_triple(&self, triple: &StarTriple) -> String {
        format!(
            "<{:?} {:?} {:?}>",
            triple.subject, triple.predicate, triple.object
        )
    }

    /// Export diff to JSON
    pub fn export_json(&self, diff: &GraphDiff) -> Result<String, DiffError> {
        serde_json::to_string_pretty(diff).map_err(|e| DiffError::SerializationError(e.to_string()))
    }

    /// Get options
    pub fn options(&self) -> &DiffOptions {
        &self.options
    }
}

impl Default for GraphDiffTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for graph comparison
pub mod utils {
    use super::*;

    /// Quick comparison of two graphs (basic stats only)
    pub fn quick_compare(old_graph: &StarGraph, new_graph: &StarGraph) -> DiffSummary {
        let old_triples: HashSet<StarTriple> = old_graph.triples().iter().cloned().collect();
        let new_triples: HashSet<StarTriple> = new_graph.triples().iter().cloned().collect();

        let added = new_triples.difference(&old_triples).count();
        let removed = old_triples.difference(&new_triples).count();
        let unchanged = old_triples.intersection(&new_triples).count();

        let similarity =
            DiffSummary::calculate_similarity(old_triples.len(), new_triples.len(), unchanged);

        DiffSummary {
            old_triple_count: old_triples.len(),
            new_triple_count: new_triples.len(),
            added_count: added,
            removed_count: removed,
            modified_count: 0,
            unchanged_count: unchanged,
            annotation_changes_count: 0,
            provenance_changes_count: 0,
            similarity_percentage: similarity,
        }
    }

    /// Check if two graphs are identical
    pub fn are_identical(graph1: &StarGraph, graph2: &StarGraph) -> bool {
        let triples1: HashSet<StarTriple> = graph1.triples().iter().cloned().collect();
        let triples2: HashSet<StarTriple> = graph2.triples().iter().cloned().collect();
        triples1 == triples2
    }

    /// Calculate Jaccard similarity between two graphs
    pub fn jaccard_similarity(graph1: &StarGraph, graph2: &StarGraph) -> f64 {
        let triples1: HashSet<StarTriple> = graph1.triples().iter().cloned().collect();
        let triples2: HashSet<StarTriple> = graph2.triples().iter().cloned().collect();

        let intersection = triples1.intersection(&triples2).count();
        let union = triples1.union(&triples2).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTerm;

    fn create_test_triple(subj: &str, pred: &str, obj: &str) -> StarTriple {
        StarTriple::new(
            StarTerm::iri(subj).unwrap(),
            StarTerm::iri(pred).unwrap(),
            StarTerm::iri(obj).unwrap(),
        )
    }

    #[test]
    fn test_diff_tool_creation() {
        let tool = GraphDiffTool::new();
        assert!(tool.options().compare_annotations);
        assert!(tool.options().compare_provenance);
    }

    #[test]
    fn test_quick_compare() {
        let mut graph1 = StarGraph::new();
        let mut graph2 = StarGraph::new();

        let triple1 = create_test_triple("http://ex.org/s1", "http://ex.org/p", "http://ex.org/o1");
        let triple2 = create_test_triple("http://ex.org/s2", "http://ex.org/p", "http://ex.org/o2");
        let triple3 = create_test_triple("http://ex.org/s3", "http://ex.org/p", "http://ex.org/o3");

        graph1.insert(triple1).unwrap();
        graph1.insert(triple2.clone()).unwrap();

        graph2.insert(triple2).unwrap();
        graph2.insert(triple3).unwrap();

        let summary = utils::quick_compare(&graph1, &graph2);
        assert_eq!(summary.old_triple_count, 2);
        assert_eq!(summary.new_triple_count, 2);
        assert_eq!(summary.added_count, 1);
        assert_eq!(summary.removed_count, 1);
        assert_eq!(summary.unchanged_count, 1);
    }

    #[test]
    fn test_identical_graphs() {
        let mut graph1 = StarGraph::new();
        let mut graph2 = StarGraph::new();

        let triple = create_test_triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");

        graph1.insert(triple.clone()).unwrap();
        graph2.insert(triple).unwrap();

        assert!(utils::are_identical(&graph1, &graph2));
    }

    #[test]
    fn test_jaccard_similarity() {
        let mut graph1 = StarGraph::new();
        let mut graph2 = StarGraph::new();

        let triple1 = create_test_triple("http://ex.org/s1", "http://ex.org/p", "http://ex.org/o1");
        let triple2 = create_test_triple("http://ex.org/s2", "http://ex.org/p", "http://ex.org/o2");

        graph1.insert(triple1.clone()).unwrap();
        graph1.insert(triple2).unwrap();

        graph2.insert(triple1).unwrap();

        let similarity = utils::jaccard_similarity(&graph1, &graph2);
        assert!((similarity - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_change_type_display() {
        assert_eq!(format!("{}", ChangeType::Added), "ADDED");
        assert_eq!(format!("{}", ChangeType::Removed), "REMOVED");
        assert_eq!(format!("{}", ChangeType::Modified), "MODIFIED");
        assert_eq!(format!("{}", ChangeType::Unchanged), "UNCHANGED");
    }

    #[test]
    fn test_diff_options_default() {
        let options = DiffOptions::default();
        assert!(options.compare_annotations);
        assert!(options.compare_provenance);
        assert!(!options.include_unchanged);
    }
}
