//! Indexing structures for efficient quoted triple lookups.
//!
//! This module provides B-tree based indexing for RDF-star quoted triples,
//! enabling efficient pattern-based queries and nesting depth optimization.

use std::collections::{BTreeMap, BTreeSet};

/// Indexing structure for efficient quoted triple lookups
#[derive(Debug, Clone)]
pub(super) struct QuotedTripleIndex {
    /// B-tree index mapping quoted triple signatures to triple indices
    pub(super) signature_to_indices: BTreeMap<String, BTreeSet<usize>>,
    /// Subject-based index for S?? pattern queries
    pub(super) subject_index: BTreeMap<String, BTreeSet<usize>>,
    /// Predicate-based index for ?P? pattern queries
    pub(super) predicate_index: BTreeMap<String, BTreeSet<usize>>,
    /// Object-based index for ??O pattern queries
    pub(super) object_index: BTreeMap<String, BTreeSet<usize>>,
    /// Nesting depth index for performance optimization
    pub(super) nesting_depth_index: BTreeMap<usize, BTreeSet<usize>>,
}

impl QuotedTripleIndex {
    pub(super) fn new() -> Self {
        Self {
            signature_to_indices: BTreeMap::new(),
            subject_index: BTreeMap::new(),
            predicate_index: BTreeMap::new(),
            object_index: BTreeMap::new(),
            nesting_depth_index: BTreeMap::new(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.signature_to_indices.clear();
        self.subject_index.clear();
        self.predicate_index.clear();
        self.object_index.clear();
        self.nesting_depth_index.clear();
    }

    /// Get index statistics for optimization analysis
    pub(super) fn get_statistics(&self) -> IndexStatistics {
        IndexStatistics {
            total_entries: self.signature_to_indices.len(),
            subject_index_size: self.subject_index.len(),
            predicate_index_size: self.predicate_index.len(),
            object_index_size: self.object_index.len(),
            nesting_depth_levels: self.nesting_depth_index.len(),
            average_bucket_size: self.calculate_average_bucket_size(),
        }
    }

    fn calculate_average_bucket_size(&self) -> f64 {
        if self.signature_to_indices.is_empty() {
            return 0.0;
        }
        let total_entries: usize = self.signature_to_indices.values().map(|s| s.len()).sum();
        total_entries as f64 / self.signature_to_indices.len() as f64
    }
}

/// Statistics about the indexing performance
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    pub total_entries: usize,
    pub subject_index_size: usize,
    pub predicate_index_size: usize,
    pub object_index_size: usize,
    pub nesting_depth_levels: usize,
    pub average_bucket_size: f64,
}
