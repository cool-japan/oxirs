//! # QueryExecutor - execute_single_pattern_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::dataset::{
    convert_property_path, ConcreteStoreDataset, Dataset, DatasetPathAdapter, InMemoryDataset,
};
use crate::algebra::Solution;
use anyhow::Result;

use super::types::AccessPath;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Execute a single triple pattern with index selection
    pub(super) fn execute_single_pattern(
        &self,
        pattern: &crate::algebra::TriplePattern,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let access_path = self.select_access_path(pattern);
        let solution = match access_path {
            AccessPath::SubjectIndex => self.lookup_by_subject(pattern, dataset)?,
            AccessPath::PredicateIndex => self.lookup_by_predicate(pattern, dataset)?,
            AccessPath::ObjectIndex => self.lookup_by_object(pattern, dataset)?,
            AccessPath::FullScan => self.full_scan_pattern(pattern, dataset)?,
        };
        Ok(solution)
    }
    /// Select optimal access path for a pattern
    pub(super) fn select_access_path(&self, pattern: &crate::algebra::TriplePattern) -> AccessPath {
        if !matches!(pattern.subject, crate::algebra::Term::Variable(_)) {
            return AccessPath::SubjectIndex;
        }
        if !matches!(pattern.predicate, crate::algebra::Term::Variable(_)) {
            return AccessPath::PredicateIndex;
        }
        if !matches!(pattern.object, crate::algebra::Term::Variable(_)) {
            return AccessPath::ObjectIndex;
        }
        AccessPath::FullScan
    }
    /// Lookup by subject index
    pub(super) fn lookup_by_subject(
        &self,
        pattern: &crate::algebra::TriplePattern,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        self.execute_pattern_with_dataset(pattern, dataset)
    }
    /// Lookup by predicate index
    pub(super) fn lookup_by_predicate(
        &self,
        pattern: &crate::algebra::TriplePattern,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        self.execute_pattern_with_dataset(pattern, dataset)
    }
    /// Lookup by object index
    pub(super) fn lookup_by_object(
        &self,
        pattern: &crate::algebra::TriplePattern,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        self.execute_pattern_with_dataset(pattern, dataset)
    }
    /// Full scan pattern
    pub(super) fn full_scan_pattern(
        &self,
        pattern: &crate::algebra::TriplePattern,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        self.execute_pattern_with_dataset(pattern, dataset)
    }
}
