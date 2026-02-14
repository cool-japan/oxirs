//! # QueryExecutor - execute_index_optimized_join_group Methods
//!
//! This module contains method implementations for `QueryExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::dataset::{
    convert_property_path, ConcreteStoreDataset, Dataset, DatasetPathAdapter, InMemoryDataset,
};
use crate::algebra::{Algebra, Solution};
use anyhow::Result;

use super::queryexecutor_type::QueryExecutor;

impl QueryExecutor {
    /// Execute index-optimized join
    pub(super) fn execute_index_optimized_join(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<Solution> {
        let left_results = self.execute_serial(left, dataset)?;
        let right_results = self.execute_serial(right, dataset)?;
        if left_results.len() < right_results.len() {
            self.hash_join(left_results, right_results)
        } else {
            self.hash_join(right_results, left_results)
        }
    }
}
