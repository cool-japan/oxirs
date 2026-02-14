//! # StreamingProjection - Trait Implementations
//!
//! This module contains trait implementations for `StreamingProjection`.
//!
//! ## Implemented Traits
//!
//! - `DataStream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Binding, Solution};
use anyhow::Result;

use super::functions::DataStream;
use super::types::{StreamStats, StreamingHashJoin, StreamingProjection};

impl DataStream for StreamingProjection {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if let Some(batch) = self.input.next_batch()? {
            let projected: Vec<Solution> = batch
                .into_iter()
                .map(|solution| {
                    let mut projected_binding = Binding::new();
                    for var in &self.variables {
                        if let Some(term) = StreamingHashJoin::get_solution_value(&solution, var) {
                            projected_binding.insert(var.clone(), term.clone());
                        }
                    }
                    vec![projected_binding]
                })
                .collect();
            Ok(Some(projected))
        } else {
            Ok(None)
        }
    }
    fn has_more(&self) -> bool {
        self.input.has_more()
    }
    fn estimated_size(&self) -> Option<usize> {
        self.input.estimated_size()
    }
    fn reset(&mut self) -> Result<()> {
        self.input.reset()
    }
    fn get_stats(&self) -> StreamStats {
        self.input.get_stats()
    }
}
