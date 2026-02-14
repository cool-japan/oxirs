//! # StreamingSelection - Trait Implementations
//!
//! This module contains trait implementations for `StreamingSelection`.
//!
//! ## Implemented Traits
//!
//! - `DataStream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::Result;

use super::functions::DataStream;
use super::types::{StreamStats, StreamingSelection};

impl DataStream for StreamingSelection {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if let Some(batch) = self.input.next_batch()? {
            let filtered: Vec<Solution> = batch
                .into_iter()
                .filter(|solution| self.evaluate_condition(solution).unwrap_or(false))
                .collect();
            if filtered.is_empty() && self.input.has_more() {
                self.next_batch()
            } else {
                Ok(Some(filtered))
            }
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
