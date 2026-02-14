//! # StreamingUnion - Trait Implementations
//!
//! This module contains trait implementations for `StreamingUnion`.
//!
//! ## Implemented Traits
//!
//! - `DataStream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::Result;

use super::functions::DataStream;
use super::types::{StreamStats, StreamingUnion};

impl DataStream for StreamingUnion {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if !self.left_exhausted {
            if let Some(batch) = self.left.next_batch()? {
                return Ok(Some(batch));
            } else {
                self.left_exhausted = true;
            }
        }
        self.right.next_batch()
    }
    fn has_more(&self) -> bool {
        !self.left_exhausted || self.right.has_more()
    }
    fn estimated_size(&self) -> Option<usize> {
        None
    }
    fn reset(&mut self) -> Result<()> {
        self.left.reset()?;
        self.right.reset()?;
        self.left_exhausted = false;
        Ok(())
    }
    fn get_stats(&self) -> StreamStats {
        StreamStats::default()
    }
}
