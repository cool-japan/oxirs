//! # EmptyStream - Trait Implementations
//!
//! This module contains trait implementations for `EmptyStream`.
//!
//! ## Implemented Traits
//!
//! - `DataStream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::Result;

use super::functions::DataStream;
use super::types::{EmptyStream, StreamStats};

impl DataStream for EmptyStream {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if self.exhausted {
            Ok(None)
        } else {
            self.exhausted = true;
            Ok(Some(vec![]))
        }
    }
    fn has_more(&self) -> bool {
        !self.exhausted
    }
    fn estimated_size(&self) -> Option<usize> {
        Some(0)
    }
    fn reset(&mut self) -> Result<()> {
        self.exhausted = false;
        Ok(())
    }
    fn get_stats(&self) -> StreamStats {
        StreamStats::default()
    }
}
