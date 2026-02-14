//! # BufferedPatternScan - Trait Implementations
//!
//! This module contains trait implementations for `BufferedPatternScan`.
//!
//! ## Implemented Traits
//!
//! - `DataStream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::Result;
use std::time::Duration;

use super::functions::DataStream;
use super::types::{BufferedPatternScan, StreamStats};

impl DataStream for BufferedPatternScan {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if self.exhausted || self.current_index >= self.solutions.len() {
            return Ok(None);
        }
        let end_index = (self.current_index + self.batch_size).min(self.solutions.len());
        let batch = self.solutions[self.current_index..end_index].to_vec();
        self.current_index = end_index;
        if self.current_index >= self.solutions.len() {
            self.exhausted = true;
        }
        Ok(Some(batch))
    }
    fn has_more(&self) -> bool {
        !self.exhausted && self.current_index < self.solutions.len()
    }
    fn estimated_size(&self) -> Option<usize> {
        Some(self.solutions.len())
    }
    fn reset(&mut self) -> Result<()> {
        self.current_index = 0;
        self.exhausted = false;
        Ok(())
    }
    fn get_stats(&self) -> StreamStats {
        StreamStats {
            rows_processed: self.solutions.len(),
            bytes_processed: 0,
            processing_time: Duration::from_secs(0),
            spill_operations: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}
