//! # StreamingPatternScan - Trait Implementations
//!
//! This module contains trait implementations for `StreamingPatternScan`.
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
use super::types::{StreamStats, StreamingPatternScan};

impl DataStream for StreamingPatternScan {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if !self.spilled_batches.is_empty() {
            let spill_id = self.spilled_batches.remove(0);
            let spilled_solutions: Vec<Solution> = self
                .spill_manager
                .lock()
                .expect("lock poisoned")
                .read_spill(&spill_id)?;
            return Ok(Some(spilled_solutions));
        }
        if self.batch_index < 10 {
            let solutions = self.generate_pattern_solutions()?;
            if solutions.is_empty() {
                return Ok(None);
            }
            if self.should_spill() {
                self.current_batch = solutions;
                self.spill_current_batch()?;
                self.batch_index += 1;
                return self.next_batch();
            }
            self.batch_index += 1;
            self.total_results += solutions.len();
            Ok(Some(solutions))
        } else {
            Ok(None)
        }
    }
    fn has_more(&self) -> bool {
        !self.spilled_batches.is_empty() || self.batch_index < 10
    }
    fn estimated_size(&self) -> Option<usize> {
        Some(self.total_results + self.current_batch.len())
    }
    fn reset(&mut self) -> Result<()> {
        self.batch_index = 0;
        self.total_results = 0;
        self.current_batch.clear();
        self.spilled_batches.clear();
        Ok(())
    }
    fn get_stats(&self) -> StreamStats {
        StreamStats {
            rows_processed: self.total_results,
            bytes_processed: 0,
            processing_time: Duration::from_secs(0),
            spill_operations: self.spilled_batches.len(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}
