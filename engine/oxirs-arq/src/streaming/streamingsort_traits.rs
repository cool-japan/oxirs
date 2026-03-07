//! # StreamingSort - Trait Implementations
//!
//! This module contains trait implementations for `StreamingSort`.
//!
//! ## Implemented Traits
//!
//! - `DataStream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::Result;

use super::functions::DataStream;
use super::types::{SpillDataType, StreamStats, StreamingSort};

impl DataStream for StreamingSort {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if !self.fully_sorted {
            let mut all_data = Vec::new();
            while let Some(batch) = self.input.next_batch()? {
                all_data.extend(batch);
                if all_data.len() > self.config.batch_size {
                    all_data.sort_by(|a, b| self.compare_solutions(a, b));
                    let spill_id = self
                        .spill_manager
                        .lock()
                        .expect("lock poisoned")
                        .spill_data(&all_data, SpillDataType::SortBuffer)?;
                    self.sorted_batches.push(spill_id);
                    all_data.clear();
                }
            }
            if !all_data.is_empty() {
                all_data.sort_by(|a, b| self.compare_solutions(a, b));
                let spill_id = self
                    .spill_manager
                    .lock()
                    .expect("lock poisoned")
                    .spill_data(&all_data, SpillDataType::SortBuffer)?;
                self.sorted_batches.push(spill_id);
            }
            self.fully_sorted = true;
        }
        if self.current_batch_index < self.sorted_batches.len() {
            let spill_id = &self.sorted_batches[self.current_batch_index];
            let batch: Vec<Solution> = self
                .spill_manager
                .lock()
                .expect("lock poisoned")
                .read_spill(spill_id)?;
            self.current_batch_index += 1;
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }
    fn has_more(&self) -> bool {
        if !self.fully_sorted {
            self.input.has_more()
        } else {
            self.current_batch_index < self.sorted_batches.len()
        }
    }
    fn estimated_size(&self) -> Option<usize> {
        self.input.estimated_size()
    }
    fn reset(&mut self) -> Result<()> {
        self.input.reset()?;
        for spill_id in &self.sorted_batches {
            self.spill_manager
                .lock()
                .expect("lock poisoned")
                .delete_spill(spill_id)?;
        }
        self.sorted_batches.clear();
        self.current_batch_index = 0;
        self.fully_sorted = false;
        Ok(())
    }
    fn get_stats(&self) -> StreamStats {
        self.input.get_stats()
    }
}
