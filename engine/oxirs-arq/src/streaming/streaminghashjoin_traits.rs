//! # StreamingHashJoin - Trait Implementations
//!
//! This module contains trait implementations for `StreamingHashJoin`.
//!
//! ## Implemented Traits
//!
//! - `DataStream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::{anyhow, Result};

use super::functions::DataStream;
use super::types::{StreamStats, StreamingHashJoin};

impl DataStream for StreamingHashJoin {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if !self.left_exhausted {
            while let Some(batch) = self.left_stream.next_batch()? {
                for solution in batch {
                    let key = self.extract_join_key(&solution);
                    let estimated_size = std::mem::size_of_val(&solution) + key.len();
                    if !self
                        .memory_monitor
                        .allocate(estimated_size, "hash_join_build")
                    {
                        self.spill_hash_table()?;
                        if !self
                            .memory_monitor
                            .allocate(estimated_size, "hash_join_build")
                        {
                            return Err(anyhow!("Cannot allocate memory even after spilling"));
                        }
                    }
                    self.hash_table.entry(key).or_default().push(solution);
                }
            }
            self.left_exhausted = true;
        }
        if let Some(right_batch) = self.right_stream.next_batch()? {
            let mut result_batch = Vec::new();
            for right_solution in right_batch {
                let key = self.extract_join_key(&right_solution);
                if let Some(left_solutions) = self.hash_table.get(&key) {
                    for left_solution in left_solutions {
                        if let Some(joined) = self.join_solutions(left_solution, &right_solution) {
                            result_batch.push(joined);
                        }
                    }
                }
            }
            Ok(if result_batch.is_empty() {
                None
            } else {
                Some(result_batch)
            })
        } else {
            Ok(None)
        }
    }
    fn has_more(&self) -> bool {
        !self.left_exhausted || self.right_stream.has_more()
    }
    fn estimated_size(&self) -> Option<usize> {
        None
    }
    fn reset(&mut self) -> Result<()> {
        self.left_stream.reset()?;
        self.right_stream.reset()?;
        self.hash_table.clear();
        self.left_exhausted = false;
        self.current_batch = None;
        Ok(())
    }
    fn get_stats(&self) -> StreamStats {
        StreamStats::default()
    }
}
