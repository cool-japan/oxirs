//! # StreamingMinus - Trait Implementations
//!
//! This module contains trait implementations for `StreamingMinus`.
//!
//! ## Implemented Traits
//!
//! - `DataStream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::Result;

use super::functions::DataStream;
use super::types::{StreamStats, StreamingMinus};

impl DataStream for StreamingMinus {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if let Some(left_batch) = self.left.next_batch()? {
            let mut filtered_batch = Vec::new();
            for left_solution in left_batch {
                let mut exclude = false;
                self.right.reset()?;
                while let Some(right_batch) = self.right.next_batch()? {
                    for right_solution in &right_batch {
                        if self.solutions_compatible(&left_solution, right_solution) {
                            exclude = true;
                            break;
                        }
                    }
                    if exclude {
                        break;
                    }
                }
                if !exclude {
                    filtered_batch.push(left_solution);
                }
            }
            Ok(if filtered_batch.is_empty() {
                None
            } else {
                Some(filtered_batch)
            })
        } else {
            Ok(None)
        }
    }
    fn has_more(&self) -> bool {
        self.left.has_more()
    }
    fn estimated_size(&self) -> Option<usize> {
        self.left.estimated_size()
    }
    fn reset(&mut self) -> Result<()> {
        self.left.reset()?;
        self.right.reset()?;
        Ok(())
    }
    fn get_stats(&self) -> StreamStats {
        let mut stats = self.left.get_stats();
        let right_stats = self.right.get_stats();
        stats.rows_processed += right_stats.rows_processed;
        stats.bytes_processed += right_stats.bytes_processed;
        stats.processing_time += right_stats.processing_time;
        stats
    }
}
