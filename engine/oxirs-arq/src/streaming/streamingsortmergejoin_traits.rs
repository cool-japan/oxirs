//! # StreamingSortMergeJoin - Trait Implementations
//!
//! This module contains trait implementations for `StreamingSortMergeJoin`.
//!
//! ## Implemented Traits
//!
//! - `DataStream`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Solution;
use anyhow::Result;

use super::functions::DataStream;
use super::types::{StreamStats, StreamingSortMergeJoin};

impl DataStream for StreamingSortMergeJoin {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        let mut result_batch = Vec::new();
        self.refill_sorted_buffers()?;
        while !self.left_buffer.is_empty() && !self.right_buffer.is_empty() {
            let left_solution = &self.left_buffer[0];
            let right_solution = &self.right_buffer[0];
            let comparison = self.compare_join_keys(left_solution, right_solution);
            match comparison {
                std::cmp::Ordering::Less => {
                    self.left_buffer.pop_front();
                }
                std::cmp::Ordering::Greater => {
                    self.right_buffer.pop_front();
                }
                std::cmp::Ordering::Equal => {
                    let left_key = self.extract_join_key(left_solution);
                    let mut matching_left = Vec::new();
                    while !self.left_buffer.is_empty() {
                        let current_left = &self.left_buffer[0];
                        if self.extract_join_key(current_left) == left_key {
                            matching_left.push(
                                self.left_buffer
                                    .pop_front()
                                    .expect("buffer should not be empty after is_empty check"),
                            );
                        } else {
                            break;
                        }
                    }
                    let mut matching_right = Vec::new();
                    while !self.right_buffer.is_empty() {
                        let current_right = &self.right_buffer[0];
                        if self.extract_join_key(current_right) == left_key {
                            matching_right.push(
                                self.right_buffer
                                    .pop_front()
                                    .expect("buffer should not be empty after is_empty check"),
                            );
                        } else {
                            break;
                        }
                    }
                    for left_sol in &matching_left {
                        for right_sol in &matching_right {
                            if let Some(joined) = self.join_solutions(left_sol, right_sol) {
                                result_batch.push(joined);
                            }
                        }
                    }
                    if result_batch.len() >= self.config.batch_size {
                        break;
                    }
                }
            }
            if self.left_buffer.len() < 10 || self.right_buffer.len() < 10 {
                self.refill_sorted_buffers()?;
            }
        }
        if result_batch.is_empty() && !self.has_more() {
            Ok(None)
        } else {
            Ok(Some(result_batch))
        }
    }
    fn has_more(&self) -> bool {
        !self.left_buffer.is_empty()
            || !self.right_buffer.is_empty()
            || self.left_stream.has_more()
            || self.right_stream.has_more()
    }
    fn estimated_size(&self) -> Option<usize> {
        let left_est = self.left_stream.estimated_size().unwrap_or(0);
        let right_est = self.right_stream.estimated_size().unwrap_or(0);
        Some(left_est + right_est + self.left_buffer.len() + self.right_buffer.len())
    }
    fn reset(&mut self) -> Result<()> {
        self.left_buffer.clear();
        self.right_buffer.clear();
        self.left_stream.reset()?;
        self.right_stream.reset()
    }
    fn get_stats(&self) -> StreamStats {
        StreamStats::default()
    }
}
