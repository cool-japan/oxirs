//! # StreamingExecutor - accessors Methods
//!
//! This module contains method implementations for `StreamingExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StreamingStats;

use super::streamingexecutor_type::StreamingExecutor;

impl StreamingExecutor {
    /// Get streaming execution statistics
    pub fn get_stats(&self) -> &StreamingStats {
        &self.execution_stats
    }
}
