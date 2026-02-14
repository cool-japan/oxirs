//! # StreamingExecutor - execute_streaming_group Methods
//!
//! This module contains method implementations for `StreamingExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::Algebra;
use anyhow::Result;
use std::time::Instant;
use tracing::{span, Level};

use super::functions::DataStream;

use super::streamingexecutor_type::StreamingExecutor;

impl StreamingExecutor {
    /// Execute algebra with streaming support
    pub fn execute_streaming(&mut self, algebra: &Algebra) -> Result<Box<dyn DataStream>> {
        let _span = span!(Level::INFO, "streaming_execution").entered();
        let start_time = Instant::now();
        let result_stream = self.execute_algebra_streaming(algebra)?;
        self.execution_stats.total_execution_time += start_time.elapsed();
        Ok(result_stream)
    }
}
