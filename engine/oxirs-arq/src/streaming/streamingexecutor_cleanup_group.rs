//! # StreamingExecutor - cleanup_group Methods
//!
//! This module contains method implementations for `StreamingExecutor`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::streamingexecutor_type::StreamingExecutor;

impl StreamingExecutor {
    /// Clean up temporary files and resources
    pub fn cleanup(&mut self) -> Result<()> {
        self.active_streams.clear();
        self.spill_manager
            .lock()
            .expect("lock poisoned")
            .cleanup_all()?;
        Ok(())
    }
}
