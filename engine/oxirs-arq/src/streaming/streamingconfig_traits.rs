//! # StreamingConfig - Trait Implementations
//!
//! This module contains trait implementations for `StreamingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StreamingConfig;

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: 512 * 1024 * 1024,
            spill_threshold: 0.8,
            batch_size: 10000,
            parallel_workers: 4,
            compression_level: 6,
            enable_memory_mapping: true,
            io_buffer_size: 64 * 1024,
            adaptive_batching: true,
        }
    }
}
