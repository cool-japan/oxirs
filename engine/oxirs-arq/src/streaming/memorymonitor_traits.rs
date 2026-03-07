//! # MemoryMonitor - Trait Implementations
//!
//! This module contains trait implementations for `MemoryMonitor`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::Arc;

use super::types::MemoryMonitor;

impl Clone for MemoryMonitor {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}
