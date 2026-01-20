//! # Store - is_ready_group Methods
//!
//! This module contains method implementations for `Store`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::collections::{HashMap, HashSet};

impl Store {
    /// Check if the store is ready for operations
    pub fn is_ready(&self) -> bool {
        self.default_store.try_read().is_ok()
            && self.datasets.try_read().is_ok()
            && self.metadata.try_read().is_ok()
    }
}
