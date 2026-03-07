//! # RegionManager - accessors Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Get local availability zone information
    pub fn get_local_availability_zone(&self) -> &str {
        &self.local_availability_zone
    }
}
