//! # RegionManager - compress_for_transfer_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result as ClusterResult;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Compress data for cross-region transfer using zstd
    pub fn compress_for_transfer(&self, data: &[u8]) -> ClusterResult<Vec<u8>> {
        use zstd::bulk::compress;
        compress(data, 3).map_err(|e| crate::error::ClusterError::Compression(e.to_string()))
    }
}
