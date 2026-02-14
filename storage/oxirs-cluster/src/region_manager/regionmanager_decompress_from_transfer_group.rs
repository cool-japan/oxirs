//! # RegionManager - decompress_from_transfer_group Methods
//!
//! This module contains method implementations for `RegionManager`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result as ClusterResult;

use super::regionmanager_type::RegionManager;

impl RegionManager {
    /// Decompress data received from cross-region transfer
    pub fn decompress_from_transfer(&self, data: &[u8]) -> ClusterResult<Vec<u8>> {
        use zstd::bulk::decompress;
        const MAX_DECOMPRESSED_SIZE: usize = 100 * 1024 * 1024;
        decompress(data, MAX_DECOMPRESSED_SIZE)
            .map_err(|e| crate::error::ClusterError::Compression(e.to_string()))
    }
}
