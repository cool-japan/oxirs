//! Tiered storage engine with intelligent data placement
//!
//! This module defines the public interface for a multi-tier storage system
//! that would automatically move data between tiers based on access patterns
//! and age.
//!
//! NOTE: The persistent multi-tier backend is not available in the default
//! (Pure Rust) build. [`TieredStorageEngine::new`] returns
//! [`crate::OxirsError::NotSupported`] so callers can fall back to an
//! in-memory store.

/// Tiered storage engine placeholder.
///
/// The persistent multi-tier backend is unavailable in the Pure Rust build.
pub struct TieredStorageEngine;

impl TieredStorageEngine {
    /// Create a new tiered storage engine.
    ///
    /// The persistent multi-tier backend is not available in this build, so
    /// this always returns [`crate::OxirsError::NotSupported`].
    pub async fn new(_config: crate::storage::StorageConfig) -> Result<Self, crate::OxirsError> {
        Err(crate::OxirsError::NotSupported(
            "TieredStorageEngine persistent backend is not available in this build".to_string(),
        ))
    }
}
