//! # Advanced Compression Module for TDB Storage
//!
//! Provides advanced compression algorithms including column-store optimizations,
//! bitmap compression, delta encoding, and adaptive compression selection.
//! 
//! This module has been refactored into separate sub-modules for better maintainability.
//! Each compression algorithm is now in its own focused module with comprehensive
//! testing and production-ready error handling.

// Re-export everything from the compression module
pub use compression::*;

// Declare the compression module
pub mod compression;