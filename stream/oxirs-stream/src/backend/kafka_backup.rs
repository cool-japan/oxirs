//! # Apache Kafka Backend - Ultra-High Performance
//!
//! This module has been refactored into a modular structure to comply with file size limits.
//! The implementation is now split across the kafka_backup/ directory.

// Re-export the modular kafka backup components
pub use kafka_backup::*;