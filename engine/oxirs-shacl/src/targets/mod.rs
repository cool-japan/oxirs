//! SHACL target selection module
//!
//! This module provides target selection functionality for SHACL validation.
//! It has been refactored from a single large file into smaller, focused modules
//! to comply with the 2000-line refactoring policy.

pub mod optimization;
pub mod selector;
pub mod target_impl;
pub mod types;

// Re-export main types and functionality
pub use optimization::*;
pub use selector::TargetSelector;
pub use types::*;

// Import the implementation to make methods available
// Note: Temporarily commented out due to unused import warning
// pub use target_impl::*;
