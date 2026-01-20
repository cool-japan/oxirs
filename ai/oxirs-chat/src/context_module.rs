//! Advanced Context Management for OxiRS Chat
//!
//! Implements intelligent context management with sliding windows, topic tracking,
//! context summarization, and adaptive memory optimization.
//!
//! ## Refactored Module Structure
//!
//! This module has been refactored from a single 2646-line file into a well-organized
//! modular structure. See `context` submodule for implementation details.

mod context;

// Re-export everything from the modular implementation
pub use context::*;
