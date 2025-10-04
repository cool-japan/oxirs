//! Advanced Context Management for OxiRS Chat
//!
//! Implements intelligent context management with sliding windows, topic tracking,
//! context summarization, and adaptive memory optimization.
//!
//! ## Module Organization
//!
//! - `config` - Context management configuration
//! - `types` - Shared types and structures
//! - `neuromorphic` - Brain-inspired context processing
//! - `components` - Topic tracking, importance scoring, summarization
//! - `window` - Sliding window management
//! - `manager` - Main AdvancedContextManager implementation

pub mod components;
pub mod config;
pub mod manager;
pub mod neuromorphic;
pub mod types;
pub mod window;

// Re-export main types
pub use config::ContextConfig;
pub use manager::AdvancedContextManager;
pub use types::*;

// Re-export component types for convenience
pub use components::{ImportanceScorer, MemoryOptimizer, SummarizationEngine, TopicTracker};
pub use neuromorphic::NeuromorphicContextManager;
