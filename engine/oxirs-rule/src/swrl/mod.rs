//! SWRL (Semantic Web Rule Language) Implementation
//!
//! Implementation of SWRL rule parsing, execution, and built-in predicates.
//! Supports the full SWRL specification including custom built-ins.

// Module declarations
pub mod builtins;
pub mod engine;
pub mod registry;
pub mod stats;
pub mod temporal;
pub mod types;
pub mod vocabulary;

// Re-export main types
pub use builtins::*;
pub use engine::SwrlEngine;
pub use registry::{BuiltinMetadata, CustomBuiltinRegistry};
pub use stats::SwrlStats;
pub use temporal::TemporalInterval;
pub use types::{BuiltinFunction, SwrlArgument, SwrlAtom, SwrlContext, SwrlRule};
