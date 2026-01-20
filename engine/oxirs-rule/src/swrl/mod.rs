//! SWRL (Semantic Web Rule Language) Implementation
//!
//! Implementation of SWRL rule parsing, execution, and built-in predicates.
//! Supports the full SWRL specification including custom built-ins.
//!
//! ## Built-in Functions
//!
//! SWRL built-in functions are organized into semantic modules:
//! - **Comparison** - `equal`, `notEqual`, `lessThan`, `greaterThan`, `between`
//! - **Arithmetic** - Math operations including basic, trigonometric, and statistical functions
//! - **String** - String manipulation and pattern matching
//! - **DateTime** - Date/time operations and temporal relations
//! - **Type Checking** - Type predicates and safe conversions
//! - **List** - List operations including set operations
//! - **Geographic** - Geospatial operations (distance, area, containment)
//! - **Encoding** - Hashing, base64, URI encoding
//!
//! See the [`builtins`] module for the complete function catalog (114 built-in functions).

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
