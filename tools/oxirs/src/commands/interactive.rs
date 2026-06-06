//! Interactive REPL mode for OxiRS (Alpha Implementation)
//!
//! Provides an interactive shell for SPARQL queries with real execution.
//! This module re-exports the public API from the sibling session module.

pub use crate::commands::interactive_session::execute;
