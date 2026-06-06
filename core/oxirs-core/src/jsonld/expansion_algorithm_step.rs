//! Individual expansion step helpers for the JSON-LD Expansion Algorithm.
//!
//! This module provides helper functions for specific expansion steps
//! as described by the W3C JSON-LD API specification:
//! https://www.w3.org/TR/json-ld-api/#expansion-algorithms
//!
//! The main state-machine logic remains in `expansion_algorithm.rs`.
//! The per-element helpers (`on_literal_value`, `expand_value`) that
//! are called from within `convert_event` are delegated to the
//! `JsonLdExpansionConverter` impl directly; this module documents the
//! conceptual grouping and re-exports any step-level utilities.

pub use super::expansion_algorithm::JsonLdExpansionConverter;
