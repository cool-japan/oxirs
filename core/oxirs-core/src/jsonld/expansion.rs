//! JSON-LD Expansion Algorithm
//!
//! Implements the [Expansion Algorithm](https://www.w3.org/TR/json-ld-api/#expansion-algorithms)
//! as defined by the W3C JSON-LD API specification.
//!
//! # Module layout
//!
//! - [expansion_context]: State machine types, output event enum, value type, and helpers.
//! - [expansion_algorithm]: The JsonLdExpansionConverter struct and its full implementation.
//! - [expansion_tests]: Unit tests.

pub use super::expansion_algorithm::JsonLdExpansionConverter;
pub use super::expansion_context::{JsonLdEvent, JsonLdValue};
