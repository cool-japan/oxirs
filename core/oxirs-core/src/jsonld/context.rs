//! JSON-LD context processing — thin facade module.
//!
//! The implementation lives in cohesive sibling modules:
//! - [`context_types`](super::context_types): core data types ([`JsonNode`],
//!   [`JsonLdContext`], [`JsonLdTermDefinition`], [`JsonLdContextProcessor`],
//!   [`JsonLdLoadDocumentOptions`], [`JsonLdRemoteDocument`]).
//! - [`context_core`](super::context_core): the [Context Processing
//!   Algorithm](https://www.w3.org/TR/json-ld-api/#algorithm), term-definition
//!   creation, IRI expansion and JSON event conversion helpers.
//!
//! External code importing from this module receives all public items via
//! the re-exports below.

pub use super::context_core::*;
pub use super::context_types::*;
