//! JSON-LD 1.1 processing for SAMM Aspect Models.
//!
//! This module provides compaction and framing operations as defined by the
//! [W3C JSON-LD 1.1 specification][spec] and referenced in the
//! [SAMM specification §mapping-to-json-ld][samm-spec].
//!
//! ## Modules
//!
//! - [`compaction`] – IRI compaction algorithm: replaces expanded IRIs with
//!   compact prefix:localname representations from a `@context`.
//! - [`framing`] – Document framing algorithm: reshapes a flat `@graph` into
//!   a nested, application-specific tree structure.
//!
//! ## Quick start
//!
//! ```rust
//! use oxirs_samm::jsonld::{JsonLdCompactor, JsonLdFramer};
//! use serde_json::json;
//!
//! let ctx = json!({ "ex": "http://example.org/" });
//! let doc = json!({
//!     "@id":   "http://example.org/resource",
//!     "@type": ["http://example.org/Thing"]
//! });
//!
//! let compactor = JsonLdCompactor::new(ctx.clone());
//! let compacted = compactor.compact(&doc, &ctx).expect("compaction should succeed");
//! assert_eq!(compacted["@id"], "ex:resource");
//!
//! let frame = json!({ "@type": ["http://example.org/Thing"] });
//! let framer = JsonLdFramer;
//! let framed = framer.frame(&doc, &frame).expect("framing should succeed");
//! assert!(!framed["@graph"].as_array().expect("@graph").is_empty());
//! ```
//!
//! [spec]: https://www.w3.org/TR/json-ld11/
//! [samm-spec]: https://eclipse-esmf.github.io/samm-specification/2.1.0/index.html#mapping-to-json-ld

pub mod compaction;
pub mod framing;

// Re-export the primary public types for ergonomic access.
pub use compaction::{JsonLdCompactor, JsonLdError};
pub use framing::JsonLdFramer;
