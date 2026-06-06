//! [JSON-LD](https://www.w3.org/TR/json-ld/) parsing into RDF quads — thin facade module.
//!
//! The implementation lives in cohesive sibling modules:
//! - [`to_rdf_parser`](super::to_rdf_parser): the [`JsonLdParser`] builder.
//! - [`to_rdf_readers`](super::to_rdf_readers): the streaming parser iterators
//!   ([`ReaderJsonLdParser`], [`TokioAsyncReaderJsonLdParser`],
//!   [`SliceJsonLdParser`]) and the [`JsonLdPrefixesIter`] prefix iterator.
//! - [`to_rdf_converter`](super::to_rdf_converter): the expanded-event to RDF
//!   quad conversion state machine.
//!
//! External code importing from this module receives all public items via
//! the re-exports below.

pub use super::to_rdf_parser::*;
pub use super::to_rdf_readers::*;
