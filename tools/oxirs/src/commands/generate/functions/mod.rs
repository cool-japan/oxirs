//! RDF dataset generation functions
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

mod api;
mod domain_data;
mod random_data;
mod schema_detect;
mod schema_owl;
mod schema_rdfs;
mod schema_shacl;

// Re-export public API
pub use api::{from_owl, from_rdfs, from_shacl, run};
