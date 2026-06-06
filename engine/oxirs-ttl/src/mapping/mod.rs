//! RML-inspired RDF Mapping Language support
//!
//! Maps non-RDF data sources (CSV, JSON, inline values) to RDF triples.
//! Inspired by the W3C RDF Mapping Language (RML) specification:
//! <https://rml.io/specs/rml/>
//!
//! # Example
//!
//! ```rust
//! use oxirs_ttl::mapping::{MappingEngine, MappingRuleBuilder, ObjectSpec};
//!
//! let csv_data = "id,name,age\n1,Alice,30\n2,Bob,25";
//!
//! let rule = MappingRuleBuilder::new("persons")
//!     .csv_source(csv_data)
//!     .subject_template("http://example.org/person/{id}")
//!     .map(
//!         "http://xmlns.com/foaf/0.1/name",
//!         ObjectSpec::Column("name".to_string()),
//!     )
//!     .map(
//!         "http://xmlns.com/foaf/0.1/age",
//!         ObjectSpec::TypedColumn {
//!             column: "age".to_string(),
//!             datatype: "http://www.w3.org/2001/XMLSchema#integer".to_string(),
//!         },
//!     )
//!     .build();
//!
//! let engine = MappingEngine::new();
//! let triples = engine.execute(&rule).expect("should succeed");
//! assert_eq!(triples.len(), 4); // 2 rows × 2 predicates
//! ```

pub mod mapping_tests;
pub mod mapping_transformers;
pub mod mapping_types;

pub use mapping_transformers::*;
pub use mapping_types::*;
