//! Enhanced SPARQL 1.1+ Query Algebra — thin facade module.
//!
//! All type definitions live in `sparql_algebra_types` (declared in `query/mod.rs`),
//! algebraic operations in `sparql_algebra_ops`, structural transformations in
//! `sparql_algebra_transform`.  Unit tests are in `sparql_algebra_tests`
//! (also declared in `query/mod.rs`).
//!
//! External code importing from this module receives all public items via
//! the re-exports below.

pub use super::sparql_algebra_ops::*;
pub use super::sparql_algebra_transform::*;
pub use super::sparql_algebra_types::*;
