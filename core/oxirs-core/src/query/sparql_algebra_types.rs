//! SPARQL algebra node types — thin facade module.
//!
//! The concrete enums, structs and their `Display`/SSE implementations live in
//! sibling modules (declared in `query/mod.rs`):
//!
//! - [`sparql_algebra_types_paths`](super::sparql_algebra_types_paths): `PropertyPathExpression`
//! - [`sparql_algebra_types_expr`](super::sparql_algebra_types_expr): `Expression`,
//!   `FunctionExpression`, `BuiltInFunction` and the internal SSE helpers
//! - [`sparql_algebra_types_pattern`](super::sparql_algebra_types_pattern): `GraphPattern`
//! - [`sparql_algebra_types_terms`](super::sparql_algebra_types_terms): `GroundTerm`,
//!   `GroundTriple`, `GroundSubject`, `TriplePattern`, `TermPattern`,
//!   `NamedNodePattern`, `OrderExpression`, `AggregateExpression`
//!
//! All public items are re-exported below so existing imports of
//! `crate::query::sparql_algebra_types::*` continue to resolve unchanged.

pub use super::sparql_algebra_types_expr::*;
pub use super::sparql_algebra_types_paths::*;
pub use super::sparql_algebra_types_pattern::*;
pub use super::sparql_algebra_types_terms::*;
