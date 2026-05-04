//! SHACL validation subset for OxiRS WASM.
//!
//! Supported constraint components:
//! - `sh:minCount`, `sh:maxCount`
//! - `sh:datatype`
//! - `sh:pattern`
//! - `sh:minInclusive`, `sh:maxInclusive`
//! - `sh:class`
//! - `sh:nodeKind`
//! - `sh:in`
//! - `sh:hasValue`

pub mod shapes;
pub mod validator;

pub use shapes::{NodeKind, NodeShape, PropertyShape, ShaclConstraint};
pub use validator::{
    DataGraph, RdfNode, Severity, ShaclValidator, Triple, ValidationReport, ValidationResult,
};
