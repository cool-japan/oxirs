//! SPARQL 1.1 Conformance Test Framework
//!
//! Provides W3C-aligned conformance tests for the oxirs-arq query engine.
//! Tests are organized by feature group as defined by the SPARQL 1.1 specification.

pub mod framework;
pub mod helpers;
pub mod tests_aggregates;
pub mod tests_basic;
pub mod tests_construct;
pub mod tests_datetime;
pub mod tests_describe;
pub mod tests_filter;
pub mod tests_functions;
pub mod tests_negation;
pub mod tests_optional_union;
pub mod tests_property_paths;
pub mod tests_subquery;
pub mod tests_type_system;
pub mod tests_update;
pub mod tests_values;

pub use framework::{
    ConformanceGroup, ConformanceResult, ConformanceTest, ConformanceTestError,
    ConformanceTestRunner,
};
