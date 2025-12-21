//! SPARQL 1.2 built-in functions and extensions
//!
//! This module implements the extended function library for SPARQL 1.2,
//! including new string functions, math functions, and advanced features.

mod aggregate;
mod bitwise;
mod datetime;
mod hash;
mod numeric;
mod registry;
mod string;
mod type_check;

// Re-export public types and the registry
pub use registry::{
    ArgumentType, CustomFunction, FunctionImpl, FunctionMetadata, FunctionRegistry,
    FunctionStatistics, NativeFunction, ReturnType,
};
