//! SWRL Built-in Functions Module
//!
//! This module organizes SWRL built-in functions into semantic categories:
//! - `utils`: Helper functions for value extraction
//! - `comparison`: Comparison operations (equal, less_than, etc.)
//! - `arithmetic`: Mathematical operations (add, multiply, sin, etc.)
//! - `string`: String manipulation (concat, upper_case, etc.)
//! - `datetime`: Date/time operations (date, time, temporal relations, etc.)
//! - `type_check`: Type checking and conversion (is_integer, int_value, etc.)
//! - `list`: List operations (list_concat, list_sort, etc.)
//! - `geo`: Geographic operations (distance, within, etc.)
//! - `encoding`: Encoding/decoding operations (hash, base64, uri encoding)
//! - `boolean`: Boolean operations (boolean_value)
//! - `lang`: Language tag operations (lang_matches)

pub mod arithmetic;
pub mod boolean;
pub mod comparison;
pub mod datetime;
pub mod encoding;
pub mod geo;
pub mod lang;
pub mod list;
pub mod string;
pub mod type_check;
pub mod utils;

// Re-export all functions for crate-internal access
pub use arithmetic::*; // Contains pub functions (builtin_add, builtin_multiply, builtin_pow)
pub(crate) use boolean::*;
pub(crate) use comparison::*;
pub(crate) use datetime::*;
pub(crate) use encoding::*;
pub(crate) use geo::*;
pub(crate) use lang::*;
pub(crate) use list::*;
pub use string::*; // Contains pub functions (builtin_string_concat, builtin_upper_case)
pub(crate) use type_check::*;

// Tests module
#[cfg(test)]
mod tests;
