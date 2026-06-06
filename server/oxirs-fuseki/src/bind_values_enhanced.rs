//! Enhanced BIND and VALUES clause processing for SPARQL 1.2
//!
//! This module implements advanced features for BIND expressions and VALUES clauses,
//! including expression optimization, value set management, and performance improvements.
//!
//! ## Module layout
//!
//! This module is a thin facade (Round 32 refactor). The implementation lives in
//! sibling modules:
//! - [`bind_values_enhanced_types`](crate::bind_values_enhanced_types): all type and enum definitions
//! - [`bind_values_enhanced_bind`](crate::bind_values_enhanced_bind): [`EnhancedBindProcessor`] impl
//! - [`bind_values_enhanced_values`](crate::bind_values_enhanced_values): [`EnhancedValuesProcessor`] impl
//! - [`bind_values_enhanced_optim`](crate::bind_values_enhanced_optim): supporting optimizer/cache/strategy impls

pub use crate::bind_values_enhanced_types::*;
