//! SHACL validation engine implementation
//!
//! This module implements the core validation engine that orchestrates SHACL validation.
//! 
//! This file re-exports the validation module for backward compatibility.

// Re-export everything from the validation module
pub use crate::validation::{
    engine::*,
    constraint_validators::*,
    stats::*,
    cache::*,
    utils::*,
    ConstraintCacheKey,
    ConstraintEvaluationResult,
    ValidationViolation,
};

// For backward compatibility, also export the main types directly
pub use crate::validation::{
    ValidationEngine,
    ValidationStats,
    QualifiedValidationStats,
    ConstraintCache,
    InheritanceCache,
    CacheManager,
    CacheConfig,
    CacheStatistics,
};

// Re-export utility functions
pub use crate::validation::utils::{
    format_term_for_sparql,
    format_term_for_message,
    is_numeric_term,
    is_boolean_term,
    is_datetime_term,
    parse_numeric_value,
    parse_boolean_value,
    terms_equivalent,
    term_to_canonical_string,
};