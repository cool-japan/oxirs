//! Aggregate functions for SPARQL 1.2
//!
//! These functions require special handling in query evaluation
//! as they operate on groups of solutions rather than individual terms.

use crate::model::Term;
use crate::OxirsError;

/// COUNT - Count solutions
pub(super) fn fn_count(_args: &[Term]) -> Result<Term, OxirsError> {
    // Aggregate functions need special handling in query evaluation
    Err(OxirsError::Query(
        "COUNT is an aggregate function".to_string(),
    ))
}

/// SUM - Sum numeric values
pub(super) fn fn_sum(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "SUM is an aggregate function".to_string(),
    ))
}

/// AVG - Calculate average
pub(super) fn fn_avg(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "AVG is an aggregate function".to_string(),
    ))
}

/// MIN - Find minimum value
pub(super) fn fn_min(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "MIN is an aggregate function".to_string(),
    ))
}

/// MAX - Find maximum value
pub(super) fn fn_max(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "MAX is an aggregate function".to_string(),
    ))
}

/// GROUP_CONCAT - Concatenate group values
pub(super) fn fn_group_concat(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "GROUP_CONCAT is an aggregate function".to_string(),
    ))
}

/// SAMPLE - Get sample value from group
pub(super) fn fn_sample(_args: &[Term]) -> Result<Term, OxirsError> {
    Err(OxirsError::Query(
        "SAMPLE is an aggregate function".to_string(),
    ))
}
