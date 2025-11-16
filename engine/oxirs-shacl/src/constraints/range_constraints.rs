//! SHACL range constraints for validating numeric and comparable literal values
//!
//! This module implements range-based SHACL constraints that validate whether
//! literal values fall within specified bounds. These constraints work with
//! comparable datatypes including numbers, dates, and times:
//!
//! - [`MinExclusiveConstraint`] - Validates values are greater than a minimum (`sh:minExclusive`)
//! - [`MaxExclusiveConstraint`] - Validates values are less than a maximum (`sh:maxExclusive`)
//! - [`MinInclusiveConstraint`] - Validates values are greater than or equal to a minimum (`sh:minInclusive`)
//! - [`MaxInclusiveConstraint`] - Validates values are less than or equal to a maximum (`sh:maxInclusive`)
//!
//! # Usage
//!
//! ```rust
//! use oxirs_shacl::constraints::range_constraints::*;
//! use oxirs_core::model::Literal;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a constraint for values greater than 0
//! let min_constraint = MinExclusiveConstraint {
//!     min_value: Literal::new_simple_literal("0"),
//! };
//!
//! // Create a constraint for values at most 100
//! let max_constraint = MaxInclusiveConstraint {
//!     max_value: Literal::new_simple_literal("100"),
//! };
//!
//! // Common pattern: percentage range (0 <= value <= 100)
//! let min_percentage = MinInclusiveConstraint {
//!     min_value: Literal::new_simple_literal("0"),
//! };
//! let max_percentage = MaxInclusiveConstraint {
//!     max_value: Literal::new_simple_literal("100"),
//! };
//! # Ok(())
//! # }
//! ```
//!
//! # SHACL Specification
//!
//! These constraints implement the range constraint components from the
//! [SHACL specification](https://www.w3.org/TR/shacl/#core-components-range):
//!
//! - `sh:minExclusive` - Specifies the exclusive minimum value that each value node must have
//! - `sh:maxExclusive` - Specifies the exclusive maximum value that each value node must have
//! - `sh:minInclusive` - Specifies the inclusive minimum value that each value node must have
//! - `sh:maxInclusive` - Specifies the inclusive maximum value that each value node must have
//!
//! # Supported Datatypes
//!
//! Range constraints work with comparable XSD datatypes:
//! - **Numeric**: `xsd:integer`, `xsd:decimal`, `xsd:double`, `xsd:float`
//! - **Temporal**: `xsd:date`, `xsd:dateTime`, `xsd:time`
//! - **Other**: Any datatype with a natural ordering

use serde::{Deserialize, Serialize};

use oxirs_core::{
    model::{Literal, Term},
    Store,
};
use std::cmp::Ordering;

use super::{
    ConstraintContext, ConstraintEvaluationResult, ConstraintEvaluator, ConstraintValidator,
};
use crate::Result;

/// SHACL `sh:minExclusive` constraint that validates values are greater than a minimum.
///
/// This constraint ensures that each literal value is strictly greater than the specified
/// minimum value. The comparison is exclusive, meaning the minimum value itself is not allowed.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - MinExclusive Constraint Component](https://www.w3.org/TR/shacl/#MinExclusiveConstraintComponent):
/// "Specifies the exclusive minimum value that each value node must have."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::range_constraints::MinExclusiveConstraint;
/// use oxirs_core::model::Literal;
///
/// // Values must be greater than 0 (positive numbers only)
/// let positive_constraint = MinExclusiveConstraint {
///     min_value: Literal::new_simple_literal("0"),
/// };
///
/// // Values must be greater than 18 (adult age validation)
/// let adult_age_constraint = MinExclusiveConstraint {
///     min_value: Literal::new_simple_literal("18"),
/// };
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When literal values are > `min_value`
/// - **Fails**: When literal values are <= `min_value`
/// - **Fails**: When values are not literals (IRIs or blank nodes)
/// - **Note**: Currently uses string comparison; typed comparison for numbers/dates is planned
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinExclusiveConstraint {
    /// The exclusive minimum value that all values must exceed
    pub min_value: Literal,
}

impl ConstraintValidator for MinExclusiveConstraint {
    fn validate(&self) -> Result<()> {
        // Value should be a comparable literal (number, date, etc.)
        Ok(())
    }
}

impl ConstraintEvaluator for MinExclusiveConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !self.compare_values_gt(literal, &self.min_value)? {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is not greater than minimum value {}",
                            literal, self.min_value
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value must be a literal for range comparison".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

impl MinExclusiveConstraint {
    fn compare_values_gt(&self, value: &Literal, min_value: &Literal) -> Result<bool> {
        compare_literals(value, min_value).map(|ord| ord == Ordering::Greater)
    }
}

/// SHACL `sh:maxExclusive` constraint that validates values are less than a maximum.
///
/// This constraint ensures that each literal value is strictly less than the specified
/// maximum value. The comparison is exclusive, meaning the maximum value itself is not allowed.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - MaxExclusive Constraint Component](https://www.w3.org/TR/shacl/#MaxExclusiveConstraintComponent):
/// "Specifies the exclusive maximum value that each value node must have."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::range_constraints::MaxExclusiveConstraint;
/// use oxirs_core::model::Literal;
///
/// // Values must be less than 100 (percentage below 100)
/// let below_hundred_constraint = MaxExclusiveConstraint {
///     max_value: Literal::new_simple_literal("100"),
/// };
///
/// // Values must be less than current date (historical dates only)
/// let historical_constraint = MaxExclusiveConstraint {
///     max_value: Literal::new_simple_literal("2025-01-01"),
/// };
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When literal values are < `max_value`
/// - **Fails**: When literal values are >= `max_value`
/// - **Fails**: When values are not literals (IRIs or blank nodes)
/// - **Note**: Currently uses string comparison; typed comparison for numbers/dates is planned
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaxExclusiveConstraint {
    /// The exclusive maximum value that all values must be less than
    pub max_value: Literal,
}

impl ConstraintValidator for MaxExclusiveConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MaxExclusiveConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !self.compare_values_lt(literal, &self.max_value)? {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is not less than maximum value {}",
                            literal, self.max_value
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value must be a literal for range comparison".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

impl MaxExclusiveConstraint {
    fn compare_values_lt(&self, value: &Literal, max_value: &Literal) -> Result<bool> {
        compare_literals(value, max_value).map(|ord| ord == Ordering::Less)
    }
}

/// SHACL `sh:minInclusive` constraint that validates values are greater than or equal to a minimum.
///
/// This constraint ensures that each literal value is greater than or equal to the specified
/// minimum value. The comparison is inclusive, meaning the minimum value itself is allowed.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - MinInclusive Constraint Component](https://www.w3.org/TR/shacl/#MinInclusiveConstraintComponent):
/// "Specifies the inclusive minimum value that each value node must have."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::range_constraints::MinInclusiveConstraint;
/// use oxirs_core::model::Literal;
///
/// // Values must be at least 0 (non-negative numbers)
/// let non_negative_constraint = MinInclusiveConstraint {
///     min_value: Literal::new_simple_literal("0"),
/// };
///
/// // Values must be at least 18 (minimum age including 18)
/// let min_age_constraint = MinInclusiveConstraint {
///     min_value: Literal::new_simple_literal("18"),
/// };
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When literal values are >= `min_value`
/// - **Fails**: When literal values are < `min_value`
/// - **Fails**: When values are not literals (IRIs or blank nodes)
/// - **Note**: Currently uses string comparison; typed comparison for numbers/dates is planned
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinInclusiveConstraint {
    /// The inclusive minimum value that all values must meet or exceed
    pub min_value: Literal,
}

impl ConstraintValidator for MinInclusiveConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MinInclusiveConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !self.compare_values_gte(literal, &self.min_value)? {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is less than minimum value {}",
                            literal, self.min_value
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value must be a literal for range comparison".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

impl MinInclusiveConstraint {
    fn compare_values_gte(&self, value: &Literal, min_value: &Literal) -> Result<bool> {
        compare_literals(value, min_value).map(|ord| ord != Ordering::Less)
    }
}

/// SHACL `sh:maxInclusive` constraint that validates values are less than or equal to a maximum.
///
/// This constraint ensures that each literal value is less than or equal to the specified
/// maximum value. The comparison is inclusive, meaning the maximum value itself is allowed.
///
/// # SHACL Specification
///
/// From [SHACL Core Components - MaxInclusive Constraint Component](https://www.w3.org/TR/shacl/#MaxInclusiveConstraintComponent):
/// "Specifies the inclusive maximum value that each value node must have."
///
/// # Example
///
/// ```rust
/// use oxirs_shacl::constraints::range_constraints::MaxInclusiveConstraint;
/// use oxirs_core::model::Literal;
///
/// // Values must be at most 100 (percentage validation)
/// let percentage_constraint = MaxInclusiveConstraint {
///     max_value: Literal::new_simple_literal("100"),
/// };
///
/// // Values must be at most 5 (rating scale 1-5)
/// let rating_constraint = MaxInclusiveConstraint {
///     max_value: Literal::new_simple_literal("5"),
/// };
/// ```
///
/// # Validation Behavior
///
/// - **Passes**: When literal values are <= `max_value`
/// - **Fails**: When literal values are > `max_value`
/// - **Fails**: When values are not literals (IRIs or blank nodes)
/// - **Note**: Currently uses string comparison; typed comparison for numbers/dates is planned
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaxInclusiveConstraint {
    /// The inclusive maximum value that all values must not exceed
    pub max_value: Literal,
}

impl ConstraintValidator for MaxInclusiveConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MaxInclusiveConstraint {
    fn evaluate(
        &self,
        _store: &dyn Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        for value in &context.values {
            if let Term::Literal(literal) = value {
                if !self.compare_values_lte(literal, &self.max_value)? {
                    return Ok(ConstraintEvaluationResult::violated(
                        Some(value.clone()),
                        Some(format!(
                            "Value {} is greater than maximum value {}",
                            literal, self.max_value
                        )),
                    ));
                }
            } else {
                return Ok(ConstraintEvaluationResult::violated(
                    Some(value.clone()),
                    Some("Value must be a literal for range comparison".to_string()),
                ));
            }
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

impl MaxInclusiveConstraint {
    fn compare_values_lte(&self, value: &Literal, max_value: &Literal) -> Result<bool> {
        compare_literals(value, max_value).map(|ord| ord != Ordering::Greater)
    }
}

/// Compare two literals with proper type handling
fn compare_literals(left: &Literal, right: &Literal) -> Result<Ordering> {
    use oxirs_core::vocab::xsd;

    let left_datatype = left.datatype();
    let right_datatype = right.datatype();

    // Both literals should have the same datatype for meaningful comparison
    if left_datatype != right_datatype {
        // Fall back to lexical comparison if datatypes differ
        return Ok(left.value().cmp(right.value()));
    }

    // Handle numeric types
    if left_datatype == *xsd::INTEGER
        || left_datatype == *xsd::INT
        || left_datatype == *xsd::LONG
        || left_datatype == *xsd::SHORT
        || left_datatype == *xsd::BYTE
    {
        // Integer comparison
        let left_val = left.value().parse::<i64>().map_err(|e| {
            crate::ShaclError::ConstraintValidation(format!("Invalid integer value: {}", e))
        })?;
        let right_val = right.value().parse::<i64>().map_err(|e| {
            crate::ShaclError::ConstraintValidation(format!("Invalid integer value: {}", e))
        })?;
        return Ok(left_val.cmp(&right_val));
    }

    if left_datatype == *xsd::DECIMAL
        || left_datatype == *xsd::DOUBLE
        || left_datatype == *xsd::FLOAT
    {
        // Floating point comparison
        let left_val = left.value().parse::<f64>().map_err(|e| {
            crate::ShaclError::ConstraintValidation(format!("Invalid decimal value: {}", e))
        })?;
        let right_val = right.value().parse::<f64>().map_err(|e| {
            crate::ShaclError::ConstraintValidation(format!("Invalid decimal value: {}", e))
        })?;

        // Handle NaN and infinity
        if left_val.is_nan() || right_val.is_nan() {
            return Err(crate::ShaclError::ConstraintValidation(
                "Cannot compare NaN values".to_string(),
            ));
        }

        return Ok(left_val.partial_cmp(&right_val).unwrap_or(Ordering::Equal));
    }

    if left_datatype == *xsd::BOOLEAN {
        // Boolean comparison (false < true)
        let left_val = left.value().parse::<bool>().map_err(|e| {
            crate::ShaclError::ConstraintValidation(format!("Invalid boolean value: {}", e))
        })?;
        let right_val = right.value().parse::<bool>().map_err(|e| {
            crate::ShaclError::ConstraintValidation(format!("Invalid boolean value: {}", e))
        })?;
        return Ok(left_val.cmp(&right_val));
    }

    if left_datatype == *xsd::DATE
        || left_datatype == *xsd::DATE_TIME
        || left_datatype == *xsd::TIME
    {
        // Date/time comparison - use lexical ordering which works for ISO 8601 format
        return Ok(left.value().cmp(right.value()));
    }

    // Default: lexical comparison for strings and other types
    Ok(left.value().cmp(right.value()))
}
