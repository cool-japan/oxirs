//! Range constraints (MinInclusive, MaxInclusive, MinExclusive, MaxExclusive)

use serde::{Deserialize, Serialize};

use oxirs_core::{model::{Literal, Term}, Store};

use crate::Result;
use super::{ConstraintValidator, ConstraintEvaluator, ConstraintContext, ConstraintEvaluationResult};

/// sh:minExclusive constraint - validates minimum exclusive value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinExclusiveConstraint {
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
        _store: &Store,
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
        // Basic comparison - for now just compare string representations
        // TODO: Implement proper typed comparison for numbers, dates, etc.
        Ok(value.value() > min_value.value())
    }
}

/// sh:maxExclusive constraint - validates maximum exclusive value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaxExclusiveConstraint {
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
        _store: &Store,
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
        // Basic comparison - for now just compare string representations
        // TODO: Implement proper typed comparison for numbers, dates, etc.
        Ok(value.value() < max_value.value())
    }
}

/// sh:minInclusive constraint - validates minimum inclusive value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MinInclusiveConstraint {
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
        _store: &Store,
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
        // Basic comparison - for now just compare string representations
        // TODO: Implement proper typed comparison for numbers, dates, etc.
        Ok(value.value() >= min_value.value())
    }
}

/// sh:maxInclusive constraint - validates maximum inclusive value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaxInclusiveConstraint {
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
        _store: &Store,
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
        // Basic comparison - for now just compare string representations
        // TODO: Implement proper typed comparison for numbers, dates, etc.
        Ok(value.value() <= max_value.value())
    }
}