//! Cardinality constraints (MinCount, MaxCount)

use serde::{Deserialize, Serialize};

use oxirs_core::Store;

use crate::Result;
use super::{ConstraintValidator, ConstraintEvaluator, ConstraintContext, ConstraintEvaluationResult};

/// sh:minCount constraint - validates minimum number of values
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinCountConstraint {
    pub min_count: u32,
}

impl ConstraintValidator for MinCountConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MinCountConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let value_count = context.values.len() as u32;
        if value_count < self.min_count {
            return Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Expected at least {} values, but found {}",
                    self.min_count, value_count
                )),
            ));
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}

/// sh:maxCount constraint - validates maximum number of values
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxCountConstraint {
    pub max_count: u32,
}

impl ConstraintValidator for MaxCountConstraint {
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl ConstraintEvaluator for MaxCountConstraint {
    fn evaluate(
        &self,
        store: &Store,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let value_count = context.values.len() as u32;
        if value_count > self.max_count {
            return Ok(ConstraintEvaluationResult::violated(
                None,
                Some(format!(
                    "Expected at most {} values, but found {}",
                    self.max_count, value_count
                )),
            ));
        }
        Ok(ConstraintEvaluationResult::satisfied())
    }
}