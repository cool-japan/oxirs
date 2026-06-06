//! The SHACL `sh:ExpressionConstraintComponent` and its evaluation result.
//!
//! Implements the constraint component itself: applying a [`ShaclExpression`]
//! to each value node and reporting a violation whenever the expression
//! evaluates to a falsy value.
//!
//! Reference: <https://www.w3.org/TR/shacl-af/#ExpressionConstraintComponent>

use serde::{Deserialize, Serialize};

use crate::Result;

use super::expression_constraint_evaluator::{ExpressionContext, ExpressionEvaluator};
use super::expression_constraint_types::{ShaclExpression, ShaclValue};

// ---------------------------------------------------------------------------
// ExpressionConstraintComponent
// ---------------------------------------------------------------------------

/// The SHACL expression constraint component.
///
/// Applies a SHACL expression to each value node. The constraint is violated
/// whenever the expression evaluates to a falsy value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpressionConstraintComponent {
    /// The expression to evaluate
    pub expression: ShaclExpression,
    /// Optional violation message
    pub message: Option<String>,
    /// Whether this constraint is deactivated
    pub deactivated: bool,
}

impl ExpressionConstraintComponent {
    /// Create a new expression constraint.
    pub fn new(expression: ShaclExpression) -> Self {
        Self {
            expression,
            message: None,
            deactivated: false,
        }
    }

    /// Set a violation message (builder pattern).
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Deactivate this constraint (builder pattern).
    pub fn deactivated(mut self) -> Self {
        self.deactivated = true;
        self
    }

    /// Evaluate the expression for a focus node and return whether it is valid.
    pub fn evaluate(&self, ctx: &ExpressionContext) -> Result<ExpressionConstraintResult> {
        if self.deactivated {
            return Ok(ExpressionConstraintResult {
                focus_node: ctx.this_node.clone(),
                is_valid: true,
                value: ShaclValue::Null,
                message: None,
            });
        }

        let value = ExpressionEvaluator::evaluate(&self.expression, ctx)?;
        let is_valid = value.is_truthy();

        let message = if is_valid {
            None
        } else {
            Some(
                self.message
                    .clone()
                    .unwrap_or_else(|| format!("Expression constraint failed: {value}")),
            )
        };

        Ok(ExpressionConstraintResult {
            focus_node: ctx.this_node.clone(),
            is_valid,
            value,
            message,
        })
    }
}

/// Result of evaluating an `ExpressionConstraintComponent`.
#[derive(Debug, Clone)]
pub struct ExpressionConstraintResult {
    /// The focus node that was validated
    pub focus_node: String,
    /// Whether the expression evaluated to a truthy value
    pub is_valid: bool,
    /// The actual evaluated value
    pub value: ShaclValue,
    /// Violation message (None when valid)
    pub message: Option<String>,
}
