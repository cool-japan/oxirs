//! Neural-Symbolic Integration — custom loss functions
//!
//! This module implements custom loss functions for neural-symbolic learning:
//! - Semantic loss (MSE + constraint violation + rule consistency)
//! - Constraint satisfaction loss
//! - Symbolic regularization terms

use anyhow::Result;
use scirs2_core::ndarray_ext::Array1;
use std::collections::HashMap;

use crate::neural_symbolic_integration::{KnowledgeRule, LogicalFormula};

/// Compute the full semantic loss combining MSE, constraint violation, and rule
/// consistency terms.
///
/// # Arguments
/// * `predictions` – model output vector
/// * `targets`     – ground-truth vector
/// * `constraints` – logical constraints that should be satisfied by predictions
/// * `knowledge_base` – rules whose consequents should be consistent with predictions
///
/// # Returns
/// Scalar loss value (≥ 0).
pub fn compute_semantic_loss(
    predictions: &Array1<f32>,
    targets: &Array1<f32>,
    constraints: &[LogicalFormula],
    knowledge_base: &[KnowledgeRule],
) -> Result<f32> {
    let mse_loss = compute_mse_loss(predictions, targets);
    let constraint_loss = compute_constraint_violation_loss(predictions, constraints);
    let rule_loss = compute_rule_consistency_loss(predictions, knowledge_base);

    let total_loss = mse_loss + 0.1 * constraint_loss + 0.1 * rule_loss;
    Ok(total_loss)
}

/// Mean Squared Error loss between predictions and targets.
pub fn compute_mse_loss(predictions: &Array1<f32>, targets: &Array1<f32>) -> f32 {
    let diff = predictions - targets;
    diff.dot(&diff) / predictions.len() as f32
}

/// Constraint violation loss: squared unsatisfaction averaged over constraints.
///
/// For each constraint c and assignment from `predictions`:
///   violation = (1 - c.evaluate(facts))^2  if not fully satisfied
pub fn compute_constraint_violation_loss(
    predictions: &Array1<f32>,
    constraints: &[LogicalFormula],
) -> f32 {
    if constraints.is_empty() {
        return 0.0;
    }

    let mut facts = HashMap::new();
    for (i, &value) in predictions.iter().enumerate() {
        facts.insert(format!("output_{i}"), value);
    }

    let total_violation: f32 = constraints
        .iter()
        .map(|constraint| {
            let satisfaction: f32 = constraint.evaluate(&facts);
            if satisfaction < 1.0 {
                (1.0 - satisfaction).powi(2)
            } else {
                0.0
            }
        })
        .sum();

    total_violation / constraints.len() as f32
}

/// Rule consistency loss: measures how inconsistent rule-predicted outputs are
/// with actual prediction values, weighted by rule importance.
///
/// For each rule that fires on `predictions` (treated as inputs):
///   inconsistency = (predicted_value - actual_prediction[index])^2 * rule.weight
pub fn compute_rule_consistency_loss(
    predictions: &Array1<f32>,
    knowledge_base: &[KnowledgeRule],
) -> f32 {
    if knowledge_base.is_empty() {
        return 0.0;
    }

    let mut facts = HashMap::new();
    for (i, &value) in predictions.iter().enumerate() {
        facts.insert(format!("input_{i}"), value);
    }

    let total_inconsistency: f32 = knowledge_base
        .iter()
        .filter_map(|rule| {
            rule.apply(&facts).and_then(|(predicate, predicted_value)| {
                predicate
                    .strip_prefix("output_")
                    .and_then(|s| s.parse::<usize>().ok())
                    .and_then(|index| {
                        if index < predictions.len() {
                            let actual_value = predictions[index];
                            Some((predicted_value - actual_value).powi(2) * rule.weight)
                        } else {
                            None
                        }
                    })
            })
        })
        .sum();

    total_inconsistency / knowledge_base.len() as f32
}

/// Symbolic regularization term: encourages model outputs to satisfy logical
/// constraints by penalizing constraint unsatisfaction directly (L1-style).
pub fn symbolic_regularization(
    predictions: &Array1<f32>,
    constraints: &[LogicalFormula],
    lambda: f32,
) -> f32 {
    if constraints.is_empty() {
        return 0.0;
    }

    let mut facts = HashMap::new();
    for (i, &value) in predictions.iter().enumerate() {
        facts.insert(format!("output_{i}"), value);
    }

    let total: f32 = constraints
        .iter()
        .map(|c| {
            let satisfaction: f32 = c.evaluate(&facts);
            (1.0 - satisfaction).max(0.0)
        })
        .sum();

    lambda * total / constraints.len() as f32
}

/// Weighted logic loss for Logic Tensor Networks.
///
/// Given an array of (formula, weight) pairs and a fact assignment,
/// returns the aggregate weighted unsatisfaction.
pub fn logic_tensor_loss(
    predictions: &Array1<f32>,
    weighted_constraints: &[(LogicalFormula, f32)],
) -> f32 {
    if weighted_constraints.is_empty() {
        return 0.0;
    }

    let mut facts = HashMap::new();
    for (i, &value) in predictions.iter().enumerate() {
        facts.insert(format!("output_{i}"), value);
    }

    weighted_constraints
        .iter()
        .map(|(formula, weight)| {
            let satisfaction: f32 = formula.evaluate(&facts);
            weight * (1.0 - satisfaction).powi(2)
        })
        .sum::<f32>()
        / weighted_constraints.len() as f32
}
