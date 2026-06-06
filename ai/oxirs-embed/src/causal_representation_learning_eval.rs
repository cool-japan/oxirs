//! Causal Representation Learning — Evaluation
//!
//! Evaluation utilities: causal disentanglement score, intervention accuracy, and
//! counterfactual consistency metrics.

use crate::causal_representation_learning_model::CausalRepresentationModel;
use crate::causal_representation_learning_types::{CounterfactualQuery, Intervention};
use anyhow::Result;
use scirs2_core::ndarray_ext::Array1;
use std::collections::HashMap;

/// Type alias to reduce repetition in function signatures.
type InterventionTriple = (HashMap<String, f32>, Intervention, HashMap<String, f32>);

/// Summary of evaluation results for a causal representation model.
#[derive(Debug, Clone)]
pub struct CausalEvalResult {
    /// Mean absolute error of predicted vs true intervention effects.
    pub intervention_mae: f32,
    /// Fraction of counterfactual queries that are self-consistent.
    pub counterfactual_consistency: f32,
    /// MIG-style disentanglement score (higher is better).
    pub disentanglement_score: f32,
}

/// Evaluate a trained model against a set of held-out interventional observations.
///
/// `interventions` is a slice of `(factual data, intervention, expected post-intervention values)`
/// triples used to measure how well the model predicts intervention effects.
pub fn evaluate_intervention_accuracy(
    model: &CausalRepresentationModel,
    interventions: &[InterventionTriple],
) -> Result<f32> {
    if interventions.is_empty() {
        return Ok(0.0);
    }

    let mut total_error = 0.0f32;
    let mut count = 0usize;

    for (_factual, intervention, expected) in interventions {
        let predicted = model.intervene(intervention)?;
        for (var, &expected_val) in expected {
            if let Some(&predicted_val) = predicted.get(var) {
                total_error += (predicted_val - expected_val).abs();
                count += 1;
            }
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(total_error / count as f32)
    }
}

/// Evaluate counterfactual consistency: a model is consistent if, when we intervene with the
/// factual values, we approximately recover the factual outcome.
///
/// Returns the fraction of query variables where the counterfactual output (under the identity
/// intervention) is within `tolerance` of the observed factual value.
pub fn evaluate_counterfactual_consistency(
    model: &CausalRepresentationModel,
    factual_observations: &[HashMap<String, f32>],
    tolerance: f32,
) -> Result<f32> {
    if factual_observations.is_empty() {
        return Ok(0.0);
    }

    let mut consistent = 0usize;
    let mut total = 0usize;

    for observation in factual_observations {
        // Build a counterfactual query that re-applies the factual values as intervention.
        let query_vars: Vec<String> = observation.keys().cloned().collect();
        if query_vars.is_empty() {
            continue;
        }
        let target_var = query_vars[0].clone();
        if let Some(&target_val) = observation.get(&target_var) {
            let intervention = Intervention::new(
                vec![target_var.clone()],
                Array1::from_vec(vec![target_val]),
                crate::causal_representation_learning_types::InterventionType::Do,
            );
            let query = CounterfactualQuery {
                factual_evidence: observation.clone(),
                intervention,
                query_variables: query_vars.iter().skip(1).cloned().collect(),
            };
            let cf_result = model.answer_counterfactual(&query)?;
            for (var, &cf_val) in &cf_result {
                if let Some(&factual_val) = observation.get(var) {
                    total += 1;
                    if (cf_val - factual_val).abs() <= tolerance {
                        consistent += 1;
                    }
                }
            }
        }
    }

    if total == 0 {
        Ok(1.0) // vacuously consistent
    } else {
        Ok(consistent as f32 / total as f32)
    }
}

/// Compute a proxy disentanglement score based on the variance of latent factors.
///
/// A higher score suggests more independent (disentangled) factors.  This is a simplified
/// Mutual Information Gap (MIG) proxy: we compute the ratio of the top factor variance to
/// the mean variance over all factors.
pub fn compute_disentanglement_score(model: &CausalRepresentationModel) -> Result<f32> {
    let factors = &model.latent_factors;
    if factors.nrows() == 0 || factors.ncols() == 0 {
        return Ok(0.0);
    }

    let n_factors = factors.ncols();
    let mut variances = Vec::with_capacity(n_factors);

    for j in 0..n_factors {
        let col: Vec<f32> = factors.column(j).to_vec();
        let mean = col.iter().sum::<f32>() / col.len() as f32;
        let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / col.len() as f32;
        variances.push(var);
    }

    variances.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let top_var = variances[0];
    let mean_var = variances.iter().sum::<f32>() / variances.len() as f32;

    if mean_var == 0.0 {
        Ok(0.0)
    } else {
        // Ratio clipped to [0, 1]
        Ok((top_var / mean_var - 1.0).clamp(0.0, 1.0))
    }
}

/// Run the full evaluation suite and return a summary.
pub fn full_evaluation(
    model: &CausalRepresentationModel,
    interventions: &[InterventionTriple],
    factual_observations: &[HashMap<String, f32>],
    cf_tolerance: f32,
) -> Result<CausalEvalResult> {
    let intervention_mae = evaluate_intervention_accuracy(model, interventions)?;
    let counterfactual_consistency =
        evaluate_counterfactual_consistency(model, factual_observations, cf_tolerance)?;
    let disentanglement_score = compute_disentanglement_score(model)?;

    Ok(CausalEvalResult {
        intervention_mae,
        counterfactual_consistency,
        disentanglement_score,
    })
}
