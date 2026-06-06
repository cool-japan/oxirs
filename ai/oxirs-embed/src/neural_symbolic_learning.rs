//! Neural-Symbolic Learning
//!
//! Learning algorithms: rule weight learning, differentiable ILP (Inductive Logic Programming),
//! neural theorem proving helpers, semantic loss, and symbolic rule learning.

use super::neural_symbolic_reasoner::NeuralSymbolicModel;
use super::neural_symbolic_types::*;
use anyhow::Result;
use scirs2_core::ndarray_ext::Array1;
use std::collections::HashMap;

impl NeuralSymbolicModel {
    /// Learn symbolic rules from input-output examples.
    ///
    /// Performs a simple differentiable ILP scan: for every dimension pair
    /// (j, k) where both input and output exceed 0.5 a candidate rule is
    /// generated.  Rules are retained when they achieve ≥3 supporting
    /// examples with mean confidence > 0.7.
    pub fn learn_symbolic_rules(&mut self, examples: &[(Array1<f32>, Array1<f32>)]) -> Result<()> {
        let mut candidate_rules = Vec::new();

        for (input, output) in examples.iter() {
            for j in 0..input.len() {
                for k in 0..output.len() {
                    if input[j] > 0.5 && output[k] > 0.5 {
                        let antecedent = LogicalFormula::new_atom(format!("input_{j}"));
                        let consequent = LogicalFormula::new_atom(format!("output_{k}"));
                        let rule =
                            KnowledgeRule::new(format!("rule_{j}_{k}"), antecedent, consequent);
                        candidate_rules.push(rule);
                    }
                }
            }
        }

        for rule in candidate_rules {
            let mut support = 0usize;
            let mut confidence_sum = 0.0f32;

            for (input, output) in examples {
                let mut facts = HashMap::new();
                for (i, &value) in input.iter().enumerate() {
                    facts.insert(format!("input_{i}"), value);
                }

                if let Some((predicate, predicted_value)) = rule.apply(&facts) {
                    if let Some(index) = predicate
                        .strip_prefix("output_")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        if index < output.len() {
                            let actual_value = output[index];
                            let error = (predicted_value - actual_value).abs();
                            if error < 0.2 {
                                support += 1;
                                confidence_sum += 1.0 - error;
                            }
                        }
                    }
                }
            }

            if support >= 3 && confidence_sum / support as f32 > 0.7 {
                self.add_knowledge_rule(rule);
            }
        }

        Ok(())
    }

    /// Compute the combined semantic loss for a prediction against targets.
    ///
    /// The total loss is a convex combination of:
    /// - MSE loss (data fit)
    /// - Constraint violation loss (logical consistency)
    /// - Rule inconsistency loss (symbolic coherence)
    pub fn compute_semantic_loss(
        &self,
        predictions: &Array1<f32>,
        targets: &Array1<f32>,
    ) -> Result<f32> {
        // Standard MSE
        let mse_loss = {
            let diff = predictions - targets;
            diff.dot(&diff) / predictions.len() as f32
        };

        // Constraint violation loss
        let constraint_loss = {
            let mut facts = HashMap::new();
            for (i, &value) in predictions.iter().enumerate() {
                facts.insert(format!("output_{i}"), value);
            }

            let mut total_violation = 0.0f32;
            for constraint in &self.constraints {
                let satisfaction = constraint.evaluate(&facts);
                if satisfaction < 1.0 {
                    total_violation += (1.0 - satisfaction).powi(2);
                }
            }
            total_violation / self.constraints.len().max(1) as f32
        };

        // Rule consistency loss
        let rule_loss = {
            let mut facts = HashMap::new();
            for (i, &value) in predictions.iter().enumerate() {
                facts.insert(format!("input_{i}"), value);
            }

            let mut total_inconsistency = 0.0f32;
            for rule in &self.knowledge_base {
                if let Some((predicate, predicted_value)) = rule.apply(&facts) {
                    if let Some(index) = predicate
                        .strip_prefix("output_")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        if index < predictions.len() {
                            let actual_value = predictions[index];
                            let inconsistency = (predicted_value - actual_value).powi(2);
                            total_inconsistency += inconsistency * rule.weight;
                        }
                    }
                }
            }
            total_inconsistency / self.knowledge_base.len().max(1) as f32
        };

        Ok(mse_loss + 0.1 * constraint_loss + 0.1 * rule_loss)
    }

    /// Update rule weights using gradient-like feedback.
    ///
    /// For each rule, the weight is adjusted proportionally to how often
    /// it fired and how accurate its predictions were against the provided
    /// examples.  This implements a rudimentary differentiable ILP update.
    pub fn update_rule_weights(
        &mut self,
        examples: &[(Array1<f32>, Array1<f32>)],
        learning_rate: f32,
    ) -> Result<()> {
        for rule in self.knowledge_base.iter_mut() {
            let mut gradient_sum = 0.0f32;
            let mut count = 0usize;

            for (input, output) in examples {
                let mut facts = HashMap::new();
                for (i, &value) in input.iter().enumerate() {
                    facts.insert(format!("input_{i}"), value);
                }

                if let Some((predicate, predicted_value)) = rule.apply(&facts) {
                    if let Some(index) = predicate
                        .strip_prefix("output_")
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        if index < output.len() {
                            let target = output[index];
                            // Gradient w.r.t. rule weight: -(target - pred) * pred
                            let grad = -(target - predicted_value) * predicted_value;
                            gradient_sum += grad;
                            count += 1;
                        }
                    }
                }
            }

            if count > 0 {
                let mean_grad = gradient_sum / count as f32;
                rule.weight = (rule.weight - learning_rate * mean_grad).clamp(0.0, 10.0);
            }
        }

        Ok(())
    }

    /// Neural theorem proving: attempt to prove a target predicate.
    ///
    /// Implements backward chaining with depth-limited search, using neural
    /// confidence scores to prune low-probability proof paths.
    pub fn prove_predicate(
        &self,
        target: &str,
        known_facts: &HashMap<String, f32>,
        max_depth: usize,
    ) -> Option<f32> {
        self.prove_recursive(target, known_facts, max_depth, 1.0)
    }

    fn prove_recursive(
        &self,
        target: &str,
        known_facts: &HashMap<String, f32>,
        depth: usize,
        confidence_so_far: f32,
    ) -> Option<f32> {
        // Base case: target is already a known fact
        if let Some(&v) = known_facts.get(target) {
            return Some(v * confidence_so_far);
        }

        if depth == 0 {
            return None;
        }

        // Try to find a rule whose consequent matches the target
        let mut best: Option<f32> = None;

        for rule in &self.knowledge_base {
            if let FormulaStructure::Atom(consequent_pred) = &rule.consequent.structure {
                if consequent_pred == target {
                    // Try to prove the antecedent
                    let ante_val = rule.antecedent.evaluate(known_facts);
                    if ante_val > 0.0 {
                        let branch_confidence =
                            confidence_so_far * rule.confidence * ante_val;
                        if branch_confidence > best.unwrap_or(0.0) {
                            best = Some(branch_confidence);
                        }
                    } else if let FormulaStructure::Atom(ante_pred) = &rule.antecedent.structure {
                        // Recurse on the antecedent predicate
                        if let Some(proved) = self.prove_recursive(
                            ante_pred,
                            known_facts,
                            depth - 1,
                            confidence_so_far * rule.confidence,
                        ) {
                            if proved > best.unwrap_or(0.0) {
                                best = Some(proved);
                            }
                        }
                    }
                }
            }
        }

        best
    }
}
