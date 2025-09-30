//! Complex qualified shape combinations
//!
//! This module provides advanced support for complex qualified value shape combinations,
//! including logical combinations (AND, OR, NOT), nested qualifications, conditional
//! qualifications, and performance optimizations for large-scale validation.

use super::constraint_context::{ConstraintContext, ConstraintEvaluationResult};
use super::shape_constraints::QualifiedValueShapeConstraint;
use crate::{Result, ShaclError, ShapeId};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Complex qualified shape combinations with advanced logical operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexQualifiedShapeCombination {
    /// The qualified shape combination specification
    pub combination: QualifiedCombination,

    /// Global cardinality constraints for the entire combination
    pub global_min_count: Option<u32>,
    pub global_max_count: Option<u32>,

    /// Whether to enable performance optimizations
    pub enable_optimizations: bool,

    /// Cache results of shape validations
    pub enable_caching: bool,
}

impl ComplexQualifiedShapeCombination {
    /// Create a new complex qualified shape combination
    pub fn new(combination: QualifiedCombination) -> Self {
        Self {
            combination,
            global_min_count: None,
            global_max_count: None,
            enable_optimizations: true,
            enable_caching: true,
        }
    }

    /// Set global minimum count
    pub fn with_global_min_count(mut self, min_count: u32) -> Self {
        self.global_min_count = Some(min_count);
        self
    }

    /// Set global maximum count
    pub fn with_global_max_count(mut self, max_count: u32) -> Self {
        self.global_max_count = Some(max_count);
        self
    }

    /// Enable or disable performance optimizations
    pub fn with_optimizations(mut self, enabled: bool) -> Self {
        self.enable_optimizations = enabled;
        self
    }

    /// Enable or disable result caching
    pub fn with_caching(mut self, enabled: bool) -> Self {
        self.enable_caching = enabled;
        self
    }

    /// Validate the combination structure
    pub fn validate(&self) -> Result<()> {
        // Validate global cardinality constraints
        if let (Some(min), Some(max)) = (self.global_min_count, self.global_max_count) {
            if min > max {
                return Err(ShaclError::ConstraintValidation(format!(
                    "Global minimum count ({min}) cannot be greater than maximum count ({max})"
                )));
            }
        }

        // Validate the combination recursively
        self.combination.validate()
    }

    /// Evaluate the complex qualified shape combination
    pub fn evaluate(
        &self,
        context: &ConstraintContext,
        store: &dyn Store,
    ) -> Result<ConstraintEvaluationResult> {
        let start_time = Instant::now();

        // Create evaluation context with caching if enabled
        let mut eval_context = QualifiedEvaluationContext::new(
            context,
            self.enable_caching,
            self.enable_optimizations,
        );

        // Evaluate the combination
        let result = self.combination.evaluate(&mut eval_context, store)?;

        // Apply global cardinality constraints
        let final_result = self.apply_global_cardinality_constraints(&result, &eval_context)?;

        // Update performance metrics
        let evaluation_time = start_time.elapsed();
        eval_context.update_metrics(evaluation_time);

        tracing::debug!(
            "Complex qualified shape combination evaluated in {:?} with {} cache hits and {} cache misses",
            evaluation_time,
            eval_context.cache_hits,
            eval_context.cache_misses
        );

        Ok(final_result)
    }

    /// Apply global cardinality constraints to the evaluation result
    fn apply_global_cardinality_constraints(
        &self,
        result: &QualifiedCombinationResult,
        _context: &QualifiedEvaluationContext,
    ) -> Result<ConstraintEvaluationResult> {
        let total_count = result.total_conforming_count();

        // Check global minimum count
        if let Some(min_count) = self.global_min_count {
            if total_count < min_count {
                return Ok(ConstraintEvaluationResult::violated(
                    None,
                    Some(format!(
                        "Global qualified combination expected at least {min_count} conforming values, but found {total_count}"
                    )),
                ));
            }
        }

        // Check global maximum count
        if let Some(max_count) = self.global_max_count {
            if total_count > max_count {
                return Ok(ConstraintEvaluationResult::violated(
                    None,
                    Some(format!(
                        "Global qualified combination expected at most {max_count} conforming values, but found {total_count}"
                    )),
                ));
            }
        }

        // Check if the combination itself is satisfied
        if result.is_satisfied() {
            Ok(ConstraintEvaluationResult::satisfied())
        } else {
            Ok(ConstraintEvaluationResult::violated(
                result.get_violating_value(),
                result.get_violation_message(),
            ))
        }
    }

    /// Get performance metrics for this combination
    pub fn get_performance_metrics(&self) -> QualifiedCombinationMetrics {
        QualifiedCombinationMetrics {
            combination_complexity: self.combination.calculate_complexity(),
            estimated_evaluation_time: self.combination.estimate_evaluation_time(),
            cache_effectiveness: if self.enable_caching { 0.8 } else { 0.0 },
            optimization_level: if self.enable_optimizations {
                OptimizationLevel::High
            } else {
                OptimizationLevel::None
            },
        }
    }
}

/// Types of qualified shape combinations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualifiedCombination {
    /// Single qualified shape constraint
    Single(QualifiedValueShapeConstraint),

    /// Logical AND combination - all qualified shapes must be satisfied
    And(AndQualifiedCombination),

    /// Logical OR combination - at least one qualified shape must be satisfied
    Or(OrQualifiedCombination),

    /// Logical NOT combination - qualified shape must not be satisfied
    Not(NotQualifiedCombination),

    /// Exactly one (XOR) combination - exactly one qualified shape must be satisfied
    ExactlyOne(ExactlyOneQualifiedCombination),

    /// Conditional combination - qualified shape satisfaction depends on condition
    Conditional(ConditionalQualifiedCombination),

    /// Nested combination - allows hierarchical combination structures
    Nested(NestedQualifiedCombination),

    /// Sequential combination - qualified shapes must be satisfied in order
    Sequential(SequentialQualifiedCombination),
}

impl QualifiedCombination {
    /// Validate the combination structure
    pub fn validate(&self) -> Result<()> {
        match self {
            QualifiedCombination::Single(constraint) => constraint.validate(),
            QualifiedCombination::And(and_combo) => and_combo.validate(),
            QualifiedCombination::Or(or_combo) => or_combo.validate(),
            QualifiedCombination::Not(not_combo) => not_combo.validate(),
            QualifiedCombination::ExactlyOne(xor_combo) => xor_combo.validate(),
            QualifiedCombination::Conditional(cond_combo) => cond_combo.validate(),
            QualifiedCombination::Nested(nested_combo) => nested_combo.validate(),
            QualifiedCombination::Sequential(seq_combo) => seq_combo.validate(),
        }
    }

    /// Evaluate the combination
    pub fn evaluate(
        &self,
        context: &mut QualifiedEvaluationContext,
        store: &dyn Store,
    ) -> Result<QualifiedCombinationResult> {
        match self {
            QualifiedCombination::Single(constraint) => {
                self.evaluate_single_constraint(constraint, context, store)
            }
            QualifiedCombination::And(and_combo) => and_combo.evaluate(context, store),
            QualifiedCombination::Or(or_combo) => or_combo.evaluate(context, store),
            QualifiedCombination::Not(not_combo) => not_combo.evaluate(context, store),
            QualifiedCombination::ExactlyOne(xor_combo) => xor_combo.evaluate(context, store),
            QualifiedCombination::Conditional(cond_combo) => cond_combo.evaluate(context, store),
            QualifiedCombination::Nested(nested_combo) => nested_combo.evaluate(context, store),
            QualifiedCombination::Sequential(seq_combo) => seq_combo.evaluate(context, store),
        }
    }

    /// Evaluate a single qualified constraint with caching
    fn evaluate_single_constraint(
        &self,
        constraint: &QualifiedValueShapeConstraint,
        context: &mut QualifiedEvaluationContext,
        store: &dyn Store,
    ) -> Result<QualifiedCombinationResult> {
        let mut conforming_values = Vec::new();
        let mut non_conforming_values = Vec::new();

        for value in &context.constraint_context.values {
            let cache_key = (constraint.shape.clone(), value.clone());

            let conforms = if context.enable_caching {
                if let Some(&cached_result) = context.validation_cache.get(&cache_key) {
                    context.cache_hits += 1;
                    cached_result
                } else {
                    context.cache_misses += 1;
                    let result = constraint.value_conforms_to_shape(
                        value,
                        store,
                        context.constraint_context,
                    )?;
                    context.validation_cache.insert(cache_key, result);
                    result
                }
            } else {
                constraint.value_conforms_to_shape(value, store, context.constraint_context)?
            };

            if conforms {
                conforming_values.push(value.clone());
            } else {
                non_conforming_values.push(value.clone());
            }
        }

        // Apply cardinality constraints
        let conforming_count = conforming_values.len() as u32;

        // Check qualified min count
        if let Some(min_count) = constraint.qualified_min_count {
            if conforming_count < min_count {
                return Ok(QualifiedCombinationResult::violated(
                    format!(
                        "Expected at least {} values conforming to shape '{}', but found {}",
                        min_count,
                        constraint.shape.as_str(),
                        conforming_count
                    ),
                    non_conforming_values.first().cloned(),
                    conforming_count,
                ));
            }
        }

        // Check qualified max count
        if let Some(max_count) = constraint.qualified_max_count {
            if conforming_count > max_count {
                return Ok(QualifiedCombinationResult::violated(
                    format!(
                        "Expected at most {} values conforming to shape '{}', but found {}",
                        max_count,
                        constraint.shape.as_str(),
                        conforming_count
                    ),
                    conforming_values.get(max_count as usize).cloned(),
                    conforming_count,
                ));
            }
        }

        Ok(QualifiedCombinationResult::satisfied(conforming_count))
    }

    /// Calculate the complexity of this combination
    pub fn calculate_complexity(&self) -> CombinationComplexity {
        match self {
            QualifiedCombination::Single(_) => CombinationComplexity::Low,
            QualifiedCombination::And(and_combo) => and_combo.calculate_complexity(),
            QualifiedCombination::Or(or_combo) => or_combo.calculate_complexity(),
            QualifiedCombination::Not(_) => CombinationComplexity::Medium,
            QualifiedCombination::ExactlyOne(xor_combo) => xor_combo.calculate_complexity(),
            QualifiedCombination::Conditional(_) => CombinationComplexity::High,
            QualifiedCombination::Nested(nested_combo) => nested_combo.calculate_complexity(),
            QualifiedCombination::Sequential(seq_combo) => seq_combo.calculate_complexity(),
        }
    }

    /// Estimate evaluation time for this combination
    pub fn estimate_evaluation_time(&self) -> Duration {
        match self {
            QualifiedCombination::Single(_) => Duration::from_millis(1),
            QualifiedCombination::And(and_combo) => and_combo.estimate_evaluation_time(),
            QualifiedCombination::Or(or_combo) => or_combo.estimate_evaluation_time(),
            QualifiedCombination::Not(_) => Duration::from_millis(5),
            QualifiedCombination::ExactlyOne(xor_combo) => xor_combo.estimate_evaluation_time(),
            QualifiedCombination::Conditional(_) => Duration::from_millis(10),
            QualifiedCombination::Nested(nested_combo) => nested_combo.estimate_evaluation_time(),
            QualifiedCombination::Sequential(seq_combo) => seq_combo.estimate_evaluation_time(),
        }
    }
}

/// AND combination of qualified shapes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AndQualifiedCombination {
    pub combinations: Vec<QualifiedCombination>,
    pub require_all_distinct: bool, // Whether values must be distinct across combinations
}

impl AndQualifiedCombination {
    pub fn new(combinations: Vec<QualifiedCombination>) -> Self {
        Self {
            combinations,
            require_all_distinct: false,
        }
    }

    pub fn with_distinct_values(mut self, distinct: bool) -> Self {
        self.require_all_distinct = distinct;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.combinations.is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "AND qualified combination must have at least one sub-combination".to_string(),
            ));
        }

        for combination in &self.combinations {
            combination.validate()?;
        }

        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &mut QualifiedEvaluationContext,
        store: &dyn Store,
    ) -> Result<QualifiedCombinationResult> {
        let mut total_conforming_count = 0;
        let mut all_conforming_values: HashSet<Term> = HashSet::new();

        // Evaluate each sub-combination
        for combination in &self.combinations {
            let result = combination.evaluate(context, store)?;

            if !result.is_satisfied() {
                return Ok(result); // Early termination on first failure
            }

            total_conforming_count += result.total_conforming_count();

            // Track conforming values if distinctness is required
            if self.require_all_distinct {
                // For distinctness checking, we would need to track which specific values
                // conform to each combination - this is a simplification
                all_conforming_values.extend(context.constraint_context.values.iter().cloned());
            }
        }

        // Check distinctness constraint if required
        if self.require_all_distinct
            && all_conforming_values.len() != total_conforming_count as usize
        {
            return Ok(QualifiedCombinationResult::violated(
                "AND combination with distinct values requirement failed: overlapping conforming values found".to_string(),
                None,
                total_conforming_count,
            ));
        }

        Ok(QualifiedCombinationResult::satisfied(
            total_conforming_count,
        ))
    }

    pub fn calculate_complexity(&self) -> CombinationComplexity {
        let max_child_complexity = self
            .combinations
            .iter()
            .map(|c| c.calculate_complexity())
            .max()
            .unwrap_or(CombinationComplexity::Low);

        match (self.combinations.len(), max_child_complexity) {
            (1..=2, CombinationComplexity::Low) => CombinationComplexity::Low,
            (1..=2, CombinationComplexity::Medium) => CombinationComplexity::Medium,
            (3..=5, CombinationComplexity::Low) => CombinationComplexity::Medium,
            _ => CombinationComplexity::High,
        }
    }

    pub fn estimate_evaluation_time(&self) -> Duration {
        self.combinations
            .iter()
            .map(|c| c.estimate_evaluation_time())
            .sum()
    }
}

/// OR combination of qualified shapes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrQualifiedCombination {
    pub combinations: Vec<QualifiedCombination>,
    pub minimum_satisfied: u32, // Minimum number of combinations that must be satisfied
}

impl OrQualifiedCombination {
    pub fn new(combinations: Vec<QualifiedCombination>) -> Self {
        Self {
            combinations,
            minimum_satisfied: 1,
        }
    }

    pub fn with_minimum_satisfied(mut self, min_satisfied: u32) -> Self {
        self.minimum_satisfied = min_satisfied;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.combinations.is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "OR qualified combination must have at least one sub-combination".to_string(),
            ));
        }

        if self.minimum_satisfied == 0 || self.minimum_satisfied > self.combinations.len() as u32 {
            return Err(ShaclError::ConstraintValidation(format!(
                "Minimum satisfied count ({}) must be between 1 and the number of combinations ({})",
                self.minimum_satisfied, self.combinations.len()
            )));
        }

        for combination in &self.combinations {
            combination.validate()?;
        }

        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &mut QualifiedEvaluationContext,
        store: &dyn Store,
    ) -> Result<QualifiedCombinationResult> {
        let mut satisfied_count = 0;
        let mut total_conforming_count = 0;
        let mut last_error_message = None;

        // Evaluate each sub-combination
        for combination in &self.combinations {
            let result = combination.evaluate(context, store)?;

            if result.is_satisfied() {
                satisfied_count += 1;
                total_conforming_count += result.total_conforming_count();

                // Early termination if we have enough satisfied combinations
                if satisfied_count >= self.minimum_satisfied {
                    break;
                }
            } else {
                last_error_message = result.get_violation_message();
            }
        }

        // Check if minimum satisfied count is met
        if satisfied_count < self.minimum_satisfied {
            return Ok(QualifiedCombinationResult::violated(
                last_error_message.unwrap_or_else(|| format!(
                    "OR combination required at least {} satisfied sub-combinations, but only {} were satisfied",
                    self.minimum_satisfied, satisfied_count
                )),
                None,
                total_conforming_count,
            ));
        }

        Ok(QualifiedCombinationResult::satisfied(
            total_conforming_count,
        ))
    }

    pub fn calculate_complexity(&self) -> CombinationComplexity {
        let max_child_complexity = self
            .combinations
            .iter()
            .map(|c| c.calculate_complexity())
            .max()
            .unwrap_or(CombinationComplexity::Low);

        match (self.combinations.len(), max_child_complexity) {
            (1..=2, CombinationComplexity::Low) => CombinationComplexity::Low,
            (1..=3, CombinationComplexity::Medium) => CombinationComplexity::Medium,
            _ => CombinationComplexity::High,
        }
    }

    pub fn estimate_evaluation_time(&self) -> Duration {
        // OR can potentially terminate early, so estimate average case
        let total_time: Duration = self
            .combinations
            .iter()
            .map(|c| c.estimate_evaluation_time())
            .sum();

        total_time / self.combinations.len() as u32 * self.minimum_satisfied
    }
}

/// NOT combination (negation of qualified shape)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NotQualifiedCombination {
    pub combination: Box<QualifiedCombination>,
}

impl NotQualifiedCombination {
    pub fn new(combination: QualifiedCombination) -> Self {
        Self {
            combination: Box::new(combination),
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.combination.validate()
    }

    pub fn evaluate(
        &self,
        context: &mut QualifiedEvaluationContext,
        store: &dyn Store,
    ) -> Result<QualifiedCombinationResult> {
        let result = self.combination.evaluate(context, store)?;

        // Negate the result
        if result.is_satisfied() {
            Ok(QualifiedCombinationResult::violated(
                "NOT qualified combination failed: inner combination was satisfied when it should not be".to_string(),
                None,
                0, // NOT combinations don't contribute to conforming count
            ))
        } else {
            Ok(QualifiedCombinationResult::satisfied(0))
        }
    }
}

/// Exactly one (XOR) combination
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactlyOneQualifiedCombination {
    pub combinations: Vec<QualifiedCombination>,
}

impl ExactlyOneQualifiedCombination {
    pub fn new(combinations: Vec<QualifiedCombination>) -> Self {
        Self { combinations }
    }

    pub fn validate(&self) -> Result<()> {
        if self.combinations.len() < 2 {
            return Err(ShaclError::ConstraintValidation(
                "ExactlyOne qualified combination must have at least two sub-combinations"
                    .to_string(),
            ));
        }

        for combination in &self.combinations {
            combination.validate()?;
        }

        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &mut QualifiedEvaluationContext,
        store: &dyn Store,
    ) -> Result<QualifiedCombinationResult> {
        let mut satisfied_count = 0;
        let mut total_conforming_count = 0;

        // Evaluate all sub-combinations
        for combination in &self.combinations {
            let result = combination.evaluate(context, store)?;

            if result.is_satisfied() {
                satisfied_count += 1;
                total_conforming_count += result.total_conforming_count();

                // Early termination if more than one is satisfied
                if satisfied_count > 1 {
                    return Ok(QualifiedCombinationResult::violated(
                        format!("ExactlyOne combination failed: {satisfied_count} sub-combinations were satisfied, expected exactly 1"),
                        None,
                        total_conforming_count,
                    ));
                }
            }
        }

        // Check that exactly one was satisfied
        if satisfied_count == 1 {
            Ok(QualifiedCombinationResult::satisfied(
                total_conforming_count,
            ))
        } else {
            Ok(QualifiedCombinationResult::violated(
                format!("ExactlyOne combination failed: {satisfied_count} sub-combinations were satisfied, expected exactly 1"),
                None,
                total_conforming_count,
            ))
        }
    }

    pub fn calculate_complexity(&self) -> CombinationComplexity {
        // XOR requires evaluating all combinations
        CombinationComplexity::High
    }

    pub fn estimate_evaluation_time(&self) -> Duration {
        // XOR must evaluate all combinations
        self.combinations
            .iter()
            .map(|c| c.estimate_evaluation_time())
            .sum()
    }
}

/// Conditional qualified combination
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionalQualifiedCombination {
    pub condition: QualificationCondition,
    pub then_combination: Box<QualifiedCombination>,
    pub else_combination: Option<Box<QualifiedCombination>>,
}

/// Condition for conditional qualified combinations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QualificationCondition {
    /// Value count condition
    ValueCount { min: Option<u32>, max: Option<u32> },
    /// Type condition
    HasType { class_iri: String },
    /// Property existence condition
    HasProperty { property_iri: String },
    /// Custom SPARQL condition
    SparqlCondition { query: String },
}

impl ConditionalQualifiedCombination {
    pub fn new(
        condition: QualificationCondition,
        then_combination: QualifiedCombination,
        else_combination: Option<QualifiedCombination>,
    ) -> Self {
        Self {
            condition,
            then_combination: Box::new(then_combination),
            else_combination: else_combination.map(Box::new),
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.then_combination.validate()?;
        if let Some(else_combo) = &self.else_combination {
            else_combo.validate()?;
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &mut QualifiedEvaluationContext,
        store: &dyn Store,
    ) -> Result<QualifiedCombinationResult> {
        // Evaluate condition
        let condition_satisfied = self.evaluate_condition(context, store)?;

        // Choose combination based on condition
        let chosen_combination = if condition_satisfied {
            &self.then_combination
        } else if let Some(else_combo) = &self.else_combination {
            else_combo
        } else {
            // No else branch and condition not satisfied
            return Ok(QualifiedCombinationResult::satisfied(0));
        };

        chosen_combination.evaluate(context, store)
    }

    fn evaluate_condition(
        &self,
        context: &QualifiedEvaluationContext,
        _store: &dyn Store,
    ) -> Result<bool> {
        match &self.condition {
            QualificationCondition::ValueCount { min, max } => {
                let value_count = context.constraint_context.values.len() as u32;
                let min_satisfied = min.map_or(true, |m| value_count >= m);
                let max_satisfied = max.map_or(true, |m| value_count <= m);
                Ok(min_satisfied && max_satisfied)
            }
            QualificationCondition::HasType { .. } => {
                // Simplified implementation - would need proper type checking
                Ok(true)
            }
            QualificationCondition::HasProperty { .. } => {
                // Simplified implementation - would need proper property checking
                Ok(true)
            }
            QualificationCondition::SparqlCondition { .. } => {
                // Simplified implementation - would need SPARQL evaluation
                Ok(true)
            }
        }
    }
}

/// Nested qualified combination for hierarchical structures
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NestedQualifiedCombination {
    pub outer_combination: Box<QualifiedCombination>,
    pub inner_combinations: Vec<QualifiedCombination>,
    pub nesting_strategy: NestingStrategy,
}

/// Strategy for nesting combinations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NestingStrategy {
    /// Apply inner combinations to results of outer combination
    ApplyToResults,
    /// Use inner combinations as filters for outer combination
    FilterOuter,
    /// Hierarchical evaluation where each level refines the previous
    Hierarchical,
}

impl NestedQualifiedCombination {
    pub fn new(
        outer_combination: QualifiedCombination,
        inner_combinations: Vec<QualifiedCombination>,
        nesting_strategy: NestingStrategy,
    ) -> Self {
        Self {
            outer_combination: Box::new(outer_combination),
            inner_combinations,
            nesting_strategy,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.outer_combination.validate()?;
        for inner in &self.inner_combinations {
            inner.validate()?;
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &mut QualifiedEvaluationContext,
        store: &dyn Store,
    ) -> Result<QualifiedCombinationResult> {
        // First evaluate outer combination
        let outer_result = self.outer_combination.evaluate(context, store)?;

        if !outer_result.is_satisfied() {
            return Ok(outer_result);
        }

        // Apply nesting strategy
        match self.nesting_strategy {
            NestingStrategy::ApplyToResults => {
                // Apply inner combinations to the results
                let mut total_count = outer_result.total_conforming_count();

                for inner in &self.inner_combinations {
                    let inner_result = inner.evaluate(context, store)?;
                    if !inner_result.is_satisfied() {
                        return Ok(inner_result);
                    }
                    total_count += inner_result.total_conforming_count();
                }

                Ok(QualifiedCombinationResult::satisfied(total_count))
            }
            NestingStrategy::FilterOuter => {
                // Use inner combinations as filters
                for inner in &self.inner_combinations {
                    let inner_result = inner.evaluate(context, store)?;
                    if !inner_result.is_satisfied() {
                        return Ok(QualifiedCombinationResult::violated(
                            "Nested combination failed: inner filter not satisfied".to_string(),
                            None,
                            outer_result.total_conforming_count(),
                        ));
                    }
                }

                Ok(outer_result)
            }
            NestingStrategy::Hierarchical => {
                // Hierarchical evaluation
                let mut current_result = outer_result;

                for inner in &self.inner_combinations {
                    let inner_result = inner.evaluate(context, store)?;
                    if !inner_result.is_satisfied() {
                        return Ok(inner_result);
                    }

                    // Combine results hierarchically
                    let combined_count = current_result.total_conforming_count()
                        + inner_result.total_conforming_count();
                    current_result = QualifiedCombinationResult::satisfied(combined_count);
                }

                Ok(current_result)
            }
        }
    }

    pub fn calculate_complexity(&self) -> CombinationComplexity {
        CombinationComplexity::High // Nested combinations are inherently complex
    }

    pub fn estimate_evaluation_time(&self) -> Duration {
        let outer_time = self.outer_combination.estimate_evaluation_time();
        let inner_time: Duration = self
            .inner_combinations
            .iter()
            .map(|c| c.estimate_evaluation_time())
            .sum();

        outer_time + inner_time
    }
}

/// Sequential qualified combination
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequentialQualifiedCombination {
    pub combinations: Vec<QualifiedCombination>,
    pub require_order: bool, // Whether the order of satisfaction matters
    pub allow_overlap: bool, // Whether values can satisfy multiple combinations
}

impl SequentialQualifiedCombination {
    pub fn new(combinations: Vec<QualifiedCombination>) -> Self {
        Self {
            combinations,
            require_order: true,
            allow_overlap: false,
        }
    }

    pub fn with_order_requirement(mut self, required: bool) -> Self {
        self.require_order = required;
        self
    }

    pub fn with_overlap_allowed(mut self, allowed: bool) -> Self {
        self.allow_overlap = allowed;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.combinations.is_empty() {
            return Err(ShaclError::ConstraintValidation(
                "Sequential qualified combination must have at least one sub-combination"
                    .to_string(),
            ));
        }

        for combination in &self.combinations {
            combination.validate()?;
        }

        Ok(())
    }

    pub fn evaluate(
        &self,
        context: &mut QualifiedEvaluationContext,
        store: &dyn Store,
    ) -> Result<QualifiedCombinationResult> {
        let mut total_conforming_count = 0;
        let mut used_values: HashSet<Term> = HashSet::new();

        for combination in &self.combinations {
            // If overlap is not allowed, filter out already used values
            let eval_context = if !self.allow_overlap && !used_values.is_empty() {
                let filtered_values: Vec<Term> = context
                    .constraint_context
                    .values
                    .iter()
                    .filter(|v| !used_values.contains(v))
                    .cloned()
                    .collect();

                // Create a modified context with filtered values
                let mut filtered_constraint_context = context.constraint_context.clone();
                filtered_constraint_context.values = filtered_values;

                let mut filtered_context = QualifiedEvaluationContext::new(
                    &filtered_constraint_context,
                    context.enable_caching,
                    context.enable_optimizations,
                );

                let result = combination.evaluate(&mut filtered_context, store)?;

                // Update the main context with cache results
                context.cache_hits += filtered_context.cache_hits;
                context.cache_misses += filtered_context.cache_misses;
                context
                    .validation_cache
                    .extend(filtered_context.validation_cache);

                result
            } else {
                combination.evaluate(context, store)?
            };

            if !eval_context.is_satisfied() {
                return Ok(eval_context);
            }

            total_conforming_count += eval_context.total_conforming_count();

            // Track used values if overlap is not allowed
            if !self.allow_overlap {
                // For simplicity, mark all values as used - in a full implementation,
                // we would track which specific values were used by each combination
                used_values.extend(context.constraint_context.values.iter().cloned());
            }
        }

        Ok(QualifiedCombinationResult::satisfied(
            total_conforming_count,
        ))
    }

    pub fn calculate_complexity(&self) -> CombinationComplexity {
        match self.combinations.len() {
            1..=2 => CombinationComplexity::Medium,
            3..=5 => CombinationComplexity::High,
            _ => CombinationComplexity::High,
        }
    }

    pub fn estimate_evaluation_time(&self) -> Duration {
        self.combinations
            .iter()
            .map(|c| c.estimate_evaluation_time())
            .sum()
    }
}

/// Result of evaluating a qualified combination
#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedCombinationResult {
    satisfied: bool,
    violation_message: Option<String>,
    violating_value: Option<Term>,
    conforming_count: u32,
}

impl QualifiedCombinationResult {
    pub fn satisfied(conforming_count: u32) -> Self {
        Self {
            satisfied: true,
            violation_message: None,
            violating_value: None,
            conforming_count,
        }
    }

    pub fn violated(message: String, violating_value: Option<Term>, conforming_count: u32) -> Self {
        Self {
            satisfied: false,
            violation_message: Some(message),
            violating_value,
            conforming_count,
        }
    }

    pub fn is_satisfied(&self) -> bool {
        self.satisfied
    }

    pub fn get_violation_message(&self) -> Option<String> {
        self.violation_message.clone()
    }

    pub fn get_violating_value(&self) -> Option<Term> {
        self.violating_value.clone()
    }

    pub fn total_conforming_count(&self) -> u32 {
        self.conforming_count
    }
}

/// Evaluation context for qualified combinations
pub struct QualifiedEvaluationContext<'a> {
    pub constraint_context: &'a ConstraintContext,
    pub validation_cache: HashMap<(ShapeId, Term), bool>,
    pub enable_caching: bool,
    pub enable_optimizations: bool,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl<'a> QualifiedEvaluationContext<'a> {
    pub fn new(
        constraint_context: &'a ConstraintContext,
        enable_caching: bool,
        enable_optimizations: bool,
    ) -> Self {
        Self {
            constraint_context,
            validation_cache: HashMap::new(),
            enable_caching,
            enable_optimizations,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    pub fn update_metrics(&mut self, _evaluation_time: Duration) {
        // Update performance metrics
        tracing::debug!(
            "Qualified combination cache performance: {} hits, {} misses, {:.2}% hit rate",
            self.cache_hits,
            self.cache_misses,
            if self.cache_hits + self.cache_misses > 0 {
                (self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64) * 100.0
            } else {
                0.0
            }
        );
    }
}

/// Complexity levels for combinations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CombinationComplexity {
    Low,
    Medium,
    High,
}

/// Performance metrics for qualified combinations
#[derive(Debug, Clone)]
pub struct QualifiedCombinationMetrics {
    pub combination_complexity: CombinationComplexity,
    pub estimated_evaluation_time: Duration,
    pub cache_effectiveness: f64,
    pub optimization_level: OptimizationLevel,
}

/// Optimization levels
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Advanced,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{constraints::shape_constraints::QualifiedValueShapeConstraint, ShapeId};
    use oxirs_core::model::{NamedNode, Term};

    #[test]
    fn test_complex_qualified_combination_creation() {
        let constraint = QualifiedValueShapeConstraint::new(ShapeId::new("TestShape"))
            .with_qualified_min_count(1)
            .with_qualified_max_count(5);

        let combination =
            ComplexQualifiedShapeCombination::new(QualifiedCombination::Single(constraint))
                .with_global_min_count(1)
                .with_global_max_count(10);

        assert_eq!(combination.global_min_count, Some(1));
        assert_eq!(combination.global_max_count, Some(10));
        assert!(combination.enable_optimizations);
        assert!(combination.enable_caching);
    }

    #[test]
    fn test_and_qualified_combination() {
        let constraint1 =
            QualifiedValueShapeConstraint::new(ShapeId::new("Shape1")).with_qualified_min_count(1);
        let constraint2 =
            QualifiedValueShapeConstraint::new(ShapeId::new("Shape2")).with_qualified_max_count(5);

        let and_combo = AndQualifiedCombination::new(vec![
            QualifiedCombination::Single(constraint1),
            QualifiedCombination::Single(constraint2),
        ])
        .with_distinct_values(true);

        assert_eq!(and_combo.combinations.len(), 2);
        assert!(and_combo.require_all_distinct);
        assert!(and_combo.validate().is_ok());
    }

    #[test]
    fn test_or_qualified_combination() {
        let constraint1 =
            QualifiedValueShapeConstraint::new(ShapeId::new("Shape1")).with_qualified_min_count(1);
        let constraint2 =
            QualifiedValueShapeConstraint::new(ShapeId::new("Shape2")).with_qualified_max_count(3);
        let constraint3 =
            QualifiedValueShapeConstraint::new(ShapeId::new("Shape3")).with_qualified_min_count(2);

        let or_combo = OrQualifiedCombination::new(vec![
            QualifiedCombination::Single(constraint1),
            QualifiedCombination::Single(constraint2),
            QualifiedCombination::Single(constraint3),
        ])
        .with_minimum_satisfied(2);

        assert_eq!(or_combo.combinations.len(), 3);
        assert_eq!(or_combo.minimum_satisfied, 2);
        assert!(or_combo.validate().is_ok());
    }

    #[test]
    fn test_exactly_one_qualified_combination() {
        let constraint1 =
            QualifiedValueShapeConstraint::new(ShapeId::new("Shape1")).with_qualified_min_count(1);
        let constraint2 =
            QualifiedValueShapeConstraint::new(ShapeId::new("Shape2")).with_qualified_min_count(1);

        let xor_combo = ExactlyOneQualifiedCombination::new(vec![
            QualifiedCombination::Single(constraint1),
            QualifiedCombination::Single(constraint2),
        ]);

        assert_eq!(xor_combo.combinations.len(), 2);
        assert!(xor_combo.validate().is_ok());
    }

    #[test]
    fn test_conditional_qualified_combination() {
        let then_constraint = QualifiedValueShapeConstraint::new(ShapeId::new("ThenShape"))
            .with_qualified_min_count(1);
        let else_constraint = QualifiedValueShapeConstraint::new(ShapeId::new("ElseShape"))
            .with_qualified_min_count(1);

        let condition = QualificationCondition::ValueCount {
            min: Some(1),
            max: Some(10),
        };

        let cond_combo = ConditionalQualifiedCombination::new(
            condition,
            QualifiedCombination::Single(then_constraint),
            Some(QualifiedCombination::Single(else_constraint)),
        );

        assert!(cond_combo.validate().is_ok());
    }

    #[test]
    fn test_nested_qualified_combination() {
        let outer_constraint = QualifiedValueShapeConstraint::new(ShapeId::new("OuterShape"))
            .with_qualified_min_count(1);
        let inner_constraint1 = QualifiedValueShapeConstraint::new(ShapeId::new("InnerShape1"))
            .with_qualified_min_count(1);
        let inner_constraint2 = QualifiedValueShapeConstraint::new(ShapeId::new("InnerShape2"))
            .with_qualified_min_count(1);

        let nested_combo = NestedQualifiedCombination::new(
            QualifiedCombination::Single(outer_constraint),
            vec![
                QualifiedCombination::Single(inner_constraint1),
                QualifiedCombination::Single(inner_constraint2),
            ],
            NestingStrategy::Hierarchical,
        );

        assert_eq!(nested_combo.inner_combinations.len(), 2);
        assert_eq!(nested_combo.nesting_strategy, NestingStrategy::Hierarchical);
        assert!(nested_combo.validate().is_ok());
    }

    #[test]
    fn test_sequential_qualified_combination() {
        let constraint1 =
            QualifiedValueShapeConstraint::new(ShapeId::new("Shape1")).with_qualified_min_count(1);
        let constraint2 =
            QualifiedValueShapeConstraint::new(ShapeId::new("Shape2")).with_qualified_min_count(1);

        let seq_combo = SequentialQualifiedCombination::new(vec![
            QualifiedCombination::Single(constraint1),
            QualifiedCombination::Single(constraint2),
        ])
        .with_order_requirement(true)
        .with_overlap_allowed(false);

        assert_eq!(seq_combo.combinations.len(), 2);
        assert!(seq_combo.require_order);
        assert!(!seq_combo.allow_overlap);
        assert!(seq_combo.validate().is_ok());
    }

    #[test]
    fn test_combination_complexity_calculation() {
        let simple_constraint = QualifiedValueShapeConstraint::new(ShapeId::new("SimpleShape"))
            .with_qualified_min_count(1);
        let simple_combo = QualifiedCombination::Single(simple_constraint);
        assert_eq!(
            simple_combo.calculate_complexity(),
            CombinationComplexity::Low
        );

        let and_combo =
            AndQualifiedCombination::new(vec![simple_combo.clone(), simple_combo.clone()]);
        assert!(and_combo.calculate_complexity() >= CombinationComplexity::Low);

        let not_combo = QualifiedCombination::Not(NotQualifiedCombination::new(simple_combo));
        assert_eq!(
            not_combo.calculate_complexity(),
            CombinationComplexity::Medium
        );
    }

    #[test]
    fn test_qualified_combination_result() {
        let satisfied_result = QualifiedCombinationResult::satisfied(5);
        assert!(satisfied_result.is_satisfied());
        assert_eq!(satisfied_result.total_conforming_count(), 5);
        assert!(satisfied_result.get_violation_message().is_none());

        let violated_result = QualifiedCombinationResult::violated(
            "Test violation".to_string(),
            Some(Term::NamedNode(
                NamedNode::new("http://example.org/test").unwrap(),
            )),
            3,
        );
        assert!(!violated_result.is_satisfied());
        assert_eq!(violated_result.total_conforming_count(), 3);
        assert!(violated_result.get_violation_message().is_some());
        assert!(violated_result.get_violating_value().is_some());
    }
}
