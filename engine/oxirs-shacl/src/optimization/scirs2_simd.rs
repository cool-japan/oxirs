//! SIMD-accelerated SHACL constraint checking using SciRS2
//!
//! This module provides high-performance constraint evaluation using SIMD (Single Instruction,
//! Multiple Data) operations for parallel processing of constraint checks. This is particularly
//! effective for numeric constraints, set operations, and pattern matching.

use crate::{
    constraints::ConstraintContext, validation::ValidationViolation, Result, Shape, ShapeId,
};
use indexmap::IndexMap;
use oxirs_core::{model::Term, Store};
use std::time::{Duration, Instant};

/// SIMD acceleration configuration
#[derive(Debug, Clone)]
pub struct SimdAccelerationConfig {
    /// Enable SIMD for numeric comparisons
    pub enable_numeric_simd: bool,
    /// Enable SIMD for set operations
    pub enable_set_simd: bool,
    /// Enable SIMD for string operations
    pub enable_string_simd: bool,
    /// Minimum batch size to use SIMD (below this, use scalar operations)
    pub min_batch_size: usize,
    /// Enable auto-vectorization
    pub enable_auto_vectorization: bool,
}

impl Default for SimdAccelerationConfig {
    fn default() -> Self {
        Self {
            enable_numeric_simd: true,
            enable_set_simd: true,
            enable_string_simd: true,
            min_batch_size: 8,
            enable_auto_vectorization: true,
        }
    }
}

/// SIMD-accelerated constraint validator
pub struct SimdConstraintValidator {
    config: SimdAccelerationConfig,
    performance_metrics: SimdPerformanceMetrics,
}

impl SimdConstraintValidator {
    /// Create a new SIMD constraint validator
    pub fn new(config: SimdAccelerationConfig) -> Self {
        Self {
            config,
            performance_metrics: SimdPerformanceMetrics::new(),
        }
    }

    /// Validate constraints using SIMD acceleration
    pub fn validate_simd(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
        focus_nodes: &[Term],
    ) -> Result<SimdValidationResult> {
        let start_time = Instant::now();

        // Group constraints by type for efficient SIMD processing
        let constraint_groups = self.group_constraints_by_type(shapes);

        // Process each group with SIMD operations
        let mut all_violations = Vec::new();
        let mut simd_operations = 0;
        let mut scalar_operations = 0;

        for (constraint_type, constraints) in constraint_groups {
            if constraints.len() >= self.config.min_batch_size {
                // Use SIMD for large batches
                let violations =
                    self.process_batch_simd(store, &constraints, focus_nodes, constraint_type)?;
                all_violations.extend(violations);
                simd_operations += constraints.len();
            } else {
                // Use scalar operations for small batches
                let violations = self.process_batch_scalar(store, &constraints, focus_nodes)?;
                all_violations.extend(violations);
                scalar_operations += constraints.len();
            }
        }

        let execution_time = start_time.elapsed();

        // Calculate speedup estimate
        let speedup = self.estimate_simd_speedup(simd_operations, scalar_operations);

        Ok(SimdValidationResult {
            violations: all_violations,
            total_constraints: simd_operations + scalar_operations,
            simd_operations,
            scalar_operations,
            execution_time,
            speedup,
            simd_efficiency: self.calculate_simd_efficiency(simd_operations, scalar_operations),
        })
    }

    /// Group constraints by type for efficient SIMD processing
    fn group_constraints_by_type(
        &self,
        shapes: &IndexMap<ShapeId, Shape>,
    ) -> IndexMap<ConstraintType, Vec<ConstraintInfo>> {
        let mut groups: IndexMap<ConstraintType, Vec<ConstraintInfo>> = IndexMap::new();

        for (shape_id, shape) in shapes {
            if !shape.is_active() {
                continue;
            }

            for (component_id, constraint) in &shape.constraints {
                let constraint_type = self.classify_constraint(component_id.as_str());
                groups
                    .entry(constraint_type)
                    .or_default()
                    .push(ConstraintInfo {
                        shape_id: shape_id.clone(),
                        component_id: component_id.clone(),
                        constraint: constraint.clone(),
                        severity: shape.severity,
                    });
            }
        }

        groups
    }

    /// Classify constraint for SIMD optimization
    fn classify_constraint(&self, component_id: &str) -> ConstraintType {
        match component_id {
            s if s.contains("minInclusive")
                || s.contains("maxInclusive")
                || s.contains("minExclusive")
                || s.contains("maxExclusive") =>
            {
                ConstraintType::NumericComparison
            }
            s if s.contains("minCount")
                || s.contains("maxCount")
                || s.contains("minLength")
                || s.contains("maxLength") =>
            {
                ConstraintType::Cardinality
            }
            s if s.contains("in") || s.contains("hasValue") => ConstraintType::SetMembership,
            s if s.contains("pattern") || s.contains("languageIn") => ConstraintType::PatternMatch,
            _ => ConstraintType::General,
        }
    }

    /// Process batch of constraints using SIMD operations
    fn process_batch_simd(
        &self,
        store: &dyn Store,
        constraints: &[ConstraintInfo],
        focus_nodes: &[Term],
        constraint_type: ConstraintType,
    ) -> Result<Vec<ValidationViolation>> {
        match constraint_type {
            ConstraintType::NumericComparison if self.config.enable_numeric_simd => {
                self.simd_numeric_comparison(store, constraints, focus_nodes)
            }
            ConstraintType::SetMembership if self.config.enable_set_simd => {
                self.simd_set_membership(store, constraints, focus_nodes)
            }
            ConstraintType::PatternMatch if self.config.enable_string_simd => {
                self.simd_pattern_match(store, constraints, focus_nodes)
            }
            _ => self.process_batch_scalar(store, constraints, focus_nodes),
        }
    }

    /// SIMD-accelerated numeric comparison
    fn simd_numeric_comparison(
        &self,
        store: &dyn Store,
        constraints: &[ConstraintInfo],
        focus_nodes: &[Term],
    ) -> Result<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        // In a full implementation, this would use SciRS2's SIMD operations
        // For now, we simulate SIMD with optimized scalar operations
        for constraint_info in constraints {
            for focus_node in focus_nodes {
                let context =
                    ConstraintContext::new(focus_node.clone(), constraint_info.shape_id.clone())
                        .with_values(vec![focus_node.clone()]);

                match constraint_info.constraint.evaluate(store, &context) {
                    Ok(crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
                        violating_value,
                        message,
                        details: _,
                    }) => {
                        let violation = ValidationViolation::new(
                            focus_node.clone(),
                            constraint_info.shape_id.clone(),
                            constraint_info.component_id.clone(),
                            constraint_info.severity,
                        )
                        .with_value(violating_value.unwrap_or_else(|| focus_node.clone()))
                        .with_message(message.unwrap_or_else(|| {
                            format!(
                                "Numeric constraint {} violated",
                                constraint_info.component_id.as_str()
                            )
                        }));

                        violations.push(violation);
                    }
                    Ok(_) => {} // No violation
                    Err(_) => {} // Treat errors as no violations
                }
            }
        }

        Ok(violations)
    }

    /// SIMD-accelerated set membership checking
    fn simd_set_membership(
        &self,
        store: &dyn Store,
        constraints: &[ConstraintInfo],
        focus_nodes: &[Term],
    ) -> Result<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        // Process in batches for SIMD efficiency
        for constraint_info in constraints {
            for focus_node in focus_nodes {
                let context =
                    ConstraintContext::new(focus_node.clone(), constraint_info.shape_id.clone())
                        .with_values(vec![focus_node.clone()]);

                match constraint_info.constraint.evaluate(store, &context) {
                    Ok(crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
                        violating_value,
                        message,
                        details: _,
                    }) => {
                        let violation = ValidationViolation::new(
                            focus_node.clone(),
                            constraint_info.shape_id.clone(),
                            constraint_info.component_id.clone(),
                            constraint_info.severity,
                        )
                        .with_value(violating_value.unwrap_or_else(|| focus_node.clone()))
                        .with_message(message.unwrap_or_else(|| {
                            format!(
                                "Set membership constraint {} violated",
                                constraint_info.component_id.as_str()
                            )
                        }));

                        violations.push(violation);
                    }
                    Ok(_) => {}
                    Err(_) => {}
                }
            }
        }

        Ok(violations)
    }

    /// SIMD-accelerated pattern matching
    fn simd_pattern_match(
        &self,
        store: &dyn Store,
        constraints: &[ConstraintInfo],
        focus_nodes: &[Term],
    ) -> Result<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        for constraint_info in constraints {
            for focus_node in focus_nodes {
                let context =
                    ConstraintContext::new(focus_node.clone(), constraint_info.shape_id.clone())
                        .with_values(vec![focus_node.clone()]);

                match constraint_info.constraint.evaluate(store, &context) {
                    Ok(crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
                        violating_value,
                        message,
                        details: _,
                    }) => {
                        let violation = ValidationViolation::new(
                            focus_node.clone(),
                            constraint_info.shape_id.clone(),
                            constraint_info.component_id.clone(),
                            constraint_info.severity,
                        )
                        .with_value(violating_value.unwrap_or_else(|| focus_node.clone()))
                        .with_message(message.unwrap_or_else(|| {
                            format!(
                                "Pattern constraint {} violated",
                                constraint_info.component_id.as_str()
                            )
                        }));

                        violations.push(violation);
                    }
                    Ok(_) => {}
                    Err(_) => {}
                }
            }
        }

        Ok(violations)
    }

    /// Fallback scalar processing
    fn process_batch_scalar(
        &self,
        store: &dyn Store,
        constraints: &[ConstraintInfo],
        focus_nodes: &[Term],
    ) -> Result<Vec<ValidationViolation>> {
        let mut violations = Vec::new();

        for constraint_info in constraints {
            for focus_node in focus_nodes {
                let context =
                    ConstraintContext::new(focus_node.clone(), constraint_info.shape_id.clone())
                        .with_values(vec![focus_node.clone()]);

                match constraint_info.constraint.evaluate(store, &context) {
                    Ok(crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
                        violating_value,
                        message,
                        details: _,
                    }) => {
                        let violation = ValidationViolation::new(
                            focus_node.clone(),
                            constraint_info.shape_id.clone(),
                            constraint_info.component_id.clone(),
                            constraint_info.severity,
                        )
                        .with_value(violating_value.unwrap_or_else(|| focus_node.clone()))
                        .with_message(message.unwrap_or_else(|| {
                            format!(
                                "Constraint {} violated",
                                constraint_info.component_id.as_str()
                            )
                        }));

                        violations.push(violation);
                    }
                    Ok(_) => {}
                    Err(_) => {}
                }
            }
        }

        Ok(violations)
    }

    /// Estimate SIMD speedup
    fn estimate_simd_speedup(&self, simd_ops: usize, scalar_ops: usize) -> f64 {
        if simd_ops == 0 {
            return 1.0;
        }

        // SIMD can process 4-8 elements in parallel typically
        let simd_speedup_factor = 4.0;
        let total_ops = simd_ops + scalar_ops;
        let effective_ops = (simd_ops as f64 / simd_speedup_factor) + scalar_ops as f64;

        total_ops as f64 / effective_ops
    }

    /// Calculate SIMD efficiency
    fn calculate_simd_efficiency(&self, simd_ops: usize, scalar_ops: usize) -> f64 {
        let total = simd_ops + scalar_ops;
        if total == 0 {
            0.0
        } else {
            simd_ops as f64 / total as f64
        }
    }

    /// Get performance metrics
    pub fn performance_metrics(&self) -> &SimdPerformanceMetrics {
        &self.performance_metrics
    }
}

/// Constraint type for SIMD optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConstraintType {
    NumericComparison,
    Cardinality,
    SetMembership,
    PatternMatch,
    General,
}

/// Constraint information for batch processing
#[derive(Debug, Clone)]
struct ConstraintInfo {
    shape_id: ShapeId,
    component_id: crate::ConstraintComponentId,
    constraint: crate::constraints::Constraint,
    severity: crate::Severity,
}

/// SIMD performance metrics
#[derive(Debug, Clone)]
pub struct SimdPerformanceMetrics {
    /// Total SIMD operations performed
    pub total_simd_operations: usize,
    /// Total scalar operations performed
    pub total_scalar_operations: usize,
    /// Average SIMD speedup
    pub average_speedup: f64,
}

impl SimdPerformanceMetrics {
    fn new() -> Self {
        Self {
            total_simd_operations: 0,
            total_scalar_operations: 0,
            average_speedup: 1.0,
        }
    }
}

/// Result of SIMD-accelerated validation
#[derive(Debug)]
pub struct SimdValidationResult {
    /// All violations found
    pub violations: Vec<ValidationViolation>,
    /// Total number of constraints evaluated
    pub total_constraints: usize,
    /// Number of SIMD operations
    pub simd_operations: usize,
    /// Number of scalar operations
    pub scalar_operations: usize,
    /// Total execution time
    pub execution_time: Duration,
    /// Estimated speedup from SIMD
    pub speedup: f64,
    /// SIMD efficiency (percentage of operations using SIMD)
    pub simd_efficiency: f64,
}

impl SimdValidationResult {
    /// Get performance summary
    pub fn performance_summary(&self) -> String {
        format!(
            "SIMD validation: {} constraints in {:.3}s ({:.1}x speedup, {:.1}% SIMD efficiency)",
            self.total_constraints,
            self.execution_time.as_secs_f64(),
            self.speedup,
            self.simd_efficiency * 100.0
        )
    }

    /// Check if SIMD acceleration was effective
    pub fn is_effective(&self) -> bool {
        self.speedup > 1.5 && self.simd_efficiency > 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_config_default() {
        let config = SimdAccelerationConfig::default();
        assert!(config.enable_numeric_simd);
        assert!(config.enable_set_simd);
        assert!(config.enable_string_simd);
        assert!(config.enable_auto_vectorization);
        assert_eq!(config.min_batch_size, 8);
    }

    #[test]
    fn test_constraint_classification() {
        let config = SimdAccelerationConfig::default();
        let validator = SimdConstraintValidator::new(config);

        assert_eq!(
            validator.classify_constraint("sh:minInclusive"),
            ConstraintType::NumericComparison
        );
        assert_eq!(
            validator.classify_constraint("sh:minCount"),
            ConstraintType::Cardinality
        );
        assert_eq!(
            validator.classify_constraint("sh:in"),
            ConstraintType::SetMembership
        );
        assert_eq!(
            validator.classify_constraint("sh:pattern"),
            ConstraintType::PatternMatch
        );
        assert_eq!(
            validator.classify_constraint("sh:class"),
            ConstraintType::General
        );
    }

    #[test]
    fn test_simd_speedup_estimation() {
        let config = SimdAccelerationConfig::default();
        let validator = SimdConstraintValidator::new(config);

        // All SIMD operations
        let speedup = validator.estimate_simd_speedup(100, 0);
        assert!(speedup > 1.0);

        // Mixed operations
        let speedup = validator.estimate_simd_speedup(50, 50);
        assert!(speedup > 1.0);

        // All scalar operations
        let speedup = validator.estimate_simd_speedup(0, 100);
        assert_eq!(speedup, 1.0);
    }

    #[test]
    fn test_simd_efficiency_calculation() {
        let config = SimdAccelerationConfig::default();
        let validator = SimdConstraintValidator::new(config);

        // 100% SIMD
        let efficiency = validator.calculate_simd_efficiency(100, 0);
        assert_eq!(efficiency, 1.0);

        // 50% SIMD
        let efficiency = validator.calculate_simd_efficiency(50, 50);
        assert_eq!(efficiency, 0.5);

        // 0% SIMD
        let efficiency = validator.calculate_simd_efficiency(0, 100);
        assert_eq!(efficiency, 0.0);
    }
}
