//! Constraint ordering and evaluation optimization
//!
//! This module provides advanced constraint ordering strategies to minimize
//! validation time by evaluating the most selective constraints first and
//! implementing early termination for performance.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use crate::{
    constraints::{Constraint, ConstraintContext, ConstraintEvaluationResult},
    ConstraintComponentId, Result, ShaclError, ShapeId,
};

/// Constraint selectivity analyzer for optimal evaluation ordering
#[derive(Debug, Clone)]
pub struct ConstraintSelectivityAnalyzer {
    /// Selectivity statistics for constraint types
    constraint_selectivity: HashMap<ConstraintTypeKey, SelectivityStats>,
    /// Historical performance data
    performance_history: HashMap<ConstraintTypeKey, PerformanceStats>,
    /// Total evaluations for normalization
    total_evaluations: usize,
    /// Configuration for selectivity analysis
    config: SelectivityConfig,
}

/// Key for grouping constraints by type for selectivity analysis
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ConstraintTypeKey {
    /// Constraint component ID
    component_id: ConstraintComponentId,
    /// Constraint complexity class
    complexity_class: ConstraintComplexity,
}

/// Constraint complexity classification for performance prediction
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
enum ConstraintComplexity {
    /// Very fast constraints (e.g., node kind, simple value checks)
    Simple,
    /// Moderately complex constraints (e.g., cardinality, range checks)
    Moderate,
    /// Complex constraints (e.g., pattern matching, SPARQL)
    Complex,
    /// Very expensive constraints (e.g., shape-based, qualified cardinality)
    Expensive,
}

/// Selectivity statistics for a constraint type
#[derive(Debug, Clone)]
struct SelectivityStats {
    /// Total number of evaluations
    total_evaluations: usize,
    /// Number of violations found
    total_violations: usize,
    /// Selectivity ratio (violations / evaluations)
    selectivity: f64,
    /// Confidence level in selectivity measurement
    confidence: f64,
    /// Last updated timestamp
    last_updated: Instant,
}

/// Performance statistics for constraint evaluation
#[derive(Debug, Clone)]
struct PerformanceStats {
    /// Average evaluation time
    avg_evaluation_time: Duration,
    /// Total evaluation time
    total_evaluation_time: Duration,
    /// Number of evaluations
    evaluation_count: usize,
    /// Standard deviation of evaluation times
    time_std_dev: Duration,
    /// Performance trend (improving/degrading)
    performance_trend: f64,
}

/// Configuration for selectivity analysis
#[derive(Debug, Clone)]
pub struct SelectivityConfig {
    /// Minimum evaluations needed for reliable selectivity
    pub min_evaluations_for_confidence: usize,
    /// Weight for selectivity in ordering decision (0-1)
    pub selectivity_weight: f64,
    /// Weight for performance in ordering decision (0-1)
    pub performance_weight: f64,
    /// Enable adaptive threshold adjustment
    pub enable_adaptive_thresholds: bool,
    /// Early termination threshold (stop if selectivity > threshold)
    pub early_termination_threshold: f64,
    /// Maximum time to spend on ordering analysis (milliseconds)
    pub max_ordering_time_ms: u64,
}

impl Default for SelectivityConfig {
    fn default() -> Self {
        Self {
            min_evaluations_for_confidence: 100,
            selectivity_weight: 0.7,
            performance_weight: 0.3,
            enable_adaptive_thresholds: true,
            early_termination_threshold: 0.9,
            max_ordering_time_ms: 10,
        }
    }
}

/// Ordered constraint with evaluation metadata
#[derive(Debug, Clone)]
pub struct OrderedConstraint {
    /// The constraint to evaluate
    pub constraint: Constraint,
    /// Constraint context
    pub context: ConstraintContext,
    /// Expected selectivity (0-1, higher = more selective)
    pub selectivity: f64,
    /// Expected evaluation time
    pub expected_duration: Duration,
    /// Priority score (higher = evaluate first)
    pub priority_score: f64,
    /// Early termination potential (higher = better for early stopping)
    pub early_termination_potential: f64,
}

/// Result of constraint ordering optimization
#[derive(Debug)]
pub struct ConstraintOrderingResult {
    /// Ordered constraints by priority
    pub ordered_constraints: Vec<OrderedConstraint>,
    /// Total time spent on ordering analysis
    pub ordering_time: Duration,
    /// Expected performance improvement
    pub expected_improvement: f64,
    /// Early termination strategy applied
    pub early_termination_strategy: EarlyTerminationStrategy,
}

/// Early termination strategies
#[derive(Debug, Clone)]
pub enum EarlyTerminationStrategy {
    /// No early termination
    None,
    /// Stop after first violation
    FirstViolation,
    /// Stop after selectivity threshold reached
    SelectivityThreshold(f64),
    /// Stop after time limit
    TimeLimit(Duration),
    /// Adaptive strategy based on constraint patterns
    Adaptive,
}

impl ConstraintSelectivityAnalyzer {
    /// Create a new constraint selectivity analyzer
    pub fn new(config: SelectivityConfig) -> Self {
        Self {
            constraint_selectivity: HashMap::new(),
            performance_history: HashMap::new(),
            total_evaluations: 0,
            config,
        }
    }

    /// Analyze and order constraints for optimal evaluation
    pub fn order_constraints(
        &mut self,
        constraints_with_contexts: Vec<(Constraint, ConstraintContext)>,
    ) -> Result<ConstraintOrderingResult> {
        let start_time = Instant::now();

        // Convert to ordered constraints with metadata
        let mut ordered_constraints = Vec::with_capacity(constraints_with_contexts.len());

        for (constraint, context) in constraints_with_contexts {
            let constraint_key = self.get_constraint_type_key(&constraint);
            let selectivity = self.get_constraint_selectivity(&constraint_key);
            let expected_duration = self.get_expected_duration(&constraint_key);
            let early_termination_potential =
                self.calculate_early_termination_potential(&constraint, selectivity);

            // Calculate priority score: higher selectivity and lower duration = higher priority
            let priority_score = self.calculate_priority_score(selectivity, expected_duration);

            ordered_constraints.push(OrderedConstraint {
                constraint,
                context,
                selectivity,
                expected_duration,
                priority_score,
                early_termination_potential,
            });
        }

        // Sort by priority score (descending - higher priority first)
        ordered_constraints.sort_by(|a, b| {
            b.priority_score
                .partial_cmp(&a.priority_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply secondary optimization based on constraint dependencies
        self.optimize_for_dependencies(&mut ordered_constraints)?;

        // Determine early termination strategy
        let early_termination_strategy =
            self.determine_early_termination_strategy(&ordered_constraints);

        // Calculate expected performance improvement
        let expected_improvement = self.calculate_expected_improvement(&ordered_constraints);

        let ordering_time = start_time.elapsed();

        Ok(ConstraintOrderingResult {
            ordered_constraints,
            ordering_time,
            expected_improvement,
            early_termination_strategy,
        })
    }

    /// Update selectivity statistics based on evaluation results
    pub fn update_selectivity(
        &mut self,
        constraint: &Constraint,
        result: &ConstraintEvaluationResult,
        evaluation_time: Duration,
    ) {
        let constraint_key = self.get_constraint_type_key(constraint);

        // Update selectivity stats
        let selectivity_stats = self
            .constraint_selectivity
            .entry(constraint_key.clone())
            .or_insert_with(|| SelectivityStats {
                total_evaluations: 0,
                total_violations: 0,
                selectivity: 0.5, // Start with neutral selectivity
                confidence: 0.0,
                last_updated: Instant::now(),
            });

        selectivity_stats.total_evaluations += 1;
        if matches!(result, ConstraintEvaluationResult::Violated { .. }) {
            selectivity_stats.total_violations += 1;
        }

        // Recalculate selectivity
        selectivity_stats.selectivity =
            selectivity_stats.total_violations as f64 / selectivity_stats.total_evaluations as f64;

        // Update confidence based on sample size
        let total_evaluations = selectivity_stats.total_evaluations;
        let confidence = self.calculate_confidence(total_evaluations);
        selectivity_stats.confidence = confidence;
        selectivity_stats.last_updated = Instant::now();

        // Update performance stats
        let performance_stats = self
            .performance_history
            .entry(constraint_key)
            .or_insert_with(|| PerformanceStats {
                avg_evaluation_time: Duration::from_millis(1),
                total_evaluation_time: Duration::ZERO,
                evaluation_count: 0,
                time_std_dev: Duration::ZERO,
                performance_trend: 0.0,
            });

        performance_stats.evaluation_count += 1;
        performance_stats.total_evaluation_time += evaluation_time;

        // Update average with exponential moving average for recent responsiveness
        let alpha = 0.1; // Smoothing factor
        let new_avg_ms = performance_stats.avg_evaluation_time.as_millis() as f64 * (1.0 - alpha)
            + evaluation_time.as_millis() as f64 * alpha;
        performance_stats.avg_evaluation_time = Duration::from_millis(new_avg_ms as u64);

        self.total_evaluations += 1;

        // Update adaptive thresholds if enabled
        if self.config.enable_adaptive_thresholds {
            self.update_adaptive_thresholds();
        }
    }

    /// Get constraint type key for classification
    fn get_constraint_type_key(&self, constraint: &Constraint) -> ConstraintTypeKey {
        let complexity_class = match constraint {
            Constraint::NodeKind(_) | Constraint::HasValue(_) => ConstraintComplexity::Simple,
            Constraint::MinCount(_)
            | Constraint::MaxCount(_)
            | Constraint::Class(_)
            | Constraint::Datatype(_) => ConstraintComplexity::Moderate,
            Constraint::Pattern(_) | Constraint::MinLength(_) | Constraint::MaxLength(_) => {
                ConstraintComplexity::Moderate
            }
            Constraint::Sparql(_) | Constraint::And(_) | Constraint::Or(_) => {
                ConstraintComplexity::Complex
            }
            Constraint::Node(_) | Constraint::QualifiedValueShape(_) => {
                ConstraintComplexity::Expensive
            }
            _ => ConstraintComplexity::Moderate,
        };

        ConstraintTypeKey {
            component_id: constraint.component_id(),
            complexity_class,
        }
    }

    /// Get selectivity for a constraint type
    fn get_constraint_selectivity(&self, key: &ConstraintTypeKey) -> f64 {
        if let Some(stats) = self.constraint_selectivity.get(key) {
            if stats.confidence > 0.5 {
                stats.selectivity
            } else {
                // Use default selectivity based on complexity class
                match key.complexity_class {
                    ConstraintComplexity::Simple => 0.1, // Simple constraints rarely fail
                    ConstraintComplexity::Moderate => 0.3, // Moderate selectivity
                    ConstraintComplexity::Complex => 0.5, // Neutral selectivity
                    ConstraintComplexity::Expensive => 0.7, // Expensive constraints often find issues
                }
            }
        } else {
            // Default selectivity for unknown constraints
            0.5
        }
    }

    /// Get expected evaluation duration for a constraint type
    fn get_expected_duration(&self, key: &ConstraintTypeKey) -> Duration {
        if let Some(stats) = self.performance_history.get(key) {
            stats.avg_evaluation_time
        } else {
            // Default durations based on complexity class
            match key.complexity_class {
                ConstraintComplexity::Simple => Duration::from_micros(10),
                ConstraintComplexity::Moderate => Duration::from_micros(100),
                ConstraintComplexity::Complex => Duration::from_millis(1),
                ConstraintComplexity::Expensive => Duration::from_millis(10),
            }
        }
    }

    /// Calculate early termination potential for a constraint
    fn calculate_early_termination_potential(
        &self,
        constraint: &Constraint,
        selectivity: f64,
    ) -> f64 {
        // Higher selectivity + faster execution = higher early termination potential
        let base_potential = selectivity;

        // Boost for constraints that are known to be good early terminators
        let constraint_boost = match constraint {
            Constraint::NodeKind(_) => 0.2, // Very fast, good for early checks
            Constraint::Class(_) => 0.15,   // Good type-based filtering
            Constraint::HasValue(_) => 0.1, // Simple value checks
            Constraint::MinCount(_) | Constraint::MaxCount(_) => 0.05, // Cardinality checks
            _ => 0.0,
        };

        (base_potential + constraint_boost).min(1.0)
    }

    /// Calculate priority score for constraint ordering
    fn calculate_priority_score(&self, selectivity: f64, expected_duration: Duration) -> f64 {
        // Higher selectivity = higher priority (more likely to find violations early)
        // Lower duration = higher priority (faster to execute)

        let selectivity_component = selectivity * self.config.selectivity_weight;

        // Normalize duration to 0-1 scale (assuming max reasonable duration is 100ms)
        let duration_ms = expected_duration.as_millis() as f64;
        let normalized_duration = (100.0 - duration_ms.min(100.0)) / 100.0;
        let performance_component = normalized_duration * self.config.performance_weight;

        selectivity_component + performance_component
    }

    /// Optimize constraint order considering dependencies
    fn optimize_for_dependencies(
        &self,
        ordered_constraints: &mut [OrderedConstraint],
    ) -> Result<()> {
        // For now, implement a simple dependency-aware optimization
        // More sophisticated dependency analysis could be added later

        // Group constraints by dependency patterns
        let mut independent_constraints = Vec::new();
        let mut dependent_constraints = Vec::new();

        for constraint in ordered_constraints.iter() {
            if self.is_independent_constraint(&constraint.constraint) {
                independent_constraints.push(constraint.clone());
            } else {
                dependent_constraints.push(constraint.clone());
            }
        }

        // Reorder: independent constraints first (for early termination potential),
        // then dependent constraints
        ordered_constraints.clear();
        ordered_constraints.extend(independent_constraints);
        ordered_constraints.extend(dependent_constraints);

        Ok(())
    }

    /// Check if a constraint is independent (doesn't depend on other constraint results)
    fn is_independent_constraint(&self, constraint: &Constraint) -> bool {
        match constraint {
            Constraint::NodeKind(_)
            | Constraint::Class(_)
            | Constraint::Datatype(_)
            | Constraint::HasValue(_)
            | Constraint::MinCount(_)
            | Constraint::MaxCount(_) => true,
            Constraint::And(_) | Constraint::Or(_) | Constraint::Node(_) => false,
            _ => true, // Assume independent by default
        }
    }

    /// Determine the best early termination strategy
    fn determine_early_termination_strategy(
        &self,
        ordered_constraints: &[OrderedConstraint],
    ) -> EarlyTerminationStrategy {
        if ordered_constraints.is_empty() {
            return EarlyTerminationStrategy::None;
        }

        // Analyze constraint patterns to determine best strategy
        let avg_selectivity: f64 = ordered_constraints
            .iter()
            .map(|c| c.selectivity)
            .sum::<f64>()
            / ordered_constraints.len() as f64;

        let avg_early_termination_potential: f64 = ordered_constraints
            .iter()
            .map(|c| c.early_termination_potential)
            .sum::<f64>()
            / ordered_constraints.len() as f64;

        if avg_early_termination_potential > 0.7 {
            EarlyTerminationStrategy::FirstViolation
        } else if avg_selectivity > self.config.early_termination_threshold {
            EarlyTerminationStrategy::SelectivityThreshold(self.config.early_termination_threshold)
        } else {
            EarlyTerminationStrategy::Adaptive
        }
    }

    /// Calculate expected performance improvement from constraint ordering
    fn calculate_expected_improvement(&self, ordered_constraints: &[OrderedConstraint]) -> f64 {
        if ordered_constraints.is_empty() {
            return 0.0;
        }

        // Calculate potential time savings from early termination
        let total_expected_time: Duration = ordered_constraints
            .iter()
            .map(|c| c.expected_duration)
            .sum();

        let early_termination_time: Duration = ordered_constraints
            .iter()
            .take(3) // Assume early termination after first few constraints
            .map(|c| c.expected_duration)
            .sum();

        let avg_selectivity: f64 = ordered_constraints
            .iter()
            .take(3)
            .map(|c| c.selectivity)
            .sum::<f64>()
            / 3.0.min(ordered_constraints.len() as f64);

        // Expected improvement is the probability of early termination * time saved
        if total_expected_time.as_millis() > 0 {
            avg_selectivity
                * (1.0
                    - early_termination_time.as_millis() as f64
                        / total_expected_time.as_millis() as f64)
        } else {
            0.0
        }
    }

    /// Calculate confidence level based on sample size
    fn calculate_confidence(&self, sample_size: usize) -> f64 {
        // Simple confidence calculation based on sample size
        let min_samples = self.config.min_evaluations_for_confidence as f64;
        (sample_size as f64 / min_samples).min(1.0)
    }

    /// Update adaptive thresholds based on performance history
    fn update_adaptive_thresholds(&mut self) {
        if self.total_evaluations < 1000 {
            return; // Need more data for meaningful adaptation
        }

        // Calculate overall performance metrics
        let mut total_selectivity = 0.0;
        let mut total_confidence = 0.0;
        let mut constraint_count = 0;

        for stats in self.constraint_selectivity.values() {
            if stats.confidence > 0.5 {
                total_selectivity += stats.selectivity;
                total_confidence += stats.confidence;
                constraint_count += 1;
            }
        }

        if constraint_count > 0 {
            let avg_selectivity = total_selectivity / constraint_count as f64;

            // Adjust early termination threshold based on observed selectivity patterns
            if avg_selectivity > 0.8 {
                // High selectivity overall, can be more aggressive with early termination
                self.config.early_termination_threshold *= 0.95;
            } else if avg_selectivity < 0.2 {
                // Low selectivity overall, be more conservative
                self.config.early_termination_threshold *= 1.05;
            }

            // Keep threshold in reasonable bounds
            self.config.early_termination_threshold =
                self.config.early_termination_threshold.clamp(0.5, 0.95);
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> ConstraintOrderingStats {
        let mut stats = ConstraintOrderingStats {
            total_constraint_types: self.constraint_selectivity.len(),
            total_evaluations: self.total_evaluations,
            avg_selectivity: 0.0,
            avg_confidence: 0.0,
            performance_improvements: HashMap::new(),
        };

        if !self.constraint_selectivity.is_empty() {
            let total_selectivity: f64 = self
                .constraint_selectivity
                .values()
                .map(|s| s.selectivity)
                .sum();
            let total_confidence: f64 = self
                .constraint_selectivity
                .values()
                .map(|s| s.confidence)
                .sum();

            stats.avg_selectivity = total_selectivity / self.constraint_selectivity.len() as f64;
            stats.avg_confidence = total_confidence / self.constraint_selectivity.len() as f64;
        }

        stats
    }
}

/// Statistics for constraint ordering optimization
#[derive(Debug)]
pub struct ConstraintOrderingStats {
    /// Number of different constraint types tracked
    pub total_constraint_types: usize,
    /// Total number of constraint evaluations
    pub total_evaluations: usize,
    /// Average selectivity across all constraints
    pub avg_selectivity: f64,
    /// Average confidence in selectivity measurements
    pub avg_confidence: f64,
    /// Performance improvements by constraint type
    pub performance_improvements: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::value_constraints::NodeKindConstraint;
    use oxirs_core::model::Term;

    #[test]
    fn test_constraint_ordering() {
        let mut analyzer = ConstraintSelectivityAnalyzer::new(SelectivityConfig::default());

        // Create some test constraints
        let constraints = vec![(
            Constraint::NodeKind(NodeKindConstraint {
                node_kind: crate::constraints::value_constraints::NodeKind::Iri,
            }),
            ConstraintContext::new(
                Term::NamedNode(
                    oxirs_core::model::NamedNode::new("http://example.org/test").unwrap(),
                ),
                crate::ShapeId::new("test_shape"),
            ),
        )];

        let result = analyzer.order_constraints(constraints).unwrap();

        assert!(!result.ordered_constraints.is_empty());
        assert!(result.ordering_time.as_millis() < 100); // Should be fast
    }

    #[test]
    fn test_selectivity_updates() {
        let mut analyzer = ConstraintSelectivityAnalyzer::new(SelectivityConfig::default());

        let constraint = Constraint::NodeKind(NodeKindConstraint {
            node_kind: crate::constraints::value_constraints::NodeKind::Iri,
        });

        // Simulate several evaluations
        for i in 0..10 {
            let result = if i < 3 {
                ConstraintEvaluationResult::Violated {
                    violating_value: None,
                    message: Some("Test violation".to_string()),
                }
            } else {
                ConstraintEvaluationResult::Satisfied
            };

            analyzer.update_selectivity(&constraint, &result, Duration::from_micros(50));
        }

        let stats = analyzer.get_statistics();
        assert_eq!(stats.total_evaluations, 10);
        assert!(stats.avg_selectivity > 0.0);
    }
}
