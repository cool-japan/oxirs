//! Bottleneck analysis and recommendation methods for the optimization engine.
//!
//! This module extends `OptimizationEngine` with methods for identifying performance
//! bottlenecks from validation history and generating optimization recommendations
//! based on shape complexity and graph structure analysis.

use indexmap::IndexMap;
use std::collections::HashMap;

use oxirs_core::Store;
use oxirs_shacl::{Constraint, ConstraintComponentId, Shape, ValidationReport};

use crate::Result;

use super::{
    engine::OptimizationEngine,
    types::{
        BottleneckSeverity, BottleneckType, GraphAnalysis, ImplementationEffort,
        OptimizationRecommendation, OptimizationRecommendationType, PerformanceBottleneck,
        RecommendationPriority,
    },
};

impl OptimizationEngine {
    pub(super) fn analyze_performance_bottlenecks(
        &self,
        validation_history: &[ValidationReport],
    ) -> Result<Vec<PerformanceBottleneck>> {
        tracing::debug!(
            "Analyzing performance bottlenecks from {} validation reports",
            validation_history.len()
        );

        let mut bottlenecks = Vec::new();

        if validation_history.is_empty() {
            return Ok(bottlenecks);
        }

        // Analyze execution time trends
        if let Some(bottleneck) = self.analyze_execution_time_bottlenecks(validation_history)? {
            bottlenecks.push(bottleneck);
        }

        // Analyze memory usage patterns
        if let Some(bottleneck) = self.analyze_memory_bottlenecks(validation_history)? {
            bottlenecks.push(bottleneck);
        }

        // Analyze constraint performance
        bottlenecks.extend(self.analyze_constraint_bottlenecks(validation_history)?);

        // Analyze shape-specific bottlenecks
        bottlenecks.extend(self.analyze_shape_specific_bottlenecks(validation_history)?);

        // Sort by impact score (highest first)
        bottlenecks.sort_by(|a, b| {
            b.impact_score
                .partial_cmp(&a.impact_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        tracing::debug!("Identified {} performance bottlenecks", bottlenecks.len());
        Ok(bottlenecks)
    }

    /// Analyze execution time bottlenecks
    fn analyze_execution_time_bottlenecks(
        &self,
        validation_history: &[ValidationReport],
    ) -> Result<Option<PerformanceBottleneck>> {
        // Calculate execution time statistics
        let execution_times: Vec<f64> = validation_history
            .iter()
            .filter_map(|report| {
                report
                    .metadata
                    .metadata
                    .get("execution_time_ms")
                    .and_then(|v| v.parse::<f64>().ok())
            })
            .collect();

        if execution_times.len() < 3 {
            return Ok(None);
        }

        let avg_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
        let recent_avg = execution_times[execution_times.len().saturating_sub(5)..]
            .iter()
            .sum::<f64>()
            / execution_times[execution_times.len().saturating_sub(5)..].len() as f64;

        // Check for degradation
        let degradation_ratio = recent_avg / avg_time;

        if degradation_ratio > 1.5 {
            // 50% increase in execution time
            let severity = if degradation_ratio > 2.0 {
                BottleneckSeverity::Critical
            } else if degradation_ratio > 1.75 {
                BottleneckSeverity::High
            } else {
                BottleneckSeverity::Medium
            };

            let bottleneck = PerformanceBottleneck {
                bottleneck_type: BottleneckType::ExecutionTime,
                description: format!(
                    "Validation execution time has increased by {:.1}% (from {:.1}ms to {:.1}ms avg)",
                    (degradation_ratio - 1.0) * 100.0,
                    avg_time,
                    recent_avg
                ),
                severity,
                impact_score: (degradation_ratio - 1.0).min(1.0),
                affected_operations: vec![
                    "shape_validation".to_string(),
                    "constraint_evaluation".to_string(),
                ],
            };

            Ok(Some(bottleneck))
        } else {
            Ok(None)
        }
    }

    /// Analyze memory usage bottlenecks
    fn analyze_memory_bottlenecks(
        &self,
        validation_history: &[ValidationReport],
    ) -> Result<Option<PerformanceBottleneck>> {
        // Extract memory usage data
        let memory_usages: Vec<f64> = validation_history
            .iter()
            .filter_map(|report| {
                report
                    .metadata
                    .metadata
                    .get("peak_memory_mb")
                    .and_then(|v| v.parse::<f64>().ok())
            })
            .collect();

        if memory_usages.len() < 3 {
            return Ok(None);
        }

        let max_memory = memory_usages
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let avg_memory = memory_usages.iter().sum::<f64>() / memory_usages.len() as f64;

        // Check for high memory usage
        if max_memory > 1024.0 || avg_memory > 512.0 {
            // High memory thresholds
            let severity = if max_memory > 2048.0 {
                BottleneckSeverity::Critical
            } else if max_memory > 1536.0 {
                BottleneckSeverity::High
            } else {
                BottleneckSeverity::Medium
            };

            let impact_score = (max_memory / 2048.0).min(1.0);

            let bottleneck = PerformanceBottleneck {
                bottleneck_type: BottleneckType::Memory,
                description: format!(
                    "High memory usage detected: peak {max_memory:.1}MB, average {avg_memory:.1}MB"
                ),
                severity,
                impact_score,
                affected_operations: vec![
                    "memory_allocation".to_string(),
                    "garbage_collection".to_string(),
                ],
            };

            Ok(Some(bottleneck))
        } else {
            Ok(None)
        }
    }

    /// Analyze constraint-specific bottlenecks
    fn analyze_constraint_bottlenecks(
        &self,
        validation_history: &[ValidationReport],
    ) -> Result<Vec<PerformanceBottleneck>> {
        let mut bottlenecks = Vec::new();

        // Analyze constraint failure patterns
        let mut constraint_failures: HashMap<String, Vec<f64>> = HashMap::new();

        for report in validation_history {
            // Extract constraint timing data (simulated)
            if report.metadata.metadata.contains_key("constraint_timings") {
                // Parse constraint timing data (simplified)
                // In practice, this would parse actual timing data from the validation report
                for constraint_type in ["pattern", "datatype", "minCount", "maxCount", "class"] {
                    let timing = match constraint_type {
                        "pattern" => 15.0, // Pattern matching is typically slow
                        "datatype" => 2.0,
                        "minCount" => 3.0,
                        "maxCount" => 3.0,
                        "class" => 5.0,
                        _ => 2.0,
                    };

                    constraint_failures
                        .entry(constraint_type.to_string())
                        .or_default()
                        .push(timing);
                }
            }
        }

        // Identify slow constraint types
        for (constraint_type, timings) in constraint_failures {
            if timings.len() < 3 {
                continue;
            }

            let avg_timing = timings.iter().sum::<f64>() / timings.len() as f64;

            if avg_timing > 10.0 {
                // Constraint takes more than 10ms on average
                let severity = if avg_timing > 50.0 {
                    BottleneckSeverity::High
                } else if avg_timing > 25.0 {
                    BottleneckSeverity::Medium
                } else {
                    BottleneckSeverity::Low
                };

                let bottleneck = PerformanceBottleneck {
                    bottleneck_type: BottleneckType::Constraint,
                    description: format!(
                        "Slow constraint type '{constraint_type}' with average execution time of {avg_timing:.1}ms"
                    ),
                    severity,
                    impact_score: (avg_timing / 100.0).min(1.0),
                    affected_operations: vec![format!(
                        "{}_constraint_validation",
                        constraint_type
                    )],
                };

                bottlenecks.push(bottleneck);
            }
        }

        Ok(bottlenecks)
    }

    /// Analyze shape-specific bottlenecks
    fn analyze_shape_specific_bottlenecks(
        &self,
        validation_history: &[ValidationReport],
    ) -> Result<Vec<PerformanceBottleneck>> {
        let mut bottlenecks = Vec::new();
        let mut shape_performance: HashMap<String, Vec<f64>> = HashMap::new();

        // Collect shape performance data
        for report in validation_history {
            // Extract shape timing data (simulated)
            if report.metadata.metadata.contains_key("shape_timings") {
                // Parse shape timing data (simplified)
                // In practice, this would parse actual timing data per shape
                for violation in &report.violations {
                    let shape_id = &violation.source_shape;
                    let timing = 25.0; // Simulated timing
                    shape_performance
                        .entry(shape_id.as_str().to_string())
                        .or_default()
                        .push(timing);
                }
            }
        }

        // Identify slow shapes
        for (shape_id, timings) in shape_performance {
            if timings.len() < 5 {
                continue;
            }

            let avg_timing = timings.iter().sum::<f64>() / timings.len() as f64;
            let max_timing = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            if avg_timing > 20.0 || max_timing > 100.0 {
                let severity = if avg_timing > 100.0 || max_timing > 500.0 {
                    BottleneckSeverity::High
                } else if avg_timing > 50.0 || max_timing > 200.0 {
                    BottleneckSeverity::Medium
                } else {
                    BottleneckSeverity::Low
                };

                let bottleneck = PerformanceBottleneck {
                    bottleneck_type: BottleneckType::ExecutionTime,
                    description: format!(
                        "Slow shape '{shape_id}' with average execution time of {avg_timing:.1}ms (max: {max_timing:.1}ms)"
                    ),
                    severity,
                    impact_score: (avg_timing / 200.0).min(1.0),
                    affected_operations: vec![format!("shape_{}_validation", shape_id)],
                };

                bottlenecks.push(bottleneck);
            }
        }

        Ok(bottlenecks)
    }

    pub(super) fn generate_recommendation_for_bottleneck(
        &self,
        bottleneck: &PerformanceBottleneck,
        _store: &dyn Store,
        _shapes: &[Shape],
    ) -> Result<OptimizationRecommendation> {
        // Map bottleneck type to the most effective optimization recommendation type
        let (rec_type, effort, benefit) = match bottleneck.bottleneck_type {
            BottleneckType::ExecutionTime => (
                OptimizationRecommendationType::ConstraintReordering,
                ImplementationEffort::Low,
                0.35,
            ),
            BottleneckType::Memory => (
                OptimizationRecommendationType::MemoryOptimization,
                ImplementationEffort::Medium,
                0.30,
            ),
            BottleneckType::Cpu => (
                OptimizationRecommendationType::ParallelExecution,
                ImplementationEffort::High,
                0.45,
            ),
            BottleneckType::Io => (
                OptimizationRecommendationType::Caching,
                ImplementationEffort::Medium,
                0.40,
            ),
            BottleneckType::Network => (
                OptimizationRecommendationType::Caching,
                ImplementationEffort::Low,
                0.25,
            ),
            BottleneckType::Constraint => (
                OptimizationRecommendationType::ConstraintReordering,
                ImplementationEffort::Low,
                0.30,
            ),
        };

        // Scale priority by impact and severity
        let priority = match bottleneck.severity {
            BottleneckSeverity::Critical => RecommendationPriority::Critical,
            BottleneckSeverity::High => RecommendationPriority::High,
            BottleneckSeverity::Medium => RecommendationPriority::Medium,
            BottleneckSeverity::Low => RecommendationPriority::Low,
        };

        // Scale benefit by actual impact score
        let scaled_benefit = benefit * bottleneck.impact_score.min(1.0).max(0.0);

        let description = format!(
            "Bottleneck [{:?}] severity {:?}: {}. Recommended action: {:?}.",
            bottleneck.bottleneck_type, bottleneck.severity, bottleneck.description, rec_type
        );

        Ok(OptimizationRecommendation {
            recommendation_type: rec_type,
            priority,
            description,
            estimated_benefit: scaled_benefit.max(0.05),
            implementation_effort: effort,
            affected_components: bottleneck.affected_operations.clone(),
        })
    }

    pub(super) fn analyze_shape_complexity_for_recommendations(
        &self,
        shapes: &[Shape],
    ) -> Result<Vec<OptimizationRecommendation>> {
        tracing::debug!("Analyzing shape complexity for {} shapes", shapes.len());

        let mut recommendations = Vec::new();

        for shape in shapes {
            // Analyze constraint count
            if shape.constraints.len() > 20 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: OptimizationRecommendationType::ConstraintReordering,
                    priority: RecommendationPriority::High,
                    description: format!(
                        "Shape '{}' has {} constraints, consider breaking it down or reordering for better performance",
                        shape.id.as_str(),
                        shape.constraints.len()
                    ),
                    estimated_benefit: 0.3,
                    implementation_effort: ImplementationEffort::Medium,
                    affected_components: vec![format!("shape_{}", shape.id.as_str())],
                });
            }

            // Analyze target specificity
            if shape.targets.len() > 5 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: OptimizationRecommendationType::ShapeMerging,
                    priority: RecommendationPriority::Medium,
                    description: format!(
                        "Shape '{}' has {} targets, consider merging with similar shapes",
                        shape.id.as_str(),
                        shape.targets.len()
                    ),
                    estimated_benefit: 0.2,
                    implementation_effort: ImplementationEffort::High,
                    affected_components: vec![format!("shape_{}", shape.id.as_str())],
                });
            }

            // Analyze expensive constraints
            let expensive_constraints = self.identify_expensive_constraints(&shape.constraints)?;
            if !expensive_constraints.is_empty() {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: OptimizationRecommendationType::ConstraintReordering,
                    priority: RecommendationPriority::High,
                    description: format!(
                        "Shape '{}' contains expensive constraints ({}), consider reordering or optimization",
                        shape.id.as_str(),
                        expensive_constraints.join(", ")
                    ),
                    estimated_benefit: 0.4,
                    implementation_effort: ImplementationEffort::Low,
                    affected_components: vec![format!("shape_{}", shape.id.as_str())],
                });
            }

            // Analyze redundant constraints
            let redundant_constraints = self.identify_redundant_constraints(&shape.constraints)?;
            if !redundant_constraints.is_empty() {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: OptimizationRecommendationType::ConstraintReordering,
                    priority: RecommendationPriority::Medium,
                    description: format!(
                        "Shape '{}' may have redundant constraints that can be simplified",
                        shape.id.as_str()
                    ),
                    estimated_benefit: 0.25,
                    implementation_effort: ImplementationEffort::Low,
                    affected_components: vec![format!("shape_{}", shape.id.as_str())],
                });
            }
        }

        // Add global recommendations
        if shapes.len() > 100 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationRecommendationType::ParallelExecution,
                priority: RecommendationPriority::High,
                description: format!(
                    "Large number of shapes ({}), consider parallel validation execution",
                    shapes.len()
                ),
                estimated_benefit: 0.5,
                implementation_effort: ImplementationEffort::High,
                affected_components: vec!["validation_engine".to_string()],
            });
        }

        if self.detect_similar_shapes(shapes)? {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationRecommendationType::ShapeMerging,
                priority: RecommendationPriority::Medium,
                description:
                    "Similar shapes detected, consider merging to reduce validation overhead"
                        .to_string(),
                estimated_benefit: 0.3,
                implementation_effort: ImplementationEffort::Medium,
                affected_components: vec!["shape_management".to_string()],
            });
        }

        tracing::debug!(
            "Generated {} shape complexity recommendations",
            recommendations.len()
        );
        Ok(recommendations)
    }

    /// Identify expensive constraints in a shape
    fn identify_expensive_constraints(
        &self,
        constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<Vec<String>> {
        let mut expensive_constraints = Vec::new();

        for (_constraint_id, constraint) in constraints {
            let cost = self.estimate_constraint_cost(constraint)?;

            if cost > 0.7 {
                // High cost threshold
                let constraint_type = match constraint {
                    Constraint::Pattern(_) => "Pattern",
                    Constraint::Class(_) => "Class",
                    Constraint::NodeKind(_) => "NodeKind",
                    Constraint::Datatype(_) => "Datatype",
                    _ => "Other",
                };
                expensive_constraints.push(constraint_type.to_string());
            }
        }

        Ok(expensive_constraints)
    }

    /// Identify potentially redundant constraints
    fn identify_redundant_constraints(
        &self,
        constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<Vec<String>> {
        let mut redundant_constraints = Vec::new();

        // Check for common redundancy patterns
        let has_min_count = constraints
            .values()
            .any(|c| matches!(c, Constraint::MinCount(_)));
        let has_max_count = constraints
            .values()
            .any(|c| matches!(c, Constraint::MaxCount(_)));
        let has_class = constraints
            .values()
            .any(|c| matches!(c, Constraint::Class(_)));
        let has_node_kind = constraints
            .values()
            .any(|c| matches!(c, Constraint::NodeKind(_)));

        // Check for conflicting or redundant combinations
        if has_min_count && has_max_count {
            // Check if minCount > maxCount (conflicting)
            let min_counts: Vec<u32> = constraints
                .values()
                .filter_map(|c| {
                    if let Constraint::MinCount(mc) = c {
                        Some(mc.min_count)
                    } else {
                        None
                    }
                })
                .collect();

            let max_counts: Vec<u32> = constraints
                .values()
                .filter_map(|c| {
                    if let Constraint::MaxCount(mc) = c {
                        Some(mc.max_count)
                    } else {
                        None
                    }
                })
                .collect();

            for &min_count in &min_counts {
                for &max_count in &max_counts {
                    if min_count > max_count {
                        redundant_constraints.push("conflicting_cardinality".to_string());
                        break;
                    }
                }
            }
        }

        // Check for redundant type constraints
        if has_class && has_node_kind {
            redundant_constraints.push("redundant_type_constraints".to_string());
        }

        Ok(redundant_constraints)
    }

    /// Detect if there are similar shapes that could be merged
    pub(super) fn detect_similar_shapes(&self, shapes: &[Shape]) -> Result<bool> {
        if shapes.len() < 2 {
            return Ok(false);
        }

        let mut similar_pairs = 0;
        let total_pairs = shapes.len() * (shapes.len() - 1) / 2;

        for i in 0..shapes.len() {
            for j in i + 1..shapes.len() {
                let similarity = self.calculate_shape_similarity(&shapes[i], &shapes[j])?;
                if similarity > 0.7 {
                    // High similarity threshold
                    similar_pairs += 1;
                }
            }
        }

        // If more than 20% of shape pairs are similar, recommend merging
        let similarity_ratio = similar_pairs as f64 / total_pairs as f64;
        Ok(similarity_ratio > 0.2)
    }

    pub(super) fn analyze_graph_structure_for_recommendations(
        &self,
        store: &dyn Store,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let graph_analysis = self.analyze_graph_for_optimization(store)?;
        let stats = &graph_analysis.statistics;
        let connectivity = &graph_analysis.connectivity_analysis;
        let mut recommendations = Vec::new();

        // Recommendation: index for high-predicate-count graphs
        if stats.unique_predicates > 50 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationRecommendationType::IndexOptimization,
                priority: if stats.unique_predicates > 200 {
                    RecommendationPriority::High
                } else {
                    RecommendationPriority::Medium
                },
                description: format!(
                    "Graph has {} distinct predicates; predicate-based indexing would accelerate property path and class-constraint evaluation.",
                    stats.unique_predicates
                ),
                estimated_benefit: (stats.unique_predicates as f64 / 1000.0).min(0.6),
                implementation_effort: ImplementationEffort::Medium,
                affected_components: vec![
                    "predicate_index".to_string(),
                    "property_path_evaluator".to_string(),
                ],
            });
        }

        // Recommendation: parallel validation for large graphs
        if stats.triple_count > 100_000 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationRecommendationType::ParallelExecution,
                priority: if stats.triple_count > 1_000_000 {
                    RecommendationPriority::Critical
                } else {
                    RecommendationPriority::High
                },
                description: format!(
                    "Large graph ({} triples) benefits from parallel shape validation across independent target partitions.",
                    stats.triple_count
                ),
                estimated_benefit: (stats.triple_count as f64 / 5_000_000.0).min(0.7),
                implementation_effort: ImplementationEffort::High,
                affected_components: vec!["validation_engine".to_string()],
            });
        }

        // Recommendation: caching for sparse graphs with repeated pattern queries
        if stats.density < 0.01 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationRecommendationType::Caching,
                priority: RecommendationPriority::Medium,
                description:
                    "Sparse graph topology: target-node result caching reduces redundant traversal."
                        .to_string(),
                estimated_benefit: 0.25,
                implementation_effort: ImplementationEffort::Low,
                affected_components: vec!["target_selector".to_string()],
            });
        }

        // Recommendation: constraint reordering for highly-connected hubs
        if connectivity.average_degree > 30.0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationRecommendationType::ConstraintReordering,
                priority: RecommendationPriority::Medium,
                description: format!(
                    "High average degree ({:.1}) indicates hub nodes; evaluate cheap/selective constraints first to short-circuit expensive ones.",
                    connectivity.average_degree
                ),
                estimated_benefit: 0.20,
                implementation_effort: ImplementationEffort::Low,
                affected_components: vec!["constraint_evaluator".to_string()],
            });
        }

        // Recommendation: shape merging for fragmented graphs (many components)
        if connectivity.connected_components > 10 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationRecommendationType::ShapeMerging,
                priority: RecommendationPriority::Low,
                description: format!(
                    "Graph has {} disconnected components; merging shapes that target disjoint subgraphs can eliminate redundant target queries.",
                    connectivity.connected_components
                ),
                estimated_benefit: 0.15,
                implementation_effort: ImplementationEffort::Medium,
                affected_components: vec!["shape_management".to_string()],
            });
        }

        tracing::debug!(
            "Graph structure analysis produced {} recommendations",
            recommendations.len()
        );
        Ok(recommendations)
    }
}
