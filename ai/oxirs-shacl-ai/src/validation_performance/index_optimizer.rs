//! Index optimization for SHACL validation performance
//!
//! This module provides intelligent index recommendations and optimization
//! to improve query performance during SHACL validation.

use crate::{PropertyConstraint, ShaclAiError, Shape};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::IndexUsagePatterns;

/// Index optimization manager
#[derive(Debug, Clone)]
pub struct IndexOptimizer {
    index_usage_stats: HashMap<String, IndexUsageStats>,
}

/// Index usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsageStats {
    pub index_name: String,
    pub usage_count: u64,
    pub last_used: DateTime<Utc>,
    pub effectiveness_score: f64,
}

/// Index recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRecommendation {
    pub index_name: String,
    pub index_type: IndexType,
    pub columns: Vec<String>,
    pub estimated_benefit: f64,
    pub creation_cost: IndexCreationCost,
    pub maintenance_overhead: f64,
}

/// Types of indexes that can be recommended
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// Hash index for equality comparisons
    Hash,
    /// B-tree index for range queries and sorting
    BTree,
    /// Full-text index for text search
    FullText,
    /// Bitmap index for low-cardinality data
    Bitmap,
}

/// Index creation cost estimation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexCreationCost {
    Low,
    Medium,
    High,
}

impl Default for IndexOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexOptimizer {
    pub fn new() -> Self {
        Self {
            index_usage_stats: HashMap::new(),
        }
    }

    /// Recommend indexes for a set of shapes
    pub fn recommend_indexes(
        &self,
        shapes: &[Shape],
    ) -> Result<Vec<IndexRecommendation>, ShaclAiError> {
        let mut recommendations = Vec::new();
        let usage_patterns = self.analyze_usage_patterns(shapes);

        for shape in shapes {
            let shape_recommendations = self.analyze_shape_indexes(shape, &usage_patterns);
            recommendations.extend(shape_recommendations);
        }

        // Sort recommendations by priority (estimated benefit)
        recommendations.sort_by(|a, b| {
            b.estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(recommendations)
    }

    /// Analyze usage patterns across all shapes to identify common access patterns
    fn analyze_usage_patterns(&self, shapes: &[Shape]) -> IndexUsagePatterns {
        let mut patterns = IndexUsagePatterns {
            index_name: "global_patterns".to_string(),
            access_count: 0,
            last_accessed: Utc::now(),
            selectivity: 0.0,
            cost_estimate: 0.0,
        };

        for shape in shapes {
            for constraint in &shape.property_constraints {
                patterns.access_count += 1;

                // Update last accessed time
                patterns.last_accessed = Utc::now();

                // Calculate selectivity based on constraint types
                if self.is_equality_constraint(constraint) {
                    patterns.selectivity += 0.8; // High selectivity for equality
                } else if self.is_range_constraint(constraint) {
                    patterns.selectivity += 0.4; // Medium selectivity for ranges
                } else {
                    patterns.selectivity += 0.2; // Low selectivity for others
                }
            }
        }

        // Normalize selectivity
        if patterns.access_count > 0 {
            patterns.selectivity /= patterns.access_count as f64;
        }

        patterns
    }

    /// Analyze and recommend indexes for a specific shape
    fn analyze_shape_indexes(
        &self,
        shape: &Shape,
        patterns: &IndexUsagePatterns,
    ) -> Vec<IndexRecommendation> {
        let mut recommendations = Vec::new();

        // Single-column indexes
        for constraint in &shape.property_constraints {
            if let Some(recommendation) =
                self.recommend_single_column_index(shape, constraint, patterns)
            {
                recommendations.push(recommendation);
            }
        }

        // Multi-column indexes for related constraints
        let composite_recommendations = self.recommend_composite_indexes(shape, patterns);
        recommendations.extend(composite_recommendations);

        // Full-text indexes for text search
        let fulltext_recommendations = self.recommend_fulltext_indexes(shape);
        recommendations.extend(fulltext_recommendations);

        recommendations
    }

    /// Recommend single-column index for a constraint
    fn recommend_single_column_index(
        &self,
        shape: &Shape,
        constraint: &PropertyConstraint,
        patterns: &IndexUsagePatterns,
    ) -> Option<IndexRecommendation> {
        let access_frequency = patterns.access_count;

        // Only recommend if accessed frequently enough
        if access_frequency < 2 {
            return None;
        }

        let index_type = self.determine_optimal_index_type(constraint);
        let estimated_benefit =
            self.calculate_index_benefit(constraint, &index_type, access_frequency as usize);
        let creation_cost = self.estimate_creation_cost(&index_type, 1);

        Some(IndexRecommendation {
            index_name: format!(
                "idx_{}_{}",
                shape.id.replace([':', '/', '#'], "_"),
                constraint.path.replace([':', '/', '#'], "_")
            ),
            index_type: index_type.clone(),
            columns: vec![constraint.path.clone()],
            estimated_benefit,
            creation_cost,
            maintenance_overhead: self.calculate_maintenance_overhead(&index_type),
        })
    }

    /// Recommend composite indexes for multiple related constraints
    fn recommend_composite_indexes(
        &self,
        shape: &Shape,
        patterns: &IndexUsagePatterns,
    ) -> Vec<IndexRecommendation> {
        let mut recommendations = Vec::new();

        // Find constraints that are often used together
        let constraint_groups = self.find_constraint_groups(shape, patterns);

        for group in constraint_groups {
            if group.len() >= 2 && group.len() <= 4 {
                // Optimal composite index size
                let index_name = format!(
                    "idx_{}_composite_{}",
                    shape.id.replace([':', '/', '#'], "_"),
                    group.len()
                );

                let estimated_benefit = self.calculate_composite_index_benefit(&group, patterns);
                let creation_cost = self.estimate_creation_cost(&IndexType::BTree, group.len());

                recommendations.push(IndexRecommendation {
                    index_name,
                    index_type: IndexType::BTree, // Composite indexes are typically B-tree
                    columns: group,
                    estimated_benefit,
                    creation_cost,
                    maintenance_overhead: 0.08, // Higher overhead for composite indexes
                });
            }
        }

        recommendations
    }

    /// Recommend full-text indexes for text search constraints
    fn recommend_fulltext_indexes(&self, shape: &Shape) -> Vec<IndexRecommendation> {
        let mut recommendations = Vec::new();

        for constraint in &shape.property_constraints {
            if self.is_text_search_constraint(constraint) {
                recommendations.push(IndexRecommendation {
                    index_name: format!(
                        "idx_{}_{}_fulltext",
                        shape.id.replace([':', '/', '#'], "_"),
                        constraint.path.replace([':', '/', '#'], "_")
                    ),
                    index_type: IndexType::FullText,
                    columns: vec![constraint.path.clone()],
                    estimated_benefit: 0.8, // High benefit for text search
                    creation_cost: IndexCreationCost::Medium,
                    maintenance_overhead: 0.15, // Higher overhead for full-text indexes
                });
            }
        }

        recommendations
    }

    /// Determine optimal index type for a constraint
    fn determine_optimal_index_type(&self, constraint: &PropertyConstraint) -> IndexType {
        if self.is_equality_constraint(constraint) {
            IndexType::Hash // Hash indexes are optimal for equality checks
        } else if self.is_range_constraint(constraint) {
            IndexType::BTree // B-tree indexes are optimal for range queries
        } else if self.is_text_search_constraint(constraint) {
            IndexType::FullText
        } else if self.is_high_cardinality_constraint(constraint) {
            IndexType::Bitmap // Bitmap indexes for low-cardinality data
        } else {
            IndexType::BTree // Default to B-tree
        }
    }

    /// Check if constraint is equality-based
    fn is_equality_constraint(&self, constraint: &PropertyConstraint) -> bool {
        constraint.path.contains("=")
            || constraint.path.contains("eq")
            || constraint.path.contains("equals")
    }

    /// Check if constraint is range-based
    fn is_range_constraint(&self, constraint: &PropertyConstraint) -> bool {
        constraint.path.contains(">")
            || constraint.path.contains("<")
            || constraint.path.contains("range")
            || constraint.path.contains("between")
    }

    /// Check if constraint is pattern-based
    fn is_pattern_constraint(&self, constraint: &PropertyConstraint) -> bool {
        constraint.path.contains("pattern")
            || constraint.path.contains("regex")
            || constraint.path.contains("match")
    }

    /// Check if constraint is for text search
    fn is_text_search_constraint(&self, constraint: &PropertyConstraint) -> bool {
        constraint.path.contains("search")
            || constraint.path.contains("text")
            || constraint.path.contains("contains")
            || constraint.path.contains("fulltext")
    }

    /// Check if constraint has high cardinality
    fn is_high_cardinality_constraint(&self, constraint: &PropertyConstraint) -> bool {
        // Heuristic: long paths or specific keywords indicate high cardinality
        constraint.path.len() > 30
            || constraint.path.contains("unique")
            || constraint.path.contains("distinct")
    }

    /// Calculate estimated benefit of an index
    fn calculate_index_benefit(
        &self,
        constraint: &PropertyConstraint,
        index_type: &IndexType,
        access_frequency: usize,
    ) -> f64 {
        let base_benefit = match index_type {
            IndexType::Hash => 0.7,     // 70% improvement for equality
            IndexType::BTree => 0.5,    // 50% improvement for ranges
            IndexType::FullText => 0.8, // 80% improvement for text search
            IndexType::Bitmap => 0.4,   // 40% improvement for low cardinality
        };

        // Adjust benefit based on access frequency
        let frequency_multiplier = (access_frequency as f64).ln().max(1.0);

        (base_benefit * frequency_multiplier).min(0.95) // Cap at 95% improvement
    }

    /// Calculate benefit of composite index
    fn calculate_composite_index_benefit(
        &self,
        columns: &[String],
        patterns: &IndexUsagePatterns,
    ) -> f64 {
        let combined_frequency = patterns.access_count;

        let base_benefit = 0.6; // Base 60% improvement for composite indexes
        let frequency_factor = (combined_frequency as f64 / columns.len() as f64)
            .ln()
            .max(1.0);

        (base_benefit * frequency_factor).min(0.9) // Cap at 90% improvement
    }

    /// Estimate index creation cost
    fn estimate_creation_cost(
        &self,
        index_type: &IndexType,
        column_count: usize,
    ) -> IndexCreationCost {
        match (index_type, column_count) {
            (IndexType::Hash, 1) => IndexCreationCost::Low,
            (IndexType::BTree, 1) => IndexCreationCost::Low,
            (IndexType::BTree, 2..=3) => IndexCreationCost::Medium,
            (IndexType::BTree, _) => IndexCreationCost::High,
            (IndexType::FullText, _) => IndexCreationCost::High,
            (IndexType::Bitmap, _) => IndexCreationCost::Medium,
            _ => IndexCreationCost::Medium,
        }
    }

    /// Calculate maintenance overhead for index type
    fn calculate_maintenance_overhead(&self, index_type: &IndexType) -> f64 {
        match index_type {
            IndexType::Hash => 0.03,     // 3% overhead
            IndexType::BTree => 0.05,    // 5% overhead
            IndexType::FullText => 0.15, // 15% overhead
            IndexType::Bitmap => 0.08,   // 8% overhead
        }
    }

    /// Find groups of constraints that are often used together
    fn find_constraint_groups(
        &self,
        shape: &Shape,
        _patterns: &IndexUsagePatterns,
    ) -> Vec<Vec<String>> {
        let mut groups = Vec::new();

        // Simple grouping: group constraints by similar prefixes
        let mut prefix_groups: HashMap<String, Vec<String>> = HashMap::new();

        for constraint in &shape.property_constraints {
            let prefix = if let Some(pos) = constraint.path.rfind('/') {
                constraint.path[..pos].to_string()
            } else if let Some(pos) = constraint.path.rfind('#') {
                constraint.path[..pos].to_string()
            } else {
                "default".to_string()
            };

            prefix_groups
                .entry(prefix)
                .or_insert_with(Vec::new)
                .push(constraint.path.clone());
        }

        // Only include groups with multiple constraints
        for (_, paths) in prefix_groups {
            if paths.len() >= 2 {
                groups.push(paths);
            }
        }

        groups
    }

    /// Update index usage statistics
    pub fn record_index_usage(&mut self, index_name: &str) {
        let stats = self
            .index_usage_stats
            .entry(index_name.to_string())
            .or_insert_with(|| IndexUsageStats {
                index_name: index_name.to_string(),
                usage_count: 0,
                last_used: Utc::now(),
                effectiveness_score: 0.5,
            });

        stats.usage_count += 1;
        stats.last_used = Utc::now();

        // Update effectiveness score based on usage patterns
        stats.effectiveness_score = (stats.usage_count as f64).ln() / 10.0;
        stats.effectiveness_score = stats.effectiveness_score.min(1.0).max(0.0);
    }

    /// Get index usage statistics
    pub fn get_usage_stats(&self) -> &HashMap<String, IndexUsageStats> {
        &self.index_usage_stats
    }

    /// Get most effective indexes
    pub fn get_most_effective_indexes(&self, limit: usize) -> Vec<&IndexUsageStats> {
        let mut stats: Vec<&IndexUsageStats> = self.index_usage_stats.values().collect();
        stats.sort_by(|a, b| {
            b.effectiveness_score
                .partial_cmp(&a.effectiveness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        stats.into_iter().take(limit).collect()
    }

    /// Get underutilized indexes
    pub fn get_underutilized_indexes(&self, threshold: f64) -> Vec<&IndexUsageStats> {
        self.index_usage_stats
            .values()
            .filter(|stats| stats.effectiveness_score < threshold)
            .collect()
    }
}
