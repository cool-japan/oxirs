//! Cost analysis for materialized views
//!
//! This module provides cost analysis and benefit estimation for materialized view decisions.

use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;
use tracing::debug;

use crate::planner::planning::TriplePattern;
use crate::ServiceRegistry;
use crate::planner::planning::types::QueryInfo;

use super::types::*;

/// Cost analyzer for materialized views
#[derive(Debug)]
pub struct ViewCostAnalyzer {
    cost_models: HashMap<String, CostModel>,
    historical_data: HashMap<String, HistoricalCostData>,
}

impl ViewCostAnalyzer {
    /// Create a new cost analyzer
    pub fn new() -> Self {
        Self {
            cost_models: HashMap::new(),
            historical_data: HashMap::new(),
        }
    }

    /// Estimate the cost of creating a materialized view
    pub async fn estimate_creation_cost(
        &self,
        definition: &ViewDefinition,
        registry: &ServiceRegistry,
    ) -> Result<ViewCreationCost> {
        debug!("Estimating creation cost for view: {}", definition.name);

        let mut total_cost = 0.0;
        let mut storage_cost = 0.0;
        let mut computation_cost = 0.0;
        let mut network_cost = 0.0;

        // Estimate computation cost based on query complexity
        computation_cost += self.estimate_query_complexity_cost(definition);

        // Estimate storage cost based on expected data size
        storage_cost += self.estimate_storage_cost(definition, registry).await?;

        // Estimate network cost for data transfer
        network_cost += self.estimate_network_cost(definition, registry).await?;

        total_cost = computation_cost + storage_cost + network_cost;

        Ok(ViewCreationCost {
            total_cost,
            computation_cost,
            storage_cost,
            network_cost,
            estimated_duration: Duration::from_secs((total_cost * 60.0) as u64),
        })
    }

    /// Estimate the benefit of using a materialized view for a query
    pub async fn estimate_query_benefit(
        &self,
        view: &MaterializedView,
        query_info: &QueryInfo,
        registry: &ServiceRegistry,
    ) -> Result<ViewBenefit> {
        debug!("Estimating query benefit for view: {}", view.id);

        let direct_query_cost = self
            .estimate_direct_query_cost(query_info, registry)
            .await?;
        let view_query_cost = self.estimate_view_query_cost(view, query_info)?;

        let cost_saving = direct_query_cost - view_query_cost;
        let performance_improvement = if view_query_cost > 0.0 {
            direct_query_cost / view_query_cost
        } else {
            1.0
        };

        Ok(ViewBenefit {
            cost_saving,
            performance_improvement,
            cache_hit_probability: self.estimate_cache_hit_probability(view, query_info),
            data_freshness_score: self.calculate_freshness_score(view),
        })
    }

    /// Estimate the maintenance cost for a materialized view
    pub fn estimate_maintenance_cost(&self, view: &MaterializedView) -> MaintenanceCost {
        let base_cost = view.size_bytes as f64 * 0.001; // $0.001 per MB
        let refresh_frequency =
            24.0 / view.definition.estimate_freshness_requirement().as_secs() as f64 * 3600.0;
        let complexity_multiplier = view.definition.complexity_score() / 10.0;

        MaintenanceCost {
            daily_cost: base_cost * refresh_frequency * complexity_multiplier,
            refresh_cost: base_cost * complexity_multiplier,
            storage_cost: view.size_bytes as f64 * 0.0001, // $0.0001 per MB per day
        }
    }

    /// Recommend whether to create a materialized view
    pub async fn recommend_view_creation(
        &self,
        definition: &ViewDefinition,
        query_patterns: &[TriplePattern],
        registry: &ServiceRegistry,
    ) -> Result<ViewRecommendation> {
        let creation_cost = self.estimate_creation_cost(definition, registry).await?;

        // Estimate benefit based on pattern analysis
        let estimated_benefit = self.estimate_pattern_benefit(definition, query_patterns);

        let confidence = self.calculate_recommendation_confidence(definition, query_patterns);

        let reason = if estimated_benefit > creation_cost.total_cost * 2.0 {
            RecommendationReason::HighQueryFrequency
        } else if definition.query_patterns().len() > 5 {
            RecommendationReason::ExpensiveJoins
        } else {
            RecommendationReason::ImprovedCacheHitRatio
        };

        Ok(ViewRecommendation {
            view_id: definition.name.clone(),
            reason,
            estimated_benefit,
            implementation_cost: creation_cost.total_cost,
            confidence,
        })
    }

    // Private helper methods

    fn estimate_query_complexity_cost(&self, definition: &ViewDefinition) -> f64 {
        let pattern_count = definition.query_patterns().len();
        let filter_count = definition.filters().len();
        let dependency_count = definition.dependencies.len();

        // Base cost calculation
        (pattern_count * 10 + filter_count * 15 + dependency_count * 20) as f64
    }

    async fn estimate_storage_cost(
        &self,
        definition: &ViewDefinition,
        registry: &ServiceRegistry,
    ) -> Result<f64> {
        let mut estimated_rows = 1000.0; // Default estimate
        let avg_row_size = 256.0; // Average row size in bytes

        // Estimate based on service data if available
        for pattern in &definition.source_patterns {
            if let Some(service) = registry.get_service(&pattern.service_id) {
                // Use service metadata to improve estimates
                estimated_rows *= pattern.estimated_selectivity;
            }
        }

        let estimated_size_bytes = estimated_rows * avg_row_size;
        Ok(estimated_size_bytes * 0.0001) // $0.0001 per MB
    }

    async fn estimate_network_cost(
        &self,
        definition: &ViewDefinition,
        registry: &ServiceRegistry,
    ) -> Result<f64> {
        let mut total_transfer = 0.0;

        for pattern in &definition.source_patterns {
            if let Some(service) = registry.get_service(&pattern.service_id) {
                // Estimate data transfer based on service location and pattern selectivity
                total_transfer += 1000.0 * pattern.estimated_selectivity; // KB
            }
        }

        Ok(total_transfer * 0.01) // $0.01 per GB
    }

    async fn estimate_direct_query_cost(
        &self,
        query_info: &QueryInfo,
        registry: &ServiceRegistry,
    ) -> Result<f64> {
        let base_cost = query_info.patterns.len() as f64 * 10.0;
        let filter_cost = query_info.filters.len() as f64 * 5.0;
        let complexity_cost = query_info.complexity as f64 * 0.1;

        Ok(base_cost + filter_cost + complexity_cost)
    }

    fn estimate_view_query_cost(
        &self,
        view: &MaterializedView,
        query_info: &QueryInfo,
    ) -> Result<f64> {
        // Cost is much lower when using a materialized view
        let base_view_cost = 1.0; // Fixed cost for view access
        let pattern_coverage = self.calculate_pattern_coverage(view, &query_info.patterns);

        Ok(base_view_cost * (1.0 - pattern_coverage + 0.1))
    }

    fn estimate_cache_hit_probability(
        &self,
        view: &MaterializedView,
        query_info: &QueryInfo,
    ) -> f64 {
        // Simplified cache hit probability based on pattern matching
        let pattern_match_score = self.calculate_pattern_coverage(view, &query_info.patterns);
        let freshness_factor = if view.is_stale { 0.5 } else { 1.0 };

        pattern_match_score * freshness_factor
    }

    fn calculate_freshness_score(&self, view: &MaterializedView) -> f64 {
        if view.is_stale {
            0.5
        } else if let Some(last_refresh) = view.last_refresh {
            let age = chrono::Utc::now().signed_duration_since(last_refresh);
            let max_age = view.definition.estimate_freshness_requirement();

            let age_ratio = age.num_seconds() as f64 / max_age.as_secs() as f64;
            (1.0 - age_ratio).max(0.0).min(1.0)
        } else {
            0.0
        }
    }

    fn calculate_pattern_coverage(
        &self,
        view: &MaterializedView,
        query_patterns: &[TriplePattern],
    ) -> f64 {
        if query_patterns.is_empty() {
            return 0.0;
        }

        let view_patterns = view.definition.query_patterns();
        let covered_patterns = query_patterns
            .iter()
            .filter(|qp| view_patterns.iter().any(|vp| patterns_match(qp, vp)))
            .count();

        covered_patterns as f64 / query_patterns.len() as f64
    }

    fn estimate_pattern_benefit(
        &self,
        definition: &ViewDefinition,
        query_patterns: &[TriplePattern],
    ) -> f64 {
        let pattern_count = query_patterns.len();
        let view_patterns = definition.query_patterns();

        let coverage = if pattern_count == 0 {
            0.0
        } else {
            let covered = query_patterns
                .iter()
                .filter(|qp| view_patterns.iter().any(|vp| patterns_match(qp, vp)))
                .count();
            covered as f64 / pattern_count as f64
        };

        coverage * 100.0 // Base benefit score
    }

    fn calculate_recommendation_confidence(
        &self,
        definition: &ViewDefinition,
        query_patterns: &[TriplePattern],
    ) -> f64 {
        let mut confidence = 0.5; // Base confidence

        // Increase confidence based on pattern coverage
        let coverage = self.estimate_pattern_benefit(definition, query_patterns) / 100.0;
        confidence += coverage * 0.3;

        // Increase confidence based on query complexity
        if definition.complexity_score() > 50.0 {
            confidence += 0.2;
        }

        confidence.min(1.0)
    }
}

impl Default for ViewCostAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern matching helper function
fn patterns_match(query_pattern: &TriplePattern, view_pattern: &TriplePattern) -> bool {
    (query_pattern.subject.is_none() || query_pattern.subject == view_pattern.subject)
        && (query_pattern.predicate.is_none() || query_pattern.predicate == view_pattern.predicate)
        && (query_pattern.object.is_none() || query_pattern.object == view_pattern.object)
}

/// Cost model for different types of operations
#[derive(Debug, Clone)]
pub struct CostModel {
    pub computation_factor: f64,
    pub storage_factor: f64,
    pub network_factor: f64,
    pub base_cost: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            computation_factor: 1.0,
            storage_factor: 1.0,
            network_factor: 1.0,
            base_cost: 10.0,
        }
    }
}

/// Historical cost data for learning and optimization
#[derive(Debug, Clone)]
pub struct HistoricalCostData {
    pub average_creation_cost: f64,
    pub average_maintenance_cost: f64,
    pub average_benefit: f64,
    pub sample_count: u64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Cost breakdown for view creation
#[derive(Debug, Clone)]
pub struct ViewCreationCost {
    pub total_cost: f64,
    pub computation_cost: f64,
    pub storage_cost: f64,
    pub network_cost: f64,
    pub estimated_duration: Duration,
}

/// Benefit analysis for using a materialized view
#[derive(Debug, Clone)]
pub struct ViewBenefit {
    pub cost_saving: f64,
    pub performance_improvement: f64,
    pub cache_hit_probability: f64,
    pub data_freshness_score: f64,
}

/// Maintenance cost breakdown
#[derive(Debug, Clone)]
pub struct MaintenanceCost {
    pub daily_cost: f64,
    pub refresh_cost: f64,
    pub storage_cost: f64,
}
