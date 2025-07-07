//! # Advanced Query Optimizer for TDB Storage
//!
//! Provides intelligent query optimization using statistics, cost models,
//! and machine learning-based optimization hints.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::nodes::NodeId;
use crate::triple_store::TripleStoreStats;

/// Query pattern for optimization analysis
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryPattern {
    pub subject: Option<NodeId>,
    pub predicate: Option<NodeId>,
    pub object: Option<NodeId>,
}

impl QueryPattern {
    /// Create a new query pattern
    pub fn new(subject: Option<NodeId>, predicate: Option<NodeId>, object: Option<NodeId>) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Get selectivity estimate (1.0 = most selective, 0.0 = least selective)
    pub fn selectivity(&self) -> f64 {
        let bound_vars = [&self.subject, &self.predicate, &self.object]
            .iter()
            .filter(|opt| opt.is_some())
            .count();

        match bound_vars {
            3 => 1.0, // Most selective - all bound
            2 => 0.7, // High selectivity - two bound
            1 => 0.3, // Medium selectivity - one bound
            0 => 0.0, // Least selective - all unbound
            _ => 1.0, // Default to high selectivity for unexpected values
        }
    }

    /// Get pattern type for index selection
    pub fn pattern_type(&self) -> PatternType {
        match (&self.subject, &self.predicate, &self.object) {
            (Some(_), Some(_), Some(_)) => PatternType::FullyBound,
            (Some(_), Some(_), None) => PatternType::SubjectPredicate,
            (Some(_), None, Some(_)) => PatternType::SubjectObject,
            (None, Some(_), Some(_)) => PatternType::PredicateObject,
            (Some(_), None, None) => PatternType::SubjectOnly,
            (None, Some(_), None) => PatternType::PredicateOnly,
            (None, None, Some(_)) => PatternType::ObjectOnly,
            (None, None, None) => PatternType::Unbound,
        }
    }
}

/// Pattern types for index selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    FullyBound,
    SubjectPredicate,
    SubjectObject,
    PredicateObject,
    SubjectOnly,
    PredicateOnly,
    ObjectOnly,
    Unbound,
}

/// Available index types for query execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    SPO, // Subject-Predicate-Object
    POS, // Predicate-Object-Subject
    OSP, // Object-Subject-Predicate
    SOP, // Subject-Object-Predicate
    PSO, // Predicate-Subject-Object
    OPS, // Object-Predicate-Subject
}

impl IndexType {
    /// Get the optimal index for a given pattern type
    pub fn optimal_for_pattern(pattern_type: PatternType) -> Self {
        match pattern_type {
            PatternType::FullyBound => IndexType::SPO, // Any index works, SPO is default
            PatternType::SubjectPredicate => IndexType::SPO, // SPO index optimal
            PatternType::SubjectObject => IndexType::SOP, // SOP index optimal
            PatternType::PredicateObject => IndexType::POS, // POS index optimal
            PatternType::SubjectOnly => IndexType::SPO, // SPO or PSO works
            PatternType::PredicateOnly => IndexType::POS, // POS or PSO works
            PatternType::ObjectOnly => IndexType::OSP, // OSP or OPS works
            PatternType::Unbound => IndexType::SPO,    // Default to SPO
        }
    }

    /// Get index efficiency score for a pattern (0.0-1.0)
    pub fn efficiency_for_pattern(&self, pattern_type: PatternType) -> f64 {
        match (self, pattern_type) {
            (IndexType::SPO, PatternType::SubjectPredicate) => 1.0,
            (IndexType::SPO, PatternType::SubjectOnly) => 0.9,
            (IndexType::SPO, PatternType::FullyBound) => 0.8,

            (IndexType::POS, PatternType::PredicateObject) => 1.0,
            (IndexType::POS, PatternType::PredicateOnly) => 0.9,
            (IndexType::POS, PatternType::FullyBound) => 0.8,

            (IndexType::OSP, PatternType::ObjectOnly) => 0.9,
            (IndexType::OSP, PatternType::FullyBound) => 0.8,

            (IndexType::SOP, PatternType::SubjectObject) => 1.0,
            (IndexType::SOP, PatternType::SubjectOnly) => 0.7,
            (IndexType::SOP, PatternType::FullyBound) => 0.8,

            (IndexType::PSO, PatternType::PredicateOnly) => 0.8,
            (IndexType::PSO, PatternType::FullyBound) => 0.7,

            (IndexType::OPS, PatternType::ObjectOnly) => 0.8,
            (IndexType::OPS, PatternType::FullyBound) => 0.7,

            _ => 0.3, // Poor match
        }
    }
}

/// Query execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStats {
    pub pattern: QueryPattern,
    pub execution_time: Duration,
    pub result_count: u64,
    pub index_used: IndexType,
    pub cost: f64,
    pub timestamp: DateTime<Utc>,
}

/// Query cost model for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCostModel {
    pub index_cost_factor: f64,
    pub result_size_factor: f64,
    pub selectivity_factor: f64,
    pub base_cost: f64,
}

impl Default for QueryCostModel {
    fn default() -> Self {
        Self {
            index_cost_factor: 1.0,
            result_size_factor: 0.1,
            selectivity_factor: 0.5,
            base_cost: 10.0,
        }
    }
}

impl QueryCostModel {
    /// Estimate cost for a query pattern using a specific index
    pub fn estimate_cost(
        &self,
        pattern: &QueryPattern,
        index: IndexType,
        stats: &TripleStoreStats,
    ) -> f64 {
        let selectivity = pattern.selectivity();
        let efficiency = index.efficiency_for_pattern(pattern.pattern_type());
        let estimated_results = (stats.total_triples as f64 * (1.0 - selectivity)) as u64;

        self.base_cost
            + (self.index_cost_factor * (1.0 - efficiency))
            + (self.result_size_factor * estimated_results as f64)
            + (self.selectivity_factor * (1.0 - selectivity))
    }
}

/// Query optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub pattern: QueryPattern,
    pub recommended_index: IndexType,
    pub estimated_cost: f64,
    pub confidence: f64,
    pub reasoning: String,
}

/// Index creation recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRecommendation {
    pub index_type: IndexType,
    pub pattern_frequency: u64,
    pub average_cost: f64,
    pub potential_savings: f64,
    pub confidence: f64,
    pub reason: String,
}

/// Adaptive index manager for smart indexing
#[derive(Debug, Clone)]
pub struct AdaptiveIndexManager {
    pattern_frequency: HashMap<QueryPattern, u64>,
    pattern_costs: HashMap<QueryPattern, Vec<f64>>,
    index_usage: HashMap<IndexType, u64>,
    creation_threshold: u64,
    cost_improvement_threshold: f64,
}

impl AdaptiveIndexManager {
    /// Create a new adaptive index manager
    pub fn new() -> Self {
        Self {
            pattern_frequency: HashMap::new(),
            pattern_costs: HashMap::new(),
            index_usage: HashMap::new(),
            creation_threshold: 100, // Create index after 100 queries of same pattern
            cost_improvement_threshold: 0.3, // 30% cost improvement required
        }
    }

    /// Record a query pattern execution
    pub fn record_pattern_execution(
        &mut self,
        pattern: &QueryPattern,
        cost: f64,
        index_used: IndexType,
    ) {
        // Update pattern frequency
        *self.pattern_frequency.entry(pattern.clone()).or_insert(0) += 1;

        // Record cost for this pattern
        self.pattern_costs
            .entry(pattern.clone())
            .or_default()
            .push(cost);

        // Update index usage
        *self.index_usage.entry(index_used).or_insert(0) += 1;
    }

    /// Get index creation recommendations based on usage patterns
    pub fn get_index_recommendations(&self) -> Vec<IndexRecommendation> {
        let mut recommendations = Vec::new();

        for (pattern, frequency) in &self.pattern_frequency {
            if *frequency >= self.creation_threshold {
                let optimal_index = IndexType::optimal_for_pattern(pattern.pattern_type());
                let current_usage = self.index_usage.get(&optimal_index).copied().unwrap_or(0);

                // Only recommend if the optimal index is underused
                if current_usage < *frequency / 2 {
                    let empty_costs = Vec::new();
                    let costs = self.pattern_costs.get(pattern).unwrap_or(&empty_costs);
                    let average_cost = if costs.is_empty() {
                        0.0
                    } else {
                        costs.iter().sum::<f64>() / costs.len() as f64
                    };

                    let potential_savings = average_cost * self.cost_improvement_threshold;
                    let confidence = (*frequency as f64 / self.creation_threshold as f64).min(1.0);

                    if potential_savings > 10.0 {
                        // Only recommend if significant savings
                        recommendations.push(IndexRecommendation {
                            index_type: optimal_index,
                            pattern_frequency: *frequency,
                            average_cost,
                            potential_savings,
                            confidence,
                            reason: format!(
                                "Pattern {:?} executed {} times with average cost {:.2}, optimal index {} could reduce cost by {:.2}",
                                pattern.pattern_type(), frequency, average_cost,
                                format!("{:?}", optimal_index), potential_savings
                            ),
                        });
                    }
                }
            }
        }

        // Sort by potential savings (highest first)
        recommendations.sort_by(|a, b| {
            b.potential_savings
                .partial_cmp(&a.potential_savings)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        recommendations
    }

    /// Get pattern analysis summary
    pub fn get_pattern_analysis(&self) -> PatternAnalysisSummary {
        let total_patterns = self.pattern_frequency.len();
        let total_executions: u64 = self.pattern_frequency.values().sum();
        let most_frequent_pattern = self
            .pattern_frequency
            .iter()
            .max_by_key(|(_, freq)| *freq)
            .map(|(pattern, freq)| (pattern.clone(), *freq));

        let index_efficiency: HashMap<IndexType, f64> = self
            .index_usage
            .iter()
            .map(|(index, usage)| {
                let efficiency = *usage as f64 / total_executions as f64;
                (*index, efficiency)
            })
            .collect();

        PatternAnalysisSummary {
            total_patterns,
            total_executions,
            most_frequent_pattern,
            index_efficiency,
            recommendations_available: self.get_index_recommendations().len(),
        }
    }
}

/// Pattern analysis summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysisSummary {
    pub total_patterns: usize,
    pub total_executions: u64,
    pub most_frequent_pattern: Option<(QueryPattern, u64)>,
    pub index_efficiency: HashMap<IndexType, f64>,
    pub recommendations_available: usize,
}

/// Advanced query optimizer with ML-based recommendations and adaptive indexing
pub struct QueryOptimizer {
    stats_history: Arc<RwLock<Vec<QueryStats>>>,
    cost_model: QueryCostModel,
    pattern_cache: Arc<RwLock<HashMap<QueryPattern, OptimizationRecommendation>>>,
    adaptive_index_manager: Arc<RwLock<AdaptiveIndexManager>>,
    max_history_size: usize,
}

impl QueryOptimizer {
    /// Create a new query optimizer
    pub fn new() -> Self {
        Self {
            stats_history: Arc::new(RwLock::new(Vec::new())),
            cost_model: QueryCostModel::default(),
            pattern_cache: Arc::new(RwLock::new(HashMap::new())),
            adaptive_index_manager: Arc::new(RwLock::new(AdaptiveIndexManager::new())),
            max_history_size: 10_000,
        }
    }

    /// Record query execution statistics
    pub fn record_execution(&self, stats: QueryStats) -> Result<()> {
        let mut history = self
            .stats_history
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on stats history"))?;

        history.push(stats.clone());

        // Limit history size
        if history.len() > self.max_history_size {
            let history_len = history.len();
            history.drain(0..history_len - self.max_history_size);
        }

        drop(history); // Release lock early

        // Record pattern execution for adaptive indexing
        if let Ok(mut manager) = self.adaptive_index_manager.write() {
            manager.record_pattern_execution(&stats.pattern, stats.cost, stats.index_used);
        }

        Ok(())
    }

    /// Get optimization recommendation for a query pattern
    pub fn recommend_optimization(
        &self,
        pattern: &QueryPattern,
        store_stats: &TripleStoreStats,
    ) -> Result<OptimizationRecommendation> {
        // Check cache first
        if let Ok(cache) = self.pattern_cache.read() {
            if let Some(cached_rec) = cache.get(pattern) {
                return Ok(cached_rec.clone());
            }
        }

        let pattern_type = pattern.pattern_type();
        let optimal_index = IndexType::optimal_for_pattern(pattern_type);

        // Calculate costs for all possible indices
        let mut best_index = optimal_index;
        let mut best_cost = self
            .cost_model
            .estimate_cost(pattern, optimal_index, store_stats);
        let mut reasoning = format!(
            "Optimal index {:?} for pattern {:?}",
            optimal_index, pattern_type
        );

        // Check if historical data suggests a better index
        if let Ok(history) = self.stats_history.read() {
            let similar_patterns: Vec<_> = history
                .iter()
                .filter(|stats| stats.pattern.pattern_type() == pattern_type)
                .collect();

            if !similar_patterns.is_empty() {
                // Find the index with best average performance
                let mut index_performance: HashMap<IndexType, Vec<f64>> = HashMap::new();

                for stats in similar_patterns {
                    let performance = 1.0 / (stats.execution_time.as_secs_f64() + 0.001);
                    index_performance
                        .entry(stats.index_used)
                        .or_default()
                        .push(performance);
                }

                if let Some((historical_best_index, performance_scores)) =
                    index_performance.iter().max_by(|(_, a), (_, b)| {
                        let avg_a = a.iter().sum::<f64>() / a.len() as f64;
                        let avg_b = b.iter().sum::<f64>() / b.len() as f64;
                        avg_a
                            .partial_cmp(&avg_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    if performance_scores.len() >= 3 {
                        // Use historical data if we have enough samples
                        best_index = *historical_best_index;
                        best_cost = self
                            .cost_model
                            .estimate_cost(pattern, best_index, store_stats);
                        reasoning = format!(
                            "Historical data shows {:?} performs best for {:?} (avg performance: {:.3})",
                            best_index,
                            pattern_type,
                            performance_scores.iter().sum::<f64>() / performance_scores.len() as f64
                        );
                    }
                }
            }
        }

        let confidence = if best_index == optimal_index {
            0.9
        } else {
            0.7
        };

        let recommendation = OptimizationRecommendation {
            pattern: pattern.clone(),
            recommended_index: best_index,
            estimated_cost: best_cost,
            confidence,
            reasoning,
        };

        // Cache the recommendation
        if let Ok(mut cache) = self.pattern_cache.write() {
            cache.insert(pattern.clone(), recommendation.clone());
        }

        Ok(recommendation)
    }

    /// Get query statistics summary
    pub fn get_statistics_summary(&self) -> Result<QueryStatisticsSummary> {
        let history = self
            .stats_history
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on stats history"))?;

        if history.is_empty() {
            return Ok(QueryStatisticsSummary::default());
        }

        let total_queries = history.len();
        let avg_execution_time = history
            .iter()
            .map(|s| s.execution_time.as_secs_f64())
            .sum::<f64>()
            / total_queries as f64;

        let avg_result_count =
            history.iter().map(|s| s.result_count as f64).sum::<f64>() / total_queries as f64;

        // Index usage statistics
        let mut index_usage: HashMap<IndexType, u64> = HashMap::new();
        for stats in history.iter() {
            *index_usage.entry(stats.index_used).or_insert(0) += 1;
        }

        // Pattern type statistics
        let mut pattern_type_usage: HashMap<PatternType, u64> = HashMap::new();
        for stats in history.iter() {
            *pattern_type_usage
                .entry(stats.pattern.pattern_type())
                .or_insert(0) += 1;
        }

        Ok(QueryStatisticsSummary {
            total_queries: total_queries as u64,
            avg_execution_time_ms: (avg_execution_time * 1000.0) as u64,
            avg_result_count: avg_result_count as u64,
            index_usage,
            pattern_type_usage,
        })
    }

    /// Clear statistics history
    pub fn clear_history(&self) -> Result<()> {
        let mut history = self
            .stats_history
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on stats history"))?;
        history.clear();

        let mut cache = self
            .pattern_cache
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on pattern cache"))?;
        cache.clear();

        Ok(())
    }

    /// Update cost model parameters based on observed performance
    pub fn adapt_cost_model(&mut self) -> Result<()> {
        let history = self
            .stats_history
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on stats history"))?;

        if history.len() < 100 {
            return Ok(()); // Need enough data for adaptation
        }

        // Simple adaptation: adjust factors based on observed vs predicted performance
        let mut total_error = 0.0;
        let mut samples = 0;

        for stats in history.iter().rev().take(100) {
            let predicted_cost = self.cost_model.estimate_cost(
                &stats.pattern,
                stats.index_used,
                &TripleStoreStats::default(), // Would need actual stats here
            );
            let actual_cost = stats.execution_time.as_secs_f64() * 1000.0; // Convert to cost units

            let error = (predicted_cost - actual_cost).abs();
            total_error += error;
            samples += 1;
        }

        let avg_error = total_error / samples as f64;

        // Adapt model if error is high
        if avg_error > 10.0 {
            self.cost_model.index_cost_factor *= 0.95;
            self.cost_model.result_size_factor *= 1.05;
        }

        Ok(())
    }

    /// Get index creation recommendations based on query patterns
    pub fn get_index_recommendations(&self) -> Result<Vec<IndexRecommendation>> {
        let manager = self.adaptive_index_manager.read().map_err(|_| {
            anyhow::anyhow!("Failed to acquire read lock on adaptive index manager")
        })?;

        Ok(manager.get_index_recommendations())
    }

    /// Get pattern analysis summary for understanding query behavior
    pub fn get_pattern_analysis(&self) -> Result<PatternAnalysisSummary> {
        let manager = self.adaptive_index_manager.read().map_err(|_| {
            anyhow::anyhow!("Failed to acquire read lock on adaptive index manager")
        })?;

        Ok(manager.get_pattern_analysis())
    }

    /// Check if any new indices should be created based on current patterns
    pub fn should_create_indices(&self) -> Result<bool> {
        let recommendations = self.get_index_recommendations()?;
        Ok(!recommendations.is_empty())
    }

    /// Get the most beneficial index to create next
    pub fn get_next_index_to_create(&self) -> Result<Option<IndexRecommendation>> {
        let recommendations = self.get_index_recommendations()?;
        Ok(recommendations.into_iter().next())
    }

    /// Reset adaptive indexing data (useful for testing or fresh starts)
    pub fn reset_adaptive_indexing(&self) -> Result<()> {
        let mut manager = self.adaptive_index_manager.write().map_err(|_| {
            anyhow::anyhow!("Failed to acquire write lock on adaptive index manager")
        })?;

        *manager = AdaptiveIndexManager::new();
        Ok(())
    }

    /// Get comprehensive optimization report including adaptive indexing insights
    pub fn generate_optimization_report(&self) -> Result<OptimizationReport> {
        let stats_summary = self.get_statistics_summary()?;
        let pattern_analysis = self.get_pattern_analysis()?;
        let index_recommendations = self.get_index_recommendations()?;

        Ok(OptimizationReport {
            query_statistics: stats_summary,
            pattern_analysis,
            index_recommendations,
            report_timestamp: chrono::Utc::now(),
        })
    }
}

/// Comprehensive optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub query_statistics: QueryStatisticsSummary,
    pub pattern_analysis: PatternAnalysisSummary,
    pub index_recommendations: Vec<IndexRecommendation>,
    pub report_timestamp: DateTime<Utc>,
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of query execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatisticsSummary {
    pub total_queries: u64,
    pub avg_execution_time_ms: u64,
    pub avg_result_count: u64,
    pub index_usage: HashMap<IndexType, u64>,
    pub pattern_type_usage: HashMap<PatternType, u64>,
}

impl Default for QueryStatisticsSummary {
    fn default() -> Self {
        Self {
            total_queries: 0,
            avg_execution_time_ms: 0,
            avg_result_count: 0,
            index_usage: HashMap::new(),
            pattern_type_usage: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_pattern_selectivity() {
        let fully_bound = QueryPattern::new(Some(1), Some(2), Some(3));
        assert_eq!(fully_bound.selectivity(), 1.0);

        let two_bound = QueryPattern::new(Some(1), Some(2), None);
        assert_eq!(two_bound.selectivity(), 0.7);

        let one_bound = QueryPattern::new(Some(1), None, None);
        assert_eq!(one_bound.selectivity(), 0.3);

        let unbound = QueryPattern::new(None, None, None);
        assert_eq!(unbound.selectivity(), 0.0);
    }

    #[test]
    fn test_pattern_type_detection() {
        let pattern = QueryPattern::new(Some(1), Some(2), None);
        assert_eq!(pattern.pattern_type(), PatternType::SubjectPredicate);

        let pattern = QueryPattern::new(None, Some(2), Some(3));
        assert_eq!(pattern.pattern_type(), PatternType::PredicateObject);

        let pattern = QueryPattern::new(Some(1), None, Some(3));
        assert_eq!(pattern.pattern_type(), PatternType::SubjectObject);
    }

    #[test]
    fn test_optimal_index_selection() {
        assert_eq!(
            IndexType::optimal_for_pattern(PatternType::SubjectPredicate),
            IndexType::SPO
        );
        assert_eq!(
            IndexType::optimal_for_pattern(PatternType::PredicateObject),
            IndexType::POS
        );
        assert_eq!(
            IndexType::optimal_for_pattern(PatternType::SubjectObject),
            IndexType::SOP
        );
    }

    #[test]
    fn test_index_efficiency() {
        let efficiency = IndexType::SPO.efficiency_for_pattern(PatternType::SubjectPredicate);
        assert_eq!(efficiency, 1.0);

        let efficiency = IndexType::POS.efficiency_for_pattern(PatternType::SubjectPredicate);
        assert!(efficiency < 0.5);
    }

    #[test]
    fn test_cost_estimation() {
        let cost_model = QueryCostModel::default();
        let pattern = QueryPattern::new(Some(1), Some(2), None);
        let stats = TripleStoreStats {
            total_triples: 10000,
            total_quads: 10000,
            named_graphs: 5,
            active_transactions: 0,
            completed_transactions: 100,
            query_count: 50,
            insert_count: 5000,
            delete_count: 500,
            index_hits: HashMap::new(),
            avg_query_time_ms: 10.0,
        };

        let cost = cost_model.estimate_cost(&pattern, IndexType::SPO, &stats);
        assert!(cost > 0.0);
        assert!(cost < 10000.0);
    }

    #[test]
    fn test_query_optimizer() {
        let optimizer = QueryOptimizer::new();
        let pattern = QueryPattern::new(Some(1), Some(2), None);
        let stats = TripleStoreStats {
            total_triples: 10000,
            total_quads: 10000,
            named_graphs: 5,
            active_transactions: 0,
            completed_transactions: 100,
            query_count: 50,
            insert_count: 5000,
            delete_count: 500,
            index_hits: HashMap::new(),
            avg_query_time_ms: 10.0,
        };

        let recommendation = optimizer.recommend_optimization(&pattern, &stats).unwrap();
        assert_eq!(recommendation.recommended_index, IndexType::SPO);
        assert!(recommendation.confidence > 0.0);
        assert!(recommendation.estimated_cost > 0.0);
    }

    #[test]
    fn test_statistics_recording() {
        let optimizer = QueryOptimizer::new();
        let pattern = QueryPattern::new(Some(1), Some(2), None);

        let stats = QueryStats {
            pattern,
            execution_time: Duration::from_millis(50),
            result_count: 100,
            index_used: IndexType::SPO,
            cost: 25.0,
            timestamp: Utc::now(),
        };

        optimizer.record_execution(stats).unwrap();

        let summary = optimizer.get_statistics_summary().unwrap();
        assert_eq!(summary.total_queries, 1);
        assert_eq!(summary.avg_execution_time_ms, 50);
        assert_eq!(summary.avg_result_count, 100);
    }
}
