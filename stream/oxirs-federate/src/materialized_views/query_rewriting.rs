//! Query Rewriting with Materialized Views
//!
//! This module provides query rewriting capabilities to utilize materialized views
//! for improved query performance in federated scenarios.

use anyhow::Result;
use std::time::Instant;
use tracing::{debug, info, warn};

use super::types::*;
use crate::planner::planning::types::{QueryInfo, TriplePattern};
use crate::service_registry::ServiceRegistry;

/// Query rewriter that utilizes materialized views
#[derive(Debug)]
pub struct QueryRewriter {
    config: RewritingConfig,
    containment_checker: ContainmentChecker,
    cost_estimator: RewritingCostEstimator,
    view_matcher: ViewMatcher,
}

/// Configuration for query rewriting
#[derive(Debug, Clone)]
pub struct RewritingConfig {
    /// Enable aggressive rewriting optimizations
    pub enable_aggressive_rewriting: bool,
    /// Maximum views to consider per query
    pub max_views_per_query: usize,
    /// Minimum cost savings threshold to trigger rewriting
    pub min_cost_savings_threshold: f64,
    /// Enable partial view matching
    pub enable_partial_matching: bool,
    /// Enable view composition (using multiple views)
    pub enable_view_composition: bool,
    /// Timeout for rewriting analysis
    pub rewriting_timeout_ms: u64,
}

impl Default for RewritingConfig {
    fn default() -> Self {
        Self {
            enable_aggressive_rewriting: true,
            max_views_per_query: 10,
            min_cost_savings_threshold: 0.2, // 20% cost reduction
            enable_partial_matching: true,
            enable_view_composition: true,
            rewriting_timeout_ms: 500,
        }
    }
}

/// Result of query rewriting analysis
#[derive(Debug, Clone)]
pub struct RewritingResult {
    /// Original query info
    pub original_query: QueryInfo,
    /// Rewritten query using views
    pub rewritten_query: Option<RewrittenQuery>,
    /// Views used in the rewriting
    pub views_used: Vec<ViewUsage>,
    /// Estimated cost reduction
    pub cost_reduction: f64,
    /// Confidence in the rewriting
    pub confidence: f64,
    /// Rewriting strategy applied
    pub strategy: RewritingStrategy,
    /// Analysis time taken
    pub analysis_time: std::time::Duration,
}

/// Rewritten query structure
#[derive(Debug, Clone)]
pub struct RewrittenQuery {
    /// New query patterns after rewriting
    pub patterns: Vec<TriplePattern>,
    /// Filters that still need to be applied
    pub remaining_filters: Vec<crate::planner::planning::FilterExpression>,
    /// View access patterns
    pub view_accesses: Vec<ViewAccess>,
    /// Additional joins required
    pub additional_joins: Vec<JoinRequirement>,
}

/// How a view is used in query rewriting
#[derive(Debug, Clone)]
pub struct ViewUsage {
    /// View ID
    pub view_id: String,
    /// Type of usage
    pub usage_type: ViewUsageType,
    /// Patterns covered by this view
    pub covered_patterns: Vec<TriplePattern>,
    /// Estimated cost savings
    pub cost_savings: f64,
    /// Coverage percentage (0.0-1.0)
    pub coverage_percentage: f64,
}

/// Types of view usage
#[derive(Debug, Clone, PartialEq)]
pub enum ViewUsageType {
    /// Complete replacement of query patterns
    CompleteReplacement,
    /// Partial coverage requiring additional processing
    PartialCoverage,
    /// Used as one component in view composition
    CompositionComponent,
    /// Provides optimization hints
    OptimizationHint,
}

/// View access pattern in rewritten query
#[derive(Debug, Clone)]
pub struct ViewAccess {
    /// View being accessed
    pub view_id: String,
    /// Selection criteria for the view
    pub selection_criteria: Vec<String>,
    /// Projection columns needed
    pub projection: Vec<String>,
    /// Estimated result size from view
    pub estimated_result_size: u64,
}

/// Join requirement for view composition
#[derive(Debug, Clone)]
pub struct JoinRequirement {
    /// Left side of join (view or table)
    pub left_side: JoinSide,
    /// Right side of join
    pub right_side: JoinSide,
    /// Join variables
    pub join_variables: Vec<String>,
    /// Join type
    pub join_type: JoinType,
}

/// Side of a join operation
#[derive(Debug, Clone)]
pub enum JoinSide {
    View(String),
    OriginalPattern(TriplePattern),
    IntermediateResult(String),
}

/// Types of joins in view composition
#[derive(Debug, Clone)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    FullOuter,
}

/// Rewriting strategies
#[derive(Debug, Clone, PartialEq)]
pub enum RewritingStrategy {
    /// No rewriting applied
    None,
    /// Single view covers entire query
    SingleViewComplete,
    /// Single view covers part of query
    SingleViewPartial,
    /// Multiple views composed together
    MultiViewComposition,
    /// Hybrid approach with views and original patterns
    HybridApproach,
}

impl QueryRewriter {
    /// Create a new query rewriter
    pub fn new() -> Self {
        Self::with_config(RewritingConfig::default())
    }

    /// Create a new query rewriter with custom configuration
    pub fn with_config(config: RewritingConfig) -> Self {
        Self {
            config,
            containment_checker: ContainmentChecker::new(),
            cost_estimator: RewritingCostEstimator::new(),
            view_matcher: ViewMatcher::new(),
        }
    }

    /// Attempt to rewrite a query using available materialized views
    pub async fn rewrite_query(
        &self,
        query_info: &QueryInfo,
        available_views: &[MaterializedView],
        registry: &ServiceRegistry,
    ) -> Result<RewritingResult> {
        let start_time = Instant::now();

        debug!(
            "Attempting to rewrite query with {} patterns using {} views",
            query_info.patterns.len(),
            available_views.len()
        );

        // Find candidate views that might be useful
        let candidates = self
            .find_candidate_views(query_info, available_views)
            .await?;

        if candidates.is_empty() {
            return Ok(RewritingResult {
                original_query: query_info.clone(),
                rewritten_query: None,
                views_used: vec![],
                cost_reduction: 0.0,
                confidence: 0.0,
                strategy: RewritingStrategy::None,
                analysis_time: start_time.elapsed(),
            });
        }

        // Try different rewriting strategies
        let strategies = vec![
            self.try_single_view_complete_rewriting(query_info, &candidates, registry)
                .await?,
            self.try_single_view_partial_rewriting(query_info, &candidates, registry)
                .await?,
            self.try_multi_view_composition(query_info, &candidates, registry)
                .await?,
            self.try_hybrid_rewriting(query_info, &candidates, registry)
                .await?,
        ];

        // Select the best strategy
        let best_strategy = strategies
            .into_iter()
            .max_by(|a, b| {
                a.cost_reduction
                    .partial_cmp(&b.cost_reduction)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| RewritingResult {
                original_query: query_info.clone(),
                rewritten_query: None,
                views_used: vec![],
                cost_reduction: 0.0,
                confidence: 0.0,
                strategy: RewritingStrategy::None,
                analysis_time: start_time.elapsed(),
            });

        // Only use rewriting if it meets the cost savings threshold
        if best_strategy.cost_reduction >= self.config.min_cost_savings_threshold {
            info!(
                "Query rewriting successful: {:.1}% cost reduction using strategy {:?}",
                best_strategy.cost_reduction * 100.0,
                best_strategy.strategy
            );
            Ok(best_strategy)
        } else {
            warn!(
                "Query rewriting cost savings ({:.1}%) below threshold ({:.1}%)",
                best_strategy.cost_reduction * 100.0,
                self.config.min_cost_savings_threshold * 100.0
            );
            Ok(RewritingResult {
                original_query: query_info.clone(),
                rewritten_query: None,
                views_used: vec![],
                cost_reduction: 0.0,
                confidence: 0.0,
                strategy: RewritingStrategy::None,
                analysis_time: start_time.elapsed(),
            })
        }
    }

    /// Find views that might be candidates for rewriting
    async fn find_candidate_views<'a>(
        &self,
        query_info: &QueryInfo,
        available_views: &'a [MaterializedView],
    ) -> Result<Vec<&'a MaterializedView>> {
        let mut candidates = Vec::new();

        for view in available_views {
            // Skip stale views unless configured otherwise
            if view.is_stale && !self.config.enable_aggressive_rewriting {
                continue;
            }

            // Check if view patterns might overlap with query patterns
            if self
                .view_matcher
                .has_potential_overlap(&view.definition, query_info)
                .await?
            {
                candidates.push(view);
            }
        }

        // Limit the number of candidates
        candidates.truncate(self.config.max_views_per_query);

        debug!("Found {} candidate views for rewriting", candidates.len());
        Ok(candidates)
    }

    /// Try complete rewriting with a single view
    async fn try_single_view_complete_rewriting(
        &self,
        query_info: &QueryInfo,
        candidates: &[&MaterializedView],
        registry: &ServiceRegistry,
    ) -> Result<RewritingResult> {
        for view in candidates {
            // Check if view completely contains the query
            if self
                .containment_checker
                .is_query_contained(&view.definition, query_info)
                .await?
            {
                let cost_reduction = self
                    .cost_estimator
                    .estimate_cost_reduction(query_info, &[view], registry)
                    .await?;

                let rewritten_query = RewrittenQuery {
                    patterns: vec![],          // No original patterns needed
                    remaining_filters: vec![], // All handled by view
                    view_accesses: vec![ViewAccess {
                        view_id: view.id.clone(),
                        selection_criteria: self
                            .extract_selection_criteria(query_info, view)
                            .await?,
                        projection: self.extract_projection(query_info).await?,
                        estimated_result_size: self
                            .estimate_view_result_size(view, query_info)
                            .await?,
                    }],
                    additional_joins: vec![],
                };

                return Ok(RewritingResult {
                    original_query: query_info.clone(),
                    rewritten_query: Some(rewritten_query),
                    views_used: vec![ViewUsage {
                        view_id: view.id.clone(),
                        usage_type: ViewUsageType::CompleteReplacement,
                        covered_patterns: query_info.patterns.clone(),
                        cost_savings: cost_reduction,
                        coverage_percentage: 1.0,
                    }],
                    cost_reduction,
                    confidence: 0.9, // High confidence for complete containment
                    strategy: RewritingStrategy::SingleViewComplete,
                    analysis_time: std::time::Duration::from_millis(0), // Will be set by caller
                });
            }
        }

        Ok(RewritingResult {
            original_query: query_info.clone(),
            rewritten_query: None,
            views_used: vec![],
            cost_reduction: 0.0,
            confidence: 0.0,
            strategy: RewritingStrategy::None,
            analysis_time: std::time::Duration::from_millis(0),
        })
    }

    /// Try partial rewriting with a single view
    async fn try_single_view_partial_rewriting(
        &self,
        query_info: &QueryInfo,
        candidates: &[&MaterializedView],
        registry: &ServiceRegistry,
    ) -> Result<RewritingResult> {
        if !self.config.enable_partial_matching {
            return Ok(RewritingResult {
                original_query: query_info.clone(),
                rewritten_query: None,
                views_used: vec![],
                cost_reduction: 0.0,
                confidence: 0.0,
                strategy: RewritingStrategy::None,
                analysis_time: std::time::Duration::from_millis(0),
            });
        }

        let mut best_result = None;
        let mut best_coverage = 0.0;

        for view in candidates {
            // Find patterns that can be covered by this view
            let coverage = self
                .view_matcher
                .calculate_pattern_coverage(&view.definition, &query_info.patterns)
                .await?;

            if coverage.coverage_percentage > best_coverage && coverage.coverage_percentage > 0.3 {
                // Need at least 30% coverage to be worthwhile

                let cost_reduction = self
                    .cost_estimator
                    .estimate_partial_cost_reduction(query_info, view, &coverage, registry)
                    .await?;

                if cost_reduction > 0.1 {
                    // At least 10% improvement
                    let rewritten_query = RewrittenQuery {
                        patterns: coverage.uncovered_patterns.clone(),
                        remaining_filters: query_info.filters.clone(), // May need filtering
                        view_accesses: vec![ViewAccess {
                            view_id: view.id.clone(),
                            selection_criteria: self
                                .extract_selection_criteria(query_info, view)
                                .await?,
                            projection: self.extract_projection(query_info).await?,
                            estimated_result_size: self
                                .estimate_view_result_size(view, query_info)
                                .await?,
                        }],
                        additional_joins: self
                            .plan_view_integration_joins(query_info, view, &coverage)
                            .await?,
                    };

                    best_result = Some(RewritingResult {
                        original_query: query_info.clone(),
                        rewritten_query: Some(rewritten_query),
                        views_used: vec![ViewUsage {
                            view_id: view.id.clone(),
                            usage_type: ViewUsageType::PartialCoverage,
                            covered_patterns: coverage.covered_patterns.clone(),
                            cost_savings: cost_reduction,
                            coverage_percentage: coverage.coverage_percentage,
                        }],
                        cost_reduction,
                        confidence: coverage.coverage_percentage * 0.8, // Lower confidence for partial
                        strategy: RewritingStrategy::SingleViewPartial,
                        analysis_time: std::time::Duration::from_millis(0),
                    });

                    best_coverage = coverage.coverage_percentage;
                }
            }
        }

        Ok(best_result.unwrap_or_else(|| RewritingResult {
            original_query: query_info.clone(),
            rewritten_query: None,
            views_used: vec![],
            cost_reduction: 0.0,
            confidence: 0.0,
            strategy: RewritingStrategy::None,
            analysis_time: std::time::Duration::from_millis(0),
        }))
    }

    /// Try rewriting using multiple views in composition
    async fn try_multi_view_composition(
        &self,
        query_info: &QueryInfo,
        candidates: &[&MaterializedView],
        registry: &ServiceRegistry,
    ) -> Result<RewritingResult> {
        if !self.config.enable_view_composition || candidates.len() < 2 {
            return Ok(RewritingResult {
                original_query: query_info.clone(),
                rewritten_query: None,
                views_used: vec![],
                cost_reduction: 0.0,
                confidence: 0.0,
                strategy: RewritingStrategy::None,
                analysis_time: std::time::Duration::from_millis(0),
            });
        }

        // Try combinations of views
        let mut best_composition = None;
        let mut best_cost_reduction = 0.0;

        // Try pairs of views first (most common case)
        for i in 0..candidates.len() {
            for j in i + 1..candidates.len() {
                let view_pair = &[candidates[i], candidates[j]];

                if let Some(composition) = self
                    .try_view_pair_composition(query_info, view_pair, registry)
                    .await?
                {
                    if composition.cost_reduction > best_cost_reduction {
                        best_cost_reduction = composition.cost_reduction;
                        best_composition = Some(composition);
                    }
                }
            }
        }

        // Could extend to try triplets, etc., but keep it simple for now

        Ok(best_composition.unwrap_or_else(|| RewritingResult {
            original_query: query_info.clone(),
            rewritten_query: None,
            views_used: vec![],
            cost_reduction: 0.0,
            confidence: 0.0,
            strategy: RewritingStrategy::None,
            analysis_time: std::time::Duration::from_millis(0),
        }))
    }

    /// Try hybrid rewriting (views + original patterns)
    async fn try_hybrid_rewriting(
        &self,
        query_info: &QueryInfo,
        candidates: &[&MaterializedView],
        registry: &ServiceRegistry,
    ) -> Result<RewritingResult> {
        // Select the best single view and combine with original patterns
        let mut best_hybrid = None;
        let mut best_cost_reduction = 0.0;

        for view in candidates {
            let coverage = self
                .view_matcher
                .calculate_pattern_coverage(&view.definition, &query_info.patterns)
                .await?;

            if coverage.coverage_percentage > 0.2 {
                // At least 20% coverage
                let cost_reduction = self
                    .cost_estimator
                    .estimate_hybrid_cost_reduction(query_info, view, &coverage, registry)
                    .await?;

                if cost_reduction > best_cost_reduction {
                    let rewritten_query = RewrittenQuery {
                        patterns: coverage.uncovered_patterns.clone(),
                        remaining_filters: query_info.filters.clone(),
                        view_accesses: vec![ViewAccess {
                            view_id: view.id.clone(),
                            selection_criteria: self
                                .extract_selection_criteria(query_info, view)
                                .await?,
                            projection: self.extract_projection(query_info).await?,
                            estimated_result_size: self
                                .estimate_view_result_size(view, query_info)
                                .await?,
                        }],
                        additional_joins: self
                            .plan_hybrid_joins(query_info, view, &coverage)
                            .await?,
                    };

                    best_hybrid = Some(RewritingResult {
                        original_query: query_info.clone(),
                        rewritten_query: Some(rewritten_query),
                        views_used: vec![ViewUsage {
                            view_id: view.id.clone(),
                            usage_type: ViewUsageType::CompositionComponent,
                            covered_patterns: coverage.covered_patterns.clone(),
                            cost_savings: cost_reduction,
                            coverage_percentage: coverage.coverage_percentage,
                        }],
                        cost_reduction,
                        confidence: 0.6, // Medium confidence for hybrid
                        strategy: RewritingStrategy::HybridApproach,
                        analysis_time: std::time::Duration::from_millis(0),
                    });

                    best_cost_reduction = cost_reduction;
                }
            }
        }

        Ok(best_hybrid.unwrap_or_else(|| RewritingResult {
            original_query: query_info.clone(),
            rewritten_query: None,
            views_used: vec![],
            cost_reduction: 0.0,
            confidence: 0.0,
            strategy: RewritingStrategy::None,
            analysis_time: std::time::Duration::from_millis(0),
        }))
    }

    // Helper methods

    async fn try_view_pair_composition(
        &self,
        query_info: &QueryInfo,
        views: &[&MaterializedView],
        registry: &ServiceRegistry,
    ) -> Result<Option<RewritingResult>> {
        // Simplified composition logic
        // In practice, this would need sophisticated join planning

        let coverage1 = self
            .view_matcher
            .calculate_pattern_coverage(&views[0].definition, &query_info.patterns)
            .await?;
        let coverage2 = self
            .view_matcher
            .calculate_pattern_coverage(&views[1].definition, &query_info.patterns)
            .await?;

        let combined_coverage = coverage1.coverage_percentage + coverage2.coverage_percentage;

        if combined_coverage > 0.8 {
            // Good combined coverage
            let cost_reduction = self
                .cost_estimator
                .estimate_composition_cost_reduction(query_info, views, registry)
                .await?;

            if cost_reduction > 0.2 {
                let rewritten_query = RewrittenQuery {
                    patterns: vec![], // Assuming views cover everything
                    remaining_filters: query_info.filters.clone(),
                    view_accesses: vec![
                        ViewAccess {
                            view_id: views[0].id.clone(),
                            selection_criteria: vec![],
                            projection: vec![],
                            estimated_result_size: 1000, // Placeholder
                        },
                        ViewAccess {
                            view_id: views[1].id.clone(),
                            selection_criteria: vec![],
                            projection: vec![],
                            estimated_result_size: 1000, // Placeholder
                        },
                    ],
                    additional_joins: vec![JoinRequirement {
                        left_side: JoinSide::View(views[0].id.clone()),
                        right_side: JoinSide::View(views[1].id.clone()),
                        join_variables: vec!["id".to_string()], // Simplified
                        join_type: JoinType::Inner,
                    }],
                };

                return Ok(Some(RewritingResult {
                    original_query: query_info.clone(),
                    rewritten_query: Some(rewritten_query),
                    views_used: vec![
                        ViewUsage {
                            view_id: views[0].id.clone(),
                            usage_type: ViewUsageType::CompositionComponent,
                            covered_patterns: coverage1.covered_patterns,
                            cost_savings: cost_reduction / 2.0,
                            coverage_percentage: coverage1.coverage_percentage,
                        },
                        ViewUsage {
                            view_id: views[1].id.clone(),
                            usage_type: ViewUsageType::CompositionComponent,
                            covered_patterns: coverage2.covered_patterns,
                            cost_savings: cost_reduction / 2.0,
                            coverage_percentage: coverage2.coverage_percentage,
                        },
                    ],
                    cost_reduction,
                    confidence: 0.7,
                    strategy: RewritingStrategy::MultiViewComposition,
                    analysis_time: std::time::Duration::from_millis(0),
                }));
            }
        }

        Ok(None)
    }

    async fn extract_selection_criteria(
        &self,
        _query_info: &QueryInfo,
        _view: &MaterializedView,
    ) -> Result<Vec<String>> {
        // Placeholder implementation
        Ok(vec![])
    }

    async fn extract_projection(&self, query_info: &QueryInfo) -> Result<Vec<String>> {
        // Extract variable names from query
        Ok(query_info.variables.iter().cloned().collect())
    }

    async fn estimate_view_result_size(
        &self,
        _view: &MaterializedView,
        _query_info: &QueryInfo,
    ) -> Result<u64> {
        // Placeholder implementation
        Ok(1000)
    }

    async fn plan_view_integration_joins(
        &self,
        _query_info: &QueryInfo,
        _view: &MaterializedView,
        _coverage: &PatternCoverage,
    ) -> Result<Vec<JoinRequirement>> {
        // Placeholder implementation
        Ok(vec![])
    }

    async fn plan_hybrid_joins(
        &self,
        _query_info: &QueryInfo,
        _view: &MaterializedView,
        _coverage: &PatternCoverage,
    ) -> Result<Vec<JoinRequirement>> {
        // Placeholder implementation
        Ok(vec![])
    }
}

impl Default for QueryRewriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper struct for checking query containment in views
#[derive(Debug)]
struct ContainmentChecker;

impl ContainmentChecker {
    fn new() -> Self {
        Self
    }

    async fn is_query_contained(
        &self,
        _view_def: &ViewDefinition,
        _query: &QueryInfo,
    ) -> Result<bool> {
        // Simplified containment check
        // In practice, this requires sophisticated logic
        Ok(false)
    }
}

/// Helper struct for estimating rewriting costs
#[derive(Debug)]
struct RewritingCostEstimator;

impl RewritingCostEstimator {
    fn new() -> Self {
        Self
    }

    async fn estimate_cost_reduction(
        &self,
        _query: &QueryInfo,
        _views: &[&MaterializedView],
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        // Placeholder cost estimation
        Ok(0.5) // 50% cost reduction
    }

    async fn estimate_partial_cost_reduction(
        &self,
        _query: &QueryInfo,
        _view: &MaterializedView,
        coverage: &PatternCoverage,
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        // Cost reduction proportional to coverage
        Ok(coverage.coverage_percentage * 0.6)
    }

    async fn estimate_composition_cost_reduction(
        &self,
        _query: &QueryInfo,
        _views: &[&MaterializedView],
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        // Placeholder for composition cost estimation
        Ok(0.4)
    }

    async fn estimate_hybrid_cost_reduction(
        &self,
        _query: &QueryInfo,
        _view: &MaterializedView,
        coverage: &PatternCoverage,
        _registry: &ServiceRegistry,
    ) -> Result<f64> {
        // Hybrid typically has lower savings due to complexity
        Ok(coverage.coverage_percentage * 0.3)
    }
}

/// Helper struct for matching views with query patterns
#[derive(Debug)]
struct ViewMatcher;

impl ViewMatcher {
    fn new() -> Self {
        Self
    }

    async fn has_potential_overlap(
        &self,
        _view_def: &ViewDefinition,
        _query: &QueryInfo,
    ) -> Result<bool> {
        // Simplified overlap detection
        Ok(true) // Assume potential overlap for simplicity
    }

    async fn calculate_pattern_coverage(
        &self,
        _view_def: &ViewDefinition,
        patterns: &[TriplePattern],
    ) -> Result<PatternCoverage> {
        // Simplified coverage calculation
        let coverage_percentage = 0.7; // 70% coverage
        let split_point = (patterns.len() as f64 * coverage_percentage) as usize;

        Ok(PatternCoverage {
            covered_patterns: patterns[..split_point].to_vec(),
            uncovered_patterns: patterns[split_point..].to_vec(),
            coverage_percentage,
        })
    }
}

/// Pattern coverage result
#[derive(Debug, Clone)]
struct PatternCoverage {
    covered_patterns: Vec<TriplePattern>,
    uncovered_patterns: Vec<TriplePattern>,
    coverage_percentage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewriting_config_default() {
        let config = RewritingConfig::default();
        assert!(config.enable_aggressive_rewriting);
        assert_eq!(config.max_views_per_query, 10);
        assert_eq!(config.min_cost_savings_threshold, 0.2);
    }

    #[tokio::test]
    async fn test_query_rewriter_creation() {
        let _rewriter = QueryRewriter::new();
        // Should create without errors
    }

    #[test]
    fn test_rewriting_strategy_types() {
        assert_eq!(RewritingStrategy::None, RewritingStrategy::None);
        assert_ne!(
            RewritingStrategy::SingleViewComplete,
            RewritingStrategy::MultiViewComposition
        );
    }

    #[test]
    fn test_view_usage_types() {
        let usage = ViewUsage {
            view_id: "test_view".to_string(),
            usage_type: ViewUsageType::CompleteReplacement,
            covered_patterns: vec![],
            cost_savings: 0.5,
            coverage_percentage: 1.0,
        };

        assert_eq!(usage.usage_type, ViewUsageType::CompleteReplacement);
        assert_eq!(usage.coverage_percentage, 1.0);
    }
}
