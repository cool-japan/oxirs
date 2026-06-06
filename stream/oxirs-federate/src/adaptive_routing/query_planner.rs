//! Adaptive query planner for federated SPARQL routing.
//!
//! Builds [`RoutingPlan`] objects by greedily assigning each [`SubQuery`] to
//! the cheapest available endpoint whose availability score exceeds a minimum
//! threshold.

use std::collections::HashMap;

use super::{
    cost_model::{EndpointCostEstimator, QueryCostFactors},
    stats::EndpointStats,
    EndpointId, RoutingConfig,
};

// ---------------------------------------------------------------------------
// Triple pattern
// ---------------------------------------------------------------------------

/// A single triple pattern within a sub-query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriplePattern {
    /// Pattern constrained only on the subject.
    Subject(String),
    /// Pattern constrained only on the predicate.
    Predicate(String),
    /// Pattern constrained only on the object.
    Object(String),
    /// Fully-bound triple `(subject, predicate, object)`.
    Full(String, String, String),
}

// ---------------------------------------------------------------------------
// Sub-query
// ---------------------------------------------------------------------------

/// A single decomposed sub-query that can be sent to one endpoint.
#[derive(Debug, Clone)]
pub struct SubQuery {
    /// Unique identifier within the parent [`FederatedQuery`].
    pub id: usize,
    /// Triple patterns forming this sub-query's BGP.
    pub triple_patterns: Vec<TriplePattern>,
    /// Estimated result cardinality (used by the cost model).
    pub estimated_cardinality: u64,
}

impl SubQuery {
    /// Derive [`QueryCostFactors`] from this sub-query's metadata.
    pub fn cost_factors(&self) -> QueryCostFactors {
        QueryCostFactors {
            // Simple heuristic: fully-bound patterns have low selectivity
            selectivity: if self.triple_patterns.is_empty() {
                0.5
            } else {
                let full_count = self
                    .triple_patterns
                    .iter()
                    .filter(|p| matches!(p, TriplePattern::Full(_, _, _)))
                    .count();
                1.0 - (full_count as f64 / self.triple_patterns.len() as f64) * 0.5
            },
            triple_pattern_count: self.triple_patterns.len(),
            result_cardinality_hint: self.estimated_cardinality,
        }
    }
}

// ---------------------------------------------------------------------------
// Federated query
// ---------------------------------------------------------------------------

/// A federated query composed of multiple [`SubQuery`] fragments, each of
/// which can be dispatched to a potentially different SPARQL endpoint.
#[derive(Debug, Clone)]
pub struct FederatedQuery {
    /// Ordered list of decomposed sub-queries.
    pub sub_queries: Vec<SubQuery>,
    /// Endpoints the caller explicitly wants to consider (empty = all registered).
    pub optional_endpoints: Vec<EndpointId>,
}

// ---------------------------------------------------------------------------
// Routing plan
// ---------------------------------------------------------------------------

/// An assignment of sub-queries to endpoints, ready for parallel dispatch.
#[derive(Debug, Clone)]
pub struct RoutingPlan {
    /// `(endpoint_id, sub_query)` pairs in the order they were planned.
    pub assignments: Vec<(EndpointId, SubQuery)>,
    /// Fallback endpoint used when the primary assignment may be unavailable.
    pub fallback_endpoint: Option<EndpointId>,
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// Minimum availability score an endpoint must have to be selected as primary.
const MIN_AVAILABILITY_SCORE: f64 = 0.5;

/// Adaptive greedy query planner.
#[derive(Debug, Clone)]
pub struct AdaptivePlanner {
    config: RoutingConfig,
}

impl AdaptivePlanner {
    /// Create a new planner with the supplied routing configuration.
    pub fn new(config: RoutingConfig) -> Self {
        Self { config }
    }

    /// Build a greedy routing plan.
    ///
    /// For each sub-query the planner picks the cheapest endpoint whose
    /// availability score is above [`MIN_AVAILABILITY_SCORE`].  If no
    /// endpoint satisfies the threshold the first endpoint in the ranked list
    /// is used as a last resort (the caller may inspect the plan and decide to
    /// reject it).
    ///
    /// Returns an empty plan if there are no registered endpoints.
    pub fn plan(
        &self,
        query: &FederatedQuery,
        endpoint_stats: &HashMap<EndpointId, EndpointStats>,
        estimator: &EndpointCostEstimator,
    ) -> RoutingPlan {
        let candidate_endpoints = self.resolve_candidates(query, endpoint_stats);

        if candidate_endpoints.is_empty() {
            return RoutingPlan {
                assignments: Vec::new(),
                fallback_endpoint: None,
            };
        }

        let assignments = self.assign_sub_queries(
            &query.sub_queries,
            &candidate_endpoints,
            endpoint_stats,
            estimator,
            false,
        );

        RoutingPlan {
            assignments,
            fallback_endpoint: None,
        }
    }

    /// Build a routing plan that includes a fallback endpoint for each primary
    /// assignment.
    ///
    /// The fallback is the second-cheapest available endpoint.  When fewer
    /// than two endpoints are registered the `fallback_endpoint` field will be
    /// `None`.
    pub fn plan_with_fallback(
        &self,
        query: &FederatedQuery,
        endpoint_stats: &HashMap<EndpointId, EndpointStats>,
        estimator: &EndpointCostEstimator,
    ) -> RoutingPlan {
        let candidate_endpoints = self.resolve_candidates(query, endpoint_stats);

        if candidate_endpoints.is_empty() {
            return RoutingPlan {
                assignments: Vec::new(),
                fallback_endpoint: None,
            };
        }

        let assignments = self.assign_sub_queries(
            &query.sub_queries,
            &candidate_endpoints,
            endpoint_stats,
            estimator,
            false,
        );

        // Pick the overall cheapest endpoint as a global fallback.
        let fallback_endpoint = self.pick_fallback(&candidate_endpoints, endpoint_stats, estimator);

        RoutingPlan {
            assignments,
            fallback_endpoint,
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Resolve the set of candidate endpoint IDs from the query's optional
    /// endpoint list or the full stats registry, capped at
    /// `config.max_endpoints_per_query`.
    fn resolve_candidates(
        &self,
        query: &FederatedQuery,
        endpoint_stats: &HashMap<EndpointId, EndpointStats>,
    ) -> Vec<EndpointId> {
        let max = self.config.max_endpoints_per_query;
        let full: Vec<EndpointId> = if !query.optional_endpoints.is_empty() {
            query
                .optional_endpoints
                .iter()
                .filter(|ep| endpoint_stats.contains_key(*ep))
                .cloned()
                .collect()
        } else {
            endpoint_stats.keys().cloned().collect()
        };
        // Honour the per-query endpoint cap configured in RoutingConfig.
        if full.len() > max {
            full.into_iter().take(max).collect()
        } else {
            full
        }
    }

    /// Assign each sub-query to the best available endpoint.
    fn assign_sub_queries(
        &self,
        sub_queries: &[SubQuery],
        candidates: &[EndpointId],
        endpoint_stats: &HashMap<EndpointId, EndpointStats>,
        estimator: &EndpointCostEstimator,
        _with_fallback: bool,
    ) -> Vec<(EndpointId, SubQuery)> {
        let _ = estimator; // used via EndpointCostEstimator::rank_endpoints

        sub_queries
            .iter()
            .map(|sq| {
                let factors = sq.cost_factors();
                let ranked =
                    EndpointCostEstimator::rank_endpoints(candidates, &factors, endpoint_stats);

                // Prefer the first endpoint that passes the availability threshold.
                let chosen = ranked
                    .iter()
                    .find(|(ep, _cost)| {
                        endpoint_stats
                            .get(ep)
                            .map(|s| s.availability_score() > MIN_AVAILABILITY_SCORE)
                            .unwrap_or(false)
                    })
                    .or_else(|| ranked.first())
                    .map(|(ep, _)| ep.clone())
                    .unwrap_or_else(|| candidates[0].clone());

                (chosen, sq.clone())
            })
            .collect()
    }

    /// Pick the globally cheapest endpoint as a fallback (different from the
    /// primary where possible).
    fn pick_fallback(
        &self,
        candidates: &[EndpointId],
        endpoint_stats: &HashMap<EndpointId, EndpointStats>,
        _estimator: &EndpointCostEstimator,
    ) -> Option<EndpointId> {
        if candidates.len() < 2 {
            return candidates.first().cloned();
        }

        // Use generic cost factors for global fallback selection.
        let generic_factors = QueryCostFactors::default();
        let ranked =
            EndpointCostEstimator::rank_endpoints(candidates, &generic_factors, endpoint_stats);

        // Return the second-best candidate as fallback to avoid routing
        // everything to the single cheapest endpoint.
        ranked.get(1).map(|(ep, _)| ep.clone())
    }
}
