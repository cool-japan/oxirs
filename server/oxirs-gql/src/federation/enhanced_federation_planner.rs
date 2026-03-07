// Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Enhanced Federation Planner with cost-based optimization.
//!
//! This module provides `EnhancedFederationPlanner`, a cost-aware query planner
//! that routes GraphQL fields to the appropriate sources, batches requests to
//! the same source, and estimates query cost across multiple GraphQL sources.
//!
//! ## Features
//!
//! - **Field-level source routing**: Route individual fields to their owning source
//! - **Request batching**: Group fields going to the same source into a single request
//! - **Cost-based optimization**: Estimate and minimize query cost across sources
//! - **Parallel execution planning**: Identify independent sub-plans for parallelism
//! - **Source health weighting**: Prefer low-latency sources when possible

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

// ---------------------------------------------------------------------------
// Source descriptors
// ---------------------------------------------------------------------------

/// Statistics collected about a remote GraphQL source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStats {
    /// Estimated average response latency in milliseconds.
    pub avg_latency_ms: u64,
    /// Estimated cost per field resolved by this source (abstract unit).
    pub cost_per_field: f64,
    /// Whether this source supports batched requests.
    pub supports_batching: bool,
    /// Maximum fields per batch (0 = unlimited).
    pub max_batch_size: usize,
}

impl Default for SourceStats {
    fn default() -> Self {
        Self {
            avg_latency_ms: 50,
            cost_per_field: 1.0,
            supports_batching: true,
            max_batch_size: 0,
        }
    }
}

/// A registered GraphQL source (subgraph or remote schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationSource {
    /// Unique identifier for this source.
    pub id: String,
    /// The GraphQL endpoint URL.
    pub url: String,
    /// Types owned by this source.
    pub owned_types: Vec<String>,
    /// Fields owned by this source, keyed by `TypeName.fieldName`.
    pub owned_fields: Vec<String>,
    /// Performance and cost statistics.
    pub stats: SourceStats,
}

impl FederationSource {
    /// Create a new source with default stats.
    pub fn new(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            owned_types: Vec::new(),
            owned_fields: Vec::new(),
            stats: SourceStats::default(),
        }
    }

    /// Add an owned type.
    pub fn with_type(mut self, type_name: impl Into<String>) -> Self {
        self.owned_types.push(type_name.into());
        self
    }

    /// Add an owned field (format: `TypeName.fieldName`).
    pub fn with_field(mut self, field: impl Into<String>) -> Self {
        self.owned_fields.push(field.into());
        self
    }

    /// Set source statistics.
    pub fn with_stats(mut self, stats: SourceStats) -> Self {
        self.stats = stats;
        self
    }

    /// Returns `true` if this source owns the given type.
    pub fn owns_type(&self, type_name: &str) -> bool {
        self.owned_types.iter().any(|t| t == type_name)
    }

    /// Returns `true` if this source owns the given field (`TypeName.fieldName`).
    pub fn owns_field(&self, field: &str) -> bool {
        self.owned_fields.iter().any(|f| f == field) || {
            // Also match by type prefix
            let type_part = field.split('.').next().unwrap_or("");
            self.owns_type(type_part)
        }
    }
}

// ---------------------------------------------------------------------------
// Field routing
// ---------------------------------------------------------------------------

/// A request for a specific field routed to a specific source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FieldRequest {
    /// The type that owns the field.
    pub type_name: String,
    /// The field name.
    pub field_name: String,
    /// The source that should resolve this field.
    pub source_id: String,
}

impl FieldRequest {
    /// Create a new field request.
    pub fn new(
        type_name: impl Into<String>,
        field_name: impl Into<String>,
        source_id: impl Into<String>,
    ) -> Self {
        Self {
            type_name: type_name.into(),
            field_name: field_name.into(),
            source_id: source_id.into(),
        }
    }

    /// Returns the qualified field key (`TypeName.fieldName`).
    pub fn qualified_name(&self) -> String {
        format!("{}.{}", self.type_name, self.field_name)
    }
}

// ---------------------------------------------------------------------------
// Batched sub-plan
// ---------------------------------------------------------------------------

/// A set of fields batched for execution against a single source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedSubPlan {
    /// The source this batch targets.
    pub source_id: String,
    /// Fields included in this batch.
    pub fields: Vec<FieldRequest>,
    /// Estimated cost for this batch.
    pub estimated_cost: f64,
    /// Estimated latency for this batch (milliseconds).
    pub estimated_latency_ms: u64,
    /// Indices of other sub-plans that must complete before this one.
    pub depends_on: Vec<usize>,
}

impl BatchedSubPlan {
    /// Returns the number of fields in the batch.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

// ---------------------------------------------------------------------------
// Enhanced plan
// ---------------------------------------------------------------------------

/// A cost-optimized execution plan across multiple sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedFederationPlan {
    /// Ordered list of batched sub-plans.
    pub sub_plans: Vec<BatchedSubPlan>,
    /// Total estimated cost across all sub-plans.
    pub total_cost: f64,
    /// Estimated critical-path latency (milliseconds).
    pub critical_path_latency_ms: u64,
    /// Whether all sub-plans are independent (no dependencies).
    pub is_fully_parallel: bool,
    /// Sources that contributed at least one field to the plan.
    pub contributing_sources: Vec<String>,
}

impl EnhancedFederationPlan {
    /// Returns `true` if the plan has no sub-plans.
    pub fn is_empty(&self) -> bool {
        self.sub_plans.is_empty()
    }

    /// Returns the total number of fields across all sub-plans.
    pub fn total_field_count(&self) -> usize {
        self.sub_plans.iter().map(|sp| sp.field_count()).sum()
    }
}

// ---------------------------------------------------------------------------
// Planner configuration
// ---------------------------------------------------------------------------

/// Configuration for the enhanced federation planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedPlannerConfig {
    /// Penalty factor added when splitting fields across sources.
    pub cross_source_penalty: f64,
    /// Whether to prefer sources with lower latency.
    pub prefer_low_latency: bool,
    /// Maximum acceptable cost before raising an error.
    pub max_plan_cost: Option<f64>,
    /// Whether to enable request batching.
    pub enable_batching: bool,
}

impl Default for EnhancedPlannerConfig {
    fn default() -> Self {
        Self {
            cross_source_penalty: 1.5,
            prefer_low_latency: true,
            max_plan_cost: None,
            enable_batching: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// Cost-based federation planner with field-level routing and batching.
///
/// The planner operates in three phases:
/// 1. **Routing**: assign each requested field to its owning source.
/// 2. **Batching**: group fields going to the same source into a single sub-plan.
/// 3. **Cost estimation**: compute cost and latency estimates for the full plan.
#[derive(Debug, Default)]
pub struct EnhancedFederationPlanner {
    config: EnhancedPlannerConfig,
    /// Registered sources, indexed by source id.
    sources: HashMap<String, FederationSource>,
}

impl EnhancedFederationPlanner {
    /// Create a planner with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a planner with custom configuration.
    pub fn with_config(config: EnhancedPlannerConfig) -> Self {
        Self {
            config,
            sources: HashMap::new(),
        }
    }

    /// Register a source with the planner.
    pub fn register_source(&mut self, source: FederationSource) {
        self.sources.insert(source.id.clone(), source);
    }

    /// Route a single field to its owning source.
    ///
    /// Returns `None` if no registered source owns the field.
    pub fn route_field(&self, type_name: &str, field_name: &str) -> Option<&FederationSource> {
        let qualified = format!("{type_name}.{field_name}");

        // Prefer exact field-level ownership over type-level ownership.
        let exact = self
            .sources
            .values()
            .filter(|s| s.owned_fields.iter().any(|f| f == &qualified))
            .min_by_key(|s| {
                if self.config.prefer_low_latency {
                    s.stats.avg_latency_ms
                } else {
                    0
                }
            });

        if exact.is_some() {
            return exact;
        }

        // Fall back to type-level ownership.
        self.sources
            .values()
            .filter(|s| s.owns_type(type_name))
            .min_by_key(|s| {
                if self.config.prefer_low_latency {
                    s.stats.avg_latency_ms
                } else {
                    0
                }
            })
    }

    /// Plan a set of field requests, returning a cost-optimised
    /// `EnhancedFederationPlan`.
    pub fn plan_fields(&self, requests: &[FieldRequest]) -> Result<EnhancedFederationPlan> {
        if requests.is_empty() {
            return Ok(EnhancedFederationPlan {
                sub_plans: Vec::new(),
                total_cost: 0.0,
                critical_path_latency_ms: 0,
                is_fully_parallel: true,
                contributing_sources: Vec::new(),
            });
        }

        // Phase 1 – route every field to a source.
        let mut source_to_fields: BTreeMap<String, Vec<FieldRequest>> = BTreeMap::new();
        for req in requests {
            let source = self
                .route_field(&req.type_name, &req.field_name)
                .ok_or_else(|| {
                    anyhow!("No source owns field {}.{}", req.type_name, req.field_name)
                })?;
            source_to_fields
                .entry(source.id.clone())
                .or_default()
                .push(req.clone());
        }

        // Phase 2 – batch fields per source, respecting max_batch_size.
        let mut sub_plans: Vec<BatchedSubPlan> = Vec::new();

        for (source_id, fields) in &source_to_fields {
            let source = self
                .sources
                .get(source_id)
                .ok_or_else(|| anyhow!("Source '{}' registered but missing from map", source_id))?;

            let chunks = if self.config.enable_batching && source.stats.supports_batching {
                let max = if source.stats.max_batch_size == 0 {
                    fields.len()
                } else {
                    source.stats.max_batch_size
                };
                fields.chunks(max.max(1)).collect::<Vec<_>>()
            } else {
                fields.iter().map(std::slice::from_ref).collect::<Vec<_>>()
            };

            for chunk in chunks {
                let n = chunk.len() as f64;
                let cost = n * source.stats.cost_per_field
                    + if sub_plans.is_empty() {
                        0.0
                    } else {
                        self.config.cross_source_penalty
                    };

                sub_plans.push(BatchedSubPlan {
                    source_id: source_id.clone(),
                    fields: chunk.to_vec(),
                    estimated_cost: cost,
                    estimated_latency_ms: source.stats.avg_latency_ms,
                    depends_on: Vec::new(),
                });
            }
        }

        // Phase 3 – resolve cross-source dependencies.
        // If a field in plan B depends on data from plan A (detected by type
        // ownership cross-reference), mark that dependency.
        // Collect source_order as owned Strings to avoid borrowing sub_plans.
        let source_order: Vec<String> = sub_plans.iter().map(|sp| sp.source_id.clone()).collect();

        let mut seen_sources: HashSet<String> = HashSet::new();
        for (plan_idx, plan) in sub_plans.iter_mut().enumerate() {
            let plan_source = plan.source_id.clone();
            let plan_fields: Vec<String> =
                plan.fields.iter().map(|f| f.type_name.clone()).collect();

            // A sub-plan depends on all previous sub-plans whose types overlap
            // with the fields it resolves — simplified heuristic.
            let mut new_deps: Vec<usize> = Vec::new();
            for type_name in &plan_fields {
                for (idx, src) in source_order.iter().enumerate() {
                    if idx < plan_idx
                        && seen_sources.contains(src)
                        && src != &plan_source
                        && self
                            .sources
                            .get(src.as_str())
                            .is_some_and(|s| s.owns_type(type_name))
                        && !new_deps.contains(&idx)
                    {
                        new_deps.push(idx);
                    }
                }
            }
            for dep in new_deps {
                if !plan.depends_on.contains(&dep) {
                    plan.depends_on.push(dep);
                }
            }
            seen_sources.insert(plan_source);
        }

        // Sort depends_on lists for determinism.
        for plan in &mut sub_plans {
            plan.depends_on.sort_unstable();
            plan.depends_on.dedup();
        }

        let total_cost: f64 = sub_plans.iter().map(|sp| sp.estimated_cost).sum();

        if let Some(max_cost) = self.config.max_plan_cost {
            if total_cost > max_cost {
                return Err(anyhow!(
                    "Plan cost {total_cost:.2} exceeds configured maximum {max_cost:.2}"
                ));
            }
        }

        let critical_path_latency_ms = self.compute_critical_path(&sub_plans);
        let is_fully_parallel = sub_plans.iter().all(|sp| sp.depends_on.is_empty());

        let contributing_sources: Vec<String> = sub_plans
            .iter()
            .map(|sp| sp.source_id.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        Ok(EnhancedFederationPlan {
            sub_plans,
            total_cost,
            critical_path_latency_ms,
            is_fully_parallel,
            contributing_sources,
        })
    }

    /// Compute critical-path latency using DP over the dependency DAG.
    fn compute_critical_path(&self, sub_plans: &[BatchedSubPlan]) -> u64 {
        if sub_plans.is_empty() {
            return 0;
        }
        let mut finish_time = vec![0u64; sub_plans.len()];
        for (i, sp) in sub_plans.iter().enumerate() {
            let earliest_start = sp
                .depends_on
                .iter()
                .map(|&dep| finish_time[dep])
                .max()
                .unwrap_or(0);
            finish_time[i] = earliest_start + sp.estimated_latency_ms;
        }
        finish_time.into_iter().max().unwrap_or(0)
    }

    /// Return all registered sources.
    pub fn sources(&self) -> impl Iterator<Item = &FederationSource> {
        self.sources.values()
    }

    /// Estimate the cost of executing a set of field requests without building
    /// a full plan. Useful for admission-control decisions.
    pub fn estimate_cost(&self, requests: &[FieldRequest]) -> f64 {
        let mut total = 0.0;
        let mut seen_sources: HashSet<&str> = HashSet::new();
        for req in requests {
            if let Some(source) = self.route_field(&req.type_name, &req.field_name) {
                total += source.stats.cost_per_field;
                if !seen_sources.contains(source.id.as_str()) {
                    if !seen_sources.is_empty() {
                        total += self.config.cross_source_penalty;
                    }
                    seen_sources.insert(source.id.as_str());
                }
            }
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn user_source() -> FederationSource {
        FederationSource::new("users", "https://users.example.com/graphql")
            .with_type("User")
            .with_field("User.id")
            .with_field("User.name")
            .with_field("User.email")
            .with_stats(SourceStats {
                avg_latency_ms: 20,
                cost_per_field: 1.0,
                supports_batching: true,
                max_batch_size: 0,
            })
    }

    fn product_source() -> FederationSource {
        FederationSource::new("products", "https://products.example.com/graphql")
            .with_type("Product")
            .with_field("Product.sku")
            .with_field("Product.price")
            .with_stats(SourceStats {
                avg_latency_ms: 30,
                cost_per_field: 2.0,
                supports_batching: true,
                max_batch_size: 10,
            })
    }

    fn review_source() -> FederationSource {
        FederationSource::new("reviews", "https://reviews.example.com/graphql")
            .with_type("Review")
            .with_field("Review.rating")
            .with_field("Review.body")
            .with_stats(SourceStats {
                avg_latency_ms: 40,
                cost_per_field: 1.5,
                supports_batching: false,
                max_batch_size: 0,
            })
    }

    fn make_planner() -> EnhancedFederationPlanner {
        let mut planner = EnhancedFederationPlanner::new();
        planner.register_source(user_source());
        planner.register_source(product_source());
        planner.register_source(review_source());
        planner
    }

    // --- FederationSource ---

    #[test]
    fn test_source_owns_type() {
        let src = user_source();
        assert!(src.owns_type("User"));
        assert!(!src.owns_type("Product"));
    }

    #[test]
    fn test_source_owns_field_exact() {
        let src = user_source();
        assert!(src.owns_field("User.name"));
        // Fields on types not owned by this source should return false
        assert!(!src.owns_field("Product.unknown"));
    }

    #[test]
    fn test_source_owns_field_by_type() {
        // A field not explicitly listed but on an owned type should resolve by type.
        let src = user_source();
        // "User.anything" should resolve because User type is owned.
        assert!(src.owns_field("User.anything"));
    }

    #[test]
    fn test_source_builder_chaining() {
        let src = FederationSource::new("test", "http://test")
            .with_type("Foo")
            .with_field("Foo.bar");
        assert_eq!(src.owned_types.len(), 1);
        assert_eq!(src.owned_fields.len(), 1);
    }

    // --- FieldRequest ---

    #[test]
    fn test_field_request_qualified_name() {
        let req = FieldRequest::new("User", "name", "users");
        assert_eq!(req.qualified_name(), "User.name");
    }

    #[test]
    fn test_field_request_equality() {
        let a = FieldRequest::new("User", "id", "users");
        let b = FieldRequest::new("User", "id", "users");
        assert_eq!(a, b);
    }

    // --- EnhancedFederationPlanner routing ---

    #[test]
    fn test_route_field_exact() {
        let planner = make_planner();
        let src = planner.route_field("User", "name");
        assert!(src.is_some());
        assert_eq!(src.unwrap().id, "users");
    }

    #[test]
    fn test_route_field_type_fallback() {
        let planner = make_planner();
        // "User.bio" is not explicitly listed but User type is owned by "users".
        let src = planner.route_field("User", "bio");
        assert!(src.is_some());
        assert_eq!(src.unwrap().id, "users");
    }

    #[test]
    fn test_route_field_unknown_returns_none() {
        let planner = make_planner();
        let src = planner.route_field("Ghost", "field");
        assert!(src.is_none());
    }

    #[test]
    fn test_route_field_product() {
        let planner = make_planner();
        let src = planner.route_field("Product", "price");
        assert!(src.is_some());
        assert_eq!(src.unwrap().id, "products");
    }

    // --- Plan construction ---

    #[test]
    fn test_plan_empty_requests() {
        let planner = make_planner();
        let plan = planner.plan_fields(&[]).unwrap();
        assert!(plan.is_empty());
        assert_eq!(plan.total_cost, 0.0);
    }

    #[test]
    fn test_plan_single_source() {
        let planner = make_planner();
        let requests = vec![
            FieldRequest::new("User", "id", "users"),
            FieldRequest::new("User", "name", "users"),
        ];
        let plan = planner.plan_fields(&requests).unwrap();
        assert_eq!(plan.sub_plans.len(), 1);
        assert_eq!(plan.sub_plans[0].source_id, "users");
        assert_eq!(plan.sub_plans[0].field_count(), 2);
    }

    #[test]
    fn test_plan_multiple_sources() {
        let planner = make_planner();
        let requests = vec![
            FieldRequest::new("User", "name", "users"),
            FieldRequest::new("Product", "price", "products"),
        ];
        let plan = planner.plan_fields(&requests).unwrap();
        assert_eq!(plan.sub_plans.len(), 2);
        assert_eq!(plan.contributing_sources.len(), 2);
    }

    #[test]
    fn test_plan_total_field_count() {
        let planner = make_planner();
        let requests = vec![
            FieldRequest::new("User", "id", "users"),
            FieldRequest::new("User", "name", "users"),
            FieldRequest::new("Product", "sku", "products"),
        ];
        let plan = planner.plan_fields(&requests).unwrap();
        assert_eq!(plan.total_field_count(), 3);
    }

    #[test]
    fn test_plan_cost_nonzero() {
        let planner = make_planner();
        let requests = vec![FieldRequest::new("User", "id", "users")];
        let plan = planner.plan_fields(&requests).unwrap();
        assert!(plan.total_cost > 0.0);
    }

    #[test]
    fn test_plan_latency_nonzero() {
        let planner = make_planner();
        let requests = vec![FieldRequest::new("User", "id", "users")];
        let plan = planner.plan_fields(&requests).unwrap();
        assert!(plan.critical_path_latency_ms > 0);
    }

    #[test]
    fn test_plan_single_source_fully_parallel() {
        let planner = make_planner();
        let requests = vec![
            FieldRequest::new("User", "id", "users"),
            FieldRequest::new("Product", "sku", "products"),
        ];
        let plan = planner.plan_fields(&requests).unwrap();
        // Without explicit cross-type dependencies, sub-plans are independent.
        assert!(plan.is_fully_parallel);
    }

    #[test]
    fn test_plan_max_cost_enforced() {
        let config = EnhancedPlannerConfig {
            max_plan_cost: Some(0.5), // very low limit
            ..Default::default()
        };
        let mut planner = EnhancedFederationPlanner::with_config(config);
        planner.register_source(user_source());
        let requests = vec![
            FieldRequest::new("User", "id", "users"),
            FieldRequest::new("User", "name", "users"),
        ];
        let result = planner.plan_fields(&requests);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds"));
    }

    #[test]
    fn test_plan_unknown_field_returns_error() {
        let planner = make_planner();
        let requests = vec![FieldRequest::new("Phantom", "field", "???")];
        let result = planner.plan_fields(&requests);
        assert!(result.is_err());
    }

    // --- Batching ---

    #[test]
    fn test_batching_respects_max_batch_size() {
        let mut planner = EnhancedFederationPlanner::new();
        let src = FederationSource::new("src", "http://src")
            .with_type("Item")
            .with_stats(SourceStats {
                avg_latency_ms: 10,
                cost_per_field: 1.0,
                supports_batching: true,
                max_batch_size: 2,
            });
        planner.register_source(src);

        let requests: Vec<FieldRequest> = (0..5)
            .map(|i| FieldRequest::new("Item", format!("field{i}").as_str(), "src"))
            .collect();
        let plan = planner.plan_fields(&requests).unwrap();
        // 5 fields with max_batch_size=2 → 3 sub-plans (2+2+1)
        assert_eq!(plan.sub_plans.len(), 3);
    }

    #[test]
    fn test_batching_disabled_when_source_does_not_support_it() {
        let mut planner = EnhancedFederationPlanner::new();
        let src = FederationSource::new("no_batch", "http://nb")
            .with_type("Foo")
            .with_stats(SourceStats {
                avg_latency_ms: 10,
                cost_per_field: 1.0,
                supports_batching: false,
                max_batch_size: 0,
            });
        planner.register_source(src);

        let requests = vec![
            FieldRequest::new("Foo", "a", "no_batch"),
            FieldRequest::new("Foo", "b", "no_batch"),
        ];
        let plan = planner.plan_fields(&requests).unwrap();
        // Each field becomes its own sub-plan when batching is unsupported.
        assert_eq!(plan.sub_plans.len(), 2);
    }

    // --- Cost estimation ---

    #[test]
    fn test_estimate_cost_single_source() {
        let planner = make_planner();
        let requests = vec![
            FieldRequest::new("User", "id", "users"),
            FieldRequest::new("User", "name", "users"),
        ];
        // 2 fields × 1.0 cost_per_field, no cross-source penalty.
        let cost = planner.estimate_cost(&requests);
        assert!((cost - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_cost_cross_source_adds_penalty() {
        let planner = make_planner();
        let requests = vec![
            FieldRequest::new("User", "id", "users"),
            FieldRequest::new("Product", "sku", "products"),
        ];
        // 1.0 (user) + 2.0 (product) + 1.5 cross-source penalty = 4.5
        let cost = planner.estimate_cost(&requests);
        assert!(cost > 3.0); // at minimum cost_per_field sum
    }

    #[test]
    fn test_sources_iterator() {
        let planner = make_planner();
        let ids: Vec<&str> = planner.sources().map(|s| s.id.as_str()).collect();
        assert!(ids.contains(&"users"));
        assert!(ids.contains(&"products"));
        assert!(ids.contains(&"reviews"));
    }

    #[test]
    fn test_config_prefer_low_latency() {
        // When prefer_low_latency=true and two sources own the same type,
        // the lower-latency source should be preferred.
        let config = EnhancedPlannerConfig {
            prefer_low_latency: true,
            ..Default::default()
        };
        let mut planner = EnhancedFederationPlanner::with_config(config);
        let slow = FederationSource::new("slow", "http://slow")
            .with_type("Shared")
            .with_stats(SourceStats {
                avg_latency_ms: 100,
                ..Default::default()
            });
        let fast = FederationSource::new("fast", "http://fast")
            .with_type("Shared")
            .with_stats(SourceStats {
                avg_latency_ms: 10,
                ..Default::default()
            });
        planner.register_source(slow);
        planner.register_source(fast);

        // Should prefer "fast"
        let src = planner.route_field("Shared", "field");
        assert!(src.is_some());
        assert_eq!(src.unwrap().id, "fast");
    }

    #[test]
    fn test_plan_is_not_empty() {
        let planner = make_planner();
        let requests = vec![FieldRequest::new("User", "id", "users")];
        let plan = planner.plan_fields(&requests).unwrap();
        assert!(!plan.is_empty());
    }
}
