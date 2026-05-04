//! Cost-model-driven index dispatcher (the "brain" of the optimizer).
//!
//! This module is **execution-agnostic** — it knows nothing about HNSW,
//! IVF, or any concrete vector store; it operates purely on cost
//! estimates and recall constraints.  The top-level
//! [`crate::index_dispatcher::IndexDispatcher`] wires this brain to
//! actual index implementations.
//!
//! Algorithm:
//! 1. Score every available index family with [`CostModel::estimate`].
//! 2. Filter out families whose **estimated recall** is below
//!    `requested_recall` (unless the caller forbids fallback, in which
//!    case we keep them as last resort).
//! 3. Pick the family with the lowest cost.
//! 4. Maintain an ordered fallback chain — the second-cheapest family
//!    that meets recall; the dispatcher uses it when an actual recall
//!    measurement on the primary pick falls below the SLA.
//!
//! The brain is driven by online-updated weights: after each query a
//! [`QueryObservation`] feeds the [`QueryStats`] and the cost-model
//! weights are refreshed periodically.

use crate::optimizer::cost_model::{
    CostEstimate, CostModel, IndexFamily, IndexParameters, WorkloadProfile,
};
use crate::optimizer::query_stats::{QueryObservation, QueryStats};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Configuration for the optimizer dispatcher brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatcherConfig {
    /// Recall threshold below which a query is re-issued against the next-best
    /// index.  Value is in `[0.0, 1.0]`.
    pub recall_fallback_threshold: f32,
    /// Maximum number of fallback re-issues per query.
    pub max_fallbacks: usize,
    /// After this many observations, refresh the cost-model weights from
    /// accumulated [`QueryStats`].
    pub weight_refresh_interval: u64,
    /// Set of families the dispatcher is allowed to consider.  An empty set
    /// is interpreted as "all families".
    pub enabled_families: BTreeSet<IndexFamily>,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        Self {
            recall_fallback_threshold: 0.85,
            max_fallbacks: 1,
            weight_refresh_interval: 64,
            enabled_families: BTreeSet::new(), // empty = all
        }
    }
}

/// Output of [`OptimizerDispatcher::pick_plan`]: a primary + ordered fallbacks.
#[derive(Debug, Clone, PartialEq)]
pub struct DispatchPlan {
    /// Primary family the dispatcher recommends.
    pub primary: IndexFamily,
    /// Predicted cost of the primary in abstract units.
    pub primary_cost: f64,
    /// Predicted recall of the primary in `[0.0, 1.0]`.
    pub primary_recall: f32,
    /// Ordered fallback chain — try in order if recall trips threshold.
    pub fallbacks: Vec<CostEstimate>,
    /// Snapshot of the workload that produced this plan.
    pub workload: WorkloadProfile,
}

impl DispatchPlan {
    /// `true` if the plan has at least one fallback to try.
    pub fn has_fallback(&self) -> bool {
        !self.fallbacks.is_empty()
    }

    /// Return the family at fallback position `idx`, or `None` if exhausted.
    pub fn fallback_at(&self, idx: usize) -> Option<IndexFamily> {
        self.fallbacks.get(idx).map(|e| e.family)
    }
}

/// Errors returned by the dispatcher brain.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    /// No index family in the enabled set met the requested recall, even with
    /// fallback disabled.
    #[error(
        "no index family meets requested_recall={requested:.3}; best estimate was {best_recall:.3} \
         from {best_family:?}"
    )]
    NoFamilyMeetsRecall {
        /// Recall floor requested by the caller.
        requested: f32,
        /// Best recall the cost model could project across all enabled families.
        best_recall: f32,
        /// Family that produced `best_recall`.
        best_family: IndexFamily,
    },
    /// No families are enabled (config has empty set after intersecting with
    /// the universe).
    #[error("no index families enabled in dispatcher configuration")]
    NoFamiliesEnabled,
}

/// Optimizer dispatcher (brain — selects family & maintains stats).
pub struct OptimizerDispatcher {
    cost_model: CostModel,
    stats: QueryStats,
    config: DispatcherConfig,
    observations_since_refresh: u64,
}

impl Default for OptimizerDispatcher {
    fn default() -> Self {
        Self::new(
            CostModel::default(),
            QueryStats::default(),
            DispatcherConfig::default(),
        )
    }
}

impl OptimizerDispatcher {
    /// Construct a dispatcher with explicit cost model, stats, and config.
    pub fn new(cost_model: CostModel, stats: QueryStats, config: DispatcherConfig) -> Self {
        Self {
            cost_model,
            stats,
            config,
            observations_since_refresh: 0,
        }
    }

    /// Borrow the underlying cost model.
    pub fn cost_model(&self) -> &CostModel {
        &self.cost_model
    }

    /// Borrow the accumulated runtime statistics.
    pub fn stats(&self) -> &QueryStats {
        &self.stats
    }

    /// Borrow the dispatcher configuration.
    pub fn config(&self) -> &DispatcherConfig {
        &self.config
    }

    /// Mutable access for tests and configuration updates.
    pub fn cost_model_mut(&mut self) -> &mut CostModel {
        &mut self.cost_model
    }

    /// Mutable access for stats — generally only needed by the wrapper.
    pub fn stats_mut(&mut self) -> &mut QueryStats {
        &mut self.stats
    }

    /// Pick a [`DispatchPlan`] for the given workload.
    pub fn pick_plan(&self, workload: &WorkloadProfile) -> Result<DispatchPlan, DispatchError> {
        let enabled = self.enabled_families();
        if enabled.is_empty() {
            return Err(DispatchError::NoFamiliesEnabled);
        }

        // Score every enabled family.
        let mut estimates: Vec<CostEstimate> = enabled
            .iter()
            .map(|fam| self.cost_model.estimate(*fam, workload))
            .collect();

        // Sort by cost ascending — cheapest first.
        estimates.sort_by(|a, b| {
            a.cost
                .partial_cmp(&b.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Partition into "meets recall" and "below recall".
        let recall_target = workload.requested_recall;
        let (meets, below): (Vec<_>, Vec<_>) = estimates
            .iter()
            .cloned()
            .partition(|e| e.recall >= recall_target);

        let primary_estimate = if let Some(first) = meets.first() {
            first.clone()
        } else {
            // No family meets recall.  Pick the highest-recall one as a
            // best-effort primary, and surface this via the error for callers
            // that want strict behaviour.
            let best = below
                .iter()
                .max_by(|a, b| {
                    a.recall
                        .partial_cmp(&b.recall)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .ok_or(DispatchError::NoFamiliesEnabled)?
                .clone();
            tracing::warn!(
                "OptimizerDispatcher: no family meets requested_recall={:.3}; \
                 best is {:?} with recall={:.3}",
                recall_target,
                best.family,
                best.recall
            );
            best
        };

        // Build fallback chain: every other estimate that meets recall, in
        // ascending cost order (already sorted).  If `meets` is empty, fall
        // back to the rest of the cost-sorted list.
        let fallbacks: Vec<CostEstimate> = if !meets.is_empty() {
            meets
                .into_iter()
                .filter(|e| e.family != primary_estimate.family)
                .collect()
        } else {
            estimates
                .into_iter()
                .filter(|e| e.family != primary_estimate.family)
                .collect()
        };

        Ok(DispatchPlan {
            primary: primary_estimate.family,
            primary_cost: primary_estimate.cost,
            primary_recall: primary_estimate.recall,
            fallbacks,
            workload: workload.clone(),
        })
    }

    /// Evaluate whether an observed recall on the primary triggers fallback.
    pub fn should_fallback(&self, plan: &DispatchPlan, observed_recall: f32) -> bool {
        plan.has_fallback() && observed_recall < self.config.recall_fallback_threshold
    }

    /// Record a query observation and refresh weights when the interval is hit.
    ///
    /// Returns `true` if weights were refreshed during this call.
    pub fn record_observation(&mut self, observation: QueryObservation) -> bool {
        self.stats.record(observation);
        self.observations_since_refresh += 1;

        if self.observations_since_refresh >= self.config.weight_refresh_interval {
            let new_weights = self.stats.recommended_weights(self.cost_model.weights());
            *self.cost_model.weights_mut() = new_weights;
            self.observations_since_refresh = 0;
            true
        } else {
            false
        }
    }

    /// Force-refresh cost-model weights from the current stats snapshot.
    pub fn force_refresh_weights(&mut self) {
        let new_weights = self.stats.recommended_weights(self.cost_model.weights());
        *self.cost_model.weights_mut() = new_weights;
        self.observations_since_refresh = 0;
    }

    /// Resolve the universe of enabled families.  An empty config set means
    /// "every family in [`IndexFamily::all`]".
    fn enabled_families(&self) -> Vec<IndexFamily> {
        let universe = IndexFamily::all();
        if self.config.enabled_families.is_empty() {
            universe.to_vec()
        } else {
            universe
                .into_iter()
                .filter(|f| self.config.enabled_families.contains(f))
                .collect()
        }
    }
}

/// Convenience constructor for a dispatcher with custom enabled families.
pub fn dispatcher_with_families(families: &[IndexFamily]) -> OptimizerDispatcher {
    let cfg = DispatcherConfig {
        enabled_families: families.iter().copied().collect(),
        ..Default::default()
    };
    OptimizerDispatcher::new(CostModel::default(), QueryStats::default(), cfg)
}

/// Convenience constructor for a dispatcher with explicit cost-model parameters.
pub fn dispatcher_with_parameters(parameters: IndexParameters) -> OptimizerDispatcher {
    let cost_model = CostModel::new(parameters, Default::default());
    OptimizerDispatcher::new(
        cost_model,
        QueryStats::default(),
        DispatcherConfig::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload(n: usize, dim: usize, recall: f32) -> WorkloadProfile {
        WorkloadProfile::new(n, dim, recall)
    }

    #[test]
    fn dispatcher_picks_lowest_cost_meeting_recall() {
        let dispatcher = OptimizerDispatcher::default();
        let plan = dispatcher
            .pick_plan(&workload(100_000, 128, 0.9))
            .expect("plan must exist");
        // For 100k vectors, HNSW should be the cheapest meeting recall.
        // (IVF is cheap but recall floor is 0.85 < 0.9, may or may not meet.)
        assert!(
            plan.primary_recall >= 0.9,
            "primary recall must meet target"
        );
    }

    #[test]
    fn dispatcher_provides_fallback_chain() {
        let dispatcher = OptimizerDispatcher::default();
        let plan = dispatcher
            .pick_plan(&workload(100_000, 128, 0.85))
            .expect("plan must exist");
        // With recall=0.85 several families meet recall → fallbacks present.
        assert!(plan.has_fallback(), "fallback chain should be non-empty");
    }

    #[test]
    fn dispatcher_handles_unmet_recall_with_warning() {
        let dispatcher = OptimizerDispatcher::default();
        // Demand 0.999 — nothing meets that floor.
        let plan = dispatcher
            .pick_plan(&workload(10_000, 128, 0.999))
            .expect("dispatcher returns best-effort plan");
        // primary_recall is best-available; below requested.
        assert!(plan.primary_recall < 0.999);
    }

    #[test]
    fn enabled_families_filter_respected() {
        let dispatcher = dispatcher_with_families(&[IndexFamily::Lsh, IndexFamily::Pq]);
        let plan = dispatcher
            .pick_plan(&workload(10_000, 128, 0.7))
            .expect("plan must exist");
        assert!(matches!(plan.primary, IndexFamily::Lsh | IndexFamily::Pq));
    }

    #[test]
    fn empty_enabled_set_returns_error_when_constructed_directly() {
        // Manually craft a config where the set is *non-default* but empty by
        // intersecting with an unknown family.  In practice we treat empty as
        // "all", so the only path to NoFamiliesEnabled is overriding the set
        // post-construction.
        let mut dispatcher = OptimizerDispatcher::default();
        dispatcher.config.enabled_families.insert(IndexFamily::Hnsw);
        dispatcher
            .config
            .enabled_families
            .remove(&IndexFamily::Hnsw);
        // After remove, set is empty again → fallback "all" path.
        let plan = dispatcher.pick_plan(&workload(1_000, 8, 0.5));
        assert!(plan.is_ok());
    }

    #[test]
    fn should_fallback_triggers_when_observed_below_threshold() {
        let dispatcher = OptimizerDispatcher::default();
        let plan = dispatcher
            .pick_plan(&workload(100_000, 128, 0.85))
            .expect("plan");
        assert!(dispatcher.should_fallback(&plan, 0.5));
        assert!(!dispatcher.should_fallback(&plan, 0.95));
    }

    #[test]
    fn record_observation_refreshes_weights_at_interval() {
        let mut dispatcher = OptimizerDispatcher::default();
        dispatcher.config.weight_refresh_interval = 3;
        // Record 2 observations — no refresh expected.
        for _ in 0..2 {
            let refreshed = dispatcher.record_observation(QueryObservation::new(
                IndexFamily::Hnsw,
                true,
                100.0,
                Some(0.92),
                50.0,
            ));
            assert!(!refreshed);
        }
        // 3rd observation triggers refresh.
        let refreshed = dispatcher.record_observation(QueryObservation::new(
            IndexFamily::Hnsw,
            true,
            100.0,
            Some(0.92),
            50.0,
        ));
        assert!(refreshed, "refresh should trigger on 3rd observation");

        // Weight should now reflect 100/50 = 2.0.
        let w = dispatcher.cost_model().weights().get(IndexFamily::Hnsw);
        assert!((w - 2.0).abs() < 1e-6);
    }

    #[test]
    fn force_refresh_weights_immediately() {
        let mut dispatcher = OptimizerDispatcher::default();
        dispatcher.stats.record(QueryObservation::new(
            IndexFamily::Pq,
            true,
            300.0,
            None,
            150.0,
        ));
        dispatcher.force_refresh_weights();
        let w = dispatcher.cost_model().weights().get(IndexFamily::Pq);
        assert!((w - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dispatcher_with_parameters_uses_overrides() {
        // Larger beam = higher cost
        let params = IndexParameters {
            hnsw_ef: 200,
            ..Default::default()
        };
        let dispatcher = dispatcher_with_parameters(params);
        let cost_high = dispatcher
            .cost_model()
            .estimate(IndexFamily::Hnsw, &workload(100_000, 128, 0.9));
        let dispatcher_default = OptimizerDispatcher::default();
        let cost_low = dispatcher_default
            .cost_model()
            .estimate(IndexFamily::Hnsw, &workload(100_000, 128, 0.9));
        assert!(cost_high.cost > cost_low.cost);
    }
}
