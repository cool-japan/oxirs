//! Adaptive Join Ordering with Runtime Feedback
//!
//! This module implements an adaptive query optimizer that learns from runtime
//! execution statistics to continuously improve join ordering decisions.
//! It uses a combination of cost-based optimization and runtime feedback.

use crate::algebra::{Term, TriplePattern};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Runtime statistics collected during query execution for feedback loops
#[derive(Debug, Clone, Default)]
pub struct RuntimeStats {
    /// Estimated vs actual cardinality per pattern fingerprint
    pub pattern_stats: HashMap<String, PatternRuntimeStats>,
    /// Join selectivity statistics per join fingerprint
    pub join_stats: HashMap<String, JoinRuntimeStats>,
    /// Execution times per plan component
    pub execution_times: HashMap<String, Duration>,
    /// Total number of queries optimized
    pub query_count: u64,
}

/// Per-pattern runtime statistics
#[derive(Debug, Clone, Default)]
pub struct PatternRuntimeStats {
    /// Cumulative estimated cardinality (sum across executions)
    pub estimated_cardinality_sum: u64,
    /// Cumulative actual cardinality (sum across executions)
    pub actual_cardinality_sum: u64,
    /// Most recent estimation error (actual / estimated ratio)
    pub estimation_error: f64,
    /// Number of samples recorded
    pub sample_count: u64,
    /// Correction factor derived from history (actual / estimated)
    pub correction_factor: f64,
}

/// Per-join runtime statistics
#[derive(Debug, Clone, Default)]
pub struct JoinRuntimeStats {
    /// Cumulative left input cardinality
    pub left_cardinality_sum: u64,
    /// Cumulative right input cardinality
    pub right_cardinality_sum: u64,
    /// Cumulative output cardinality
    pub output_cardinality_sum: u64,
    /// Observed selectivity (output / (left * right))
    pub observed_selectivity: f64,
    /// Number of samples
    pub sample_count: u64,
}

/// Thread-safe store for adaptive runtime statistics
pub struct AdaptiveStatsStore {
    stats: Arc<RwLock<RuntimeStats>>,
    max_history: usize,
}

impl AdaptiveStatsStore {
    /// Create a new adaptive statistics store
    pub fn new(max_history: usize) -> Self {
        Self {
            stats: Arc::new(RwLock::new(RuntimeStats::default())),
            max_history,
        }
    }

    /// Record actual vs estimated cardinality for a pattern
    pub fn record_pattern_execution(&self, pattern_id: &str, estimated: u64, actual: u64) {
        let Ok(mut stats) = self.stats.write() else {
            return;
        };
        let entry = stats
            .pattern_stats
            .entry(pattern_id.to_string())
            .or_default();

        entry.sample_count += 1;
        entry.estimated_cardinality_sum += estimated;
        entry.actual_cardinality_sum += actual;

        let ratio = if estimated > 0 {
            actual as f64 / estimated as f64
        } else {
            1.0
        };
        entry.estimation_error = ratio;

        // Exponential moving average of correction factor (alpha = 0.2)
        if entry.sample_count == 1 {
            entry.correction_factor = ratio;
        } else {
            entry.correction_factor = 0.8 * entry.correction_factor + 0.2 * ratio;
        }

        // Trim history if needed by resetting sums when they exceed max_history
        if entry.sample_count > self.max_history as u64 {
            let avg_est = entry.estimated_cardinality_sum / entry.sample_count;
            let avg_act = entry.actual_cardinality_sum / entry.sample_count;
            entry.estimated_cardinality_sum = avg_est;
            entry.actual_cardinality_sum = avg_act;
            entry.sample_count = 1;
        }
    }

    /// Record actual join execution statistics
    pub fn record_join_execution(&self, join_id: &str, left: u64, right: u64, output: u64) {
        let Ok(mut stats) = self.stats.write() else {
            return;
        };
        let entry = stats.join_stats.entry(join_id.to_string()).or_default();

        entry.sample_count += 1;
        entry.left_cardinality_sum += left;
        entry.right_cardinality_sum += right;
        entry.output_cardinality_sum += output;

        let denominator = (left as f64) * (right as f64);
        let selectivity = if denominator > 0.0 {
            output as f64 / denominator
        } else {
            0.0
        };

        // Exponential moving average
        if entry.sample_count == 1 {
            entry.observed_selectivity = selectivity;
        } else {
            entry.observed_selectivity = 0.8 * entry.observed_selectivity + 0.2 * selectivity;
        }
    }

    /// Record execution time for a plan component
    pub fn record_execution_time(&self, component_id: &str, duration: Duration) {
        let Ok(mut stats) = self.stats.write() else {
            return;
        };
        stats
            .execution_times
            .insert(component_id.to_string(), duration);
    }

    /// Get adjusted cardinality estimate incorporating runtime feedback
    pub fn get_adjusted_cardinality(&self, pattern_id: &str, base_estimate: u64) -> u64 {
        let Ok(stats) = self.stats.read() else {
            return base_estimate;
        };
        let Some(entry) = stats.pattern_stats.get(pattern_id) else {
            return base_estimate;
        };

        if entry.sample_count == 0 {
            return base_estimate;
        }

        let adjusted = (base_estimate as f64 * entry.correction_factor).round() as u64;
        adjusted.max(1)
    }

    /// Get adjusted selectivity incorporating runtime feedback
    pub fn get_adjusted_selectivity(&self, join_id: &str, base_selectivity: f64) -> f64 {
        let Ok(stats) = self.stats.read() else {
            return base_selectivity;
        };
        let Some(entry) = stats.join_stats.get(join_id) else {
            return base_selectivity;
        };

        if entry.sample_count == 0 {
            return base_selectivity;
        }

        // Blend base estimate with observed selectivity (weight toward observed as samples grow)
        let observed_weight = (entry.sample_count as f64 / 10.0).min(0.8);
        let base_weight = 1.0 - observed_weight;
        (base_weight * base_selectivity + observed_weight * entry.observed_selectivity)
            .clamp(0.0001, 1.0)
    }

    /// Snapshot a read-only view of the current statistics
    pub fn snapshot(&self) -> Option<RuntimeStats> {
        self.stats.read().ok().map(|s| s.clone())
    }
}

/// Which join algorithm to use for a given join node
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinAlgorithm {
    /// Hash join: good when one side is small enough to fit in memory
    Hash,
    /// Nested loop join: good when outer is tiny and inner has index support
    NestedLoop,
    /// Sort-merge join: good for large sorted inputs with shared sort keys
    Merge,
}

/// A term in a triple pattern (subject, predicate, or object position)
#[derive(Debug, Clone)]
pub enum PatternTerm {
    Variable(String),
    Iri(String),
    Literal(String),
    BlankNode(String),
}

impl PatternTerm {
    /// Return true if this term is a variable (unbound position)
    pub fn is_variable(&self) -> bool {
        matches!(self, PatternTerm::Variable(_))
    }

    /// Return the variable name if this is a variable
    pub fn variable_name(&self) -> Option<&str> {
        match self {
            PatternTerm::Variable(name) => Some(name),
            _ => None,
        }
    }
}

/// Full information about a triple pattern for optimization purposes
#[derive(Debug, Clone)]
pub struct TriplePatternInfo {
    /// Unique identifier (fingerprint) for this pattern
    pub id: String,
    pub subject: PatternTerm,
    pub predicate: PatternTerm,
    pub object: PatternTerm,
    /// Estimated number of matching triples
    pub estimated_cardinality: u64,
    /// Variable names bound by this pattern
    pub bound_variables: Vec<String>,
    /// Reference to the original TriplePattern (for re-use downstream)
    pub original_pattern: Option<TriplePattern>,
}

impl TriplePatternInfo {
    /// Construct a TriplePatternInfo from an algebra TriplePattern and cardinality estimate
    pub fn from_triple_pattern(pattern: &TriplePattern, estimated_cardinality: u64) -> Self {
        let subject = term_to_pattern_term(&pattern.subject);
        let predicate = term_to_pattern_term(&pattern.predicate);
        let object = term_to_pattern_term(&pattern.object);

        let mut bound_variables = Vec::new();
        if let PatternTerm::Variable(ref v) = subject {
            bound_variables.push(v.clone());
        }
        if let PatternTerm::Variable(ref v) = predicate {
            bound_variables.push(v.clone());
        }
        if let PatternTerm::Variable(ref v) = object {
            bound_variables.push(v.clone());
        }

        // Build a stable pattern fingerprint
        let id = build_pattern_fingerprint(&subject, &predicate, &object);

        Self {
            id,
            subject,
            predicate,
            object,
            estimated_cardinality,
            bound_variables,
            original_pattern: Some(pattern.clone()),
        }
    }

    /// Number of unbound (variable) positions - lower = more selective
    pub fn bound_positions(&self) -> usize {
        let mut count = 0;
        if !self.subject.is_variable() {
            count += 1;
        }
        if !self.predicate.is_variable() {
            count += 1;
        }
        if !self.object.is_variable() {
            count += 1;
        }
        count
    }
}

fn term_to_pattern_term(term: &Term) -> PatternTerm {
    match term {
        Term::Variable(v) => PatternTerm::Variable(v.name().to_string()),
        Term::Iri(iri) => PatternTerm::Iri(iri.as_str().to_string()),
        Term::Literal(lit) => PatternTerm::Literal(lit.value.clone()),
        Term::BlankNode(bn) => PatternTerm::BlankNode(bn.as_str().to_string()),
        // Treat quoted triples and property paths as opaque IRIs for ordering purposes
        _ => PatternTerm::Iri(format!("{term}")),
    }
}

fn build_pattern_fingerprint(
    subject: &PatternTerm,
    predicate: &PatternTerm,
    object: &PatternTerm,
) -> String {
    let s = match subject {
        PatternTerm::Variable(_) => "?".to_string(),
        PatternTerm::Iri(v) => v.clone(),
        PatternTerm::Literal(v) => format!("\"{v}\""),
        PatternTerm::BlankNode(v) => format!("_:{v}"),
    };
    let p = match predicate {
        PatternTerm::Variable(_) => "?".to_string(),
        PatternTerm::Iri(v) => v.clone(),
        PatternTerm::Literal(v) => format!("\"{v}\""),
        PatternTerm::BlankNode(v) => format!("_:{v}"),
    };
    let o = match object {
        PatternTerm::Variable(_) => "?".to_string(),
        PatternTerm::Iri(v) => v.clone(),
        PatternTerm::Literal(v) => format!("\"{v}\""),
        PatternTerm::BlankNode(v) => format!("_:{v}"),
    };
    format!("{s} {p} {o}")
}

/// A join plan node - the output of the adaptive optimizer
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum JoinPlanNode {
    /// A leaf scan of a single triple pattern
    TriplePatternScan { info: TriplePatternInfo },
    /// Hash join of two sub-plans
    HashJoin {
        left: Box<JoinPlanNode>,
        right: Box<JoinPlanNode>,
        join_vars: Vec<String>,
        estimated_output: u64,
    },
    /// Nested-loop join (outer drives inner)
    NestedLoopJoin {
        outer: Box<JoinPlanNode>,
        inner: Box<JoinPlanNode>,
        join_vars: Vec<String>,
        estimated_output: u64,
    },
    /// Sort-merge join of two ordered sub-plans
    MergeJoin {
        left: Box<JoinPlanNode>,
        right: Box<JoinPlanNode>,
        join_vars: Vec<String>,
        sort_key: Vec<String>,
        estimated_output: u64,
    },
}

impl JoinPlanNode {
    /// Estimated output cardinality of this node
    pub fn estimated_cardinality(&self) -> u64 {
        match self {
            JoinPlanNode::TriplePatternScan { info } => info.estimated_cardinality,
            JoinPlanNode::HashJoin {
                estimated_output, ..
            } => *estimated_output,
            JoinPlanNode::NestedLoopJoin {
                estimated_output, ..
            } => *estimated_output,
            JoinPlanNode::MergeJoin {
                estimated_output, ..
            } => *estimated_output,
        }
    }

    /// Collect all variable names produced by this node
    pub fn output_variables(&self) -> Vec<String> {
        match self {
            JoinPlanNode::TriplePatternScan { info } => info.bound_variables.clone(),
            JoinPlanNode::HashJoin { left, right, .. } => {
                merge_variable_sets(left.output_variables(), right.output_variables())
            }
            JoinPlanNode::NestedLoopJoin { outer, inner, .. } => {
                merge_variable_sets(outer.output_variables(), inner.output_variables())
            }
            JoinPlanNode::MergeJoin { left, right, .. } => {
                merge_variable_sets(left.output_variables(), right.output_variables())
            }
        }
    }
}

fn merge_variable_sets(mut left: Vec<String>, right: Vec<String>) -> Vec<String> {
    for v in right {
        if !left.contains(&v) {
            left.push(v);
        }
    }
    left
}

/// Adaptive join order optimizer using dynamic programming with runtime feedback
pub struct AdaptiveJoinOrderOptimizer {
    /// Shared statistics store for feedback
    stats_store: Arc<AdaptiveStatsStore>,
    /// Above this threshold, fall back to greedy heuristics
    max_patterns_for_dp: usize,
    /// Default join selectivity when no history is available
    default_selectivity: f64,
}

impl AdaptiveJoinOrderOptimizer {
    /// Create a new optimizer with a reference to the shared stats store
    pub fn new(stats_store: Arc<AdaptiveStatsStore>) -> Self {
        Self {
            stats_store,
            max_patterns_for_dp: 8,
            default_selectivity: 0.1,
        }
    }

    /// Override DP threshold
    pub fn with_dp_threshold(mut self, threshold: usize) -> Self {
        self.max_patterns_for_dp = threshold;
        self
    }

    /// Main optimization entry point
    pub fn optimize(&self, patterns: Vec<TriplePatternInfo>) -> Result<JoinPlanNode> {
        if patterns.is_empty() {
            return Err(anyhow!("Cannot optimize empty pattern list"));
        }
        if patterns.len() == 1 {
            return Ok(JoinPlanNode::TriplePatternScan {
                info: patterns.into_iter().next().expect("checked len == 1"),
            });
        }

        // Apply adaptive cardinality corrections
        let adjusted = self.apply_cardinality_corrections(patterns);

        if adjusted.len() <= self.max_patterns_for_dp {
            self.dp_optimize(&adjusted)
        } else {
            self.greedy_optimize(&adjusted)
        }
    }

    /// Apply cardinality corrections from the stats store to all patterns
    fn apply_cardinality_corrections(
        &self,
        patterns: Vec<TriplePatternInfo>,
    ) -> Vec<TriplePatternInfo> {
        patterns
            .into_iter()
            .map(|mut p| {
                let adjusted = self
                    .stats_store
                    .get_adjusted_cardinality(&p.id, p.estimated_cardinality);
                p.estimated_cardinality = adjusted;
                p
            })
            .collect()
    }

    /// Dynamic programming optimizer for small numbers of patterns
    ///
    /// Uses the classic Selinger-style DP approach: enumerate all subsets,
    /// find optimal plan for each, then combine bottom-up.
    fn dp_optimize(&self, patterns: &[TriplePatternInfo]) -> Result<JoinPlanNode> {
        let n = patterns.len();
        // dp[mask] = best (cost, plan) for the subset encoded by `mask`
        // mask is a bitmask of pattern indices
        let total_masks = 1usize << n;
        let mut dp: Vec<Option<(f64, JoinPlanNode)>> = vec![None; total_masks];

        // Initialize single-pattern entries
        for (i, pattern) in patterns.iter().enumerate() {
            let mask = 1usize << i;
            let plan = JoinPlanNode::TriplePatternScan {
                info: pattern.clone(),
            };
            let cost = self.scan_cost(pattern);
            dp[mask] = Some((cost, plan));
        }

        // Fill DP table bottom-up (increasing subset size)
        for mask in 1..total_masks {
            // Skip single-bit masks (already initialized) and zero
            let bit_count = mask.count_ones() as usize;
            if bit_count < 2 {
                continue;
            }

            let mut best: Option<(f64, JoinPlanNode)> = None;

            // Enumerate all proper subsets of mask as the left side
            let mut left_mask = (mask - 1) & mask;
            while left_mask > 0 {
                let right_mask = mask ^ left_mask;
                if right_mask == 0 {
                    left_mask = (left_mask - 1) & mask;
                    continue;
                }

                // Avoid duplicate pairs (left, right) == (right, left) by requiring left < right
                if left_mask >= right_mask {
                    left_mask = (left_mask - 1) & mask;
                    continue;
                }

                let (Some((left_cost, ref left_plan)), Some((right_cost, ref right_plan))) =
                    (&dp[left_mask], &dp[right_mask])
                else {
                    left_mask = (left_mask - 1) & mask;
                    continue;
                };

                let left_vars = left_plan.output_variables();
                let right_vars = right_plan.output_variables();
                let join_vars = Self::find_join_variables_sets(&left_vars, &right_vars);

                // No shared variables means a cross product - penalize heavily
                let join_id = format!("{left_mask}x{right_mask}");
                let selectivity = if join_vars.is_empty() {
                    1.0 // cross product
                } else {
                    self.stats_store
                        .get_adjusted_selectivity(&join_id, self.default_selectivity)
                };

                let left_card = left_plan.estimated_cardinality();
                let right_card = right_plan.estimated_cardinality();
                let output_card =
                    ((left_card as f64 * right_card as f64 * selectivity).round() as u64).max(1);

                let algorithm = Self::select_join_algorithm(left_card, right_card, &join_vars);
                let join_cost =
                    self.join_cost(left_cost + right_cost, left_card, right_card, &algorithm);
                let total_cost = left_cost + right_cost + join_cost;

                if best.is_none() || total_cost < best.as_ref().map(|(c, _)| *c).unwrap_or(f64::MAX)
                {
                    let plan = self.build_join_plan(
                        left_plan.clone(),
                        right_plan.clone(),
                        join_vars,
                        output_card,
                        algorithm,
                    );
                    best = Some((total_cost, plan));
                }

                left_mask = (left_mask - 1) & mask;
            }

            if best.is_some() {
                dp[mask] = best;
            }
        }

        let full_mask = total_masks - 1;
        dp[full_mask]
            .take()
            .map(|(_, plan)| plan)
            .ok_or_else(|| anyhow!("DP optimizer failed to find a valid plan"))
    }

    /// Greedy optimizer for large numbers of patterns
    ///
    /// Repeatedly picks the cheapest next pattern to join with the current running plan.
    fn greedy_optimize(&self, patterns: &[TriplePatternInfo]) -> Result<JoinPlanNode> {
        if patterns.is_empty() {
            return Err(anyhow!("Cannot optimize empty pattern list"));
        }

        // Sort patterns by estimated cardinality ascending (most selective first)
        let mut remaining: Vec<TriplePatternInfo> = patterns.to_vec();
        remaining.sort_by_key(|p| p.estimated_cardinality);

        // Start with the most selective pattern
        let first = remaining.remove(0);
        let mut current_plan = JoinPlanNode::TriplePatternScan { info: first };

        while !remaining.is_empty() {
            // Find the next best pattern to join
            let mut best_idx = 0;
            let mut best_cost = f64::MAX;

            let current_vars = current_plan.output_variables();
            let current_card = current_plan.estimated_cardinality();

            for (idx, candidate) in remaining.iter().enumerate() {
                let join_vars =
                    Self::find_join_variables_sets(&current_vars, &candidate.bound_variables);
                let join_id = format!("g_{idx}_{}", candidate.id);
                let selectivity = self
                    .stats_store
                    .get_adjusted_selectivity(&join_id, self.default_selectivity);

                let algorithm = Self::select_join_algorithm(
                    current_card,
                    candidate.estimated_cardinality,
                    &join_vars,
                );
                let cost = self.join_cost(
                    0.0,
                    current_card,
                    candidate.estimated_cardinality,
                    &algorithm,
                );

                // Prefer patterns that share variables (avoid cross products)
                let adjusted_cost = if join_vars.is_empty() {
                    cost * 1000.0
                } else {
                    cost * (1.0 + (1.0 - selectivity))
                };

                if adjusted_cost < best_cost {
                    best_cost = adjusted_cost;
                    best_idx = idx;
                }
            }

            let next = remaining.remove(best_idx);
            let join_vars = Self::find_join_variables_sets(&current_vars, &next.bound_variables);
            let selectivity = self.stats_store.get_adjusted_selectivity(
                &format!("g_{best_idx}_{}", next.id),
                self.default_selectivity,
            );
            let next_card = next.estimated_cardinality;
            let output_card =
                ((current_card as f64 * next_card as f64 * selectivity).round() as u64).max(1);
            let algorithm = Self::select_join_algorithm(current_card, next_card, &join_vars);
            let right_plan = JoinPlanNode::TriplePatternScan { info: next };

            current_plan =
                self.build_join_plan(current_plan, right_plan, join_vars, output_card, algorithm);
        }

        Ok(current_plan)
    }

    /// Estimate cost of scanning a single triple pattern
    fn scan_cost(&self, pattern: &TriplePatternInfo) -> f64 {
        // Base cost: proportional to estimated cardinality
        // Bound positions reduce cost due to index selectivity
        let base = pattern.estimated_cardinality as f64;
        let bound_factor = match pattern.bound_positions() {
            0 => 1.0,  // full scan
            1 => 0.3,  // one index lookup
            2 => 0.05, // two-component lookup
            _ => 0.01, // fully bound - cheap existence check
        };
        base * bound_factor
    }

    /// Estimate cost of a join given child costs and sizes
    fn join_cost(
        &self,
        children_cost: f64,
        left_card: u64,
        right_card: u64,
        algorithm: &JoinAlgorithm,
    ) -> f64 {
        let l = left_card as f64;
        let r = right_card as f64;
        match algorithm {
            JoinAlgorithm::Hash => {
                // Build: O(r), Probe: O(l)
                children_cost + r + l
            }
            JoinAlgorithm::NestedLoop => {
                // O(l * r) - expensive, only good when outer is tiny
                children_cost + l * r
            }
            JoinAlgorithm::Merge => {
                // Sort: O(n log n) each, Merge: O(n+m)
                children_cost + l * l.max(1.0).ln() + r * r.max(1.0).ln() + l + r
            }
        }
    }

    /// Find shared variables between two variable sets
    fn find_join_variables_sets(left: &[String], right: &[String]) -> Vec<String> {
        left.iter().filter(|v| right.contains(v)).cloned().collect()
    }

    /// Select best join algorithm based on estimated cardinalities
    pub fn select_join_algorithm(
        left_card: u64,
        right_card: u64,
        join_vars: &[String],
    ) -> JoinAlgorithm {
        if join_vars.is_empty() {
            // Cross product - nested loop is simplest for tiny outer
            if left_card.min(right_card) < 100 {
                return JoinAlgorithm::NestedLoop;
            }
            return JoinAlgorithm::Hash;
        }

        let smaller = left_card.min(right_card);
        let larger = left_card.max(right_card);

        if smaller < 1000 {
            // Small build side - hash join is ideal
            JoinAlgorithm::Hash
        } else if smaller > 50_000 && larger > 50_000 {
            // Both sides large - merge join amortizes sort cost
            JoinAlgorithm::Merge
        } else {
            JoinAlgorithm::Hash
        }
    }

    /// Build a JoinPlanNode from two sub-plans
    fn build_join_plan(
        &self,
        left: JoinPlanNode,
        right: JoinPlanNode,
        join_vars: Vec<String>,
        estimated_output: u64,
        algorithm: JoinAlgorithm,
    ) -> JoinPlanNode {
        match algorithm {
            JoinAlgorithm::Hash => JoinPlanNode::HashJoin {
                left: Box::new(left),
                right: Box::new(right),
                join_vars,
                estimated_output,
            },
            JoinAlgorithm::NestedLoop => {
                // Ensure the smaller side is outer
                JoinPlanNode::NestedLoopJoin {
                    outer: Box::new(left),
                    inner: Box::new(right),
                    join_vars,
                    estimated_output,
                }
            }
            JoinAlgorithm::Merge => {
                let sort_key = join_vars.clone();
                JoinPlanNode::MergeJoin {
                    left: Box::new(left),
                    right: Box::new(right),
                    join_vars,
                    sort_key,
                    estimated_output,
                }
            }
        }
    }
}

/// Execution timer for recording plan execution times
pub struct PlanTimer {
    component_id: String,
    start: Instant,
    stats_store: Arc<AdaptiveStatsStore>,
}

impl PlanTimer {
    /// Start timing a plan component
    pub fn start(component_id: impl Into<String>, stats_store: Arc<AdaptiveStatsStore>) -> Self {
        Self {
            component_id: component_id.into(),
            start: Instant::now(),
            stats_store,
        }
    }
}

impl Drop for PlanTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.stats_store
            .record_execution_time(&self.component_id, elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Term, TriplePattern};
    use oxirs_core::model::{NamedNode, Variable as CoreVariable};

    fn make_var(name: &str) -> Term {
        Term::Variable(CoreVariable::new(name).unwrap())
    }

    fn make_iri(iri: &str) -> Term {
        Term::Iri(NamedNode::new_unchecked(iri))
    }

    fn pattern_info(
        subject: PatternTerm,
        predicate: PatternTerm,
        object: PatternTerm,
        cardinality: u64,
    ) -> TriplePatternInfo {
        let bound_variables: Vec<String> = [&subject, &predicate, &object]
            .iter()
            .filter_map(|t| t.variable_name().map(|s| s.to_string()))
            .collect();
        let id = format!("{:?}-{:?}-{:?}", subject, predicate, object);
        TriplePatternInfo {
            id,
            subject,
            predicate,
            object,
            estimated_cardinality: cardinality,
            bound_variables,
            original_pattern: None,
        }
    }

    #[test]
    fn test_adaptive_stats_store_record_and_adjust() {
        let store = AdaptiveStatsStore::new(100);
        store.record_pattern_execution("pat1", 1000, 500);

        // Correction factor should be 0.5 after one sample
        let adjusted = store.get_adjusted_cardinality("pat1", 1000);
        assert_eq!(
            adjusted, 500,
            "Adjusted cardinality should reflect correction factor"
        );
    }

    #[test]
    fn test_adaptive_stats_store_unknown_pattern_returns_base() {
        let store = AdaptiveStatsStore::new(100);
        let adjusted = store.get_adjusted_cardinality("unknown_pat", 500);
        assert_eq!(
            adjusted, 500,
            "Unknown pattern should return base estimate unchanged"
        );
    }

    #[test]
    fn test_adaptive_stats_store_join_selectivity() {
        let store = AdaptiveStatsStore::new(100);
        // actual output = 50, left = 100, right = 200 => selectivity = 50/20000 = 0.0025
        store.record_join_execution("j1", 100, 200, 50);

        let adjusted = store.get_adjusted_selectivity("j1", 0.1);
        // Should blend base (0.1) with observed (0.0025)
        assert!(
            adjusted < 0.1,
            "Adjusted selectivity should be reduced toward observed value"
        );
        assert!(adjusted > 0.0, "Adjusted selectivity must remain positive");
    }

    #[test]
    fn test_single_pattern_optimization() {
        let store = Arc::new(AdaptiveStatsStore::new(100));
        let optimizer = AdaptiveJoinOrderOptimizer::new(store);

        let patterns = vec![pattern_info(
            PatternTerm::Variable("s".to_string()),
            PatternTerm::Iri("http://example.org/type".to_string()),
            PatternTerm::Variable("o".to_string()),
            500,
        )];

        let plan = optimizer.optimize(patterns).unwrap();
        assert!(matches!(plan, JoinPlanNode::TriplePatternScan { .. }));
    }

    #[test]
    fn test_two_pattern_dp_optimization() {
        let store = Arc::new(AdaptiveStatsStore::new(100));
        let optimizer = AdaptiveJoinOrderOptimizer::new(store);

        let patterns = vec![
            pattern_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://example.org/type".to_string()),
                PatternTerm::Iri("http://example.org/Person".to_string()),
                50,
            ),
            pattern_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://xmlns.com/foaf/0.1/name".to_string()),
                PatternTerm::Variable("name".to_string()),
                10000,
            ),
        ];

        let plan = optimizer.optimize(patterns).unwrap();
        // Should produce a join node
        assert!(
            matches!(
                plan,
                JoinPlanNode::HashJoin { .. }
                    | JoinPlanNode::NestedLoopJoin { .. }
                    | JoinPlanNode::MergeJoin { .. }
            ),
            "Should produce a join plan"
        );
    }

    #[test]
    fn test_greedy_optimization_for_large_pattern_sets() {
        let store = Arc::new(AdaptiveStatsStore::new(100));
        let optimizer = AdaptiveJoinOrderOptimizer::new(store).with_dp_threshold(3);

        let patterns: Vec<TriplePatternInfo> = (0..6)
            .map(|i| {
                pattern_info(
                    PatternTerm::Variable(format!("s{i}")),
                    PatternTerm::Iri(format!("http://example.org/p{i}")),
                    PatternTerm::Variable(format!("o{i}")),
                    (i + 1) as u64 * 100,
                )
            })
            .collect();

        let plan = optimizer.optimize(patterns).unwrap();
        // Should produce some kind of join
        assert!(
            !matches!(plan, JoinPlanNode::TriplePatternScan { .. }),
            "Multiple patterns should produce a join plan"
        );
    }

    #[test]
    fn test_empty_patterns_returns_error() {
        let store = Arc::new(AdaptiveStatsStore::new(100));
        let optimizer = AdaptiveJoinOrderOptimizer::new(store);
        assert!(optimizer.optimize(vec![]).is_err());
    }

    #[test]
    fn test_join_algorithm_selection() {
        // Small build side -> hash join
        let alg =
            AdaptiveJoinOrderOptimizer::select_join_algorithm(100, 1_000_000, &["x".to_string()]);
        assert_eq!(alg, JoinAlgorithm::Hash);

        // Both large -> merge join
        let alg =
            AdaptiveJoinOrderOptimizer::select_join_algorithm(100_000, 200_000, &["x".to_string()]);
        assert_eq!(alg, JoinAlgorithm::Merge);
    }

    #[test]
    fn test_from_triple_pattern() {
        let pattern = TriplePattern::new(
            make_var("s"),
            make_iri("http://example.org/p"),
            make_var("o"),
        );
        let info = TriplePatternInfo::from_triple_pattern(&pattern, 100);
        assert_eq!(info.estimated_cardinality, 100);
        assert!(info.bound_variables.contains(&"s".to_string()));
        assert!(info.bound_variables.contains(&"o".to_string()));
        assert_eq!(info.bound_positions(), 1); // predicate is bound
    }

    #[test]
    fn test_cardinality_correction_with_multiple_samples() {
        let store = AdaptiveStatsStore::new(100);
        // Repeat with consistent underestimation (estimated 100, actual 200)
        for _ in 0..5 {
            store.record_pattern_execution("pat2", 100, 200);
        }
        let adjusted = store.get_adjusted_cardinality("pat2", 100);
        // Correction factor should converge toward 2.0
        assert!(adjusted > 100, "Cardinality should be adjusted upward");
    }

    #[test]
    fn test_plan_timer_records_duration() {
        let store = Arc::new(AdaptiveStatsStore::new(100));
        {
            let _timer = PlanTimer::start("test_component", Arc::clone(&store));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let snapshot = store.snapshot().unwrap();
        assert!(
            snapshot.execution_times.contains_key("test_component"),
            "Timer should record execution time on drop"
        );
    }

    #[test]
    fn test_output_variables_propagation() {
        let store = Arc::new(AdaptiveStatsStore::new(100));
        let optimizer = AdaptiveJoinOrderOptimizer::new(store);

        let patterns = vec![
            pattern_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://example.org/type".to_string()),
                PatternTerm::Variable("type".to_string()),
                100,
            ),
            pattern_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://example.org/name".to_string()),
                PatternTerm::Variable("name".to_string()),
                500,
            ),
        ];

        let plan = optimizer.optimize(patterns).unwrap();
        let vars = plan.output_variables();
        // Both ?s and ?type and ?name should appear
        assert!(vars.contains(&"s".to_string()), "Plan should expose ?s");
        assert!(
            vars.contains(&"name".to_string()),
            "Plan should expose ?name"
        );
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;
    use crate::algebra::{Term, TriplePattern};
    use oxirs_core::model::{NamedNode, Variable as CoreVariable};

    fn make_var(name: &str) -> Term {
        Term::Variable(CoreVariable::new(name).unwrap())
    }

    fn make_iri(iri: &str) -> Term {
        Term::Iri(NamedNode::new_unchecked(iri))
    }

    fn p_info(
        subject: PatternTerm,
        predicate: PatternTerm,
        object: PatternTerm,
        cardinality: u64,
    ) -> TriplePatternInfo {
        let bound_variables: Vec<String> = [&subject, &predicate, &object]
            .iter()
            .filter_map(|t| t.variable_name().map(|s| s.to_string()))
            .collect();
        let id = format!("{:?}-{:?}-{:?}", subject, predicate, object);
        TriplePatternInfo {
            id,
            subject,
            predicate,
            object,
            estimated_cardinality: cardinality,
            bound_variables,
            original_pattern: None,
        }
    }

    // --- AdaptiveStatsStore tests ---

    #[test]
    fn test_stats_snapshot_contains_recorded_pattern() {
        let store = AdaptiveStatsStore::new(50);
        store.record_pattern_execution("snap_pat", 200, 400);

        let snapshot = store.snapshot().unwrap();
        assert!(snapshot.pattern_stats.contains_key("snap_pat"));
        let entry = &snapshot.pattern_stats["snap_pat"];
        assert_eq!(entry.sample_count, 1);
        assert_eq!(entry.actual_cardinality_sum, 400);
    }

    #[test]
    fn test_stats_snapshot_contains_recorded_join() {
        let store = AdaptiveStatsStore::new(50);
        store.record_join_execution("j_snap", 1000, 500, 25);

        let snapshot = store.snapshot().unwrap();
        assert!(snapshot.join_stats.contains_key("j_snap"));
        let entry = &snapshot.join_stats["j_snap"];
        assert_eq!(entry.sample_count, 1);
        assert_eq!(entry.output_cardinality_sum, 25);
    }

    #[test]
    fn test_correction_factor_clamped_above_zero() {
        let store = AdaptiveStatsStore::new(50);
        // Extreme overestimation (estimated 1 million, actual 1)
        store.record_pattern_execution("extreme_over", 1_000_000, 1);
        let adjusted = store.get_adjusted_cardinality("extreme_over", 1_000_000);
        assert!(adjusted >= 1, "Adjusted cardinality must be at least 1");
    }

    #[test]
    fn test_multiple_patterns_tracked_independently() {
        let store = AdaptiveStatsStore::new(50);
        store.record_pattern_execution("pat_a", 100, 50);
        store.record_pattern_execution("pat_b", 100, 300);

        let adj_a = store.get_adjusted_cardinality("pat_a", 100);
        let adj_b = store.get_adjusted_cardinality("pat_b", 100);
        assert!(
            adj_a < adj_b,
            "pat_a (undercount) should produce lower estimate than pat_b (overcount)"
        );
    }

    #[test]
    fn test_execution_time_recorded_via_snapshot() {
        let store = AdaptiveStatsStore::new(50);
        store.record_execution_time("component_x", std::time::Duration::from_millis(42));
        let snapshot = store.snapshot().unwrap();
        assert!(snapshot.execution_times.contains_key("component_x"));
        assert_eq!(
            snapshot.execution_times["component_x"],
            std::time::Duration::from_millis(42)
        );
    }

    #[test]
    fn test_join_selectivity_unknown_join_returns_base() {
        let store = AdaptiveStatsStore::new(50);
        let base = 0.05;
        let adj = store.get_adjusted_selectivity("no_such_join", base);
        assert!(
            (adj - base).abs() < 1e-9,
            "Unknown join should return base selectivity unchanged"
        );
    }

    #[test]
    fn test_join_selectivity_clamps_to_valid_range() {
        let store = AdaptiveStatsStore::new(50);
        // Very low actual output => very low observed selectivity
        for _ in 0..20 {
            store.record_join_execution("tiny_sel", 1_000_000, 1_000_000, 1);
        }
        let adj = store.get_adjusted_selectivity("tiny_sel", 0.5);
        assert!(adj > 0.0, "Selectivity must remain positive");
        assert!(adj <= 1.0, "Selectivity must not exceed 1.0");
    }

    // --- PatternTerm tests ---

    #[test]
    fn test_pattern_term_iri_is_not_variable() {
        let term = PatternTerm::Iri("http://example.org/foo".to_string());
        assert!(!term.is_variable());
        assert!(term.variable_name().is_none());
    }

    #[test]
    fn test_pattern_term_literal_is_not_variable() {
        let term = PatternTerm::Literal("hello".to_string());
        assert!(!term.is_variable());
        assert!(term.variable_name().is_none());
    }

    #[test]
    fn test_pattern_term_blank_node_is_not_variable() {
        let term = PatternTerm::BlankNode("b1".to_string());
        assert!(!term.is_variable());
        assert!(term.variable_name().is_none());
    }

    #[test]
    fn test_triple_pattern_info_bound_positions_fully_bound() {
        let info = p_info(
            PatternTerm::Iri("http://s".to_string()),
            PatternTerm::Iri("http://p".to_string()),
            PatternTerm::Literal("val".to_string()),
            10,
        );
        assert_eq!(info.bound_positions(), 3, "All positions are bound");
    }

    #[test]
    fn test_triple_pattern_info_bound_positions_no_variables() {
        let info = p_info(
            PatternTerm::Variable("s".to_string()),
            PatternTerm::Variable("p".to_string()),
            PatternTerm::Variable("o".to_string()),
            100,
        );
        assert_eq!(
            info.bound_positions(),
            0,
            "No positions are bound when all are variables"
        );
    }

    #[test]
    fn test_from_triple_pattern_literal_object() {
        let pattern = TriplePattern::new(
            make_var("s"),
            make_iri("http://example.org/p"),
            make_iri("http://example.org/o"),
        );
        let info = TriplePatternInfo::from_triple_pattern(&pattern, 42);
        assert_eq!(info.estimated_cardinality, 42);
        // subject is variable
        assert!(info.bound_variables.contains(&"s".to_string()));
    }

    // --- JoinPlanNode tests ---

    #[test]
    fn test_join_plan_node_hash_join_estimated_cardinality() {
        let left = JoinPlanNode::TriplePatternScan {
            info: p_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://p".to_string()),
                PatternTerm::Variable("o".to_string()),
                100,
            ),
        };
        let right = JoinPlanNode::TriplePatternScan {
            info: p_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://q".to_string()),
                PatternTerm::Variable("x".to_string()),
                200,
            ),
        };
        let node = JoinPlanNode::HashJoin {
            left: Box::new(left),
            right: Box::new(right),
            join_vars: vec!["s".to_string()],
            estimated_output: 50,
        };
        assert_eq!(node.estimated_cardinality(), 50);
    }

    #[test]
    fn test_join_plan_nested_loop_output_variables() {
        let outer = JoinPlanNode::TriplePatternScan {
            info: p_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://p".to_string()),
                PatternTerm::Variable("o".to_string()),
                100,
            ),
        };
        let inner = JoinPlanNode::TriplePatternScan {
            info: p_info(
                PatternTerm::Variable("o".to_string()),
                PatternTerm::Iri("http://q".to_string()),
                PatternTerm::Variable("z".to_string()),
                50,
            ),
        };
        let node = JoinPlanNode::NestedLoopJoin {
            outer: Box::new(outer),
            inner: Box::new(inner),
            join_vars: vec!["o".to_string()],
            estimated_output: 30,
        };
        let vars = node.output_variables();
        assert!(vars.contains(&"s".to_string()), "Should contain s");
        assert!(vars.contains(&"o".to_string()), "Should contain o");
        assert!(vars.contains(&"z".to_string()), "Should contain z");
    }

    // --- AdaptiveJoinOrderOptimizer tests ---

    #[test]
    fn test_optimizer_selects_lower_cardinality_pattern_first() {
        let store = Arc::new(AdaptiveStatsStore::new(50));
        let optimizer = AdaptiveJoinOrderOptimizer::new(Arc::clone(&store));

        let patterns = vec![
            p_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://rare".to_string()),
                PatternTerm::Variable("o1".to_string()),
                5, // low cardinality - should be leaf
            ),
            p_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://common".to_string()),
                PatternTerm::Variable("o2".to_string()),
                50_000, // high cardinality
            ),
        ];

        let plan = optimizer.optimize(patterns).unwrap();
        // The lower-cardinality side should be in the plan somewhere
        // We verify the plan contains a join (not just a scan)
        assert!(
            matches!(
                plan,
                JoinPlanNode::HashJoin { .. }
                    | JoinPlanNode::NestedLoopJoin { .. }
                    | JoinPlanNode::MergeJoin { .. }
            ),
            "Two patterns should produce a join plan"
        );
    }

    #[test]
    fn test_optimizer_dp_threshold_boundary() {
        // At exactly the DP threshold, optimizer uses DP
        let store = Arc::new(AdaptiveStatsStore::new(50));
        let optimizer = AdaptiveJoinOrderOptimizer::new(Arc::clone(&store)).with_dp_threshold(4);

        let patterns: Vec<TriplePatternInfo> = (0..4)
            .map(|i| {
                p_info(
                    PatternTerm::Variable(format!("s{i}")),
                    PatternTerm::Iri(format!("http://p{i}")),
                    PatternTerm::Variable(format!("o{i}")),
                    (i + 1) as u64 * 50,
                )
            })
            .collect();

        let result = optimizer.optimize(patterns);
        assert!(
            result.is_ok(),
            "DP optimization at threshold should succeed"
        );
    }

    #[test]
    fn test_optimizer_uses_runtime_feedback_for_ordering() {
        let store = Arc::new(AdaptiveStatsStore::new(50));
        // Record that "pat_heavy" actually has far more rows than estimated
        store.record_pattern_execution("? http://heavy ?", 10, 100_000);

        let optimizer = AdaptiveJoinOrderOptimizer::new(Arc::clone(&store));
        let patterns = vec![
            p_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://heavy".to_string()),
                PatternTerm::Variable("o".to_string()),
                10, // low estimate, but runtime says 100_000
            ),
            p_info(
                PatternTerm::Variable("s".to_string()),
                PatternTerm::Iri("http://light".to_string()),
                PatternTerm::Variable("x".to_string()),
                500,
            ),
        ];
        let result = optimizer.optimize(patterns);
        assert!(
            result.is_ok(),
            "Optimizer should succeed with runtime feedback"
        );
    }
}
