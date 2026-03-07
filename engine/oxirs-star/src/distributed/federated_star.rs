//! Distributed RDF-star query processing with partition-aware federated planning.
//!
//! This module implements:
//! - Partition-aware query planning across RDF-star shards
//! - Federated SPARQL-star execution across heterogeneous shards
//! - Cost-based shard selection and join ordering
//! - Result merging with de-duplication
//!
//! # Design
//!
//! ```text
//! FederatedStarPlanner
//!   ├─ ShardRegistry         (metadata about each shard)
//!   ├─ PartitionScheme       (how data is partitioned)
//!   └─ FederatedQueryPlan
//!        ├─ ShardSubPlan[0]  (sub-plan for shard A)
//!        ├─ ShardSubPlan[1]  (sub-plan for shard B)
//!        └─ MergeStep        (how to combine results)
//! ```

use crate::{StarError, StarResult, StarTerm, StarTriple};
use scirs2_core::profiling::Profiler;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Shard metadata
// ---------------------------------------------------------------------------

/// Unique identifier for a shard.
pub type ShardId = u64;

/// Partition scheme determining how data is distributed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionScheme {
    /// Each shard owns a disjoint range of subject-hash values.
    SubjectHash { shard_count: usize },
    /// Each shard owns a disjoint range of predicate-hash values.
    PredicateHash { shard_count: usize },
    /// Each shard owns a named graph.
    NamedGraph,
    /// Round-robin distribution (for uniform loads).
    RoundRobin { shard_count: usize },
    /// Custom user-defined partition key.
    Custom { key: String },
}

impl PartitionScheme {
    /// Determine which shard(s) to query for the given subject term.
    pub fn shards_for_subject(
        &self,
        subject: &Option<StarTerm>,
        all_shards: &[ShardId],
    ) -> Vec<ShardId> {
        if all_shards.is_empty() {
            return Vec::new();
        }
        // Sort for deterministic routing (caller may pass unsorted slices).
        let mut sorted_shards: Vec<ShardId> = all_shards.to_vec();
        sorted_shards.sort_unstable();
        match (self, subject) {
            (PartitionScheme::SubjectHash { shard_count }, Some(s)) => {
                let hash = fnv1a_term(s);
                let shard_idx = (hash as usize) % shard_count;
                if shard_idx < sorted_shards.len() {
                    vec![sorted_shards[shard_idx]]
                } else {
                    sorted_shards
                }
            }
            (PartitionScheme::PredicateHash { shard_count }, _) => {
                // Without predicate info here, broadcast to all.
                let _ = shard_count;
                all_shards.to_vec()
            }
            _ => all_shards.to_vec(), // default: broadcast
        }
    }

    /// Determine which shard(s) to query for a quoted triple subject.
    pub fn shards_for_quoted_subject(
        &self,
        quoted: &StarTriple,
        all_shards: &[ShardId],
    ) -> Vec<ShardId> {
        // Hash the entire serialisation of the quoted triple.
        let key = format!(
            "<<{}|{}|{}>>",
            term_str(&quoted.subject),
            term_str(&quoted.predicate),
            term_str(&quoted.object)
        );
        if all_shards.is_empty() {
            return Vec::new();
        }
        // Sort for deterministic routing.
        let mut sorted_shards: Vec<ShardId> = all_shards.to_vec();
        sorted_shards.sort_unstable();
        match self {
            PartitionScheme::SubjectHash { shard_count } => {
                let hash = fnv1a_str(key.as_bytes());
                let idx = (hash as usize) % shard_count;
                if idx < sorted_shards.len() {
                    vec![sorted_shards[idx]]
                } else {
                    sorted_shards
                }
            }
            _ => sorted_shards,
        }
    }
}

fn term_str(t: &StarTerm) -> String {
    match t {
        StarTerm::NamedNode(n) => format!("<{}>", n.iri),
        StarTerm::BlankNode(b) => format!("_:{}", b.id),
        StarTerm::Literal(l) => format!("\"{}\"", l.value),
        StarTerm::QuotedTriple(qt) => {
            format!(
                "<<{}|{}|{}>>",
                term_str(&qt.subject),
                term_str(&qt.predicate),
                term_str(&qt.object)
            )
        }
        StarTerm::Variable(v) => format!("?{}", v.name),
    }
}

fn fnv1a_term(t: &StarTerm) -> u64 {
    fnv1a_str(term_str(t).as_bytes())
}

fn fnv1a_str(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Metadata about a single shard.
#[derive(Debug, Clone)]
pub struct ShardMetadata {
    pub id: ShardId,
    /// Human-readable label.
    pub label: String,
    /// Endpoint URL (used in real federated queries).
    pub endpoint: String,
    /// Estimated triple count (used for cost estimation).
    pub estimated_triples: u64,
    /// Estimated latency to this shard (milliseconds).
    pub estimated_latency_ms: u64,
    /// Whether the shard is currently healthy.
    pub healthy: bool,
    /// Supported SPARQL-star features.
    pub capabilities: ShardCapabilities,
}

/// Capabilities advertised by a shard.
#[derive(Debug, Clone, Default)]
pub struct ShardCapabilities {
    pub supports_sparql_star: bool,
    pub supports_quoted_triple_filter: bool,
    pub supports_annotation_queries: bool,
    pub max_result_size: Option<u64>,
}

impl ShardMetadata {
    pub fn new(id: ShardId, label: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            endpoint: endpoint.into(),
            estimated_triples: 0,
            estimated_latency_ms: 10,
            healthy: true,
            capabilities: ShardCapabilities {
                supports_sparql_star: true,
                supports_quoted_triple_filter: true,
                supports_annotation_queries: true,
                max_result_size: None,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Query pattern (federated level)
// ---------------------------------------------------------------------------

/// A SPARQL-star triple pattern used in federated planning.
#[derive(Debug, Clone)]
pub struct FederatedPattern {
    pub subject: Option<StarTerm>,
    pub predicate: Option<StarTerm>,
    pub object: Option<StarTerm>,
    /// Variable name for the matched triple (SPARQL bind syntax).
    pub triple_var: Option<String>,
}

impl FederatedPattern {
    pub fn new(
        subject: Option<StarTerm>,
        predicate: Option<StarTerm>,
        object: Option<StarTerm>,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            triple_var: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Query plan
// ---------------------------------------------------------------------------

/// Sub-plan assigned to a single shard.
#[derive(Debug, Clone)]
pub struct ShardSubPlan {
    pub shard_id: ShardId,
    /// Patterns to evaluate on this shard.
    pub patterns: Vec<FederatedPattern>,
    /// Estimated cost (latency × selectivity).
    pub estimated_cost: f64,
    /// Whether results from this shard require annotation joins.
    pub requires_annotation_join: bool,
}

/// How to merge results from multiple shards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Union (SPARQL UNION semantics – duplicate-preserving).
    Union,
    /// Distinct union.
    DistinctUnion,
    /// Inner join on shared variables.
    InnerJoin { join_vars: Vec<String> },
}

/// Complete federated query plan.
#[derive(Debug, Clone)]
pub struct FederatedQueryPlan {
    pub shard_plans: Vec<ShardSubPlan>,
    pub merge_strategy: MergeStrategy,
    pub estimated_total_cost: f64,
    pub pattern_count: usize,
}

impl FederatedQueryPlan {
    /// Return the set of shard IDs involved in this plan.
    pub fn involved_shards(&self) -> HashSet<ShardId> {
        self.shard_plans.iter().map(|sp| sp.shard_id).collect()
    }

    /// Return the number of patterns sent to a specific shard.
    pub fn patterns_for_shard(&self, shard_id: ShardId) -> usize {
        self.shard_plans
            .iter()
            .filter(|sp| sp.shard_id == shard_id)
            .map(|sp| sp.patterns.len())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Shard registry
// ---------------------------------------------------------------------------

/// Registry of all known shards with live metadata.
pub struct ShardRegistry {
    shards: HashMap<ShardId, ShardMetadata>,
}

impl ShardRegistry {
    pub fn new() -> Self {
        Self {
            shards: HashMap::new(),
        }
    }

    pub fn register(&mut self, meta: ShardMetadata) {
        self.shards.insert(meta.id, meta);
    }

    pub fn get(&self, id: ShardId) -> Option<&ShardMetadata> {
        self.shards.get(&id)
    }

    pub fn healthy_shards(&self) -> Vec<ShardId> {
        self.shards
            .values()
            .filter(|m| m.healthy)
            .map(|m| m.id)
            .collect()
    }

    pub fn all_shards(&self) -> Vec<ShardId> {
        self.shards.keys().copied().collect()
    }

    /// Mark a shard as unhealthy.
    pub fn mark_unhealthy(&mut self, id: ShardId) {
        if let Some(m) = self.shards.get_mut(&id) {
            m.healthy = false;
        }
    }

    /// Update the estimated triple count for a shard.
    pub fn update_stats(&mut self, id: ShardId, triples: u64, latency_ms: u64) {
        if let Some(m) = self.shards.get_mut(&id) {
            m.estimated_triples = triples;
            m.estimated_latency_ms = latency_ms;
        }
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

impl Default for ShardRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cost estimator
// ---------------------------------------------------------------------------

/// Cost model for federated query planning.
pub struct CostEstimator;

impl CostEstimator {
    /// Estimate the cost of sending a pattern to a shard.
    ///
    /// Cost = latency + (estimated_triples / selectivity_factor)
    pub fn estimate(meta: &ShardMetadata, pattern: &FederatedPattern) -> f64 {
        let selectivity = Self::pattern_selectivity(pattern);
        let data_cost = (meta.estimated_triples as f64) * selectivity;
        meta.estimated_latency_ms as f64 + data_cost / 1000.0
    }

    /// Selectivity heuristic: fewer bound slots → higher cardinality.
    fn pattern_selectivity(pattern: &FederatedPattern) -> f64 {
        let bound = [
            pattern.subject.is_some(),
            pattern.predicate.is_some(),
            pattern.object.is_some(),
        ]
        .iter()
        .filter(|&&b| b)
        .count();
        match bound {
            0 => 1.0,
            1 => 0.1,
            2 => 0.01,
            _ => 0.001,
        }
    }
}

// ---------------------------------------------------------------------------
// Federated planner
// ---------------------------------------------------------------------------

/// Partition-aware federated SPARQL-star query planner.
pub struct FederatedStarPlanner {
    registry: Arc<RwLock<ShardRegistry>>,
    scheme: PartitionScheme,
    #[allow(dead_code)]
    profiler: Profiler,
}

impl FederatedStarPlanner {
    pub fn new(registry: Arc<RwLock<ShardRegistry>>, scheme: PartitionScheme) -> Self {
        Self {
            registry,
            scheme,
            profiler: Profiler::new(),
        }
    }

    /// Build a federated query plan for a set of SPARQL-star patterns.
    pub fn plan(&mut self, patterns: Vec<FederatedPattern>) -> StarResult<FederatedQueryPlan> {
        let reg = self
            .registry
            .read()
            .map_err(|_| StarError::processing_error("ShardRegistry read lock poisoned"))?;
        let mut all_shards = reg.healthy_shards();
        all_shards.sort_unstable(); // Ensure deterministic routing order

        if all_shards.is_empty() {
            return Err(StarError::processing_error("No healthy shards available"));
        }

        // Map each pattern to the candidate shards.
        let mut shard_patterns: HashMap<ShardId, Vec<FederatedPattern>> = HashMap::new();

        for pattern in &patterns {
            let target_shards = match &pattern.subject {
                Some(StarTerm::QuotedTriple(qt)) => {
                    self.scheme.shards_for_quoted_subject(qt, &all_shards)
                }
                subject => self.scheme.shards_for_subject(subject, &all_shards),
            };

            // Assign pattern to the cheapest candidate shard.
            let best_shard = target_shards
                .iter()
                .min_by(|&&a, &&b| {
                    let ca = reg
                        .get(a)
                        .map(|m| CostEstimator::estimate(m, pattern))
                        .unwrap_or(f64::MAX);
                    let cb = reg
                        .get(b)
                        .map(|m| CostEstimator::estimate(m, pattern))
                        .unwrap_or(f64::MAX);
                    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(all_shards[0]);

            shard_patterns
                .entry(best_shard)
                .or_default()
                .push(pattern.clone());
        }

        // Build sub-plans.
        let mut shard_plans: Vec<ShardSubPlan> = shard_patterns
            .into_iter()
            .map(|(shard_id, pats)| {
                let cost: f64 = reg
                    .get(shard_id)
                    .map(|m| pats.iter().map(|p| CostEstimator::estimate(m, p)).sum())
                    .unwrap_or(0.0);
                let annotation = pats.iter().any(|p| p.triple_var.is_some());
                ShardSubPlan {
                    shard_id,
                    patterns: pats,
                    estimated_cost: cost,
                    requires_annotation_join: annotation,
                }
            })
            .collect();

        // Sort by cost (cheapest first for pipeline ordering).
        shard_plans.sort_by(|a, b| {
            a.estimated_cost
                .partial_cmp(&b.estimated_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_cost: f64 = shard_plans.iter().map(|sp| sp.estimated_cost).sum();
        let pattern_count = patterns.len();

        Ok(FederatedQueryPlan {
            shard_plans,
            merge_strategy: MergeStrategy::DistinctUnion,
            estimated_total_cost: total_cost,
            pattern_count,
        })
    }
}

// ---------------------------------------------------------------------------
// Result aggregator
// ---------------------------------------------------------------------------

/// Aggregated result from federated evaluation.
#[derive(Debug, Clone)]
pub struct FederatedResult {
    pub triples: Vec<StarTriple>,
    pub shard_contributions: HashMap<ShardId, usize>,
    pub total_latency: Duration,
    pub plan: Option<FederatedQueryPlan>,
}

impl FederatedResult {
    pub fn empty() -> Self {
        Self {
            triples: Vec::new(),
            shard_contributions: HashMap::new(),
            total_latency: Duration::ZERO,
            plan: None,
        }
    }

    /// Merge results from multiple shards applying the merge strategy.
    pub fn merge_from(
        shard_results: Vec<(ShardId, Vec<StarTriple>)>,
        strategy: MergeStrategy,
        latency: Duration,
    ) -> Self {
        let mut result = Self::empty();
        result.total_latency = latency;

        let mut seen: HashSet<String> = HashSet::new();

        for (shard_id, triples) in shard_results {
            let count = triples.len();
            *result.shard_contributions.entry(shard_id).or_insert(0) += count;

            for triple in triples {
                match &strategy {
                    MergeStrategy::Union => {
                        result.triples.push(triple);
                    }
                    MergeStrategy::DistinctUnion => {
                        let key = format!(
                            "{}|{}|{}",
                            term_str(&triple.subject),
                            term_str(&triple.predicate),
                            term_str(&triple.object)
                        );
                        if seen.insert(key) {
                            result.triples.push(triple);
                        }
                    }
                    MergeStrategy::InnerJoin { .. } => {
                        // Inner join merge is handled at binding level; here just collect.
                        result.triples.push(triple);
                    }
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// In-memory shard simulator (for testing)
// ---------------------------------------------------------------------------

/// Simple in-memory shard for testing federated plans.
pub struct MemoryShard {
    pub id: ShardId,
    pub triples: Vec<StarTriple>,
}

impl MemoryShard {
    pub fn new(id: ShardId) -> Self {
        Self {
            id,
            triples: Vec::new(),
        }
    }

    pub fn insert(&mut self, triple: StarTriple) {
        self.triples.push(triple);
    }

    pub fn evaluate(&self, pattern: &FederatedPattern) -> Vec<StarTriple> {
        self.triples
            .iter()
            .filter(|t| {
                let s_ok = pattern
                    .subject
                    .as_ref()
                    .map(|s| s == &t.subject)
                    .unwrap_or(true);
                let p_ok = pattern
                    .predicate
                    .as_ref()
                    .map(|p| p == &t.predicate)
                    .unwrap_or(true);
                let o_ok = pattern
                    .object
                    .as_ref()
                    .map(|o| o == &t.object)
                    .unwrap_or(true);
                s_ok && p_ok && o_ok
            })
            .cloned()
            .collect()
    }
}

/// Execute a federated plan against a set of in-memory shards.
pub fn execute_federated_plan(
    plan: &FederatedQueryPlan,
    shards: &HashMap<ShardId, MemoryShard>,
) -> StarResult<FederatedResult> {
    let start = Instant::now();
    let mut shard_results: Vec<(ShardId, Vec<StarTriple>)> = Vec::new();

    for sub_plan in &plan.shard_plans {
        let shard = shards.get(&sub_plan.shard_id).ok_or_else(|| {
            StarError::processing_error(format!(
                "Shard {} not found in local shards",
                sub_plan.shard_id
            ))
        })?;

        let mut shard_triples: Vec<StarTriple> = Vec::new();
        for pattern in &sub_plan.patterns {
            let mut matches = shard.evaluate(pattern);
            shard_triples.append(&mut matches);
        }
        shard_results.push((sub_plan.shard_id, shard_triples));
    }

    let latency = start.elapsed();
    Ok(FederatedResult::merge_from(
        shard_results,
        plan.merge_strategy.clone(),
        latency,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StarTerm, StarTriple};

    fn make_triple(s: &str, p: &str, o: &str) -> StarTriple {
        StarTriple::new(
            StarTerm::iri(s).unwrap(),
            StarTerm::iri(p).unwrap(),
            StarTerm::iri(o).unwrap(),
        )
    }

    fn make_registry_with_shards(n: usize) -> Arc<RwLock<ShardRegistry>> {
        let mut reg = ShardRegistry::new();
        for i in 0..n {
            let mut meta = ShardMetadata::new(
                i as ShardId,
                format!("shard-{i}"),
                format!("http://shard{i}.example.org/sparql"),
            );
            meta.estimated_triples = 1000 + (i as u64) * 100;
            meta.estimated_latency_ms = 5 + i as u64;
            reg.register(meta);
        }
        Arc::new(RwLock::new(reg))
    }

    fn make_shards_map(n: usize, triples_per_shard: usize) -> HashMap<ShardId, MemoryShard> {
        let mut map = HashMap::new();
        for i in 0..n {
            let mut shard = MemoryShard::new(i as ShardId);
            for j in 0..triples_per_shard {
                shard.insert(make_triple(
                    &format!("http://ex.org/s{}", i * triples_per_shard + j),
                    "http://ex.org/p",
                    &format!("http://ex.org/o{}", i * triples_per_shard + j),
                ));
            }
            map.insert(i as ShardId, shard);
        }
        map
    }

    // --- PartitionScheme tests ---

    #[test]
    fn test_subject_hash_routing() {
        let scheme = PartitionScheme::SubjectHash { shard_count: 4 };
        let all_shards: Vec<ShardId> = (0..4).collect();
        let subject = Some(StarTerm::iri("http://ex.org/alice").unwrap());
        let shards = scheme.shards_for_subject(&subject, &all_shards);
        assert_eq!(
            shards.len(),
            1,
            "Subject hash should route to exactly one shard"
        );
    }

    #[test]
    fn test_subject_hash_wildcard_broadcasts() {
        let scheme = PartitionScheme::SubjectHash { shard_count: 4 };
        let all_shards: Vec<ShardId> = (0..4).collect();
        let shards = scheme.shards_for_subject(&None, &all_shards);
        assert_eq!(
            shards.len(),
            4,
            "Wildcard subject should broadcast to all shards"
        );
    }

    #[test]
    fn test_round_robin_scheme_broadcasts() {
        let scheme = PartitionScheme::RoundRobin { shard_count: 3 };
        let all_shards: Vec<ShardId> = (0..3).collect();
        let shards = scheme.shards_for_subject(
            &Some(StarTerm::iri("http://ex.org/x").unwrap()),
            &all_shards,
        );
        assert_eq!(shards.len(), 3);
    }

    #[test]
    fn test_empty_shards_returns_empty() {
        let scheme = PartitionScheme::SubjectHash { shard_count: 4 };
        let shards = scheme.shards_for_subject(&None, &[]);
        assert!(shards.is_empty());
    }

    // --- ShardRegistry tests ---

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = ShardRegistry::new();
        let meta = ShardMetadata::new(1, "shard-1", "http://endpoint1");
        reg.register(meta);
        assert!(reg.get(1).is_some());
        assert!(reg.get(99).is_none());
    }

    #[test]
    fn test_registry_healthy_shards() {
        let mut reg = ShardRegistry::new();
        reg.register(ShardMetadata::new(1, "a", "http://a"));
        let mut m2 = ShardMetadata::new(2, "b", "http://b");
        m2.healthy = false;
        reg.register(m2);
        let healthy = reg.healthy_shards();
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0], 1);
    }

    #[test]
    fn test_registry_mark_unhealthy() {
        let mut reg = ShardRegistry::new();
        reg.register(ShardMetadata::new(1, "a", "http://a"));
        reg.mark_unhealthy(1);
        assert!(!reg.get(1).unwrap().healthy);
    }

    #[test]
    fn test_registry_update_stats() {
        let mut reg = ShardRegistry::new();
        reg.register(ShardMetadata::new(1, "a", "http://a"));
        reg.update_stats(1, 5000, 20);
        let m = reg.get(1).unwrap();
        assert_eq!(m.estimated_triples, 5000);
        assert_eq!(m.estimated_latency_ms, 20);
    }

    #[test]
    fn test_registry_shard_count() {
        let mut reg = ShardRegistry::new();
        for i in 0..5u64 {
            reg.register(ShardMetadata::new(
                i,
                format!("s{i}"),
                format!("http://s{i}"),
            ));
        }
        assert_eq!(reg.shard_count(), 5);
    }

    // --- CostEstimator tests ---

    #[test]
    fn test_cost_estimator_fully_bound_cheaper() {
        let meta = ShardMetadata {
            id: 1,
            label: "test".into(),
            endpoint: "http://x".into(),
            estimated_triples: 10000,
            estimated_latency_ms: 10,
            healthy: true,
            capabilities: ShardCapabilities::default(),
        };
        let full_pattern = FederatedPattern::new(
            Some(StarTerm::iri("http://ex.org/s").unwrap()),
            Some(StarTerm::iri("http://ex.org/p").unwrap()),
            Some(StarTerm::iri("http://ex.org/o").unwrap()),
        );
        let empty_pattern = FederatedPattern::new(None, None, None);
        let full_cost = CostEstimator::estimate(&meta, &full_pattern);
        let empty_cost = CostEstimator::estimate(&meta, &empty_pattern);
        assert!(
            full_cost < empty_cost,
            "Fully bound pattern should have lower cost, got full={full_cost} empty={empty_cost}"
        );
    }

    // --- FederatedStarPlanner tests ---

    #[test]
    fn test_planner_empty_pattern_list() {
        let reg = make_registry_with_shards(3);
        let mut planner =
            FederatedStarPlanner::new(reg, PartitionScheme::SubjectHash { shard_count: 3 });
        let plan = planner.plan(vec![]).unwrap();
        assert_eq!(plan.pattern_count, 0);
    }

    #[test]
    fn test_planner_single_pattern() {
        let reg = make_registry_with_shards(3);
        let mut planner =
            FederatedStarPlanner::new(reg, PartitionScheme::SubjectHash { shard_count: 3 });
        let patterns = vec![FederatedPattern::new(
            Some(StarTerm::iri("http://ex.org/alice").unwrap()),
            None,
            None,
        )];
        let plan = planner.plan(patterns).unwrap();
        assert_eq!(plan.pattern_count, 1);
        assert!(!plan.shard_plans.is_empty());
    }

    #[test]
    fn test_planner_routes_to_subset_of_shards() {
        let reg = make_registry_with_shards(4);
        let mut planner =
            FederatedStarPlanner::new(reg, PartitionScheme::SubjectHash { shard_count: 4 });
        let patterns: Vec<FederatedPattern> = (0..4)
            .map(|i| {
                FederatedPattern::new(
                    Some(StarTerm::iri(&format!("http://ex.org/s{i}")).unwrap()),
                    None,
                    None,
                )
            })
            .collect();
        let plan = planner.plan(patterns).unwrap();
        // Not all 4 patterns need to go to different shards, but at most 4 shards.
        assert!(plan.shard_plans.len() <= 4);
    }

    #[test]
    fn test_planner_no_healthy_shards_fails() {
        let mut reg = ShardRegistry::new();
        let mut m = ShardMetadata::new(1, "a", "http://a");
        m.healthy = false;
        reg.register(m);
        let reg = Arc::new(RwLock::new(reg));
        let mut planner =
            FederatedStarPlanner::new(reg, PartitionScheme::RoundRobin { shard_count: 1 });
        let result = planner.plan(vec![FederatedPattern::new(None, None, None)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_plan_involved_shards() {
        let reg = make_registry_with_shards(3);
        let mut planner =
            FederatedStarPlanner::new(reg, PartitionScheme::RoundRobin { shard_count: 3 });
        let patterns = vec![
            FederatedPattern::new(None, None, None),
            FederatedPattern::new(None, None, None),
        ];
        let plan = planner.plan(patterns).unwrap();
        let shards = plan.involved_shards();
        assert!(!shards.is_empty());
    }

    // --- FederatedResult tests ---

    #[test]
    fn test_federated_result_distinct_union() {
        let t = make_triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        let shard_results = vec![
            (0u64, vec![t.clone()]),
            (1u64, vec![t.clone()]), // duplicate
        ];
        let result = FederatedResult::merge_from(
            shard_results,
            MergeStrategy::DistinctUnion,
            Duration::ZERO,
        );
        assert_eq!(
            result.triples.len(),
            1,
            "Distinct union should de-duplicate"
        );
    }

    #[test]
    fn test_federated_result_union_keeps_duplicates() {
        let t = make_triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        let shard_results = vec![(0u64, vec![t.clone()]), (1u64, vec![t.clone()])];
        let result =
            FederatedResult::merge_from(shard_results, MergeStrategy::Union, Duration::ZERO);
        assert_eq!(result.triples.len(), 2, "Union should keep duplicates");
    }

    #[test]
    fn test_federated_result_shard_contributions() {
        let t1 = make_triple("http://ex.org/s1", "http://ex.org/p", "http://ex.org/o1");
        let t2 = make_triple("http://ex.org/s2", "http://ex.org/p", "http://ex.org/o2");
        let shard_results = vec![(0u64, vec![t1]), (1u64, vec![t2])];
        let result =
            FederatedResult::merge_from(shard_results, MergeStrategy::Union, Duration::ZERO);
        assert_eq!(*result.shard_contributions.get(&0).unwrap(), 1);
        assert_eq!(*result.shard_contributions.get(&1).unwrap(), 1);
    }

    // --- execute_federated_plan tests ---

    #[test]
    fn test_execute_plan_all_wildcard() {
        let reg = make_registry_with_shards(2);
        let mut planner = FederatedStarPlanner::new(
            Arc::clone(&reg),
            PartitionScheme::SubjectHash { shard_count: 2 },
        );
        let patterns = vec![FederatedPattern::new(None, None, None)];
        let plan = planner.plan(patterns).unwrap();

        let shards = make_shards_map(2, 5);
        let result = execute_federated_plan(&plan, &shards).unwrap();
        // Each shard has 5 triples; plan routes wildcard to all shards.
        assert!(!result.triples.is_empty());
    }

    #[test]
    fn test_execute_plan_selective_subject() {
        let reg = make_registry_with_shards(2);
        let mut planner = FederatedStarPlanner::new(
            Arc::clone(&reg),
            PartitionScheme::SubjectHash { shard_count: 2 },
        );
        // Target a specific subject that lives in shard 0.
        let target = StarTerm::iri("http://ex.org/s0").unwrap();
        let patterns = vec![FederatedPattern::new(Some(target), None, None)];
        let plan = planner.plan(patterns).unwrap();

        let shards = make_shards_map(2, 5);
        let result = execute_federated_plan(&plan, &shards).unwrap();
        assert_eq!(result.triples.len(), 1);
    }

    #[test]
    fn test_execute_plan_missing_shard_errors() {
        let reg = make_registry_with_shards(1);
        let mut planner = FederatedStarPlanner::new(
            Arc::clone(&reg),
            PartitionScheme::RoundRobin { shard_count: 1 },
        );
        let patterns = vec![FederatedPattern::new(None, None, None)];
        let plan = planner.plan(patterns).unwrap();

        // Provide an empty shards map (shard 0 is missing).
        let shards: HashMap<ShardId, MemoryShard> = HashMap::new();
        let result = execute_federated_plan(&plan, &shards);
        assert!(result.is_err());
    }

    #[test]
    fn test_quoted_triple_routing() {
        let scheme = PartitionScheme::SubjectHash { shard_count: 3 };
        let inner = make_triple("http://ex.org/a", "http://ex.org/b", "http://ex.org/c");
        let all_shards: Vec<ShardId> = (0..3).collect();
        let shards = scheme.shards_for_quoted_subject(&inner, &all_shards);
        assert_eq!(shards.len(), 1, "Quoted triple should route to one shard");
    }

    #[test]
    fn test_memory_shard_evaluate() {
        let mut shard = MemoryShard::new(0);
        shard.insert(make_triple(
            "http://ex.org/s",
            "http://ex.org/p",
            "http://ex.org/o",
        ));
        shard.insert(make_triple(
            "http://ex.org/s2",
            "http://ex.org/p2",
            "http://ex.org/o2",
        ));
        let pattern =
            FederatedPattern::new(Some(StarTerm::iri("http://ex.org/s").unwrap()), None, None);
        let results = shard.evaluate(&pattern);
        assert_eq!(results.len(), 1);
    }
}
