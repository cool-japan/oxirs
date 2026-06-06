//! # Rule Index Types
//!
//! Data structures, key types, and configuration for rule index lookup.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration for rule indexing behavior
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Enable indexing by predicate (recommended)
    pub predicate_indexing: bool,
    /// Enable indexing by first argument
    pub first_arg_indexing: bool,
    /// Enable indexing by predicate + first arg combination
    pub combined_indexing: bool,
    /// Maximum number of rules before switching to hash indexing
    pub hash_threshold: usize,
    /// Enable statistics collection
    pub collect_statistics: bool,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            predicate_indexing: true,
            first_arg_indexing: true,
            combined_indexing: true,
            hash_threshold: 10,
            collect_statistics: true,
        }
    }
}

impl IndexConfig {
    /// Create a new index configuration with predicate indexing
    pub fn with_predicate_indexing(mut self, enabled: bool) -> Self {
        self.predicate_indexing = enabled;
        self
    }

    /// Create a new index configuration with first-arg indexing
    pub fn with_first_arg_indexing(mut self, enabled: bool) -> Self {
        self.first_arg_indexing = enabled;
        self
    }

    /// Create a new index configuration with combined indexing
    pub fn with_combined_indexing(mut self, enabled: bool) -> Self {
        self.combined_indexing = enabled;
        self
    }

    /// Set hash threshold for index type selection
    pub fn with_hash_threshold(mut self, threshold: usize) -> Self {
        self.hash_threshold = threshold;
        self
    }

    /// Enable or disable statistics collection
    pub fn with_statistics(mut self, enabled: bool) -> Self {
        self.collect_statistics = enabled;
        self
    }
}

/// Index key for predicate-based lookups
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PredicateKey(pub String);

/// Index key for first-argument lookups
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FirstArgKey {
    pub predicate: String,
    pub first_arg: Option<String>,
}

/// Combined key for predicate + subject + object patterns
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CombinedKey {
    pub predicate: String,
    pub subject_type: ArgType,
    pub object_type: ArgType,
}

/// Type of argument in a triple pattern
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ArgType {
    /// Variable argument (matches anything)
    Variable,
    /// Constant argument with specific value
    Constant(String),
    /// Any type (for wildcard matching)
    Any,
}

/// Statistics for index performance tracking
#[derive(Debug, Default)]
pub struct IndexStatistics {
    /// Total number of lookups performed
    pub total_lookups: AtomicU64,
    /// Number of lookups that hit the predicate index
    pub predicate_hits: AtomicU64,
    /// Number of lookups that hit the first-arg index
    pub first_arg_hits: AtomicU64,
    /// Number of lookups that hit the combined index
    pub combined_hits: AtomicU64,
    /// Number of lookups that fell back to full scan
    pub full_scans: AtomicU64,
    /// Total rules checked during lookups
    pub rules_checked: AtomicU64,
    /// Total rules matched during lookups
    pub rules_matched: AtomicU64,
}

impl IndexStatistics {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Get hit rate for predicate index
    pub fn predicate_hit_rate(&self) -> f64 {
        let total = self.total_lookups.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.predicate_hits.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get hit rate for first-arg index
    pub fn first_arg_hit_rate(&self) -> f64 {
        let total = self.total_lookups.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.first_arg_hits.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get combined hit rate
    pub fn combined_hit_rate(&self) -> f64 {
        let total = self.total_lookups.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.combined_hits.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get selectivity (matched / checked)
    pub fn selectivity(&self) -> f64 {
        let checked = self.rules_checked.load(Ordering::Relaxed);
        if checked == 0 {
            return 0.0;
        }
        self.rules_matched.load(Ordering::Relaxed) as f64 / checked as f64
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.total_lookups.store(0, Ordering::Relaxed);
        self.predicate_hits.store(0, Ordering::Relaxed);
        self.first_arg_hits.store(0, Ordering::Relaxed);
        self.combined_hits.store(0, Ordering::Relaxed);
        self.full_scans.store(0, Ordering::Relaxed);
        self.rules_checked.store(0, Ordering::Relaxed);
        self.rules_matched.store(0, Ordering::Relaxed);
    }

    /// Get snapshot of current statistics
    pub fn snapshot(&self) -> IndexStatisticsSnapshot {
        IndexStatisticsSnapshot {
            total_lookups: self.total_lookups.load(Ordering::Relaxed),
            predicate_hits: self.predicate_hits.load(Ordering::Relaxed),
            first_arg_hits: self.first_arg_hits.load(Ordering::Relaxed),
            combined_hits: self.combined_hits.load(Ordering::Relaxed),
            full_scans: self.full_scans.load(Ordering::Relaxed),
            rules_checked: self.rules_checked.load(Ordering::Relaxed),
            rules_matched: self.rules_matched.load(Ordering::Relaxed),
        }
    }
}

/// Immutable snapshot of index statistics
#[derive(Debug, Clone)]
pub struct IndexStatisticsSnapshot {
    pub total_lookups: u64,
    pub predicate_hits: u64,
    pub first_arg_hits: u64,
    pub combined_hits: u64,
    pub full_scans: u64,
    pub rules_checked: u64,
    pub rules_matched: u64,
}

/// Rule identifier for index storage
pub type RuleId = usize;

/// An edge in the rule dependency graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DependencyEdge {
    /// The rule that produces the predicate.
    pub from: RuleId,
    /// The rule that consumes the predicate.
    pub to: RuleId,
    /// The predicate IRI connecting the two rules.
    pub predicate: String,
}

/// A rule dependency graph: which rules trigger which other rules.
///
/// Rule A "triggers" Rule B when A produces a predicate in its head that B
/// consumes in its body.
#[derive(Debug, Default)]
pub struct RuleDependencyGraph {
    /// All edges (from -> to via predicate).
    pub(crate) edges: Vec<DependencyEdge>,
    /// Forward adjacency: rule_id -> list of rules it triggers.
    pub(crate) forward: HashMap<RuleId, Vec<RuleId>>,
    /// Backward adjacency: rule_id -> list of rules that trigger it.
    pub(crate) backward: HashMap<RuleId, Vec<RuleId>>,
}

impl RuleDependencyGraph {
    /// Add an edge to the graph.
    pub(crate) fn add_edge(&mut self, edge: DependencyEdge) {
        self.forward.entry(edge.from).or_default().push(edge.to);
        self.backward.entry(edge.to).or_default().push(edge.from);
        self.edges.push(edge);
    }

    /// Return the rules triggered by the given rule.
    pub fn triggered_by(&self, rule_id: RuleId) -> Vec<RuleId> {
        self.forward.get(&rule_id).cloned().unwrap_or_default()
    }

    /// Return the rules that trigger the given rule.
    pub fn triggers_of(&self, rule_id: RuleId) -> Vec<RuleId> {
        self.backward.get(&rule_id).cloned().unwrap_or_default()
    }

    /// Return all edges.
    pub fn edges(&self) -> &[DependencyEdge] {
        &self.edges
    }

    /// Return the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return all root rules (rules with no incoming dependency).
    pub fn roots(&self) -> Vec<RuleId> {
        let all_targets: HashSet<RuleId> = self.edges.iter().map(|e| e.to).collect();
        let all_sources: HashSet<RuleId> = self.edges.iter().map(|e| e.from).collect();
        all_sources.difference(&all_targets).copied().collect()
    }

    /// Detect cycles in the dependency graph using DFS.
    pub fn has_cycle(&self) -> bool {
        let all_nodes: HashSet<RuleId> = self.edges.iter().flat_map(|e| [e.from, e.to]).collect();

        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();

        for &node in &all_nodes {
            if !visited.contains(&node) && self.dfs_cycle(node, &mut visited, &mut on_stack) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(
        &self,
        node: RuleId,
        visited: &mut HashSet<RuleId>,
        on_stack: &mut HashSet<RuleId>,
    ) -> bool {
        visited.insert(node);
        on_stack.insert(node);

        if let Some(neighbours) = self.forward.get(&node) {
            for &next in neighbours {
                if !visited.contains(&next) {
                    if self.dfs_cycle(next, visited, on_stack) {
                        return true;
                    }
                } else if on_stack.contains(&next) {
                    return true;
                }
            }
        }

        on_stack.remove(&node);
        false
    }

    /// Return a topological ordering of rules (or empty if cycle detected).
    pub fn topological_order(&self) -> Vec<RuleId> {
        let all_nodes: HashSet<RuleId> = self.edges.iter().flat_map(|e| [e.from, e.to]).collect();

        let mut in_degree: HashMap<RuleId, usize> = HashMap::new();
        for &node in &all_nodes {
            in_degree.entry(node).or_insert(0);
        }
        for edge in &self.edges {
            *in_degree.entry(edge.to).or_insert(0) += 1;
        }

        let mut queue: Vec<RuleId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort_unstable(); // deterministic

        let mut result = Vec::new();
        while let Some(node) = queue.pop() {
            result.push(node);
            if let Some(neighbours) = self.forward.get(&node) {
                for &next in neighbours {
                    if let Some(deg) = in_degree.get_mut(&next) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push(next);
                            queue.sort_unstable();
                        }
                    }
                }
            }
        }

        if result.len() < all_nodes.len() {
            // Cycle detected
            Vec::new()
        } else {
            result
        }
    }
}

/// A prioritized rule reference.
#[derive(Debug, Clone)]
pub struct PrioritizedRule {
    /// Rule identifier.
    pub id: RuleId,
    /// Rule name.
    pub name: String,
    /// Priority (higher = executed first).
    pub priority: i32,
}
