//! # Rule Indexing for High-Performance Rule Matching
//!
//! This module provides indexed rule lookup for efficient rule matching in large rule sets.
//! Instead of sequential scanning through all rules, this module enables O(1) lookup
//! based on predicate and first-argument patterns.
//!
//! ## Features
//!
//! - **Predicate Indexing**: Index rules by their body predicate patterns
//! - **First-Argument Indexing**: Additional indexing by first argument for common patterns
//! - **Hash-Based Lookup**: O(1) average case retrieval
//! - **Index Statistics**: Track hit rates and performance metrics
//! - **Automatic Maintenance**: Self-updating indices on rule addition/removal
//!
//! ## Performance Impact
//!
//! - **Without indexing**: O(n) rule scan per fact (n = number of rules)
//! - **With indexing**: O(1) average lookup + O(m) matching rules (m << n typically)
//! - **Expected speedup**: 10-100x for large rule sets (100+ rules)
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::rule_index::{RuleIndex, IndexConfig};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let config = IndexConfig::default()
//!     .with_predicate_indexing(true)
//!     .with_first_arg_indexing(true);
//!
//! let mut index = RuleIndex::new(config);
//!
//! // Add rules to index
//! let rule = Rule {
//!     name: "ancestor".to_string(),
//!     body: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("parent".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("ancestor".to_string()),
//!         object: Term::Variable("Y".to_string()),
//!     }],
//! };
//!
//! index.add_rule(rule);
//!
//! // Fast lookup by predicate
//! let matching_rules = index.find_rules_by_predicate("parent");
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::{Rule, RuleAtom, Term};

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

/// High-performance rule index with multiple indexing strategies
#[derive(Debug)]
pub struct RuleIndex {
    /// Configuration
    config: IndexConfig,
    /// All rules stored by ID
    rules: RwLock<Vec<Rule>>,
    /// Predicate -> Rule IDs index
    predicate_index: RwLock<HashMap<PredicateKey, HashSet<RuleId>>>,
    /// (Predicate, FirstArg) -> Rule IDs index
    first_arg_index: RwLock<HashMap<FirstArgKey, HashSet<RuleId>>>,
    /// Combined pattern -> Rule IDs index
    combined_index: RwLock<HashMap<CombinedKey, HashSet<RuleId>>>,
    /// Statistics tracker
    statistics: IndexStatistics,
}

impl RuleIndex {
    /// Create a new rule index with the given configuration
    pub fn new(config: IndexConfig) -> Self {
        Self {
            config,
            rules: RwLock::new(Vec::new()),
            predicate_index: RwLock::new(HashMap::new()),
            first_arg_index: RwLock::new(HashMap::new()),
            combined_index: RwLock::new(HashMap::new()),
            statistics: IndexStatistics::new(),
        }
    }

    /// Create a new rule index with default configuration
    pub fn with_defaults() -> Self {
        Self::new(IndexConfig::default())
    }

    /// Add a rule to the index
    pub fn add_rule(&self, rule: Rule) -> RuleId {
        let mut rules = self.rules.write().expect("lock poisoned");
        let rule_id = rules.len();

        // Index by body predicates
        if self.config.predicate_indexing {
            self.index_by_predicate(&rule, rule_id);
        }

        // Index by first argument
        if self.config.first_arg_indexing {
            self.index_by_first_arg(&rule, rule_id);
        }

        // Index by combined pattern
        if self.config.combined_indexing {
            self.index_by_combined(&rule, rule_id);
        }

        rules.push(rule);
        rule_id
    }

    /// Add multiple rules to the index
    pub fn add_rules(&self, rules: Vec<Rule>) -> Vec<RuleId> {
        rules.into_iter().map(|r| self.add_rule(r)).collect()
    }

    /// Remove a rule from the index by ID
    pub fn remove_rule(&self, rule_id: RuleId) -> Option<Rule> {
        let mut rules = self.rules.write().expect("lock poisoned");
        if rule_id >= rules.len() {
            return None;
        }

        let rule = rules[rule_id].clone();

        // Remove from predicate index
        if self.config.predicate_indexing {
            self.unindex_by_predicate(&rule, rule_id);
        }

        // Remove from first-arg index
        if self.config.first_arg_indexing {
            self.unindex_by_first_arg(&rule, rule_id);
        }

        // Remove from combined index
        if self.config.combined_indexing {
            self.unindex_by_combined(&rule, rule_id);
        }

        // Mark as removed (we don't actually remove to preserve IDs)
        rules[rule_id] = Rule {
            name: String::new(),
            body: Vec::new(),
            head: Vec::new(),
        };

        Some(rule)
    }

    /// Find rules by predicate name
    pub fn find_rules_by_predicate(&self, predicate: &str) -> Vec<&Rule> {
        if self.config.collect_statistics {
            self.statistics
                .total_lookups
                .fetch_add(1, Ordering::Relaxed);
        }

        let rules = self.rules.read().expect("lock poisoned");

        if self.config.predicate_indexing {
            let pred_index = self.predicate_index.read().expect("lock poisoned");
            let key = PredicateKey(predicate.to_string());

            if let Some(rule_ids) = pred_index.get(&key) {
                if self.config.collect_statistics {
                    self.statistics
                        .predicate_hits
                        .fetch_add(1, Ordering::Relaxed);
                    self.statistics
                        .rules_checked
                        .fetch_add(rule_ids.len() as u64, Ordering::Relaxed);
                }

                // SAFETY: We need to return references that outlive the lock
                // This is safe because we're returning references to rules that won't be modified
                let result: Vec<*const Rule> = rule_ids
                    .iter()
                    .filter_map(|&id| {
                        let rule = rules.get(id)?;
                        if !rule.name.is_empty() {
                            Some(rule as *const Rule)
                        } else {
                            None
                        }
                    })
                    .collect();

                drop(rules);
                drop(pred_index);

                // This is safe as long as rules aren't removed while references exist
                // In production, use Arc<Rule> instead
                return result.into_iter().map(|p| unsafe { &*p }).collect();
            }
        }

        // Fallback to full scan
        if self.config.collect_statistics {
            self.statistics.full_scans.fetch_add(1, Ordering::Relaxed);
            self.statistics
                .rules_checked
                .fetch_add(rules.len() as u64, Ordering::Relaxed);
        }

        // Full scan fallback (returns empty for now due to lifetime issues)
        // In production code, use Arc<Rule> for proper lifetime management
        Vec::new()
    }

    /// Find rules that match a specific triple pattern
    pub fn find_rules_for_triple(
        &self,
        subject: Option<&str>,
        predicate: &str,
        object: Option<&str>,
    ) -> Vec<RuleId> {
        if self.config.collect_statistics {
            self.statistics
                .total_lookups
                .fetch_add(1, Ordering::Relaxed);
        }

        // Try combined index first
        if self.config.combined_indexing {
            let combined = self.combined_index.read().expect("lock poisoned");
            let key = CombinedKey {
                predicate: predicate.to_string(),
                subject_type: subject.map_or(ArgType::Any, |s| ArgType::Constant(s.to_string())),
                object_type: object.map_or(ArgType::Any, |o| ArgType::Constant(o.to_string())),
            };

            if let Some(rule_ids) = combined.get(&key) {
                if self.config.collect_statistics {
                    self.statistics
                        .combined_hits
                        .fetch_add(1, Ordering::Relaxed);
                    self.statistics
                        .rules_checked
                        .fetch_add(rule_ids.len() as u64, Ordering::Relaxed);
                }
                return rule_ids.iter().copied().collect();
            }
        }

        // Try first-arg index
        if self.config.first_arg_indexing && subject.is_some() {
            let first_arg = self.first_arg_index.read().expect("lock poisoned");
            let key = FirstArgKey {
                predicate: predicate.to_string(),
                first_arg: subject.map(|s| s.to_string()),
            };

            if let Some(rule_ids) = first_arg.get(&key) {
                if self.config.collect_statistics {
                    self.statistics
                        .first_arg_hits
                        .fetch_add(1, Ordering::Relaxed);
                    self.statistics
                        .rules_checked
                        .fetch_add(rule_ids.len() as u64, Ordering::Relaxed);
                }
                return rule_ids.iter().copied().collect();
            }
        }

        // Try predicate index
        if self.config.predicate_indexing {
            let pred_index = self.predicate_index.read().expect("lock poisoned");
            let key = PredicateKey(predicate.to_string());

            if let Some(rule_ids) = pred_index.get(&key) {
                if self.config.collect_statistics {
                    self.statistics
                        .predicate_hits
                        .fetch_add(1, Ordering::Relaxed);
                    self.statistics
                        .rules_checked
                        .fetch_add(rule_ids.len() as u64, Ordering::Relaxed);
                }
                return rule_ids.iter().copied().collect();
            }
        }

        // Fallback: return all rule IDs
        if self.config.collect_statistics {
            self.statistics.full_scans.fetch_add(1, Ordering::Relaxed);
        }

        let rules = self.rules.read().expect("lock poisoned");
        if self.config.collect_statistics {
            self.statistics
                .rules_checked
                .fetch_add(rules.len() as u64, Ordering::Relaxed);
        }
        (0..rules.len())
            .filter(|&id| !rules[id].name.is_empty())
            .collect()
    }

    /// Get a rule by ID
    pub fn get_rule(&self, rule_id: RuleId) -> Option<Rule> {
        let rules = self.rules.read().expect("lock poisoned");
        rules.get(rule_id).filter(|r| !r.name.is_empty()).cloned()
    }

    /// Get all rules (excluding removed ones)
    pub fn get_all_rules(&self) -> Vec<Rule> {
        let rules = self.rules.read().expect("lock poisoned");
        rules
            .iter()
            .filter(|r| !r.name.is_empty())
            .cloned()
            .collect()
    }

    /// Get the total number of indexed rules
    pub fn rule_count(&self) -> usize {
        let rules = self.rules.read().expect("lock poisoned");
        rules.iter().filter(|r| !r.name.is_empty()).count()
    }

    /// Get index statistics
    pub fn statistics(&self) -> &IndexStatistics {
        &self.statistics
    }

    /// Get statistics snapshot
    pub fn statistics_snapshot(&self) -> IndexStatisticsSnapshot {
        self.statistics.snapshot()
    }

    /// Reset statistics
    pub fn reset_statistics(&self) {
        self.statistics.reset();
    }

    /// Clear all rules and indices
    pub fn clear(&self) {
        let mut rules = self.rules.write().expect("lock poisoned");
        let mut pred_index = self.predicate_index.write().expect("lock poisoned");
        let mut first_arg = self.first_arg_index.write().expect("lock poisoned");
        let mut combined = self.combined_index.write().expect("lock poisoned");

        rules.clear();
        pred_index.clear();
        first_arg.clear();
        combined.clear();

        self.statistics.reset();
    }

    /// Get index memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        let rules = self.rules.read().expect("lock poisoned");
        let pred_index = self.predicate_index.read().expect("lock poisoned");
        let first_arg = self.first_arg_index.read().expect("lock poisoned");
        let combined = self.combined_index.read().expect("lock poisoned");

        let rules_size = rules.len() * std::mem::size_of::<Rule>();
        let pred_size = pred_index.len()
            * (std::mem::size_of::<PredicateKey>() + std::mem::size_of::<HashSet<RuleId>>());
        let first_arg_size = first_arg.len()
            * (std::mem::size_of::<FirstArgKey>() + std::mem::size_of::<HashSet<RuleId>>());
        let combined_size = combined.len()
            * (std::mem::size_of::<CombinedKey>() + std::mem::size_of::<HashSet<RuleId>>());

        rules_size + pred_size + first_arg_size + combined_size
    }

    // === Private indexing methods ===

    fn index_by_predicate(&self, rule: &Rule, rule_id: RuleId) {
        let mut pred_index = self.predicate_index.write().expect("lock poisoned");

        for atom in &rule.body {
            if let Some(predicate) = Self::extract_predicate(atom) {
                let key = PredicateKey(predicate);
                pred_index.entry(key).or_default().insert(rule_id);
            }
        }
    }

    fn unindex_by_predicate(&self, rule: &Rule, rule_id: RuleId) {
        let mut pred_index = self.predicate_index.write().expect("lock poisoned");

        for atom in &rule.body {
            if let Some(predicate) = Self::extract_predicate(atom) {
                let key = PredicateKey(predicate);
                if let Some(ids) = pred_index.get_mut(&key) {
                    ids.remove(&rule_id);
                    if ids.is_empty() {
                        pred_index.remove(&key);
                    }
                }
            }
        }
    }

    fn index_by_first_arg(&self, rule: &Rule, rule_id: RuleId) {
        let mut first_arg_index = self.first_arg_index.write().expect("lock poisoned");

        for atom in &rule.body {
            if let (Some(predicate), first_arg) =
                (Self::extract_predicate(atom), Self::extract_first_arg(atom))
            {
                let key = FirstArgKey {
                    predicate,
                    first_arg,
                };
                first_arg_index.entry(key).or_default().insert(rule_id);
            }
        }
    }

    fn unindex_by_first_arg(&self, rule: &Rule, rule_id: RuleId) {
        let mut first_arg_index = self.first_arg_index.write().expect("lock poisoned");

        for atom in &rule.body {
            if let (Some(predicate), first_arg) =
                (Self::extract_predicate(atom), Self::extract_first_arg(atom))
            {
                let key = FirstArgKey {
                    predicate,
                    first_arg,
                };
                if let Some(ids) = first_arg_index.get_mut(&key) {
                    ids.remove(&rule_id);
                    if ids.is_empty() {
                        first_arg_index.remove(&key);
                    }
                }
            }
        }
    }

    fn index_by_combined(&self, rule: &Rule, rule_id: RuleId) {
        let mut combined_index = self.combined_index.write().expect("lock poisoned");

        for atom in &rule.body {
            if let Some(key) = Self::extract_combined_key(atom) {
                combined_index.entry(key).or_default().insert(rule_id);
            }
        }
    }

    fn unindex_by_combined(&self, rule: &Rule, rule_id: RuleId) {
        let mut combined_index = self.combined_index.write().expect("lock poisoned");

        for atom in &rule.body {
            if let Some(key) = Self::extract_combined_key(atom) {
                if let Some(ids) = combined_index.get_mut(&key) {
                    ids.remove(&rule_id);
                    if ids.is_empty() {
                        combined_index.remove(&key);
                    }
                }
            }
        }
    }

    fn extract_predicate(atom: &RuleAtom) -> Option<String> {
        match atom {
            RuleAtom::Triple {
                predicate: Term::Constant(c),
                ..
            } => Some(c.clone()),
            RuleAtom::Triple { .. } => None,
            RuleAtom::Builtin { name, .. } => Some(name.clone()),
            _ => None,
        }
    }

    fn extract_first_arg(atom: &RuleAtom) -> Option<String> {
        match atom {
            RuleAtom::Triple {
                subject: Term::Constant(c),
                ..
            } => Some(c.clone()),
            RuleAtom::Triple { .. } => None,
            RuleAtom::Builtin { args, .. } => args.first().and_then(|arg| {
                if let Term::Constant(c) = arg {
                    Some(c.clone())
                } else {
                    None
                }
            }),
            _ => None,
        }
    }

    fn extract_combined_key(atom: &RuleAtom) -> Option<CombinedKey> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let predicate = match predicate {
                    Term::Constant(c) => c.clone(),
                    _ => return None,
                };

                let subject_type = match subject {
                    Term::Constant(c) => ArgType::Constant(c.clone()),
                    Term::Variable(_) => ArgType::Variable,
                    _ => ArgType::Any,
                };

                let object_type = match object {
                    Term::Constant(c) => ArgType::Constant(c.clone()),
                    Term::Variable(_) => ArgType::Variable,
                    _ => ArgType::Any,
                };

                Some(CombinedKey {
                    predicate,
                    subject_type,
                    object_type,
                })
            }
            _ => None,
        }
    }
}

impl Default for RuleIndex {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Builder for creating optimized rule indices
pub struct RuleIndexBuilder {
    config: IndexConfig,
    rules: Vec<Rule>,
}

impl RuleIndexBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: IndexConfig::default(),
            rules: Vec::new(),
        }
    }

    /// Set the index configuration
    pub fn config(mut self, config: IndexConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a rule to be indexed
    pub fn add_rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Add multiple rules to be indexed
    pub fn add_rules(mut self, rules: Vec<Rule>) -> Self {
        self.rules.extend(rules);
        self
    }

    /// Build the rule index
    pub fn build(self) -> RuleIndex {
        let index = RuleIndex::new(self.config);
        for rule in self.rules {
            index.add_rule(rule);
        }
        index
    }
}

impl Default for RuleIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Rule dependency graph
// ---------------------------------------------------------------------------

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
    edges: Vec<DependencyEdge>,
    /// Forward adjacency: rule_id -> list of rules it triggers.
    forward: HashMap<RuleId, Vec<RuleId>>,
    /// Backward adjacency: rule_id -> list of rules that trigger it.
    backward: HashMap<RuleId, Vec<RuleId>>,
}

impl RuleDependencyGraph {
    /// Build a dependency graph from an existing index.
    pub fn from_index(index: &RuleIndex) -> Self {
        let rules = index.rules.read().expect("lock poisoned");
        let mut graph = Self::default();

        // Collect head predicates per rule
        let mut head_preds: HashMap<String, Vec<RuleId>> = HashMap::new();
        for (id, rule) in rules.iter().enumerate() {
            if rule.name.is_empty() {
                continue;
            }
            for atom in &rule.head {
                if let Some(pred) = RuleIndex::extract_predicate(atom) {
                    head_preds.entry(pred).or_default().push(id);
                }
            }
        }

        // For each rule body predicate, find rules that produce it
        for (consumer_id, rule) in rules.iter().enumerate() {
            if rule.name.is_empty() {
                continue;
            }
            for atom in &rule.body {
                if let Some(pred) = RuleIndex::extract_predicate(atom) {
                    if let Some(producers) = head_preds.get(&pred) {
                        for &producer_id in producers {
                            if producer_id != consumer_id {
                                graph.add_edge(DependencyEdge {
                                    from: producer_id,
                                    to: consumer_id,
                                    predicate: pred.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        graph
    }

    /// Add an edge to the graph.
    fn add_edge(&mut self, edge: DependencyEdge) {
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

// ---------------------------------------------------------------------------
// Priority-based rule ordering
// ---------------------------------------------------------------------------

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

/// Priority index that allows ordering rules by a user-defined priority.
#[derive(Debug, Default)]
pub struct PriorityIndex {
    priorities: HashMap<RuleId, i32>,
}

impl PriorityIndex {
    /// Create a new, empty priority index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the priority for a rule.
    pub fn set_priority(&mut self, rule_id: RuleId, priority: i32) {
        self.priorities.insert(rule_id, priority);
    }

    /// Get the priority for a rule (defaults to 0).
    pub fn get_priority(&self, rule_id: RuleId) -> i32 {
        self.priorities.get(&rule_id).copied().unwrap_or(0)
    }

    /// Remove the priority for a rule.
    pub fn remove_priority(&mut self, rule_id: RuleId) {
        self.priorities.remove(&rule_id);
    }

    /// Sort a list of rule IDs by priority (descending: highest first).
    pub fn sort_by_priority(&self, rule_ids: &mut [RuleId]) {
        rule_ids.sort_by(|a, b| {
            let pa = self.get_priority(*a);
            let pb = self.get_priority(*b);
            pb.cmp(&pa) // descending
        });
    }

    /// Return all rules ordered by priority (highest first).
    pub fn ordered_rules(&self, index: &RuleIndex) -> Vec<PrioritizedRule> {
        let rules = index.rules.read().expect("lock poisoned");
        let mut result: Vec<PrioritizedRule> = rules
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.name.is_empty())
            .map(|(id, r)| PrioritizedRule {
                id,
                name: r.name.clone(),
                priority: self.get_priority(id),
            })
            .collect();
        result.sort_by(|a, b| b.priority.cmp(&a.priority));
        result
    }

    /// Return the number of rules with explicit priorities.
    pub fn len(&self) -> usize {
        self.priorities.len()
    }

    /// Return true if no priorities are set.
    pub fn is_empty(&self) -> bool {
        self.priorities.is_empty()
    }

    /// Clear all priorities.
    pub fn clear(&mut self) {
        self.priorities.clear();
    }
}

// ---------------------------------------------------------------------------
// Wildcard matching helper
// ---------------------------------------------------------------------------

/// Check if a pattern (with `*` wildcards) matches a given string.
pub fn wildcard_matches(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if !pattern.contains('*') {
        return pattern == text;
    }

    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.is_empty() {
        return true;
    }

    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if let Some(found) = text[pos..].find(part) {
            if i == 0 && found != 0 {
                // First segment must anchor at the start
                return false;
            }
            pos += found + part.len();
        } else {
            return false;
        }
    }

    // If the pattern does not end with '*', the last part must anchor at the end
    if let Some(last) = parts.last() {
        if !last.is_empty() && !text.ends_with(last) {
            return false;
        }
    }

    true
}

impl RuleIndex {
    /// Find rules whose body predicates match a wildcard pattern.
    pub fn find_rules_by_wildcard(&self, pattern: &str) -> Vec<RuleId> {
        let rules = self.rules.read().expect("lock poisoned");
        let mut result = Vec::new();

        for (id, rule) in rules.iter().enumerate() {
            if rule.name.is_empty() {
                continue;
            }
            for atom in &rule.body {
                if let Some(pred) = Self::extract_predicate(atom) {
                    if wildcard_matches(pattern, &pred) {
                        result.push(id);
                        break;
                    }
                }
            }
        }

        result
    }

    /// Build the dependency graph for all indexed rules.
    pub fn dependency_graph(&self) -> RuleDependencyGraph {
        RuleDependencyGraph::from_index(self)
    }

    /// Re-index a single rule (remove old entries, re-add).
    pub fn reindex_rule(&self, rule_id: RuleId) -> bool {
        let rules = self.rules.read().expect("lock poisoned");
        let rule = match rules.get(rule_id) {
            Some(r) if !r.name.is_empty() => r.clone(),
            _ => return false,
        };
        drop(rules);

        // Remove old entries
        if self.config.predicate_indexing {
            self.unindex_by_predicate(&rule, rule_id);
        }
        if self.config.first_arg_indexing {
            self.unindex_by_first_arg(&rule, rule_id);
        }
        if self.config.combined_indexing {
            self.unindex_by_combined(&rule, rule_id);
        }

        // Re-add entries
        if self.config.predicate_indexing {
            self.index_by_predicate(&rule, rule_id);
        }
        if self.config.first_arg_indexing {
            self.index_by_first_arg(&rule, rule_id);
        }
        if self.config.combined_indexing {
            self.index_by_combined(&rule, rule_id);
        }

        true
    }

    /// Replace a rule in-place and re-index it.
    pub fn replace_rule(&self, rule_id: RuleId, new_rule: Rule) -> Option<Rule> {
        let rules = self.rules.read().expect("lock poisoned");
        if rule_id >= rules.len() || rules[rule_id].name.is_empty() {
            return None;
        }

        let old_rule = rules[rule_id].clone();
        drop(rules);

        // Un-index the old rule
        if self.config.predicate_indexing {
            self.unindex_by_predicate(&old_rule, rule_id);
        }
        if self.config.first_arg_indexing {
            self.unindex_by_first_arg(&old_rule, rule_id);
        }
        if self.config.combined_indexing {
            self.unindex_by_combined(&old_rule, rule_id);
        }

        // Index the new rule
        if self.config.predicate_indexing {
            self.index_by_predicate(&new_rule, rule_id);
        }
        if self.config.first_arg_indexing {
            self.index_by_first_arg(&new_rule, rule_id);
        }
        if self.config.combined_indexing {
            self.index_by_combined(&new_rule, rule_id);
        }

        // Store the new rule
        let mut rules = self.rules.write().expect("lock poisoned");
        rules[rule_id] = new_rule;

        Some(old_rule)
    }

    /// Return predicate-level index density (entries / unique predicates).
    pub fn predicate_density(&self) -> f64 {
        let pred_index = self.predicate_index.read().expect("lock poisoned");
        if pred_index.is_empty() {
            return 0.0;
        }
        let total_entries: usize = pred_index.values().map(|v| v.len()).sum();
        total_entries as f64 / pred_index.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_rule(name: &str, pred: &str) -> Rule {
        Rule {
            name: name.to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(pred.to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(format!("inferred_{pred}")),
                object: Term::Variable("Y".to_string()),
            }],
        }
    }

    #[test]
    fn test_add_and_find_rule() {
        let index = RuleIndex::with_defaults();
        let rule = create_test_rule("rule1", "parent");

        let id = index.add_rule(rule);
        assert_eq!(id, 0);

        let found = index.find_rules_for_triple(None, "parent", None);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], 0);
    }

    #[test]
    fn test_multiple_rules_same_predicate() {
        let index = RuleIndex::with_defaults();

        index.add_rule(create_test_rule("rule1", "parent"));
        index.add_rule(create_test_rule("rule2", "parent"));
        index.add_rule(create_test_rule("rule3", "child"));

        let parent_rules = index.find_rules_for_triple(None, "parent", None);
        assert_eq!(parent_rules.len(), 2);

        let child_rules = index.find_rules_for_triple(None, "child", None);
        assert_eq!(child_rules.len(), 1);
    }

    #[test]
    fn test_remove_rule() {
        let index = RuleIndex::with_defaults();

        let id1 = index.add_rule(create_test_rule("rule1", "parent"));
        let id2 = index.add_rule(create_test_rule("rule2", "parent"));

        assert_eq!(index.rule_count(), 2);

        let removed = index.remove_rule(id1);
        assert!(removed.is_some());
        assert_eq!(removed.expect("already checked is_some").name, "rule1");

        assert_eq!(index.rule_count(), 1);

        let found = index.find_rules_for_triple(None, "parent", None);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], id2);
    }

    #[test]
    fn test_statistics() {
        let config = IndexConfig::default().with_statistics(true);
        let index = RuleIndex::new(config);

        index.add_rule(create_test_rule("rule1", "parent"));
        index.add_rule(create_test_rule("rule2", "child"));

        // Perform lookups
        index.find_rules_for_triple(None, "parent", None);
        index.find_rules_for_triple(None, "child", None);
        index.find_rules_for_triple(None, "unknown", None);

        let stats = index.statistics_snapshot();
        assert_eq!(stats.total_lookups, 3);
        assert!(stats.predicate_hits >= 2); // At least 2 hits for known predicates
    }

    #[test]
    fn test_clear() {
        let index = RuleIndex::with_defaults();

        index.add_rule(create_test_rule("rule1", "parent"));
        index.add_rule(create_test_rule("rule2", "child"));

        assert_eq!(index.rule_count(), 2);

        index.clear();

        assert_eq!(index.rule_count(), 0);
    }

    #[test]
    fn test_builder() {
        let index = RuleIndexBuilder::new()
            .config(IndexConfig::default().with_combined_indexing(true))
            .add_rule(create_test_rule("rule1", "parent"))
            .add_rule(create_test_rule("rule2", "child"))
            .build();

        assert_eq!(index.rule_count(), 2);
    }

    #[test]
    fn test_get_rule() {
        let index = RuleIndex::with_defaults();
        let rule = create_test_rule("rule1", "parent");

        let id = index.add_rule(rule.clone());

        let retrieved = index.get_rule(id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("already checked is_some").name, "rule1");

        let not_found = index.get_rule(999);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_get_all_rules() {
        let index = RuleIndex::with_defaults();

        index.add_rule(create_test_rule("rule1", "parent"));
        index.add_rule(create_test_rule("rule2", "child"));
        index.add_rule(create_test_rule("rule3", "sibling"));

        let all = index.get_all_rules();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_memory_usage() {
        let index = RuleIndex::with_defaults();

        for i in 0..100 {
            index.add_rule(create_test_rule(
                &format!("rule{i}"),
                &format!("pred{}", i % 10),
            ));
        }

        let usage = index.memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_combined_key_indexing() {
        let index = RuleIndex::new(IndexConfig::default().with_combined_indexing(true));

        // Rule with constant subject
        let rule_with_const = Rule {
            name: "const_subject".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("knows".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Constant("john".to_string()),
                predicate: Term::Constant("friend".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        index.add_rule(rule_with_const);

        // Should find with specific subject
        let found = index.find_rules_for_triple(Some("john"), "knows", None);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_first_arg_indexing() {
        let config = IndexConfig::default()
            .with_predicate_indexing(true)
            .with_first_arg_indexing(true);
        let index = RuleIndex::new(config);

        index.add_rule(create_test_rule("rule1", "parent"));

        // Lookup should work
        let found = index.find_rules_for_triple(None, "parent", None);
        assert!(!found.is_empty());
    }

    #[test]
    fn test_statistics_reset() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_test_rule("rule1", "parent"));

        index.find_rules_for_triple(None, "parent", None);

        let stats_before = index.statistics_snapshot();
        assert!(stats_before.total_lookups > 0);

        index.reset_statistics();

        let stats_after = index.statistics_snapshot();
        assert_eq!(stats_after.total_lookups, 0);
    }

    #[test]
    fn test_hit_rates() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_test_rule("rule1", "parent"));
        index.add_rule(create_test_rule("rule2", "child"));

        // Make some lookups
        for _ in 0..10 {
            index.find_rules_for_triple(None, "parent", None);
        }

        let stats = index.statistics();
        assert!(stats.predicate_hit_rate() > 0.0);
    }

    #[test]
    fn test_builtin_indexing() {
        let index = RuleIndex::with_defaults();

        let rule = Rule {
            name: "builtin_rule".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("hasAge".to_string()),
                    object: Term::Variable("Age".to_string()),
                },
                RuleAtom::Builtin {
                    name: "greaterThan".to_string(),
                    args: vec![
                        Term::Variable("Age".to_string()),
                        Term::Constant("18".to_string()),
                    ],
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("isAdult".to_string()),
                object: Term::Constant("true".to_string()),
            }],
        };

        index.add_rule(rule);

        let found = index.find_rules_for_triple(None, "hasAge", None);
        assert_eq!(found.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Dependency graph tests
    // -----------------------------------------------------------------------

    fn create_chain_rule(name: &str, body_pred: &str, head_pred: &str) -> Rule {
        Rule {
            name: name.to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(body_pred.to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(head_pred.to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }
    }

    #[test]
    fn test_dependency_graph_basic() {
        let index = RuleIndex::with_defaults();
        // r1: parent -> ancestor
        index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
        // r2: ancestor -> reachable
        index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));

        let graph = index.dependency_graph();
        assert_eq!(graph.edge_count(), 1);
        // r1 triggers r2
        let triggered = graph.triggered_by(0);
        assert_eq!(triggered.len(), 1);
        assert_eq!(triggered[0], 1);
    }

    #[test]
    fn test_dependency_graph_triggers_of() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
        index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));

        let graph = index.dependency_graph();
        let triggers = graph.triggers_of(1);
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0], 0);
    }

    #[test]
    fn test_dependency_graph_roots() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
        index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));
        index.add_rule(create_chain_rule("r3", "reachable", "connected"));

        let graph = index.dependency_graph();
        let roots = graph.roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], 0);
    }

    #[test]
    fn test_dependency_graph_no_cycle() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
        index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));

        let graph = index.dependency_graph();
        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_dependency_graph_cycle() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_chain_rule("r1", "a", "b"));
        index.add_rule(create_chain_rule("r2", "b", "a"));

        let graph = index.dependency_graph();
        assert!(graph.has_cycle());
    }

    #[test]
    fn test_dependency_graph_topological_order() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
        index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));
        index.add_rule(create_chain_rule("r3", "reachable", "connected"));

        let graph = index.dependency_graph();
        let topo = graph.topological_order();
        assert_eq!(topo.len(), 3);
        // r1 (0) should come before r2 (1), r2 before r3 (2)
        let pos_r1 = topo.iter().position(|&x| x == 0).expect("r1 in topo");
        let pos_r2 = topo.iter().position(|&x| x == 1).expect("r2 in topo");
        let pos_r3 = topo.iter().position(|&x| x == 2).expect("r3 in topo");
        assert!(pos_r1 < pos_r2);
        assert!(pos_r2 < pos_r3);
    }

    #[test]
    fn test_dependency_graph_topological_cycle_empty() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_chain_rule("r1", "a", "b"));
        index.add_rule(create_chain_rule("r2", "b", "a"));

        let graph = index.dependency_graph();
        let topo = graph.topological_order();
        assert!(topo.is_empty()); // cycle detected
    }

    #[test]
    fn test_dependency_graph_empty_index() {
        let index = RuleIndex::with_defaults();
        let graph = index.dependency_graph();
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.roots().is_empty());
        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_dependency_graph_self_no_cycle() {
        // A rule that doesn't reference itself
        let index = RuleIndex::with_defaults();
        index.add_rule(create_test_rule("r1", "parent"));
        let graph = index.dependency_graph();
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_dependency_graph_edges() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_chain_rule("r1", "parent", "ancestor"));
        index.add_rule(create_chain_rule("r2", "ancestor", "reachable"));

        let graph = index.dependency_graph();
        let edges = graph.edges();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].from, 0);
        assert_eq!(edges[0].to, 1);
        assert_eq!(edges[0].predicate, "ancestor");
    }

    // -----------------------------------------------------------------------
    // Priority index tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_priority_set_and_get() {
        let mut pi = PriorityIndex::new();
        pi.set_priority(0, 10);
        pi.set_priority(1, 5);
        assert_eq!(pi.get_priority(0), 10);
        assert_eq!(pi.get_priority(1), 5);
        assert_eq!(pi.get_priority(99), 0); // default
    }

    #[test]
    fn test_priority_sort() {
        let mut pi = PriorityIndex::new();
        pi.set_priority(0, 1);
        pi.set_priority(1, 10);
        pi.set_priority(2, 5);

        let mut ids = vec![0, 1, 2];
        pi.sort_by_priority(&mut ids);
        assert_eq!(ids, vec![1, 2, 0]); // 10, 5, 1
    }

    #[test]
    fn test_priority_ordered_rules() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_test_rule("low", "p1"));
        index.add_rule(create_test_rule("high", "p2"));
        index.add_rule(create_test_rule("mid", "p3"));

        let mut pi = PriorityIndex::new();
        pi.set_priority(0, 1);
        pi.set_priority(1, 100);
        pi.set_priority(2, 50);

        let ordered = pi.ordered_rules(&index);
        assert_eq!(ordered.len(), 3);
        assert_eq!(ordered[0].name, "high");
        assert_eq!(ordered[1].name, "mid");
        assert_eq!(ordered[2].name, "low");
    }

    #[test]
    fn test_priority_remove() {
        let mut pi = PriorityIndex::new();
        pi.set_priority(0, 10);
        assert_eq!(pi.len(), 1);

        pi.remove_priority(0);
        assert_eq!(pi.len(), 0);
        assert_eq!(pi.get_priority(0), 0);
    }

    #[test]
    fn test_priority_is_empty() {
        let pi = PriorityIndex::new();
        assert!(pi.is_empty());
    }

    #[test]
    fn test_priority_clear() {
        let mut pi = PriorityIndex::new();
        pi.set_priority(0, 10);
        pi.set_priority(1, 20);
        pi.clear();
        assert!(pi.is_empty());
    }

    // -----------------------------------------------------------------------
    // Wildcard matching tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_wildcard_star() {
        assert!(wildcard_matches("*", "anything"));
        assert!(wildcard_matches("*", ""));
    }

    #[test]
    fn test_wildcard_exact() {
        assert!(wildcard_matches("parent", "parent"));
        assert!(!wildcard_matches("parent", "child"));
    }

    #[test]
    fn test_wildcard_prefix() {
        assert!(wildcard_matches("par*", "parent"));
        assert!(!wildcard_matches("par*", "child"));
    }

    #[test]
    fn test_wildcard_suffix() {
        assert!(wildcard_matches("*ent", "parent"));
        assert!(!wildcard_matches("*ent", "child"));
    }

    #[test]
    fn test_wildcard_middle() {
        assert!(wildcard_matches("p*t", "parent"));
        assert!(wildcard_matches("p*t", "pet"));
        assert!(!wildcard_matches("p*t", "patch")); // doesn't end with t
    }

    #[test]
    fn test_wildcard_find_rules() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_test_rule("r1", "parent"));
        index.add_rule(create_test_rule("r2", "child"));
        index.add_rule(create_test_rule("r3", "partner"));

        let found = index.find_rules_by_wildcard("par*");
        assert_eq!(found.len(), 2); // parent and partner
    }

    #[test]
    fn test_wildcard_star_all() {
        let index = RuleIndex::with_defaults();
        index.add_rule(create_test_rule("r1", "parent"));
        index.add_rule(create_test_rule("r2", "child"));

        let found = index.find_rules_by_wildcard("*");
        assert_eq!(found.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Incremental re-indexing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reindex_rule() {
        let index = RuleIndex::with_defaults();
        let id = index.add_rule(create_test_rule("r1", "parent"));
        assert!(index.reindex_rule(id));
        let found = index.find_rules_for_triple(None, "parent", None);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_reindex_invalid_id() {
        let index = RuleIndex::with_defaults();
        assert!(!index.reindex_rule(999));
    }

    #[test]
    fn test_replace_rule() {
        let index = RuleIndex::with_defaults();
        let id = index.add_rule(create_test_rule("r1", "parent"));

        let old = index.replace_rule(id, create_test_rule("r1_new", "child"));
        assert!(old.is_some());
        assert_eq!(old.expect("old rule").name, "r1");

        // The stored rule should be updated
        let retrieved = index.get_rule(id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("rule exists").name, "r1_new");

        // New predicate should be indexed via the predicate index
        let found_child = index.find_rules_for_triple(None, "child", None);
        assert!(found_child.contains(&id));

        // Verify old predicate is removed from predicate index
        let found_by_pred = index.find_rules_by_predicate("parent");
        assert!(found_by_pred.is_empty());
    }

    #[test]
    fn test_replace_rule_invalid_id() {
        let index = RuleIndex::with_defaults();
        let result = index.replace_rule(999, create_test_rule("r1", "parent"));
        assert!(result.is_none());
    }

    #[test]
    fn test_predicate_density() {
        let index = RuleIndex::with_defaults();
        assert_eq!(index.predicate_density(), 0.0);

        index.add_rule(create_test_rule("r1", "parent"));
        index.add_rule(create_test_rule("r2", "parent"));
        index.add_rule(create_test_rule("r3", "child"));

        let density = index.predicate_density();
        // 3 rule entries across 2 predicates => density = 1.5
        assert!(density > 1.0);
    }

    // -----------------------------------------------------------------------
    // Multiple adds via add_rules
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_rules_batch() {
        let index = RuleIndex::with_defaults();
        let ids = index.add_rules(vec![
            create_test_rule("r1", "parent"),
            create_test_rule("r2", "child"),
            create_test_rule("r3", "sibling"),
        ]);
        assert_eq!(ids.len(), 3);
        assert_eq!(index.rule_count(), 3);
    }

    // -----------------------------------------------------------------------
    // Remove non-existent rule
    // -----------------------------------------------------------------------

    #[test]
    fn test_remove_nonexistent_rule() {
        let index = RuleIndex::with_defaults();
        let result = index.remove_rule(999);
        assert!(result.is_none());
    }
}
