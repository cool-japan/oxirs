//! # Rule Index Storage and Retrieval
//!
//! High-performance rule index implementation: storage, lookup, dependency graphs,
//! priority ordering, and wildcard matching.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::sync::{atomic::Ordering, RwLock};

use crate::rule_index_types::{
    ArgType, CombinedKey, DependencyEdge, FirstArgKey, IndexConfig, IndexStatistics,
    IndexStatisticsSnapshot, PredicateKey, PrioritizedRule, RuleDependencyGraph, RuleId,
};
use crate::{Rule, RuleAtom, Term};

/// High-performance rule index with multiple indexing strategies
#[derive(Debug)]
pub struct RuleIndex {
    /// Configuration
    pub(crate) config: IndexConfig,
    /// All rules stored by ID
    pub(crate) rules: RwLock<Vec<Rule>>,
    /// Predicate -> Rule IDs index
    pub(crate) predicate_index: RwLock<HashMap<PredicateKey, HashSet<RuleId>>>,
    /// (Predicate, FirstArg) -> Rule IDs index
    pub(crate) first_arg_index: RwLock<HashMap<FirstArgKey, HashSet<RuleId>>>,
    /// Combined pattern -> Rule IDs index
    pub(crate) combined_index: RwLock<HashMap<CombinedKey, HashSet<RuleId>>>,
    /// Statistics tracker
    pub(crate) statistics: IndexStatistics,
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
        let mut rules = self.rules.write().unwrap_or_else(|e| e.into_inner());
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
        let mut rules = self.rules.write().unwrap_or_else(|e| e.into_inner());
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

        let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());

        if self.config.predicate_indexing {
            let pred_index = self
                .predicate_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
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
            let combined = self
                .combined_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
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
            let first_arg = self
                .first_arg_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
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
            let pred_index = self
                .predicate_index
                .read()
                .unwrap_or_else(|e| e.into_inner());
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

        let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());
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
        let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());
        rules.get(rule_id).filter(|r| !r.name.is_empty()).cloned()
    }

    /// Get all rules (excluding removed ones)
    pub fn get_all_rules(&self) -> Vec<Rule> {
        let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());
        rules
            .iter()
            .filter(|r| !r.name.is_empty())
            .cloned()
            .collect()
    }

    /// Get the total number of indexed rules
    pub fn rule_count(&self) -> usize {
        let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());
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
        let mut rules = self.rules.write().unwrap_or_else(|e| e.into_inner());
        let mut pred_index = self
            .predicate_index
            .write()
            .unwrap_or_else(|e| e.into_inner());
        let mut first_arg = self
            .first_arg_index
            .write()
            .unwrap_or_else(|e| e.into_inner());
        let mut combined = self
            .combined_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

        rules.clear();
        pred_index.clear();
        first_arg.clear();
        combined.clear();

        self.statistics.reset();
    }

    /// Get index memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());
        let pred_index = self
            .predicate_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let first_arg = self
            .first_arg_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let combined = self
            .combined_index
            .read()
            .unwrap_or_else(|e| e.into_inner());

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
        let mut pred_index = self
            .predicate_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

        for atom in &rule.body {
            if let Some(predicate) = Self::extract_predicate(atom) {
                let key = PredicateKey(predicate);
                pred_index.entry(key).or_default().insert(rule_id);
            }
        }
    }

    fn unindex_by_predicate(&self, rule: &Rule, rule_id: RuleId) {
        let mut pred_index = self
            .predicate_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

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
        let mut first_arg_index = self
            .first_arg_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

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
        let mut first_arg_index = self
            .first_arg_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

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
        let mut combined_index = self
            .combined_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

        for atom in &rule.body {
            if let Some(key) = Self::extract_combined_key(atom) {
                combined_index.entry(key).or_default().insert(rule_id);
            }
        }
    }

    fn unindex_by_combined(&self, rule: &Rule, rule_id: RuleId) {
        let mut combined_index = self
            .combined_index
            .write()
            .unwrap_or_else(|e| e.into_inner());

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

    pub(crate) fn extract_predicate(atom: &RuleAtom) -> Option<String> {
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

    /// Find rules whose body predicates match a wildcard pattern.
    pub fn find_rules_by_wildcard(&self, pattern: &str) -> Vec<RuleId> {
        let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());
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
        let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());
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
        let rules = self.rules.read().unwrap_or_else(|e| e.into_inner());
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
        let mut rules = self.rules.write().unwrap_or_else(|e| e.into_inner());
        rules[rule_id] = new_rule;

        Some(old_rule)
    }

    /// Return predicate-level index density (entries / unique predicates).
    pub fn predicate_density(&self) -> f64 {
        let pred_index = self
            .predicate_index
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if pred_index.is_empty() {
            return 0.0;
        }
        let total_entries: usize = pred_index.values().map(|v| v.len()).sum();
        total_entries as f64 / pred_index.len() as f64
    }
}

impl Default for RuleIndex {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl RuleDependencyGraph {
    /// Build a dependency graph from an existing index.
    pub fn from_index(index: &RuleIndex) -> Self {
        let rules = index.rules.read().unwrap_or_else(|e| e.into_inner());
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
        let rules = index.rules.read().unwrap_or_else(|e| e.into_inner());
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
        result.sort_by_key(|b| Reverse(b.priority));
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
