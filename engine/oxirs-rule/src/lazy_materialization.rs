//! # Query-Driven Lazy Materialization Module
//!
//! This module provides on-demand materialization that only computes facts
//! needed to answer specific queries, minimizing unnecessary computation and
//! memory usage for large knowledge graphs.
//!
//! ## Features
//!
//! - **Query Pattern Analysis**: Analyzes queries to determine required inferences
//! - **Selective Materialization**: Only materializes facts relevant to queries
//! - **Dependency Tracking**: Tracks which rules can derive query patterns
//! - **Incremental Computation**: Reuses previously materialized facts
//! - **Cache Management**: Smart caching with eviction policies
//! - **Performance Optimization**: Reduces memory footprint and computation time
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::lazy_materialization::*;
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! // Create a lazy materializer
//! let mut materializer = LazyMaterializer::new();
//!
//! // Add rules
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
//! materializer.add_rule(rule);
//!
//! // Add base facts
//! let facts = vec![RuleAtom::Triple {
//!     subject: Term::Constant("john".to_string()),
//!     predicate: Term::Constant("parent".to_string()),
//!     object: Term::Constant("mary".to_string()),
//! }];
//!
//! materializer.add_facts(facts);
//!
//! // Query - only materializes what's needed
//! let query = RuleAtom::Triple {
//!     subject: Term::Constant("john".to_string()),
//!     predicate: Term::Constant("ancestor".to_string()),
//!     object: Term::Variable("Z".to_string()),
//! };
//!
//! let results = materializer.query(&query).unwrap();
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Query pattern for matching against cached facts
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryPattern {
    /// Predicate (None = wildcard)
    pub predicate: Option<String>,
    /// Subject type (constant, variable, any)
    pub subject_type: PatternType,
    /// Object type (constant, variable, any)
    pub object_type: PatternType,
}

/// Type of term in a pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatternType {
    Constant(String),
    Variable,
    Any,
}

impl QueryPattern {
    /// Extract pattern from a query atom
    pub fn from_atom(atom: &RuleAtom) -> Self {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let pred = match predicate {
                    Term::Constant(p) => Some(p.clone()),
                    _ => None,
                };

                let subj_type = match subject {
                    Term::Constant(s) => PatternType::Constant(s.clone()),
                    Term::Variable(_) => PatternType::Variable,
                    _ => PatternType::Any,
                };

                let obj_type = match object {
                    Term::Constant(o) => PatternType::Constant(o.clone()),
                    Term::Variable(_) => PatternType::Variable,
                    _ => PatternType::Any,
                };

                QueryPattern {
                    predicate: pred,
                    subject_type: subj_type,
                    object_type: obj_type,
                }
            }
            _ => QueryPattern {
                predicate: None,
                subject_type: PatternType::Any,
                object_type: PatternType::Any,
            },
        }
    }

    /// Check if this pattern matches an atom
    pub fn matches(&self, atom: &RuleAtom) -> bool {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = atom
        {
            // Check predicate
            if let Some(ref p) = self.predicate {
                if let Term::Constant(pred_val) = predicate {
                    if p != pred_val {
                        return false;
                    }
                } else {
                    return false;
                }
            }

            // Check subject type
            match &self.subject_type {
                PatternType::Constant(s) => {
                    if let Term::Constant(subj_val) = subject {
                        if s != subj_val {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                PatternType::Variable => {
                    // Variable matches anything
                }
                PatternType::Any => {}
            }

            // Check object type
            match &self.object_type {
                PatternType::Constant(o) => {
                    if let Term::Constant(obj_val) = object {
                        if o != obj_val {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                PatternType::Variable => {
                    // Variable matches anything
                }
                PatternType::Any => {}
            }

            true
        } else {
            false
        }
    }
}

/// Materialization cache entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached facts for this pattern
    facts: Vec<RuleAtom>,
    /// Timestamp of last materialization
    timestamp: std::time::Instant,
    /// Rules used to derive these facts
    #[allow(dead_code)]
    derived_by: HashSet<String>,
}

/// Query-driven lazy materializer
pub struct LazyMaterializer {
    /// Base facts (always available)
    base_facts: Vec<RuleAtom>,
    /// Rules for inference
    rules: Vec<Rule>,
    /// Cache of materialized facts by pattern
    cache: HashMap<QueryPattern, CacheEntry>,
    /// Index: predicate -> rules that can derive it
    predicate_index: HashMap<String, Vec<usize>>,
    /// Cache eviction policy
    max_cache_size: usize,
    /// Current cache size
    cache_size: usize,
    /// Cache statistics
    cache_hits: usize,
    cache_misses: usize,
    query_count: usize,
}

impl LazyMaterializer {
    /// Create a new lazy materializer
    pub fn new() -> Self {
        Self {
            base_facts: Vec::new(),
            rules: Vec::new(),
            cache: HashMap::new(),
            predicate_index: HashMap::new(),
            max_cache_size: 10000,
            cache_size: 0,
            cache_hits: 0,
            cache_misses: 0,
            query_count: 0,
        }
    }

    /// Create with custom cache size
    pub fn with_cache_size(max_size: usize) -> Self {
        let mut result = Self::new();
        result.max_cache_size = max_size;
        result
    }

    /// Add a rule to the materializer
    pub fn add_rule(&mut self, rule: Rule) {
        let rule_idx = self.rules.len();

        // Index by predicates in head
        for head_atom in &rule.head {
            if let RuleAtom::Triple {
                predicate: Term::Constant(pred),
                ..
            } = head_atom
            {
                self.predicate_index
                    .entry(pred.clone())
                    .or_default()
                    .push(rule_idx);
            }
        }

        self.rules.push(rule);
    }

    /// Add base facts
    pub fn add_facts(&mut self, facts: Vec<RuleAtom>) {
        self.base_facts.extend(facts);
        // Invalidate cache when base facts change
        self.invalidate_cache();
    }

    /// Query with lazy materialization
    pub fn query(&mut self, query: &RuleAtom) -> Result<Vec<RuleAtom>> {
        self.query_count += 1;

        // Extract query pattern
        let pattern = QueryPattern::from_atom(query);

        // Check cache first
        if let Some(entry) = self.cache.get(&pattern) {
            self.cache_hits += 1;
            // Filter cached facts by query
            let results: Vec<_> = entry
                .facts
                .iter()
                .filter(|fact| self.matches_query(query, fact))
                .cloned()
                .collect();
            return Ok(results);
        }

        self.cache_misses += 1;

        // Materialize on demand
        let materialized = self.materialize_for_pattern(&pattern)?;

        // Cache results
        self.cache_materialized_facts(pattern, materialized.clone());

        // Filter by specific query
        let results: Vec<_> = materialized
            .into_iter()
            .filter(|fact| self.matches_query(query, fact))
            .collect();

        Ok(results)
    }

    /// Materialize facts for a specific pattern
    fn materialize_for_pattern(&self, pattern: &QueryPattern) -> Result<Vec<RuleAtom>> {
        let mut results = Vec::new();

        // Start with ALL base facts (not just those matching pattern)
        // because rule bodies might need facts with different predicates
        let mut working_set = self.base_facts.clone();

        // Find applicable rules
        let applicable_rules = self.find_applicable_rules(pattern);

        // Apply rules iteratively (fixed-point computation)
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for &rule_idx in &applicable_rules {
                let rule = &self.rules[rule_idx];
                let new_facts = self.apply_rule_to_facts(rule, &working_set);

                for new_fact in new_facts {
                    if !working_set.contains(&new_fact) {
                        working_set.push(new_fact.clone());
                        changed = true;
                    }
                }
            }
        }

        // Filter results to match pattern
        for fact in &working_set {
            if pattern.matches(fact) {
                results.push(fact.clone());
            }
        }

        Ok(results)
    }

    /// Find rules that can derive facts matching the pattern
    fn find_applicable_rules(&self, pattern: &QueryPattern) -> Vec<usize> {
        if let Some(ref pred) = pattern.predicate {
            // Use index to find rules that derive this predicate
            self.predicate_index.get(pred).cloned().unwrap_or_default()
        } else {
            // No predicate specified - all rules might be applicable
            (0..self.rules.len()).collect()
        }
    }

    /// Apply a rule to a set of facts (simplified forward chaining)
    fn apply_rule_to_facts(&self, rule: &Rule, facts: &[RuleAtom]) -> Vec<RuleAtom> {
        let mut new_facts = Vec::new();

        // Simple case: single atom in body
        if rule.body.len() == 1 {
            for fact in facts {
                if let Some(substitution) = self.try_match(&rule.body[0], fact) {
                    // Apply substitution to head
                    for head_atom in &rule.head {
                        if let Some(derived) = self.apply_substitution(head_atom, &substitution) {
                            new_facts.push(derived);
                        }
                    }
                }
            }
        }

        new_facts
    }

    /// Try to match a rule atom against a fact
    fn try_match(&self, pattern: &RuleAtom, fact: &RuleAtom) -> Option<HashMap<String, Term>> {
        if let (
            RuleAtom::Triple {
                subject: ps,
                predicate: pp,
                object: po,
            },
            RuleAtom::Triple {
                subject: fs,
                predicate: fp,
                object: fo,
            },
        ) = (pattern, fact)
        {
            let mut sub = HashMap::new();

            // Match subject
            if !self.unify_term(ps, fs, &mut sub) {
                return None;
            }

            // Match predicate
            if !self.unify_term(pp, fp, &mut sub) {
                return None;
            }

            // Match object
            if !self.unify_term(po, fo, &mut sub) {
                return None;
            }

            Some(sub)
        } else {
            None
        }
    }

    /// Simple unification for terms
    fn unify_term(&self, pattern: &Term, fact: &Term, sub: &mut HashMap<String, Term>) -> bool {
        match (pattern, fact) {
            (Term::Variable(var), _) => {
                if let Some(bound) = sub.get(var) {
                    bound == fact
                } else {
                    sub.insert(var.clone(), fact.clone());
                    true
                }
            }
            (Term::Constant(p), Term::Constant(f)) => p == f,
            _ => false,
        }
    }

    /// Apply substitution to an atom
    fn apply_substitution(&self, atom: &RuleAtom, sub: &HashMap<String, Term>) -> Option<RuleAtom> {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = atom
        {
            let new_subj = self.substitute_term(subject, sub);
            let new_pred = self.substitute_term(predicate, sub);
            let new_obj = self.substitute_term(object, sub);

            Some(RuleAtom::Triple {
                subject: new_subj,
                predicate: new_pred,
                object: new_obj,
            })
        } else {
            None
        }
    }

    /// Substitute variables in a term
    fn substitute_term(&self, term: &Term, sub: &HashMap<String, Term>) -> Term {
        match term {
            Term::Variable(var) => sub.get(var).cloned().unwrap_or_else(|| term.clone()),
            _ => term.clone(),
        }
    }

    /// Check if a fact matches a query
    fn matches_query(&self, query: &RuleAtom, fact: &RuleAtom) -> bool {
        if let (
            RuleAtom::Triple {
                subject: qs,
                predicate: qp,
                object: qo,
            },
            RuleAtom::Triple {
                subject: fs,
                predicate: fp,
                object: fo,
            },
        ) = (query, fact)
        {
            self.term_matches(qs, fs) && self.term_matches(qp, fp) && self.term_matches(qo, fo)
        } else {
            false
        }
    }

    /// Check if a query term matches a fact term
    fn term_matches(&self, query_term: &Term, fact_term: &Term) -> bool {
        match (query_term, fact_term) {
            (Term::Variable(_), _) => true, // Variable matches anything
            (Term::Constant(q), Term::Constant(f)) => q == f,
            _ => false,
        }
    }

    /// Cache materialized facts
    fn cache_materialized_facts(&mut self, pattern: QueryPattern, facts: Vec<RuleAtom>) {
        // Check cache size limit
        if self.cache_size >= self.max_cache_size {
            self.evict_cache_entry();
        }

        let entry = CacheEntry {
            facts: facts.clone(),
            timestamp: std::time::Instant::now(),
            derived_by: HashSet::new(),
        };

        self.cache_size += facts.len();
        self.cache.insert(pattern, entry);
    }

    /// Evict oldest cache entry (LRU policy)
    fn evict_cache_entry(&mut self) {
        if let Some((oldest_pattern, oldest_entry)) = self
            .cache
            .iter()
            .min_by_key(|(_, entry)| entry.timestamp)
            .map(|(p, e)| (p.clone(), e.clone()))
        {
            self.cache_size = self.cache_size.saturating_sub(oldest_entry.facts.len());
            self.cache.remove(&oldest_pattern);
        }
    }

    /// Invalidate entire cache
    pub fn invalidate_cache(&mut self) {
        self.cache.clear();
        self.cache_size = 0;
    }

    /// Invalidate cache entries affected by a predicate
    pub fn invalidate_predicate(&mut self, predicate: &str) {
        let patterns_to_remove: Vec<_> = self
            .cache
            .keys()
            .filter(|p| {
                if let Some(ref pred) = p.predicate {
                    pred == predicate
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        for pattern in patterns_to_remove {
            if let Some(entry) = self.cache.remove(&pattern) {
                self.cache_size = self.cache_size.saturating_sub(entry.facts.len());
            }
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            cache_size: self.cache_size,
            num_patterns: self.cache.len(),
            max_size: self.max_cache_size,
            hit_rate: if self.cache_hits + self.cache_misses > 0 {
                self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for LazyMaterializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cache_size: usize,
    pub num_patterns: usize,
    pub max_size: usize,
    pub hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_rule() -> Rule {
        Rule {
            name: "ancestor".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }
    }

    #[test]
    fn test_lazy_materializer_creation() {
        let materializer = LazyMaterializer::new();
        assert_eq!(materializer.rules.len(), 0);
    }

    #[test]
    fn test_add_rule() {
        let mut materializer = LazyMaterializer::new();
        let rule = create_simple_rule();

        materializer.add_rule(rule);

        assert_eq!(materializer.rules.len(), 1);
        assert!(materializer.predicate_index.contains_key("ancestor"));
    }

    #[test]
    fn test_query_pattern_extraction() {
        let atom = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Variable("X".to_string()),
        };

        let pattern = QueryPattern::from_atom(&atom);

        assert_eq!(pattern.predicate, Some("parent".to_string()));
        assert_eq!(
            pattern.subject_type,
            PatternType::Constant("john".to_string())
        );
        assert_eq!(pattern.object_type, PatternType::Variable);
    }

    #[test]
    fn test_pattern_matching() {
        let pattern = QueryPattern {
            predicate: Some("parent".to_string()),
            subject_type: PatternType::Variable,
            object_type: PatternType::Variable,
        };

        let atom1 = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("mary".to_string()),
        };

        let atom2 = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("friend".to_string()),
            object: Term::Constant("bob".to_string()),
        };

        assert!(pattern.matches(&atom1));
        assert!(!pattern.matches(&atom2));
    }

    #[test]
    fn test_lazy_query() {
        let mut materializer = LazyMaterializer::new();

        // Add rule
        materializer.add_rule(create_simple_rule());

        // Add base facts
        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("mary".to_string()),
        }];

        materializer.add_facts(facts);

        // Query
        let query = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Variable("Z".to_string()),
        };

        let results = materializer.query(&query).unwrap();

        // Should derive ancestor(john, mary) from parent(john, mary)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_cache_hit() {
        let mut materializer = LazyMaterializer::new();

        materializer.add_rule(create_simple_rule());

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("mary".to_string()),
        }];

        materializer.add_facts(facts);

        let query = RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Variable("Y".to_string()),
        };

        // First query - cache miss
        let _results1 = materializer.query(&query).unwrap();

        // Second query - should hit cache
        let _results2 = materializer.query(&query).unwrap();

        let stats = materializer.cache_stats();
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_cache_invalidation() {
        let mut materializer = LazyMaterializer::new();

        materializer.add_rule(create_simple_rule());

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("mary".to_string()),
        }];

        materializer.add_facts(facts);

        let query = RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Variable("Y".to_string()),
        };

        // Query to populate cache
        let _results = materializer.query(&query).unwrap();

        // Invalidate cache
        materializer.invalidate_cache();

        let stats = materializer.cache_stats();
        assert_eq!(stats.cache_size, 0);
    }

    #[test]
    fn test_predicate_invalidation() {
        let mut materializer = LazyMaterializer::new();

        materializer.add_rule(create_simple_rule());

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("mary".to_string()),
        }];

        materializer.add_facts(facts);

        let query = RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Variable("Y".to_string()),
        };

        // Query to populate cache
        let _results = materializer.query(&query).unwrap();

        // Invalidate specific predicate
        materializer.invalidate_predicate("ancestor");

        let stats = materializer.cache_stats();
        assert_eq!(stats.cache_size, 0);
    }

    #[test]
    fn test_cache_eviction() {
        let mut materializer = LazyMaterializer::with_cache_size(10);

        materializer.add_rule(create_simple_rule());

        // Add enough facts to trigger eviction
        for i in 0..15 {
            let facts = vec![RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("parent".to_string()),
                object: Term::Constant(format!("entity_{}", i + 100)),
            }];

            materializer.add_facts(facts);

            let query = RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            };

            let _ = materializer.query(&query);
        }

        let stats = materializer.cache_stats();
        assert!(stats.cache_size <= 10);
    }
}
