//! SPARQL Query Integration
//!
//! Provides integration hooks between SPARQL query processing and rule-based reasoning.
//! Enables query-driven reasoning where SPARQL queries can trigger rule inference.
//!
//! # Features
//!
//! - **Query-Driven Reasoning**: Trigger rules based on SPARQL query patterns
//! - **Incremental Materialization**: Only materialize facts needed for query answering
//! - **Query Rewriting**: Rewrite queries to leverage derived facts
//! - **Backward Chaining Integration**: Use backward chaining for SPARQL ASK queries
//! - **Rule-Aware Optimization**: Optimize queries based on available rules
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::sparql_integration::{SparqlRuleIntegration, QueryMode};
//! use oxirs_rule::RuleEngine;
//!
//! let mut engine = RuleEngine::new();
//! let integration = SparqlRuleIntegration::new(engine);
//!
//! // Execute query with rule-based reasoning
//! // let results = integration.query_with_reasoning("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, RuleEngine, Term};
use anyhow::Result;
use scirs2_core::metrics::Timer;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

// Global metrics for query performance
lazy_static::lazy_static! {
    static ref QUERY_DIRECT_TIMER: Timer = Timer::new("sparql_query_direct".to_string());
    static ref QUERY_FORWARD_TIMER: Timer = Timer::new("sparql_query_forward".to_string());
    static ref QUERY_BACKWARD_TIMER: Timer = Timer::new("sparql_query_backward".to_string());
}

/// Query execution mode
#[derive(Debug, Clone, PartialEq)]
pub enum QueryMode {
    /// Execute query without reasoning
    Direct,
    /// Execute query with forward chaining first
    ForwardReasoning,
    /// Execute query with backward chaining (goal-driven)
    BackwardReasoning,
    /// Execute query with hybrid reasoning (forward + backward)
    Hybrid,
    /// Execute query with lazy materialization
    LazyMaterialization,
}

/// Query pattern for triggering rules
#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// Subject pattern (variable or constant)
    pub subject: Option<String>,
    /// Predicate pattern
    pub predicate: Option<String>,
    /// Object pattern
    pub object: Option<String>,
}

impl QueryPattern {
    /// Create a new query pattern
    pub fn new(subject: Option<String>, predicate: Option<String>, object: Option<String>) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Check if pattern matches an atom
    pub fn matches(&self, atom: &RuleAtom) -> bool {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = atom
        {
            self.matches_term(&self.subject, subject)
                && self.matches_term(&self.predicate, predicate)
                && self.matches_term(&self.object, object)
        } else {
            false
        }
    }

    fn matches_term(&self, pattern: &Option<String>, term: &Term) -> bool {
        match pattern {
            None => true, // Wildcard matches anything
            Some(pat) => match term {
                Term::Constant(c) => c == pat,
                Term::Literal(l) => l == pat,
                Term::Variable(_) => true, // Variables match pattern
                _ => false,
            },
        }
    }
}

/// SPARQL-Rule integration manager
pub struct SparqlRuleIntegration {
    /// Underlying rule engine
    engine: RuleEngine,
    /// Query mode
    mode: QueryMode,
    /// Pattern-to-rule mappings
    pattern_rules: HashMap<String, Vec<String>>,
    /// Query statistics
    stats: IntegrationStats,
    /// Cached materialized facts (optimization)
    materialized_cache: Option<Vec<RuleAtom>>,
    /// Hash of facts when cache was created
    facts_hash: u64,
}

impl SparqlRuleIntegration {
    /// Create new integration
    pub fn new(engine: RuleEngine) -> Self {
        Self {
            engine,
            mode: QueryMode::Hybrid,
            pattern_rules: HashMap::new(),
            stats: IntegrationStats::default(),
            materialized_cache: None,
            facts_hash: 0,
        }
    }

    /// Set query execution mode
    pub fn set_mode(&mut self, mode: QueryMode) {
        info!("Setting query mode to {:?}", mode);
        self.mode = mode;
    }

    /// Get current mode
    pub fn get_mode(&self) -> &QueryMode {
        &self.mode
    }

    /// Register a pattern-triggered rule
    pub fn register_pattern_rule(&mut self, pattern: String, rule_name: String) {
        debug!("Registering rule '{}' for pattern '{}'", rule_name, pattern);
        self.pattern_rules
            .entry(pattern)
            .or_default()
            .push(rule_name);
    }

    /// Execute query with reasoning
    pub fn query_with_reasoning(&mut self, patterns: &[QueryPattern]) -> Result<Vec<RuleAtom>> {
        self.stats.total_queries += 1;

        match self.mode {
            QueryMode::Direct => self.query_direct(patterns),
            QueryMode::ForwardReasoning => self.query_with_forward(patterns),
            QueryMode::BackwardReasoning => self.query_with_backward(patterns),
            QueryMode::Hybrid => self.query_hybrid(patterns),
            QueryMode::LazyMaterialization => self.query_lazy(patterns),
        }
    }

    /// Direct query without reasoning (optimized with SIMD)
    fn query_direct(&self, patterns: &[QueryPattern]) -> Result<Vec<RuleAtom>> {
        let _timer = QUERY_DIRECT_TIMER.start();
        let facts = self.engine.get_facts();

        // Early return for no patterns
        if patterns.is_empty() {
            return Ok(Vec::new());
        }

        // Optimize for single pattern (common case)
        if patterns.len() == 1 {
            let pattern = &patterns[0];
            let results: Vec<RuleAtom> = facts
                .into_iter()
                .filter(|fact| pattern.matches(fact))
                .collect();

            // Use SIMD deduplication for large result sets
            if results.len() > 100 {
                use crate::simd_ops::BatchProcessor;
                let processor = BatchProcessor::default();
                return Ok(processor.deduplicate(results));
            }

            return Ok(results);
        }

        // For multiple patterns, use more efficient matching
        let mut results: Vec<RuleAtom> = facts
            .into_iter()
            .filter(|fact| {
                // Check each pattern (optimized with early termination)
                for pattern in patterns {
                    if pattern.matches(fact) {
                        return true;
                    }
                }
                false
            })
            .collect();

        // Use SIMD deduplication for large result sets
        if results.len() > 100 {
            use crate::simd_ops::SimdMatcher;
            let matcher = SimdMatcher::new();
            matcher.batch_deduplicate(&mut results);
        }

        Ok(results)
    }

    /// Query with forward chaining (optimized with caching)
    fn query_with_forward(&mut self, patterns: &[QueryPattern]) -> Result<Vec<RuleAtom>> {
        let _timer = QUERY_FORWARD_TIMER.start();

        // Use cached materialization for performance
        let materialized = self.get_materialized_facts()?;

        // Filter by patterns
        let results = materialized
            .into_iter()
            .filter(|fact| patterns.iter().any(|p| p.matches(fact)))
            .collect();

        self.stats.forward_reasoning_queries += 1;
        Ok(results)
    }

    /// Query with backward chaining
    fn query_with_backward(&mut self, patterns: &[QueryPattern]) -> Result<Vec<RuleAtom>> {
        let _timer = QUERY_BACKWARD_TIMER.start();
        let mut results = Vec::new();

        // Try to prove each pattern as a goal
        for pattern in patterns {
            // Convert pattern to concrete goal (if fully bound)
            if let Some(goal) = self.pattern_to_goal(pattern) {
                if self.engine.backward_chain(&goal)? {
                    results.push(goal);
                }
            } else {
                // Pattern has variables - need to enumerate bindings
                // For now, fall back to forward reasoning
                return self.query_with_forward(patterns);
            }
        }

        self.stats.backward_reasoning_queries += 1;
        Ok(results)
    }

    /// Hybrid query execution
    fn query_hybrid(&mut self, patterns: &[QueryPattern]) -> Result<Vec<RuleAtom>> {
        // Use forward reasoning for broad queries, backward for specific ones
        let has_variables = patterns
            .iter()
            .any(|p| p.subject.is_none() || p.predicate.is_none() || p.object.is_none());

        if has_variables {
            self.query_with_forward(patterns)
        } else {
            self.query_with_backward(patterns)
        }
    }

    /// Lazy materialization query
    fn query_lazy(&mut self, patterns: &[QueryPattern]) -> Result<Vec<RuleAtom>> {
        // Only materialize facts relevant to the query patterns
        let relevant_rules = self.find_relevant_rules(patterns);
        let facts = self.engine.get_facts();

        // Apply only relevant rules
        let mut results = facts.clone();
        for _rule_name in relevant_rules {
            // In a full implementation, we would selectively apply rules
            // For now, use forward chaining as baseline
            results = self.engine.forward_chain(&results)?;
        }

        // Filter by patterns
        let filtered = results
            .into_iter()
            .filter(|fact| patterns.iter().any(|p| p.matches(fact)))
            .collect();

        self.stats.lazy_queries += 1;
        Ok(filtered)
    }

    /// Find rules relevant to query patterns
    fn find_relevant_rules(&self, patterns: &[QueryPattern]) -> Vec<String> {
        let mut relevant = HashSet::new();

        for pattern in patterns {
            // Check if any registered patterns match
            for (pattern_str, rules) in &self.pattern_rules {
                // Simple string matching for now
                if let Some(pred) = &pattern.predicate {
                    if pattern_str.contains(pred) {
                        relevant.extend(rules.clone());
                    }
                }
            }
        }

        relevant.into_iter().collect()
    }

    /// Convert query pattern to concrete goal
    fn pattern_to_goal(&self, pattern: &QueryPattern) -> Option<RuleAtom> {
        if pattern.subject.is_some() && pattern.predicate.is_some() && pattern.object.is_some() {
            Some(RuleAtom::Triple {
                subject: Term::Constant(
                    pattern
                        .subject
                        .clone()
                        .expect("subject verified to be Some"),
                ),
                predicate: Term::Constant(
                    pattern
                        .predicate
                        .clone()
                        .expect("predicate verified to be Some"),
                ),
                object: Term::Constant(pattern.object.clone().expect("object verified to be Some")),
            })
        } else {
            None
        }
    }

    /// Get underlying engine (mutable)
    pub fn engine_mut(&mut self) -> &mut RuleEngine {
        &mut self.engine
    }

    /// Get underlying engine (immutable)
    pub fn engine(&self) -> &RuleEngine {
        &self.engine
    }

    /// Get integration statistics
    pub fn get_stats(&self) -> &IntegrationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = IntegrationStats::default();
    }

    /// Compute hash of facts for cache invalidation
    fn compute_facts_hash(&self, facts: &[RuleAtom]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        facts.len().hash(&mut hasher);

        // Hash first and last few facts for performance
        // (hashing all facts would be too expensive for large knowledge bases)
        let sample_size = facts.len().min(10);
        for fact in facts.iter().take(sample_size) {
            // Hash the fact structure (simplified - in production we'd use proper serialization)
            format!("{:?}", fact).hash(&mut hasher);
        }
        if facts.len() > sample_size {
            for fact in facts.iter().skip(facts.len() - sample_size) {
                format!("{:?}", fact).hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Invalidate materialization cache
    pub fn invalidate_cache(&mut self) {
        self.materialized_cache = None;
        self.facts_hash = 0;
        debug!("Materialization cache invalidated");
    }

    /// Get or compute materialized facts with caching
    fn get_materialized_facts(&mut self) -> Result<Vec<RuleAtom>> {
        let facts = self.engine.get_facts();
        let current_hash = self.compute_facts_hash(&facts);

        // Check if cache is valid
        if let Some(ref cached) = self.materialized_cache {
            if current_hash == self.facts_hash {
                debug!("Using cached materialized facts ({} facts)", cached.len());
                self.stats.cache_hits += 1;
                return Ok(cached.clone());
            }
        }

        // Cache miss - compute materialization
        debug!("Cache miss - materializing facts");
        self.stats.cache_misses += 1;

        let materialized = self.engine.forward_chain(&facts)?;

        // Update cache
        self.materialized_cache = Some(materialized.clone());
        self.facts_hash = current_hash;

        Ok(materialized)
    }
}

/// Integration statistics
#[derive(Debug, Clone, Default)]
pub struct IntegrationStats {
    /// Total queries executed
    pub total_queries: usize,
    /// Queries using forward reasoning
    pub forward_reasoning_queries: usize,
    /// Queries using backward reasoning
    pub backward_reasoning_queries: usize,
    /// Queries using lazy materialization
    pub lazy_queries: usize,
    /// Materialization cache hits
    pub cache_hits: usize,
    /// Materialization cache misses
    pub cache_misses: usize,
}

impl std::fmt::Display for IntegrationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Total: {}, Forward: {}, Backward: {}, Lazy: {}, Cache(hits/misses): {}/{}",
            self.total_queries,
            self.forward_reasoning_queries,
            self.backward_reasoning_queries,
            self.lazy_queries,
            self.cache_hits,
            self.cache_misses
        )
    }
}

/// Query rewriter for rule-aware optimization
pub struct QueryRewriter {
    /// Available rules
    rules: Vec<Rule>,
    /// Rewrite statistics
    rewrites: usize,
}

impl QueryRewriter {
    /// Create new query rewriter
    pub fn new(rules: Vec<Rule>) -> Self {
        Self { rules, rewrites: 0 }
    }

    /// Analyze if query can benefit from rewriting
    pub fn can_rewrite(&self, patterns: &[QueryPattern]) -> bool {
        // Check if any rule heads match query patterns
        for pattern in patterns {
            for rule in &self.rules {
                if self.rule_derives_pattern(rule, pattern) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if rule can derive a pattern
    fn rule_derives_pattern(&self, rule: &Rule, pattern: &QueryPattern) -> bool {
        rule.head.iter().any(|atom| pattern.matches(atom))
    }

    /// Rewrite query to leverage derived facts
    pub fn rewrite(&mut self, patterns: Vec<QueryPattern>) -> Vec<QueryPattern> {
        // In a full implementation, we would:
        // 1. Identify which patterns can be derived from rules
        // 2. Replace those patterns with rule body patterns
        // 3. Add unions for both direct and derived facts
        //
        // For now, return patterns unchanged
        self.rewrites += 1;
        patterns
    }

    /// Get rewrite statistics
    pub fn get_rewrite_count(&self) -> usize {
        self.rewrites
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_pattern_matching() {
        let pattern = QueryPattern::new(
            Some("john".to_string()),
            Some("knows".to_string()),
            None, // Wildcard
        );

        let atom = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("knows".to_string()),
            object: Term::Constant("mary".to_string()),
        };

        assert!(pattern.matches(&atom));
    }

    #[test]
    fn test_sparql_integration_creation() {
        let engine = RuleEngine::new();
        let integration = SparqlRuleIntegration::new(engine);

        assert_eq!(*integration.get_mode(), QueryMode::Hybrid);
    }

    #[test]
    fn test_query_mode_setting() {
        let engine = RuleEngine::new();
        let mut integration = SparqlRuleIntegration::new(engine);

        integration.set_mode(QueryMode::ForwardReasoning);
        assert_eq!(*integration.get_mode(), QueryMode::ForwardReasoning);
    }

    #[test]
    fn test_pattern_rule_registration() {
        let engine = RuleEngine::new();
        let mut integration = SparqlRuleIntegration::new(engine);

        integration.register_pattern_rule("?s rdf:type ?o".to_string(), "typing_rule".to_string());

        assert_eq!(integration.pattern_rules.len(), 1);
    }

    #[test]
    fn test_direct_query() {
        let mut engine = RuleEngine::new();
        engine.add_fact(RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("knows".to_string()),
            object: Term::Constant("mary".to_string()),
        });

        let mut integration = SparqlRuleIntegration::new(engine);
        integration.set_mode(QueryMode::Direct);

        let patterns = vec![QueryPattern::new(
            Some("john".to_string()),
            Some("knows".to_string()),
            None,
        )];

        let results = integration.query_with_reasoning(&patterns).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_rewriter() {
        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("derived".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        };

        let rewriter = QueryRewriter::new(vec![rule]);

        let pattern = QueryPattern::new(None, Some("derived".to_string()), None);

        assert!(rewriter.can_rewrite(&[pattern]));
    }

    #[test]
    fn test_integration_stats() {
        let engine = RuleEngine::new();
        let mut integration = SparqlRuleIntegration::new(engine);

        let patterns = vec![QueryPattern::new(None, None, None)];
        integration.set_mode(QueryMode::ForwardReasoning);
        let _ = integration.query_with_reasoning(&patterns);

        let stats = integration.get_stats();
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.forward_reasoning_queries, 1);
    }
}
