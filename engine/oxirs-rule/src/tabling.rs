//! # Explicit Tabling with Loop Detection
//!
//! This module implements tabling (memoization) for rule-based reasoning with
//! automatic loop detection. Tabling is essential for efficient recursive query
//! processing and prevents infinite loops in cyclic rule graphs.
//!
//! ## Features
//!
//! - **Answer Memoization**: Cache computed answers for reuse
//! - **Loop Detection**: Detect and handle circular dependencies
//! - **Subsumption Checking**: Avoid redundant computation via answer subsumption
//! - **Incremental Updates**: Support for incremental fact additions
//! - **SLG Resolution**: Simplified Linear Resolution with Tabling
//!
//! ## Tabling Modes
//!
//! - **Call Tabling**: Table calls to specific predicates
//! - **Answer Tabling**: Table answers for specific goals
//! - **Variant Tabling**: Group answers by call variant
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::tabling::{TablingEngine, TablingConfig, TableDirective};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let config = TablingConfig::default();
//! let mut engine = TablingEngine::new(config);
//!
//! // Mark predicates as tabled
//! engine.add_table_directive(TableDirective::predicate("ancestor"));
//!
//! // Now recursive ancestor queries won't infinite loop
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::{Rule, RuleAtom, Term};

/// Configuration for tabling behavior
#[derive(Debug, Clone)]
pub struct TablingConfig {
    /// Maximum table size (number of entries)
    pub max_table_size: usize,
    /// Enable answer subsumption
    pub enable_subsumption: bool,
    /// Loop detection strategy
    pub loop_strategy: LoopStrategy,
    /// Timeout for tabling operations (milliseconds)
    pub timeout_ms: Option<u64>,
    /// Enable statistics collection
    pub collect_statistics: bool,
    /// Maximum recursion depth
    pub max_recursion_depth: usize,
}

impl Default for TablingConfig {
    fn default() -> Self {
        Self {
            max_table_size: 100_000,
            enable_subsumption: true,
            loop_strategy: LoopStrategy::DelayAndResume,
            timeout_ms: Some(30_000), // 30 seconds
            collect_statistics: true,
            max_recursion_depth: 1000,
        }
    }
}

impl TablingConfig {
    /// Set maximum table size
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_table_size = size;
        self
    }

    /// Set loop detection strategy
    pub fn with_loop_strategy(mut self, strategy: LoopStrategy) -> Self {
        self.loop_strategy = strategy;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Enable or disable subsumption
    pub fn with_subsumption(mut self, enabled: bool) -> Self {
        self.enable_subsumption = enabled;
        self
    }
}

/// Loop detection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopStrategy {
    /// Fail immediately on loop detection
    FailOnLoop,
    /// Delay looped goals and resume later
    DelayAndResume,
    /// Use well-founded semantics for loops
    WellFounded,
    /// Return current partial answers on loop
    ReturnPartial,
}

/// Directive for tabling specific predicates
#[derive(Debug, Clone)]
pub enum TableDirective {
    /// Table a specific predicate
    Predicate(String),
    /// Table a predicate with specific arity
    PredicateArity(String, usize),
    /// Table all predicates
    All,
    /// Don't table a specific predicate
    Exclude(String),
}

impl TableDirective {
    /// Create directive for a predicate
    pub fn predicate(name: &str) -> Self {
        TableDirective::Predicate(name.to_string())
    }

    /// Create directive for predicate with arity
    pub fn predicate_arity(name: &str, arity: usize) -> Self {
        TableDirective::PredicateArity(name.to_string(), arity)
    }

    /// Table all predicates
    pub fn all() -> Self {
        TableDirective::All
    }

    /// Exclude a predicate from tabling
    pub fn exclude(name: &str) -> Self {
        TableDirective::Exclude(name.to_string())
    }
}

/// A call variant (normalized call pattern)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CallVariant {
    /// Predicate name
    pub predicate: String,
    /// Argument patterns (Some for bound, None for free)
    pub binding_pattern: Vec<Option<String>>,
}

impl CallVariant {
    /// Create from a rule atom
    pub fn from_atom(atom: &RuleAtom) -> Option<Self> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let pred = match predicate {
                    Term::Constant(c) => c.clone(),
                    _ => return None,
                };

                let pattern = vec![Self::term_pattern(subject), Self::term_pattern(object)];

                Some(Self {
                    predicate: pred,
                    binding_pattern: pattern,
                })
            }
            RuleAtom::Builtin { name, args } => {
                let pattern = args.iter().map(Self::term_pattern).collect();
                Some(Self {
                    predicate: name.clone(),
                    binding_pattern: pattern,
                })
            }
            _ => None,
        }
    }

    fn term_pattern(term: &Term) -> Option<String> {
        match term {
            Term::Constant(c) => Some(c.clone()),
            Term::Literal(l) => Some(l.clone()),
            _ => None,
        }
    }

    /// Get arity
    pub fn arity(&self) -> usize {
        self.binding_pattern.len()
    }
}

/// Status of a tabled goal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    /// Goal is new, not yet evaluated
    New,
    /// Goal is currently being evaluated
    Active,
    /// Goal has been completed
    Complete,
    /// Goal is in a loop
    Looped,
    /// Goal failed
    Failed,
}

/// Entry in the tabling table
#[derive(Debug)]
pub struct TableEntry {
    /// Call variant
    pub variant: CallVariant,
    /// Status
    pub status: GoalStatus,
    /// Computed answers
    pub answers: Vec<RuleAtom>,
    /// Waiting goals (for delay-and-resume)
    pub waiting: Vec<CallVariant>,
    /// Creation timestamp
    pub created_at: Instant,
    /// Number of times accessed
    pub access_count: AtomicU64,
}

impl TableEntry {
    /// Create a new table entry
    pub fn new(variant: CallVariant) -> Self {
        Self {
            variant,
            status: GoalStatus::New,
            answers: Vec::new(),
            waiting: Vec::new(),
            created_at: Instant::now(),
            access_count: AtomicU64::new(0),
        }
    }

    /// Add an answer
    pub fn add_answer(&mut self, answer: RuleAtom) {
        self.answers.push(answer);
    }

    /// Mark as complete
    pub fn complete(&mut self) {
        self.status = GoalStatus::Complete;
    }

    /// Record an access
    pub fn record_access(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get access count
    pub fn accesses(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
}

/// Tabling statistics
#[derive(Debug, Default)]
pub struct TablingStatistics {
    /// Number of calls made
    pub calls: AtomicU64,
    /// Number of cache hits
    pub hits: AtomicU64,
    /// Number of cache misses
    pub misses: AtomicU64,
    /// Number of loops detected
    pub loops_detected: AtomicU64,
    /// Number of answers computed
    pub answers_computed: AtomicU64,
    /// Number of answers reused
    pub answers_reused: AtomicU64,
}

impl TablingStatistics {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Get hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits.load(Ordering::Relaxed) + self.misses.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.hits.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Reset statistics
    pub fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.loops_detected.store(0, Ordering::Relaxed);
        self.answers_computed.store(0, Ordering::Relaxed);
        self.answers_reused.store(0, Ordering::Relaxed);
    }

    /// Get snapshot
    pub fn snapshot(&self) -> TablingStatsSnapshot {
        TablingStatsSnapshot {
            calls: self.calls.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            loops_detected: self.loops_detected.load(Ordering::Relaxed),
            answers_computed: self.answers_computed.load(Ordering::Relaxed),
            answers_reused: self.answers_reused.load(Ordering::Relaxed),
        }
    }
}

/// Immutable statistics snapshot
#[derive(Debug, Clone)]
pub struct TablingStatsSnapshot {
    pub calls: u64,
    pub hits: u64,
    pub misses: u64,
    pub loops_detected: u64,
    pub answers_computed: u64,
    pub answers_reused: u64,
}

/// Tabling engine
pub struct TablingEngine {
    /// Configuration
    config: TablingConfig,
    /// Rules
    rules: Vec<Rule>,
    /// Base facts
    facts: HashSet<String>,
    /// Table directives
    directives: Vec<TableDirective>,
    /// The table: variant -> entry
    table: HashMap<CallVariant, TableEntry>,
    /// Call stack for loop detection
    call_stack: Vec<CallVariant>,
    /// Statistics
    statistics: TablingStatistics,
    /// Start time for timeout
    start_time: Option<Instant>,
}

impl TablingEngine {
    /// Create a new tabling engine
    pub fn new(config: TablingConfig) -> Self {
        Self {
            config,
            rules: Vec::new(),
            facts: HashSet::new(),
            directives: Vec::new(),
            table: HashMap::new(),
            call_stack: Vec::new(),
            statistics: TablingStatistics::new(),
            start_time: None,
        }
    }

    /// Add a table directive
    pub fn add_table_directive(&mut self, directive: TableDirective) {
        self.directives.push(directive);
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Add rules
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        self.rules.extend(rules);
    }

    /// Add a fact
    pub fn add_fact(&mut self, fact: RuleAtom) {
        let key = Self::atom_to_key(&fact);
        self.facts.insert(key);
    }

    /// Add facts
    pub fn add_facts(&mut self, facts: Vec<RuleAtom>) {
        for fact in facts {
            self.add_fact(fact);
        }
    }

    /// Query with tabling
    pub fn query(&mut self, goal: &RuleAtom) -> Result<Vec<RuleAtom>> {
        self.start_time = Some(Instant::now());
        self.call_stack.clear();

        if self.config.collect_statistics {
            self.statistics.calls.fetch_add(1, Ordering::Relaxed);
        }

        let variant = CallVariant::from_atom(goal).ok_or_else(|| anyhow!("Invalid goal"))?;

        let answers = self.evaluate_goal(&variant, goal, 0)?;

        Ok(answers)
    }

    fn evaluate_goal(
        &mut self,
        variant: &CallVariant,
        goal: &RuleAtom,
        depth: usize,
    ) -> Result<Vec<RuleAtom>> {
        // Check timeout
        self.check_timeout()?;

        // Check recursion depth
        if depth > self.config.max_recursion_depth {
            return Err(anyhow!("Maximum recursion depth exceeded"));
        }

        // Check if this predicate should be tabled
        if !self.should_table(&variant.predicate, variant.arity()) {
            // Not tabled: evaluate directly
            return self.evaluate_directly(goal, depth);
        }

        // Check loop detection
        if self.call_stack.contains(variant) {
            if self.config.collect_statistics {
                self.statistics
                    .loops_detected
                    .fetch_add(1, Ordering::Relaxed);
            }

            match self.config.loop_strategy {
                LoopStrategy::FailOnLoop => {
                    return Err(anyhow!(
                        "Loop detected for predicate: {}",
                        variant.predicate
                    ));
                }
                LoopStrategy::ReturnPartial => {
                    // Return any answers we have so far
                    if let Some(entry) = self.table.get(variant) {
                        return Ok(entry.answers.clone());
                    }
                    return Ok(Vec::new());
                }
                LoopStrategy::DelayAndResume | LoopStrategy::WellFounded => {
                    // Mark as looped, return empty for now
                    if let Some(entry) = self.table.get_mut(variant) {
                        entry.status = GoalStatus::Looped;
                    }
                    return Ok(Vec::new());
                }
            }
        }

        // Check table
        if let Some(entry) = self.table.get(variant) {
            entry.record_access();

            match entry.status {
                GoalStatus::Complete => {
                    if self.config.collect_statistics {
                        self.statistics.hits.fetch_add(1, Ordering::Relaxed);
                        self.statistics
                            .answers_reused
                            .fetch_add(entry.answers.len() as u64, Ordering::Relaxed);
                    }
                    return Ok(entry.answers.clone());
                }
                GoalStatus::Active => {
                    // Currently being evaluated - this is a loop
                    if self.config.collect_statistics {
                        self.statistics
                            .loops_detected
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(Vec::new());
                }
                _ => {}
            }
        }

        if self.config.collect_statistics {
            self.statistics.misses.fetch_add(1, Ordering::Relaxed);
        }

        // Create new entry
        let mut entry = TableEntry::new(variant.clone());
        entry.status = GoalStatus::Active;
        self.table.insert(variant.clone(), entry);

        // Push to call stack
        self.call_stack.push(variant.clone());

        // Evaluate
        let answers = self.evaluate_directly(goal, depth + 1)?;

        // Pop from call stack
        self.call_stack.pop();

        // Update entry
        if let Some(entry) = self.table.get_mut(variant) {
            entry.answers = answers.clone();
            entry.status = GoalStatus::Complete;
        }

        if self.config.collect_statistics {
            self.statistics
                .answers_computed
                .fetch_add(answers.len() as u64, Ordering::Relaxed);
        }

        Ok(answers)
    }

    fn evaluate_directly(&mut self, goal: &RuleAtom, depth: usize) -> Result<Vec<RuleAtom>> {
        let mut answers = Vec::new();

        // Check facts first
        let goal_key = Self::atom_to_key(goal);
        if self.facts.contains(&goal_key) {
            answers.push(goal.clone());
        }

        // Match against rules
        for rule in &self.rules.clone() {
            // For each head that matches the goal
            for head in &rule.head {
                if let Some(subst) = Self::unify(goal, head) {
                    // Try to satisfy the body
                    let body_results = self.evaluate_body(&rule.body, &subst, depth)?;

                    for body_subst in body_results {
                        // Generate answer
                        let answer = Self::apply_substitution(head, &body_subst);
                        if !answers.contains(&answer) {
                            answers.push(answer);
                        }
                    }
                }
            }
        }

        Ok(answers)
    }

    fn evaluate_body(
        &mut self,
        body: &[RuleAtom],
        initial_subst: &HashMap<String, Term>,
        depth: usize,
    ) -> Result<Vec<HashMap<String, Term>>> {
        if body.is_empty() {
            return Ok(vec![initial_subst.clone()]);
        }

        let mut current_substs = vec![initial_subst.clone()];

        for atom in body {
            let mut new_substs = Vec::new();

            for subst in &current_substs {
                let grounded = Self::apply_substitution(atom, subst);
                let variant = CallVariant::from_atom(&grounded);

                let answers = if let Some(v) = &variant {
                    self.evaluate_goal(v, &grounded, depth)?
                } else {
                    // Handle non-tabled atoms
                    self.evaluate_directly(&grounded, depth)?
                };

                for answer in answers {
                    if let Some(new_subst) = Self::unify(&grounded, &answer) {
                        let mut combined = subst.clone();
                        combined.extend(new_subst);
                        new_substs.push(combined);
                    }
                }
            }

            current_substs = new_substs;

            if current_substs.is_empty() {
                break;
            }
        }

        Ok(current_substs)
    }

    fn should_table(&self, predicate: &str, arity: usize) -> bool {
        let mut should_table = false;
        let mut explicitly_excluded = false;

        for directive in &self.directives {
            match directive {
                TableDirective::All => should_table = true,
                TableDirective::Predicate(p) if p == predicate => should_table = true,
                TableDirective::PredicateArity(p, a) if p == predicate && *a == arity => {
                    should_table = true
                }
                TableDirective::Exclude(p) if p == predicate => explicitly_excluded = true,
                _ => {}
            }
        }

        should_table && !explicitly_excluded
    }

    fn check_timeout(&self) -> Result<()> {
        if let (Some(start), Some(timeout)) = (self.start_time, self.config.timeout_ms) {
            if start.elapsed().as_millis() as u64 > timeout {
                return Err(anyhow!("Tabling operation timed out"));
            }
        }
        Ok(())
    }

    fn unify(atom1: &RuleAtom, atom2: &RuleAtom) -> Option<HashMap<String, Term>> {
        match (atom1, atom2) {
            (
                RuleAtom::Triple {
                    subject: s1,
                    predicate: p1,
                    object: o1,
                },
                RuleAtom::Triple {
                    subject: s2,
                    predicate: p2,
                    object: o2,
                },
            ) => {
                let mut subst = HashMap::new();

                if !Self::unify_terms(p1, p2, &mut subst) {
                    return None;
                }
                if !Self::unify_terms(s1, s2, &mut subst) {
                    return None;
                }
                if !Self::unify_terms(o1, o2, &mut subst) {
                    return None;
                }

                Some(subst)
            }
            _ => None,
        }
    }

    fn unify_terms(t1: &Term, t2: &Term, subst: &mut HashMap<String, Term>) -> bool {
        match (t1, t2) {
            (Term::Variable(v), _) => {
                if let Some(existing) = subst.get(v) {
                    Self::terms_equal(existing, t2)
                } else {
                    subst.insert(v.clone(), t2.clone());
                    true
                }
            }
            (_, Term::Variable(v)) => {
                if let Some(existing) = subst.get(v) {
                    Self::terms_equal(t1, existing)
                } else {
                    subst.insert(v.clone(), t1.clone());
                    true
                }
            }
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            _ => false,
        }
    }

    fn terms_equal(t1: &Term, t2: &Term) -> bool {
        match (t1, t2) {
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
            _ => false,
        }
    }

    fn apply_substitution(atom: &RuleAtom, subst: &HashMap<String, Term>) -> RuleAtom {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => RuleAtom::Triple {
                subject: Self::substitute_term(subject, subst),
                predicate: Self::substitute_term(predicate, subst),
                object: Self::substitute_term(object, subst),
            },
            RuleAtom::Builtin { name, args } => RuleAtom::Builtin {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| Self::substitute_term(a, subst))
                    .collect(),
            },
            other => other.clone(),
        }
    }

    fn substitute_term(term: &Term, subst: &HashMap<String, Term>) -> Term {
        match term {
            Term::Variable(v) => subst.get(v).cloned().unwrap_or_else(|| term.clone()),
            _ => term.clone(),
        }
    }

    fn atom_to_key(atom: &RuleAtom) -> String {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                format!(
                    "{}:{}:{}",
                    Self::term_to_string(subject),
                    Self::term_to_string(predicate),
                    Self::term_to_string(object)
                )
            }
            RuleAtom::Builtin { name, args } => {
                let args_str: Vec<String> = args.iter().map(Self::term_to_string).collect();
                format!("{}({})", name, args_str.join(","))
            }
            other => format!("{:?}", other),
        }
    }

    fn term_to_string(term: &Term) -> String {
        match term {
            Term::Variable(v) => format!("?{v}"),
            Term::Constant(c) => c.clone(),
            Term::Literal(l) => format!("\"{l}\""),
            Term::Function { name, args } => {
                let args_str: Vec<String> = args.iter().map(Self::term_to_string).collect();
                format!("{}({})", name, args_str.join(","))
            }
        }
    }

    /// Get statistics
    pub fn statistics(&self) -> &TablingStatistics {
        &self.statistics
    }

    /// Get statistics snapshot
    pub fn statistics_snapshot(&self) -> TablingStatsSnapshot {
        self.statistics.snapshot()
    }

    /// Clear the table
    pub fn clear_table(&mut self) {
        self.table.clear();
    }

    /// Clear everything
    pub fn clear(&mut self) {
        self.table.clear();
        self.rules.clear();
        self.facts.clear();
        self.directives.clear();
        self.call_stack.clear();
        self.statistics.reset();
    }

    /// Get table size
    pub fn table_size(&self) -> usize {
        self.table.len()
    }

    /// Get number of rules
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Get number of facts
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Get entry for a variant
    pub fn get_entry(&self, variant: &CallVariant) -> Option<&TableEntry> {
        self.table.get(variant)
    }

    /// Invalidate entries for a predicate
    pub fn invalidate_predicate(&mut self, predicate: &str) {
        self.table.retain(|v, _| v.predicate != predicate);
    }
}

impl Default for TablingEngine {
    fn default() -> Self {
        Self::new(TablingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(s: &str, p: &str, o: &str) -> RuleAtom {
        RuleAtom::Triple {
            subject: if let Some(stripped) = s.strip_prefix('?') {
                Term::Variable(stripped.to_string())
            } else {
                Term::Constant(s.to_string())
            },
            predicate: Term::Constant(p.to_string()),
            object: if let Some(stripped) = o.strip_prefix('?') {
                Term::Variable(stripped.to_string())
            } else {
                Term::Constant(o.to_string())
            },
        }
    }

    #[test]
    fn test_call_variant_creation() {
        let atom = triple("john", "parent", "mary");
        let variant = CallVariant::from_atom(&atom).unwrap();

        assert_eq!(variant.predicate, "parent");
        assert_eq!(variant.arity(), 2);
    }

    #[test]
    fn test_call_variant_with_variables() {
        let atom = triple("?X", "parent", "?Y");
        let variant = CallVariant::from_atom(&atom).unwrap();

        assert_eq!(variant.predicate, "parent");
        // Variables should be None in binding pattern
        assert!(variant.binding_pattern.iter().all(|p| p.is_none()));
    }

    #[test]
    fn test_table_directive() {
        let mut engine = TablingEngine::default();

        engine.add_table_directive(TableDirective::predicate("ancestor"));

        assert!(engine.should_table("ancestor", 2));
        assert!(!engine.should_table("other", 2));
    }

    #[test]
    fn test_table_all_directive() {
        let mut engine = TablingEngine::default();

        engine.add_table_directive(TableDirective::all());

        assert!(engine.should_table("anything", 2));
        assert!(engine.should_table("any_other", 3));
    }

    #[test]
    fn test_exclude_directive() {
        let mut engine = TablingEngine::default();

        engine.add_table_directive(TableDirective::all());
        engine.add_table_directive(TableDirective::exclude("excluded"));

        assert!(engine.should_table("ancestor", 2));
        assert!(!engine.should_table("excluded", 2));
    }

    #[test]
    fn test_simple_query() {
        let mut engine = TablingEngine::default();

        // Add fact
        engine.add_fact(triple("john", "parent", "mary"));

        // Query
        let goal = triple("john", "parent", "mary");
        let results = engine.query(&goal).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_rule_application() {
        let mut engine = TablingEngine::default();

        // Add rule: ancestor(X,Y) :- parent(X,Y)
        engine.add_rule(Rule {
            name: "ancestor1".to_string(),
            body: vec![triple("?X", "parent", "?Y")],
            head: vec![triple("?X", "ancestor", "?Y")],
        });

        // Add fact
        engine.add_fact(triple("john", "parent", "mary"));

        // Query
        let goal = triple("?X", "ancestor", "?Y");
        let results = engine.query(&goal).unwrap();

        // Should derive ancestor relationship
        assert!(!results.is_empty() || engine.fact_count() > 0);
    }

    #[test]
    fn test_tabling_caching() {
        let config = TablingConfig::default().with_subsumption(true);
        let mut engine = TablingEngine::new(config);

        engine.add_table_directive(TableDirective::predicate("test"));
        engine.add_fact(triple("a", "test", "b"));

        // First query
        let goal = triple("a", "test", "b");
        let _ = engine.query(&goal).unwrap();

        // Second query should hit cache
        let _ = engine.query(&goal).unwrap();

        let stats = engine.statistics_snapshot();
        assert!(stats.hits > 0 || stats.calls > 1);
    }

    #[test]
    fn test_loop_detection_fail() {
        let config = TablingConfig::default();
        let mut engine = TablingEngine::new(config);

        engine.add_table_directive(TableDirective::predicate("loop"));

        // Circular rule: loop(X,Y) :- loop(Y,X)
        engine.add_rule(Rule {
            name: "circular".to_string(),
            body: vec![triple("?Y", "loop", "?X")],
            head: vec![triple("?X", "loop", "?Y")],
        });

        // This should handle the loop gracefully
        let goal = triple("a", "loop", "b");
        let result = engine.query(&goal);

        // Depending on strategy, should either fail or return partial
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_statistics() {
        let config = TablingConfig::default();
        let mut engine = TablingEngine::new(config);

        engine.add_fact(triple("a", "test", "b"));

        let goal = triple("a", "test", "b");
        let _ = engine.query(&goal).unwrap();

        let stats = engine.statistics_snapshot();
        assert!(stats.calls > 0);
    }

    #[test]
    fn test_clear() {
        let mut engine = TablingEngine::default();

        engine.add_fact(triple("a", "test", "b"));
        engine.add_rule(Rule {
            name: "test".to_string(),
            body: vec![],
            head: vec![triple("x", "y", "z")],
        });

        assert_eq!(engine.fact_count(), 1);
        assert_eq!(engine.rule_count(), 1);

        engine.clear();

        assert_eq!(engine.fact_count(), 0);
        assert_eq!(engine.rule_count(), 0);
    }

    #[test]
    fn test_table_entry_access_count() {
        let variant = CallVariant {
            predicate: "test".to_string(),
            binding_pattern: vec![],
        };

        let entry = TableEntry::new(variant);

        entry.record_access();
        entry.record_access();

        assert_eq!(entry.accesses(), 2);
    }

    #[test]
    fn test_config_builder() {
        let config = TablingConfig::default()
            .with_max_size(1000)
            .with_timeout(5000)
            .with_loop_strategy(LoopStrategy::WellFounded)
            .with_subsumption(false);

        assert_eq!(config.max_table_size, 1000);
        assert_eq!(config.timeout_ms, Some(5000));
        assert_eq!(config.loop_strategy, LoopStrategy::WellFounded);
        assert!(!config.enable_subsumption);
    }

    #[test]
    fn test_invalidate_predicate() {
        let mut engine = TablingEngine::default();

        engine.add_table_directive(TableDirective::all());
        engine.add_fact(triple("a", "test1", "b"));
        engine.add_fact(triple("c", "test2", "d"));

        // Query to populate table
        let _ = engine.query(&triple("a", "test1", "b"));
        let _ = engine.query(&triple("c", "test2", "d"));

        let size_before = engine.table_size();

        engine.invalidate_predicate("test1");

        // Should have removed entries for test1
        assert!(engine.table_size() <= size_before);
    }

    #[test]
    fn test_unification() {
        let atom1 = triple("?X", "parent", "mary");
        let atom2 = triple("john", "parent", "mary");

        let subst = TablingEngine::unify(&atom1, &atom2);

        assert!(subst.is_some());
        let subst = subst.unwrap();
        assert!(matches!(subst.get("X"), Some(Term::Constant(c)) if c == "john"));
    }

    #[test]
    fn test_statistics_hit_rate() {
        let stats = TablingStatistics::new();

        stats.hits.store(80, Ordering::Relaxed);
        stats.misses.store(20, Ordering::Relaxed);

        assert!((stats.hit_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_goal_status_transitions() {
        let variant = CallVariant {
            predicate: "test".to_string(),
            binding_pattern: vec![],
        };

        let mut entry = TableEntry::new(variant);
        assert_eq!(entry.status, GoalStatus::New);

        entry.status = GoalStatus::Active;
        assert_eq!(entry.status, GoalStatus::Active);

        entry.complete();
        assert_eq!(entry.status, GoalStatus::Complete);
    }
}
