//! N3 Backward Chaining Engine
//!
//! This module provides goal-driven backward chaining inference for N3 rules.
//! Backward chaining works by starting from a goal and working backwards
//! through rules to find supporting facts.
//!
//! # Algorithm
//!
//! Given a goal statement, the engine:
//! 1. Tries to unify the goal directly with known facts
//! 2. For each rule whose consequent unifies with the goal, recursively
//!    resolves each antecedent sub-goal with bindings propagated forward
//! 3. Depth limiting and visited-set tracking prevent infinite loops
//!
//! # Examples
//!
//! ```rust
//! use oxirs_ttl::n3::backward_chaining::{BackwardChainer, BackwardChainingEngine};
//! use oxirs_ttl::n3::{N3Formula, N3Implication, N3Statement, N3Term, N3Variable};
//! use oxirs_core::model::NamedNode;
//!
//! // Facts: alice parent bob, bob parent carol
//! let alice = N3Term::NamedNode(NamedNode::new("http://ex.org/alice").expect("should succeed"));
//! let bob = N3Term::NamedNode(NamedNode::new("http://ex.org/bob").expect("should succeed"));
//! let parent_iri = N3Term::NamedNode(NamedNode::new("http://ex.org/parent").expect("should succeed"));
//!
//! let facts = vec![
//!     N3Statement::new(alice.clone(), parent_iri.clone(), bob.clone()),
//! ];
//!
//! // Goal: who is a parent of bob?
//! let goal = N3Statement::new(
//!     N3Term::Variable(N3Variable::universal("who")),
//!     parent_iri.clone(),
//!     bob.clone(),
//! );
//!
//! let chainer = BackwardChainer::new(100);
//! let results = chainer.resolve(&goal, &[], &facts);
//! assert!(!results.is_empty());
//! ```

use crate::formats::n3_reasoning::{Matcher, Substitution, VariableBindings};
use crate::formats::n3_types::{N3Formula, N3Implication, N3Statement, N3Term, N3Variable};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global counter for generating unique rule-instance variable suffixes.
static RULE_INSTANCE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Rename all variables in a rule to fresh names to avoid unification capture.
fn rename_rule(rule: &N3Implication) -> N3Implication {
    let id = RULE_INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut renames: HashMap<String, String> = HashMap::new();

    // Collect all variable names
    for stmt in rule
        .antecedent
        .triples
        .iter()
        .chain(rule.consequent.triples.iter())
    {
        collect_vars_in_term(&stmt.subject, &mut renames, id);
        collect_vars_in_term(&stmt.predicate, &mut renames, id);
        collect_vars_in_term(&stmt.object, &mut renames, id);
    }

    let new_ant = rename_formula(&rule.antecedent, &renames);
    let new_con = rename_formula(&rule.consequent, &renames);
    N3Implication::new(new_ant, new_con)
}

fn collect_vars_in_term(term: &N3Term, renames: &mut HashMap<String, String>, id: usize) {
    if let N3Term::Variable(v) = term {
        renames
            .entry(v.name.clone())
            .or_insert_with(|| format!("{}_r{}", v.name, id));
    }
}

fn rename_term(term: &N3Term, renames: &HashMap<String, String>) -> N3Term {
    match term {
        N3Term::Variable(v) => {
            let new_name = renames
                .get(&v.name)
                .cloned()
                .unwrap_or_else(|| v.name.clone());
            N3Term::Variable(N3Variable::universal(&new_name))
        }
        N3Term::Formula(f) => N3Term::Formula(Box::new(rename_formula(f, renames))),
        other => other.clone(),
    }
}

fn rename_stmt(stmt: &N3Statement, renames: &HashMap<String, String>) -> N3Statement {
    N3Statement::new(
        rename_term(&stmt.subject, renames),
        rename_term(&stmt.predicate, renames),
        rename_term(&stmt.object, renames),
    )
}

fn rename_formula(formula: &N3Formula, renames: &HashMap<String, String>) -> N3Formula {
    N3Formula::with_statements(
        formula
            .triples
            .iter()
            .map(|s| rename_stmt(s, renames))
            .collect(),
    )
}

// ── Solving context ───────────────────────────────────────────────────────────

/// Tracks the current state of a backward-chaining search to prevent infinite loops.
#[derive(Debug, Clone)]
pub struct SolvingContext {
    /// Current recursion depth
    pub depth: u32,
    /// Maximum allowed recursion depth
    pub max_depth: u32,
    /// String representations of goals already on the stack (cycle detection)
    pub visited: HashSet<String>,
}

impl SolvingContext {
    /// Create a new solving context with the given depth limit.
    pub fn new(max_depth: u32) -> Self {
        Self {
            depth: 0,
            max_depth,
            visited: HashSet::new(),
        }
    }

    /// Return a child context one level deeper.
    fn descend(&self) -> Self {
        Self {
            depth: self.depth + 1,
            max_depth: self.max_depth,
            visited: self.visited.clone(),
        }
    }

    /// Return true if we have exceeded the depth limit.
    fn is_too_deep(&self) -> bool {
        self.depth > self.max_depth
    }
}

// ── Proof tracing ─────────────────────────────────────────────────────────────

/// A single step recorded during backward-chaining proof search.
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// A human-readable summary of the rule (or "fact") that was applied.
    pub rule_applied: String,
    /// A human-readable summary of the bindings in effect at this step.
    pub bindings_summary: String,
    /// Recursion depth at which this step was taken.
    pub depth: u32,
}

impl ProofStep {
    fn new(
        rule_applied: impl Into<String>,
        bindings_summary: impl Into<String>,
        depth: u32,
    ) -> Self {
        Self {
            rule_applied: rule_applied.into(),
            bindings_summary: bindings_summary.into(),
            depth,
        }
    }
}

/// A complete proof trace for a single goal resolution attempt.
#[derive(Debug, Clone)]
pub struct ProofTrace {
    /// Steps taken during this proof search.
    pub steps: Vec<ProofStep>,
    /// Human-readable summary of the top-level goal.
    pub goal_summary: String,
    /// Whether the proof succeeded (at least one solution was found).
    pub succeeded: bool,
}

impl ProofTrace {
    fn new(goal_summary: impl Into<String>) -> Self {
        Self {
            steps: Vec::new(),
            goal_summary: goal_summary.into(),
            succeeded: false,
        }
    }

    /// Record a new proof step.
    pub fn record_step(&mut self, step: ProofStep) {
        self.steps.push(step);
    }
}

// ── Unification ───────────────────────────────────────────────────────────────

/// Attempt to unify a pattern term (which may contain variables) against a
/// concrete term, extending `bindings` on success.
fn unify_term(pattern: &N3Term, concrete: &N3Term, bindings: &mut VariableBindings) -> bool {
    match pattern {
        N3Term::Variable(var) => {
            if let Some(already_bound) = bindings.get(&var.name) {
                // Variable already bound — check consistency
                already_bound == concrete
            } else {
                // Bind variable to the concrete term
                bindings.bind(var.name.clone(), concrete.clone());
                true
            }
        }
        N3Term::Formula(_) => {
            // Formula-level unification is not supported in this implementation
            false
        }
        _ => {
            // Concrete terms must match exactly
            pattern == concrete
        }
    }
}

/// Try to unify a pattern statement against a concrete statement.
///
/// Returns `Some(bindings)` on success, `None` on failure.
fn unify_statements(pattern: &N3Statement, concrete: &N3Statement) -> Option<VariableBindings> {
    let mut bindings = VariableBindings::new();
    if unify_term(&pattern.subject, &concrete.subject, &mut bindings)
        && unify_term(&pattern.predicate, &concrete.predicate, &mut bindings)
        && unify_term(&pattern.object, &concrete.object, &mut bindings)
    {
        Some(bindings)
    } else {
        None
    }
}

/// Apply `outer` bindings to a statement (replace variables with bound values).
fn apply_bindings_to_statement(stmt: &N3Statement, bindings: &VariableBindings) -> N3Statement {
    N3Statement::new(
        bindings.substitute(&stmt.subject),
        bindings.substitute(&stmt.predicate),
        bindings.substitute(&stmt.object),
    )
}

/// Produce a short string key for a statement (used for cycle detection).
fn statement_key(stmt: &N3Statement) -> String {
    format!("{} {} {}", stmt.subject, stmt.predicate, stmt.object)
}

// ── Core resolver ─────────────────────────────────────────────────────────────

/// Merge two compatible `VariableBindings` into a new binding set.
///
/// Returns `None` if the bindings conflict.
fn merge_bindings(a: &VariableBindings, b: &VariableBindings) -> Option<VariableBindings> {
    if a.is_compatible(b) {
        let mut merged = a.clone();
        merged.merge(b);
        Some(merged)
    } else {
        None
    }
}

/// Recursively resolve a single goal statement, returning all satisfying bindings.
fn resolve_goal(
    goal: &N3Statement,
    rules: &[N3Implication],
    facts: &[N3Statement],
    ctx: &SolvingContext,
    trace: &mut ProofTrace,
) -> Vec<VariableBindings> {
    if ctx.is_too_deep() {
        return vec![];
    }

    let key = statement_key(goal);
    if ctx.visited.contains(&key) {
        return vec![];
    }

    let mut child_ctx = ctx.descend();
    child_ctx.visited.insert(key.clone());

    let mut results: Vec<VariableBindings> = Vec::new();

    // ── Step 1: Try to satisfy the goal directly from facts ────────────────
    for fact in facts {
        if let Some(bindings) = unify_statements(goal, fact) {
            trace.record_step(ProofStep::new(
                "fact",
                format!("bound via fact: {}", statement_key(fact)),
                ctx.depth,
            ));
            results.push(bindings);
        }
    }

    // ── Step 2: Try each rule ──────────────────────────────────────────────
    for rule in rules {
        // Rename variables to fresh names to avoid capture across rule applications
        let fresh_rule = rename_rule(rule);
        for consequent_stmt in &fresh_rule.consequent.triples {
            // Unify the rule consequent (which has variables) against the goal.
            // The consequent is the "pattern" — its variables get bound to goal terms.
            if let Some(head_bindings) = unify_statements(consequent_stmt, goal) {
                // Substitute head_bindings into the antecedent body
                let body_goals: Vec<N3Statement> = fresh_rule
                    .antecedent
                    .triples
                    .iter()
                    .map(|s| apply_bindings_to_statement(s, &head_bindings))
                    .collect();

                // Resolve all sub-goals
                let sub_solutions =
                    resolve_goals_conjunction(&body_goals, rules, facts, &child_ctx, trace);

                for sub_bindings in sub_solutions {
                    if let Some(merged) = merge_bindings(&head_bindings, &sub_bindings) {
                        trace.record_step(ProofStep::new(
                            format!("rule: {} => {}", rule.antecedent, rule.consequent),
                            format!("depth={}", ctx.depth),
                            ctx.depth,
                        ));
                        results.push(merged);
                    }
                }
            }
        }
    }

    results
}

/// Resolve a conjunction of goals, returning all satisfying combined bindings.
fn resolve_goals_conjunction(
    goals: &[N3Statement],
    rules: &[N3Implication],
    facts: &[N3Statement],
    ctx: &SolvingContext,
    trace: &mut ProofTrace,
) -> Vec<VariableBindings> {
    if goals.is_empty() {
        return vec![VariableBindings::new()];
    }

    let first = &goals[0];
    let rest = &goals[1..];

    let first_solutions = resolve_goal(first, rules, facts, ctx, trace);

    let mut combined: Vec<VariableBindings> = Vec::new();
    for first_binding in first_solutions {
        // Apply first_binding to remaining goals before resolving them
        let substituted_rest: Vec<N3Statement> = rest
            .iter()
            .map(|s| apply_bindings_to_statement(s, &first_binding))
            .collect();

        let rest_solutions = resolve_goals_conjunction(&substituted_rest, rules, facts, ctx, trace);

        for rest_binding in rest_solutions {
            if let Some(merged) = merge_bindings(&first_binding, &rest_binding) {
                combined.push(merged);
            }
        }
    }
    combined
}

// ── Public API ────────────────────────────────────────────────────────────────

/// A backward chaining engine for N3 rules.
///
/// Given a goal statement, the engine searches through rules and facts to find
/// all variable bindings that satisfy the goal.
#[derive(Debug, Clone)]
pub struct BackwardChainer {
    /// Maximum recursion depth to prevent infinite loops
    max_depth: u32,
    /// Matcher used for pattern matching
    #[allow(dead_code)]
    matcher: Matcher,
}

impl BackwardChainer {
    /// Create a new `BackwardChainer` with the given maximum recursion depth.
    pub fn new(max_depth: u32) -> Self {
        Self {
            max_depth,
            matcher: Matcher::new(),
        }
    }

    /// Create a `BackwardChainer` with the default depth limit of 100.
    pub fn default_depth() -> Self {
        Self::new(100)
    }

    /// Resolve a single goal, returning all satisfying variable bindings.
    pub fn resolve(
        &self,
        goal: &N3Statement,
        rules: &[N3Implication],
        facts: &[N3Statement],
    ) -> Vec<VariableBindings> {
        let ctx = SolvingContext::new(self.max_depth);
        let mut trace = ProofTrace::new(statement_key(goal));
        resolve_goal(goal, rules, facts, &ctx, &mut trace)
    }

    /// Resolve a conjunction of goals, returning all satisfying combined bindings.
    pub fn resolve_all(
        &self,
        goals: &[N3Statement],
        rules: &[N3Implication],
        facts: &[N3Statement],
    ) -> Vec<VariableBindings> {
        let ctx = SolvingContext::new(self.max_depth);
        let mut trace = ProofTrace::new("multi-goal conjunction");
        resolve_goals_conjunction(goals, rules, facts, &ctx, &mut trace)
    }

    /// Resolve a single goal, returning both the bindings and a proof trace.
    pub fn resolve_with_trace(
        &self,
        goal: &N3Statement,
        rules: &[N3Implication],
        facts: &[N3Statement],
    ) -> (Vec<VariableBindings>, ProofTrace) {
        let ctx = SolvingContext::new(self.max_depth);
        let mut trace = ProofTrace::new(statement_key(goal));
        let results = resolve_goal(goal, rules, facts, &ctx, &mut trace);
        trace.succeeded = !results.is_empty();
        (results, trace)
    }
}

impl Default for BackwardChainer {
    fn default() -> Self {
        Self::default_depth()
    }
}

/// High-level backward chaining engine with built-in proof tracing support.
#[derive(Debug, Clone, Default)]
pub struct BackwardChainingEngine {
    chainer: BackwardChainer,
}

impl BackwardChainingEngine {
    /// Create a new engine with the given maximum depth.
    pub fn new(max_depth: u32) -> Self {
        Self {
            chainer: BackwardChainer::new(max_depth),
        }
    }

    /// Solve a goal and return bindings together with a proof trace.
    pub fn solve_with_trace(
        &self,
        goal: &N3Statement,
        rules: &[N3Implication],
        facts: &[N3Statement],
    ) -> (Vec<VariableBindings>, ProofTrace) {
        self.chainer.resolve_with_trace(goal, rules, facts)
    }

    /// Solve a goal without recording a proof trace.
    pub fn solve(
        &self,
        goal: &N3Statement,
        rules: &[N3Implication],
        facts: &[N3Statement],
    ) -> Vec<VariableBindings> {
        self.chainer.resolve(goal, rules, facts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::NamedNode;

    // ─── helpers ───────────────────────────────────────────────────────────

    fn iri(s: &str) -> N3Term {
        N3Term::NamedNode(NamedNode::new(s).expect("valid IRI"))
    }

    fn var(name: &str) -> N3Term {
        use crate::formats::n3_types::N3Variable;
        N3Term::Variable(N3Variable::universal(name))
    }

    fn stmt(s: N3Term, p: N3Term, o: N3Term) -> N3Statement {
        N3Statement::new(s, p, o)
    }

    fn rule(ant_stmts: Vec<N3Statement>, con_stmts: Vec<N3Statement>) -> N3Implication {
        use crate::formats::n3_types::N3Formula;
        N3Implication::new(
            N3Formula::with_statements(ant_stmts),
            N3Formula::with_statements(con_stmts),
        )
    }

    // ─── basic fact lookup ─────────────────────────────────────────────────

    #[test]
    fn test_resolve_ground_fact() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let chainer = BackwardChainer::new(10);
        let results = chainer.resolve(&stmt(a, p, b), &[], &facts);
        assert_eq!(
            results.len(),
            1,
            "ground goal should match exactly one fact"
        );
    }

    #[test]
    fn test_resolve_no_match() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let c = iri("http://ex.org/c");
        let facts = vec![stmt(a.clone(), p.clone(), b)];
        let chainer = BackwardChainer::new(10);
        let results = chainer.resolve(&stmt(a, p, c), &[], &facts);
        assert!(
            results.is_empty(),
            "goal with no matching fact returns empty"
        );
    }

    #[test]
    fn test_resolve_variable_subject() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let chainer = BackwardChainer::new(10);
        let goal = stmt(var("x"), p, b);
        let results = chainer.resolve(&goal, &[], &facts);
        assert!(!results.is_empty());
        let binding = results[0].get("x").expect("x must be bound");
        assert_eq!(binding, &a);
    }

    #[test]
    fn test_resolve_variable_object() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let chainer = BackwardChainer::new(10);
        let goal = stmt(a, p, var("y"));
        let results = chainer.resolve(&goal, &[], &facts);
        assert!(!results.is_empty());
        let binding = results[0].get("y").expect("y must be bound");
        assert_eq!(binding, &b);
    }

    #[test]
    fn test_resolve_both_variables() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![
            stmt(a.clone(), p.clone(), b.clone()),
            stmt(b.clone(), p.clone(), a.clone()),
        ];
        let chainer = BackwardChainer::new(10);
        let goal = stmt(var("x"), p, var("y"));
        let results = chainer.resolve(&goal, &[], &facts);
        assert_eq!(results.len(), 2, "should match both facts");
    }

    // ─── rule-based chaining ───────────────────────────────────────────────

    #[test]
    fn test_symmetry_rule() {
        // { ?x :knows ?y } => { ?y :knows ?x }
        let knows = iri("http://ex.org/knows");
        let alice = iri("http://ex.org/alice");
        let bob = iri("http://ex.org/bob");

        let facts = vec![stmt(alice.clone(), knows.clone(), bob.clone())];
        let rules = vec![rule(
            vec![stmt(var("x"), knows.clone(), var("y"))],
            vec![stmt(var("y"), knows.clone(), var("x"))],
        )];

        let chainer = BackwardChainer::new(10);
        // Goal: is bob :knows alice derivable?
        let goal = stmt(bob.clone(), knows.clone(), alice.clone());
        let results = chainer.resolve(&goal, &rules, &facts);
        assert!(
            !results.is_empty(),
            "symmetry rule should derive bob knows alice"
        );
    }

    #[test]
    fn test_transitivity_two_hops() {
        // { ?x :parent ?y . ?y :parent ?z } => { ?x :grandparent ?z }
        let parent = iri("http://ex.org/parent");
        let grandparent = iri("http://ex.org/grandparent");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let c = iri("http://ex.org/c");

        let facts = vec![
            stmt(a.clone(), parent.clone(), b.clone()),
            stmt(b.clone(), parent.clone(), c.clone()),
        ];
        let rules = vec![rule(
            vec![
                stmt(var("x"), parent.clone(), var("y")),
                stmt(var("y"), parent.clone(), var("z")),
            ],
            vec![stmt(var("x"), grandparent.clone(), var("z"))],
        )];

        let chainer = BackwardChainer::new(20);
        let goal = stmt(a.clone(), grandparent.clone(), c.clone());
        let results = chainer.resolve(&goal, &rules, &facts);
        assert!(
            !results.is_empty(),
            "grandparent should be derivable via two hops"
        );
    }

    #[test]
    fn test_transitivity_three_hops() {
        let ancestor = iri("http://ex.org/ancestor");
        let parent = iri("http://ex.org/parent");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let c = iri("http://ex.org/c");
        let d = iri("http://ex.org/d");

        let facts = vec![
            stmt(a.clone(), parent.clone(), b.clone()),
            stmt(b.clone(), parent.clone(), c.clone()),
            stmt(c.clone(), parent.clone(), d.clone()),
        ];
        // Base: parent => ancestor
        // Inductive: ancestor + parent => ancestor
        let rules = vec![
            rule(
                vec![stmt(var("x"), parent.clone(), var("y"))],
                vec![stmt(var("x"), ancestor.clone(), var("y"))],
            ),
            rule(
                vec![
                    stmt(var("x"), ancestor.clone(), var("y")),
                    stmt(var("y"), parent.clone(), var("z")),
                ],
                vec![stmt(var("x"), ancestor.clone(), var("z"))],
            ),
        ];

        let chainer = BackwardChainer::new(30);
        let goal = stmt(a.clone(), ancestor.clone(), d.clone());
        let results = chainer.resolve(&goal, &rules, &facts);
        assert!(
            !results.is_empty(),
            "ancestor should be derivable across three hops"
        );
    }

    // ─── depth limit ──────────────────────────────────────────────────────

    #[test]
    fn test_depth_limit_prevents_stack_overflow() {
        // A rule that is cyclic: { ?x :p ?y } => { ?y :p ?x }
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let rules = vec![rule(
            vec![stmt(var("x"), p.clone(), var("y"))],
            vec![stmt(var("y"), p.clone(), var("x"))],
        )];
        // Very small depth limit — should terminate without panicking
        let chainer = BackwardChainer::new(3);
        let goal = stmt(b.clone(), p.clone(), a.clone());
        let results = chainer.resolve(&goal, &rules, &facts);
        // Result may or may not be found but must not recurse infinitely
        let _ = results;
    }

    #[test]
    fn test_zero_depth_limit() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        // With depth 0 only direct fact matches succeed
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let rules = vec![rule(
            vec![stmt(var("x"), p.clone(), var("y"))],
            vec![stmt(var("y"), p.clone(), var("x"))],
        )];
        let chainer = BackwardChainer::new(0);
        // Goal that matches a fact directly should still succeed
        let goal = stmt(a.clone(), p.clone(), b.clone());
        let results = chainer.resolve(&goal, &[], &facts);
        assert!(!results.is_empty());
        // Goal that requires rule application should fail at depth 0
        let rule_goal = stmt(b.clone(), p.clone(), a.clone());
        let rule_results = chainer.resolve(&rule_goal, &rules, &facts);
        // depth 0 means we cannot apply any rules (body resolution is at depth 1)
        // This either returns empty or some results depending on if visited tracking cuts it
        let _ = rule_results;
    }

    // ─── resolve_all (conjunction) ─────────────────────────────────────────

    #[test]
    fn test_resolve_all_conjunction() {
        let p = iri("http://ex.org/p");
        let q = iri("http://ex.org/q");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");

        let facts = vec![
            stmt(a.clone(), p.clone(), b.clone()),
            stmt(b.clone(), q.clone(), a.clone()),
        ];

        let chainer = BackwardChainer::new(10);
        let goals = vec![
            stmt(var("x"), p.clone(), var("y")),
            stmt(var("y"), q.clone(), var("x")),
        ];
        let results = chainer.resolve_all(&goals, &[], &facts);
        assert!(!results.is_empty(), "conjunction should be satisfiable");
        let binding = &results[0];
        assert!(binding.is_bound("x"));
        assert!(binding.is_bound("y"));
    }

    #[test]
    fn test_resolve_all_empty_goals() {
        let chainer = BackwardChainer::new(10);
        let results = chainer.resolve_all(&[], &[], &[]);
        // Empty conjunction is trivially true with empty bindings
        assert_eq!(results.len(), 1);
        assert!(results[0].all_bindings().is_empty());
    }

    #[test]
    fn test_resolve_all_unsatisfiable() {
        let p = iri("http://ex.org/p");
        let q = iri("http://ex.org/q");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let c = iri("http://ex.org/c");

        // a p b exists but there is no binding for ?y such that ?y q ?x = (b q a)
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let chainer = BackwardChainer::new(10);
        // Require: ?x p ?y AND ?y q ?z (no q facts exist)
        let goals = vec![
            stmt(var("x"), p.clone(), var("y")),
            stmt(var("y"), q.clone(), var("z")),
        ];
        let results = chainer.resolve_all(&goals, &[], &facts);
        assert!(results.is_empty(), "conjunction should be unsatisfiable");
        let _ = c;
    }

    // ─── proof tracing ─────────────────────────────────────────────────────

    #[test]
    fn test_proof_trace_fact_match() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let chainer = BackwardChainer::new(10);
        let (results, trace) = chainer.resolve_with_trace(&stmt(a, p, b), &[], &facts);
        assert!(!results.is_empty());
        assert!(trace.succeeded);
        assert!(!trace.steps.is_empty());
        assert_eq!(trace.steps[0].rule_applied, "fact");
    }

    #[test]
    fn test_proof_trace_rule_application() {
        let knows = iri("http://ex.org/knows");
        let alice = iri("http://ex.org/alice");
        let bob = iri("http://ex.org/bob");

        let facts = vec![stmt(alice.clone(), knows.clone(), bob.clone())];
        let rules = vec![rule(
            vec![stmt(var("x"), knows.clone(), var("y"))],
            vec![stmt(var("y"), knows.clone(), var("x"))],
        )];

        let engine = BackwardChainingEngine::new(10);
        let goal = stmt(bob.clone(), knows.clone(), alice.clone());
        let (results, trace) = engine.solve_with_trace(&goal, &rules, &facts);
        assert!(!results.is_empty());
        assert!(trace.succeeded);
        // Should have at least one step from the rule application
        assert!(!trace.steps.is_empty());
    }

    #[test]
    fn test_proof_trace_failed_goal() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let c = iri("http://ex.org/c");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let chainer = BackwardChainer::new(10);
        let (results, trace) = chainer.resolve_with_trace(&stmt(a, p, c), &[], &facts);
        assert!(results.is_empty());
        assert!(!trace.succeeded);
    }

    // ─── BackwardChainingEngine ────────────────────────────────────────────

    #[test]
    fn test_engine_solve() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let engine = BackwardChainingEngine::new(10);
        let results = engine.solve(&stmt(a, p, var("y")), &[], &facts);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_engine_default() {
        let engine = BackwardChainingEngine::default();
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let results = engine.solve(&stmt(a, p, b), &[], &facts);
        assert!(!results.is_empty());
    }

    // ─── multiple rules ────────────────────────────────────────────────────

    #[test]
    fn test_multiple_applicable_rules() {
        let child = iri("http://ex.org/child");
        let parent = iri("http://ex.org/parent");
        let offspring = iri("http://ex.org/offspring");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");

        // Both rules can produce offspring from child or parent
        let facts = vec![
            stmt(a.clone(), child.clone(), b.clone()),
            stmt(b.clone(), parent.clone(), a.clone()),
        ];
        let rules = vec![
            rule(
                vec![stmt(var("x"), child.clone(), var("y"))],
                vec![stmt(var("x"), offspring.clone(), var("y"))],
            ),
            rule(
                vec![stmt(var("y"), parent.clone(), var("x"))],
                vec![stmt(var("x"), offspring.clone(), var("y"))],
            ),
        ];

        let chainer = BackwardChainer::new(10);
        let goal = stmt(a.clone(), offspring.clone(), b.clone());
        let results = chainer.resolve(&goal, &rules, &facts);
        // Both rules produce the same conclusion so we may get duplicate bindings
        assert!(!results.is_empty(), "at least one derivation should exist");
    }

    #[test]
    fn test_no_rules_no_facts() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let chainer = BackwardChainer::new(10);
        let results = chainer.resolve(&stmt(a, p, b), &[], &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_variable_predicate() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let chainer = BackwardChainer::new(10);
        let goal = stmt(a.clone(), var("pred"), b.clone());
        let results = chainer.resolve(&goal, &[], &facts);
        assert!(!results.is_empty());
        let binding = results[0].get("pred").expect("pred must be bound");
        assert_eq!(binding, &p);
    }

    #[test]
    fn test_solving_context_depth_tracking() {
        let ctx = SolvingContext::new(5);
        assert_eq!(ctx.depth, 0);
        assert!(!ctx.is_too_deep());
        let child = ctx.descend();
        assert_eq!(child.depth, 1);
        // Simulate deep nesting
        let mut deep = ctx.clone();
        for _ in 0..6 {
            deep = deep.descend();
        }
        assert!(deep.is_too_deep());
    }

    #[test]
    fn test_visited_set_prevents_cycles() {
        // A perfectly cyclic rule: { ?x :r ?y } => { ?y :r ?x }
        // with only direct fact available
        let r = iri("http://ex.org/r");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), r.clone(), b.clone())];
        let rules = vec![rule(
            vec![stmt(var("x"), r.clone(), var("y"))],
            vec![stmt(var("y"), r.clone(), var("x"))],
        )];
        // depth=5 should be sufficient to terminate via cycle detection
        let chainer = BackwardChainer::new(5);
        let goal = stmt(b.clone(), r.clone(), a.clone());
        // Must not panic / infinite loop
        let results = chainer.resolve(&goal, &rules, &facts);
        let _ = results;
    }

    #[test]
    fn test_proof_step_depth_recorded() {
        let p = iri("http://ex.org/p");
        let a = iri("http://ex.org/a");
        let b = iri("http://ex.org/b");
        let facts = vec![stmt(a.clone(), p.clone(), b.clone())];
        let chainer = BackwardChainer::new(10);
        let (_, trace) = chainer.resolve_with_trace(&stmt(a, p, b), &[], &facts);
        assert!(!trace.steps.is_empty());
        assert_eq!(trace.steps[0].depth, 0, "top-level match is at depth 0");
    }
}
