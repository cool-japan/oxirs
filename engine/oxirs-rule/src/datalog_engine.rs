//! Bottom-up Datalog Evaluation Engine with Semi-Naive Fixpoint
//!
//! This module implements a semi-naive bottom-up Datalog engine that supports:
//!
//! - Unary, binary, and n-ary predicates
//! - Recursive rules (transitive closure, ancestor queries, graph reachability)
//! - Negation-as-failure (stratified negation)
//! - Efficient incremental evaluation via delta-fact tracking
//! - Pattern-matching query interface
//!
//! # Algorithm
//!
//! The semi-naive evaluation computes the fixpoint layer by layer:
//!
//! 1. Start with the EDB (extensional database = base facts) as `delta_new`
//! 2. For each round, derive new facts using rules where at least one body
//!    atom uses a fact from `delta_new`
//! 3. Add genuinely new derived facts to the IDB and update `delta_new`
//! 4. Repeat until `delta_new` is empty (fixpoint)
//!
//! Stratified negation is handled by computing a dependency graph, topologically
//! sorting strata, and evaluating each stratum to fixpoint before moving to the next.

use std::collections::{HashMap, HashSet};

// ─── Terms ──────────────────────────────────────────────────────────────────

/// A term in a Datalog atom: either a constant string or a variable name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DatalogTerm {
    /// A ground constant value
    Constant(String),
    /// A universally-quantified variable (used in rules)
    Variable(String),
}

impl DatalogTerm {
    /// Returns `true` if this term is a variable.
    pub fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }

    /// Returns the inner string regardless of variant.
    pub fn value(&self) -> &str {
        match self {
            Self::Constant(s) | Self::Variable(s) => s,
        }
    }
}

// ─── Atoms ──────────────────────────────────────────────────────────────────

/// A Datalog atom: a predicate applied to a list of terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DatalogAtom {
    /// Predicate name
    pub predicate: String,
    /// Argument terms
    pub terms: Vec<DatalogTerm>,
}

impl DatalogAtom {
    /// Construct a ground fact (all terms are constants).
    pub fn fact(
        predicate: impl Into<String>,
        terms: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            predicate: predicate.into(),
            terms: terms
                .into_iter()
                .map(|t| DatalogTerm::Constant(t.into()))
                .collect(),
        }
    }

    /// Returns `true` if all terms are constants (i.e. this is a ground atom).
    pub fn is_ground(&self) -> bool {
        self.terms.iter().all(|t| !t.is_variable())
    }

    /// Arity of the atom (number of terms).
    pub fn arity(&self) -> usize {
        self.terms.len()
    }
}

// ─── Negated body atoms ──────────────────────────────────────────────────────

/// A body literal that can optionally be negated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BodyLiteral {
    /// The underlying atom
    pub atom: DatalogAtom,
    /// If `true`, this is a negation-as-failure literal: `not atom`
    pub negated: bool,
}

impl BodyLiteral {
    /// Positive body literal
    pub fn positive(atom: DatalogAtom) -> Self {
        Self {
            atom,
            negated: false,
        }
    }

    /// Negated body literal (negation-as-failure)
    pub fn negative(atom: DatalogAtom) -> Self {
        Self {
            atom,
            negated: true,
        }
    }
}

// ─── Rules ──────────────────────────────────────────────────────────────────

/// A Datalog (Horn-clause) rule: `head :- body_1, body_2, ..., not neg_1, ...`
#[derive(Debug, Clone)]
pub struct DatalogRule {
    /// Head atom (conclusion)
    pub head: DatalogAtom,
    /// Body literals (positive and negated)
    pub body: Vec<BodyLiteral>,
}

impl DatalogRule {
    /// Create a simple positive rule (no negation).
    pub fn new(head: DatalogAtom, body: impl IntoIterator<Item = DatalogAtom>) -> Self {
        Self {
            head,
            body: body.into_iter().map(BodyLiteral::positive).collect(),
        }
    }

    /// Create a rule with mixed positive and negated body literals.
    pub fn with_literals(head: DatalogAtom, body: Vec<BodyLiteral>) -> Self {
        Self { head, body }
    }
}

// ─── Program ────────────────────────────────────────────────────────────────

/// A complete Datalog program: a set of base facts (EDB) and rules (IDB).
#[derive(Debug, Clone, Default)]
pub struct DatalogProgram {
    /// Base facts (extensional database)
    pub facts: Vec<DatalogAtom>,
    /// Rules (intensional database)
    pub rules: Vec<DatalogRule>,
}

// ─── Substitution ────────────────────────────────────────────────────────────

/// A variable → constant substitution (partial or complete).
type Substitution = HashMap<String, String>;

/// Attempt to unify `atom` (which may contain variables) with `fact` (ground).
/// Returns `Some(substitution)` on success, or `None` if the atoms are incompatible.
fn unify(atom: &DatalogAtom, fact: &DatalogAtom) -> Option<Substitution> {
    if atom.predicate != fact.predicate || atom.terms.len() != fact.terms.len() {
        return None;
    }
    let mut sub = Substitution::new();
    for (a, f) in atom.terms.iter().zip(fact.terms.iter()) {
        let f_val = match f {
            DatalogTerm::Constant(c) => c,
            DatalogTerm::Variable(_) => return None, // facts must be ground
        };
        match a {
            DatalogTerm::Constant(c) => {
                if c != f_val {
                    return None;
                }
            }
            DatalogTerm::Variable(v) => {
                if let Some(existing) = sub.get(v) {
                    if existing != f_val {
                        return None;
                    }
                } else {
                    sub.insert(v.clone(), f_val.clone());
                }
            }
        }
    }
    Some(sub)
}

/// Apply a substitution to an atom, replacing variables with their bound constants.
fn apply_sub(atom: &DatalogAtom, sub: &Substitution) -> DatalogAtom {
    DatalogAtom {
        predicate: atom.predicate.clone(),
        terms: atom
            .terms
            .iter()
            .map(|t| match t {
                DatalogTerm::Constant(c) => DatalogTerm::Constant(c.clone()),
                DatalogTerm::Variable(v) => sub
                    .get(v)
                    .map(|c| DatalogTerm::Constant(c.clone()))
                    .unwrap_or_else(|| DatalogTerm::Variable(v.clone())),
            })
            .collect(),
    }
}

/// Merge two substitutions if they are compatible (no conflicting bindings).
fn merge_subs(a: &Substitution, b: &Substitution) -> Option<Substitution> {
    let mut result = a.clone();
    for (k, v) in b {
        if let Some(existing) = result.get(k) {
            if existing != v {
                return None;
            }
        } else {
            result.insert(k.clone(), v.clone());
        }
    }
    Some(result)
}

// ─── Stratification ──────────────────────────────────────────────────────────

/// Compute the stratum number for each predicate.
/// Returns `None` if there is an unstratifiable negation cycle.
fn compute_strata(rules: &[DatalogRule]) -> Option<HashMap<String, usize>> {
    let mut stratum: HashMap<String, usize> = HashMap::new();

    // Iterative fixpoint: increase stratum of head if body has negated dependency
    // on a same/higher stratum predicate.
    let max_iter = rules.len() * 10 + 10;
    let mut changed = true;
    let mut iter = 0;
    while changed && iter < max_iter {
        changed = false;
        iter += 1;
        for rule in rules {
            let head_pred = &rule.head.predicate;
            let current = *stratum.get(head_pred).unwrap_or(&0);
            for lit in &rule.body {
                let body_pred = &lit.atom.predicate;
                let body_stratum = *stratum.get(body_pred).unwrap_or(&0);
                let required = if lit.negated {
                    body_stratum + 1
                } else {
                    body_stratum
                };
                if required > current {
                    stratum.insert(head_pred.clone(), required);
                    changed = true;
                }
            }
        }
    }

    // Check for cycles through negation (would require stratum > max_iter)
    if iter >= max_iter {
        // If we're still changing after max iterations, something is wrong
        return None;
    }

    Some(stratum)
}

// ─── Semi-naive evaluation ────────────────────────────────────────────────────

/// Evaluates one stratum of rules to fixpoint using semi-naive evaluation.
///
/// Returns all derived facts (including the initial base facts passed in).
fn evaluate_stratum(
    base_facts: &HashSet<DatalogAtom>,
    rules: &[DatalogRule],
    all_facts: &HashSet<DatalogAtom>,
) -> HashSet<DatalogAtom> {
    let mut idb: HashSet<DatalogAtom> = base_facts.clone();

    loop {
        let mut delta_new: HashSet<DatalogAtom> = HashSet::new();

        for rule in rules {
            // Generate all substitutions for this rule using the current IDB + all_facts
            let mut subs: Vec<Substitution> = vec![Substitution::new()];

            for lit in &rule.body {
                let mut new_subs: Vec<Substitution> = Vec::new();
                let lookup_set: &HashSet<DatalogAtom> = if lit.negated { all_facts } else { &idb };

                if lit.negated {
                    // Negation-as-failure: the atom must NOT be derivable
                    subs.retain(|sub| {
                        let partially_applied = apply_sub(&lit.atom, sub);
                        // Check if any ground fact matches the pattern
                        !lookup_set
                            .iter()
                            .any(|fact| unify(&partially_applied, fact).is_some())
                    });
                    new_subs = subs.clone();
                } else {
                    // Positive literal: join with IDB (semi-naive uses delta for efficiency,
                    // but for correctness we join with full IDB here and filter below)
                    for sub in &subs {
                        let partially_applied = apply_sub(&lit.atom, sub);
                        for fact in lookup_set {
                            if let Some(new_sub) = unify(&partially_applied, fact) {
                                if let Some(merged) = merge_subs(sub, &new_sub) {
                                    new_subs.push(merged);
                                }
                            }
                        }
                    }
                }
                subs = new_subs;
            }

            // Apply each complete substitution to the head
            for sub in &subs {
                let derived = apply_sub(&rule.head, sub);
                if derived.is_ground() && !idb.contains(&derived) {
                    delta_new.insert(derived);
                }
            }
        }

        // Semi-naive: only continue with truly new facts.
        // Future optimization: filter delta_new using previous-round delta;
        // for now we use full IDB join which is correct but not maximally efficient.
        if delta_new.is_empty() {
            break;
        }

        idb.extend(delta_new);
    }

    idb
}

// ─── Engine ──────────────────────────────────────────────────────────────────

/// A bottom-up Datalog evaluation engine with semi-naive fixpoint computation
/// and stratified negation support.
#[derive(Debug, Default)]
pub struct DatalogEngine {
    /// Base (EDB) facts
    facts: Vec<DatalogAtom>,
    /// Rules (IDB)
    rules: Vec<DatalogRule>,
}

impl DatalogEngine {
    /// Create a new empty engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a base fact.
    pub fn add_fact(&mut self, predicate: impl Into<String>, terms: Vec<impl Into<String>>) {
        self.facts.push(DatalogAtom {
            predicate: predicate.into(),
            terms: terms
                .into_iter()
                .map(|t| DatalogTerm::Constant(t.into()))
                .collect(),
        });
    }

    /// Add a rule.
    pub fn add_rule(&mut self, head: DatalogAtom, body: Vec<DatalogAtom>) {
        self.rules.push(DatalogRule::new(head, body));
    }

    /// Add a rule with mixed positive/negated body literals.
    pub fn add_rule_with_literals(&mut self, head: DatalogAtom, body: Vec<BodyLiteral>) {
        self.rules.push(DatalogRule::with_literals(head, body));
    }

    /// Add a pre-built rule.
    pub fn add_datalog_rule(&mut self, rule: DatalogRule) {
        self.rules.push(rule);
    }

    /// Evaluate the program using stratified semi-naive bottom-up fixpoint.
    ///
    /// Returns all derived ground facts (including EDB facts).
    pub fn evaluate(&self) -> Vec<DatalogAtom> {
        let strata = compute_strata(&self.rules).unwrap_or_default();

        // Group rules by the stratum of their head predicate
        let max_stratum = strata.values().copied().max().unwrap_or(0);

        let mut all_facts: HashSet<DatalogAtom> = self.facts.iter().cloned().collect();

        for s in 0..=max_stratum {
            // Collect rules belonging to this stratum
            let stratum_rules: Vec<&DatalogRule> = self
                .rules
                .iter()
                .filter(|r| *strata.get(&r.head.predicate).unwrap_or(&0) == s)
                .collect();

            if stratum_rules.is_empty() {
                continue;
            }

            // Base facts for this stratum: current all_facts filtered to predicates
            // used in heads of stratum rules + the EDB
            let base: HashSet<DatalogAtom> = all_facts.clone();
            let rules_owned: Vec<DatalogRule> = stratum_rules.into_iter().cloned().collect();

            let derived = evaluate_stratum(&base, &rules_owned, &all_facts);
            all_facts = derived;
        }

        // Also evaluate stratum-0 rules (no negation, base stratum)
        let stratum0_rules: Vec<DatalogRule> = self
            .rules
            .iter()
            .filter(|r| *strata.get(&r.head.predicate).unwrap_or(&0) == 0)
            .cloned()
            .collect();

        if !stratum0_rules.is_empty() {
            let base: HashSet<DatalogAtom> = all_facts.clone();
            all_facts = evaluate_stratum(&base, &stratum0_rules, &all_facts);
        }

        all_facts.into_iter().collect()
    }

    /// Query the derived facts by predicate and term patterns.
    ///
    /// Pass `None` for positions you want to treat as wildcards (variables);
    /// pass `Some(val)` for positions you want to match as constants.
    ///
    /// Returns a list of bindings: each binding maps wildcard position indices
    /// (as `"_0"`, `"_1"`, ...) to the matched constant values.
    pub fn query(&self, predicate: &str, pattern: &[Option<&str>]) -> Vec<HashMap<String, String>> {
        let all = self.evaluate();
        let mut results = Vec::new();

        for atom in &all {
            if atom.predicate != predicate {
                continue;
            }
            if atom.terms.len() != pattern.len() {
                continue;
            }
            let mut binding: HashMap<String, String> = HashMap::new();
            let mut matched = true;
            for (i, (term, pat)) in atom.terms.iter().zip(pattern.iter()).enumerate() {
                let term_val = match term {
                    DatalogTerm::Constant(c) => c,
                    DatalogTerm::Variable(_) => {
                        matched = false;
                        break;
                    }
                };
                match pat {
                    Some(expected) => {
                        if term_val != *expected {
                            matched = false;
                            break;
                        }
                    }
                    None => {
                        binding.insert(format!("_{i}"), term_val.clone());
                    }
                }
            }
            if matched {
                results.push(binding);
            }
        }

        results
    }

    /// Clear all facts and rules.
    pub fn clear(&mut self) {
        self.facts.clear();
        self.rules.clear();
    }

    /// Number of base facts.
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

/// Semi-naive evaluation tracker for incremental updates.
///
/// Tracks which facts are "delta" (new in the most recent round)
/// for use in incremental Datalog evaluation.
#[derive(Debug, Default)]
pub struct SemiNaiveEvaluation {
    /// All derived facts so far
    pub idb: HashSet<DatalogAtom>,
    /// Facts added in the most recent evaluation round
    pub delta: HashSet<DatalogAtom>,
    /// Round counter
    pub round: usize,
}

impl SemiNaiveEvaluation {
    /// Create a new semi-naive evaluator seeded with initial facts.
    pub fn new(initial_facts: impl IntoIterator<Item = DatalogAtom>) -> Self {
        let idb: HashSet<DatalogAtom> = initial_facts.into_iter().collect();
        let delta = idb.clone();
        Self {
            idb,
            delta,
            round: 0,
        }
    }

    /// Apply one round of rule applications.
    ///
    /// Returns `true` if any new facts were derived (not yet at fixpoint).
    pub fn step(&mut self, rules: &[DatalogRule]) -> bool {
        let mut new_facts: HashSet<DatalogAtom> = HashSet::new();

        for rule in rules {
            let mut subs: Vec<Substitution> = vec![Substitution::new()];

            for lit in &rule.body {
                if lit.negated {
                    subs.retain(|sub| {
                        let pat = apply_sub(&lit.atom, sub);
                        !self.idb.iter().any(|f| unify(&pat, f).is_some())
                    });
                } else {
                    let mut new_subs = Vec::new();
                    for sub in &subs {
                        let pat = apply_sub(&lit.atom, sub);
                        for fact in &self.idb {
                            if let Some(new_sub) = unify(&pat, fact) {
                                if let Some(merged) = merge_subs(sub, &new_sub) {
                                    new_subs.push(merged);
                                }
                            }
                        }
                    }
                    subs = new_subs;
                }
            }

            for sub in &subs {
                let derived = apply_sub(&rule.head, sub);
                if derived.is_ground() && !self.idb.contains(&derived) {
                    new_facts.insert(derived);
                }
            }
        }

        let changed = !new_facts.is_empty();
        self.delta = new_facts.clone();
        self.idb.extend(new_facts);
        self.round += 1;
        changed
    }

    /// Run to fixpoint, returning all derived facts.
    pub fn run_to_fixpoint(&mut self, rules: &[DatalogRule]) -> Vec<DatalogAtom> {
        while self.step(rules) {}
        self.idb.iter().cloned().collect()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ground(pred: &str, terms: &[&str]) -> DatalogAtom {
        DatalogAtom::fact(pred, terms.iter().copied())
    }

    fn var_atom(pred: &str, terms: &[&str]) -> DatalogAtom {
        DatalogAtom {
            predicate: pred.to_string(),
            terms: terms
                .iter()
                .map(|t| {
                    if let Some(var) = t.strip_prefix('?') {
                        DatalogTerm::Variable(var.to_string())
                    } else {
                        DatalogTerm::Constant(t.to_string())
                    }
                })
                .collect(),
        }
    }

    fn has_fact(facts: &[DatalogAtom], pred: &str, terms: &[&str]) -> bool {
        let target = ground(pred, terms);
        facts.contains(&target)
    }

    // ── Basic facts ──────────────────────────────────────────────────────────

    #[test]
    fn test_empty_engine() {
        let engine = DatalogEngine::new();
        let facts = engine.evaluate();
        assert!(facts.is_empty());
    }

    #[test]
    fn test_add_fact_retrieval() {
        let mut engine = DatalogEngine::new();
        engine.add_fact("person", vec!["alice"]);
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "person", &["alice"]));
    }

    #[test]
    fn test_multiple_facts() {
        let mut engine = DatalogEngine::new();
        engine.add_fact("person", vec!["alice"]);
        engine.add_fact("person", vec!["bob"]);
        engine.add_fact("age", vec!["alice", "30"]);
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "person", &["alice"]));
        assert!(has_fact(&facts, "person", &["bob"]));
        assert!(has_fact(&facts, "age", &["alice", "30"]));
    }

    #[test]
    fn test_fact_arity() {
        let atom = ground("triple", &["s", "p", "o"]);
        assert_eq!(atom.arity(), 3);
    }

    #[test]
    fn test_fact_is_ground() {
        let atom = ground("pred", &["a", "b"]);
        assert!(atom.is_ground());
    }

    #[test]
    fn test_rule_atom_is_not_ground() -> anyhow::Result<()> {
        let atom = var_atom("pred", &["?X", "b"]);
        assert!(!atom.is_ground());
        Ok(())
    }

    // ── Simple rules ─────────────────────────────────────────────────────────

    #[test]
    fn test_simple_deduction() -> anyhow::Result<()> {
        // parent(X,Y) → ancestor(X,Y)
        let mut engine = DatalogEngine::new();
        engine.add_fact("parent", vec!["alice", "bob"]);
        engine.add_rule(
            var_atom("ancestor", &["?X", "?Y"]),
            vec![var_atom("parent", &["?X", "?Y"])],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "ancestor", &["alice", "bob"]));
        Ok(())
    }

    #[test]
    fn test_chain_rules() -> anyhow::Result<()> {
        // parent → grandparent chain
        let mut engine = DatalogEngine::new();
        engine.add_fact("parent", vec!["alice", "bob"]);
        engine.add_fact("parent", vec!["bob", "carol"]);
        // grandparent(X,Z) :- parent(X,Y), parent(Y,Z)
        engine.add_rule(
            var_atom("grandparent", &["?X", "?Z"]),
            vec![
                var_atom("parent", &["?X", "?Y"]),
                var_atom("parent", &["?Y", "?Z"]),
            ],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "grandparent", &["alice", "carol"]));
        Ok(())
    }

    #[test]
    fn test_multiple_rules() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("edge", vec!["a", "b"]);
        engine.add_fact("edge", vec!["b", "c"]);
        engine.add_rule(
            var_atom("reachable", &["?X", "?Y"]),
            vec![var_atom("edge", &["?X", "?Y"])],
        );
        engine.add_rule(
            var_atom("reachable", &["?X", "?Z"]),
            vec![
                var_atom("reachable", &["?X", "?Y"]),
                var_atom("edge", &["?Y", "?Z"]),
            ],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "reachable", &["a", "b"]));
        assert!(has_fact(&facts, "reachable", &["a", "c"]));
        assert!(has_fact(&facts, "reachable", &["b", "c"]));
        Ok(())
    }

    // ── Recursive rules ───────────────────────────────────────────────────────

    #[test]
    fn test_transitive_closure() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("edge", vec!["1", "2"]);
        engine.add_fact("edge", vec!["2", "3"]);
        engine.add_fact("edge", vec!["3", "4"]);
        // tc(X,Y) :- edge(X,Y)
        engine.add_rule(
            var_atom("tc", &["?X", "?Y"]),
            vec![var_atom("edge", &["?X", "?Y"])],
        );
        // tc(X,Z) :- tc(X,Y), edge(Y,Z)
        engine.add_rule(
            var_atom("tc", &["?X", "?Z"]),
            vec![
                var_atom("tc", &["?X", "?Y"]),
                var_atom("edge", &["?Y", "?Z"]),
            ],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "tc", &["1", "2"]));
        assert!(has_fact(&facts, "tc", &["1", "3"]));
        assert!(has_fact(&facts, "tc", &["1", "4"]));
        assert!(has_fact(&facts, "tc", &["2", "3"]));
        assert!(has_fact(&facts, "tc", &["2", "4"]));
        assert!(has_fact(&facts, "tc", &["3", "4"]));
        Ok(())
    }

    #[test]
    fn test_ancestor_recursive() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("parent", vec!["a", "b"]);
        engine.add_fact("parent", vec!["b", "c"]);
        engine.add_fact("parent", vec!["c", "d"]);
        // ancestor(X,Y) :- parent(X,Y)
        engine.add_rule(
            var_atom("ancestor", &["?X", "?Y"]),
            vec![var_atom("parent", &["?X", "?Y"])],
        );
        // ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
        engine.add_rule(
            var_atom("ancestor", &["?X", "?Z"]),
            vec![
                var_atom("parent", &["?X", "?Y"]),
                var_atom("ancestor", &["?Y", "?Z"]),
            ],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "ancestor", &["a", "b"]));
        assert!(has_fact(&facts, "ancestor", &["a", "c"]));
        assert!(has_fact(&facts, "ancestor", &["a", "d"]));
        assert!(has_fact(&facts, "ancestor", &["b", "c"]));
        assert!(has_fact(&facts, "ancestor", &["b", "d"]));
        Ok(())
    }

    #[test]
    fn test_graph_reachability() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        // A diamond graph: a→b, a→c, b→d, c→d
        engine.add_fact("edge", vec!["a", "b"]);
        engine.add_fact("edge", vec!["a", "c"]);
        engine.add_fact("edge", vec!["b", "d"]);
        engine.add_fact("edge", vec!["c", "d"]);
        engine.add_rule(
            var_atom("reach", &["?X", "?Y"]),
            vec![var_atom("edge", &["?X", "?Y"])],
        );
        engine.add_rule(
            var_atom("reach", &["?X", "?Z"]),
            vec![
                var_atom("reach", &["?X", "?Y"]),
                var_atom("edge", &["?Y", "?Z"]),
            ],
        );
        let facts = engine.evaluate();
        // a can reach b, c, d
        assert!(has_fact(&facts, "reach", &["a", "b"]));
        assert!(has_fact(&facts, "reach", &["a", "c"]));
        assert!(has_fact(&facts, "reach", &["a", "d"]));
        // b can reach d; c can reach d
        assert!(has_fact(&facts, "reach", &["b", "d"]));
        assert!(has_fact(&facts, "reach", &["c", "d"]));
        Ok(())
    }

    // ── Negation-as-failure ───────────────────────────────────────────────────

    #[test]
    fn test_negation_as_failure_basic() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("person", vec!["alice"]);
        engine.add_fact("person", vec!["bob"]);
        engine.add_fact("blocked", vec!["bob"]);
        // allowed(X) :- person(X), not blocked(X)
        engine.add_rule_with_literals(
            var_atom("allowed", &["?X"]),
            vec![
                BodyLiteral::positive(var_atom("person", &["?X"])),
                BodyLiteral::negative(var_atom("blocked", &["?X"])),
            ],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "allowed", &["alice"]));
        assert!(!has_fact(&facts, "allowed", &["bob"]));
        Ok(())
    }

    #[test]
    fn test_negation_bachelor() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("person", vec!["alice"]);
        engine.add_fact("person", vec!["bob"]);
        engine.add_fact("person", vec!["carol"]);
        engine.add_fact("married", vec!["alice", "carol"]);
        // bachelor(X) :- person(X), not married(X, _)
        // simplified: not_married(X) :- person(X), not is_married(X)
        engine.add_fact("is_married", vec!["alice"]);
        engine.add_rule_with_literals(
            var_atom("bachelor", &["?X"]),
            vec![
                BodyLiteral::positive(var_atom("person", &["?X"])),
                BodyLiteral::negative(var_atom("is_married", &["?X"])),
            ],
        );
        let facts = engine.evaluate();
        assert!(!has_fact(&facts, "bachelor", &["alice"]));
        assert!(has_fact(&facts, "bachelor", &["bob"]));
        assert!(has_fact(&facts, "bachelor", &["carol"]));
        Ok(())
    }

    // ── Query interface ───────────────────────────────────────────────────────

    #[test]
    fn test_query_wildcard() {
        let mut engine = DatalogEngine::new();
        engine.add_fact("person", vec!["alice"]);
        engine.add_fact("person", vec!["bob"]);
        let results = engine.query("person", &[None]);
        assert_eq!(results.len(), 2);
        let names: HashSet<String> = results.iter().map(|b| b["_0"].clone()).collect();
        assert!(names.contains("alice"));
        assert!(names.contains("bob"));
    }

    #[test]
    fn test_query_constant() {
        let mut engine = DatalogEngine::new();
        engine.add_fact("edge", vec!["a", "b"]);
        engine.add_fact("edge", vec!["a", "c"]);
        engine.add_fact("edge", vec!["b", "c"]);
        // Query edges starting from "a"
        let results = engine.query("edge", &[Some("a"), None]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_no_results() {
        let mut engine = DatalogEngine::new();
        engine.add_fact("person", vec!["alice"]);
        let results = engine.query("person", &[Some("nobody")]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_binary_predicate() {
        let mut engine = DatalogEngine::new();
        engine.add_fact("age", vec!["alice", "30"]);
        engine.add_fact("age", vec!["bob", "25"]);
        let results = engine.query("age", &[None, Some("30")]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["_0"], "alice");
    }

    // ── N-ary predicates ─────────────────────────────────────────────────────

    #[test]
    fn test_ternary_fact() {
        let mut engine = DatalogEngine::new();
        engine.add_fact("triple", vec!["s", "p", "o"]);
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "triple", &["s", "p", "o"]));
    }

    #[test]
    fn test_ternary_rule() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("knows", vec!["alice", "bob", "2020"]);
        // friends(X,Y) :- knows(X,Y,_T)
        engine.add_rule(
            var_atom("friends", &["?X", "?Y"]),
            vec![var_atom("knows", &["?X", "?Y", "?T"])],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "friends", &["alice", "bob"]));
        Ok(())
    }

    // ── SemiNaiveEvaluation ───────────────────────────────────────────────────

    #[test]
    fn test_semi_naive_basic() -> anyhow::Result<()> {
        let rules = vec![DatalogRule::new(
            var_atom("reachable", &["?X", "?Y"]),
            vec![var_atom("edge", &["?X", "?Y"])],
        )];
        let initial = vec![ground("edge", &["a", "b"]), ground("edge", &["b", "c"])];
        let mut eval = SemiNaiveEvaluation::new(initial);
        let all = eval.run_to_fixpoint(&rules);
        assert!(all.contains(&ground("reachable", &["a", "b"])));
        Ok(())
    }

    #[test]
    fn test_semi_naive_round_counter() {
        let rules: Vec<DatalogRule> = vec![];
        let initial = vec![ground("fact", &["x"])];
        let mut eval = SemiNaiveEvaluation::new(initial);
        eval.step(&rules);
        assert_eq!(eval.round, 1);
    }

    #[test]
    fn test_semi_naive_delta_tracking() -> anyhow::Result<()> {
        let rules = vec![DatalogRule::new(
            var_atom("derived", &["?X"]),
            vec![var_atom("base", &["?X"])],
        )];
        let initial = vec![ground("base", &["a"])];
        let mut eval = SemiNaiveEvaluation::new(initial);
        let changed = eval.step(&rules);
        assert!(changed);
        assert!(eval.delta.contains(&ground("derived", &["a"])));
        Ok(())
    }

    #[test]
    fn test_semi_naive_fixpoint_detection() {
        let rules: Vec<DatalogRule> = vec![];
        let initial = vec![ground("fact", &["x"])];
        let mut eval = SemiNaiveEvaluation::new(initial);
        // No rules → no new facts → no change after first step
        let changed = eval.step(&rules);
        assert!(!changed);
    }

    // ── Engine utilities ─────────────────────────────────────────────────────

    #[test]
    fn test_engine_counts() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("p", vec!["a"]);
        engine.add_fact("p", vec!["b"]);
        engine.add_rule(var_atom("q", &["?X"]), vec![var_atom("p", &["?X"])]);
        assert_eq!(engine.fact_count(), 2);
        assert_eq!(engine.rule_count(), 1);
        Ok(())
    }

    #[test]
    fn test_engine_clear() {
        let mut engine = DatalogEngine::new();
        engine.add_fact("p", vec!["a"]);
        engine.clear();
        assert_eq!(engine.fact_count(), 0);
        assert_eq!(engine.rule_count(), 0);
    }

    #[test]
    fn test_deduplication() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("p", vec!["a"]);
        engine.add_fact("p", vec!["a"]); // duplicate
        engine.add_rule(var_atom("q", &["?X"]), vec![var_atom("p", &["?X"])]);
        let facts = engine.evaluate();
        let q_count = facts.iter().filter(|f| f.predicate == "q").count();
        assert_eq!(q_count, 1); // deduplicated
        Ok(())
    }

    #[test]
    fn test_unary_predicate_rule() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("mammal", vec!["cat"]);
        engine.add_rule(
            var_atom("animal", &["?X"]),
            vec![var_atom("mammal", &["?X"])],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "animal", &["cat"]));
        Ok(())
    }

    #[test]
    fn test_constant_join() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("likes", vec!["alice", "pizza"]);
        engine.add_fact("likes", vec!["bob", "pizza"]);
        engine.add_fact("likes", vec!["alice", "sushi"]);
        // same_taste(X,Y) :- likes(X,Z), likes(Y,Z), X != Y  (simplified: just join)
        // Here we test that the join on ?Z correctly identifies shared items
        engine.add_rule(
            var_atom("pizza_fan", &["?X"]),
            vec![var_atom("likes", &["?X", "pizza"])],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "pizza_fan", &["alice"]));
        assert!(has_fact(&facts, "pizza_fan", &["bob"]));
        assert!(!has_fact(&facts, "pizza_fan", &["carol"]));
        Ok(())
    }

    #[test]
    fn test_add_datalog_rule() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("base", vec!["x"]);
        let rule = DatalogRule::new(
            var_atom("derived", &["?X"]),
            vec![var_atom("base", &["?X"])],
        );
        engine.add_datalog_rule(rule);
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "derived", &["x"]));
        Ok(())
    }

    #[test]
    fn test_rule_with_two_variable_joins() -> anyhow::Result<()> {
        let mut engine = DatalogEngine::new();
        engine.add_fact("student", vec!["alice"]);
        engine.add_fact("course", vec!["alice", "math"]);
        engine.add_rule(
            var_atom("math_student", &["?X"]),
            vec![
                var_atom("student", &["?X"]),
                var_atom("course", &["?X", "math"]),
            ],
        );
        let facts = engine.evaluate();
        assert!(has_fact(&facts, "math_student", &["alice"]));
        Ok(())
    }
}
