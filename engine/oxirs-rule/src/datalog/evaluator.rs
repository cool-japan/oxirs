//! Datalog semi-naive evaluation engine
//!
//! ## Algorithm Overview
//!
//! ### Naive evaluation (initial pass)
//! For each rule `head :- b1, b2, ... bk`, enumerate all substitutions σ such
//! that σ(b1), σ(b2), ..., σ(bk) are all in the current database, and add
//! σ(head) to the IDB.
//!
//! ### Semi-naive evaluation (fixpoint loop)
//! In each iteration t, for each rule and for each i ∈ {1..k}, derive new head
//! facts using:
//!   - `delta[bi]` (facts derived in iteration t-1) for body atom i
//!   - full IDB (from all previous iterations) for all other body atoms j ≠ i
//!
//! This guarantees that every newly produced fact uses at least one fact from
//! the delta, avoiding re-deriving already known facts and making each iteration
//! O(|delta| × |IDB|^(k-1)) rather than O(|IDB|^k).
//!
//! ### Stratification
//! The dependency graph is computed and checked for negation-stratifiability.
//! If a negated predicate has a path back to itself, a `StratificationError`
//! is returned. Strata are computed via topological sort.

use std::collections::{HashMap, HashSet, VecDeque};

use super::{
    DatalogAtom, DatalogError, DatalogFact, DatalogProgram, DatalogRule, DatalogTerm, DatalogValue,
    FactDatabase, Substitution,
};

/// Unify a `DatalogTerm` against a ground `DatalogValue`, potentially
/// extending the substitution.
///
/// Returns `true` and extends `sub` if successful; returns `false` otherwise
/// (the substitution is left in an indeterminate state and should not be used
/// on failure).
pub fn unify(term: &DatalogTerm, value: &DatalogValue, sub: &mut Substitution) -> bool {
    match term {
        DatalogTerm::Constant(c) => c == value,
        DatalogTerm::Variable(v) => {
            if let Some(existing) = sub.get(v) {
                existing == value
            } else {
                sub.insert(v.clone(), value.clone());
                true
            }
        }
    }
}

/// Instantiate a `DatalogAtom` under a given substitution.
/// Unbound variables remain as variables in the result.
fn apply_subst_to_atom(atom: &DatalogAtom, sub: &Substitution) -> DatalogAtom {
    DatalogAtom {
        predicate: atom.predicate.clone(),
        terms: atom
            .terms
            .iter()
            .map(|t| match t {
                DatalogTerm::Variable(v) => sub
                    .get(v)
                    .map(|val| DatalogTerm::Constant(val.clone()))
                    .unwrap_or_else(|| DatalogTerm::Variable(v.clone())),
                DatalogTerm::Constant(_) => t.clone(),
            })
            .collect(),
    }
}

/// Try to extend all candidate substitutions by matching `atom` against `db`.
/// Returns the set of extended substitutions.
fn extend_substitutions(
    partial_subs: Vec<Substitution>,
    atom: &DatalogAtom,
    db: &FactDatabase,
) -> Vec<Substitution> {
    let mut result = Vec::new();
    for sub in &partial_subs {
        let atom_inst = apply_subst_to_atom(atom, sub);
        // Iterate over all ground tuples for this predicate
        for tuple in db.tuples_for(&atom_inst.predicate) {
            if atom_inst.terms.len() != tuple.len() {
                continue;
            }
            let mut new_sub = sub.clone();
            let mut ok = true;
            for (term, value) in atom_inst.terms.iter().zip(tuple.iter()) {
                if !unify(term, value, &mut new_sub) {
                    ok = false;
                    break;
                }
            }
            if ok {
                result.push(new_sub);
            }
        }
    }
    result
}

/// Apply a rule naively: enumerate all substitutions that satisfy the entire
/// body against `db` and collect derived head facts.
pub fn apply_rule(
    rule: &DatalogRule,
    edb: &FactDatabase,
    idb: &FactDatabase,
) -> HashSet<DatalogFact> {
    // Build a merged view: EDB ∪ IDB
    let mut merged = edb.clone();
    merged.merge(idb);
    apply_rule_against(&merged, rule)
}

/// Core rule application against an arbitrary database.
fn apply_rule_against(db: &FactDatabase, rule: &DatalogRule) -> HashSet<DatalogFact> {
    let mut derived = HashSet::new();

    if rule.body.is_empty() {
        // Fact-like rule: head with no body variables → emit directly if all constants
        if let Some(fact) = ground_atom(&rule.head, &HashMap::new()) {
            derived.insert(fact);
        }
        return derived;
    }

    let mut subs: Vec<Substitution> = vec![HashMap::new()];

    for atom in &rule.body {
        subs = extend_substitutions(subs, atom, db);
        if subs.is_empty() {
            return derived; // short-circuit
        }
    }

    for sub in subs {
        if let Some(fact) = ground_atom(&rule.head, &sub) {
            derived.insert(fact);
        }
    }

    derived
}

/// Try to produce a ground `DatalogFact` from `atom` under substitution `sub`.
/// Returns `None` if any term remains unbound (the rule isn't range-restricted).
fn ground_atom(atom: &DatalogAtom, sub: &Substitution) -> Option<DatalogFact> {
    let mut args = Vec::with_capacity(atom.terms.len());
    for term in &atom.terms {
        match term {
            DatalogTerm::Constant(c) => args.push(c.clone()),
            DatalogTerm::Variable(v) => {
                args.push(sub.get(v)?.clone());
            }
        }
    }
    Some(DatalogFact {
        predicate: atom.predicate.clone(),
        args,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Dependency graph and stratification
// ────────────────────────────────────────────────────────────────────────────

/// Edge in the dependency graph: `(from_predicate, to_predicate, is_negated)`.
struct DepEdge {
    from: String,
    to: String,
    negated: bool,
}

/// Build the predicate dependency graph from the program's rules.
/// An edge `p → q` (positive) exists if rule body contains `q` in the body
/// of a rule with head predicate `p`.
/// (Negation-through-failure is implicit in future extensions; here we only
/// track positive dependencies for the basic semi-naive engine.)
fn build_dependency_graph(rules: &[DatalogRule]) -> Vec<DepEdge> {
    let mut edges = Vec::new();
    for rule in rules {
        let head_pred = &rule.head.predicate;
        for body_atom in &rule.body {
            edges.push(DepEdge {
                from: head_pred.clone(),
                to: body_atom.predicate.clone(),
                negated: false, // extension point for future negation
            });
        }
    }
    edges
}

/// Compute a topological ordering of strata (predicates).
///
/// A stratum is a set of predicates that can be evaluated together without
/// depending on negation of each other. With only positive recursion, each
/// SCC forms one stratum.
///
/// Returns `Err(StratificationError)` if a negated cyclic dependency is found.
fn stratify(rules: &[DatalogRule]) -> Result<Vec<String>, DatalogError> {
    let edges = build_dependency_graph(rules);

    // Collect all predicates
    let mut predicates: HashSet<String> = HashSet::new();
    for rule in rules {
        predicates.insert(rule.head.predicate.clone());
        for atom in &rule.body {
            predicates.insert(atom.predicate.clone());
        }
    }

    // Check for negated cycles (none exist in basic Datalog, but enforce as per spec)
    for edge in &edges {
        if edge.negated {
            // Check if there's a path back from `to` to `from`
            if has_path(&edges, &edge.to, &edge.from) {
                return Err(DatalogError::StratificationError(format!(
                    "cyclic negation between '{}' and '{}'",
                    edge.from, edge.to
                )));
            }
        }
    }

    // Topological sort of predicate dependency: Kahn's algorithm
    // Build adjacency: pred → set of preds that depend on it (reverse edges)
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut reverse_adj: HashMap<String, Vec<String>> = HashMap::new();

    for pred in &predicates {
        in_degree.entry(pred.clone()).or_insert(0);
        reverse_adj.entry(pred.clone()).or_default();
    }

    for edge in &edges {
        // edge: from depends on to → in_degree[from] += 1 in the dependency ordering
        *in_degree.entry(edge.from.clone()).or_insert(0) += 1;
        reverse_adj
            .entry(edge.to.clone())
            .or_default()
            .push(edge.from.clone());
    }

    let mut queue: VecDeque<String> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(p, _)| p.clone())
        .collect();

    let mut order = Vec::new();

    while let Some(pred) = queue.pop_front() {
        order.push(pred.clone());
        if let Some(dependents) = reverse_adj.get(&pred) {
            for dep in dependents {
                let deg = in_degree.entry(dep.clone()).or_insert(0);
                if *deg > 0 {
                    *deg -= 1;
                }
                if *deg == 0 {
                    queue.push_back(dep.clone());
                }
            }
        }
    }

    // If any predicate wasn't visited, there's a cycle (mutual recursion is OK in Datalog)
    // We handle cycles by including remaining predicates at the end (they form SCCs)
    if order.len() < predicates.len() {
        for pred in &predicates {
            if !order.contains(pred) {
                order.push(pred.clone());
            }
        }
    }

    Ok(order)
}

/// Check if there is a directed path from `start` to `goal` in the edge set.
fn has_path(edges: &[DepEdge], start: &str, goal: &str) -> bool {
    let mut visited = HashSet::new();
    let mut stack = vec![start.to_string()];

    while let Some(current) = stack.pop() {
        if current == goal {
            return true;
        }
        if visited.contains(&current) {
            continue;
        }
        visited.insert(current.clone());

        for edge in edges {
            if edge.from == current {
                stack.push(edge.to.clone());
            }
        }
    }

    false
}

// ────────────────────────────────────────────────────────────────────────────
// Semi-naive evaluator
// ────────────────────────────────────────────────────────────────────────────

/// Semi-naive Datalog evaluator.
///
/// The semi-naive algorithm extends naive bottom-up evaluation by tracking,
/// at each iteration, only the *delta* (newly derived facts). In each
/// iteration it fires rules using at least one body atom from the delta,
/// which avoids redundant re-derivation of already-known facts.
pub struct SemiNaiveEvaluator {}

impl SemiNaiveEvaluator {
    /// Create a new evaluator.
    pub fn new() -> Self {
        Self {}
    }

    /// Evaluate a Datalog program and return all derived facts.
    ///
    /// Returns a `FactDatabase` containing all EDB facts plus all derived IDB facts.
    pub fn evaluate(&self, program: &DatalogProgram) -> Result<FactDatabase, DatalogError> {
        // Stratification check (returns predicate ordering)
        let _strata_order = stratify(&program.rules)?;

        // Identify IDB predicates (heads of rules)
        let idb_preds: HashSet<String> = program.idb_predicates();

        // Initialize EDB
        let mut edb = FactDatabase::new();
        for fact in &program.edb {
            edb.insert_fact(fact);
        }

        // Initialize IDB and per-predicate delta (Δ)
        let mut idb = FactDatabase::new();
        // delta[pred] = facts derived *in the last iteration* for `pred`
        let mut delta: HashMap<String, HashSet<Vec<DatalogValue>>> = HashMap::new();

        for pred in &idb_preds {
            delta.insert(pred.clone(), HashSet::new());
        }

        // ──────────────────────────────────────────────────────────────────
        // Phase 1: Naive first iteration — derive all facts reachable from EDB alone
        // ──────────────────────────────────────────────────────────────────
        for rule in &program.rules {
            let new_facts = apply_rule(rule, &edb, &idb);
            for fact in new_facts {
                let args = fact.args.clone();
                let pred = fact.predicate.clone();
                if idb.insert_fact(&fact) {
                    delta.entry(pred).or_default().insert(args);
                }
            }
        }

        // ──────────────────────────────────────────────────────────────────
        // Phase 2: Semi-naive fixpoint loop
        //
        // In each iteration, for each rule `head :- b1, b2, ..., bk` and
        // for each index i ∈ [0, k), fire the rule using:
        //   - delta[bi] for body atom i     (the "new" relation)
        //   - full IDB   for body atoms j≠i (the "old" relation)
        //
        // This covers every new derivation exactly once per iteration.
        // ──────────────────────────────────────────────────────────────────
        loop {
            // Build a snapshot of the current IDB so we can measure growth
            let prev_idb_size = idb.len();

            // new_delta accumulates all facts derived in this iteration
            let mut new_delta: HashMap<String, HashSet<Vec<DatalogValue>>> = HashMap::new();
            for pred in &idb_preds {
                new_delta.insert(pred.clone(), HashSet::new());
            }

            for rule in &program.rules {
                let head_pred = &rule.head.predicate;

                if rule.body.is_empty() {
                    // Rule with no body — already handled in phase 1
                    continue;
                }

                // Semi-naive: for each body position i, use delta[body[i]] +
                // full IDB for all other positions
                let body_len = rule.body.len();
                for i in 0..body_len {
                    let delta_pred = &rule.body[i].predicate;

                    // Get the delta set for position i
                    let delta_tuples = match delta.get(delta_pred) {
                        Some(d) if !d.is_empty() => d.clone(),
                        _ => continue, // no new facts for this predicate → skip
                    };

                    // Build a temporary database for the delta at position i
                    let mut delta_db = FactDatabase::new();
                    for tuple in &delta_tuples {
                        delta_db.insert(delta_pred, tuple.clone());
                    }

                    // Enumerate all substitutions:
                    // - positions < i:  use full IDB
                    // - position i:     use delta_db
                    // - positions > i:  use full IDB
                    let mut subs: Vec<Substitution> = vec![HashMap::new()];

                    for (j, body_atom) in rule.body.iter().enumerate() {
                        let db_to_use = if j == i { &delta_db } else { &idb };
                        // For EDB predicates, always include EDB facts regardless
                        if idb_preds.contains(&body_atom.predicate) || j == i {
                            subs = extend_substitutions(subs, body_atom, db_to_use);
                        } else {
                            // EDB atom at position j ≠ i: extend from EDB
                            subs = extend_substitutions(subs, body_atom, &edb);
                        }
                        if subs.is_empty() {
                            break;
                        }
                    }

                    for sub in subs {
                        if let Some(fact) = ground_atom(&rule.head, &sub) {
                            let args = fact.args.clone();
                            let is_new = idb.insert_fact(&fact);
                            if is_new {
                                new_delta.entry(head_pred.clone()).or_default().insert(args);
                            }
                        }
                    }
                }
            }

            // Replace delta with new_delta for next iteration
            delta = new_delta;

            // Fixpoint check: if IDB did not grow, we're done
            if idb.len() == prev_idb_size {
                break;
            }
        }

        // Produce combined output: EDB + IDB
        let mut result = edb;
        result.merge(&idb);
        Ok(result)
    }
}

impl Default for SemiNaiveEvaluator {
    fn default() -> Self {
        Self::new()
    }
}
