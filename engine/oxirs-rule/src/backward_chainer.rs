/// Backward-chaining (goal-directed) reasoner for SPARQL/RDF rule sets.
///
/// Implements SLD-resolution style backward chaining where proofs are
/// constructed by recursively resolving goals against a set of Horn clauses.
use std::collections::HashMap;

/// A logical goal: a predicate with subject and object terms.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Goal {
    /// The predicate name (e.g. "parent", "ancestor").
    pub predicate: String,
    /// Subject term (IRI, constant, or variable starting with '?').
    pub subject: String,
    /// Object term (IRI, constant, or variable starting with '?').
    pub object: String,
}

impl Goal {
    /// Create a new goal.
    pub fn new(
        predicate: impl Into<String>,
        subject: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            predicate: predicate.into(),
            subject: subject.into(),
            object: object.into(),
        }
    }
}

/// A Horn clause: `head :- body[0], body[1], ...`
/// When the body is empty the clause is a fact.
#[derive(Debug, Clone)]
pub struct Clause {
    /// The conclusion.
    pub head: Goal,
    /// Antecedents (empty for facts).
    pub body: Vec<Goal>,
}

/// A set of variable bindings mapping variable names (without '?') to values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Substitution(pub HashMap<String, String>);

impl Substitution {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Bind a variable name to a value.
    pub fn bind(&mut self, var: String, val: String) {
        self.0.insert(var, val);
    }

    /// Look up the value bound to a variable name (without '?').
    pub fn get(&self, var: &str) -> Option<&str> {
        self.0.get(var).map(|s| s.as_str())
    }

    /// Compose two substitutions: apply `other` on top of `self`.
    /// Variables in `self` are resolved through `other` if possible.
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();
        for (k, v) in &self.0 {
            // Apply other to the value
            let resolved = if Self::is_variable(v) {
                let var_name = &v[1..];
                other.0.get(var_name).cloned().unwrap_or_else(|| v.clone())
            } else {
                v.clone()
            };
            result.0.insert(k.clone(), resolved);
        }
        // Add bindings from other that are not already in self
        for (k, v) in &other.0 {
            result.0.entry(k.clone()).or_insert_with(|| v.clone());
        }
        result
    }

    /// Return true if the term is a variable (starts with '?').
    pub fn is_variable(s: &str) -> bool {
        s.starts_with('?')
    }
}

/// A node in a proof tree showing how a goal was proved.
#[derive(Debug, Clone)]
pub struct ProofTree {
    /// The goal that was proved.
    pub goal: Goal,
    /// The clause label used (None for facts).
    pub used_clause: Option<String>,
    /// Sub-proofs for each body goal.
    pub children: Vec<ProofTree>,
    /// The substitution at this node.
    pub substitution: Substitution,
}

/// Backward-chaining reasoner.
#[derive(Debug, Clone)]
pub struct BackwardChainer {
    clauses: Vec<Clause>,
    max_depth: usize,
}

impl BackwardChainer {
    /// Create a new backward chainer with the given recursion limit.
    pub fn new(max_depth: usize) -> Self {
        Self {
            clauses: Vec::new(),
            max_depth,
        }
    }

    /// Add a Horn clause (head :- body).
    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }

    /// Add a ground fact (clause with empty body).
    pub fn add_fact(&mut self, predicate: &str, subject: &str, object: &str) {
        self.clauses.push(Clause {
            head: Goal::new(predicate, subject, object),
            body: vec![],
        });
    }

    /// Attempt to prove a goal using DFS. Returns the first proof tree found.
    pub fn prove(&self, goal: &Goal) -> Option<ProofTree> {
        let sub = Substitution::new();
        self.prove_internal(goal, &sub, 0)
    }

    /// Collect all proofs for a goal (DFS, enumerates all solutions).
    pub fn prove_all(&self, goal: &Goal) -> Vec<ProofTree> {
        let sub = Substitution::new();
        self.prove_all_internal(goal, &sub, 0)
    }

    /// Return true if the goal can be proved.
    pub fn can_prove(&self, goal: &Goal) -> bool {
        self.prove(goal).is_some()
    }

    /// Return the total number of clauses (facts + rules).
    pub fn clause_count(&self) -> usize {
        self.clauses.len()
    }

    /// Unify two goals. Returns the most general unifier or None if they clash.
    pub fn unify(goal: &Goal, head: &Goal) -> Option<Substitution> {
        if goal.predicate != head.predicate {
            return None;
        }
        let mut sub = Substitution::new();
        Self::unify_term(&goal.subject, &head.subject, &mut sub)?;
        Self::unify_term(&goal.object, &head.object, &mut sub)?;
        Some(sub)
    }

    /// Apply a substitution to a goal, replacing variables with their bound values.
    pub fn apply_substitution(goal: &Goal, sub: &Substitution) -> Goal {
        Goal {
            predicate: goal.predicate.clone(),
            subject: Self::apply_term(&goal.subject, sub),
            object: Self::apply_term(&goal.object, sub),
        }
    }

    // ---- private helpers ----

    fn apply_term(term: &str, sub: &Substitution) -> String {
        if Substitution::is_variable(term) {
            let var_name = &term[1..];
            sub.get(var_name)
                .map(|v| {
                    // Recursively resolve if the value is itself a variable
                    if Substitution::is_variable(v) {
                        Self::apply_term(v, sub)
                    } else {
                        v.to_string()
                    }
                })
                .unwrap_or_else(|| term.to_string())
        } else {
            term.to_string()
        }
    }

    fn unify_term(t1: &str, t2: &str, sub: &mut Substitution) -> Option<()> {
        let r1 = if Substitution::is_variable(t1) {
            sub.get(&t1[1..])
                .map(|s| s.to_string())
                .unwrap_or_else(|| t1.to_string())
        } else {
            t1.to_string()
        };
        let r2 = if Substitution::is_variable(t2) {
            sub.get(&t2[1..])
                .map(|s| s.to_string())
                .unwrap_or_else(|| t2.to_string())
        } else {
            t2.to_string()
        };

        if r1 == r2 {
            return Some(());
        }
        if Substitution::is_variable(&r1) {
            sub.0.insert(r1[1..].to_string(), r2);
            return Some(());
        }
        if Substitution::is_variable(&r2) {
            sub.0.insert(r2[1..].to_string(), r1);
            return Some(());
        }
        // Both are ground terms and they differ
        None
    }

    fn prove_internal(
        &self,
        goal: &Goal,
        current_sub: &Substitution,
        depth: usize,
    ) -> Option<ProofTree> {
        if depth > self.max_depth {
            return None;
        }
        let resolved_goal = Self::apply_substitution(goal, current_sub);

        for clause in &self.clauses {
            // Rename variables in clause to avoid capture
            let renamed = self.rename_clause(clause, depth);

            if let Some(mgu) = Self::unify(&resolved_goal, &renamed.head) {
                let combined = current_sub.compose(&mgu);

                // Try to prove all body goals
                if let Some((children, final_sub)) =
                    self.prove_body(&renamed.body, &combined, depth + 1)
                {
                    let label = if renamed.body.is_empty() {
                        None
                    } else {
                        Some(format!(
                            "{} :- {}",
                            renamed.head.predicate,
                            renamed
                                .body
                                .iter()
                                .map(|g| g.predicate.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ))
                    };
                    return Some(ProofTree {
                        goal: resolved_goal,
                        used_clause: label,
                        children,
                        substitution: final_sub,
                    });
                }
            }
        }
        None
    }

    fn prove_all_internal(
        &self,
        goal: &Goal,
        current_sub: &Substitution,
        depth: usize,
    ) -> Vec<ProofTree> {
        if depth > self.max_depth {
            return vec![];
        }
        let resolved_goal = Self::apply_substitution(goal, current_sub);
        let mut proofs = Vec::new();

        for clause in &self.clauses {
            let renamed = self.rename_clause(clause, depth);

            if let Some(mgu) = Self::unify(&resolved_goal, &renamed.head) {
                let combined = current_sub.compose(&mgu);

                if renamed.body.is_empty() {
                    proofs.push(ProofTree {
                        goal: resolved_goal.clone(),
                        used_clause: None,
                        children: vec![],
                        substitution: combined,
                    });
                } else {
                    // Enumerate all body proofs
                    let all_body = self.prove_all_body(&renamed.body, &combined, depth + 1);
                    for (children, final_sub) in all_body {
                        let label = Some(format!(
                            "{} :- {}",
                            renamed.head.predicate,
                            renamed
                                .body
                                .iter()
                                .map(|g| g.predicate.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ));
                        proofs.push(ProofTree {
                            goal: resolved_goal.clone(),
                            used_clause: label,
                            children,
                            substitution: final_sub,
                        });
                    }
                }
            }
        }
        proofs
    }

    fn prove_body(
        &self,
        body: &[Goal],
        sub: &Substitution,
        depth: usize,
    ) -> Option<(Vec<ProofTree>, Substitution)> {
        if body.is_empty() {
            return Some((vec![], sub.clone()));
        }
        let first = &body[0];
        let rest = &body[1..];

        if let Some(tree) = self.prove_internal(first, sub, depth) {
            let new_sub = sub.compose(&tree.substitution);
            if let Some((mut rest_trees, final_sub)) = self.prove_body(rest, &new_sub, depth) {
                rest_trees.insert(0, tree);
                return Some((rest_trees, final_sub));
            }
        }
        None
    }

    fn prove_all_body(
        &self,
        body: &[Goal],
        sub: &Substitution,
        depth: usize,
    ) -> Vec<(Vec<ProofTree>, Substitution)> {
        if body.is_empty() {
            return vec![(vec![], sub.clone())];
        }
        let first = &body[0];
        let rest = &body[1..];
        let first_proofs = self.prove_all_internal(first, sub, depth);
        let mut results = Vec::new();
        for tree in first_proofs {
            let new_sub = sub.compose(&tree.substitution);
            let rest_results = self.prove_all_body(rest, &new_sub, depth);
            for (mut rest_trees, final_sub) in rest_results {
                rest_trees.insert(0, tree.clone());
                results.push((rest_trees, final_sub));
            }
        }
        results
    }

    /// Rename all variables in a clause with a depth-suffixed version.
    fn rename_clause(&self, clause: &Clause, depth: usize) -> Clause {
        let suffix = format!("_{depth}");
        let rename_goal = |g: &Goal| Goal {
            predicate: g.predicate.clone(),
            subject: Self::rename_term(&g.subject, &suffix),
            object: Self::rename_term(&g.object, &suffix),
        };
        Clause {
            head: rename_goal(&clause.head),
            body: clause.body.iter().map(rename_goal).collect(),
        }
    }

    fn rename_term(term: &str, suffix: &str) -> String {
        if Substitution::is_variable(term) {
            format!("{}{}", term, suffix)
        } else {
            term.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_chainer() -> BackwardChainer {
        let mut bc = BackwardChainer::new(10);
        // Facts: parent(tom, bob), parent(bob, ann)
        bc.add_fact("parent", "tom", "bob");
        bc.add_fact("parent", "bob", "ann");
        // Rule: ancestor(X, Y) :- parent(X, Y)
        bc.add_clause(Clause {
            head: Goal::new("ancestor", "?X", "?Y"),
            body: vec![Goal::new("parent", "?X", "?Y")],
        });
        // Rule: ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)
        bc.add_clause(Clause {
            head: Goal::new("ancestor", "?X", "?Z"),
            body: vec![
                Goal::new("parent", "?X", "?Y"),
                Goal::new("ancestor", "?Y", "?Z"),
            ],
        });
        bc
    }

    // --- basic fact proving ---

    #[test]
    fn test_prove_simple_fact() {
        let bc = simple_chainer();
        let goal = Goal::new("parent", "tom", "bob");
        assert!(bc.can_prove(&goal));
    }

    #[test]
    fn test_prove_unknown_fact() {
        let bc = simple_chainer();
        let goal = Goal::new("parent", "ann", "tom");
        assert!(!bc.can_prove(&goal));
    }

    #[test]
    fn test_prove_direct_ancestor() {
        let bc = simple_chainer();
        let goal = Goal::new("ancestor", "tom", "bob");
        assert!(bc.can_prove(&goal));
    }

    #[test]
    fn test_prove_chained_ancestor() {
        let bc = simple_chainer();
        let goal = Goal::new("ancestor", "tom", "ann");
        assert!(bc.can_prove(&goal));
    }

    #[test]
    fn test_prove_returns_proof_tree() {
        let bc = simple_chainer();
        let goal = Goal::new("parent", "tom", "bob");
        let proof = bc.prove(&goal);
        assert!(proof.is_some());
        let tree = proof.unwrap();
        assert_eq!(tree.goal.predicate, "parent");
    }

    #[test]
    fn test_proof_tree_has_correct_goal() {
        let bc = simple_chainer();
        let goal = Goal::new("ancestor", "tom", "ann");
        let tree = bc.prove(&goal).unwrap();
        assert_eq!(tree.goal.subject, "tom");
        assert_eq!(tree.goal.object, "ann");
    }

    // --- clause_count ---

    #[test]
    fn test_clause_count_initial() {
        let bc = BackwardChainer::new(5);
        assert_eq!(bc.clause_count(), 0);
    }

    #[test]
    fn test_clause_count_after_facts() {
        let mut bc = BackwardChainer::new(5);
        bc.add_fact("p", "a", "b");
        bc.add_fact("p", "b", "c");
        assert_eq!(bc.clause_count(), 2);
    }

    #[test]
    fn test_clause_count_with_rules() {
        let bc = simple_chainer();
        // 2 facts + 2 rules = 4
        assert_eq!(bc.clause_count(), 4);
    }

    // --- add_fact ---

    #[test]
    fn test_add_fact_increases_count() {
        let mut bc = BackwardChainer::new(5);
        bc.add_fact("likes", "alice", "bob");
        assert_eq!(bc.clause_count(), 1);
    }

    #[test]
    fn test_prove_added_fact() {
        let mut bc = BackwardChainer::new(5);
        bc.add_fact("likes", "alice", "bob");
        assert!(bc.can_prove(&Goal::new("likes", "alice", "bob")));
    }

    // --- unification ---

    #[test]
    fn test_unify_same_ground_terms() {
        let g1 = Goal::new("p", "a", "b");
        let g2 = Goal::new("p", "a", "b");
        assert!(BackwardChainer::unify(&g1, &g2).is_some());
    }

    #[test]
    fn test_unify_different_predicate() {
        let g1 = Goal::new("p", "a", "b");
        let g2 = Goal::new("q", "a", "b");
        assert!(BackwardChainer::unify(&g1, &g2).is_none());
    }

    #[test]
    fn test_unify_variable_binds() {
        let g1 = Goal::new("p", "?X", "b");
        let g2 = Goal::new("p", "a", "b");
        let sub = BackwardChainer::unify(&g1, &g2).unwrap();
        assert_eq!(sub.get("X"), Some("a"));
    }

    #[test]
    fn test_unify_conflict_fails() {
        let g1 = Goal::new("p", "a", "b");
        let g2 = Goal::new("p", "c", "b");
        assert!(BackwardChainer::unify(&g1, &g2).is_none());
    }

    #[test]
    fn test_unify_both_variables() {
        let g1 = Goal::new("p", "?X", "?Y");
        let g2 = Goal::new("p", "tom", "ann");
        let sub = BackwardChainer::unify(&g1, &g2).unwrap();
        assert_eq!(sub.get("X"), Some("tom"));
        assert_eq!(sub.get("Y"), Some("ann"));
    }

    // --- apply_substitution ---

    #[test]
    fn test_apply_substitution_binds_variable() {
        let goal = Goal::new("p", "?X", "?Y");
        let mut sub = Substitution::new();
        sub.bind("X".to_string(), "a".to_string());
        sub.bind("Y".to_string(), "b".to_string());
        let result = BackwardChainer::apply_substitution(&goal, &sub);
        assert_eq!(result.subject, "a");
        assert_eq!(result.object, "b");
    }

    #[test]
    fn test_apply_substitution_keeps_constant() {
        let goal = Goal::new("p", "const", "?Y");
        let mut sub = Substitution::new();
        sub.bind("Y".to_string(), "val".to_string());
        let result = BackwardChainer::apply_substitution(&goal, &sub);
        assert_eq!(result.subject, "const");
        assert_eq!(result.object, "val");
    }

    #[test]
    fn test_apply_substitution_unbound_variable_stays() {
        let goal = Goal::new("p", "?X", "b");
        let sub = Substitution::new();
        let result = BackwardChainer::apply_substitution(&goal, &sub);
        assert_eq!(result.subject, "?X");
    }

    // --- Substitution ---

    #[test]
    fn test_substitution_is_variable_true() {
        assert!(Substitution::is_variable("?X"));
    }

    #[test]
    fn test_substitution_is_variable_false() {
        assert!(!Substitution::is_variable("alice"));
    }

    #[test]
    fn test_substitution_bind_and_get() {
        let mut sub = Substitution::new();
        sub.bind("X".to_string(), "hello".to_string());
        assert_eq!(sub.get("X"), Some("hello"));
    }

    #[test]
    fn test_substitution_get_missing() {
        let sub = Substitution::new();
        assert!(sub.get("Z").is_none());
    }

    #[test]
    fn test_substitution_compose() {
        let mut s1 = Substitution::new();
        s1.bind("X".to_string(), "a".to_string());

        let mut s2 = Substitution::new();
        s2.bind("Y".to_string(), "b".to_string());

        let composed = s1.compose(&s2);
        assert_eq!(composed.get("X"), Some("a"));
        assert_eq!(composed.get("Y"), Some("b"));
    }

    #[test]
    fn test_substitution_compose_resolves_variable_chain() {
        let mut s1 = Substitution::new();
        s1.bind("X".to_string(), "?Y".to_string());

        let mut s2 = Substitution::new();
        s2.bind("Y".to_string(), "val".to_string());

        let composed = s1.compose(&s2);
        assert_eq!(composed.get("X"), Some("val"));
    }

    // --- prove_all ---

    #[test]
    fn test_prove_all_multiple_solutions() {
        let mut bc = BackwardChainer::new(5);
        bc.add_fact("color", "rose", "red");
        bc.add_fact("color", "sky", "blue");
        bc.add_fact("color", "grass", "green");
        // Query: color(?X, ?Y) - should find all 3
        let goal = Goal::new("color", "?X", "?Y");
        let proofs = bc.prove_all(&goal);
        assert_eq!(proofs.len(), 3);
    }

    #[test]
    fn test_prove_all_no_solutions() {
        let bc = BackwardChainer::new(5);
        let goal = Goal::new("unknown", "x", "y");
        let proofs = bc.prove_all(&goal);
        assert!(proofs.is_empty());
    }

    // --- max_depth ---

    #[test]
    fn test_max_depth_limits_recursion() {
        // Very deep chain with max_depth=1 should fail
        let mut bc = BackwardChainer::new(1);
        bc.add_fact("parent", "tom", "bob");
        bc.add_fact("parent", "bob", "ann");
        bc.add_clause(Clause {
            head: Goal::new("ancestor", "?X", "?Z"),
            body: vec![
                Goal::new("parent", "?X", "?Y"),
                Goal::new("ancestor", "?Y", "?Z"),
            ],
        });
        // Direct parent is ok
        assert!(bc.can_prove(&Goal::new("parent", "tom", "bob")));
    }

    #[test]
    fn test_proof_tree_used_clause_none_for_fact() {
        let mut bc = BackwardChainer::new(5);
        bc.add_fact("likes", "alice", "bob");
        let tree = bc.prove(&Goal::new("likes", "alice", "bob")).unwrap();
        assert!(tree.used_clause.is_none());
    }

    #[test]
    fn test_proof_tree_used_clause_some_for_rule() {
        let bc = simple_chainer();
        let tree = bc.prove(&Goal::new("ancestor", "tom", "bob")).unwrap();
        assert!(tree.used_clause.is_some());
    }

    #[test]
    fn test_can_prove_false() {
        let bc = simple_chainer();
        assert!(!bc.can_prove(&Goal::new("parent", "ann", "tom")));
    }

    #[test]
    fn test_prove_all_single_solution() {
        let bc = simple_chainer();
        let goal = Goal::new("parent", "tom", "bob");
        let proofs = bc.prove_all(&goal);
        assert!(!proofs.is_empty());
    }

    #[test]
    fn test_proof_tree_children_for_rule() {
        let bc = simple_chainer();
        let tree = bc.prove(&Goal::new("ancestor", "tom", "bob")).unwrap();
        // The rule "ancestor :- parent" should have one child
        assert!(!tree.children.is_empty());
    }

    #[test]
    fn test_goal_new() {
        let g = Goal::new("foo", "a", "b");
        assert_eq!(g.predicate, "foo");
        assert_eq!(g.subject, "a");
        assert_eq!(g.object, "b");
    }
}
