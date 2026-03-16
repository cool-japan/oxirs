//! Rule syntax and semantic validation for the OxiRS rule engine.
//!
//! Validates individual rules for structural correctness (bound variables,
//! non-empty head/body) and validates rule sets for semantic issues such as
//! duplicate IDs and circular dependencies.

use std::collections::{HashMap, HashSet};

// ─── types ────────────────────────────────────────────────────────────────────

/// The logical role a rule plays in the knowledge base.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RuleKind {
    /// Classic if-then implication (body ⇒ head).
    Implication,
    /// Bidirectional rule (body ⟺ head).
    Equivalence,
    /// Integrity constraint: head must be empty or a sentinel.
    Integrity,
    /// Transformation / rewrite rule.
    Transformation,
}

/// The consequent of a rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleHead {
    pub predicate: String,
    pub variables: Vec<String>,
}

/// A single triple-pattern condition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Condition {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Condition {
    /// Collect all variable references (tokens starting with `?`) in this condition.
    pub fn variables(&self) -> Vec<String> {
        [&self.subject, &self.predicate, &self.object]
            .iter()
            .filter(|t| t.starts_with('?'))
            .map(|t| t[1..].to_string())
            .collect()
    }
}

/// The antecedent of a rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleBody {
    /// Positive conditions that must hold.
    pub conditions: Vec<Condition>,
    /// Negated conditions (NAF / MINUS).
    pub negations: Vec<Condition>,
}

impl RuleBody {
    /// All variables referenced in positive conditions.
    pub fn positive_vars(&self) -> HashSet<String> {
        self.conditions.iter().flat_map(|c| c.variables()).collect()
    }
}

/// A single rule in the knowledge base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    pub id: String,
    pub kind: RuleKind,
    pub head: RuleHead,
    pub body: RuleBody,
}

// ─── validation errors ────────────────────────────────────────────────────────

/// Errors that may be reported during validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// The rule head has no predicate or variables (empty consequent).
    EmptyHead,
    /// The rule body has no conditions (empty antecedent).
    EmptyBody,
    /// A variable in the head is not bound by any body condition.
    UnboundVariable(String),
    /// Two rules in a ruleset share the same ID.
    DuplicateId(String),
    /// A cycle exists among rule dependencies; the vector contains the IDs in cycle order.
    CircularDependency(Vec<String>),
    /// A predicate string is syntactically invalid (e.g., empty).
    InvalidPredicate(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyHead => write!(f, "Rule head is empty"),
            Self::EmptyBody => write!(f, "Rule body is empty"),
            Self::UnboundVariable(v) => write!(f, "Unbound head variable: ?{}", v),
            Self::DuplicateId(id) => write!(f, "Duplicate rule ID: {}", id),
            Self::CircularDependency(cycle) => {
                write!(f, "Circular dependency: {}", cycle.join(" → "))
            }
            Self::InvalidPredicate(p) => write!(f, "Invalid predicate: {}", p),
        }
    }
}

// ─── validator ────────────────────────────────────────────────────────────────

/// Validates rules individually and as a set.
#[derive(Debug, Clone, Default)]
pub struct RuleValidator;

impl RuleValidator {
    /// Create a new validator (stateless).
    pub fn new() -> Self {
        Self
    }

    // ── single-rule checks ─────────────────────────────────────────────────

    /// Validate a single rule, returning all errors found.
    ///
    /// Checks performed:
    /// 1. Head predicate must be non-empty.
    /// 2. Body must have at least one positive condition.
    /// 3. Every variable in `head.variables` must appear in at least one body condition.
    /// 4. All condition predicates must be non-empty.
    pub fn validate_rule(&self, rule: &Rule) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // (1) Head predicate must be non-empty.
        if rule.head.predicate.is_empty() {
            errors.push(ValidationError::EmptyHead);
        } else if !Self::is_valid_predicate(&rule.head.predicate) {
            errors.push(ValidationError::InvalidPredicate(
                rule.head.predicate.clone(),
            ));
        }

        // (2) Body must have at least one positive condition.
        if rule.body.conditions.is_empty() {
            errors.push(ValidationError::EmptyBody);
        }

        // (3) Check head variables are bound.
        let body_vars = rule.body.positive_vars();
        for var in &rule.head.variables {
            if !body_vars.contains(var) {
                errors.push(ValidationError::UnboundVariable(var.clone()));
            }
        }

        // (4) All condition predicates must be non-empty.
        for cond in rule
            .body
            .conditions
            .iter()
            .chain(rule.body.negations.iter())
        {
            if cond.predicate.is_empty() {
                errors.push(ValidationError::InvalidPredicate(cond.predicate.clone()));
            }
        }

        errors
    }

    // ── ruleset checks ─────────────────────────────────────────────────────

    /// Validate a slice of rules as a set.
    ///
    /// In addition to per-rule checks, this detects:
    /// - Duplicate rule IDs.
    /// - Circular predicate dependencies (head predicate referenced in another rule's body).
    pub fn validate_ruleset(&self, rules: &[Rule]) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Per-rule validation.
        for rule in rules {
            errors.extend(self.validate_rule(rule));
        }

        // Duplicate ID detection.
        let mut seen_ids: HashMap<&str, usize> = HashMap::new();
        for rule in rules {
            let count = seen_ids.entry(&rule.id).or_insert(0);
            *count += 1;
            if *count == 2 {
                errors.push(ValidationError::DuplicateId(rule.id.clone()));
            }
        }

        // Circular dependency detection.
        let cycles = self.detect_cycles(rules);
        for cycle in cycles {
            errors.push(ValidationError::CircularDependency(cycle));
        }

        errors
    }

    // ── cycle detection ────────────────────────────────────────────────────

    /// Detect circular predicate dependencies among rules.
    ///
    /// Builds a dependency graph where an edge `A → B` means rule A's head
    /// predicate appears in a body condition predicate of rule B.  Then runs
    /// DFS-based cycle detection and returns each cycle as an ordered list of
    /// rule IDs.
    pub fn detect_cycles(&self, rules: &[Rule]) -> Vec<Vec<String>> {
        // Map predicate → set of rule IDs that define it (head).
        let mut pred_to_rules: HashMap<&str, Vec<&str>> = HashMap::new();
        for rule in rules {
            pred_to_rules
                .entry(&rule.head.predicate)
                .or_default()
                .push(&rule.id);
        }

        // Build adjacency: rule_id → set of rule_ids it depends on (body predicates).
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for rule in rules {
            let deps: Vec<&str> = rule
                .body
                .conditions
                .iter()
                .flat_map(|c| {
                    pred_to_rules
                        .get(c.predicate.as_str())
                        .map(|v| v.as_slice())
                        .unwrap_or(&[])
                })
                .copied()
                .collect();
            adj.entry(&rule.id).or_default().extend(deps);
        }

        // Iterative DFS cycle detection (Tarjan-like coloring: 0=white,1=grey,2=black).
        let mut color: HashMap<&str, u8> = HashMap::new();
        let mut cycles: Vec<Vec<String>> = Vec::new();

        for rule in rules {
            if color.get(rule.id.as_str()).copied().unwrap_or(0) == 0 {
                Self::dfs_detect_cycles(&rule.id, &adj, &mut color, &mut Vec::new(), &mut cycles);
            }
        }

        cycles
    }

    // ─ private helpers ─────────────────────────────────────────────────────

    fn dfs_detect_cycles<'a>(
        node: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        color: &mut HashMap<&'a str, u8>,
        path: &mut Vec<&'a str>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        // Use an explicit stack to avoid recursion depth issues.
        // Each stack frame: (node, neighbour_index).
        let mut stack: Vec<(&str, usize)> = Vec::new();
        color.insert(node, 1);
        path.push(node);
        stack.push((node, 0));

        'outer: while let Some((current, idx)) = stack.last_mut() {
            let current = *current;
            let neighbours = adj.get(current).map(|v| v.as_slice()).unwrap_or(&[]);

            if *idx < neighbours.len() {
                let next = neighbours[*idx];
                *idx += 1;

                match color.get(next).copied().unwrap_or(0) {
                    1 => {
                        // Back edge → cycle found.
                        if let Some(start) = path.iter().position(|&p| p == next) {
                            let cycle: Vec<String> =
                                path[start..].iter().map(|s| s.to_string()).collect();
                            // Avoid duplicate cycle reports.
                            if !cycles.contains(&cycle) {
                                cycles.push(cycle);
                            }
                        }
                    }
                    0 => {
                        color.insert(next, 1);
                        path.push(next);
                        stack.push((next, 0));
                        continue 'outer;
                    }
                    _ => {} // Already fully explored.
                }
            } else {
                // Done with this node.
                color.insert(current, 2);
                stack.pop();
                path.pop();
            }
        }
    }

    fn is_valid_predicate(pred: &str) -> bool {
        !pred.is_empty()
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn cond(s: &str, p: &str, o: &str) -> Condition {
        Condition {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
        }
    }

    fn make_rule(
        id: &str,
        head_pred: &str,
        head_vars: Vec<&str>,
        body_conds: Vec<Condition>,
        negations: Vec<Condition>,
        kind: RuleKind,
    ) -> Rule {
        Rule {
            id: id.to_string(),
            kind,
            head: RuleHead {
                predicate: head_pred.to_string(),
                variables: head_vars.iter().map(|v| v.to_string()).collect(),
            },
            body: RuleBody {
                conditions: body_conds,
                negations,
            },
        }
    }

    fn simple_rule(
        id: &str,
        head_pred: &str,
        head_vars: Vec<&str>,
        body_conds: Vec<Condition>,
    ) -> Rule {
        make_rule(
            id,
            head_pred,
            head_vars,
            body_conds,
            vec![],
            RuleKind::Implication,
        )
    }

    // ── Condition tests ───────────────────────────────────────────────────────

    #[test]
    fn test_condition_variables_with_variables() -> anyhow::Result<()> {
        let c = cond("?x", "rdf:type", "?y");
        let mut vars = c.variables();
        vars.sort();
        assert_eq!(vars, vec!["x", "y"]);
        Ok(())
    }

    #[test]
    fn test_condition_variables_no_variables() {
        let c = cond(":alice", "rdf:type", ":Person");
        assert!(c.variables().is_empty());
    }

    #[test]
    fn test_condition_variables_mixed() -> anyhow::Result<()> {
        let c = cond("?s", "rdfs:label", ":literal");
        let vars = c.variables();
        assert_eq!(vars, vec!["s"]);
        Ok(())
    }

    // ── RuleBody tests ────────────────────────────────────────────────────────

    #[test]
    fn test_rule_body_positive_vars() -> anyhow::Result<()> {
        let body = RuleBody {
            conditions: vec![
                cond("?x", "rdf:type", "?y"),
                cond("?y", "rdfs:subClassOf", "?z"),
            ],
            negations: vec![],
        };
        let vars = body.positive_vars();
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(vars.contains("z"));
        Ok(())
    }

    #[test]
    fn test_rule_body_positive_vars_ignores_negations() -> anyhow::Result<()> {
        let body = RuleBody {
            conditions: vec![cond("?x", "rdf:type", ":Person")],
            negations: vec![cond("?x", "rdf:type", "?z")],
        };
        let vars = body.positive_vars();
        assert!(vars.contains("x"));
        assert!(!vars.contains("z")); // z only in negation
        Ok(())
    }

    // ── validate_rule: valid rules ────────────────────────────────────────────

    #[test]
    fn test_validate_rule_valid_simple() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = simple_rule(
            "r1",
            "ancestor",
            vec!["x", "y"],
            vec![cond("?x", "parent", "?y")],
        );
        let errors = validator.validate_rule(&rule);
        assert!(errors.is_empty());
        Ok(())
    }

    #[test]
    fn test_validate_rule_valid_no_head_vars() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = simple_rule("r1", "exists", vec![], vec![cond("?x", "rdf:type", ":T")]);
        let errors = validator.validate_rule(&rule);
        assert!(errors.is_empty());
        Ok(())
    }

    #[test]
    fn test_validate_rule_valid_with_negation() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = make_rule(
            "r1",
            "safe",
            vec!["x"],
            vec![cond("?x", "rdf:type", ":Agent")],
            vec![cond("?x", "rdf:type", ":Banned")],
            RuleKind::Implication,
        );
        let errors = validator.validate_rule(&rule);
        assert!(errors.is_empty());
        Ok(())
    }

    // ── validate_rule: invalid rules ─────────────────────────────────────────

    #[test]
    fn test_validate_rule_empty_head_predicate() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = simple_rule("r1", "", vec![], vec![cond("?x", "rdf:type", ":T")]);
        let errors = validator.validate_rule(&rule);
        assert!(errors.contains(&ValidationError::EmptyHead));
        Ok(())
    }

    #[test]
    fn test_validate_rule_empty_body() {
        let validator = RuleValidator::new();
        let rule = simple_rule("r1", "foo", vec![], vec![]);
        let errors = validator.validate_rule(&rule);
        assert!(errors.contains(&ValidationError::EmptyBody));
    }

    #[test]
    fn test_validate_rule_unbound_variable() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        // Head has variable "z" but body only binds "x" and "y".
        let rule = simple_rule(
            "r1",
            "rel",
            vec!["x", "z"],
            vec![cond("?x", "parent", "?y")],
        );
        let errors = validator.validate_rule(&rule);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnboundVariable(v) if v == "z")));
        Ok(())
    }

    #[test]
    fn test_validate_rule_all_three_errors() {
        let validator = RuleValidator::new();
        let rule = simple_rule("r1", "", vec!["z"], vec![]);
        let errors = validator.validate_rule(&rule);
        assert!(errors.contains(&ValidationError::EmptyHead));
        assert!(errors.contains(&ValidationError::EmptyBody));
        // z unbound (body empty, so no positive vars)
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnboundVariable(_))));
    }

    #[test]
    fn test_validate_rule_invalid_body_predicate() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = simple_rule("r1", "foo", vec![], vec![cond("?x", "", ":T")]);
        let errors = validator.validate_rule(&rule);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidPredicate(_))));
        Ok(())
    }

    #[test]
    fn test_validate_rule_all_vars_bound() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = simple_rule(
            "r1",
            "triple_rel",
            vec!["x", "y", "z"],
            vec![cond("?x", "p", "?y"), cond("?y", "q", "?z")],
        );
        let errors = validator.validate_rule(&rule);
        assert!(errors.is_empty());
        Ok(())
    }

    // ── validate_ruleset: valid sets ──────────────────────────────────────────

    #[test]
    fn test_validate_ruleset_empty() {
        let validator = RuleValidator::new();
        let errors = validator.validate_ruleset(&[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_ruleset_single_valid_rule() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = simple_rule(
            "r1",
            "ancestor",
            vec!["x"],
            vec![cond("?x", "parent", "?y")],
        );
        let errors = validator.validate_ruleset(&[rule]);
        assert!(errors.is_empty());
        Ok(())
    }

    #[test]
    fn test_validate_ruleset_multiple_valid_rules() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let r1 = simple_rule(
            "r1",
            "ancestor",
            vec!["x"],
            vec![cond("?x", "parent", "?y")],
        );
        let r2 = simple_rule(
            "r2",
            "human",
            vec![],
            vec![cond("?x", "rdf:type", ":Person")],
        );
        let errors = validator.validate_ruleset(&[r1, r2]);
        assert!(errors.is_empty());
        Ok(())
    }

    // ── validate_ruleset: duplicate IDs ──────────────────────────────────────

    #[test]
    fn test_validate_ruleset_duplicate_id() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let r1 = simple_rule("dup", "a", vec![], vec![cond("?x", "p", ":o")]);
        let r2 = simple_rule("dup", "b", vec![], vec![cond("?y", "q", ":o")]);
        let errors = validator.validate_ruleset(&[r1, r2]);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::DuplicateId(id) if id == "dup")));
        Ok(())
    }

    #[test]
    fn test_validate_ruleset_unique_ids_no_error() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rules: Vec<Rule> = (0..5)
            .map(|i| {
                simple_rule(
                    &format!("r{}", i),
                    "foo",
                    vec![],
                    vec![cond("?x", "p", ":o")],
                )
            })
            .collect();
        let errors = validator.validate_ruleset(&rules);
        assert!(!errors
            .iter()
            .any(|e| matches!(e, ValidationError::DuplicateId(_))));
        Ok(())
    }

    // ── detect_cycles ─────────────────────────────────────────────────────────

    #[test]
    fn test_detect_cycles_no_cycles() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        // r1: body uses "parent", head defines "ancestor"
        // r2: body uses "ancestor", head defines "knows"
        let r1 = simple_rule("r1", "ancestor", vec![], vec![cond("?x", "parent", "?y")]);
        let r2 = simple_rule("r2", "knows", vec![], vec![cond("?x", "ancestor", "?y")]);
        let cycles = validator.detect_cycles(&[r1, r2]);
        assert!(cycles.is_empty());
        Ok(())
    }

    #[test]
    fn test_detect_cycles_self_loop() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        // Rule's body references its own head predicate.
        let rule = simple_rule("r1", "loop", vec![], vec![cond("?x", "loop", "?y")]);
        let cycles = validator.detect_cycles(&[rule]);
        assert!(!cycles.is_empty());
        assert!(cycles[0].contains(&"r1".to_string()));
        Ok(())
    }

    #[test]
    fn test_detect_cycles_two_node_cycle() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let r1 = simple_rule("r1", "a", vec![], vec![cond("?x", "b", "?y")]);
        let r2 = simple_rule("r2", "b", vec![], vec![cond("?x", "a", "?y")]);
        let cycles = validator.detect_cycles(&[r1, r2]);
        assert!(!cycles.is_empty());
        Ok(())
    }

    #[test]
    fn test_detect_cycles_no_cycle_chain() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let r1 = simple_rule("r1", "a", vec![], vec![cond("?x", "base", "?y")]);
        let r2 = simple_rule("r2", "b", vec![], vec![cond("?x", "a", "?y")]);
        let r3 = simple_rule("r3", "c", vec![], vec![cond("?x", "b", "?y")]);
        let cycles = validator.detect_cycles(&[r1, r2, r3]);
        assert!(cycles.is_empty());
        Ok(())
    }

    #[test]
    fn test_validate_ruleset_reports_cycle() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let r1 = simple_rule("r1", "a", vec![], vec![cond("?x", "b", "?y")]);
        let r2 = simple_rule("r2", "b", vec![], vec![cond("?x", "a", "?y")]);
        let errors = validator.validate_ruleset(&[r1, r2]);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::CircularDependency(_))));
        Ok(())
    }

    // ── display tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_display_empty_head() {
        let e = ValidationError::EmptyHead;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn test_display_empty_body() {
        let e = ValidationError::EmptyBody;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn test_display_unbound_variable() {
        let e = ValidationError::UnboundVariable("z".to_string());
        assert!(e.to_string().contains("z"));
    }

    #[test]
    fn test_display_duplicate_id() {
        let e = ValidationError::DuplicateId("r1".to_string());
        assert!(e.to_string().contains("r1"));
    }

    #[test]
    fn test_display_circular_dependency() {
        let e = ValidationError::CircularDependency(vec!["r1".to_string(), "r2".to_string()]);
        let s = e.to_string();
        assert!(s.contains("r1"));
        assert!(s.contains("r2"));
    }

    #[test]
    fn test_display_invalid_predicate() {
        let e = ValidationError::InvalidPredicate("bad::pred".to_string());
        assert!(e.to_string().contains("bad::pred"));
    }

    // ── RuleKind exhaustiveness ───────────────────────────────────────────────

    #[test]
    fn test_all_rule_kinds_accepted() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        for kind in [
            RuleKind::Implication,
            RuleKind::Equivalence,
            RuleKind::Integrity,
            RuleKind::Transformation,
        ] {
            let rule = make_rule(
                "r1",
                "foo",
                vec![],
                vec![cond("?x", "p", ":o")],
                vec![],
                kind,
            );
            let errors = validator.validate_rule(&rule);
            assert!(errors.is_empty());
        }
        Ok(())
    }

    // ── extra coverage ────────────────────────────────────────────────────────

    #[test]
    fn test_validate_ruleset_combines_per_rule_and_set_errors() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        // r1 is valid; r2 has duplicate id with r1 and also empty body.
        let r1 = simple_rule("dup", "foo", vec![], vec![cond("?x", "p", ":o")]);
        let r2 = simple_rule("dup", "bar", vec![], vec![]);
        let errors = validator.validate_ruleset(&[r1, r2]);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::DuplicateId(_))));
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::EmptyBody)));
        Ok(())
    }

    #[test]
    fn test_rule_validator_default() -> anyhow::Result<()> {
        let v = RuleValidator;
        let rule = simple_rule("r1", "p", vec![], vec![cond("?x", "q", ":o")]);
        assert!(v.validate_rule(&rule).is_empty());
        Ok(())
    }

    #[test]
    fn test_unbound_variable_multiple() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = simple_rule(
            "r1",
            "rel",
            vec!["x", "y", "z"],
            vec![cond("?w", "p", ":o")],
        );
        let errors = validator.validate_rule(&rule);
        let unbound: Vec<&str> = errors
            .iter()
            .filter_map(|e| {
                if let ValidationError::UnboundVariable(v) = e {
                    Some(v.as_str())
                } else {
                    None
                }
            })
            .collect();
        // x, y, z are all unbound
        assert!(unbound.contains(&"x"));
        assert!(unbound.contains(&"y"));
        assert!(unbound.contains(&"z"));
        Ok(())
    }

    #[test]
    fn test_three_node_cycle() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let r1 = simple_rule("r1", "a", vec![], vec![cond("?x", "c", "?y")]);
        let r2 = simple_rule("r2", "b", vec![], vec![cond("?x", "a", "?y")]);
        let r3 = simple_rule("r3", "c", vec![], vec![cond("?x", "b", "?y")]);
        let cycles = validator.detect_cycles(&[r1, r2, r3]);
        assert!(!cycles.is_empty());
        Ok(())
    }

    #[test]
    fn test_acyclic_diamond() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        // a → b, a → c, b → d, c → d (diamond, no cycle)
        let r1 = simple_rule("r1", "b", vec![], vec![cond("?x", "a", "?y")]);
        let r2 = simple_rule("r2", "c", vec![], vec![cond("?x", "a", "?y")]);
        let r3 = simple_rule("r3", "d", vec![], vec![cond("?x", "b", "?y")]);
        let r4 = simple_rule("r4", "d2", vec![], vec![cond("?x", "c", "?y")]);
        let cycles = validator.detect_cycles(&[r1, r2, r3, r4]);
        assert!(cycles.is_empty());
        Ok(())
    }

    #[test]
    fn test_validate_rule_equivalence_kind() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = make_rule(
            "r1",
            "equiv",
            vec![],
            vec![cond("?x", "p", ":o")],
            vec![],
            RuleKind::Equivalence,
        );
        assert!(validator.validate_rule(&rule).is_empty());
        Ok(())
    }

    #[test]
    fn test_validate_rule_integrity_kind() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = make_rule(
            "r1",
            "integrity_check",
            vec![],
            vec![cond("?x", "rdf:type", ":T")],
            vec![],
            RuleKind::Integrity,
        );
        assert!(validator.validate_rule(&rule).is_empty());
        Ok(())
    }

    #[test]
    fn test_validate_rule_transformation_kind() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = make_rule(
            "r1",
            "transform_out",
            vec![],
            vec![cond("?x", "transform_in", ":val")],
            vec![],
            RuleKind::Transformation,
        );
        assert!(validator.validate_rule(&rule).is_empty());
        Ok(())
    }

    #[test]
    fn test_validate_ruleset_no_duplicate_different_predicates() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let r1 = simple_rule("rule_alpha", "foo", vec![], vec![cond("?x", "p", ":o")]);
        let r2 = simple_rule("rule_beta", "bar", vec![], vec![cond("?y", "q", ":o")]);
        let errors = validator.validate_ruleset(&[r1, r2]);
        assert!(!errors
            .iter()
            .any(|e| matches!(e, ValidationError::DuplicateId(_))));
        Ok(())
    }

    #[test]
    fn test_condition_variable_in_predicate_position() -> anyhow::Result<()> {
        // Predicate positions starting with '?' are also variables.
        let c = cond(":s", "?p", ":o");
        let vars = c.variables();
        assert_eq!(vars, vec!["p"]);
        Ok(())
    }

    #[test]
    fn test_validate_rule_body_object_binds_head_var() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        let rule = simple_rule(
            "r1",
            "found",
            vec!["o"],
            vec![cond(":subject", "rdf:type", "?o")],
        );
        let errors = validator.validate_rule(&rule);
        assert!(errors.is_empty());
        Ok(())
    }

    #[test]
    fn test_validate_ruleset_single_invalid_rule() {
        let validator = RuleValidator::new();
        let rule = simple_rule("r1", "", vec![], vec![]);
        let errors = validator.validate_ruleset(&[rule]);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::EmptyHead)));
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::EmptyBody)));
    }

    #[test]
    fn test_detect_cycles_empty_ruleset() {
        let validator = RuleValidator::new();
        let cycles = validator.detect_cycles(&[]);
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_validate_rule_negation_invalid_predicate() -> anyhow::Result<()> {
        let validator = RuleValidator::new();
        // Empty predicate in negation condition.
        let rule = make_rule(
            "r1",
            "head_pred",
            vec![],
            vec![cond("?x", "rdf:type", ":T")],
            vec![cond("?x", "", ":banned")],
            RuleKind::Implication,
        );
        let errors = validator.validate_rule(&rule);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidPredicate(_))));
        Ok(())
    }
}
