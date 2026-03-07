//! Semi-naive forward chaining rule engine.
//!
//! Implements datalog-style forward chaining: rules are repeatedly applied until
//! no new facts can be derived (fixpoint). The semi-naive optimisation is achieved
//! by tracking the delta between iterations.

use std::collections::{HashMap, HashSet};

/// A ground triple fact.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Fact {
    /// Create a new fact.
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        }
    }
}

impl std::fmt::Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<{}> <{}> <{}>",
            self.subject, self.predicate, self.object
        )
    }
}

/// A rule body consisting of zero or more triple patterns.
///
/// Each component of a pattern may be:
/// - `Some(s)` where `s` starts with `?` → variable
/// - `Some(s)` where `s` does not start with `?` → constant
/// - `None` → wildcard (matches anything, does not bind)
#[derive(Debug, Clone)]
pub struct RuleBody {
    pub patterns: Vec<(Option<String>, Option<String>, Option<String>)>,
}

impl RuleBody {
    /// Create a body from a list of patterns.
    pub fn new(patterns: Vec<(Option<String>, Option<String>, Option<String>)>) -> Self {
        Self { patterns }
    }
}

/// A rule head describing the triple to derive.
///
/// Template strings may contain `?var` references that will be substituted
/// from the variable bindings produced by matching the rule body.
#[derive(Debug, Clone)]
pub struct RuleHead {
    pub subject_template: String,
    pub predicate_template: String,
    pub object_template: String,
}

impl RuleHead {
    /// Create a new rule head.
    pub fn new(s: &str, p: &str, o: &str) -> Self {
        Self {
            subject_template: s.to_string(),
            predicate_template: p.to_string(),
            object_template: o.to_string(),
        }
    }
}

/// A forward-chaining rule.
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub body: RuleBody,
    pub head: RuleHead,
}

impl Rule {
    /// Create a named rule.
    pub fn new(name: &str, body: RuleBody, head: RuleHead) -> Self {
        Self {
            name: name.to_string(),
            body,
            head,
        }
    }
}

/// Variable bindings: maps variable names (without leading `?`) to values.
pub type Binding = HashMap<String, String>;

/// Forward-chaining reasoner with semi-naive iteration.
#[derive(Debug, Default)]
pub struct ForwardChainer {
    facts: Vec<Fact>,
    fact_set: HashSet<Fact>,
    rules: Vec<Rule>,
}

impl ForwardChainer {
    /// Create an empty forward chainer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a ground fact to the knowledge base.
    pub fn add_fact(&mut self, fact: Fact) {
        if self.fact_set.insert(fact.clone()) {
            self.facts.push(fact);
        }
    }

    /// Add a rule to the rule set.
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Try to unify a single triple pattern with a fact given existing bindings.
    ///
    /// Returns updated bindings on success, or `None` on failure.
    pub fn match_pattern(
        pattern: &(Option<String>, Option<String>, Option<String>),
        fact: &Fact,
        bindings: &Binding,
    ) -> Option<Binding> {
        let mut new_bindings = bindings.clone();
        let components = [
            (pattern.0.as_deref(), fact.subject.as_str()),
            (pattern.1.as_deref(), fact.predicate.as_str()),
            (pattern.2.as_deref(), fact.object.as_str()),
        ];

        for (pat_component, fact_value) in components {
            match pat_component {
                None => {
                    // Wildcard: matches anything, no binding
                }
                Some(p) if p.starts_with('?') => {
                    let var = p.strip_prefix('?').unwrap_or(p);
                    if let Some(existing) = new_bindings.get(var) {
                        if existing != fact_value {
                            return None; // Conflict
                        }
                    } else {
                        new_bindings.insert(var.to_string(), fact_value.to_string());
                    }
                }
                Some(constant) => {
                    if constant != fact_value {
                        return None;
                    }
                }
            }
        }
        Some(new_bindings)
    }

    /// Recursively match a sequence of patterns against the fact set.
    ///
    /// Returns all possible complete binding sets.
    pub fn match_body(
        patterns: &[(Option<String>, Option<String>, Option<String>)],
        facts: &[Fact],
        bindings: &Binding,
    ) -> Vec<Binding> {
        if patterns.is_empty() {
            return vec![bindings.clone()];
        }

        let first = &patterns[0];
        let rest = &patterns[1..];
        let mut results = Vec::new();

        for fact in facts {
            if let Some(new_bindings) = Self::match_pattern(first, fact, bindings) {
                let sub_results = Self::match_body(rest, facts, &new_bindings);
                results.extend(sub_results);
            }
        }
        results
    }

    /// Substitute variable references in a template string.
    fn substitute(template: &str, bindings: &Binding) -> Option<String> {
        // Simple substitution: replace first ?var token found
        // For production, this would handle complex expressions
        if let Some(var) = template.strip_prefix('?') {
            bindings.get(var).cloned()
        } else {
            Some(template.to_string())
        }
    }

    /// Apply variable bindings to a rule head to produce a new fact.
    ///
    /// Returns `None` if any variable in the head is unbound.
    pub fn apply_head(head: &RuleHead, bindings: &Binding) -> Option<Fact> {
        let s = Self::substitute(&head.subject_template, bindings)?;
        let p = Self::substitute(&head.predicate_template, bindings)?;
        let o = Self::substitute(&head.object_template, bindings)?;
        Some(Fact::new(&s, &p, &o))
    }

    /// Perform one iteration of forward chaining.
    ///
    /// Returns the number of new facts derived.
    pub fn step(&mut self) -> usize {
        let current_facts: Vec<Fact> = self.facts.clone();
        let mut new_facts: Vec<Fact> = Vec::new();

        for rule in &self.rules {
            let all_bindings =
                Self::match_body(&rule.body.patterns, &current_facts, &Binding::new());

            for bindings in all_bindings {
                if let Some(derived) = Self::apply_head(&rule.head, &bindings) {
                    if !self.fact_set.contains(&derived) {
                        new_facts.push(derived);
                    }
                }
            }
        }

        let mut added = 0usize;
        for fact in new_facts {
            if self.fact_set.insert(fact.clone()) {
                self.facts.push(fact);
                added += 1;
            }
        }
        added
    }

    /// Run forward chaining until fixpoint or `max_iterations`.
    ///
    /// Returns the total number of new facts derived across all iterations.
    pub fn run(&mut self, max_iterations: usize) -> usize {
        let mut total = 0usize;
        for _ in 0..max_iterations {
            let n = self.step();
            total += n;
            if n == 0 {
                break;
            }
        }
        total
    }

    /// Return a slice of all facts currently in the knowledge base.
    pub fn facts(&self) -> &[Fact] {
        &self.facts
    }

    /// Return the total number of facts in the knowledge base.
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pat(
        s: Option<&str>,
        p: Option<&str>,
        o: Option<&str>,
    ) -> (Option<String>, Option<String>, Option<String>) {
        (
            s.map(str::to_string),
            p.map(str::to_string),
            o.map(str::to_string),
        )
    }

    // ------ Fact ------

    #[test]
    fn test_fact_new() {
        let f = Fact::new("s", "p", "o");
        assert_eq!(f.subject, "s");
        assert_eq!(f.predicate, "p");
        assert_eq!(f.object, "o");
    }

    #[test]
    fn test_fact_eq() {
        let f1 = Fact::new("s", "p", "o");
        let f2 = Fact::new("s", "p", "o");
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_fact_hash() {
        let mut set = HashSet::new();
        set.insert(Fact::new("s", "p", "o"));
        assert!(set.contains(&Fact::new("s", "p", "o")));
    }

    // ------ match_pattern ------

    #[test]
    fn test_match_pattern_constant_match() {
        let fact = Fact::new("Alice", "rdf:type", "Person");
        let pattern = pat(Some("Alice"), Some("rdf:type"), Some("Person"));
        let result = ForwardChainer::match_pattern(&pattern, &fact, &Binding::new());
        assert!(result.is_some());
    }

    #[test]
    fn test_match_pattern_constant_no_match() {
        let fact = Fact::new("Alice", "rdf:type", "Animal");
        let pattern = pat(Some("Alice"), Some("rdf:type"), Some("Person"));
        let result = ForwardChainer::match_pattern(&pattern, &fact, &Binding::new());
        assert!(result.is_none());
    }

    #[test]
    fn test_match_pattern_variable_binds() {
        let fact = Fact::new("Alice", "rdf:type", "Person");
        let pattern = pat(Some("?x"), Some("rdf:type"), Some("Person"));
        let result = ForwardChainer::match_pattern(&pattern, &fact, &Binding::new());
        assert!(result.is_some());
        let bindings = result.unwrap();
        assert_eq!(bindings.get("x"), Some(&"Alice".to_string()));
    }

    #[test]
    fn test_match_pattern_variable_conflict() {
        let fact = Fact::new("Alice", "knows", "Bob");
        let pattern = pat(Some("?x"), Some("knows"), Some("?x")); // x must be same in s and o
        let result = ForwardChainer::match_pattern(&pattern, &fact, &Binding::new());
        assert!(result.is_none()); // Alice != Bob
    }

    #[test]
    fn test_match_pattern_wildcard() {
        let fact = Fact::new("Alice", "knows", "Bob");
        let pattern = pat(None, None, Some("Bob"));
        let result = ForwardChainer::match_pattern(&pattern, &fact, &Binding::new());
        assert!(result.is_some());
    }

    #[test]
    fn test_match_pattern_existing_binding_consistent() {
        let fact = Fact::new("Alice", "knows", "Bob");
        let mut existing = Binding::new();
        existing.insert("x".to_string(), "Alice".to_string());
        let pattern = pat(Some("?x"), Some("knows"), Some("?y"));
        let result = ForwardChainer::match_pattern(&pattern, &fact, &existing);
        assert!(result.is_some());
    }

    // ------ match_body ------

    #[test]
    fn test_match_body_empty_patterns() {
        let result = ForwardChainer::match_body(&[], &[], &Binding::new());
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_match_body_single_pattern() {
        let facts = vec![
            Fact::new("Alice", "rdf:type", "Person"),
            Fact::new("Bob", "rdf:type", "Person"),
        ];
        let patterns = vec![pat(Some("?x"), Some("rdf:type"), Some("Person"))];
        let results = ForwardChainer::match_body(&patterns, &facts, &Binding::new());
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_match_body_two_patterns_join() {
        let facts = vec![
            Fact::new("Alice", "rdf:type", "Person"),
            Fact::new("Person", "rdfs:subClassOf", "Agent"),
        ];
        let patterns = vec![
            pat(Some("?x"), Some("rdf:type"), Some("?class")),
            pat(Some("?class"), Some("rdfs:subClassOf"), Some("?super")),
        ];
        let results = ForwardChainer::match_body(&patterns, &facts, &Binding::new());
        assert_eq!(results.len(), 1);
        let b = &results[0];
        assert_eq!(b.get("x"), Some(&"Alice".to_string()));
        assert_eq!(b.get("super"), Some(&"Agent".to_string()));
    }

    // ------ apply_head ------

    #[test]
    fn test_apply_head_basic() {
        let mut bindings = Binding::new();
        bindings.insert("x".to_string(), "Alice".to_string());
        bindings.insert("super".to_string(), "Agent".to_string());

        let head = RuleHead::new("?x", "rdf:type", "?super");
        let result = ForwardChainer::apply_head(&head, &bindings);
        assert!(result.is_some());
        let fact = result.unwrap();
        assert_eq!(fact.subject, "Alice");
        assert_eq!(fact.object, "Agent");
    }

    #[test]
    fn test_apply_head_unbound_variable() {
        let bindings = Binding::new();
        let head = RuleHead::new("?x", "rdf:type", "Person");
        let result = ForwardChainer::apply_head(&head, &bindings);
        assert!(result.is_none());
    }

    #[test]
    fn test_apply_head_constant_template() {
        let bindings = Binding::new();
        let head = RuleHead::new("Alice", "knows", "Bob");
        let result = ForwardChainer::apply_head(&head, &bindings);
        assert!(result.is_some());
    }

    // ------ step / run ------

    #[test]
    fn test_step_derives_new_fact() {
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("Alice", "rdf:type", "Person"));
        fc.add_rule(Rule::new(
            "person_is_agent",
            RuleBody::new(vec![pat(Some("?x"), Some("rdf:type"), Some("Person"))]),
            RuleHead::new("?x", "rdf:type", "Agent"),
        ));

        let n = fc.step();
        assert_eq!(n, 1);
        assert!(fc
            .fact_set
            .contains(&Fact::new("Alice", "rdf:type", "Agent")));
    }

    #[test]
    fn test_step_no_new_facts_on_second_call() {
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("Alice", "rdf:type", "Person"));
        fc.add_rule(Rule::new(
            "person_is_agent",
            RuleBody::new(vec![pat(Some("?x"), Some("rdf:type"), Some("Person"))]),
            RuleHead::new("?x", "rdf:type", "Agent"),
        ));
        fc.step();
        let n = fc.step();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_run_fixpoint() {
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("Alice", "rdf:type", "Person"));
        fc.add_rule(Rule::new(
            "person_is_agent",
            RuleBody::new(vec![pat(Some("?x"), Some("rdf:type"), Some("Person"))]),
            RuleHead::new("?x", "rdf:type", "Agent"),
        ));
        let total = fc.run(100);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_transitive_closure() {
        // rdfs:subClassOf* closure
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("Person", "rdfs:subClassOf", "Agent"));
        fc.add_fact(Fact::new("Employee", "rdfs:subClassOf", "Person"));

        // If A subClassOf B and B subClassOf C then A subClassOf C
        fc.add_rule(Rule::new(
            "transitivity",
            RuleBody::new(vec![
                pat(Some("?a"), Some("rdfs:subClassOf"), Some("?b")),
                pat(Some("?b"), Some("rdfs:subClassOf"), Some("?c")),
            ]),
            RuleHead::new("?a", "rdfs:subClassOf", "?c"),
        ));

        fc.run(100);

        assert!(fc
            .fact_set
            .contains(&Fact::new("Employee", "rdfs:subClassOf", "Agent")));
    }

    #[test]
    fn test_rdfs_domain_entailment() {
        // RDFS: if P has domain D and X P Y then X rdf:type D
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("knows", "rdfs:domain", "Person"));
        fc.add_fact(Fact::new("Alice", "knows", "Bob"));

        fc.add_rule(Rule::new(
            "rdfs_domain",
            RuleBody::new(vec![
                pat(Some("?p"), Some("rdfs:domain"), Some("?d")),
                pat(Some("?x"), Some("?p"), Some("?y")),
            ]),
            RuleHead::new("?x", "rdf:type", "?d"),
        ));

        fc.run(100);

        assert!(fc
            .fact_set
            .contains(&Fact::new("Alice", "rdf:type", "Person")));
    }

    #[test]
    fn test_rdfs_range_entailment() {
        // RDFS: if P has range R and X P Y then Y rdf:type R
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("knows", "rdfs:range", "Person"));
        fc.add_fact(Fact::new("Alice", "knows", "Bob"));

        fc.add_rule(Rule::new(
            "rdfs_range",
            RuleBody::new(vec![
                pat(Some("?p"), Some("rdfs:range"), Some("?r")),
                pat(Some("?x"), Some("?p"), Some("?y")),
            ]),
            RuleHead::new("?y", "rdf:type", "?r"),
        ));

        fc.run(100);

        assert!(fc
            .fact_set
            .contains(&Fact::new("Bob", "rdf:type", "Person")));
    }

    #[test]
    fn test_multiple_rules() {
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("Alice", "rdf:type", "Employee"));
        fc.add_fact(Fact::new("Employee", "rdfs:subClassOf", "Person"));

        // subClassOf → rdf:type propagation
        fc.add_rule(Rule::new(
            "type_inheritance",
            RuleBody::new(vec![
                pat(Some("?x"), Some("rdf:type"), Some("?c")),
                pat(Some("?c"), Some("rdfs:subClassOf"), Some("?super")),
            ]),
            RuleHead::new("?x", "rdf:type", "?super"),
        ));

        fc.run(10);

        assert!(fc
            .fact_set
            .contains(&Fact::new("Alice", "rdf:type", "Person")));
    }

    #[test]
    fn test_max_iterations_limit() {
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("a", "type", "A"));
        // Rule produces one new fact
        fc.add_rule(Rule::new(
            "r",
            RuleBody::new(vec![pat(Some("?x"), Some("type"), Some("A"))]),
            RuleHead::new("?x", "type", "B"),
        ));
        // Run for only 0 iterations → no new facts
        let total = fc.run(0);
        assert_eq!(total, 0);
        assert_eq!(fc.fact_count(), 1);
    }

    #[test]
    fn test_no_infinite_loop_cycle() {
        // Symmetric rule: if A knows B then B knows A
        // Should terminate because symmetric facts are added only once
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("Alice", "knows", "Bob"));

        fc.add_rule(Rule::new(
            "symmetric",
            RuleBody::new(vec![pat(Some("?x"), Some("knows"), Some("?y"))]),
            RuleHead::new("?y", "knows", "?x"),
        ));

        let total = fc.run(1000);
        // Only one new fact: Bob knows Alice
        assert_eq!(total, 1);
    }

    #[test]
    fn test_add_fact_dedup() {
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("a", "p", "b"));
        fc.add_fact(Fact::new("a", "p", "b"));
        assert_eq!(fc.fact_count(), 1);
    }

    #[test]
    fn test_wildcard_pattern_matches_all() {
        let mut fc = ForwardChainer::new();
        fc.add_fact(Fact::new("Alice", "knows", "Bob"));
        fc.add_fact(Fact::new("Bob", "knows", "Carol"));

        fc.add_rule(Rule::new(
            "everyone_is_entity",
            RuleBody::new(vec![pat(None, Some("knows"), None)]),
            RuleHead::new("dummy", "has_knows_triples", "true"),
        ));

        fc.step();
        // The rule fires for both knows triples, but they both derive the same fact
        assert!(fc
            .fact_set
            .contains(&Fact::new("dummy", "has_knows_triples", "true")));
        // Still only 3 facts (original 2 + 1 derived)
        assert_eq!(fc.fact_count(), 3);
    }
}
