//! N3 reasoning primitives
//!
//! This module provides basic reasoning capabilities for N3 formulas:
//! - Pattern matching against formulas
//! - Variable substitution and unification
//! - Simple forward chaining with implications
//!
//! # Examples
//!
//! ## Pattern Matching
//!
//! ```rust
//! use oxirs_ttl::n3::reasoning::{FormulaPattern, Matcher};
//! use oxirs_ttl::n3::{N3Formula, N3Statement, N3Term, N3Variable};
//! use oxirs_core::model::NamedNode;
//!
//! // Create a pattern: { ?x :knows ?y }
//! let pattern = FormulaPattern::new_with_statement(
//!     N3Term::Variable(N3Variable::universal("x")),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/knows")?),
//!     N3Term::Variable(N3Variable::universal("y"))
//! );
//!
//! // Create a formula to match against
//! let mut formula = N3Formula::new();
//! formula.add_statement(N3Statement::new(
//!     N3Term::NamedNode(NamedNode::new("http://example.org/alice")?),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/knows")?),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/bob")?)
//! ));
//!
//! // Match pattern against formula
//! let matcher = Matcher::new();
//! let bindings = matcher.match_pattern(&pattern, &formula);
//! assert!(bindings.is_some());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Variable Substitution
//!
//! ```rust
//! use oxirs_ttl::n3::reasoning::{Substitution, VariableBindings};
//! use oxirs_ttl::n3::{N3Term, N3Variable};
//! use oxirs_core::model::NamedNode;
//!
//! let mut bindings = VariableBindings::new();
//! bindings.bind(
//!     "x".to_string(),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/alice")?)
//! );
//!
//! // Apply substitution to a term containing variable ?x
//! let var_term = N3Term::Variable(N3Variable::universal("x"));
//! let substituted = bindings.substitute(&var_term);
//! assert!(!substituted.is_variable());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Simple Forward Chaining
//!
//! ```rust
//! use oxirs_ttl::n3::reasoning::{ReasoningEngine, KnowledgeBase};
//! use oxirs_ttl::n3::{N3Formula, N3Implication, N3Statement, N3Term, N3Variable};
//! use oxirs_core::model::NamedNode;
//!
//! let mut kb = KnowledgeBase::new();
//!
//! // Add a fact: alice knows bob
//! kb.add_fact(N3Statement::new(
//!     N3Term::NamedNode(NamedNode::new("http://example.org/alice")?),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/knows")?),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/bob")?)
//! ));
//!
//! // Add a rule: { ?x :knows ?y } => { ?y :knows ?x } (symmetric relation)
//! let mut antecedent = N3Formula::new();
//! antecedent.add_statement(N3Statement::new(
//!     N3Term::Variable(N3Variable::universal("x")),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/knows")?),
//!     N3Term::Variable(N3Variable::universal("y"))
//! ));
//!
//! let mut consequent = N3Formula::new();
//! consequent.add_statement(N3Statement::new(
//!     N3Term::Variable(N3Variable::universal("y")),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/knows")?),
//!     N3Term::Variable(N3Variable::universal("x"))
//! ));
//!
//! kb.add_rule(N3Implication::new(antecedent, consequent));
//!
//! // Apply forward chaining
//! let engine = ReasoningEngine::new();
//! let new_facts = engine.forward_chain(&kb, 1); // One iteration
//!
//! // Should derive: bob knows alice
//! assert!(!new_facts.is_empty());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#[allow(unused_imports)] // Used in doc tests and unit tests
use crate::formats::n3_types::{N3Formula, N3Implication, N3Statement, N3Term, N3Variable};
use std::collections::HashMap;

/// Variable bindings for substitution
#[derive(Debug, Clone, Default)]
pub struct VariableBindings {
    /// Map from variable name to bound term
    bindings: HashMap<String, N3Term>,
}

impl VariableBindings {
    /// Create a new empty set of bindings
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Bind a variable to a term
    pub fn bind(&mut self, var_name: String, term: N3Term) {
        self.bindings.insert(var_name, term);
    }

    /// Get the binding for a variable
    pub fn get(&self, var_name: &str) -> Option<&N3Term> {
        self.bindings.get(var_name)
    }

    /// Check if a variable is bound
    pub fn is_bound(&self, var_name: &str) -> bool {
        self.bindings.contains_key(var_name)
    }

    /// Get all bindings
    pub fn all_bindings(&self) -> &HashMap<String, N3Term> {
        &self.bindings
    }

    /// Check if two bindings are compatible (no conflicting bindings)
    pub fn is_compatible(&self, other: &VariableBindings) -> bool {
        for (var, term) in &self.bindings {
            if let Some(other_term) = other.get(var) {
                if term != other_term {
                    return false;
                }
            }
        }
        true
    }

    /// Merge two compatible bindings
    pub fn merge(&mut self, other: &VariableBindings) {
        for (var, term) in &other.bindings {
            self.bindings.insert(var.clone(), term.clone());
        }
    }
}

/// Trait for variable substitution
pub trait Substitution {
    /// Substitute variables in a term using bindings
    fn substitute(&self, term: &N3Term) -> N3Term;

    /// Substitute variables in a statement
    fn substitute_statement(&self, stmt: &N3Statement) -> N3Statement {
        N3Statement::new(
            self.substitute(&stmt.subject),
            self.substitute(&stmt.predicate),
            self.substitute(&stmt.object),
        )
    }

    /// Substitute variables in a formula
    fn substitute_formula(&self, formula: &N3Formula) -> N3Formula {
        let mut new_formula = N3Formula::new();
        for stmt in &formula.triples {
            new_formula.add_statement(self.substitute_statement(stmt));
        }
        new_formula
    }
}

impl Substitution for VariableBindings {
    fn substitute(&self, term: &N3Term) -> N3Term {
        match term {
            N3Term::Variable(var) => {
                if let Some(bound_term) = self.get(&var.name) {
                    bound_term.clone()
                } else {
                    term.clone()
                }
            }
            N3Term::Formula(formula) => N3Term::Formula(Box::new(self.substitute_formula(formula))),
            _ => term.clone(),
        }
    }
}

/// Formula pattern for matching
#[derive(Debug, Clone)]
pub struct FormulaPattern {
    /// Statements in the pattern (may contain variables)
    pub statements: Vec<N3Statement>,
}

impl FormulaPattern {
    /// Create a new empty pattern
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
        }
    }

    /// Create a pattern with a single statement
    pub fn new_with_statement(subject: N3Term, predicate: N3Term, object: N3Term) -> Self {
        let mut pattern = Self::new();
        pattern
            .statements
            .push(N3Statement::new(subject, predicate, object));
        pattern
    }

    /// Add a statement to the pattern
    pub fn add_statement(&mut self, stmt: N3Statement) {
        self.statements.push(stmt);
    }

    /// Convert from a formula
    pub fn from_formula(formula: &N3Formula) -> Self {
        Self {
            statements: formula.triples.clone(),
        }
    }
}

impl Default for FormulaPattern {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern matcher for formulas
#[derive(Debug, Clone, Default)]
pub struct Matcher;

impl Matcher {
    /// Create a new matcher
    pub fn new() -> Self {
        Self
    }

    /// Try to match a pattern against a formula
    pub fn match_pattern(
        &self,
        pattern: &FormulaPattern,
        formula: &N3Formula,
    ) -> Option<VariableBindings> {
        let mut bindings = VariableBindings::new();

        // Try to match each pattern statement against formula statements
        for pattern_stmt in &pattern.statements {
            let mut found_match = false;

            for formula_stmt in &formula.triples {
                if let Some(stmt_bindings) = self.match_statement(pattern_stmt, formula_stmt) {
                    // Check compatibility with existing bindings
                    if bindings.is_compatible(&stmt_bindings) {
                        bindings.merge(&stmt_bindings);
                        found_match = true;
                        break;
                    }
                }
            }

            // If any pattern statement doesn't match, the whole pattern fails
            if !found_match {
                return None;
            }
        }

        Some(bindings)
    }

    /// Try to match a pattern statement against a concrete statement
    fn match_statement(
        &self,
        pattern: &N3Statement,
        concrete: &N3Statement,
    ) -> Option<VariableBindings> {
        let mut bindings = VariableBindings::new();

        // Match subject
        if !self.match_term(&pattern.subject, &concrete.subject, &mut bindings) {
            return None;
        }

        // Match predicate
        if !self.match_term(&pattern.predicate, &concrete.predicate, &mut bindings) {
            return None;
        }

        // Match object
        if !self.match_term(&pattern.object, &concrete.object, &mut bindings) {
            return None;
        }

        Some(bindings)
    }

    /// Try to match a pattern term against a concrete term
    fn match_term(
        &self,
        pattern: &N3Term,
        concrete: &N3Term,
        bindings: &mut VariableBindings,
    ) -> bool {
        match pattern {
            N3Term::Variable(var) => {
                if let Some(bound_term) = bindings.get(&var.name) {
                    // Variable already bound, check consistency
                    bound_term == concrete
                } else {
                    // Bind variable to concrete term
                    bindings.bind(var.name.clone(), concrete.clone());
                    true
                }
            }
            _ => {
                // Non-variable terms must match exactly
                pattern == concrete
            }
        }
    }
}

/// Knowledge base for storing facts and rules
#[derive(Debug, Clone, Default)]
pub struct KnowledgeBase {
    /// Facts (ground statements)
    facts: Vec<N3Statement>,
    /// Rules (implications)
    rules: Vec<N3Implication>,
}

impl KnowledgeBase {
    /// Create a new empty knowledge base
    pub fn new() -> Self {
        Self {
            facts: Vec::new(),
            rules: Vec::new(),
        }
    }

    /// Add a fact to the knowledge base
    pub fn add_fact(&mut self, fact: N3Statement) {
        if !fact.has_variables() {
            self.facts.push(fact);
        }
    }

    /// Add a rule to the knowledge base
    pub fn add_rule(&mut self, rule: N3Implication) {
        self.rules.push(rule);
    }

    /// Get all facts
    pub fn facts(&self) -> &[N3Statement] {
        &self.facts
    }

    /// Get all rules
    pub fn rules(&self) -> &[N3Implication] {
        &self.rules
    }

    /// Convert facts to a formula
    pub fn facts_as_formula(&self) -> N3Formula {
        N3Formula::with_statements(self.facts.clone())
    }
}

/// Simple reasoning engine
#[derive(Debug, Clone, Default)]
pub struct ReasoningEngine {
    matcher: Matcher,
}

impl ReasoningEngine {
    /// Create a new reasoning engine
    pub fn new() -> Self {
        Self {
            matcher: Matcher::new(),
        }
    }

    /// Perform forward chaining for a given number of iterations
    pub fn forward_chain(&self, kb: &KnowledgeBase, max_iterations: usize) -> Vec<N3Statement> {
        let mut new_facts = Vec::new();
        let mut current_kb = kb.clone();

        for _ in 0..max_iterations {
            let mut derived_facts = Vec::new();

            // Try to apply each rule
            for rule in kb.rules() {
                // Convert antecedent to pattern
                let pattern = FormulaPattern::from_formula(&rule.antecedent);

                // Try to match against current facts
                let formula = current_kb.facts_as_formula();
                if let Some(bindings) = self.matcher.match_pattern(&pattern, &formula) {
                    // Apply bindings to consequent
                    let instantiated = bindings.substitute_formula(&rule.consequent);

                    // Add new facts
                    for stmt in instantiated.triples {
                        if !stmt.has_variables() && !current_kb.facts.contains(&stmt) {
                            derived_facts.push(stmt.clone());
                            new_facts.push(stmt);
                        }
                    }
                }
            }

            // If no new facts were derived, we've reached a fixed point
            if derived_facts.is_empty() {
                break;
            }

            // Add derived facts to knowledge base
            for fact in derived_facts {
                current_kb.add_fact(fact);
            }
        }

        new_facts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_variable_bindings() {
        let mut bindings = VariableBindings::new();
        let term = N3Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap());

        bindings.bind("x".to_string(), term.clone());
        assert!(bindings.is_bound("x"));
        assert_eq!(bindings.get("x"), Some(&term));
    }

    #[test]
    fn test_substitution() {
        let mut bindings = VariableBindings::new();
        bindings.bind(
            "x".to_string(),
            N3Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap()),
        );

        let var_term = N3Term::Variable(N3Variable::universal("x"));
        let substituted = bindings.substitute(&var_term);
        assert!(!substituted.is_variable());
    }

    #[test]
    fn test_pattern_matching() {
        let pattern = FormulaPattern::new_with_statement(
            N3Term::Variable(N3Variable::universal("x")),
            N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap()),
            N3Term::Variable(N3Variable::universal("y")),
        );

        let mut formula = N3Formula::new();
        formula.add_statement(N3Statement::new(
            N3Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap()),
            N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap()),
            N3Term::NamedNode(NamedNode::new("http://example.org/bob").unwrap()),
        ));

        let matcher = Matcher::new();
        let bindings = matcher.match_pattern(&pattern, &formula);
        assert!(bindings.is_some());
    }

    #[test]
    fn test_forward_chaining() {
        let mut kb = KnowledgeBase::new();

        // Add fact: alice knows bob
        kb.add_fact(N3Statement::new(
            N3Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap()),
            N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap()),
            N3Term::NamedNode(NamedNode::new("http://example.org/bob").unwrap()),
        ));

        // Add rule: { ?x :knows ?y } => { ?y :knows ?x }
        let mut antecedent = N3Formula::new();
        antecedent.add_statement(N3Statement::new(
            N3Term::Variable(N3Variable::universal("x")),
            N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap()),
            N3Term::Variable(N3Variable::universal("y")),
        ));

        let mut consequent = N3Formula::new();
        consequent.add_statement(N3Statement::new(
            N3Term::Variable(N3Variable::universal("y")),
            N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap()),
            N3Term::Variable(N3Variable::universal("x")),
        ));

        kb.add_rule(N3Implication::new(antecedent, consequent));

        // Apply forward chaining
        let engine = ReasoningEngine::new();
        let new_facts = engine.forward_chain(&kb, 1);

        // Should derive: bob knows alice
        assert!(!new_facts.is_empty());
    }
}
