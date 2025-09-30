//! Forward Chaining Inference Engine
//!
//! Implementation of forward chaining rule application with fixpoint calculation.
//! Applies rules from facts to derive new facts until no more facts can be derived.

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, trace, warn};

/// Variable substitution mapping
pub type Substitution = HashMap<String, Term>;

/// Forward chaining inference engine
#[derive(Debug)]
pub struct ForwardChainer {
    /// Rules to apply
    rules: Vec<Rule>,
    /// Known facts
    facts: HashSet<RuleAtom>,
    /// Maximum number of iterations to prevent infinite loops
    max_iterations: usize,
    /// Enable detailed logging
    debug_mode: bool,
}

impl Default for ForwardChainer {
    fn default() -> Self {
        Self::new()
    }
}

impl ForwardChainer {
    /// Create a new forward chainer
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            facts: HashSet::new(),
            max_iterations: 1000,
            debug_mode: false,
        }
    }

    /// Create a new forward chainer with custom configuration
    pub fn with_config(max_iterations: usize, debug_mode: bool) -> Self {
        Self {
            rules: Vec::new(),
            facts: HashSet::new(),
            max_iterations,
            debug_mode,
        }
    }

    /// Add a rule to the engine
    pub fn add_rule(&mut self, rule: Rule) {
        if self.debug_mode {
            debug!("Adding rule: {}", rule.name);
        }
        self.rules.push(rule);
    }

    /// Add multiple rules to the engine
    pub fn add_rules(&mut self, rules: Vec<Rule>) {
        for rule in rules {
            self.add_rule(rule);
        }
    }

    /// Add a fact to the knowledge base
    pub fn add_fact(&mut self, fact: RuleAtom) {
        if self.debug_mode {
            trace!("Adding fact: {:?}", fact);
        }
        self.facts.insert(fact);
    }

    /// Add multiple facts to the knowledge base
    pub fn add_facts(&mut self, facts: Vec<RuleAtom>) {
        for fact in facts {
            self.add_fact(fact);
        }
    }

    /// Get all current facts
    pub fn get_facts(&self) -> Vec<RuleAtom> {
        self.facts.iter().cloned().collect()
    }

    /// Clear all facts
    pub fn clear_facts(&mut self) {
        self.facts.clear();
    }

    /// Perform forward chaining inference
    pub fn infer(&mut self) -> Result<Vec<RuleAtom>> {
        let initial_fact_count = self.facts.len();
        let mut iteration = 0;
        let mut new_facts_added = true;

        info!(
            "Starting forward chaining with {} initial facts and {} rules",
            initial_fact_count,
            self.rules.len()
        );

        while new_facts_added && iteration < self.max_iterations {
            new_facts_added = false;
            iteration += 1;

            if self.debug_mode {
                debug!(
                    "Forward chaining iteration {} with {} facts",
                    iteration,
                    self.facts.len()
                );
            }

            // Apply all rules to current facts
            for rule in &self.rules {
                let new_facts = self.apply_rule(rule)?;
                for fact in new_facts {
                    if !self.facts.contains(&fact) {
                        if self.debug_mode {
                            trace!("Derived new fact from rule '{}': {:?}", rule.name, fact);
                        }
                        self.facts.insert(fact);
                        new_facts_added = true;
                    }
                }
            }
        }

        if iteration >= self.max_iterations {
            warn!(
                "Forward chaining reached maximum iterations ({}), may not have reached fixpoint",
                self.max_iterations
            );
        }

        let final_fact_count = self.facts.len();
        info!(
            "Forward chaining completed after {} iterations: {} -> {} facts",
            iteration, initial_fact_count, final_fact_count
        );

        Ok(self.get_facts())
    }

    /// Apply a single rule to current facts
    fn apply_rule(&self, rule: &Rule) -> Result<Vec<RuleAtom>> {
        let mut new_facts = Vec::new();

        // Find all possible substitutions that satisfy the rule body
        let substitutions = self.find_substitutions(&rule.body)?;

        // Apply each substitution to the rule head
        for substitution in substitutions {
            for head_atom in &rule.head {
                let instantiated = self.apply_substitution(head_atom, &substitution)?;
                new_facts.push(instantiated);
            }
        }

        if self.debug_mode && !new_facts.is_empty() {
            debug!(
                "Rule '{}' produced {} new facts",
                rule.name,
                new_facts.len()
            );
        }

        Ok(new_facts)
    }

    /// Find all substitutions that satisfy the rule body
    fn find_substitutions(&self, body: &[RuleAtom]) -> Result<Vec<Substitution>> {
        if body.is_empty() {
            return Ok(vec![HashMap::new()]);
        }

        // Start with the first atom in the body
        let mut substitutions = self.match_atom(&body[0], &HashMap::new())?;

        // Extend substitutions with remaining atoms
        for atom in &body[1..] {
            let mut new_substitutions = Vec::new();
            for substitution in substitutions {
                let extended = self.match_atom(atom, &substitution)?;
                new_substitutions.extend(extended);
            }
            substitutions = new_substitutions;
        }

        Ok(substitutions)
    }

    /// Match an atom against facts with a given partial substitution
    fn match_atom(&self, atom: &RuleAtom, partial_sub: &Substitution) -> Result<Vec<Substitution>> {
        let mut substitutions = Vec::new();

        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                // Match against all facts
                for fact in &self.facts {
                    if let RuleAtom::Triple {
                        subject: fact_subject,
                        predicate: fact_predicate,
                        object: fact_object,
                    } = fact
                    {
                        if let Some(substitution) = self.unify_triple(
                            (subject, predicate, object),
                            (fact_subject, fact_predicate, fact_object),
                            partial_sub.clone(),
                        )? {
                            substitutions.push(substitution);
                        }
                    }
                }
            }
            RuleAtom::Builtin { name, args } => {
                // Handle built-in predicates
                if let Some(substitution) =
                    self.evaluate_builtin(name, args, partial_sub.clone())?
                {
                    substitutions.push(substitution);
                }
            }
            RuleAtom::NotEqual { left, right } => {
                // Handle not-equal constraint
                let left_term = self.substitute_term(left, partial_sub);
                let right_term = self.substitute_term(right, partial_sub);
                if !self.terms_equal(&left_term, &right_term) {
                    substitutions.push(partial_sub.clone());
                }
            }
            RuleAtom::GreaterThan { left, right } => {
                // Handle greater-than constraint
                let left_term = self.substitute_term(left, partial_sub);
                let right_term = self.substitute_term(right, partial_sub);
                if self.compare_terms(&left_term, &right_term) > 0 {
                    substitutions.push(partial_sub.clone());
                }
            }
            RuleAtom::LessThan { left, right } => {
                // Handle less-than constraint
                let left_term = self.substitute_term(left, partial_sub);
                let right_term = self.substitute_term(right, partial_sub);
                if self.compare_terms(&left_term, &right_term) < 0 {
                    substitutions.push(partial_sub.clone());
                }
            }
        }

        Ok(substitutions)
    }

    /// Unify two triples and extend the substitution
    fn unify_triple(
        &self,
        pattern: (&Term, &Term, &Term),
        fact: (&Term, &Term, &Term),
        mut substitution: Substitution,
    ) -> Result<Option<Substitution>> {
        // Unify subject
        if !self.unify_terms(pattern.0, fact.0, &mut substitution)? {
            return Ok(None);
        }

        // Unify predicate
        if !self.unify_terms(pattern.1, fact.1, &mut substitution)? {
            return Ok(None);
        }

        // Unify object
        if !self.unify_terms(pattern.2, fact.2, &mut substitution)? {
            return Ok(None);
        }

        Ok(Some(substitution))
    }

    /// Unify two terms and update the substitution
    fn unify_terms(
        &self,
        pattern_term: &Term,
        fact_term: &Term,
        substitution: &mut Substitution,
    ) -> Result<bool> {
        match (pattern_term, fact_term) {
            // Variable in pattern
            (Term::Variable(var), fact_term) => {
                if let Some(existing) = substitution.get(var) {
                    // Check if consistent with existing binding
                    Ok(self.terms_equal(existing, fact_term))
                } else {
                    // Add new binding
                    substitution.insert(var.clone(), fact_term.clone());
                    Ok(true)
                }
            }
            // Variable in fact (shouldn't happen in forward chaining, but handle anyway)
            (fact_term, Term::Variable(var)) => {
                if let Some(existing) = substitution.get(var) {
                    Ok(self.terms_equal(existing, fact_term))
                } else {
                    substitution.insert(var.clone(), fact_term.clone());
                    Ok(true)
                }
            }
            // Both constants - must match exactly
            (Term::Constant(c1), Term::Constant(c2)) => Ok(c1 == c2),
            (Term::Literal(l1), Term::Literal(l2)) => Ok(l1 == l2),
            (Term::Constant(c), Term::Literal(l)) | (Term::Literal(l), Term::Constant(c)) => {
                Ok(c == l) // Allow constants and literals to unify if equal
            }
            // Function terms unify if name and args match
            (Term::Function { name: n1, args: a1 }, Term::Function { name: n2, args: a2 }) => {
                if n1 != n2 || a1.len() != a2.len() {
                    Ok(false)
                } else {
                    // Recursively unify all arguments
                    for (arg1, arg2) in a1.iter().zip(a2.iter()) {
                        if !self.unify_terms(arg1, arg2, substitution)? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
            }
            // Other combinations don't unify
            _ => Ok(false),
        }
    }

    /// Check if two terms are equal
    fn terms_equal(&self, term1: &Term, term2: &Term) -> bool {
        match (term1, term2) {
            (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            (Term::Constant(c), Term::Literal(l)) | (Term::Literal(l), Term::Constant(c)) => c == l,
            (Term::Function { name: n1, args: a1 }, Term::Function { name: n2, args: a2 }) => {
                n1 == n2 && a1 == a2
            }
            _ => false,
        }
    }

    /// Compare two terms for ordering (-1: left < right, 0: equal, 1: left > right)
    fn compare_terms(&self, term1: &Term, term2: &Term) -> i32 {
        match (term1, term2) {
            (Term::Constant(c1), Term::Constant(c2)) => {
                // Try to parse as numbers first
                if let (Ok(n1), Ok(n2)) = (c1.parse::<f64>(), c2.parse::<f64>()) {
                    if n1 < n2 {
                        -1
                    } else if n1 > n2 {
                        1
                    } else {
                        0
                    }
                } else {
                    // Fallback to string comparison
                    if c1 < c2 {
                        -1
                    } else if c1 > c2 {
                        1
                    } else {
                        0
                    }
                }
            }
            (Term::Literal(l1), Term::Literal(l2)) => {
                // Try to parse as numbers first
                if let (Ok(n1), Ok(n2)) = (l1.parse::<f64>(), l2.parse::<f64>()) {
                    if n1 < n2 {
                        -1
                    } else if n1 > n2 {
                        1
                    } else {
                        0
                    }
                } else {
                    // Fallback to string comparison
                    if l1 < l2 {
                        -1
                    } else if l1 > l2 {
                        1
                    } else {
                        0
                    }
                }
            }
            (Term::Constant(c), Term::Literal(l)) | (Term::Literal(l), Term::Constant(c)) => {
                // Try to parse as numbers first
                if let (Ok(n1), Ok(n2)) = (c.parse::<f64>(), l.parse::<f64>()) {
                    if n1 < n2 {
                        -1
                    } else if n1 > n2 {
                        1
                    } else {
                        0
                    }
                } else {
                    // Fallback to string comparison
                    if c < l {
                        -1
                    } else if c > l {
                        1
                    } else {
                        0
                    }
                }
            }
            (Term::Function { name: n1, args: a1 }, Term::Function { name: n2, args: a2 }) => {
                // Compare function names first
                if n1 < n2 {
                    -1
                } else if n1 > n2 {
                    1
                } else {
                    // If names are equal, compare arg counts
                    if a1.len() < a2.len() {
                        -1
                    } else if a1.len() > a2.len() {
                        1
                    } else {
                        0
                    } // Equal functions
                }
            }
            // Variables and mixed types can't be compared meaningfully
            _ => 0,
        }
    }

    /// Apply substitution to an atom
    fn apply_substitution(&self, atom: &RuleAtom, substitution: &Substitution) -> Result<RuleAtom> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => Ok(RuleAtom::Triple {
                subject: self.substitute_term(subject, substitution),
                predicate: self.substitute_term(predicate, substitution),
                object: self.substitute_term(object, substitution),
            }),
            RuleAtom::Builtin { name, args } => {
                let substituted_args = args
                    .iter()
                    .map(|arg| self.substitute_term(arg, substitution))
                    .collect();
                Ok(RuleAtom::Builtin {
                    name: name.clone(),
                    args: substituted_args,
                })
            }
            RuleAtom::NotEqual { left, right } => Ok(RuleAtom::NotEqual {
                left: self.substitute_term(left, substitution),
                right: self.substitute_term(right, substitution),
            }),
            RuleAtom::GreaterThan { left, right } => Ok(RuleAtom::GreaterThan {
                left: self.substitute_term(left, substitution),
                right: self.substitute_term(right, substitution),
            }),
            RuleAtom::LessThan { left, right } => Ok(RuleAtom::LessThan {
                left: self.substitute_term(left, substitution),
                right: self.substitute_term(right, substitution),
            }),
        }
    }

    /// Substitute variables in a term
    #[allow(clippy::only_used_in_recursion)]
    fn substitute_term(&self, term: &Term, substitution: &Substitution) -> Term {
        match term {
            Term::Variable(var) => substitution
                .get(var)
                .cloned()
                .unwrap_or_else(|| term.clone()),
            Term::Function { name, args } => {
                let substituted_args = args
                    .iter()
                    .map(|arg| self.substitute_term(arg, substitution))
                    .collect();
                Term::Function {
                    name: name.clone(),
                    args: substituted_args,
                }
            }
            _ => term.clone(),
        }
    }

    /// Evaluate built-in predicates
    fn evaluate_builtin(
        &self,
        name: &str,
        args: &[Term],
        substitution: Substitution,
    ) -> Result<Option<Substitution>> {
        match name {
            "equal" => {
                if args.len() != 2 {
                    return Err(anyhow::anyhow!("equal/2 requires exactly 2 arguments"));
                }
                let arg1 = self.substitute_term(&args[0], &substitution);
                let arg2 = self.substitute_term(&args[1], &substitution);
                if self.terms_equal(&arg1, &arg2) {
                    Ok(Some(substitution))
                } else {
                    Ok(None)
                }
            }
            "notEqual" => {
                if args.len() != 2 {
                    return Err(anyhow::anyhow!("notEqual/2 requires exactly 2 arguments"));
                }
                let arg1 = self.substitute_term(&args[0], &substitution);
                let arg2 = self.substitute_term(&args[1], &substitution);
                if !self.terms_equal(&arg1, &arg2) {
                    Ok(Some(substitution))
                } else {
                    Ok(None)
                }
            }
            "bound" => {
                if args.len() != 1 {
                    return Err(anyhow::anyhow!("bound/1 requires exactly 1 argument"));
                }
                match &args[0] {
                    Term::Variable(var) => {
                        if substitution.contains_key(var) {
                            Ok(Some(substitution))
                        } else {
                            Ok(None)
                        }
                    }
                    _ => Ok(Some(substitution)), // Non-variables are always "bound"
                }
            }
            "unbound" => {
                if args.len() != 1 {
                    return Err(anyhow::anyhow!("unbound/1 requires exactly 1 argument"));
                }
                match &args[0] {
                    Term::Variable(var) => {
                        if !substitution.contains_key(var) {
                            Ok(Some(substitution))
                        } else {
                            Ok(None)
                        }
                    }
                    _ => Ok(None), // Non-variables are always "bound"
                }
            }
            _ => {
                warn!("Unknown built-in predicate: {}", name);
                Ok(None)
            }
        }
    }

    /// Get statistics about the inference process
    pub fn get_stats(&self) -> ForwardChainingStats {
        ForwardChainingStats {
            total_facts: self.facts.len(),
            total_rules: self.rules.len(),
        }
    }

    /// Check if a specific fact is derivable
    pub fn can_derive(&mut self, target: &RuleAtom) -> Result<bool> {
        let initial_facts = self.facts.clone();
        self.infer()?;
        let result = self.facts.contains(target);
        self.facts = initial_facts; // Restore original state
        Ok(result)
    }

    /// Derive all facts and return only the newly derived ones
    pub fn derive_new_facts(&mut self) -> Result<Vec<RuleAtom>> {
        let initial_facts = self.facts.clone();
        self.infer()?;
        let new_facts = self.facts.difference(&initial_facts).cloned().collect();
        Ok(new_facts)
    }
}

/// Statistics about forward chaining inference
#[derive(Debug, Clone)]
pub struct ForwardChainingStats {
    pub total_facts: usize,
    pub total_rules: usize,
}

impl std::fmt::Display for ForwardChainingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Facts: {}, Rules: {}",
            self.total_facts, self.total_rules
        )
    }
}

/// Forward chaining result
#[derive(Debug, Clone)]
pub struct ForwardChainingResult {
    pub facts: Vec<RuleAtom>,
    pub iterations: usize,
    pub new_facts_derived: usize,
}

impl PartialEq for RuleAtom {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
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
            ) => s1 == s2 && p1 == p2 && o1 == o2,
            (
                RuleAtom::Builtin { name: n1, args: a1 },
                RuleAtom::Builtin { name: n2, args: a2 },
            ) => n1 == n2 && a1 == a2,
            (
                RuleAtom::NotEqual {
                    left: l1,
                    right: r1,
                },
                RuleAtom::NotEqual {
                    left: l2,
                    right: r2,
                },
            ) => l1 == l2 && r1 == r2,
            (
                RuleAtom::GreaterThan {
                    left: l1,
                    right: r1,
                },
                RuleAtom::GreaterThan {
                    left: l2,
                    right: r2,
                },
            ) => l1 == l2 && r1 == r2,
            (
                RuleAtom::LessThan {
                    left: l1,
                    right: r1,
                },
                RuleAtom::LessThan {
                    left: l2,
                    right: r2,
                },
            ) => l1 == l2 && r1 == r2,
            _ => false,
        }
    }
}

impl Eq for RuleAtom {}

impl std::hash::Hash for RuleAtom {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                0.hash(state);
                subject.hash(state);
                predicate.hash(state);
                object.hash(state);
            }
            RuleAtom::Builtin { name, args } => {
                1.hash(state);
                name.hash(state);
                args.hash(state);
            }
            RuleAtom::NotEqual { left, right } => {
                2.hash(state);
                left.hash(state);
                right.hash(state);
            }
            RuleAtom::GreaterThan { left, right } => {
                3.hash(state);
                left.hash(state);
                right.hash(state);
            }
            RuleAtom::LessThan { left, right } => {
                4.hash(state);
                left.hash(state);
                right.hash(state);
            }
        }
    }
}

impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            (Term::Function { name: n1, args: a1 }, Term::Function { name: n2, args: a2 }) => {
                n1 == n2 && a1 == a2
            }
            _ => false,
        }
    }
}

impl Eq for Term {}

impl std::hash::Hash for Term {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Term::Variable(v) => {
                0.hash(state);
                v.hash(state);
            }
            Term::Constant(c) => {
                1.hash(state);
                c.hash(state);
            }
            Term::Literal(l) => {
                2.hash(state);
                l.hash(state);
            }
            Term::Function { name, args } => {
                3.hash(state);
                name.hash(state);
                args.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_forward_chaining() {
        let mut chainer = ForwardChainer::new();

        // Add rule: mortal(X) :- human(X)
        chainer.add_rule(Rule {
            name: "mortality_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("human".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("mortal".to_string()),
            }],
        });

        // Add fact: human(socrates)
        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        });

        // Run inference
        let facts = chainer.infer().unwrap();

        // Should derive: mortal(socrates)
        let expected = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("mortal".to_string()),
        };

        assert!(facts.contains(&expected));
    }

    #[test]
    fn test_transitive_chaining() {
        let mut chainer = ForwardChainer::new();

        // Add rule: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)
        chainer.add_rule(Rule {
            name: "transitive_ancestor".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("ancestor".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        });

        // Add rule: ancestor(X,Y) :- parent(X,Y)
        chainer.add_rule(Rule {
            name: "direct_ancestor".to_string(),
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
        });

        // Add facts
        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("mary".to_string()),
        });
        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("mary".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("bob".to_string()),
        });

        // Run inference
        let facts = chainer.infer().unwrap();

        // Should derive ancestor relationships
        assert!(facts.contains(&RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Constant("mary".to_string()),
        }));
        assert!(facts.contains(&RuleAtom::Triple {
            subject: Term::Constant("mary".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Constant("bob".to_string()),
        }));
        assert!(facts.contains(&RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Constant("bob".to_string()),
        }));
    }

    #[test]
    fn test_builtin_predicates() {
        let mut chainer = ForwardChainer::new();

        // Add rule with built-in: same(X,X) :- bound(X)
        chainer.add_rule(Rule {
            name: "reflexive_same".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("exists".to_string()),
                    object: Term::Constant("true".to_string()),
                },
                RuleAtom::Builtin {
                    name: "bound".to_string(),
                    args: vec![Term::Variable("X".to_string())],
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("same".to_string()),
                object: Term::Variable("X".to_string()),
            }],
        });

        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("exists".to_string()),
            object: Term::Constant("true".to_string()),
        });

        let facts = chainer.infer().unwrap();

        assert!(facts.contains(&RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("same".to_string()),
            object: Term::Constant("a".to_string()),
        }));
    }
}
