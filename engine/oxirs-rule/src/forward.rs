//! Forward Chaining Inference Engine
//!
//! Implementation of forward chaining rule application with fixpoint calculation.
//! Applies rules from facts to derive new facts until no more facts can be derived.

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use once_cell::sync::Lazy;
use scirs2_core::metrics::{Counter, Gauge};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, trace, warn};

// Global metrics for memory tracking
static SUBSTITUTION_CLONES: Lazy<Counter> =
    Lazy::new(|| Counter::new("forward_chain_substitution_clones".to_string()));
static FACT_SET_CLONES: Lazy<Counter> =
    Lazy::new(|| Counter::new("forward_chain_fact_set_clones".to_string()));
static ACTIVE_SUBSTITUTIONS: Lazy<Gauge> =
    Lazy::new(|| Gauge::new("forward_chain_active_substitutions".to_string()));

/// Variable substitution mapping
pub type Substitution = HashMap<String, Term>;

/// Predicate-keyed index over ground triple facts.
///
/// Groups facts by their predicate term so a body atom with a bound predicate
/// only scans facts sharing that predicate instead of the entire fact set.
/// Non-triple atoms are never facts and are not indexed.
#[derive(Debug, Default)]
struct PredicateIndex {
    by_predicate: HashMap<Term, Vec<RuleAtom>>,
}

impl PredicateIndex {
    /// Build an index from an iterator of facts.
    fn from_facts<'a, I>(facts: I) -> Self
    where
        I: IntoIterator<Item = &'a RuleAtom>,
    {
        let mut index = PredicateIndex::default();
        for fact in facts {
            index.insert(fact);
        }
        index
    }

    /// Insert a fact into the index (no-op for non-triple atoms).
    fn insert(&mut self, fact: &RuleAtom) {
        if let RuleAtom::Triple { predicate, .. } = fact {
            self.by_predicate
                .entry(predicate.clone())
                .or_default()
                .push(fact.clone());
        }
    }
}

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

    /// Perform forward chaining inference.
    ///
    /// Uses **semi-naive** evaluation: facts are indexed by predicate so each
    /// body atom only scans candidate facts sharing its predicate (rather than
    /// the whole fact set), and each iteration joins only against the *delta*
    /// of facts derived in the previous round instead of re-deriving everything
    /// over the entire set. This computes the same least fixpoint as the naive
    /// algorithm but avoids the O(rules * facts * iterations) blow-up.
    pub fn infer(&mut self) -> Result<Vec<RuleAtom>> {
        let initial_fact_count = self.facts.len();

        info!(
            "Starting forward chaining with {} initial facts and {} rules",
            initial_fact_count,
            self.rules.len()
        );

        // Build a predicate index over the current facts.
        let mut index = PredicateIndex::from_facts(self.facts.iter());

        // Semi-naive delta: facts newly derived in the previous round. Seed it
        // with every current fact so the first round considers the full set
        // (equivalent to the naive first pass).
        let mut delta: Vec<RuleAtom> = self.facts.iter().cloned().collect();

        let mut iteration = 0;
        while !delta.is_empty() && iteration < self.max_iterations {
            iteration += 1;

            if self.debug_mode {
                debug!(
                    "Forward chaining iteration {} with {} facts ({} in delta)",
                    iteration,
                    self.facts.len(),
                    delta.len()
                );
            }

            let delta_index = PredicateIndex::from_facts(delta.iter());
            let mut next_delta: Vec<RuleAtom> = Vec::new();

            for rule in &self.rules {
                let derived = self.apply_rule_semi_naive(rule, &index, &delta_index, iteration)?;
                for fact in derived {
                    if !self.facts.contains(&fact) {
                        if self.debug_mode {
                            trace!("Derived new fact from rule '{}': {:?}", rule.name, fact);
                        }
                        index.insert(&fact);
                        self.facts.insert(fact.clone());
                        next_delta.push(fact);
                    }
                }
            }

            delta = next_delta;
        }

        if iteration >= self.max_iterations && !delta.is_empty() {
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

    /// Apply a single rule under semi-naive evaluation.
    ///
    /// For a rule whose body contains fact-consuming (triple) atoms, this fires
    /// the rule once per triple-atom position, drawing that position from the
    /// `delta` index and all other positions from the `full` index. The union
    /// over positions yields exactly the derivations that use at least one
    /// newly-derived fact — derivations that use only older facts were already
    /// produced in an earlier round. Rules with no triple atoms in their body
    /// (pure builtin/constraint bodies) are evaluated once, in the first round.
    fn apply_rule_semi_naive(
        &self,
        rule: &Rule,
        full: &PredicateIndex,
        delta: &PredicateIndex,
        iteration: usize,
    ) -> Result<Vec<RuleAtom>> {
        let triple_positions: Vec<usize> = rule
            .body
            .iter()
            .enumerate()
            .filter(|(_, atom)| matches!(atom, RuleAtom::Triple { .. }))
            .map(|(i, _)| i)
            .collect();

        let mut new_facts = Vec::new();

        if triple_positions.is_empty() {
            // No fact-consuming atoms: the body depends only on builtins /
            // constraints over ground terms, so it can only fire once. Evaluate
            // it in the first round against the full fact set.
            if iteration == 1 {
                let substitutions = self.find_substitutions(&rule.body)?;
                for substitution in substitutions {
                    for head_atom in &rule.head {
                        new_facts.push(self.apply_substitution(head_atom, &substitution)?);
                    }
                }
            }
            return Ok(new_facts);
        }

        for &delta_pos in &triple_positions {
            let substitutions =
                self.find_substitutions_semi_naive(&rule.body, full, delta, delta_pos)?;
            for substitution in substitutions {
                for head_atom in &rule.head {
                    new_facts.push(self.apply_substitution(head_atom, &substitution)?);
                }
            }
        }

        if self.debug_mode && !new_facts.is_empty() {
            debug!(
                "Rule '{}' produced {} candidate facts",
                rule.name,
                new_facts.len()
            );
        }

        Ok(new_facts)
    }

    /// Find substitutions satisfying `body`, drawing the atom at `delta_pos`
    /// from the `delta` index and all other atoms from the `full` index.
    fn find_substitutions_semi_naive(
        &self,
        body: &[RuleAtom],
        full: &PredicateIndex,
        delta: &PredicateIndex,
        delta_pos: usize,
    ) -> Result<Vec<Substitution>> {
        if body.is_empty() {
            return Ok(vec![HashMap::new()]);
        }

        let source_for = |i: usize| if i == delta_pos { delta } else { full };

        let mut substitutions =
            self.match_atom_indexed(&body[0], &HashMap::new(), source_for(0))?;
        ACTIVE_SUBSTITUTIONS.set(substitutions.len() as f64);

        for (i, atom) in body.iter().enumerate().skip(1) {
            let source = source_for(i);
            let mut new_substitutions = Vec::new();
            for substitution in substitutions {
                let extended = self.match_atom_indexed(atom, &substitution, source)?;
                new_substitutions.extend(extended);
            }
            substitutions = new_substitutions;
            ACTIVE_SUBSTITUTIONS.set(substitutions.len() as f64);
        }

        Ok(substitutions)
    }

    /// Find all substitutions that satisfy the rule body
    fn find_substitutions(&self, body: &[RuleAtom]) -> Result<Vec<Substitution>> {
        if body.is_empty() {
            return Ok(vec![HashMap::new()]);
        }

        // Start with the first atom in the body
        let mut substitutions = self.match_atom(&body[0], &HashMap::new())?;

        // Track active substitutions
        ACTIVE_SUBSTITUTIONS.set(substitutions.len() as f64);

        // Extend substitutions with remaining atoms
        for atom in &body[1..] {
            let mut new_substitutions = Vec::new();
            for substitution in substitutions {
                let extended = self.match_atom(atom, &substitution)?;
                new_substitutions.extend(extended);
            }
            substitutions = new_substitutions;

            // Update gauge with current count
            ACTIVE_SUBSTITUTIONS.set(substitutions.len() as f64);
        }

        Ok(substitutions)
    }

    /// Match an atom against all facts with a given partial substitution.
    ///
    /// Retained for bodies evaluated outside the predicate-indexed path (e.g.
    /// pure builtin/constraint bodies via [`find_substitutions`]).
    fn match_atom(&self, atom: &RuleAtom, partial_sub: &Substitution) -> Result<Vec<Substitution>> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let mut substitutions = Vec::new();
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
                            partial_sub,
                        )? {
                            SUBSTITUTION_CLONES.inc();
                            substitutions.push(substitution);
                        }
                    }
                }
                Ok(substitutions)
            }
            _ => self.match_filter_atom(atom, partial_sub),
        }
    }

    /// Match a triple atom against a predicate-indexed fact set, or evaluate a
    /// builtin/constraint filter atom. Only candidate facts sharing the atom's
    /// (substituted) predicate are scanned; an unbound predicate falls back to
    /// scanning every bucket.
    fn match_atom_indexed(
        &self,
        atom: &RuleAtom,
        partial_sub: &Substitution,
        index: &PredicateIndex,
    ) -> Result<Vec<Substitution>> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let mut substitutions = Vec::new();
                let pred_term = self.substitute_term(predicate, partial_sub);
                let pattern = (subject, predicate, object);
                if matches!(pred_term, Term::Variable(_)) {
                    for bucket in index.by_predicate.values() {
                        self.unify_bucket(pattern, bucket, partial_sub, &mut substitutions)?;
                    }
                } else if let Some(bucket) = index.by_predicate.get(&pred_term) {
                    self.unify_bucket(pattern, bucket, partial_sub, &mut substitutions)?;
                }
                Ok(substitutions)
            }
            _ => self.match_filter_atom(atom, partial_sub),
        }
    }

    /// Unify a triple pattern against every fact in a candidate bucket,
    /// pushing successful extended substitutions.
    fn unify_bucket(
        &self,
        pattern: (&Term, &Term, &Term),
        bucket: &[RuleAtom],
        partial_sub: &Substitution,
        substitutions: &mut Vec<Substitution>,
    ) -> Result<()> {
        for fact in bucket {
            if let RuleAtom::Triple {
                subject: fact_subject,
                predicate: fact_predicate,
                object: fact_object,
            } = fact
            {
                if let Some(substitution) = self.unify_triple(
                    pattern,
                    (fact_subject, fact_predicate, fact_object),
                    partial_sub,
                )? {
                    SUBSTITUTION_CLONES.inc();
                    substitutions.push(substitution);
                }
            }
        }
        Ok(())
    }

    /// Evaluate a non-triple (builtin / constraint) body atom against a partial
    /// substitution. These atoms filter or bind variables but never consult the
    /// fact store.
    fn match_filter_atom(
        &self,
        atom: &RuleAtom,
        partial_sub: &Substitution,
    ) -> Result<Vec<Substitution>> {
        let mut substitutions = Vec::new();

        match atom {
            RuleAtom::Triple { .. } => {
                // Triples are handled by the fact-matching paths, not here.
            }
            RuleAtom::Builtin { name, args } => {
                if let Some(substitution) = self.evaluate_builtin(name, args, partial_sub)? {
                    SUBSTITUTION_CLONES.inc();
                    substitutions.push(substitution);
                }
            }
            RuleAtom::NotEqual { left, right } => {
                let left_term = self.substitute_term(left, partial_sub);
                let right_term = self.substitute_term(right, partial_sub);
                if !self.terms_equal(&left_term, &right_term) {
                    SUBSTITUTION_CLONES.inc();
                    substitutions.push(partial_sub.clone());
                }
            }
            RuleAtom::GreaterThan { left, right } => {
                let left_term = self.substitute_term(left, partial_sub);
                let right_term = self.substitute_term(right, partial_sub);
                if self.compare_terms(&left_term, &right_term) > 0 {
                    SUBSTITUTION_CLONES.inc();
                    substitutions.push(partial_sub.clone());
                }
            }
            RuleAtom::LessThan { left, right } => {
                let left_term = self.substitute_term(left, partial_sub);
                let right_term = self.substitute_term(right, partial_sub);
                if self.compare_terms(&left_term, &right_term) < 0 {
                    SUBSTITUTION_CLONES.inc();
                    substitutions.push(partial_sub.clone());
                }
            }
        }

        Ok(substitutions)
    }

    /// Unify two triples and extend the substitution
    /// OPTIMIZED: Takes reference to avoid unnecessary clones
    fn unify_triple(
        &self,
        pattern: (&Term, &Term, &Term),
        fact: (&Term, &Term, &Term),
        substitution: &Substitution,
    ) -> Result<Option<Substitution>> {
        // Clone only once at the start - will be cheaper than cloning for every match attempt
        let mut new_substitution = substitution.clone();

        // Unify subject
        if !self.unify_terms(pattern.0, fact.0, &mut new_substitution)? {
            return Ok(None);
        }

        // Unify predicate
        if !self.unify_terms(pattern.1, fact.1, &mut new_substitution)? {
            return Ok(None);
        }

        // Unify object
        if !self.unify_terms(pattern.2, fact.2, &mut new_substitution)? {
            return Ok(None);
        }

        Ok(Some(new_substitution))
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
    /// OPTIMIZED: Takes reference to avoid unnecessary clones
    fn evaluate_builtin(
        &self,
        name: &str,
        args: &[Term],
        substitution: &Substitution,
    ) -> Result<Option<Substitution>> {
        match name {
            "equal" => {
                if args.len() != 2 {
                    return Err(anyhow::anyhow!("equal/2 requires exactly 2 arguments"));
                }
                let arg1 = self.substitute_term(&args[0], substitution);
                let arg2 = self.substitute_term(&args[1], substitution);
                if self.terms_equal(&arg1, &arg2) {
                    Ok(Some(substitution.clone()))
                } else {
                    Ok(None)
                }
            }
            "notEqual" => {
                if args.len() != 2 {
                    return Err(anyhow::anyhow!("notEqual/2 requires exactly 2 arguments"));
                }
                let arg1 = self.substitute_term(&args[0], substitution);
                let arg2 = self.substitute_term(&args[1], substitution);
                if !self.terms_equal(&arg1, &arg2) {
                    Ok(Some(substitution.clone()))
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
                            Ok(Some(substitution.clone()))
                        } else {
                            Ok(None)
                        }
                    }
                    _ => Ok(Some(substitution.clone())), // Non-variables are always "bound"
                }
            }
            "unbound" => {
                if args.len() != 1 {
                    return Err(anyhow::anyhow!("unbound/1 requires exactly 1 argument"));
                }
                match &args[0] {
                    Term::Variable(var) => {
                        if !substitution.contains_key(var) {
                            Ok(Some(substitution.clone()))
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
    /// OPTIMIZED: Use count-based restoration instead of full clone
    pub fn can_derive(&mut self, target: &RuleAtom) -> Result<bool> {
        // Store initial count instead of cloning entire set
        let initial_count = self.facts.len();

        // Quick check: if already present, no need to infer
        if self.facts.contains(target) {
            return Ok(true);
        }

        // Collect initial facts into a vector for efficient restoration
        let initial_facts: Vec<RuleAtom> = self.facts.iter().cloned().collect();

        self.infer()?;
        let result = self.facts.contains(target);

        // Restore by removing new facts (cheaper than full clone for small deltas)
        if self.facts.len() > initial_count {
            FACT_SET_CLONES.inc();
            self.facts.clear();
            self.facts.extend(initial_facts);
        }

        Ok(result)
    }

    /// Derive all facts and return only the newly derived ones
    /// OPTIMIZED: Avoid full clone by using set difference efficiently
    pub fn derive_new_facts(&mut self) -> Result<Vec<RuleAtom>> {
        // Store initial facts as a vector for efficient difference computation
        let initial_facts: Vec<RuleAtom> = self.facts.iter().cloned().collect();
        let initial_set: HashSet<RuleAtom> = initial_facts.iter().cloned().collect();

        FACT_SET_CLONES.inc();

        self.infer()?;

        // Only collect the difference
        let new_facts: Vec<RuleAtom> = self.facts.difference(&initial_set).cloned().collect();

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
    fn test_basic_forward_chaining() -> Result<(), Box<dyn std::error::Error>> {
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
        let facts = chainer.infer()?;

        // Should derive: mortal(socrates)
        let expected = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("mortal".to_string()),
        };

        assert!(facts.contains(&expected));
        Ok(())
    }

    #[test]
    fn test_transitive_chaining() -> Result<(), Box<dyn std::error::Error>> {
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
        let facts = chainer.infer()?;

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
        Ok(())
    }

    /// Semi-naive evaluation must still reach the full transitive closure over
    /// a multi-hop chain that requires several iterations to saturate.
    #[test]
    fn test_semi_naive_deep_transitive_closure() -> Result<(), Box<dyn std::error::Error>> {
        let mut chainer = ForwardChainer::new();

        // path(X,Z) :- edge(X,Y), path(Y,Z)
        chainer.add_rule(Rule {
            name: "transitive_path".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("edge".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("path".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        });
        // path(X,Y) :- edge(X,Y)
        chainer.add_rule(Rule {
            name: "base_path".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("path".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        });

        // Chain n0 -> n1 -> n2 -> n3 -> n4
        let nodes = ["n0", "n1", "n2", "n3", "n4"];
        for pair in nodes.windows(2) {
            chainer.add_fact(RuleAtom::Triple {
                subject: Term::Constant(pair[0].to_string()),
                predicate: Term::Constant("edge".to_string()),
                object: Term::Constant(pair[1].to_string()),
            });
        }

        let facts = chainer.infer()?;

        // Every ordered pair (ni, nj) with i < j must be reachable.
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let expected = RuleAtom::Triple {
                    subject: Term::Constant(nodes[i].to_string()),
                    predicate: Term::Constant("path".to_string()),
                    object: Term::Constant(nodes[j].to_string()),
                };
                assert!(
                    facts.contains(&expected),
                    "missing transitive path {} -> {}",
                    nodes[i],
                    nodes[j]
                );
            }
        }
        // 5 nodes -> 10 ordered reachable pairs.
        let path_count = facts
            .iter()
            .filter(|f| matches!(f, RuleAtom::Triple { predicate: Term::Constant(p), .. } if p == "path"))
            .count();
        assert_eq!(path_count, 10);
        Ok(())
    }

    /// A body atom with an unbound (variable) predicate must still match facts
    /// across all predicate buckets under the indexed evaluation path.
    #[test]
    fn test_semi_naive_variable_predicate() -> Result<(), Box<dyn std::error::Error>> {
        let mut chainer = ForwardChainer::new();

        // related(X,Y) :- link(X, P, Y)  -- here the predicate position is a variable
        chainer.add_rule(Rule {
            name: "variable_predicate".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Variable("P".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("related".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        });

        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("knows".to_string()),
            object: Term::Constant("b".to_string()),
        });
        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("c".to_string()),
            predicate: Term::Constant("likes".to_string()),
            object: Term::Constant("d".to_string()),
        });

        let facts = chainer.infer()?;

        assert!(facts.contains(&RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("related".to_string()),
            object: Term::Constant("b".to_string()),
        }));
        assert!(facts.contains(&RuleAtom::Triple {
            subject: Term::Constant("c".to_string()),
            predicate: Term::Constant("related".to_string()),
            object: Term::Constant("d".to_string()),
        }));
        Ok(())
    }

    #[test]
    fn test_builtin_predicates() -> Result<(), Box<dyn std::error::Error>> {
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

        let facts = chainer.infer()?;

        assert!(facts.contains(&RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("same".to_string()),
            object: Term::Constant("a".to_string()),
        }));
        Ok(())
    }
}
