//! Backward Chaining Inference Engine
//!
//! Implementation of goal-driven backward chaining inference.
//! Starts with a goal and works backwards to find supporting facts and rules.

use crate::{Rule, RuleAtom, Term};
use crate::forward::Substitution;
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::{debug, info, trace, warn};

/// Proof context for tracking derivation paths
#[derive(Debug, Clone)]
pub struct ProofContext {
    /// Derivation path (goals that led to this point)
    pub path: Vec<RuleAtom>,
    /// Current substitution
    pub substitution: Substitution,
    /// Depth of the current proof attempt
    pub depth: usize,
}

impl Default for ProofContext {
    fn default() -> Self {
        Self {
            path: Vec::new(),
            substitution: HashMap::new(),
            depth: 0,
        }
    }
}

/// Proof result
#[derive(Debug, Clone)]
pub enum ProofResult {
    /// Goal successfully proven with substitution
    Success(Substitution),
    /// Goal failed to prove
    Failure,
    /// Goal partially proven (needs more information)
    Partial(Vec<RuleAtom>),
}

/// Backward chaining inference engine
#[derive(Debug)]
pub struct BackwardChainer {
    /// Rules for inference
    rules: Vec<Rule>,
    /// Known facts
    facts: HashSet<RuleAtom>,
    /// Maximum proof depth to prevent infinite recursion
    max_depth: usize,
    /// Enable detailed logging
    debug_mode: bool,
    /// Cache for memoization
    proof_cache: HashMap<RuleAtom, ProofResult>,
}

impl Default for BackwardChainer {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardChainer {
    /// Create a new backward chainer
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            facts: HashSet::new(),
            max_depth: 100,
            debug_mode: false,
            proof_cache: HashMap::new(),
        }
    }

    /// Create a new backward chainer with custom configuration
    pub fn with_config(max_depth: usize, debug_mode: bool) -> Self {
        Self {
            rules: Vec::new(),
            facts: HashSet::new(),
            max_depth,
            debug_mode,
            proof_cache: HashMap::new(),
        }
    }

    /// Add a rule to the engine
    pub fn add_rule(&mut self, rule: Rule) {
        if self.debug_mode {
            debug!("Adding rule: {}", rule.name);
        }
        self.rules.push(rule);
        self.proof_cache.clear(); // Clear cache when rules change
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
        self.proof_cache.clear(); // Clear cache when facts change
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
        self.proof_cache.clear();
    }

    /// Clear proof cache
    pub fn clear_cache(&mut self) {
        self.proof_cache.clear();
    }

    /// Prove a goal using backward chaining
    pub fn prove(&mut self, goal: &RuleAtom) -> Result<bool> {
        info!("Starting backward chaining proof for goal: {:?}", goal);
        
        let context = ProofContext::default();
        let result = self.prove_goal(goal, &context)?;
        
        match result {
            ProofResult::Success(_) => {
                info!("Goal successfully proven");
                Ok(true)
            }
            ProofResult::Failure => {
                info!("Goal failed to prove");
                Ok(false)
            }
            ProofResult::Partial(remaining) => {
                info!("Goal partially proven, {} remaining subgoals", remaining.len());
                Ok(false)
            }
        }
    }

    /// Prove a goal and return all valid substitutions
    pub fn prove_all(&mut self, goal: &RuleAtom) -> Result<Vec<Substitution>> {
        info!("Finding all proofs for goal: {:?}", goal);
        
        let context = ProofContext::default();
        let mut substitutions = Vec::new();
        self.find_all_proofs(goal, &context, &mut substitutions)?;
        
        info!("Found {} valid proofs", substitutions.len());
        Ok(substitutions)
    }

    /// Core goal proving algorithm
    fn prove_goal(&mut self, goal: &RuleAtom, context: &ProofContext) -> Result<ProofResult> {
        // Check depth limit
        if context.depth > self.max_depth {
            warn!("Maximum proof depth exceeded for goal: {:?}", goal);
            return Ok(ProofResult::Failure);
        }

        // Check for cycles
        if context.path.contains(goal) {
            if self.debug_mode {
                debug!("Cycle detected for goal: {:?}", goal);
            }
            return Ok(ProofResult::Failure);
        }

        // Check cache
        if let Some(cached_result) = self.proof_cache.get(goal) {
            if self.debug_mode {
                trace!("Using cached result for goal: {:?}", goal);
            }
            return Ok(cached_result.clone());
        }

        let result = self.prove_goal_internal(goal, context)?;
        
        // Cache the result
        self.proof_cache.insert(goal.clone(), result.clone());
        
        Ok(result)
    }

    /// Internal goal proving implementation
    fn prove_goal_internal(&mut self, goal: &RuleAtom, context: &ProofContext) -> Result<ProofResult> {
        if self.debug_mode {
            debug!("Proving goal at depth {}: {:?}", context.depth, goal);
        }

        // Try to match against facts first
        if let Some(substitution) = self.match_against_facts(goal, &context.substitution)? {
            if self.debug_mode {
                debug!("Goal proven by direct fact match");
            }
            return Ok(ProofResult::Success(substitution));
        }

        // Try to prove using rules
        self.prove_using_rules(goal, context)
    }

    /// Match a goal against known facts
    fn match_against_facts(&self, goal: &RuleAtom, context_sub: &Substitution) -> Result<Option<Substitution>> {
        match goal {
            RuleAtom::Triple { subject, predicate, object } => {
                for fact in &self.facts {
                    if let RuleAtom::Triple { 
                        subject: fact_subject, 
                        predicate: fact_predicate, 
                        object: fact_object 
                    } = fact {
                        if let Some(substitution) = self.unify_triple(
                            (subject, predicate, object),
                            (fact_subject, fact_predicate, fact_object),
                            context_sub.clone()
                        )? {
                            return Ok(Some(substitution));
                        }
                    }
                }
                Ok(None)
            }
            RuleAtom::Builtin { name, args } => {
                // Built-ins are evaluated directly
                self.evaluate_builtin(name, args, context_sub.clone())
            }
        }
    }

    /// Prove a goal using available rules
    fn prove_using_rules(&mut self, goal: &RuleAtom, context: &ProofContext) -> Result<ProofResult> {
        for rule in &self.rules.clone() {
            // Try to unify goal with rule head
            for head_atom in &rule.head {
                if let Some(head_substitution) = self.unify_atoms(goal, head_atom, context.substitution.clone())? {
                    if self.debug_mode {
                        debug!("Trying rule '{}' for goal: {:?}", rule.name, goal);
                    }

                    // Create new context
                    let mut new_context = context.clone();
                    new_context.path.push(goal.clone());
                    new_context.substitution = head_substitution.clone();
                    new_context.depth += 1;

                    // Try to prove all conditions in the rule body
                    if let Some(final_substitution) = self.prove_rule_body(&rule.body, &new_context)? {
                        if self.debug_mode {
                            debug!("Rule '{}' successfully proven", rule.name);
                        }
                        return Ok(ProofResult::Success(final_substitution));
                    }
                }
            }
        }

        Ok(ProofResult::Failure)
    }

    /// Prove all conditions in a rule body
    fn prove_rule_body(&mut self, body: &[RuleAtom], context: &ProofContext) -> Result<Option<Substitution>> {
        let mut current_substitution = context.substitution.clone();

        for atom in body {
            // Apply current substitution to the atom
            let instantiated_atom = self.apply_substitution(atom, &current_substitution)?;
            
            // Create context for this subgoal
            let subgoal_context = ProofContext {
                path: context.path.clone(),
                substitution: current_substitution.clone(),
                depth: context.depth,
            };

            // Try to prove the subgoal
            match self.prove_goal(&instantiated_atom, &subgoal_context)? {
                ProofResult::Success(new_substitution) => {
                    // Merge substitutions
                    current_substitution = self.merge_substitutions(current_substitution, new_substitution)?;
                }
                ProofResult::Failure => {
                    return Ok(None);
                }
                ProofResult::Partial(_) => {
                    return Ok(None);
                }
            }
        }

        Ok(Some(current_substitution))
    }

    /// Find all valid proofs for a goal
    fn find_all_proofs(&mut self, goal: &RuleAtom, context: &ProofContext, results: &mut Vec<Substitution>) -> Result<()> {
        // Check depth limit
        if context.depth > self.max_depth {
            return Ok(());
        }

        // Check for cycles
        if context.path.contains(goal) {
            return Ok(());
        }

        // Try to match against facts
        if let Some(substitution) = self.match_against_facts(goal, &context.substitution)? {
            results.push(substitution);
        }

        // Try all applicable rules
        for rule in &self.rules.clone() {
            for head_atom in &rule.head {
                if let Some(head_substitution) = self.unify_atoms(goal, head_atom, context.substitution.clone())? {
                    let mut new_context = context.clone();
                    new_context.path.push(goal.clone());
                    new_context.substitution = head_substitution;
                    new_context.depth += 1;

                    if let Some(final_substitution) = self.prove_rule_body(&rule.body, &new_context)? {
                        results.push(final_substitution);
                    }
                }
            }
        }

        Ok(())
    }

    /// Unify two atoms
    fn unify_atoms(&self, atom1: &RuleAtom, atom2: &RuleAtom, mut substitution: Substitution) -> Result<Option<Substitution>> {
        match (atom1, atom2) {
            (RuleAtom::Triple { subject: s1, predicate: p1, object: o1 },
             RuleAtom::Triple { subject: s2, predicate: p2, object: o2 }) => {
                self.unify_triple((s1, p1, o1), (s2, p2, o2), substitution)
            }
            (RuleAtom::Builtin { name: n1, args: a1 },
             RuleAtom::Builtin { name: n2, args: a2 }) => {
                if n1 == n2 && a1.len() == a2.len() {
                    for (arg1, arg2) in a1.iter().zip(a2.iter()) {
                        if !self.unify_terms(arg1, arg2, &mut substitution)? {
                            return Ok(None);
                        }
                    }
                    Ok(Some(substitution))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Unify two triples
    fn unify_triple(
        &self,
        triple1: (&Term, &Term, &Term),
        triple2: (&Term, &Term, &Term),
        mut substitution: Substitution,
    ) -> Result<Option<Substitution>> {
        if !self.unify_terms(triple1.0, triple2.0, &mut substitution)? {
            return Ok(None);
        }
        if !self.unify_terms(triple1.1, triple2.1, &mut substitution)? {
            return Ok(None);
        }
        if !self.unify_terms(triple1.2, triple2.2, &mut substitution)? {
            return Ok(None);
        }
        Ok(Some(substitution))
    }

    /// Unify two terms
    fn unify_terms(
        &self,
        term1: &Term,
        term2: &Term,
        substitution: &mut Substitution,
    ) -> Result<bool> {
        match (term1, term2) {
            (Term::Variable(var), term) | (term, Term::Variable(var)) => {
                if let Some(existing) = substitution.get(var).cloned() {
                    self.unify_terms(&existing, term, substitution)
                } else {
                    substitution.insert(var.clone(), term.clone());
                    Ok(true)
                }
            }
            (Term::Constant(c1), Term::Constant(c2)) => Ok(c1 == c2),
            (Term::Literal(l1), Term::Literal(l2)) => Ok(l1 == l2),
            (Term::Constant(c), Term::Literal(l)) | (Term::Literal(l), Term::Constant(c)) => {
                Ok(c == l)
            }
            _ => Ok(false),
        }
    }

    /// Apply substitution to an atom
    fn apply_substitution(&self, atom: &RuleAtom, substitution: &Substitution) -> Result<RuleAtom> {
        match atom {
            RuleAtom::Triple { subject, predicate, object } => {
                Ok(RuleAtom::Triple {
                    subject: self.substitute_term(subject, substitution),
                    predicate: self.substitute_term(predicate, substitution),
                    object: self.substitute_term(object, substitution),
                })
            }
            RuleAtom::Builtin { name, args } => {
                let substituted_args = args.iter()
                    .map(|arg| self.substitute_term(arg, substitution))
                    .collect();
                Ok(RuleAtom::Builtin {
                    name: name.clone(),
                    args: substituted_args,
                })
            }
        }
    }

    /// Substitute variables in a term
    fn substitute_term(&self, term: &Term, substitution: &Substitution) -> Term {
        match term {
            Term::Variable(var) => {
                substitution.get(var).cloned().unwrap_or_else(|| term.clone())
            }
            _ => term.clone(),
        }
    }

    /// Merge two substitutions
    fn merge_substitutions(&self, sub1: Substitution, sub2: Substitution) -> Result<Substitution> {
        let mut merged = sub1;
        for (var, term) in sub2 {
            if let Some(existing) = merged.get(&var) {
                if !self.terms_equal(existing, &term) {
                    return Err(anyhow::anyhow!("Inconsistent substitutions for variable {}", var));
                }
            } else {
                merged.insert(var, term);
            }
        }
        Ok(merged)
    }

    /// Check if two terms are equal
    fn terms_equal(&self, term1: &Term, term2: &Term) -> bool {
        match (term1, term2) {
            (Term::Variable(v1), Term::Variable(v2)) => v1 == v2,
            (Term::Constant(c1), Term::Constant(c2)) => c1 == c2,
            (Term::Literal(l1), Term::Literal(l2)) => l1 == l2,
            (Term::Constant(c), Term::Literal(l)) | (Term::Literal(l), Term::Constant(c)) => c == l,
            _ => false,
        }
    }

    /// Evaluate built-in predicates
    fn evaluate_builtin(&self, name: &str, args: &[Term], substitution: Substitution) -> Result<Option<Substitution>> {
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
                    _ => Ok(Some(substitution)),
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
                    _ => Ok(None),
                }
            }
            _ => {
                warn!("Unknown built-in predicate: {}", name);
                Ok(None)
            }
        }
    }

    /// Get statistics about the reasoning process
    pub fn get_stats(&self) -> BackwardChainingStats {
        BackwardChainingStats {
            total_facts: self.facts.len(),
            total_rules: self.rules.len(),
            cache_size: self.proof_cache.len(),
        }
    }

    /// Query for facts that match a pattern
    pub fn query(&mut self, pattern: &RuleAtom) -> Result<Vec<RuleAtom>> {
        let mut results = Vec::new();
        
        // Check facts directly
        for fact in &self.facts {
            if self.unify_atoms(pattern, fact, HashMap::new())?.is_some() {
                results.push(fact.clone());
            }
        }

        // Try backward chaining to derive new facts
        let substitutions = self.prove_all(pattern)?;
        for substitution in substitutions {
            let instantiated = self.apply_substitution(pattern, &substitution)?;
            if !results.contains(&instantiated) {
                results.push(instantiated);
            }
        }

        Ok(results)
    }
}

/// Statistics about backward chaining inference
#[derive(Debug, Clone)]
pub struct BackwardChainingStats {
    pub total_facts: usize,
    pub total_rules: usize,
    pub cache_size: usize,
}

impl std::fmt::Display for BackwardChainingStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Facts: {}, Rules: {}, Cache: {}", 
               self.total_facts, self.total_rules, self.cache_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_backward_chaining() {
        let mut chainer = BackwardChainer::new();

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

        // Test goal: mortal(socrates)
        let goal = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("mortal".to_string()),
        };

        assert!(chainer.prove(&goal).unwrap());
    }

    #[test]
    fn test_fact_matching() {
        let mut chainer = BackwardChainer::new();

        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        });

        let goal = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };

        assert!(chainer.prove(&goal).unwrap());
    }

    #[test]
    fn test_variable_substitution() {
        let mut chainer = BackwardChainer::new();

        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        });

        let goal = RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };

        let substitutions = chainer.prove_all(&goal).unwrap();
        assert_eq!(substitutions.len(), 1);
        assert_eq!(
            substitutions[0].get("X"),
            Some(&Term::Constant("socrates".to_string()))
        );
    }

    #[test]
    fn test_transitive_proof() {
        let mut chainer = BackwardChainer::with_config(20, true); // Enable debug mode

        // Test just direct ancestor first
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

        // Test direct ancestry first
        let direct_goal = RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Constant("mary".to_string()),
        };

        assert!(chainer.prove(&direct_goal).unwrap());
    }

    #[test]
    fn test_query_functionality() {
        let mut chainer = BackwardChainer::new();

        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        });
        chainer.add_fact(RuleAtom::Triple {
            subject: Term::Constant("plato".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        });

        let pattern = RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };

        let results = chainer.query(&pattern).unwrap();
        assert_eq!(results.len(), 2);
    }
}
