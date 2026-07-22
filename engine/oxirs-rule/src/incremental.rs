//! Incremental Reasoning Engine
//!
//! Provides efficient incremental reasoning by computing only the delta (new facts)
//! when the knowledge base is updated, rather than recomputing everything from scratch.
//!
//! # Features
//!
//! - **Delta Computation**: Track changes and compute only affected inferences
//! - **Dependency Tracking**: Maintain dependencies between facts and rules
//! - **Efficient Updates**: Add/remove facts with minimal recomputation
//! - **Undo Support**: Rollback to previous states
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::incremental::IncrementalReasoner;
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let mut reasoner = IncrementalReasoner::new();
//!
//! // Add rule
//! reasoner.add_rule(Rule {
//!     name: "mortal".to_string(),
//!     body: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("type".to_string()),
//!         object: Term::Constant("human".to_string()),
//!     }],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("type".to_string()),
//!         object: Term::Constant("mortal".to_string()),
//!     }],
//! });
//!
//! // Add fact incrementally
//! let delta = reasoner.add_fact(RuleAtom::Triple {
//!     subject: Term::Constant("socrates".to_string()),
//!     predicate: Term::Constant("type".to_string()),
//!     object: Term::Constant("human".to_string()),
//! }).expect("should succeed");
//!
//! // Only newly derived facts are returned
//! assert!(!delta.is_empty());
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use once_cell::sync::Lazy;
use scirs2_core::metrics::Counter;
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::{debug, info, trace};

// Global metrics for memory tracking
static INCREMENTAL_SUBSTITUTION_CLONES: Lazy<Counter> =
    Lazy::new(|| Counter::new("incremental_substitution_clones".to_string()));
static INCREMENTAL_RULE_CLONES: Lazy<Counter> =
    Lazy::new(|| Counter::new("incremental_rule_clones".to_string()));

/// Fact identifier for dependency tracking
type FactId = usize;

/// Rule identifier for dependency tracking
type RuleId = usize;

/// Dependency information for a derived fact
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FactDependency {
    /// Facts that this fact depends on
    depends_on: HashSet<FactId>,
    /// Rule that derived this fact
    derived_by: Option<RuleId>,
    /// Generation when this fact was added
    generation: usize,
}

/// Incremental reasoning engine
#[derive(Debug)]
pub struct IncrementalReasoner {
    /// All known facts (indexed by ID)
    facts: HashMap<FactId, RuleAtom>,
    /// Reverse index: fact content -> fact ID
    fact_index: HashMap<RuleAtom, FactId>,
    /// Next fact ID
    next_fact_id: FactId,

    /// All rules (indexed by ID)
    rules: HashMap<RuleId, Rule>,
    /// Next rule ID
    next_rule_id: RuleId,

    /// Dependencies for each fact
    dependencies: HashMap<FactId, FactDependency>,
    /// Reverse dependencies: which facts depend on this fact
    reverse_dependencies: HashMap<FactId, HashSet<FactId>>,

    /// Facts that are explicitly asserted (not derived)
    asserted_facts: HashSet<FactId>,
    /// Facts that were derived (not asserted)
    derived_facts: HashSet<FactId>,

    /// Current generation counter
    generation: usize,
    /// Snapshot history for undo support
    snapshots: Vec<Snapshot>,

    /// Maximum derivation depth to prevent infinite loops
    max_depth: usize,
}

/// Snapshot of reasoner state for undo support
#[derive(Debug, Clone)]
struct Snapshot {
    generation: usize,
    asserted_facts: HashSet<FactId>,
    derived_facts: HashSet<FactId>,
}

impl Default for IncrementalReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalReasoner {
    /// Create a new incremental reasoner
    pub fn new() -> Self {
        Self {
            facts: HashMap::new(),
            fact_index: HashMap::new(),
            next_fact_id: 0,
            rules: HashMap::new(),
            next_rule_id: 0,
            dependencies: HashMap::new(),
            reverse_dependencies: HashMap::new(),
            asserted_facts: HashSet::new(),
            derived_facts: HashSet::new(),
            generation: 0,
            snapshots: Vec::new(),
            max_depth: 100,
        }
    }

    /// Create a new incremental reasoner with custom configuration
    pub fn with_config(max_depth: usize) -> Self {
        Self {
            max_depth,
            ..Self::new()
        }
    }

    /// Add a rule to the reasoner
    pub fn add_rule(&mut self, rule: Rule) -> RuleId {
        let rule_id = self.next_rule_id;
        self.next_rule_id += 1;
        self.rules.insert(rule_id, rule);
        debug!("Added rule with ID {}", rule_id);
        rule_id
    }

    /// Add a fact incrementally and return newly derived facts (delta)
    pub fn add_fact(&mut self, fact: RuleAtom) -> Result<Vec<RuleAtom>> {
        self.generation += 1;

        // Check if fact already exists
        if let Some(&fact_id) = self.fact_index.get(&fact) {
            trace!("Fact already exists with ID {}", fact_id);
            return Ok(vec![]);
        }

        // Add the fact
        let fact_id = self.add_fact_internal(fact.clone(), None, HashSet::new())?;
        self.asserted_facts.insert(fact_id);

        // Compute delta (newly derived facts)
        let delta = self.compute_delta_from_fact(fact_id)?;

        info!("Added fact {} derived {} new facts", fact_id, delta.len());

        Ok(delta)
    }

    /// Remove a fact and all facts that depend on it
    pub fn remove_fact(&mut self, fact: &RuleAtom) -> Result<Vec<RuleAtom>> {
        self.generation += 1;

        // Find fact ID
        let fact_id = match self.fact_index.get(fact) {
            Some(&id) => id,
            None => {
                trace!("Fact not found for removal");
                return Ok(vec![]);
            }
        };

        // Can only remove asserted facts
        if !self.asserted_facts.contains(&fact_id) {
            return Err(anyhow::anyhow!(
                "Cannot remove derived fact {}; remove its dependencies instead",
                fact_id
            ));
        }

        // Find all facts that depend on this fact (transitively)
        let affected_facts = self.find_affected_facts(fact_id);

        // Remove all affected facts
        let mut removed = Vec::new();
        for affected_id in affected_facts {
            if let Some(fact) = self.facts.remove(&affected_id) {
                self.fact_index.remove(&fact);
                self.dependencies.remove(&affected_id);
                self.reverse_dependencies.remove(&affected_id);
                self.asserted_facts.remove(&affected_id);
                self.derived_facts.remove(&affected_id);
                removed.push(fact);
            }
        }

        info!("Removed {} facts", removed.len());
        Ok(removed)
    }

    /// Create a snapshot for undo support
    pub fn create_snapshot(&mut self) {
        let snapshot = Snapshot {
            generation: self.generation,
            asserted_facts: self.asserted_facts.clone(),
            derived_facts: self.derived_facts.clone(),
        };
        self.snapshots.push(snapshot);
        debug!("Created snapshot at generation {}", self.generation);
    }

    /// Restore to the last snapshot
    pub fn restore_snapshot(&mut self) -> Result<()> {
        let snapshot = self
            .snapshots
            .pop()
            .ok_or_else(|| anyhow::anyhow!("No snapshot available to restore"))?;

        // Remove facts that were added after the snapshot
        let facts_to_remove: Vec<FactId> = self
            .facts
            .keys()
            .filter(|&id| {
                !snapshot.asserted_facts.contains(id) && !snapshot.derived_facts.contains(id)
            })
            .copied()
            .collect();

        for fact_id in facts_to_remove {
            if let Some(fact) = self.facts.remove(&fact_id) {
                self.fact_index.remove(&fact);
                self.dependencies.remove(&fact_id);
                self.reverse_dependencies.remove(&fact_id);
            }
        }

        self.generation = snapshot.generation;
        self.asserted_facts = snapshot.asserted_facts;
        self.derived_facts = snapshot.derived_facts;

        info!("Restored snapshot to generation {}", self.generation);
        Ok(())
    }

    /// Get all current facts
    pub fn get_facts(&self) -> Vec<RuleAtom> {
        self.facts.values().cloned().collect()
    }

    /// Get statistics about the reasoner
    pub fn get_stats(&self) -> IncrementalStats {
        IncrementalStats {
            total_facts: self.facts.len(),
            asserted_facts: self.asserted_facts.len(),
            derived_facts: self.derived_facts.len(),
            total_rules: self.rules.len(),
            generation: self.generation,
            snapshots: self.snapshots.len(),
        }
    }

    // Internal methods

    /// Add a fact internally with dependency tracking
    fn add_fact_internal(
        &mut self,
        fact: RuleAtom,
        derived_by: Option<RuleId>,
        depends_on: HashSet<FactId>,
    ) -> Result<FactId> {
        let fact_id = self.next_fact_id;
        self.next_fact_id += 1;

        // Store fact
        self.facts.insert(fact_id, fact.clone());
        self.fact_index.insert(fact, fact_id);

        // Store dependency
        self.dependencies.insert(
            fact_id,
            FactDependency {
                depends_on: depends_on.clone(),
                derived_by,
                generation: self.generation,
            },
        );

        // Update reverse dependencies
        for dep_id in depends_on {
            self.reverse_dependencies
                .entry(dep_id)
                .or_default()
                .insert(fact_id);
        }

        Ok(fact_id)
    }

    /// Compute delta (newly derived facts) from adding a fact
    fn compute_delta_from_fact(&mut self, new_fact_id: FactId) -> Result<Vec<RuleAtom>> {
        let mut delta = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((new_fact_id, 0));

        let mut processed = HashSet::new();

        while let Some((fact_id, depth)) = queue.pop_front() {
            if depth >= self.max_depth {
                continue;
            }

            if !processed.insert(fact_id) {
                continue;
            }

            // OPTIMIZATION: Collect only rule IDs and body/head references instead of cloning entire rules
            let rules_to_process: Vec<(RuleId, Vec<RuleAtom>, Vec<RuleAtom>)> = self
                .rules
                .iter()
                .map(|(&id, rule)| {
                    INCREMENTAL_RULE_CLONES.inc();
                    (id, rule.body.clone(), rule.head.clone())
                })
                .collect();

            // Try to apply each rule
            for (rule_id, body, head) in rules_to_process {
                let derived = self.try_apply_rule_parts(rule_id, &body, &head, fact_id)?;

                for new_fact in derived {
                    // Check if this fact already exists
                    if self.fact_index.contains_key(&new_fact) {
                        continue;
                    }

                    // Add the derived fact
                    let new_id = self.add_fact_internal(
                        new_fact.clone(),
                        Some(rule_id),
                        [fact_id].iter().copied().collect(),
                    )?;

                    self.derived_facts.insert(new_id);
                    delta.push(new_fact);
                    queue.push_back((new_id, depth + 1));
                }
            }
        }

        Ok(delta)
    }

    /// Try to apply a rule given a new fact (takes body and head separately)
    /// OPTIMIZED: Takes body/head instead of entire Rule to avoid cloning rule names
    fn try_apply_rule_parts(
        &self,
        _rule_id: RuleId,
        body: &[RuleAtom],
        head: &[RuleAtom],
        new_fact_id: FactId,
    ) -> Result<Vec<RuleAtom>> {
        // True semi-naive delta join: only substitutions that actually involve
        // the newly added fact can yield genuinely new derivations (older
        // combinations were already materialized in prior generations). So we
        // seed the join by binding the new fact against each body-atom position,
        // then join the *remaining* atoms against the full fact set — instead of
        // re-joining the entire body against everything (which was an O(rules *
        // facts) full pass masquerading as an incremental update).
        let new_fact = match self.facts.get(&new_fact_id) {
            Some(fact) => fact,
            None => return Ok(Vec::new()),
        };

        let mut substitutions: Vec<HashMap<String, Term>> = Vec::new();

        for (seed_pos, seed_atom) in body.iter().enumerate() {
            // Bind the new fact against this body position (if it can match).
            for seed_sub in self.match_atom_against_fact(seed_atom, new_fact)? {
                // Join every other body atom against the full fact index,
                // carrying the seed bindings.
                let mut partials = vec![seed_sub];
                for (pos, atom) in body.iter().enumerate() {
                    if pos == seed_pos {
                        continue;
                    }
                    let mut extended = Vec::new();
                    for sub in &partials {
                        extended.extend(self.match_atom(atom, sub)?);
                    }
                    partials = extended;
                    if partials.is_empty() {
                        break;
                    }
                }
                for sub in partials {
                    if !substitutions.contains(&sub) {
                        substitutions.push(sub);
                    }
                }
            }
        }

        let mut derived = Vec::new();
        for substitution in &substitutions {
            for head_atom in head {
                derived.push(self.apply_substitution(head_atom, substitution)?);
            }
        }

        Ok(derived)
    }

    /// Bind a single fact against a body atom, returning the (at most one)
    /// substitution under which they unify. Non-triple atoms never match a
    /// stored triple fact and yield no bindings.
    fn match_atom_against_fact(
        &self,
        atom: &RuleAtom,
        fact: &RuleAtom,
    ) -> Result<Vec<HashMap<String, Term>>> {
        match (atom, fact) {
            (
                RuleAtom::Triple {
                    subject,
                    predicate,
                    object,
                },
                RuleAtom::Triple {
                    subject: fs,
                    predicate: fp,
                    object: fo,
                },
            ) => {
                let mut out = Vec::new();
                if let Some(sub) =
                    self.unify_triple((subject, predicate, object), (fs, fp, fo), &HashMap::new())?
                {
                    out.push(sub);
                }
                Ok(out)
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Match an atom against known facts
    /// OPTIMIZED: Pass reference to unify_triple instead of cloning
    fn match_atom(
        &self,
        atom: &RuleAtom,
        partial_sub: &HashMap<String, Term>,
    ) -> Result<Vec<HashMap<String, Term>>> {
        let mut substitutions = Vec::new();

        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                for fact in self.facts.values() {
                    if let RuleAtom::Triple {
                        subject: fs,
                        predicate: fp,
                        object: fo,
                    } = fact
                    {
                        // OPTIMIZATION: Pass reference - only clone on unification success
                        if let Some(sub) = self.unify_triple(
                            (subject, predicate, object),
                            (fs, fp, fo),
                            partial_sub, // Reference instead of clone!
                        )? {
                            substitutions.push(sub);
                        }
                    }
                }
            }
            _ => {
                // Handle other atom types
            }
        }

        Ok(substitutions)
    }

    /// Unify two triples
    /// OPTIMIZED: Takes reference, only clones on success
    fn unify_triple(
        &self,
        pattern: (&Term, &Term, &Term),
        fact: (&Term, &Term, &Term),
        substitution: &HashMap<String, Term>,
    ) -> Result<Option<HashMap<String, Term>>> {
        // Clone once at start - only this clone survives if unification succeeds
        let mut new_substitution = substitution.clone();

        if !self.unify_terms(pattern.0, fact.0, &mut new_substitution)? {
            return Ok(None);
        }
        if !self.unify_terms(pattern.1, fact.1, &mut new_substitution)? {
            return Ok(None);
        }
        if !self.unify_terms(pattern.2, fact.2, &mut new_substitution)? {
            return Ok(None);
        }

        // Track successful unification (only clones that survive)
        INCREMENTAL_SUBSTITUTION_CLONES.inc();
        Ok(Some(new_substitution))
    }

    /// Unify two terms
    fn unify_terms(
        &self,
        pattern: &Term,
        fact: &Term,
        substitution: &mut HashMap<String, Term>,
    ) -> Result<bool> {
        match (pattern, fact) {
            (Term::Variable(var), fact_term) => {
                if let Some(existing) = substitution.get(var) {
                    Ok(existing == fact_term)
                } else {
                    substitution.insert(var.clone(), fact_term.clone());
                    Ok(true)
                }
            }
            (Term::Constant(c1), Term::Constant(c2)) => Ok(c1 == c2),
            (Term::Literal(l1), Term::Literal(l2)) => Ok(l1 == l2),
            _ => Ok(false),
        }
    }

    /// Apply substitution to an atom
    fn apply_substitution(
        &self,
        atom: &RuleAtom,
        substitution: &HashMap<String, Term>,
    ) -> Result<RuleAtom> {
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
            _ => Ok(atom.clone()),
        }
    }

    /// Substitute variables in a term
    fn substitute_term(&self, term: &Term, substitution: &HashMap<String, Term>) -> Term {
        match term {
            Term::Variable(var) => substitution
                .get(var)
                .cloned()
                .unwrap_or_else(|| term.clone()),
            _ => term.clone(),
        }
    }

    /// Find all facts that depend on the given fact (transitively)
    fn find_affected_facts(&self, fact_id: FactId) -> HashSet<FactId> {
        let mut affected = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(fact_id);

        while let Some(id) = queue.pop_front() {
            if !affected.insert(id) {
                continue;
            }

            if let Some(dependents) = self.reverse_dependencies.get(&id) {
                for &dependent_id in dependents {
                    queue.push_back(dependent_id);
                }
            }
        }

        affected
    }
}

/// Statistics about incremental reasoning
#[derive(Debug, Clone)]
pub struct IncrementalStats {
    pub total_facts: usize,
    pub asserted_facts: usize,
    pub derived_facts: usize,
    pub total_rules: usize,
    pub generation: usize,
    pub snapshots: usize,
}

impl std::fmt::Display for IncrementalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Facts: {} (asserted: {}, derived: {}), Rules: {}, Generation: {}, Snapshots: {}",
            self.total_facts,
            self.asserted_facts,
            self.derived_facts,
            self.total_rules,
            self.generation,
            self.snapshots
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_add_fact() -> Result<(), Box<dyn std::error::Error>> {
        let mut reasoner = IncrementalReasoner::new();

        // Add a simple rule
        reasoner.add_rule(Rule {
            name: "mortal".to_string(),
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

        // Add a fact
        let delta = reasoner.add_fact(RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        })?;

        // Should derive that socrates is mortal
        assert!(!delta.is_empty());

        let stats = reasoner.get_stats();
        assert_eq!(stats.asserted_facts, 1);
        assert!(stats.derived_facts > 0);
        Ok(())
    }

    #[test]
    fn test_incremental_remove_fact() -> Result<(), Box<dyn std::error::Error>> {
        let mut reasoner = IncrementalReasoner::new();

        let fact = RuleAtom::Triple {
            subject: Term::Constant("socrates".to_string()),
            predicate: Term::Constant("type".to_string()),
            object: Term::Constant("human".to_string()),
        };

        reasoner.add_fact(fact.clone())?;
        let removed = reasoner.remove_fact(&fact)?;

        assert!(!removed.is_empty());
        assert_eq!(reasoner.get_stats().total_facts, 0);
        Ok(())
    }

    #[test]
    fn test_snapshot_restore() -> Result<(), Box<dyn std::error::Error>> {
        let mut reasoner = IncrementalReasoner::new();

        reasoner.create_snapshot();

        reasoner.add_fact(RuleAtom::Triple {
            subject: Term::Constant("test".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("o".to_string()),
        })?;

        assert_eq!(reasoner.get_stats().total_facts, 1);

        reasoner.restore_snapshot()?;

        assert_eq!(reasoner.get_stats().total_facts, 0);
        Ok(())
    }

    /// Regression: the incremental delta join must actually join a two-atom
    /// rule body across the newly added fact — deriving results that only exist
    /// because of the new fact, and only when the join is genuinely satisfied.
    #[test]
    fn regression_incremental_delta_join_two_atoms() -> Result<(), Box<dyn std::error::Error>> {
        let mut reasoner = IncrementalReasoner::new();
        // grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
        reasoner.add_rule(Rule {
            name: "grandparent".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("parent".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("grandparent".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        });

        // First fact: no join partner yet, so no grandparent derived.
        let d1 = reasoner.add_fact(RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("b".to_string()),
        })?;
        assert!(d1.is_empty(), "no grandparent should exist yet: {d1:?}");

        // Second fact closes the chain a->b->c: delta must contain grandparent(a,c).
        let d2 = reasoner.add_fact(RuleAtom::Triple {
            subject: Term::Constant("b".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("c".to_string()),
        })?;

        let expected = RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("grandparent".to_string()),
            object: Term::Constant("c".to_string()),
        };
        assert!(
            d2.contains(&expected),
            "delta join across the new fact must derive grandparent(a,c); got: {d2:?}"
        );
        // And must not over-derive a spurious self grandparent.
        assert!(
            !d2.iter().any(|f| matches!(f, RuleAtom::Triple { subject: Term::Constant(s), object: Term::Constant(o), .. } if s == "a" && o == "a")),
            "must not derive grandparent(a,a): {d2:?}"
        );
        Ok(())
    }
}
