//! Core types for reasoning-aware SHACL validation.
//!
//! Defines entailment regimes, reasoning configuration, the inferred-triple
//! representation, the inference cache, validation results / statistics, and
//! the closed-world / negation-as-failure helper structures.

use crate::Result;
use oxirs_core::{
    model::{NamedNode, Term},
    Store,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Entailment regime for reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntailmentRegime {
    /// No entailment (simple RDF semantics)
    Simple,
    /// RDFS entailment
    RDFS,
    /// OWL 2 RDF-Based Semantics
    OWL2Full,
    /// OWL 2 RL (Rule Language) profile
    OWL2RL,
    /// OWL 2 QL (Query Language) profile
    OWL2QL,
    /// OWL 2 EL (Existential Language) profile
    OWL2EL,
    /// Custom reasoning rules
    Custom(CustomReasoning),
}

/// Custom reasoning configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomReasoning {
    /// Enable transitive property reasoning
    pub transitive: bool,
    /// Enable symmetric property reasoning
    pub symmetric: bool,
    /// Enable inverse property reasoning
    pub inverse: bool,
    /// Enable functional property reasoning
    pub functional: bool,
}

impl Default for CustomReasoning {
    fn default() -> Self {
        Self {
            transitive: true,
            symmetric: true,
            inverse: true,
            functional: true,
        }
    }
}

/// Reasoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Entailment regime to use
    pub entailment_regime: EntailmentRegime,
    /// Enable closed-world assumption
    pub closed_world_assumption: bool,
    /// Maximum reasoning depth (for recursion control)
    pub max_reasoning_depth: usize,
    /// Cache inferred triples
    pub cache_inferences: bool,
    /// Timeout for reasoning (milliseconds)
    pub reasoning_timeout_ms: Option<u64>,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            entailment_regime: EntailmentRegime::RDFS,
            closed_world_assumption: false,
            max_reasoning_depth: 100,
            cache_inferences: true,
            reasoning_timeout_ms: Some(30000), // 30 seconds
        }
    }
}

/// Inferred triple representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InferredTriple {
    /// Subject
    pub subject: Term,
    /// Predicate
    pub predicate: NamedNode,
    /// Object
    pub object: Term,
    /// Rule that derived this triple
    pub derived_by: String,
}

impl InferredTriple {
    /// Create a new inferred triple
    pub fn new(
        subject: Term,
        predicate: NamedNode,
        object: Term,
        derived_by: impl Into<String>,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            derived_by: derived_by.into(),
        }
    }
}

/// Cache for inferred triples
pub(crate) struct InferenceCache {
    /// Map from focus node to inferred triples
    cache: HashMap<Term, Vec<InferredTriple>>,
}

impl InferenceCache {
    pub(crate) fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub(crate) fn get(&self, focus_node: &Term) -> Option<&Vec<InferredTriple>> {
        self.cache.get(focus_node)
    }

    pub(crate) fn put(&mut self, focus_node: Term, inferred: Vec<InferredTriple>) {
        self.cache.insert(focus_node, inferred);
    }

    pub(crate) fn clear(&mut self) {
        self.cache.clear();
    }
}

/// Reasoning validation result
#[derive(Debug, Clone)]
pub struct ReasoningValidationResult {
    /// Whether validation succeeded
    pub conforms: bool,
    /// Number of inferred triples used
    pub inferred_triple_count: usize,
    /// Time spent on reasoning (milliseconds)
    pub reasoning_time_ms: u64,
    /// Whether result was from cache
    pub cache_hit: bool,
}

/// Reasoning statistics
#[derive(Debug, Clone, Default)]
pub struct ReasoningStats {
    /// Total validations performed
    pub total_validations: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Total inferred triples
    pub total_inferred_triples: usize,
    /// Total reasoning time (milliseconds)
    pub total_reasoning_time_ms: u64,
}

impl ReasoningStats {
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Calculate average reasoning time
    pub fn average_reasoning_time_ms(&self) -> f64 {
        if self.total_validations == 0 {
            0.0
        } else {
            self.total_reasoning_time_ms as f64 / self.total_validations as f64
        }
    }
}

/// Closed-world assumption validator
pub struct ClosedWorldValidator {
    /// Known predicates in the schema
    known_predicates: HashSet<NamedNode>,
    /// Known classes in the schema
    known_classes: HashSet<NamedNode>,
}

impl ClosedWorldValidator {
    /// Create a new CWA validator
    pub fn new() -> Self {
        Self {
            known_predicates: HashSet::new(),
            known_classes: HashSet::new(),
        }
    }

    /// Register a known predicate
    pub fn register_predicate(&mut self, predicate: NamedNode) {
        self.known_predicates.insert(predicate);
    }

    /// Register a known class
    pub fn register_class(&mut self, class: NamedNode) {
        self.known_classes.insert(class);
    }

    /// Check if a statement is false under CWA.
    ///
    /// Under the closed-world assumption, a triple that cannot be found in the
    /// store is considered false. This queries the store for the exact triple
    /// pattern and reports `true` (the statement is false) when no matching
    /// quad exists.
    pub fn is_false_under_cwa(
        &self,
        subject: &Term,
        predicate: &NamedNode,
        object: &Term,
        store: &dyn Store,
    ) -> Result<bool> {
        use oxirs_core::model::{Object, Predicate, Subject};

        // A literal or variable cannot be the subject of an RDF triple, so any
        // such statement is trivially absent from a well-formed store.
        let subject_term = match subject {
            Term::NamedNode(n) => Subject::from(n.clone()),
            Term::BlankNode(b) => Subject::from(b.clone()),
            _ => return Ok(true),
        };
        let predicate_term = Predicate::from(predicate.clone());
        let object_term = match object {
            Term::NamedNode(n) => Object::from(n.clone()),
            Term::BlankNode(b) => Object::from(b.clone()),
            Term::Literal(l) => Object::from(l.clone()),
            Term::Variable(_) | Term::QuotedTriple(_) => return Ok(true),
        };

        let quads = store.find_quads(
            Some(&subject_term),
            Some(&predicate_term),
            Some(&object_term),
            None,
        )?;

        // Closed-world: absent triple ⇒ false.
        Ok(quads.is_empty())
    }

    /// Check if a predicate is known
    pub fn is_known_predicate(&self, predicate: &NamedNode) -> bool {
        self.known_predicates.contains(predicate)
    }

    /// Check if a class is known
    pub fn is_known_class(&self, class: &NamedNode) -> bool {
        self.known_classes.contains(class)
    }
}

impl Default for ClosedWorldValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Negation as failure support
pub struct NegationAsFailure {
    /// Maximum depth for NAF checking
    max_depth: usize,
}

impl NegationAsFailure {
    /// Create a new NAF checker
    pub fn new() -> Self {
        Self { max_depth: 10 }
    }

    /// Check if a goal fails (cannot be proven).
    ///
    /// Negation-as-failure treats a goal as failing when no triple in the store
    /// matches its (subject, predicate, object) pattern. `None` components act
    /// as wildcards. This performs a single-step (depth-1) pattern match against
    /// the store; deeper rule-based proof search is deferred to a future round.
    pub fn fails(&self, goal: &NafGoal, store: &dyn Store) -> Result<bool> {
        use oxirs_core::model::{Object, Predicate, Subject};

        // max_depth currently bounds the (depth-1) match; recorded for future
        // multi-step proof search.
        let _ = self.max_depth;

        let subject_term = match &goal.subject {
            None => None,
            Some(Term::NamedNode(n)) => Some(Subject::from(n.clone())),
            Some(Term::BlankNode(b)) => Some(Subject::from(b.clone())),
            // A non-subject-capable term can never match: the goal fails.
            Some(_) => return Ok(true),
        };
        let predicate_term = goal.predicate.clone().map(Predicate::from);
        let object_term = match &goal.object {
            None => None,
            Some(Term::NamedNode(n)) => Some(Object::from(n.clone())),
            Some(Term::BlankNode(b)) => Some(Object::from(b.clone())),
            Some(Term::Literal(l)) => Some(Object::from(l.clone())),
            Some(Term::Variable(_)) | Some(Term::QuotedTriple(_)) => return Ok(true),
        };

        let quads = store.find_quads(
            subject_term.as_ref(),
            predicate_term.as_ref(),
            object_term.as_ref(),
            None,
        )?;

        // The goal fails (under NAF) exactly when nothing proves it.
        Ok(quads.is_empty())
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }
}

impl Default for NegationAsFailure {
    fn default() -> Self {
        Self::new()
    }
}

/// NAF goal representation
#[derive(Debug, Clone)]
pub struct NafGoal {
    /// Subject pattern
    pub subject: Option<Term>,
    /// Predicate pattern
    pub predicate: Option<NamedNode>,
    /// Object pattern
    pub object: Option<Term>,
}

impl NafGoal {
    /// Create a new NAF goal
    pub fn new(subject: Option<Term>, predicate: Option<NamedNode>, object: Option<Term>) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}
