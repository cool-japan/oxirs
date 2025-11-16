//! # Description Logic Reasoning with Tableaux Algorithm
//!
//! This module implements description logic (DL) reasoning using the tableaux algorithm,
//! which is a decision procedure for checking consistency and subsumption in DL knowledge bases.
//!
//! ## Supported DL Constructs
//!
//! - **Atomic concepts**: A, B, C
//! - **Top/Bottom**: ⊤ (Thing), ⊥ (Nothing)
//! - **Conjunction**: C ⊓ D (intersection)
//! - **Disjunction**: C ⊔ D (union)
//! - **Negation**: ¬C (complement)
//! - **Existential restriction**: ∃R.C (some values from)
//! - **Universal restriction**: ∀R.C (all values from)
//! - **Role**: R (object property)
//!
//! ## Tableaux Algorithm
//!
//! The tableaux algorithm constructs a completion tree by applying expansion rules:
//! 1. **⊓-rule**: If C ⊓ D ∈ L(x), add C and D to L(x)
//! 2. **⊔-rule**: If C ⊔ D ∈ L(x), branch with C or D in L(x)
//! 3. **∃-rule**: If ∃R.C ∈ L(x), create y with (x,y) : R and C ∈ L(y)
//! 4. **∀-rule**: If ∀R.C ∈ L(x) and (x,y) : R, add C to L(y)
//! 5. **Clash detection**: If C ∈ L(x) and ¬C ∈ L(x), mark branch as closed
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::description_logic::{TableauxReasoner, Concept, Role};
//!
//! let mut reasoner = TableauxReasoner::new();
//!
//! // Define concepts: Person ⊓ ∃hasChild.Person
//! let person = Concept::Atomic("Person".to_string());
//! let has_child = Role::new("hasChild".to_string());
//! let parent = Concept::And(
//!     Box::new(person.clone()),
//!     Box::new(Concept::Exists(has_child, Box::new(person)))
//! );
//!
//! // Check satisfiability
//! assert!(reasoner.is_satisfiable(&parent)?);
//! # Ok::<(), anyhow::Error>(())
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::metrics::{Counter, Gauge};
use std::collections::{HashMap, HashSet, VecDeque};

// Global metrics for tableaux reasoning
lazy_static::lazy_static! {
    static ref TABLEAUX_EXPANSIONS: Counter = Counter::new("tableaux_expansions".to_string());
    static ref TABLEAUX_CLASHES: Counter = Counter::new("tableaux_clashes".to_string());
    static ref TABLEAUX_NODE_COUNT: Gauge = Gauge::new("tableaux_node_count".to_string());
}

/// Description Logic concept
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Concept {
    /// Top concept (⊤, owl:Thing)
    Top,
    /// Bottom concept (⊥, owl:Nothing)
    Bottom,
    /// Atomic concept (named class)
    Atomic(String),
    /// Negation (¬C)
    Not(Box<Concept>),
    /// Conjunction (C ⊓ D)
    And(Box<Concept>, Box<Concept>),
    /// Disjunction (C ⊔ D)
    Or(Box<Concept>, Box<Concept>),
    /// Existential restriction (∃R.C)
    Exists(Role, Box<Concept>),
    /// Universal restriction (∀R.C)
    ForAll(Role, Box<Concept>),
    /// At-least cardinality restriction (≥nR.C)
    AtLeast(usize, Role, Box<Concept>),
    /// At-most cardinality restriction (≤nR.C)
    AtMost(usize, Role, Box<Concept>),
    /// Exactly cardinality restriction (=nR.C)
    Exactly(usize, Role, Box<Concept>),
}

/// Role (object property)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Role {
    pub name: String,
}

impl Role {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

/// Individual (node in completion tree)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Individual {
    pub id: usize,
    pub name: Option<String>,
}

impl Individual {
    pub fn new(id: usize) -> Self {
        Self { id, name: None }
    }

    pub fn named(id: usize, name: String) -> Self {
        Self {
            id,
            name: Some(name),
        }
    }
}

/// Node label in completion tree
#[derive(Debug, Clone)]
pub struct NodeLabel {
    /// Individual this label belongs to
    pub individual: Individual,
    /// Concepts in label L(x)
    pub concepts: HashSet<Concept>,
}

impl NodeLabel {
    pub fn new(individual: Individual) -> Self {
        Self {
            individual,
            concepts: HashSet::new(),
        }
    }

    /// Add concept to label
    pub fn add_concept(&mut self, concept: Concept) -> bool {
        self.concepts.insert(concept)
    }

    /// Check if label contains concept
    pub fn contains(&self, concept: &Concept) -> bool {
        self.concepts.contains(concept)
    }

    /// Check for clash (C and ¬C both present)
    pub fn has_clash(&self) -> bool {
        for concept in &self.concepts {
            let negation = Concept::Not(Box::new(concept.clone()));
            if self.concepts.contains(&negation) {
                return true;
            }
        }
        false
    }
}

/// Edge in completion tree (role assertion)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edge {
    pub from: Individual,
    pub role: Role,
    pub to: Individual,
}

impl Edge {
    pub fn new(from: Individual, role: Role, to: Individual) -> Self {
        Self { from, role, to }
    }
}

/// Completion tree for tableaux algorithm
#[derive(Debug, Clone)]
pub struct CompletionTree {
    /// Node labels L(x)
    pub labels: HashMap<Individual, NodeLabel>,
    /// Role edges R(x,y)
    pub edges: Vec<Edge>,
    /// Next individual ID
    next_id: usize,
}

impl CompletionTree {
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
            edges: Vec::new(),
            next_id: 0,
        }
    }

    /// Create new individual
    pub fn create_individual(&mut self) -> Individual {
        let individual = Individual::new(self.next_id);
        self.next_id += 1;
        self.labels
            .insert(individual.clone(), NodeLabel::new(individual.clone()));
        individual
    }

    /// Add concept to individual's label
    pub fn add_concept(&mut self, individual: &Individual, concept: Concept) -> bool {
        if let Some(label) = self.labels.get_mut(individual) {
            label.add_concept(concept)
        } else {
            false
        }
    }

    /// Add role edge
    pub fn add_edge(&mut self, from: Individual, role: Role, to: Individual) {
        self.edges.push(Edge::new(from, role, to));
    }

    /// Get successors via role
    pub fn get_successors(&self, individual: &Individual, role: &Role) -> Vec<Individual> {
        self.edges
            .iter()
            .filter(|e| &e.from == individual && &e.role == role)
            .map(|e| e.to.clone())
            .collect()
    }

    /// Check for any clash in tree
    pub fn has_clash(&self) -> bool {
        self.labels.values().any(|label| label.has_clash())
    }

    /// Get label for individual
    pub fn get_label(&self, individual: &Individual) -> Option<&NodeLabel> {
        self.labels.get(individual)
    }
}

impl Default for CompletionTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Expansion rule result
#[derive(Debug, Clone)]
pub enum ExpansionResult {
    /// No changes made
    NoChange,
    /// Single branch modified
    Modified,
    /// Branching into multiple trees
    Branching(Vec<CompletionTree>),
}

/// Tableaux reasoner for description logic
pub struct TableauxReasoner {
    /// Maximum expansion depth
    max_depth: usize,
    /// Maximum number of branches
    max_branches: usize,
    /// Statistics
    pub stats: TableauxStats,
}

impl Default for TableauxReasoner {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for tableaux reasoning
#[derive(Debug, Clone, Default)]
pub struct TableauxStats {
    pub expansions: usize,
    pub clashes: usize,
    pub branches: usize,
    pub max_depth_reached: usize,
}

impl TableauxReasoner {
    pub fn new() -> Self {
        Self {
            max_depth: 100,
            max_branches: 1000,
            stats: TableauxStats::default(),
        }
    }

    /// Set maximum expansion depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set maximum branches
    pub fn with_max_branches(mut self, branches: usize) -> Self {
        self.max_branches = branches;
        self
    }

    /// Check if concept is satisfiable
    pub fn is_satisfiable(&mut self, concept: &Concept) -> Result<bool> {
        // Create initial tree with concept
        let mut tree = CompletionTree::new();
        let root = tree.create_individual();
        tree.add_concept(&root, concept.clone());

        // Apply expansion rules until saturation or clash
        self.expand_tree(tree, 0)
    }

    /// Check if concept C is subsumed by concept D (C ⊑ D)
    pub fn is_subsumed_by(&mut self, c: &Concept, d: &Concept) -> Result<bool> {
        // C ⊑ D iff C ⊓ ¬D is unsatisfiable
        let negated_d = Concept::Not(Box::new(d.clone()));
        let conjunction = Concept::And(Box::new(c.clone()), Box::new(negated_d));

        let satisfiable = self.is_satisfiable(&conjunction)?;
        Ok(!satisfiable)
    }

    /// Check if two concepts are equivalent (C ≡ D)
    pub fn is_equivalent(&mut self, c: &Concept, d: &Concept) -> Result<bool> {
        let c_subsumes_d = self.is_subsumed_by(c, d)?;
        let d_subsumes_c = self.is_subsumed_by(d, c)?;
        Ok(c_subsumes_d && d_subsumes_c)
    }

    /// Expand completion tree using tableaux rules
    fn expand_tree(&mut self, mut tree: CompletionTree, depth: usize) -> Result<bool> {
        if depth > self.max_depth {
            return Err(anyhow!("Maximum expansion depth exceeded"));
        }

        // Check for clash
        if tree.has_clash() {
            self.stats.clashes += 1;
            TABLEAUX_CLASHES.inc();
            return Ok(false);
        }

        // Try to apply expansion rules
        let mut queue: VecDeque<Individual> = tree.labels.keys().cloned().collect();
        let mut changed = true;

        while changed {
            changed = false;

            // Process each individual
            while let Some(individual) = queue.pop_front() {
                let label = match tree.get_label(&individual) {
                    Some(l) => l.clone(),
                    None => continue,
                };

                // Apply ⊓-rule (conjunction)
                for concept in &label.concepts {
                    if let Concept::And(c1, c2) = concept {
                        if tree.add_concept(&individual, (**c1).clone()) {
                            changed = true;
                            queue.push_back(individual.clone());
                        }
                        if tree.add_concept(&individual, (**c2).clone()) {
                            changed = true;
                            queue.push_back(individual.clone());
                        }
                        self.stats.expansions += 1;
                        TABLEAUX_EXPANSIONS.inc();
                    }
                }

                // Apply ∃-rule (existential restriction)
                for concept in &label.concepts {
                    if let Concept::Exists(role, c) = concept {
                        // Check if there's already a suitable successor
                        let successors = tree.get_successors(&individual, role);
                        let has_successor = successors.iter().any(|succ| {
                            tree.get_label(succ).map(|l| l.contains(c)).unwrap_or(false)
                        });

                        if !has_successor {
                            // Create new individual with concept
                            let new_individual = tree.create_individual();
                            tree.add_edge(individual.clone(), role.clone(), new_individual.clone());
                            tree.add_concept(&new_individual, (**c).clone());
                            changed = true;
                            queue.push_back(new_individual);
                            self.stats.expansions += 1;
                            TABLEAUX_EXPANSIONS.inc();
                        }
                    }
                }

                // Apply ∀-rule (universal restriction)
                for concept in &label.concepts {
                    if let Concept::ForAll(role, c) = concept {
                        let successors = tree.get_successors(&individual, role);
                        for successor in successors {
                            if tree.add_concept(&successor, (**c).clone()) {
                                changed = true;
                                queue.push_back(successor);
                                self.stats.expansions += 1;
                                TABLEAUX_EXPANSIONS.inc();
                            }
                        }
                    }
                }

                // Check for clash after expansions
                if tree.has_clash() {
                    self.stats.clashes += 1;
                    TABLEAUX_CLASHES.inc();
                    return Ok(false);
                }
            }
        }

        // Handle ⊔-rule (disjunction) with branching
        for individual in tree.labels.keys().cloned().collect::<Vec<_>>() {
            let label = match tree.get_label(&individual) {
                Some(l) => l.clone(),
                None => continue,
            };

            for concept in &label.concepts {
                if let Concept::Or(c1, c2) = concept {
                    // Check if already satisfied
                    if label.contains(c1) || label.contains(c2) {
                        continue;
                    }

                    // Branch: try c1
                    let mut branch1 = tree.clone();
                    branch1.add_concept(&individual, (**c1).clone());

                    // Branch: try c2
                    let mut branch2 = tree.clone();
                    branch2.add_concept(&individual, (**c2).clone());

                    self.stats.branches += 2;

                    // Check if either branch is satisfiable
                    let satisfiable1 = self.expand_tree(branch1, depth + 1)?;
                    if satisfiable1 {
                        return Ok(true);
                    }

                    let satisfiable2 = self.expand_tree(branch2, depth + 1)?;
                    return Ok(satisfiable2);
                }
            }
        }

        // No more rules applicable - tree is complete and clash-free
        Ok(true)
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = TableauxStats::default();
    }
}

/// Normalize concept to negation normal form (NNF)
pub fn to_nnf(concept: &Concept) -> Concept {
    match concept {
        Concept::Top | Concept::Bottom | Concept::Atomic(_) => concept.clone(),

        Concept::Not(c) => match c.as_ref() {
            Concept::Top => Concept::Bottom,
            Concept::Bottom => Concept::Top,
            Concept::Atomic(_) => concept.clone(),
            Concept::Not(c2) => to_nnf(c2),
            Concept::And(c1, c2) => {
                let nnf1 = to_nnf(&Concept::Not(c1.clone()));
                let nnf2 = to_nnf(&Concept::Not(c2.clone()));
                Concept::Or(Box::new(nnf1), Box::new(nnf2))
            }
            Concept::Or(c1, c2) => {
                let nnf1 = to_nnf(&Concept::Not(c1.clone()));
                let nnf2 = to_nnf(&Concept::Not(c2.clone()));
                Concept::And(Box::new(nnf1), Box::new(nnf2))
            }
            Concept::Exists(r, c2) => {
                let nnf = to_nnf(&Concept::Not(c2.clone()));
                Concept::ForAll(r.clone(), Box::new(nnf))
            }
            Concept::ForAll(r, c2) => {
                let nnf = to_nnf(&Concept::Not(c2.clone()));
                Concept::Exists(r.clone(), Box::new(nnf))
            }
            _ => concept.clone(),
        },

        Concept::And(c1, c2) => Concept::And(Box::new(to_nnf(c1)), Box::new(to_nnf(c2))),

        Concept::Or(c1, c2) => Concept::Or(Box::new(to_nnf(c1)), Box::new(to_nnf(c2))),

        Concept::Exists(r, c) => Concept::Exists(r.clone(), Box::new(to_nnf(c))),

        Concept::ForAll(r, c) => Concept::ForAll(r.clone(), Box::new(to_nnf(c))),

        _ => concept.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_creation() {
        let person = Concept::Atomic("Person".to_string());
        let animal = Concept::Atomic("Animal".to_string());
        let conjunction = Concept::And(Box::new(person), Box::new(animal));

        match conjunction {
            Concept::And(c1, c2) => {
                assert!(matches!(*c1, Concept::Atomic(_)));
                assert!(matches!(*c2, Concept::Atomic(_)));
            }
            _ => panic!("Expected And concept"),
        }
    }

    #[test]
    fn test_completion_tree_creation() {
        let mut tree = CompletionTree::new();
        let ind1 = tree.create_individual();
        let ind2 = tree.create_individual();

        assert_eq!(ind1.id, 0);
        assert_eq!(ind2.id, 1);
        assert_eq!(tree.labels.len(), 2);
    }

    #[test]
    fn test_clash_detection() {
        let mut tree = CompletionTree::new();
        let ind = tree.create_individual();

        let person = Concept::Atomic("Person".to_string());
        let not_person = Concept::Not(Box::new(person.clone()));

        tree.add_concept(&ind, person);
        tree.add_concept(&ind, not_person);

        assert!(tree.has_clash());
    }

    #[test]
    fn test_satisfiability_simple() -> Result<()> {
        let mut reasoner = TableauxReasoner::new();

        // Person should be satisfiable
        let person = Concept::Atomic("Person".to_string());
        assert!(reasoner.is_satisfiable(&person)?);

        Ok(())
    }

    #[test]
    fn test_satisfiability_conjunction() -> Result<()> {
        let mut reasoner = TableauxReasoner::new();

        // Person ⊓ Animal should be satisfiable
        let person = Concept::Atomic("Person".to_string());
        let animal = Concept::Atomic("Animal".to_string());
        let both = Concept::And(Box::new(person), Box::new(animal));

        assert!(reasoner.is_satisfiable(&both)?);

        Ok(())
    }

    #[test]
    fn test_unsatisfiability_contradiction() -> Result<()> {
        let mut reasoner = TableauxReasoner::new();

        // Person ⊓ ¬Person should be unsatisfiable
        let person = Concept::Atomic("Person".to_string());
        let not_person = Concept::Not(Box::new(person.clone()));
        let contradiction = Concept::And(Box::new(person), Box::new(not_person));

        assert!(!reasoner.is_satisfiable(&contradiction)?);

        Ok(())
    }

    #[test]
    fn test_existential_restriction() -> Result<()> {
        let mut reasoner = TableauxReasoner::new();

        // ∃hasChild.Person should be satisfiable
        let person = Concept::Atomic("Person".to_string());
        let has_child = Role::new("hasChild".to_string());
        let parent = Concept::Exists(has_child, Box::new(person));

        assert!(reasoner.is_satisfiable(&parent)?);

        Ok(())
    }

    #[test]
    fn test_subsumption() -> Result<()> {
        let mut reasoner = TableauxReasoner::new();

        let person = Concept::Atomic("Person".to_string());
        let animal = Concept::Atomic("Animal".to_string());

        // Person ⊓ Animal ⊑ Person
        let both = Concept::And(Box::new(person.clone()), Box::new(animal));
        assert!(reasoner.is_subsumed_by(&both, &person)?);

        Ok(())
    }

    #[test]
    fn test_nnf_conversion() {
        let person = Concept::Atomic("Person".to_string());
        let animal = Concept::Atomic("Animal".to_string());

        // ¬(Person ⊓ Animal) → ¬Person ⊔ ¬Animal
        let and = Concept::And(Box::new(person.clone()), Box::new(animal.clone()));
        let negated = Concept::Not(Box::new(and));
        let nnf = to_nnf(&negated);

        match nnf {
            Concept::Or(c1, c2) => {
                assert!(matches!(*c1, Concept::Not(_)));
                assert!(matches!(*c2, Concept::Not(_)));
            }
            _ => panic!("Expected Or concept"),
        }
    }

    #[test]
    fn test_disjunction_branching() -> Result<()> {
        let mut reasoner = TableauxReasoner::new();

        // Person ⊔ Animal should be satisfiable
        let person = Concept::Atomic("Person".to_string());
        let animal = Concept::Atomic("Animal".to_string());
        let either = Concept::Or(Box::new(person), Box::new(animal));

        assert!(reasoner.is_satisfiable(&either)?);
        assert!(reasoner.stats.branches > 0);

        Ok(())
    }

    #[test]
    fn test_universal_restriction() -> Result<()> {
        let mut reasoner = TableauxReasoner::new();

        // ∀hasChild.Person ⊓ ∃hasChild.⊤ should be satisfiable
        let person = Concept::Atomic("Person".to_string());
        let has_child = Role::new("hasChild".to_string());

        let forall = Concept::ForAll(has_child.clone(), Box::new(person));
        let exists = Concept::Exists(has_child, Box::new(Concept::Top));

        let combined = Concept::And(Box::new(forall), Box::new(exists));

        assert!(reasoner.is_satisfiable(&combined)?);

        Ok(())
    }
}
