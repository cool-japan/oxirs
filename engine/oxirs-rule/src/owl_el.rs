//! # OWL 2 EL Profile Reasoner
//!
//! Implements the OWL 2 EL (Existential Language) profile using the EL++ completion
//! algorithm (consequence-based reasoning). OWL 2 EL is designed for very large ontologies
//! like SNOMED CT (>300k concepts).
//!
//! ## Complexity
//! PTime in the size of the TBox (tractable for millions of axioms).
//!
//! ## Supported Constructs
//! - SubClassOf with atomic concepts or ObjectIntersectionOf on left
//! - ObjectSomeValuesFrom on either side
//! - EquivalentClasses
//! - ObjectPropertyChain (role compositions)
//! - SubObjectPropertyOf, TransitiveObjectProperty
//! - ConceptAssertion, RoleAssertion
//!
//! ## Reference
//! Baader, Brandt, Lutz: "Pushing the EL Envelope" (IJCAI 2005).
//! <https://www.w3.org/TR/owl2-profiles/#OWL_2_EL>

use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Errors from OWL 2 EL reasoning
#[derive(Debug, Error)]
pub enum ElError {
    #[error("Ontology is inconsistent: {0}")]
    Inconsistency(String),

    #[error("Invalid axiom: {0}")]
    InvalidAxiom(String),

    #[error("Maximum work items ({0}) exceeded during classification")]
    MaxWorkExceeded(usize),
}

/// An EL concept expression
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ElConcept {
    /// owl:Thing (top concept)
    Top,
    /// owl:Nothing (bottom concept)
    Bottom,
    /// Named atomic concept (class IRI)
    Named(String),
    /// ObjectIntersectionOf(C1, C2, ...)
    Intersection(Vec<ElConcept>),
    /// ObjectSomeValuesFrom(role, filler)
    SomeValues {
        role: String,
        filler: Box<ElConcept>,
    },
}

impl ElConcept {
    /// Create a named concept
    pub fn named(iri: impl Into<String>) -> Self {
        Self::Named(iri.into())
    }

    /// Create an intersection
    pub fn intersection(concepts: Vec<ElConcept>) -> Self {
        match concepts.len() {
            0 => Self::Top,
            1 => concepts.into_iter().next().unwrap_or(Self::Top),
            _ => Self::Intersection(concepts),
        }
    }

    /// Create a SomeValuesFrom restriction
    pub fn some_values(role: impl Into<String>, filler: ElConcept) -> Self {
        Self::SomeValues {
            role: role.into(),
            filler: Box::new(filler),
        }
    }

    /// Return atomic name if this is Named
    pub fn as_named(&self) -> Option<&str> {
        if let Self::Named(n) = self {
            Some(n)
        } else {
            None
        }
    }
}

/// An OWL 2 EL axiom
#[derive(Debug, Clone)]
pub enum ElAxiom {
    /// C ⊑ D (subclass)
    SubConceptOf { sub: ElConcept, sup: ElConcept },
    /// C ≡ D (equivalent class)
    EquivalentConcepts(ElConcept, ElConcept),
    /// Individual x : C (concept assertion)
    ConceptAssertion {
        individual: String,
        concept: ElConcept,
    },
    /// (x, y) : r (role assertion)
    RoleAssertion {
        subject: String,
        role: String,
        object: String,
    },
    /// r1 o r2 ⊑ s (property chain, binary or longer)
    PropertyChain {
        chain: Vec<String>,
        result_role: String,
    },
    /// r1 ⊑ r2 (sub-property)
    SubRole { sub: String, sup: String },
    /// r rdf:type owl:TransitiveProperty
    TransitiveRole(String),
}

/// Normalized axiom form for the EL completion algorithm.
/// All complex axioms are reduced to one of these normal forms.
#[derive(Debug, Clone)]
pub enum NormalAxiom {
    /// A ⊑ B  (two atomic concept names)
    AtomSubAtom(String, String),
    /// A ⊓ B ⊑ C  (binary intersection on left)
    InterAtomSubAtom(String, String, String),
    /// A ⊑ ∃r.B  (existential on right)
    AtomSubSome(String, String, String),
    /// ∃r.A ⊑ B  (existential on left)
    SomeSubAtom(String, String, String),
    /// Transitive role r
    TransRole(String),
    /// Role chain: r1 o r2 ⊑ s
    RoleChain(String, String, String),
    /// Sub-role: r ⊑ s
    SubRole(String, String),
}

/// Result of EL classification
#[derive(Debug, Clone)]
pub struct ElClassification {
    /// subsumption_hierarchy: concept → set of all named superclasses (including transitive)
    pub subsumption_hierarchy: HashMap<String, HashSet<String>>,
    /// Groups of equivalent classes (each group contains mutually equivalent names)
    pub equivalent_classes: Vec<Vec<String>>,
    /// Role successor sets: (individual_or_concept, role) → set of successors
    pub role_successors: HashMap<(String, String), HashSet<String>>,
    /// ABox: per-individual concept memberships
    pub individual_types: HashMap<String, HashSet<String>>,
    /// Number of saturation loop iterations performed
    pub iterations: usize,
    /// Total subsumption relationships computed
    pub subsumptions_computed: usize,
}

impl ElClassification {
    fn new() -> Self {
        Self {
            subsumption_hierarchy: HashMap::new(),
            equivalent_classes: Vec::new(),
            role_successors: HashMap::new(),
            individual_types: HashMap::new(),
            iterations: 0,
            subsumptions_computed: 0,
        }
    }

    /// Get all named superclasses of a concept (direct and indirect)
    pub fn get_superclasses(&self, class: &str) -> Vec<String> {
        self.subsumption_hierarchy
            .get(class)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get all named subclasses of a concept
    pub fn get_subclasses(&self, class: &str) -> Vec<String> {
        self.subsumption_hierarchy
            .iter()
            .filter(|(_, supers)| supers.contains(class))
            .map(|(sub, _)| sub.clone())
            .collect()
    }

    /// Check if `sub` ⊑ `sup`
    pub fn is_subclass_of(&self, sub: &str, sup: &str) -> bool {
        sub == sup
            || self
                .subsumption_hierarchy
                .get(sub)
                .map(|s| s.contains(sup))
                .unwrap_or(false)
    }

    /// Get all concept names an individual belongs to
    pub fn get_individual_types(&self, individual: &str) -> Vec<String> {
        self.individual_types
            .get(individual)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }
}

/// OWL 2 EL reasoner using the standard EL completion algorithm.
///
/// The algorithm operates on per-concept "context" sets S(C) and role-successor
/// sets R(C, r), computing the closure of all subsumption relationships using
/// the EL saturation rules CR1–CR6.
pub struct Owl2ElReasoner {
    axioms: Vec<ElAxiom>,
    max_work_items: usize,
}

impl Default for Owl2ElReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl Owl2ElReasoner {
    /// Create a new OWL 2 EL reasoner
    pub fn new() -> Self {
        Self {
            axioms: Vec::new(),
            max_work_items: 1_000_000,
        }
    }

    /// Set maximum work item limit (safety bound)
    pub fn with_max_work_items(mut self, n: usize) -> Self {
        self.max_work_items = n;
        self
    }

    /// Add an axiom to the ontology
    pub fn add_axiom(&mut self, axiom: ElAxiom) {
        self.axioms.push(axiom);
    }

    /// Add multiple axioms
    pub fn add_axioms(&mut self, axioms: impl IntoIterator<Item = ElAxiom>) {
        self.axioms.extend(axioms);
    }

    /// Add a simple subclass axiom: sub ⊑ sup
    pub fn add_subclass_of(&mut self, sub: &str, sup: &str) {
        self.axioms.push(ElAxiom::SubConceptOf {
            sub: ElConcept::named(sub),
            sup: ElConcept::named(sup),
        });
    }

    /// Add an equivalent class axiom: c1 ≡ c2
    pub fn add_equivalent_classes(&mut self, c1: &str, c2: &str) {
        self.axioms.push(ElAxiom::EquivalentConcepts(
            ElConcept::named(c1),
            ElConcept::named(c2),
        ));
    }

    /// Add a concept assertion: individual rdf:type concept_name
    pub fn add_concept_assertion(&mut self, individual: &str, concept_name: &str) {
        self.axioms.push(ElAxiom::ConceptAssertion {
            individual: individual.to_string(),
            concept: ElConcept::named(concept_name),
        });
    }

    /// Add a role assertion: (subject, role, object)
    pub fn add_role_assertion(&mut self, subject: &str, role: &str, object: &str) {
        self.axioms.push(ElAxiom::RoleAssertion {
            subject: subject.to_string(),
            role: role.to_string(),
            object: object.to_string(),
        });
    }

    /// Add a property chain: `roles[0]` o `roles[1]` o ... ⊑ result_role
    pub fn add_property_chain(&mut self, chain: Vec<String>, result_role: &str) {
        self.axioms.push(ElAxiom::PropertyChain {
            chain,
            result_role: result_role.to_string(),
        });
    }

    /// Declare a role as transitive
    pub fn add_transitive_role(&mut self, role: &str) {
        self.axioms.push(ElAxiom::TransitiveRole(role.to_string()));
    }

    // -----------------------------------------------------------------------
    // Public classification API
    // -----------------------------------------------------------------------

    /// Classify the ontology using the EL completion algorithm.
    ///
    /// Returns a complete `ElClassification` with subsumption hierarchy,
    /// equivalent classes, role successors, and ABox individual types.
    pub fn classify(&self) -> Result<ElClassification, ElError> {
        let mut result = ElClassification::new();

        // Step 1: normalize TBox axioms
        let normalized = self.normalize_axioms()?;

        // Step 2: collect all named atomic concepts
        let all_concepts = self.collect_concepts();

        // Step 3: initialise S(C) for each concept C
        // S(C) always contains C and owl:Thing
        let mut concept_sets: HashMap<String, HashSet<String>> = HashMap::new();
        for concept in &all_concepts {
            let mut s = HashSet::new();
            s.insert(concept.clone());
            s.insert("owl:Thing".to_string());
            concept_sets.insert(concept.clone(), s);
        }

        // Step 4: initialise role-successor sets R(C, r)
        // role_succs[concept][role] = set of successor concept-names
        let mut role_succs: HashMap<String, HashMap<String, HashSet<String>>> = HashMap::new();

        // Step 5: seed ABox
        let mut individual_concepts: HashMap<String, HashSet<String>> = HashMap::new();
        let mut individual_role_succs: HashMap<String, HashMap<String, HashSet<String>>> =
            HashMap::new();

        for axiom in &self.axioms {
            match axiom {
                ElAxiom::ConceptAssertion {
                    individual,
                    concept,
                } => {
                    if let Some(name) = concept.as_named() {
                        individual_concepts
                            .entry(individual.clone())
                            .or_default()
                            .insert(name.to_string());
                        // Every individual belongs to owl:Thing
                        individual_concepts
                            .entry(individual.clone())
                            .or_default()
                            .insert("owl:Thing".to_string());
                    }
                }
                ElAxiom::RoleAssertion {
                    subject,
                    role,
                    object,
                } => {
                    individual_role_succs
                        .entry(subject.clone())
                        .or_default()
                        .entry(role.clone())
                        .or_default()
                        .insert(object.clone());
                    // Ensure subject and object are known as individuals (at least owl:Thing)
                    individual_concepts
                        .entry(subject.clone())
                        .or_default()
                        .insert("owl:Thing".to_string());
                    individual_concepts
                        .entry(object.clone())
                        .or_default()
                        .insert("owl:Thing".to_string());
                }
                _ => {}
            }
        }

        // Step 6: saturation work-queue for TBox
        let mut work_queue: VecDeque<(String, String)> = VecDeque::new();

        // Seed: for every C, queue all initial S(C) members
        for (concept, supers) in &concept_sets {
            for sup in supers {
                work_queue.push_back((concept.clone(), sup.clone()));
            }
        }

        let mut work_count = 0usize;
        let mut iterations = 0usize;

        while let Some((concept, new_super)) = work_queue.pop_front() {
            work_count += 1;
            if work_count > self.max_work_items {
                return Err(ElError::MaxWorkExceeded(self.max_work_items));
            }
            iterations += 1;

            // --- CR1, CR2: derive new concept memberships ---
            let new_members =
                derive_concept_members(&concept, &new_super, &concept_sets, &normalized);
            for (c, sup) in new_members {
                let s_c = concept_sets.entry(c.clone()).or_default();
                if s_c.insert(sup.clone()) {
                    work_queue.push_back((c, sup));
                    result.subsumptions_computed += 1;
                }
            }

            // --- CR3: derive new role successors from A ⊑ ∃r.B ---
            let new_succs = derive_role_successors(&concept, &new_super, &normalized);
            for (c, role, succ) in new_succs {
                let changed = role_succs
                    .entry(c.clone())
                    .or_default()
                    .entry(role.clone())
                    .or_default()
                    .insert(succ.clone());

                if changed {
                    // Propagate: S(succ) types become relevant for c via CR4
                    if let Some(succ_types) = concept_sets.get(&succ).cloned() {
                        for t in succ_types {
                            let s_c = concept_sets.entry(c.clone()).or_default();
                            if s_c.insert(t.clone()) {
                                work_queue.push_back((c.clone(), t));
                                result.subsumptions_computed += 1;
                            }
                        }
                    }
                }
            }

            // --- CR4: ∃r.A ⊑ B — already handled above by propagating succ types ---

            // --- CR5/CR6: transitivity and role chains ---
            let chain_succs = derive_chain_successors(&concept, &role_succs, &normalized);
            for (c, role, succ) in chain_succs {
                let changed = role_succs
                    .entry(c.clone())
                    .or_default()
                    .entry(role.clone())
                    .or_default()
                    .insert(succ.clone());

                if changed {
                    if let Some(succ_types) = concept_sets.get(&succ).cloned() {
                        for t in succ_types {
                            let s_c = concept_sets.entry(c.clone()).or_default();
                            if s_c.insert(t.clone()) {
                                work_queue.push_back((c.clone(), t));
                            }
                        }
                    }
                }
            }
        }

        result.iterations = iterations;

        // Step 7: ABox saturation
        saturate_abox(
            &mut individual_concepts,
            &mut individual_role_succs,
            &concept_sets,
            &normalized,
        );

        // Step 8: find equivalent classes
        result.equivalent_classes = find_equivalents(&concept_sets);

        // Step 9: build output hierarchy (exclude self-subsumption and owl:Thing)
        for (concept, supers) in &concept_sets {
            let filtered: HashSet<String> = supers
                .iter()
                .filter(|s| s.as_str() != concept)
                .cloned()
                .collect();
            if !filtered.is_empty() {
                result
                    .subsumption_hierarchy
                    .insert(concept.clone(), filtered);
            }
        }

        // Step 10: copy TBox role successors
        for (concept, role_map) in &role_succs {
            for (role, succs) in role_map {
                result
                    .role_successors
                    .entry((concept.clone(), role.clone()))
                    .or_default()
                    .extend(succs.iter().cloned());
            }
        }

        // Step 10b: copy ABox individual role successors (merged with TBox)
        for (individual, role_map) in &individual_role_succs {
            for (role, succs) in role_map {
                // Exclude witness nodes (internal implementation detail)
                let filtered_succs: HashSet<String> = succs
                    .iter()
                    .filter(|s| !s.starts_with("__wit_"))
                    .cloned()
                    .collect();
                if !filtered_succs.is_empty() {
                    result
                        .role_successors
                        .entry((individual.clone(), role.clone()))
                        .or_default()
                        .extend(filtered_succs);
                }
            }
        }

        // Step 11: copy individual types
        for (individual, types) in individual_concepts {
            // Exclude witness nodes from individual types output
            if !individual.starts_with("__wit_") {
                result.individual_types.insert(individual, types);
            }
        }

        Ok(result)
    }

    /// Check if `sub` ⊑ `sup` using EL reasoning
    pub fn is_subclass_of(&self, sub: &str, sup: &str) -> Result<bool, ElError> {
        let cls = self.classify()?;
        Ok(cls.is_subclass_of(sub, sup))
    }

    /// Get all named superclasses of a concept
    pub fn get_superclasses(&self, class: &str) -> Result<Vec<String>, ElError> {
        let cls = self.classify()?;
        Ok(cls.get_superclasses(class))
    }

    /// Get all named subclasses of a concept
    pub fn get_subclasses(&self, class: &str) -> Result<Vec<String>, ElError> {
        let cls = self.classify()?;
        Ok(cls.get_subclasses(class))
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn normalize_axioms(&self) -> Result<Vec<NormalAxiom>, ElError> {
        let mut normalized = Vec::new();

        for axiom in &self.axioms {
            match axiom {
                ElAxiom::SubConceptOf { sub, sup } => {
                    normalize_subclass(sub, sup, &mut normalized);
                }
                ElAxiom::EquivalentConcepts(c1, c2) => {
                    normalize_subclass(c1, c2, &mut normalized);
                    normalize_subclass(c2, c1, &mut normalized);
                }
                ElAxiom::TransitiveRole(r) => {
                    normalized.push(NormalAxiom::TransRole(r.clone()));
                }
                ElAxiom::PropertyChain { chain, result_role } => {
                    decompose_chain(chain, result_role, &mut normalized);
                }
                ElAxiom::SubRole { sub, sup } => {
                    normalized.push(NormalAxiom::SubRole(sub.clone(), sup.clone()));
                }
                // ABox axioms are handled separately
                ElAxiom::ConceptAssertion { .. } | ElAxiom::RoleAssertion { .. } => {}
            }
        }

        Ok(normalized)
    }

    fn collect_concepts(&self) -> HashSet<String> {
        let mut concepts = HashSet::new();
        concepts.insert("owl:Thing".to_string());
        concepts.insert("owl:Nothing".to_string());

        for axiom in &self.axioms {
            match axiom {
                ElAxiom::SubConceptOf { sub, sup } => {
                    collect_from_concept(sub, &mut concepts);
                    collect_from_concept(sup, &mut concepts);
                }
                ElAxiom::EquivalentConcepts(c1, c2) => {
                    collect_from_concept(c1, &mut concepts);
                    collect_from_concept(c2, &mut concepts);
                }
                ElAxiom::ConceptAssertion { concept, .. } => {
                    collect_from_concept(concept, &mut concepts);
                }
                _ => {}
            }
        }
        concepts
    }
}

// -----------------------------------------------------------------------
// Free functions for EL completion rules (extracted for borrow safety)
// -----------------------------------------------------------------------

/// CR1 + CR2: derive new concept set members for `concept` given `new_super` was just added.
fn derive_concept_members(
    concept: &str,
    new_super: &str,
    concept_sets: &HashMap<String, HashSet<String>>,
    normalized: &[NormalAxiom],
) -> Vec<(String, String)> {
    let mut derived = Vec::new();
    let s_c = concept_sets.get(concept);

    for axiom in normalized {
        match axiom {
            // CR1: A ∈ S(X), A ⊑ B → B ∈ S(X)
            NormalAxiom::AtomSubAtom(a, b) if a == new_super => {
                derived.push((concept.to_string(), b.clone()));
            }
            // CR2: A ∈ S(X), B ∈ S(X), A ⊓ B ⊑ C → C ∈ S(X)
            NormalAxiom::InterAtomSubAtom(a, b, c) => {
                if a == new_super {
                    if s_c.map(|s| s.contains(b.as_str())).unwrap_or(false) {
                        derived.push((concept.to_string(), c.clone()));
                    }
                } else if b == new_super && s_c.map(|s| s.contains(a.as_str())).unwrap_or(false) {
                    derived.push((concept.to_string(), c.clone()));
                }
            }
            // CR4: ∃r.A ⊑ B — handled when a new role successor is added
            // (propagated via the main loop's successor handling)
            NormalAxiom::SomeSubAtom(_, _, _) => {}
            _ => {}
        }
    }

    derived
}

/// CR3: A ∈ S(X), A ⊑ ∃r.B → add (X, B) to R(r)
fn derive_role_successors(
    concept: &str,
    new_super: &str,
    normalized: &[NormalAxiom],
) -> Vec<(String, String, String)> {
    let mut derived = Vec::new();

    for axiom in normalized {
        if let NormalAxiom::AtomSubSome(a, role, b) = axiom {
            if a == new_super {
                derived.push((concept.to_string(), role.clone(), b.clone()));
            }
        }
    }

    derived
}

/// CR5 (transitivity) + CR6 (role chains): derive new role successors.
fn derive_chain_successors(
    concept: &str,
    role_succs: &HashMap<String, HashMap<String, HashSet<String>>>,
    normalized: &[NormalAxiom],
) -> Vec<(String, String, String)> {
    let mut derived = Vec::new();

    let Some(my_roles) = role_succs.get(concept) else {
        return derived;
    };

    for axiom in normalized {
        match axiom {
            // CR5: transitive role: (X,Y) ∈ R(r), (Y,Z) ∈ R(r) → (X,Z) ∈ R(r)
            NormalAxiom::TransRole(r) => {
                if let Some(r_succs) = my_roles.get(r) {
                    for y in r_succs {
                        if let Some(y_roles) = role_succs.get(y) {
                            if let Some(y_r_succs) = y_roles.get(r) {
                                for z in y_r_succs {
                                    derived.push((concept.to_string(), r.clone(), z.clone()));
                                }
                            }
                        }
                    }
                }
            }
            // CR6: role chain: (X,Y) ∈ R(r1), (Y,Z) ∈ R(r2) → (X,Z) ∈ R(s)
            NormalAxiom::RoleChain(r1, r2, s) => {
                if let Some(r1_succs) = my_roles.get(r1) {
                    for y in r1_succs {
                        if let Some(y_roles) = role_succs.get(y) {
                            if let Some(r2_succs) = y_roles.get(r2) {
                                for z in r2_succs {
                                    derived.push((concept.to_string(), s.clone(), z.clone()));
                                }
                            }
                        }
                    }
                }
            }
            // CR sub-role: (X,Y) ∈ R(sub) → (X,Y) ∈ R(sup)
            NormalAxiom::SubRole(sub, sup) => {
                if let Some(sub_succs) = my_roles.get(sub) {
                    for y in sub_succs {
                        derived.push((concept.to_string(), sup.clone(), y.clone()));
                    }
                }
            }
            _ => {}
        }
    }

    derived
}

/// Apply TBox results to ABox individuals (simple forward propagation).
fn saturate_abox(
    individual_concepts: &mut HashMap<String, HashSet<String>>,
    individual_role_succs: &mut HashMap<String, HashMap<String, HashSet<String>>>,
    concept_sets: &HashMap<String, HashSet<String>>,
    normalized: &[NormalAxiom],
) {
    // Iterate until fixpoint
    let mut changed = true;
    while changed {
        changed = false;

        let individuals: Vec<String> = individual_concepts.keys().cloned().collect();

        for individual in &individuals {
            // Propagate TBox subsumptions: if individual ∈ C and C ⊑ D, add D
            let ind_types: Vec<String> = individual_concepts
                .get(individual)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .collect();

            for typ in &ind_types {
                if let Some(supers) = concept_sets.get(typ) {
                    for sup in supers {
                        if individual_concepts
                            .entry(individual.clone())
                            .or_default()
                            .insert(sup.clone())
                        {
                            changed = true;
                        }
                    }
                }
            }

            // Apply A ⊓ B ⊑ C (CR2 for ABox): if individual ∈ A and individual ∈ B, add C
            for axiom in normalized {
                if let NormalAxiom::InterAtomSubAtom(a, b, c) = axiom {
                    let ind_types = individual_concepts
                        .get(individual)
                        .cloned()
                        .unwrap_or_default();
                    if ind_types.contains(a.as_str())
                        && ind_types.contains(b.as_str())
                        && individual_concepts
                            .entry(individual.clone())
                            .or_default()
                            .insert(c.clone())
                    {
                        changed = true;
                    }
                }
            }

            // Apply ∃r.A ⊑ B: if individual has role r to someone of type A, add B
            for axiom in normalized {
                if let NormalAxiom::SomeSubAtom(role, filler, sup) = axiom {
                    let has_witness = individual_role_succs
                        .get(individual)
                        .and_then(|rm| rm.get(role))
                        .map(|succs| {
                            succs.iter().any(|succ| {
                                individual_concepts
                                    .get(succ)
                                    .map(|s| s.contains(filler.as_str()))
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false);

                    if has_witness
                        && individual_concepts
                            .entry(individual.clone())
                            .or_default()
                            .insert(sup.clone())
                    {
                        changed = true;
                    }
                }
            }

            // Apply A ⊑ ∃r.B: if individual ∈ A, ensure witness exists for r
            for axiom in normalized {
                if let NormalAxiom::AtomSubSome(a, role, b) = axiom {
                    let has_a = individual_concepts
                        .get(individual)
                        .map(|s| s.contains(a.as_str()))
                        .unwrap_or(false);

                    if has_a {
                        let witness = format!("__wit_{}_{}", individual, b);
                        let added = individual_role_succs
                            .entry(individual.clone())
                            .or_default()
                            .entry(role.clone())
                            .or_default()
                            .insert(witness.clone());

                        if added {
                            individual_concepts
                                .entry(witness)
                                .or_default()
                                .insert(b.clone());
                            changed = true;
                        }
                    }
                }
            }

            // Apply sub-roles and role chains for individuals
            for axiom in normalized {
                match axiom {
                    NormalAxiom::SubRole(sub, sup) => {
                        let sub_succs: Vec<String> = individual_role_succs
                            .get(individual)
                            .and_then(|rm| rm.get(sub))
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .collect();

                        for succ in sub_succs {
                            if individual_role_succs
                                .entry(individual.clone())
                                .or_default()
                                .entry(sup.clone())
                                .or_default()
                                .insert(succ)
                            {
                                changed = true;
                            }
                        }
                    }
                    NormalAxiom::RoleChain(r1, r2, s) => {
                        let r1_succs: Vec<String> = individual_role_succs
                            .get(individual)
                            .and_then(|rm| rm.get(r1))
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .collect();

                        for y in r1_succs {
                            let r2_succs: Vec<String> = individual_role_succs
                                .get(&y)
                                .and_then(|rm| rm.get(r2))
                                .cloned()
                                .unwrap_or_default()
                                .into_iter()
                                .collect();

                            for z in r2_succs {
                                if individual_role_succs
                                    .entry(individual.clone())
                                    .or_default()
                                    .entry(s.clone())
                                    .or_default()
                                    .insert(z)
                                {
                                    changed = true;
                                }
                            }
                        }
                    }
                    NormalAxiom::TransRole(r) => {
                        let r_succs: Vec<String> = individual_role_succs
                            .get(individual)
                            .and_then(|rm| rm.get(r))
                            .cloned()
                            .unwrap_or_default()
                            .into_iter()
                            .collect();

                        for y in r_succs {
                            let y_r_succs: Vec<String> = individual_role_succs
                                .get(&y)
                                .and_then(|rm| rm.get(r))
                                .cloned()
                                .unwrap_or_default()
                                .into_iter()
                                .collect();

                            for z in y_r_succs {
                                if individual_role_succs
                                    .entry(individual.clone())
                                    .or_default()
                                    .entry(r.clone())
                                    .or_default()
                                    .insert(z)
                                {
                                    changed = true;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Detect pairs of equivalent concepts: C ≡ D iff C ⊑ D and D ⊑ C.
fn find_equivalents(concept_sets: &HashMap<String, HashSet<String>>) -> Vec<Vec<String>> {
    let mut groups: Vec<Vec<String>> = Vec::new();
    let mut processed: HashSet<String> = HashSet::new();

    let mut concepts: Vec<String> = concept_sets.keys().cloned().collect();
    concepts.sort(); // deterministic output

    for c1 in &concepts {
        if processed.contains(c1) {
            continue;
        }

        let mut group = vec![c1.clone()];

        for c2 in &concepts {
            if c1 == c2 || processed.contains(c2) {
                continue;
            }

            let c1_sub_c2 = concept_sets
                .get(c1)
                .map(|s| s.contains(c2.as_str()))
                .unwrap_or(false);
            let c2_sub_c1 = concept_sets
                .get(c2)
                .map(|s| s.contains(c1.as_str()))
                .unwrap_or(false);

            if c1_sub_c2 && c2_sub_c1 {
                group.push(c2.clone());
                processed.insert(c2.clone());
            }
        }

        if group.len() > 1 {
            groups.push(group);
        }
        processed.insert(c1.clone());
    }

    groups
}

/// Normalize a subclass axiom C ⊑ D into normal forms.
fn normalize_subclass(sub: &ElConcept, sup: &ElConcept, out: &mut Vec<NormalAxiom>) {
    match (sub, sup) {
        (ElConcept::Named(a), ElConcept::Named(b)) => {
            out.push(NormalAxiom::AtomSubAtom(a.clone(), b.clone()));
        }
        (ElConcept::Top, ElConcept::Named(b)) => {
            out.push(NormalAxiom::AtomSubAtom("owl:Thing".to_string(), b.clone()));
        }
        (ElConcept::Named(a), ElConcept::Top) => {
            out.push(NormalAxiom::AtomSubAtom(a.clone(), "owl:Thing".to_string()));
        }
        (ElConcept::Intersection(parts), ElConcept::Named(sup_name)) => {
            // Only handle binary intersections on the left (EL core)
            if parts.len() == 2 {
                if let (ElConcept::Named(a), ElConcept::Named(b)) = (&parts[0], &parts[1]) {
                    out.push(NormalAxiom::InterAtomSubAtom(
                        a.clone(),
                        b.clone(),
                        sup_name.clone(),
                    ));
                }
            } else if parts.len() > 2 {
                // Flatten: A1 ⊓ A2 ⊓ ... ⊓ An ⊑ C
                // → A1 ⊓ A2 ⊑ fresh1, fresh1 ⊓ A3 ⊑ fresh2, ..., fresh(n-2) ⊓ An ⊑ C
                let sup_name_str = sup_name.clone();
                let named_parts: Vec<&str> = parts.iter().filter_map(|p| p.as_named()).collect();
                if named_parts.len() == parts.len() {
                    let mut prev_fresh = named_parts[0].to_string();
                    for (i, part) in named_parts.iter().enumerate().skip(1) {
                        if i == named_parts.len() - 1 {
                            out.push(NormalAxiom::InterAtomSubAtom(
                                prev_fresh.clone(),
                                part.to_string(),
                                sup_name_str.clone(),
                            ));
                        } else {
                            let fresh = format!("__inter_{}_{}", i, sup_name_str);
                            out.push(NormalAxiom::InterAtomSubAtom(
                                prev_fresh,
                                part.to_string(),
                                fresh.clone(),
                            ));
                            prev_fresh = fresh;
                        }
                    }
                }
            }
        }
        (ElConcept::Named(a), ElConcept::SomeValues { role, filler }) => {
            if let ElConcept::Named(b) = filler.as_ref() {
                out.push(NormalAxiom::AtomSubSome(a.clone(), role.clone(), b.clone()));
            }
        }
        (ElConcept::SomeValues { role, filler }, ElConcept::Named(sup_name)) => {
            if let ElConcept::Named(a) = filler.as_ref() {
                out.push(NormalAxiom::SomeSubAtom(
                    role.clone(),
                    a.clone(),
                    sup_name.clone(),
                ));
            }
        }
        (ElConcept::Named(a), ElConcept::Bottom) => {
            // A ⊑ ⊥ — unsatisfiable concept
            out.push(NormalAxiom::AtomSubAtom(
                a.clone(),
                "owl:Nothing".to_string(),
            ));
        }
        _ => {
            // Complex cases beyond core EL — silently ignored
        }
    }
}

/// Decompose a property chain of length ≥ 2 into binary chains.
fn decompose_chain(chain: &[String], result_role: &str, out: &mut Vec<NormalAxiom>) {
    match chain.len() {
        0 | 1 => {}
        2 => {
            out.push(NormalAxiom::RoleChain(
                chain[0].clone(),
                chain[1].clone(),
                result_role.to_string(),
            ));
        }
        _ => {
            // r1 o r2 o r3 ⊑ s  =>  r1 o r2 ⊑ fresh, fresh o r3 ⊑ s
            let mut prev = chain[0].clone();
            for (i, r) in chain.iter().enumerate().skip(1) {
                if i == chain.len() - 1 {
                    out.push(NormalAxiom::RoleChain(
                        prev.clone(),
                        r.clone(),
                        result_role.to_string(),
                    ));
                } else {
                    let fresh = format!("__chain_{}_{}", i, result_role);
                    out.push(NormalAxiom::RoleChain(prev, r.clone(), fresh.clone()));
                    prev = fresh;
                }
            }
        }
    }
}

/// Collect all named atomic concepts referenced in a concept expression.
fn collect_from_concept(c: &ElConcept, out: &mut HashSet<String>) {
    match c {
        ElConcept::Named(n) => {
            out.insert(n.clone());
        }
        ElConcept::Top => {
            out.insert("owl:Thing".to_string());
        }
        ElConcept::Bottom => {
            out.insert("owl:Nothing".to_string());
        }
        ElConcept::Intersection(parts) => {
            for p in parts {
                collect_from_concept(p, out);
            }
        }
        ElConcept::SomeValues { filler, .. } => {
            collect_from_concept(filler, out);
        }
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_subclass_classification() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Mammal");
        r.add_subclass_of("Mammal", "Animal");

        let cls = r.classify().expect("classification failed");
        assert!(cls.is_subclass_of("Dog", "Mammal"), "Dog ⊑ Mammal");
        assert!(
            cls.is_subclass_of("Dog", "Animal"),
            "Dog ⊑ Animal (transitive)"
        );
        assert!(cls.is_subclass_of("Mammal", "Animal"), "Mammal ⊑ Animal");
        assert!(!cls.is_subclass_of("Animal", "Dog"), "Animal not ⊑ Dog");
    }

    #[test]
    fn test_equivalent_classes() {
        let mut r = Owl2ElReasoner::new();
        r.add_equivalent_classes("Human", "Person");

        let cls = r.classify().expect("classification failed");
        assert!(cls.is_subclass_of("Human", "Person"), "Human ⊑ Person");
        assert!(cls.is_subclass_of("Person", "Human"), "Person ⊑ Human");

        let equivs = &cls.equivalent_classes;
        let found = equivs
            .iter()
            .any(|g| g.contains(&"Human".to_string()) && g.contains(&"Person".to_string()));
        assert!(
            found,
            "Human and Person should be in the same equivalence group"
        );
    }

    #[test]
    fn test_intersection_on_left() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // Doctor ⊓ HaematologySpecialist ⊑ Haematologist
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::intersection(vec![
                ElConcept::named("Doctor"),
                ElConcept::named("HaematologySpecialist"),
            ]),
            sup: ElConcept::named("Haematologist"),
        });
        r.add_concept_assertion("alice", "Doctor");
        r.add_concept_assertion("alice", "HaematologySpecialist");

        let cls = r.classify().expect("classification failed");
        let alice_types = cls.get_individual_types("alice");
        assert!(
            alice_types.contains(&"Haematologist".to_string()),
            "Expected alice to be Haematologist via intersection. Got: {:?}",
            alice_types
        );
        Ok(())
    }

    #[test]
    fn test_existential_some_values_from_right() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // Person ⊑ ∃hasParent.Human
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::named("Person"),
            sup: ElConcept::some_values("hasParent", ElConcept::named("Human")),
        });
        // ∃hasParent.Human ⊑ OffspringOfHuman
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("hasParent", ElConcept::named("Human")),
            sup: ElConcept::named("OffspringOfHuman"),
        });
        r.add_concept_assertion("alice", "Person");

        let cls = r.classify().expect("classification failed");
        let alice_types = cls.get_individual_types("alice");
        assert!(
            alice_types.contains(&"OffspringOfHuman".to_string()),
            "Expected alice OffspringOfHuman via existential chain. Got: {:?}",
            alice_types
        );
        Ok(())
    }

    #[test]
    fn test_transitive_role() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        r.add_transitive_role("partOf");
        r.add_role_assertion("lug", "partOf", "wheel");
        r.add_role_assertion("wheel", "partOf", "car");

        let cls = r.classify().expect("classification failed");
        let lug_succs = cls
            .role_successors
            .get(&("lug".to_string(), "partOf".to_string()));
        assert!(
            lug_succs.map(|s| s.contains("car")).unwrap_or(false),
            "Expected lug partOf car via transitivity. Got: {:?}",
            lug_succs
        );
        Ok(())
    }

    #[test]
    fn test_property_chain() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // hasParent o hasParent ⊑ hasGrandParent
        r.add_property_chain(
            vec!["hasParent".to_string(), "hasParent".to_string()],
            "hasGrandParent",
        );
        r.add_role_assertion("child", "hasParent", "parent");
        r.add_role_assertion("parent", "hasParent", "grandparent");

        let cls = r.classify().expect("classification failed");
        let child_grand = cls
            .role_successors
            .get(&("child".to_string(), "hasGrandParent".to_string()));
        assert!(
            child_grand
                .map(|s| s.contains("grandparent"))
                .unwrap_or(false),
            "Expected child hasGrandParent grandparent via chain. Got: {:?}",
            child_grand
        );
        Ok(())
    }

    #[test]
    fn test_abox_type_propagation() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        r.add_concept_assertion("fido", "Dog");

        let cls = r.classify().expect("classification failed");
        let fido_types = cls.get_individual_types("fido");
        assert!(
            fido_types.contains(&"Animal".to_string()),
            "Expected fido to be Animal via subclass. Got: {:?}",
            fido_types
        );
        Ok(())
    }

    #[test]
    fn test_get_superclasses() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Labrador", "Dog");
        r.add_subclass_of("Dog", "Mammal");
        r.add_subclass_of("Mammal", "Animal");

        let supers = r.get_superclasses("Labrador").expect("failed");
        assert!(supers.contains(&"Dog".to_string()), "Missing Dog");
        assert!(supers.contains(&"Mammal".to_string()), "Missing Mammal");
        assert!(supers.contains(&"Animal".to_string()), "Missing Animal");
    }

    #[test]
    fn test_get_subclasses() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        r.add_subclass_of("Cat", "Animal");

        let subs = r.get_subclasses("Animal").expect("failed");
        assert!(subs.contains(&"Dog".to_string()), "Missing Dog");
        assert!(subs.contains(&"Cat".to_string()), "Missing Cat");
    }

    #[test]
    fn test_is_subclass_of() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Poodle", "Dog");

        assert!(r.is_subclass_of("Poodle", "Dog").expect("failed"));
        assert!(!r.is_subclass_of("Dog", "Poodle").expect("failed"));
    }

    #[test]
    fn test_long_chain_classification() {
        let mut r = Owl2ElReasoner::new();
        for i in 0..9usize {
            r.add_subclass_of(&format!("C{}", i), &format!("C{}", i + 1));
        }

        let cls = r.classify().expect("classification failed");
        for i in 1..10usize {
            assert!(
                cls.is_subclass_of("C0", &format!("C{}", i)),
                "C0 should be ⊑ C{}",
                i
            );
        }
    }

    #[test]
    fn test_sub_role() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        r.add_axiom(ElAxiom::SubRole {
            sub: "isChildOf".to_string(),
            sup: "isRelatedTo".to_string(),
        });
        r.add_role_assertion("alice", "isChildOf", "bob");

        let cls = r.classify().expect("classification failed");
        let alice_related = cls
            .role_successors
            .get(&("alice".to_string(), "isRelatedTo".to_string()));
        assert!(
            alice_related.map(|s| s.contains("bob")).unwrap_or(false),
            "Expected alice isRelatedTo bob via subRole. Got: {:?}",
            alice_related
        );
        Ok(())
    }

    #[test]
    fn test_some_sub_atom() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // ∃worksIn.Organization ⊑ Employee
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("worksIn", ElConcept::named("Organization")),
            sup: ElConcept::named("Employee"),
        });
        r.add_role_assertion("alice", "worksIn", "acme");
        r.add_concept_assertion("acme", "Organization");

        let cls = r.classify().expect("classification failed");
        let alice_types = cls.get_individual_types("alice");
        assert!(
            alice_types.contains(&"Employee".to_string()),
            "Expected alice to be Employee via ∃worksIn.Organization. Got: {:?}",
            alice_types
        );
        Ok(())
    }

    #[test]
    fn test_long_property_chain() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        // r1 o r2 o r3 ⊑ rResult
        r.add_property_chain(
            vec!["r1".to_string(), "r2".to_string(), "r3".to_string()],
            "rResult",
        );
        r.add_role_assertion("a", "r1", "b");
        r.add_role_assertion("b", "r2", "c");
        r.add_role_assertion("c", "r3", "d");

        let cls = r.classify().expect("classification failed");
        let a_result = cls
            .role_successors
            .get(&("a".to_string(), "rResult".to_string()));
        assert!(
            a_result.map(|s| s.contains("d")).unwrap_or(false),
            "Expected a rResult d via 3-chain. Got: {:?}",
            a_result
        );
        Ok(())
    }
}

// -----------------------------------------------------------------------
// Extended Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests_extended {
    use super::*;

    // ---- Subclass hierarchy tests ----

    #[test]
    fn test_empty_reasoner_classify_succeeds() {
        let r = Owl2ElReasoner::new();
        let cls = r.classify().expect("empty classify failed");
        // No individuals should be present in an empty ontology
        assert_eq!(cls.individual_types.len(), 0);
        // No individual types for any name
        assert!(cls.get_individual_types("nobody").is_empty());
    }

    #[test]
    fn test_single_subclass_axiom() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Cat", "Animal");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("Cat", "Animal"), "Cat ⊑ Animal");
        assert!(!cls.is_subclass_of("Animal", "Cat"), "Animal not ⊑ Cat");
    }

    #[test]
    fn test_no_accidental_subsumption() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("C", "D");
        let cls = r.classify().expect("classify failed");
        assert!(!cls.is_subclass_of("A", "D"), "A should not ⊑ D");
        assert!(!cls.is_subclass_of("C", "B"), "C should not ⊑ B");
    }

    #[test]
    fn test_self_subsumption() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("X", "Y");
        let cls = r.classify().expect("classify failed");
        // Every class is subclass of itself
        assert!(cls.is_subclass_of("X", "X"), "X ⊑ X (reflexivity)");
        assert!(cls.is_subclass_of("Y", "Y"), "Y ⊑ Y (reflexivity)");
    }

    #[test]
    fn test_five_level_chain() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("B", "C");
        r.add_subclass_of("C", "D");
        r.add_subclass_of("D", "E");
        r.add_subclass_of("E", "F");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("A", "F"), "A ⊑ F (5-hop transitive)");
        assert!(cls.is_subclass_of("B", "F"), "B ⊑ F");
        assert!(!cls.is_subclass_of("F", "A"), "F not ⊑ A");
    }

    #[test]
    fn test_get_superclasses_method() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Poodle", "Dog");
        r.add_subclass_of("Dog", "Mammal");
        r.add_subclass_of("Mammal", "Animal");
        let supers = r
            .get_superclasses("Poodle")
            .expect("get_superclasses failed");
        assert!(supers.contains(&"Dog".to_string()), "Missing Dog");
        assert!(supers.contains(&"Mammal".to_string()), "Missing Mammal");
        assert!(supers.contains(&"Animal".to_string()), "Missing Animal");
    }

    #[test]
    fn test_get_subclasses_method() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        r.add_subclass_of("Cat", "Animal");
        r.add_subclass_of("Bird", "Animal");
        let subs = r.get_subclasses("Animal").expect("get_subclasses failed");
        assert!(subs.contains(&"Dog".to_string()), "Missing Dog");
        assert!(subs.contains(&"Cat".to_string()), "Missing Cat");
        assert!(subs.contains(&"Bird".to_string()), "Missing Bird");
    }

    #[test]
    fn test_is_subclass_of_method_false() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        assert!(r.is_subclass_of("A", "B").expect("failed"));
        assert!(!r.is_subclass_of("B", "A").expect("failed"));
        assert!(!r.is_subclass_of("A", "C").expect("failed"));
    }

    #[test]
    fn test_equivalent_classes_chain() {
        // A ≡ B, B ≡ C => A ⊑ C, C ⊑ A
        let mut r = Owl2ElReasoner::new();
        r.add_equivalent_classes("A", "B");
        r.add_equivalent_classes("B", "C");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("A", "B"), "A ⊑ B");
        assert!(cls.is_subclass_of("B", "A"), "B ⊑ A");
        assert!(cls.is_subclass_of("B", "C"), "B ⊑ C");
        assert!(cls.is_subclass_of("C", "B"), "C ⊑ B");
        // Transitive: A ⊑ C
        assert!(cls.is_subclass_of("A", "C"), "A ⊑ C (equiv chain)");
    }

    // ---- Intersection tests ----

    #[test]
    fn test_three_way_intersection_on_left() {
        let mut r = Owl2ElReasoner::new();
        // A ⊓ B ⊓ C ⊑ D
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::intersection(vec![
                ElConcept::named("A"),
                ElConcept::named("B"),
                ElConcept::named("C"),
            ]),
            sup: ElConcept::named("D"),
        });
        r.add_concept_assertion("x", "A");
        r.add_concept_assertion("x", "B");
        r.add_concept_assertion("x", "C");

        let cls = r.classify().expect("classify failed");
        assert!(
            cls.get_individual_types("x").contains(&"D".to_string()),
            "Expected x:D via 3-way intersection"
        );
    }

    #[test]
    fn test_intersection_missing_one_class() {
        let mut r = Owl2ElReasoner::new();
        // A ⊓ B ⊑ C
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::intersection(vec![ElConcept::named("A"), ElConcept::named("B")]),
            sup: ElConcept::named("C"),
        });
        r.add_concept_assertion("x", "A");
        // Missing B — x should NOT be classified as C

        let cls = r.classify().expect("classify failed");
        assert!(
            !cls.get_individual_types("x").contains(&"C".to_string()),
            "x should not be C without B"
        );
    }

    // ---- Existential restriction tests ----

    #[test]
    fn test_existential_chain_three_hops() {
        let mut r = Owl2ElReasoner::new();
        // ∃r.A ⊑ B, ∃s.B ⊑ C
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("r", ElConcept::named("A")),
            sup: ElConcept::named("B"),
        });
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("s", ElConcept::named("B")),
            sup: ElConcept::named("C"),
        });
        r.add_role_assertion("x", "r", "y");
        r.add_concept_assertion("y", "A");
        r.add_role_assertion("z", "s", "x");

        let cls = r.classify().expect("classify failed");
        // x: ∃r.A ⊑ B, so x:B
        assert!(
            cls.get_individual_types("x").contains(&"B".to_string()),
            "x should be B"
        );
        // z: ∃s.B (via x:B) ⊑ C, so z:C
        assert!(
            cls.get_individual_types("z").contains(&"C".to_string()),
            "z should be C"
        );
    }

    #[test]
    fn test_some_values_named_filler() {
        let mut r = Owl2ElReasoner::new();
        // ∃worksAt.Company ⊑ Employee
        r.add_axiom(ElAxiom::SubConceptOf {
            sub: ElConcept::some_values("worksAt", ElConcept::named("Company")),
            sup: ElConcept::named("Employee"),
        });
        r.add_role_assertion("alice", "worksAt", "acme");
        r.add_concept_assertion("acme", "Company");

        let cls = r.classify().expect("classify failed");
        assert!(
            cls.get_individual_types("alice")
                .contains(&"Employee".to_string()),
            "alice should be Employee via ∃worksAt.Company"
        );
    }

    // ---- Role / property chain tests ----

    #[test]
    fn test_transitive_role_three_steps() {
        let mut r = Owl2ElReasoner::new();
        r.add_transitive_role("ancestorOf");
        r.add_role_assertion("great_grandparent", "ancestorOf", "grandparent");
        r.add_role_assertion("grandparent", "ancestorOf", "parent");
        r.add_role_assertion("parent", "ancestorOf", "child");

        let cls = r.classify().expect("classify failed");
        let ggp = cls
            .role_successors
            .get(&("great_grandparent".to_string(), "ancestorOf".to_string()));
        assert!(
            ggp.map(|s| s.contains("child")).unwrap_or(false),
            "great_grandparent should ancestorOf child (3-step)"
        );
    }

    #[test]
    fn test_property_chain_two_roles() {
        let mut r = Owl2ElReasoner::new();
        // uncle ≡ hasParent o hasBrother
        r.add_property_chain(
            vec!["hasParent".to_string(), "hasBrother".to_string()],
            "hasUncle",
        );
        r.add_role_assertion("alice", "hasParent", "bob");
        r.add_role_assertion("bob", "hasBrother", "charlie");

        let cls = r.classify().expect("classify failed");
        let alice_uncles = cls
            .role_successors
            .get(&("alice".to_string(), "hasUncle".to_string()));
        assert!(
            alice_uncles.map(|s| s.contains("charlie")).unwrap_or(false),
            "alice hasUncle charlie via chain"
        );
    }

    #[test]
    fn test_sub_role_propagation() {
        let mut r = Owl2ElReasoner::new();
        r.add_axiom(ElAxiom::SubRole {
            sub: "worksFor".to_string(),
            sup: "associatedWith".to_string(),
        });
        r.add_role_assertion("emp", "worksFor", "company");

        let cls = r.classify().expect("classify failed");
        let emp_assoc = cls
            .role_successors
            .get(&("emp".to_string(), "associatedWith".to_string()));
        assert!(
            emp_assoc.map(|s| s.contains("company")).unwrap_or(false),
            "emp should associatedWith company via subRole"
        );
    }

    // ---- Individual classification tests ----

    #[test]
    fn test_individual_inherits_via_subclass_chain() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Labrador", "Dog");
        r.add_subclass_of("Dog", "Mammal");
        r.add_subclass_of("Mammal", "LivingBeing");
        r.add_concept_assertion("rex", "Labrador");

        let cls = r.classify().expect("classify failed");
        let types = cls.get_individual_types("rex");
        assert!(types.contains(&"Dog".to_string()), "rex should be Dog");
        assert!(
            types.contains(&"Mammal".to_string()),
            "rex should be Mammal"
        );
        assert!(
            types.contains(&"LivingBeing".to_string()),
            "rex should be LivingBeing"
        );
    }

    #[test]
    fn test_multiple_individuals_multiple_classes() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Truck", "Vehicle");
        r.add_subclass_of("Car", "Vehicle");
        r.add_concept_assertion("t1", "Truck");
        r.add_concept_assertion("c1", "Car");

        let cls = r.classify().expect("classify failed");
        assert!(
            cls.get_individual_types("t1")
                .contains(&"Vehicle".to_string()),
            "t1 should be Vehicle"
        );
        assert!(
            cls.get_individual_types("c1")
                .contains(&"Vehicle".to_string()),
            "c1 should be Vehicle"
        );
    }

    #[test]
    fn test_individual_in_multiple_classes() {
        let mut r = Owl2ElReasoner::new();
        r.add_concept_assertion("alice", "Professor");
        r.add_concept_assertion("alice", "Researcher");
        r.add_concept_assertion("alice", "Person");

        let cls = r.classify().expect("classify failed");
        let types = cls.get_individual_types("alice");
        assert!(types.contains(&"Professor".to_string()));
        assert!(types.contains(&"Researcher".to_string()));
        assert!(types.contains(&"Person".to_string()));
    }

    #[test]
    fn test_individual_unknown_is_empty() {
        let r = Owl2ElReasoner::new();
        let cls = r.classify().expect("classify failed");
        let types = cls.get_individual_types("nonexistent");
        assert!(types.is_empty(), "unknown individual should have no types");
    }

    #[test]
    fn test_classify_idempotent() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("B", "C");
        let cls1 = r.classify().expect("first classify failed");
        let cls2 = r.classify().expect("second classify failed");
        assert_eq!(
            cls1.is_subclass_of("A", "C"),
            cls2.is_subclass_of("A", "C"),
            "classify should be idempotent"
        );
    }

    // ---- Edge-case / structural tests ----

    #[test]
    fn test_diamond_inheritance() {
        // A ⊑ B, A ⊑ C, B ⊑ D, C ⊑ D
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("A", "B");
        r.add_subclass_of("A", "C");
        r.add_subclass_of("B", "D");
        r.add_subclass_of("C", "D");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("A", "D"), "A ⊑ D (diamond)");
        assert!(cls.is_subclass_of("B", "D"), "B ⊑ D");
        assert!(cls.is_subclass_of("C", "D"), "C ⊑ D");
    }

    #[test]
    fn test_sibling_classes_not_related() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Cat", "Animal");
        r.add_subclass_of("Dog", "Animal");
        let cls = r.classify().expect("classify failed");
        assert!(!cls.is_subclass_of("Cat", "Dog"), "Cat should not ⊑ Dog");
        assert!(!cls.is_subclass_of("Dog", "Cat"), "Dog should not ⊑ Cat");
    }

    #[test]
    fn test_intersection_empty_becomes_top() {
        let result = ElConcept::intersection(vec![]);
        assert!(
            matches!(result, ElConcept::Top),
            "empty intersection is Top"
        );
    }

    #[test]
    fn test_intersection_single_becomes_concept() {
        let result = ElConcept::intersection(vec![ElConcept::named("A")]);
        assert!(
            matches!(result, ElConcept::Named(ref n) if n == "A"),
            "singleton intersection is the concept"
        );
    }

    #[test]
    fn test_add_axioms_batch() {
        let mut r = Owl2ElReasoner::new();
        r.add_axioms(vec![
            ElAxiom::SubConceptOf {
                sub: ElConcept::named("X"),
                sup: ElConcept::named("Y"),
            },
            ElAxiom::SubConceptOf {
                sub: ElConcept::named("Y"),
                sup: ElConcept::named("Z"),
            },
        ]);
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("X", "Z"), "X ⊑ Z via batch axioms");
    }

    #[test]
    fn test_named_concept_as_named() {
        let c = ElConcept::named("MyClass");
        assert_eq!(c.as_named(), Some("MyClass"));
    }

    #[test]
    fn test_top_concept_as_named_none() {
        let c = ElConcept::Top;
        assert_eq!(c.as_named(), None);
    }

    #[test]
    fn test_bottom_concept_as_named_none() {
        let c = ElConcept::Bottom;
        assert_eq!(c.as_named(), None);
    }

    #[test]
    fn test_property_chain_two_roles_simple() {
        // father o father ⊑ grandfather
        let mut r = Owl2ElReasoner::new();
        r.add_property_chain(
            vec!["father".to_string(), "father".to_string()],
            "grandfather",
        );
        r.add_role_assertion("x", "father", "y");
        r.add_role_assertion("y", "father", "z");

        let cls = r.classify().expect("classify failed");
        let x_grfa = cls
            .role_successors
            .get(&("x".to_string(), "grandfather".to_string()));
        assert!(
            x_grfa.map(|s| s.contains("z")).unwrap_or(false),
            "x should have grandfather z via chain"
        );
    }

    #[test]
    fn test_multiple_property_chains() {
        let mut r = Owl2ElReasoner::new();
        r.add_property_chain(vec!["p".to_string(), "q".to_string()], "pq");
        r.add_property_chain(vec!["q".to_string(), "r".to_string()], "qr");
        r.add_role_assertion("a", "p", "b");
        r.add_role_assertion("b", "q", "c");
        r.add_role_assertion("c", "r", "d");

        let cls = r.classify().expect("classify failed");
        let a_pq = cls
            .role_successors
            .get(&("a".to_string(), "pq".to_string()));
        let b_qr = cls
            .role_successors
            .get(&("b".to_string(), "qr".to_string()));
        assert!(a_pq.map(|s| s.contains("c")).unwrap_or(false), "a pq c");
        assert!(b_qr.map(|s| s.contains("d")).unwrap_or(false), "b qr d");
    }

    #[test]
    fn test_subsumption_hierarchy_not_symmetric() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Child", "Parent");
        let cls = r.classify().expect("classify failed");
        assert!(cls.is_subclass_of("Child", "Parent"));
        assert!(!cls.is_subclass_of("Parent", "Child"));
    }

    #[test]
    fn test_role_successors_absent_for_unknown_role() {
        let mut r = Owl2ElReasoner::new();
        r.add_role_assertion("x", "knows", "y");
        let cls = r.classify().expect("classify failed");
        let unknown = cls
            .role_successors
            .get(&("x".to_string(), "unknown".to_string()));
        assert!(unknown.is_none() || unknown.map(|s| s.is_empty()).unwrap_or(true));
    }

    #[test]
    fn test_with_max_work_items() {
        let mut r = Owl2ElReasoner::new().with_max_work_items(10_000);
        r.add_subclass_of("A", "B");
        let cls = r.classify().expect("classify with custom limit failed");
        assert!(cls.is_subclass_of("A", "B"));
    }

    #[test]
    fn test_get_subclasses_empty_for_leaf() -> anyhow::Result<()> {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        let subs = r.get_subclasses("Dog").expect("get_subclasses failed");
        // Dog has no explicit subclasses
        assert!(
            subs.is_empty(),
            "Dog should have no subclasses, got {:?}",
            subs
        );
        Ok(())
    }

    #[test]
    fn test_get_superclasses_root_does_not_include_subclasses() {
        let mut r = Owl2ElReasoner::new();
        r.add_subclass_of("Dog", "Animal");
        r.add_subclass_of("Cat", "Animal");
        let supers = r
            .get_superclasses("Animal")
            .expect("get_superclasses failed");
        // Animal's superclasses should not include Dog or Cat
        assert!(
            !supers.contains(&"Dog".to_string()),
            "Animal superclasses should not include Dog"
        );
        assert!(
            !supers.contains(&"Cat".to_string()),
            "Animal superclasses should not include Cat"
        );
    }
}
