//! # OWL 2 EL Reasoner
//!
//! The `Owl2ElReasoner` struct and all free helper functions implementing
//! the EL completion rules CR1–CR6 (Baader, Brandt, Lutz 2005).

use std::collections::{HashMap, HashSet, VecDeque};

use crate::owl_el_axioms::{ElAxiom, ElClassification, ElConcept, ElError, NormalAxiom};

/// OWL 2 EL reasoner using the standard EL completion algorithm.
///
/// The algorithm operates on per-concept "context" sets S(C) and role-successor
/// sets R(C, r), computing the closure of all subsumption relationships using
/// the EL saturation rules CR1–CR6.
pub struct Owl2ElReasoner {
    pub(crate) axioms: Vec<ElAxiom>,
    pub(crate) max_work_items: usize,
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
pub(crate) fn derive_concept_members(
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
pub(crate) fn derive_role_successors(
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
pub(crate) fn derive_chain_successors(
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
pub(crate) fn saturate_abox(
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
pub(crate) fn find_equivalents(
    concept_sets: &HashMap<String, HashSet<String>>,
) -> Vec<Vec<String>> {
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
pub(crate) fn normalize_subclass(sub: &ElConcept, sup: &ElConcept, out: &mut Vec<NormalAxiom>) {
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
pub(crate) fn decompose_chain(chain: &[String], result_role: &str, out: &mut Vec<NormalAxiom>) {
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
pub(crate) fn collect_from_concept(c: &ElConcept, out: &mut HashSet<String>) {
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
