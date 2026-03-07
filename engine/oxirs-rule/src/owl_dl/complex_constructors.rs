//! Complex class constructor rules for OWL 2 DL (80% milestone).
//!
//! Implements rules 14–17 in the materialization fixpoint:
//!
//! - **Rule 14** `ObjectIntersectionOf`: x ∈ C1 ∩ C2 ↔ x ∈ C1 ∧ x ∈ C2.
//! - **Rule 15** `FunctionalProperty`: x P a ∧ x P b → a owl:sameAs b.
//! - **Rule 16** `InverseFunctionalProperty`: a P x ∧ b P x → a owl:sameAs b.
//! - **Rule 17** `HasKey`: individuals of a class sharing all key-property values
//!   are identical (owl:sameAs).
//!
//! Plus post-loop inconsistency checks:
//! - `ObjectComplementOf`: x ∈ C ∧ x ∈ ¬C → inconsistency.
//! - `DisjointUnionOf` pairwise disjointness violations.
//!
//! The `expand_disjoint_unions` method runs once before the fixpoint loop to
//! materialise the TBox-level consequences (subclass + disjointness pairs)
//! that follow from `owl:disjointUnionOf` declarations.

use std::collections::HashMap;
use std::collections::HashSet;

use super::{mk_triple, vocab, Owl2DLReasoner, RuleFirings, Triple};

impl Owl2DLReasoner {
    // ── Pre-loop TBox expansion ───────────────────────────────────────────────

    /// Expand `owl:disjointUnionOf` axioms into their TBox consequences:
    ///
    /// For `A owl:disjointUnionOf (B C D …)`:
    /// - Each operand is a subclass of `A`.
    /// - Each pair of operands is mutually disjoint.
    ///
    /// This is called once before the fixpoint loop.
    pub(crate) fn expand_disjoint_unions(&mut self) {
        // Clone to avoid borrowing self mutably while iterating
        let unions: Vec<_> = self.disjoint_unions.clone();
        for du in &unions {
            // Each operand ⊑ parent class
            for operand in &du.operands {
                self.subclass_of
                    .insert((operand.clone(), du.class_iri.clone()));
                self.abox
                    .insert(mk_triple(operand, vocab::RDFS_SUBCLASS_OF, &du.class_iri));
            }

            // All pairs of operands are disjoint
            let ops = &du.operands;
            for i in 0..ops.len() {
                for j in (i + 1)..ops.len() {
                    self.disjoint_classes
                        .insert((ops[i].clone(), ops[j].clone()));
                    self.disjoint_classes
                        .insert((ops[j].clone(), ops[i].clone()));
                    self.abox
                        .insert(mk_triple(&ops[i], vocab::OWL_DISJOINT_WITH, &ops[j]));
                    self.abox
                        .insert(mk_triple(&ops[j], vocab::OWL_DISJOINT_WITH, &ops[i]));
                }
            }
        }
    }

    // ── Rule 14: ObjectIntersectionOf ────────────────────────────────────────

    /// Rule 14a (forward): if x ∈ all operands of an intersection class → x ∈ class_iri.
    /// Rule 14b (backward): if x ∈ class_iri (intersection) → x ∈ each operand.
    pub(crate) fn apply_intersection_of(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        // Collect per-individual type set for fast membership tests
        let mut ind_types: HashMap<&str, HashSet<&str>> = HashMap::new();
        for (s, p, o) in triples {
            if p == vocab::RDF_TYPE {
                ind_types.entry(s.as_str()).or_default().insert(o.as_str());
            }
        }

        for ix in &self.intersection_of_classes {
            // --- 14a: forward — all operands present → intersection class -------
            for (ind, types) in &ind_types {
                let all_present = ix.operands.iter().all(|op| types.contains(op.as_str()));

                if all_present {
                    let t = mk_triple(ind, vocab::RDF_TYPE, &ix.class_iri);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.intersection_of += 1;
                    }
                }
            }

            // --- 14b: backward — intersection class → each operand --------------
            for (ind, types) in &ind_types {
                if types.contains(ix.class_iri.as_str()) {
                    for operand in &ix.operands {
                        let t = mk_triple(ind, vocab::RDF_TYPE, operand);
                        if !triples.contains(&t) {
                            new_triples.insert(t);
                            firings.intersection_of += 1;
                        }
                    }
                }
            }
        }
    }

    // ── Rule 15: FunctionalProperty ──────────────────────────────────────────

    /// Rule 15: if P is functional and `x P a` and `x P b` then `a owl:sameAs b`.
    ///
    /// Per the OWL 2 semantics, a functional property allows at most one distinct
    /// value per subject.  When two values are found they must be identical.
    pub(crate) fn apply_functional_property(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        let functional_props: Vec<String> = self
            .property_chars
            .iter()
            .filter(|(_, c)| c.is_functional)
            .map(|(p, _)| p.clone())
            .collect();

        for prop in &functional_props {
            // Group objects by subject: subject → [objects]
            let mut subj_to_objs: HashMap<&str, Vec<&str>> = HashMap::new();
            for (s, p, o) in triples {
                if p == prop {
                    subj_to_objs.entry(s.as_str()).or_default().push(o.as_str());
                }
            }

            for objs in subj_to_objs.values() {
                if objs.len() < 2 {
                    continue;
                }
                // All pairs of objects must be sameAs
                for i in 0..objs.len() {
                    for j in (i + 1)..objs.len() {
                        let a = objs[i];
                        let b = objs[j];
                        if a == b {
                            continue;
                        }
                        let t1 = mk_triple(a, vocab::OWL_SAME_AS, b);
                        let t2 = mk_triple(b, vocab::OWL_SAME_AS, a);
                        if !triples.contains(&t1) {
                            new_triples.insert(t1);
                            firings.functional_property += 1;
                        }
                        if !triples.contains(&t2) {
                            new_triples.insert(t2);
                            firings.functional_property += 1;
                        }
                    }
                }
            }
        }
    }

    // ── Rule 16: InverseFunctionalProperty ───────────────────────────────────

    /// Rule 16: if P is inverse-functional and `a P x` and `b P x` then `a owl:sameAs b`.
    ///
    /// An inverse-functional property uniquely identifies the subject: no two distinct
    /// subjects may share the same object for that property.
    pub(crate) fn apply_inverse_functional_property(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        let inv_func_props: Vec<String> = self
            .property_chars
            .iter()
            .filter(|(_, c)| c.is_inverse_functional)
            .map(|(p, _)| p.clone())
            .collect();

        for prop in &inv_func_props {
            // Group subjects by object: object → [subjects]
            let mut obj_to_subjs: HashMap<&str, Vec<&str>> = HashMap::new();
            for (s, p, o) in triples {
                if p == prop {
                    obj_to_subjs.entry(o.as_str()).or_default().push(s.as_str());
                }
            }

            for subjs in obj_to_subjs.values() {
                if subjs.len() < 2 {
                    continue;
                }
                // All pairs of subjects must be sameAs
                for i in 0..subjs.len() {
                    for j in (i + 1)..subjs.len() {
                        let a = subjs[i];
                        let b = subjs[j];
                        if a == b {
                            continue;
                        }
                        let t1 = mk_triple(a, vocab::OWL_SAME_AS, b);
                        let t2 = mk_triple(b, vocab::OWL_SAME_AS, a);
                        if !triples.contains(&t1) {
                            new_triples.insert(t1);
                            firings.inverse_functional_property += 1;
                        }
                        if !triples.contains(&t2) {
                            new_triples.insert(t2);
                            firings.inverse_functional_property += 1;
                        }
                    }
                }
            }
        }
    }

    // ── Rule 17: HasKey ───────────────────────────────────────────────────────

    /// Rule 17: `owl:hasKey` unique-key entailment.
    ///
    /// For a `HasKeyAxiom { class_iri, key_properties }`:
    /// If two individuals `a` and `b` are both of type `class_iri`, and for every
    /// property `p` in `key_properties` there exists a value `v` such that both
    /// `a p v` and `b p v`, then `a owl:sameAs b`.
    ///
    /// Implementation collects a key-tuple per individual (only those with values
    /// for ALL key properties), then groups by identical key-tuple to find collisions.
    pub(crate) fn apply_has_key(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for hk in &self.has_key_axioms {
            if hk.key_properties.is_empty() {
                continue;
            }

            // Collect members of class_iri
            let members: Vec<&str> = triples
                .iter()
                .filter(|(_, p, o)| p == vocab::RDF_TYPE && o == &hk.class_iri)
                .map(|(s, _, _)| s.as_str())
                .collect();

            if members.len() < 2 {
                continue;
            }

            // For each member, build a canonical key: Vec of sorted property values
            // (one value per key property; skip member if any property is unbound)
            let mut key_to_individuals: HashMap<Vec<String>, Vec<&str>> = HashMap::new();

            'member: for &member in &members {
                let mut key_parts: Vec<String> = Vec::with_capacity(hk.key_properties.len());
                for kp in &hk.key_properties {
                    // Collect all values for this property from this member, sort for canonicality
                    let mut values: Vec<&str> = triples
                        .iter()
                        .filter(|(s, p, _)| s.as_str() == member && p == kp)
                        .map(|(_, _, o)| o.as_str())
                        .collect();

                    if values.is_empty() {
                        // Member has no value for this key property — cannot participate
                        continue 'member;
                    }
                    values.sort_unstable();
                    key_parts.push(values.join("|"));
                }
                key_to_individuals
                    .entry(key_parts)
                    .or_default()
                    .push(member);
            }

            // Any key-group with 2+ members → they must be sameAs
            for group in key_to_individuals.values() {
                if group.len() < 2 {
                    continue;
                }
                for i in 0..group.len() {
                    for j in (i + 1)..group.len() {
                        let a = group[i];
                        let b = group[j];
                        if a == b {
                            continue;
                        }
                        let t1 = mk_triple(a, vocab::OWL_SAME_AS, b);
                        let t2 = mk_triple(b, vocab::OWL_SAME_AS, a);
                        if !triples.contains(&t1) {
                            new_triples.insert(t1);
                            firings.has_key += 1;
                        }
                        if !triples.contains(&t2) {
                            new_triples.insert(t2);
                            firings.has_key += 1;
                        }
                    }
                }
            }
        }
    }

    // ── Post-loop inconsistency checks ────────────────────────────────────────

    /// Check for `ObjectComplementOf` violations:
    /// if `x rdf:type class_iri` and `x rdf:type base_class` both hold,
    /// that is a direct contradiction (C ∧ ¬C).
    pub(crate) fn check_complement_of_violations(&mut self, firings: &mut RuleFirings) {
        let complements: Vec<_> = self.complement_of_classes.clone();
        for comp in &complements {
            // Individuals that are asserted/inferred to be of type class_iri (¬base)
            let complement_members: Vec<String> = self
                .abox
                .iter()
                .filter(|(_, p, o)| p == vocab::RDF_TYPE && o == &comp.class_iri)
                .map(|(s, _, _)| s.clone())
                .collect();

            for ind in complement_members {
                // Contradiction: also a member of base_class
                if self
                    .abox
                    .contains(&mk_triple(&ind, vocab::RDF_TYPE, &comp.base_class))
                {
                    let msg = format!(
                        "ComplementOf violation: {ind} is both {} and {} (complement contradiction)",
                        comp.base_class, comp.class_iri
                    );
                    if !self.inconsistencies.contains(&msg) {
                        self.inconsistencies.push(msg);
                        firings.complement_of += 1;
                    }
                }
            }
        }
    }

    /// Check for `DisjointUnionOf` operand-pair disjointness violations.
    ///
    /// This is a superset of the standard disjoint-class check that fires
    /// specifically for pairs derived from `owl:disjointUnionOf`.
    /// (The standard `check_disjoint_violations` covers the same pairs once
    /// `expand_disjoint_unions` has run, but we report a more informative message here.)
    pub(crate) fn check_disjoint_union_violations(&mut self) {
        let unions: Vec<_> = self.disjoint_unions.clone();
        for du in &unions {
            let ops = &du.operands;
            for i in 0..ops.len() {
                for j in (i + 1)..ops.len() {
                    let c1 = &ops[i];
                    let c2 = &ops[j];

                    let c1_members: Vec<String> = self
                        .abox
                        .iter()
                        .filter(|(_, p, o)| p == vocab::RDF_TYPE && o == c1)
                        .map(|(s, _, _)| s.clone())
                        .collect();

                    for ind in c1_members {
                        if self.abox.contains(&mk_triple(&ind, vocab::RDF_TYPE, c2)) {
                            let msg = format!(
                                "DisjointUnionOf violation: {ind} is both {c1} and {c2} \
                                 (disjoint union of {})",
                                du.class_iri
                            );
                            if !self.inconsistencies.contains(&msg) {
                                self.inconsistencies.push(msg);
                            }
                        }
                    }
                }
            }
        }
    }
}
