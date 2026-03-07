//! Property hierarchy and advanced role reasoning rules for OWL 2 DL (90% milestone).
//!
//! Implements rules 18–21 in the materialization fixpoint:
//!
//! - **Rule 18** `SubObjectPropertyOf` / `SubDataPropertyOf`:
//!   if `x P1 y` and `P1 ⊑ P2`, then `x P2 y`.
//! - **Rule 19** `EquivalentObjectProperties`:
//!   `P1 ≡ P2` — bidirectional property subsumption.
//! - **Rule 20** `ReflexiveObjectProperty`:
//!   `x P x` for every individual in the ABox when `P` is declared reflexive.
//! - **Rule 21** `hasSelf` (local-reflexivity restriction):
//!   `x P x ↔ x ∈ RestrictionClass`.
//!
//! Post-loop inconsistency checks:
//! - **DisjointObjectProperties**: `x P1 y ∧ x P2 y` when `P1 owl:propertyDisjointWith P2`.
//! - **NegativePropertyAssertion**: explicit non-assertion contradicted by an ABox triple.

use std::collections::HashSet;

use super::{mk_triple, vocab, Owl2DLReasoner, RuleFirings, Triple};

impl Owl2DLReasoner {
    // ── Rule 18: SubObjectPropertyOf / SubDataPropertyOf ──────────────────────

    /// Rule 18: Property subsumption — if `x P1 y` and `P1 ⊑ P2` then `x P2 y`.
    ///
    /// Covers both object-property and data-property hierarchies; the combined
    /// index `sub_object_property_of ∪ sub_data_property_of` is scanned.
    /// The transitive closure has already been pre-computed by
    /// `close_sub_property_hierarchy()` before the first iteration, so a single
    /// linear scan per triple suffices here.
    pub(crate) fn apply_sub_property_of(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        // Collect ALL subject → object pairs per property for fast lookup.
        // We iterate sub_object_property_of and sub_data_property_of together.
        for (sub_prop, sup_prop) in self
            .sub_object_property_of
            .iter()
            .chain(self.sub_data_property_of.iter())
        {
            let is_data = self
                .sub_data_property_of
                .contains(&(sub_prop.clone(), sup_prop.clone()));

            for (s, p, o) in triples {
                if p != sub_prop {
                    continue;
                }
                // Skip meta-triples (rdf:type, rdfs:subPropertyOf) — these are
                // schema triples, not ABox property assertions between individuals.
                if p == vocab::RDF_TYPE
                    || p == vocab::RDFS_SUBPROPERTY_OF
                    || p == vocab::RDFS_SUBCLASS_OF
                {
                    continue;
                }
                let t = mk_triple(s, sup_prop, o);
                if !triples.contains(&t) {
                    new_triples.insert(t);
                    if is_data {
                        firings.sub_data_property += 1;
                    } else {
                        firings.sub_object_property += 1;
                    }
                }
            }
        }
    }

    // ── Rule 19: EquivalentObjectProperties ───────────────────────────────────

    /// Rule 19: Equivalent property expansion — if `P1 ≡ P2` and `x P1 y` then `x P2 y`
    /// (and vice versa — both directions stored in `equivalent_properties`).
    pub(crate) fn apply_equivalent_properties(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for (p1, p2) in &self.equivalent_properties {
            for (s, p, o) in triples {
                if p != p1 {
                    continue;
                }
                // Skip meta-triples
                if p == vocab::RDF_TYPE
                    || p == vocab::RDFS_SUBPROPERTY_OF
                    || p == vocab::RDFS_SUBCLASS_OF
                    || p == vocab::OWL_EQUIVALENT_PROPERTY
                {
                    continue;
                }
                let t = mk_triple(s, p2, o);
                if !triples.contains(&t) {
                    new_triples.insert(t);
                    firings.equivalent_properties += 1;
                }
            }
        }
    }

    // ── Rule 20: ReflexiveObjectProperty ─────────────────────────────────────

    /// Rule 20: Self-loop entailment for `owl:ReflexiveProperty`.
    ///
    /// Per the OWL 2 direct semantics, if `P` is reflexive, every individual in
    /// the *interpretation domain* satisfies `x P x`.  In the ABox-only setting
    /// we approximate the domain as all individuals mentioned anywhere in the ABox
    /// (either as the subject of a triple or as the object of an `rdf:type` triple).
    pub(crate) fn apply_reflexive_property(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        let reflexive_props: Vec<&String> = self
            .property_chars
            .iter()
            .filter(|(_, c)| c.is_reflexive)
            .map(|(p, _)| p)
            .collect();

        if reflexive_props.is_empty() {
            return;
        }

        // Collect all individuals: any subject in the ABox, plus objects of
        // rdf:type triples (the typed individuals).
        let mut individuals: HashSet<&str> = HashSet::new();
        for (s, p, o) in triples {
            individuals.insert(s.as_str());
            if p == vocab::RDF_TYPE {
                // The class IRI is not an individual — only the subject is.
                let _ = o; // intentionally not added
            }
            // Objects of non-type triples could also be individuals if they
            // appear elsewhere as subjects; they are collected via the subject pass.
        }
        // Also include objects of rdf:type triples' subjects — already covered by subject pass.
        // Include objects of any non-schema property triple as potential individuals:
        for (_, p, o) in triples {
            if p != vocab::RDFS_SUBCLASS_OF
                && p != vocab::RDFS_SUBPROPERTY_OF
                && p != vocab::OWL_EQUIVALENT_CLASS
                && p != vocab::OWL_EQUIVALENT_PROPERTY
                && p != vocab::OWL_INVERSE_OF
                && p != vocab::RDFS_DOMAIN
                && p != vocab::RDFS_RANGE
                && p != vocab::OWL_PROPERTY_DISJOINT_WITH
                && p != vocab::OWL_DISJOINT_WITH
            {
                individuals.insert(o.as_str());
            }
        }

        for prop in &reflexive_props {
            for ind in &individuals {
                let t = mk_triple(ind, prop, ind);
                if !triples.contains(&t) {
                    new_triples.insert(t);
                    firings.reflexive_self += 1;
                }
            }
        }
    }

    // ── Rule 21: hasSelf (local reflexivity) ──────────────────────────────────

    /// Rule 21: `hasSelf` restriction — local reflexivity class membership.
    ///
    /// *Forward*: if `x property x` then `x rdf:type restriction_class`.
    /// *Backward*: if `x rdf:type restriction_class` then `x property x`.
    pub(crate) fn apply_has_self(
        &self,
        triples: &HashSet<Triple>,
        new_triples: &mut HashSet<Triple>,
        firings: &mut RuleFirings,
    ) {
        for hs in &self.has_self_restrictions {
            // --- Forward direction -------------------------------------------
            // Collect all (s, p, o) where p == hs.property and s == o
            for (s, p, o) in triples {
                if p == &hs.property && s == o {
                    let t = mk_triple(s, vocab::RDF_TYPE, &hs.restriction_class);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.has_self += 1;
                    }
                }
            }

            // --- Backward direction ------------------------------------------
            // For each individual typed as restriction_class, infer the self-loop.
            for (s, p, o) in triples {
                if p == vocab::RDF_TYPE && o == &hs.restriction_class {
                    let t = mk_triple(s, &hs.property, s);
                    if !triples.contains(&t) {
                        new_triples.insert(t);
                        firings.has_self += 1;
                    }
                }
            }
        }
    }

    // ── Post-loop inconsistency checks ────────────────────────────────────────

    /// Check for `owl:propertyDisjointWith` violations.
    ///
    /// For each disjoint pair `(P1, P2)`: if `x P1 y` and `x P2 y` both hold
    /// in the fully-materialised ABox, that is an inconsistency.
    ///
    /// We process canonical pairs only (avoid reporting each violation twice by
    /// checking only the lexicographically-first ordering of each pair).
    pub(crate) fn check_disjoint_property_violations(&mut self) {
        // Build a deduplicated set of canonical disjoint pairs to avoid O(2n) reporting.
        let canonical_pairs: HashSet<(String, String)> = self
            .disjoint_properties
            .iter()
            .filter(|(p1, p2)| p1 < p2)
            .cloned()
            .collect();

        for (p1, p2) in &canonical_pairs {
            // Collect all (s, o) pairs for p1
            let p1_pairs: Vec<(String, String)> = self
                .abox
                .iter()
                .filter(|(_, p, _)| p == p1)
                .map(|(s, _, o)| (s.clone(), o.clone()))
                .collect();

            for (s, o) in &p1_pairs {
                if self.abox.contains(&mk_triple(s, p2, o)) {
                    let msg = format!(
                        "DisjointProperties violation: {s} {p1} {o} AND {s} {p2} {o} \
                         (properties are declared disjoint)"
                    );
                    if !self.inconsistencies.contains(&msg) {
                        self.inconsistencies.push(msg);
                    }
                }
            }
        }
    }

    /// Check for `owl:NegativePropertyAssertion` violations.
    ///
    /// A `NegativeObjectPropertyAssertion(source, property, target)` declares that
    /// the triple `(source, property, target)` must NOT hold.  If it exists in the
    /// fully-materialised ABox, the ABox is inconsistent.
    ///
    /// Similarly for `NegativeDataPropertyAssertion`.
    pub(crate) fn check_negative_property_assertion_violations(
        &mut self,
        firings: &mut RuleFirings,
    ) {
        // Object property assertions
        let obj_negatives: Vec<_> = self.negative_object_assertions.clone();
        for npa in &obj_negatives {
            if self.abox.contains(&mk_triple(
                &npa.source_individual,
                &npa.assertion_property,
                &npa.target_individual,
            )) {
                let msg = format!(
                    "NegativePropertyAssertion violation: {} {} {} is asserted \
                     but declared as a negative object property assertion",
                    npa.source_individual, npa.assertion_property, npa.target_individual
                );
                if !self.inconsistencies.contains(&msg) {
                    self.inconsistencies.push(msg);
                    firings.negative_property_assertion += 1;
                }
            }
        }

        // Data property assertions
        let data_negatives: Vec<_> = self.negative_data_assertions.clone();
        for npa in &data_negatives {
            if self.abox.contains(&mk_triple(
                &npa.source_individual,
                &npa.assertion_property,
                &npa.target_value,
            )) {
                let msg = format!(
                    "NegativeDataPropertyAssertion violation: {} {} {} is asserted \
                     but declared as a negative data property assertion",
                    npa.source_individual, npa.assertion_property, npa.target_value
                );
                if !self.inconsistencies.contains(&msg) {
                    self.inconsistencies.push(msg);
                    firings.negative_property_assertion += 1;
                }
            }
        }
    }
}
