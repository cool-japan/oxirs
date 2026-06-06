//! OWL 2 RL Entailment Engine — implements a subset of the W3C OWL 2 RL profile
//! rules via the [`EntailmentEngine`] trait.
//!
//! OWL 2 RL is a tractable profile of OWL 2 designed for scalable rule-based
//! forward chaining. This module implements the key rules from the W3C OWL 2
//! Profiles specification (https://www.w3.org/TR/owl2-profiles/#Reasoning_in_OWL_2_RL_and_RDF_Graphs_using_Rules).
//!
//! # Rules implemented
//!
//! | Rule        | Description                                          |
//! |-------------|------------------------------------------------------|
//! | cls-svf1    | someValuesFrom restriction membership                |
//! | cls-avf     | allValuesFrom restriction propagation                |
//! | cls-hv1     | hasValue type entailment                             |
//! | cls-hv2     | hasValue instance matching                           |
//! | cls-int1    | intersectionOf membership (N-ary)                    |
//! | cls-int2    | intersectionOf decomposition                         |
//! | prp-symp    | SymmetricProperty inversion                          |
//! | prp-trp     | TransitiveProperty chaining                          |
//! | prp-eqp1    | equivalentProperty forward                           |
//! | prp-eqp2    | equivalentProperty backward                          |
//! | cax-sco     | subClassOf type propagation                          |
//! | cax-eqc1    | equivalentClass forward                              |
//! | cax-eqc2    | equivalentClass backward                             |
//!
//! # References
//!
//! - <https://www.w3.org/TR/owl2-profiles/#Reasoning_in_OWL_2_RL_and_RDF_Graphs_using_Rules>
//! - <https://www.w3.org/TR/sparql11-entailment/>

use super::{EntailmentEngine, EntailmentError, Triple, TripleStore};
use crate::entailment::rdf_entailment::{rdfs_iri, rdf_iri};

// ── OWL vocabulary helpers ────────────────────────────────────────────────────

const OWL_NS: &str = "http://www.w3.org/2002/07/owl#";

#[inline]
fn owl_iri(local: &str) -> String {
    format!("{OWL_NS}{local}")
}

// ── OwlRlRuleSet ─────────────────────────────────────────────────────────────

/// Configuration flags to enable or disable individual OWL 2 RL rule groups.
///
/// By default all rule groups are enabled.
#[derive(Debug, Clone)]
pub struct OwlRlRuleSet {
    /// Enable class restriction rules (cls-svf1, cls-avf, cls-hv1/2, cls-int1/2)
    pub class_restrictions: bool,
    /// Enable property characteristic rules (prp-symp, prp-trp)
    pub property_characteristics: bool,
    /// Enable equivalent-property rules (prp-eqp1/2)
    pub equivalent_properties: bool,
    /// Enable class axiom rules (cax-sco, cax-eqc1/2)
    pub class_axioms: bool,
}

impl Default for OwlRlRuleSet {
    fn default() -> Self {
        Self {
            class_restrictions: true,
            property_characteristics: true,
            equivalent_properties: true,
            class_axioms: true,
        }
    }
}

// ── OwlRlEntailmentEngine ─────────────────────────────────────────────────────

/// An [`EntailmentEngine`] that applies the W3C OWL 2 RL profile rules.
///
/// This engine operates on the flat [`TripleStore`] API and performs a single
/// forward-chaining pass. To reach the full OWL 2 RL closure, the caller must
/// iterate via [`crate::entailment::ClosureGraph`] (or [`crate::entailment::EntailmentGraph`])
/// until no new triples are produced.
///
/// The engine also includes RDFS entailment as a subset (cax-sco is equivalent
/// to rdfs9; rdfs rules are applied first so that OWL rules can build on them).
#[derive(Debug, Clone)]
pub struct OwlRlEntailmentEngine {
    /// Which rule groups are active
    pub rule_set: OwlRlRuleSet,
}

impl OwlRlEntailmentEngine {
    /// Create a new `OwlRlEntailmentEngine` with all rule groups enabled.
    pub fn new() -> Self {
        Self {
            rule_set: OwlRlRuleSet::default(),
        }
    }

    /// Create a new engine with a custom rule set.
    pub fn with_rule_set(rule_set: OwlRlRuleSet) -> Self {
        Self { rule_set }
    }

    // ── Helper ────────────────────────────────────────────────────────────

    /// Add a triple to `result` if not already in `store` or `result`.
    fn maybe_add(store: &TripleStore, result: &mut Vec<Triple>, t: Triple) {
        if !store.contains(&t.subject, &t.predicate, &t.object)
            && !result.iter().any(|r| r == &t)
        {
            result.push(t);
        }
    }

    // ── RDFS subset rules (needed as foundation for OWL RL) ───────────────

    /// rdfs9 / cax-sco: `(?c rdfs:subClassOf ?d), (?x rdf:type ?c)` → `(?x rdf:type ?d)`
    fn rule_cax_sco(store: &TripleStore, result: &mut Vec<Triple>) {
        let sub_class = rdfs_iri("subClassOf");
        let rdf_type = rdf_iri("type");

        let class_pairs: Vec<(String, String)> = store
            .get_all_p(&sub_class)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (c, d) in &class_pairs {
            if c == d {
                continue;
            }
            for triple in store.get_all_p(&rdf_type) {
                if &triple.object == c {
                    let inferred = Triple::new(&triple.subject, &rdf_type, d);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    // ── Class axiom rules ─────────────────────────────────────────────────

    /// cax-eqc1: `(?c owl:equivalentClass ?d), (?x rdf:type ?c)` → `(?x rdf:type ?d)`
    fn rule_cax_eqc1(store: &TripleStore, result: &mut Vec<Triple>) {
        let equiv_class = owl_iri("equivalentClass");
        let sub_class = rdfs_iri("subClassOf");
        let rdf_type = rdf_iri("type");

        let equiv_pairs: Vec<(String, String)> = store
            .get_all_p(&equiv_class)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (c, d) in &equiv_pairs {
            // equivalentClass implies bidirectional subClassOf
            let t1 = Triple::new(c, &sub_class, d);
            Self::maybe_add(store, result, t1);
            let t2 = Triple::new(d, &sub_class, c);
            Self::maybe_add(store, result, t2);

            // Also forward-apply type membership
            for triple in store.get_all_p(&rdf_type) {
                if &triple.object == c {
                    let inferred = Triple::new(&triple.subject, &rdf_type, d);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    /// cax-eqc2: `(?c owl:equivalentClass ?d), (?x rdf:type ?d)` → `(?x rdf:type ?c)`
    fn rule_cax_eqc2(store: &TripleStore, result: &mut Vec<Triple>) {
        let equiv_class = owl_iri("equivalentClass");
        let rdf_type = rdf_iri("type");

        let equiv_pairs: Vec<(String, String)> = store
            .get_all_p(&equiv_class)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (c, d) in &equiv_pairs {
            for triple in store.get_all_p(&rdf_type) {
                if &triple.object == d {
                    let inferred = Triple::new(&triple.subject, &rdf_type, c);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    // ── Property characteristic rules ─────────────────────────────────────

    /// prp-symp: `(?p rdf:type owl:SymmetricProperty), (?x ?p ?y)` → `(?y ?p ?x)`
    fn rule_prp_symp(store: &TripleStore, result: &mut Vec<Triple>) {
        let rdf_type = rdf_iri("type");
        let symmetric = owl_iri("SymmetricProperty");

        let sym_props: Vec<String> = store
            .get_all_p(&rdf_type)
            .into_iter()
            .filter(|t| t.object == symmetric)
            .map(|t| t.subject.clone())
            .collect();

        for p in &sym_props {
            for triple in store.get_all_p(p) {
                let inferred = Triple::new(&triple.object, p, &triple.subject);
                Self::maybe_add(store, result, inferred);
            }
        }
    }

    /// prp-trp: `(?p rdf:type owl:TransitiveProperty), (?x ?p ?y), (?y ?p ?z)` → `(?x ?p ?z)`
    fn rule_prp_trp(store: &TripleStore, result: &mut Vec<Triple>) {
        let rdf_type = rdf_iri("type");
        let transitive = owl_iri("TransitiveProperty");

        let trans_props: Vec<String> = store
            .get_all_p(&rdf_type)
            .into_iter()
            .filter(|t| t.object == transitive)
            .map(|t| t.subject.clone())
            .collect();

        for p in &trans_props {
            let pairs: Vec<(String, String)> = store
                .get_all_p(p)
                .into_iter()
                .map(|t| (t.subject.clone(), t.object.clone()))
                .collect();

            for (x, y1) in &pairs {
                for (y2, z) in &pairs {
                    if y1 == y2 && x != z {
                        let inferred = Triple::new(x, p, z);
                        Self::maybe_add(store, result, inferred);
                    }
                }
            }
        }
    }

    // ── Property equivalence rules ────────────────────────────────────────

    /// prp-eqp1: `(?p owl:equivalentProperty ?q), (?x ?p ?y)` → `(?x ?q ?y)`
    fn rule_prp_eqp1(store: &TripleStore, result: &mut Vec<Triple>) {
        let equiv_prop = owl_iri("equivalentProperty");
        let sub_prop = rdfs_iri("subPropertyOf");

        let equiv_pairs: Vec<(String, String)> = store
            .get_all_p(&equiv_prop)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (p, q) in &equiv_pairs {
            // equivalentProperty implies bidirectional subPropertyOf
            let t1 = Triple::new(p, &sub_prop, q);
            Self::maybe_add(store, result, t1);
            let t2 = Triple::new(q, &sub_prop, p);
            Self::maybe_add(store, result, t2);

            // Forward-apply: every triple with predicate p also holds with q
            for triple in store.get_all_p(p) {
                let inferred = Triple::new(&triple.subject, q, &triple.object);
                Self::maybe_add(store, result, inferred);
            }
        }
    }

    /// prp-eqp2: `(?p owl:equivalentProperty ?q), (?x ?q ?y)` → `(?x ?p ?y)`
    fn rule_prp_eqp2(store: &TripleStore, result: &mut Vec<Triple>) {
        let equiv_prop = owl_iri("equivalentProperty");

        let equiv_pairs: Vec<(String, String)> = store
            .get_all_p(&equiv_prop)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (p, q) in &equiv_pairs {
            for triple in store.get_all_p(q) {
                let inferred = Triple::new(&triple.subject, p, &triple.object);
                Self::maybe_add(store, result, inferred);
            }
        }
    }

    // ── Class restriction rules ───────────────────────────────────────────

    /// cls-hv1: `(?x owl:hasValue ?y), (?x owl:onProperty ?p), (?u rdf:type ?x)` → `(?u ?p ?y)`
    fn rule_cls_hv1(store: &TripleStore, result: &mut Vec<Triple>) {
        let has_value = owl_iri("hasValue");
        let on_property = owl_iri("onProperty");
        let rdf_type = rdf_iri("type");

        // Collect hasValue restrictions: restriction → (value, property)
        let restrictions: Vec<(String, String, String)> = {
            let mut v = Vec::new();
            for hv_triple in store.get_all_p(&has_value) {
                let x = &hv_triple.subject;
                let y = &hv_triple.object;
                for op_triple in store.get_by_sp(x, &on_property) {
                    v.push((x.clone(), y.clone(), op_triple.object.clone()));
                }
            }
            v
        };

        for (x, y, p) in &restrictions {
            // For every u rdf:type x, infer u p y
            for triple in store.get_by_po(&rdf_type, x) {
                let u = &triple.subject;
                let inferred = Triple::new(u, p, y);
                Self::maybe_add(store, result, inferred);
            }
        }
    }

    /// cls-hv2: `(?x owl:hasValue ?y), (?x owl:onProperty ?p), (?u ?p ?y)` → `(?u rdf:type ?x)`
    fn rule_cls_hv2(store: &TripleStore, result: &mut Vec<Triple>) {
        let has_value = owl_iri("hasValue");
        let on_property = owl_iri("onProperty");
        let rdf_type = rdf_iri("type");

        let restrictions: Vec<(String, String, String)> = {
            let mut v = Vec::new();
            for hv_triple in store.get_all_p(&has_value) {
                let x = &hv_triple.subject;
                let y = &hv_triple.object;
                for op_triple in store.get_by_sp(x, &on_property) {
                    v.push((x.clone(), y.clone(), op_triple.object.clone()));
                }
            }
            v
        };

        for (x, y, p) in &restrictions {
            // For every u p y, infer u rdf:type x
            for triple in store.get_by_po(p, y) {
                let u = &triple.subject;
                let inferred = Triple::new(u, &rdf_type, x);
                Self::maybe_add(store, result, inferred);
            }
        }
    }

    /// cls-svf1: `(?x owl:someValuesFrom ?y), (?x owl:onProperty ?p),
    ///            (?u ?p ?v), (?v rdf:type ?y)` → `(?u rdf:type ?x)`
    fn rule_cls_svf1(store: &TripleStore, result: &mut Vec<Triple>) {
        let some_values_from = owl_iri("someValuesFrom");
        let on_property = owl_iri("onProperty");
        let rdf_type = rdf_iri("type");

        // restrictions: (x, y=filler, p=property)
        let restrictions: Vec<(String, String, String)> = {
            let mut v = Vec::new();
            for svf_triple in store.get_all_p(&some_values_from) {
                let x = &svf_triple.subject;
                let y = &svf_triple.object;
                for op_triple in store.get_by_sp(x, &on_property) {
                    v.push((x.clone(), y.clone(), op_triple.object.clone()));
                }
            }
            v
        };

        for (x, y, p) in &restrictions {
            // For every (u p v) where v rdf:type y, infer u rdf:type x
            for p_triple in store.get_all_p(p) {
                let u = &p_triple.subject;
                let v = &p_triple.object;
                if store.contains(v, &rdf_type, y) {
                    let inferred = Triple::new(u, &rdf_type, x);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    /// cls-avf: `(?x owl:allValuesFrom ?y), (?x owl:onProperty ?p),
    ///           (?u rdf:type ?x), (?u ?p ?v)` → `(?v rdf:type ?y)`
    fn rule_cls_avf(store: &TripleStore, result: &mut Vec<Triple>) {
        let all_values_from = owl_iri("allValuesFrom");
        let on_property = owl_iri("onProperty");
        let rdf_type = rdf_iri("type");

        let restrictions: Vec<(String, String, String)> = {
            let mut v = Vec::new();
            for avf_triple in store.get_all_p(&all_values_from) {
                let x = &avf_triple.subject;
                let y = &avf_triple.object;
                for op_triple in store.get_by_sp(x, &on_property) {
                    v.push((x.clone(), y.clone(), op_triple.object.clone()));
                }
            }
            v
        };

        for (x, y, p) in &restrictions {
            // For every u rdf:type x and u p v, infer v rdf:type y
            for type_triple in store.get_by_po(&rdf_type, x) {
                let u = &type_triple.subject;
                for p_triple in store.get_by_sp(u, p) {
                    let v = &p_triple.object;
                    let inferred = Triple::new(v, &rdf_type, y);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    /// cls-int2: `(?c owl:intersectionOf ?list), (?y rdf:type ?c)`
    ///           → `(?y rdf:type ?ci)` for each `?ci` in the intersection list.
    ///
    /// This rule decomposes an intersection class: if something is typed as
    /// an intersection, it must be typed as each component.
    fn rule_cls_int2(store: &TripleStore, result: &mut Vec<Triple>) {
        let intersection_of = owl_iri("intersectionOf");
        let rdf_type = rdf_iri("type");
        let first_pred = rdf_iri("first");
        let rest_pred = rdf_iri("rest");
        let rdf_nil = rdf_iri("nil");

        // For each intersection class c
        for int_triple in store.get_all_p(&intersection_of) {
            let c = &int_triple.subject;
            let mut list_node = int_triple.object.clone();

            // Collect all members of the RDF list
            let mut members: Vec<String> = Vec::new();
            while list_node != rdf_nil {
                let firsts = store.get_by_sp(&list_node, &first_pred);
                if firsts.is_empty() {
                    break;
                }
                members.push(firsts[0].object.clone());
                let rests = store.get_by_sp(&list_node, &rest_pred);
                if rests.is_empty() {
                    break;
                }
                list_node = rests[0].object.clone();
            }

            if members.is_empty() {
                continue;
            }

            // For every y rdf:type c, infer y rdf:type each member
            for type_triple in store.get_by_po(&rdf_type, c) {
                let y = &type_triple.subject;
                for member in &members {
                    let inferred = Triple::new(y, &rdf_type, member);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    /// cls-int1: `(?c owl:intersectionOf ?list), members = {c1, c2, ...},
    ///            (?y rdf:type c1), (?y rdf:type c2), ...` → `(?y rdf:type ?c)`
    ///
    /// If something is typed as every member of an intersection, it is typed
    /// as the intersection class.
    fn rule_cls_int1(store: &TripleStore, result: &mut Vec<Triple>) {
        let intersection_of = owl_iri("intersectionOf");
        let rdf_type = rdf_iri("type");
        let first_pred = rdf_iri("first");
        let rest_pred = rdf_iri("rest");
        let rdf_nil = rdf_iri("nil");

        for int_triple in store.get_all_p(&intersection_of) {
            let c = &int_triple.subject;
            let mut list_node = int_triple.object.clone();

            let mut members: Vec<String> = Vec::new();
            while list_node != rdf_nil {
                let firsts = store.get_by_sp(&list_node, &first_pred);
                if firsts.is_empty() {
                    break;
                }
                members.push(firsts[0].object.clone());
                let rests = store.get_by_sp(&list_node, &rest_pred);
                if rests.is_empty() {
                    break;
                }
                list_node = rests[0].object.clone();
            }

            if members.is_empty() {
                continue;
            }

            // Find all individuals that are typed as *every* member of the intersection
            // Start with individuals typed as the first member
            let first_typed: Vec<String> = store
                .get_by_po(&rdf_type, &members[0])
                .into_iter()
                .map(|t| t.subject.clone())
                .collect();

            for y in first_typed {
                let all_typed = members
                    .iter()
                    .all(|m| store.contains(&y, &rdf_type, m));
                if all_typed {
                    let inferred = Triple::new(&y, &rdf_type, c);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }
}

impl Default for OwlRlEntailmentEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl EntailmentEngine for OwlRlEntailmentEngine {
    fn entail(&self, store: &TripleStore) -> Result<Vec<Triple>, EntailmentError> {
        let mut result: Vec<Triple> = Vec::new();

        // Apply RDFS foundation rules first (needed by OWL RL on top)
        if self.rule_set.class_axioms {
            Self::rule_cax_sco(store, &mut result);
        }

        // Apply OWL RL class axiom rules
        if self.rule_set.class_axioms {
            Self::rule_cax_eqc1(store, &mut result);
            Self::rule_cax_eqc2(store, &mut result);
        }

        // Apply property characteristic rules
        if self.rule_set.property_characteristics {
            Self::rule_prp_symp(store, &mut result);
            Self::rule_prp_trp(store, &mut result);
        }

        // Apply equivalent property rules
        if self.rule_set.equivalent_properties {
            Self::rule_prp_eqp1(store, &mut result);
            Self::rule_prp_eqp2(store, &mut result);
        }

        // Apply class restriction rules
        if self.rule_set.class_restrictions {
            Self::rule_cls_hv1(store, &mut result);
            Self::rule_cls_hv2(store, &mut result);
            Self::rule_cls_svf1(store, &mut result);
            Self::rule_cls_avf(store, &mut result);
            Self::rule_cls_int1(store, &mut result);
            Self::rule_cls_int2(store, &mut result);
        }

        Ok(result)
    }
}

// ── EntailmentRegime trait implementation for OwlRlEntailmentEngine ───────────
//
// This allows OwlRlEntailmentEngine to be used with the rich-term EntailmentGraph
// API (via the EntailmentRegime trait) as well as the flat-string ClosureGraph API
// (via the EntailmentEngine trait).

impl super::EntailmentRegime for OwlRlEntailmentEngine {
    fn name(&self) -> &str {
        "OWL 2 RL Entailment"
    }

    fn entail(
        &self,
        triples: &[super::RichEntailmentTriple],
    ) -> Vec<super::RichEntailmentTriple> {
        // Convert rich triples → flat TripleStore, run rules, convert back
        let mut store = TripleStore::new();
        for rt in triples {
            let s = rich_term_to_str(&rt.subject);
            let o = rich_term_to_str(&rt.object);
            store.add(Triple::new(&s, &rt.predicate, &o));
        }

        let flat_results = match EntailmentEngine::entail(self, &store) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        // Filter out triples that are already present in the input
        flat_results
            .into_iter()
            .filter_map(|t| {
                let already = triples.iter().any(|rt| {
                    let s = rich_term_to_str(&rt.subject);
                    let o = rich_term_to_str(&rt.object);
                    s == t.subject && rt.predicate == t.predicate && o == t.object
                });
                if already {
                    None
                } else {
                    Some(super::RichEntailmentTriple::named_triple(
                        &t.subject,
                        &t.predicate,
                        &t.object,
                    ))
                }
            })
            .collect()
    }

    fn is_consistent(&self, _triples: &[super::RichEntailmentTriple]) -> bool {
        // OWL 2 RL consistency checking is a separate concern (requires full ABox reasoning).
        // A conservative implementation always returns true here.
        true
    }
}

/// Convert a rich [`EntailmentTerm`] to a flat string for use with the flat-string API.
fn rich_term_to_str(term: &super::EntailmentTerm) -> String {
    match term {
        super::EntailmentTerm::NamedNode(s) | super::EntailmentTerm::BlankNode(s) => s.clone(),
        super::EntailmentTerm::Literal { value, .. } => value.clone(),
    }
}

// ── Impl EntailmentRegime for RdfEntailmentEngine ────────────────────────────
//
// Provide EntailmentRegime blanket impls for the other engines so they can also
// be used with the rich-term EntailmentGraph API. Placed here to avoid
// circular module deps (rdf_entailment.rs would need to import regime types).

impl super::EntailmentRegime for super::rdf_entailment::RdfEntailmentEngine {
    fn name(&self) -> &str {
        "RDF Entailment"
    }

    fn entail(
        &self,
        triples: &[super::RichEntailmentTriple],
    ) -> Vec<super::RichEntailmentTriple> {
        let mut store = TripleStore::new();
        for rt in triples {
            let s = rich_term_to_str(&rt.subject);
            let o = rich_term_to_str(&rt.object);
            store.add(Triple::new(&s, &rt.predicate, &o));
        }

        let flat_results = match EntailmentEngine::entail(self, &store) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        flat_results
            .into_iter()
            .filter_map(|t| {
                let already = triples.iter().any(|rt| {
                    let s = rich_term_to_str(&rt.subject);
                    let o = rich_term_to_str(&rt.object);
                    s == t.subject && rt.predicate == t.predicate && o == t.object
                });
                if already {
                    None
                } else {
                    Some(super::RichEntailmentTriple::named_triple(
                        &t.subject,
                        &t.predicate,
                        &t.object,
                    ))
                }
            })
            .collect()
    }
}

// ── Impl EntailmentRegime for RdfsEntailmentEngine ────────────────────────────

impl super::EntailmentRegime for super::rdfs_entailment::RdfsEntailmentEngine {
    fn name(&self) -> &str {
        "RDFS Entailment"
    }

    fn entail(
        &self,
        triples: &[super::RichEntailmentTriple],
    ) -> Vec<super::RichEntailmentTriple> {
        let mut store = TripleStore::new();
        for rt in triples {
            let s = rich_term_to_str(&rt.subject);
            let o = rich_term_to_str(&rt.object);
            store.add(Triple::new(&s, &rt.predicate, &o));
        }

        let flat_results = match EntailmentEngine::entail(self, &store) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        flat_results
            .into_iter()
            .filter_map(|t| {
                let already = triples.iter().any(|rt| {
                    let s = rich_term_to_str(&rt.subject);
                    let o = rich_term_to_str(&rt.object);
                    s == t.subject && rt.predicate == t.predicate && o == t.object
                });
                if already {
                    None
                } else {
                    Some(super::RichEntailmentTriple::named_triple(
                        &t.subject,
                        &t.predicate,
                        &t.object,
                    ))
                }
            })
            .collect()
    }
}
