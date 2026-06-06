//! RDFS Entailment Engine — implements the W3C RDFS entailment rules via the
//! [`EntailmentEngine`] trait.
//!
//! # Rules implemented
//!
//! | Rule    | Premise(s)                                                           | Conclusion                         |
//! |---------|----------------------------------------------------------------------|------------------------------------|
//! | rdfs2   | `?a rdfs:domain ?x`, `?u ?a ?y`                                      | `?u rdf:type ?x`                   |
//! | rdfs3   | `?a rdfs:range ?x`, `?u ?a ?v`                                       | `?v rdf:type ?x`                   |
//! | rdfs5   | `?u rdfs:subPropertyOf ?v`, `?v rdfs:subPropertyOf ?w`               | `?u rdfs:subPropertyOf ?w`         |
//! | rdfs6   | `?u rdf:type rdf:Property`                                           | `?u rdfs:subPropertyOf ?u`         |
//! | rdfs7   | `?a rdfs:subPropertyOf ?b`, `?u ?a ?y`                               | `?u ?b ?y`                         |
//! | rdfs8   | `?u rdf:type rdfs:Class`                                             | `?u rdfs:subClassOf rdfs:Resource` |
//! | rdfs9   | `?u rdfs:subClassOf ?x`, `?v rdf:type ?u`                            | `?v rdf:type ?x`                   |
//! | rdfs10  | `?u rdf:type rdfs:Class`                                             | `?u rdfs:subClassOf ?u`            |
//! | rdfs11  | `?u rdfs:subClassOf ?v`, `?v rdfs:subClassOf ?w`                     | `?u rdfs:subClassOf ?w`            |
//! | rdfs12  | `?u rdf:type rdfs:ContainerMembershipProperty`                       | `?u rdfs:subPropertyOf rdfs:member`|
//!
//! # References
//!
//! - <https://www.w3.org/TR/rdf11-mt/#rdfs-entailment>

use super::{EntailmentEngine, EntailmentError, Triple, TripleStore};
use crate::entailment::rdf_entailment::{rdfs_iri, rdf_iri};

// ── RdfsEntailmentEngine ──────────────────────────────────────────────────────

/// An [`EntailmentEngine`] that applies the W3C RDFS entailment rules (rdfs2–rdfs12)
/// to a single pass. The caller should iterate to fixpoint via [`EntailmentGraph::close`].
#[derive(Debug, Default, Clone)]
pub struct RdfsEntailmentEngine;

impl RdfsEntailmentEngine {
    /// Create a new `RdfsEntailmentEngine`
    pub fn new() -> Self {
        Self
    }

    // ── Internal helper: add triple if not already in store or result ─────

    fn maybe_add(store: &TripleStore, result: &mut Vec<Triple>, t: Triple) {
        if !store.contains(&t.subject, &t.predicate, &t.object)
            && !result.iter().any(|r| r == &t)
        {
            result.push(t);
        }
    }

    // ── rdfs2: domain inference ───────────────────────────────────────────

    fn rule_rdfs2(store: &TripleStore, result: &mut Vec<Triple>) {
        let domain_pred = rdfs_iri("domain");
        let rdf_type = rdf_iri("type");

        // Collect (property, class) domain declarations
        let domain_pairs: Vec<(String, String)> = store
            .get_all_p(&domain_pred)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (prop, class) in &domain_pairs {
            // For every triple that uses `prop` as predicate, its subject gets rdf:type `class`
            for triple in store.iter() {
                if &triple.predicate == prop {
                    let inferred = Triple::new(&triple.subject, &rdf_type, class);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    // ── rdfs3: range inference ────────────────────────────────────────────

    fn rule_rdfs3(store: &TripleStore, result: &mut Vec<Triple>) {
        let range_pred = rdfs_iri("range");
        let rdf_type = rdf_iri("type");

        let range_pairs: Vec<(String, String)> = store
            .get_all_p(&range_pred)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (prop, class) in &range_pairs {
            for triple in store.iter() {
                if &triple.predicate == prop {
                    let inferred = Triple::new(&triple.object, &rdf_type, class);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    // ── rdfs5: subPropertyOf transitivity ────────────────────────────────

    fn rule_rdfs5(store: &TripleStore, result: &mut Vec<Triple>) {
        let sub_prop = rdfs_iri("subPropertyOf");

        let pairs: Vec<(String, String)> = store
            .get_all_p(&sub_prop)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (p, q) in &pairs {
            for (q2, r) in &pairs {
                if q == q2 && p != r {
                    let inferred = Triple::new(p, &sub_prop, r);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    // ── rdfs6: every rdf:Property is subPropertyOf itself ────────────────

    fn rule_rdfs6(store: &TripleStore, result: &mut Vec<Triple>) {
        let rdf_type = rdf_iri("type");
        let rdf_property = rdf_iri("Property");
        let sub_prop = rdfs_iri("subPropertyOf");

        for triple in store.get_all_p(&rdf_type) {
            if triple.object == rdf_property {
                let u = &triple.subject;
                let inferred = Triple::new(u, &sub_prop, u);
                Self::maybe_add(store, result, inferred);
            }
        }
    }

    // ── rdfs7: subPropertyOf propagation ─────────────────────────────────

    fn rule_rdfs7(store: &TripleStore, result: &mut Vec<Triple>) {
        let sub_prop = rdfs_iri("subPropertyOf");

        let pairs: Vec<(String, String)> = store
            .get_all_p(&sub_prop)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (p, q) in &pairs {
            if p == q {
                continue; // reflexive, would duplicate everything
            }
            for triple in store.iter() {
                if &triple.predicate == p {
                    let inferred = Triple::new(&triple.subject, q, &triple.object);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    // ── rdfs8: every rdfs:Class is subClassOf rdfs:Resource ──────────────

    fn rule_rdfs8(store: &TripleStore, result: &mut Vec<Triple>) {
        let rdf_type = rdf_iri("type");
        let rdfs_class = rdfs_iri("Class");
        let rdfs_resource = rdfs_iri("Resource");
        let sub_class = rdfs_iri("subClassOf");

        for triple in store.get_all_p(&rdf_type) {
            if triple.object == rdfs_class {
                let u = &triple.subject;
                let inferred = Triple::new(u, &sub_class, &rdfs_resource);
                Self::maybe_add(store, result, inferred);
            }
        }
    }

    // ── rdfs9: subClassOf instance inheritance ────────────────────────────

    fn rule_rdfs9(store: &TripleStore, result: &mut Vec<Triple>) {
        let rdf_type = rdf_iri("type");
        let sub_class = rdfs_iri("subClassOf");

        // (c rdfs:subClassOf d) pairs
        let class_pairs: Vec<(String, String)> = store
            .get_all_p(&sub_class)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (c, d) in &class_pairs {
            if c == d {
                continue; // reflexive pair; no new instances
            }
            // For every v rdf:type c, infer v rdf:type d
            for triple in store.get_all_p(&rdf_type) {
                if &triple.object == c {
                    let inferred = Triple::new(&triple.subject, &rdf_type, d);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    // ── rdfs10: every rdfs:Class is subClassOf itself ─────────────────────

    fn rule_rdfs10(store: &TripleStore, result: &mut Vec<Triple>) {
        let rdf_type = rdf_iri("type");
        let rdfs_class = rdfs_iri("Class");
        let sub_class = rdfs_iri("subClassOf");

        for triple in store.get_all_p(&rdf_type) {
            if triple.object == rdfs_class {
                let u = &triple.subject;
                let inferred = Triple::new(u, &sub_class, u);
                Self::maybe_add(store, result, inferred);
            }
        }
    }

    // ── rdfs11: subClassOf transitivity ──────────────────────────────────

    fn rule_rdfs11(store: &TripleStore, result: &mut Vec<Triple>) {
        let sub_class = rdfs_iri("subClassOf");

        let pairs: Vec<(String, String)> = store
            .get_all_p(&sub_class)
            .into_iter()
            .map(|t| (t.subject.clone(), t.object.clone()))
            .collect();

        for (c, d) in &pairs {
            for (d2, e) in &pairs {
                if d == d2 && c != e {
                    let inferred = Triple::new(c, &sub_class, e);
                    Self::maybe_add(store, result, inferred);
                }
            }
        }
    }

    // ── rdfs12: ContainerMembershipProperty subPropertyOf rdfs:member ────

    fn rule_rdfs12(store: &TripleStore, result: &mut Vec<Triple>) {
        let rdf_type = rdf_iri("type");
        let rdfs_cmp = rdfs_iri("ContainerMembershipProperty");
        let rdfs_member = rdfs_iri("member");
        let sub_prop = rdfs_iri("subPropertyOf");

        for triple in store.get_all_p(&rdf_type) {
            if triple.object == rdfs_cmp {
                let u = &triple.subject;
                let inferred = Triple::new(u, &sub_prop, &rdfs_member);
                Self::maybe_add(store, result, inferred);
            }
        }
    }
}

impl EntailmentEngine for RdfsEntailmentEngine {
    fn entail(&self, store: &TripleStore) -> Result<Vec<Triple>, EntailmentError> {
        let mut result: Vec<Triple> = Vec::new();

        Self::rule_rdfs2(store, &mut result);
        Self::rule_rdfs3(store, &mut result);
        Self::rule_rdfs5(store, &mut result);
        Self::rule_rdfs6(store, &mut result);
        Self::rule_rdfs7(store, &mut result);
        Self::rule_rdfs8(store, &mut result);
        Self::rule_rdfs9(store, &mut result);
        Self::rule_rdfs10(store, &mut result);
        Self::rule_rdfs11(store, &mut result);
        Self::rule_rdfs12(store, &mut result);

        Ok(result)
    }
}
