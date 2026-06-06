//! RDF Entailment Engine — implements W3C RDF entailment rules via the
//! [`EntailmentEngine`] trait.
//!
//! # Rules implemented
//!
//! - **rdf1**: Every predicate `<p>` used in a triple entails `<p> rdf:type rdf:Property`.
//! - **rdf4**: `rdf:type rdf:type rdf:Property` is an axiomatic triple.
//! - **Container membership properties**: `rdf:_1`, `rdf:_2`, … are instances of
//!   `rdfs:ContainerMembershipProperty` and `rdfs:subPropertyOf rdfs:member`.
//!
//! # References
//!
//! - <https://www.w3.org/TR/rdf11-mt/#rdfs-entailment>

use super::{EntailmentEngine, EntailmentError, Triple, TripleStore};

// ── Vocabulary shortcuts ──────────────────────────────────────────────────────

pub(super) const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
pub(super) const RDFS_NS: &str = "http://www.w3.org/2000/01/rdf-schema#";
#[allow(dead_code)]
pub(super) const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

#[inline]
pub(super) fn rdf_iri(local: &str) -> String {
    format!("{RDF_NS}{local}")
}

#[inline]
pub(super) fn rdfs_iri(local: &str) -> String {
    format!("{RDFS_NS}{local}")
}

// ── RdfEntailmentEngine ───────────────────────────────────────────────────────

/// An [`EntailmentEngine`] that applies the W3C RDF entailment rules.
///
/// Implements the following rules from the RDF 1.1 Semantics specification:
///
/// | Rule   | Conclusion                                        |
/// |--------|---------------------------------------------------|
/// | rdf1   | `?p rdf:type rdf:Property` for every predicate    |
/// | rdf4   | `rdf:type rdf:type rdf:Property` (axiomatic)      |
/// | cmprop | Container membership properties type and subOf    |
#[derive(Debug, Default, Clone)]
pub struct RdfEntailmentEngine;

impl RdfEntailmentEngine {
    /// Create a new `RdfEntailmentEngine`
    pub fn new() -> Self {
        Self
    }

    /// Determine whether `s` looks like an RDF container membership property IRI
    /// (`rdf:_1`, `rdf:_2`, …).
    fn is_container_membership_prop(iri: &str) -> bool {
        if let Some(local) = iri.strip_prefix(RDF_NS) {
            // local must be "_" followed by one or more decimal digits
            local
                .strip_prefix('_')
                .is_some_and(|digits| !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()))
        } else {
            false
        }
    }
}

impl EntailmentEngine for RdfEntailmentEngine {
    fn entail(&self, store: &TripleStore) -> Result<Vec<Triple>, EntailmentError> {
        let rdf_type = rdf_iri("type");
        let rdf_property = rdf_iri("Property");
        let rdfs_cmp = rdfs_iri("ContainerMembershipProperty");
        let rdfs_member = rdfs_iri("member");
        let rdfs_sub_prop = rdfs_iri("subPropertyOf");

        let mut result: Vec<Triple> = Vec::new();

        // rdf4: rdf:type is itself an rdf:Property (axiomatic)
        let rdf4 = Triple::new(&rdf_type, &rdf_type, &rdf_property);
        if !store.contains(&rdf4.subject, &rdf4.predicate, &rdf4.object) {
            result.push(rdf4);
        }

        // rdf1: for every triple <s p o>, p rdf:type rdf:Property
        // rdf container membership props: additional typing
        for triple in store.iter() {
            let p = &triple.predicate;

            // rdf1
            let t = Triple::new(p, &rdf_type, &rdf_property);
            if !store.contains(&t.subject, &t.predicate, &t.object) {
                // avoid duplicates within result
                if !result.iter().any(|r| r == &t) {
                    result.push(t);
                }
            }

            // Container membership property rules (rdf5a-j generalised)
            if Self::is_container_membership_prop(p) {
                // rdf:_n rdf:type rdfs:ContainerMembershipProperty
                let cmp_type = Triple::new(p, &rdf_type, &rdfs_cmp);
                if !store.contains(&cmp_type.subject, &cmp_type.predicate, &cmp_type.object)
                    && !result.iter().any(|r| r == &cmp_type)
                {
                    result.push(cmp_type);
                }
                // rdf:_n rdfs:subPropertyOf rdfs:member
                let sub_member = Triple::new(p, &rdfs_sub_prop, &rdfs_member);
                if !store.contains(&sub_member.subject, &sub_member.predicate, &sub_member.object)
                    && !result.iter().any(|r| r == &sub_member)
                {
                    result.push(sub_member);
                }
            }
        }

        Ok(result)
    }
}
