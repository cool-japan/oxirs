//! RDFS entailment regime implementation.
//!
//! Implements all standard RDFS entailment rules (rdf1, rdfs2–rdfs13)
//! via a forward-chaining fixpoint materialisation.

use std::collections::HashSet;

// ── Vocabulary constants ──────────────────────────────────────────────────────

const RDF_TYPE: &str = "rdf:type";
const RDF_PROPERTY: &str = "rdf:Property";
#[allow(dead_code)]
const RDF_STATEMENT: &str = "rdf:Statement";
const RDFS_DOMAIN: &str = "rdfs:domain";
const RDFS_RANGE: &str = "rdfs:range";
const RDFS_RESOURCE: &str = "rdfs:Resource";
const RDFS_CLASS: &str = "rdfs:Class";
const RDFS_LITERAL: &str = "rdfs:Literal";
const RDFS_DATATYPE: &str = "rdfs:Datatype";
const RDFS_SUB_CLASS_OF: &str = "rdfs:subClassOf";
const RDFS_SUB_PROPERTY_OF: &str = "rdfs:subPropertyOf";
const RDFS_MEMBER: &str = "rdfs:member";
const RDFS_CONTAINER_MEMBERSHIP_PROPERTY: &str = "rdfs:ContainerMembershipProperty";

// ── Public types ──────────────────────────────────────────────────────────────

/// A single RDF triple (using short prefixed IRIs as strings).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    fn new(s: &str, p: &str, o: &str) -> Self {
        Triple {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
        }
    }
}

/// RDFS reasoner: materialises all RDFS-entailed triples from the base set.
pub struct RdfsReasoner {
    /// Asserted + inferred triples.
    triples: Vec<Triple>,
    /// Triples that were derived by inference (as a set for fast lookup).
    inferred: HashSet<(String, String, String)>,
    /// All known triples as a set for deduplication.
    all: HashSet<(String, String, String)>,
}

impl Default for RdfsReasoner {
    fn default() -> Self {
        Self::new()
    }
}

impl RdfsReasoner {
    /// Create an empty reasoner.
    pub fn new() -> Self {
        RdfsReasoner {
            triples: Vec::new(),
            inferred: HashSet::new(),
            all: HashSet::new(),
        }
    }

    /// Add an asserted triple.
    pub fn add_triple(&mut self, s: &str, p: &str, o: &str) {
        let key = (s.to_string(), p.to_string(), o.to_string());
        if self.all.insert(key) {
            self.triples.push(Triple::new(s, p, o));
        }
    }

    /// Run all RDFS entailment rules to a fixpoint.
    ///
    /// Returns the number of newly inferred triples.
    pub fn materialize(&mut self) -> usize {
        let mut changed = true;
        let before = self.inferred.len();

        while changed {
            changed = false;
            let snapshot: Vec<Triple> = self.triples.clone();

            for new in Self::apply_rules_once(&snapshot) {
                let key = (
                    new.subject.clone(),
                    new.predicate.clone(),
                    new.object.clone(),
                );
                if self.all.insert(key.clone()) {
                    self.inferred.insert(key);
                    self.triples.push(new);
                    changed = true;
                }
            }
        }

        self.inferred.len() - before
    }

    /// Query triples with optional subject / predicate / object filters.
    pub fn query<'a>(
        &'a self,
        s: Option<&str>,
        p: Option<&str>,
        o: Option<&str>,
    ) -> Vec<&'a Triple> {
        self.triples
            .iter()
            .filter(|t| {
                s.map_or(true, |v| t.subject == v)
                    && p.map_or(true, |v| t.predicate == v)
                    && o.map_or(true, |v| t.object == v)
            })
            .collect()
    }

    /// Number of triples inferred by the reasoner (not asserted).
    pub fn inferred_count(&self) -> usize {
        self.inferred.len()
    }

    // ── Rule application ──────────────────────────────────────────────────

    /// Apply one round of all RDFS rules to `triples`, returning new triples.
    fn apply_rules_once(triples: &[Triple]) -> Vec<Triple> {
        let mut new_triples: Vec<Triple> = Vec::new();

        // Build lookup helpers (predicate → list of (s,o) pairs).
        let get_pairs = |pred: &str| -> Vec<(&str, &str)> {
            triples
                .iter()
                .filter(|t| t.predicate == pred)
                .map(|t| (t.subject.as_str(), t.object.as_str()))
                .collect()
        };

        // rdf1: ?s ?p ?o  →  ?p rdf:type rdf:Property
        for t in triples {
            new_triples.push(Triple::new(&t.predicate, RDF_TYPE, RDF_PROPERTY));
        }

        // rdfs2: ?p rdfs:domain ?c . ?s ?p ?o  →  ?s rdf:type ?c
        for (p, c) in get_pairs(RDFS_DOMAIN) {
            for t in triples {
                if t.predicate == p {
                    new_triples.push(Triple::new(&t.subject, RDF_TYPE, c));
                }
            }
        }

        // rdfs3: ?p rdfs:range ?c . ?s ?p ?o  →  ?o rdf:type ?c
        for (p, c) in get_pairs(RDFS_RANGE) {
            for t in triples {
                if t.predicate == p {
                    new_triples.push(Triple::new(&t.object, RDF_TYPE, c));
                }
            }
        }

        // rdfs4a: ?s ?p ?o  →  ?s rdf:type rdfs:Resource
        for t in triples {
            new_triples.push(Triple::new(&t.subject, RDF_TYPE, RDFS_RESOURCE));
        }

        // rdfs4b: ?s ?p ?o  →  ?o rdf:type rdfs:Resource
        for t in triples {
            new_triples.push(Triple::new(&t.object, RDF_TYPE, RDFS_RESOURCE));
        }

        // rdfs5: ?p rdfs:subPropertyOf ?q . ?q rdfs:subPropertyOf ?r
        //         →  ?p rdfs:subPropertyOf ?r
        let sub_prop_pairs: Vec<(&str, &str)> = get_pairs(RDFS_SUB_PROPERTY_OF);
        for (p, q) in &sub_prop_pairs {
            for (q2, r) in &sub_prop_pairs {
                if p != r && q == q2 {
                    new_triples.push(Triple::new(p, RDFS_SUB_PROPERTY_OF, r));
                }
            }
        }

        // rdfs6: ?p rdf:type rdf:Property  →  ?p rdfs:subPropertyOf ?p
        for t in triples {
            if t.predicate == RDF_TYPE && t.object == RDF_PROPERTY {
                new_triples.push(Triple::new(&t.subject, RDFS_SUB_PROPERTY_OF, &t.subject));
            }
        }

        // rdfs7: ?s ?p ?o . ?p rdfs:subPropertyOf ?q  →  ?s ?q ?o
        for (p, q) in get_pairs(RDFS_SUB_PROPERTY_OF) {
            for t in triples {
                if t.predicate == p {
                    new_triples.push(Triple::new(&t.subject, q, &t.object));
                }
            }
        }

        // rdfs8: ?c rdf:type rdfs:Class  →  ?c rdfs:subClassOf rdfs:Resource
        for t in triples {
            if t.predicate == RDF_TYPE && t.object == RDFS_CLASS {
                new_triples.push(Triple::new(&t.subject, RDFS_SUB_CLASS_OF, RDFS_RESOURCE));
            }
        }

        // rdfs9: ?s rdf:type ?c . ?c rdfs:subClassOf ?d  →  ?s rdf:type ?d
        let type_pairs: Vec<(&str, &str)> = get_pairs(RDF_TYPE);
        let sub_class_pairs: Vec<(&str, &str)> = get_pairs(RDFS_SUB_CLASS_OF);
        for (s, c) in &type_pairs {
            for (c2, d) in &sub_class_pairs {
                if c == c2 {
                    new_triples.push(Triple::new(s, RDF_TYPE, d));
                }
            }
        }

        // rdfs10: ?c rdf:type rdfs:Class  →  ?c rdfs:subClassOf ?c
        for t in triples {
            if t.predicate == RDF_TYPE && t.object == RDFS_CLASS {
                new_triples.push(Triple::new(&t.subject, RDFS_SUB_CLASS_OF, &t.subject));
            }
        }

        // rdfs11: ?c rdfs:subClassOf ?d . ?d rdfs:subClassOf ?e
        //          →  ?c rdfs:subClassOf ?e
        for (c, d) in &sub_class_pairs {
            for (d2, e) in &sub_class_pairs {
                if c != e && d == d2 {
                    new_triples.push(Triple::new(c, RDFS_SUB_CLASS_OF, e));
                }
            }
        }

        // rdfs12: ?p rdf:type rdfs:ContainerMembershipProperty
        //          →  ?p rdfs:subPropertyOf rdfs:member
        for t in triples {
            if t.predicate == RDF_TYPE && t.object == RDFS_CONTAINER_MEMBERSHIP_PROPERTY {
                new_triples.push(Triple::new(&t.subject, RDFS_SUB_PROPERTY_OF, RDFS_MEMBER));
            }
        }

        // rdfs13: ?c rdf:type rdfs:Datatype
        //          →  ?c rdfs:subClassOf rdfs:Literal
        for t in triples {
            if t.predicate == RDF_TYPE && t.object == RDFS_DATATYPE {
                new_triples.push(Triple::new(&t.subject, RDFS_SUB_CLASS_OF, RDFS_LITERAL));
            }
        }

        new_triples
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────

    fn fresh() -> RdfsReasoner {
        RdfsReasoner::new()
    }

    fn has(r: &RdfsReasoner, s: &str, p: &str, o: &str) -> bool {
        !r.query(Some(s), Some(p), Some(o)).is_empty()
    }

    // ── rdf1 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rdf1_predicate_is_property() {
        let mut r = fresh();
        r.add_triple("ex:bob", "ex:knows", "ex:alice");
        r.materialize();
        assert!(has(&r, "ex:knows", RDF_TYPE, RDF_PROPERTY));
    }

    // ── rdfs2 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs2_domain() {
        let mut r = fresh();
        r.add_triple("ex:knows", RDFS_DOMAIN, "ex:Person");
        r.add_triple("ex:bob", "ex:knows", "ex:alice");
        r.materialize();
        assert!(has(&r, "ex:bob", RDF_TYPE, "ex:Person"));
    }

    // ── rdfs3 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs3_range() {
        let mut r = fresh();
        r.add_triple("ex:knows", RDFS_RANGE, "ex:Agent");
        r.add_triple("ex:bob", "ex:knows", "ex:alice");
        r.materialize();
        assert!(has(&r, "ex:alice", RDF_TYPE, "ex:Agent"));
    }

    // ── rdfs4a / rdfs4b ───────────────────────────────────────────────────

    #[test]
    fn test_rdfs4a_subject_is_resource() {
        let mut r = fresh();
        r.add_triple("ex:bob", "ex:knows", "ex:alice");
        r.materialize();
        assert!(has(&r, "ex:bob", RDF_TYPE, RDFS_RESOURCE));
    }

    #[test]
    fn test_rdfs4b_object_is_resource() {
        let mut r = fresh();
        r.add_triple("ex:bob", "ex:knows", "ex:alice");
        r.materialize();
        assert!(has(&r, "ex:alice", RDF_TYPE, RDFS_RESOURCE));
    }

    // ── rdfs5 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs5_sub_property_transitivity() {
        let mut r = fresh();
        r.add_triple("ex:p", RDFS_SUB_PROPERTY_OF, "ex:q");
        r.add_triple("ex:q", RDFS_SUB_PROPERTY_OF, "ex:r");
        r.materialize();
        assert!(has(&r, "ex:p", RDFS_SUB_PROPERTY_OF, "ex:r"));
    }

    #[test]
    fn test_rdfs5_chain_of_three() {
        let mut r = fresh();
        r.add_triple("ex:p1", RDFS_SUB_PROPERTY_OF, "ex:p2");
        r.add_triple("ex:p2", RDFS_SUB_PROPERTY_OF, "ex:p3");
        r.add_triple("ex:p3", RDFS_SUB_PROPERTY_OF, "ex:p4");
        r.materialize();
        assert!(has(&r, "ex:p1", RDFS_SUB_PROPERTY_OF, "ex:p4"));
    }

    // ── rdfs6 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs6_property_subproperty_self() {
        let mut r = fresh();
        r.add_triple("ex:p", RDF_TYPE, RDF_PROPERTY);
        r.materialize();
        assert!(has(&r, "ex:p", RDFS_SUB_PROPERTY_OF, "ex:p"));
    }

    // ── rdfs7 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs7_subproperty_inheritance() {
        let mut r = fresh();
        r.add_triple("ex:p", RDFS_SUB_PROPERTY_OF, "ex:q");
        r.add_triple("ex:bob", "ex:p", "ex:alice");
        r.materialize();
        assert!(has(&r, "ex:bob", "ex:q", "ex:alice"));
    }

    // ── rdfs8 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs8_class_subclass_of_resource() {
        let mut r = fresh();
        r.add_triple("ex:Person", RDF_TYPE, RDFS_CLASS);
        r.materialize();
        assert!(has(&r, "ex:Person", RDFS_SUB_CLASS_OF, RDFS_RESOURCE));
    }

    // ── rdfs9 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs9_type_inheritance() {
        let mut r = fresh();
        r.add_triple("ex:Person", RDFS_SUB_CLASS_OF, "ex:Agent");
        r.add_triple("ex:bob", RDF_TYPE, "ex:Person");
        r.materialize();
        assert!(has(&r, "ex:bob", RDF_TYPE, "ex:Agent"));
    }

    // ── rdfs10 ────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs10_class_subclass_self() {
        let mut r = fresh();
        r.add_triple("ex:Person", RDF_TYPE, RDFS_CLASS);
        r.materialize();
        assert!(has(&r, "ex:Person", RDFS_SUB_CLASS_OF, "ex:Person"));
    }

    // ── rdfs11 ────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs11_subclass_transitivity() {
        let mut r = fresh();
        r.add_triple("ex:Cat", RDFS_SUB_CLASS_OF, "ex:Animal");
        r.add_triple("ex:Animal", RDFS_SUB_CLASS_OF, "ex:Thing");
        r.materialize();
        assert!(has(&r, "ex:Cat", RDFS_SUB_CLASS_OF, "ex:Thing"));
    }

    #[test]
    fn test_rdfs11_chain_of_three() {
        let mut r = fresh();
        r.add_triple("ex:C1", RDFS_SUB_CLASS_OF, "ex:C2");
        r.add_triple("ex:C2", RDFS_SUB_CLASS_OF, "ex:C3");
        r.add_triple("ex:C3", RDFS_SUB_CLASS_OF, "ex:C4");
        r.materialize();
        assert!(has(&r, "ex:C1", RDFS_SUB_CLASS_OF, "ex:C4"));
    }

    // ── rdfs12 ────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs12_container_membership_property() {
        let mut r = fresh();
        r.add_triple("rdf:_1", RDF_TYPE, RDFS_CONTAINER_MEMBERSHIP_PROPERTY);
        r.materialize();
        assert!(has(&r, "rdf:_1", RDFS_SUB_PROPERTY_OF, RDFS_MEMBER));
    }

    // ── rdfs13 ────────────────────────────────────────────────────────────

    #[test]
    fn test_rdfs13_datatype_subclass_literal() {
        let mut r = fresh();
        r.add_triple("xsd:integer", RDF_TYPE, RDFS_DATATYPE);
        r.materialize();
        assert!(has(&r, "xsd:integer", RDFS_SUB_CLASS_OF, RDFS_LITERAL));
    }

    // ── Combined rules ────────────────────────────────────────────────────

    #[test]
    fn test_combined_domain_range_type() {
        let mut r = fresh();
        r.add_triple("ex:age", RDFS_DOMAIN, "ex:Person");
        r.add_triple("ex:age", RDFS_RANGE, "xsd:integer");
        r.add_triple("ex:alice", "ex:age", "42");
        r.materialize();
        assert!(has(&r, "ex:alice", RDF_TYPE, "ex:Person"));
        assert!(has(&r, "42", RDF_TYPE, "xsd:integer"));
    }

    #[test]
    fn test_combined_subclass_and_domain() {
        let mut r = fresh();
        r.add_triple("ex:Employee", RDFS_SUB_CLASS_OF, "ex:Person");
        r.add_triple("ex:worksFor", RDFS_DOMAIN, "ex:Employee");
        r.add_triple("ex:bob", "ex:worksFor", "ex:Acme");
        r.materialize();
        // rdfs2: bob is Employee
        assert!(has(&r, "ex:bob", RDF_TYPE, "ex:Employee"));
        // rdfs9: bob is Person (via subClassOf)
        assert!(has(&r, "ex:bob", RDF_TYPE, "ex:Person"));
    }

    #[test]
    fn test_combined_subproperty_and_domain() {
        let mut r = fresh();
        r.add_triple("ex:hasMother", RDFS_SUB_PROPERTY_OF, "ex:hasParent");
        r.add_triple("ex:hasParent", RDFS_DOMAIN, "ex:Person");
        r.add_triple("ex:alice", "ex:hasMother", "ex:eve");
        r.materialize();
        // rdfs7: alice hasMother → alice hasParent eve
        assert!(has(&r, "ex:alice", "ex:hasParent", "ex:eve"));
        // rdfs2: alice is Person (via hasParent domain)
        assert!(has(&r, "ex:alice", RDF_TYPE, "ex:Person"));
    }

    // ── Empty graph ───────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph_materializes_nothing() {
        let mut r = fresh();
        let n = r.materialize();
        assert_eq!(n, 0);
        assert_eq!(r.inferred_count(), 0);
    }

    // ── Query ─────────────────────────────────────────────────────────────

    #[test]
    fn test_query_by_subject() {
        let mut r = fresh();
        r.add_triple("ex:alice", RDF_TYPE, "ex:Person");
        r.add_triple("ex:bob", RDF_TYPE, "ex:Person");
        let results = r.query(Some("ex:alice"), None, None);
        assert!(!results.is_empty());
        assert!(results.iter().all(|t| t.subject == "ex:alice"));
    }

    #[test]
    fn test_query_by_predicate() {
        let mut r = fresh();
        r.add_triple("ex:alice", RDF_TYPE, "ex:Person");
        r.add_triple("ex:alice", "ex:name", "Alice");
        let results = r.query(None, Some(RDF_TYPE), None);
        assert!(!results.is_empty());
        assert!(results.iter().all(|t| t.predicate == RDF_TYPE));
    }

    #[test]
    fn test_query_by_object() {
        let mut r = fresh();
        r.add_triple("ex:alice", RDF_TYPE, "ex:Person");
        r.add_triple("ex:bob", RDF_TYPE, "ex:Animal");
        let results = r.query(None, None, Some("ex:Person"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "ex:alice");
    }

    #[test]
    fn test_query_full_match() {
        let mut r = fresh();
        r.add_triple("ex:alice", RDF_TYPE, "ex:Person");
        let results = r.query(Some("ex:alice"), Some(RDF_TYPE), Some("ex:Person"));
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_no_match() {
        let mut r = fresh();
        r.add_triple("ex:alice", RDF_TYPE, "ex:Person");
        let results = r.query(Some("ex:bob"), None, None);
        assert!(results.is_empty());
    }

    // ── Deduplication ─────────────────────────────────────────────────────

    #[test]
    fn test_no_duplicate_triples() {
        let mut r = fresh();
        r.add_triple("ex:p", RDFS_DOMAIN, "ex:C");
        r.add_triple("ex:p", RDFS_DOMAIN, "ex:C"); // duplicate
        r.add_triple("ex:alice", "ex:p", "ex:x");
        r.materialize();
        let results = r.query(Some("ex:alice"), Some(RDF_TYPE), Some("ex:C"));
        assert_eq!(results.len(), 1);
    }

    // ── Inferred count ────────────────────────────────────────────────────

    #[test]
    fn test_inferred_count_increases() {
        let mut r = fresh();
        r.add_triple("ex:alice", "ex:knows", "ex:bob");
        let n = r.materialize();
        assert!(n > 0);
        assert_eq!(r.inferred_count(), n);
    }

    #[test]
    fn test_double_materialize_stable() {
        let mut r = fresh();
        r.add_triple("ex:alice", "ex:knows", "ex:bob");
        r.materialize();
        let count1 = r.inferred_count();
        let n2 = r.materialize();
        assert_eq!(n2, 0); // no new triples
        assert_eq!(r.inferred_count(), count1);
    }

    // ── rdf:Statement (from rdf1 on rdf:type itself) ──────────────────────

    #[test]
    fn test_rdf1_type_predicate_is_property() {
        let mut r = fresh();
        r.add_triple("ex:x", RDF_TYPE, "ex:C");
        r.materialize();
        // rdf:type used as predicate → rdf:type rdf:type rdf:Property
        assert!(has(&r, RDF_TYPE, RDF_TYPE, RDF_PROPERTY));
    }

    // ── rdfs:subClassOf reflexivity from rdfs10 ───────────────────────────

    #[test]
    fn test_rdfs10_direct() {
        let mut r = fresh();
        r.add_triple("ex:Foo", RDF_TYPE, RDFS_CLASS);
        r.materialize();
        assert!(has(&r, "ex:Foo", RDFS_SUB_CLASS_OF, "ex:Foo"));
    }

    // ── rdfs:Datatype → Literal via rdfs13 + rdfs9 ───────────────────────

    #[test]
    fn test_rdfs13_plus_rdfs9() {
        let mut r = fresh();
        r.add_triple("xsd:string", RDF_TYPE, RDFS_DATATYPE);
        r.add_triple("ex:v", RDF_TYPE, "xsd:string");
        r.materialize();
        // xsd:string rdfs:subClassOf rdfs:Literal (rdfs13)
        assert!(has(&r, "xsd:string", RDFS_SUB_CLASS_OF, RDFS_LITERAL));
        // ex:v rdf:type rdfs:Literal (rdfs9)
        assert!(has(&r, "ex:v", RDF_TYPE, RDFS_LITERAL));
    }
}
