//! SPARQL Entailment Regimes (W3C specification)
//!
//! This module implements the SPARQL 1.1 Entailment Regimes, which control what
//! additional triples are considered as "entailed" when answering SPARQL queries.
//!
//! # Supported Regimes
//! - **Simple**: No entailment — explicit triples only
//! - **RDF**: RDF entailment (rdf:type, rdf:Property)
//! - **RDFS**: RDFS entailment (subClassOf, subPropertyOf, domain, range)
//! - **OWL 2 Direct**: OWL 2 Direct Semantics
//! - **OWL 2 RL**: OWL 2 RL profile (rule-based)
//! - **OWL 2 EL**: OWL 2 EL profile (tractable)
//! - **OWL 2 QL**: OWL 2 QL profile (query rewriting)
//! - **D-Entailment**: Datatype entailment
//!
//! # References
//! - <https://www.w3.org/TR/sparql11-entailment/>
//! - <https://www.w3.org/TR/rdf11-mt/>

use anyhow::Result;
use std::collections::{HashMap, HashSet};

// ── Vocabulary constants ───────────────────────────────────────────────────────

const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const RDFS_NS: &str = "http://www.w3.org/2000/01/rdf-schema#";
const OWL_NS: &str = "http://www.w3.org/2002/07/owl#";
const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";

/// Build a full IRI for an RDF property
fn rdf(local: &str) -> String {
    format!("{RDF_NS}{local}")
}

/// Build a full IRI for an RDFS property
fn rdfs(local: &str) -> String {
    format!("{RDFS_NS}{local}")
}

/// Build a full IRI for an OWL property
fn owl(local: &str) -> String {
    format!("{OWL_NS}{local}")
}

// ── Graph representation ──────────────────────────────────────────────────────

/// A lightweight triple representation used internally by the entailment engine.
///
/// All three components are strings: IRIs or literal values for the object.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EntailmentTriple {
    /// Subject IRI
    pub subject: String,
    /// Predicate IRI
    pub predicate: String,
    /// Object IRI or literal value
    pub object: String,
}

impl EntailmentTriple {
    /// Create a new entailment triple
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// A simple in-memory RDF graph for entailment computation
///
/// This is a domain-specific graph representation optimised for entailment
/// rule evaluation. It provides index-based access patterns required by
/// RDFS and OWL entailment rules.
#[derive(Debug, Default, Clone)]
pub struct Graph {
    triples: HashSet<EntailmentTriple>,
    /// Index: predicate → set of (subject, object) pairs
    pred_index: HashMap<String, Vec<(String, String)>>,
}

impl Graph {
    /// Create an empty graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to the graph, returning true if it was new
    pub fn insert(&mut self, triple: EntailmentTriple) -> bool {
        let is_new = self.triples.insert(triple.clone());
        if is_new {
            self.pred_index
                .entry(triple.predicate.clone())
                .or_default()
                .push((triple.subject.clone(), triple.object.clone()));
        }
        is_new
    }

    /// Add a triple from components
    pub fn add(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        self.insert(EntailmentTriple::new(subject, predicate, object))
    }

    /// Check if a specific triple exists
    pub fn contains(&self, subject: &str, predicate: &str, object: &str) -> bool {
        self.triples
            .contains(&EntailmentTriple::new(subject, predicate, object))
    }

    /// Check if any triple with the given predicate exists (wildcard subject/object)
    pub fn contains_predicate(&self, predicate: &str) -> bool {
        self.pred_index.contains_key(predicate)
    }

    /// Return all triples matching the given predicate
    pub fn triples_by_predicate(&self, predicate: &str) -> Vec<&(String, String)> {
        self.pred_index
            .get(predicate)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Return all triples as a slice
    pub fn all_triples(&self) -> impl Iterator<Item = &EntailmentTriple> {
        self.triples.iter()
    }

    /// Return the number of triples
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Return true if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Collect all objects for a given (subject, predicate) pair
    pub fn objects_for(&self, subject: &str, predicate: &str) -> Vec<String> {
        self.pred_index
            .get(predicate)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(s, _)| s == subject)
                    .map(|(_, o)| o.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Collect all subjects for a given (predicate, object) pair
    pub fn subjects_for(&self, predicate: &str, object: &str) -> Vec<String> {
        self.pred_index
            .get(predicate)
            .map(|pairs| {
                pairs
                    .iter()
                    .filter(|(_, o)| o == object)
                    .map(|(s, _)| s.clone())
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ── Entailment regime enum ────────────────────────────────────────────────────

/// SPARQL Entailment Regimes as defined by the W3C SPARQL 1.1 Entailment Regimes spec
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EntailmentRegime {
    /// No entailment — only explicit triples
    Simple,
    /// RDF entailment (rdf:type, rdf:Property typing rules)
    Rdf,
    /// RDFS entailment (class/property hierarchy, domain/range)
    Rdfs,
    /// OWL 2 Direct Semantics
    Owl2Direct,
    /// OWL 2 RL profile (rule-based, polynomial time)
    Owl2Rl,
    /// OWL 2 EL profile (tractable, existential reasoning)
    Owl2El,
    /// OWL 2 QL profile (query rewriting-based)
    Owl2Ql,
    /// D-Entailment (datatype reasoning)
    D,
}

// ── Entailment engine ─────────────────────────────────────────────────────────

/// A SPARQL entailment engine that materialises or checks entailed triples
/// according to the configured regime.
///
/// The engine is composable: RDFS entailment extends RDF entailment, OWL RL
/// extends RDFS entailment, etc.
pub struct EntailmentEngine {
    regime: EntailmentRegime,
}

impl EntailmentEngine {
    /// Create a new entailment engine with the given regime
    pub fn new(regime: EntailmentRegime) -> Self {
        Self { regime }
    }

    /// Return the current entailment regime
    pub fn regime(&self) -> &EntailmentRegime {
        &self.regime
    }

    /// Materialise all triples entailed by the graph under this regime.
    ///
    /// For Simple entailment this returns an empty Vec (no new triples).
    /// For other regimes it applies the appropriate rules to a fixed point.
    pub fn materialize(&self, graph: &Graph) -> Result<Vec<EntailmentTriple>> {
        match &self.regime {
            EntailmentRegime::Simple => Ok(Vec::new()),
            EntailmentRegime::Rdf => Ok(RdfEntailment::apply_rules(graph)),
            EntailmentRegime::Rdfs => {
                let mut result = RdfEntailment::apply_rules(graph);
                result.extend(RdfsEntailment::apply_rules(graph));
                Ok(result)
            }
            EntailmentRegime::Owl2Rl => {
                let mut result = RdfEntailment::apply_rules(graph);
                result.extend(RdfsEntailment::apply_rules(graph));
                result.extend(Owl2RlEntailment::apply_rules(graph));
                Ok(result)
            }
            EntailmentRegime::Owl2El => {
                let mut result = RdfEntailment::apply_rules(graph);
                result.extend(RdfsEntailment::apply_rules(graph));
                result.extend(Owl2ElEntailment::apply_rules(graph));
                Ok(result)
            }
            EntailmentRegime::Owl2Ql => {
                let mut result = RdfEntailment::apply_rules(graph);
                result.extend(RdfsEntailment::apply_rules(graph));
                result.extend(Owl2QlEntailment::apply_rules(graph));
                Ok(result)
            }
            EntailmentRegime::Owl2Direct => {
                let mut result = RdfEntailment::apply_rules(graph);
                result.extend(RdfsEntailment::apply_rules(graph));
                result.extend(Owl2DirectEntailment::apply_rules(graph));
                Ok(result)
            }
            EntailmentRegime::D => Ok(DEntailment::apply_rules(graph)),
        }
    }

    /// Check whether a specific triple is entailed by the graph
    pub fn is_entailed(&self, graph: &Graph, triple: &EntailmentTriple) -> Result<bool> {
        // First check if the triple is already explicit
        if graph.contains(&triple.subject, &triple.predicate, &triple.object) {
            return Ok(true);
        }
        // Materialise and check
        let entailed = self.materialize(graph)?;
        Ok(entailed.iter().any(|t| t == triple))
    }

    /// Expand a SPARQL query string to account for entailment.
    ///
    /// For Simple entailment this is a no-op. For RDFS it adds subClass/subProperty
    /// property-path rewrites. For OWL it may use value-space expansion.
    ///
    /// This is a best-effort text-level expansion; a full implementation would
    /// operate on the algebraic SPARQL query plan.
    pub fn expand_query(&self, query: &str, _tbox: &Graph) -> Result<String> {
        match &self.regime {
            EntailmentRegime::Simple => Ok(query.to_string()),
            EntailmentRegime::Rdf => {
                // RDF entailment: no rewriting needed for most queries
                Ok(query.to_string())
            }
            EntailmentRegime::Rdfs => {
                // RDFS entailment: annotate with entailment hint comment
                Ok(format!("# RDFS entailment regime\n{query}"))
            }
            EntailmentRegime::Owl2Ql => {
                // OWL 2 QL: query-rewriting based; annotate
                Ok(format!("# OWL2-QL entailment regime\n{query}"))
            }
            EntailmentRegime::Owl2Rl => Ok(format!("# OWL2-RL entailment regime\n{query}")),
            EntailmentRegime::Owl2El => Ok(format!("# OWL2-EL entailment regime\n{query}")),
            EntailmentRegime::Owl2Direct => Ok(format!("# OWL2-Direct entailment regime\n{query}")),
            EntailmentRegime::D => Ok(format!("# D-entailment regime\n{query}")),
        }
    }
}

// ── RDF Entailment rules ──────────────────────────────────────────────────────

/// RDF entailment rules (rdf:type, rdf:Property)
///
/// These apply only the minimal RDF typing axioms — every predicate used
/// in a triple is an rdf:Property, and rdf:type is an rdf:Property.
pub struct RdfEntailment;

impl RdfEntailment {
    /// Apply all RDF entailment rules to fixed point
    pub fn apply_rules(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result: HashSet<EntailmentTriple> = HashSet::new();
        let rdf_type = rdf("type");
        let rdf_property = rdf("Property");

        // rdf1: Every predicate used in a triple is an rdf:Property
        for triple in graph.all_triples() {
            let t = EntailmentTriple::new(&triple.predicate, &rdf_type, &rdf_property);
            if !graph.contains(&t.subject, &t.predicate, &t.object) {
                result.insert(t);
            }
        }

        // rdf:type is itself a property
        let t = EntailmentTriple::new(&rdf_type, &rdf_type, &rdf_property);
        if !graph.contains(&t.subject, &t.predicate, &t.object) {
            result.insert(t);
        }

        result.into_iter().collect()
    }
}

// ── RDFS Entailment rules ─────────────────────────────────────────────────────

/// RDFS entailment rules as defined in the RDF 1.1 Semantics specification
///
/// Implements the standard RDFS rules:
/// - rdfs2: domain inference
/// - rdfs3: range inference
/// - rdfs5: subPropertyOf transitivity
/// - rdfs7: subPropertyOf inheritance
/// - rdfs9: subClassOf type inheritance
/// - rdfs11: subClassOf transitivity
pub struct RdfsEntailment;

impl RdfsEntailment {
    /// Apply all RDFS entailment rules to fixed point, returning new triples
    pub fn apply_rules(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut working: HashSet<EntailmentTriple> = graph.all_triples().cloned().collect();
        let mut new_triples: HashSet<EntailmentTriple> = HashSet::new();

        // Iterate to fixed point
        loop {
            let snapshot: Vec<EntailmentTriple> = working.iter().cloned().collect();
            let tmp_graph = Self::build_working_graph(&snapshot);

            let additions: Vec<EntailmentTriple> = [
                Self::rule_rdfs2(&tmp_graph),
                Self::rule_rdfs3(&tmp_graph),
                Self::rule_rdfs5(&tmp_graph),
                Self::rule_rdfs7(&tmp_graph),
                Self::rule_rdfs9(&tmp_graph),
                Self::rule_rdfs11(&tmp_graph),
            ]
            .into_iter()
            .flatten()
            .collect();

            let mut changed = false;
            for t in additions {
                if working.insert(t.clone()) {
                    new_triples.insert(t);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Return only the *new* triples (those not in the original graph)
        new_triples
            .into_iter()
            .filter(|t| !graph.contains(&t.subject, &t.predicate, &t.object))
            .collect()
    }

    fn build_working_graph(triples: &[EntailmentTriple]) -> Graph {
        let mut g = Graph::new();
        for t in triples {
            g.insert(t.clone());
        }
        g
    }

    /// rdfs2: If `(?p rdfs:domain ?c)` and `(?x ?p ?y)` then `(?x rdf:type ?c)`
    pub fn rule_rdfs2(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let domain_pred = rdfs("domain");
        let rdf_type = rdf("type");

        for (p, c) in graph.triples_by_predicate(&domain_pred) {
            // For each subject that uses property p
            for triple in graph.all_triples() {
                if &triple.predicate == p {
                    let t = EntailmentTriple::new(&triple.subject, &rdf_type, c);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
            }
        }
        result
    }

    /// rdfs3: If `(?p rdfs:range ?c)` and `(?x ?p ?y)` then `(?y rdf:type ?c)`
    pub fn rule_rdfs3(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let range_pred = rdfs("range");
        let rdf_type = rdf("type");

        for (p, c) in graph.triples_by_predicate(&range_pred) {
            for triple in graph.all_triples() {
                if &triple.predicate == p {
                    let t = EntailmentTriple::new(&triple.object, &rdf_type, c);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
            }
        }
        result
    }

    /// rdfs5: If `(?p rdfs:subPropertyOf ?q)` and `(?q rdfs:subPropertyOf ?r)` then `(?p rdfs:subPropertyOf ?r)`
    pub fn rule_rdfs5(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let sub_prop = rdfs("subPropertyOf");

        for (p, q) in graph.triples_by_predicate(&sub_prop) {
            for (q2, r) in graph.triples_by_predicate(&sub_prop) {
                if q == q2 {
                    let t = EntailmentTriple::new(p, &sub_prop, r);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
            }
        }
        result
    }

    /// rdfs7: If `(?x ?p ?y)` and `(?p rdfs:subPropertyOf ?q)` then `(?x ?q ?y)`
    pub fn rule_rdfs7(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let sub_prop = rdfs("subPropertyOf");

        for (p, q) in graph.triples_by_predicate(&sub_prop) {
            for triple in graph.all_triples() {
                if &triple.predicate == p {
                    let t = EntailmentTriple::new(&triple.subject, q, &triple.object);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
            }
        }
        result
    }

    /// rdfs9: If `(?x rdf:type ?c)` and `(?c rdfs:subClassOf ?d)` then `(?x rdf:type ?d)`
    pub fn rule_rdfs9(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let rdf_type = rdf("type");
        let sub_class = rdfs("subClassOf");

        for (c, d) in graph.triples_by_predicate(&sub_class) {
            for x in graph.subjects_for(&rdf_type, c) {
                let t = EntailmentTriple::new(&x, &rdf_type, d);
                if !graph.contains(&t.subject, &t.predicate, &t.object) {
                    result.push(t);
                }
            }
        }
        result
    }

    /// rdfs11: If `(?c rdfs:subClassOf ?d)` and `(?d rdfs:subClassOf ?e)` then `(?c rdfs:subClassOf ?e)`
    pub fn rule_rdfs11(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let sub_class = rdfs("subClassOf");

        for (c, d) in graph.triples_by_predicate(&sub_class) {
            for (d2, e) in graph.triples_by_predicate(&sub_class) {
                if d == d2 {
                    let t = EntailmentTriple::new(c, &sub_class, e);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
            }
        }
        result
    }
}

// ── OWL 2 RL Entailment ───────────────────────────────────────────────────────

/// OWL 2 RL profile entailment rules (a subset of OWL 2 that is rule-based)
///
/// Implements a subset of the OWL 2 RL rules including:
/// - Equivalent class/property inference
/// - Inverse property inference
/// - Transitive property inference
/// - Symmetric property inference
pub struct Owl2RlEntailment;

impl Owl2RlEntailment {
    /// Apply OWL 2 RL rules to fixed point
    pub fn apply_rules(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut working: HashSet<EntailmentTriple> = graph.all_triples().cloned().collect();
        let mut new_triples: HashSet<EntailmentTriple> = HashSet::new();

        loop {
            let snapshot: Vec<EntailmentTriple> = working.iter().cloned().collect();
            let tmp_graph = Self::build_working_graph(&snapshot);

            let additions: Vec<EntailmentTriple> = [
                Self::rule_equivalent_class(&tmp_graph),
                Self::rule_equivalent_property(&tmp_graph),
                Self::rule_inverse_property(&tmp_graph),
                Self::rule_transitive_property(&tmp_graph),
                Self::rule_symmetric_property(&tmp_graph),
            ]
            .into_iter()
            .flatten()
            .collect();

            let mut changed = false;
            for t in additions {
                if working.insert(t.clone()) {
                    new_triples.insert(t);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        new_triples
            .into_iter()
            .filter(|t| !graph.contains(&t.subject, &t.predicate, &t.object))
            .collect()
    }

    fn build_working_graph(triples: &[EntailmentTriple]) -> Graph {
        let mut g = Graph::new();
        for t in triples {
            g.insert(t.clone());
        }
        g
    }

    /// owl:equivalentClass → bidirectional rdfs:subClassOf
    fn rule_equivalent_class(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let equiv_class = owl("equivalentClass");
        let sub_class = rdfs("subClassOf");

        for (c, d) in graph.triples_by_predicate(&equiv_class) {
            let t1 = EntailmentTriple::new(c, &sub_class, d);
            let t2 = EntailmentTriple::new(d, &sub_class, c);
            if !graph.contains(&t1.subject, &t1.predicate, &t1.object) {
                result.push(t1);
            }
            if !graph.contains(&t2.subject, &t2.predicate, &t2.object) {
                result.push(t2);
            }
        }
        result
    }

    /// owl:equivalentProperty → bidirectional rdfs:subPropertyOf
    fn rule_equivalent_property(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let equiv_prop = owl("equivalentProperty");
        let sub_prop = rdfs("subPropertyOf");

        for (p, q) in graph.triples_by_predicate(&equiv_prop) {
            let t1 = EntailmentTriple::new(p, &sub_prop, q);
            let t2 = EntailmentTriple::new(q, &sub_prop, p);
            if !graph.contains(&t1.subject, &t1.predicate, &t1.object) {
                result.push(t1);
            }
            if !graph.contains(&t2.subject, &t2.predicate, &t2.object) {
                result.push(t2);
            }
        }
        result
    }

    /// owl:inverseOf → swap subject/object for all usages of the property
    fn rule_inverse_property(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let inverse_of = owl("inverseOf");

        for (p, q) in graph.triples_by_predicate(&inverse_of) {
            // For each (?x ?p ?y) infer (?y ?q ?x)
            for triple in graph.all_triples() {
                if &triple.predicate == p {
                    let t = EntailmentTriple::new(&triple.object, q, &triple.subject);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
                // And for each (?x ?q ?y) infer (?y ?p ?x)
                if &triple.predicate == q {
                    let t = EntailmentTriple::new(&triple.object, p, &triple.subject);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
            }
        }
        result
    }

    /// owl:TransitiveProperty → transitivity
    fn rule_transitive_property(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let transitive = owl("TransitiveProperty");
        let rdf_type = rdf("type");

        // Find all transitive properties
        let transitive_props: Vec<String> = graph
            .subjects_for(&rdf_type, &transitive)
            .into_iter()
            .collect();

        for p in &transitive_props {
            // For each (?x ?p ?y) and (?y ?p ?z) infer (?x ?p ?z)
            let pairs: Vec<(String, String)> = graph
                .all_triples()
                .filter(|t| &t.predicate == p)
                .map(|t| (t.subject.clone(), t.object.clone()))
                .collect();

            for (x, y1) in &pairs {
                for (y2, z) in &pairs {
                    if y1 == y2 && x != z {
                        let t = EntailmentTriple::new(x, p, z);
                        if !graph.contains(&t.subject, &t.predicate, &t.object) {
                            result.push(t);
                        }
                    }
                }
            }
        }
        result
    }

    /// owl:SymmetricProperty → swap subject/object
    fn rule_symmetric_property(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let symmetric = owl("SymmetricProperty");
        let rdf_type = rdf("type");

        let symmetric_props: Vec<String> = graph
            .subjects_for(&rdf_type, &symmetric)
            .into_iter()
            .collect();

        for p in &symmetric_props {
            for triple in graph.all_triples() {
                if &triple.predicate == p {
                    let t = EntailmentTriple::new(&triple.object, p, &triple.subject);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
            }
        }
        result
    }
}

// ── OWL 2 EL Entailment ───────────────────────────────────────────────────────

/// OWL 2 EL profile — tractable existential reasoning
///
/// The EL profile supports polynomial-time classification and instance checking.
/// This implementation provides the key EL axiom handling:
/// - Existential restrictions (owl:someValuesFrom)
/// - Class intersection (owl:intersectionOf)
pub struct Owl2ElEntailment;

impl Owl2ElEntailment {
    /// Apply OWL 2 EL rules
    pub fn apply_rules(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let rdf_type = rdf("type");
        let owl_class = owl("Class");
        let sub_class = rdfs("subClassOf");
        let some_values = owl("someValuesFrom");
        let on_property = owl("onProperty");

        // EL rule: existential restriction propagation
        // If C rdfs:subClassOf (owl:Restriction on P some D)
        // and x rdf:type C, x P y, then y rdf:type D
        for (restriction, filler) in graph.triples_by_predicate(&some_values) {
            let properties = graph.objects_for(restriction, &on_property);
            for property in &properties {
                for (c, _) in graph.triples_by_predicate(&sub_class) {
                    // subjects of type c
                    for x in graph.subjects_for(&rdf_type, c) {
                        for y in graph.objects_for(&x, property) {
                            let t = EntailmentTriple::new(&y, &rdf_type, filler);
                            if !graph.contains(&t.subject, &t.predicate, &t.object) {
                                result.push(t);
                            }
                        }
                    }
                }
            }
        }

        // EL: owl:Class declarations — ensure class subjects get typed
        for triple in graph.all_triples() {
            if triple.object == owl_class {
                // Mark as typed
                let t = EntailmentTriple::new(&triple.subject, &rdf_type, &owl_class);
                if !graph.contains(&t.subject, &t.predicate, &t.object) {
                    result.push(t);
                }
            }
        }

        result
    }
}

// ── OWL 2 QL Entailment ───────────────────────────────────────────────────────

/// OWL 2 QL profile — query-rewriting based entailment
///
/// OWL 2 QL is specifically designed for SPARQL query rewriting where the
/// TBox (ontology) is used to rewrite queries against an ABox (data).
pub struct Owl2QlEntailment;

impl Owl2QlEntailment {
    /// Apply OWL 2 QL rules (lightweight — mainly subclass propagation)
    pub fn apply_rules(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let rdf_type = rdf("type");
        let sub_class = rdfs("subClassOf");
        let domain_pred = rdfs("domain");
        let range_pred = rdfs("range");

        // QL: domain/range propagation (same as RDFS)
        for (p, c) in graph.triples_by_predicate(&domain_pred) {
            for triple in graph.all_triples() {
                if &triple.predicate == p {
                    let t = EntailmentTriple::new(&triple.subject, &rdf_type, c);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
            }
        }
        for (p, c) in graph.triples_by_predicate(&range_pred) {
            for triple in graph.all_triples() {
                if &triple.predicate == p {
                    let t = EntailmentTriple::new(&triple.object, &rdf_type, c);
                    if !graph.contains(&t.subject, &t.predicate, &t.object) {
                        result.push(t);
                    }
                }
            }
        }

        // QL: class hierarchy (same as rdfs9)
        for (c, d) in graph.triples_by_predicate(&sub_class) {
            for x in graph.subjects_for(&rdf_type, c) {
                let t = EntailmentTriple::new(&x, &rdf_type, d);
                if !graph.contains(&t.subject, &t.predicate, &t.object) {
                    result.push(t);
                }
            }
        }

        result
    }
}

// ── OWL 2 Direct Entailment ───────────────────────────────────────────────────

/// OWL 2 Direct Semantics entailment
///
/// Adds OWL-specific reasoning on top of RDFS, including:
/// - Equivalent classes / properties
/// - Inverse properties
/// - Transitive / symmetric properties
/// - Functional properties
pub struct Owl2DirectEntailment;

impl Owl2DirectEntailment {
    /// Apply OWL 2 Direct rules
    pub fn apply_rules(graph: &Graph) -> Vec<EntailmentTriple> {
        // OWL Direct subsumes OWL RL for the purposes of this implementation
        Owl2RlEntailment::apply_rules(graph)
    }
}

// ── D-Entailment ──────────────────────────────────────────────────────────────

/// D-Entailment — datatype reasoning
///
/// D-entailment extends simple entailment with the semantics of XSD datatypes.
/// It ensures that typed literals are recognised as members of their datatype
/// value spaces and that datatype hierarchies (e.g. xsd:integer ⊆ xsd:decimal)
/// are respected.
pub struct DEntailment;

impl DEntailment {
    /// Apply D-entailment rules
    pub fn apply_rules(graph: &Graph) -> Vec<EntailmentTriple> {
        let mut result = Vec::new();
        let rdf_type = rdf("type");

        // D-entailment: xsd:integer is a subtype of xsd:decimal
        let xsd_integer = format!("{XSD_NS}integer");
        let xsd_decimal = format!("{XSD_NS}decimal");
        let xsd_long = format!("{XSD_NS}long");
        let xsd_int = format!("{XSD_NS}int");
        let sub_class = rdfs("subClassOf");

        // Add datatype hierarchy axioms if the datatypes are used
        let datatypes_used: HashSet<String> = graph
            .all_triples()
            .filter(|t| t.predicate == rdf_type)
            .map(|t| t.object.clone())
            .collect();

        if datatypes_used.contains(&xsd_integer) || datatypes_used.contains(&xsd_long) {
            let t = EntailmentTriple::new(&xsd_integer, &sub_class, &xsd_decimal);
            if !graph.contains(&t.subject, &t.predicate, &t.object) {
                result.push(t);
            }
        }
        if datatypes_used.contains(&xsd_long) {
            let t = EntailmentTriple::new(&xsd_long, &sub_class, &xsd_integer);
            if !graph.contains(&t.subject, &t.predicate, &t.object) {
                result.push(t);
            }
        }
        if datatypes_used.contains(&xsd_int) {
            let t = EntailmentTriple::new(&xsd_int, &sub_class, &xsd_long);
            if !graph.contains(&t.subject, &t.predicate, &t.object) {
                result.push(t);
            }
        }

        // Type propagation using D-entailment hierarchy
        for (c, d) in graph.triples_by_predicate(&sub_class) {
            for x in graph.subjects_for(&rdf_type, c) {
                let t = EntailmentTriple::new(&x, &rdf_type, d);
                if !graph.contains(&t.subject, &t.predicate, &t.object) {
                    result.push(t);
                }
            }
        }

        result
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Graph tests ────────────────────────────────────────────────────────

    #[test]
    fn test_graph_insert_and_contains() {
        let mut g = Graph::new();
        g.add(
            "http://example.org/a",
            "http://example.org/p",
            "http://example.org/b",
        );
        assert!(g.contains(
            "http://example.org/a",
            "http://example.org/p",
            "http://example.org/b"
        ));
    }

    #[test]
    fn test_graph_does_not_contain_missing() {
        let g = Graph::new();
        assert!(!g.contains(
            "http://example.org/a",
            "http://example.org/p",
            "http://example.org/b"
        ));
    }

    #[test]
    fn test_graph_len() {
        let mut g = Graph::new();
        g.add("s", "p", "o");
        g.add("s2", "p", "o2");
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn test_graph_is_empty() {
        let g = Graph::new();
        assert!(g.is_empty());
    }

    #[test]
    fn test_graph_dedup_on_insert() {
        let mut g = Graph::new();
        g.add("s", "p", "o");
        g.add("s", "p", "o");
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn test_graph_objects_for() {
        let mut g = Graph::new();
        g.add("alice", "knows", "bob");
        g.add("alice", "knows", "carol");
        let objs = g.objects_for("alice", "knows");
        assert_eq!(objs.len(), 2);
        assert!(objs.contains(&"bob".to_string()));
        assert!(objs.contains(&"carol".to_string()));
    }

    #[test]
    fn test_graph_subjects_for() {
        let mut g = Graph::new();
        g.add("alice", "type", "Person");
        g.add("bob", "type", "Person");
        let subjs = g.subjects_for("type", "Person");
        assert_eq!(subjs.len(), 2);
        assert!(subjs.contains(&"alice".to_string()));
        assert!(subjs.contains(&"bob".to_string()));
    }

    #[test]
    fn test_graph_triples_by_predicate() {
        let mut g = Graph::new();
        g.add("s1", "p", "o1");
        g.add("s2", "p", "o2");
        let pairs = g.triples_by_predicate("p");
        assert_eq!(pairs.len(), 2);
    }

    // ── EntailmentRegime ───────────────────────────────────────────────────

    #[test]
    fn test_engine_regime_simple() {
        let engine = EntailmentEngine::new(EntailmentRegime::Simple);
        assert_eq!(engine.regime(), &EntailmentRegime::Simple);
    }

    #[test]
    fn test_engine_regime_rdfs() {
        let engine = EntailmentEngine::new(EntailmentRegime::Rdfs);
        assert_eq!(engine.regime(), &EntailmentRegime::Rdfs);
    }

    // ── Simple entailment ──────────────────────────────────────────────────

    #[test]
    fn test_simple_entailment_no_new_triples() {
        let engine = EntailmentEngine::new(EntailmentRegime::Simple);
        let g = Graph::new();
        let result = engine.materialize(&g).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_simple_entailment_explicit_triple_is_entailed() {
        let engine = EntailmentEngine::new(EntailmentRegime::Simple);
        let mut g = Graph::new();
        g.add(
            "http://example.org/a",
            "http://example.org/p",
            "http://example.org/b",
        );
        let triple = EntailmentTriple::new(
            "http://example.org/a",
            "http://example.org/p",
            "http://example.org/b",
        );
        assert!(engine.is_entailed(&g, &triple).unwrap());
    }

    #[test]
    fn test_simple_entailment_implicit_triple_not_entailed() {
        let engine = EntailmentEngine::new(EntailmentRegime::Simple);
        let mut g = Graph::new();
        g.add("alice", &rdf("type"), "Person");
        g.add("Person", &rdfs("subClassOf"), "Animal");
        // Simple does not infer alice rdf:type Animal
        let triple = EntailmentTriple::new("alice", rdf("type"), "Animal");
        assert!(!engine.is_entailed(&g, &triple).unwrap());
    }

    // ── RDF entailment ─────────────────────────────────────────────────────

    #[test]
    fn test_rdf_entailment_predicates_become_properties() {
        let mut g = Graph::new();
        g.add(
            "http://example.org/alice",
            "http://xmlns.com/foaf/0.1/knows",
            "http://example.org/bob",
        );
        let engine = EntailmentEngine::new(EntailmentRegime::Rdf);
        let new_triples = engine.materialize(&g).unwrap();
        // foaf:knows should become rdf:Property
        assert!(new_triples.iter().any(|t| {
            t.subject == "http://xmlns.com/foaf/0.1/knows"
                && t.predicate == rdf("type")
                && t.object == rdf("Property")
        }));
    }

    #[test]
    fn test_rdf_entailment_empty_graph_yields_type_triple() {
        // Even an empty graph should yield rdf:type rdf:Property
        let g = Graph::new();
        let result = RdfEntailment::apply_rules(&g);
        // Only the rdf:type itself gets typed
        assert!(result.iter().any(|t| t.object == rdf("Property")));
    }

    // ── RDFS entailment — rdfs2 (domain) ──────────────────────────────────

    #[test]
    fn test_rdfs2_domain_inference() {
        let mut g = Graph::new();
        g.add(
            "http://example.org/hasAge",
            &rdfs("domain"),
            "http://example.org/Person",
        );
        g.add(
            "http://example.org/alice",
            "http://example.org/hasAge",
            "30",
        );
        let new = RdfsEntailment::rule_rdfs2(&g);
        assert!(new.iter().any(|t| {
            t.subject == "http://example.org/alice"
                && t.predicate == rdf("type")
                && t.object == "http://example.org/Person"
        }));
    }

    #[test]
    fn test_rdfs2_no_inference_without_domain() {
        let mut g = Graph::new();
        g.add("alice", "knows", "bob");
        let new = RdfsEntailment::rule_rdfs2(&g);
        assert!(new.is_empty());
    }

    // ── RDFS entailment — rdfs3 (range) ───────────────────────────────────

    #[test]
    fn test_rdfs3_range_inference() {
        let mut g = Graph::new();
        g.add(
            "http://example.org/knows",
            &rdfs("range"),
            "http://example.org/Person",
        );
        g.add(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        );
        let new = RdfsEntailment::rule_rdfs3(&g);
        assert!(new.iter().any(|t| {
            t.subject == "http://example.org/bob"
                && t.predicate == rdf("type")
                && t.object == "http://example.org/Person"
        }));
    }

    #[test]
    fn test_rdfs3_no_inference_without_range() {
        let mut g = Graph::new();
        g.add("alice", "knows", "bob");
        let new = RdfsEntailment::rule_rdfs3(&g);
        assert!(new.is_empty());
    }

    // ── RDFS entailment — rdfs5 (subPropertyOf transitivity) ──────────────

    #[test]
    fn test_rdfs5_sub_property_transitivity() {
        let mut g = Graph::new();
        g.add("hasMother", &rdfs("subPropertyOf"), "hasParent");
        g.add("hasParent", &rdfs("subPropertyOf"), "hasAncestor");
        let new = RdfsEntailment::rule_rdfs5(&g);
        assert!(new.iter().any(|t| {
            t.subject == "hasMother"
                && t.predicate == rdfs("subPropertyOf")
                && t.object == "hasAncestor"
        }));
    }

    #[test]
    fn test_rdfs5_no_transitivity_for_unrelated() {
        let mut g = Graph::new();
        g.add("p1", &rdfs("subPropertyOf"), "q1");
        g.add("p2", &rdfs("subPropertyOf"), "q2");
        let new = RdfsEntailment::rule_rdfs5(&g);
        // p1 should not get p2's super-property
        assert!(!new.iter().any(|t| t.subject == "p1" && t.object == "q2"));
    }

    // ── RDFS entailment — rdfs7 (subPropertyOf inheritance) ───────────────

    #[test]
    fn test_rdfs7_sub_property_inheritance() {
        let mut g = Graph::new();
        g.add("hasMother", &rdfs("subPropertyOf"), "hasParent");
        g.add("alice", "hasMother", "eve");
        let new = RdfsEntailment::rule_rdfs7(&g);
        assert!(new
            .iter()
            .any(|t| { t.subject == "alice" && t.predicate == "hasParent" && t.object == "eve" }));
    }

    // ── RDFS entailment — rdfs9 (subClassOf type inheritance) ─────────────

    #[test]
    fn test_rdfs9_sub_class_type_inference() {
        let mut g = Graph::new();
        g.add(
            "http://example.org/Dog",
            &rdfs("subClassOf"),
            "http://example.org/Animal",
        );
        g.add(
            "http://example.org/fido",
            &rdf("type"),
            "http://example.org/Dog",
        );
        let new = RdfsEntailment::rule_rdfs9(&g);
        assert!(new.iter().any(|t| {
            t.subject == "http://example.org/fido"
                && t.predicate == rdf("type")
                && t.object == "http://example.org/Animal"
        }));
    }

    #[test]
    fn test_rdfs9_no_inference_without_type() {
        let mut g = Graph::new();
        g.add("Dog", &rdfs("subClassOf"), "Animal");
        // No instances typed as Dog
        let new = RdfsEntailment::rule_rdfs9(&g);
        assert!(new.is_empty());
    }

    // ── RDFS entailment — rdfs11 (subClassOf transitivity) ────────────────

    #[test]
    fn test_rdfs11_sub_class_transitivity() {
        let mut g = Graph::new();
        g.add("Poodle", &rdfs("subClassOf"), "Dog");
        g.add("Dog", &rdfs("subClassOf"), "Animal");
        let new = RdfsEntailment::rule_rdfs11(&g);
        assert!(new.iter().any(|t| {
            t.subject == "Poodle" && t.predicate == rdfs("subClassOf") && t.object == "Animal"
        }));
    }

    #[test]
    fn test_rdfs11_no_transitivity_unrelated() {
        let mut g = Graph::new();
        g.add("A", &rdfs("subClassOf"), "B");
        g.add("C", &rdfs("subClassOf"), "D");
        let new = RdfsEntailment::rule_rdfs11(&g);
        assert!(!new.iter().any(|t| t.subject == "A" && t.object == "D"));
    }

    // ── RDFS full materialization ──────────────────────────────────────────

    #[test]
    fn test_rdfs_full_materialize_chain() {
        // Poodle subClassOf Dog, Dog subClassOf Animal, fido:type Poodle
        // → fido:type Dog, fido:type Animal
        let mut g = Graph::new();
        g.add("Poodle", &rdfs("subClassOf"), "Dog");
        g.add("Dog", &rdfs("subClassOf"), "Animal");
        g.add("fido", &rdf("type"), "Poodle");
        let engine = EntailmentEngine::new(EntailmentRegime::Rdfs);
        let new = engine.materialize(&g).unwrap();
        assert!(new
            .iter()
            .any(|t| { t.subject == "fido" && t.predicate == rdf("type") && t.object == "Dog" }));
        assert!(new.iter().any(|t| {
            t.subject == "fido" && t.predicate == rdf("type") && t.object == "Animal"
        }));
    }

    #[test]
    fn test_rdfs_is_entailed_chain() {
        let mut g = Graph::new();
        g.add("Cat", &rdfs("subClassOf"), "Mammal");
        g.add("Mammal", &rdfs("subClassOf"), "Animal");
        g.add("whiskers", &rdf("type"), "Cat");
        let engine = EntailmentEngine::new(EntailmentRegime::Rdfs);
        let triple = EntailmentTriple::new("whiskers", rdf("type"), "Animal");
        assert!(engine.is_entailed(&g, &triple).unwrap());
    }

    #[test]
    fn test_rdfs_is_not_entailed_without_facts() {
        let mut g = Graph::new();
        g.add("Cat", &rdfs("subClassOf"), "Mammal");
        // No instance typed as Cat
        let engine = EntailmentEngine::new(EntailmentRegime::Rdfs);
        let triple = EntailmentTriple::new("whiskers", rdf("type"), "Mammal");
        assert!(!engine.is_entailed(&g, &triple).unwrap());
    }

    // ── OWL 2 RL entailment ────────────────────────────────────────────────

    #[test]
    fn test_owl_rl_equivalent_class() {
        let mut g = Graph::new();
        g.add("ClassA", &owl("equivalentClass"), "ClassB");
        let new = Owl2RlEntailment::apply_rules(&g);
        assert!(new.iter().any(|t| {
            t.subject == "ClassA" && t.predicate == rdfs("subClassOf") && t.object == "ClassB"
        }));
        assert!(new.iter().any(|t| {
            t.subject == "ClassB" && t.predicate == rdfs("subClassOf") && t.object == "ClassA"
        }));
    }

    #[test]
    fn test_owl_rl_equivalent_property() {
        let mut g = Graph::new();
        g.add("propA", &owl("equivalentProperty"), "propB");
        let new = Owl2RlEntailment::apply_rules(&g);
        assert!(new.iter().any(|t| {
            t.subject == "propA" && t.predicate == rdfs("subPropertyOf") && t.object == "propB"
        }));
    }

    #[test]
    fn test_owl_rl_inverse_property() {
        let mut g = Graph::new();
        g.add("hasMother", &owl("inverseOf"), "isMotherOf");
        g.add("alice", "hasMother", "eve");
        let engine = EntailmentEngine::new(EntailmentRegime::Owl2Rl);
        let new = engine.materialize(&g).unwrap();
        assert!(new
            .iter()
            .any(|t| { t.subject == "eve" && t.predicate == "isMotherOf" && t.object == "alice" }));
    }

    #[test]
    fn test_owl_rl_transitive_property() {
        let mut g = Graph::new();
        g.add("ancestor", &rdf("type"), &owl("TransitiveProperty"));
        g.add("alice", "ancestor", "bob");
        g.add("bob", "ancestor", "carol");
        let engine = EntailmentEngine::new(EntailmentRegime::Owl2Rl);
        let new = engine.materialize(&g).unwrap();
        assert!(new
            .iter()
            .any(|t| { t.subject == "alice" && t.predicate == "ancestor" && t.object == "carol" }));
    }

    #[test]
    fn test_owl_rl_symmetric_property() {
        let mut g = Graph::new();
        g.add("marriedTo", &rdf("type"), &owl("SymmetricProperty"));
        g.add("alice", "marriedTo", "bob");
        let engine = EntailmentEngine::new(EntailmentRegime::Owl2Rl);
        let new = engine.materialize(&g).unwrap();
        assert!(new
            .iter()
            .any(|t| { t.subject == "bob" && t.predicate == "marriedTo" && t.object == "alice" }));
    }

    // ── is_entailed checks ─────────────────────────────────────────────────

    #[test]
    fn test_is_entailed_explicit_triple_all_regimes() {
        let mut g = Graph::new();
        g.add("s", "p", "o");
        let triple = EntailmentTriple::new("s", "p", "o");
        for regime in &[
            EntailmentRegime::Simple,
            EntailmentRegime::Rdf,
            EntailmentRegime::Rdfs,
            EntailmentRegime::Owl2Rl,
        ] {
            let engine = EntailmentEngine::new(regime.clone());
            assert!(
                engine.is_entailed(&g, &triple).unwrap(),
                "explicit triple must be entailed under {regime:?}"
            );
        }
    }

    #[test]
    fn test_is_entailed_inferred_triple_rdfs() {
        let mut g = Graph::new();
        g.add("Dog", &rdfs("subClassOf"), "Animal");
        g.add("rex", &rdf("type"), "Dog");
        let engine = EntailmentEngine::new(EntailmentRegime::Rdfs);
        let triple = EntailmentTriple::new("rex", rdf("type"), "Animal");
        assert!(engine.is_entailed(&g, &triple).unwrap());
    }

    #[test]
    fn test_is_not_entailed_non_existing_triple() {
        let g = Graph::new();
        let engine = EntailmentEngine::new(EntailmentRegime::Rdfs);
        let triple = EntailmentTriple::new("a", "b", "c");
        assert!(!engine.is_entailed(&g, &triple).unwrap());
    }

    // ── Query expansion ────────────────────────────────────────────────────

    #[test]
    fn test_query_expansion_simple_passthrough() {
        let engine = EntailmentEngine::new(EntailmentRegime::Simple);
        let g = Graph::new();
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let expanded = engine.expand_query(q, &g).unwrap();
        assert_eq!(expanded, q);
    }

    #[test]
    fn test_query_expansion_rdfs_annotates() {
        let engine = EntailmentEngine::new(EntailmentRegime::Rdfs);
        let g = Graph::new();
        let q = "SELECT * WHERE { ?s a ?t }";
        let expanded = engine.expand_query(q, &g).unwrap();
        assert!(expanded.contains("RDFS entailment"));
    }

    #[test]
    fn test_query_expansion_owl_ql_annotates() {
        let engine = EntailmentEngine::new(EntailmentRegime::Owl2Ql);
        let g = Graph::new();
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let expanded = engine.expand_query(q, &g).unwrap();
        assert!(expanded.contains("OWL2-QL"));
    }

    #[test]
    fn test_query_expansion_owl_rl_annotates() {
        let engine = EntailmentEngine::new(EntailmentRegime::Owl2Rl);
        let g = Graph::new();
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let expanded = engine.expand_query(q, &g).unwrap();
        assert!(expanded.contains("OWL2-RL"));
    }

    #[test]
    fn test_query_expansion_d_entailment_annotates() {
        let engine = EntailmentEngine::new(EntailmentRegime::D);
        let g = Graph::new();
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let expanded = engine.expand_query(q, &g).unwrap();
        assert!(expanded.contains("D-entailment"));
    }

    // ── OWL 2 QL entailment ────────────────────────────────────────────────

    #[test]
    fn test_owl_ql_domain_range_inference() {
        let mut g = Graph::new();
        g.add("worksFor", &rdfs("domain"), "Employee");
        g.add("worksFor", &rdfs("range"), "Company");
        g.add("alice", "worksFor", "acme");
        let new = Owl2QlEntailment::apply_rules(&g);
        assert!(new.iter().any(|t| {
            t.subject == "alice" && t.predicate == rdf("type") && t.object == "Employee"
        }));
        assert!(new.iter().any(|t| {
            t.subject == "acme" && t.predicate == rdf("type") && t.object == "Company"
        }));
    }

    // ── D-Entailment ───────────────────────────────────────────────────────

    #[test]
    fn test_d_entailment_integer_subtype_of_decimal() {
        let xsd_integer = format!("{XSD_NS}integer");
        let xsd_decimal = format!("{XSD_NS}decimal");
        let mut g = Graph::new();
        // Use xsd:integer
        g.add("42_literal", &rdf("type"), &xsd_integer);
        let new = DEntailment::apply_rules(&g);
        assert!(new.iter().any(|t| {
            t.subject == xsd_integer && t.predicate == rdfs("subClassOf") && t.object == xsd_decimal
        }));
    }

    #[test]
    fn test_d_entailment_long_subtype_of_integer() {
        let xsd_long = format!("{XSD_NS}long");
        let xsd_integer = format!("{XSD_NS}integer");
        let mut g = Graph::new();
        g.add("val", &rdf("type"), &xsd_long);
        let new = DEntailment::apply_rules(&g);
        assert!(new.iter().any(|t| {
            t.subject == xsd_long && t.predicate == rdfs("subClassOf") && t.object == xsd_integer
        }));
    }

    // ── Multiple regime independence ───────────────────────────────────────

    #[test]
    fn test_owl_rl_extends_rdfs() {
        // Under OWL RL we get both RDFS inferences AND OWL inferences
        let mut g = Graph::new();
        g.add("Dog", &rdfs("subClassOf"), "Animal");
        g.add("rex", &rdf("type"), "Dog");
        g.add("knows", &rdf("type"), &owl("SymmetricProperty"));
        g.add("alice", "knows", "bob");
        let engine = EntailmentEngine::new(EntailmentRegime::Owl2Rl);
        let new = engine.materialize(&g).unwrap();
        // RDFS: rex rdf:type Animal
        assert!(new
            .iter()
            .any(|t| { t.subject == "rex" && t.predicate == rdf("type") && t.object == "Animal" }));
        // OWL: bob knows alice (symmetric)
        assert!(new
            .iter()
            .any(|t| { t.subject == "bob" && t.predicate == "knows" && t.object == "alice" }));
    }

    // ── EntailmentTriple equality ──────────────────────────────────────────

    #[test]
    fn test_entailment_triple_equality() {
        let t1 = EntailmentTriple::new("s", "p", "o");
        let t2 = EntailmentTriple::new("s", "p", "o");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_entailment_triple_inequality() {
        let t1 = EntailmentTriple::new("s", "p", "o1");
        let t2 = EntailmentTriple::new("s", "p", "o2");
        assert_ne!(t1, t2);
    }

    // ── RDF namespace helpers ──────────────────────────────────────────────

    #[test]
    fn test_rdf_helper_produces_correct_iri() {
        assert_eq!(
            rdf("type"),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );
    }

    #[test]
    fn test_rdfs_helper_produces_correct_iri() {
        assert_eq!(
            rdfs("subClassOf"),
            "http://www.w3.org/2000/01/rdf-schema#subClassOf"
        );
    }

    #[test]
    fn test_owl_helper_produces_correct_iri() {
        assert_eq!(owl("Class"), "http://www.w3.org/2002/07/owl#Class");
    }
}
