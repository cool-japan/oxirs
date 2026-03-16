//! # SKOS Entailment Rules
//!
//! Implements SKOS (Simple Knowledge Organization System) entailment rules
//! as defined in the W3C SKOS Reference specification.
//!
//! ## Reference
//! - <https://www.w3.org/TR/skos-reference/>
//! - <https://www.w3.org/TR/skos-primer/>
//!
//! ## Rules Implemented
//!
//! | ID | Rule | Description |
//! |----|------|-------------|
//! | S1 | broaderTransitive transitivity | broaderTransitive chain propagation |
//! | S2 | narrowerTransitive transitivity | narrowerTransitive chain propagation |
//! | S3 | broader/narrower symmetry | broader implies narrower and vice-versa |
//! | S4 | related symmetry | related is symmetric |
//! | S5 | topConceptOf/hasTopConcept symmetry | inverse relationship |
//! | S6 | exactMatch reflexive, symmetric, transitive | equivalence-like mapping |
//! | S7 | inScheme via topConceptOf | topConceptOf implies inScheme |
//! | S8 | broader implies broaderTransitive | broader/narrower lift to transitive closure |
//! | S9 | closeMatch symmetry | closeMatch is symmetric |
//! | S10 | broader/narrower from match relations | broadMatch implies broadMatch inverse |

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// SKOS vocabulary constants (mirrors oxirs-core::vocab::skos)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
const SKOS_NS: &str = "http://www.w3.org/2004/02/skos/core#";

const BROADER: &str = "http://www.w3.org/2004/02/skos/core#broader";
const NARROWER: &str = "http://www.w3.org/2004/02/skos/core#narrower";
const BROADER_TRANSITIVE: &str = "http://www.w3.org/2004/02/skos/core#broaderTransitive";
const NARROWER_TRANSITIVE: &str = "http://www.w3.org/2004/02/skos/core#narrowerTransitive";
const RELATED: &str = "http://www.w3.org/2004/02/skos/core#related";
const EXACT_MATCH: &str = "http://www.w3.org/2004/02/skos/core#exactMatch";
const CLOSE_MATCH: &str = "http://www.w3.org/2004/02/skos/core#closeMatch";
const BROAD_MATCH: &str = "http://www.w3.org/2004/02/skos/core#broadMatch";
const NARROW_MATCH: &str = "http://www.w3.org/2004/02/skos/core#narrowMatch";
const IN_SCHEME: &str = "http://www.w3.org/2004/02/skos/core#inScheme";
const HAS_TOP_CONCEPT: &str = "http://www.w3.org/2004/02/skos/core#hasTopConcept";
const TOP_CONCEPT_OF: &str = "http://www.w3.org/2004/02/skos/core#topConceptOf";
const PREF_LABEL: &str = "http://www.w3.org/2004/02/skos/core#prefLabel";
const ALT_LABEL: &str = "http://www.w3.org/2004/02/skos/core#altLabel";

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A triple using owned string values: (subject IRI, predicate IRI, object IRI or literal)
pub type Triple = (String, String, String);

/// Named node (IRI) string alias for clarity
pub type NamedNode = String;

/// Error type for SKOS reasoning operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum SkosError {
    #[error("Cycle detected in SKOS hierarchy involving concept: {0}")]
    CycleDetected(String),
    #[error("Invalid SKOS graph: {0}")]
    InvalidGraph(String),
    #[error("Maximum iteration limit ({0}) exceeded during entailment")]
    MaxIterationsExceeded(usize),
}

/// Result alias for SKOS operations
pub type SkosResult<T> = Result<T, SkosError>;

// ---------------------------------------------------------------------------
// Graph — a simple in-memory triple store for SKOS reasoning
// ---------------------------------------------------------------------------

/// A lightweight in-memory RDF graph sufficient for SKOS entailment.
///
/// Triples are stored as (subject, predicate, object) string tuples.
/// Subject and object are IRIs; literals are represented as plain strings
/// (for labels, the value is the lexical form, optionally with a `@lang` suffix).
#[derive(Debug, Clone, Default)]
pub struct Graph {
    triples: HashSet<Triple>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            triples: HashSet::new(),
        }
    }

    /// Add a triple to the graph.  Returns `true` if the triple is new.
    pub fn add_triple(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> bool {
        self.triples
            .insert((subject.into(), predicate.into(), object.into()))
    }

    /// Test membership of a triple.
    pub fn contains(&self, subject: &str, predicate: &str, object: &str) -> bool {
        self.triples
            .contains(&(subject.to_owned(), predicate.to_owned(), object.to_owned()))
    }

    /// Iterate over all triples with the given predicate.
    pub fn triples_with_predicate<'a>(
        &'a self,
        predicate: &str,
    ) -> impl Iterator<Item = &'a Triple> {
        let predicate = predicate.to_string();
        self.triples.iter().filter(move |(_, p, _)| p == &predicate)
    }

    /// Iterate over all triples whose subject matches `subject`.
    pub fn triples_with_subject<'a>(&'a self, subject: &str) -> impl Iterator<Item = &'a Triple> {
        let subject = subject.to_string();
        self.triples.iter().filter(move |(s, _, _)| s == &subject)
    }

    /// Iterate over all triples whose object matches `object`.
    pub fn triples_with_object<'a>(&'a self, object: &str) -> impl Iterator<Item = &'a Triple> {
        let object = object.to_string();
        self.triples.iter().filter(move |(_, _, o)| o == &object)
    }

    /// Return all triples.
    pub fn triples(&self) -> impl Iterator<Item = &Triple> {
        self.triples.iter()
    }

    /// Number of triples in the graph.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Returns `true` if the graph has no triples.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Merge another graph's triples into this one.
    pub fn merge(&mut self, other: &Graph) {
        for triple in &other.triples {
            self.triples.insert(triple.clone());
        }
    }

    /// Return an adjacency map for a specific predicate: subject → set of objects.
    pub fn adjacency_map(&self, predicate: &str) -> HashMap<String, HashSet<String>> {
        let mut map: HashMap<String, HashSet<String>> = HashMap::new();
        for (s, p, o) in &self.triples {
            if p == predicate {
                map.entry(s.clone()).or_default().insert(o.clone());
            }
        }
        map
    }
}

// ---------------------------------------------------------------------------
// SkosReasoner
// ---------------------------------------------------------------------------

/// Implements SKOS entailment rules per W3C SKOS Reference §8.
///
/// Each rule method computes the set of *new* triples entailed under that rule
/// alone, given the current graph state.  [`SkosReasoner::apply_rules`] runs
/// all rules to fixpoint.
pub struct SkosReasoner;

impl SkosReasoner {
    /// Apply all SKOS entailment rules to `graph` until no new triples are
    /// derived (fixpoint iteration).
    ///
    /// Returns the set of all newly entailed triples (not present in the
    /// original graph).
    ///
    /// # Errors
    /// Returns [`SkosError::MaxIterationsExceeded`] if fixpoint is not reached
    /// within 1 000 iterations (guards against unexpected cycles in the rule
    /// application order).
    pub fn apply_rules(graph: &Graph) -> SkosResult<Vec<Triple>> {
        const MAX_ITER: usize = 1_000;

        let mut working = graph.clone();
        let mut all_new: HashSet<Triple> = HashSet::new();

        for _iter in 0..MAX_ITER {
            let batch: Vec<Triple> = [
                Self::rule_broader_to_transitive(&working),
                Self::rule_narrower_to_transitive(&working),
                Self::rule_broader_transitive_chain(&working),
                Self::rule_narrower_transitive_chain(&working),
                Self::rule_broader_narrower_symmetry(&working),
                Self::rule_related_symmetry(&working),
                Self::rule_top_concept_symmetry(&working),
                Self::rule_exact_match_symmetry(&working),
                Self::rule_exact_match_transitivity(&working),
                Self::rule_close_match_symmetry(&working),
                Self::rule_in_scheme_via_top_concept(&working),
                Self::rule_broad_match_narrow_match_inverse(&working),
            ]
            .into_iter()
            .flatten()
            .collect();

            let mut added = false;
            for triple in batch {
                if !working.triples.contains(&triple) {
                    all_new.insert(triple.clone());
                    working.triples.insert(triple);
                    added = true;
                }
            }

            if !added {
                return Ok(all_new.into_iter().collect());
            }
        }

        Err(SkosError::MaxIterationsExceeded(MAX_ITER))
    }

    // -----------------------------------------------------------------------
    // S8: broader → broaderTransitive  (and narrower → narrowerTransitive)
    // -----------------------------------------------------------------------

    /// Rule S8a: `X skos:broader Y` ⟹ `X skos:broaderTransitive Y`
    pub fn rule_broader_to_transitive(graph: &Graph) -> Vec<Triple> {
        graph
            .triples_with_predicate(BROADER)
            .map(|(s, _, o)| (s.clone(), BROADER_TRANSITIVE.to_owned(), o.clone()))
            .collect()
    }

    /// Rule S8b: `X skos:narrower Y` ⟹ `X skos:narrowerTransitive Y`
    pub fn rule_narrower_to_transitive(graph: &Graph) -> Vec<Triple> {
        graph
            .triples_with_predicate(NARROWER)
            .map(|(s, _, o)| (s.clone(), NARROWER_TRANSITIVE.to_owned(), o.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // S1: broaderTransitive is transitive
    // -----------------------------------------------------------------------

    /// Rule S1: If `X skos:broaderTransitive Y` and `Y skos:broaderTransitive Z`
    /// then `X skos:broaderTransitive Z`.
    pub fn rule_broader_transitive_chain(graph: &Graph) -> Vec<Triple> {
        let bt_map = graph.adjacency_map(BROADER_TRANSITIVE);
        let mut new_triples = Vec::new();
        for (x, y_set) in &bt_map {
            for y in y_set {
                if let Some(z_set) = bt_map.get(y) {
                    for z in z_set {
                        if z != x {
                            new_triples.push((x.clone(), BROADER_TRANSITIVE.to_owned(), z.clone()));
                        }
                    }
                }
            }
        }
        new_triples
    }

    // -----------------------------------------------------------------------
    // S2: narrowerTransitive is transitive
    // -----------------------------------------------------------------------

    /// Rule S2: If `X skos:narrowerTransitive Y` and `Y skos:narrowerTransitive Z`
    /// then `X skos:narrowerTransitive Z`.
    pub fn rule_narrower_transitive_chain(graph: &Graph) -> Vec<Triple> {
        let nt_map = graph.adjacency_map(NARROWER_TRANSITIVE);
        let mut new_triples = Vec::new();
        for (x, y_set) in &nt_map {
            for y in y_set {
                if let Some(z_set) = nt_map.get(y) {
                    for z in z_set {
                        if z != x {
                            new_triples.push((
                                x.clone(),
                                NARROWER_TRANSITIVE.to_owned(),
                                z.clone(),
                            ));
                        }
                    }
                }
            }
        }
        new_triples
    }

    // -----------------------------------------------------------------------
    // S3: broader/narrower inverse symmetry
    // -----------------------------------------------------------------------

    /// Rule S3: If `X skos:broader Y` then `Y skos:narrower X`, and vice-versa.
    pub fn rule_broader_narrower_symmetry(graph: &Graph) -> Vec<Triple> {
        let mut new_triples: Vec<Triple> = Vec::new();

        // broader → narrower
        for (s, _, o) in graph.triples_with_predicate(BROADER) {
            new_triples.push((o.clone(), NARROWER.to_owned(), s.clone()));
        }

        // narrower → broader
        for (s, _, o) in graph.triples_with_predicate(NARROWER) {
            new_triples.push((o.clone(), BROADER.to_owned(), s.clone()));
        }

        new_triples
    }

    // -----------------------------------------------------------------------
    // S4: related is symmetric
    // -----------------------------------------------------------------------

    /// Rule S4: If `X skos:related Y` then `Y skos:related X`.
    pub fn rule_related_symmetry(graph: &Graph) -> Vec<Triple> {
        graph
            .triples_with_predicate(RELATED)
            .map(|(s, _, o)| (o.clone(), RELATED.to_owned(), s.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // S5: topConceptOf / hasTopConcept symmetry
    // -----------------------------------------------------------------------

    /// Rule S5: `X skos:topConceptOf S` ⟺ `S skos:hasTopConcept X`.
    pub fn rule_top_concept_symmetry(graph: &Graph) -> Vec<Triple> {
        let mut new_triples: Vec<Triple> = Vec::new();

        // topConceptOf(X, S) → hasTopConcept(S, X)
        for (x, _, s) in graph.triples_with_predicate(TOP_CONCEPT_OF) {
            new_triples.push((s.clone(), HAS_TOP_CONCEPT.to_owned(), x.clone()));
        }

        // hasTopConcept(S, X) → topConceptOf(X, S)
        for (s, _, x) in graph.triples_with_predicate(HAS_TOP_CONCEPT) {
            new_triples.push((x.clone(), TOP_CONCEPT_OF.to_owned(), s.clone()));
        }

        new_triples
    }

    // -----------------------------------------------------------------------
    // S6a: exactMatch is symmetric
    // -----------------------------------------------------------------------

    /// Rule S6a: If `X skos:exactMatch Y` then `Y skos:exactMatch X`.
    pub fn rule_exact_match_symmetry(graph: &Graph) -> Vec<Triple> {
        graph
            .triples_with_predicate(EXACT_MATCH)
            .map(|(s, _, o)| (o.clone(), EXACT_MATCH.to_owned(), s.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // S6b: exactMatch is transitive
    // -----------------------------------------------------------------------

    /// Rule S6b: If `X skos:exactMatch Y` and `Y skos:exactMatch Z`
    /// then `X skos:exactMatch Z`.
    pub fn rule_exact_match_transitivity(graph: &Graph) -> Vec<Triple> {
        let em_map = graph.adjacency_map(EXACT_MATCH);
        let mut new_triples = Vec::new();
        for (x, y_set) in &em_map {
            for y in y_set {
                if let Some(z_set) = em_map.get(y) {
                    for z in z_set {
                        if z != x {
                            new_triples.push((x.clone(), EXACT_MATCH.to_owned(), z.clone()));
                        }
                    }
                }
            }
        }
        new_triples
    }

    // -----------------------------------------------------------------------
    // S9: closeMatch is symmetric
    // -----------------------------------------------------------------------

    /// Rule S9: If `X skos:closeMatch Y` then `Y skos:closeMatch X`.
    pub fn rule_close_match_symmetry(graph: &Graph) -> Vec<Triple> {
        graph
            .triples_with_predicate(CLOSE_MATCH)
            .map(|(s, _, o)| (o.clone(), CLOSE_MATCH.to_owned(), s.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // S7: topConceptOf implies inScheme
    // -----------------------------------------------------------------------

    /// Rule S7: If `X skos:topConceptOf S` then `X skos:inScheme S`.
    pub fn rule_in_scheme_via_top_concept(graph: &Graph) -> Vec<Triple> {
        graph
            .triples_with_predicate(TOP_CONCEPT_OF)
            .map(|(x, _, s)| (x.clone(), IN_SCHEME.to_owned(), s.clone()))
            .collect()
    }

    // -----------------------------------------------------------------------
    // S10: broadMatch ↔ narrowMatch inverse
    // -----------------------------------------------------------------------

    /// Rule S10: If `X skos:broadMatch Y` then `Y skos:narrowMatch X`, and vice-versa.
    pub fn rule_broad_match_narrow_match_inverse(graph: &Graph) -> Vec<Triple> {
        let mut new_triples: Vec<Triple> = Vec::new();

        for (x, _, y) in graph.triples_with_predicate(BROAD_MATCH) {
            new_triples.push((y.clone(), NARROW_MATCH.to_owned(), x.clone()));
        }

        for (x, _, y) in graph.triples_with_predicate(NARROW_MATCH) {
            new_triples.push((y.clone(), BROAD_MATCH.to_owned(), x.clone()));
        }

        new_triples
    }

    // -----------------------------------------------------------------------
    // Transitive closure utilities
    // -----------------------------------------------------------------------

    /// Compute all ancestors of `concept` via BFS over `skos:broaderTransitive`.
    ///
    /// The returned set does **not** include `concept` itself (non-reflexive).
    ///
    /// # Errors
    /// Returns [`SkosError::CycleDetected`] if a cycle is found in the hierarchy.
    pub fn broader_transitive_closure(
        graph: &Graph,
        concept: &NamedNode,
    ) -> SkosResult<Vec<NamedNode>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut result: Vec<NamedNode> = Vec::new();

        // Seed with direct broader and broaderTransitive
        for (_, p, o) in graph.triples_with_subject(concept) {
            if (p == BROADER || p == BROADER_TRANSITIVE) && o != concept && !visited.contains(o) {
                visited.insert(o.clone());
                queue.push_back(o.clone());
                result.push(o.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            for (_, p, o) in graph.triples_with_subject(&current) {
                if p == BROADER || p == BROADER_TRANSITIVE {
                    if o == concept {
                        return Err(SkosError::CycleDetected(concept.clone()));
                    }
                    if !visited.contains(o) {
                        visited.insert(o.clone());
                        queue.push_back(o.clone());
                        result.push(o.clone());
                    }
                }
            }
        }

        Ok(result)
    }

    /// Compute all descendants of `concept` via BFS over `skos:narrowerTransitive`.
    ///
    /// The returned set does **not** include `concept` itself.
    ///
    /// # Errors
    /// Returns [`SkosError::CycleDetected`] if a cycle is found in the hierarchy.
    pub fn narrower_transitive_closure(
        graph: &Graph,
        concept: &NamedNode,
    ) -> SkosResult<Vec<NamedNode>> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        let mut result: Vec<NamedNode> = Vec::new();

        for (_, p, o) in graph.triples_with_subject(concept) {
            if (p == NARROWER || p == NARROWER_TRANSITIVE) && o != concept && !visited.contains(o) {
                visited.insert(o.clone());
                queue.push_back(o.clone());
                result.push(o.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            for (_, p, o) in graph.triples_with_subject(&current) {
                if p == NARROWER || p == NARROWER_TRANSITIVE {
                    if o == concept {
                        return Err(SkosError::CycleDetected(concept.clone()));
                    }
                    if !visited.contains(o) {
                        visited.insert(o.clone());
                        queue.push_back(o.clone());
                        result.push(o.clone());
                    }
                }
            }
        }

        Ok(result)
    }

    /// Find all concepts whose `skos:prefLabel` or `skos:altLabel` matches
    /// the given `label` string.
    ///
    /// If `lang` is `Some("en")`, only triples whose object ends with `@en`
    /// (or is equal to `label` without a language tag) are matched.
    pub fn find_by_label(graph: &Graph, label: &str, lang: Option<&str>) -> Vec<NamedNode> {
        let mut results: HashSet<String> = HashSet::new();

        for predicate in &[PREF_LABEL, ALT_LABEL] {
            for (s, _, o) in graph.triples_with_predicate(predicate) {
                if label_matches(o, label, lang) {
                    results.insert(s.clone());
                }
            }
        }

        results.into_iter().collect()
    }
}

/// Internal helper: test whether an RDF literal object matches a label and
/// optional language tag.
///
/// Literals are stored as `"value"` or `"value"@lang`.
fn label_matches(literal: &str, label: &str, lang: Option<&str>) -> bool {
    match lang {
        None => {
            // Match if the literal value (stripped of language tag) equals label
            if let Some(at_pos) = literal.rfind('@') {
                &literal[..at_pos] == label
            } else {
                literal == label
            }
        }
        Some(expected_lang) => {
            if let Some(at_pos) = literal.rfind('@') {
                let val = &literal[..at_pos];
                let lit_lang = &literal[at_pos + 1..];
                val == label && lit_lang.eq_ignore_ascii_case(expected_lang)
            } else {
                // No language tag — only match if caller did not request specific lang
                literal == label
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConceptSchemeAnalyzer
// ---------------------------------------------------------------------------

/// A tree node representing a concept and its narrower concepts (children).
#[derive(Debug, Clone)]
pub struct ConceptNode {
    /// The IRI of this concept
    pub iri: NamedNode,
    /// The preferred label (if any)
    pub pref_label: Option<String>,
    /// Narrower (child) concepts in the hierarchy
    pub children: Vec<ConceptNode>,
}

/// The root of a concept tree for a particular `ConceptScheme`.
#[derive(Debug, Clone)]
pub struct ConceptTree {
    /// Top-level concepts of the scheme (concepts without a broader concept within the scheme)
    pub root_concepts: Vec<ConceptNode>,
}

/// Analyzes a `skos:ConceptScheme` within a graph.
pub struct ConceptSchemeAnalyzer;

impl ConceptSchemeAnalyzer {
    /// Return all concepts declared as top concepts of `scheme` via either
    /// `skos:hasTopConcept` or `skos:topConceptOf`.
    pub fn top_concepts(graph: &Graph, scheme: &NamedNode) -> Vec<NamedNode> {
        let mut result: HashSet<String> = HashSet::new();

        // scheme skos:hasTopConcept X
        for (_, _, o) in graph
            .triples_with_subject(scheme)
            .filter(|(_, p, _)| p == HAS_TOP_CONCEPT)
        {
            result.insert(o.clone());
        }

        // X skos:topConceptOf scheme
        for (s, _, _) in graph
            .triples_with_object(scheme)
            .filter(|(_, p, _)| p == TOP_CONCEPT_OF)
        {
            result.insert(s.clone());
        }

        result.into_iter().collect()
    }

    /// Return **all** concepts in `scheme` — i.e., every concept `C` for which
    /// `C skos:inScheme scheme` is present (after entailment, this includes top
    /// concepts via rule S7).
    pub fn all_concepts(graph: &Graph, scheme: &NamedNode) -> Vec<NamedNode> {
        let mut result: HashSet<String> = HashSet::new();

        // Direct inScheme
        for (s, _, _) in graph
            .triples_with_object(scheme)
            .filter(|(_, p, _)| p == IN_SCHEME)
        {
            result.insert(s.clone());
        }

        // Also pick up concepts declared via hasTopConcept / topConceptOf
        for c in Self::top_concepts(graph, scheme) {
            result.insert(c);
        }

        result.into_iter().collect()
    }

    /// Build a `ConceptTree` rooted at the top concepts of `scheme`.
    ///
    /// Recursively traverses `skos:narrower` edges.  Cycles are broken by
    /// tracking visited concept IRIs per branch.
    pub fn concept_tree(graph: &Graph, scheme: &NamedNode) -> ConceptTree {
        let top = Self::top_concepts(graph, scheme);
        let root_concepts = top
            .into_iter()
            .map(|iri| build_node(graph, &iri, &mut HashSet::new()))
            .collect();

        ConceptTree { root_concepts }
    }
}

/// Recursively build a `ConceptNode` for `iri`, walking `skos:narrower` edges.
fn build_node(graph: &Graph, iri: &str, visited: &mut HashSet<String>) -> ConceptNode {
    visited.insert(iri.to_owned());

    let pref_label = graph
        .triples_with_subject(iri)
        .find(|(_, p, _)| p == PREF_LABEL)
        .map(|(_, _, o)| {
            // Strip @lang suffix for display
            if let Some(at) = o.rfind('@') {
                o[..at].to_owned()
            } else {
                o.clone()
            }
        });

    let children: Vec<ConceptNode> = graph
        .triples_with_subject(iri)
        .filter(|(_, p, _)| p == NARROWER)
        .map(|(_, _, child_iri)| child_iri.clone())
        .filter(|child| !visited.contains(child.as_str()))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|child_iri| {
            let mut vis = visited.clone();
            build_node(graph, &child_iri, &mut vis)
        })
        .collect();

    ConceptNode {
        iri: iri.to_owned(),
        pref_label,
        children,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Graph helpers
    // ------------------------------------------------------------------

    fn g() -> Graph {
        Graph::new()
    }

    fn add(graph: &mut Graph, s: &str, p: &str, o: &str) {
        graph.add_triple(s, p, o);
    }

    // ------------------------------------------------------------------
    // Vocabulary constant correctness
    // ------------------------------------------------------------------

    #[test]
    fn test_skos_ns_constant() {
        assert_eq!(SKOS_NS, "http://www.w3.org/2004/02/skos/core#");
    }

    #[test]
    fn test_skos_predicate_iris() {
        assert_eq!(BROADER, "http://www.w3.org/2004/02/skos/core#broader");
        assert_eq!(NARROWER, "http://www.w3.org/2004/02/skos/core#narrower");
        assert_eq!(
            BROADER_TRANSITIVE,
            "http://www.w3.org/2004/02/skos/core#broaderTransitive"
        );
        assert_eq!(
            NARROWER_TRANSITIVE,
            "http://www.w3.org/2004/02/skos/core#narrowerTransitive"
        );
        assert_eq!(RELATED, "http://www.w3.org/2004/02/skos/core#related");
        assert_eq!(
            EXACT_MATCH,
            "http://www.w3.org/2004/02/skos/core#exactMatch"
        );
        assert_eq!(
            CLOSE_MATCH,
            "http://www.w3.org/2004/02/skos/core#closeMatch"
        );
        assert_eq!(
            BROAD_MATCH,
            "http://www.w3.org/2004/02/skos/core#broadMatch"
        );
        assert_eq!(
            NARROW_MATCH,
            "http://www.w3.org/2004/02/skos/core#narrowMatch"
        );
        assert_eq!(IN_SCHEME, "http://www.w3.org/2004/02/skos/core#inScheme");
        assert_eq!(
            HAS_TOP_CONCEPT,
            "http://www.w3.org/2004/02/skos/core#hasTopConcept"
        );
        assert_eq!(
            TOP_CONCEPT_OF,
            "http://www.w3.org/2004/02/skos/core#topConceptOf"
        );
        assert_eq!(PREF_LABEL, "http://www.w3.org/2004/02/skos/core#prefLabel");
        assert_eq!(ALT_LABEL, "http://www.w3.org/2004/02/skos/core#altLabel");
    }

    // ------------------------------------------------------------------
    // Graph basic operations
    // ------------------------------------------------------------------

    #[test]
    fn test_graph_add_and_contains() {
        let mut graph = g();
        assert!(graph.add_triple("ex:A", BROADER, "ex:B"));
        assert!(graph.contains("ex:A", BROADER, "ex:B"));
        assert!(!graph.contains("ex:B", BROADER, "ex:A"));
    }

    #[test]
    fn test_graph_duplicate_triple_not_added() {
        let mut graph = g();
        assert!(graph.add_triple("ex:A", BROADER, "ex:B"));
        assert!(!graph.add_triple("ex:A", BROADER, "ex:B")); // duplicate
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_graph_triples_with_predicate() {
        let mut graph = g();
        add(&mut graph, "ex:A", BROADER, "ex:B");
        add(&mut graph, "ex:C", BROADER, "ex:D");
        add(&mut graph, "ex:E", RELATED, "ex:F");

        let bt: Vec<_> = graph.triples_with_predicate(BROADER).collect();
        assert_eq!(bt.len(), 2);
    }

    #[test]
    fn test_graph_len_and_is_empty() {
        let mut graph = g();
        assert!(graph.is_empty());
        add(&mut graph, "ex:A", BROADER, "ex:B");
        assert!(!graph.is_empty());
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_graph_merge() {
        let mut g1 = g();
        add(&mut g1, "ex:A", BROADER, "ex:B");

        let mut g2 = g();
        add(&mut g2, "ex:C", BROADER, "ex:D");

        g1.merge(&g2);
        assert_eq!(g1.len(), 2);
    }

    #[test]
    fn test_graph_adjacency_map() {
        let mut graph = g();
        add(&mut graph, "ex:A", BROADER, "ex:B");
        add(&mut graph, "ex:A", BROADER, "ex:C");
        add(&mut graph, "ex:B", BROADER, "ex:D");

        let map = graph.adjacency_map(BROADER);
        assert_eq!(map.get("ex:A").map(|s| s.len()), Some(2));
        assert_eq!(map.get("ex:B").map(|s| s.len()), Some(1));
        assert!(!map.contains_key("ex:D"));
    }

    // ------------------------------------------------------------------
    // Rule S8: broader/narrower → broaderTransitive/narrowerTransitive
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_broader_to_transitive() {
        let mut graph = g();
        add(&mut graph, "ex:A", BROADER, "ex:B");
        add(&mut graph, "ex:B", BROADER, "ex:C");

        let new = SkosReasoner::rule_broader_to_transitive(&graph);
        assert!(new
            .iter()
            .any(|(s, p, o)| s == "ex:A" && p == BROADER_TRANSITIVE && o == "ex:B"));
        assert!(new
            .iter()
            .any(|(s, p, o)| s == "ex:B" && p == BROADER_TRANSITIVE && o == "ex:C"));
    }

    #[test]
    fn test_rule_narrower_to_transitive() {
        let mut graph = g();
        add(&mut graph, "ex:A", NARROWER, "ex:B");

        let new = SkosReasoner::rule_narrower_to_transitive(&graph);
        assert!(new
            .iter()
            .any(|(s, p, o)| s == "ex:A" && p == NARROWER_TRANSITIVE && o == "ex:B"));
    }

    // ------------------------------------------------------------------
    // Rule S1: broaderTransitive is transitive
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_broader_transitive_chain_simple() -> anyhow::Result<()> {
        let mut graph = g();
        add(&mut graph, "ex:A", BROADER_TRANSITIVE, "ex:B");
        add(&mut graph, "ex:B", BROADER_TRANSITIVE, "ex:C");

        let new = SkosReasoner::rule_broader_transitive_chain(&graph);
        assert!(
            new.iter()
                .any(|(s, p, o)| s == "ex:A" && p == BROADER_TRANSITIVE && o == "ex:C"),
            "Expected ex:A broaderTransitive ex:C; got {new:?}"
        );
        Ok(())
    }

    #[test]
    fn test_rule_broader_transitive_chain_three_hops() {
        let mut graph = g();
        add(&mut graph, "ex:A", BROADER_TRANSITIVE, "ex:B");
        add(&mut graph, "ex:B", BROADER_TRANSITIVE, "ex:C");
        add(&mut graph, "ex:C", BROADER_TRANSITIVE, "ex:D");

        // One pass gives A→C and B→D; next fixpoint pass gives A→D
        let new1 = SkosReasoner::rule_broader_transitive_chain(&graph);
        for triple in &new1 {
            graph.add_triple(triple.0.clone(), triple.1.clone(), triple.2.clone());
        }
        let new2 = SkosReasoner::rule_broader_transitive_chain(&graph);
        for triple in &new2 {
            graph.add_triple(triple.0.clone(), triple.1.clone(), triple.2.clone());
        }
        assert!(graph.contains("ex:A", BROADER_TRANSITIVE, "ex:D"));
    }

    // ------------------------------------------------------------------
    // Rule S2: narrowerTransitive is transitive
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_narrower_transitive_chain_simple() {
        let mut graph = g();
        add(&mut graph, "ex:A", NARROWER_TRANSITIVE, "ex:B");
        add(&mut graph, "ex:B", NARROWER_TRANSITIVE, "ex:C");

        let new = SkosReasoner::rule_narrower_transitive_chain(&graph);
        assert!(new
            .iter()
            .any(|(s, p, o)| s == "ex:A" && p == NARROWER_TRANSITIVE && o == "ex:C"));
    }

    // ------------------------------------------------------------------
    // Rule S3: broader ↔ narrower symmetry
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_broader_narrower_symmetry_broader_direction() {
        let mut graph = g();
        add(&mut graph, "ex:A", BROADER, "ex:B");

        let new = SkosReasoner::rule_broader_narrower_symmetry(&graph);
        assert!(
            new.iter()
                .any(|(s, p, o)| s == "ex:B" && p == NARROWER && o == "ex:A"),
            "Expected ex:B narrower ex:A"
        );
    }

    #[test]
    fn test_rule_broader_narrower_symmetry_narrower_direction() {
        let mut graph = g();
        add(&mut graph, "ex:B", NARROWER, "ex:A");

        let new = SkosReasoner::rule_broader_narrower_symmetry(&graph);
        assert!(
            new.iter()
                .any(|(s, p, o)| s == "ex:A" && p == BROADER && o == "ex:B"),
            "Expected ex:A broader ex:B"
        );
    }

    // ------------------------------------------------------------------
    // Rule S4: related is symmetric
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_related_symmetry() {
        let mut graph = g();
        add(&mut graph, "ex:A", RELATED, "ex:B");

        let new = SkosReasoner::rule_related_symmetry(&graph);
        assert!(new
            .iter()
            .any(|(s, p, o)| s == "ex:B" && p == RELATED && o == "ex:A"));
    }

    #[test]
    fn test_rule_related_symmetry_multiple() {
        let mut graph = g();
        add(&mut graph, "ex:A", RELATED, "ex:B");
        add(&mut graph, "ex:C", RELATED, "ex:D");

        let new = SkosReasoner::rule_related_symmetry(&graph);
        assert_eq!(new.len(), 2);
    }

    // ------------------------------------------------------------------
    // Rule S5: topConceptOf / hasTopConcept symmetry
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_top_concept_symmetry_top_concept_of() {
        let mut graph = g();
        add(&mut graph, "ex:Art", TOP_CONCEPT_OF, "ex:Scheme1");

        let new = SkosReasoner::rule_top_concept_symmetry(&graph);
        assert!(
            new.iter()
                .any(|(s, p, o)| s == "ex:Scheme1" && p == HAS_TOP_CONCEPT && o == "ex:Art"),
            "Expected ex:Scheme1 hasTopConcept ex:Art"
        );
    }

    #[test]
    fn test_rule_top_concept_symmetry_has_top_concept() {
        let mut graph = g();
        add(&mut graph, "ex:Scheme1", HAS_TOP_CONCEPT, "ex:Art");

        let new = SkosReasoner::rule_top_concept_symmetry(&graph);
        assert!(
            new.iter()
                .any(|(s, p, o)| s == "ex:Art" && p == TOP_CONCEPT_OF && o == "ex:Scheme1"),
            "Expected ex:Art topConceptOf ex:Scheme1"
        );
    }

    // ------------------------------------------------------------------
    // Rule S6: exactMatch symmetric + transitive
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_exact_match_symmetry() {
        let mut graph = g();
        add(&mut graph, "ex:A", EXACT_MATCH, "ex:B");

        let new = SkosReasoner::rule_exact_match_symmetry(&graph);
        assert!(new
            .iter()
            .any(|(s, p, o)| s == "ex:B" && p == EXACT_MATCH && o == "ex:A"));
    }

    #[test]
    fn test_rule_exact_match_transitivity() {
        let mut graph = g();
        add(&mut graph, "ex:A", EXACT_MATCH, "ex:B");
        add(&mut graph, "ex:B", EXACT_MATCH, "ex:C");

        let new = SkosReasoner::rule_exact_match_transitivity(&graph);
        assert!(
            new.iter()
                .any(|(s, p, o)| s == "ex:A" && p == EXACT_MATCH && o == "ex:C"),
            "Expected ex:A exactMatch ex:C"
        );
    }

    #[test]
    fn test_rule_exact_match_chain_three() {
        let mut graph = g();
        add(&mut graph, "ex:A", EXACT_MATCH, "ex:B");
        add(&mut graph, "ex:B", EXACT_MATCH, "ex:C");
        add(&mut graph, "ex:C", EXACT_MATCH, "ex:D");

        // Two fixpoint iterations needed for A→D
        let new1 = SkosReasoner::rule_exact_match_transitivity(&graph);
        for t in &new1 {
            graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }
        let new2 = SkosReasoner::rule_exact_match_transitivity(&graph);
        for t in &new2 {
            graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }

        assert!(
            graph.contains("ex:A", EXACT_MATCH, "ex:D"),
            "Expected A exactMatch D after two fixpoint passes"
        );
    }

    // ------------------------------------------------------------------
    // Rule S9: closeMatch symmetry
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_close_match_symmetry() {
        let mut graph = g();
        add(&mut graph, "ex:A", CLOSE_MATCH, "ex:B");

        let new = SkosReasoner::rule_close_match_symmetry(&graph);
        assert!(new
            .iter()
            .any(|(s, p, o)| s == "ex:B" && p == CLOSE_MATCH && o == "ex:A"));
    }

    // ------------------------------------------------------------------
    // Rule S7: inScheme via topConceptOf
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_in_scheme_via_top_concept() {
        let mut graph = g();
        add(&mut graph, "ex:Art", TOP_CONCEPT_OF, "ex:Scheme1");

        let new = SkosReasoner::rule_in_scheme_via_top_concept(&graph);
        assert!(
            new.iter()
                .any(|(s, p, o)| s == "ex:Art" && p == IN_SCHEME && o == "ex:Scheme1"),
            "Expected ex:Art inScheme ex:Scheme1"
        );
    }

    // ------------------------------------------------------------------
    // Rule S10: broadMatch ↔ narrowMatch inverse
    // ------------------------------------------------------------------

    #[test]
    fn test_rule_broad_match_narrow_match_inverse_from_broad() {
        let mut graph = g();
        add(&mut graph, "ex:A", BROAD_MATCH, "ex:B");

        let new = SkosReasoner::rule_broad_match_narrow_match_inverse(&graph);
        assert!(new
            .iter()
            .any(|(s, p, o)| s == "ex:B" && p == NARROW_MATCH && o == "ex:A"));
    }

    #[test]
    fn test_rule_broad_match_narrow_match_inverse_from_narrow() {
        let mut graph = g();
        add(&mut graph, "ex:A", NARROW_MATCH, "ex:B");

        let new = SkosReasoner::rule_broad_match_narrow_match_inverse(&graph);
        assert!(new
            .iter()
            .any(|(s, p, o)| s == "ex:B" && p == BROAD_MATCH && o == "ex:A"));
    }

    // ------------------------------------------------------------------
    // apply_rules — fixpoint integration
    // ------------------------------------------------------------------

    #[test]
    fn test_apply_rules_broader_chain_fixpoint() -> anyhow::Result<()> {
        let mut graph = g();
        add(&mut graph, "ex:A", BROADER, "ex:B");
        add(&mut graph, "ex:B", BROADER, "ex:C");
        add(&mut graph, "ex:C", BROADER, "ex:D");

        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");

        // After fixpoint: A→BT→B, B→BT→C, C→BT→D, A→BT→C, A→BT→D, B→BT→D
        // plus symmetric narrower
        let contains_bt = |s: &str, o: &str| {
            new.iter()
                .any(|(ns, np, no)| ns == s && np == BROADER_TRANSITIVE && no == o)
                || graph.contains(s, BROADER_TRANSITIVE, o)
        };

        // Build final graph for checking
        let mut final_graph = graph.clone();
        for t in &new {
            final_graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }

        assert!(
            final_graph.contains("ex:A", BROADER_TRANSITIVE, "ex:B") || contains_bt("ex:A", "ex:B")
        );
        assert!(
            final_graph.contains("ex:A", BROADER_TRANSITIVE, "ex:C") || contains_bt("ex:A", "ex:C"),
            "Expected A broaderTransitive C; new triples = {new:?}"
        );
        assert!(
            final_graph.contains("ex:A", BROADER_TRANSITIVE, "ex:D") || contains_bt("ex:A", "ex:D"),
            "Expected A broaderTransitive D; new triples = {new:?}"
        );
        Ok(())
    }

    #[test]
    fn test_apply_rules_symmetry_closure() {
        let mut graph = g();
        add(&mut graph, "ex:X", RELATED, "ex:Y");
        add(&mut graph, "ex:P", BROADER, "ex:Q");

        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");
        let mut final_graph = graph.clone();
        for t in &new {
            final_graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }

        assert!(
            final_graph.contains("ex:Y", RELATED, "ex:X"),
            "related symmetry"
        );
        assert!(
            final_graph.contains("ex:Q", NARROWER, "ex:P"),
            "narrower from broader"
        );
    }

    #[test]
    fn test_apply_rules_top_concept_scheme() {
        let mut graph = g();
        add(&mut graph, "ex:Art", TOP_CONCEPT_OF, "ex:Scheme");

        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");
        let mut final_graph = graph.clone();
        for t in &new {
            final_graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }

        assert!(
            final_graph.contains("ex:Scheme", HAS_TOP_CONCEPT, "ex:Art"),
            "hasTopConcept"
        );
        assert!(
            final_graph.contains("ex:Art", IN_SCHEME, "ex:Scheme"),
            "inScheme"
        );
    }

    #[test]
    fn test_apply_rules_exact_match_closure() {
        let mut graph = g();
        add(&mut graph, "ex:A", EXACT_MATCH, "ex:B");
        add(&mut graph, "ex:B", EXACT_MATCH, "ex:C");

        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");
        let mut final_graph = graph.clone();
        for t in &new {
            final_graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }

        assert!(
            final_graph.contains("ex:B", EXACT_MATCH, "ex:A"),
            "exactMatch symmetry"
        );
        assert!(
            final_graph.contains("ex:C", EXACT_MATCH, "ex:B"),
            "exactMatch symmetry"
        );
        assert!(
            final_graph.contains("ex:A", EXACT_MATCH, "ex:C"),
            "exactMatch transitivity"
        );
        assert!(
            final_graph.contains("ex:C", EXACT_MATCH, "ex:A"),
            "exactMatch transitivity+symmetry"
        );
    }

    #[test]
    fn test_apply_rules_empty_graph() {
        let graph = g();
        let new =
            SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed on empty graph");
        assert!(new.is_empty(), "No new triples from empty graph");
    }

    #[test]
    fn test_apply_rules_no_new_triples_when_already_entailed() {
        let mut graph = g();
        add(&mut graph, "ex:A", RELATED, "ex:B");
        add(&mut graph, "ex:B", RELATED, "ex:A"); // already symmetric

        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");
        // No genuinely new triples
        assert!(new.is_empty(), "No new triples when already fully entailed");
    }

    // ------------------------------------------------------------------
    // broader_transitive_closure
    // ------------------------------------------------------------------

    #[test]
    fn test_broader_transitive_closure_single_hop() {
        let mut graph = g();
        add(&mut graph, "ex:Art", BROADER, "ex:Culture");

        let ancestors = SkosReasoner::broader_transitive_closure(&graph, &"ex:Art".to_owned())
            .expect("closure should succeed");
        assert!(ancestors.contains(&"ex:Culture".to_owned()));
        assert!(!ancestors.contains(&"ex:Art".to_owned()));
    }

    #[test]
    fn test_broader_transitive_closure_chain() {
        let mut graph = g();
        add(&mut graph, "ex:A", BROADER, "ex:B");
        add(&mut graph, "ex:B", BROADER, "ex:C");
        add(&mut graph, "ex:C", BROADER, "ex:D");

        let ancestors = SkosReasoner::broader_transitive_closure(&graph, &"ex:A".to_owned())
            .expect("closure should succeed");
        assert!(ancestors.contains(&"ex:B".to_owned()));
        assert!(ancestors.contains(&"ex:C".to_owned()));
        assert!(ancestors.contains(&"ex:D".to_owned()));
    }

    #[test]
    fn test_broader_transitive_closure_no_ancestors() {
        let mut graph = g();
        add(&mut graph, "ex:A", BROADER, "ex:B");

        let ancestors = SkosReasoner::broader_transitive_closure(&graph, &"ex:B".to_owned())
            .expect("closure should succeed");
        assert!(ancestors.is_empty(), "ex:B has no broader ancestors");
    }

    #[test]
    fn test_broader_transitive_closure_uses_bt_edges() {
        let mut graph = g();
        // Use broaderTransitive directly
        add(&mut graph, "ex:A", BROADER_TRANSITIVE, "ex:B");
        add(&mut graph, "ex:B", BROADER_TRANSITIVE, "ex:C");

        let ancestors = SkosReasoner::broader_transitive_closure(&graph, &"ex:A".to_owned())
            .expect("closure should succeed");
        assert!(ancestors.contains(&"ex:B".to_owned()));
        assert!(ancestors.contains(&"ex:C".to_owned()));
    }

    // ------------------------------------------------------------------
    // narrower_transitive_closure
    // ------------------------------------------------------------------

    #[test]
    fn test_narrower_transitive_closure_chain() {
        let mut graph = g();
        add(&mut graph, "ex:D", NARROWER, "ex:C");
        add(&mut graph, "ex:C", NARROWER, "ex:B");
        add(&mut graph, "ex:B", NARROWER, "ex:A");

        let descendants = SkosReasoner::narrower_transitive_closure(&graph, &"ex:D".to_owned())
            .expect("closure should succeed");
        assert!(descendants.contains(&"ex:C".to_owned()));
        assert!(descendants.contains(&"ex:B".to_owned()));
        assert!(descendants.contains(&"ex:A".to_owned()));
    }

    #[test]
    fn test_narrower_transitive_closure_no_descendants() {
        let mut graph = g();
        add(&mut graph, "ex:A", NARROWER, "ex:B");

        let descendants = SkosReasoner::narrower_transitive_closure(&graph, &"ex:B".to_owned())
            .expect("closure should succeed");
        assert!(descendants.is_empty());
    }

    // ------------------------------------------------------------------
    // find_by_label
    // ------------------------------------------------------------------

    #[test]
    fn test_find_by_label_pref_label() {
        let mut graph = g();
        add(&mut graph, "ex:Art", PREF_LABEL, "Art@en");
        add(&mut graph, "ex:Science", PREF_LABEL, "Science@en");

        let found = SkosReasoner::find_by_label(&graph, "Art", Some("en"));
        assert!(found.contains(&"ex:Art".to_owned()));
        assert!(!found.contains(&"ex:Science".to_owned()));
    }

    #[test]
    fn test_find_by_label_alt_label() {
        let mut graph = g();
        add(&mut graph, "ex:Art", ALT_LABEL, "Fine Art@en");

        let found = SkosReasoner::find_by_label(&graph, "Fine Art", Some("en"));
        assert!(found.contains(&"ex:Art".to_owned()));
    }

    #[test]
    fn test_find_by_label_no_lang_filter() {
        let mut graph = g();
        add(&mut graph, "ex:Art", PREF_LABEL, "Art@en");
        add(&mut graph, "ex:Kunst", PREF_LABEL, "Kunst@de");

        // Without lang filter, both can match
        let found_art = SkosReasoner::find_by_label(&graph, "Art", None);
        assert!(found_art.contains(&"ex:Art".to_owned()));

        let found_kunst = SkosReasoner::find_by_label(&graph, "Kunst", None);
        assert!(found_kunst.contains(&"ex:Kunst".to_owned()));
    }

    #[test]
    fn test_find_by_label_lang_case_insensitive() {
        let mut graph = g();
        add(&mut graph, "ex:Art", PREF_LABEL, "Art@EN");

        let found = SkosReasoner::find_by_label(&graph, "Art", Some("en"));
        assert!(
            found.contains(&"ex:Art".to_owned()),
            "Language tag comparison should be case-insensitive"
        );
    }

    #[test]
    fn test_find_by_label_no_match() {
        let mut graph = g();
        add(&mut graph, "ex:Art", PREF_LABEL, "Art@en");

        let found = SkosReasoner::find_by_label(&graph, "Music", Some("en"));
        assert!(found.is_empty());
    }

    #[test]
    fn test_find_by_label_no_lang_tag_in_literal() {
        let mut graph = g();
        add(&mut graph, "ex:Art", PREF_LABEL, "Art");

        let found = SkosReasoner::find_by_label(&graph, "Art", None);
        assert!(found.contains(&"ex:Art".to_owned()));
    }

    // ------------------------------------------------------------------
    // ConceptSchemeAnalyzer::top_concepts
    // ------------------------------------------------------------------

    #[test]
    fn test_top_concepts_via_has_top_concept() {
        let mut graph = g();
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Science");

        let tops = ConceptSchemeAnalyzer::top_concepts(&graph, &"ex:Scheme".to_owned());
        assert_eq!(tops.len(), 2);
        assert!(tops.contains(&"ex:Art".to_owned()));
        assert!(tops.contains(&"ex:Science".to_owned()));
    }

    #[test]
    fn test_top_concepts_via_top_concept_of() {
        let mut graph = g();
        add(&mut graph, "ex:Art", TOP_CONCEPT_OF, "ex:Scheme");

        let tops = ConceptSchemeAnalyzer::top_concepts(&graph, &"ex:Scheme".to_owned());
        assert!(tops.contains(&"ex:Art".to_owned()));
    }

    #[test]
    fn test_top_concepts_both_directions() {
        let mut graph = g();
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        add(&mut graph, "ex:Music", TOP_CONCEPT_OF, "ex:Scheme");

        let tops = ConceptSchemeAnalyzer::top_concepts(&graph, &"ex:Scheme".to_owned());
        assert!(tops.contains(&"ex:Art".to_owned()));
        assert!(tops.contains(&"ex:Music".to_owned()));
    }

    #[test]
    fn test_top_concepts_empty_scheme() {
        let graph = g();
        let tops = ConceptSchemeAnalyzer::top_concepts(&graph, &"ex:Scheme".to_owned());
        assert!(tops.is_empty());
    }

    // ------------------------------------------------------------------
    // ConceptSchemeAnalyzer::all_concepts
    // ------------------------------------------------------------------

    #[test]
    fn test_all_concepts_in_scheme() {
        let mut graph = g();
        add(&mut graph, "ex:Art", IN_SCHEME, "ex:Scheme");
        add(&mut graph, "ex:Music", IN_SCHEME, "ex:Scheme");
        add(&mut graph, "ex:Physics", IN_SCHEME, "ex:OtherScheme");

        let all = ConceptSchemeAnalyzer::all_concepts(&graph, &"ex:Scheme".to_owned());
        assert!(all.contains(&"ex:Art".to_owned()));
        assert!(all.contains(&"ex:Music".to_owned()));
        assert!(!all.contains(&"ex:Physics".to_owned()));
    }

    #[test]
    fn test_all_concepts_includes_top_concepts() {
        let mut graph = g();
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        // Art is not explicitly in_scheme but should be included via top concept

        let all = ConceptSchemeAnalyzer::all_concepts(&graph, &"ex:Scheme".to_owned());
        assert!(all.contains(&"ex:Art".to_owned()));
    }

    #[test]
    fn test_all_concepts_empty_scheme() {
        let graph = g();
        let all = ConceptSchemeAnalyzer::all_concepts(&graph, &"ex:Scheme".to_owned());
        assert!(all.is_empty());
    }

    // ------------------------------------------------------------------
    // ConceptTree / concept_tree
    // ------------------------------------------------------------------

    #[test]
    fn test_concept_tree_single_level() {
        let mut graph = g();
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        add(&mut graph, "ex:Art", PREF_LABEL, "Art@en");

        let tree = ConceptSchemeAnalyzer::concept_tree(&graph, &"ex:Scheme".to_owned());
        assert_eq!(tree.root_concepts.len(), 1);
        let root = &tree.root_concepts[0];
        assert_eq!(root.iri, "ex:Art");
        assert_eq!(root.pref_label.as_deref(), Some("Art"));
        assert!(root.children.is_empty());
    }

    #[test]
    fn test_concept_tree_with_children() {
        let mut graph = g();
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        add(&mut graph, "ex:Art", NARROWER, "ex:Painting");
        add(&mut graph, "ex:Art", NARROWER, "ex:Sculpture");
        add(&mut graph, "ex:Painting", PREF_LABEL, "Painting@en");
        add(&mut graph, "ex:Sculpture", PREF_LABEL, "Sculpture@en");

        let tree = ConceptSchemeAnalyzer::concept_tree(&graph, &"ex:Scheme".to_owned());
        assert_eq!(tree.root_concepts.len(), 1);
        let root = &tree.root_concepts[0];
        assert_eq!(root.iri, "ex:Art");
        assert_eq!(root.children.len(), 2);
    }

    #[test]
    fn test_concept_tree_three_levels() {
        let mut graph = g();
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        add(&mut graph, "ex:Art", NARROWER, "ex:VisualArts");
        add(&mut graph, "ex:VisualArts", NARROWER, "ex:Painting");

        let tree = ConceptSchemeAnalyzer::concept_tree(&graph, &"ex:Scheme".to_owned());
        assert_eq!(tree.root_concepts.len(), 1);
        let level1 = &tree.root_concepts[0];
        assert_eq!(level1.iri, "ex:Art");
        assert_eq!(level1.children.len(), 1);
        let level2 = &level1.children[0];
        assert_eq!(level2.iri, "ex:VisualArts");
        assert_eq!(level2.children.len(), 1);
        let level3 = &level2.children[0];
        assert_eq!(level3.iri, "ex:Painting");
        assert!(level3.children.is_empty());
    }

    #[test]
    fn test_concept_tree_empty_scheme() {
        let graph = g();
        let tree = ConceptSchemeAnalyzer::concept_tree(&graph, &"ex:Scheme".to_owned());
        assert!(tree.root_concepts.is_empty());
    }

    #[test]
    fn test_concept_tree_multiple_top_concepts() {
        let mut graph = g();
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Science");
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Technology");

        let tree = ConceptSchemeAnalyzer::concept_tree(&graph, &"ex:Scheme".to_owned());
        assert_eq!(tree.root_concepts.len(), 3);
    }

    // ------------------------------------------------------------------
    // ConceptNode structure
    // ------------------------------------------------------------------

    #[test]
    fn test_concept_node_no_pref_label() {
        let mut graph = g();
        add(&mut graph, "ex:Scheme", HAS_TOP_CONCEPT, "ex:Art");
        // No prefLabel triple

        let tree = ConceptSchemeAnalyzer::concept_tree(&graph, &"ex:Scheme".to_owned());
        let root = &tree.root_concepts[0];
        assert!(root.pref_label.is_none());
    }

    // ------------------------------------------------------------------
    // label_matches helper
    // ------------------------------------------------------------------

    #[test]
    fn test_label_matches_with_lang() {
        assert!(label_matches("Art@en", "Art", Some("en")));
        assert!(!label_matches("Art@de", "Art", Some("en")));
        assert!(!label_matches("Music@en", "Art", Some("en")));
    }

    #[test]
    fn test_label_matches_without_lang() {
        assert!(label_matches("Art@en", "Art", None));
        assert!(label_matches("Art", "Art", None));
        assert!(!label_matches("Art@en", "Music", None));
    }

    #[test]
    fn test_label_matches_case_insensitive_lang() {
        assert!(label_matches("Art@EN", "Art", Some("en")));
        assert!(label_matches("Art@En", "Art", Some("EN")));
    }

    // ------------------------------------------------------------------
    // Integration: real-world thesaurus scenario
    // ------------------------------------------------------------------

    /// Tests a simplified AGROVOC-like concept hierarchy with 4 levels
    #[test]
    fn test_integration_agriculture_hierarchy() {
        let mut graph = g();

        // Scheme
        let scheme = "ex:AgriScheme";
        add(&mut graph, scheme, HAS_TOP_CONCEPT, "ex:Agriculture");

        // Top level
        add(&mut graph, "ex:Agriculture", PREF_LABEL, "Agriculture@en");
        add(&mut graph, "ex:Agriculture", IN_SCHEME, scheme);

        // Level 2
        add(&mut graph, "ex:Agriculture", NARROWER, "ex:CropProduction");
        add(
            &mut graph,
            "ex:Agriculture",
            NARROWER,
            "ex:LivestockFarming",
        );
        add(
            &mut graph,
            "ex:CropProduction",
            PREF_LABEL,
            "Crop production@en",
        );
        add(
            &mut graph,
            "ex:LivestockFarming",
            PREF_LABEL,
            "Livestock farming@en",
        );

        // Level 3
        add(
            &mut graph,
            "ex:CropProduction",
            NARROWER,
            "ex:CerealProduction",
        );
        add(
            &mut graph,
            "ex:CerealProduction",
            PREF_LABEL,
            "Cereal production@en",
        );

        // Level 4
        add(
            &mut graph,
            "ex:CerealProduction",
            NARROWER,
            "ex:WheatProduction",
        );
        add(
            &mut graph,
            "ex:WheatProduction",
            PREF_LABEL,
            "Wheat production@en",
        );

        // Cross-concept relation
        add(
            &mut graph,
            "ex:CropProduction",
            RELATED,
            "ex:LivestockFarming",
        );

        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");
        let mut final_graph = graph.clone();
        for t in &new {
            final_graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }

        // broader/narrower symmetry
        assert!(
            final_graph.contains("ex:CropProduction", BROADER, "ex:Agriculture"),
            "CropProduction should have broader Agriculture"
        );
        assert!(final_graph.contains("ex:CerealProduction", BROADER, "ex:CropProduction"));
        assert!(final_graph.contains("ex:WheatProduction", BROADER, "ex:CerealProduction"));

        // related symmetry
        assert!(
            final_graph.contains("ex:LivestockFarming", RELATED, "ex:CropProduction"),
            "related should be symmetric"
        );

        // hasTopConcept generated
        assert!(
            final_graph.contains(scheme, HAS_TOP_CONCEPT, "ex:Agriculture") || {
                graph.contains(scheme, HAS_TOP_CONCEPT, "ex:Agriculture")
            }
        );

        // Transitive closure
        let ancestors = SkosReasoner::broader_transitive_closure(
            &final_graph,
            &"ex:WheatProduction".to_owned(),
        )
        .expect("closure should succeed");
        assert!(
            ancestors.contains(&"ex:Agriculture".to_owned()),
            "WheatProduction's ancestors should include Agriculture"
        );
        assert!(ancestors.contains(&"ex:CerealProduction".to_owned()));
        assert!(ancestors.contains(&"ex:CropProduction".to_owned()));

        // Label search
        let found = SkosReasoner::find_by_label(&final_graph, "Wheat production", Some("en"));
        assert!(
            found.contains(&"ex:WheatProduction".to_owned()),
            "find_by_label should find WheatProduction"
        );

        // Concept tree
        let tree = ConceptSchemeAnalyzer::concept_tree(&final_graph, &scheme.to_owned());
        assert!(!tree.root_concepts.is_empty());
        let root = &tree.root_concepts[0];
        assert_eq!(root.iri, "ex:Agriculture");
        assert!(!root.children.is_empty());
    }

    /// Tests cross-scheme exactMatch entailment
    #[test]
    fn test_integration_cross_scheme_exact_match() {
        let mut graph = g();
        add(&mut graph, "thesA:Art", EXACT_MATCH, "thesB:Art");
        add(&mut graph, "thesB:Art", EXACT_MATCH, "thesC:Art");

        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");
        let mut final_graph = graph.clone();
        for t in &new {
            final_graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }

        assert!(
            final_graph.contains("thesA:Art", EXACT_MATCH, "thesC:Art"),
            "exactMatch should be transitive across schemes"
        );
        assert!(
            final_graph.contains("thesC:Art", EXACT_MATCH, "thesA:Art"),
            "exactMatch should be transitive and symmetric"
        );
        assert!(
            final_graph.contains("thesB:Art", EXACT_MATCH, "thesA:Art"),
            "exactMatch symmetry"
        );
        assert!(
            final_graph.contains("thesC:Art", EXACT_MATCH, "thesB:Art"),
            "exactMatch symmetry"
        );
    }

    /// Test broadMatch/narrowMatch cross-vocabulary mapping
    #[test]
    fn test_integration_mapping_relations() {
        let mut graph = g();
        // In thesA "Art" is broader than thesB's "Painting" (from thesA perspective)
        add(&mut graph, "thesA:Art", BROAD_MATCH, "thesB:Painting");

        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");
        let mut final_graph = graph.clone();
        for t in &new {
            final_graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }

        // From thesB's perspective, Painting has a narrowMatch to thesA's Art
        assert!(
            final_graph.contains("thesB:Painting", NARROW_MATCH, "thesA:Art"),
            "broadMatch should imply inverse narrowMatch"
        );
    }

    /// Edge case: singleton graph with a self-related concept is a no-op
    #[test]
    fn test_integration_no_entailment_from_type_only() {
        let mut graph = g();
        add(&mut graph, "ex:Art", "rdf:type", "skos:Concept");

        // No SKOS relational triples, so no entailment
        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");
        assert!(new.is_empty(), "Type-only graph entails nothing");
    }

    /// Diamond hierarchy: A narrower B and C; B and C broader D
    #[test]
    fn test_integration_diamond_hierarchy() {
        let mut graph = g();
        add(&mut graph, "ex:A", NARROWER, "ex:B");
        add(&mut graph, "ex:A", NARROWER, "ex:C");
        add(&mut graph, "ex:B", BROADER, "ex:D");
        add(&mut graph, "ex:C", BROADER, "ex:D");

        let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");
        let mut final_graph = graph.clone();
        for t in &new {
            final_graph.add_triple(t.0.clone(), t.1.clone(), t.2.clone());
        }

        // B broader A (from narrower symmetry), D has narrower B and C
        assert!(
            final_graph.contains("ex:B", BROADER, "ex:A")
                || final_graph.contains("ex:A", NARROWER, "ex:B")
        );
        // D narrower B and C from broader symmetry
        assert!(
            final_graph.contains("ex:D", NARROWER, "ex:B"),
            "D should have narrower B (from broader symmetry)"
        );
        assert!(
            final_graph.contains("ex:D", NARROWER, "ex:C"),
            "D should have narrower C (from broader symmetry)"
        );
    }
}
