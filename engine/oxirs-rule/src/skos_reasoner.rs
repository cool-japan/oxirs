//! # SKOS Reasoner
//!
//! Implements SKOS entailment rules per W3C SKOS Reference §8 and provides
//! hierarchy traversal utilities (broader/narrower transitive closures and
//! label search).

use std::collections::{HashSet, VecDeque};

use crate::skos_types::{
    label_matches, Graph, NamedNode, SkosError, SkosResult, Triple, ALT_LABEL, BROADER,
    BROADER_TRANSITIVE, BROAD_MATCH, CLOSE_MATCH, EXACT_MATCH, HAS_TOP_CONCEPT, IN_SCHEME,
    NARROWER, NARROWER_TRANSITIVE, NARROW_MATCH, PREF_LABEL, RELATED, TOP_CONCEPT_OF,
};

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
