//! # SKOS Concept Scheme Mappings / Analyzer
//!
//! Analyzes `skos:ConceptScheme` membership and builds concept trees from
//! the `skos:narrower` hierarchy.

use std::collections::HashSet;

use crate::skos_types::{
    ConceptNode, ConceptTree, Graph, NamedNode, HAS_TOP_CONCEPT, IN_SCHEME, NARROWER, PREF_LABEL,
    TOP_CONCEPT_OF,
};

// ---------------------------------------------------------------------------
// ConceptSchemeAnalyzer
// ---------------------------------------------------------------------------

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
