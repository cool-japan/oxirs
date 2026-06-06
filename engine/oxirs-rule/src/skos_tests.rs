//! Tests for the SKOS reasoner.
//!
//! Flattened from the original `skos::tests` inline module so the source file
//! stays under the project's 2000-line refactor threshold.
#![cfg(test)]

use crate::skos_mappings::ConceptSchemeAnalyzer;
use crate::skos_reasoner::SkosReasoner;
use crate::skos_types::{
    label_matches, Graph, ALT_LABEL, BROADER, BROADER_TRANSITIVE, BROAD_MATCH, CLOSE_MATCH,
    EXACT_MATCH, HAS_TOP_CONCEPT, IN_SCHEME, NARROWER, NARROWER_TRANSITIVE, NARROW_MATCH,
    PREF_LABEL, RELATED, SKOS_NS, TOP_CONCEPT_OF,
};

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
// Rule S8: broader/narrower -> broaderTransitive/narrowerTransitive
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

    // One pass gives A->C and B->D; next fixpoint pass gives A->D
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
// Rule S3: broader <-> narrower symmetry
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

    // Two fixpoint iterations needed for A->D
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
// Rule S10: broadMatch <-> narrowMatch inverse
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
// apply_rules -- fixpoint integration
// ------------------------------------------------------------------

#[test]
fn test_apply_rules_broader_chain_fixpoint() -> anyhow::Result<()> {
    let mut graph = g();
    add(&mut graph, "ex:A", BROADER, "ex:B");
    add(&mut graph, "ex:B", BROADER, "ex:C");
    add(&mut graph, "ex:C", BROADER, "ex:D");

    let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed");

    // After fixpoint: A->BT->B, B->BT->C, C->BT->D, A->BT->C, A->BT->D, B->BT->D
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
    let new = SkosReasoner::apply_rules(&graph).expect("apply_rules should succeed on empty graph");
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
    let ancestors =
        SkosReasoner::broader_transitive_closure(&final_graph, &"ex:WheatProduction".to_owned())
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
