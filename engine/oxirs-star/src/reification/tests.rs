use std::collections::HashMap;

use crate::model::{StarGraph, StarTerm, StarTriple};

use super::converter::{AdvancedReificator, Reificator};
use super::mapper::{
    count_reifications, has_reifications, validate_reifications, ReificationBridge,
};
use super::types::{
    AdvancedReificationStrategy, AnnotationStyle, EmbeddedTriple, ReificationCondition,
    ReificationRule, ReificationStrategy,
};
use super::vocab;

fn iri(s: &str) -> StarTerm {
    StarTerm::iri(s).expect("iri")
}
fn lit(s: &str) -> StarTerm {
    StarTerm::literal(s).expect("lit")
}

fn sample_triple() -> StarTriple {
    StarTriple::new(
        iri("http://example.org/alice"),
        iri("http://example.org/age"),
        lit("30"),
    )
}

#[test]
fn test_basic_reification() {
    let mut reificator = Reificator::new(ReificationStrategy::StandardReification, None);

    let quoted_triple = StarTriple::new(
        StarTerm::iri("http://example.org/subject").unwrap(),
        StarTerm::iri("http://example.org/predicate").unwrap(),
        StarTerm::iri("http://example.org/object").unwrap(),
    );

    let triple_with_quoted = StarTriple::new(
        StarTerm::QuotedTriple(Box::new(quoted_triple)),
        StarTerm::iri("http://example.org/hasMetadata").unwrap(),
        StarTerm::literal("metadata").unwrap(),
    );

    let reified_triples = reificator.reify_triple(&triple_with_quoted).unwrap();
    assert!(reified_triples.len() > 1);
}

#[test]
fn test_dereification() {
    let mut reificator = Reificator::new(ReificationStrategy::StandardReification, None);

    let mut star_graph = StarGraph::new();
    let quoted_triple = StarTriple::new(
        StarTerm::iri("http://example.org/subject").unwrap(),
        StarTerm::iri("http://example.org/predicate").unwrap(),
        StarTerm::iri("http://example.org/object").unwrap(),
    );

    let triple_with_quoted = StarTriple::new(
        StarTerm::QuotedTriple(Box::new(quoted_triple)),
        StarTerm::iri("http://example.org/hasMetadata").unwrap(),
        StarTerm::literal("metadata").unwrap(),
    );

    star_graph.insert(triple_with_quoted).unwrap();

    let reified_graph = reificator.reify_graph(&star_graph).unwrap();
    let dereified_graph = reificator.dereify_graph(&reified_graph).unwrap();

    assert_eq!(star_graph.len(), dereified_graph.len());
}

#[test]
fn test_advanced_reification_strategies() {
    let hybrid_strategy = AdvancedReificationStrategy::Hybrid {
        simple_strategy: ReificationStrategy::StandardReification,
        nested_strategy: ReificationStrategy::BlankNodes,
        predicate_strategies: HashMap::new(),
    };

    let mut advanced_reificator = AdvancedReificator::new(hybrid_strategy);

    let mut star_graph = StarGraph::new();
    let quoted_triple = StarTriple::new(
        StarTerm::iri("http://example.org/subject").unwrap(),
        StarTerm::iri("http://example.org/predicate").unwrap(),
        StarTerm::iri("http://example.org/object").unwrap(),
    );

    let triple_with_quoted = StarTriple::new(
        StarTerm::QuotedTriple(Box::new(quoted_triple)),
        StarTerm::iri("http://example.org/hasMetadata").unwrap(),
        StarTerm::literal("metadata").unwrap(),
    );

    star_graph.insert(triple_with_quoted).unwrap();

    let reified_graph = advanced_reificator
        .reify_graph_advanced(&star_graph)
        .unwrap();
    assert!(!reified_graph.is_empty());

    let stats = advanced_reificator.get_statistics();
    assert!(stats.total_triples > 0);
}

#[test]
fn test_conditional_reification() {
    let rules = vec![ReificationRule {
        condition: ReificationCondition::PredicateIri("http://example.org/special".to_string()),
        strategy: ReificationStrategy::BlankNodes,
        priority: 10,
    }];

    let conditional_strategy = AdvancedReificationStrategy::Conditional {
        default_strategy: ReificationStrategy::StandardReification,
        rules,
    };

    let mut advanced_reificator = AdvancedReificator::new(conditional_strategy);

    let mut star_graph = StarGraph::new();
    let quoted_triple = StarTriple::new(
        StarTerm::iri("http://example.org/subject").unwrap(),
        StarTerm::iri("http://example.org/special").unwrap(),
        StarTerm::iri("http://example.org/object").unwrap(),
    );

    let triple_with_quoted = StarTriple::new(
        StarTerm::QuotedTriple(Box::new(quoted_triple)),
        StarTerm::iri("http://example.org/hasMetadata").unwrap(),
        StarTerm::literal("metadata").unwrap(),
    );

    star_graph.insert(triple_with_quoted).unwrap();

    let reified_graph = advanced_reificator
        .reify_graph_advanced(&star_graph)
        .unwrap();
    assert!(!reified_graph.is_empty());

    let stats = advanced_reificator.get_statistics();
    assert!(!stats.strategy_usage.is_empty());
}

#[test]
fn test_additional_basic_reification() {
    let mut reificator = Reificator::new(
        ReificationStrategy::StandardReification,
        Some("http://example.org/stmt/".to_string()),
    );

    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );

    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );

    let mut star_graph = StarGraph::new();
    star_graph.insert(outer).unwrap();

    let reified = reificator.reify_graph(&star_graph).unwrap();
    assert!(reified.len() > 1);

    let has_type_triple = reified.triples().iter().any(|t| {
        if let (StarTerm::NamedNode(p), StarTerm::NamedNode(o)) = (&t.predicate, &t.object) {
            p.iri == vocab::RDF_TYPE && o.iri == vocab::RDF_STATEMENT
        } else {
            false
        }
    });
    assert!(has_type_triple);
}

#[test]
fn test_additional_dereification() {
    let mut reified_graph = StarGraph::new();

    let stmt_iri = "http://example.org/stmt/1";

    reified_graph
        .insert(StarTriple::new(
            StarTerm::iri(stmt_iri).unwrap(),
            StarTerm::iri(vocab::RDF_TYPE).unwrap(),
            StarTerm::iri(vocab::RDF_STATEMENT).unwrap(),
        ))
        .unwrap();

    reified_graph
        .insert(StarTriple::new(
            StarTerm::iri(stmt_iri).unwrap(),
            StarTerm::iri(vocab::RDF_SUBJECT).unwrap(),
            StarTerm::iri("http://example.org/alice").unwrap(),
        ))
        .unwrap();

    reified_graph
        .insert(StarTriple::new(
            StarTerm::iri(stmt_iri).unwrap(),
            StarTerm::iri(vocab::RDF_PREDICATE).unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
        ))
        .unwrap();

    reified_graph
        .insert(StarTriple::new(
            StarTerm::iri(stmt_iri).unwrap(),
            StarTerm::iri(vocab::RDF_OBJECT).unwrap(),
            StarTerm::literal("25").unwrap(),
        ))
        .unwrap();

    reified_graph
        .insert(StarTriple::new(
            StarTerm::iri(stmt_iri).unwrap(),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal("0.9").unwrap(),
        ))
        .unwrap();

    let mut dereificator = Reificator::new(
        ReificationStrategy::StandardReification,
        Some("http://example.org/stmt/".to_string()),
    );

    let star_graph = dereificator.dereify_graph(&reified_graph).unwrap();
    assert_eq!(star_graph.len(), 1);

    let triple = &star_graph.triples()[0];
    assert!(triple.subject.is_quoted_triple());
}

#[test]
fn test_reification_roundtrip() {
    let inner = StarTriple::new(
        StarTerm::iri("http://example.org/alice").unwrap(),
        StarTerm::iri("http://example.org/age").unwrap(),
        StarTerm::literal("25").unwrap(),
    );

    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner),
        StarTerm::iri("http://example.org/certainty").unwrap(),
        StarTerm::literal("0.9").unwrap(),
    );

    let mut original_graph = StarGraph::new();
    original_graph.insert(outer).unwrap();

    let mut reificator = Reificator::new(
        ReificationStrategy::StandardReification,
        Some("http://example.org/stmt/".to_string()),
    );
    let reified_graph = reificator.reify_graph(&original_graph).unwrap();

    let mut dereificator = Reificator::new(
        ReificationStrategy::StandardReification,
        Some("http://example.org/stmt/".to_string()),
    );
    let recovered_graph = dereificator.dereify_graph(&reified_graph).unwrap();

    assert_eq!(recovered_graph.len(), original_graph.len());

    let recovered_triple = &recovered_graph.triples()[0];
    assert!(recovered_triple.subject.is_quoted_triple());
}

#[test]
fn test_utils() {
    let mut graph = StarGraph::new();

    graph
        .insert(StarTriple::new(
            StarTerm::iri("http://example.org/stmt1").unwrap(),
            StarTerm::iri(vocab::RDF_SUBJECT).unwrap(),
            StarTerm::iri("http://example.org/alice").unwrap(),
        ))
        .unwrap();

    assert!(has_reifications(&graph));
    assert_eq!(count_reifications(&graph), 1);
    assert!(validate_reifications(&graph).is_err());
}

#[test]
fn test_bridge_reification_style_generates_four_triples() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let triples = bridge.star_to_reification(&sample_triple());
    assert_eq!(
        triples.len(),
        4,
        "rdf:Statement reification needs 4 triples"
    );
}

#[test]
fn test_bridge_reification_contains_rdf_type() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let triples = bridge.star_to_reification(&sample_triple());
    let has_type = triples.iter().any(|t| {
        if let (StarTerm::NamedNode(p), StarTerm::NamedNode(o)) = (&t.predicate, &t.object) {
            p.iri == vocab::RDF_TYPE && o.iri == vocab::RDF_STATEMENT
        } else {
            false
        }
    });
    assert!(has_type, "must include rdf:type rdf:Statement triple");
}

#[test]
fn test_bridge_reification_contains_rdf_subject() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let triples = bridge.star_to_reification(&sample_triple());
    let has_subject = triples.iter().any(|t| {
        if let StarTerm::NamedNode(p) = &t.predicate {
            p.iri == vocab::RDF_SUBJECT
        } else {
            false
        }
    });
    assert!(has_subject);
}

#[test]
fn test_bridge_reification_contains_rdf_predicate() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let triples = bridge.star_to_reification(&sample_triple());
    let has_predicate = triples.iter().any(|t| {
        if let StarTerm::NamedNode(p) = &t.predicate {
            p.iri == vocab::RDF_PREDICATE
        } else {
            false
        }
    });
    assert!(has_predicate);
}

#[test]
fn test_bridge_reification_contains_rdf_object() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let triples = bridge.star_to_reification(&sample_triple());
    let has_object = triples.iter().any(|t| {
        if let StarTerm::NamedNode(p) = &t.predicate {
            p.iri == vocab::RDF_OBJECT
        } else {
            false
        }
    });
    assert!(has_object);
}

#[test]
fn test_bridge_singleton_style_generates_two_triples() {
    let bridge = ReificationBridge::new(AnnotationStyle::Singleton);
    let triples = bridge.star_to_reification(&sample_triple());
    assert_eq!(triples.len(), 2, "singleton style needs 2 triples");
}

#[test]
fn test_bridge_nary_relation_style_generates_three_triples() {
    let bridge = ReificationBridge::new(AnnotationStyle::NaryRelation);
    let triples = bridge.star_to_reification(&sample_triple());
    assert_eq!(triples.len(), 3, "nary relation style needs 3 triples");
}

#[test]
fn test_bridge_reification_roundtrip() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let original = sample_triple();
    let reified = bridge.star_to_reification(&original);
    let recovered = bridge.reification_to_star(&reified);
    assert!(recovered.is_some(), "should recover the embedded triple");
    let recovered = recovered.unwrap();
    assert_eq!(recovered.subject, original.subject);
    assert_eq!(recovered.predicate, original.predicate);
    assert_eq!(recovered.object, original.object);
}

#[test]
fn test_bridge_reification_to_star_incomplete_returns_none() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let partial = vec![StarTriple::new(
        iri("http://example.org/stmt1"),
        iri(vocab::RDF_SUBJECT),
        iri("http://example.org/alice"),
    )];
    let recovered = bridge.reification_to_star(&partial);
    assert!(
        recovered.is_none(),
        "incomplete reification should return None"
    );
}

#[test]
fn test_bridge_reification_to_star_empty_returns_none() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let recovered = bridge.reification_to_star(&[]);
    assert!(recovered.is_none());
}

#[test]
fn test_convert_graph_to_reification_expands_quoted_subjects() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let inner = sample_triple();
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner),
        iri("http://example.org/certainty"),
        lit("high"),
    );
    let expanded = bridge.convert_graph_to_reification(&[outer]);
    assert!(expanded.len() >= 4);
}

#[test]
fn test_convert_graph_plain_triples_pass_through() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let plain = sample_triple();
    let expanded = bridge.convert_graph_to_reification(std::slice::from_ref(&plain));
    assert_eq!(expanded.len(), 1);
    assert_eq!(expanded[0], plain);
}

#[test]
fn test_convert_reification_to_star_roundtrip() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let inner = sample_triple();
    let reified = bridge.star_to_reification(&inner);
    let recovered = bridge.convert_reification_to_star(&reified);
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].subject, inner.subject);
    assert_eq!(recovered[0].predicate, inner.predicate);
    assert_eq!(recovered[0].object, inner.object);
}

#[test]
fn test_convert_reification_to_star_multiple_clusters() {
    let bridge = ReificationBridge::new(AnnotationStyle::Reification);
    let t1 = sample_triple();
    let t2 = StarTriple::new(
        iri("http://example.org/bob"),
        iri("http://example.org/age"),
        lit("25"),
    );
    let mut all_reified = bridge.star_to_reification(&t1);
    all_reified.extend(bridge.star_to_reification(&t2));
    let recovered = bridge.convert_reification_to_star(&all_reified);
    assert_eq!(recovered.len(), 2, "should recover both embedded triples");
}

#[test]
fn test_annotation_style_default_is_reification() {
    assert_eq!(AnnotationStyle::default(), AnnotationStyle::Reification);
}

#[test]
fn test_bridge_with_base_iri() {
    let bridge =
        ReificationBridge::with_base_iri(AnnotationStyle::Reification, "http://my.org/stmts/");
    let triples = bridge.star_to_reification(&sample_triple());
    let stmt_node = triples
        .iter()
        .filter_map(|t| {
            if let StarTerm::NamedNode(n) = &t.subject {
                Some(n.iri.clone())
            } else {
                None
            }
        })
        .next();
    assert!(stmt_node.is_some());
    assert!(
        stmt_node.unwrap().starts_with("http://my.org/stmts/"),
        "statement node should use custom base IRI"
    );
}

#[test]
fn test_embedded_triple_type_alias() {
    let et: EmbeddedTriple = sample_triple();
    assert!(et.validate().is_ok());
}
