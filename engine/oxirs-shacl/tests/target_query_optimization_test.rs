//! Tests for target query optimization and edge cases
//!
//! This module tests advanced optimization scenarios and edge cases
//! for target query generation and selection.

use oxirs_core::model::{NamedNode, Term};
use oxirs_shacl::{
    paths::PropertyPath,
    targets::{
        selector::TargetSelector,
        types::{
            ConditionalTarget, HierarchicalRelationship, HierarchicalTarget, PathBasedTarget,
            PathDirection, PropertyDirection, Target, TargetCondition, TargetDifference,
            TargetIntersection, TargetUnion,
        },
    },
};

/// Test empty union target
#[test]
fn test_empty_union_target() {
    let selector = TargetSelector::new();

    let union_target = Target::Union(TargetUnion {
        targets: vec![],
        optimization_hints: None,
    });

    let query = selector.generate_target_query(&union_target, None).unwrap();

    // Should return empty query for empty union
    assert!(query.contains("SELECT DISTINCT ?target WHERE { }"));
}

/// Test single target intersection (should optimize to direct query)
#[test]
fn test_single_target_intersection() {
    let selector = TargetSelector::new();

    let class_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let intersection_target = Target::Intersection(TargetIntersection {
        targets: vec![class_target],
        optimization_hints: None,
    });

    let query = selector
        .generate_target_query(&intersection_target, None)
        .unwrap();

    // Should be optimized to direct class query
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/Person"));
    assert!(!query.contains("FILTER")); // Should not have intersection logic for single target
}

/// Test target query with graph context
#[test]
fn test_target_query_with_graph_context() {
    let selector = TargetSelector::new();

    let class_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let graph_name = "http://example.org/myGraph";

    let query = selector
        .generate_target_query(&class_target, Some(graph_name))
        .unwrap();

    // Should include GRAPH clause
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("GRAPH"));
    assert!(query.contains("http://example.org/myGraph"));
    assert!(query.contains("http://example.org/Person"));
}

/// Test hierarchical target with unlimited depth
#[test]
fn test_hierarchical_target_unlimited_depth() {
    let selector = TargetSelector::new();

    let root_target = Target::Class(NamedNode::new("http://example.org/Concept").unwrap());

    let hierarchical_target = Target::Hierarchical(HierarchicalTarget {
        root_target: Box::new(root_target),
        relationship: HierarchicalRelationship::SubclassOf,
        max_depth: -1, // Unlimited
        include_intermediate: true,
    });

    let query = selector
        .generate_target_query(&hierarchical_target, None)
        .unwrap();

    // Should use default reasonable limit
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("50")); // Default limit
}

/// Test inverse property hierarchical relationship
#[test]
fn test_inverse_property_hierarchical() {
    let selector = TargetSelector::new();

    let root_target = Target::Node(Term::NamedNode(
        NamedNode::new("http://example.org/leaf").unwrap(),
    ));

    let hierarchical_target = Target::Hierarchical(HierarchicalTarget {
        root_target: Box::new(root_target),
        relationship: HierarchicalRelationship::InverseProperty(
            NamedNode::new("http://example.org/parent").unwrap(),
        ),
        max_depth: 2,
        include_intermediate: true,
    });

    let query = selector
        .generate_target_query(&hierarchical_target, None)
        .unwrap();

    println!("Generated inverse property hierarchical query: {}", query);

    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(
        query.contains("http://example.org/leaf"),
        "Query should contain leaf IRI. Actual query: {}",
        query
    );
    assert!(
        query.contains("^<http://example.org/parent>"),
        "Query should have inverse syntax. Actual query: {}",
        query
    );
}

/// Test custom SPARQL path in hierarchical relationship
#[test]
fn test_custom_sparql_path_hierarchical() {
    let selector = TargetSelector::new();

    let root_target = Target::Class(NamedNode::new("http://example.org/Organization").unwrap());

    let hierarchical_target = Target::Hierarchical(HierarchicalTarget {
        root_target: Box::new(root_target),
        relationship: HierarchicalRelationship::CustomPath(
            "<http://example.org/partOf>+".to_string(),
        ),
        max_depth: 3,
        include_intermediate: false,
    });

    let query = selector
        .generate_target_query(&hierarchical_target, None)
        .unwrap();

    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/Organization"));
    assert!(query.contains("<http://example.org/partOf>+"));
}

/// Test path-based target with backward direction
#[test]
fn test_path_based_target_backward_direction() {
    let selector = TargetSelector::new();

    let start_target = Target::ObjectsOf(NamedNode::new("http://example.org/hasAuthor").unwrap());
    let authored_path =
        PropertyPath::predicate(NamedNode::new("http://example.org/wrote").unwrap());

    let path_target = Target::PathBased(PathBasedTarget {
        start_target: Box::new(start_target),
        path: authored_path,
        direction: PathDirection::Backward,
        filters: vec![],
    });

    let query = selector.generate_target_query(&path_target, None).unwrap();

    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/hasAuthor"));
    assert!(query.contains("?target"));
    assert!(query.contains("http://example.org/wrote"));
    assert!(query.contains("?startNode"));
}

/// Test conditional target with SPARQL ASK condition
#[test]
fn test_conditional_target_sparql_ask() {
    let selector = TargetSelector::new();

    let base_target = Target::Class(NamedNode::new("http://example.org/Product").unwrap());
    let condition = TargetCondition::SparqlAsk {
        query: "?target <http://example.org/inStock> true".to_string(),
        prefixes: Some("PREFIX ex: <http://example.org/>".to_string()),
    };

    let conditional_target = Target::Conditional(ConditionalTarget {
        base_target: Box::new(base_target),
        condition,
        context: None,
    });

    let query = selector
        .generate_target_query(&conditional_target, None)
        .unwrap();

    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/Product"));
    assert!(query.contains("FILTER EXISTS"));
    assert!(query.contains("PREFIX ex:"));
    assert!(query.contains("inStock"));
}

/// Test property direction variations in conditional targets
#[test]
fn test_conditional_target_property_directions() {
    let selector = TargetSelector::new();

    // Test Object direction
    let base_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let condition = TargetCondition::PropertyExists {
        property: NamedNode::new("http://example.org/likes").unwrap(),
        direction: PropertyDirection::Object,
    };

    let conditional_target = Target::Conditional(ConditionalTarget {
        base_target: Box::new(base_target),
        condition,
        context: None,
    });

    let query = selector
        .generate_target_query(&conditional_target, None)
        .unwrap();

    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("?conditionValue <http://example.org/likes> ?target"));

    // Test Either direction
    let condition_either = TargetCondition::PropertyExists {
        property: NamedNode::new("http://example.org/knows").unwrap(),
        direction: PropertyDirection::Either,
    };

    let conditional_target_either = Target::Conditional(ConditionalTarget {
        base_target: Box::new(Target::Class(
            NamedNode::new("http://example.org/Person").unwrap(),
        )),
        condition: condition_either,
        context: None,
    });

    let query_either = selector
        .generate_target_query(&conditional_target_either, None)
        .unwrap();

    assert!(query_either.contains("UNION"));
    assert!(query_either.contains("?target <http://example.org/knows> ?conditionValue"));
    assert!(query_either.contains("?conditionValue <http://example.org/knows> ?target"));
}

/// Test multiple complex intersection with optimization
#[test]
fn test_multiple_complex_intersection() {
    let selector = TargetSelector::new();

    // Create intersection of multiple different target types
    let class_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let objects_target =
        Target::ObjectsOf(NamedNode::new("http://example.org/hasEmployee").unwrap());
    let node_target = Target::Node(Term::NamedNode(
        NamedNode::new("http://example.org/specific").unwrap(),
    ));

    let intersection_target = Target::Intersection(TargetIntersection {
        targets: vec![class_target, objects_target, node_target],
        optimization_hints: None,
    });

    let query = selector
        .generate_target_query(&intersection_target, None)
        .unwrap();

    println!("Generated multiple complex intersection query: {}", query);

    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(
        query.contains("?target_0"),
        "Query should contain target_0 variable. Actual query: {}",
        query
    );
    assert!(
        query.contains("?target_1"),
        "Query should contain target_1 variable. Actual query: {}",
        query
    );
    assert!(
        query.contains("?target_2"),
        "Query should contain target_2 variable. Actual query: {}",
        query
    );
    assert!(
        query.contains("FILTER(?target = ?target_1)"),
        "Query should contain target comparison filter. Actual query: {}",
        query
    );
    assert!(
        query.contains("FILTER(?target = ?target_2)"),
        "Query should contain target comparison filter. Actual query: {}",
        query
    );
    assert!(
        query.contains("BIND(?target_0 AS ?target)"),
        "Query should contain target binding. Actual query: {}",
        query
    );
}

/// Test WHERE clause extraction edge cases
#[test]
fn test_where_clause_extraction_edge_cases() {
    let selector = TargetSelector::new();

    // Test with malformed query (should handle gracefully)
    let primary_target = Target::Class(NamedNode::new("http://example.org/ValidClass").unwrap());
    let exclusion_target =
        Target::Class(NamedNode::new("http://example.org/ExcludedClass").unwrap());

    let difference_target = Target::Difference(TargetDifference {
        primary_target: Box::new(primary_target),
        exclusion_target: Box::new(exclusion_target),
    });

    let query = selector
        .generate_target_query(&difference_target, None)
        .unwrap();

    // Should still generate valid difference query
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("FILTER NOT EXISTS"));
    assert!(query.contains("http://example.org/ValidClass"));
    assert!(query.contains("http://example.org/ExcludedClass"));
}

/// Test large union target performance
#[test]
fn test_large_union_target_performance() {
    let selector = TargetSelector::new();

    // Create union with many targets
    let mut targets = Vec::new();
    for i in 1..=20 {
        targets.push(Target::Class(
            NamedNode::new(&format!("http://example.org/Class{}", i)).unwrap(),
        ));
    }

    let union_target = Target::Union(TargetUnion {
        targets,
        optimization_hints: None,
    });

    let query = selector.generate_target_query(&union_target, None).unwrap();

    // Should handle large union efficiently
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("UNION"));
    assert!(query.contains("http://example.org/Class1"));
    assert!(query.contains("http://example.org/Class20"));

    // Query should not be excessively large
    assert!(query.len() < 10000); // Reasonable size limit
}

/// Test nested target combinations with complex paths
#[test]
fn test_nested_target_with_complex_paths() {
    let selector = TargetSelector::new();

    // Create path-based target with sequence path
    let start_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let complex_path = PropertyPath::sequence(vec![
        PropertyPath::predicate(NamedNode::new("http://example.org/worksFor").unwrap()),
        PropertyPath::predicate(NamedNode::new("http://example.org/hasLocation").unwrap()),
    ]);

    let path_target = Target::PathBased(PathBasedTarget {
        start_target: Box::new(start_target),
        path: complex_path,
        direction: PathDirection::Forward,
        filters: vec![],
    });

    // Wrap in intersection to test nested complex targets
    let intersection_target = Target::Intersection(TargetIntersection {
        targets: vec![
            path_target,
            Target::Class(NamedNode::new("http://example.org/Location").unwrap()),
        ],
        optimization_hints: None,
    });

    let query = selector
        .generate_target_query(&intersection_target, None)
        .unwrap();

    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/Person"));
    assert!(query.contains("http://example.org/worksFor"));
    assert!(query.contains("http://example.org/hasLocation"));
    assert!(query.contains("http://example.org/Location"));
}
