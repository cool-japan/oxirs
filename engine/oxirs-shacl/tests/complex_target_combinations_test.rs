//! Tests for complex target combinations in SHACL validation
//!
//! This module tests the advanced target combination features including
//! union, intersection, difference, conditional, hierarchical, and path-based targets.

use oxirs_core::model::{Literal, NamedNode, Term};
use oxirs_core::vocab::xsd;
use oxirs_shacl::{
    paths::PropertyPath,
    targets::{
        selector::TargetSelector,
        types::{
            ConditionalTarget, HierarchicalRelationship, HierarchicalTarget, NodeTypeFilter,
            PathBasedTarget, PathDirection, PathFilter, PropertyDirection, Target, TargetCondition,
            TargetContext, TargetDifference, TargetIntersection, TargetUnion,
        },
    },
};
use std::collections::HashMap;

/// Test union target query generation
#[test]
fn test_union_target_query_generation() {
    let selector = TargetSelector::new();

    // Create union of class and node targets
    let class_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let node_target = Target::Node(Term::NamedNode(
        NamedNode::new("http://example.org/john").unwrap(),
    ));

    let union_target = Target::Union(TargetUnion {
        targets: vec![class_target, node_target],
        optimization_hints: None,
    });

    let query = selector.generate_target_query(&union_target, None).unwrap();

    println!("Generated union query: {query}");

    // Should contain UNION clause
    assert!(
        query.contains("UNION"),
        "Query should contain UNION clause. Actual query: {query}"
    );
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/Person"));
    assert!(query.contains("http://example.org/john"));
}

/// Test intersection target query generation
#[test]
fn test_intersection_target_query_generation() {
    let selector = TargetSelector::new();

    // Create intersection of two class targets
    let class1_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let class2_target = Target::Class(NamedNode::new("http://example.org/Employee").unwrap());

    let intersection_target = Target::Intersection(TargetIntersection {
        targets: vec![class1_target, class2_target],
        optimization_hints: None,
    });

    let query = selector
        .generate_target_query(&intersection_target, None)
        .unwrap();

    // Should contain filtered constraints for intersection
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("FILTER"));
    assert!(query.contains("http://example.org/Person"));
    assert!(query.contains("http://example.org/Employee"));
}

/// Test difference target query generation
#[test]
fn test_difference_target_query_generation() {
    let selector = TargetSelector::new();

    // Create difference: all Persons except Employees
    let person_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let employee_target = Target::Class(NamedNode::new("http://example.org/Employee").unwrap());

    let difference_target = Target::Difference(TargetDifference {
        primary_target: Box::new(person_target),
        exclusion_target: Box::new(employee_target),
    });

    let query = selector
        .generate_target_query(&difference_target, None)
        .unwrap();

    // Should contain FILTER NOT EXISTS clause
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("FILTER NOT EXISTS"));
    assert!(query.contains("http://example.org/Person"));
    assert!(query.contains("http://example.org/Employee"));
}

/// Test conditional target query generation
#[test]
fn test_conditional_target_query_generation() {
    let selector = TargetSelector::new();

    // Create conditional target: Persons that have a name property
    let base_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let condition = TargetCondition::PropertyExists {
        property: NamedNode::new("http://foaf.org/spec/name").unwrap(),
        direction: PropertyDirection::Subject,
    };

    // Create context with custom bindings
    let mut bindings = HashMap::new();
    bindings.insert(
        "customVar".to_string(),
        Term::NamedNode(NamedNode::new("http://example.org/value").unwrap()),
    );

    let context = TargetContext {
        bindings,
        prefixes: HashMap::new(),
    };

    let conditional_target = Target::Conditional(ConditionalTarget {
        base_target: Box::new(base_target),
        condition,
        context: Some(context),
    });

    let query = selector
        .generate_target_query(&conditional_target, None)
        .unwrap();

    // Should contain the base class and the condition
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/Person"));
    assert!(query.contains("http://foaf.org/spec/name"));
    assert!(query.contains("BIND"));
    assert!(query.contains("customVar"));
}

/// Test hierarchical target query generation
#[test]
fn test_hierarchical_target_query_generation() {
    let selector = TargetSelector::new();

    // Create hierarchical target following subclass relationships
    let root_target = Target::Class(NamedNode::new("http://example.org/Animal").unwrap());

    let hierarchical_target = Target::Hierarchical(HierarchicalTarget {
        root_target: Box::new(root_target),
        relationship: HierarchicalRelationship::SubclassOf,
        max_depth: 5,
        include_intermediate: true,
    });

    let query = selector
        .generate_target_query(&hierarchical_target, None)
        .unwrap();

    println!("Generated hierarchical subclass query: {query}");

    // Should contain hierarchical relationship pattern
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/Animal"));
    assert!(query.contains("subClassOf"));
    assert!(
        query.contains("BIND"),
        "Query should contain BIND statement. Actual query: {query}"
    );
}

/// Test path-based target query generation
#[test]
fn test_path_based_target_query_generation() {
    let selector = TargetSelector::new();

    // Create path-based target: people reachable through friendship
    let start_target = Target::Node(Term::NamedNode(
        NamedNode::new("http://example.org/alice").unwrap(),
    ));
    let friendship_path =
        PropertyPath::predicate(NamedNode::new("http://foaf.org/spec/knows").unwrap());

    let path_target = Target::PathBased(PathBasedTarget {
        start_target: Box::new(start_target),
        path: friendship_path,
        direction: PathDirection::Forward,
        filters: vec![
            PathFilter::NodeType(NodeTypeFilter::IriOnly),
            PathFilter::PropertyValue {
                property: NamedNode::new("http://example.org/active").unwrap(),
                value: Term::Literal(Literal::new_typed("true", xsd::BOOLEAN.clone())),
            },
        ],
    });

    let query = selector.generate_target_query(&path_target, None).unwrap();

    println!("Generated path-based query: {query}");

    // Should contain path traversal and filters
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(
        query.contains("http://example.org/alice"),
        "Query should contain alice IRI. Actual query: {query}"
    );
    assert!(
        query.contains("http://foaf.org/spec/knows"),
        "Query should contain knows property. Actual query: {query}"
    );
    assert!(
        query.contains("FILTER(isIRI(?target))"),
        "Query should contain IRI filter. Actual query: {query}"
    );
    assert!(
        query.contains("http://example.org/active"),
        "Query should contain active property. Actual query: {query}"
    );
}

/// Test conditional target with property value condition
#[test]
fn test_conditional_target_with_property_value() {
    let selector = TargetSelector::new();

    let base_target = Target::Class(NamedNode::new("http://example.org/Document").unwrap());
    let condition = TargetCondition::PropertyValue {
        property: NamedNode::new("http://example.org/status").unwrap(),
        value: Term::Literal(Literal::new_simple_literal("published")),
        direction: PropertyDirection::Subject,
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
    assert!(query.contains("http://example.org/Document"));
    assert!(query.contains("http://example.org/status"));
    assert!(query.contains("\"published\""));
}

/// Test conditional target with type condition
#[test]
fn test_conditional_target_with_type_condition() {
    let selector = TargetSelector::new();

    let base_target = Target::ObjectsOf(NamedNode::new("http://example.org/hasMember").unwrap());
    let condition = TargetCondition::HasType {
        class_iri: NamedNode::new("http://example.org/ActiveMember").unwrap(),
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
    assert!(query.contains("http://example.org/hasMember"));
    assert!(query.contains("http://example.org/ActiveMember"));
    assert!(query.contains("rdf-syntax-ns#type"));
}

/// Test conditional target with cardinality condition
#[test]
fn test_conditional_target_with_cardinality_condition() {
    let selector = TargetSelector::new();

    let base_target = Target::Class(NamedNode::new("http://example.org/Group").unwrap());
    let condition = TargetCondition::Cardinality {
        property: NamedNode::new("http://example.org/hasMember").unwrap(),
        min_count: Some(5),
        max_count: Some(20),
        direction: PropertyDirection::Subject,
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
    assert!(query.contains("http://example.org/Group"));
    assert!(query.contains("http://example.org/hasMember"));
    assert!(query.contains("COUNT"));
    assert!(query.contains("GROUP BY"));
    assert!(query.contains(">= 5"));
    assert!(query.contains("<= 20"));
}

/// Test hierarchical target with custom property
#[test]
fn test_hierarchical_target_with_custom_property() {
    let selector = TargetSelector::new();

    let root_target = Target::Node(Term::NamedNode(
        NamedNode::new("http://example.org/ceo").unwrap(),
    ));

    let hierarchical_target = Target::Hierarchical(HierarchicalTarget {
        root_target: Box::new(root_target),
        relationship: HierarchicalRelationship::Property(
            NamedNode::new("http://example.org/reportsTo").unwrap(),
        ),
        max_depth: 3,
        include_intermediate: false,
    });

    let query = selector
        .generate_target_query(&hierarchical_target, None)
        .unwrap();

    println!("Generated hierarchical query: {query}");

    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(
        query.contains("http://example.org/ceo"),
        "Query should contain CEO IRI. Actual query: {query}"
    );
    assert!(
        query.contains("http://example.org/reportsTo"),
        "Query should contain reportsTo property. Actual query: {query}"
    );
    assert!(
        query.contains("+"),
        "Query should use + pattern for descendants only. Actual query: {query}"
    );
}

/// Test path-based target with bidirectional traversal
#[test]
fn test_path_based_target_bidirectional() {
    let selector = TargetSelector::new();

    let start_target = Target::Class(NamedNode::new("http://example.org/City").unwrap());
    let connection_path =
        PropertyPath::predicate(NamedNode::new("http://example.org/connectedTo").unwrap());

    let path_target = Target::PathBased(PathBasedTarget {
        start_target: Box::new(start_target),
        path: connection_path,
        direction: PathDirection::Both,
        filters: vec![PathFilter::PropertyValue {
            property: NamedNode::new("http://example.org/population").unwrap(),
            value: Term::Literal(Literal::new_typed("100000", xsd::INTEGER.clone())),
        }],
    });

    let query = selector.generate_target_query(&path_target, None).unwrap();

    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/City"));
    assert!(query.contains("http://example.org/connectedTo"));
    assert!(query.contains("UNION")); // Should have UNION for both directions
    assert!(query.contains("http://example.org/population"));
}

/// Test complex nested target combinations
#[test]
fn test_complex_nested_target_combinations() {
    let selector = TargetSelector::new();

    // Create a complex nested combination:
    // (Persons OR Documents) AND (NOT Archived) AND HasActiveStatus
    let person_target = Target::Class(NamedNode::new("http://example.org/Person").unwrap());
    let document_target = Target::Class(NamedNode::new("http://example.org/Document").unwrap());

    let union_target = Target::Union(TargetUnion {
        targets: vec![person_target, document_target],
        optimization_hints: None,
    });

    let archived_target = Target::Class(NamedNode::new("http://example.org/Archived").unwrap());
    let difference_target = Target::Difference(TargetDifference {
        primary_target: Box::new(union_target),
        exclusion_target: Box::new(archived_target),
    });

    let condition = TargetCondition::PropertyValue {
        property: NamedNode::new("http://example.org/status").unwrap(),
        value: Term::Literal(Literal::new_simple_literal("active")),
        direction: PropertyDirection::Subject,
    };

    let final_target = Target::Conditional(ConditionalTarget {
        base_target: Box::new(difference_target),
        condition,
        context: None,
    });

    let query = selector.generate_target_query(&final_target, None).unwrap();

    // Should be a complex query with multiple clauses
    assert!(query.contains("SELECT DISTINCT ?target"));
    assert!(query.contains("http://example.org/Person"));
    assert!(query.contains("http://example.org/Document"));
    assert!(query.contains("http://example.org/Archived"));
    assert!(query.contains("http://example.org/status"));
    assert!(query.contains("\"active\""));
}
