//! Store-backed tests for recursive shape validation.
//!
//! These exercise the real depth-first / breadth-first / optimized traversal,
//! cycle detection, depth bounding, and dependency extraction. They live in a
//! sibling file (declared from `advanced_features/mod.rs`) to keep
//! `recursive_shapes.rs` under the workspace size policy.

use super::recursive_shapes::{
    RecursionStrategy, RecursiveShapeValidator, RecursiveValidationConfig, ShapeDependencyAnalyzer,
    ShapeResolver,
};
use crate::constraints::logical_constraints::AndConstraint;
use crate::constraints::shape_constraints::NodeConstraint;
use crate::constraints::value_constraints::ClassConstraint;
use crate::{Constraint, ConstraintComponentId, PropertyPath, Shape, ShapeId};
use oxirs_core::{
    model::{GraphName, NamedNode, Object, Predicate, Quad, Subject, Term},
    ConcreteStore,
};
use std::collections::HashMap;

const EX: &str = "http://example.org/";
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

fn iri(local: &str) -> NamedNode {
    NamedNode::new(format!("{EX}{local}")).expect("valid IRI")
}

fn term(local: &str) -> Term {
    Term::NamedNode(iri(local))
}

fn cid(s: &str) -> ConstraintComponentId {
    ConstraintComponentId::new(s)
}

fn insert(store: &ConcreteStore, subject: &str, predicate: &str, object_local: &str) {
    let quad = Quad::new(
        Subject::from(iri(subject)),
        Predicate::from(NamedNode::new(predicate).expect("predicate IRI")),
        Object::from(iri(object_local)),
        GraphName::DefaultGraph,
    );
    store.insert_quad(quad).expect("insert quad");
}

fn insert_next(store: &ConcreteStore, from: &str, to: &str) {
    insert(store, from, &format!("{EX}next"), to);
}

fn insert_type_item(store: &ConcreteStore, subject: &str) {
    insert(store, subject, RDF_TYPE, "Item");
}

/// In-memory shape resolver backed by a `HashMap`.
struct MapResolver {
    shapes: HashMap<ShapeId, Shape>,
}

impl MapResolver {
    fn new() -> Self {
        Self {
            shapes: HashMap::new(),
        }
    }

    fn add(&mut self, shape: Shape) {
        self.shapes.insert(shape.id.clone(), shape);
    }
}

impl ShapeResolver for MapResolver {
    fn resolve_shape(&self, shape_id: &ShapeId) -> Option<&Shape> {
        self.shapes.get(shape_id)
    }

    fn get_referencing_shapes(&self, _shape_id: &ShapeId) -> Vec<&Shape> {
        Vec::new()
    }

    fn detect_circular_dependencies(&self) -> Vec<Vec<ShapeId>> {
        Vec::new()
    }
}

/// Build a linked-list style recursive shape pair:
/// * `ItemShape` (node shape): `sh:class :Item` + linked property shape `NextShape`.
/// * `NextShape` (property shape, path `:next`): `sh:node ItemShape`.
///
/// Returns a resolver pre-populated with both shapes.
fn linked_list_resolver() -> MapResolver {
    let mut item_shape = Shape::node_shape(ShapeId::new("ItemShape"));
    item_shape.add_constraint(
        cid("sh:ClassConstraintComponent"),
        Constraint::Class(ClassConstraint {
            class_iri: iri("Item"),
        }),
    );
    item_shape.property_shapes.push(ShapeId::new("NextShape"));

    let mut next_shape = Shape::property_shape(
        ShapeId::new("NextShape"),
        PropertyPath::Predicate(iri("next")),
    );
    next_shape.add_constraint(
        cid("sh:NodeConstraintComponent"),
        Constraint::Node(NodeConstraint::new(ShapeId::new("ItemShape"))),
    );

    let mut resolver = MapResolver::new();
    resolver.add(item_shape);
    resolver.add(next_shape);
    resolver
}

fn item_shape_clone(resolver: &MapResolver) -> Shape {
    resolver
        .resolve_shape(&ShapeId::new("ItemShape"))
        .expect("ItemShape present")
        .clone()
}

// ---------------------------------------------------------------------------
// Depth-first
// ---------------------------------------------------------------------------

#[test]
fn test_dfs_three_level_chain_conforms() {
    // n1 -> n2 -> n3, all typed :Item. n3 has no next (chain terminates).
    let store = ConcreteStore::new().expect("store");
    for n in ["n1", "n2", "n3"] {
        insert_type_item(&store, n);
    }
    insert_next(&store, "n1", "n2");
    insert_next(&store, "n2", "n3");

    let resolver = linked_list_resolver();
    let shape = item_shape_clone(&resolver);

    let mut validator = RecursiveShapeValidator::new(RecursiveValidationConfig {
        max_depth: 100,
        detect_cycles: true,
        cache_results: true,
        strategy: RecursionStrategy::DepthFirst,
    });

    let result = validator
        .validate_recursive(&term("n1"), &shape, &store, &resolver)
        .expect("validate");
    assert!(result.conforms, "well-formed 3-level chain must conform");
    assert!(
        result.depth_reached >= 3,
        "depth should reflect the chain length, got {}",
        result.depth_reached
    );
}

#[test]
fn test_dfs_nested_non_conforming() {
    // n2 is NOT typed :Item => the recursive sh:node check on n2 fails.
    let store = ConcreteStore::new().expect("store");
    insert_type_item(&store, "n1");
    // n2 deliberately missing rdf:type :Item
    insert_type_item(&store, "n3");
    insert_next(&store, "n1", "n2");
    insert_next(&store, "n2", "n3");

    let resolver = linked_list_resolver();
    let shape = item_shape_clone(&resolver);

    let mut validator = RecursiveShapeValidator::new(RecursiveValidationConfig {
        max_depth: 100,
        detect_cycles: true,
        cache_results: false,
        strategy: RecursionStrategy::DepthFirst,
    });

    let result = validator
        .validate_recursive(&term("n1"), &shape, &store, &resolver)
        .expect("validate");
    assert!(
        !result.conforms,
        "a non-conforming nested node must fail the whole validation"
    );
}

#[test]
fn test_dfs_cyclic_graph_terminates() {
    // n1 -> n2 -> n1 cycle, both typed :Item. Must terminate via cycle detection
    // and conform (the finite structure is valid).
    let store = ConcreteStore::new().expect("store");
    insert_type_item(&store, "n1");
    insert_type_item(&store, "n2");
    insert_next(&store, "n1", "n2");
    insert_next(&store, "n2", "n1");

    let resolver = linked_list_resolver();
    let shape = item_shape_clone(&resolver);

    let mut validator = RecursiveShapeValidator::new(RecursiveValidationConfig {
        max_depth: 1000,
        detect_cycles: true,
        cache_results: true,
        strategy: RecursionStrategy::DepthFirst,
    });

    let result = validator
        .validate_recursive(&term("n1"), &shape, &store, &resolver)
        .expect("cyclic validation must terminate, not overflow");
    assert!(result.conforms, "valid cyclic structure conforms");
    assert!(
        result.cycles_detected,
        "the back-edge n2->n1 must be detected as a cycle"
    );
    assert!(
        validator.stats().cycles_detected >= 1,
        "cycle statistic must be recorded"
    );
}

#[test]
fn test_dfs_exceeds_max_depth_errors() {
    // A 5-node chain with max_depth = 2 must hit the recursion limit.
    let store = ConcreteStore::new().expect("store");
    let nodes = ["m1", "m2", "m3", "m4", "m5"];
    for n in nodes {
        insert_type_item(&store, n);
    }
    for pair in nodes.windows(2) {
        insert_next(&store, pair[0], pair[1]);
    }

    let resolver = linked_list_resolver();
    let shape = item_shape_clone(&resolver);

    let mut validator = RecursiveShapeValidator::new(RecursiveValidationConfig {
        max_depth: 2,
        detect_cycles: true,
        cache_results: false,
        strategy: RecursionStrategy::DepthFirst,
    });

    let result = validator.validate_recursive(&term("m1"), &shape, &store, &resolver);
    assert!(
        result.is_err(),
        "chain deeper than max_depth must return a recursion-limit error"
    );
}

// ---------------------------------------------------------------------------
// Breadth-first
// ---------------------------------------------------------------------------

#[test]
fn test_bfs_three_level_chain_conforms() {
    let store = ConcreteStore::new().expect("store");
    for n in ["n1", "n2", "n3"] {
        insert_type_item(&store, n);
    }
    insert_next(&store, "n1", "n2");
    insert_next(&store, "n2", "n3");

    let resolver = linked_list_resolver();
    let shape = item_shape_clone(&resolver);

    let mut validator = RecursiveShapeValidator::new(RecursiveValidationConfig {
        max_depth: 100,
        detect_cycles: true,
        cache_results: true,
        strategy: RecursionStrategy::BreadthFirst,
    });

    let result = validator
        .validate_recursive(&term("n1"), &shape, &store, &resolver)
        .expect("validate");
    assert!(result.conforms, "BFS over a valid chain must conform");
    assert!(result.depth_reached >= 3, "BFS should reach chain depth");
}

#[test]
fn test_bfs_cyclic_graph_terminates() {
    let store = ConcreteStore::new().expect("store");
    insert_type_item(&store, "n1");
    insert_type_item(&store, "n2");
    insert_next(&store, "n1", "n2");
    insert_next(&store, "n2", "n1");

    let resolver = linked_list_resolver();
    let shape = item_shape_clone(&resolver);

    let mut validator = RecursiveShapeValidator::new(RecursiveValidationConfig {
        max_depth: 1000,
        detect_cycles: true,
        cache_results: true,
        strategy: RecursionStrategy::BreadthFirst,
    });

    let result = validator
        .validate_recursive(&term("n1"), &shape, &store, &resolver)
        .expect("BFS cyclic validation must terminate");
    assert!(result.conforms);
    assert!(
        result.cycles_detected,
        "BFS must flag the cycle via its seen-set"
    );
}

#[test]
fn test_bfs_nested_non_conforming() {
    let store = ConcreteStore::new().expect("store");
    insert_type_item(&store, "n1");
    // n2 missing :Item type
    insert_next(&store, "n1", "n2");

    let resolver = linked_list_resolver();
    let shape = item_shape_clone(&resolver);

    let mut validator = RecursiveShapeValidator::new(RecursiveValidationConfig {
        max_depth: 100,
        detect_cycles: true,
        cache_results: false,
        strategy: RecursionStrategy::BreadthFirst,
    });

    let result = validator
        .validate_recursive(&term("n1"), &shape, &store, &resolver)
        .expect("validate");
    assert!(!result.conforms, "BFS must detect non-conforming child");
}

// ---------------------------------------------------------------------------
// Optimized (cache-heavy)
// ---------------------------------------------------------------------------

#[test]
fn test_optimized_chain_conforms_and_uses_cache() {
    let store = ConcreteStore::new().expect("store");
    for n in ["n1", "n2", "n3"] {
        insert_type_item(&store, n);
    }
    insert_next(&store, "n1", "n2");
    insert_next(&store, "n2", "n3");

    let resolver = linked_list_resolver();
    let shape = item_shape_clone(&resolver);

    let mut validator = RecursiveShapeValidator::new(RecursiveValidationConfig {
        max_depth: 100,
        detect_cycles: true,
        cache_results: true,
        strategy: RecursionStrategy::OptimizedCycleBreaking,
    });

    // First run populates the cache.
    let first = validator
        .validate_recursive(&term("n1"), &shape, &store, &resolver)
        .expect("validate");
    assert!(first.conforms);

    // Second run on the same root should serve from the cache => a cache hit and
    // the `cached` flag set on the top-level result.
    let second = validator
        .validate_recursive(&term("n1"), &shape, &store, &resolver)
        .expect("validate");
    assert!(second.conforms);
    assert!(second.cached, "repeat top-level validation must be cached");
    assert!(
        validator.stats().cache_hits >= 1,
        "optimized strategy must register cache hits"
    );
}

#[test]
fn test_optimized_cyclic_terminates() {
    let store = ConcreteStore::new().expect("store");
    insert_type_item(&store, "n1");
    insert_type_item(&store, "n2");
    insert_next(&store, "n1", "n2");
    insert_next(&store, "n2", "n1");

    let resolver = linked_list_resolver();
    let shape = item_shape_clone(&resolver);

    let mut validator = RecursiveShapeValidator::new(RecursiveValidationConfig {
        max_depth: 1000,
        detect_cycles: true,
        cache_results: true,
        strategy: RecursionStrategy::OptimizedCycleBreaking,
    });

    let result = validator
        .validate_recursive(&term("n1"), &shape, &store, &resolver)
        .expect("optimized cyclic validation must terminate");
    assert!(result.conforms);
}

// ---------------------------------------------------------------------------
// Dependency extraction (ShapeDependencyAnalyzer::build_from_shapes)
// ---------------------------------------------------------------------------

#[test]
fn test_extract_dependencies_all_reference_kinds() {
    // A shape that references other shapes via property, extends, sh:node and sh:and.
    let mut shape = Shape::node_shape(ShapeId::new("Root"));
    shape.property_shapes.push(ShapeId::new("PropShape"));
    shape.extends.push(ShapeId::new("Parent"));
    shape.add_constraint(
        cid("sh:NodeConstraintComponent"),
        Constraint::Node(NodeConstraint::new(ShapeId::new("NodeRef"))),
    );
    shape.add_constraint(
        cid("sh:AndConstraintComponent"),
        Constraint::And(AndConstraint::new(vec![
            ShapeId::new("AndA"),
            ShapeId::new("AndB"),
        ])),
    );

    let mut analyzer = ShapeDependencyAnalyzer::new();
    analyzer.build_from_shapes(std::slice::from_ref(&shape));

    // Every referenced shape must now be a dependency edge from Root, so the
    // dependency depth must exceed 1 (Root -> something).
    let depth = analyzer.get_dependency_depth(&ShapeId::new("Root"));
    assert!(
        depth >= 2,
        "Root must depend on referenced shapes (depth >= 2), got {depth}"
    );
}

#[test]
fn test_extract_dependencies_detects_cycle() {
    // Two node shapes referencing each other via sh:node => a dependency cycle.
    let mut a = Shape::node_shape(ShapeId::new("A"));
    a.add_constraint(
        cid("sh:NodeConstraintComponent"),
        Constraint::Node(NodeConstraint::new(ShapeId::new("B"))),
    );
    let mut b = Shape::node_shape(ShapeId::new("B"));
    b.add_constraint(
        cid("sh:NodeConstraintComponent"),
        Constraint::Node(NodeConstraint::new(ShapeId::new("A"))),
    );

    let mut analyzer = ShapeDependencyAnalyzer::new();
    analyzer.build_from_shapes(&[a, b]);

    let cycles = analyzer.detect_cycles();
    assert!(
        !cycles.is_empty(),
        "mutually-referencing shapes must yield a dependency cycle"
    );
}

#[test]
fn test_extract_dependencies_empty_for_leaf() {
    // A shape with only a local (non-shape) constraint has no dependencies.
    let mut leaf = Shape::node_shape(ShapeId::new("Leaf"));
    leaf.add_constraint(
        cid("sh:ClassConstraintComponent"),
        Constraint::Class(ClassConstraint {
            class_iri: iri("Item"),
        }),
    );

    let mut analyzer = ShapeDependencyAnalyzer::new();
    analyzer.build_from_shapes(std::slice::from_ref(&leaf));

    // No edges recorded => topological sort succeeds and Leaf depth is 1 (itself).
    assert!(analyzer.topological_sort().is_ok());
    assert_eq!(analyzer.get_dependency_depth(&ShapeId::new("Leaf")), 1);
}
