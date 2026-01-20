//! Tests for RDF Collection Serialization in Turtle
//!
//! Tests for compact collection syntax using `(item1 item2 item3)` instead of
//! verbose rdf:first/rdf:rest/rdf:nil representation.

use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple};
use oxirs_ttl::formats::turtle::TurtleSerializer;
use oxirs_ttl::toolkit::Serializer;

/// Helper to create rdf:first predicate
fn rdf_first() -> Predicate {
    Predicate::NamedNode(
        NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#first").unwrap(),
    )
}

/// Helper to create rdf:rest predicate
fn rdf_rest() -> Predicate {
    Predicate::NamedNode(NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest").unwrap())
}

/// Helper to create rdf:nil object
fn rdf_nil() -> Object {
    Object::NamedNode(NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil").unwrap())
}

#[test]
fn test_simple_collection() {
    // Create a simple collection: (ex:a ex:b ex:c)
    // Represented as:
    // ex:alice ex:likes _:b1 .
    // _:b1 rdf:first ex:a ; rdf:rest _:b2 .
    // _:b2 rdf:first ex:b ; rdf:rest _:b3 .
    // _:b3 rdf:first ex:c ; rdf:rest rdf:nil .

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let likes_pred = Predicate::NamedNode(NamedNode::new("http://example.org/likes").unwrap());

    let b1 = BlankNode::new("b1").unwrap();
    let b2 = BlankNode::new("b2").unwrap();
    let b3 = BlankNode::new("b3").unwrap();

    let item_a = Object::NamedNode(NamedNode::new("http://example.org/a").unwrap());
    let item_b = Object::NamedNode(NamedNode::new("http://example.org/b").unwrap());
    let item_c = Object::NamedNode(NamedNode::new("http://example.org/c").unwrap());

    let triples = vec![
        // ex:alice ex:likes _:b1
        Triple::new(alice, likes_pred, Object::BlankNode(b1.clone())),
        // _:b1 rdf:first ex:a
        Triple::new(Subject::BlankNode(b1.clone()), rdf_first(), item_a),
        // _:b1 rdf:rest _:b2
        Triple::new(
            Subject::BlankNode(b1.clone()),
            rdf_rest(),
            Object::BlankNode(b2.clone()),
        ),
        // _:b2 rdf:first ex:b
        Triple::new(Subject::BlankNode(b2.clone()), rdf_first(), item_b),
        // _:b2 rdf:rest _:b3
        Triple::new(
            Subject::BlankNode(b2.clone()),
            rdf_rest(),
            Object::BlankNode(b3.clone()),
        ),
        // _:b3 rdf:first ex:c
        Triple::new(Subject::BlankNode(b3.clone()), rdf_first(), item_c),
        // _:b3 rdf:rest rdf:nil
        Triple::new(Subject::BlankNode(b3), rdf_rest(), rdf_nil()),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Simple collection output:\n{}", result);

    // Verify collection syntax is used
    assert!(
        result.contains("("),
        "Should use opening parenthesis for collection"
    );
    assert!(
        result.contains(")"),
        "Should use closing parenthesis for collection"
    );

    // Verify items are present
    assert!(
        result.contains("a") || result.contains("http://example.org/a"),
        "Should contain item a"
    );
    assert!(
        result.contains("b") || result.contains("http://example.org/b"),
        "Should contain item b"
    );
    assert!(
        result.contains("c") || result.contains("http://example.org/c"),
        "Should contain item c"
    );

    // Verify rdf:first and rdf:rest are NOT explicitly shown
    assert!(
        !result.contains("rdf:first") && !result.contains("#first"),
        "Should not show rdf:first when using collection syntax"
    );
    assert!(
        !result.contains("rdf:rest") && !result.contains("#rest"),
        "Should not show rdf:rest when using collection syntax"
    );
}

#[test]
fn test_empty_collection() {
    // Empty collection: ()
    // Represented as: ex:alice ex:likes rdf:nil .

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let likes_pred = Predicate::NamedNode(NamedNode::new("http://example.org/likes").unwrap());

    let triples = vec![Triple::new(alice, likes_pred, rdf_nil())];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Empty collection output:\n{}", result);

    // Empty collection should be represented as rdf:nil
    assert!(
        result.contains("nil") || result.contains("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"),
        "Should contain rdf:nil for empty collection"
    );
}

#[test]
fn test_single_item_collection() {
    // Collection with one item: (ex:a)
    // _:b1 rdf:first ex:a ; rdf:rest rdf:nil .

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let likes_pred = Predicate::NamedNode(NamedNode::new("http://example.org/likes").unwrap());

    let b1 = BlankNode::new("b1").unwrap();
    let item_a = Object::NamedNode(NamedNode::new("http://example.org/a").unwrap());

    let triples = vec![
        Triple::new(alice, likes_pred, Object::BlankNode(b1.clone())),
        Triple::new(Subject::BlankNode(b1.clone()), rdf_first(), item_a),
        Triple::new(Subject::BlankNode(b1), rdf_rest(), rdf_nil()),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Single item collection output:\n{}", result);

    // Verify collection syntax
    assert!(result.contains("("), "Should use collection syntax");
    assert!(result.contains(")"), "Should use collection syntax");
    assert!(
        result.contains("a") || result.contains("http://example.org/a"),
        "Should contain item a"
    );
}

#[test]
fn test_collection_with_literals() {
    // Collection of literals: ("Alice" "Bob" "Charlie")

    let person = Subject::NamedNode(NamedNode::new("http://example.org/person1").unwrap());
    let names_pred = Predicate::NamedNode(NamedNode::new("http://example.org/names").unwrap());

    let b1 = BlankNode::new("b1").unwrap();
    let b2 = BlankNode::new("b2").unwrap();
    let b3 = BlankNode::new("b3").unwrap();

    let triples = vec![
        Triple::new(person, names_pred, Object::BlankNode(b1.clone())),
        Triple::new(
            Subject::BlankNode(b1.clone()),
            rdf_first(),
            Object::Literal(Literal::new_simple_literal("Alice")),
        ),
        Triple::new(
            Subject::BlankNode(b1.clone()),
            rdf_rest(),
            Object::BlankNode(b2.clone()),
        ),
        Triple::new(
            Subject::BlankNode(b2.clone()),
            rdf_first(),
            Object::Literal(Literal::new_simple_literal("Bob")),
        ),
        Triple::new(
            Subject::BlankNode(b2.clone()),
            rdf_rest(),
            Object::BlankNode(b3.clone()),
        ),
        Triple::new(
            Subject::BlankNode(b3.clone()),
            rdf_first(),
            Object::Literal(Literal::new_simple_literal("Charlie")),
        ),
        Triple::new(Subject::BlankNode(b3), rdf_rest(), rdf_nil()),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Collection with literals output:\n{}", result);

    // Verify all names are present
    assert!(result.contains("Alice"), "Should contain Alice");
    assert!(result.contains("Bob"), "Should contain Bob");
    assert!(result.contains("Charlie"), "Should contain Charlie");
    assert!(result.contains("("), "Should use collection syntax");
    assert!(result.contains(")"), "Should use collection syntax");
}

#[test]
fn test_nested_collections() {
    // Nested collection: (ex:a (ex:b ex:c))
    // This creates a collection where one item is itself a collection

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let data_pred = Predicate::NamedNode(NamedNode::new("http://example.org/data").unwrap());

    let b1 = BlankNode::new("b1").unwrap(); // Outer collection head
    let b2 = BlankNode::new("b2").unwrap(); // Outer collection rest
    let b3 = BlankNode::new("b3").unwrap(); // Inner collection head
    let b4 = BlankNode::new("b4").unwrap(); // Inner collection rest

    let item_a = Object::NamedNode(NamedNode::new("http://example.org/a").unwrap());
    let item_b = Object::NamedNode(NamedNode::new("http://example.org/b").unwrap());
    let item_c = Object::NamedNode(NamedNode::new("http://example.org/c").unwrap());

    let triples = vec![
        // ex:alice ex:data _:b1
        Triple::new(alice, data_pred, Object::BlankNode(b1.clone())),
        // _:b1 rdf:first ex:a
        Triple::new(Subject::BlankNode(b1.clone()), rdf_first(), item_a),
        // _:b1 rdf:rest _:b2
        Triple::new(
            Subject::BlankNode(b1.clone()),
            rdf_rest(),
            Object::BlankNode(b2.clone()),
        ),
        // _:b2 rdf:first _:b3 (the inner collection)
        Triple::new(
            Subject::BlankNode(b2.clone()),
            rdf_first(),
            Object::BlankNode(b3.clone()),
        ),
        // _:b2 rdf:rest rdf:nil
        Triple::new(Subject::BlankNode(b2), rdf_rest(), rdf_nil()),
        // Inner collection: _:b3 rdf:first ex:b
        Triple::new(Subject::BlankNode(b3.clone()), rdf_first(), item_b),
        // _:b3 rdf:rest _:b4
        Triple::new(
            Subject::BlankNode(b3.clone()),
            rdf_rest(),
            Object::BlankNode(b4.clone()),
        ),
        // _:b4 rdf:first ex:c
        Triple::new(Subject::BlankNode(b4.clone()), rdf_first(), item_c),
        // _:b4 rdf:rest rdf:nil
        Triple::new(Subject::BlankNode(b4), rdf_rest(), rdf_nil()),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Nested collections output:\n{}", result);

    // Verify nested parentheses
    let paren_count = result.matches('(').count();
    assert!(
        paren_count >= 2,
        "Should have at least 2 opening parentheses for nested collections"
    );

    // Verify all items present
    assert!(result.contains("a") || result.contains("http://example.org/a"));
    assert!(result.contains("b") || result.contains("http://example.org/b"));
    assert!(result.contains("c") || result.contains("http://example.org/c"));
}

#[test]
fn test_collection_with_mixed_types() {
    // Collection with mixed types: (ex:iri "literal" 42)

    let subject = Subject::NamedNode(NamedNode::new("http://example.org/subject").unwrap());
    let pred = Predicate::NamedNode(NamedNode::new("http://example.org/mixed").unwrap());

    let b1 = BlankNode::new("b1").unwrap();
    let b2 = BlankNode::new("b2").unwrap();
    let b3 = BlankNode::new("b3").unwrap();

    let triples = vec![
        Triple::new(subject, pred, Object::BlankNode(b1.clone())),
        Triple::new(
            Subject::BlankNode(b1.clone()),
            rdf_first(),
            Object::NamedNode(NamedNode::new("http://example.org/iri").unwrap()),
        ),
        Triple::new(
            Subject::BlankNode(b1.clone()),
            rdf_rest(),
            Object::BlankNode(b2.clone()),
        ),
        Triple::new(
            Subject::BlankNode(b2.clone()),
            rdf_first(),
            Object::Literal(Literal::new_simple_literal("literal")),
        ),
        Triple::new(
            Subject::BlankNode(b2.clone()),
            rdf_rest(),
            Object::BlankNode(b3.clone()),
        ),
        Triple::new(
            Subject::BlankNode(b3.clone()),
            rdf_first(),
            Object::Literal(Literal::new_typed_literal(
                "42",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ),
        Triple::new(Subject::BlankNode(b3), rdf_rest(), rdf_nil()),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Mixed types collection output:\n{}", result);

    // Verify all items
    assert!(result.contains("iri") || result.contains("http://example.org/iri"));
    assert!(result.contains("literal"));
    assert!(result.contains("42"));
    assert!(result.contains("("));
    assert!(result.contains(")"));
}

#[test]
fn test_multiple_collections() {
    // Two separate collections in the same dataset

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let bob = Subject::NamedNode(NamedNode::new("http://example.org/bob").unwrap());
    let likes_pred = Predicate::NamedNode(NamedNode::new("http://example.org/likes").unwrap());

    let b1 = BlankNode::new("b1").unwrap();
    let b2 = BlankNode::new("b2").unwrap();
    let b3 = BlankNode::new("b3").unwrap();
    let b4 = BlankNode::new("b4").unwrap();

    let item_a = Object::NamedNode(NamedNode::new("http://example.org/a").unwrap());
    let item_b = Object::NamedNode(NamedNode::new("http://example.org/b").unwrap());
    let item_c = Object::NamedNode(NamedNode::new("http://example.org/c").unwrap());
    let item_d = Object::NamedNode(NamedNode::new("http://example.org/d").unwrap());

    let triples = vec![
        // Alice's collection: (a b)
        Triple::new(alice, likes_pred.clone(), Object::BlankNode(b1.clone())),
        Triple::new(Subject::BlankNode(b1.clone()), rdf_first(), item_a),
        Triple::new(
            Subject::BlankNode(b1.clone()),
            rdf_rest(),
            Object::BlankNode(b2.clone()),
        ),
        Triple::new(Subject::BlankNode(b2.clone()), rdf_first(), item_b),
        Triple::new(Subject::BlankNode(b2), rdf_rest(), rdf_nil()),
        // Bob's collection: (c d)
        Triple::new(bob, likes_pred, Object::BlankNode(b3.clone())),
        Triple::new(Subject::BlankNode(b3.clone()), rdf_first(), item_c),
        Triple::new(
            Subject::BlankNode(b3.clone()),
            rdf_rest(),
            Object::BlankNode(b4.clone()),
        ),
        Triple::new(Subject::BlankNode(b4.clone()), rdf_first(), item_d),
        Triple::new(Subject::BlankNode(b4), rdf_rest(), rdf_nil()),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Multiple collections output:\n{}", result);

    // Should have at least 2 collections
    let paren_count = result.matches('(').count();
    assert!(paren_count >= 2, "Should have at least 2 collections");

    // Verify all items
    assert!(result.contains("a") || result.contains("http://example.org/a"));
    assert!(result.contains("b") || result.contains("http://example.org/b"));
    assert!(result.contains("c") || result.contains("http://example.org/c"));
    assert!(result.contains("d") || result.contains("http://example.org/d"));
}

#[test]
fn test_round_trip_with_collections() {
    use oxirs_ttl::formats::turtle::TurtleParser;

    // Create a simple collection
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let likes_pred = Predicate::NamedNode(NamedNode::new("http://example.org/likes").unwrap());

    let b1 = BlankNode::new("b1").unwrap();
    let b2 = BlankNode::new("b2").unwrap();

    let item_a = Object::NamedNode(NamedNode::new("http://example.org/a").unwrap());
    let item_b = Object::NamedNode(NamedNode::new("http://example.org/b").unwrap());

    let triples = vec![
        Triple::new(alice, likes_pred, Object::BlankNode(b1.clone())),
        Triple::new(Subject::BlankNode(b1.clone()), rdf_first(), item_a),
        Triple::new(
            Subject::BlankNode(b1.clone()),
            rdf_rest(),
            Object::BlankNode(b2.clone()),
        ),
        Triple::new(Subject::BlankNode(b2.clone()), rdf_first(), item_b),
        Triple::new(Subject::BlankNode(b2), rdf_rest(), rdf_nil()),
    ];

    // Serialize with collection optimization
    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let turtle_str = String::from_utf8(output).unwrap();
    println!("Serialized with collections:\n{}", turtle_str);

    // Parse it back
    let parser = TurtleParser::new();
    let parsed_triples = parser.parse_document(&turtle_str).unwrap();

    // Should get the same triples back (blank node labels may differ)
    assert_eq!(
        parsed_triples.len(),
        triples.len(),
        "Round-trip should preserve all triples"
    );
}

#[test]
fn test_comparison_with_regular_serialization() {
    // Compare output size between regular and collection-optimized serialization

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let likes_pred = Predicate::NamedNode(NamedNode::new("http://example.org/likes").unwrap());

    let b1 = BlankNode::new("b1").unwrap();
    let b2 = BlankNode::new("b2").unwrap();
    let b3 = BlankNode::new("b3").unwrap();

    let item_a = Object::NamedNode(NamedNode::new("http://example.org/a").unwrap());
    let item_b = Object::NamedNode(NamedNode::new("http://example.org/b").unwrap());
    let item_c = Object::NamedNode(NamedNode::new("http://example.org/c").unwrap());

    let triples = vec![
        Triple::new(alice, likes_pred, Object::BlankNode(b1.clone())),
        Triple::new(Subject::BlankNode(b1.clone()), rdf_first(), item_a),
        Triple::new(
            Subject::BlankNode(b1.clone()),
            rdf_rest(),
            Object::BlankNode(b2.clone()),
        ),
        Triple::new(Subject::BlankNode(b2.clone()), rdf_first(), item_b),
        Triple::new(
            Subject::BlankNode(b2.clone()),
            rdf_rest(),
            Object::BlankNode(b3.clone()),
        ),
        Triple::new(Subject::BlankNode(b3.clone()), rdf_first(), item_c),
        Triple::new(Subject::BlankNode(b3), rdf_rest(), rdf_nil()),
    ];

    let serializer = TurtleSerializer::new();

    // Regular serialization (no optimizations)
    let mut regular_output = Vec::new();
    serializer.serialize(&triples, &mut regular_output).unwrap();

    // Optimized serialization with collections
    let mut optimized_output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut optimized_output)
        .unwrap();

    println!(
        "Regular output ({} bytes):\n{}",
        regular_output.len(),
        String::from_utf8_lossy(&regular_output)
    );
    println!(
        "Optimized output ({} bytes):\n{}",
        optimized_output.len(),
        String::from_utf8_lossy(&optimized_output)
    );

    // Optimized output should be significantly more compact
    assert!(
        optimized_output.len() < regular_output.len(),
        "Optimized output should be more compact"
    );
}
