//! Tests for Blank Node Optimization in Turtle Serialization
//!
//! Tests for inline blank node property lists using `[]` syntax.

use oxirs_core::model::{BlankNode, Literal, NamedNode, Object, Predicate, Subject, Triple};
use oxirs_ttl::formats::turtle::TurtleSerializer;

#[test]
fn test_simple_blank_node_property_list() {
    // Create triples: ex:alice ex:address _:b1 . _:b1 ex:city "Wonderland" . _:b1 ex:zip "12345" .
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let address_pred = Predicate::NamedNode(NamedNode::new("http://example.org/address").unwrap());
    let city_pred = Predicate::NamedNode(NamedNode::new("http://example.org/city").unwrap());
    let zip_pred = Predicate::NamedNode(NamedNode::new("http://example.org/zip").unwrap());

    let bn = BlankNode::new("b1").unwrap();

    let triples = vec![
        Triple::new(alice, address_pred, Object::BlankNode(bn.clone())),
        Triple::new(
            Subject::BlankNode(bn.clone()),
            city_pred,
            Object::Literal(Literal::new_simple_literal("Wonderland")),
        ),
        Triple::new(
            Subject::BlankNode(bn),
            zip_pred,
            Object::Literal(Literal::new_simple_literal("12345")),
        ),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Blank node property list output:\n{}", result);

    // Verify property list syntax is used
    assert!(
        result.contains("["),
        "Should use opening bracket for blank node property list"
    );
    assert!(
        result.contains("]"),
        "Should use closing bracket for blank node property list"
    );

    // Verify the blank node label is NOT used (it's inlined)
    assert!(
        !result.contains("_:b1"),
        "Should not contain blank node label when inlined"
    );

    // Verify properties are inside the brackets
    assert!(
        result.contains("city") || result.contains("http://example.org/city"),
        "Should contain city property"
    );
    assert!(
        result.contains("Wonderland"),
        "Should contain Wonderland value"
    );
}

#[test]
fn test_nested_blank_nodes() {
    // Create nested blank nodes: ex:alice ex:address _:b1 . _:b1 ex:location _:b2 . _:b2 ex:city "Wonderland" .
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let address_pred = Predicate::NamedNode(NamedNode::new("http://example.org/address").unwrap());
    let location_pred =
        Predicate::NamedNode(NamedNode::new("http://example.org/location").unwrap());
    let city_pred = Predicate::NamedNode(NamedNode::new("http://example.org/city").unwrap());

    let bn1 = BlankNode::new("b1").unwrap();
    let bn2 = BlankNode::new("b2").unwrap();

    let triples = vec![
        Triple::new(alice, address_pred, Object::BlankNode(bn1.clone())),
        Triple::new(
            Subject::BlankNode(bn1),
            location_pred,
            Object::BlankNode(bn2.clone()),
        ),
        Triple::new(
            Subject::BlankNode(bn2),
            city_pred,
            Object::Literal(Literal::new_simple_literal("Wonderland")),
        ),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Nested blank nodes output:\n{}", result);

    // Verify nested property lists
    let bracket_count = result.matches('[').count();
    assert!(
        bracket_count >= 2,
        "Should have at least 2 opening brackets for nested blank nodes"
    );
}

#[test]
fn test_blank_node_used_multiple_times() {
    // Blank node used as object multiple times should NOT be inlined
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let bob = Subject::NamedNode(NamedNode::new("http://example.org/bob").unwrap());
    let address_pred = Predicate::NamedNode(NamedNode::new("http://example.org/address").unwrap());
    let city_pred = Predicate::NamedNode(NamedNode::new("http://example.org/city").unwrap());

    let bn = BlankNode::new("b1").unwrap();

    let triples = vec![
        Triple::new(alice, address_pred.clone(), Object::BlankNode(bn.clone())),
        Triple::new(
            bob,
            address_pred,
            Object::BlankNode(bn.clone()), // Same blank node used twice
        ),
        Triple::new(
            Subject::BlankNode(bn),
            city_pred,
            Object::Literal(Literal::new_simple_literal("Wonderland")),
        ),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Blank node used multiple times output:\n{}", result);

    // Verify blank node label IS used (not inlined because it appears twice)
    assert!(
        result.contains("_:b1"),
        "Should contain blank node label when used multiple times"
    );
}

#[test]
fn test_blank_node_with_multiple_properties() {
    // Test blank node with 3+ properties
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let address_pred = Predicate::NamedNode(NamedNode::new("http://example.org/address").unwrap());
    let city_pred = Predicate::NamedNode(NamedNode::new("http://example.org/city").unwrap());
    let street_pred = Predicate::NamedNode(NamedNode::new("http://example.org/street").unwrap());
    let zip_pred = Predicate::NamedNode(NamedNode::new("http://example.org/zip").unwrap());

    let bn = BlankNode::new("b1").unwrap();

    let triples = vec![
        Triple::new(alice, address_pred, Object::BlankNode(bn.clone())),
        Triple::new(
            Subject::BlankNode(bn.clone()),
            city_pred,
            Object::Literal(Literal::new_simple_literal("Wonderland")),
        ),
        Triple::new(
            Subject::BlankNode(bn.clone()),
            street_pred,
            Object::Literal(Literal::new_simple_literal("Main St")),
        ),
        Triple::new(
            Subject::BlankNode(bn),
            zip_pred,
            Object::Literal(Literal::new_simple_literal("12345")),
        ),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Blank node with multiple properties output:\n{}", result);

    // Verify all properties are serialized
    assert!(result.contains("Wonderland"));
    assert!(result.contains("Main St"));
    assert!(result.contains("12345"));

    // Verify semicolons are used between properties
    assert!(
        result.contains(";"),
        "Should use semicolons between properties"
    );
}

#[test]
fn test_round_trip_with_blank_nodes() {
    use oxirs_ttl::formats::turtle::TurtleParser;

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let address_pred = Predicate::NamedNode(NamedNode::new("http://example.org/address").unwrap());
    let city_pred = Predicate::NamedNode(NamedNode::new("http://example.org/city").unwrap());

    let bn = BlankNode::new("b1").unwrap();

    let triples = vec![
        Triple::new(alice, address_pred, Object::BlankNode(bn.clone())),
        Triple::new(
            Subject::BlankNode(bn),
            city_pred,
            Object::Literal(Literal::new_simple_literal("Wonderland")),
        ),
    ];

    // Serialize with blank node optimization
    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let turtle_str = String::from_utf8(output).unwrap();
    println!("Serialized:\n{}", turtle_str);

    // Parse it back
    let parser = TurtleParser::new();
    let parsed_triples = parser.parse_document(&turtle_str).unwrap();

    // Verify we get the same number of triples back
    assert_eq!(
        parsed_triples.len(),
        triples.len(),
        "Round-trip should preserve all triples"
    );
}

#[test]
fn test_mixed_blank_and_named_nodes() {
    // Test with both blank nodes and named nodes as objects
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let bob_named = NamedNode::new("http://example.org/bob").unwrap();
    let address_pred = Predicate::NamedNode(NamedNode::new("http://example.org/address").unwrap());
    let knows_pred = Predicate::NamedNode(NamedNode::new("http://example.org/knows").unwrap());
    let city_pred = Predicate::NamedNode(NamedNode::new("http://example.org/city").unwrap());

    let bn = BlankNode::new("b1").unwrap();

    let triples = vec![
        Triple::new(alice.clone(), address_pred, Object::BlankNode(bn.clone())),
        Triple::new(
            Subject::BlankNode(bn),
            city_pred,
            Object::Literal(Literal::new_simple_literal("Wonderland")),
        ),
        Triple::new(alice, knows_pred, Object::NamedNode(bob_named)),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Mixed blank and named nodes output:\n{}", result);

    // Verify both blank node property list and named node are present
    assert!(
        result.contains("["),
        "Should contain blank node property list"
    );
    assert!(
        result.contains("bob") || result.contains("http://example.org/bob"),
        "Should contain named node reference"
    );
}

#[test]
fn test_comparison_with_regular_serialization() {
    // Compare output size between regular and optimized serialization
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let address_pred = Predicate::NamedNode(NamedNode::new("http://example.org/address").unwrap());
    let city_pred = Predicate::NamedNode(NamedNode::new("http://example.org/city").unwrap());
    let zip_pred = Predicate::NamedNode(NamedNode::new("http://example.org/zip").unwrap());

    let bn = BlankNode::new("b1").unwrap();

    let triples = vec![
        Triple::new(alice, address_pred, Object::BlankNode(bn.clone())),
        Triple::new(
            Subject::BlankNode(bn.clone()),
            city_pred,
            Object::Literal(Literal::new_simple_literal("Wonderland")),
        ),
        Triple::new(
            Subject::BlankNode(bn),
            zip_pred,
            Object::Literal(Literal::new_simple_literal("12345")),
        ),
    ];

    let serializer = TurtleSerializer::new();

    // Regular serialization
    let mut regular_output = Vec::new();
    serializer
        .serialize_optimized(&triples, &mut regular_output)
        .unwrap();

    // Optimized serialization with blank nodes
    let mut optimized_output = Vec::new();
    serializer
        .serialize_with_blank_node_optimization(&triples, &mut optimized_output)
        .unwrap();

    println!(
        "Regular output:\n{}",
        String::from_utf8_lossy(&regular_output)
    );
    println!(
        "Optimized output:\n{}",
        String::from_utf8_lossy(&optimized_output)
    );

    // Optimized output should be more compact (fewer characters)
    assert!(
        optimized_output.len() < regular_output.len(),
        "Optimized output should be more compact"
    );
}
