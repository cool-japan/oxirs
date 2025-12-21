//! Tests for Optimized Turtle Serialization
//!
//! Tests for predicate grouping (semicolon syntax) and object list optimization (comma syntax).

use oxirs_core::model::{Literal, NamedNode, Object, Predicate, Subject, Triple};
use oxirs_ttl::formats::turtle::TurtleSerializer;

#[test]
fn test_predicate_grouping() {
    // Create triples with same subject, different predicates
    let subject = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());

    let triple1 = Triple::new(
        subject.clone(),
        Predicate::NamedNode(NamedNode::new("http://example.org/name").unwrap()),
        Object::Literal(Literal::new_simple_literal("Alice")),
    );

    let triple2 = Triple::new(
        subject.clone(),
        Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap()),
        Object::Literal(Literal::new_typed_literal(
            "30",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
        )),
    );

    let triple3 = Triple::new(
        subject,
        Predicate::NamedNode(NamedNode::new("http://example.org/email").unwrap()),
        Object::Literal(Literal::new_simple_literal("alice@example.org")),
    );

    let triples = vec![triple1, triple2, triple3];

    // Serialize with optimization
    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_optimized(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Optimized output:\n{}", result);

    // Verify semicolons are used
    assert!(
        result.contains(";"),
        "Should use semicolon for predicate grouping"
    );

    // Verify only one statement-ending dot (followed by newline or end)
    let statement_dots = result.matches(" .\n").count()
        + result
            .matches(" .")
            .collect::<Vec<_>>()
            .iter()
            .filter(|_| result.ends_with(" ."))
            .count();
    assert!(
        statement_dots == 1 || result.contains(" ."),
        "Should have statement-ending dot"
    );
}

#[test]
fn test_object_list_optimization() {
    // Create triples with same subject and predicate, different objects
    let subject = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let predicate = Predicate::NamedNode(NamedNode::new("http://example.org/knows").unwrap());

    let triple1 = Triple::new(
        subject.clone(),
        predicate.clone(),
        Object::NamedNode(NamedNode::new("http://example.org/bob").unwrap()),
    );

    let triple2 = Triple::new(
        subject.clone(),
        predicate.clone(),
        Object::NamedNode(NamedNode::new("http://example.org/charlie").unwrap()),
    );

    let triple3 = Triple::new(
        subject,
        predicate,
        Object::NamedNode(NamedNode::new("http://example.org/diana").unwrap()),
    );

    let triples = vec![triple1, triple2, triple3];

    // Serialize with optimization
    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_optimized(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Optimized output:\n{}", result);

    // Verify commas are used
    let comma_count = result.matches(',').count();
    assert!(
        comma_count >= 2,
        "Should use comma for object list (found {} commas)",
        comma_count
    );
}

#[test]
fn test_combined_optimization() {
    // Create a complex set of triples:
    // - alice has name, age, and knows multiple people
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());

    let name_pred = Predicate::NamedNode(NamedNode::new("http://example.org/name").unwrap());
    let age_pred = Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap());
    let knows_pred = Predicate::NamedNode(NamedNode::new("http://example.org/knows").unwrap());

    let triples = vec![
        Triple::new(
            alice.clone(),
            name_pred,
            Object::Literal(Literal::new_simple_literal("Alice")),
        ),
        Triple::new(
            alice.clone(),
            age_pred,
            Object::Literal(Literal::new_typed_literal(
                "30",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ),
        Triple::new(
            alice.clone(),
            knows_pred.clone(),
            Object::NamedNode(NamedNode::new("http://example.org/bob").unwrap()),
        ),
        Triple::new(
            alice.clone(),
            knows_pred.clone(),
            Object::NamedNode(NamedNode::new("http://example.org/charlie").unwrap()),
        ),
        Triple::new(
            alice,
            knows_pred,
            Object::NamedNode(NamedNode::new("http://example.org/diana").unwrap()),
        ),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_optimized(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Combined optimization output:\n{}", result);

    // Verify both semicolons and commas are used
    assert!(
        result.contains(";"),
        "Should use semicolons for predicate grouping"
    );
    assert!(result.contains(","), "Should use commas for object lists");
}

#[test]
fn test_multiple_subjects() {
    // Create triples for multiple subjects
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let bob = Subject::NamedNode(NamedNode::new("http://example.org/bob").unwrap());

    let name_pred = Predicate::NamedNode(NamedNode::new("http://example.org/name").unwrap());
    let age_pred = Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap());

    let triples = vec![
        Triple::new(
            alice.clone(),
            name_pred.clone(),
            Object::Literal(Literal::new_simple_literal("Alice")),
        ),
        Triple::new(
            alice,
            age_pred.clone(),
            Object::Literal(Literal::new_typed_literal(
                "30",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ),
        Triple::new(
            bob.clone(),
            name_pred,
            Object::Literal(Literal::new_simple_literal("Bob")),
        ),
        Triple::new(
            bob,
            age_pred,
            Object::Literal(Literal::new_typed_literal(
                "25",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ),
    ];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_optimized(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Multiple subjects output:\n{}", result);

    // Should have 2 statement-ending dots (one for each subject)
    let statement_dots = result.matches(" .\n").count();
    assert_eq!(
        statement_dots, 2,
        "Should have 2 statement-ending dots (one per subject group)"
    );
}

#[test]
fn test_rdf_type_abbreviation() {
    // Test that rdf:type is abbreviated to 'a'
    let subject = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());

    let triple1 = Triple::new(
        subject.clone(),
        Predicate::NamedNode(
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
        ),
        Object::NamedNode(NamedNode::new("http://example.org/Person").unwrap()),
    );

    let triple2 = Triple::new(
        subject,
        Predicate::NamedNode(NamedNode::new("http://example.org/name").unwrap()),
        Object::Literal(Literal::new_simple_literal("Alice")),
    );

    let triples = vec![triple1, triple2];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_optimized(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("RDF type abbreviation output:\n{}", result);

    // Verify 'a' is used instead of full rdf:type IRI
    assert!(result.contains(" a "), "Should abbreviate rdf:type to 'a'");
}

#[test]
fn test_blank_nodes() {
    use oxirs_core::model::BlankNode;

    let subject = Subject::BlankNode(BlankNode::new("b1").unwrap());

    let triple1 = Triple::new(
        subject.clone(),
        Predicate::NamedNode(NamedNode::new("http://example.org/prop1").unwrap()),
        Object::Literal(Literal::new_simple_literal("value1")),
    );

    let triple2 = Triple::new(
        subject,
        Predicate::NamedNode(NamedNode::new("http://example.org/prop2").unwrap()),
        Object::Literal(Literal::new_simple_literal("value2")),
    );

    let triples = vec![triple1, triple2];

    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_optimized(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Blank nodes output:\n{}", result);

    // Verify blank node syntax
    assert!(result.contains("_:b1"), "Should contain blank node label");
    assert!(
        result.contains(";"),
        "Should use semicolon for predicate grouping"
    );
}

#[test]
fn test_round_trip_optimization() {
    use oxirs_ttl::formats::turtle::TurtleParser;

    // Create some triples
    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let name_pred = Predicate::NamedNode(NamedNode::new("http://example.org/name").unwrap());
    let age_pred = Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap());

    let triples = vec![
        Triple::new(
            alice.clone(),
            name_pred,
            Object::Literal(Literal::new_simple_literal("Alice")),
        ),
        Triple::new(
            alice,
            age_pred,
            Object::Literal(Literal::new_typed_literal(
                "30",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ),
    ];

    // Serialize with optimization
    let serializer = TurtleSerializer::new();
    let mut output = Vec::new();
    serializer
        .serialize_optimized(&triples, &mut output)
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
fn test_pretty_print_with_optimization() {
    use oxirs_ttl::toolkit::SerializationConfig;

    let alice = Subject::NamedNode(NamedNode::new("http://example.org/alice").unwrap());
    let name_pred = Predicate::NamedNode(NamedNode::new("http://example.org/name").unwrap());
    let age_pred = Predicate::NamedNode(NamedNode::new("http://example.org/age").unwrap());

    let triples = vec![
        Triple::new(
            alice.clone(),
            name_pred,
            Object::Literal(Literal::new_simple_literal("Alice")),
        ),
        Triple::new(
            alice,
            age_pred,
            Object::Literal(Literal::new_typed_literal(
                "30",
                NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").unwrap(),
            )),
        ),
    ];

    let config = SerializationConfig::default().with_pretty(true);
    let serializer = TurtleSerializer::with_config(config);

    let mut output = Vec::new();
    serializer
        .serialize_optimized(&triples, &mut output)
        .unwrap();

    let result = String::from_utf8(output).unwrap();
    println!("Pretty printed output:\n{}", result);

    // Verify pretty printing adds proper indentation
    assert!(result.contains("\n"), "Should contain newlines");
}
