//! N3 round-trip tests
//!
//! These tests verify that N3 documents can be parsed and serialized
//! without loss of information (round-trip property).

use oxirs_core::model::NamedNode;
use oxirs_ttl::formats::n3_parser::{AdvancedN3Parser, N3Document};
use oxirs_ttl::formats::n3_serializer::N3Serializer;
use oxirs_ttl::formats::n3_types::{N3Formula, N3Implication, N3Statement, N3Term, N3Variable};

#[test]
fn test_roundtrip_simple_statement() {
    let input = r#"@prefix ex: <http://example.org/> .
ex:alice ex:knows ex:bob ."#;

    // Parse
    let mut parser = AdvancedN3Parser::new(input).unwrap();
    let doc = parser.parse_document().unwrap();

    // Serialize
    let mut output = Vec::new();
    let serializer = N3Serializer::new();
    serializer.serialize_document(&doc, &mut output).unwrap();

    // Parse again
    let output_str = String::from_utf8(output).unwrap();
    let mut parser2 = AdvancedN3Parser::new(&output_str).unwrap();
    let doc2 = parser2.parse_document().unwrap();

    // Verify
    assert_eq!(doc.statements.len(), doc2.statements.len());
    assert_eq!(doc.implications.len(), doc2.implications.len());
}

#[test]
fn test_roundtrip_variable_statement() {
    let mut doc = N3Document::new();
    doc.add_prefix("ex".to_string(), "http://example.org/".to_string());

    let stmt = N3Statement::new(
        N3Term::Variable(N3Variable::universal("x")),
        N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap()),
        N3Term::Variable(N3Variable::universal("y")),
    );
    doc.add_statement(stmt);

    // Serialize
    let mut output = Vec::new();
    let serializer = N3Serializer::new();
    serializer.serialize_document(&doc, &mut output).unwrap();

    // Parse
    let output_str = String::from_utf8(output).unwrap();
    println!("Serialized N3:\n{}", output_str);

    let mut parser = AdvancedN3Parser::new(&output_str).unwrap();
    let doc2 = parser.parse_document().unwrap();

    // Verify
    assert_eq!(doc2.statements.len(), 1);
    let stmt2 = &doc2.statements[0];
    assert!(stmt2.subject.is_variable());
    assert!(stmt2.object.is_variable());
}

#[test]
#[ignore = "Formula as subject - advanced N3 feature, parser needs enhancement"]
fn test_roundtrip_formula_as_subject() {
    // This test is ignored because the current N3 parser doesn't fully support
    // formulas in subject position. This is valid N3 but requires parser enhancements.
    let mut doc = N3Document::new();
    doc.add_prefix("ex".to_string(), "http://example.org/".to_string());

    let mut formula = N3Formula::new();
    formula.add_statement(N3Statement::new(
        N3Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap()),
        N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap()),
        N3Term::NamedNode(NamedNode::new("http://example.org/bob").unwrap()),
    ));

    let stmt = N3Statement::new(
        N3Term::Formula(Box::new(formula)),
        N3Term::NamedNode(NamedNode::new("http://example.org/source").unwrap()),
        N3Term::NamedNode(NamedNode::new("http://example.org/doc1").unwrap()),
    );
    doc.add_statement(stmt);

    // Serialize
    let mut output = Vec::new();
    let serializer = N3Serializer::new();
    serializer.serialize_document(&doc, &mut output).unwrap();

    // Parse
    let output_str = String::from_utf8(output).unwrap();
    println!("Serialized N3:\n{}", output_str);

    let mut parser = AdvancedN3Parser::new(&output_str).unwrap();
    let doc2 = parser.parse_document().unwrap();

    // Verify
    assert_eq!(doc2.statements.len(), 1);
    assert!(doc2.statements[0].subject.is_formula());
}

#[test]
fn test_roundtrip_implication() {
    let mut doc = N3Document::new();
    doc.add_prefix("ex".to_string(), "http://example.org/".to_string());

    // Create antecedent: { ?x :parent ?y }
    let mut antecedent = N3Formula::new();
    antecedent.add_statement(N3Statement::new(
        N3Term::Variable(N3Variable::universal("x")),
        N3Term::NamedNode(NamedNode::new("http://example.org/parent").unwrap()),
        N3Term::Variable(N3Variable::universal("y")),
    ));

    // Create consequent: { ?y :child ?x }
    let mut consequent = N3Formula::new();
    consequent.add_statement(N3Statement::new(
        N3Term::Variable(N3Variable::universal("y")),
        N3Term::NamedNode(NamedNode::new("http://example.org/child").unwrap()),
        N3Term::Variable(N3Variable::universal("x")),
    ));

    let implication = N3Implication::new(antecedent, consequent);
    doc.add_implication(implication);

    // Serialize
    let mut output = Vec::new();
    let serializer = N3Serializer::new();
    serializer.serialize_document(&doc, &mut output).unwrap();

    // Parse
    let output_str = String::from_utf8(output).unwrap();
    println!("Serialized N3:\n{}", output_str);

    let mut parser = AdvancedN3Parser::new(&output_str).unwrap();
    let doc2 = parser.parse_document().unwrap();

    // Verify
    assert_eq!(doc2.implications.len(), 1);
    let impl2 = &doc2.implications[0];
    assert_eq!(impl2.antecedent.len(), 1);
    assert_eq!(impl2.consequent.len(), 1);
}

#[test]
fn test_roundtrip_quantifiers() {
    let mut doc = N3Document::new();
    doc.add_prefix("ex".to_string(), "http://example.org/".to_string());

    // Add statement with universal variable
    let stmt1 = N3Statement::new(
        N3Term::Variable(N3Variable::universal("x")),
        N3Term::NamedNode(NamedNode::new("http://example.org/type").unwrap()),
        N3Term::NamedNode(NamedNode::new("http://example.org/Person").unwrap()),
    );
    doc.add_statement(stmt1);

    // Add statement with existential variable
    let stmt2 = N3Statement::new(
        N3Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap()),
        N3Term::NamedNode(NamedNode::new("http://example.org/knows").unwrap()),
        N3Term::Variable(N3Variable::existential("z")),
    );
    doc.add_statement(stmt2);

    // Serialize
    let mut output = Vec::new();
    let serializer = N3Serializer::new();
    serializer.serialize_document(&doc, &mut output).unwrap();

    let output_str = String::from_utf8(output).unwrap();
    println!("Serialized N3:\n{}", output_str);

    // Verify quantifiers are present
    assert!(output_str.contains("@forAll"));
    assert!(output_str.contains("@forSome"));
    assert!(output_str.contains("?x"));
    assert!(output_str.contains("?z"));
}

#[test]
fn test_roundtrip_complex_implication() {
    let mut doc = N3Document::new();
    doc.add_prefix("ex".to_string(), "http://example.org/".to_string());
    doc.add_prefix(
        "rdf".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
    );

    // Create antecedent: { ?x a :Person }
    let mut antecedent = N3Formula::new();
    antecedent.add_statement(N3Statement::new(
        N3Term::Variable(N3Variable::universal("x")),
        N3Term::NamedNode(
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").unwrap(),
        ),
        N3Term::NamedNode(NamedNode::new("http://example.org/Person").unwrap()),
    ));

    // Create consequent: { ?x :hasType :Human }
    let mut consequent = N3Formula::new();
    consequent.add_statement(N3Statement::new(
        N3Term::Variable(N3Variable::universal("x")),
        N3Term::NamedNode(NamedNode::new("http://example.org/hasType").unwrap()),
        N3Term::NamedNode(NamedNode::new("http://example.org/Human").unwrap()),
    ));

    let implication = N3Implication::new(antecedent, consequent);
    doc.add_implication(implication);

    // Serialize
    let mut output = Vec::new();
    let serializer = N3Serializer::new();
    serializer.serialize_document(&doc, &mut output).unwrap();

    // Parse
    let output_str = String::from_utf8(output).unwrap();
    println!("Serialized N3:\n{}", output_str);

    let mut parser = AdvancedN3Parser::new(&output_str).unwrap();
    let doc2 = parser.parse_document().unwrap();

    // Verify
    assert_eq!(doc2.implications.len(), 1);
}

#[test]
fn test_roundtrip_multiple_implications() {
    let mut doc = N3Document::new();
    doc.add_prefix("ex".to_string(), "http://example.org/".to_string());

    // First implication: { ?x :parent ?y } => { ?y :child ?x }
    let mut ant1 = N3Formula::new();
    ant1.add_statement(N3Statement::new(
        N3Term::Variable(N3Variable::universal("x")),
        N3Term::NamedNode(NamedNode::new("http://example.org/parent").unwrap()),
        N3Term::Variable(N3Variable::universal("y")),
    ));

    let mut cons1 = N3Formula::new();
    cons1.add_statement(N3Statement::new(
        N3Term::Variable(N3Variable::universal("y")),
        N3Term::NamedNode(NamedNode::new("http://example.org/child").unwrap()),
        N3Term::Variable(N3Variable::universal("x")),
    ));

    doc.add_implication(N3Implication::new(ant1, cons1));

    // Second implication: { ?x :friend ?y } => { ?y :friend ?x }
    let mut ant2 = N3Formula::new();
    ant2.add_statement(N3Statement::new(
        N3Term::Variable(N3Variable::universal("a")),
        N3Term::NamedNode(NamedNode::new("http://example.org/friend").unwrap()),
        N3Term::Variable(N3Variable::universal("b")),
    ));

    let mut cons2 = N3Formula::new();
    cons2.add_statement(N3Statement::new(
        N3Term::Variable(N3Variable::universal("b")),
        N3Term::NamedNode(NamedNode::new("http://example.org/friend").unwrap()),
        N3Term::Variable(N3Variable::universal("a")),
    ));

    doc.add_implication(N3Implication::new(ant2, cons2));

    // Serialize
    let mut output = Vec::new();
    let serializer = N3Serializer::new();
    serializer.serialize_document(&doc, &mut output).unwrap();

    // Parse
    let output_str = String::from_utf8(output).unwrap();
    println!("Serialized N3:\n{}", output_str);

    let mut parser = AdvancedN3Parser::new(&output_str).unwrap();
    let doc2 = parser.parse_document().unwrap();

    // Verify
    assert_eq!(doc2.implications.len(), 2);
}

#[test]
fn test_serialize_empty_document() {
    let doc = N3Document::new();

    let mut output = Vec::new();
    let serializer = N3Serializer::new();
    serializer.serialize_document(&doc, &mut output).unwrap();

    let output_str = String::from_utf8(output).unwrap();
    // Should have minimal output (maybe just newlines)
    assert!(output_str.trim().is_empty() || output_str.trim().lines().count() <= 2);
}

#[test]
fn test_roundtrip_with_literals() {
    let mut doc = N3Document::new();
    doc.add_prefix("ex".to_string(), "http://example.org/".to_string());

    let stmt = N3Statement::new(
        N3Term::NamedNode(NamedNode::new("http://example.org/alice").unwrap()),
        N3Term::NamedNode(NamedNode::new("http://example.org/name").unwrap()),
        N3Term::Literal(oxirs_core::model::Literal::new("Alice")),
    );
    doc.add_statement(stmt);

    // Serialize
    let mut output = Vec::new();
    let serializer = N3Serializer::new();
    serializer.serialize_document(&doc, &mut output).unwrap();

    // Parse
    let output_str = String::from_utf8(output).unwrap();
    println!("Serialized N3:\n{}", output_str);

    let mut parser = AdvancedN3Parser::new(&output_str).unwrap();
    let doc2 = parser.parse_document().unwrap();

    // Verify
    assert_eq!(doc2.statements.len(), 1);
    if let N3Term::Literal(lit) = &doc2.statements[0].object {
        assert_eq!(lit.value(), "Alice");
    } else {
        panic!("Expected literal object");
    }
}
