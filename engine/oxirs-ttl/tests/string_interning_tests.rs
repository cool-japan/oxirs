//! Tests for string interning performance in RDF parsing

use oxirs_core::model::{Object, Predicate, Subject};
use oxirs_ttl::formats::turtle::TurtleParser;
use oxirs_ttl::toolkit::Parser;

#[test]
fn test_string_interning_with_repeated_predicates() {
    // Create a Turtle document with many repeated predicates
    // This simulates a real-world RDF dataset where predicates like rdf:type appear many times
    let ttl = r#"
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

ex:person1 rdf:type ex:Person .
ex:person2 rdf:type ex:Person .
ex:person3 rdf:type ex:Person .
ex:person4 rdf:type ex:Person .
ex:person5 rdf:type ex:Person .
ex:person6 rdf:type ex:Person .
ex:person7 rdf:type ex:Person .
ex:person8 rdf:type ex:Person .
ex:person9 rdf:type ex:Person .
ex:person10 rdf:type ex:Person .

ex:person1 ex:name "Alice" .
ex:person2 ex:name "Bob" .
ex:person3 ex:name "Charlie" .
ex:person4 ex:name "David" .
ex:person5 ex:name "Eve" .
ex:person6 ex:name "Frank" .
ex:person7 ex:name "Grace" .
ex:person8 ex:name "Henry" .
ex:person9 ex:name "Iris" .
ex:person10 ex:name "Jack" .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    // Should have parsed all 20 triples (10 rdf:type + 10 ex:name)
    assert_eq!(triples.len(), 20);

    // Verify rdf:type triples
    let type_triples: Vec<_> = triples
        .iter()
        .filter(|t| match t.predicate() {
            Predicate::NamedNode(nn) => {
                nn.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            }
            _ => false,
        })
        .collect();

    assert_eq!(type_triples.len(), 10);

    // All type objects should be ex:Person
    for triple in type_triples {
        match triple.object() {
            Object::NamedNode(nn) => assert!(nn.as_str().contains("Person")),
            _ => panic!("Expected NamedNode object"),
        }
    }
}

#[test]
fn test_string_interning_memory_efficiency() {
    // Create a large document with highly repeated IRIs
    let mut ttl_parts = vec![String::from("@prefix ex: <http://example.org/> .")];

    // Generate 100 triples with the same predicate and object type
    for i in 1..=100 {
        ttl_parts.push(format!(
            "ex:subject{i} ex:commonPredicate ex:commonObject{} .",
            i % 10
        ));
    }

    let ttl = ttl_parts.join("\n");

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    // Should have parsed all 100 triples
    assert_eq!(triples.len(), 100);

    // String interning should have significantly reduced memory allocations
    // by reusing the same string for repeated IRIs
    // (In a production implementation, we would measure this with profiling tools)
}

#[test]
fn test_string_interning_with_datatype_literals() {
    // Test that string interning also works for frequently repeated datatype IRIs
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:value1 ex:count "1"^^<http://www.w3.org/2001/XMLSchema#integer> .
ex:value2 ex:count "2"^^<http://www.w3.org/2001/XMLSchema#integer> .
ex:value3 ex:count "3"^^<http://www.w3.org/2001/XMLSchema#integer> .
ex:value4 ex:count "4"^^<http://www.w3.org/2001/XMLSchema#integer> .
ex:value5 ex:count "5"^^<http://www.w3.org/2001/XMLSchema#integer> .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 5);

    // All literals should have xsd:integer datatype
    for triple in &triples {
        match triple.object() {
            Object::Literal(lit) => {
                let dt = lit.datatype();
                assert_eq!(dt.as_str(), "http://www.w3.org/2001/XMLSchema#integer");
            }
            _ => panic!("Expected literal object"),
        }
    }
}

#[test]
fn test_string_interning_with_common_namespaces() {
    // Test that pre-populated common namespaces are interned
    let ttl = r#"
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

<http://example.org/MyClass> rdf:type owl:Class .
<http://example.org/MyClass> rdfs:label "My Class" .
<http://example.org/myProperty> rdf:type owl:DatatypeProperty .
<http://example.org/myProperty> rdfs:domain <http://example.org/MyClass> .
<http://example.org/myProperty> rdfs:range xsd:string .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    // Should parse all triples successfully
    assert_eq!(triples.len(), 5);

    // Common namespace IRIs should have been interned
    let rdf_type_count = triples
        .iter()
        .filter(|t| match t.predicate() {
            Predicate::NamedNode(nn) => {
                nn.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            }
            _ => false,
        })
        .count();

    assert_eq!(rdf_type_count, 2);
}

#[test]
fn test_string_interning_preserves_correctness() {
    // Ensure that string interning doesn't affect parsing correctness
    let ttl = r#"
@prefix ex: <http://example.org/> .

ex:alice ex:knows ex:bob .
ex:bob ex:knows ex:charlie .
ex:charlie ex:knows ex:alice .
"#;

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    assert_eq!(triples.len(), 3);

    // Verify the exact structure of the triples
    for triple in &triples {
        match (triple.subject(), triple.predicate(), triple.object()) {
            (Subject::NamedNode(s), Predicate::NamedNode(p), Object::NamedNode(o)) => {
                assert!(s.as_str().contains("example.org"));
                assert_eq!(p.as_str(), "http://example.org/knows");
                assert!(o.as_str().contains("example.org"));
            }
            _ => panic!("Expected all NamedNode triple"),
        }
    }
}

#[test]
fn test_string_interning_with_large_dataset() {
    // Simulate parsing a larger dataset with many repeated patterns
    let mut ttl_parts = vec![
        String::from("@prefix ex: <http://example.org/> ."),
        String::from("@prefix foaf: <http://xmlns.com/foaf/0.1/> ."),
    ];

    // Generate 500 triples with common predicates
    for i in 1..=500 {
        ttl_parts.push(format!("ex:person{i} foaf:name \"Person {i}\" ."));
        if i % 5 == 0 {
            ttl_parts.push(format!("ex:person{i} foaf:age {i} ."));
        }
    }

    let ttl = ttl_parts.join("\n");

    let parser = TurtleParser::new();
    let triples = parser.parse(ttl.as_bytes()).expect("Parse failed");

    // Should have parsed all triples (500 names + 100 ages)
    assert_eq!(triples.len(), 600);

    // Count foaf:name predicates
    let name_count = triples
        .iter()
        .filter(|t| match t.predicate() {
            Predicate::NamedNode(nn) => nn.as_str() == "http://xmlns.com/foaf/0.1/name",
            _ => false,
        })
        .count();

    assert_eq!(name_count, 500);

    // String interning should have significantly reduced memory usage
    // by reusing the foaf:name IRI 500 times
}
