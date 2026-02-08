//! Tests for the TurtleParser fix

use oxirs_core::format::turtle::TurtleParser;

#[test]
fn test_semicolon_continuation() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:subject :pred1 :obj1 ; :pred2 :obj2 .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 2);
}

#[test]
fn test_comma_object_list() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:subject :pred :obj1 , :obj2 , :obj3 .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 3);
}

#[test]
fn test_string_literal_with_spaces() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s :p "hello world" .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 1);
}

#[test]
fn test_blank_node_property_list() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s :p [ :nested :value ] .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 2);
}

#[test]
fn test_collection() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s :p ( :a :b :c ) .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert!(triples.len() >= 6);
}

#[test]
fn test_typed_literal() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <urn:test:> .
:s :p "-90.0"^^xsd:double .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 1);
}

#[test]
fn test_language_tagged_string() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s :p "hello"@en .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 1);
}

#[test]
fn test_a_shorthand() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s a :Type .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 1);
}

#[test]
fn test_numeric_literals() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s :p 42 .
:s :q -90.0 .
:s :r 1.5e10 .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 3);
}

#[test]
fn test_boolean_literals() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s :p true .
:s :q false .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 2);
}

#[test]
fn test_multiline_string() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s :p """line one
line two""" .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 1);
}

#[test]
fn test_sparql_style_prefix() {
    let parser = TurtleParser::new();
    let ttl = r#"
PREFIX : <urn:test:>
:s :p :o .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 1);
}

#[test]
fn test_deeply_nested_blank_nodes() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s :p [ :a [ :b [ :c :d ] ] ] .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 4);
}

#[test]
fn test_mixed_shacl_syntax() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix : <urn:shapes:> .

:EntityShape a sh:NodeShape ;
    sh:targetClass :Entity ;
    sh:property [
        sh:path :lat ;
        sh:datatype xsd:double ;
        sh:minInclusive -90.0 ;
        sh:maxInclusive 90.0 ;
        sh:maxCount 1 ;
        sh:message "Latitude must be between -90 and 90" ;
    ] .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    // Expected triples:
    // 1. :EntityShape a sh:NodeShape
    // 2. :EntityShape sh:targetClass :Entity
    // 3. :EntityShape sh:property _:b0
    // 4. _:b0 sh:path :lat
    // 5. _:b0 sh:datatype xsd:double
    // 6. _:b0 sh:minInclusive -90.0
    // 7. _:b0 sh:maxInclusive 90.0
    // 8. _:b0 sh:maxCount 1
    // 9. _:b0 sh:message "..."
    assert!(
        triples.len() >= 9,
        "Expected at least 9 triples, got {}",
        triples.len()
    );
}

#[test]
fn test_empty_collection() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
:s :p () .
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 1);
}

#[test]
fn test_comments() {
    let parser = TurtleParser::new();
    let ttl = r#"
@prefix : <urn:test:> .
# This is a comment
:s :p :o . # inline comment
"#;
    let triples = parser.parse_str(ttl).unwrap();
    assert_eq!(triples.len(), 1);
}
