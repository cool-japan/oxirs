//! Comprehensive N-Triples parser and serializer tests

use oxirs_core::model::{Literal, NamedNode, Object, Subject, Triple};
use oxirs_ttl::ntriples::{NTriplesParser, NTriplesSerializer};
use oxirs_ttl::Parser;
use std::io::Cursor;

#[test]
fn test_simple_triple() {
    let nt =
        "<http://example.org/subject> <http://example.org/predicate> <http://example.org/object> .";
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    let triple = &triples[0];
    assert!(matches!(triple.subject(), Subject::NamedNode(_)));
    assert!(matches!(triple.object(), Object::NamedNode(_)));
}

#[test]
fn test_string_literal() {
    let nt = r#"<http://example.org/subject> <http://example.org/predicate> "literal value" ."#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    if let Object::Literal(lit) = triples[0].object() {
        assert_eq!(lit.value(), "literal value");
    } else {
        panic!("Expected literal");
    }
}

#[test]
fn test_literal_with_language_tag() {
    let nt = r#"<http://example.org/subject> <http://example.org/predicate> "Hello"@en ."#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    if let Object::Literal(lit) = triples[0].object() {
        assert_eq!(lit.value(), "Hello");
        assert_eq!(lit.language(), Some("en"));
    } else {
        panic!("Expected literal with language tag");
    }
}

#[test]
fn test_typed_literal() {
    let nt = r#"<http://example.org/subject> <http://example.org/predicate> "42"^^<http://www.w3.org/2001/XMLSchema#integer> ."#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    if let Object::Literal(lit) = triples[0].object() {
        assert_eq!(lit.value(), "42");
        assert!(lit.datatype().as_str().ends_with("integer"));
    } else {
        panic!("Expected typed literal");
    }
}

#[test]
fn test_blank_node_subject() {
    let nt = r#"_:blank1 <http://example.org/predicate> "object" ."#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    assert!(matches!(triples[0].subject(), Subject::BlankNode(_)));
}

#[test]
fn test_blank_node_object() {
    let nt = r#"<http://example.org/subject> <http://example.org/predicate> _:blank1 ."#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    assert!(matches!(triples[0].object(), Object::BlankNode(_)));
}

#[test]
fn test_multiple_triples() {
    let nt = r#"
<http://example.org/s1> <http://example.org/p1> "o1" .
<http://example.org/s2> <http://example.org/p2> "o2" .
<http://example.org/s3> <http://example.org/p3> "o3" .
"#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 3);
}

#[test]
fn test_empty_document() {
    let nt = "";
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    assert_eq!(triples.unwrap().len(), 0);
}

#[test]
fn test_comments() {
    let nt = r#"
# This is a comment
<http://example.org/s1> <http://example.org/p1> "o1" .
# Another comment
<http://example.org/s2> <http://example.org/p2> "o2" . # Inline comment
"#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 2);
}

#[test]
fn test_whitespace_handling() {
    let nt = "\n\n  <http://example.org/s>   <http://example.org/p>   \"o\"  .  \n\n";
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    assert_eq!(triples.unwrap().len(), 1);
}

#[test]
fn test_escape_sequences() {
    let nt =
        r#"<http://example.org/s> <http://example.org/p> "line1\nline2\ttab\r\n\"quote\"\\" ."#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    if let Object::Literal(lit) = triples[0].object() {
        let value = lit.value();
        assert!(value.contains("\n"));
        assert!(value.contains("\t"));
        assert!(value.contains("\""));
        assert!(value.contains("\\"));
    }
}

#[test]
fn test_unicode_escapes() {
    let nt = r#"<http://example.org/s> <http://example.org/p> "Hello \u0041\U00000042" ."#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    if let Object::Literal(lit) = triples[0].object() {
        assert!(lit.value().contains('A'));
        assert!(lit.value().contains('B'));
    }
}

#[test]
fn test_unicode_characters() {
    let nt = "<http://example.org/s> <http://example.org/p> \"日本語🦀Rust\" .";
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    if let Object::Literal(lit) = triples[0].object() {
        assert_eq!(lit.value(), "日本語🦀Rust");
    }
}

#[test]
fn test_long_uris() {
    let long_uri = format!("<http://example.org/{}>", "a".repeat(1000));
    let nt = format!("{} {} \"object\" .", long_uri, long_uri);
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    assert_eq!(triples.unwrap().len(), 1);
}

#[test]
fn test_long_literal() {
    let long_value = "a".repeat(10000);
    let nt = format!(
        r#"<http://example.org/s> <http://example.org/p> "{}" ."#,
        long_value
    );
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    let triples = triples.unwrap();
    assert_eq!(triples.len(), 1);

    if let Object::Literal(lit) = triples[0].object() {
        assert_eq!(lit.value().len(), 10000);
    }
}

#[test]
fn test_error_missing_period() {
    let nt = "<http://example.org/s> <http://example.org/p> \"object\"";
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    // Should fail due to missing period
    assert!(triples.is_err());
}

#[test]
fn test_error_invalid_iri() {
    let nt = "<invalid iri with spaces> <http://example.org/p> \"object\" .";
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    // Should fail due to invalid IRI
    assert!(triples.is_err());
}

#[test]
fn test_error_unclosed_literal() {
    let nt = r#"<http://example.org/s> <http://example.org/p> "unclosed literal ."#;
    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    // Should fail due to unclosed literal
    assert!(triples.is_err());
}

#[test]
fn test_serialization_roundtrip() {
    use oxirs_ttl::Serializer;

    let original_nt = r#"<http://example.org/alice> <http://example.org/name> "Alice" .
<http://example.org/bob> <http://example.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
_:blank1 <http://example.org/type> "Person" ."#;

    // Parse
    let parser = NTriplesParser::new();
    let triples: Vec<_> = parser
        .for_reader(Cursor::new(original_nt))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Serialize
    let mut buffer = Vec::new();
    let serializer = NTriplesSerializer::new();
    serializer.serialize(&triples, &mut buffer).unwrap();

    // Parse serialized output
    let serialized = String::from_utf8(buffer).unwrap();
    let parser2 = NTriplesParser::new();
    let triples2: Vec<_> = parser2
        .for_reader(Cursor::new(serialized))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Compare
    assert_eq!(triples.len(), triples2.len());
}

#[test]
fn test_serializer_escaping() {
    use oxirs_ttl::Serializer;

    let triple = Triple::new(
        NamedNode::new_unchecked("http://example.org/s"),
        NamedNode::new_unchecked("http://example.org/p"),
        Literal::new_simple_literal("line1\nline2\ttab\"quote\\"),
    );

    let mut buffer = Vec::new();
    let serializer = NTriplesSerializer::new();
    serializer.serialize(&[triple], &mut buffer).unwrap();

    let serialized = String::from_utf8(buffer).unwrap();

    // Check that special characters are escaped
    assert!(serialized.contains("\\n"));
    assert!(serialized.contains("\\t"));
    assert!(serialized.contains("\\\""));
    assert!(serialized.contains("\\\\"));
}

#[test]
fn test_large_document() {
    let mut nt = String::new();
    for i in 0..10_000 {
        nt.push_str(&format!(
            "<http://example.org/s{}> <http://example.org/p> \"object{}\" .\n",
            i, i
        ));
    }

    let parser = NTriplesParser::new();
    let triples: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nt)).collect();

    assert!(triples.is_ok());
    assert_eq!(triples.unwrap().len(), 10_000);
}

#[test]
fn test_streaming_parsing() {
    use oxirs_ttl::streaming::{StreamingConfig, StreamingParser};

    let mut nt = String::new();
    for i in 0..1000 {
        nt.push_str(&format!(
            "<http://example.org/s{}> <http://example.org/p> \"object{}\" .\n",
            i, i
        ));
    }

    let config = StreamingConfig::default().with_batch_size(100);
    let mut parser = StreamingParser::with_config(Cursor::new(nt), config);

    let mut total = 0;
    while let Some(batch) = parser.next_batch().unwrap() {
        total += batch.len();
    }

    assert_eq!(total, 1000);
}
