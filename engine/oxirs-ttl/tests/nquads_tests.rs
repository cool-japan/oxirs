//! Comprehensive N-Quads parser and serializer tests

use oxirs_core::model::{GraphName, Literal, NamedNode, Object, Quad, Subject};
use oxirs_ttl::nquads::{NQuadsParser, NQuadsSerializer};
use oxirs_ttl::Parser;
use std::io::Cursor;

#[test]
fn test_simple_quad() {
    let nq = "<http://example.org/subject> <http://example.org/predicate> <http://example.org/object> <http://example.org/graph> .";
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    let quad = &quads[0];
    assert!(matches!(quad.subject(), Subject::NamedNode(_)));
    assert!(matches!(quad.object(), Object::NamedNode(_)));
    assert!(matches!(quad.graph_name(), GraphName::NamedNode(_)));
}

#[test]
fn test_default_graph() {
    let nq =
        "<http://example.org/subject> <http://example.org/predicate> <http://example.org/object> .";
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    assert!(matches!(quads[0].graph_name(), GraphName::DefaultGraph));
}

#[test]
fn test_quad_with_literal() {
    let nq = r#"<http://example.org/subject> <http://example.org/predicate> "literal value" <http://example.org/graph> ."#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        assert_eq!(lit.value(), "literal value");
    } else {
        panic!("Expected literal");
    }
}

#[test]
fn test_quad_with_language_tag() {
    let nq = r#"<http://example.org/subject> <http://example.org/predicate> "Hello"@en <http://example.org/graph> ."#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        assert_eq!(lit.value(), "Hello");
        assert_eq!(lit.language(), Some("en"));
    } else {
        panic!("Expected literal with language tag");
    }
}

#[test]
fn test_quad_with_typed_literal() {
    let nq = r#"<http://example.org/subject> <http://example.org/predicate> "42"^^<http://www.w3.org/2001/XMLSchema#integer> <http://example.org/graph> ."#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        assert_eq!(lit.value(), "42");
        assert!(lit.datatype().as_str().ends_with("integer"));
    } else {
        panic!("Expected typed literal");
    }
}

#[test]
fn test_blank_node_subject() {
    let nq = r#"_:blank1 <http://example.org/predicate> "object" <http://example.org/graph> ."#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    assert!(matches!(quads[0].subject(), Subject::BlankNode(_)));
}

#[test]
fn test_blank_node_object() {
    let nq = r#"<http://example.org/subject> <http://example.org/predicate> _:blank1 <http://example.org/graph> ."#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    assert!(matches!(quads[0].object(), Object::BlankNode(_)));
}

#[test]
fn test_blank_node_graph() {
    let nq =
        r#"<http://example.org/subject> <http://example.org/predicate> "object" _:graphBlank ."#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    assert!(matches!(quads[0].graph_name(), GraphName::BlankNode(_)));
}

#[test]
fn test_multiple_quads_different_graphs() {
    let nq = r#"
<http://example.org/s1> <http://example.org/p1> "o1" <http://example.org/g1> .
<http://example.org/s2> <http://example.org/p2> "o2" <http://example.org/g2> .
<http://example.org/s3> <http://example.org/p3> "o3" <http://example.org/g1> .
"#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 3);

    // Check that different graph names are preserved
    if let (GraphName::NamedNode(g1), GraphName::NamedNode(g2)) =
        (quads[0].graph_name(), quads[1].graph_name())
    {
        assert_ne!(g1.as_str(), g2.as_str());
    }
}

#[test]
fn test_mixed_default_and_named_graphs() {
    let nq = r#"
<http://example.org/s1> <http://example.org/p1> "o1" .
<http://example.org/s2> <http://example.org/p2> "o2" <http://example.org/graph> .
<http://example.org/s3> <http://example.org/p3> "o3" .
"#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 3);

    // Check that default graph and named graph are distinguished
    assert!(matches!(quads[0].graph_name(), GraphName::DefaultGraph));
    assert!(matches!(quads[1].graph_name(), GraphName::NamedNode(_)));
    assert!(matches!(quads[2].graph_name(), GraphName::DefaultGraph));
}

#[test]
fn test_empty_document() {
    let nq = "";
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    assert_eq!(quads.unwrap().len(), 0);
}

#[test]
fn test_comments() {
    let nq = r#"
# This is a comment
<http://example.org/s1> <http://example.org/p1> "o1" <http://example.org/g1> .
# Another comment
<http://example.org/s2> <http://example.org/p2> "o2" . # Inline comment
"#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 2);
}

#[test]
fn test_whitespace_handling() {
    let nq = "\n\n  <http://example.org/s>   <http://example.org/p>   \"o\"  <http://example.org/g>  .  \n\n";
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    assert_eq!(quads.unwrap().len(), 1);
}

#[test]
fn test_escape_sequences() {
    let nq = r#"<http://example.org/s> <http://example.org/p> "line1\nline2\ttab\r\n\"quote\"\\" <http://example.org/g> ."#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        let value = lit.value();
        assert!(value.contains("\n"));
        assert!(value.contains("\t"));
        assert!(value.contains("\""));
        assert!(value.contains("\\"));
    }
}

#[test]
fn test_unicode_escapes() {
    let nq = r#"<http://example.org/s> <http://example.org/p> "Hello \u0041\U00000042" <http://example.org/g> ."#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        assert!(lit.value().contains('A'));
        assert!(lit.value().contains('B'));
    }
}

#[test]
fn test_unicode_characters() {
    let nq =
        "<http://example.org/s> <http://example.org/p> \"日本語🦀Rust\" <http://example.org/g> .";
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        assert_eq!(lit.value(), "日本語🦀Rust");
    }
}

#[test]
fn test_long_uris() {
    let long_uri = format!("<http://example.org/{}>", "a".repeat(1000));
    let nq = format!("{} {} \"object\" {} .", long_uri, long_uri, long_uri);
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    assert_eq!(quads.unwrap().len(), 1);
}

#[test]
fn test_long_literal() {
    let long_value = "a".repeat(10000);
    let nq = format!(
        r#"<http://example.org/s> <http://example.org/p> "{}" <http://example.org/g> ."#,
        long_value
    );
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        assert_eq!(lit.value().len(), 10000);
    }
}

#[test]
fn test_error_missing_period() {
    let nq = "<http://example.org/s> <http://example.org/p> \"object\" <http://example.org/g>";
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    // Should fail due to missing period
    assert!(quads.is_err());
}

#[test]
fn test_error_invalid_iri() {
    let nq = "<invalid iri with spaces> <http://example.org/p> \"object\" <http://example.org/g> .";
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    // Should fail due to invalid IRI
    assert!(quads.is_err());
}

#[test]
fn test_error_unclosed_literal() {
    let nq = r#"<http://example.org/s> <http://example.org/p> "unclosed literal <http://example.org/g> ."#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    // Should fail due to unclosed literal
    assert!(quads.is_err());
}

#[test]
fn test_serialization_roundtrip() {
    use oxirs_ttl::Serializer;

    let original_nq = r#"<http://example.org/alice> <http://example.org/name> "Alice" <http://example.org/g1> .
<http://example.org/bob> <http://example.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> <http://example.org/g2> .
_:blank1 <http://example.org/type> "Person" .
<http://example.org/charlie> <http://example.org/city> "Tokyo" <http://example.org/g1> ."#;

    // Parse
    let parser = NQuadsParser::new();
    let quads: Vec<_> = parser
        .for_reader(Cursor::new(original_nq))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Serialize
    let mut buffer = Vec::new();
    let serializer = NQuadsSerializer::new();
    serializer.serialize(&quads, &mut buffer).unwrap();

    // Parse serialized output
    let serialized = String::from_utf8(buffer).unwrap();
    let parser2 = NQuadsParser::new();
    let quads2: Vec<_> = parser2
        .for_reader(Cursor::new(serialized))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Compare
    assert_eq!(quads.len(), quads2.len());
}

#[test]
fn test_serializer_escaping() {
    use oxirs_ttl::Serializer;

    let quad = Quad::new(
        NamedNode::new_unchecked("http://example.org/s"),
        NamedNode::new_unchecked("http://example.org/p"),
        Literal::new_simple_literal("line1\nline2\ttab\"quote\\"),
        NamedNode::new_unchecked("http://example.org/g"),
    );

    let mut buffer = Vec::new();
    let serializer = NQuadsSerializer::new();
    serializer.serialize(&[quad], &mut buffer).unwrap();

    let serialized = String::from_utf8(buffer).unwrap();

    // Check that special characters are escaped
    assert!(serialized.contains("\\n"));
    assert!(serialized.contains("\\t"));
    assert!(serialized.contains("\\\""));
    assert!(serialized.contains("\\\\"));
}

#[test]
fn test_large_document() {
    let mut nq = String::new();
    for i in 0..10_000 {
        let graph = if i % 2 == 0 {
            format!("<http://example.org/g{}>", i % 10)
        } else {
            String::new()
        };
        nq.push_str(&format!(
            "<http://example.org/s{}> <http://example.org/p> \"object{}\" {} .\n",
            i, i, graph
        ));
    }

    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    assert_eq!(quads.unwrap().len(), 10_000);
}

#[test]
#[ignore = "StreamingParser currently only supports Turtle (triples), not N-Quads (quads). Requires format-aware streaming parser - deferred to beta.2"]
fn test_streaming_parsing() {
    use oxirs_ttl::streaming::{StreamingConfig, StreamingParser};

    let mut nq = String::new();
    for i in 0..1000 {
        let graph = if i % 2 == 0 {
            format!("<http://example.org/g{}>", i % 5)
        } else {
            String::new()
        };
        nq.push_str(&format!(
            "<http://example.org/s{}> <http://example.org/p> \"object{}\" {} .\n",
            i, i, graph
        ));
    }

    let config = StreamingConfig::default().with_batch_size(100);
    let mut parser = StreamingParser::with_config(Cursor::new(nq), config);

    let mut total = 0;
    while let Some(batch) = parser.next_batch().unwrap() {
        total += batch.len();
    }

    assert_eq!(total, 1000);
}

#[test]
fn test_graph_grouping() {
    let nq = r#"
<http://example.org/s1> <http://example.org/p1> "o1" <http://example.org/g1> .
<http://example.org/s2> <http://example.org/p2> "o2" <http://example.org/g1> .
<http://example.org/s3> <http://example.org/p3> "o3" <http://example.org/g2> .
<http://example.org/s4> <http://example.org/p4> "o4" <http://example.org/g1> .
"#;
    let parser = NQuadsParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(nq)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();

    // Count quads per graph
    let g1_count = quads
        .iter()
        .filter(|q| matches!(q.graph_name(), GraphName::NamedNode(n) if n.as_str().ends_with("g1")))
        .count();

    let g2_count = quads
        .iter()
        .filter(|q| matches!(q.graph_name(), GraphName::NamedNode(n) if n.as_str().ends_with("g2")))
        .count();

    assert_eq!(g1_count, 3);
    assert_eq!(g2_count, 1);
}
