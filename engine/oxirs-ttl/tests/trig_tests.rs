//! Comprehensive TriG parser and serializer tests

use oxirs_core::model::{GraphName, Object, Predicate, Subject};
use oxirs_ttl::trig::{TriGParser, TriGSerializer};
use oxirs_ttl::Parser;
use std::io::Cursor;

#[test]
fn test_simple_default_graph() {
    let trig = r#"
        @prefix ex: <http://example.org/> .
        ex:subject ex:predicate "object" .
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);
    assert!(matches!(quads[0].graph_name(), GraphName::DefaultGraph));
}

#[test]
fn test_named_graph_basic() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:alice ex:name "Alice" .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);
    assert!(matches!(quads[0].graph_name(), GraphName::NamedNode(_)));
}

#[test]
fn test_multiple_named_graphs() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:alice ex:name "Alice" .
            ex:alice ex:age "30" .
        }

        <http://example.org/graph2> {
            ex:bob ex:name "Bob" .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 3);

    // Check graph distribution
    let g1_count = quads
        .iter()
        .filter(
            |q| matches!(q.graph_name(), GraphName::NamedNode(n) if n.as_str().ends_with("graph1")),
        )
        .count();

    let g2_count = quads
        .iter()
        .filter(
            |q| matches!(q.graph_name(), GraphName::NamedNode(n) if n.as_str().ends_with("graph2")),
        )
        .count();

    assert_eq!(g1_count, 2);
    assert_eq!(g2_count, 1);
}

#[test]
fn test_mixed_default_and_named_graphs() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        ex:default1 ex:prop "In default graph" .

        <http://example.org/graph1> {
            ex:named1 ex:prop "In named graph" .
        }

        ex:default2 ex:prop "Also in default graph" .
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 3);

    let default_count = quads
        .iter()
        .filter(|q| matches!(q.graph_name(), GraphName::DefaultGraph))
        .count();

    assert_eq!(default_count, 2);
}

#[test]
fn test_blank_node_graph_name() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        _:graph1 {
            ex:subject ex:predicate "object" .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);
    assert!(matches!(quads[0].graph_name(), GraphName::BlankNode(_)));
}

#[test]
fn test_prefix_inheritance() {
    let trig = r#"
        @prefix ex: <http://example.org/> .
        @prefix foaf: <http://xmlns.com/foaf/0.1/> .

        <http://example.org/graph1> {
            ex:alice foaf:name "Alice" .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    // Check that both prefixes were resolved correctly
    match quads[0].subject() {
        Subject::NamedNode(n) => assert!(n.as_str().starts_with("http://example.org/")),
        _ => panic!("Expected NamedNode subject"),
    }
    match quads[0].predicate() {
        Predicate::NamedNode(n) => assert!(n.as_str().starts_with("http://xmlns.com/foaf/")),
        _ => panic!("Expected NamedNode predicate"),
    }
}

#[test]
fn test_graph_specific_prefixes() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            @prefix local: <http://local.example.org/> .
            local:subject ex:predicate "object" .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);
}

#[test]
fn test_base_directive() {
    let trig = r#"
        @base <http://example.org/> .
        @prefix ex: <vocab#> .

        <graph1> {
            <subject1> ex:predicate "object" .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);
}

#[test]
fn test_turtle_syntax_in_graph() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:alice ex:name "Alice" ;
                     ex:age "30" ;
                     ex:city "Tokyo" .

            ex:bob ex:knows ex:alice , ex:charlie .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 5); // alice has 3 properties, bob knows 2 people
}

#[test]
fn test_blank_node_subjects() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            [ ex:name "Anonymous" ; ex:age "25" ] .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 2);

    // Both quads should have the same blank node subject
    if let (Subject::BlankNode(bn1), Subject::BlankNode(bn2)) =
        (quads[0].subject(), quads[1].subject())
    {
        assert_eq!(bn1.id(), bn2.id());
    } else {
        panic!("Expected blank node subjects");
    }
}

#[test]
fn test_blank_node_property_list() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:alice ex:friend [ ex:name "Friend" ] .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 2);
}

#[test]
fn test_collections() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:list ex:contains ( "one" "two" "three" ) .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    // Collections generate multiple triples (RDF list structure)
    assert!(quads.len() > 1);
}

#[test]
fn test_empty_graph() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/emptyGraph> {
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    assert_eq!(quads.unwrap().len(), 0);
}

#[test]
fn test_comments() {
    let trig = r#"
        # Top-level comment
        @prefix ex: <http://example.org/> .

        # Comment before graph
        <http://example.org/graph1> {
            # Comment in graph
            ex:alice ex:name "Alice" . # Inline comment
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    assert_eq!(quads.unwrap().len(), 1);
}

#[test]
fn test_literals_with_datatypes() {
    let trig = r#"
        @prefix ex: <http://example.org/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

        <http://example.org/graph1> {
            ex:subject ex:int "42"^^xsd:integer ;
                      ex:float "3.14"^^xsd:float ;
                      ex:bool "true"^^xsd:boolean ;
                      ex:date "2023-01-01"^^xsd:date .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 4);
}

#[test]
fn test_language_tagged_literals() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:book ex:title "Hello"@en ;
                    ex:title "Bonjour"@fr ;
                    ex:title "Hola"@es ;
                    ex:title "こんにちは"@ja .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 4);

    // Check that all have language tags
    for quad in &quads {
        if let Object::Literal(lit) = quad.object() {
            assert!(lit.language().is_some());
        }
    }
}

#[test]
fn test_long_literals() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:book ex:description """
                This is a long literal
                that spans multiple lines
                with preserved whitespace.
            """ .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        assert!(lit.value().contains('\n'));
    }
}

#[test]
fn test_escape_sequences() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:subject ex:text "line1\nline2\ttab\"quote\\" .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        let value = lit.value();
        assert!(value.contains('\n'));
        assert!(value.contains('\t'));
        assert!(value.contains('"'));
        assert!(value.contains('\\'));
    }
}

#[test]
fn test_unicode_characters() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:subject ex:text "日本語 Русский 中文 🦀" .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    if let Err(ref e) = quads {
        eprintln!("DEBUG ERROR: {:?}", e);
    }
    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);

    if let Object::Literal(lit) = quads[0].object() {
        assert!(lit.value().contains("日本語"));
        assert!(lit.value().contains("🦀"));
    }
}

#[test]
fn test_numeric_shortcuts() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:subject ex:int 42 ;
                      ex:float 3.14 ;
                      ex:neg -5 ;
                      ex:sci 1.5e10 .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 4);
}

#[test]
fn test_boolean_shortcuts() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:subject1 ex:bool true .
            ex:subject2 ex:bool false .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 2);
}

#[test]
fn test_relative_iris() {
    let trig = r#"
        @base <http://example.org/> .
        @prefix ex: <vocab#> .

        <graph1> {
            <subject1> ex:predicate "object" .
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    let quads = quads.unwrap();
    assert_eq!(quads.len(), 1);
}

#[test]
fn test_large_document() {
    let mut trig = String::from(
        r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
    "#,
    );

    for i in 0..1000 {
        trig.push_str(&format!(
            "    ex:subject{} ex:predicate \"object{}\" .\n",
            i, i
        ));
    }

    trig.push_str("}\n");

    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    assert_eq!(quads.unwrap().len(), 1000);
}

#[test]
fn test_multiple_large_graphs() {
    let mut trig = String::from("@prefix ex: <http://example.org/> .\n");

    for graph_i in 0..5 {
        trig.push_str(&format!("<http://example.org/graph{}> {{\n", graph_i));
        for i in 0..100 {
            trig.push_str(&format!(
                "    ex:subject{} ex:predicate \"object{}\" .\n",
                i, i
            ));
        }
        trig.push_str("}\n");
    }

    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    assert!(quads.is_ok());
    assert_eq!(quads.unwrap().len(), 500);
}

#[test]
fn test_error_unclosed_graph() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:subject ex:predicate "object" .
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    // Should fail due to unclosed graph
    assert!(quads.is_err());
}

#[test]
fn test_error_invalid_syntax() {
    let trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            invalid syntax here
        }
    "#;
    let parser = TriGParser::new();
    let quads: Result<Vec<_>, _> = parser.for_reader(Cursor::new(trig)).collect();

    // Should fail due to invalid syntax
    assert!(quads.is_err());
}

#[test]
fn test_serialization_roundtrip() {
    use oxirs_ttl::Serializer;

    let original_trig = r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
            ex:alice ex:name "Alice" .
            ex:bob ex:name "Bob" .
        }

        <http://example.org/graph2> {
            ex:charlie ex:name "Charlie" .
        }
    "#;

    // Parse
    let parser = TriGParser::new();
    let quads: Vec<_> = parser
        .for_reader(Cursor::new(original_trig))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Serialize
    let mut buffer = Vec::new();
    let serializer = TriGSerializer::new();
    serializer.serialize(&quads, &mut buffer).unwrap();

    // Parse serialized output
    let serialized = String::from_utf8(buffer).unwrap();
    let parser2 = TriGParser::new();
    let quads2: Vec<_> = parser2
        .for_reader(Cursor::new(serialized))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    // Compare
    assert_eq!(quads.len(), quads2.len());
}

#[test]
fn test_streaming_parsing() {
    use oxirs_ttl::streaming::{StreamingConfig, StreamingParser};

    let mut trig = String::from(
        r#"
        @prefix ex: <http://example.org/> .

        <http://example.org/graph1> {
    "#,
    );

    for i in 0..1000 {
        trig.push_str(&format!(
            "    ex:subject{} ex:predicate \"object{}\" .\n",
            i, i
        ));
    }

    trig.push_str("}\n");

    let config = StreamingConfig::default().with_batch_size(100);
    let mut parser = StreamingParser::with_config(Cursor::new(trig), config);

    let mut total = 0;
    while let Some(batch) = parser.next_batch().unwrap() {
        total += batch.len();
    }

    assert_eq!(total, 1000);
}
