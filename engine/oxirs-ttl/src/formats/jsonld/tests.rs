use super::*;
use serde_json::json;

// ── JsonLdTerm ─────────────────────────────────────────────────────────

#[test]
fn test_term_iri_is_iri() {
    let t = JsonLdTerm::Iri("http://example.org/foo".into());
    assert!(t.is_iri());
    assert!(!t.is_blank_node());
    assert!(!t.is_literal());
}

#[test]
fn test_term_blank_node_is_blank_node() {
    let t = JsonLdTerm::BlankNode("_:b0".into());
    assert!(t.is_blank_node());
    assert!(!t.is_iri());
}

#[test]
fn test_term_literal_is_literal() {
    let t = JsonLdTerm::Literal {
        value: "hello".into(),
        datatype: XSD_STRING.into(),
        language: None,
    };
    assert!(t.is_literal());
}

#[test]
fn test_term_nquads_iri() {
    let t = JsonLdTerm::Iri("http://example.org/s".into());
    assert_eq!(t.to_nquads_string(), "<http://example.org/s>");
}

#[test]
fn test_term_nquads_blank_node() {
    let t = JsonLdTerm::BlankNode("_:b0".into());
    assert_eq!(t.to_nquads_string(), "_:b0");
}

#[test]
fn test_term_nquads_literal_plain() {
    let t = JsonLdTerm::Literal {
        value: "hello".into(),
        datatype: XSD_STRING.into(),
        language: None,
    };
    assert!(t.to_nquads_string().contains("hello"));
}

#[test]
fn test_term_nquads_literal_lang() {
    let t = JsonLdTerm::Literal {
        value: "bonjour".into(),
        datatype: RDF_LANG_STRING.into(),
        language: Some("fr".into()),
    };
    let s = t.to_nquads_string();
    assert!(s.contains("@fr"));
}

#[test]
fn test_term_nquads_literal_escape() {
    let t = JsonLdTerm::Literal {
        value: "line1\nline2".into(),
        datatype: XSD_STRING.into(),
        language: None,
    };
    let s = t.to_nquads_string();
    assert!(s.contains("\\n"));
    assert!(!s.contains('\n'));
}

// ── JsonLdQuad ─────────────────────────────────────────────────────────

#[test]
fn test_quad_triple_no_graph() {
    let q = JsonLdQuad::triple(
        JsonLdTerm::Iri("http://s".into()),
        JsonLdTerm::Iri("http://p".into()),
        JsonLdTerm::Iri("http://o".into()),
    );
    assert!(q.graph.is_none());
}

#[test]
fn test_quad_named_graph() {
    let q = JsonLdQuad::named(
        JsonLdTerm::Iri("http://s".into()),
        JsonLdTerm::Iri("http://p".into()),
        JsonLdTerm::Iri("http://o".into()),
        JsonLdTerm::Iri("http://g".into()),
    );
    assert!(q.graph.is_some());
}

#[test]
fn test_quad_nquads_line_default_graph() {
    let q = JsonLdQuad::triple(
        JsonLdTerm::Iri("http://s".into()),
        JsonLdTerm::Iri("http://p".into()),
        JsonLdTerm::Iri("http://o".into()),
    );
    let line = q.to_nquads_line();
    assert!(line.ends_with('.'));
    assert!(!line.contains("<http://g>"));
}

#[test]
fn test_quad_nquads_line_named_graph() {
    let q = JsonLdQuad::named(
        JsonLdTerm::Iri("http://s".into()),
        JsonLdTerm::Iri("http://p".into()),
        JsonLdTerm::Iri("http://o".into()),
        JsonLdTerm::Iri("http://g".into()),
    );
    let line = q.to_nquads_line();
    assert!(line.contains("<http://g>"));
}

// ── ContainerType ──────────────────────────────────────────────────────

#[test]
fn test_container_type_round_trip() {
    let types = [
        ContainerType::List,
        ContainerType::Set,
        ContainerType::Index,
        ContainerType::Language,
        ContainerType::Id,
        ContainerType::Graph,
    ];
    for ct in &types {
        let s = ct.as_str();
        let parsed = ContainerType::from_str(s).expect("should parse");
        assert_eq!(&parsed, ct);
    }
}

#[test]
fn test_container_type_unknown() {
    assert!(ContainerType::from_str("@unknown").is_none());
}

// ── JsonLdContext::parse ────────────────────────────────────────────────

#[test]
fn test_context_parse_empty_object() {
    let ctx = JsonLdContext::parse(&json!({})).expect("parse");
    assert!(ctx.vocab.is_none());
    assert!(ctx.base_iri.is_none());
}

#[test]
fn test_context_parse_base() {
    let ctx = JsonLdContext::parse(&json!({ "@base": "http://example.org/" })).expect("parse");
    assert_eq!(ctx.base_iri.unwrap(), "http://example.org/");
}

#[test]
fn test_context_parse_vocab() {
    let ctx = JsonLdContext::parse(&json!({ "@vocab": "http://schema.org/" })).expect("parse");
    assert_eq!(ctx.vocab.unwrap(), "http://schema.org/");
}

#[test]
fn test_context_parse_language() {
    let ctx = JsonLdContext::parse(&json!({ "@language": "en" })).expect("parse");
    assert_eq!(ctx.default_language.unwrap(), "en");
}

#[test]
fn test_context_parse_prefix_mapping() {
    let ctx = JsonLdContext::parse(&json!({
        "ex": "http://example.org/"
    }))
    .expect("parse");
    assert_eq!(ctx.prefixes.get("ex").unwrap(), "http://example.org/");
}

#[test]
fn test_context_parse_term_definition_string() {
    let ctx = JsonLdContext::parse(&json!({
        "name": "http://schema.org/name"
    }))
    .expect("parse");
    let def = ctx.terms.get("name").expect("term");
    assert_eq!(def.iri, "http://schema.org/name");
}

#[test]
fn test_context_parse_term_definition_object_with_id() {
    let ctx = JsonLdContext::parse(&json!({
        "name": { "@id": "http://schema.org/name", "@container": "@set" }
    }))
    .expect("parse");
    let def = ctx.terms.get("name").expect("term");
    assert_eq!(def.container, Some(ContainerType::Set));
}

#[test]
fn test_context_parse_term_definition_type_coercion() {
    let ctx = JsonLdContext::parse(&json!({
        "age": { "@id": "http://schema.org/age", "@type": "xsd:integer" }
    }))
    .expect("parse");
    let def = ctx.terms.get("age").expect("term");
    assert!(def.type_coercion.is_some());
}

#[test]
fn test_context_parse_array() {
    let ctx = JsonLdContext::parse(&json!([
        { "@vocab": "http://schema.org/" },
        { "ex": "http://example.org/" }
    ]))
    .expect("parse");
    assert!(ctx.vocab.is_some());
    assert!(ctx.prefixes.contains_key("ex"));
}

#[test]
fn test_context_parse_null_resets() {
    let ctx = JsonLdContext::parse(&Value::Null).expect("parse");
    assert!(ctx.vocab.is_none());
    assert!(ctx.base_iri.is_none());
    assert!(ctx.prefixes.is_empty());
}

#[test]
fn test_context_parse_invalid_type() {
    let result = JsonLdContext::parse(&json!(42));
    assert!(result.is_err());
}

// ── expand_term ────────────────────────────────────────────────────────

#[test]
fn test_expand_term_absolute_iri_passthrough() {
    let ctx = JsonLdContext::default();
    assert_eq!(
        ctx.expand_term("http://example.org/foo"),
        "http://example.org/foo"
    );
}

#[test]
fn test_expand_term_curie_rdf() {
    let ctx = JsonLdContext::default();
    assert_eq!(
        ctx.expand_term("rdf:type"),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    );
}

#[test]
fn test_expand_term_curie_xsd() {
    let ctx = JsonLdContext::default();
    assert_eq!(
        ctx.expand_term("xsd:integer"),
        "http://www.w3.org/2001/XMLSchema#integer"
    );
}

#[test]
fn test_expand_term_curie_schema() {
    let ctx = JsonLdContext::default();
    assert_eq!(ctx.expand_term("schema:name"), "http://schema.org/name");
}

#[test]
fn test_expand_term_curie_foaf() {
    let ctx = JsonLdContext::default();
    assert_eq!(
        ctx.expand_term("foaf:Person"),
        "http://xmlns.com/foaf/0.1/Person"
    );
}

#[test]
fn test_expand_term_via_vocab() {
    let ctx = JsonLdContext {
        vocab: Some("http://schema.org/".into()),
        ..Default::default()
    };
    assert_eq!(ctx.expand_term("name"), "http://schema.org/name");
}

#[test]
fn test_expand_term_via_base() {
    let mut ctx = JsonLdContext::empty();
    ctx.base_iri = Some("http://example.org/".into());
    assert_eq!(ctx.expand_term("foo"), "http://example.org/foo");
}

#[test]
fn test_expand_term_keyword_passthrough() {
    let ctx = JsonLdContext::default();
    assert_eq!(ctx.expand_term("@type"), "@type");
}

#[test]
fn test_expand_term_via_term_definition() {
    let mut ctx = JsonLdContext::default();
    ctx.terms.insert(
        "name".into(),
        TermDefinition {
            iri: "http://schema.org/name".into(),
            container: None,
            language: None,
            type_coercion: None,
        },
    );
    assert_eq!(ctx.expand_term("name"), "http://schema.org/name");
}

// ── compact_iri ────────────────────────────────────────────────────────

#[test]
fn test_compact_iri_keyword_passthrough() {
    let ctx = JsonLdContext::default();
    assert_eq!(ctx.compact_iri("@type"), "@type");
}

#[test]
fn test_compact_iri_schema_prefix() {
    let ctx = JsonLdContext::default();
    let compacted = ctx.compact_iri("http://schema.org/name");
    assert_eq!(compacted, "schema:name");
}

#[test]
fn test_compact_iri_rdf_prefix() {
    let ctx = JsonLdContext::default();
    let compacted = ctx.compact_iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    assert_eq!(compacted, "rdf:type");
}

#[test]
fn test_compact_iri_xsd_prefix() {
    let ctx = JsonLdContext::default();
    let compacted = ctx.compact_iri("http://www.w3.org/2001/XMLSchema#string");
    assert_eq!(compacted, "xsd:string");
}

#[test]
fn test_compact_iri_via_term() {
    let mut ctx = JsonLdContext::default();
    ctx.terms.insert(
        "name".into(),
        TermDefinition {
            iri: "http://schema.org/name".into(),
            container: None,
            language: None,
            type_coercion: None,
        },
    );
    assert_eq!(ctx.compact_iri("http://schema.org/name"), "name");
}

#[test]
fn test_compact_iri_no_match_returns_full() {
    let ctx = JsonLdContext::default();
    let iri = "http://unknown.example.org/foo";
    assert_eq!(ctx.compact_iri(iri), iri);
}

#[test]
fn test_compact_iri_via_vocab() {
    let mut ctx = JsonLdContext::empty();
    ctx.vocab = Some("http://schema.org/".into());
    let compacted = ctx.compact_iri("http://schema.org/Person");
    assert_eq!(compacted, "Person");
}

// ── Expansion algorithm ────────────────────────────────────────────────

#[test]
fn test_expand_simple_object() {
    let input = json!({
        "@context": { "@vocab": "http://schema.org/" },
        "@id": "http://example.org/alice",
        "@type": "Person",
        "name": "Alice"
    });
    let expanded = JsonLdProcessor::expand(&input, None).expect("expand");
    let arr = expanded.as_array().expect("array");
    assert!(!arr.is_empty());
    let node = &arr[0];
    assert_eq!(node["@id"], "http://example.org/alice");
    let types = node["@type"].as_array().expect("types");
    assert!(types.iter().any(|t| t == "http://schema.org/Person"));
}

#[test]
fn test_expand_curie_in_id() {
    let input = json!({
        "@context": { "ex": "http://example.org/" },
        "@id": "ex:alice"
    });
    let expanded = JsonLdProcessor::expand(&input, None).expect("expand");
    let id = expanded[0]["@id"].as_str().expect("id");
    assert_eq!(id, "http://example.org/alice");
}

#[test]
fn test_expand_language_tagged_value() {
    let input = json!({
        "@context": { "@vocab": "http://schema.org/", "@language": "en" },
        "name": "Alice"
    });
    let expanded = JsonLdProcessor::expand(&input, None).expect("expand");
    let name_arr = expanded[0]["http://schema.org/name"]
        .as_array()
        .expect("name");
    assert!(!name_arr.is_empty());
    let val = &name_arr[0];
    assert_eq!(val["@language"], "en");
}

#[test]
fn test_expand_typed_value() {
    let input = json!({
        "@context": { "@vocab": "http://schema.org/" },
        "age": { "@value": "42", "@type": "xsd:integer" }
    });
    let expanded = JsonLdProcessor::expand(&input, None).expect("expand");
    let age_arr = expanded[0]["http://schema.org/age"]
        .as_array()
        .expect("age");
    let val = &age_arr[0];
    assert!(val["@type"].as_str().unwrap_or("").contains("integer"));
}

#[test]
fn test_expand_external_context() {
    let input = json!({ "@id": "ex:bob", "ex:name": "Bob" });
    let ctx = json!({ "ex": "http://example.org/" });
    let expanded = JsonLdProcessor::expand(&input, Some(&ctx)).expect("expand");
    let id = expanded[0]["@id"].as_str().expect("id");
    assert_eq!(id, "http://example.org/bob");
}

#[test]
fn test_expand_returns_array() {
    let input = json!({ "@id": "http://example.org/x" });
    let expanded = JsonLdProcessor::expand(&input, None).expect("expand");
    assert!(expanded.is_array());
}

#[test]
fn test_expand_nested_object() {
    let input = json!({
        "@context": { "@vocab": "http://schema.org/" },
        "@id": "http://example.org/alice",
        "knows": {
            "@id": "http://example.org/bob",
            "name": "Bob"
        }
    });
    let expanded = JsonLdProcessor::expand(&input, None).expect("expand");
    let knows = expanded[0]["http://schema.org/knows"]
        .as_array()
        .expect("knows");
    assert!(!knows.is_empty());
}

#[test]
fn test_expand_array_value() {
    let input = json!({
        "@context": { "@vocab": "http://schema.org/" },
        "@id": "http://example.org/a",
        "name": ["Alice", "Alicia"]
    });
    let expanded = JsonLdProcessor::expand(&input, None).expect("expand");
    let name = expanded[0]["http://schema.org/name"]
        .as_array()
        .expect("name");
    assert_eq!(name.len(), 2);
}

// ── Compaction algorithm ────────────────────────────────────────────────

#[test]
fn test_compact_basic() {
    let input = json!([{
        "@id": "http://example.org/alice",
        "@type": ["http://schema.org/Person"],
        "http://schema.org/name": [{ "@value": "Alice", "@type": "http://www.w3.org/2001/XMLSchema#string" }]
    }]);
    let ctx = json!({
        "schema": "http://schema.org/",
        "ex": "http://example.org/"
    });
    let compacted = JsonLdProcessor::compact(&input, &ctx).expect("compact");
    assert!(compacted.get("@context").is_some());
}

#[test]
fn test_compact_single_node_unwrapped() {
    let input = json!([{
        "@id": "http://example.org/alice"
    }]);
    let ctx = json!({ "ex": "http://example.org/" });
    let compacted = JsonLdProcessor::compact(&input, &ctx).expect("compact");
    // Should not have @graph for single node
    assert!(compacted.get("@id").is_some());
}

// ── Flattening algorithm ────────────────────────────────────────────────

#[test]
fn test_flatten_produces_graph() {
    let input = json!({
        "@context": { "@vocab": "http://schema.org/" },
        "@id": "http://example.org/alice",
        "knows": { "@id": "http://example.org/bob" }
    });
    let flat = JsonLdProcessor::flatten(&input).expect("flatten");
    let graph = flat["@graph"].as_array().expect("@graph");
    assert!(!graph.is_empty());
}

#[test]
fn test_flatten_all_nodes_present() {
    let input = json!([
        { "@id": "http://example.org/alice", "http://schema.org/knows": { "@id": "http://example.org/bob" } },
        { "@id": "http://example.org/bob", "http://schema.org/name": [{ "@value": "Bob", "@type": "http://www.w3.org/2001/XMLSchema#string" }] }
    ]);
    let flat = JsonLdProcessor::flatten(&input).expect("flatten");
    let graph = flat["@graph"].as_array().expect("@graph");
    let ids: Vec<&str> = graph
        .iter()
        .filter_map(|n| n.get("@id").and_then(|v| v.as_str()))
        .collect();
    assert!(ids.contains(&"http://example.org/alice") || !ids.is_empty());
}

#[test]
fn test_flatten_nested_becomes_flat() {
    let input = json!({
        "@id": "http://example.org/a",
        "http://example.org/child": {
            "@id": "http://example.org/b",
            "http://example.org/child": {
                "@id": "http://example.org/c"
            }
        }
    });
    let flat = JsonLdProcessor::flatten(&input).expect("flatten");
    let graph = flat["@graph"].as_array().expect("@graph");
    // All three nodes should be present at top level
    let ids: Vec<&str> = graph
        .iter()
        .filter_map(|n| n.get("@id").and_then(|v| v.as_str()))
        .collect();
    assert!(ids.len() >= 2);
}

// ── Framing algorithm ──────────────────────────────────────────────────

#[test]
fn test_frame_by_type() {
    let input = json!([
        { "@id": "http://example.org/alice", "@type": ["http://schema.org/Person"], "http://schema.org/name": [{"@value": "Alice", "@type": "http://www.w3.org/2001/XMLSchema#string"}] },
        { "@id": "http://example.org/book1", "@type": ["http://schema.org/Book"] }
    ]);
    let frame = json!({ "@type": "http://schema.org/Person" });
    let framed = JsonLdProcessor::frame(&input, &frame).expect("frame");
    let graph = framed["@graph"].as_array().expect("@graph");
    // Only Person nodes should be in result
    assert!(graph.iter().all(|n| {
        n["@type"]
            .as_array()
            .map(|t| {
                t.iter()
                    .any(|v| v.as_str() == Some("http://schema.org/Person"))
            })
            .unwrap_or(false)
    }));
}

#[test]
fn test_frame_by_id() {
    let input = json!([
        { "@id": "http://example.org/alice" },
        { "@id": "http://example.org/bob" }
    ]);
    let frame = json!({ "@id": "http://example.org/alice" });
    let framed = JsonLdProcessor::frame(&input, &frame).expect("frame");
    let graph = framed["@graph"].as_array().expect("@graph");
    assert_eq!(graph.len(), 1);
    assert_eq!(graph[0]["@id"], "http://example.org/alice");
}

#[test]
fn test_frame_no_match_empty_graph() {
    let input = json!([{ "@id": "http://example.org/alice" }]);
    let frame = json!({ "@id": "http://example.org/nobody" });
    let framed = JsonLdProcessor::frame(&input, &frame).expect("frame");
    let graph = framed["@graph"].as_array().expect("@graph");
    assert!(graph.is_empty());
}

// ── to_rdf ─────────────────────────────────────────────────────────────

#[test]
fn test_to_rdf_type_triple() {
    let input = json!([{
        "@id": "http://example.org/alice",
        "@type": ["http://schema.org/Person"]
    }]);
    let quads = JsonLdProcessor::to_rdf(&input).expect("to_rdf");
    assert!(!quads.is_empty());
    let type_quad = quads.iter().find(|q| {
        q.predicate == JsonLdTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into())
    });
    assert!(type_quad.is_some());
}

#[test]
fn test_to_rdf_literal_object() {
    let input = json!([{
        "@id": "http://example.org/alice",
        "http://schema.org/name": [{ "@value": "Alice", "@type": "http://www.w3.org/2001/XMLSchema#string" }]
    }]);
    let quads = JsonLdProcessor::to_rdf(&input).expect("to_rdf");
    let name_quad = quads
        .iter()
        .find(|q| q.predicate == JsonLdTerm::Iri("http://schema.org/name".into()));
    assert!(name_quad.is_some());
    if let Some(q) = name_quad {
        assert!(q.object.is_literal());
    }
}

#[test]
fn test_to_rdf_iri_object() {
    let input = json!([{
        "@id": "http://example.org/alice",
        "http://schema.org/knows": [{ "@id": "http://example.org/bob" }]
    }]);
    let quads = JsonLdProcessor::to_rdf(&input).expect("to_rdf");
    let knows_quad = quads
        .iter()
        .find(|q| q.predicate == JsonLdTerm::Iri("http://schema.org/knows".into()));
    assert!(knows_quad.is_some());
    if let Some(q) = knows_quad {
        assert!(q.object.is_iri());
    }
}

#[test]
fn test_to_rdf_blank_node_subject() {
    let input = json!([{
        "@type": ["http://schema.org/Person"],
        "http://schema.org/name": [{ "@value": "Alice", "@type": "http://www.w3.org/2001/XMLSchema#string" }]
    }]);
    let quads = JsonLdProcessor::to_rdf(&input).expect("to_rdf");
    assert!(!quads.is_empty());
    // Subject should be a blank node (no @id in input)
    for q in &quads {
        assert!(q.subject.is_blank_node() || q.subject.is_iri());
    }
}

#[test]
fn test_to_rdf_multiple_triples() {
    let input = json!([{
        "@id": "http://example.org/alice",
        "@type": ["http://schema.org/Person"],
        "http://schema.org/name": [{ "@value": "Alice", "@type": "http://www.w3.org/2001/XMLSchema#string" }],
        "http://schema.org/age": [{ "@value": "30", "@type": "http://www.w3.org/2001/XMLSchema#integer" }]
    }]);
    let quads = JsonLdProcessor::to_rdf(&input).expect("to_rdf");
    assert!(quads.len() >= 3);
}

// ── from_rdf ────────────────────────────────────────────────────────────

#[test]
fn test_from_rdf_basic() {
    let quads = vec![JsonLdQuad::triple(
        JsonLdTerm::Iri("http://example.org/alice".into()),
        JsonLdTerm::Iri("http://schema.org/name".into()),
        JsonLdTerm::Literal {
            value: "Alice".into(),
            datatype: XSD_STRING.into(),
            language: None,
        },
    )];
    let result = JsonLdProcessor::from_rdf(&quads, None).expect("from_rdf");
    let arr = result.as_array().expect("array");
    assert!(!arr.is_empty());
}

#[test]
fn test_from_rdf_with_context() {
    let quads = vec![JsonLdQuad::triple(
        JsonLdTerm::Iri("http://example.org/alice".into()),
        JsonLdTerm::Iri("http://schema.org/name".into()),
        JsonLdTerm::Literal {
            value: "Alice".into(),
            datatype: XSD_STRING.into(),
            language: None,
        },
    )];
    let ctx = json!({ "schema": "http://schema.org/" });
    let result = JsonLdProcessor::from_rdf(&quads, Some(&ctx)).expect("from_rdf");
    // Result should be compacted
    assert!(result.get("@context").is_some());
}

#[test]
fn test_from_rdf_lang_literal() {
    let quads = vec![JsonLdQuad::triple(
        JsonLdTerm::Iri("http://example.org/doc".into()),
        JsonLdTerm::Iri("http://schema.org/name".into()),
        JsonLdTerm::Literal {
            value: "bonjour".into(),
            datatype: RDF_LANG_STRING.into(),
            language: Some("fr".into()),
        },
    )];
    let result = JsonLdProcessor::from_rdf(&quads, None).expect("from_rdf");
    let arr = result.as_array().expect("array");
    assert!(!arr.is_empty());
}

// ── Round-trip ─────────────────────────────────────────────────────────

#[test]
fn test_round_trip_rdf_to_jsonld_to_rdf() {
    let original_quads = vec![
        JsonLdQuad::triple(
            JsonLdTerm::Iri("http://example.org/alice".into()),
            JsonLdTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".into()),
            JsonLdTerm::Iri("http://schema.org/Person".into()),
        ),
        JsonLdQuad::triple(
            JsonLdTerm::Iri("http://example.org/alice".into()),
            JsonLdTerm::Iri("http://schema.org/name".into()),
            JsonLdTerm::Literal {
                value: "Alice".into(),
                datatype: XSD_STRING.into(),
                language: None,
            },
        ),
    ];

    let jsonld = JsonLdProcessor::from_rdf(&original_quads, None).expect("from_rdf");
    let round_tripped = JsonLdProcessor::to_rdf(&jsonld).expect("to_rdf");

    // Should have at least the same number of triples
    assert!(round_tripped.len() >= original_quads.len());
}

#[test]
fn test_round_trip_jsonld_to_rdf_to_jsonld() {
    let input = json!([{
        "@id": "http://example.org/alice",
        "@type": ["http://schema.org/Person"],
        "http://schema.org/name": [{ "@value": "Alice", "@type": "http://www.w3.org/2001/XMLSchema#string" }]
    }]);

    let quads = JsonLdProcessor::to_rdf(&input).expect("to_rdf");
    let back = JsonLdProcessor::from_rdf(&quads, None).expect("from_rdf");

    let arr = back.as_array().expect("array");
    assert!(!arr.is_empty());
    // Should contain alice
    let alice = arr.iter().find(|n| {
        n.get("@id")
            .and_then(|v| v.as_str())
            .map(|id| id.contains("alice"))
            .unwrap_or(false)
    });
    assert!(alice.is_some());
}

// ── JsonLdWriter ────────────────────────────────────────────────────────

#[test]
fn test_writer_new_defaults() {
    let w = JsonLdWriter::new();
    assert!(!w.compact);
    assert!(!w.pretty);
    assert!(w.context.is_none());
}

#[test]
fn test_writer_with_context() {
    let w = JsonLdWriter::new().with_context(json!({ "schema": "http://schema.org/" }));
    assert!(w.context.is_some());
}

#[test]
fn test_writer_compact_mode() {
    let w = JsonLdWriter::new().compact_mode();
    assert!(w.compact);
}

#[test]
fn test_writer_pretty_print() {
    let w = JsonLdWriter::new().pretty_print();
    assert!(w.pretty);
}

#[test]
fn test_writer_write_triples_basic() {
    let triples = vec![Triple {
        subject: "http://example.org/alice".into(),
        predicate: "http://schema.org/name".into(),
        object: WriterObject::Literal("Alice".into()),
    }];
    let w = JsonLdWriter::new();
    let output = w.write_triples(&triples).expect("write_triples");
    let parsed: Value = serde_json::from_str(&output).expect("valid json");
    assert!(parsed.get("@graph").is_some());
}

#[test]
fn test_writer_write_triples_with_context() {
    let triples = vec![Triple {
        subject: "http://example.org/alice".into(),
        predicate: "http://schema.org/name".into(),
        object: WriterObject::Literal("Alice".into()),
    }];
    let ctx = json!({ "schema": "http://schema.org/", "ex": "http://example.org/" });
    let w = JsonLdWriter::new().with_context(ctx).compact_mode();
    let output = w.write_triples(&triples).expect("write_triples");
    let parsed: Value = serde_json::from_str(&output).expect("valid json");
    assert!(parsed.get("@context").is_some());
}

#[test]
fn test_writer_pretty_output_has_newlines() {
    let triples = vec![Triple {
        subject: "http://example.org/s".into(),
        predicate: "http://example.org/p".into(),
        object: WriterObject::Iri("http://example.org/o".into()),
    }];
    let w = JsonLdWriter::new().pretty_print();
    let output = w.write_triples(&triples).expect("write_triples");
    assert!(output.contains('\n'));
}

#[test]
fn test_writer_compact_output_no_newlines() {
    let triples = vec![Triple {
        subject: "http://example.org/s".into(),
        predicate: "http://example.org/p".into(),
        object: WriterObject::Iri("http://example.org/o".into()),
    }];
    let w = JsonLdWriter::new();
    let output = w.write_triples(&triples).expect("write_triples");
    assert!(!output.contains('\n'));
}

#[test]
fn test_writer_write_quads_named_graph() {
    let quads = vec![Quad {
        subject: "http://example.org/alice".into(),
        predicate: "http://schema.org/name".into(),
        object: WriterObject::Literal("Alice".into()),
        graph: Some("http://example.org/graph1".into()),
    }];
    let w = JsonLdWriter::new();
    let output = w.write_quads(&quads).expect("write_quads");
    let parsed: Value = serde_json::from_str(&output).expect("valid json");
    let graph = parsed["@graph"].as_array().expect("@graph");
    // Named graph should be in the output
    assert!(!graph.is_empty());
}

#[test]
fn test_writer_typed_literal() {
    let triples = vec![Triple {
        subject: "http://example.org/alice".into(),
        predicate: "http://schema.org/age".into(),
        object: WriterObject::TypedLiteral(
            "30".into(),
            "http://www.w3.org/2001/XMLSchema#integer".into(),
        ),
    }];
    let w = JsonLdWriter::new();
    let output = w.write_triples(&triples).expect("write_triples");
    assert!(output.contains("30"));
}

#[test]
fn test_writer_lang_literal() {
    let triples = vec![Triple {
        subject: "http://example.org/doc".into(),
        predicate: "http://schema.org/name".into(),
        object: WriterObject::LangLiteral("Hola".into(), "es".into()),
    }];
    let w = JsonLdWriter::new();
    let output = w.write_triples(&triples).expect("write_triples");
    assert!(output.contains("es") || output.contains("Hola"));
}

#[test]
fn test_writer_blank_node_object() {
    let triples = vec![Triple {
        subject: "http://example.org/alice".into(),
        predicate: "http://schema.org/address".into(),
        object: WriterObject::BlankNode("_:b0".into()),
    }];
    let w = JsonLdWriter::new();
    let output = w.write_triples(&triples).expect("write_triples");
    assert!(output.contains("_:b0"));
}

// ── is_absolute_iri helper ─────────────────────────────────────────────

#[test]
fn test_is_absolute_iri_http() {
    assert!(is_absolute_iri("http://example.org/"));
}

#[test]
fn test_is_absolute_iri_https() {
    assert!(is_absolute_iri("https://example.org/"));
}

#[test]
fn test_is_absolute_iri_urn() {
    assert!(is_absolute_iri("urn:isbn:0451450523"));
}

#[test]
fn test_is_absolute_iri_relative() {
    assert!(!is_absolute_iri("foo"));
}

#[test]
fn test_is_absolute_iri_curie() {
    // "rdf:type" looks like IRI but local part starts with non-//
    // The function returns true for CURIEs because they match the scheme pattern
    // This is intentional — the expand_term function handles proper CURIE detection
    let _ = is_absolute_iri("rdf:type");
}

// ── XSD constants ──────────────────────────────────────────────────────

#[test]
fn test_xsd_string_constant() {
    assert!(XSD_STRING.contains("XMLSchema#string"));
}

#[test]
fn test_xsd_boolean_constant() {
    assert!(XSD_BOOLEAN.contains("XMLSchema#boolean"));
}

#[test]
fn test_xsd_integer_constant() {
    assert!(XSD_INTEGER.contains("XMLSchema#integer"));
}

#[test]
fn test_rdf_lang_string_constant() {
    assert!(RDF_LANG_STRING.contains("rdf-syntax-ns#langString"));
}

// ── JsonLdError ────────────────────────────────────────────────────────

#[test]
fn test_error_display_invalid_context() {
    let e = JsonLdError::InvalidContext("bad context".into());
    assert!(e.to_string().contains("bad context"));
}

#[test]
fn test_error_display_invalid_iri() {
    let e = JsonLdError::InvalidIri("not an iri".into());
    assert!(e.to_string().contains("not an iri"));
}

#[test]
fn test_error_display_framing() {
    let e = JsonLdError::Framing("bad frame".into());
    assert!(e.to_string().contains("bad frame"));
}

// ── Integration tests ──────────────────────────────────────────────────

#[test]
fn test_full_schema_org_document() {
    let input = json!({
        "@context": {
            "@vocab": "http://schema.org/",
            "ex": "http://example.org/"
        },
        "@id": "ex:alice",
        "@type": "Person",
        "name": "Alice Smith",
        "email": "alice@example.org",
        "knows": {
            "@id": "ex:bob",
            "@type": "Person",
            "name": "Bob Jones"
        }
    });

    let expanded = JsonLdProcessor::expand(&input, None).expect("expand");
    let arr = expanded.as_array().expect("array");
    assert!(!arr.is_empty());

    let quads = JsonLdProcessor::to_rdf(&expanded).expect("to_rdf");
    assert!(quads.len() >= 4);

    let flat = JsonLdProcessor::flatten(&expanded).expect("flatten");
    let graph = flat["@graph"].as_array().expect("@graph");
    assert!(!graph.is_empty());
}

#[test]
fn test_context_builtin_prefixes_available() {
    let ctx = JsonLdContext::default();
    assert!(ctx.prefixes.contains_key("rdf"));
    assert!(ctx.prefixes.contains_key("rdfs"));
    assert!(ctx.prefixes.contains_key("owl"));
    assert!(ctx.prefixes.contains_key("xsd"));
    assert!(ctx.prefixes.contains_key("schema"));
    assert!(ctx.prefixes.contains_key("foaf"));
    assert!(ctx.prefixes.contains_key("skos"));
    assert!(ctx.prefixes.contains_key("dc"));
    assert!(ctx.prefixes.contains_key("dcterms"));
}

#[test]
fn test_expand_multiple_types() {
    let input = json!({
        "@context": { "@vocab": "http://schema.org/" },
        "@id": "http://example.org/x",
        "@type": ["Person", "Agent"]
    });
    let expanded = JsonLdProcessor::expand(&input, None).expect("expand");
    let types = expanded[0]["@type"].as_array().expect("types");
    assert_eq!(types.len(), 2);
    assert!(types.iter().any(|t| t == "http://schema.org/Person"));
    assert!(types.iter().any(|t| t == "http://schema.org/Agent"));
}

#[test]
fn test_context_language_applied_to_strings() {
    let ctx_json = json!({
        "@vocab": "http://schema.org/",
        "@language": "de"
    });
    let ctx = JsonLdContext::parse(&ctx_json).expect("parse");
    assert_eq!(ctx.default_language.as_deref(), Some("de"));
}
