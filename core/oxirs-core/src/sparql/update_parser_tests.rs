//! Tests for the SPARQL UPDATE statement parser.

use super::update_parser::*;
use std::collections::HashMap;

// ── INSERT DATA ─────────────────────────────────────────────────────────────

#[test]
fn test_insert_data_single_triple() {
    let input = r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> <http://ex.org/o> }"#;
    let req = parse_update(input).expect("should parse");
    assert_eq!(req.operations.len(), 1);
    match &req.operations[0] {
        UpdateOperation::InsertData { triples, graph } => {
            assert_eq!(triples.len(), 1);
            assert_eq!(triples[0].subject, "<http://ex.org/s>");
            assert_eq!(triples[0].predicate, "<http://ex.org/p>");
            assert_eq!(triples[0].object, "<http://ex.org/o>");
            assert!(graph.is_none());
        }
        other => panic!("expected InsertData, got {:?}", other),
    }
}

#[test]
fn test_insert_data_multiple_triples() {
    let input = r#"INSERT DATA {
        <http://ex.org/s1> <http://ex.org/p1> <http://ex.org/o1> .
        <http://ex.org/s2> <http://ex.org/p2> "hello"
    }"#;
    let req = parse_update(input).expect("should parse");
    assert_eq!(req.operations.len(), 1);
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[1].object, "\"hello\"");
    }
}

#[test]
fn test_insert_data_with_graph() {
    let input = r#"INSERT DATA { GRAPH <http://ex.org/g> { <http://ex.org/s> <http://ex.org/p> <http://ex.org/o> } }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::InsertData { graph, triples } = &req.operations[0] {
        assert_eq!(graph.as_deref(), Some("http://ex.org/g"));
        assert_eq!(triples.len(), 1);
    }
}

#[test]
fn test_insert_data_with_prefix() {
    let input = r#"PREFIX ex: <http://ex.org/>
    INSERT DATA { ex:s ex:p ex:o }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert_eq!(triples[0].subject, "<http://ex.org/s>");
        assert_eq!(triples[0].predicate, "<http://ex.org/p>");
        assert_eq!(triples[0].object, "<http://ex.org/o>");
    }
}

#[test]
fn test_insert_data_with_literal_datatype() {
    let input = r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "42"^^<http://www.w3.org/2001/XMLSchema#integer> }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert!(triples[0].object.contains("integer"));
    }
}

#[test]
fn test_insert_data_with_lang_tag() {
    let input = r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "bonjour"@fr }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert_eq!(triples[0].object, "\"bonjour\"@fr");
    }
}

// ── DELETE DATA ─────────────────────────────────────────────────────────────

#[test]
fn test_delete_data_single_triple() {
    let input = r#"DELETE DATA { <http://ex.org/s> <http://ex.org/p> <http://ex.org/o> }"#;
    let req = parse_update(input).expect("should parse");
    assert_eq!(req.operations.len(), 1);
    match &req.operations[0] {
        UpdateOperation::DeleteData { triples, graph } => {
            assert_eq!(triples.len(), 1);
            assert!(graph.is_none());
        }
        other => panic!("expected DeleteData, got {:?}", other),
    }
}

#[test]
fn test_delete_data_with_graph() {
    let input = r#"DELETE DATA { GRAPH <http://ex.org/g> { <http://ex.org/s> <http://ex.org/p> <http://ex.org/o> } }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::DeleteData { graph, .. } = &req.operations[0] {
        assert_eq!(graph.as_deref(), Some("http://ex.org/g"));
    }
}

#[test]
fn test_delete_data_multiple_triples() {
    let input = r#"DELETE DATA {
        <http://ex.org/s1> <http://ex.org/p1> <http://ex.org/o1> .
        <http://ex.org/s2> <http://ex.org/p2> <http://ex.org/o2>
    }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::DeleteData { triples, .. } = &req.operations[0] {
        assert_eq!(triples.len(), 2);
    }
}

// ── DELETE/INSERT WHERE ─────────────────────────────────────────────────────

#[test]
fn test_delete_insert_where() {
    let input = r#"DELETE { ?s <http://ex.org/old> ?o }
    INSERT { ?s <http://ex.org/new> ?o }
    WHERE { ?s <http://ex.org/old> ?o }"#;
    let req = parse_update(input).expect("should parse");
    match &req.operations[0] {
        UpdateOperation::DeleteInsertWhere {
            delete_triples,
            insert_triples,
            where_triples,
            ..
        } => {
            assert_eq!(delete_triples.len(), 1);
            assert_eq!(insert_triples.len(), 1);
            assert_eq!(where_triples.len(), 1);
            assert_eq!(delete_triples[0].predicate, "<http://ex.org/old>");
            assert_eq!(insert_triples[0].predicate, "<http://ex.org/new>");
        }
        other => panic!("expected DeleteInsertWhere, got {:?}", other),
    }
}

#[test]
fn test_delete_where_only() {
    let input = r#"DELETE { ?s <http://ex.org/p> ?o }
    WHERE { ?s <http://ex.org/p> ?o }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::DeleteInsertWhere { insert_triples, .. } = &req.operations[0] {
        assert!(insert_triples.is_empty());
    }
}

#[test]
fn test_delete_insert_where_with_variables() {
    let input = r#"DELETE { ?person <http://ex.org/age> ?old }
    INSERT { ?person <http://ex.org/age> ?new }
    WHERE { ?person <http://ex.org/age> ?old }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::DeleteInsertWhere {
        delete_triples,
        insert_triples,
        ..
    } = &req.operations[0]
    {
        assert_eq!(delete_triples[0].subject, "?person");
        assert_eq!(insert_triples[0].object, "?new");
    }
}

// ── LOAD ────────────────────────────────────────────────────────────────────

#[test]
fn test_load_basic() {
    let input = r#"LOAD <http://example.org/data.ttl>"#;
    let req = parse_update(input).expect("should parse");
    match &req.operations[0] {
        UpdateOperation::Load {
            source_uri,
            target_graph,
            silent,
        } => {
            assert_eq!(source_uri, "http://example.org/data.ttl");
            assert!(target_graph.is_none());
            assert!(!silent);
        }
        other => panic!("expected Load, got {:?}", other),
    }
}

#[test]
fn test_load_into_graph() {
    let input = r#"LOAD <http://example.org/data.ttl> INTO GRAPH <http://example.org/g>"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Load { target_graph, .. } = &req.operations[0] {
        assert_eq!(target_graph.as_deref(), Some("http://example.org/g"));
    }
}

#[test]
fn test_load_silent() {
    let input = r#"LOAD SILENT <http://example.org/data.ttl>"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Load { silent, .. } = &req.operations[0] {
        assert!(*silent);
    }
}

#[test]
fn test_load_silent_into_graph() {
    let input = r#"LOAD SILENT <http://example.org/data.ttl> INTO GRAPH <http://example.org/g>"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Load {
        silent,
        target_graph,
        ..
    } = &req.operations[0]
    {
        assert!(*silent);
        assert_eq!(target_graph.as_deref(), Some("http://example.org/g"));
    }
}

// ── CLEAR ───────────────────────────────────────────────────────────────────

#[test]
fn test_clear_all() {
    let input = "CLEAR ALL";
    let req = parse_update(input).expect("should parse");
    match &req.operations[0] {
        UpdateOperation::Clear { target, silent } => {
            assert_eq!(*target, GraphTarget::All);
            assert!(!silent);
        }
        other => panic!("expected Clear, got {:?}", other),
    }
}

#[test]
fn test_clear_default() {
    let input = "CLEAR DEFAULT";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Clear { target, .. } = &req.operations[0] {
        assert_eq!(*target, GraphTarget::Default);
    }
}

#[test]
fn test_clear_named() {
    let input = "CLEAR NAMED";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Clear { target, .. } = &req.operations[0] {
        assert_eq!(*target, GraphTarget::Named);
    }
}

#[test]
fn test_clear_graph() {
    let input = "CLEAR GRAPH <http://example.org/g>";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Clear { target, .. } = &req.operations[0] {
        assert_eq!(
            *target,
            GraphTarget::Graph("http://example.org/g".to_string())
        );
    }
}

#[test]
fn test_clear_silent() {
    let input = "CLEAR SILENT ALL";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Clear { silent, .. } = &req.operations[0] {
        assert!(*silent);
    }
}

// ── DROP ────────────────────────────────────────────────────────────────────

#[test]
fn test_drop_all() {
    let input = "DROP ALL";
    let req = parse_update(input).expect("should parse");
    match &req.operations[0] {
        UpdateOperation::Drop { target, silent } => {
            assert_eq!(*target, GraphTarget::All);
            assert!(!silent);
        }
        other => panic!("expected Drop, got {:?}", other),
    }
}

#[test]
fn test_drop_graph() {
    let input = "DROP GRAPH <http://ex.org/g>";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Drop { target, .. } = &req.operations[0] {
        assert_eq!(*target, GraphTarget::Graph("http://ex.org/g".to_string()));
    }
}

#[test]
fn test_drop_silent() {
    let input = "DROP SILENT DEFAULT";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Drop { target, silent } = &req.operations[0] {
        assert_eq!(*target, GraphTarget::Default);
        assert!(*silent);
    }
}

// ── CREATE GRAPH ────────────────────────────────────────────────────────────

#[test]
fn test_create_graph() {
    let input = "CREATE GRAPH <http://example.org/new-graph>";
    let req = parse_update(input).expect("should parse");
    match &req.operations[0] {
        UpdateOperation::CreateGraph { graph_iri, silent } => {
            assert_eq!(graph_iri, "http://example.org/new-graph");
            assert!(!silent);
        }
        other => panic!("expected CreateGraph, got {:?}", other),
    }
}

#[test]
fn test_create_graph_silent() {
    let input = "CREATE SILENT GRAPH <http://example.org/g>";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::CreateGraph { silent, .. } = &req.operations[0] {
        assert!(*silent);
    }
}

// ── COPY ────────────────────────────────────────────────────────────────────

#[test]
fn test_copy_default_to_graph() {
    let input = "COPY DEFAULT TO GRAPH <http://ex.org/backup>";
    let req = parse_update(input).expect("should parse");
    match &req.operations[0] {
        UpdateOperation::Copy {
            source,
            destination,
            silent,
        } => {
            assert_eq!(*source, GraphTarget::Default);
            assert_eq!(
                *destination,
                GraphTarget::Graph("http://ex.org/backup".to_string())
            );
            assert!(!silent);
        }
        other => panic!("expected Copy, got {:?}", other),
    }
}

#[test]
fn test_copy_graph_to_default() {
    let input = "COPY GRAPH <http://ex.org/src> TO DEFAULT";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Copy {
        source,
        destination,
        ..
    } = &req.operations[0]
    {
        assert_eq!(*source, GraphTarget::Graph("http://ex.org/src".to_string()));
        assert_eq!(*destination, GraphTarget::Default);
    }
}

#[test]
fn test_copy_silent() {
    let input = "COPY SILENT DEFAULT TO GRAPH <http://ex.org/dst>";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Copy { silent, .. } = &req.operations[0] {
        assert!(*silent);
    }
}

// ── MOVE ────────────────────────────────────────────────────────────────────

#[test]
fn test_move_graph_to_graph() {
    let input = "MOVE GRAPH <http://ex.org/a> TO GRAPH <http://ex.org/b>";
    let req = parse_update(input).expect("should parse");
    match &req.operations[0] {
        UpdateOperation::Move {
            source,
            destination,
            silent,
        } => {
            assert_eq!(*source, GraphTarget::Graph("http://ex.org/a".to_string()));
            assert_eq!(
                *destination,
                GraphTarget::Graph("http://ex.org/b".to_string())
            );
            assert!(!silent);
        }
        other => panic!("expected Move, got {:?}", other),
    }
}

#[test]
fn test_move_silent() {
    let input = "MOVE SILENT GRAPH <http://ex.org/a> TO DEFAULT";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Move { silent, .. } = &req.operations[0] {
        assert!(*silent);
    }
}

// ── ADD ─────────────────────────────────────────────────────────────────────

#[test]
fn test_add_default_to_graph() {
    let input = "ADD DEFAULT TO GRAPH <http://ex.org/combined>";
    let req = parse_update(input).expect("should parse");
    match &req.operations[0] {
        UpdateOperation::Add {
            source,
            destination,
            silent,
        } => {
            assert_eq!(*source, GraphTarget::Default);
            assert_eq!(
                *destination,
                GraphTarget::Graph("http://ex.org/combined".to_string())
            );
            assert!(!silent);
        }
        other => panic!("expected Add, got {:?}", other),
    }
}

#[test]
fn test_add_silent() {
    let input = "ADD SILENT GRAPH <http://ex.org/src> TO DEFAULT";
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::Add { silent, .. } = &req.operations[0] {
        assert!(*silent);
    }
}

// ── Multiple operations ─────────────────────────────────────────────────────

#[test]
fn test_multiple_operations_semicolon_separated() {
    let input = r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> <http://ex.org/o> } ;
    CLEAR ALL"#;
    let req = parse_update(input).expect("should parse");
    assert_eq!(req.operations.len(), 2);
    assert_eq!(req.operations[0].kind_label(), "INSERT DATA");
    assert_eq!(req.operations[1].kind_label(), "CLEAR");
}

#[test]
fn test_three_operations() {
    let input = r#"
        CREATE GRAPH <http://ex.org/g> ;
        LOAD <http://ex.org/data.ttl> INTO GRAPH <http://ex.org/g> ;
        DROP GRAPH <http://ex.org/old>
    "#;
    let req = parse_update(input).expect("should parse");
    assert_eq!(req.operations.len(), 3);
    assert_eq!(req.operations[0].kind_label(), "CREATE GRAPH");
    assert_eq!(req.operations[1].kind_label(), "LOAD");
    assert_eq!(req.operations[2].kind_label(), "DROP");
}

// ── PREFIX handling ─────────────────────────────────────────────────────────

#[test]
fn test_multiple_prefixes() {
    let input = r#"
        PREFIX ex: <http://example.org/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        INSERT DATA { ex:alice foaf:name "Alice" }
    "#;
    let req = parse_update(input).expect("should parse");
    assert_eq!(req.prefixes.len(), 2);
    assert_eq!(
        req.prefixes.get("ex"),
        Some(&"http://example.org/".to_string())
    );
    assert_eq!(
        req.prefixes.get("foaf"),
        Some(&"http://xmlns.com/foaf/0.1/".to_string())
    );
}

#[test]
fn test_prefix_expansion_in_triples() {
    let input = r#"
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        INSERT DATA { <http://ex.org/alice> foaf:knows <http://ex.org/bob> }
    "#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert_eq!(triples[0].predicate, "<http://xmlns.com/foaf/0.1/knows>");
    }
}

// ── Error cases ─────────────────────────────────────────────────────────────

#[test]
fn test_error_empty_input() {
    let result = parse_update("");
    assert!(result.is_err());
    let err = result.expect_err("should be error");
    assert!(err.message.contains("empty"));
}

#[test]
fn test_error_unknown_keyword() {
    let result = parse_update("FROBNICATE ALL");
    assert!(result.is_err());
    let err = result.expect_err("should be error");
    assert!(err.message.contains("expected update keyword"));
    assert!(err.position == 0);
}

#[test]
fn test_error_missing_brace() {
    let result = parse_update("INSERT DATA <http://ex.org/s> <http://ex.org/p> <http://ex.org/o>");
    assert!(result.is_err());
}

#[test]
fn test_error_unterminated_brace() {
    let result =
        parse_update("INSERT DATA { <http://ex.org/s> <http://ex.org/p> <http://ex.org/o>");
    assert!(result.is_err());
    let err = result.expect_err("should be error");
    assert!(err.message.contains("'}'"));
}

#[test]
fn test_error_position_tracking() {
    let result = parse_update("CLEAR BADTARGET");
    assert!(result.is_err());
    let err = result.expect_err("should be error");
    assert!(err.position > 0);
    assert!(err.line.is_some());
    assert!(err.column.is_some());
}

#[test]
fn test_error_unterminated_iri() {
    let result = parse_update("LOAD <http://example.org/unterminated");
    assert!(result.is_err());
    let err = result.expect_err("should be error");
    assert!(err.message.contains("unterminated"));
}

// ── Special terms ───────────────────────────────────────────────────────────

#[test]
fn test_rdf_type_shorthand() {
    let input = r#"INSERT DATA { <http://ex.org/alice> a <http://ex.org/Person> }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert_eq!(
            triples[0].predicate,
            "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
        );
    }
}

#[test]
fn test_blank_node() {
    let input = r#"INSERT DATA { _:b1 <http://ex.org/p> <http://ex.org/o> }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert_eq!(triples[0].subject, "_:b1");
    }
}

#[test]
fn test_numeric_literal_integer() {
    let input = r#"INSERT DATA { <http://ex.org/s> <http://ex.org/age> 42 }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert!(triples[0].object.contains("42"));
        assert!(triples[0].object.contains("integer"));
    }
}

#[test]
fn test_numeric_literal_decimal() {
    let input = r#"INSERT DATA { <http://ex.org/s> <http://ex.org/weight> 3.14 }"#;
    let req = parse_update(input).expect("should parse");
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert!(triples[0].object.contains("3.14"));
    }
}

// ── GraphTarget display ─────────────────────────────────────────────────────

#[test]
fn test_graph_target_display() {
    assert_eq!(GraphTarget::Default.to_string(), "DEFAULT");
    assert_eq!(GraphTarget::Named.to_string(), "NAMED");
    assert_eq!(GraphTarget::All.to_string(), "ALL");
    assert_eq!(
        GraphTarget::Graph("http://ex.org/g".to_string()).to_string(),
        "GRAPH <http://ex.org/g>"
    );
}

// ── UpdateParseError display ────────────────────────────────────────────────

#[test]
fn test_error_display_with_location() {
    let err = UpdateParseError::new("bad token", 10).with_location(2, 5);
    let msg = err.to_string();
    assert!(msg.contains("2:5"));
    assert!(msg.contains("bad token"));
}

#[test]
fn test_error_display_without_location() {
    let err = UpdateParseError::new("bad token", 10);
    let msg = err.to_string();
    assert!(msg.contains("byte 10"));
    assert!(msg.contains("bad token"));
}

// ── Kind label ──────────────────────────────────────────────────────────────

#[test]
fn test_kind_labels() {
    assert_eq!(
        UpdateOperation::InsertData {
            triples: vec![],
            graph: None
        }
        .kind_label(),
        "INSERT DATA"
    );
    assert_eq!(
        UpdateOperation::DeleteData {
            triples: vec![],
            graph: None
        }
        .kind_label(),
        "DELETE DATA"
    );
    assert_eq!(
        UpdateOperation::Load {
            source_uri: String::new(),
            target_graph: None,
            silent: false
        }
        .kind_label(),
        "LOAD"
    );
    assert_eq!(
        UpdateOperation::Clear {
            target: GraphTarget::All,
            silent: false
        }
        .kind_label(),
        "CLEAR"
    );
    assert_eq!(
        UpdateOperation::Drop {
            target: GraphTarget::All,
            silent: false
        }
        .kind_label(),
        "DROP"
    );
    assert_eq!(
        UpdateOperation::CreateGraph {
            graph_iri: String::new(),
            silent: false
        }
        .kind_label(),
        "CREATE GRAPH"
    );
    assert_eq!(
        UpdateOperation::Copy {
            source: GraphTarget::Default,
            destination: GraphTarget::Default,
            silent: false
        }
        .kind_label(),
        "COPY"
    );
    assert_eq!(
        UpdateOperation::Move {
            source: GraphTarget::Default,
            destination: GraphTarget::Default,
            silent: false
        }
        .kind_label(),
        "MOVE"
    );
    assert_eq!(
        UpdateOperation::Add {
            source: GraphTarget::Default,
            destination: GraphTarget::Default,
            silent: false
        }
        .kind_label(),
        "ADD"
    );
}

// ── Comment handling ────────────────────────────────────────────────────────

#[test]
fn test_comments_are_skipped() {
    let input = r#"
        # This is a comment
        INSERT DATA {
            # Another comment
            <http://ex.org/s> <http://ex.org/p> <http://ex.org/o>
        }
    "#;
    let req = parse_update(input).expect("should parse");
    assert_eq!(req.operations.len(), 1);
}

// ── with_prefixes constructor ───────────────────────────────────────────────

#[test]
fn test_with_prefixes_constructor() {
    let mut prefixes = HashMap::new();
    prefixes.insert("ex".to_string(), "http://example.org/".to_string());

    let input = "INSERT DATA { ex:s ex:p ex:o }";
    let result = parse_update_with_prefixes(input, prefixes);
    let req = result.expect("should parse");
    if let UpdateOperation::InsertData { triples, .. } = &req.operations[0] {
        assert_eq!(triples[0].subject, "<http://example.org/s>");
    }
}

// ── Triple pattern construction ─────────────────────────────────────────────

#[test]
fn test_triple_pattern_new() {
    let tp = TriplePattern::new("s", "p", "o");
    assert_eq!(tp.subject, "s");
    assert_eq!(tp.predicate, "p");
    assert_eq!(tp.object, "o");
}

// ── Case insensitivity ──────────────────────────────────────────────────────

#[test]
fn test_keywords_case_insensitive() {
    let input = "clear all";
    let req = parse_update(input).expect("should parse");
    assert_eq!(req.operations[0].kind_label(), "CLEAR");
}

#[test]
fn test_mixed_case_keywords() {
    let input = "Insert Data { <http://ex.org/s> <http://ex.org/p> <http://ex.org/o> }";
    let req = parse_update(input).expect("should parse");
    assert_eq!(req.operations[0].kind_label(), "INSERT DATA");
}

// ── line_col helper ─────────────────────────────────────────────────────────

#[test]
fn test_line_col_first_line() {
    let (ln, col) = line_col("hello world", 6);
    assert_eq!(ln, 1);
    assert_eq!(col, 7);
}

#[test]
fn test_line_col_second_line() {
    let (ln, col) = line_col("hello\nworld", 6);
    assert_eq!(ln, 2);
    assert_eq!(col, 1);
}

#[test]
fn test_line_col_empty() {
    let (ln, col) = line_col("", 0);
    assert_eq!(ln, 1);
    assert_eq!(col, 1);
}

// ── UpdateParser default ────────────────────────────────────────────────────

#[test]
fn test_parser_default_trait() {
    let parser = UpdateParser::default();
    assert!(parser.prefixes().is_empty());
}
