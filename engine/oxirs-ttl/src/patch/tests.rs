use super::*;
use crate::writer::TermType;

// Helper to build a simple triple
fn triple(s: &str, p: &str, o: &str) -> PatchTriple {
    PatchTriple::new(PatchTerm::iri(s), PatchTerm::iri(p), PatchTerm::iri(o))
}

fn triple_lit(s: &str, p: &str, o: &str) -> PatchTriple {
    PatchTriple::new(PatchTerm::iri(s), PatchTerm::iri(p), PatchTerm::literal(o))
}

// ── Header parsing ────────────────────────────────────────────────────

#[test]
fn test_parse_header_id() {
    let patch = PatchParser::parse("H id <urn:uuid:1234>\n").expect("should succeed");
    assert_eq!(patch.headers.len(), 1);
    assert_eq!(patch.id(), Some("urn:uuid:1234"));
}

#[test]
fn test_parse_header_prev() {
    let patch = PatchParser::parse("H prev <urn:uuid:abcd>\n").expect("should succeed");
    assert_eq!(patch.previous(), Some("urn:uuid:abcd"));
}

#[test]
fn test_parse_header_version() {
    let patch = PatchParser::parse("H version 1\n").expect("should succeed");
    matches!(&patch.headers[0], PatchHeader::Version(v) if v == "1");
}

#[test]
fn test_parse_header_unknown() {
    let patch = PatchParser::parse("H custom myval\n").expect("should succeed");
    assert!(matches!(&patch.headers[0], PatchHeader::Unknown { key, .. } if key == "custom"));
}

#[test]
fn test_parse_multiple_headers() {
    let input = "H id <urn:1>\nH prev <urn:0>\nH version 2\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    assert_eq!(patch.headers.len(), 3);
}

// ── Transaction control ───────────────────────────────────────────────

#[test]
fn test_parse_tx_tc() {
    let patch = PatchParser::parse("TX\nTC\n").expect("should succeed");
    assert_eq!(patch.changes.len(), 2);
    assert!(matches!(patch.changes[0], PatchChange::TransactionBegin));
    assert!(matches!(patch.changes[1], PatchChange::TransactionCommit));
}

#[test]
fn test_parse_ta() {
    let patch = PatchParser::parse("TX\nTA\n").expect("should succeed");
    assert!(matches!(patch.changes[1], PatchChange::TransactionAbort));
}

#[test]
fn test_transaction_control_predicates() {
    assert!(PatchChange::TransactionBegin.is_transaction_control());
    assert!(PatchChange::TransactionCommit.is_transaction_control());
    assert!(PatchChange::TransactionAbort.is_transaction_control());
}

// ── Prefix parsing ────────────────────────────────────────────────────

#[test]
fn test_parse_prefix_add() {
    let patch = PatchParser::parse("PA ex <http://example.org/>\n").expect("should succeed");
    assert_eq!(patch.changes.len(), 1);
    match &patch.changes[0] {
        PatchChange::AddPrefix { prefix, iri } => {
            assert_eq!(prefix, "ex");
            assert_eq!(iri, "http://example.org/");
        }
        _ => panic!("unexpected change type"),
    }
}

#[test]
fn test_parse_prefix_delete() {
    let patch = PatchParser::parse("PD ex <http://example.org/>\n").expect("should succeed");
    assert!(
        matches!(&patch.changes[0], PatchChange::DeletePrefix { prefix, .. } if prefix == "ex")
    );
}

#[test]
fn test_prefix_resolution_in_triple() {
    let input = "PA ex <http://example.org/>\nA ex:s ex:p ex:o .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    assert_eq!(patch.changes.len(), 2);
    if let PatchChange::AddTriple(t) = &patch.changes[1] {
        assert_eq!(t.subject.value(), "http://example.org/s");
    } else {
        panic!("expected AddTriple");
    }
}

// ── Triple operations ─────────────────────────────────────────────────

#[test]
fn test_parse_add_triple() {
    let input = "A <http://s> <http://p> <http://o> .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    assert!(matches!(&patch.changes[0], PatchChange::AddTriple(_)));
}

#[test]
fn test_parse_delete_triple() {
    let input = "D <http://s> <http://p> <http://o> .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    assert!(matches!(&patch.changes[0], PatchChange::DeleteTriple(_)));
}

#[test]
fn test_parse_triple_with_literal() {
    let input = "A <http://s> <http://p> \"hello\" .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    if let PatchChange::AddTriple(t) = &patch.changes[0] {
        assert!(
            t.object.0.term_type
                == TermType::Literal {
                    datatype: None,
                    lang: None
                }
        );
        assert_eq!(t.object.value(), "hello");
    } else {
        panic!("expected AddTriple");
    }
}

#[test]
fn test_parse_literal_with_language() {
    let input = "A <http://s> <http://p> \"hello\"@en .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    if let PatchChange::AddTriple(t) = &patch.changes[0] {
        assert!(matches!(
            &t.object.0.term_type,
            TermType::Literal { lang: Some(l), .. } if l == "en"
        ));
    } else {
        panic!("expected AddTriple");
    }
}

#[test]
fn test_parse_literal_with_datatype() {
    let input = "A <http://s> <http://p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    if let PatchChange::AddTriple(t) = &patch.changes[0] {
        assert!(matches!(
            &t.object.0.term_type,
            TermType::Literal { datatype: Some(dt), .. }
            if dt == "http://www.w3.org/2001/XMLSchema#integer"
        ));
    } else {
        panic!("expected AddTriple");
    }
}

#[test]
fn test_parse_triple_blank_node() {
    let input = "A _:b0 <http://p> <http://o> .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    if let PatchChange::AddTriple(t) = &patch.changes[0] {
        assert!(t.subject.is_blank_node());
        assert_eq!(t.subject.value(), "b0");
    } else {
        panic!("expected AddTriple");
    }
}

// ── Quad operations ───────────────────────────────────────────────────

#[test]
fn test_parse_add_quad() {
    let input = "A <http://s> <http://p> <http://o> <http://g> .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    assert!(matches!(&patch.changes[0], PatchChange::AddQuad(_)));
}

#[test]
fn test_parse_delete_quad() {
    let input = "D <http://s> <http://p> <http://o> <http://g> .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    assert!(matches!(&patch.changes[0], PatchChange::DeleteQuad(_)));
}

#[test]
fn test_quad_graph_term() {
    let input = "A <http://s> <http://p> <http://o> <http://graph1> .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    if let PatchChange::AddQuad(q) = &patch.changes[0] {
        assert_eq!(q.graph.value(), "http://graph1");
    } else {
        panic!("expected AddQuad");
    }
}

// ── Serialization ─────────────────────────────────────────────────────

#[test]
fn test_serialize_header() {
    let patch = RdfPatch {
        headers: vec![PatchHeader::Id("urn:1".to_string())],
        changes: vec![],
    };
    let s = PatchSerializer::serialize(&patch);
    assert!(s.contains("H id urn:1"));
}

#[test]
fn test_serialize_add_triple() {
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![PatchChange::AddTriple(triple(
            "http://s", "http://p", "http://o",
        ))],
    };
    let s = PatchSerializer::serialize(&patch);
    assert!(s.contains("A <http://s> <http://p> <http://o>"));
}

#[test]
fn test_serialize_delete_triple() {
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![PatchChange::DeleteTriple(triple(
            "http://s", "http://p", "http://o",
        ))],
    };
    let s = PatchSerializer::serialize(&patch);
    assert!(s.starts_with("D "));
}

#[test]
fn test_serialize_prefix_add() {
    let change = PatchChange::AddPrefix {
        prefix: "ex".to_string(),
        iri: "http://example.org/".to_string(),
    };
    let s = PatchSerializer::serialize_change(&change);
    assert_eq!(s, "PA ex <http://example.org/>");
}

#[test]
fn test_serialize_transaction_control() {
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![
            PatchChange::TransactionBegin,
            PatchChange::TransactionCommit,
        ],
    };
    let s = PatchSerializer::serialize(&patch);
    assert!(s.contains("TX"));
    assert!(s.contains("TC"));
}

#[test]
fn test_serialize_literal() {
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![PatchChange::AddTriple(triple_lit(
            "http://s", "http://p", "hello",
        ))],
    };
    let s = PatchSerializer::serialize(&patch);
    assert!(s.contains("\"hello\""));
}

// ── Apply patch ───────────────────────────────────────────────────────

#[test]
fn test_apply_add_triple() {
    let mut graph = Graph::new();
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![PatchChange::AddTriple(triple(
            "http://s", "http://p", "http://o",
        ))],
    };
    let stats = apply_patch(&mut graph, &patch).expect("should succeed");
    assert_eq!(stats.triples_added, 1);
    assert_eq!(graph.len(), 1);
}

#[test]
fn test_apply_delete_triple() {
    let mut graph = Graph::new();
    let t = triple("http://s", "http://p", "http://o");
    graph.add_triple(t.clone());
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![PatchChange::DeleteTriple(t)],
    };
    let stats = apply_patch(&mut graph, &patch).expect("should succeed");
    assert_eq!(stats.triples_deleted, 1);
    assert_eq!(graph.len(), 0);
}

#[test]
fn test_apply_idempotent_add() {
    let mut graph = Graph::new();
    let t = triple("http://s", "http://p", "http://o");
    graph.add_triple(t.clone());
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![PatchChange::AddTriple(t)],
    };
    let stats = apply_patch(&mut graph, &patch).expect("should succeed");
    // Should not double-count
    assert_eq!(stats.triples_added, 0);
    assert_eq!(graph.len(), 1);
}

#[test]
fn test_apply_prefix_add() {
    let mut graph = Graph::new();
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![PatchChange::AddPrefix {
            prefix: "ex".to_string(),
            iri: "http://example.org/".to_string(),
        }],
    };
    let stats = apply_patch(&mut graph, &patch).expect("should succeed");
    assert_eq!(stats.prefixes_added, 1);
    assert_eq!(
        graph.prefixes.get("ex").map(String::as_str),
        Some("http://example.org/")
    );
}

#[test]
fn test_apply_transaction_commit() {
    let mut graph = Graph::new();
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![
            PatchChange::TransactionBegin,
            PatchChange::AddTriple(triple("http://s", "http://p", "http://o")),
            PatchChange::TransactionCommit,
        ],
    };
    let stats = apply_patch(&mut graph, &patch).expect("should succeed");
    assert_eq!(stats.triples_added, 1);
    assert_eq!(stats.transactions, 1);
    assert_eq!(graph.len(), 1);
}

#[test]
fn test_apply_transaction_abort() {
    let mut graph = Graph::new();
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![
            PatchChange::TransactionBegin,
            PatchChange::AddTriple(triple("http://s", "http://p", "http://o")),
            PatchChange::TransactionAbort,
        ],
    };
    let stats = apply_patch(&mut graph, &patch).expect("should succeed");
    assert_eq!(stats.aborts, 1);
    // Graph must remain empty — abort rolls back staged changes
    assert_eq!(graph.len(), 0);
}

#[test]
fn test_apply_multiple_changes() {
    let mut graph = Graph::new();
    let t1 = triple("http://a", "http://p", "http://x");
    let t2 = triple("http://b", "http://p", "http://y");
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![
            PatchChange::AddTriple(t1.clone()),
            PatchChange::AddTriple(t2.clone()),
            PatchChange::DeleteTriple(t1),
        ],
    };
    let stats = apply_patch(&mut graph, &patch).expect("should succeed");
    assert_eq!(stats.triples_added, 2);
    assert_eq!(stats.triples_deleted, 1);
    assert_eq!(graph.len(), 1);
}

// ── Graph diff ────────────────────────────────────────────────────────

#[test]
fn test_diff_to_patch_add() {
    let old = Graph::new();
    let mut new_graph = Graph::new();
    new_graph.add_triple(triple("http://s", "http://p", "http://o"));
    let patch = diff_to_patch(&old, &new_graph);
    assert_eq!(patch.add_count(), 1);
    assert_eq!(patch.delete_count(), 0);
}

#[test]
fn test_diff_to_patch_delete() {
    let mut old = Graph::new();
    old.add_triple(triple("http://s", "http://p", "http://o"));
    let new_graph = Graph::new();
    let patch = diff_to_patch(&old, &new_graph);
    assert_eq!(patch.add_count(), 0);
    assert_eq!(patch.delete_count(), 1);
}

#[test]
fn test_diff_to_patch_no_change() {
    let mut old = Graph::new();
    old.add_triple(triple("http://s", "http://p", "http://o"));
    let new_graph = old.clone();
    let patch = diff_to_patch(&old, &new_graph);
    assert!(patch.changes.is_empty());
}

#[test]
fn test_diff_to_patch_prefix_added() {
    let old = Graph::new();
    let mut new_graph = Graph::new();
    new_graph
        .prefixes
        .insert("ex".to_string(), "http://example.org/".to_string());
    let patch = diff_to_patch(&old, &new_graph);
    assert!(patch
        .changes
        .iter()
        .any(|c| matches!(c, PatchChange::AddPrefix { .. })));
}

#[test]
fn test_diff_to_patch_prefix_removed() {
    let mut old = Graph::new();
    old.prefixes
        .insert("ex".to_string(), "http://example.org/".to_string());
    let new_graph = Graph::new();
    let patch = diff_to_patch(&old, &new_graph);
    assert!(patch
        .changes
        .iter()
        .any(|c| matches!(c, PatchChange::DeletePrefix { .. })));
}

// ── Round-trip ────────────────────────────────────────────────────────

#[test]
fn test_round_trip_simple() {
    let input = "H id <urn:1>\nA <http://s> <http://p> <http://o> .\nD <http://s> <http://p> <http://old> .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    let serialized = PatchSerializer::serialize(&patch);
    let reparsed = PatchParser::parse(&serialized).expect("should succeed");
    assert_eq!(reparsed.headers.len(), patch.headers.len());
    assert_eq!(reparsed.changes.len(), patch.changes.len());
}

#[test]
fn test_round_trip_with_prefixes() {
    let input = "PA ex <http://example.org/>\nA ex:s ex:p ex:o .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    let serialized = PatchSerializer::serialize(&patch);
    // After serialisation ex:s becomes <http://example.org/s>
    assert!(serialized.contains("<http://example.org/s>"));
    // Re-parse the serialised form
    let reparsed = PatchParser::parse(&serialized).expect("should succeed");
    assert_eq!(reparsed.changes.len(), 2);
}

#[test]
fn test_round_trip_transaction() {
    let input = "TX\nA <http://s> <http://p> <http://o> .\nTC\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    let serialized = PatchSerializer::serialize(&patch);
    let reparsed = PatchParser::parse(&serialized).expect("should succeed");
    assert_eq!(reparsed.changes.len(), 3);
    assert!(matches!(reparsed.changes[0], PatchChange::TransactionBegin));
    assert!(matches!(
        reparsed.changes[2],
        PatchChange::TransactionCommit
    ));
}

#[test]
fn test_round_trip_with_blank_nodes() {
    let input = "A _:b0 <http://p> <http://o> .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    let s = PatchSerializer::serialize(&patch);
    let reparsed = PatchParser::parse(&s).expect("should succeed");
    if let PatchChange::AddTriple(t) = &reparsed.changes[0] {
        assert!(t.subject.is_blank_node());
    } else {
        panic!("expected AddTriple");
    }
}

#[test]
fn test_round_trip_literal_with_lang() {
    let input = "A <http://s> <http://p> \"bonjour\"@fr .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    let s = PatchSerializer::serialize(&patch);
    let reparsed = PatchParser::parse(&s).expect("should succeed");
    if let PatchChange::AddTriple(t) = &reparsed.changes[0] {
        assert!(matches!(
            &t.object.0.term_type,
            TermType::Literal { lang: Some(l), .. } if l == "fr"
        ));
    } else {
        panic!("expected AddTriple");
    }
}

#[test]
fn test_round_trip_literal_with_datatype() {
    let dt = "http://www.w3.org/2001/XMLSchema#integer";
    let input = format!("A <http://s> <http://p> \"42\"^^<{dt}> .\n");
    let patch = PatchParser::parse(&input).expect("should succeed");
    let s = PatchSerializer::serialize(&patch);
    let reparsed = PatchParser::parse(&s).expect("should succeed");
    if let PatchChange::AddTriple(t) = &reparsed.changes[0] {
        assert!(matches!(
            &t.object.0.term_type,
            TermType::Literal { datatype: Some(d), .. } if d == dt
        ));
    } else {
        panic!("expected AddTriple");
    }
}

// ── Streaming parser ──────────────────────────────────────────────────

#[test]
fn test_streaming_parser_basic() {
    let input = "TX\nA <http://s> <http://p> <http://o> .\nTC\n";
    let changes: Vec<_> = PatchParser::parse_streaming(input.as_bytes()).collect();
    assert_eq!(changes.len(), 3);
    assert!(changes[0]
        .as_ref()
        .map(|c| matches!(c, PatchChange::TransactionBegin))
        .unwrap_or(false));
}

#[test]
fn test_streaming_skips_headers() {
    let input = "H id <urn:1>\nA <http://s> <http://p> <http://o> .\n";
    let changes: Vec<_> = PatchParser::parse_streaming(input.as_bytes()).collect();
    // Header is skipped in streaming mode
    assert_eq!(changes.len(), 1);
}

#[test]
fn test_streaming_parser_prefixes() {
    let input = "PA ex <http://example.org/>\nA ex:s ex:p ex:o .\n";
    let changes: Vec<_> = PatchParser::parse_streaming(input.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .expect("should succeed");
    assert_eq!(changes.len(), 2);
}

#[test]
fn test_streaming_parser_multiple_batches() {
    let input = "A <http://s1> <http://p> <http://o1> .\nA <http://s2> <http://p> <http://o2> .\nD <http://s1> <http://p> <http://o1> .\n";
    let changes: Vec<_> = PatchParser::parse_streaming(input.as_bytes())
        .collect::<Result<Vec<_>, _>>()
        .expect("should succeed");
    assert_eq!(changes.len(), 3);
}

// ── Edge cases ────────────────────────────────────────────────────────

#[test]
fn test_empty_patch() {
    let patch = PatchParser::parse("").expect("should succeed");
    assert!(patch.is_empty());
}

#[test]
fn test_comments_ignored() {
    let input = "# This is a comment\nA <http://s> <http://p> <http://o> .\n# Another comment\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    assert_eq!(patch.changes.len(), 1);
}

#[test]
fn test_blank_lines_ignored() {
    let input = "\n\nA <http://s> <http://p> <http://o> .\n\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    assert_eq!(patch.changes.len(), 1);
}

#[test]
fn test_error_unknown_line() {
    let result = PatchParser::parse("UNKNOWN_CMD <http://x>\n");
    assert!(result.is_err());
}

#[test]
fn test_error_unterminated_iri() {
    let result = PatchParser::parse("A <http://s <http://p> <http://o> .\n");
    assert!(result.is_err());
}

#[test]
fn test_patch_change_predicates() {
    assert!(PatchChange::AddTriple(triple("h://s", "h://p", "h://o")).is_add());
    assert!(PatchChange::DeleteTriple(triple("h://s", "h://p", "h://o")).is_delete());
    assert!(!PatchChange::AddTriple(triple("h://s", "h://p", "h://o")).is_delete());
    assert!(!PatchChange::DeleteTriple(triple("h://s", "h://p", "h://o")).is_add());
}

#[test]
fn test_graph_contains() {
    let mut g = Graph::new();
    let t = triple("http://s", "http://p", "http://o");
    assert!(!g.contains(&t));
    g.add_triple(t.clone());
    assert!(g.contains(&t));
    g.remove_triple(&t);
    assert!(!g.contains(&t));
}

#[test]
fn test_graph_len() {
    let mut g = Graph::new();
    assert_eq!(g.len(), 0);
    assert!(g.is_empty());
    g.add_triple(triple("http://s", "http://p", "http://o"));
    assert_eq!(g.len(), 1);
    assert!(!g.is_empty());
}

#[test]
fn test_apply_patch_from_parsed_text() {
    let input = "PA ex <http://example.org/>\nTX\nA ex:alice <http://type> <http://Person> .\nTC\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    let mut graph = Graph::new();
    let stats = apply_patch(&mut graph, &patch).expect("should succeed");
    assert_eq!(stats.triples_added, 1);
    assert_eq!(stats.transactions, 1);
}

#[test]
fn test_patch_stats_default() {
    let stats = PatchStats::default();
    assert_eq!(stats.triples_added, 0);
    assert_eq!(stats.triples_deleted, 0);
    assert_eq!(stats.prefixes_added, 0);
    assert_eq!(stats.prefixes_deleted, 0);
    assert_eq!(stats.transactions, 0);
    assert_eq!(stats.aborts, 0);
}

#[test]
fn test_patch_header_key_value() {
    let h = PatchHeader::Id("urn:test".to_string());
    assert_eq!(h.key(), "id");
    assert_eq!(h.value(), "urn:test");
}

#[test]
fn test_diff_then_apply_round_trip() {
    let mut old = Graph::new();
    old.add_triple(triple("http://s", "http://p", "http://o1"));
    old.add_triple(triple("http://s", "http://p", "http://o2"));

    let mut new_graph = Graph::new();
    new_graph.add_triple(triple("http://s", "http://p", "http://o2"));
    new_graph.add_triple(triple("http://s", "http://p", "http://o3"));

    let patch = diff_to_patch(&old, &new_graph);
    // Apply patch to old to get new
    let mut result = old.clone();
    apply_patch(&mut result, &patch).expect("should succeed");

    assert_eq!(result.len(), new_graph.len());
    for t in new_graph.iter() {
        assert!(result.contains(t), "missing triple: {t}");
    }
}

#[test]
fn test_serialize_then_parse_complete_patch() {
    let mut patch = RdfPatch::new();
    patch
        .headers
        .push(PatchHeader::Id("urn:test:42".to_string()));
    patch
        .headers
        .push(PatchHeader::Previous("urn:test:41".to_string()));
    patch.changes.push(PatchChange::AddPrefix {
        prefix: "foaf".to_string(),
        iri: "http://xmlns.com/foaf/0.1/".to_string(),
    });
    patch.changes.push(PatchChange::TransactionBegin);
    patch.changes.push(PatchChange::AddTriple(triple(
        "http://example.org/alice",
        "http://xmlns.com/foaf/0.1/name",
        "http://example.org/literal_placeholder",
    )));
    patch.changes.push(PatchChange::TransactionCommit);

    let serialized = PatchSerializer::serialize(&patch);
    let reparsed = PatchParser::parse(&serialized).expect("should succeed");

    assert_eq!(reparsed.id(), Some("urn:test:42"));
    assert_eq!(reparsed.previous(), Some("urn:test:41"));
    assert_eq!(reparsed.changes.len(), 4);
}

#[test]
fn test_quad_apply_to_simple_graph() {
    let mut graph = Graph::new();
    let q = PatchQuad::new(
        PatchTerm::iri("http://s"),
        PatchTerm::iri("http://p"),
        PatchTerm::iri("http://o"),
        PatchTerm::iri("http://graph1"),
    );
    let patch = RdfPatch {
        headers: vec![],
        changes: vec![PatchChange::AddQuad(q)],
    };
    let stats = apply_patch(&mut graph, &patch).expect("should succeed");
    assert_eq!(stats.triples_added, 1);
}

#[test]
fn test_escaped_literal() {
    let input = "A <http://s> <http://p> \"say \\\"hello\\\"\" .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    if let PatchChange::AddTriple(t) = &patch.changes[0] {
        assert_eq!(t.object.value(), "say \"hello\"");
    } else {
        panic!("expected AddTriple");
    }
}

#[test]
fn test_newline_in_literal_escape() {
    let input = "A <http://s> <http://p> \"line1\\nline2\" .\n";
    let patch = PatchParser::parse(input).expect("should succeed");
    if let PatchChange::AddTriple(t) = &patch.changes[0] {
        assert!(t.object.value().contains('\n'));
    } else {
        panic!("expected AddTriple");
    }
}

#[test]
fn test_patch_change_line_prefix() {
    assert_eq!(PatchChange::TransactionBegin.line_prefix(), "TX");
    assert_eq!(PatchChange::TransactionCommit.line_prefix(), "TC");
    assert_eq!(PatchChange::TransactionAbort.line_prefix(), "TA");
    assert_eq!(
        PatchChange::AddTriple(triple("http://s", "http://p", "http://o")).line_prefix(),
        "A"
    );
    assert_eq!(
        PatchChange::DeleteTriple(triple("http://s", "http://p", "http://o")).line_prefix(),
        "D"
    );
}
