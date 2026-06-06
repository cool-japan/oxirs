//! Tests for SPARQL CONSTRUCT query support.

#![cfg(test)]

use super::construct_engine::ConstructEngine;
use super::construct_parser::{
    parse_construct_query, parse_prefixes, parse_template_term, parse_template_triples,
    split_template_statements, tokenize_template,
};
use super::construct_serialize::{
    abbreviate_term, extract_literal_content, serialize_construct, serialize_construct_jsonld,
    serialize_ntriples, serialize_turtle, ConstructOutputFormat,
};
use super::construct_types::{ConstructConfig, ConstructStats, TemplateTerm};
use crate::store::OxiRSStore;
use crate::Triple;
use std::collections::{HashMap, HashSet};

fn make_social_store() -> OxiRSStore {
    let mut store = OxiRSStore::new();
    store.insert("http://ex/alice", "http://ex/knows", "http://ex/bob");
    store.insert("http://ex/alice", "http://ex/name", "\"Alice\"");
    store.insert("http://ex/bob", "http://ex/knows", "http://ex/carol");
    store.insert("http://ex/bob", "http://ex/name", "\"Bob\"");
    store.insert("http://ex/carol", "http://ex/name", "\"Carol\"");
    store.insert(
        "http://ex/alice",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://ex/Person",
    );
    store.insert(
        "http://ex/bob",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://ex/Person",
    );
    store
}

// ── Config tests ──

#[test]
fn test_config_default() {
    let config = ConstructConfig::default();
    assert!(config.deduplicate);
    assert!(config.max_triples.is_none());
    assert!(config.collect_stats);
    assert_eq!(config.blank_node_prefix, "b");
}

#[test]
fn test_config_custom() {
    let config = ConstructConfig {
        deduplicate: false,
        max_triples: Some(100),
        collect_stats: false,
        blank_node_prefix: "gen".to_string(),
    };
    assert!(!config.deduplicate);
    assert_eq!(config.max_triples, Some(100));
}

// ── Template term parsing ──

#[test]
fn test_parse_variable_term() {
    let term = parse_template_term("?name");
    assert_eq!(term, TemplateTerm::Variable("name".to_string()));
}

#[test]
fn test_parse_dollar_variable_term() {
    let term = parse_template_term("$x");
    assert_eq!(term, TemplateTerm::Variable("x".to_string()));
}

#[test]
fn test_parse_iri_term() {
    let term = parse_template_term("<http://example.org/foo>");
    assert_eq!(
        term,
        TemplateTerm::Iri("http://example.org/foo".to_string())
    );
}

#[test]
fn test_parse_blank_node_term() {
    let term = parse_template_term("_:b0");
    assert_eq!(term, TemplateTerm::BlankNode("b0".to_string()));
}

#[test]
fn test_parse_plain_literal() {
    let term = parse_template_term("\"hello\"");
    assert_eq!(term, TemplateTerm::Literal("hello".to_string()));
}

#[test]
fn test_parse_lang_literal() {
    let term = parse_template_term("\"hello\"@en");
    assert_eq!(
        term,
        TemplateTerm::LangLiteral {
            value: "hello".to_string(),
            lang: "en".to_string(),
        }
    );
}

#[test]
fn test_parse_typed_literal() {
    let term = parse_template_term("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>");
    assert_eq!(
        term,
        TemplateTerm::TypedLiteral {
            value: "42".to_string(),
            datatype: "http://www.w3.org/2001/XMLSchema#integer".to_string(),
        }
    );
}

#[test]
fn test_parse_a_keyword() {
    let term = parse_template_term("a");
    assert_eq!(
        term,
        TemplateTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())
    );
}

// ── Template parsing ──

#[test]
fn test_parse_template_single_triple() {
    let body = "?s <http://ex/p> ?o";
    let triples = parse_template_triples(body).expect("parse");
    assert_eq!(triples.len(), 1);
    assert_eq!(triples[0].subject, TemplateTerm::Variable("s".to_string()));
}

#[test]
fn test_parse_template_multiple_triples() {
    let body = "?s <http://ex/p> ?o . ?s <http://ex/q> ?z";
    let triples = parse_template_triples(body).expect("parse");
    assert_eq!(triples.len(), 2);
}

#[test]
fn test_parse_template_with_blank_nodes() {
    let body = "_:b0 <http://ex/p> ?o . _:b0 <http://ex/q> _:b1";
    let triples = parse_template_triples(body).expect("parse");
    assert_eq!(triples.len(), 2);
    assert_eq!(
        triples[0].subject,
        TemplateTerm::BlankNode("b0".to_string())
    );
}

#[test]
fn test_parse_template_empty_body() {
    let triples = parse_template_triples("").expect("parse");
    assert!(triples.is_empty());
}

// ── CONSTRUCT query parsing ──

#[test]
fn test_parse_construct_basic() {
    let sparql = "CONSTRUCT { ?s <http://ex/p> ?o } WHERE { ?s <http://ex/knows> ?o }";
    let query = parse_construct_query(sparql).expect("parse");
    assert_eq!(query.template.len(), 1);
    assert!(!query.where_patterns.is_empty());
}

#[test]
fn test_parse_construct_with_prefix() {
    let sparql = r#"
        PREFIX ex: <http://ex/>
        CONSTRUCT { ?s ex:p ?o }
        WHERE { ?s ex:knows ?o }
    "#;
    let query = parse_construct_query(sparql).expect("parse");
    assert!(query.prefixes.contains_key("ex"));
    assert_eq!(
        query.prefixes.get("ex").map(|s| s.as_str()),
        Some("http://ex/")
    );
}

#[test]
fn test_parse_construct_where_shorthand() {
    let sparql = "CONSTRUCT WHERE { ?s <http://ex/knows> ?o }";
    let query = parse_construct_query(sparql).expect("parse");
    assert_eq!(query.template.len(), 1);
    assert!(!query.where_patterns.is_empty());
}

#[test]
fn test_parse_construct_with_limit() {
    let sparql = "CONSTRUCT { ?s <http://ex/p> ?o } WHERE { ?s <http://ex/knows> ?o } LIMIT 5";
    let query = parse_construct_query(sparql).expect("parse");
    assert_eq!(query.limit, Some(5));
}

#[test]
fn test_parse_construct_with_offset() {
    let sparql = "CONSTRUCT { ?s <http://ex/p> ?o } WHERE { ?s <http://ex/knows> ?o } OFFSET 2";
    let query = parse_construct_query(sparql).expect("parse");
    assert_eq!(query.offset, Some(2));
}

// ── Engine execution tests ──

#[test]
fn test_construct_basic_execution() {
    let store = make_social_store();
    let engine = ConstructEngine::new();
    let sparql = "CONSTRUCT { ?s <http://ex/friendOf> ?o } WHERE { ?s <http://ex/knows> ?o }";
    let (triples, stats) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(triples.len(), 2); // alice->bob, bob->carol
    assert_eq!(stats.solution_count, 2);
    assert_eq!(stats.template_triple_count, 1);
}

#[test]
fn test_construct_multi_template() {
    let store = make_social_store();
    let engine = ConstructEngine::new();
    let sparql = r#"
        CONSTRUCT {
            ?s <http://ex/friendOf> ?o .
            ?o <http://ex/knownBy> ?s
        } WHERE {
            ?s <http://ex/knows> ?o
        }
    "#;
    let (triples, stats) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(triples.len(), 4); // 2 solutions x 2 template triples
    assert_eq!(stats.template_triple_count, 2);
}

#[test]
fn test_construct_deduplication() {
    let mut store = OxiRSStore::new();
    store.insert("http://ex/a", "http://ex/p", "http://ex/b");
    store.insert("http://ex/a", "http://ex/q", "http://ex/b");

    let engine = ConstructEngine::new();
    // Both solutions map ?s to the same value, so the constructed triple is identical
    let sparql =
        "CONSTRUCT { <http://ex/a> <http://ex/r> <http://ex/b> } WHERE { ?s ?p <http://ex/b> }";
    let (triples, stats) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(triples.len(), 1); // Deduplicated
    assert_eq!(stats.raw_triple_count, 2);
    assert_eq!(stats.deduped_triple_count, 1);
}

#[test]
fn test_construct_no_deduplication() {
    let mut store = OxiRSStore::new();
    store.insert("http://ex/a", "http://ex/p", "http://ex/b");
    store.insert("http://ex/a", "http://ex/q", "http://ex/b");

    let config = ConstructConfig {
        deduplicate: false,
        ..Default::default()
    };
    let engine = ConstructEngine::with_config(config);
    let sparql =
        "CONSTRUCT { <http://ex/a> <http://ex/r> <http://ex/b> } WHERE { ?s ?p <http://ex/b> }";
    let (triples, _) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(triples.len(), 2); // Not deduplicated
}

#[test]
fn test_construct_unbound_variable_skipped() {
    let mut store = OxiRSStore::new();
    store.insert("http://ex/a", "http://ex/p", "http://ex/b");

    let engine = ConstructEngine::new();
    // ?name is unbound -> triple should be skipped
    let sparql = r#"
        CONSTRUCT {
            ?s <http://ex/named> ?name .
            ?s <http://ex/linked> ?o
        } WHERE {
            ?s <http://ex/p> ?o
        }
    "#;
    let (triples, stats) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(triples.len(), 1); // Only the ?s linked ?o triple
    assert_eq!(stats.skipped_unbound, 1);
}

#[test]
fn test_construct_blank_node_scoping() {
    let mut store = OxiRSStore::new();
    store.insert("http://ex/alice", "http://ex/knows", "http://ex/bob");
    store.insert("http://ex/carol", "http://ex/knows", "http://ex/dave");

    let engine = ConstructEngine::new();
    let sparql = r#"
        CONSTRUCT {
            _:node <http://ex/from> ?s .
            _:node <http://ex/to> ?o
        } WHERE {
            ?s <http://ex/knows> ?o
        }
    "#;
    let (triples, stats) = engine.execute(sparql, &store).expect("execute");
    // 2 solutions x 2 template triples = 4 triples
    assert_eq!(triples.len(), 4);
    // Each solution's _:node should be different
    let subjects: Vec<String> = triples.iter().map(|t| t.subject()).collect();
    let unique_blanks: HashSet<&String> = subjects.iter().filter(|s| s.starts_with("_:")).collect();
    assert_eq!(unique_blanks.len(), 2); // Two distinct blank nodes
    assert!(stats.blank_nodes_generated >= 2);
}

#[test]
fn test_construct_with_limit() {
    let store = make_social_store();
    let engine = ConstructEngine::new();
    let sparql = "CONSTRUCT { ?s <http://ex/f> ?o } WHERE { ?s <http://ex/knows> ?o } LIMIT 1";
    let (triples, stats) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(triples.len(), 1);
    assert_eq!(stats.solution_count, 1);
}

#[test]
fn test_construct_with_offset() {
    let store = make_social_store();
    let engine = ConstructEngine::new();
    let sparql = "CONSTRUCT { ?s <http://ex/f> ?o } WHERE { ?s <http://ex/knows> ?o } OFFSET 1";
    let (triples, stats) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(triples.len(), 1);
    assert_eq!(stats.solution_count, 1);
}

#[test]
fn test_construct_max_triples_limit() {
    let store = make_social_store();
    let config = ConstructConfig {
        max_triples: Some(1),
        ..Default::default()
    };
    let engine = ConstructEngine::with_config(config);
    let sparql = "CONSTRUCT { ?s <http://ex/f> ?o } WHERE { ?s <http://ex/knows> ?o }";
    let (triples, _) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(triples.len(), 1);
}

#[test]
fn test_construct_empty_result() {
    let store = OxiRSStore::new();
    let engine = ConstructEngine::new();
    let sparql = "CONSTRUCT { ?s <http://ex/p> ?o } WHERE { ?s <http://ex/nonexistent> ?o }";
    let (triples, stats) = engine.execute(sparql, &store).expect("execute");
    assert!(triples.is_empty());
    assert_eq!(stats.solution_count, 0);
}

#[test]
fn test_construct_where_shorthand_execution() {
    let store = make_social_store();
    let engine = ConstructEngine::new();
    let sparql = "CONSTRUCT WHERE { ?s <http://ex/knows> ?o }";
    let (triples, _) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(triples.len(), 2);
}

// ── Serialization tests ──

#[test]
fn test_serialize_ntriples() {
    let triples = vec![
        Triple::new("http://ex/alice", "http://ex/knows", "http://ex/bob"),
        Triple::new("http://ex/alice", "http://ex/name", "\"Alice\""),
    ];
    let output = serialize_ntriples(&triples).expect("serialize");
    assert!(output.contains("<http://ex/alice>"));
    assert!(output.contains("<http://ex/knows>"));
    assert!(output.contains("<http://ex/bob>"));
    assert!(output.contains("\"Alice\""));
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 2);
}

#[test]
fn test_serialize_turtle_with_prefixes() {
    let triples = vec![Triple::new(
        "http://ex/alice",
        "http://ex/knows",
        "http://ex/bob",
    )];
    let mut prefixes = HashMap::new();
    prefixes.insert("ex".to_string(), "http://ex/".to_string());
    let output = serialize_turtle(&triples, &prefixes).expect("serialize");
    assert!(output.contains("@prefix ex: <http://ex/>"));
    assert!(output.contains("ex:alice"));
    assert!(output.contains("ex:knows"));
    assert!(output.contains("ex:bob"));
}

#[test]
fn test_serialize_turtle_rdf_type_abbreviation() {
    let triples = vec![Triple::new(
        "http://ex/alice",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://ex/Person",
    )];
    let prefixes = HashMap::new();
    let output = serialize_turtle(&triples, &prefixes).expect("serialize");
    assert!(output.contains(" a "));
}

#[test]
fn test_serialize_turtle_subject_grouping() {
    let triples = vec![
        Triple::new("http://ex/alice", "http://ex/knows", "http://ex/bob"),
        Triple::new("http://ex/alice", "http://ex/name", "\"Alice\""),
    ];
    let prefixes = HashMap::new();
    let output = serialize_turtle(&triples, &prefixes).expect("serialize");
    // Both predicates should be grouped under the same subject with ';'
    assert!(output.contains(';'));
}

#[test]
fn test_serialize_jsonld() {
    let triples = vec![Triple::new(
        "http://ex/alice",
        "http://ex/knows",
        "http://ex/bob",
    )];
    let output = serialize_construct_jsonld(&triples).expect("serialize");
    assert!(output.contains("@id"));
    assert!(output.contains("http://ex/alice"));
}

#[test]
fn test_serialize_construct_all_formats() {
    let triples = vec![Triple::new("http://ex/a", "http://ex/b", "http://ex/c")];
    let prefixes = HashMap::new();

    let nt = serialize_construct(&triples, ConstructOutputFormat::NTriples, &prefixes).expect("nt");
    assert!(nt.contains("<http://ex/a>"));

    let ttl = serialize_construct(&triples, ConstructOutputFormat::Turtle, &prefixes).expect("ttl");
    assert!(ttl.contains("<http://ex/a>"));

    let jld =
        serialize_construct(&triples, ConstructOutputFormat::JsonLd, &prefixes).expect("jsonld");
    assert!(jld.contains("http://ex/a"));
}

// ── Blank node N-Triples serialization ──

#[test]
fn test_serialize_ntriples_blank_nodes() {
    let triples = vec![Triple::new("_:b1", "http://ex/p", "http://ex/o")];
    let output = serialize_ntriples(&triples).expect("serialize");
    assert!(output.contains("_:b1"));
}

// ── PREFIX parsing ──

#[test]
fn test_parse_prefixes_single() {
    let sparql = "PREFIX ex: <http://example.org/> SELECT * WHERE { ?s ?p ?o }";
    let prefixes = parse_prefixes(sparql);
    assert_eq!(
        prefixes.get("ex").map(|s| s.as_str()),
        Some("http://example.org/")
    );
}

#[test]
fn test_parse_prefixes_multiple() {
    let sparql = r#"
        PREFIX ex: <http://example.org/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        SELECT * WHERE { ?s ?p ?o }
    "#;
    let prefixes = parse_prefixes(sparql);
    assert_eq!(prefixes.len(), 2);
    assert!(prefixes.contains_key("ex"));
    assert!(prefixes.contains_key("foaf"));
}

#[test]
fn test_parse_prefixes_empty() {
    let sparql = "SELECT * WHERE { ?s ?p ?o }";
    let prefixes = parse_prefixes(sparql);
    assert!(prefixes.is_empty());
}

// ── Extract literal content ──

#[test]
fn test_extract_literal_content_plain() {
    assert_eq!(extract_literal_content("\"hello\""), "hello");
}

#[test]
fn test_extract_literal_content_lang() {
    assert_eq!(extract_literal_content("\"hello\"@en"), "hello");
}

#[test]
fn test_extract_literal_content_typed() {
    assert_eq!(
        extract_literal_content("\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>"),
        "42"
    );
}

// ── Stats ──

#[test]
fn test_construct_stats_tracking() {
    let store = make_social_store();
    let engine = ConstructEngine::new();
    let sparql = r#"
        CONSTRUCT {
            ?s <http://ex/friendOf> ?o .
            ?o <http://ex/knownBy> ?s
        } WHERE {
            ?s <http://ex/knows> ?o
        }
    "#;
    let (_, stats) = engine.execute(sparql, &store).expect("execute");
    assert_eq!(stats.solution_count, 2);
    assert_eq!(stats.template_triple_count, 2);
    assert_eq!(stats.raw_triple_count, 4);
    assert_eq!(stats.deduped_triple_count, 4);
    assert_eq!(stats.skipped_unbound, 0);
}

// ── Abbreviation ──

#[test]
fn test_abbreviate_term_with_prefix() {
    let mut prefixes = HashMap::new();
    prefixes.insert("ex".to_string(), "http://example.org/".to_string());
    assert_eq!(
        abbreviate_term("http://example.org/alice", &prefixes),
        "ex:alice"
    );
}

#[test]
fn test_abbreviate_term_no_match() {
    let prefixes = HashMap::new();
    assert_eq!(
        abbreviate_term("http://other.org/foo", &prefixes),
        "<http://other.org/foo>"
    );
}

#[test]
fn test_abbreviate_blank_node() {
    let prefixes = HashMap::new();
    assert_eq!(abbreviate_term("_:b1", &prefixes), "_:b1");
}

// ── ConstructEngine default ──

#[test]
fn test_engine_default() {
    let engine = ConstructEngine::default();
    assert!(engine.config.deduplicate);
}

// ── Config serialization ──

#[test]
fn test_config_serialization_roundtrip() {
    let config = ConstructConfig {
        deduplicate: false,
        max_triples: Some(50),
        collect_stats: true,
        blank_node_prefix: "test".to_string(),
    };
    let json = serde_json::to_string(&config).expect("serialize");
    let deserialized: ConstructConfig = serde_json::from_str(&json).expect("deserialize");
    assert!(!deserialized.deduplicate);
    assert_eq!(deserialized.max_triples, Some(50));
    assert_eq!(deserialized.blank_node_prefix, "test");
}

#[test]
fn test_stats_serialization_roundtrip() {
    let stats = ConstructStats {
        solution_count: 10,
        template_triple_count: 3,
        raw_triple_count: 30,
        deduped_triple_count: 25,
        skipped_unbound: 2,
        blank_nodes_generated: 5,
    };
    let json = serde_json::to_string(&stats).expect("serialize");
    let deserialized: ConstructStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.solution_count, 10);
    assert_eq!(deserialized.deduped_triple_count, 25);
}

// ── Split template statements ──

#[test]
fn test_split_template_single() {
    let stmts = split_template_statements("?s <http://p> ?o");
    assert_eq!(stmts.len(), 1);
}

#[test]
fn test_split_template_multiple() {
    let stmts = split_template_statements("?s <http://p> ?o . ?a <http://q> ?b");
    assert_eq!(stmts.len(), 2);
}

#[test]
fn test_split_template_quoted_dot() {
    let stmts = split_template_statements("?s <http://p> \"hello. world\"");
    assert_eq!(stmts.len(), 1); // Dot inside quotes should not split
}

// ── Tokenize template ──

#[test]
fn test_tokenize_template_basic() {
    let tokens = tokenize_template("?s <http://p> ?o");
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0], "?s");
    assert_eq!(tokens[1], "<http://p>");
    assert_eq!(tokens[2], "?o");
}

#[test]
fn test_tokenize_template_with_semicolon() {
    let tokens = tokenize_template("?s <http://p> ?o ; <http://q> ?z");
    assert_eq!(tokens.len(), 6);
    assert_eq!(tokens[3], ";");
}
