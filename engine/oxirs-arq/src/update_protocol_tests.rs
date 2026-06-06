//! Tests for the standalone SPARQL 1.1 Update protocol module.

#![cfg(test)]

use crate::update_protocol::{
    ClearType, DropType, PatternTerm, SparqlUpdate, SparqlUpdateParser, Triple, UpdateExecutor,
};

// ------------------------------------------------------------------
// Parser – InsertData
// ------------------------------------------------------------------

#[test]
fn test_parse_insert_data_single_triple() {
    let input = "INSERT DATA { <http://a> <http://b> <http://c> }";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::InsertData(triples) => {
            assert_eq!(triples.len(), 1);
            assert_eq!(triples[0].s, "http://a");
            assert_eq!(triples[0].p, "http://b");
            assert_eq!(triples[0].o, "http://c");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_parse_insert_data_multiple_triples() {
    let input = "INSERT DATA { <s1> <p1> <o1> . <s2> <p2> <o2> }";
    let result = SparqlUpdateParser::parse_one(input).expect("parse failed");
    if let SparqlUpdate::InsertData(triples) = result {
        assert_eq!(triples.len(), 2);
    } else {
        panic!("wrong variant");
    }
}

// ------------------------------------------------------------------
// Parser – DeleteData
// ------------------------------------------------------------------

#[test]
fn test_parse_delete_data() {
    let input = "DELETE DATA { <http://x> <http://y> <http://z> }";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::DeleteData(triples) => {
            assert_eq!(triples.len(), 1);
            assert_eq!(triples[0].s, "http://x");
        }
        _ => panic!("wrong variant"),
    }
}

// ------------------------------------------------------------------
// Parser – CreateGraph
// ------------------------------------------------------------------

#[test]
fn test_parse_create_graph() {
    let input = "CREATE GRAPH <http://example.org/g>";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::CreateGraph { iri, silent } => {
            assert_eq!(iri, "http://example.org/g");
            assert!(!silent);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_parse_create_graph_silent() {
    let input = "CREATE SILENT GRAPH <http://example.org/g>";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::CreateGraph { iri: _, silent } => {
            assert!(silent);
        }
        _ => panic!("wrong variant"),
    }
}

// ------------------------------------------------------------------
// Parser – DropGraph
// ------------------------------------------------------------------

#[test]
fn test_parse_drop_graph_named() {
    let input = "DROP GRAPH <http://g>";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::DropGraph {
            iri,
            silent,
            drop_type,
        } => {
            assert_eq!(iri, Some("http://g".to_string()));
            assert!(!silent);
            assert_eq!(drop_type, DropType::Graph);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_parse_drop_all() {
    let input = "DROP ALL";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::DropGraph { drop_type, .. } => {
            assert_eq!(drop_type, DropType::All);
        }
        _ => panic!("wrong variant"),
    }
}

// ------------------------------------------------------------------
// Parser – ClearGraph
// ------------------------------------------------------------------

#[test]
fn test_parse_clear_default() {
    let input = "CLEAR DEFAULT";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::ClearGraph { clear_type, .. } => {
            assert_eq!(clear_type, ClearType::Default);
        }
        _ => panic!("wrong variant"),
    }
}

// ------------------------------------------------------------------
// Parser – CopyGraph, MoveGraph, AddGraph
// ------------------------------------------------------------------

#[test]
fn test_parse_copy_graph() {
    let input = "COPY <http://src> TO <http://dst>";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::CopyGraph {
            source,
            target,
            silent,
        } => {
            assert_eq!(source, "http://src");
            assert_eq!(target, "http://dst");
            assert!(!silent);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_parse_move_graph() {
    let input = "MOVE <http://old> TO <http://new>";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::MoveGraph { source, target, .. } => {
            assert_eq!(source, "http://old");
            assert_eq!(target, "http://new");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_parse_add_graph() {
    let input = "ADD <http://src> TO <http://dst>";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::AddGraph { .. } => {}
        _ => panic!("wrong variant"),
    }
}

// ------------------------------------------------------------------
// Parser – LOAD
// ------------------------------------------------------------------

#[test]
fn test_parse_load_basic() {
    let input = "LOAD <http://data.example.org/data.ttl>";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::Load { iri, into, silent } => {
            assert_eq!(iri, "http://data.example.org/data.ttl");
            assert!(into.is_none());
            assert!(!silent);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_parse_load_into_graph() {
    let input = "LOAD <http://src.ttl> INTO GRAPH <http://target>";
    let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
    match update {
        SparqlUpdate::Load { into, .. } => {
            assert_eq!(into, Some("http://target".to_string()));
        }
        _ => panic!("wrong variant"),
    }
}

// ------------------------------------------------------------------
// Parser – multiple operations
// ------------------------------------------------------------------

#[test]
fn test_parse_multiple_operations() {
    let input = "INSERT DATA { <s1> <p1> <o1> } ; DELETE DATA { <s2> <p2> <o2> }";
    let updates = SparqlUpdateParser::parse(input).expect("parse failed");
    assert_eq!(updates.len(), 2);
}

#[test]
fn test_parse_empty_input() {
    let updates = SparqlUpdateParser::parse("").expect("parse failed");
    assert!(updates.is_empty());
}

#[test]
fn test_parse_unknown_keyword_returns_error() {
    let result = SparqlUpdateParser::parse_one("FROBULATE { }");
    assert!(result.is_err());
}

// ------------------------------------------------------------------
// UpdateExecutor – InsertData / DeleteData
// ------------------------------------------------------------------

#[test]
fn test_executor_insert_data() {
    let mut exec = UpdateExecutor::new();
    let update = SparqlUpdate::InsertData(vec![Triple::new("s", "p", "o")]);
    let result = exec.execute(&update).expect("execute failed");
    assert_eq!(result.triples_inserted, 1);
    assert_eq!(exec.triple_count(), 1);
}

#[test]
fn test_executor_delete_data_existing() {
    let mut exec = UpdateExecutor::new();
    let t = Triple::new("s", "p", "o");
    exec.execute(&SparqlUpdate::InsertData(vec![t.clone()]))
        .expect("insert failed");
    let result = exec
        .execute(&SparqlUpdate::DeleteData(vec![t]))
        .expect("delete failed");
    assert_eq!(result.triples_deleted, 1);
    assert_eq!(exec.triple_count(), 0);
}

#[test]
fn test_executor_delete_data_nonexistent() {
    let mut exec = UpdateExecutor::new();
    let result = exec
        .execute(&SparqlUpdate::DeleteData(vec![Triple::new("x", "y", "z")]))
        .expect("delete failed");
    assert_eq!(result.triples_deleted, 0);
}

// ------------------------------------------------------------------
// UpdateExecutor – CreateGraph / DropGraph
// ------------------------------------------------------------------

#[test]
fn test_executor_create_graph() {
    let mut exec = UpdateExecutor::new();
    exec.execute(&SparqlUpdate::CreateGraph {
        iri: "http://g".to_string(),
        silent: false,
    })
    .expect("create failed");
    assert_eq!(exec.graph_count(), 1);
    assert!(exec.get_graph("http://g").is_some());
}

#[test]
fn test_executor_create_duplicate_graph_silent() {
    let mut exec = UpdateExecutor::new();
    let update = SparqlUpdate::CreateGraph {
        iri: "http://g".to_string(),
        silent: true,
    };
    exec.execute(&update).expect("first create failed");
    exec.execute(&update)
        .expect("second create (silent) should not error");
}

#[test]
fn test_executor_create_duplicate_graph_non_silent_errors() {
    let mut exec = UpdateExecutor::new();
    let update = SparqlUpdate::CreateGraph {
        iri: "http://g".to_string(),
        silent: false,
    };
    exec.execute(&update).expect("first create failed");
    let result = exec.execute(&update);
    assert!(result.is_err());
}

#[test]
fn test_executor_drop_named_graph() {
    let mut exec = UpdateExecutor::new();
    exec.execute(&SparqlUpdate::CreateGraph {
        iri: "http://g".to_string(),
        silent: false,
    })
    .expect("create failed");
    exec.execute(&SparqlUpdate::DropGraph {
        iri: Some("http://g".to_string()),
        silent: false,
        drop_type: DropType::Graph,
    })
    .expect("drop failed");
    assert_eq!(exec.graph_count(), 0);
}

// ------------------------------------------------------------------
// UpdateExecutor – ClearGraph
// ------------------------------------------------------------------

#[test]
fn test_executor_clear_default_graph() {
    let mut exec = UpdateExecutor::new();
    exec.execute(&SparqlUpdate::InsertData(vec![
        Triple::new("s1", "p1", "o1"),
        Triple::new("s2", "p2", "o2"),
    ]))
    .expect("insert failed");
    exec.execute(&SparqlUpdate::ClearGraph {
        iri: None,
        silent: false,
        clear_type: ClearType::Default,
    })
    .expect("clear failed");
    assert_eq!(exec.triple_count(), 0);
}

// ------------------------------------------------------------------
// UpdateExecutor – execute_all
// ------------------------------------------------------------------

#[test]
fn test_executor_execute_all() {
    let mut exec = UpdateExecutor::new();
    let updates = vec![
        SparqlUpdate::InsertData(vec![Triple::new("a", "b", "c")]),
        SparqlUpdate::InsertData(vec![Triple::new("d", "e", "f")]),
    ];
    let results = exec.execute_all(&updates).expect("execute_all failed");
    assert_eq!(results.len(), 2);
    assert_eq!(exec.triple_count(), 2);
}

// ------------------------------------------------------------------
// Triple / PatternTerm helpers
// ------------------------------------------------------------------

#[test]
fn test_triple_equality() {
    let t1 = Triple::new("s", "p", "o");
    let t2 = Triple::new("s", "p", "o");
    assert_eq!(t1, t2);
}

#[test]
fn test_pattern_term_is_variable() {
    let var = PatternTerm::Variable("x".to_string());
    assert!(var.is_variable());
    let iri = PatternTerm::Iri("http://a".to_string());
    assert!(!iri.is_variable());
}

#[test]
fn test_pattern_term_variable_name() {
    let var = PatternTerm::Variable("myVar".to_string());
    assert_eq!(var.variable_name(), Some("myVar"));
    let iri = PatternTerm::Iri("http://a".to_string());
    assert_eq!(iri.variable_name(), None);
}

// ------------------------------------------------------------------
// UpdateExecutor – CopyGraph / MoveGraph / AddGraph
// ------------------------------------------------------------------

#[test]
fn test_executor_copy_graph() {
    let mut exec = UpdateExecutor::new();
    exec.execute(&SparqlUpdate::CreateGraph {
        iri: "http://src".into(),
        silent: false,
    })
    .expect("create src");
    // Manually add a triple to the named graph via the HashMap.
    exec.named_graphs
        .get_mut("http://src")
        .expect("src exists")
        .push(Triple::new("a", "b", "c"));

    exec.execute(&SparqlUpdate::CopyGraph {
        source: "http://src".into(),
        target: "http://dst".into(),
        silent: false,
    })
    .expect("copy failed");
    assert_eq!(exec.get_graph("http://dst").expect("dst exists").len(), 1);
}

#[test]
fn test_executor_move_graph() {
    let mut exec = UpdateExecutor::new();
    exec.execute(&SparqlUpdate::CreateGraph {
        iri: "http://src".into(),
        silent: false,
    })
    .expect("create src");
    exec.named_graphs
        .get_mut("http://src")
        .expect("src exists")
        .push(Triple::new("x", "y", "z"));

    exec.execute(&SparqlUpdate::MoveGraph {
        source: "http://src".into(),
        target: "http://dst".into(),
        silent: false,
    })
    .expect("move failed");

    assert!(exec.get_graph("http://src").is_none());
    assert_eq!(exec.get_graph("http://dst").expect("dst exists").len(), 1);
}
