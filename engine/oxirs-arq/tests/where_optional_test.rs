//! WHERE-keyword omission (SPARQL 1.1 §16.2 / §18).
//!
//! `SELECT` and the explicit-template `CONSTRUCT` may drop the `WHERE` keyword
//! (`SELECT ?s { … }`, `CONSTRUCT { tmpl } { … }`); the `CONSTRUCT WHERE { BGP }`
//! shorthand must keep it (there the WHERE block *is* the template). These tests
//! also guard the still-with-WHERE forms against regression.

use oxirs_arq::query::{parse_query, QueryType};

#[test]
fn select_star_without_where_parses() {
    let q =
        parse_query("SELECT * { ?s ?p ?o }").expect("`SELECT * { … }` must parse without WHERE");
    assert_eq!(q.query_type, QueryType::Select);
}

#[test]
fn select_vars_without_where_parses() {
    let q = parse_query("SELECT ?s ?o { ?s ?p ?o }")
        .expect("`SELECT ?s ?o { … }` must parse without WHERE");
    assert_eq!(q.query_type, QueryType::Select);
    assert_eq!(
        q.select_variables.len(),
        2,
        "both projected variables must be retained when WHERE is omitted"
    );
}

#[test]
fn select_with_where_still_parses() {
    let q = parse_query("SELECT * WHERE { ?s ?p ?o }")
        .expect("`SELECT * WHERE { … }` must still parse");
    assert_eq!(q.query_type, QueryType::Select);
}

#[test]
fn construct_explicit_template_without_where_parses() {
    let q = parse_query("CONSTRUCT { ?s ?p ?o } { ?s ?p ?o }")
        .expect("`CONSTRUCT { tmpl } { … }` (explicit template, no WHERE) must parse");
    assert_eq!(q.query_type, QueryType::Construct);
    assert_eq!(q.construct_template.len(), 1);
}

#[test]
fn construct_explicit_template_with_where_parses() {
    let q = parse_query("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
        .expect("`CONSTRUCT { tmpl } WHERE { … }` must still parse");
    assert_eq!(q.query_type, QueryType::Construct);
    assert_eq!(q.construct_template.len(), 1);
}

#[test]
fn construct_shorthand_with_where_parses() {
    // The shorthand keeps WHERE mandatory; its WHERE BGP becomes the template.
    let q = parse_query("CONSTRUCT WHERE { ?s ?p ?o }")
        .expect("`CONSTRUCT WHERE { BGP }` shorthand must parse");
    assert_eq!(q.query_type, QueryType::Construct);
    assert_eq!(
        q.construct_template.len(),
        1,
        "the shorthand's WHERE BGP is the CONSTRUCT template"
    );
}
