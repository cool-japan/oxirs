//! Full-IRI (`<scheme:...>`) acceptance in the SPARQL tokenizer (R7.6 gap a).
//!
//! Before this fix the `<` tokenizer only entered IRIREF mode when the first
//! character after `<` was `h` or `/`, so `<urn:p>`, `<mailto:x>` and the
//! relative `<p>` were mis-tokenized as the `<` comparison operator and the
//! parser failed with "Expected term". The tokenizer now scans a full
//!   IRIREF ::= '<' ([^<>"{}|^`\] - [#x00-#x20])* '>'
//! (per SPARQL 1.1) and only commits an IRI when the closing `>` is reached
//! before any excluded character — so genuine `<` / `<=` comparisons (which are
//! always separated from their operands by whitespace or hit an excluded char
//! before any `>`) still lex as operators.
//!
//! These are pure parser assertions driven through the public `parse_query`.

use oxirs_arq::query::parse_query;

/// Debug rendering of the parsed WHERE algebra. `NamedNode`'s derived `Debug`
/// embeds the raw IRI string verbatim, so substring checks confirm the IRI was
/// captured as a term rather than swallowed as an operator.
fn where_debug(query: &str) -> String {
    let q = parse_query(query).unwrap_or_else(|e| panic!("must parse `{query}`: {e}"));
    format!("{:?}", q.where_clause)
}

// ── `<urn:…>` full IRIs in every triple position ─────────────────────────────

#[test]
fn urn_iri_in_predicate_position_parses() {
    let dbg = where_debug("SELECT ?a WHERE { ?a <urn:p> ?x }");
    assert!(
        dbg.contains("urn:p"),
        "predicate `<urn:p>` must be captured as an IRI term, got: {dbg}"
    );
}

#[test]
fn urn_iri_in_subject_position_parses() {
    let dbg = where_debug("SELECT ?x WHERE { <urn:s> <urn:p> ?x }");
    assert!(
        dbg.contains("urn:s"),
        "subject `<urn:s>` must be captured as an IRI term, got: {dbg}"
    );
}

#[test]
fn urn_iri_in_object_position_parses() {
    let dbg = where_debug("SELECT ?a WHERE { ?a <urn:p> <urn:o> }");
    assert!(
        dbg.contains("urn:o"),
        "object `<urn:o>` must be captured as an IRI term, got: {dbg}"
    );
}

// ── http/https IRIs (the previously-working forms must keep working) ──────────

#[test]
fn http_iri_predicate_parses() {
    let dbg = where_debug("SELECT ?a WHERE { ?a <http://example.org/p> ?x }");
    assert!(
        dbg.contains("http://example.org/p"),
        "http predicate IRI must be captured, got: {dbg}"
    );
}

#[test]
fn https_iri_with_fragment_predicate_parses() {
    let dbg = where_debug("SELECT ?a WHERE { ?a <https://example.org/vocab#field> ?x }");
    assert!(
        dbg.contains("https://example.org/vocab#field"),
        "https `#fragment` predicate IRI must be captured intact, got: {dbg}"
    );
}

// ── Relative IRI `<p>` (BASE-resolution target; stored verbatim here) ─────────

#[test]
fn relative_iri_predicate_parses() {
    // No BASE is declared, so `parse_query` stores the reference verbatim via
    // `NamedNode::new_unchecked` (BASE resolution, when present, is a separate
    // concern this fix does not touch). The point is that `<relpred>` is no
    // longer rejected as a `<` operator.
    let dbg = where_debug("SELECT ?a WHERE { ?a <relpred> ?x }");
    assert!(
        dbg.contains("relpred"),
        "relative IRI `<relpred>` must parse as an IRI term, got: {dbg}"
    );
    // The single-letter form from the task description must also parse.
    assert!(
        parse_query("SELECT ?a WHERE { ?a <p> ?x }").is_ok(),
        "relative IRI `<p>` must parse"
    );
}

// ── Property-path position (predicate becomes a path, not a plain BGP term) ───

#[test]
fn full_iris_in_property_paths_parse() {
    for q in [
        "SELECT ?a WHERE { ?a <urn:p>/<urn:q> ?x }", // sequence
        "SELECT ?a WHERE { ?a <urn:p>|<urn:q> ?x }", // alternative
        "SELECT ?a WHERE { ?a <urn:p>+ ?x }",        // one-or-more
        "SELECT ?a WHERE { ?a <urn:p>* ?x }",        // zero-or-more
        "SELECT ?a WHERE { ?a ^<urn:p> ?x }",        // inverse
    ] {
        assert!(
            parse_query(q).is_ok(),
            "property path with full IRIs must parse: `{q}`"
        );
    }
}

// ── The `<` / `<=` / `>` comparison operators must NOT regress ────────────────

#[test]
fn comparison_operators_still_parse() {
    for q in [
        "SELECT ?a WHERE { ?a <urn:p> ?x . FILTER(?x < 5) }",
        "SELECT ?a WHERE { ?a <urn:p> ?x . FILTER(?x <= 5) }",
        "SELECT ?a WHERE { ?a <urn:p> ?x . FILTER(?x > 5) }",
        "SELECT ?a WHERE { ?a <urn:p> ?x . FILTER(?x >= 5) }",
        "SELECT ?a WHERE { ?a <urn:p> ?x . ?b <urn:q> ?y . FILTER(?x < ?y) }",
        "SELECT ?a WHERE { ?a <urn:p> ?x . FILTER(?x > 1 && ?x < 9) }",
    ] {
        assert!(
            parse_query(q).is_ok(),
            "comparison operator query must still parse: `{q}`"
        );
    }
}
