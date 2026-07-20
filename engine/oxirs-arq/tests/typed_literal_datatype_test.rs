//! Datatype IRIs of typed literals `"…"^^dt` — angle-bracket vs prefixed form.
//!
//! R7.6 gap: `"5"^^<urn:myint>` (and any authority-less scheme such as `tag:`)
//! failed with "Undefined prefix 'urn'". The lexer had erased whether the
//! datatype was written as `^^<iri>` or `^^prefix:local`, and resolution then
//! treated only `scheme://…` as an absolute IRI — so `<urn:myint>` was
//! mis-resolved as the prefixed name `urn:myint`.
//!
//! The fix carries the written origin through `Token::RdfLiteral` (a
//! `DatatypeRef`): the angle-bracket form is an absolute IRI used verbatim; only
//! the `prefix:local` form is prefix-resolved. These tests pin every combination
//! and guard the previously-working forms against regression.

use oxirs_arq::algebra::{Algebra, Term};
use oxirs_arq::query::parse_query;

/// Parse `SELECT ?s WHERE { ?s ?p <object> }` and return the datatype IRI of the
/// object literal (its `NamedNode` rendered as a string), or `None` if the
/// object is not a typed literal.
fn object_datatype(query: &str) -> Option<String> {
    let q = parse_query(query).unwrap_or_else(|e| panic!("must parse `{query}`: {e}"));
    let bgp = match &q.where_clause {
        Algebra::Bgp(triples) => triples,
        other => panic!("expected a BGP WHERE, got {other:?}"),
    };
    let triple = bgp.first().expect("one triple expected");
    match &triple.object {
        Term::Literal(lit) => lit.datatype.as_ref().map(|dt| dt.as_str().to_string()),
        other => panic!("expected a literal object, got {other:?}"),
    }
}

const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";

// ── `^^<iri>` — angle-bracket absolute IRIs, authority-less included ──────────

#[test]
fn urn_scheme_datatype_is_kept_verbatim() {
    // The core regression: `urn:` has no `//` authority yet must be an absolute
    // IRI, not a prefixed name.
    assert_eq!(
        object_datatype("SELECT ?s WHERE { ?s ?p \"5\"^^<urn:myint> }").as_deref(),
        Some("urn:myint"),
        "`^^<urn:myint>` must be kept as the absolute IRI urn:myint"
    );
}

#[test]
fn tag_scheme_datatype_is_kept_verbatim() {
    assert_eq!(
        object_datatype("SELECT ?s WHERE { ?s ?p \"x\"^^<tag:example.com,2020:t> }").as_deref(),
        Some("tag:example.com,2020:t"),
        "`^^<tag:…>` must be kept as an absolute IRI"
    );
}

#[test]
fn http_angle_bracket_datatype_is_kept_verbatim() {
    // The previously-working `scheme://` form must not regress.
    assert_eq!(
        object_datatype(&format!(
            "SELECT ?s WHERE {{ ?s ?p \"5\"^^<{XSD_INTEGER}> }}"
        ))
        .as_deref(),
        Some(XSD_INTEGER),
        "`^^<http://…>` must be kept verbatim"
    );
}

// ── `^^prefix:local` — prefixed names resolved against the prologue ──────────

#[test]
fn prefixed_datatype_is_resolved_against_prologue() {
    let q = format!(
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> \
         SELECT ?s WHERE {{ ?s ?p \"5\"^^xsd:integer }}"
    );
    assert_eq!(
        object_datatype(&q).as_deref(),
        Some(XSD_INTEGER),
        "`^^xsd:integer` must resolve to the full XSD IRI"
    );
}

#[test]
fn undefined_prefix_datatype_is_a_parse_error() {
    // A genuine prefixed datatype with no matching PREFIX must still fail loud.
    assert!(
        parse_query("SELECT ?s WHERE { ?s ?p \"5\"^^nope:bar }").is_err(),
        "an undefined datatype prefix must be a parse error"
    );
    // But the same local part inside angle brackets is an absolute IRI and MUST
    // parse — proving the two forms are no longer conflated.
    assert_eq!(
        object_datatype("SELECT ?s WHERE { ?s ?p \"5\"^^<nope:bar> }").as_deref(),
        Some("nope:bar"),
        "`^^<nope:bar>` is an absolute IRI, not a prefixed name"
    );
}

// ── the same distinction inside an expression (FILTER) position ──────────────

#[test]
fn typed_literal_in_filter_expression_keeps_urn_datatype() {
    // Exercises the expression-grammar RdfLiteral path (not just the term path).
    // Before the fix this 400-ed on "Undefined prefix 'urn'".
    assert!(
        parse_query("SELECT ?s WHERE { ?s ?p ?o FILTER(?o = \"5\"^^<urn:myint>) }").is_ok(),
        "a `^^<urn:…>` typed literal in a FILTER must parse"
    );
    assert!(
        parse_query(
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> \
             SELECT ?s WHERE { ?s ?p ?o FILTER(?o = \"5\"^^xsd:integer) }"
        )
        .is_ok(),
        "a `^^xsd:integer` typed literal in a FILTER must still parse"
    );
}
