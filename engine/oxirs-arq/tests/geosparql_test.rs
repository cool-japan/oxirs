//! GeoSPARQL minimal-implementation tests: CURIE-expanded function-call parsing,
//! `geof:distance` over WKT `POINT` literals with unit conversion, the OxiRS
//! `oxgeo:distanceLatLon` convenience function over bare decimal coordinates,
//! error semantics inside `FILTER`/`BIND`, and a regression guard proving the
//! prefixed-name-call parser fix does not disturb aggregate handling when a
//! default namespace (`PREFIX : <...>`) is declared.
//!
//! Expected distances are the Haversine great-circle values on a spherical Earth
//! of mean radius 6,371,000 m (independently reproduced with a Python oracle).

use oxirs_arq::algebra::{Expression, PropertyPath, Term};
use oxirs_arq::query::{ProjectionItem, QueryParser};
use oxirs_arq::{Dataset, Literal, QueryExecutor, TriplePattern, Variable};
use oxirs_core::model::NamedNode;
use std::collections::HashMap;

// ── Namespaces used throughout the tests ─────────────────────────────────────

const GEOF: &str = "http://www.opengis.net/def/function/geosparql/";
const UOM: &str = "http://www.opengis.net/def/uom/OGC/1.0/";
const OXGEO: &str = "http://oxirs.io/fn/geo#";
const WKT_LITERAL: &str = "http://www.opengis.net/ont/geosparql#wktLiteral";
const XSD_DECIMAL: &str = "http://www.w3.org/2001/XMLSchema#decimal";

const GEOF_DISTANCE_IRI: &str = "http://www.opengis.net/def/function/geosparql/distance";

// The standard SPARQL prologue shared by the execution tests.
fn prologue() -> String {
    format!(
        "PREFIX ex: <http://ex/>\n\
         PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>\n\
         PREFIX geof: <{GEOF}>\n\
         PREFIX uom: <{UOM}>\n\
         PREFIX oxgeo: <{OXGEO}>\n"
    )
}

// ── In-memory dataset (mirrors the arq compliance-test mock) ─────────────────

struct MemDataset {
    triples: Vec<(Term, Term, Term)>,
}

impl MemDataset {
    fn add(&mut self, s: Term, p: Term, o: Term) {
        self.triples.push((s, p, o));
    }
}

fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

fn typed_lit(value: &str, datatype: &str) -> Term {
    Term::Literal(Literal {
        value: value.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(datatype)),
    })
}

fn slot_matches(slot: &Term, value: &Term) -> bool {
    matches!(slot, Term::Variable(_)) || slot == value
}

fn pred_matches(slot: &Term, stored: &Term) -> bool {
    match slot {
        Term::Variable(_) => true,
        Term::PropertyPath(PropertyPath::Variable(_)) => true,
        Term::PropertyPath(PropertyPath::Iri(n)) => *stored == Term::Iri(n.clone()),
        other => other == stored,
    }
}

impl Dataset for MemDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> anyhow::Result<Vec<(Term, Term, Term)>> {
        Ok(self
            .triples
            .iter()
            .filter(|(s, p, o)| {
                slot_matches(&pattern.subject, s)
                    && pred_matches(&pattern.predicate, p)
                    && slot_matches(&pattern.object, o)
            })
            .cloned()
            .collect())
    }

    fn contains_triple(&self, s: &Term, p: &Term, o: &Term) -> anyhow::Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(a, b, c)| a == s && b == p && c == o))
    }

    fn subjects(&self) -> anyhow::Result<Vec<Term>> {
        Ok(self.triples.iter().map(|(s, _, _)| s.clone()).collect())
    }

    fn predicates(&self) -> anyhow::Result<Vec<Term>> {
        Ok(self.triples.iter().map(|(_, p, _)| p.clone()).collect())
    }

    fn objects(&self) -> anyhow::Result<Vec<Term>> {
        Ok(self.triples.iter().map(|(_, _, o)| o.clone()).collect())
    }
}

fn var(name: &str) -> Variable {
    Variable::new(name).expect("valid variable")
}

/// Parse `query`, execute its WHERE clause against `dataset`, return the rows.
fn run(query: &str, dataset: &MemDataset) -> Vec<HashMap<Variable, Term>> {
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse(query)
        .unwrap_or_else(|e| panic!("query should parse: {e}\n{query}"));
    let mut executor = QueryExecutor::new();
    let (solution, _stats) = executor
        .execute(&parsed.where_clause, dataset)
        .unwrap_or_else(|e| panic!("query should execute: {e}\n{query}"));
    solution
        .iter()
        .map(|binding| {
            binding
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .collect()
}

/// Extract the single numeric binding of `name` from a one-row result.
fn single_number(rows: &[HashMap<Variable, Term>], name: &str) -> f64 {
    assert_eq!(
        rows.len(),
        1,
        "expected exactly one row, got {}",
        rows.len()
    );
    match rows[0].get(&var(name)) {
        Some(Term::Literal(lit)) => lit
            .value
            .parse::<f64>()
            .unwrap_or_else(|_| panic!("binding ?{name} not numeric: {}", lit.value)),
        other => panic!("binding ?{name} missing or non-literal: {other:?}"),
    }
}

// Tokyo Station and Osaka Station (WGS84). Great-circle distance:
//   403057.527 m == 403.058 km == 250.448 statute miles (Python oracle).
const TOKYO_LAT: &str = "35.681236";
const TOKYO_LON: &str = "139.767125";
const OSAKA_LAT: &str = "34.702485";
const OSAKA_LON: &str = "135.495951";
const TOKYO_OSAKA_METRES: f64 = 403_057.527;

fn wkt_dataset() -> MemDataset {
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    ds.add(
        iri("http://ex/tokyo"),
        iri("http://ex/wkt"),
        typed_lit(&format!("POINT({TOKYO_LON} {TOKYO_LAT})"), WKT_LITERAL),
    );
    ds.add(
        iri("http://ex/osaka"),
        iri("http://ex/wkt"),
        // A CRS-prefixed wktLiteral, to exercise the `<...> POINT(...)` path.
        typed_lit(
            &format!(
                "<http://www.opengis.net/def/crs/OGC/1.3/CRS84> POINT({OSAKA_LON} {OSAKA_LAT})"
            ),
            WKT_LITERAL,
        ),
    );
    ds
}

// ── Parser-level tests ───────────────────────────────────────────────────────

/// Parse `SELECT (<expr> AS ?y) WHERE { ?x ?p ?o }` and return the projected AST.
fn projected_expr(prefixes: &str, select_expr: &str) -> Expression {
    let query = format!("{prefixes}SELECT ({select_expr} AS ?y) WHERE {{ ?x ?p ?o }}");
    let mut parser = QueryParser::new();
    let parsed = parser
        .parse(&query)
        .unwrap_or_else(|e| panic!("query should parse: {e}\n{query}"));
    match parsed.projection_items.into_iter().next() {
        Some(ProjectionItem::Expression { expr, .. }) => expr,
        other => panic!("expected an expression projection, got {other:?}"),
    }
}

#[test]
fn parser_expands_geof_curie_to_full_iri() {
    let prefixes = format!("PREFIX geof: <{GEOF}>\nPREFIX uom: <{UOM}>\n");
    match projected_expr(&prefixes, "geof:distance(?a, ?b, uom:metre)") {
        Expression::Function { name, args } => {
            assert_eq!(
                name, GEOF_DISTANCE_IRI,
                "geof:distance must expand to its full IRI"
            );
            assert_eq!(args.len(), 3);
        }
        other => panic!("geof:distance must lower to a Function, got {other:?}"),
    }
}

#[test]
fn parser_prefix_alias_resolves_to_same_iri() {
    // The chosen prefix label is irrelevant: any label bound to the GeoSPARQL
    // function namespace resolves the call to the identical full IRI.
    let a = projected_expr(
        &format!("PREFIX geof: <{GEOF}>\n"),
        "geof:distance(?a, ?b, ?u)",
    );
    let b = projected_expr(
        &format!("PREFIX spatialfn: <{GEOF}>\n"),
        "spatialfn:distance(?a, ?b, ?u)",
    );
    match (a, b) {
        (Expression::Function { name: na, .. }, Expression::Function { name: nb, .. }) => {
            assert_eq!(na, GEOF_DISTANCE_IRI);
            assert_eq!(nb, GEOF_DISTANCE_IRI);
            assert_eq!(na, nb, "different prefix labels must resolve identically");
        }
        other => panic!("both must lower to a Function, got {other:?}"),
    }
}

#[test]
fn parser_unregistered_prefix_call_is_left_unexpanded() {
    // Mirrors the non-call branch: an undeclared prefix keeps `prefix:local`.
    match projected_expr("", "unknownpfx:someFn(?a)") {
        Expression::Function { name, .. } => {
            assert_eq!(name, "unknownpfx:someFn");
        }
        other => panic!("expected a Function, got {other:?}"),
    }
}

#[test]
fn parser_preserves_aggregates_under_default_namespace() {
    // Regression: a declared default namespace must NOT cause the empty-prefix
    // aggregate names reaching parse_primary (in HAVING/expression context) to
    // be rewritten to `<ns>SUM`, which `aggregate_function_name` would no longer
    // recognise. `SUM` stays an aggregate, so a two-argument `SUM(?x, ?y)` in
    // HAVING is still rejected for wrong arity even with `PREFIX : <...>` set.
    let head = "PREFIX : <http://ex/>\nSELECT (SUM(?x) AS ?s) WHERE { ?a :p ?x } GROUP BY ?a";

    let mut parser = QueryParser::new();
    assert!(
        parser
            .parse(&format!("{head} HAVING(SUM(?x, ?y) > 10)"))
            .is_err(),
        "SUM must still be recognised as an aggregate (wrong arity) under a default namespace"
    );

    // The well-formed aggregates parse cleanly.
    let mut parser = QueryParser::new();
    assert!(
        parser
            .parse(&format!("{head} HAVING(SUM(?x) > 10)"))
            .is_ok(),
        "HAVING(SUM(?x) > 10) must parse under a default namespace"
    );
    let mut parser = QueryParser::new();
    let count_head =
        "PREFIX : <http://ex/>\nSELECT (COUNT(*) AS ?c) WHERE { ?a :p ?x } GROUP BY ?a";
    assert!(
        parser
            .parse(&format!("{count_head} HAVING(COUNT(*) > 1)"))
            .is_ok(),
        "HAVING(COUNT(*) > 1) must parse under a default namespace"
    );
}

// ── geof:distance execution + unit conversion ────────────────────────────────

#[test]
fn geof_distance_metres_matches_oracle() {
    let ds = wkt_dataset();
    let query = format!(
        "{}SELECT ?d WHERE {{ ex:tokyo ex:wkt ?wa . ex:osaka ex:wkt ?wb . \
         BIND(geof:distance(?wa, ?wb, uom:metre) AS ?d) }}",
        prologue()
    );
    let got = single_number(&run(&query, &ds), "d");
    assert!(
        (got - TOKYO_OSAKA_METRES).abs() < 1.0,
        "distance {got} m should be ~{TOKYO_OSAKA_METRES} m"
    );
}

#[test]
fn geof_distance_kilometres_and_miles() {
    let ds = wkt_dataset();
    let km_query = format!(
        "{}SELECT ?d WHERE {{ ex:tokyo ex:wkt ?wa . ex:osaka ex:wkt ?wb . \
         BIND(geof:distance(?wa, ?wb, uom:kilometre) AS ?d) }}",
        prologue()
    );
    let km = single_number(&run(&km_query, &ds), "d");
    assert!(
        (km - TOKYO_OSAKA_METRES / 1000.0).abs() < 0.001,
        "kilometre conversion wrong: {km}"
    );

    let mi_query = format!(
        "{}SELECT ?d WHERE {{ ex:tokyo ex:wkt ?wa . ex:osaka ex:wkt ?wb . \
         BIND(geof:distance(?wa, ?wb, uom:mile) AS ?d) }}",
        prologue()
    );
    let mi = single_number(&run(&mi_query, &ds), "d");
    assert!(
        (mi - TOKYO_OSAKA_METRES / 1_609.344).abs() < 0.001,
        "mile conversion wrong: {mi}"
    );
}

#[test]
fn geof_distance_unknown_units_excludes_row_in_filter() {
    // An unrecognised unit is an ordinary evaluation error, so SPARQL §17.3
    // excludes the row rather than failing the whole query.
    let ds = wkt_dataset();
    let query = format!(
        "{}PREFIX bad: <http://example.com/unit/>\n\
         SELECT ?d WHERE {{ ex:tokyo ex:wkt ?wa . ex:osaka ex:wkt ?wb . \
         BIND(geof:distance(?wa, ?wb, uom:metre) AS ?d) \
         FILTER(geof:distance(?wa, ?wb, bad:furlong) < 999999999.0) }}",
        prologue()
    );
    let rows = run(&query, &ds);
    assert!(rows.is_empty(), "unknown-unit FILTER must drop the row");
}

// ── oxgeo:distanceLatLon convenience function ────────────────────────────────

#[test]
fn distance_lat_lon_literal_args_match_oracle() {
    let ds = wkt_dataset(); // dataset unused by this query but `run` needs one
    let query = format!(
        "{}SELECT ?d WHERE {{ BIND(oxgeo:distanceLatLon({TOKYO_LAT}, {TOKYO_LON}, {OSAKA_LAT}, {OSAKA_LON}) AS ?d) }}",
        prologue()
    );
    let got = single_number(&run(&query, &ds), "d");
    assert!(
        (got - TOKYO_OSAKA_METRES).abs() < 1.0,
        "distanceLatLon {got} m should be ~{TOKYO_OSAKA_METRES} m"
    );
}

/// A center point plus four candidates; the two `near*` are within 2 km, the two
/// `far*` are not (distances precomputed with the Python oracle).
fn radius_dataset() -> MemDataset {
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    let geo = "http://www.w3.org/2003/01/geo/wgs84_pos#";
    let lat = format!("{geo}lat");
    let lon = format!("{geo}long");
    let place = |ds: &mut MemDataset, id: &str, la: &str, lo: &str| {
        ds.add(iri(id), iri(&lat), typed_lit(la, XSD_DECIMAL));
        ds.add(iri(id), iri(&lon), typed_lit(lo, XSD_DECIMAL));
    };
    place(&mut ds, "http://ex/center", "35.681236", "139.767125");
    place(&mut ds, "http://ex/near1", "35.690000", "139.767125"); // 974.5 m
    place(&mut ds, "http://ex/near2", "35.681236", "139.780000"); // 1162.9 m
    place(&mut ds, "http://ex/far1", "35.750000", "139.800000"); // 8202.1 m
    place(&mut ds, "http://ex/far2", "35.681236", "139.900000"); // 12001.4 m
    ds
}

#[test]
fn distance_lat_lon_filter_selects_within_radius() {
    let ds = radius_dataset();
    let query = format!(
        "{}SELECT ?p WHERE {{ \
         ex:center geo:lat ?clat ; geo:long ?clon . \
         ?p geo:lat ?plat ; geo:long ?plon . \
         FILTER(oxgeo:distanceLatLon(?clat, ?clon, ?plat, ?plon) <= 2000.0) }}",
        prologue()
    );
    let rows = run(&query, &ds);
    let mut found: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get(&var("p")) {
            Some(Term::Iri(n)) => Some(n.as_str().to_string()),
            _ => None,
        })
        .collect();
    found.sort();
    // center (0 m), near1, near2 are all <= 2 km; far1/far2 are excluded.
    assert_eq!(
        found,
        vec![
            "http://ex/center".to_string(),
            "http://ex/near1".to_string(),
            "http://ex/near2".to_string(),
        ],
        "only points within 2 km must survive the FILTER"
    );
}

#[test]
fn distance_lat_lon_wrong_arity_excludes_row_in_filter() {
    let ds = radius_dataset();
    // Three args instead of four -> ordinary error -> row excluded.
    let query = format!(
        "{}SELECT ?p WHERE {{ \
         ex:center geo:lat ?clat ; geo:long ?clon . \
         ?p geo:lat ?plat ; geo:long ?plon . \
         FILTER(oxgeo:distanceLatLon(?clat, ?clon, ?plat) <= 2000.0) }}",
        prologue()
    );
    assert!(
        run(&query, &ds).is_empty(),
        "wrong-arity geo call must drop every row, not crash the query"
    );
}

#[test]
fn unknown_non_geo_function_still_errors() {
    // A genuinely unknown function name must remain an UnknownFunctionError, so a
    // FILTER over it fails the query loudly (no silent empty result).
    let ds = radius_dataset();
    let query = format!(
        "{}SELECT ?p WHERE {{ ?p geo:lat ?plat . \
         FILTER(ex:notAFunction(?plat) = 1) }}",
        prologue()
    );
    let mut parser = QueryParser::new();
    let parsed = parser.parse(&query).expect("should parse");
    let mut executor = QueryExecutor::new();
    assert!(
        executor.execute(&parsed.where_clause, &ds).is_err(),
        "an unknown function in FILTER must fail the query"
    );
}
