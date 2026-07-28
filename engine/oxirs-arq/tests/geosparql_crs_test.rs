//! GeoSPARQL CRS axis-order tests for `geof:distance` (adversarial finding P2).
//!
//! A `geo:wktLiteral` may carry a leading CRS URI, and the two WGS84 CRSs in
//! practical use disagree on coordinate order:
//!
//! * **CRS84** (`…/OGC/1.3/CRS84`) and an untagged literal → `(longitude, latitude)`.
//! * **EPSG:4326** (`…/EPSG/0/4326`) → `(latitude, longitude)` — the *opposite* order.
//!
//! `parse_wkt_point` previously discarded the CRS URI and always read
//! `(lon, lat)`, so an EPSG:4326 point had its axes swapped and `geof:distance`
//! returned a large, plausible-looking wrong answer. These tests pin down that a
//! point written under either CRS denotes the *same* location (identical
//! distance), and that an unrecognised CRS is an ordinary evaluation error — so
//! inside a `FILTER` the row is excluded rather than silently mis-projected.
//!
//! Expected distance is the Haversine great-circle value on a spherical Earth of
//! mean radius 6,371,000 m (the same oracle as `geosparql_test.rs`).

use oxirs_arq::algebra::{PropertyPath, Term};
use oxirs_arq::query::QueryParser;
use oxirs_arq::{Dataset, Literal, QueryExecutor, TriplePattern, Variable};
use oxirs_core::model::NamedNode;
use std::collections::HashMap;

// ── Namespaces ───────────────────────────────────────────────────────────────

const GEOF: &str = "http://www.opengis.net/def/function/geosparql/";
const UOM: &str = "http://www.opengis.net/def/uom/OGC/1.0/";
const WKT_LITERAL: &str = "http://www.opengis.net/ont/geosparql#wktLiteral";

const CRS84_URI: &str = "http://www.opengis.net/def/crs/OGC/1.3/CRS84";
const EPSG_4326_URI: &str = "http://www.opengis.net/def/crs/EPSG/0/4326";
// A recognised-but-unsupported CRS (Web Mercator) to exercise the error path.
const EPSG_3857_URI: &str = "http://www.opengis.net/def/crs/EPSG/0/3857";

fn prologue() -> String {
    format!(
        "PREFIX ex: <http://ex/>\n\
         PREFIX geof: <{GEOF}>\n\
         PREFIX uom: <{UOM}>\n"
    )
}

// ── In-memory dataset mock (mirrors geosparql_test.rs) ───────────────────────

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

fn wkt(value: &str) -> Term {
    typed_lit(value, WKT_LITERAL)
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

// Tokyo Station and Osaka Station (WGS84); great-circle distance 403057.527 m.
const TOKYO_LAT: &str = "35.681236";
const TOKYO_LON: &str = "139.767125";
const OSAKA_LAT: &str = "34.702485";
const OSAKA_LON: &str = "135.495951";
const TOKYO_OSAKA_METRES: f64 = 403_057.527;

/// Run `geof:distance(ex:a, ex:b, uom:metre)` over two WKT literals.
fn distance_metres(a_wkt: &str, b_wkt: &str) -> f64 {
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    ds.add(iri("http://ex/a"), iri("http://ex/wkt"), wkt(a_wkt));
    ds.add(iri("http://ex/b"), iri("http://ex/wkt"), wkt(b_wkt));
    let query = format!(
        "{}SELECT ?d WHERE {{ ex:a ex:wkt ?wa . ex:b ex:wkt ?wb . \
         BIND(geof:distance(?wa, ?wb, uom:metre) AS ?d) }}",
        prologue()
    );
    single_number(&run(&query, &ds), "d")
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// An EPSG:4326 point (axis order lat/lon) must resolve to the same location as
/// the CRS84 encoding of the same place: distance between the two encodings of
/// Tokyo is essentially zero. If the CRS were ignored (the old bug), the
/// EPSG:4326 coordinates would be read as (lon, lat) and land far away.
#[test]
fn epsg4326_and_crs84_encode_the_same_location() {
    let crs84_tokyo = format!("<{CRS84_URI}> POINT({TOKYO_LON} {TOKYO_LAT})");
    let epsg_tokyo = format!("<{EPSG_4326_URI}> POINT({TOKYO_LAT} {TOKYO_LON})");
    let d = distance_metres(&crs84_tokyo, &epsg_tokyo);
    assert!(
        d < 1.0,
        "CRS84 and EPSG:4326 encodings of Tokyo must coincide, got {d} m apart"
    );
}

/// An untagged WKT literal defaults to CRS84 (lon, lat): it must coincide with an
/// explicit CRS84 literal of the same point.
#[test]
fn untagged_wkt_defaults_to_crs84() {
    let untagged = format!("POINT({TOKYO_LON} {TOKYO_LAT})");
    let crs84 = format!("<{CRS84_URI}> POINT({TOKYO_LON} {TOKYO_LAT})");
    let d = distance_metres(&untagged, &crs84);
    assert!(
        d < 1.0,
        "an untagged POINT must default to CRS84, got {d} m apart"
    );
}

/// The Tokyo–Osaka distance is the same whichever CRS each endpoint is written
/// in — mixed CRS84 / EPSG:4326 / untagged encodings all yield the oracle value.
#[test]
fn tokyo_osaka_distance_is_crs_independent() {
    let crs84_tokyo = format!("<{CRS84_URI}> POINT({TOKYO_LON} {TOKYO_LAT})");
    let untagged_tokyo = format!("POINT({TOKYO_LON} {TOKYO_LAT})");
    let epsg_tokyo = format!("<{EPSG_4326_URI}> POINT({TOKYO_LAT} {TOKYO_LON})");
    let crs84_osaka = format!("<{CRS84_URI}> POINT({OSAKA_LON} {OSAKA_LAT})");
    let epsg_osaka = format!("<{EPSG_4326_URI}> POINT({OSAKA_LAT} {OSAKA_LON})");

    for (a, b, label) in [
        (&crs84_tokyo, &crs84_osaka, "CRS84/CRS84"),
        (&epsg_tokyo, &epsg_osaka, "EPSG4326/EPSG4326"),
        (&untagged_tokyo, &epsg_osaka, "untagged/EPSG4326"),
        (&epsg_tokyo, &crs84_osaka, "EPSG4326/CRS84"),
    ] {
        let d = distance_metres(a, b);
        assert!(
            (d - TOKYO_OSAKA_METRES).abs() < 1.0,
            "{label}: distance {d} m should be ~{TOKYO_OSAKA_METRES} m regardless of CRS"
        );
    }
}

/// A recognised-but-unsupported CRS (here EPSG:3857, Web Mercator) is an ordinary
/// evaluation error, so under SPARQL §17.3 a `FILTER` referencing such a point
/// drops the row rather than silently mis-projecting it or aborting the query.
#[test]
fn unknown_crs_excludes_row_in_filter() {
    let mut ds = MemDataset {
        triples: Vec::new(),
    };
    ds.add(
        iri("http://ex/tokyo"),
        iri("http://ex/wkt"),
        wkt(&format!("<{CRS84_URI}> POINT({TOKYO_LON} {TOKYO_LAT})")),
    );
    ds.add(
        iri("http://ex/osaka"),
        iri("http://ex/wkt"),
        wkt(&format!("<{CRS84_URI}> POINT({OSAKA_LON} {OSAKA_LAT})")),
    );
    ds.add(
        iri("http://ex/weird"),
        iri("http://ex/wkt"),
        wkt(&format!("<{EPSG_3857_URI}> POINT(15557410 4257480)")),
    );

    let query = format!(
        "{}SELECT ?d WHERE {{ \
         ex:tokyo ex:wkt ?wa . ex:osaka ex:wkt ?wb . ex:weird ex:wkt ?wc . \
         BIND(geof:distance(?wa, ?wb, uom:metre) AS ?d) \
         FILTER(geof:distance(?wa, ?wc, uom:metre) < 9999999999.0) }}",
        prologue()
    );
    assert!(
        run(&query, &ds).is_empty(),
        "an unsupported-CRS point in a FILTER must drop the row, not crash the query"
    );
}
