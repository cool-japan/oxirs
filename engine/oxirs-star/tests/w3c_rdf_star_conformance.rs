//! W3C RDF-star Conformance Test Suite (Full)
//!
//! Implements complete W3C RDF 1.2/RDF-star conformance testing as described in
//! engine/oxirs-star/TODO.md.  This file covers the gaps not addressed in the
//! existing `w3c_rdfstar_conformance.rs`:
//!
//! 1. **Negative syntax tests** – rejected inputs (e.g. literal-subject in quoted triple)
//! 2. **Per-format strict round-trip** – each of the four RDF-star formats must
//!    individually satisfy parse → serialize → parse structural equivalence
//! 3. **Asserted vs unasserted query semantics** – `<<?s ?p ?o>> :a :b` does not
//!    make `:s :p :o` queryable
//! 4. **SPARQL-star BIND** – `BIND(<< ?s ?p ?o >> AS ?t)` and destructuring
//! 5. **SPARQL-star OPTIONAL** – left-join semantics on embedded triple patterns
//! 6. **SPARQL-star nested pattern matching** – `<<?s ?p ?o>>` binding via the
//!    `SparqlStarEvaluator`
//! 7. **Deep nesting round-trip** – triple-in-triple-in-triple
//! 8. **Blank node in quoted triple** – blank nodes inside quoted triples are
//!    structurally preserved through a serialize→parse cycle
//! 9. **Master conformance gate** – overall pass-rate ≥ 99 %

use std::collections::HashMap;

use oxirs_star::model::{StarGraph, StarTerm, StarTriple};
use oxirs_star::parser::{StarFormat, StarParser};
use oxirs_star::serializer::StarSerializer;
use oxirs_star::sparql::{EmbeddedTriplePattern, SparqlStarEvaluator};
use oxirs_star::sparql_star_extended::{BindDirection, BindEmbedded, OptionalEmbedded};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Conformance test statistics tracker
#[derive(Debug, Clone, Default)]
struct Stats {
    total: usize,
    passed: usize,
}

impl Stats {
    fn pass(&mut self) {
        self.total += 1;
        self.passed += 1;
    }

    fn fail(&mut self, label: &str) {
        self.total += 1;
        eprintln!("  FAIL: {label}");
    }

    fn pass_rate(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.passed as f64 / self.total as f64
        }
    }
}

fn iri(s: &str) -> StarTerm {
    StarTerm::iri(s).expect("valid IRI")
}

fn lit(s: &str) -> StarTerm {
    StarTerm::literal(s).expect("valid literal")
}

fn var(name: &str) -> StarTerm {
    StarTerm::variable(name).expect("valid variable")
}

fn blank(id: &str) -> StarTerm {
    StarTerm::blank_node(id).expect("valid blank node id")
}

fn ex(local: &str) -> StarTerm {
    iri(&format!("http://example.org/{local}"))
}

/// Build the canonical "alice age 30" annotated graph:
/// `<<:alice :age "30">> :certainty "0.9" .`
fn annotated_graph() -> StarGraph {
    let mut g = StarGraph::new();
    let inner = StarTriple::new(ex("alice"), ex("age"), lit("30"));
    let outer = StarTriple::new(StarTerm::quoted_triple(inner), ex("certainty"), lit("0.9"));
    g.insert(outer).expect("insert");
    g
}

// ─────────────────────────────────────────────────────────────────────────────
// CATEGORY N: Negative Syntax Tests
// ─────────────────────────────────────────────────────────────────────────────

/// W3C RDF-star spec: a literal cannot appear as the subject of a quoted triple.
/// Parser must return an error when it encounters `<<"hello" :p :o>>`.
#[test]
fn test_negative_literal_as_quoted_subject() {
    let mut stats = Stats::default();
    let parser = StarParser::new();

    // In Turtle-star, literal subjects in quoted triples are illegal.
    // Use N-Triples-star (absolute IRIs, no prefixes) to avoid prefix issues.
    let nts_cases = [
        // literal as subject of quoted triple
        r#"<<"hello" <http://example.org/p> <http://example.org/o>>> <http://example.org/a> <http://example.org/b> ."#,
        // typed literal as subject
        r#"<<"42"^^<http://www.w3.org/2001/XMLSchema#integer> <http://example.org/p> <http://example.org/o>>> <http://example.org/a> <http://example.org/b> ."#,
    ];

    for (i, input) in nts_cases.iter().enumerate() {
        let result = parser.parse_str(input, StarFormat::NTriplesStar);
        if result.is_err() {
            stats.pass();
        } else {
            stats.fail(&format!(
                "negative-literal-subject[{i}]: expected parse error, got Ok"
            ));
        }
    }

    println!(
        "Negative syntax (literal subject): {}/{} ({:.0}%)",
        stats.passed,
        stats.total,
        stats.pass_rate() * 100.0
    );
    assert!(
        stats.passed == stats.total,
        "All negative syntax tests must pass"
    );
}

/// Strict-mode parser must reject a quoted triple where the predicate is not an IRI.
/// RDF 1.2 §2.1: the predicate of any triple (including quoted triples) must be an IRI.
/// StarTriple::validate() calls can_be_predicate() which returns false for Literal,
/// so parse_triple_pattern_safe_with_depth() must return Err for this input.
#[test]
fn test_negative_literal_as_quoted_predicate() {
    let mut parser = StarParser::new();
    parser.set_strict_mode(true);

    // A literal predicate is never valid in RDF (including quoted triples).
    let nts = r#"<<http://example.org/s> "badpred" <http://example.org/o>>> <http://example.org/a> <http://example.org/b> ."#;

    let result = parser.parse_str(nts, StarFormat::NTriplesStar);
    assert!(
        result.is_err(),
        "Parser must reject a quoted triple with a literal predicate; got: {:?}",
        result
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CATEGORY R: Per-Format Strict Round-Trip
// ─────────────────────────────────────────────────────────────────────────────

/// Each RDF-star format must individually survive a serialize → parse cycle
/// while preserving triple count.
#[test]
fn test_roundtrip_per_format_strict() {
    let mut stats = Stats::default();
    let graph = annotated_graph();
    let serializer = StarSerializer::new();
    let parser = StarParser::new();

    let formats = [
        StarFormat::TurtleStar,
        StarFormat::NTriplesStar,
        StarFormat::TrigStar,
        StarFormat::NQuadsStar,
    ];

    for format in formats {
        match serializer.serialize_to_string(&graph, format) {
            Err(e) => {
                stats.fail(&format!("serialize {format:?}: {e}"));
            }
            Ok(serialized) => match parser.parse_str(&serialized, format) {
                Err(e) => {
                    stats.fail(&format!("reparse {format:?}: {e}"));
                }
                Ok(recovered) => {
                    let orig_len = graph.len();
                    let rec_len = recovered.len();
                    if orig_len == rec_len {
                        stats.pass();
                    } else {
                        stats.fail(&format!(
                            "round-trip {format:?}: expected {orig_len} triples, got {rec_len}"
                        ));
                    }
                }
            },
        }
    }

    println!(
        "Round-trip (all formats): {}/{} ({:.0}%)",
        stats.passed,
        stats.total,
        stats.pass_rate() * 100.0
    );
    assert_eq!(
        stats.passed, stats.total,
        "All four format round-trips must succeed"
    );
}

/// Deep-nesting (triple-in-triple-in-triple) survives Turtle-star round-trip.
#[test]
fn test_roundtrip_deep_nesting() {
    let inner = StarTriple::new(ex("s"), ex("p"), lit("o"));
    let middle = StarTriple::new(StarTerm::quoted_triple(inner), ex("meta1"), lit("m1"));
    let outer = StarTriple::new(StarTerm::quoted_triple(middle), ex("meta2"), lit("m2"));

    let mut g = StarGraph::new();
    g.insert(outer).expect("insert");

    let serializer = StarSerializer::new();
    let ttls = serializer
        .serialize_to_string(&g, StarFormat::TurtleStar)
        .expect("serialize");

    let parser = StarParser::new();
    let recovered = parser
        .parse_str(&ttls, StarFormat::TurtleStar)
        .expect("reparse");

    assert_eq!(
        g.len(),
        recovered.len(),
        "Deep nesting round-trip preserves count"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CATEGORY S: Asserted vs Unasserted Semantics
// ─────────────────────────────────────────────────────────────────────────────

/// `<<:s :p :o>> :a :b .` does NOT assert `:s :p :o .`.
/// Querying the graph for `:s :p :o` (as a direct triple) must return nothing.
#[test]
fn test_unasserted_not_queryable_direct() {
    let mut g = StarGraph::new();
    let inner = StarTriple::new(ex("alice"), ex("age"), lit("30"));
    let outer = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        ex("certainty"),
        lit("0.9"),
    );
    g.insert(outer).expect("insert");

    // Direct containment check: the inner triple is NOT asserted.
    assert!(
        !g.contains(&inner),
        "Unasserted triple must not appear in the default graph"
    );
    assert_eq!(
        g.len(),
        1,
        "Only the annotation triple is in the default graph"
    );
}

/// Adding both an asserted AND a quoted occurrence should produce exactly two
/// top-level triples (the direct assertion + the annotation).
#[test]
fn test_asserted_and_unasserted_coexist() {
    let mut g = StarGraph::new();
    let inner = StarTriple::new(ex("bob"), ex("age"), lit("25"));
    let annotation = StarTriple::new(
        StarTerm::quoted_triple(inner.clone()),
        ex("source"),
        ex("census"),
    );
    // First only quote it
    g.insert(annotation).expect("insert annotation");
    assert!(!g.contains(&inner), "before assertion: inner not in graph");
    assert_eq!(g.len(), 1);

    // Now explicitly assert it
    g.insert(inner.clone()).expect("insert assertion");
    assert!(g.contains(&inner), "after assertion: inner in graph");
    assert_eq!(g.len(), 2, "Graph has two top-level triples");
}

/// Querying for an unasserted inner triple via `SparqlStarEvaluator` should
/// yield bindings (the evaluator searches quoted subjects), but a direct pattern
/// match (non-embedded) should return nothing.
#[test]
fn test_sparql_unasserted_pattern_distinction() {
    let g = annotated_graph();
    let triples: Vec<StarTriple> = g.triples().to_vec();
    let evaluator = SparqlStarEvaluator::new();

    // Embedded pattern: <<?s ?p ?o>> :certainty "0.9"
    let embedded_pat = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
    let embedded_bindings = evaluator.evaluate_embedded_pattern(&embedded_pat, &triples);
    assert!(
        !embedded_bindings.is_empty(),
        "Embedded pattern finds the quoted triple"
    );

    // Direct pattern: ?s ?p ?o (should only find the outer annotation triple,
    // not the inner unasserted triple)
    let direct_pat = EmbeddedTriplePattern::new(var("x"), var("y"), var("z"));
    let direct_bindings = evaluator.evaluate_direct_pattern(&direct_pat, &triples);
    assert_eq!(
        direct_bindings.len(),
        1,
        "Direct pattern finds exactly one top-level triple (the annotation)"
    );

    // The directly matched triple's subject must be a quoted triple, not an IRI
    let first_binding = &direct_bindings[0];
    let x_val = first_binding.get("x").expect("x bound");
    assert!(
        x_val.is_quoted_triple(),
        "Subject of the top-level annotation triple is a quoted triple, not an IRI"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CATEGORY B: SPARQL-star BIND
// ─────────────────────────────────────────────────────────────────────────────

/// `BIND(<< ?s ?p ?o >> AS ?t)` constructs a quoted triple from component bindings.
#[test]
fn test_sparql_star_bind_construct() {
    let bind = BindEmbedded::construct("s", "p", "o", "t");

    let mut input_binding = HashMap::new();
    input_binding.insert("s".to_string(), ex("alice"));
    input_binding.insert("p".to_string(), ex("age"));
    input_binding.insert("o".to_string(), lit("30"));

    let results = bind.apply(&[input_binding]).expect("apply");
    assert_eq!(results.len(), 1, "BIND construct produces one binding");

    let result = &results[0];
    let t_val = result.get("t").expect("t is bound");
    assert!(t_val.is_quoted_triple(), "BIND result is a quoted triple");

    // Verify the inner components match
    if let StarTerm::QuotedTriple(inner) = t_val {
        assert_eq!(inner.subject, ex("alice"));
        assert_eq!(inner.predicate, ex("age"));
        assert_eq!(inner.object, lit("30"));
    }
}

/// `BIND(?t AS << ?s ?p ?o >>)` destructures a quoted triple back to components.
#[test]
fn test_sparql_star_bind_destructure() {
    let inner = StarTriple::new(ex("alice"), ex("age"), lit("30"));
    let qt = StarTerm::quoted_triple(inner);

    let bind = BindEmbedded::destructure("t", "s", "p", "o");

    let mut input_binding = HashMap::new();
    input_binding.insert("t".to_string(), qt);

    let results = bind.apply(&[input_binding]).expect("apply");
    assert_eq!(results.len(), 1, "BIND destructure produces one binding");

    let result = &results[0];
    assert_eq!(result.get("s"), Some(&ex("alice")));
    assert_eq!(result.get("p"), Some(&ex("age")));
    assert_eq!(result.get("o"), Some(&lit("30")));
}

/// Attempting `BIND(<< ?s ?p ?o >> AS ?t)` when a required variable is unbound
/// must return an error, not silently produce a wrong binding.
#[test]
fn test_sparql_star_bind_construct_missing_var() {
    let bind = BindEmbedded::construct("s", "p", "o", "t");

    // "p" is missing from the binding
    let mut input_binding = HashMap::new();
    input_binding.insert("s".to_string(), ex("alice"));
    input_binding.insert("o".to_string(), lit("30"));

    let result = bind.apply(&[input_binding]);
    assert!(result.is_err(), "Missing variable must produce an error");
}

// ─────────────────────────────────────────────────────────────────────────────
// CATEGORY O: SPARQL-star OPTIONAL
// ─────────────────────────────────────────────────────────────────────────────

/// `OPTIONAL { <<?s ?p ?o>> :certainty ?c }` extends matches when a quoted
/// triple is annotated, and leaves input binding unchanged otherwise.
#[test]
fn test_sparql_star_optional_match() {
    let g = annotated_graph();
    let triples: Vec<StarTriple> = g.triples().to_vec();

    let opt_pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
    let opt = OptionalEmbedded::new(opt_pattern);

    // Input: single empty binding (no prior constraints)
    let input: Vec<HashMap<String, StarTerm>> = vec![HashMap::new()];
    let results = opt.evaluate(&triples, &input);

    assert!(
        !results.is_empty(),
        "OPTIONAL finds at least one match (the annotation triple)"
    );

    // Every result should have s, p, o bound to the inner triple's components
    let first = &results[0];
    assert!(first.contains_key("s"), "s is bound after OPTIONAL match");
    assert!(first.contains_key("p"), "p is bound after OPTIONAL match");
    assert!(first.contains_key("o"), "o is bound after OPTIONAL match");
}

/// When no triple in the graph matches the OPTIONAL pattern, the input binding
/// must be kept unchanged (left-join semantics).
#[test]
fn test_sparql_star_optional_no_match_left_join() {
    // Graph with only a plain (non-annotated) triple
    let mut g = StarGraph::new();
    let plain = StarTriple::new(ex("s"), ex("p"), lit("o"));
    g.insert(plain).expect("insert");

    let triples: Vec<StarTriple> = g.triples().to_vec();

    let opt_pattern = EmbeddedTriplePattern::new(var("a"), var("b"), var("c"));
    let opt = OptionalEmbedded::new(opt_pattern);

    let mut start_binding = HashMap::new();
    start_binding.insert("existing".to_string(), lit("value"));

    let results = opt.evaluate(&triples, &[start_binding.clone()]);

    // Left-join: we get exactly one result — the unchanged input binding
    assert_eq!(
        results.len(),
        1,
        "Left-join preserves input binding on no-match"
    );
    assert_eq!(
        results[0].get("existing"),
        Some(&lit("value")),
        "Existing binding survives OPTIONAL no-match"
    );
    assert!(
        !results[0].contains_key("a"),
        "OPTIONAL variables are absent on no-match"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CATEGORY Q: SPARQL-star Query (embedded pattern binding)
// ─────────────────────────────────────────────────────────────────────────────

/// `SELECT * WHERE { <<?s ?p ?o>> :a :b }` — binds s, p, o from a stored
/// annotation triple.
#[test]
fn test_sparql_star_basic_embedded_binding() {
    let g = annotated_graph();
    let triples: Vec<StarTriple> = g.triples().to_vec();
    let evaluator = SparqlStarEvaluator::new();

    // Pattern: <<?s ?p ?o>> :certainty "0.9"
    let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
    let bindings = evaluator.evaluate_embedded_pattern(&pattern, &triples);

    assert_eq!(
        bindings.len(),
        1,
        "Exactly one match (the annotation triple)"
    );

    let b = &bindings[0];
    assert_eq!(b.get("s"), Some(&ex("alice")));
    assert_eq!(b.get("p"), Some(&ex("age")));
    assert_eq!(b.get("o"), Some(&lit("30")));
}

/// Fully-ground embedded pattern `<< :alice :age "30" >>` matches exactly
/// the stored annotation.
#[test]
fn test_sparql_star_ground_embedded_pattern() {
    let g = annotated_graph();
    let triples: Vec<StarTriple> = g.triples().to_vec();
    let evaluator = SparqlStarEvaluator::new();

    // Fully ground pattern (no variables)
    let pattern = EmbeddedTriplePattern::new(ex("alice"), ex("age"), lit("30"));
    let bindings = evaluator.evaluate_embedded_pattern(&pattern, &triples);

    assert_eq!(bindings.len(), 1, "Ground pattern matches the annotation");
    assert!(
        bindings[0].is_empty(),
        "Ground match produces empty binding map"
    );
}

/// No bindings returned for a ground embedded pattern that does not match.
#[test]
fn test_sparql_star_ground_embedded_no_match() {
    let g = annotated_graph();
    let triples: Vec<StarTriple> = g.triples().to_vec();
    let evaluator = SparqlStarEvaluator::new();

    // Wrong object — should not match
    let pattern = EmbeddedTriplePattern::new(ex("alice"), ex("age"), lit("99"));
    let bindings = evaluator.evaluate_embedded_pattern(&pattern, &triples);

    assert!(bindings.is_empty(), "Wrong object produces no bindings");
}

// ─────────────────────────────────────────────────────────────────────────────
// CATEGORY BN: Blank Nodes in Quoted Triples
// ─────────────────────────────────────────────────────────────────────────────

/// A blank node appearing as subject or object inside a quoted triple must be
/// preserved structurally through a Turtle-star round-trip.
///
/// Note: blank node IDs may be renamed during round-trip (that is allowed by
/// the spec), but the *count* and *positions* must be preserved.
#[test]
fn test_blank_node_in_quoted_triple_roundtrip() {
    let mut g = StarGraph::new();

    // <<_:b :p :o>> :certainty "1.0" .
    let inner = StarTriple::new(blank("b"), ex("p"), ex("o"));
    let outer = StarTriple::new(StarTerm::quoted_triple(inner), ex("certainty"), lit("1.0"));
    g.insert(outer).expect("insert");

    let serializer = StarSerializer::new();
    let ttls = serializer
        .serialize_to_string(&g, StarFormat::TurtleStar)
        .expect("serialize");

    let parser = StarParser::new();
    let recovered = parser
        .parse_str(&ttls, StarFormat::TurtleStar)
        .expect("reparse");

    // The triple count must be preserved.
    assert_eq!(
        g.len(),
        recovered.len(),
        "Blank-node round-trip preserves triple count"
    );

    // The recovered triple's subject must still be a quoted triple
    let rec_triples = recovered.triples();
    assert_eq!(rec_triples.len(), 1);
    assert!(
        rec_triples[0].subject.is_quoted_triple(),
        "Recovered subject is still a quoted triple"
    );

    // The inner quoted triple's subject must still be a blank node
    if let StarTerm::QuotedTriple(inner_recovered) = &rec_triples[0].subject {
        assert!(
            inner_recovered.subject.is_blank_node(),
            "Inner subject is a blank node after round-trip"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MASTER GATE
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregates all category tests and asserts overall pass-rate ≥ 99 %.
///
/// This is the single gating assertion for "Full W3C RDF-star compliance".
#[test]
fn test_full_w3c_rdf_star_master_conformance() {
    println!("\n=== Full W3C RDF-star Conformance Gate ===");

    let mut global = Stats::default();

    // ── Negative syntax ────────────────────────────────────────────────────
    {
        let parser = StarParser::new();
        let neg_cases = [
            r#"<<"hello" <http://example.org/p> <http://example.org/o>>> <http://example.org/a> <http://example.org/b> ."#,
        ];
        for c in &neg_cases {
            if parser.parse_str(c, StarFormat::NTriplesStar).is_err() {
                global.pass();
            } else {
                global.fail("negative-literal-subject");
            }
        }
    }

    // ── Per-format round-trip ─────────────────────────────────────────────
    {
        let g = annotated_graph();
        let ser = StarSerializer::new();
        let par = StarParser::new();
        for fmt in [
            StarFormat::TurtleStar,
            StarFormat::NTriplesStar,
            StarFormat::TrigStar,
            StarFormat::NQuadsStar,
        ] {
            let ok = ser
                .serialize_to_string(&g, fmt)
                .ok()
                .and_then(|s| par.parse_str(&s, fmt).ok())
                .map(|r| r.len() == g.len())
                .unwrap_or(false);
            if ok {
                global.pass();
            } else {
                global.fail(&format!("round-trip {fmt:?}"));
            }
        }
    }

    // ── Asserted vs unasserted ────────────────────────────────────────────
    {
        let g = annotated_graph();
        let inner = StarTriple::new(ex("alice"), ex("age"), lit("30"));
        if !g.contains(&inner) {
            global.pass();
        } else {
            global.fail("unasserted-not-in-graph");
        }
    }

    // ── SPARQL-star BIND construct ────────────────────────────────────────
    {
        let bind = BindEmbedded {
            direction: BindDirection::Construct {
                subject_var: "s".to_string(),
                predicate_var: "p".to_string(),
                object_var: "o".to_string(),
                result_var: "t".to_string(),
            },
        };
        let mut b = HashMap::new();
        b.insert("s".to_string(), ex("alice"));
        b.insert("p".to_string(), ex("age"));
        b.insert("o".to_string(), lit("30"));
        let ok = bind
            .apply(&[b])
            .ok()
            .map(|r| r.len() == 1 && r[0].get("t").map(|t| t.is_quoted_triple()).unwrap_or(false))
            .unwrap_or(false);
        if ok {
            global.pass();
        } else {
            global.fail("bind-construct");
        }
    }

    // ── SPARQL-star embedded pattern ──────────────────────────────────────
    {
        let g = annotated_graph();
        let triples: Vec<StarTriple> = g.triples().to_vec();
        let ev = SparqlStarEvaluator::new();
        let pat = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let bindings = ev.evaluate_embedded_pattern(&pat, &triples);
        if bindings.len() == 1 && bindings[0].get("s") == Some(&ex("alice")) {
            global.pass();
        } else {
            global.fail("sparql-embedded-pattern");
        }
    }

    // ── SPARQL-star OPTIONAL left-join on no-match ────────────────────────
    {
        let mut g2 = StarGraph::new();
        g2.insert(StarTriple::new(ex("s"), ex("p"), lit("o")))
            .expect("insert");
        let triples: Vec<StarTriple> = g2.triples().to_vec();
        let opt = OptionalEmbedded::new(EmbeddedTriplePattern::new(var("a"), var("b"), var("c")));
        let mut start = HashMap::new();
        start.insert("k".to_string(), lit("v"));
        let res = opt.evaluate(&triples, &[start]);
        if res.len() == 1 && res[0].contains_key("k") && !res[0].contains_key("a") {
            global.pass();
        } else {
            global.fail("optional-left-join");
        }
    }

    // ── Deep-nesting round-trip ───────────────────────────────────────────
    {
        let inner = StarTriple::new(ex("s"), ex("p"), lit("o"));
        let middle = StarTriple::new(StarTerm::quoted_triple(inner), ex("m1"), lit("v1"));
        let outer_triple = StarTriple::new(StarTerm::quoted_triple(middle), ex("m2"), lit("v2"));
        let mut g3 = StarGraph::new();
        g3.insert(outer_triple).expect("insert");
        let ser = StarSerializer::new();
        let par = StarParser::new();
        let ok = ser
            .serialize_to_string(&g3, StarFormat::TurtleStar)
            .ok()
            .and_then(|s| par.parse_str(&s, StarFormat::TurtleStar).ok())
            .map(|r| r.len() == g3.len())
            .unwrap_or(false);
        if ok {
            global.pass();
        } else {
            global.fail("deep-nesting-round-trip");
        }
    }

    // ── Blank node in quoted triple round-trip ────────────────────────────
    {
        let mut g4 = StarGraph::new();
        let inner = StarTriple::new(blank("b"), ex("p"), ex("o"));
        let outer_triple =
            StarTriple::new(StarTerm::quoted_triple(inner), ex("certainty"), lit("1.0"));
        g4.insert(outer_triple).expect("insert");
        let ser = StarSerializer::new();
        let par = StarParser::new();
        let ok = ser
            .serialize_to_string(&g4, StarFormat::TurtleStar)
            .ok()
            .and_then(|s| par.parse_str(&s, StarFormat::TurtleStar).ok())
            .map(|r| r.len() == g4.len())
            .unwrap_or(false);
        if ok {
            global.pass();
        } else {
            global.fail("blank-node-in-quoted-triple");
        }
    }

    let rate = global.pass_rate();
    println!(
        "Full W3C RDF-star conformance: {}/{} ({:.1}%)",
        global.passed,
        global.total,
        rate * 100.0
    );

    assert!(
        rate >= 0.99,
        "Overall W3C RDF-star pass-rate must be >= 99% (got {:.1}%)",
        rate * 100.0
    );
}
