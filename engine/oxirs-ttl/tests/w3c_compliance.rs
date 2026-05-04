//! W3C RDF 1.1 / 1.2 compliance driver for `oxirs-ttl`.
//!
//! This integration test loads the fixture corpus rooted at
//! `tests/fixtures/w3c-rdf-tests/` and dispatches each entry by its
//! `mf:Test` subtype, asserting a per-format pass rate of at least 99%.
//!
//! The corpus is hand-curated to mirror the structure of the official W3C RDF
//! Test Suite (see manifest header comments for the pinned reference snapshot).
//! It exercises each of the four mainstream Turtle-family formats — Turtle,
//! N-Triples, N-Quads, TriG — plus the RDF-star (RDF 1.2) extensions where
//! applicable. N3 is intentionally not part of the W3C suite; see the
//! crate-level documentation.
//!
//! # Test categories
//!
//! - `*PositiveSyntax`  — input must parse without error
//! - `*NegativeSyntax`  — input must fail to parse with a typed error
//! - `*Eval`            — input must parse and the resulting graph must match
//!   the expected N-Triples / N-Quads result (compared as a set, modulo
//!   blank-node label renaming)
//!
//! # Running
//!
//! ```text
//! cargo nextest run -p oxirs-ttl --test w3c_compliance
//! ```

use oxirs_core::model::{GraphName, NamedNode, Object, Predicate, Quad, Subject, Triple};
use oxirs_ttl::nquads::{NQuadsParser, NQuadsSerializer};
use oxirs_ttl::ntriples::{NTriplesParser, NTriplesSerializer};
use oxirs_ttl::toolkit::{Parser as ToolkitParser, Serializer as ToolkitSerializer};
use oxirs_ttl::trig::TriGParser;
use oxirs_ttl::turtle::{TurtleParser, TurtleSerializer};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// Required pass rate (W3C compliance certification target).
const PASS_RATE_TARGET: f64 = 0.99;

/// Per-test outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Pass,
    Fail(String),
}

/// Per-test record kept by the driver for reporting.
#[derive(Debug, Clone)]
struct TestResult {
    id: String,
    name: String,
    category: TestCategory,
    outcome: Outcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TestCategory {
    PositiveSyntax,
    NegativeSyntax,
    Eval,
}

/// Manifest entry parsed from `manifest.ttl`.
#[derive(Debug, Clone)]
struct ManifestEntry {
    id: String,
    test_type: String,
    /// Human-readable name from `mf:name`, surfaced in failure diagnostics.
    name: String,
    action: PathBuf,
    result: Option<PathBuf>,
}

/// Locate the fixture root (works regardless of cwd at test time).
fn fixture_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("tests")
        .join("fixtures")
        .join("w3c-rdf-tests")
}

/// Read a fixture file as UTF-8 string.
fn read_fixture(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {e}", path.display()))
}

/// Parse a manifest.ttl file into a list of [`ManifestEntry`] in declared order.
///
/// The manifest is a Turtle document with the W3C `mf:`/`rdft:` vocabulary; we
/// re-use the crate's own Turtle parser to avoid needing a second tool here.
fn load_manifest(format_dir: &Path) -> Vec<ManifestEntry> {
    let manifest_path = format_dir.join("manifest.ttl");
    let manifest_text = read_fixture(&manifest_path);

    let parser = TurtleParser::new();
    let triples = parser.parse_document(&manifest_text).unwrap_or_else(|e| {
        panic!(
            "Manifest parse failed for {}: {e:?}",
            manifest_path.display()
        )
    });

    // We need ordered traversal of mf:entries, but we also need the per-entry
    // mf:name / mf:action / mf:result / rdf:type fields.
    let mut by_subject: HashMap<String, HashMap<String, Vec<Object>>> = HashMap::new();
    for triple in &triples {
        let subj = format_subject(triple.subject());
        let pred = match triple.predicate() {
            Predicate::NamedNode(n) => n.as_str().to_string(),
            Predicate::Variable(_) => continue,
        };
        by_subject
            .entry(subj)
            .or_default()
            .entry(pred)
            .or_default()
            .push(triple.object().clone());
    }

    // Find the manifest subject (the one that has an `mf:entries` predicate).
    let entries_iri = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#entries";
    let (entries_subject, entries_objects) = by_subject
        .iter()
        .find_map(|(subj, preds)| {
            preds
                .get(entries_iri)
                .map(|objs| (subj.clone(), objs.clone()))
        })
        .unwrap_or_else(|| {
            panic!(
                "Manifest {} has no mf:entries predicate",
                manifest_path.display()
            )
        });

    let entries_head = entries_objects
        .first()
        .unwrap_or_else(|| panic!("Manifest {entries_subject} mf:entries is empty"))
        .clone();
    let ordered_ids = collect_rdf_list(&entries_head, &by_subject);

    let mut entries = Vec::new();
    for id in ordered_ids {
        let preds = by_subject
            .get(&id)
            .unwrap_or_else(|| panic!("Manifest entry {id} has no associated triples"));

        let test_type = first_iri(preds, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
            .unwrap_or_else(|| panic!("Entry {id} has no rdf:type"));
        let name = first_string(
            preds,
            "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#name",
        )
        .unwrap_or_else(|| id.clone());
        let action_iri = first_iri(
            preds,
            "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#action",
        )
        .unwrap_or_else(|| panic!("Entry {id} has no mf:action"));
        let action_path = format_dir.join(strip_iri_to_filename(&action_iri));

        let result_path = first_iri(
            preds,
            "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#result",
        )
        .map(|iri| format_dir.join(strip_iri_to_filename(&iri)));

        entries.push(ManifestEntry {
            id,
            test_type,
            name,
            action: action_path,
            result: result_path,
        });
    }

    entries
}

/// Walk an RDF list (rdf:first / rdf:rest / rdf:nil) starting from a head node.
///
/// The returned IDs are formatted as `<iri>` (matching the keys produced by
/// [`format_subject`]) so they can be used directly as `by_subject` lookup keys.
fn collect_rdf_list(
    head: &Object,
    by_subject: &HashMap<String, HashMap<String, Vec<Object>>>,
) -> Vec<String> {
    let rdf_first = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first";
    let rdf_rest = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest";
    let rdf_nil = "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil";

    let mut out = Vec::new();
    let mut current = head.clone();
    loop {
        let key = format_object(&current);
        if key == format!("<{rdf_nil}>") {
            break;
        }
        let preds = match by_subject.get(&key) {
            Some(v) => v,
            None => break,
        };
        if let Some(Object::NamedNode(nn)) = preds.get(rdf_first).and_then(|v| v.first()) {
            // Each entry id is a NamedNode; format it as <iri> to match
            // format_subject's output.
            out.push(format!("<{}>", nn.as_str()));
        }
        match preds.get(rdf_rest).and_then(|v| v.first()) {
            Some(rest) => current = rest.clone(),
            None => break,
        }
    }
    out
}

fn first_iri(preds: &HashMap<String, Vec<Object>>, pred: &str) -> Option<String> {
    preds.get(pred).and_then(|objs| {
        objs.iter().find_map(|o| match o {
            Object::NamedNode(nn) => Some(nn.as_str().to_string()),
            _ => None,
        })
    })
}

fn first_string(preds: &HashMap<String, Vec<Object>>, pred: &str) -> Option<String> {
    preds.get(pred).and_then(|objs| {
        objs.iter().find_map(|o| match o {
            Object::Literal(l) => Some(l.value().to_string()),
            _ => None,
        })
    })
}

fn format_subject(subject: &Subject) -> String {
    match subject {
        Subject::NamedNode(n) => format!("<{}>", n.as_str()),
        Subject::BlankNode(b) => format!("_:{}", b.as_str()),
        Subject::Variable(v) => format!("?{}", v.as_str()),
        Subject::QuotedTriple(_) => "<<quoted>>".to_string(),
    }
}

fn format_object(object: &Object) -> String {
    match object {
        Object::NamedNode(n) => format!("<{}>", n.as_str()),
        Object::BlankNode(b) => format!("_:{}", b.as_str()),
        Object::Literal(l) => format!("\"{}\"", l.value()),
        Object::Variable(v) => format!("?{}", v.as_str()),
        Object::QuotedTriple(_) => "<<quoted>>".to_string(),
    }
}

/// Strip a relative IRI down to its filename component.
///
/// Manifest entries reference fixtures via `<filename.ttl>` (relative IRIs),
/// which the parser resolves into something like the file's path stub. We just
/// take everything after the last `/`.
fn strip_iri_to_filename(iri: &str) -> String {
    iri.rsplit('/').next().unwrap_or(iri).to_string()
}

/// Compute pass rate over a vector of [`TestResult`].
fn pass_rate(results: &[TestResult]) -> f64 {
    if results.is_empty() {
        return 1.0;
    }
    let passes = results
        .iter()
        .filter(|r| matches!(r.outcome, Outcome::Pass))
        .count();
    passes as f64 / results.len() as f64
}

/// Print the failures and assert pass rate.
fn assert_pass_rate(format: &str, results: &[TestResult]) {
    let rate = pass_rate(results);
    let total = results.len();
    let passed = results
        .iter()
        .filter(|r| matches!(r.outcome, Outcome::Pass))
        .count();

    let failures: Vec<&TestResult> = results
        .iter()
        .filter(|r| !matches!(r.outcome, Outcome::Pass))
        .collect();

    eprintln!(
        "[w3c-{format}] pass {passed}/{total} ({:.2}%)",
        rate * 100.0
    );
    for failure in &failures {
        if let Outcome::Fail(reason) = &failure.outcome {
            eprintln!(
                "  FAIL {} ({:?}) [{}]: {reason}",
                failure.id, failure.category, failure.name,
            );
        }
    }

    assert!(
        rate >= PASS_RATE_TARGET,
        "[w3c-{format}] pass rate {:.4} < target {:.4} ({} of {} failures)",
        rate,
        PASS_RATE_TARGET,
        failures.len(),
        total,
    );
}

/// Categorise a manifest entry by its `rdft:` test type.
fn categorize(test_type: &str) -> Option<TestCategory> {
    if test_type.ends_with("PositiveSyntax") {
        Some(TestCategory::PositiveSyntax)
    } else if test_type.ends_with("NegativeSyntax") {
        Some(TestCategory::NegativeSyntax)
    } else if test_type.ends_with("Eval") {
        Some(TestCategory::Eval)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Triple / quad set normalisation for evaluation comparisons
// ---------------------------------------------------------------------------

/// Canonicalise a triple set for comparison, mapping blank-node labels to a
/// stable order-independent representation. This is a simplified canonicalisation
/// suitable for the small fixture graphs in this corpus (no isomorphism
/// resolution needed in practice — fixtures use distinct blank-node labels).
fn canonical_triple_set(triples: &[Triple]) -> BTreeSet<String> {
    triples.iter().map(canonical_triple_string).collect()
}

fn canonical_quad_set(quads: &[Quad]) -> BTreeSet<String> {
    quads.iter().map(canonical_quad_string).collect()
}

fn canonical_triple_string(t: &Triple) -> String {
    format!(
        "{} {} {}",
        canonical_subject(t.subject()),
        canonical_predicate(t.predicate()),
        canonical_object(t.object()),
    )
}

fn canonical_quad_string(q: &Quad) -> String {
    let graph = match q.graph_name() {
        GraphName::DefaultGraph => "DEFAULT".to_string(),
        GraphName::NamedNode(n) => format!("<{}>", n.as_str()),
        GraphName::BlankNode(_) => "_:bn".to_string(), // collapse — fixtures
        GraphName::Variable(v) => format!("?{}", v.as_str()),
    };
    format!(
        "{} {} {} {}",
        canonical_subject(q.subject()),
        canonical_predicate(q.predicate()),
        canonical_object(q.object()),
        graph,
    )
}

fn canonical_subject(s: &Subject) -> String {
    match s {
        Subject::NamedNode(n) => format!("<{}>", n.as_str()),
        Subject::BlankNode(_) => "_:bn".to_string(),
        Subject::Variable(v) => format!("?{}", v.as_str()),
        Subject::QuotedTriple(qt) => format!("<<{}>>", canonical_triple_string(&qt.as_triple())),
    }
}

fn canonical_predicate(p: &Predicate) -> String {
    match p {
        Predicate::NamedNode(n) => format!("<{}>", n.as_str()),
        Predicate::Variable(v) => format!("?{}", v.as_str()),
    }
}

fn canonical_object(o: &Object) -> String {
    match o {
        Object::NamedNode(n) => format!("<{}>", n.as_str()),
        Object::BlankNode(_) => "_:bn".to_string(),
        Object::Literal(l) => {
            if let Some(lang) = l.language() {
                format!("\"{}\"@{lang}", l.value())
            } else if l.datatype().as_str() == "http://www.w3.org/2001/XMLSchema#string" {
                format!("\"{}\"", l.value())
            } else {
                format!("\"{}\"^^<{}>", l.value(), l.datatype().as_str())
            }
        }
        Object::Variable(v) => format!("?{}", v.as_str()),
        Object::QuotedTriple(qt) => format!("<<{}>>", canonical_triple_string(&qt.as_triple())),
    }
}

// Helper: extract inner Triple from QuotedTriple via Display-equivalent fields.
trait QuotedTripleExt {
    fn as_triple(&self) -> Triple;
}

impl QuotedTripleExt for oxirs_core::model::QuotedTriple {
    fn as_triple(&self) -> Triple {
        Triple::new(
            self.subject().clone(),
            Predicate::NamedNode(match self.predicate() {
                Predicate::NamedNode(n) => n.clone(),
                Predicate::Variable(_) => NamedNode::new("urn:variable").expect("static IRI"),
            }),
            self.object().clone(),
        )
    }
}

// ---------------------------------------------------------------------------
// Per-format runners
// ---------------------------------------------------------------------------

fn run_turtle_entry(entry: &ManifestEntry, root: &Path) -> TestResult {
    let _ = root;
    let category = match categorize(&entry.test_type) {
        Some(c) => c,
        None => {
            return TestResult {
                id: entry.id.clone(),
                name: entry.name.clone(),
                category: TestCategory::PositiveSyntax,
                outcome: Outcome::Fail(format!("unknown test type {}", entry.test_type)),
            };
        }
    };

    let input = read_fixture(&entry.action);
    let parser = TurtleParser::new();
    let parsed = parser.parse_document(&input);

    let outcome = match (&category, parsed) {
        (TestCategory::PositiveSyntax, Ok(_)) => Outcome::Pass,
        (TestCategory::PositiveSyntax, Err(e)) => {
            Outcome::Fail(format!("expected parse OK; got error: {e}"))
        }
        (TestCategory::NegativeSyntax, Ok(triples)) => Outcome::Fail(format!(
            "expected parse error; got {} triples",
            triples.len()
        )),
        (TestCategory::NegativeSyntax, Err(_)) => Outcome::Pass,
        (TestCategory::Eval, Ok(actual)) => match &entry.result {
            None => Outcome::Fail("Eval test missing mf:result".to_string()),
            Some(result_path) => {
                let expected_text = read_fixture(result_path);
                let nt_parser = NTriplesParser::new();
                let expected: Vec<Triple> =
                    match nt_parser.parse(Cursor::new(expected_text.as_bytes())) {
                        Ok(t) => t,
                        Err(e) => {
                            return TestResult {
                                id: entry.id.clone(),
                                name: entry.name.clone(),
                                category,
                                outcome: Outcome::Fail(format!("expected file parse error: {e}")),
                            };
                        }
                    };
                if canonical_triple_set(&actual) == canonical_triple_set(&expected) {
                    Outcome::Pass
                } else {
                    Outcome::Fail(format!(
                        "expected {} triples, got {} (canonical sets differ)",
                        expected.len(),
                        actual.len()
                    ))
                }
            }
        },
        (TestCategory::Eval, Err(e)) => Outcome::Fail(format!("Eval parse failed: {e}")),
    };

    TestResult {
        id: entry.id.clone(),
        name: entry.name.clone(),
        category,
        outcome,
    }
}

fn run_ntriples_entry(entry: &ManifestEntry, _root: &Path) -> TestResult {
    let category = match categorize(&entry.test_type) {
        Some(c) => c,
        None => {
            return TestResult {
                id: entry.id.clone(),
                name: entry.name.clone(),
                category: TestCategory::PositiveSyntax,
                outcome: Outcome::Fail(format!("unknown test type {}", entry.test_type)),
            };
        }
    };

    let input = read_fixture(&entry.action);
    let parser = NTriplesParser::new();
    let parsed: Result<Vec<Triple>, _> = parser.parse(Cursor::new(input.as_bytes()));

    let outcome = match (&category, parsed) {
        (TestCategory::PositiveSyntax, Ok(_)) => Outcome::Pass,
        (TestCategory::PositiveSyntax, Err(e)) => {
            Outcome::Fail(format!("expected parse OK; got error: {e}"))
        }
        (TestCategory::NegativeSyntax, Ok(triples)) => Outcome::Fail(format!(
            "expected parse error; got {} triples",
            triples.len()
        )),
        (TestCategory::NegativeSyntax, Err(_)) => Outcome::Pass,
        (TestCategory::Eval, Ok(actual)) => match &entry.result {
            None => Outcome::Fail("Eval test missing mf:result".to_string()),
            Some(result_path) => {
                let expected_text = read_fixture(result_path);
                let exp_parser = NTriplesParser::new();
                let expected: Vec<Triple> =
                    match exp_parser.parse(Cursor::new(expected_text.as_bytes())) {
                        Ok(t) => t,
                        Err(e) => {
                            return TestResult {
                                id: entry.id.clone(),
                                name: entry.name.clone(),
                                category,
                                outcome: Outcome::Fail(format!("expected file parse error: {e}")),
                            };
                        }
                    };

                // Round-trip through the serializer to validate output stability too.
                let serializer = NTriplesSerializer::new();
                let mut buf = Vec::new();
                if let Err(e) = serializer.serialize(&actual, &mut buf) {
                    return TestResult {
                        id: entry.id.clone(),
                        name: entry.name.clone(),
                        category,
                        outcome: Outcome::Fail(format!("re-serialise failed: {e}")),
                    };
                }
                let reparsed: Result<Vec<Triple>, _> =
                    NTriplesParser::new().parse(Cursor::new(buf.as_slice()));
                let reparsed = match reparsed {
                    Ok(t) => t,
                    Err(e) => {
                        return TestResult {
                            id: entry.id.clone(),
                            name: entry.name.clone(),
                            category,
                            outcome: Outcome::Fail(format!("re-parse failed: {e}")),
                        };
                    }
                };
                if canonical_triple_set(&reparsed) == canonical_triple_set(&expected) {
                    Outcome::Pass
                } else {
                    Outcome::Fail(format!(
                        "round-trip differs: expected {} triples, got {}",
                        expected.len(),
                        reparsed.len()
                    ))
                }
            }
        },
        (TestCategory::Eval, Err(e)) => Outcome::Fail(format!("Eval parse failed: {e}")),
    };

    TestResult {
        id: entry.id.clone(),
        name: entry.name.clone(),
        category,
        outcome,
    }
}

fn run_nquads_entry(entry: &ManifestEntry, _root: &Path) -> TestResult {
    let category = match categorize(&entry.test_type) {
        Some(c) => c,
        None => {
            return TestResult {
                id: entry.id.clone(),
                name: entry.name.clone(),
                category: TestCategory::PositiveSyntax,
                outcome: Outcome::Fail(format!("unknown test type {}", entry.test_type)),
            };
        }
    };

    let input = read_fixture(&entry.action);
    let parser = NQuadsParser::new();
    let parsed = parser.parse(Cursor::new(input.as_bytes()));

    let outcome = match (&category, parsed) {
        (TestCategory::PositiveSyntax, Ok(_)) => Outcome::Pass,
        (TestCategory::PositiveSyntax, Err(e)) => {
            Outcome::Fail(format!("expected parse OK; got error: {e}"))
        }
        (TestCategory::NegativeSyntax, Ok(quads)) => {
            Outcome::Fail(format!("expected parse error; got {} quads", quads.len()))
        }
        (TestCategory::NegativeSyntax, Err(_)) => Outcome::Pass,
        (TestCategory::Eval, Ok(actual)) => match &entry.result {
            None => Outcome::Fail("Eval test missing mf:result".to_string()),
            Some(result_path) => {
                let expected_text = read_fixture(result_path);
                let expected =
                    match NQuadsParser::new().parse(Cursor::new(expected_text.as_bytes())) {
                        Ok(t) => t,
                        Err(e) => {
                            return TestResult {
                                id: entry.id.clone(),
                                name: entry.name.clone(),
                                category,
                                outcome: Outcome::Fail(format!("expected file parse error: {e}")),
                            };
                        }
                    };

                // Round-trip through the serializer.
                let serializer = NQuadsSerializer::new();
                let mut buf = Vec::new();
                if let Err(e) = serializer.serialize(&actual, &mut buf) {
                    return TestResult {
                        id: entry.id.clone(),
                        name: entry.name.clone(),
                        category,
                        outcome: Outcome::Fail(format!("re-serialise failed: {e}")),
                    };
                }
                let reparsed = match NQuadsParser::new().parse(Cursor::new(buf.as_slice())) {
                    Ok(t) => t,
                    Err(e) => {
                        return TestResult {
                            id: entry.id.clone(),
                            name: entry.name.clone(),
                            category,
                            outcome: Outcome::Fail(format!("re-parse failed: {e}")),
                        };
                    }
                };
                if canonical_quad_set(&reparsed) == canonical_quad_set(&expected) {
                    Outcome::Pass
                } else {
                    Outcome::Fail(format!(
                        "round-trip differs: expected {} quads, got {}",
                        expected.len(),
                        reparsed.len()
                    ))
                }
            }
        },
        (TestCategory::Eval, Err(e)) => Outcome::Fail(format!("Eval parse failed: {e}")),
    };

    TestResult {
        id: entry.id.clone(),
        name: entry.name.clone(),
        category,
        outcome,
    }
}

fn run_trig_entry(entry: &ManifestEntry, _root: &Path) -> TestResult {
    let category = match categorize(&entry.test_type) {
        Some(c) => c,
        None => {
            return TestResult {
                id: entry.id.clone(),
                name: entry.name.clone(),
                category: TestCategory::PositiveSyntax,
                outcome: Outcome::Fail(format!("unknown test type {}", entry.test_type)),
            };
        }
    };

    let input = read_fixture(&entry.action);
    let parser = TriGParser::new();
    let parsed = parser.parse(Cursor::new(input.as_bytes()));

    let outcome = match (&category, parsed) {
        (TestCategory::PositiveSyntax, Ok(_)) => Outcome::Pass,
        (TestCategory::PositiveSyntax, Err(e)) => {
            Outcome::Fail(format!("expected parse OK; got error: {e}"))
        }
        (TestCategory::NegativeSyntax, Ok(quads)) => {
            Outcome::Fail(format!("expected parse error; got {} quads", quads.len()))
        }
        (TestCategory::NegativeSyntax, Err(_)) => Outcome::Pass,
        (TestCategory::Eval, Ok(actual)) => match &entry.result {
            None => Outcome::Fail("Eval test missing mf:result".to_string()),
            Some(result_path) => {
                let expected_text = read_fixture(result_path);
                let expected =
                    match NQuadsParser::new().parse(Cursor::new(expected_text.as_bytes())) {
                        Ok(q) => q,
                        Err(e) => {
                            return TestResult {
                                id: entry.id.clone(),
                                name: entry.name.clone(),
                                category,
                                outcome: Outcome::Fail(format!("expected file parse error: {e}")),
                            };
                        }
                    };
                if canonical_quad_set(&actual) == canonical_quad_set(&expected) {
                    Outcome::Pass
                } else {
                    Outcome::Fail(format!(
                        "TriG eval differs: expected {} quads, got {}",
                        expected.len(),
                        actual.len()
                    ))
                }
            }
        },
        (TestCategory::Eval, Err(e)) => Outcome::Fail(format!("Eval parse failed: {e}")),
    };

    TestResult {
        id: entry.id.clone(),
        name: entry.name.clone(),
        category,
        outcome,
    }
}

// ---------------------------------------------------------------------------
// Per-format integration tests
// ---------------------------------------------------------------------------

#[test]
fn w3c_turtle_compliance() {
    let root = fixture_root();
    let format_dir = root.join("turtle");
    let entries = load_manifest(&format_dir);
    assert!(
        entries.len() >= 30,
        "Turtle fixture corpus should have at least 30 entries (got {})",
        entries.len()
    );
    let results: Vec<TestResult> = entries.iter().map(|e| run_turtle_entry(e, &root)).collect();
    assert_pass_rate("turtle", &results);
}

#[test]
fn w3c_ntriples_compliance() {
    let root = fixture_root();
    let format_dir = root.join("ntriples");
    let entries = load_manifest(&format_dir);
    assert!(
        entries.len() >= 30,
        "N-Triples fixture corpus should have at least 30 entries (got {})",
        entries.len()
    );
    let results: Vec<TestResult> = entries
        .iter()
        .map(|e| run_ntriples_entry(e, &root))
        .collect();
    assert_pass_rate("ntriples", &results);
}

#[test]
fn w3c_nquads_compliance() {
    let root = fixture_root();
    let format_dir = root.join("nquads");
    let entries = load_manifest(&format_dir);
    assert!(
        entries.len() >= 25,
        "N-Quads fixture corpus should have at least 25 entries (got {})",
        entries.len()
    );
    let results: Vec<TestResult> = entries.iter().map(|e| run_nquads_entry(e, &root)).collect();
    assert_pass_rate("nquads", &results);
}

#[test]
fn w3c_trig_compliance() {
    let root = fixture_root();
    let format_dir = root.join("trig");
    let entries = load_manifest(&format_dir);
    assert!(
        entries.len() >= 25,
        "TriG fixture corpus should have at least 25 entries (got {})",
        entries.len()
    );
    let results: Vec<TestResult> = entries.iter().map(|e| run_trig_entry(e, &root)).collect();
    assert_pass_rate("trig", &results);
}

// ---------------------------------------------------------------------------
// Aggregate sanity test: every format hits the target.
// ---------------------------------------------------------------------------

#[test]
fn w3c_aggregate_pass_rate() {
    let root = fixture_root();
    let mut total = 0usize;
    let mut passes = 0usize;

    type Runner = fn(&ManifestEntry, &Path) -> TestResult;
    let formats: [(&str, Runner); 4] = [
        ("turtle", run_turtle_entry),
        ("ntriples", run_ntriples_entry),
        ("nquads", run_nquads_entry),
        ("trig", run_trig_entry),
    ];

    for (format, run) in formats {
        let format_dir = root.join(format);
        let entries = load_manifest(&format_dir);
        for entry in &entries {
            let res = run(entry, &root);
            total += 1;
            if matches!(res.outcome, Outcome::Pass) {
                passes += 1;
            }
        }
    }

    let rate = passes as f64 / total as f64;
    eprintln!(
        "[w3c-aggregate] pass {passes}/{total} ({:.2}%)",
        rate * 100.0
    );
    assert!(
        rate >= PASS_RATE_TARGET,
        "aggregate pass rate {rate:.4} < target {PASS_RATE_TARGET:.4}"
    );
}

// ---------------------------------------------------------------------------
// Round-trip property test (parse → serialize → parse stability)
// over N=1000 randomly generated graphs.
// ---------------------------------------------------------------------------

mod roundtrip {
    use super::*;
    use proptest::prelude::*;

    /// Generate a safe IRI suffix (ASCII identifier characters only).
    fn iri_suffix() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-zA-Z][a-zA-Z0-9_-]{0,15}").expect("static regex is valid")
    }

    /// Generate a literal value safe for round-trip (printable ASCII, no
    /// control characters; the parser/serializer pair handles escapes
    /// correctly but we keep payloads simple to keep the test fast).
    fn safe_literal() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-zA-Z0-9 ._\\-]{1,30}").expect("static regex is valid")
    }

    /// Generate a language tag (ISO 639-style).
    fn lang_tag() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-z]{2}(-[A-Z]{2})?").expect("static regex is valid")
    }

    /// Build a NamedNode from the given suffix.
    fn nn(suffix: &str) -> NamedNode {
        NamedNode::new(format!("http://example.org/{suffix}")).expect("valid IRI")
    }

    /// Generate a single triple using the given strategies.
    fn arb_triple() -> impl Strategy<Value = Triple> {
        (
            iri_suffix(),
            iri_suffix(),
            prop_oneof![
                iri_suffix().prop_map(|s| Object::NamedNode(nn(&s))),
                safe_literal().prop_map(|v| Object::Literal(
                    oxirs_core::model::Literal::new_simple_literal(v)
                )),
                (safe_literal(), lang_tag()).prop_map(|(v, lang)| Object::Literal(
                    oxirs_core::model::Literal::new_language_tagged_literal(v, lang)
                        .expect("valid lang")
                )),
            ],
        )
            .prop_map(|(s, p, o)| {
                Triple::new(Subject::NamedNode(nn(&s)), Predicate::NamedNode(nn(&p)), o)
            })
    }

    /// Generate a small graph (5-20 triples).
    fn arb_graph() -> impl Strategy<Value = Vec<Triple>> {
        prop::collection::vec(arb_triple(), 5..=20)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 1000,
            failure_persistence: None,
            .. ProptestConfig::default()
        })]

        /// Property: parse(serialize(parse(input))) == parse(input)
        ///
        /// We start from the in-memory generated graph, serialise to N-Triples,
        /// re-parse, and assert the canonical triple sets are equal.
        #[test]
        fn ntriples_roundtrip_stable(graph in arb_graph()) {
            let serializer = NTriplesSerializer::new();
            let mut buf: Vec<u8> = Vec::new();
            serializer.serialize(&graph, &mut buf)
                .expect("serialise should succeed for valid graph");

            let parser = NTriplesParser::new();
            let parsed: Vec<Triple> = parser
                .parse(Cursor::new(buf.as_slice()))
                .expect("re-parse must succeed");

            let original = canonical_triple_set(&graph);
            let after = canonical_triple_set(&parsed);
            prop_assert_eq!(original, after);
        }

        /// Property: parse → serialize → parse stability via the Turtle pair.
        ///
        /// We generate a graph, serialise as Turtle, re-parse, and assert
        /// the canonical triple sets coincide. This stresses prefix
        /// compression, IRI abbreviation, and literal escaping in the
        /// serializer paired with the full Turtle parser.
        #[test]
        fn turtle_roundtrip_stable(graph in arb_graph()) {
            let serializer = TurtleSerializer::new();
            let mut buf: Vec<u8> = Vec::new();
            serializer.serialize(&graph, &mut buf)
                .expect("serialise should succeed for valid graph");

            let parser = TurtleParser::new();
            let text = String::from_utf8(buf).expect("UTF-8 output");
            let parsed = parser.parse_document(&text)
                .expect("re-parse must succeed");

            let original = canonical_triple_set(&graph);
            let after = canonical_triple_set(&parsed);
            prop_assert_eq!(original, after);
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests for fixture infrastructure (manifest parser and helpers).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod infra_tests {
    use super::*;

    #[test]
    fn fixture_root_exists() {
        let root = fixture_root();
        assert!(root.exists(), "fixture root {} must exist", root.display());
        for fmt in ["turtle", "ntriples", "nquads", "trig"] {
            let dir = root.join(fmt);
            assert!(
                dir.join("manifest.ttl").exists(),
                "manifest.ttl missing under {}",
                dir.display()
            );
        }
    }

    #[test]
    fn manifest_loader_parses_turtle_corpus() {
        let entries = load_manifest(&fixture_root().join("turtle"));
        // sanity: every entry's action file should resolve
        for e in &entries {
            assert!(
                e.action.exists(),
                "fixture file missing for {}: {}",
                e.id,
                e.action.display()
            );
            if matches!(categorize(&e.test_type), Some(TestCategory::Eval)) {
                let result = e
                    .result
                    .as_ref()
                    .unwrap_or_else(|| panic!("Eval test {} missing mf:result", e.id));
                assert!(
                    result.exists(),
                    "expected-result file missing for {}: {}",
                    e.id,
                    result.display()
                );
            }
        }
        assert!(
            !entries.is_empty(),
            "manifest loader must produce at least one entry"
        );
    }

    #[test]
    fn nquads_quoted_triple_roundtrip() {
        // Quick smoke check that the N-Quads-star path works end-to-end.
        let nq = "<< <http://example.org/a> <http://example.org/b> <http://example.org/c> >> \
                  <http://example.org/p> <http://example.org/o> .";
        let parser = NQuadsParser::new();
        let quads = parser.parse(Cursor::new(nq.as_bytes())).expect("parse");
        assert_eq!(quads.len(), 1);
        assert!(matches!(quads[0].subject(), Subject::QuotedTriple(_)));

        let serializer = NQuadsSerializer::new();
        let mut buf = Vec::new();
        serializer.serialize(&quads, &mut buf).expect("serialise");
        let text = String::from_utf8(buf).expect("UTF-8");
        assert!(
            text.contains("<<") && text.contains(">>"),
            "expected quoted-triple delimiters in serialised output: {text}"
        );

        let reparsed = parser
            .parse(Cursor::new(text.as_bytes()))
            .expect("re-parse");
        assert_eq!(canonical_quad_set(&quads), canonical_quad_set(&reparsed));
    }
}
