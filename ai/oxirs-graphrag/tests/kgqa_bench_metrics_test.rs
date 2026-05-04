//! Integration tests for the KGQA benchmark metric helpers.
//!
//! The criterion benchmark file (`benches/langchain_kgqa.rs`) defines the
//! same helpers inline so it can run as a `harness = false` binary, but
//! criterion takes over the test runner of that binary, which means
//! `#[cfg(test)]` modules inside the bench are not executed by `cargo test`
//! / `cargo nextest`. This file therefore re-implements the small pure
//! helpers (Hits@K, MRR, percentile, deterministic LCG, fixture loader,
//! synthetic generator) and exercises them directly so the metric layer is
//! covered by the workspace test suite.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Pure metric helpers (mirror of benches/langchain_kgqa.rs)
// ---------------------------------------------------------------------------

fn hits_at_k(ranked: &[String], gold: &HashSet<String>, k: usize) -> f64 {
    if gold.is_empty() {
        return 0.0;
    }
    ranked.iter().take(k).any(|iri| gold.contains(iri)) as u8 as f64
}

fn reciprocal_rank(ranked: &[String], gold: &HashSet<String>) -> f64 {
    for (i, iri) in ranked.iter().enumerate() {
        if gold.contains(iri) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

fn percentile(samples: &[Duration], pct: f64) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let mut sorted: Vec<Duration> = samples.to_vec();
    sorted.sort();
    let clamped = pct.clamp(0.0, 100.0);
    let idx = ((clamped / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
}

// ---------------------------------------------------------------------------
// Fixture types (subset of benches/langchain_kgqa.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct FixtureTriple {
    s: String,
    p: String,
    o: String,
}

#[derive(Debug, Clone, Deserialize)]
struct FixtureQuestion {
    qid: String,
    question: String,
    topic_entity: String,
    predicate: String,
    answer_entities: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Fixture {
    kg: Vec<FixtureTriple>,
    #[serde(default)]
    labels: HashMap<String, String>,
    questions: Vec<FixtureQuestion>,
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches/fixtures/kgqa-bench")
        .join(name)
}

fn load_webqsp_fixture() -> Fixture {
    let path = fixture_path("webqsp_subset.json");
    let bytes = fs::read(&path).expect("read webqsp_subset.json");
    serde_json::from_slice::<Fixture>(&bytes).expect("parse webqsp_subset.json")
}

#[derive(Debug, Clone, Deserialize)]
struct LangChainReference {
    qid: String,
    ranked_answers: Vec<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn hits_at_k_returns_one_when_gold_within_top_k() {
    let ranked = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let gold: HashSet<String> = ["b".to_string()].iter().cloned().collect();
    assert_eq!(hits_at_k(&ranked, &gold, 1), 0.0);
    assert_eq!(hits_at_k(&ranked, &gold, 2), 1.0);
    assert_eq!(hits_at_k(&ranked, &gold, 5), 1.0);
}

#[test]
fn hits_at_k_handles_empty_gold() {
    let ranked = vec!["a".to_string()];
    let gold: HashSet<String> = HashSet::new();
    assert_eq!(hits_at_k(&ranked, &gold, 1), 0.0);
}

#[test]
fn reciprocal_rank_returns_inverse_position() {
    let ranked = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let gold: HashSet<String> = ["c".to_string()].iter().cloned().collect();
    let rr = reciprocal_rank(&ranked, &gold);
    assert!((rr - (1.0 / 3.0)).abs() < 1e-9);
}

#[test]
fn reciprocal_rank_zero_when_gold_missing() {
    let ranked = vec!["a".to_string()];
    let gold: HashSet<String> = ["b".to_string()].iter().cloned().collect();
    assert_eq!(reciprocal_rank(&ranked, &gold), 0.0);
}

#[test]
fn percentile_orders_correctly() {
    let samples: Vec<Duration> = (1..=100).map(|x| Duration::from_millis(x as u64)).collect();
    let p50 = percentile(&samples, 50.0);
    let p95 = percentile(&samples, 95.0);
    let p99 = percentile(&samples, 99.0);
    assert!(p50 <= p95);
    assert!(p95 <= p99);
    assert!(p99 >= Duration::from_millis(95));
}

#[test]
fn percentile_zero_for_empty_samples() {
    assert_eq!(percentile(&[], 50.0), Duration::ZERO);
}

#[test]
fn lcg_is_deterministic_with_same_seed() {
    let mut a = Lcg64::new(42);
    let mut b = Lcg64::new(42);
    for _ in 0..16 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn lcg_diverges_with_different_seeds() {
    let mut a = Lcg64::new(1);
    let mut b = Lcg64::new(2);
    let mut differs = false;
    for _ in 0..16 {
        if a.next_u64() != b.next_u64() {
            differs = true;
            break;
        }
    }
    assert!(differs, "two distinct LCG seeds must diverge");
}

#[test]
fn webqsp_fixture_loads_and_has_expected_shape() {
    let fixture = load_webqsp_fixture();
    assert!(!fixture.kg.is_empty(), "KG must not be empty");
    assert!(
        fixture.questions.len() >= 10,
        "WebQSP-derived fixture must have at least 10 questions"
    );
    for question in &fixture.questions {
        assert!(!question.qid.is_empty());
        assert!(!question.question.is_empty());
        assert!(!question.topic_entity.is_empty());
        assert!(!question.predicate.is_empty());
        assert!(
            !question.answer_entities.is_empty(),
            "every question must have at least one gold answer"
        );
    }
    // Every gold answer must be reachable via the gold predicate from the
    // topic entity in the bundled KG.  This protects the fixture from
    // accidental drift between question / triple edits.
    let mut triples_idx: HashMap<(String, String), HashSet<String>> = HashMap::new();
    for triple in &fixture.kg {
        triples_idx
            .entry((triple.s.clone(), triple.p.clone()))
            .or_default()
            .insert(triple.o.clone());
    }
    for question in &fixture.questions {
        let key = (question.topic_entity.clone(), question.predicate.clone());
        let triple_objects = triples_idx
            .get(&key)
            .unwrap_or_else(|| panic!("missing edge for {:?}", key));
        for gold in &question.answer_entities {
            assert!(
                triple_objects.contains(gold),
                "{} → {} → {} not present in fixture KG",
                question.topic_entity,
                question.predicate,
                gold
            );
        }
    }
}

#[test]
fn webqsp_fixture_labels_cover_all_kg_iris() {
    let fixture = load_webqsp_fixture();
    let mut iris: HashSet<&str> = HashSet::new();
    for triple in &fixture.kg {
        iris.insert(&triple.s);
        iris.insert(&triple.p);
        iris.insert(&triple.o);
    }
    // We only require labels for entity IRIs (subjects/objects). Predicates
    // intentionally do not need pretty labels because the linker indexes
    // entities, not relations.
    let entity_iris: HashSet<&str> = fixture
        .kg
        .iter()
        .flat_map(|t| [t.s.as_str(), t.o.as_str()])
        .collect();
    for iri in entity_iris {
        assert!(
            fixture.labels.contains_key(iri),
            "missing label for entity IRI {iri}"
        );
    }
}

#[test]
fn langchain_reference_parses_expected_shape() {
    let raw = r#"{"qid":"WebQSP-001","ranked_answers":["ent:Ulm","ent:Germany"]}"#;
    let parsed: LangChainReference = serde_json::from_str(raw).expect("parse reference");
    assert_eq!(parsed.qid, "WebQSP-001");
    assert_eq!(parsed.ranked_answers.len(), 2);
}

#[test]
fn langchain_reference_round_trips_through_temp_dir() {
    let dir = std::env::temp_dir().join("oxirs_graphrag_kgqa_bench_metric_test");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("temp dir create");
    let payload = r#"{"qid":"WebQSP-001","ranked_answers":["ent:Ulm"]}"#;
    let target = dir.join("WebQSP-001.json");
    fs::write(&target, payload).expect("write payload");
    let bytes = fs::read(&target).expect("read payload");
    let parsed: LangChainReference =
        serde_json::from_slice(&bytes).expect("parse roundtrip payload");
    assert_eq!(parsed.qid, "WebQSP-001");
    assert_eq!(parsed.ranked_answers, vec!["ent:Ulm".to_string()]);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn webqsp_fixture_uses_only_consistent_predicates() {
    // No question should reference a predicate that doesn't appear in the
    // KG triples — guards the bench from silently scoring 0 because of an
    // unsupported predicate.
    let fixture = load_webqsp_fixture();
    let kg_predicates: HashSet<&str> = fixture.kg.iter().map(|t| t.p.as_str()).collect();
    for q in &fixture.questions {
        assert!(
            kg_predicates.contains(q.predicate.as_str()),
            "predicate {} for {} is not present in KG triples",
            q.predicate,
            q.qid
        );
    }
}
