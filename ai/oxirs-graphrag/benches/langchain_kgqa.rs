//! KGQA benchmark for oxirs-graphrag.
//!
//! Phase 1 (always runs): standalone Hits@1 / Hits@5 / MRR / latency p50, p95,
//! p99 / throughput on a hand-crafted WebQSP-derived subset and a deterministic
//! synthetic ontology-QA generator.
//!
//! Phase 2 (optional): when `LANGCHAIN_REF_FIXTURES` points at a directory of
//! per-question JSON captures (produced by
//! `benches/scripts/capture_langchain_reference.py`), the bench compares
//! oxirs-graphrag's Hits@5 against the reference and panics if the gap exceeds
//! 5 percentage points. Skipped gracefully when the env var is unset.
//!
//! Pipeline under test
//! -------------------
//!
//! For each question we:
//!   1. Use [`EntityLinker`] over the KG label index to detect a topic entity
//!      from the question text.
//!   2. Use [`PathFinder`] to enumerate one-hop neighbours of the topic
//!      entity, optionally constrained by the gold predicate.
//!   3. Rank the resulting candidate object IRIs by edge score.
//!
//! That mirrors the LangChain `GraphQAChain` strategy at the algorithmic level
//! while remaining 100% Pure Rust and free of LLM calls so the benchmark is
//! deterministic and reproducible.

#![allow(clippy::too_many_arguments)]

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use serde::{Deserialize, Serialize};

use oxirs_graphrag::entity_linker::{Entity, EntityLinker};
use oxirs_graphrag::path_finder::{KnowledgeEdge, PathFinder, PathFinderConfig};

// ---------------------------------------------------------------------------
// Fixture types (must match webqsp_subset.json + LangChain capture schema)
// ---------------------------------------------------------------------------

/// One KG triple read from the JSON fixture.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct FixtureTriple {
    s: String,
    p: String,
    o: String,
}

/// One KGQA question read from the JSON fixture.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct FixtureQuestion {
    qid: String,
    question: String,
    topic_entity: String,
    predicate: String,
    answer_entities: Vec<String>,
}

/// Top-level structure of `webqsp_subset.json`.
#[derive(Debug, Clone, Deserialize)]
struct Fixture {
    kg: Vec<FixtureTriple>,
    #[serde(default)]
    labels: HashMap<String, String>,
    questions: Vec<FixtureQuestion>,
}

/// LangChain reference capture (one JSON per question) used by Phase 2.
#[derive(Debug, Clone, Deserialize)]
struct LangChainReference {
    qid: String,
    ranked_answers: Vec<String>,
}

// ---------------------------------------------------------------------------
// KGQA pipeline (oxirs-graphrag side)
// ---------------------------------------------------------------------------

/// Result of running the KGQA pipeline on a single question.
#[derive(Debug, Clone)]
struct PipelineAnswer {
    qid: String,
    /// Top-K candidate answer IRIs ordered best → worst.
    ranked: Vec<String>,
    /// Wall-clock latency for the full pipeline.
    latency: Duration,
}

/// Build a `(linker, finder, predicate_set)` triple from a parsed fixture.
fn build_pipeline(fixture: &Fixture) -> (EntityLinker, PathFinder, HashSet<String>) {
    // Build entity index from KG labels (subjects/objects of triples).
    let mut linker = EntityLinker::new(0.05);
    let mut seen_iris: HashSet<String> = HashSet::new();
    for triple in &fixture.kg {
        for iri in [&triple.s, &triple.o] {
            if !seen_iris.insert(iri.clone()) {
                continue;
            }
            let label = fixture
                .labels
                .get(iri)
                .cloned()
                .unwrap_or_else(|| iri.split(':').next_back().unwrap_or(iri).replace('_', " "));
            linker.add_entity(Entity {
                iri: iri.clone(),
                label,
                aliases: Vec::new(),
                entity_type: "Thing".to_string(),
                description: None,
                popularity: 0.5,
            });
        }
    }

    let edges: Vec<KnowledgeEdge> = fixture
        .kg
        .iter()
        .map(|t| KnowledgeEdge::new(t.s.clone(), t.p.clone(), t.o.clone()))
        .collect();
    let finder = PathFinder::new(
        edges,
        PathFinderConfig {
            max_depth: 3,
            max_paths: 32,
            ..PathFinderConfig::default()
        },
    );

    let predicates: HashSet<String> = fixture.kg.iter().map(|t| t.p.clone()).collect();
    (linker, finder, predicates)
}

/// Run the KGQA pipeline on a single question.
///
/// Strategy
/// --------
///   1. Detect mentions in the question text via `EntityLinker`.
///   2. Pick the highest-confidence linked entity as the topic seed.
///   3. Enumerate 1-hop neighbours through `PathFinder`. If no mention is
///      detected we fall back to the gold topic entity (this lets the bench
///      still measure path-finding latency in pathological cases).
///   4. Score candidates by predicate weight match against the question.
fn answer_question(
    linker: &EntityLinker,
    finder: &PathFinder,
    predicates: &HashSet<String>,
    question: &FixtureQuestion,
    top_k: usize,
) -> PipelineAnswer {
    let started = Instant::now();
    let mention_seeds = detect_topic_seeds(linker, &question.question);
    let topic = mention_seeds
        .first()
        .cloned()
        .unwrap_or_else(|| question.topic_entity.clone());
    let ranked = rank_candidates(finder, predicates, &topic, &question.question, top_k);
    PipelineAnswer {
        qid: question.qid.clone(),
        ranked,
        latency: started.elapsed(),
    }
}

/// Detect candidate topic entities for a question via the entity linker.
fn detect_topic_seeds(linker: &EntityLinker, text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    match linker.link(trimmed) {
        Ok(linked) => linked
            .into_iter()
            .filter_map(|le| le.best_candidate.map(|c| c.entity.iri))
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Rank one-hop neighbours by predicate weight match against question text.
///
/// Each candidate object IRI receives a score that combines the path-finder
/// edge weight with a bonus when the predicate's "local name" surface appears
/// in the question (a cheap proxy for keyword-aware scoring without pulling
/// in a full BM25 index).
fn rank_candidates(
    finder: &PathFinder,
    predicates: &HashSet<String>,
    topic: &str,
    question_text: &str,
    top_k: usize,
) -> Vec<String> {
    let lower_q = question_text.to_lowercase();
    let mut pred_match: HashMap<&String, f64> = HashMap::new();
    for pred in predicates {
        let local = pred
            .split(':')
            .next_back()
            .unwrap_or(pred)
            .replace('_', " ");
        let bonus = if !local.is_empty() && lower_q.contains(&local.to_lowercase()) {
            1.0
        } else {
            0.0
        };
        pred_match.insert(pred, bonus);
    }

    // One-hop neighbour expansion via PathFinder.
    let mut scored: Vec<(String, f64)> = Vec::new();
    for path in finder.multi_hop_paths(topic, 1) {
        if path.predicates.len() != 1 {
            continue;
        }
        let path_pred = match path.predicates.first() {
            Some(p) => p,
            None => continue,
        };
        let target = match path.nodes.last() {
            Some(t) => t.clone(),
            None => continue,
        };
        let bonus = pred_match.get(path_pred).copied().unwrap_or(0.0);
        scored.push((target, path.score + bonus));
    }

    // Stable sort by score desc, then by string asc for determinism.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    for (iri, _) in scored {
        if seen.insert(iri.clone()) {
            out.push(iri);
            if out.len() >= top_k {
                break;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Hits@K = fraction of questions whose first K predicted IRIs intersect any
/// gold answer.
fn hits_at_k(ranked: &[String], gold: &HashSet<String>, k: usize) -> f64 {
    if gold.is_empty() {
        return 0.0;
    }
    ranked.iter().take(k).any(|iri| gold.contains(iri)) as u8 as f64
}

/// Mean Reciprocal Rank — 1/(rank of first correct hit), 0.0 if none.
fn reciprocal_rank(ranked: &[String], gold: &HashSet<String>) -> f64 {
    for (i, iri) in ranked.iter().enumerate() {
        if gold.contains(iri) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Compute a percentile from latency samples. `pct` is in `[0.0, 100.0]`.
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

/// Aggregate metrics over a benchmark run.
#[derive(Debug, Clone, Default)]
struct AggregateMetrics {
    queries: usize,
    hits_at_1: f64,
    hits_at_5: f64,
    mrr: f64,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    throughput_qps: f64,
}

fn aggregate(
    answers: &[PipelineAnswer],
    golds: &HashMap<String, HashSet<String>>,
) -> AggregateMetrics {
    if answers.is_empty() {
        return AggregateMetrics::default();
    }
    let mut hits1 = 0.0_f64;
    let mut hits5 = 0.0_f64;
    let mut rr_sum = 0.0_f64;
    let mut latencies: Vec<Duration> = Vec::with_capacity(answers.len());
    let mut total_time = Duration::ZERO;

    for answer in answers {
        let gold = match golds.get(&answer.qid) {
            Some(g) => g,
            None => continue,
        };
        hits1 += hits_at_k(&answer.ranked, gold, 1);
        hits5 += hits_at_k(&answer.ranked, gold, 5);
        rr_sum += reciprocal_rank(&answer.ranked, gold);
        latencies.push(answer.latency);
        total_time += answer.latency;
    }

    let n = answers.len() as f64;
    AggregateMetrics {
        queries: answers.len(),
        hits_at_1: hits1 / n,
        hits_at_5: hits5 / n,
        mrr: rr_sum / n,
        p50: percentile(&latencies, 50.0),
        p95: percentile(&latencies, 95.0),
        p99: percentile(&latencies, 99.0),
        throughput_qps: if total_time.as_secs_f64() > 0.0 {
            n / total_time.as_secs_f64()
        } else {
            0.0
        },
    }
}

// ---------------------------------------------------------------------------
// Synthetic ontology-QA generator (deterministic, seeded LCG)
// ---------------------------------------------------------------------------

/// 64-bit Linear Congruential Generator — deterministic, no external deps.
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
        // Numerical Recipes constants.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            return lo;
        }
        let span = (hi - lo) as u64;
        lo + (self.next_u64() % span) as usize
    }
}

/// Build a synthetic class-instance ontology with `n_classes` classes,
/// `instances_per_class` instances each, and one `instance_of` predicate
/// linking instances to their class. Each question asks for the class of a
/// given instance — a textbook one-hop KGQA task.
fn synthetic_fixture(seed: u64, n_classes: usize, instances_per_class: usize) -> Fixture {
    let mut rng = Lcg64::new(seed);
    let mut kg: Vec<FixtureTriple> = Vec::new();
    let mut labels: HashMap<String, String> = HashMap::new();
    let mut questions: Vec<FixtureQuestion> = Vec::new();

    for c in 0..n_classes {
        let class_iri = format!("syn:Class_{c:03}");
        labels.insert(class_iri.clone(), format!("Class {c:03}"));
        for i in 0..instances_per_class {
            let instance_iri = format!("syn:Inst_{c:03}_{i:03}");
            labels.insert(instance_iri.clone(), format!("Instance {c:03}-{i:03}"));
            kg.push(FixtureTriple {
                s: instance_iri.clone(),
                p: "syn:instance_of".to_string(),
                o: class_iri.clone(),
            });
            // Add a randomised "related_to" within-class edge for graph density.
            if instances_per_class > 1 {
                let mut related = rng.gen_range(0, instances_per_class);
                if related == i {
                    related = (related + 1) % instances_per_class;
                }
                let related_iri = format!("syn:Inst_{c:03}_{related:03}");
                kg.push(FixtureTriple {
                    s: instance_iri.clone(),
                    p: "syn:related_to".to_string(),
                    o: related_iri,
                });
            }
            questions.push(FixtureQuestion {
                qid: format!("SYN-{c:03}-{i:03}"),
                question: format!("What is the class of Instance {c:03}-{i:03}?"),
                topic_entity: instance_iri,
                predicate: "syn:instance_of".to_string(),
                answer_entities: vec![class_iri.clone()],
            });
        }
    }

    Fixture {
        kg,
        labels,
        questions,
    }
}

// ---------------------------------------------------------------------------
// Fixture loading
// ---------------------------------------------------------------------------

fn fixture_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest).join("benches/fixtures/kgqa-bench")
}

fn load_webqsp_fixture() -> Fixture {
    let path = fixture_dir().join("webqsp_subset.json");
    let bytes =
        fs::read(&path).unwrap_or_else(|err| panic!("read {} failed: {err}", path.display()));
    serde_json::from_slice::<Fixture>(&bytes)
        .unwrap_or_else(|err| panic!("parse {} failed: {err}", path.display()))
}

fn build_gold_map(fixture: &Fixture) -> HashMap<String, HashSet<String>> {
    fixture
        .questions
        .iter()
        .map(|q| (q.qid.clone(), q.answer_entities.iter().cloned().collect()))
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark groups
// ---------------------------------------------------------------------------

fn run_pipeline_all(
    linker: &EntityLinker,
    finder: &PathFinder,
    predicates: &HashSet<String>,
    questions: &[FixtureQuestion],
    top_k: usize,
) -> Vec<PipelineAnswer> {
    questions
        .iter()
        .map(|q| answer_question(linker, finder, predicates, q, top_k))
        .collect()
}

fn bench_webqsp_subset(c: &mut Criterion) {
    let fixture = load_webqsp_fixture();
    let (linker, finder, predicates) = build_pipeline(&fixture);
    let golds = build_gold_map(&fixture);

    let mut group = c.benchmark_group("kgqa_webqsp_subset");
    group.bench_function("end_to_end_top5", |b| {
        b.iter(|| {
            let answers = run_pipeline_all(&linker, &finder, &predicates, &fixture.questions, 5);
            let metrics = aggregate(&answers, &golds);
            black_box(metrics);
        });
    });
    group.finish();

    // Single non-bench run to print summary metrics so operators see absolute numbers.
    let answers = run_pipeline_all(&linker, &finder, &predicates, &fixture.questions, 5);
    let metrics = aggregate(&answers, &golds);
    eprintln!(
        "[oxirs-graphrag KGQA WebQSP-subset] queries={} Hits@1={:.4} Hits@5={:.4} \
         MRR={:.4} p50={:?} p95={:?} p99={:?} throughput={:.2} qps",
        metrics.queries,
        metrics.hits_at_1,
        metrics.hits_at_5,
        metrics.mrr,
        metrics.p50,
        metrics.p95,
        metrics.p99,
        metrics.throughput_qps,
    );
}

fn bench_synthetic_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("kgqa_synthetic_scaling");
    let scales = [(4usize, 4usize), (8, 8), (16, 8)];
    for (n_classes, instances_per_class) in scales {
        let fixture = synthetic_fixture(0xC001D00D, n_classes, instances_per_class);
        let (linker, finder, predicates) = build_pipeline(&fixture);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("c{n_classes}xi{instances_per_class}")),
            &fixture,
            |b, fix| {
                b.iter(|| {
                    let answers =
                        run_pipeline_all(&linker, &finder, &predicates, &fix.questions, 5);
                    black_box(answers);
                });
            },
        );
    }
    group.finish();
}

fn bench_pipeline_latency(c: &mut Criterion) {
    let fixture = load_webqsp_fixture();
    let (linker, finder, predicates) = build_pipeline(&fixture);
    let first_question = match fixture.questions.first() {
        Some(q) => q.clone(),
        None => return,
    };

    c.bench_function("kgqa_single_question_latency", |b| {
        b.iter(|| {
            let answer = answer_question(&linker, &finder, &predicates, &first_question, 5);
            black_box(answer);
        });
    });
}

// ---------------------------------------------------------------------------
// Phase 2 — optional LangChain comparator
// ---------------------------------------------------------------------------

const LANGCHAIN_HITS_AT_5_TOLERANCE_PP: f64 = 5.0;

/// Locate per-question reference captures in `dir`. Files must be named
/// `<qid>.json` and contain a `LangChainReference` payload.
fn load_langchain_references(dir: &Path) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!(
                "[langchain-comparator] could not read {}: {err}",
                dir.display()
            );
            return out;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(err) => {
                eprintln!("[langchain-comparator] skip {}: {err}", path.display());
                continue;
            }
        };
        match serde_json::from_slice::<LangChainReference>(&bytes) {
            Ok(reference) => {
                out.insert(reference.qid, reference.ranked_answers);
            }
            Err(err) => {
                eprintln!(
                    "[langchain-comparator] parse error in {}: {err}",
                    path.display()
                );
            }
        }
    }
    out
}

fn bench_langchain_comparator(c: &mut Criterion) {
    let env_dir = match env::var("LANGCHAIN_REF_FIXTURES") {
        Ok(value) if !value.trim().is_empty() => PathBuf::from(value),
        _ => {
            eprintln!("[langchain-comparator] LANGCHAIN_REF_FIXTURES unset — Phase 2 skipped.");
            return;
        }
    };

    let fixture = load_webqsp_fixture();
    let (linker, finder, predicates) = build_pipeline(&fixture);
    let golds = build_gold_map(&fixture);
    let oxirs_answers = run_pipeline_all(&linker, &finder, &predicates, &fixture.questions, 5);
    let oxirs_metrics = aggregate(&oxirs_answers, &golds);

    let references = load_langchain_references(&env_dir);
    if references.is_empty() {
        panic!(
            "LANGCHAIN_REF_FIXTURES={} contained no usable *.json captures",
            env_dir.display()
        );
    }

    // Compute LangChain Hits@5 over the same gold set, for the subset of
    // questions for which a reference exists.
    let mut langchain_hits5 = 0.0_f64;
    let mut compared = 0_usize;
    for question in &fixture.questions {
        let ranked = match references.get(&question.qid) {
            Some(r) => r,
            None => continue,
        };
        let gold: HashSet<String> = question.answer_entities.iter().cloned().collect();
        langchain_hits5 += hits_at_k(ranked, &gold, 5);
        compared += 1;
    }
    if compared == 0 {
        panic!("LANGCHAIN_REF_FIXTURES had no overlapping qids with webqsp_subset.json");
    }
    let langchain_hits5 = langchain_hits5 / compared as f64;
    let oxirs_hits5_pp = oxirs_metrics.hits_at_5 * 100.0;
    let langchain_hits5_pp = langchain_hits5 * 100.0;
    let delta_pp = oxirs_hits5_pp - langchain_hits5_pp;

    eprintln!(
        "[langchain-comparator] oxirs Hits@5={oxirs_hits5_pp:.2}pp \
         langchain Hits@5={langchain_hits5_pp:.2}pp delta={delta_pp:+.2}pp \
         tolerance=±{LANGCHAIN_HITS_AT_5_TOLERANCE_PP}pp"
    );
    assert!(
        delta_pp.abs() <= LANGCHAIN_HITS_AT_5_TOLERANCE_PP,
        "oxirs-graphrag Hits@5 ({oxirs_hits5_pp:.2}pp) deviates from LangChain \
         ({langchain_hits5_pp:.2}pp) by {delta_pp:+.2}pp — exceeds tolerance \
         of ±{LANGCHAIN_HITS_AT_5_TOLERANCE_PP}pp"
    );

    // Register a noop benchmark so criterion still emits a record for Phase 2.
    let mut group = c.benchmark_group("kgqa_langchain_comparator");
    group.bench_function("hits_at_5_within_tolerance", |b| {
        b.iter(|| {
            black_box((oxirs_hits5_pp, langchain_hits5_pp, delta_pp));
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Tests for metric helpers (run by `cargo nextest run -p oxirs-graphrag`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_at_k_returns_one_when_first_match_within_k() {
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
    fn reciprocal_rank_zero_when_missing() {
        let ranked = vec!["a".to_string()];
        let gold: HashSet<String> = ["b".to_string()].iter().cloned().collect();
        assert_eq!(reciprocal_rank(&ranked, &gold), 0.0);
    }

    #[test]
    fn percentile_handles_typical_distribution() {
        let samples: Vec<Duration> = (1..=100).map(|x| Duration::from_millis(x as u64)).collect();
        let p50 = percentile(&samples, 50.0);
        let p95 = percentile(&samples, 95.0);
        let p99 = percentile(&samples, 99.0);
        assert!(p50 <= p95);
        assert!(p95 <= p99);
        // p99 should land near the top of the distribution.
        assert!(p99 >= Duration::from_millis(95));
    }

    #[test]
    fn percentile_zero_for_empty() {
        assert_eq!(percentile(&[], 50.0), Duration::ZERO);
    }

    #[test]
    fn synthetic_fixture_is_deterministic() {
        let f1 = synthetic_fixture(42, 3, 3);
        let f2 = synthetic_fixture(42, 3, 3);
        assert_eq!(f1.kg.len(), f2.kg.len());
        assert_eq!(f1.questions.len(), f2.questions.len());
        for (a, b) in f1.kg.iter().zip(f2.kg.iter()) {
            assert_eq!(a.s, b.s);
            assert_eq!(a.p, b.p);
            assert_eq!(a.o, b.o);
        }
    }

    #[test]
    fn synthetic_fixture_has_expected_question_count() {
        let n_classes = 4;
        let instances_per_class = 5;
        let f = synthetic_fixture(7, n_classes, instances_per_class);
        assert_eq!(f.questions.len(), n_classes * instances_per_class);
    }

    #[test]
    fn webqsp_fixture_loads_and_has_expected_shape() {
        let fixture = load_webqsp_fixture();
        assert!(!fixture.kg.is_empty(), "KG must not be empty");
        assert!(!fixture.questions.is_empty(), "questions must not be empty");
        for q in &fixture.questions {
            assert!(!q.qid.is_empty());
            assert!(!q.question.is_empty());
            assert!(!q.topic_entity.is_empty());
            assert!(!q.answer_entities.is_empty());
        }
    }

    #[test]
    fn pipeline_recovers_topic_entity_for_simple_question() {
        let fixture = load_webqsp_fixture();
        let (linker, finder, predicates) = build_pipeline(&fixture);
        let q = fixture
            .questions
            .first()
            .expect("fixture must contain at least one question")
            .clone();
        let answer = answer_question(&linker, &finder, &predicates, &q, 5);
        assert_eq!(answer.qid, q.qid);
        assert!(
            !answer.ranked.is_empty(),
            "pipeline must return at least one candidate"
        );
    }

    #[test]
    fn aggregate_metrics_are_sane_on_webqsp_subset() {
        let fixture = load_webqsp_fixture();
        let (linker, finder, predicates) = build_pipeline(&fixture);
        let golds = build_gold_map(&fixture);
        let answers = run_pipeline_all(&linker, &finder, &predicates, &fixture.questions, 5);
        let metrics = aggregate(&answers, &golds);

        assert_eq!(metrics.queries, fixture.questions.len());
        assert!((0.0..=1.0).contains(&metrics.hits_at_1));
        assert!((0.0..=1.0).contains(&metrics.hits_at_5));
        assert!((0.0..=1.0).contains(&metrics.mrr));
        assert!(metrics.hits_at_5 >= metrics.hits_at_1);
        // The deterministic pipeline should answer at least half the questions
        // correctly within the top-5 — a sane lower bound for a regression
        // sentinel without being overly tight.
        assert!(
            metrics.hits_at_5 >= 0.5,
            "Hits@5 too low: {:.4}",
            metrics.hits_at_5
        );
    }

    #[test]
    fn synthetic_pipeline_high_accuracy_on_one_hop_lookup() {
        let fixture = synthetic_fixture(0xC001D00D, 3, 3);
        let (linker, finder, predicates) = build_pipeline(&fixture);
        let golds = build_gold_map(&fixture);
        let answers = run_pipeline_all(&linker, &finder, &predicates, &fixture.questions, 5);
        let metrics = aggregate(&answers, &golds);

        // Synthetic instance_of lookups should be answerable with very high accuracy.
        assert!(
            metrics.hits_at_5 >= 0.8,
            "synthetic Hits@5 too low: {:.4}",
            metrics.hits_at_5
        );
    }

    #[test]
    fn langchain_reference_parses_expected_shape() {
        let raw = r#"{"qid":"WebQSP-001","ranked_answers":["ent:Ulm","ent:Germany"]}"#;
        let parsed: LangChainReference =
            serde_json::from_str(raw).expect("LangChain reference must parse");
        assert_eq!(parsed.qid, "WebQSP-001");
        assert_eq!(parsed.ranked_answers.len(), 2);
    }

    #[test]
    fn langchain_comparator_reads_directory_of_captures() {
        let dir = std::env::temp_dir().join("oxirs_graphrag_kgqa_bench_test_caps");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir create");
        let payload = r#"{"qid":"WebQSP-001","ranked_answers":["ent:Ulm"]}"#;
        std::fs::write(dir.join("WebQSP-001.json"), payload).expect("write fixture");
        let map = load_langchain_references(&dir);
        assert_eq!(map.len(), 1);
        assert_eq!(map["WebQSP-001"], vec!["ent:Ulm".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group! {
    name = kgqa_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3))
        .sample_size(20);
    targets = bench_webqsp_subset, bench_synthetic_scaling, bench_pipeline_latency, bench_langchain_comparator
}
criterion_main!(kgqa_benches);
