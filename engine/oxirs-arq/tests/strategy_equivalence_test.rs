//! Serial / Parallel / Streaming strategy-equivalence matrix.
//!
//! The fuseki server forces `ExecutionStrategy::Serial` for every query because
//! the Parallel and Streaming evaluators historically diverged from Serial and
//! returned wrong answers. This suite pins that reality down as an executable
//! spec: it runs a representative set of operator trees under all three
//! strategies, treats **Serial as ground truth**, and asserts that each
//! `(query, strategy)` cell matches a documented expectation.
//!
//! Green means "the divergences are exactly where we say they are." An
//! *unexpected* divergence (a strategy silently drifting from Serial) fails the
//! suite loudly. As divergences are fixed, their expected cell flips from a
//! `Diverge*` verdict to `Match`.
//!
//! This is the groundwork that must land before the server's Serial-only pin can
//! ever be relaxed.

#![allow(clippy::uninlined_format_args)]

use std::collections::HashMap;

use oxirs_arq::algebra::{Aggregate, Algebra, Expression, GroupCondition, Term, TriplePattern};
use oxirs_arq::executor::ExecutionStrategy;
use oxirs_arq::query::{parse_query, ProjectionItem, Query};
use oxirs_arq::{Dataset, Literal, QueryExecutor, Variable};
use oxirs_core::model::NamedNode;

// ─────────────────────────────── fixtures ──────────────────────────────────

fn iri(s: &str) -> Term {
    Term::Iri(NamedNode::new_unchecked(s))
}

fn int(n: i64) -> Term {
    Term::Literal(Literal {
        value: n.to_string(),
        language: None,
        datatype: Some(NamedNode::new_unchecked(
            "http://www.w3.org/2001/XMLSchema#integer",
        )),
    })
}

fn plain(s: &str) -> Term {
    Term::Literal(Literal {
        value: s.to_string(),
        language: None,
        datatype: None,
    })
}

const P: &str = "http://example.org/p";
const Q: &str = "http://example.org/q";
const LINK: &str = "http://example.org/link";
const PREF: &str = "http://example.org/pref";
const ALT: &str = "http://example.org/alt";

/// A path-aware in-memory dataset. Mirrors the real store's predicate handling
/// (a length-one `PropertyPath::Iri` predicate is matched as a plain IRI) so a
/// parsed BGP predicate matches stored triples.
struct PathDataset {
    triples: Vec<(Term, Term, Term)>,
}

impl PathDataset {
    fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }
    fn add(&mut self, s: Term, p: Term, o: Term) {
        self.triples.push((s, p, o));
    }
}

fn normalize_predicate(t: &Term) -> Term {
    match t {
        Term::PropertyPath(oxirs_arq::algebra::PropertyPath::Iri(n)) => Term::Iri(n.clone()),
        other => other.clone(),
    }
}

impl Dataset for PathDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> anyhow::Result<Vec<(Term, Term, Term)>> {
        let want_pred = normalize_predicate(&pattern.predicate);
        let mut out = Vec::new();
        for (s, p, o) in &self.triples {
            let s_ok = matches!(pattern.subject, Term::Variable(_)) || &pattern.subject == s;
            let p_ok = matches!(want_pred, Term::Variable(_)) || &want_pred == p;
            let o_ok = matches!(pattern.object, Term::Variable(_)) || &pattern.object == o;
            if s_ok && p_ok && o_ok {
                out.push((s.clone(), p.clone(), o.clone()));
            }
        }
        Ok(out)
    }

    fn contains_triple(&self, s: &Term, p: &Term, o: &Term) -> anyhow::Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(a, b, c)| a == s && b == p && c == o))
    }

    fn subjects(&self) -> anyhow::Result<Vec<Term>> {
        let mut v: Vec<Term> = self.triples.iter().map(|(s, _, _)| s.clone()).collect();
        v.sort();
        v.dedup();
        Ok(v)
    }

    fn predicates(&self) -> anyhow::Result<Vec<Term>> {
        let mut v: Vec<Term> = self.triples.iter().map(|(_, p, _)| p.clone()).collect();
        v.sort();
        v.dedup();
        Ok(v)
    }

    fn objects(&self) -> anyhow::Result<Vec<Term>> {
        let mut v: Vec<Term> = self.triples.iter().map(|(_, _, o)| o.clone()).collect();
        v.sort();
        v.dedup();
        Ok(v)
    }
}

/// The shared dataset used by every query in the matrix.
///
/// * `s1 p 10`, `s2 p 20`, `s3 p 30` — the numeric FILTER / ORDER BY corpus.
/// * `s1 q 100`, `s3 q 300` — the "has a q" side for EXISTS / OPTIONAL / MINUS.
/// * `a link b`, `b link c` — the property-path chain for `link+`.
/// * the pref/alt rows — the UNION+FILTER corpus (`keep` survives, `drop` does not).
fn dataset() -> PathDataset {
    let mut d = PathDataset::new();
    d.add(iri("http://example.org/s1"), iri(P), int(10));
    d.add(iri("http://example.org/s2"), iri(P), int(20));
    d.add(iri("http://example.org/s3"), iri(P), int(30));
    d.add(iri("http://example.org/s1"), iri(Q), int(100));
    d.add(iri("http://example.org/s3"), iri(Q), int(300));
    d.add(
        iri("http://example.org/a"),
        iri(LINK),
        iri("http://example.org/b"),
    );
    d.add(
        iri("http://example.org/b"),
        iri(LINK),
        iri("http://example.org/c"),
    );
    d.add(iri("http://example.org/c1"), iri(PREF), plain("keep"));
    d.add(iri("http://example.org/c2"), iri(PREF), plain("drop"));
    d.add(iri("http://example.org/c3"), iri(ALT), plain("keep"));
    d.add(iri("http://example.org/c4"), iri(ALT), plain("drop"));
    d
}

// ────────────────────────── algebra construction ───────────────────────────

fn parse(q: &str) -> Query {
    parse_query(q).expect("query must parse")
}

/// The bare WHERE operator tree (Filter / Union / Minus / LeftJoin / …) with NO
/// solution modifiers wrapped around it. This is what exposes the Streaming
/// Filter divergence: a top-level `Filter` / `Union` is dispatched to the
/// streaming evaluator directly (wrapping it in `Project` would hide the bug
/// because the streaming evaluator delegates unknown top-level nodes to Serial).
fn where_of(q: &str) -> Algebra {
    parse(q).where_clause
}

/// Full SELECT algebra, assembled exactly like the fuseki
/// `build_select_algebra` (Group → OrderBy → Project → Slice). Used for the
/// modifier-centric cases (GROUP BY, LIMIT/OFFSET).
fn full_select(q: &str) -> Algebra {
    let query = parse(q);
    let mut alg = query.where_clause.clone();

    let aggregates: Vec<(Variable, Aggregate)> = query
        .projection_items
        .iter()
        .filter_map(|item| match item {
            ProjectionItem::Aggregate { aggregate, alias } => {
                Some((alias.clone(), aggregate.clone()))
            }
            _ => None,
        })
        .collect();
    let has_grouping = !aggregates.is_empty() || !query.group_by.is_empty();
    if has_grouping {
        let group_by: Vec<GroupCondition> = query.group_by.clone();
        alg = Algebra::Group {
            pattern: Box::new(alg),
            variables: group_by,
            aggregates,
        };
    }
    for item in &query.projection_items {
        if let ProjectionItem::Expression { expr, alias } = item {
            alg = Algebra::Extend {
                pattern: Box::new(alg),
                variable: alias.clone(),
                expr: expr.clone(),
            };
        }
    }
    if !query.order_by.is_empty() {
        alg = Algebra::OrderBy {
            pattern: Box::new(alg),
            conditions: query.order_by.clone(),
        };
    }
    if !query.select_variables.is_empty() {
        alg = Algebra::Project {
            pattern: Box::new(alg),
            variables: query.select_variables.clone(),
        };
    }
    if query.distinct {
        alg = Algebra::Distinct {
            pattern: Box::new(alg),
        };
    }
    if query.limit.is_some() || query.offset.is_some() {
        alg = Algebra::Slice {
            pattern: Box::new(alg),
            offset: query.offset,
            limit: query.limit,
        };
    }
    alg
}

// ─────────────────────────────── harness ───────────────────────────────────

/// Canonical, order-independent multiset representation of a solution.
fn canon(sol: &[HashMap<Variable, Term>]) -> Vec<String> {
    let mut rows: Vec<String> = sol
        .iter()
        .map(|b| {
            let mut parts: Vec<String> = b.iter().map(|(v, t)| format!("{}={}", v, t)).collect();
            parts.sort();
            parts.join("|")
        })
        .collect();
    rows.sort();
    rows
}

fn run(alg: &Algebra, ds: &dyn Dataset, strat: ExecutionStrategy) -> anyhow::Result<Vec<String>> {
    let mut ex = QueryExecutor::new();
    ex.set_strategy(strat);
    let (sol, _stats) = ex.execute(alg, ds)?;
    Ok(canon(&sol))
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Verdict {
    /// Both produced identical row multisets.
    Match,
    /// Both failed (equivalent observable failure).
    BothErr,
    /// Both succeeded but the row multisets differ.
    DivergeRows,
    /// Serial succeeded, the other strategy errored.
    OtherErr,
    /// Serial errored, the other strategy silently succeeded (the worst class:
    /// a wrong 200 where Serial fails loud).
    OtherOk,
}

fn verdict(serial: &anyhow::Result<Vec<String>>, other: &anyhow::Result<Vec<String>>) -> Verdict {
    match (serial, other) {
        (Ok(s), Ok(o)) => {
            if s == o {
                Verdict::Match
            } else {
                Verdict::DivergeRows
            }
        }
        (Err(_), Err(_)) => Verdict::BothErr,
        (Ok(_), Err(_)) => Verdict::OtherErr,
        (Err(_), Ok(_)) => Verdict::OtherOk,
    }
}

/// A `(query, strategy)` expectation. `Match`/`BothErr` are the "equivalent"
/// verdicts; anything else is a documented divergence.
struct Case {
    name: &'static str,
    alg: Algebra,
    parallel: Verdict,
    streaming: Verdict,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "filter_builtin",
            alg: where_of(&format!(
                "SELECT * WHERE {{ ?s <{P}> ?o FILTER(?o > 15) }}"
            )),
            // REMAINING (parallel): the parallel filter uses a distinct evaluator
            // (`crate::expression::ExpressionEvaluator`) that does not evaluate a
            // numeric `?o > 15` the way Serial's `evaluate_expression` does, so
            // every row is dropped. Root cause == the EXISTS / unknown-function
            // divergences below: the parallel path must be unified onto the
            // Serial evaluator. Streaming now applies the condition (Fix B).
            parallel: Verdict::DivergeRows,
            streaming: Verdict::Match,
        },
        Case {
            name: "union_filter",
            alg: where_of(&format!(
                "SELECT * WHERE {{ {{ ?c <{PREF}> ?l }} UNION {{ ?c <{ALT}> ?l }} FILTER(?l = \"keep\") }}"
            )),
            parallel: Verdict::Match,
            streaming: Verdict::Match,
        },
        Case {
            name: "exists",
            alg: where_of(&format!(
                "SELECT * WHERE {{ ?s <{P}> ?o FILTER EXISTS {{ ?s <{Q}> ?o2 }} }}"
            )),
            // REMAINING (parallel): the parallel evaluator bails on EXISTS/NOT
            // EXISTS ("requires query executor context") because it has no
            // dataset access on the rayon worker thread, so all rows drop.
            // Serial evaluates EXISTS via a thread-local dataset pointer that
            // does not cross the rayon boundary. Streaming routes the whole
            // Filter through Serial (Fix B), so EXISTS matches there.
            parallel: Verdict::DivergeRows,
            streaming: Verdict::Match,
        },
        Case {
            name: "minus",
            alg: where_of(&format!(
                "SELECT * WHERE {{ ?s <{P}> ?o MINUS {{ ?s <{Q}> ?o2 }} }}"
            )),
            parallel: Verdict::Match,
            streaming: Verdict::Match,
        },
        Case {
            name: "optional",
            alg: where_of(&format!(
                "SELECT * WHERE {{ ?s <{P}> ?o OPTIONAL {{ ?s <{Q}> ?o2 }} }}"
            )),
            parallel: Verdict::Match,
            streaming: Verdict::Match,
        },
        Case {
            name: "property_path",
            alg: where_of(&format!("SELECT * WHERE {{ ?s <{LINK}>+ ?o }}")),
            // REMAINING (parallel): `execute_parallel_property_path` returns no
            // rows for a `+` (OneOrMore) transitive closure where Serial's
            // `execute_property_path` returns the full closure. A separate,
            // larger fix in the parallel path engine. Streaming delegates the
            // PropertyPath node to Serial, so it matches.
            parallel: Verdict::DivergeRows,
            streaming: Verdict::Match,
        },
        Case {
            name: "two_pattern_join",
            // A 2-pattern BGP: `?s p ?o . ?s q ?o2`. If the parser emits a single
            // `Bgp[p1, p2]`, the parallel executor chunks it and `merge_bgp_results`
            // concatenates the partial solutions instead of joining them — a
            // potential wrong-answer path. If the parser emits `Join(Bgp, Bgp)`,
            // the parallel hash-join path runs instead.
            alg: where_of(&format!(
                "SELECT * WHERE {{ ?s <{P}> ?o . ?s <{Q}> ?o2 }}"
            )),
            parallel: Verdict::Match,
            streaming: Verdict::Match,
        },
        Case {
            name: "group_by",
            alg: full_select("SELECT ?p (COUNT(?o) AS ?n) WHERE { ?s ?p ?o } GROUP BY ?p"),
            parallel: Verdict::Match,
            streaming: Verdict::Match,
        },
        Case {
            name: "limit_offset",
            alg: full_select(&format!(
                "SELECT ?s WHERE {{ ?s <{P}> ?o }} ORDER BY ?s LIMIT 2 OFFSET 1"
            )),
            parallel: Verdict::Match,
            streaming: Verdict::Match,
        },
        Case {
            name: "unknown_function",
            // Built by hand: the parser does not accept the `<iri>(args)` custom
            // call syntax, so we wrap the parsed BGP in a Filter whose condition
            // is a function the evaluator does not implement. Serial raises a
            // typed `UnknownFunctionError` (a whole-query fault); the divergence
            // question is whether Parallel/Streaming also fail loud.
            alg: Algebra::Filter {
                pattern: Box::new(where_of(&format!("SELECT * WHERE {{ ?s <{P}> ?o }}"))),
                condition: Expression::Function {
                    name: "nosuchfn".to_string(),
                    args: vec![Expression::Variable(
                        Variable::new("o").expect("valid variable"),
                    )],
                },
            },
            // REMAINING (parallel): Serial raises a typed `UnknownFunctionError`
            // (whole-query fault) and fails loud; the parallel evaluator raises an
            // *untyped* `anyhow!("Unknown function")`, which Fix C's classifier
            // does not recognize as a whole-query fault, so the row is dropped and
            // parallel returns an empty `200 OK` (OtherOk — Serial errs, parallel
            // succeeds). Same root cause as filter_builtin/exists: distinct
            // evaluator. Streaming now routes the Filter through Serial (Fix B),
            // so it fails loud identically (BothErr).
            parallel: Verdict::OtherOk,
            streaming: Verdict::BothErr,
        },
    ]
}

#[test]
fn strategy_equivalence_matrix() {
    let ds = dataset();
    let cases = cases();

    let mut report = String::from(
        "\nstrategy-equivalence matrix (Serial = truth)\n\
         query                | parallel        | streaming\n\
         ---------------------+-----------------+-----------------\n",
    );
    let mut mismatches: Vec<String> = Vec::new();

    for case in &cases {
        let serial = run(&case.alg, &ds, ExecutionStrategy::Serial);
        let par = run(&case.alg, &ds, ExecutionStrategy::Parallel);
        let stream = run(&case.alg, &ds, ExecutionStrategy::Streaming);

        let par_v = verdict(&serial, &par);
        let stream_v = verdict(&serial, &stream);

        report.push_str(&format!(
            "{:<20} | {:<15?} | {:<15?}\n",
            case.name, par_v, stream_v
        ));
        if std::env::var("DUMP_ROWS").is_ok() {
            report.push_str(&format!("   serial   = {:?}\n", serial));
            report.push_str(&format!("   parallel = {:?}\n", par));
            report.push_str(&format!("   streaming= {:?}\n", stream));
        }

        if par_v != case.parallel {
            mismatches.push(format!(
                "{}: parallel expected {:?}, got {:?}",
                case.name, case.parallel, par_v
            ));
        }
        if stream_v != case.streaming {
            mismatches.push(format!(
                "{}: streaming expected {:?}, got {:?}",
                case.name, case.streaming, stream_v
            ));
        }
    }

    println!("{report}");

    assert!(
        mismatches.is_empty(),
        "strategy divergences differ from the documented expectation:\n{}\n{report}",
        mismatches.join("\n")
    );
}
