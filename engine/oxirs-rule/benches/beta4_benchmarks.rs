//! # Beta.4 Features Performance Benchmarks
//!
//! Benchmarks for RIF, CHR, and ASP features added in Beta.4
//!
//! Run with: `cargo bench --bench beta4_benchmarks`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use oxirs_rule::asp::{AspLiteral, AspSolver, AspTerm, Atom};
use oxirs_rule::chr::{ChrEngine, ChrRule, ChrTerm, Constraint};
use oxirs_rule::rif::{RifDialect, RifParser, RifSerializer};

/// Benchmark RIF parsing performance
fn bench_rif_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("RIF_Parsing");

    // Small RIF document (10 rules)
    let small_rif = generate_rif_document(10);

    // Medium RIF document (100 rules)
    let medium_rif = generate_rif_document(100);

    // Large RIF document (500 rules)
    let large_rif = generate_rif_document(500);

    group.bench_with_input(BenchmarkId::new("parse", "small"), &small_rif, |b, rif| {
        b.iter(|| {
            let mut parser = RifParser::new(RifDialect::Bld);
            parser.parse(black_box(rif)).unwrap()
        });
    });

    group.bench_with_input(
        BenchmarkId::new("parse", "medium"),
        &medium_rif,
        |b, rif| {
            b.iter(|| {
                let mut parser = RifParser::new(RifDialect::Bld);
                parser.parse(black_box(rif)).unwrap()
            });
        },
    );

    group.bench_with_input(BenchmarkId::new("parse", "large"), &large_rif, |b, rif| {
        b.iter(|| {
            let mut parser = RifParser::new(RifDialect::Bld);
            parser.parse(black_box(rif)).unwrap()
        });
    });

    group.finish();
}

/// Benchmark RIF serialization performance
fn bench_rif_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("RIF_Serialization");

    let mut parser = RifParser::new(RifDialect::Bld);

    // Parse documents once
    let small_doc = parser.parse(&generate_rif_document(10)).unwrap();
    let medium_doc = parser.parse(&generate_rif_document(100)).unwrap();
    let large_doc = parser.parse(&generate_rif_document(500)).unwrap();

    let serializer = RifSerializer::new(RifDialect::Bld);

    group.bench_with_input(
        BenchmarkId::new("serialize", "small"),
        &small_doc,
        |b, doc| {
            b.iter(|| serializer.serialize(black_box(doc)).unwrap());
        },
    );

    group.bench_with_input(
        BenchmarkId::new("serialize", "medium"),
        &medium_doc,
        |b, doc| {
            b.iter(|| serializer.serialize(black_box(doc)).unwrap());
        },
    );

    group.bench_with_input(
        BenchmarkId::new("serialize", "large"),
        &large_doc,
        |b, doc| {
            b.iter(|| serializer.serialize(black_box(doc)).unwrap());
        },
    );

    group.finish();
}

/// Benchmark RIF to OxiRS rule conversion
fn bench_rif_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("RIF_Conversion");

    let mut parser = RifParser::new(RifDialect::Bld);

    let small_doc = parser.parse(&generate_rif_document(10)).unwrap();
    let medium_doc = parser.parse(&generate_rif_document(100)).unwrap();
    let large_doc = parser.parse(&generate_rif_document(500)).unwrap();

    group.bench_with_input(
        BenchmarkId::new("to_oxirs_rules", "small"),
        &small_doc,
        |b, doc| {
            b.iter(|| black_box(doc).to_oxirs_rules().unwrap());
        },
    );

    group.bench_with_input(
        BenchmarkId::new("to_oxirs_rules", "medium"),
        &medium_doc,
        |b, doc| {
            b.iter(|| black_box(doc).to_oxirs_rules().unwrap());
        },
    );

    group.bench_with_input(
        BenchmarkId::new("to_oxirs_rules", "large"),
        &large_doc,
        |b, doc| {
            b.iter(|| black_box(doc).to_oxirs_rules().unwrap());
        },
    );

    group.finish();
}

/// Benchmark CHR constraint solving
fn bench_chr_solving(c: &mut Criterion) {
    let mut group = c.benchmark_group("CHR_Solving");

    // Benchmark LEQ constraint system with different sizes
    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("leq_constraints", size), size, |b, &n| {
            b.iter(|| {
                let mut engine = create_leq_engine();

                // Add n constraints forming a chain
                for i in 0..n {
                    engine.add_constraint(Constraint::new(
                        "leq",
                        vec![
                            ChrTerm::const_(&format!("x{}", i)),
                            ChrTerm::const_(&format!("x{}", i + 1)),
                        ],
                    ));
                }

                black_box(engine.solve().unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark CHR rule application
fn bench_chr_rules(c: &mut Criterion) {
    let mut group = c.benchmark_group("CHR_Rules");

    group.bench_function("simplification_rule", |b| {
        b.iter(|| {
            let mut engine = ChrEngine::new();

            // Add simplification rule
            engine.add_rule(ChrRule::simplification(
                "reflexivity",
                vec![Constraint::binary("leq", "X", "X")],
                vec![],
                vec![],
            ));

            // Add constraints that trigger the rule
            for i in 0..10 {
                engine.add_constraint(Constraint::binary(
                    "leq",
                    &format!("a{}", i),
                    &format!("a{}", i),
                ));
            }

            black_box(engine.solve().unwrap())
        });
    });

    group.bench_function("propagation_rule", |b| {
        b.iter(|| {
            let mut engine = ChrEngine::new();

            // Add transitivity rule
            engine.add_rule(ChrRule::propagation(
                "transitivity",
                vec![
                    Constraint::binary("leq", "X", "Y"),
                    Constraint::binary("leq", "Y", "Z"),
                ],
                vec![],
                vec![Constraint::binary("leq", "X", "Z")],
            ));

            // Add chain of constraints
            for i in 0..10 {
                engine.add_constraint(Constraint::binary(
                    "leq",
                    &format!("a{}", i),
                    &format!("a{}", i + 1),
                ));
            }

            black_box(engine.solve().unwrap())
        });
    });

    group.finish();
}

/// Benchmark ASP solving performance
fn bench_asp_solving(c: &mut Criterion) {
    let mut group = c.benchmark_group("ASP_Solving");

    // Benchmark graph coloring with different sizes
    for size in [3, 5, 7].iter() {
        group.bench_with_input(BenchmarkId::new("graph_coloring", size), size, |b, &n| {
            b.iter(|| {
                let mut solver = AspSolver::new();

                // Add nodes
                for i in 0..n {
                    solver.add_fact(&format!("node(n{})", i)).unwrap();
                }

                // Add edges (complete graph)
                for i in 0..n {
                    for j in (i + 1)..n {
                        solver.add_fact(&format!("edge(n{}, n{})", i, j)).unwrap();
                    }
                }

                // Add choice rules for 3-coloring
                for i in 0..n {
                    let colors: Vec<_> = ["red", "green", "blue"]
                        .iter()
                        .map(|c| {
                            Atom::new(
                                "color",
                                vec![AspTerm::constant(&format!("n{}", i)), AspTerm::constant(c)],
                            )
                        })
                        .collect();

                    solver.add_choice_rule(
                        colors,
                        Some(1),
                        Some(1),
                        vec![AspLiteral::positive(Atom::new(
                            "node",
                            vec![AspTerm::constant(&format!("n{}", i))],
                        ))],
                    );
                }

                // Add integrity constraints
                for i in 0..n {
                    for j in (i + 1)..n {
                        for color in &["red", "green", "blue"] {
                            solver.add_constraint(vec![
                                AspLiteral::positive(Atom::new(
                                    "edge",
                                    vec![
                                        AspTerm::constant(&format!("n{}", i)),
                                        AspTerm::constant(&format!("n{}", j)),
                                    ],
                                )),
                                AspLiteral::positive(Atom::new(
                                    "color",
                                    vec![
                                        AspTerm::constant(&format!("n{}", i)),
                                        AspTerm::constant(color),
                                    ],
                                )),
                                AspLiteral::positive(Atom::new(
                                    "color",
                                    vec![
                                        AspTerm::constant(&format!("n{}", j)),
                                        AspTerm::constant(color),
                                    ],
                                )),
                            ]);
                        }
                    }
                }

                black_box(solver.solve().unwrap())
            });
        });
    }

    group.finish();
}

/// Benchmark ASP grounding
fn bench_asp_grounding(c: &mut Criterion) {
    let mut group = c.benchmark_group("ASP_Grounding");

    for domain_size in [5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("grounding", domain_size),
            domain_size,
            |b, &n| {
                b.iter(|| {
                    let mut solver = AspSolver::new();

                    // Add domain facts
                    for i in 0..n {
                        solver.add_fact(&format!("element(e{})", i)).unwrap();
                    }

                    // Add a rule with 2 variables (creates n^2 groundings)
                    solver.add_choice_rule(
                        vec![Atom::new(
                            "related",
                            vec![AspTerm::variable("X"), AspTerm::variable("Y")],
                        )],
                        Some(0),
                        None,
                        vec![
                            AspLiteral::positive(Atom::new(
                                "element",
                                vec![AspTerm::variable("X")],
                            )),
                            AspLiteral::positive(Atom::new(
                                "element",
                                vec![AspTerm::variable("Y")],
                            )),
                        ],
                    );

                    black_box(solver.solve().unwrap())
                });
            },
        );
    }

    group.finish();
}

// Helper functions

fn generate_rif_document(num_rules: usize) -> String {
    let mut doc = String::from(
        r#"Prefix(ex <http://example.org/>)
Prefix(rdf <http://www.w3.org/1999/02/22-rdf-syntax-ns#>)

Group (
"#,
    );

    for i in 0..num_rules {
        doc.push_str(&format!(
            r#"    Forall ?x ?y (
        ex:derived{}(?x ?y) :- ex:source{}(?x ?y)
    )

"#,
            i, i
        ));
    }

    doc.push_str(")\n");
    doc
}

fn create_leq_engine() -> ChrEngine {
    let mut engine = ChrEngine::new();

    // Add standard LEQ rules
    engine.add_rule(ChrRule::simplification(
        "reflexivity",
        vec![Constraint::binary("leq", "X", "X")],
        vec![],
        vec![],
    ));

    engine.add_rule(ChrRule::simplification(
        "antisymmetry",
        vec![
            Constraint::binary("leq", "X", "Y"),
            Constraint::binary("leq", "Y", "X"),
        ],
        vec![],
        vec![Constraint::eq("X", "Y")],
    ));

    engine.add_rule(ChrRule::propagation(
        "transitivity",
        vec![
            Constraint::binary("leq", "X", "Y"),
            Constraint::binary("leq", "Y", "Z"),
        ],
        vec![],
        vec![Constraint::binary("leq", "X", "Z")],
    ));

    engine
}

criterion_group!(
    beta4_benches,
    bench_rif_parsing,
    bench_rif_serialization,
    bench_rif_conversion,
    bench_chr_solving,
    bench_chr_rules,
    bench_asp_solving,
    bench_asp_grounding
);

criterion_main!(beta4_benches);
