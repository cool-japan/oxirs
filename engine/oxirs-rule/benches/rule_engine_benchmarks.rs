//! Comprehensive benchmarks for the OxiRS rule engine
//!
//! This module provides detailed performance testing for all aspects of the rule engine
//! including forward chaining, backward chaining, RETE networks, and built-in predicates.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_rule::{
    integration::{rule_builders, RuleIntegration},
    rete::ReteNetwork,
    swrl::{SwrlArgument, SwrlEngine},
    Rule, RuleAtom, RuleEngine, Term,
};

/// Generate test facts for benchmarking
fn generate_test_facts(size: usize) -> Vec<RuleAtom> {
    let mut facts = Vec::with_capacity(size);

    for i in 0..size {
        // Generate person facts
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("http://example.org/person{i}")),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("http://example.org/Person".to_string()),
        });

        // Generate age facts
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("http://example.org/person{i}")),
            predicate: Term::Constant("http://example.org/hasAge".to_string()),
            object: Term::Literal((20 + i % 60).to_string()),
        });

        // Generate some relationships
        if i > 0 {
            facts.push(RuleAtom::Triple {
                subject: Term::Constant(format!("http://example.org/person{i}")),
                predicate: Term::Constant("http://example.org/knows".to_string()),
                object: Term::Constant(format!("http://example.org/person{prev_i}", prev_i = i - 1)),
            });
        }
    }

    facts
}

/// Generate test rules for benchmarking
fn generate_test_rules(complexity: usize) -> Vec<Rule> {
    let mut rules = Vec::new();

    // Basic type inheritance rules
    for i in 0..complexity {
        rules.push(Rule {
            name: format!("rule_{i}"),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                    ),
                    object: Term::Constant("http://example.org/Person".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("http://example.org/hasAge".to_string()),
                    object: Term::Variable("Age".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                object: Term::Constant(format!("http://example.org/Adult{i}")),
            }],
        });
    }

    // Add RDFS reasoning rules
    rules.extend(rule_builders::all_rdfs_rules());

    rules
}

/// Benchmark forward chaining performance
fn benchmark_forward_chaining(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_chaining");

    for fact_size in [100, 500, 1000, 5000].iter() {
        for rule_count in [5, 10, 20].iter() {
            let facts = generate_test_facts(*fact_size);
            let rules = generate_test_rules(*rule_count);

            group.throughput(Throughput::Elements(*fact_size as u64));
            group.bench_with_input(
                BenchmarkId::new("facts_rules", format!("{fact_size}_{rule_count}")),
                &(facts, rules),
                |b, (facts, rules)| {
                    b.iter(|| {
                        let mut engine = RuleEngine::new();
                        for rule in rules.iter() {
                            engine.add_rule(rule.clone());
                        }
                        engine.forward_chain(facts).unwrap()
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark backward chaining performance
fn benchmark_backward_chaining(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_chaining");

    for fact_size in [100, 500, 1000].iter() {
        let facts = generate_test_facts(*fact_size);
        let rules = generate_test_rules(10);

        let goal = RuleAtom::Triple {
            subject: Term::Constant("http://example.org/person0".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("http://example.org/Adult0".to_string()),
        };

        group.throughput(Throughput::Elements(*fact_size as u64));
        group.bench_with_input(
            BenchmarkId::new("goal_proving", fact_size),
            &(facts, rules, goal),
            |b, (facts, rules, goal)| {
                b.iter(|| {
                    let mut engine = RuleEngine::new();
                    engine.add_facts(facts.clone());
                    for rule in rules.iter() {
                        engine.add_rule(rule.clone());
                    }
                    engine.backward_chain(goal).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RETE network performance
fn benchmark_rete_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("rete_network");

    for fact_size in [100, 500, 1000, 2000].iter() {
        let facts = generate_test_facts(*fact_size);
        let rules = generate_test_rules(15);

        group.throughput(Throughput::Elements(*fact_size as u64));
        group.bench_with_input(
            BenchmarkId::new("rete_execution", fact_size),
            &(facts, rules),
            |b, (facts, rules)| {
                b.iter(|| {
                    let mut rete = ReteNetwork::new();

                    // Add rules to RETE network
                    for rule in rules.iter() {
                        rete.add_rule(rule).unwrap();
                    }

                    // Process facts
                    for fact in facts.iter() {
                        rete.add_fact(fact.clone()).unwrap();
                    }

                    rete.get_facts()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SWRL built-in predicates
fn benchmark_swrl_builtins(c: &mut Criterion) {
    let mut group = c.benchmark_group("swrl_builtins");

    let _engine = SwrlEngine::new();

    // Mathematical operations
    group.bench_function("math_operations", |b| {
        b.iter(|| {
            let args = vec![
                SwrlArgument::Literal("5.5".to_string()),
                SwrlArgument::Literal("3.2".to_string()),
                SwrlArgument::Literal("8.7".to_string()),
            ];

            // Test multiple mathematical operations
            for _ in 0..100 {
                oxirs_rule::swrl::builtin_add(&args).unwrap();
                oxirs_rule::swrl::builtin_multiply(&args).unwrap();

                let pow_args = vec![
                    SwrlArgument::Literal("2.0".to_string()),
                    SwrlArgument::Literal("3.0".to_string()),
                    SwrlArgument::Literal("8.0".to_string()),
                ];
                oxirs_rule::swrl::builtin_pow(&pow_args).unwrap();
            }
        });
    });

    // String operations
    group.bench_function("string_operations", |b| {
        b.iter(|| {
            for i in 0..100 {
                let concat_args = vec![
                    SwrlArgument::Literal(format!("Hello{i}")),
                    SwrlArgument::Literal(" World".to_string()),
                    SwrlArgument::Literal(format!("Hello{i} World")),
                ];
                oxirs_rule::swrl::builtin_string_concat(&concat_args).unwrap();

                let upper_args = vec![
                    SwrlArgument::Literal("test string".to_string()),
                    SwrlArgument::Literal("TEST STRING".to_string()),
                ];
                oxirs_rule::swrl::builtin_upper_case(&upper_args).unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark rule integration with oxirs-core
fn benchmark_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("integration");

    for fact_size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*fact_size as u64));
        group.bench_with_input(
            BenchmarkId::new("core_integration", fact_size),
            fact_size,
            |b, &fact_size| {
                b.iter(|| {
                    let mut integration = RuleIntegration::new();

                    // Add RDFS rules
                    integration.add_rules(rule_builders::all_rdfs_rules());

                    // Generate and add RDF triples
                    for i in 0..fact_size {
                        let subject =
                            oxirs_core::NamedNode::new(format!("http://example.org/person{i}"))
                                .unwrap();
                        let predicate = oxirs_core::NamedNode::new(
                            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                        )
                        .unwrap();
                        let object =
                            oxirs_core::NamedNode::new("http://example.org/Person").unwrap();

                        let triple = oxirs_core::Triple::new(subject, predicate, object);
                        integration.store.insert_triple(triple).unwrap();
                    }

                    // Apply rules and measure performance
                    integration.apply_rules().unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    group.bench_function("rule_engine_memory", |b| {
        b.iter(|| {
            let mut engine = RuleEngine::new();
            let facts = generate_test_facts(1000);
            let rules = generate_test_rules(20);

            // Add all facts and rules
            engine.add_facts(facts);
            for rule in rules {
                engine.add_rule(rule);
            }

            // Perform reasoning
            let initial_facts = engine.get_facts().clone();
            engine.forward_chain(&initial_facts).unwrap();
        });
    });

    group.bench_function("rete_memory", |b| {
        b.iter(|| {
            let mut rete = ReteNetwork::new();
            let facts = generate_test_facts(1000);
            let rules = generate_test_rules(15);

            // Build RETE network
            for rule in rules {
                rete.add_rule(&rule).unwrap();
            }

            // Process all facts
            for fact in facts {
                rete.add_fact(fact).unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark rule complexity scaling
fn benchmark_rule_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_complexity");

    let facts = generate_test_facts(500);

    for rule_complexity in [1, 2, 3, 5, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("complex_rules", rule_complexity),
            rule_complexity,
            |b, &complexity| {
                b.iter(|| {
                    let mut engine = RuleEngine::new();

                    // Generate complex rules with multiple body conditions
                    let rule = Rule {
                        name: "complex_rule".to_string(),
                        body: (0..complexity)
                            .map(|i| RuleAtom::Triple {
                                subject: Term::Variable(format!("X{i}")),
                                predicate: Term::Constant(
                                    "http://example.org/hasProperty".to_string(),
                                ),
                                object: Term::Variable(format!("Y{i}")),
                            })
                            .collect(),
                        head: vec![RuleAtom::Triple {
                            subject: Term::Variable("X0".to_string()),
                            predicate: Term::Constant("http://example.org/isComplex".to_string()),
                            object: Term::Constant("true".to_string()),
                        }],
                    };

                    engine.add_rule(rule);
                    engine.forward_chain(&facts).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark concurrent rule execution
fn benchmark_concurrent_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_execution");

    group.bench_function("parallel_rule_application", |b| {
        b.iter(|| {
            use std::sync::{Arc, Mutex};
            use std::thread;

            let facts = Arc::new(generate_test_facts(200));
            let rules = Arc::new(generate_test_rules(10));
            let results = Arc::new(Mutex::new(Vec::new()));

            let mut handles = Vec::new();

            // Spawn multiple threads to process rules concurrently
            for thread_id in 0..4 {
                let facts_clone = Arc::clone(&facts);
                let rules_clone = Arc::clone(&rules);
                let results_clone = Arc::clone(&results);

                let handle = thread::spawn(move || {
                    let mut engine = RuleEngine::new();

                    // Each thread processes a subset of rules
                    let chunk_size = rules_clone.len() / 4;
                    let start = thread_id * chunk_size;
                    let end = if thread_id == 3 {
                        rules_clone.len()
                    } else {
                        start + chunk_size
                    };

                    for rule in &rules_clone[start..end] {
                        engine.add_rule(rule.clone());
                    }

                    let result = engine.forward_chain(&facts_clone).unwrap();

                    let mut results_guard = results_clone.lock().unwrap();
                    results_guard.push(result);
                });

                handles.push(handle);
            }

            // Wait for all threads to complete
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

/// Benchmark W3C RDFS test suite compliance
fn benchmark_w3c_rdfs_compliance(c: &mut Criterion) {
    let mut group = c.benchmark_group("w3c_rdfs_compliance");

    // RDFS test cases based on W3C test suite
    let rdfs_test_cases = vec![
        // rdfs2: (aaa rdfs:domain xxx), (yyy aaa zzz) -> (yyy rdf:type xxx)
        (
            "rdfs2",
            vec![
                RuleAtom::Triple {
                    subject: Term::Constant("http://example.org/property".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/2000/01/rdf-schema#domain".to_string(),
                    ),
                    object: Term::Constant("http://example.org/DomainClass".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Constant("http://example.org/subject".to_string()),
                    predicate: Term::Constant("http://example.org/property".to_string()),
                    object: Term::Constant("http://example.org/object".to_string()),
                },
            ],
        ),
        // rdfs7: (aaa rdfs:subPropertyOf bbb), (xxx aaa yyy) -> (xxx bbb yyy)
        (
            "rdfs7",
            vec![
                RuleAtom::Triple {
                    subject: Term::Constant("http://example.org/subProperty".to_string()),
                    predicate: Term::Constant(
                        "http://www.w3.org/2000/01/rdf-schema#subPropertyOf".to_string(),
                    ),
                    object: Term::Constant("http://example.org/superProperty".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Constant("http://example.org/subject".to_string()),
                    predicate: Term::Constant("http://example.org/subProperty".to_string()),
                    object: Term::Constant("http://example.org/object".to_string()),
                },
            ],
        ),
    ];

    for (test_name, facts) in rdfs_test_cases {
        group.bench_function(test_name, |b| {
            b.iter(|| {
                let mut engine = RuleEngine::new();

                // Add basic reasoning rule
                engine.add_rule(Rule {
                    name: "basic_inference".to_string(),
                    body: vec![RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Variable("P".to_string()),
                        object: Term::Variable("Y".to_string()),
                    }],
                    head: vec![RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("http://example.org/related".to_string()),
                        object: Term::Variable("Y".to_string()),
                    }],
                });

                // Execute reasoning
                engine.forward_chain(&facts).unwrap();
            });
        });
    }

    group.finish();
}

/// Large-scale performance benchmark
fn benchmark_large_scale_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_performance");
    group.sample_size(10); // Fewer samples for large-scale tests

    // Test with datasets of different sizes
    for size in [1_000, 5_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("large_dataset", size), size, |b, &size| {
            b.iter(|| {
                let mut engine = RuleEngine::new();

                // Add basic reasoning rules
                engine.add_rule(Rule {
                    name: "type_inference".to_string(),
                    body: vec![RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("http://example.org/hasAge".to_string()),
                        object: Term::Variable("Age".to_string()),
                    }],
                    head: vec![RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant(
                            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                        ),
                        object: Term::Constant("http://example.org/AgedEntity".to_string()),
                    }],
                });

                // Generate large dataset
                let facts = generate_test_facts(size);

                // Perform reasoning
                engine.forward_chain(&facts).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_forward_chaining,
    benchmark_backward_chaining,
    benchmark_rete_network,
    benchmark_swrl_builtins,
    benchmark_integration,
    benchmark_memory_usage,
    benchmark_rule_complexity,
    benchmark_concurrent_execution,
    benchmark_w3c_rdfs_compliance,
    benchmark_large_scale_performance
);

criterion_main!(benches);
