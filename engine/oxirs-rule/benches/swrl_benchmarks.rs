//! SWRL-specific benchmarks for built-in predicates and rule execution
//!
//! This module focuses on benchmarking SWRL rule execution performance,
//! built-in predicate performance, and complex rule scenarios.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_rule::{
    swrl::{SwrlArgument, SwrlAtom, SwrlEngine, SwrlRule},
    RuleAtom, Term,
};
use std::collections::HashMap;

/// Generate SWRL test rules
fn generate_swrl_rules(count: usize) -> Vec<SwrlRule> {
    let mut rules = Vec::new();

    for i in 0..count {
        rules.push(SwrlRule {
            id: format!("swrl_rule_{}", i),
            body: vec![
                SwrlAtom::Class {
                    class_predicate: "http://example.org/Person".to_string(),
                    argument: SwrlArgument::Variable("x".to_string()),
                },
                SwrlAtom::Builtin {
                    builtin_predicate: "http://www.w3.org/2003/11/swrlb#greaterThan".to_string(),
                    arguments: vec![
                        SwrlArgument::Variable("age".to_string()),
                        SwrlArgument::Literal("18".to_string()),
                    ],
                },
            ],
            head: vec![SwrlAtom::Class {
                class_predicate: format!("http://example.org/Adult{}", i),
                argument: SwrlArgument::Variable("x".to_string()),
            }],
            metadata: HashMap::new(),
        });
    }

    rules
}

/// Generate complex SWRL rules with multiple built-ins
fn generate_complex_swrl_rules() -> Vec<SwrlRule> {
    vec![
        SwrlRule {
            id: "complex_math_rule".to_string(),
            body: vec![
                SwrlAtom::DatavalueProperty {
                    property_predicate: "http://example.org/hasHeight".to_string(),
                    argument1: SwrlArgument::Variable("person".to_string()),
                    argument2: SwrlArgument::Variable("height".to_string()),
                },
                SwrlAtom::DatavalueProperty {
                    property_predicate: "http://example.org/hasWeight".to_string(),
                    argument1: SwrlArgument::Variable("person".to_string()),
                    argument2: SwrlArgument::Variable("weight".to_string()),
                },
                SwrlAtom::Builtin {
                    builtin_predicate: "http://www.w3.org/2003/11/swrlb#multiply".to_string(),
                    arguments: vec![
                        SwrlArgument::Variable("height".to_string()),
                        SwrlArgument::Variable("height".to_string()),
                        SwrlArgument::Variable("height_squared".to_string()),
                    ],
                },
                SwrlAtom::Builtin {
                    builtin_predicate: "http://www.w3.org/2003/11/swrlb#divide".to_string(),
                    arguments: vec![
                        SwrlArgument::Variable("weight".to_string()),
                        SwrlArgument::Variable("height_squared".to_string()),
                        SwrlArgument::Variable("bmi".to_string()),
                    ],
                },
                SwrlAtom::Builtin {
                    builtin_predicate: "http://www.w3.org/2003/11/swrlb#greaterThan".to_string(),
                    arguments: vec![
                        SwrlArgument::Variable("bmi".to_string()),
                        SwrlArgument::Literal("25.0".to_string()),
                    ],
                },
            ],
            head: vec![SwrlAtom::Class {
                class_predicate: "http://example.org/Overweight".to_string(),
                argument: SwrlArgument::Variable("person".to_string()),
            }],
            metadata: HashMap::new(),
        },
        SwrlRule {
            id: "string_processing_rule".to_string(),
            body: vec![
                SwrlAtom::DatavalueProperty {
                    property_predicate: "http://example.org/hasName".to_string(),
                    argument1: SwrlArgument::Variable("person".to_string()),
                    argument2: SwrlArgument::Variable("name".to_string()),
                },
                SwrlAtom::Builtin {
                    builtin_predicate: "http://www.w3.org/2003/11/swrlb#stringLength".to_string(),
                    arguments: vec![
                        SwrlArgument::Variable("name".to_string()),
                        SwrlArgument::Variable("name_length".to_string()),
                    ],
                },
                SwrlAtom::Builtin {
                    builtin_predicate: "http://www.w3.org/2003/11/swrlb#greaterThan".to_string(),
                    arguments: vec![
                        SwrlArgument::Variable("name_length".to_string()),
                        SwrlArgument::Literal("10".to_string()),
                    ],
                },
                SwrlAtom::Builtin {
                    builtin_predicate: "http://www.w3.org/2003/11/swrlb#upperCase".to_string(),
                    arguments: vec![
                        SwrlArgument::Variable("name".to_string()),
                        SwrlArgument::Variable("upper_name".to_string()),
                    ],
                },
            ],
            head: vec![SwrlAtom::DatavalueProperty {
                property_predicate: "http://example.org/hasDisplayName".to_string(),
                argument1: SwrlArgument::Variable("person".to_string()),
                argument2: SwrlArgument::Variable("upper_name".to_string()),
            }],
            metadata: HashMap::new(),
        },
    ]
}

/// Generate test facts for SWRL execution
fn generate_swrl_facts(size: usize) -> Vec<RuleAtom> {
    let mut facts = Vec::new();

    for i in 0..size {
        // Person type
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("http://example.org/person{}", i)),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("http://example.org/Person".to_string()),
        });

        // Age property
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("http://example.org/person{}", i)),
            predicate: Term::Constant("http://example.org/hasAge".to_string()),
            object: Term::Literal((18 + i % 50).to_string()),
        });

        // Name property
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("http://example.org/person{}", i)),
            predicate: Term::Constant("http://example.org/hasName".to_string()),
            object: Term::Literal(format!("Person Number {}", i)),
        });

        // Height and weight for complex rules
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("http://example.org/person{}", i)),
            predicate: Term::Constant("http://example.org/hasHeight".to_string()),
            object: Term::Literal(format!("{:.2}", 1.60 + (i as f64 * 0.001))),
        });

        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("http://example.org/person{}", i)),
            predicate: Term::Constant("http://example.org/hasWeight".to_string()),
            object: Term::Literal(format!("{:.1}", 60.0 + (i as f64 * 0.5))),
        });
    }

    facts
}

/// Benchmark SWRL rule execution
fn benchmark_swrl_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("swrl_execution");

    for fact_size in [50, 100, 200, 500].iter() {
        for rule_count in [5, 10, 15].iter() {
            let facts = generate_swrl_facts(*fact_size);
            let rules = generate_swrl_rules(*rule_count);

            group.throughput(Throughput::Elements(*fact_size as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    "rule_execution",
                    format!("{}facts_{}rules", fact_size, rule_count),
                ),
                &(facts, rules),
                |b, (facts, rules)| {
                    b.iter(|| {
                        let mut engine = SwrlEngine::new();

                        // Add all rules
                        for rule in rules.iter() {
                            engine.add_rule(rule.clone()).unwrap();
                        }

                        // Execute rules on facts
                        engine.execute(facts).unwrap()
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark complex SWRL rules with multiple built-ins
fn benchmark_complex_swrl_rules(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_swrl_rules");

    let facts = generate_swrl_facts(100);
    let complex_rules = generate_complex_swrl_rules();

    group.bench_function("complex_math_rules", |b| {
        b.iter(|| {
            let mut engine = SwrlEngine::new();

            for rule in &complex_rules {
                engine.add_rule(rule.clone()).unwrap();
            }

            engine.execute(&facts).unwrap()
        });
    });

    group.finish();
}

/// Benchmark SWRL built-in predicate performance individually
fn benchmark_builtin_predicates(c: &mut Criterion) {
    let mut group = c.benchmark_group("builtin_predicates");

    // Test different built-in categories
    let test_cases = vec![
        (
            "comparison",
            vec![
                ("equal", vec!["5", "5"]),
                ("greaterThan", vec!["10", "5"]),
                ("lessThan", vec!["3", "8"]),
            ],
        ),
        (
            "mathematics",
            vec![
                ("add", vec!["5", "3", "8"]),
                ("multiply", vec!["4", "6", "24"]),
                ("pow", vec!["2", "3", "8"]),
                ("sqrt", vec!["16", "4"]),
                ("sin", vec!["0", "0"]),
            ],
        ),
        (
            "string_operations",
            vec![
                ("stringLength", vec!["hello world", "11"]),
                ("upperCase", vec!["hello", "HELLO"]),
                ("lowerCase", vec!["WORLD", "world"]),
                ("substring", vec!["hello world", "0", "5", "hello"]),
            ],
        ),
    ];

    for (category, operations) in test_cases {
        group.bench_function(category, |b| {
            b.iter(|| {
                let mut engine = SwrlEngine::new();

                // Create and execute rules for each operation
                for (op_name, args) in &operations {
                    let rule = SwrlRule {
                        id: format!("{}_test", op_name),
                        body: vec![SwrlAtom::Builtin {
                            builtin_predicate: format!(
                                "http://www.w3.org/2003/11/swrlb#{}",
                                op_name
                            ),
                            arguments: args
                                .iter()
                                .map(|&s| SwrlArgument::Literal(s.to_string()))
                                .collect(),
                        }],
                        head: vec![SwrlAtom::Class {
                            class_predicate: "http://example.org/TestResult".to_string(),
                            argument: SwrlArgument::Individual("test_result".to_string()),
                        }],
                        metadata: HashMap::new(),
                    };

                    engine.add_rule(rule).unwrap();
                }

                // Execute with empty facts (built-ins should still be evaluated)
                engine.execute(&[]).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark SWRL engine statistics and overhead
fn benchmark_swrl_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("swrl_overhead");

    // Test engine creation overhead
    group.bench_function("engine_creation", |b| {
        b.iter(|| SwrlEngine::new());
    });

    // Test rule addition overhead
    group.bench_function("rule_addition", |b| {
        let rules = generate_swrl_rules(50);

        b.iter(|| {
            let mut engine = SwrlEngine::new();
            for rule in &rules {
                engine.add_rule(rule.clone()).unwrap();
            }
        });
    });

    // Test statistics computation
    group.bench_function("statistics", |b| {
        let mut engine = SwrlEngine::new();
        let rules = generate_swrl_rules(20);

        for rule in rules {
            engine.add_rule(rule).unwrap();
        }

        b.iter(|| engine.get_stats());
    });

    group.finish();
}

/// Benchmark SWRL rule conversion performance
fn benchmark_rule_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("rule_conversion");

    let swrl_rules = generate_swrl_rules(100);

    group.bench_function("swrl_to_internal_conversion", |b| {
        b.iter(|| {
            let engine = SwrlEngine::new();

            for rule in &swrl_rules {
                // This tests the internal conversion process
                let _ = engine.add_rule(rule.clone());
            }
        });
    });

    group.finish();
}

/// Benchmark variable binding and unification
fn benchmark_variable_binding(c: &mut Criterion) {
    let mut group = c.benchmark_group("variable_binding");

    // Create rules with many variables to test binding performance
    let complex_binding_rule = SwrlRule {
        id: "variable_binding_test".to_string(),
        body: vec![
            SwrlAtom::Class {
                class_predicate: "http://example.org/Person".to_string(),
                argument: SwrlArgument::Variable("person".to_string()),
            },
            SwrlAtom::DatavalueProperty {
                property_predicate: "http://example.org/hasAge".to_string(),
                argument1: SwrlArgument::Variable("person".to_string()),
                argument2: SwrlArgument::Variable("age".to_string()),
            },
            SwrlAtom::DatavalueProperty {
                property_predicate: "http://example.org/hasName".to_string(),
                argument1: SwrlArgument::Variable("person".to_string()),
                argument2: SwrlArgument::Variable("name".to_string()),
            },
            // Multiple built-ins with shared variables
            SwrlAtom::Builtin {
                builtin_predicate: "http://www.w3.org/2003/11/swrlb#greaterThan".to_string(),
                arguments: vec![
                    SwrlArgument::Variable("age".to_string()),
                    SwrlArgument::Literal("18".to_string()),
                ],
            },
            SwrlAtom::Builtin {
                builtin_predicate: "http://www.w3.org/2003/11/swrlb#stringLength".to_string(),
                arguments: vec![
                    SwrlArgument::Variable("name".to_string()),
                    SwrlArgument::Variable("name_length".to_string()),
                ],
            },
        ],
        head: vec![SwrlAtom::Class {
            class_predicate: "http://example.org/QualifiedPerson".to_string(),
            argument: SwrlArgument::Variable("person".to_string()),
        }],
        metadata: HashMap::new(),
    };

    let facts = generate_swrl_facts(100);

    group.bench_function("complex_variable_binding", |b| {
        b.iter(|| {
            let mut engine = SwrlEngine::new();
            engine.add_rule(complex_binding_rule.clone()).unwrap();
            engine.execute(&facts).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    swrl_benches,
    benchmark_swrl_execution,
    benchmark_complex_swrl_rules,
    benchmark_builtin_predicates,
    benchmark_swrl_overhead,
    benchmark_rule_conversion,
    benchmark_variable_binding
);

criterion_main!(swrl_benches);
