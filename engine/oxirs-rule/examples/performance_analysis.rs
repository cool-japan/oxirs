//! Performance analysis example for the OxiRS rule engine
//!
//! This example demonstrates how to use the performance profiling
//! and analysis utilities to measure rule engine performance.

use oxirs_rule::{
    integration::{rule_builders, RuleIntegration},
    performance::{PerformanceTestHarness, PerformanceThresholds, RuleEngineProfiler},
    Rule, RuleAtom, RuleEngine, Term,
};

fn main() -> anyhow::Result<()> {
    println!("🔍 OxiRS Rule Engine Performance Analysis Example");
    println!("================================================");

    // Example 1: Basic performance profiling
    basic_profiling_example()?;

    // Example 2: Comprehensive performance testing
    comprehensive_testing_example()?;

    // Example 3: Memory stress testing
    memory_stress_testing_example()?;

    // Example 4: Integration performance testing
    integration_performance_example()?;

    Ok(())
}

/// Example 1: Basic performance profiling
fn basic_profiling_example() -> anyhow::Result<()> {
    println!("\n📊 Example 1: Basic Performance Profiling");
    println!("-----------------------------------------");

    let mut profiler = RuleEngineProfiler::new();

    // Create rule engine and add some rules
    let rules = vec![
        Rule {
            name: "basic_rule_1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasType".to_string()),
                object: Term::Constant("Person".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant("intelligent".to_string()),
            }],
        },
        Rule {
            name: "basic_rule_2".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("hasProperty".to_string()),
                    object: Term::Constant("intelligent".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("hasAge".to_string()),
                    object: Term::Variable("Age".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("isQualified".to_string()),
                object: Term::Constant("true".to_string()),
            }],
        },
    ];

    // Generate test facts
    let facts = generate_test_facts(100);

    // Profile rule engine operations
    let mut engine = RuleEngine::new();

    profiler.profile_operation("rule_loading", || {
        for rule in rules {
            engine.add_rule(rule);
        }
    });

    profiler.profile_operation("fact_processing", || {
        engine.add_facts(facts.clone());
    });

    let _results = profiler.profile_operation("forward_chaining", || {
        engine.forward_chain(&facts).unwrap_or_default()
    });

    // Generate and print report
    profiler.print_report();

    Ok(())
}

/// Example 2: Comprehensive performance testing
fn comprehensive_testing_example() -> anyhow::Result<()> {
    println!("\n🧪 Example 2: Comprehensive Performance Testing");
    println!("-----------------------------------------------");

    let mut harness = PerformanceTestHarness::new();

    // Test with different scales
    let test_cases = vec![
        (50, 5, "Small scale"),
        (200, 10, "Medium scale"),
        (500, 15, "Large scale"),
    ];

    for (fact_count, rule_count, description) in test_cases {
        println!("\n🔬 Testing {description}: {fact_count} facts, {rule_count} rules");

        let facts = generate_test_facts(fact_count);
        let rules = generate_test_rules(rule_count);

        let metrics = harness.run_comprehensive_test(rules, facts);

        println!("  Total time: {:?}", metrics.total_time);
        println!("  Rules processed: {}", metrics.rules_processed);
        println!("  Facts processed: {}", metrics.facts_processed);
        println!("  Inferred facts: {}", metrics.inferred_facts);
        println!(
            "  Peak memory: {} bytes",
            metrics.memory_stats.peak_memory_usage
        );

        if !metrics.warnings.is_empty() {
            println!("  ⚠️  Warnings: {}", metrics.warnings.len());
        }
    }

    Ok(())
}

/// Example 3: Memory stress testing
fn memory_stress_testing_example() -> anyhow::Result<()> {
    println!("\n💾 Example 3: Memory Stress Testing");
    println!("-----------------------------------");

    let mut harness = PerformanceTestHarness::new();

    // Test with increasing memory pressure
    let stress_levels = vec![1, 2, 5]; // Scale factors

    for scale in stress_levels {
        println!("\n🧠 Memory stress test - Scale factor: {scale}");

        let metrics = harness.run_memory_stress_test(scale);

        println!("  Total time: {:?}", metrics.total_time);
        println!(
            "  Peak memory: {} MB",
            metrics.memory_stats.peak_memory_usage / (1024 * 1024)
        );
        println!(
            "  Facts memory: {} KB",
            metrics.memory_stats.facts_memory / 1024
        );
        println!(
            "  Rules memory: {} KB",
            metrics.memory_stats.rules_memory / 1024
        );

        if metrics.memory_stats.peak_memory_usage > 100 * 1024 * 1024 {
            // 100 MB
            println!("  ⚠️  High memory usage detected");
        }
    }

    Ok(())
}

/// Example 4: Integration performance testing
fn integration_performance_example() -> anyhow::Result<()> {
    println!("\n🔗 Example 4: Integration Performance Testing");
    println!("---------------------------------------------");

    let custom_thresholds = PerformanceThresholds {
        max_rule_loading_time: 500,       // 0.5 seconds
        max_forward_chaining_time: 2000,  // 2 seconds
        max_backward_chaining_time: 1000, // 1 second
        max_memory_usage: 512,            // 512 MB
        max_iterations: 500,
    };

    let mut profiler = RuleEngineProfiler::with_thresholds(custom_thresholds);

    // Test integration with oxirs-core
    let integration_result = profiler.profile_operation("integration_test", || {
        let mut integration = RuleIntegration::new();

        // Add RDFS rules
        integration.add_rules(rule_builders::all_rdfs_rules());

        // Generate RDF triples and add to store
        for i in 0..200 {
            let subject_iri = format!("http://example.org/person{i}");
            let subject = oxirs_core::NamedNode::new(&subject_iri).unwrap();
            let predicate =
                oxirs_core::NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .unwrap();
            let object = oxirs_core::NamedNode::new("http://example.org/Person").unwrap();

            let triple = oxirs_core::Triple::new(subject, predicate, object);
            integration.store.insert_triple(triple).unwrap();
        }

        // Apply rules and measure performance
        integration.apply_rules().unwrap()
    });

    println!("Integration test derived {integration_result} facts");

    // Generate detailed report
    let metrics = profiler.generate_report();

    // Export as JSON for further analysis
    if let Ok(json_report) = profiler.export_json() {
        println!("\n📄 JSON report (first 500 chars):");
        println!("{}", &json_report[..json_report.len().min(500)]);
        if json_report.len() > 500 {
            println!("... (truncated)");
        }
    }

    // Print warnings if any
    if !metrics.warnings.is_empty() {
        println!("\n⚠️  Performance Warnings:");
        for warning in &metrics.warnings {
            println!("  - {warning}");
        }
    } else {
        println!("\n✅ All performance thresholds met!");
    }

    Ok(())
}

/// Helper function to generate test facts
fn generate_test_facts(count: usize) -> Vec<RuleAtom> {
    let mut facts = Vec::with_capacity(count);

    for i in 0..count {
        // Person type facts
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("person_{i}")),
            predicate: Term::Constant("hasType".to_string()),
            object: Term::Constant("Person".to_string()),
        });

        // Age facts
        facts.push(RuleAtom::Triple {
            subject: Term::Constant(format!("person_{i}")),
            predicate: Term::Constant("hasAge".to_string()),
            object: Term::Literal((20 + i % 50).to_string()),
        });

        // Additional properties
        if i % 3 == 0 {
            facts.push(RuleAtom::Triple {
                subject: Term::Constant(format!("person_{i}")),
                predicate: Term::Constant("hasEducation".to_string()),
                object: Term::Constant("university".to_string()),
            });
        }
    }

    facts
}

/// Helper function to generate test rules
fn generate_test_rules(count: usize) -> Vec<Rule> {
    let mut rules = Vec::with_capacity(count);

    for i in 0..count {
        rules.push(Rule {
            name: format!("test_rule_{i}"),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasType".to_string()),
                object: Term::Constant("Person".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Constant(format!("property_{i}")),
            }],
        });
    }

    // Add some RDFS-style rules
    rules.extend(rule_builders::all_rdfs_rules());

    rules
}
