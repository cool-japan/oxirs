//! # OxiRS Rule Engine Comprehensive Tutorial
//!
//! This example demonstrates all major features of the OxiRS Rule Engine including:
//! - Forward and backward chaining
//! - RDFS and OWL reasoning  
//! - SWRL rule execution
//! - RETE network processing
//! - Debugging and performance analysis
//!
//! Run with: cargo run --example comprehensive_tutorial

use anyhow::Result;
use oxirs_rule::debug::DebuggableRuleEngine;
use oxirs_rule::owl::OwlReasoner;
use oxirs_rule::rdfs::RdfsReasoner;
use oxirs_rule::swrl::{SwrlArgument, SwrlAtom, SwrlEngine, SwrlRule};
use oxirs_rule::*;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("🚀 OxiRS Rule Engine Comprehensive Tutorial");
    println!("==========================================\n");

    // Demonstrate basic rule engine usage
    basic_rule_engine_example()?;

    // Demonstrate debugging capabilities
    debugging_example()?;

    // Demonstrate RDFS reasoning
    rdfs_reasoning_example()?;

    // Demonstrate OWL reasoning
    owl_reasoning_example()?;

    // Demonstrate SWRL rules
    swrl_reasoning_example()?;

    // Demonstrate performance analysis
    performance_analysis_example()?;

    println!("✅ Tutorial completed successfully!");
    Ok(())
}

/// Demonstrates basic rule engine functionality
fn basic_rule_engine_example() -> Result<()> {
    println!("📋 1. Basic Rule Engine Example");
    println!("-------------------------------\n");

    let mut engine = RuleEngine::new();

    // Define a simple rule: parent(X,Y) -> ancestor(X,Y)
    let parent_to_ancestor_rule = Rule {
        name: "parent_to_ancestor".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    };

    // Define transitive rule: ancestor(X,Y) ∧ ancestor(Y,Z) -> ancestor(X,Z)
    let transitive_ancestor_rule = Rule {
        name: "transitive_ancestor".to_string(),
        body: vec![
            RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Y".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Variable("Y".to_string()),
                predicate: Term::Constant("ancestor".to_string()),
                object: Term::Variable("Z".to_string()),
            },
        ],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("ancestor".to_string()),
            object: Term::Variable("Z".to_string()),
        }],
    };

    engine.add_rule(parent_to_ancestor_rule);
    engine.add_rule(transitive_ancestor_rule);

    // Add family facts
    let family_facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("mary".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("mary".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("tom".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("tom".to_string()),
            predicate: Term::Constant("parent".to_string()),
            object: Term::Constant("sue".to_string()),
        },
    ];

    println!("Family facts:");
    for fact in &family_facts {
        println!("  {:?}", fact);
    }

    // Perform forward chaining
    println!("\n🔄 Forward chaining results:");
    let forward_results = engine.forward_chain(&family_facts)?;
    for result in &forward_results {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = result
        {
            if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) =
                (subject, predicate, object)
            {
                if p == "ancestor" {
                    println!("  Derived: {} is ancestor of {}", s, o);
                }
            }
        }
    }

    // Test backward chaining
    println!("\n🔍 Backward chaining test:");
    let goal = RuleAtom::Triple {
        subject: Term::Constant("john".to_string()),
        predicate: Term::Constant("ancestor".to_string()),
        object: Term::Constant("sue".to_string()),
    };

    engine.add_facts(family_facts);
    let can_prove = engine.backward_chain(&goal)?;
    println!("  Can prove john is ancestor of sue: {}", can_prove);

    println!();
    Ok(())
}

/// Demonstrates debugging capabilities
fn debugging_example() -> Result<()> {
    println!("🐛 2. Debugging Example");
    println!("----------------------\n");

    let mut debug_engine = DebuggableRuleEngine::new();
    debug_engine.enable_debugging(false);

    // Add a simple rule for debugging
    debug_engine.engine.add_rule(Rule {
        name: "debug_rule".to_string(),
        body: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("input".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
        head: vec![RuleAtom::Triple {
            subject: Term::Variable("X".to_string()),
            predicate: Term::Constant("output".to_string()),
            object: Term::Variable("Y".to_string()),
        }],
    });

    // Add breakpoint
    debug_engine.add_breakpoint("debug_rule");

    let test_facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("entity1".to_string()),
            predicate: Term::Constant("input".to_string()),
            object: Term::Constant("value1".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("entity2".to_string()),
            predicate: Term::Constant("input".to_string()),
            object: Term::Constant("value2".to_string()),
        },
    ];

    // Execute with debugging
    let _results = debug_engine.debug_forward_chain(&test_facts)?;

    // Show debug information
    println!("Debug trace entries: {}", debug_engine.get_trace().len());

    let metrics = debug_engine.get_metrics();
    println!("Performance metrics:");
    println!("  Execution time: {:?}", metrics.total_execution_time);
    println!("  Facts processed: {}", metrics.facts_processed);
    println!("  Facts derived: {}", metrics.facts_derived);

    let conflicts = debug_engine.get_conflicts();
    println!("  Conflicts detected: {}", conflicts.len());

    // Generate and show debug report
    println!("\n📊 Debug Report:");
    let report = debug_engine.generate_debug_report();
    println!("{}", report);

    Ok(())
}

/// Demonstrates RDFS reasoning
fn rdfs_reasoning_example() -> Result<()> {
    println!("🔗 3. RDFS Reasoning Example");
    println!("---------------------------\n");

    let mut rdfs_reasoner = RdfsReasoner::new();

    // Create RDFS class hierarchy
    let rdfs_facts = vec![
        // Class hierarchy: Animal -> Mammal -> Dog
        RuleAtom::Triple {
            subject: Term::Constant("Mammal".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
            ),
            object: Term::Constant("Animal".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("Dog".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/2000/01/rdf-schema#subClassOf".to_string(),
            ),
            object: Term::Constant("Mammal".to_string()),
        },
        // Property hierarchy: hasChild -> hasOffspring
        RuleAtom::Triple {
            subject: Term::Constant("hasChild".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/2000/01/rdf-schema#subPropertyOf".to_string(),
            ),
            object: Term::Constant("hasOffspring".to_string()),
        },
        // Instance data
        RuleAtom::Triple {
            subject: Term::Constant("fido".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("Dog".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("fido".to_string()),
            predicate: Term::Constant("hasChild".to_string()),
            object: Term::Constant("puppy".to_string()),
        },
    ];

    println!("Input RDFS facts:");
    for fact in &rdfs_facts {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = fact
        {
            if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) =
                (subject, predicate, object)
            {
                let short_pred = p.split('#').last().unwrap_or(p);
                println!("  {} {} {}", s, short_pred, o);
            }
        }
    }

    // Perform RDFS inference
    let inferred_facts = rdfs_reasoner.infer(&rdfs_facts)?;

    println!("\n🧠 RDFS inferred facts:");
    for fact in &inferred_facts {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = fact
        {
            if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) =
                (subject, predicate, object)
            {
                // Skip input facts, show only inferred ones
                if !rdfs_facts.contains(fact) {
                    let short_pred = p.split('#').last().unwrap_or(p);
                    println!("  Inferred: {} {} {}", s, short_pred, o);
                }
            }
        }
    }

    println!();
    Ok(())
}

/// Demonstrates OWL reasoning
fn owl_reasoning_example() -> Result<()> {
    println!("🦉 4. OWL Reasoning Example");
    println!("-------------------------\n");

    let mut owl_reasoner = OwlReasoner::new();

    // OWL equivalence and transitivity examples
    let owl_facts = vec![
        // Class equivalence
        RuleAtom::Triple {
            subject: Term::Constant("Human".to_string()),
            predicate: Term::Constant("http://www.w3.org/2002/07/owl#equivalentClass".to_string()),
            object: Term::Constant("Person".to_string()),
        },
        // Property characteristics
        RuleAtom::Triple {
            subject: Term::Constant("knows".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("http://www.w3.org/2002/07/owl#SymmetricProperty".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("ancestorOf".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("http://www.w3.org/2002/07/owl#TransitiveProperty".to_string()),
        },
        // Instance data
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("Human".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("knows".to_string()),
            object: Term::Constant("bob".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("alice".to_string()),
            predicate: Term::Constant("ancestorOf".to_string()),
            object: Term::Constant("charlie".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("charlie".to_string()),
            predicate: Term::Constant("ancestorOf".to_string()),
            object: Term::Constant("diana".to_string()),
        },
    ];

    println!("Input OWL facts:");
    for fact in &owl_facts {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = fact
        {
            if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) =
                (subject, predicate, object)
            {
                let short_pred = p.split('#').last().unwrap_or(p);
                println!("  {} {} {}", s, short_pred, o);
            }
        }
    }

    // Perform OWL inference
    let owl_inferred = owl_reasoner.infer(&owl_facts)?;

    println!("\n🧠 OWL inferred facts:");
    for fact in &owl_inferred {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = fact
        {
            if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) =
                (subject, predicate, object)
            {
                // Skip input facts, show only inferred ones
                if !owl_facts.contains(fact) {
                    let short_pred = p.split('#').last().unwrap_or(p);
                    println!("  Inferred: {} {} {}", s, short_pred, o);
                }
            }
        }
    }

    println!();
    Ok(())
}

/// Demonstrates SWRL rule execution
fn swrl_reasoning_example() -> Result<()> {
    println!("📜 5. SWRL Reasoning Example");
    println!("---------------------------\n");

    let mut swrl_engine = SwrlEngine::new();

    // Create a SWRL rule: Person(?x) ∧ hasAge(?x, ?age) ∧ greaterThan(?age, 65) → Senior(?x)
    let senior_rule = SwrlRule {
        id: "senior_classification".to_string(),
        body: vec![
            SwrlAtom::Class {
                class_predicate: "Person".to_string(),
                argument: SwrlArgument::Variable("x".to_string()),
            },
            SwrlAtom::DatavalueProperty {
                property_predicate: "hasAge".to_string(),
                argument1: SwrlArgument::Variable("x".to_string()),
                argument2: SwrlArgument::Variable("age".to_string()),
            },
            SwrlAtom::Builtin {
                builtin_predicate: "http://www.w3.org/2003/11/swrlb#greaterThan".to_string(),
                arguments: vec![
                    SwrlArgument::Variable("age".to_string()),
                    SwrlArgument::Literal("65".to_string()),
                ],
            },
        ],
        head: vec![SwrlAtom::Class {
            class_predicate: "Senior".to_string(),
            argument: SwrlArgument::Variable("x".to_string()),
        }],
        metadata: HashMap::new(),
    };

    swrl_engine.add_rule(senior_rule)?;

    // Create another SWRL rule for discount eligibility
    let discount_rule = SwrlRule {
        id: "discount_eligibility".to_string(),
        body: vec![
            SwrlAtom::Class {
                class_predicate: "Senior".to_string(),
                argument: SwrlArgument::Variable("x".to_string()),
            },
            SwrlAtom::Class {
                class_predicate: "Customer".to_string(),
                argument: SwrlArgument::Variable("x".to_string()),
            },
        ],
        head: vec![SwrlAtom::DatavalueProperty {
            property_predicate: "eligibleForDiscount".to_string(),
            argument1: SwrlArgument::Variable("x".to_string()),
            argument2: SwrlArgument::Literal("true".to_string()),
        }],
        metadata: HashMap::new(),
    };

    swrl_engine.add_rule(discount_rule)?;

    // Add instance data
    let swrl_facts = vec![
        RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("Person".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("Customer".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("john".to_string()),
            predicate: Term::Constant("hasAge".to_string()),
            object: Term::Literal("72".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("mary".to_string()),
            predicate: Term::Constant(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            ),
            object: Term::Constant("Person".to_string()),
        },
        RuleAtom::Triple {
            subject: Term::Constant("mary".to_string()),
            predicate: Term::Constant("hasAge".to_string()),
            object: Term::Literal("45".to_string()),
        },
    ];

    println!("Input facts:");
    for fact in &swrl_facts {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = fact
        {
            if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) =
                (subject, predicate, object)
            {
                let short_pred = p.split('#').last().unwrap_or(p);
                println!("  {} {} {}", s, short_pred, o);
            }
        }
    }

    // Execute SWRL rules
    let swrl_results = swrl_engine.execute(&swrl_facts)?;

    println!("\n🧠 SWRL inferred facts:");
    for fact in &swrl_results {
        if let RuleAtom::Triple {
            subject,
            predicate,
            object,
        } = fact
        {
            if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) =
                (subject, predicate, object)
            {
                // Skip input facts, show only inferred ones
                if !swrl_facts.contains(fact) {
                    let short_pred = p.split('#').last().unwrap_or(p);
                    println!("  Inferred: {} {} {}", s, short_pred, o);
                }
            }
        }
    }

    println!();
    Ok(())
}

/// Demonstrates performance analysis
fn performance_analysis_example() -> Result<()> {
    println!("⚡ 6. Performance Analysis Example");
    println!("--------------------------------\n");

    let mut debug_engine = DebuggableRuleEngine::new();
    debug_engine.enable_debugging(false);

    // Add multiple rules for performance testing
    for i in 0..10 {
        debug_engine.engine.add_rule(Rule {
            name: format!("perf_rule_{}", i),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(format!("input_{}", i)),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant(format!("output_{}", i)),
                object: Term::Variable("Y".to_string()),
            }],
        });
    }

    // Generate large dataset for performance testing
    let mut large_facts = Vec::new();
    for i in 0..1000 {
        for j in 0..10 {
            large_facts.push(RuleAtom::Triple {
                subject: Term::Constant(format!("entity_{}", i)),
                predicate: Term::Constant(format!("input_{}", j)),
                object: Term::Constant(format!("value_{}_{}", i, j)),
            });
        }
    }

    println!(
        "Processing {} facts with {} rules...",
        large_facts.len(),
        10
    );

    // Execute with performance monitoring
    let start = std::time::Instant::now();
    let _results = debug_engine.debug_forward_chain(&large_facts)?;
    let total_time = start.elapsed();

    // Display performance metrics
    let metrics = debug_engine.get_metrics();
    println!("\n📊 Performance Results:");
    println!("  Total execution time: {:?}", total_time);
    println!("  Facts processed: {}", metrics.facts_processed);
    println!("  Facts derived: {}", metrics.facts_derived);
    println!("  Memory peak: {} bytes", metrics.memory_peak);

    // Show rule execution times
    println!("\n  Rule execution times:");
    let mut rule_times: Vec<_> = metrics.rule_execution_times.iter().collect();
    rule_times.sort_by(|a, b| b.1.cmp(a.1));
    for (rule, time) in rule_times.iter().take(5) {
        let count = metrics.rule_execution_counts.get(*rule).unwrap_or(&0);
        println!("    {}: {:?} ({} executions)", rule, time, count);
    }

    // Show throughput
    let throughput = metrics.facts_processed as f64 / total_time.as_secs_f64();
    println!("  Throughput: {:.0} facts/second", throughput);

    // Check for performance issues
    let conflicts = debug_engine.get_conflicts();
    if !conflicts.is_empty() {
        println!("\n⚠️  Performance issues detected:");
        for conflict in conflicts {
            println!("    {}", conflict.resolution_suggestion);
        }
    } else {
        println!("\n✅ No performance issues detected");
    }

    println!();
    Ok(())
}
